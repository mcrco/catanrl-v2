"""AlphaZero / expert iteration on cppanatron (DESIGN 1.6).

``train_alphazero(variant, writer)`` alternates, in one process:

1. **Self-play.** A VecEnv (auto-reset, the ``env`` scenarios as in PPO:
   mixed maps and player counts) plays games; every acting row is searched
   with a SearchPool, all searches in lockstep with their leaves evaluated in
   shared GPU batches (the eval's BatchEvaluator and CUDA graphs). Search is
   on the true game (``dets: 0``, fast) or on worlds sampled from the deciding
   seat's hand tracker (``dets: K``, fair; ``VecEnv(trackers=True)``). The
   move played is sampled from the search policy at a temperature that decays
   over the first ``temperature_moves`` decisions of each game (the engine's
   ``select``: visits for PUCT roots, the improved policy for Gumbel roots,
   which also add Gumbel noise themselves; PUCT roots can add Dirichlet noise).
   Playout cap randomization (KataGo): a ``full_search_prob`` fraction of
   moves get ``sims`` simulations and become policy targets, the rest get
   ``fast_sims`` and serve as value targets only.
2. **Targets.** Policy: the search's ``policy_target`` (the visit
   distribution, or Gumbel's improved policy), over legal actions. Value:
   each seat's outcome, seat-relative like the value head (slot k = the seat
   k places after the row's), 1 to the winner and 0 to the others; a
   truncated game gives every seat 1/n (the eval convention) or 0
   (``truncation_value: zero``).
3. **Replay.** A FIFO of ``replay_capacity`` positions sampled uniformly:
   float16 rows (exact for the integer topology blocks, about 1e-3 relative
   on features), float16 policy targets, float32 value targets. On a 2p BASE
   graph-s row (2,564 floats) a position takes about 5.8 KB: 200k positions
   are about 1.2 GB of RAM, and the trainer state (the buffer included, for
   an exact resume) as much on disk.
4. **Learning.** ``updates_per_iteration`` Adam(W) steps on minibatches:
   cross-entropy of the network's policy to the policy targets plus
   ``value_coef`` times each seated slot's binary cross-entropy to the value
   targets, weight decay as L2. The network is any ``build_model`` network,
   so PPO and DAgger checkpoints warm-start it through ``init_from``.

Checkpoints, select evals and ``best`` follow ``train_ppo`` (one epoch = one
iteration). A resume restores the weights, optimizer, replay buffer, counters
and RNG; games in flight restart.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cppanatron as cpp
import numpy as np
import torch
import torch.nn.functional as F

from catanrl.envs.encoding import critic_level_of, encoder_dims, info_level
from catanrl.envs.puffer import plan_env
from catanrl.models import CatanModel
from catanrl.search import BatchEvaluator, make_pool, run_pools, search_config
from catanrl.seeding import derive_seed, determinism, rng_state, seed_everything, set_rng_state
from catanrl.train.puffer import build_policy_model, check_bindable, resolve_device
from catanrl.train.select import Selector

#: The ``train`` keys AlphaZero reads (configs/train/alphazero.yaml has the same values).
ALPHAZERO_KEYS: dict[str, Any] = {
    "algo": "alphazero",
    "iterations": 100,
    "steps_per_iteration": 32,
    "root": "gumbel",
    "sims": 64,
    "fast_sims": 16,
    "full_search_prob": 0.25,
    "dets": 0,
    "c_puct": 0.75,
    "gumbel_m": 16,
    "dirichlet_noise": True,
    "temperature_start": 1.0,
    "temperature_end": 0.0,
    "temperature_moves": 30,
    "truncation_value": "share",
    "replay_capacity": 200_000,
    "min_replay": 5_000,
    "updates_per_iteration": 64,
    "minibatch_size": 512,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "value_coef": 1.0,
    "max_grad_norm": 1.0,
    "search_rows": 512,
    "checkpoint_interval": 10,
}


def check_train(train: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(set(ALPHAZERO_KEYS) - set(train))
    unknown = sorted(set(train) - set(ALPHAZERO_KEYS))
    if missing or unknown:
        raise ValueError(f"alphazero train keys: missing {missing}, unknown {unknown}")
    train = dict(train)
    if train["root"] not in ("puct", "gumbel"):
        raise ValueError(f"train.root must be puct or gumbel, not {train['root']!r}")
    if train["truncation_value"] not in ("share", "zero"):
        raise ValueError(f"train.truncation_value must be share or zero, not {train['truncation_value']!r}")
    if not 0.0 < float(train["full_search_prob"]) <= 1.0:
        raise ValueError("train.full_search_prob must be in (0, 1]")
    if int(train["sims"]) < 1 or int(train["fast_sims"]) < 1 or int(train["dets"]) < 0:
        raise ValueError("train.sims and train.fast_sims must be positive and train.dets non-negative")
    return train


def temperature(train: Mapping[str, Any], ply: int) -> float:
    """Linear from temperature_start at a game's first decision to temperature_end
    after temperature_moves decisions."""
    moves = int(train["temperature_moves"])
    start, end = float(train["temperature_start"]), float(train["temperature_end"])
    if moves <= 0 or ply >= moves:
        return end
    return start + (end - start) * ply / moves


def value_targets(winner: int | None, seat: int, num_players: int, max_players: int,
                  truncation: str) -> tuple[np.ndarray, np.ndarray]:
    """Seat-relative outcome targets of a row at ``seat`` and the slots present.

    Slot k is the seat k places after ``seat`` (the value head's order). A
    truncated game (``winner`` None) gives every seat 1/n (``share``) or 0.
    """
    targets = np.zeros(max_players, dtype=np.float32)
    present = np.zeros(max_players, dtype=np.float32)
    present[:num_players] = 1.0
    if winner is None:
        if truncation == "share":
            targets[:num_players] = 1.0 / num_players
    else:
        targets[(winner - seat) % num_players] = 1.0
    return targets, present


# ---------------------------------------------------------------- replay


class ReplayBuffer:
    """A FIFO of positions: float16 rows and policy targets, float32 value targets."""

    def __init__(self, capacity: int, obs_size: int, num_actions: int, max_players: int) -> None:
        self.capacity = int(capacity)
        self.shapes = (int(obs_size), int(num_actions), int(max_players))
        self.size = 0
        self.next = 0
        self.added = 0
        self.obs = self.policy = self.has_policy = self.value = self.present = None

    def _allocate(self) -> None:
        obs_size, num_actions, players = self.shapes
        self.obs = np.zeros((self.capacity, obs_size), dtype=np.float16)
        self.policy = np.zeros((self.capacity, num_actions), dtype=np.float16)
        self.has_policy = np.zeros(self.capacity, dtype=bool)
        self.value = np.zeros((self.capacity, players), dtype=np.float32)
        self.present = np.zeros((self.capacity, players), dtype=np.float32)

    def add(self, obs, policy, has_policy, value, present) -> None:
        if self.obs is None:
            self._allocate()
        count = len(obs)
        keep = slice(max(0, count - self.capacity), count)  # more than fits: the newest
        slots = (self.next + np.arange(count)[keep] - keep.start) % self.capacity
        for name, values in (("obs", obs), ("policy", policy), ("has_policy", has_policy), ("value", value),
                             ("present", present)):
            getattr(self, name)[slots] = np.asarray(values)[keep]
        self.next = (self.next + len(slots)) % self.capacity
        self.size = min(self.size + len(slots), self.capacity)
        self.added += count

    def batch(self, index: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        def tensor(array, dtype=torch.float32):
            return torch.from_numpy(np.ascontiguousarray(array[index])).to(device=device, dtype=dtype)

        return {"obs": tensor(self.obs), "policy": tensor(self.policy), "has_policy": tensor(self.has_policy, torch.bool),
                "value": tensor(self.value), "present": tensor(self.present)}

    def state(self) -> dict[str, Any]:
        """The buffer for the trainer state, oldest first (``weights_only``-loadable)."""
        if self.size == 0:
            return {"size": 0, "added": self.added}
        order = np.arange(self.next - self.size, self.next) % self.capacity
        arrays = {name: torch.from_numpy(np.ascontiguousarray(getattr(self, name)[order]))
                  for name in ("obs", "policy", "has_policy", "value", "present")}
        return {"size": self.size, "added": self.added, **arrays}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.size = self.next = 0
        self.added = int(state["added"])
        count = int(state["size"])
        if not count:
            return
        self._allocate()
        keep = slice(max(0, count - self.capacity), count)
        for name in ("obs", "policy", "has_policy", "value", "present"):
            values = state[name][keep].numpy()
            getattr(self, name)[: len(values)] = values
        self.size = min(count, self.capacity)
        self.next = self.size % self.capacity


# ---------------------------------------------------------------- self-play


@dataclass
class _Pending:
    obs: np.ndarray
    policy: np.ndarray | None
    seat: int
    num_players: int


class SelfPlay:
    """The VecEnv, two search pools (full and fast budgets) and the games' pending positions."""

    def __init__(self, plan: Any, train: Mapping[str, Any], *, model: CatanModel, num_envs: int, threads: int,
                 seed: int, device: str, mode: str) -> None:
        self.train = dict(train)
        self.seed = int(seed)
        self.dims = dict(plan.dims)
        levels = model.levels
        critic = critic_level_of(levels["level"], levels["critic_level"])
        self.dets = int(train["dets"])
        self.vec = cpp.VecEnv(plan.scenarios, num_envs=num_envs, dims=encoder_dims(plan.dims),
                              level=info_level(levels["level"]), seed=self.seed, num_threads=threads,
                              critic_level=None if critic is None else info_level(critic), trackers=self.dets > 0)
        self.buffers = self.vec.allocate()
        self.vec.reset(self.seed)
        self.players = int(self.vec.max_players)
        self.evaluator = BatchEvaluator(model, device=device, infer_rows=int(train["search_rows"]), mode=mode)

        def pool(sims: int) -> cpp.SearchPool:
            params = {"sims": sims, "dets": self.dets, "oracle": self.dets == 0, "root": train["root"],
                      "c_puct": train["c_puct"], "gumbel_m": train["gumbel_m"],
                      "noise": bool(train["dirichlet_noise"]) and train["root"] == "puct"}
            config = search_config(params, max_players=self.players, threads=threads)
            return make_pool(self.dims, levels["level"], config, critic_level=levels["critic_level"])

        self.full_pool, self.fast_pool = pool(int(train["sims"])), pool(int(train["fast_sims"]))
        self.pending: list[list[_Pending]] = [[] for _ in range(num_envs)]
        self.ply = np.zeros(num_envs, dtype=np.int64)
        self.decisions = 0
        self.games = 0

    def step(self, iteration: int, step: int, rng: np.random.Generator, replay: ReplayBuffer,
             stats: dict[str, float]) -> None:
        """Search every acting row, play the chosen moves and file finished games."""
        b = self.buffers
        acting = np.flatnonzero(b["acting"])
        full = rng.random(len(acting)) < float(self.train["full_search_prob"])
        searches = []
        try:
            for i, row in enumerate(acting):
                env, seat = int(b["env_index"][row]), int(b["seat"][row])
                pool = self.full_pool if full[i] else self.fast_pool
                tracker = self.vec.tracker(env, seat) if self.dets else None
                seed = derive_seed(self.seed, "selfplay", iteration, step, env)
                searches.append((pool, pool.add(self.vec.game(env), seed=seed, tracker=tracker)))
            leaves = run_pools([self.full_pool, self.fast_pool], self.evaluator,
                               batch_rows=[int(full.sum()), int((~full).sum())])
            for i, row in enumerate(acting):
                env, seat = int(b["env_index"][row]), int(b["seat"][row])
                pool, sid = searches[i]
                policy = None
                if full[i]:
                    policy = pool.policy_target(sid).astype(np.float16)
                    search = pool.stats(sid)
                    stats["depth"] += search.mean_depth
                    stats["max_depth"] += search.max_depth
                b["actions"][row] = pool.select(sid, temperature(self.train, int(self.ply[env])))
                self.pending[env].append(_Pending(b["observations"][row].astype(np.float16), policy, seat,
                                                  int(b["num_players"][env])))
                self.ply[env] += 1
        finally:
            for pool in (self.full_pool, self.fast_pool):
                if pool.pending:
                    pool.cancel()
                for sid in pool.ids:
                    pool.remove(sid)
        stats["leaves"] += leaves
        stats["full"] += int(full.sum())
        self.decisions += len(acting)
        self.vec.step()
        self._finish(replay)

    def _finish(self, replay: ReplayBuffer) -> None:
        b = self.buffers
        players = self.players
        for env in np.flatnonzero(b["terminals"].reshape(-1, players)[:, 0]):
            rows = self.pending[env]
            self.pending[env] = []
            self.ply[env] = 0
            self.games += 1
            if not rows:
                continue
            rewards = b["rewards"][env * players : (env + 1) * players]
            winner = int(np.argmax(rewards)) if rewards.max() > 0 else None
            targets = [value_targets(winner, r.seat, r.num_players, players, self.train["truncation_value"])
                       for r in rows]
            num_actions = self.full_pool.action_size
            replay.add(
                np.stack([r.obs for r in rows]),
                np.stack([r.policy if r.policy is not None else np.zeros(num_actions, np.float16) for r in rows]),
                np.array([r.policy is not None for r in rows]),
                np.stack([t for t, _ in targets]),
                np.stack([p for _, p in targets]),
            )


# ---------------------------------------------------------------- learning


def learn(model: CatanModel, optimizer: torch.optim.Optimizer, replay: ReplayBuffer, train: Mapping[str, Any],
          device: str, rng: np.random.Generator) -> dict[str, float]:
    """``updates_per_iteration`` gradient steps on uniform minibatches; their mean statistics."""
    model.train()
    sums: dict[str, float] = {}
    players = replay.shapes[2]
    by_n = np.zeros((players + 1, 3))  # count, predicted seat-sum, target seat-sum
    updates = int(train["updates_per_iteration"])
    for _ in range(updates):
        batch = replay.batch(rng.integers(0, replay.size, int(train["minibatch_size"])), device)
        logits, value_logits, _ = model.heads(batch["obs"])
        logp = F.log_softmax(logits.float(), dim=1)
        has = batch["has_policy"]
        target = batch["policy"][has]
        target = target / target.sum(1, keepdim=True).clamp_min(1e-8)
        policy_loss = -(target * logp[has]).sum(1).mean() if bool(has.any()) else logp.new_zeros(())
        present = batch["present"]
        value_loss = (F.binary_cross_entropy_with_logits(value_logits.float(), batch["value"], reduction="none")
                      * present).sum() / present.sum().clamp_min(1.0)
        loss = policy_loss + float(train["value_coef"]) * value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["max_grad_norm"]))
        optimizer.step()
        with torch.no_grad():
            probs = logp.exp()
            if bool(has.any()):
                net_logp = logp[has]
                legal = net_logp > -1e8
                target_entropy = -(target * target.clamp_min(1e-12).log()).sum(1).mean()
                net_entropy = -(probs[has] * torch.where(legal, net_logp, torch.zeros_like(net_logp))).sum(1).mean()
                kl = (target * (target.clamp_min(1e-12).log() - torch.where(legal, net_logp, torch.zeros_like(net_logp)))
                      ).sum(1).mean()
                for key, value in (("target_entropy", target_entropy), ("net_entropy", net_entropy), ("kl_target_net", kl)):
                    sums[key] = sums.get(key, 0.0) + float(value) / updates
            for key, value in (("policy_loss", policy_loss), ("value_loss", value_loss)):
                sums[key] = sums.get(key, 0.0) + float(value) / updates
            predicted = (torch.sigmoid(value_logits.float()) * present).sum(1)
            n = present.sum(1).long().cpu().numpy()
            np.add.at(by_n, (n, 0), 1.0)
            np.add.at(by_n, (n, 1), predicted.cpu().numpy())
            np.add.at(by_n, (n, 2), (batch["value"] * present).sum(1).cpu().numpy())
    for n in range(2, players + 1):
        if by_n[n, 0]:
            sums[f"value_sum_{n}p"] = by_n[n, 1] / by_n[n, 0]
            sums[f"target_sum_{n}p"] = by_n[n, 2] / by_n[n, 0]
    model.eval()
    return sums


# ---------------------------------------------------------------- the loop


def train_alphazero(variant: Any, writer: Any) -> None:
    """AlphaZero on ``variant`` (a :class:`catanrl.config.Variant`) into ``writer``'s run."""
    cfg = variant.config
    runtime = dict(cfg.get("runtime") or {})
    strict = runtime.get("determinism", "fast") == "strict"
    with determinism("strict" if strict else "fast"):
        _train(cfg, writer, check_train(cfg["train"]), runtime, strict)


def _train(cfg: Mapping[str, Any], writer: Any, train: dict[str, Any], runtime: Mapping[str, Any], strict: bool) -> None:
    seed = int(cfg["seed"])
    launch = int(writer.launch)
    env_cfg = cfg["env"]
    pool_seed = (env_cfg.get("map_pool") or {}).get("seed")
    plan = plan_env(env_cfg, pool_seed=derive_seed(seed, "map_pool") if pool_seed is None else int(pool_seed))
    seed_everything(derive_seed(seed, "init"))
    model = build_policy_model(cfg, plan.dims)
    check_bindable(model, plan)
    if cfg.get("init_from") and writer.mode == "new":
        from catanrl.checkpoint import load_parts

        load_parts(model, cfg["init_from"])
    device = resolve_device(runtime.get("device", "auto"))
    model.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]),
                                  weight_decay=float(train["weight_decay"]))
    selfplay = SelfPlay(plan, train, model=model, num_envs=int(env_cfg["num_envs"]),
                        threads=int(runtime.get("threads", 1)), seed=derive_seed(seed, "rollout", launch),
                        device=device, mode="strict" if strict else "fast")
    replay = ReplayBuffer(int(train["replay_capacity"]), selfplay.vec.observation_size, selfplay.vec.action_size,
                          selfplay.players)
    eval_runtime = {key: runtime[key] for key in ("device", "num_envs", "threads", "infer_rows") if key in runtime}
    selector = Selector((cfg.get("eval") or {}).get("select"), writer, eval_runtime)
    iteration = 0

    def save() -> None:
        ref = writer.save_checkpoint(model, epoch=iteration, agent_steps=selfplay.decisions)
        writer.save_trainer_state({
            "epoch": iteration, "decisions": selfplay.decisions, "games": selfplay.games,
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()), "replay": replay.state(), "rng": rng_state(),
            "map_pool": plan.pool["hash"],
        })
        if selector.due(iteration):
            selector.run(ref, model)

    state = writer.load_trainer_state() if writer.mode == "resume" else None
    if state is not None:
        if state.get("map_pool") != plan.pool["hash"]:
            raise RuntimeError("the map pool changed since the run started; refusing to resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        replay.restore(state["replay"])
        iteration = int(state["epoch"])
        selfplay.decisions, selfplay.games = int(state["decisions"]), int(state["games"])
        set_rng_state(state["rng"])
        selector.restore()
    else:
        save()
    for ref in writer.pending_selects():
        selector.run(ref, model)

    status, saved = "finished", iteration
    while iteration < int(train["iterations"]):
        if writer.should_stop():
            status = writer.stop_reason or "interrupted"
            break
        start = time.time()
        rng = np.random.default_rng(derive_seed(seed, "selfplay", iteration))
        decisions, games = selfplay.decisions, selfplay.games
        stats = {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}
        model.eval()
        for step in range(int(train["steps_per_iteration"])):
            selfplay.step(iteration, step, rng, replay, stats)
        played = time.time() - start
        learned = {}
        if replay.size >= int(train["min_replay"]):
            learned = learn(model, optimizer, replay, train, device,
                            np.random.default_rng(derive_seed(seed, "replay", iteration)))
        iteration += 1
        new = selfplay.decisions - decisions
        episodes = selfplay.vec.pop_stats()
        full = max(stats["full"], 1.0)
        logs = {
            "epoch": iteration, "decisions": selfplay.decisions, "games": selfplay.games, "replay": replay.size,
            "positions": replay.added, "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selfplay/decisions_per_s": new / played, "selfplay/leaves_per_s": stats["leaves"] / played,
            "selfplay/games_per_hour": (selfplay.games - games) * 3600.0 / played,
            "selfplay/full_fraction": stats["full"] / max(new, 1),
            "search/mean_depth": stats["depth"] / full, "search/max_depth": stats["max_depth"] / full,
            "environment/episodes": episodes["episodes"], "environment/truncated": episodes["truncated"],
            "environment/length": episodes["length"], "environment/learner_win": episodes["learner_win"],
            "throughput/selfplay_s": played, "throughput/iteration_s": time.time() - start,
            **{f"train/{k}": v for k, v in learned.items()}, "map_pool": plan.pool["hash"],
        }
        writer.log(logs, selfplay.decisions)
        if iteration % int(train["checkpoint_interval"]) == 0:
            save()
            saved = iteration
    if saved != iteration:
        save()
    writer.finish(status)
