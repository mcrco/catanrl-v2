"""DAgger from F: imitation warm start (v1's recipe on the VecEnv).

Each iteration plays ``steps_per_iteration`` VecEnv steps. At every acting
row the expert's action (Catanatron's ValueFunctionPlayer, the engine's
``value_action``) is the label; the env executes it with probability
``beta`` and otherwise a sample from the current policy. ``beta`` starts at
``beta0`` and is multiplied by ``beta_decay`` per iteration down to
``beta_min``. Rows are labelled with their game's outcome (each seat's win,
seat-relative; a truncation pays nobody) when it ends, and join a dataset of
at most ``dataset_cap`` rows (oldest dropped, stored as float16). The
network then trains ``epochs_per_iteration`` epochs on it: cross-entropy to
the expert plus ``vf_coef`` times the value head's binary cross-entropy to
the outcomes, so the critic is fitted alongside.

``train_imitation(variant, writer)`` has ``train_ppo``'s signature; the
``train`` section holds :data:`IMITATION_KEYS`.
"""

from __future__ import annotations

import contextlib
import copy
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import cppanatron as cpp
import numpy as np
import torch
import torch.nn.functional as F

from catanrl.envs.encoding import encoder_dims, info_level
from catanrl.envs.puffer import plan_env
from catanrl.train.puffer import (
    _derive_seed,
    _seed_everything,
    build_policy_model,
    check_bindable,
    resolve_device,
    select_eval,
)

#: The ``train`` keys DAgger reads, with v1's values.
IMITATION_KEYS: dict[str, Any] = {
    "algo": "imitation",
    "expert": "F",
    "expert_enemies": "next",
    "iterations": 24,
    "steps_per_iteration": 512,
    "beta0": 1.0,
    "beta_decay": 0.9,
    "beta_min": 0.1,
    "dataset_cap": 200_000,
    "epochs_per_iteration": 1,
    "minibatch_size": 512,
    "learning_rate": 3.0e-4,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "checkpoint_interval": 8,
}


def _check_train(train: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(set(IMITATION_KEYS) - set(train))
    unknown = sorted(set(train) - set(IMITATION_KEYS))
    if missing or unknown:
        raise ValueError(f"imitation train keys: missing {missing}, unknown {unknown}")
    if train["expert"] != "F":
        raise ValueError("the only native expert is F")
    return dict(train)


class _Dataset:
    """FIFO of (row, expert label, outcome targets, present seats)."""

    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.chunks: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque()
        self.size = 0

    def add(self, rows, labels, targets, present) -> None:
        self.chunks.append((rows.astype(np.float16), labels.astype(np.int64), targets.astype(np.float32),
                            present.astype(np.float32)))
        self.size += len(rows)
        while self.size - len(self.chunks[0][0]) >= self.cap:
            self.size -= len(self.chunks.popleft()[0])

    def arrays(self):
        return tuple(np.concatenate(part) for part in zip(*self.chunks))


class _Collector:
    """Runs the VecEnv under the DAgger mixture and labels finished games."""

    def __init__(self, plan, cfg, *, num_envs: int, threads: int, seed: int, level: str,
                 enemies: cpp.ValueEnemies) -> None:
        self.vec = cpp.VecEnv(plan.scenarios, num_envs=num_envs, dims=encoder_dims(plan.dims),
                              level=info_level(level), seed=seed, num_threads=threads)
        self.buffers = self.vec.allocate()
        self.vec.reset(seed)
        self.space = self.vec.encoder.action_space
        self.players = int(self.vec.max_players)
        self.enemies = enemies
        self.pending: list[list[tuple[np.ndarray, int, int, int]]] = [[] for _ in range(num_envs)]
        self.decisions = 0

    def collect(self, model: torch.nn.Module, device: str, steps: int, beta: float,
                rng: np.random.Generator, dataset: _Dataset) -> dict[str, float]:
        b = self.buffers
        agree = executed_expert = count = 0
        model.eval()
        for _ in range(steps):
            acting = np.flatnonzero(b["acting"])
            obs = b["observations"][acting]
            with torch.no_grad():
                logits = model.policy_logits(torch.from_numpy(obs).to(device)).float()
                sampled = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            for i, row in enumerate(acting):
                env, seat = int(b["env_index"][row]), int(b["seat"][row])
                game = self.vec.game(env)
                label = int(self.space.index(game, cpp.value_action(game, enemies=self.enemies)))
                use_expert = rng.random() < beta
                action = label if use_expert else int(sampled[i])
                executed_expert += use_expert
                agree += int(sampled[i]) == label
                b["actions"][row] = action
                self.pending[env].append((obs[i].copy(), label, seat, int(b["num_players"][env])))
            self.decisions += len(acting)
            count += len(acting)
            self.vec.step()
            self._finish(dataset)
        total = max(1, count)
        return {"policy_agreement": agree / total, "expert_executed": executed_expert / total}

    def _finish(self, dataset: _Dataset) -> None:
        b = self.buffers
        done = np.flatnonzero(b["terminals"].reshape(-1, self.players)[:, 0])
        for env in done:
            rows = self.pending[env]
            self.pending[env] = []
            if not rows:
                continue
            rewards = b["rewards"][env * self.players : (env + 1) * self.players]
            winner = int(np.argmax(rewards)) if rewards.max() > 0 else -1
            obs = np.stack([r[0] for r in rows])
            labels = np.array([r[1] for r in rows])
            targets = np.zeros((len(rows), self.players), dtype=np.float32)
            present = np.zeros((len(rows), self.players), dtype=np.float32)
            for i, (_, _, seat, n) in enumerate(rows):
                present[i, :n] = 1.0
                if winner >= 0:
                    targets[i, (winner - seat) % n] = 1.0
            dataset.add(obs, labels, targets, present)


def _fit(model, optimizer, dataset: _Dataset, cfg, device: str, rng: np.random.Generator) -> dict[str, float]:
    rows, labels, targets, present = dataset.arrays()
    model.train()
    order = rng.permutation(len(rows))
    batch = int(cfg["minibatch_size"])
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0}
    batches = 0
    for _ in range(int(cfg["epochs_per_iteration"])):
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            obs = torch.from_numpy(rows[idx].astype(np.float32)).to(device)
            label = torch.from_numpy(labels[idx]).to(device)
            target = torch.from_numpy(targets[idx]).to(device)
            seats = torch.from_numpy(present[idx]).to(device)
            logits, value_logits, _ = model.heads(obs)
            policy_loss = F.cross_entropy(logits.float(), label)
            value_loss = (F.binary_cross_entropy_with_logits(value_logits.float(), target, reduction="none")
                          * seats).sum() / seats.sum().clamp_min(1.0)
            loss = policy_loss + float(cfg["vf_coef"]) * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            optimizer.step()
            sums["policy_loss"] += policy_loss.item()
            sums["value_loss"] += value_loss.item()
            sums["accuracy"] += (logits.argmax(1) == label).float().mean().item()
            batches += 1
    return {key: value / max(1, batches) for key, value in sums.items()}


def train_imitation(variant: Any, writer: Any) -> None:
    """DAgger from F into ``writer``'s run (same contract as ``train_ppo``)."""
    cfg = variant.config
    train = _check_train(cfg["train"])
    runtime = dict(cfg.get("runtime") or {})
    seed = int(cfg["seed"])
    launch = int(getattr(writer, "launch", 0))
    env_cfg = cfg["env"]
    strict = runtime.get("determinism", "fast") == "strict"
    try:
        from catanrl.seeding import determinism
        guard = determinism("strict" if strict else "fast")
    except ImportError:
        guard = contextlib.nullcontext()
    with guard:
        _train(variant, writer, cfg, train, runtime, seed, launch, env_cfg)


def _train(variant, writer, cfg, train, runtime, seed, launch, env_cfg) -> None:
    pool_cfg = env_cfg.get("map_pool") or {}
    pool_seed = _derive_seed(seed, "map_pool") if pool_cfg.get("seed") is None else int(pool_cfg["seed"])
    plan = plan_env(env_cfg, pool_seed=pool_seed)
    _seed_everything(_derive_seed(seed, "init"))
    model = build_policy_model(cfg, plan.dims)
    check_bindable(model, plan)
    if cfg.get("init_from") and getattr(writer, "mode", "new") == "new":
        from catanrl.checkpoint import load_parts

        load_parts(model, cfg["init_from"])
    device = resolve_device(runtime.get("device", "auto"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train["learning_rate"]))
    obs = cfg.get("obs") or {}
    if obs.get("critic_level") not in (None, obs.get("level", "private")):
        raise NotImplementedError("DAgger fits the critic on the actor's rows; use one level")
    collector = _Collector(plan, train, num_envs=int(env_cfg["num_envs"]), threads=int(runtime.get("threads", 1)),
                           seed=_derive_seed(seed, "rollout", launch), level=obs.get("level", "private"),
                           enemies={"next": cpp.ValueEnemies.NEXT, "all": cpp.ValueEnemies.ALL}[train["expert_enemies"]])
    dataset = _Dataset(train["dataset_cap"])
    iteration, agent_steps = 0, 0
    beta = float(train["beta0"])
    select = (cfg.get("eval") or {}).get("select")
    eval_runtime = {key: runtime[key] for key in ("device", "num_envs", "threads", "infer_rows") if key in runtime}
    best: list[float | None] = [None]

    def run_select(ref) -> None:
        model.eval()
        record, metric = select_eval(select, ref, eval_runtime)
        if best[0] is None or metric > best[0]:
            best[0] = metric
            writer.set_alias("best", ref, record=record)

    def save() -> None:
        ref = writer.save_checkpoint(model, epoch=iteration, agent_steps=agent_steps)
        writer.save_trainer_state({
            "epoch": iteration, "agent_steps": agent_steps, "beta": beta, "decisions": collector.decisions,
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()), "map_pool": plan.pool["hash"], "best_metric": best[0],
        })
        if select and iteration % int(select.get("every", 1)) == 0:
            run_select(ref)

    state = writer.load_trainer_state() if getattr(writer, "mode", "new") == "resume" else None
    if state is not None:
        if state.get("map_pool") != plan.pool["hash"]:
            raise RuntimeError("the map pool changed since the run started; refusing to resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        iteration, agent_steps, beta = int(state["epoch"]), int(state["agent_steps"]), float(state["beta"])
        best[0] = state.get("best_metric")
    else:
        save()
    for ref in getattr(writer, "pending_selects", list)():
        if select:
            run_select(ref)
    status = "finished"
    saved = iteration
    while iteration < int(train["iterations"]):
        if writer.should_stop():
            status = getattr(writer, "stop_reason", None) or "interrupted"
            break
        start = time.time()
        rng = np.random.default_rng(_derive_seed(seed, "dagger", iteration))
        decisions = collector.decisions
        collected = collector.collect(model, device, int(train["steps_per_iteration"]), beta, rng, dataset)
        agent_steps += int(train["steps_per_iteration"]) * collector.vec.num_rows
        collect_time = time.time() - start
        fitted = _fit(model, optimizer, dataset, train, device, rng) if dataset.size else {}
        iteration += 1
        stats = collector.vec.pop_stats()
        logs = {
            "epoch": iteration, "agent_steps": agent_steps, "decisions": collector.decisions, "beta": beta,
            "dataset": dataset.size, **{f"imitation/{k}": v for k, v in {**collected, **fitted}.items()},
            "environment/learner_win": stats["learner_win"], "environment/episodes": stats["episodes"],
            "environment/truncated": stats["truncated"], "environment/length": stats["length"],
            "throughput/decisions_per_s": (collector.decisions - decisions) / collect_time,
            "throughput/iteration_s": time.time() - start, "map_pool": plan.pool["hash"],
        }
        writer.log(logs, agent_steps)
        beta = max(float(train["beta_min"]), beta * float(train["beta_decay"]))
        if iteration % int(train["checkpoint_interval"]) == 0:
            save()
            saved = iteration
    if saved != iteration:
        save()
    writer.finish(status)
