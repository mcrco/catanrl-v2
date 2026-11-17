"""How far the checkpoints of a lineage are from each other, as critics and as policies, on one
fixed set of positions.

``play`` builds the positions. ``--games`` games on boards of a map pool drawn from another seed
than the runs' training pool (refused if any board is shared; ``tools/critic_heldout.py``'s
check, against every player's run config), in the players' own env (rules, turn limit). Each
game is self-play of one player network's raw policy, no search: every decision a temperature-1
sample of its masked logits on the true game state. The players (``--players``, checkpoint
specs) get equal shares of the games: env ``i`` of ``--envs`` belongs to player ``i % players``
and keeps its first ``--games / --envs`` games (an unbiased sample: long games are not cut off);
envs past their quota finish on uniform legal moves, unrecorded. From each game a uniform sample
of ``--per-game`` decisions is kept (all of them in a shorter game), a reservoir drawn as the
game goes. The env writes rows with the dice input (``obs.dice_state``); a network without it
plays from the same rows less the dice columns, which is exactly its own row: every kept
position is checked against the encoder without the input, bit for bit. The file holds, per
position, the row with the dice input (``rows``, float32; ``plain_columns`` are the columns
that make the row without it), the legal actions, the seat to move, the turn, the decision's
index in its game, the player and the game, and the game's outcome for the seat to move (``y``:
1 for its win, 0 for a loss or a game cut at the turn limit, which ``truncated`` flags).

``score`` runs checkpoints on those positions, in the order given, which is taken as training
order: ``--ckpts`` first, then every committed checkpoint of each ``--run`` at a multiple of
``--every``, and the run's last one (az-2p-kata/main+600's e410, where its fork went on). Each
network reads the row it was trained with (with or without the dice input: the checkpoint's own
``obs.dice_state``). Per checkpoint it keeps the win probability of the seat to move as the
trainer and the search read it (the sigmoid of the value head's slot 0: ``CatanModel.forward``,
what ``BatchEvaluator`` hands the search), the policy's log-probabilities on the legal actions
(float32; the positions' legal actions are one flat list,
``legal_action`` with ``legal_start`` per position: about 10 a position, so 111 checkpoints of
20,000 positions take about 110 MB where all 314 actions in float16 would take 1.4 GB), the
top action and its probability. A run that starts at e0 from a checkpoint of a run listed
before it continues that run's count: its eN is iteration (that checkpoint's iteration) + N.

``report`` reads the scores. For every pair of checkpoints: the mean absolute difference of
their win probabilities and their correlation (``value_mad``, ``value_corr``), the share of
positions where both policies' top action is the same (``top1_agree``) and the mean
Jensen-Shannon divergence of the two policies over the legal actions, in bits (``js``: 0 for
the same policy, 1 for disjoint ones); over all positions and over each player's positions
alone, so a conclusion can be checked against the positions' source. Per checkpoint: the Brier
score and log loss of its win probability against who won (truncated games left out), with
game-clustered bootstrap 95% CIs, over all positions and per player; and its distance to the
first checkpoint and to the previous one, as flat series. ``--figures DIR`` draws the three
matrices as heatmaps (matplotlib, which the project's environment does not have: without it the
figures are skipped and the report says so; ``report`` itself needs only numpy, so any Python
with both draws them, e.g. ``uv run --no-project --with numpy --with matplotlib``).

Nothing trains and nothing is written under the runs directory.

    uv run python tools/checkpoint_similarity.py play --games 2000 --device cuda --out positions.npz \\
        --players az-2p-kata/main@e0 az-2p-kata/main+600-fast-noselect@e900 az-2p-kata/main+1500@e1800 \\
        az-2p-bundle/main@latest
    uv run python tools/checkpoint_similarity.py score positions.npz --device cuda --out scores.npz \\
        --run az-2p-kata/main --run az-2p-kata/main+600 --run az-2p-kata/main+600-fast \\
        --run az-2p-kata/main+600-fast-noselect --run az-2p-kata/main+1500 --run az-2p-bundle/main
    uv run python tools/checkpoint_similarity.py report scores.npz --json similarity.json --figures figures/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-7
#: The pairwise measures, in the order of :func:`pair_matrices`' leading axis.
MEASURES = ("value_mad", "top1_agree", "js")
#: The one-hue ramp of the heatmaps, light to dark, from the figure's surface.
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
RAMP = (SURFACE, "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b")


def _critic_heldout():
    """tools/critic_heldout.py: its held-out map pool and its game-clustered bootstrap."""
    spec = importlib.util.spec_from_file_location("critic_heldout", Path(__file__).with_name("critic_heldout.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Progress:
    """A counter on stderr, at most every ``every`` seconds: done of total, the rate and the ETA."""

    def __init__(self, total: int, unit: str, *, every: float = 5.0, log: Callable[[str], None] | None = None) -> None:
        self.total, self.unit, self.every = int(total), unit, float(every)
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.start = self.last = time.time()

    def update(self, done: int, note: str = "", *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = f"{(self.total - done) / rate / 60:.1f} min" if rate > 0 else "?"
        self.log(f"{self.unit} {done}/{self.total} ({done / max(self.total, 1):.0%}), {rate * 60:.1f}/min, "
                 f"{elapsed / 60:.1f} min elapsed, ETA {eta}{note}")


# ---------------------------------------------------------------- play


def plain_columns(dims: Mapping[str, Any], level: str) -> np.ndarray:
    """The columns of a row with the dice input that make the row without it, in its order:
    every column but the dice features of the global block and of each tile."""
    from catanrl.envs.encoding import DICE_PREFIX, row_layout, row_size
    from catanrl.models.rows import feature_names

    layout = row_layout(dims, level, dice_state=True)
    names = feature_names(dims, level, True)
    dice = np.zeros(row_size(layout), dtype=bool)
    for block in ("global", "tiles"):
        offset, shape = layout[block]
        flags = np.array([name.startswith(DICE_PREFIX) for name in names[block]])
        slots = int(np.prod(shape)) // len(flags)
        dice[offset:offset + slots * len(flags)] = np.tile(flags, slots)
    columns = np.flatnonzero(~dice)
    if len(columns) != row_size(row_layout(dims, level)):
        raise ValueError("the dice features are not all a dice row has over a plain one")
    return columns


def heldout_plan(configs: Sequence[Mapping[str, Any]], pool_seed: int | None):
    """One held-out plan for the run configs of several networks: the first's env on a map pool
    from ``pool_seed`` (default: derived from the first's seed), refused if a board is in any
    config's training pool or a config's env gives another pool (other rules or maps)."""
    from catanrl.canon import digest
    from catanrl.seeding import derive_seed

    heldout = _critic_heldout().heldout_plan
    if pool_seed is None:
        pool_seed = derive_seed(int(configs[0]["seed"]), "checkpoint_similarity")
    plan, pools, seen = None, {}, set()
    for config in configs:
        key = digest([config["env"], int(config["seed"])])
        if key in seen:
            continue
        seen.add(key)
        other, theirs = heldout(config["env"], seed=int(config["seed"]), pool_seed=pool_seed)
        if plan is None:
            plan, pools = other, {**theirs, "train_pools": []}
        elif theirs["pool"] != pools["pool"] or other.dims != plan.dims:
            raise ValueError("the networks' runs play different envs (rules, maps or dims): one set of positions "
                             "cannot be held out for all of them")
        if theirs["train_pool"] not in pools["train_pools"]:
            pools["train_pools"].append(theirs["train_pool"])
    return plan, pools


def play_positions(players: Sequence[Any], plan, *, games: int, envs: int, per_game: int, seed: int,
                   threads: int = 1, device: str = "cpu", every: float = 5.0, log=None) -> dict[str, np.ndarray]:
    """``games`` self-play games on ``plan``, env ``i`` played by ``players[i % len(players)]``'s
    sampled policy, and a uniform sample of up to ``per_game`` decisions of each. Arrays: per
    position ``rows`` [N, W] (with the dice input), ``legal`` [N, A], ``game``, ``ply`` (the
    decision's index in its game), ``turn``, ``seat``, ``player``, ``y`` (the seat to move won)
    and ``truncated``; ``plain_columns``; per game ``g_id``, ``g_player``, ``g_T`` (decisions),
    ``g_winner`` (-1 none) and ``g_trunc``."""
    import cppanatron as cpp
    import torch

    from catanrl.envs.encoding import (
        check_dims,
        dice_state_of,
        encoder_dims,
        info_level,
        make_encoder,
        row_layout,
    )

    count = len(players)
    if games % envs or envs % count:
        raise ValueError(f"--games {games} must be a multiple of --envs {envs}, and that of the {count} players")
    per_env = games // envs
    level = players[0].levels["level"]
    for model in players:
        if model.levels["level"] != level or model.has_critic_block or getattr(model, "recurrent", False) \
                or check_dims(model.dims) != check_dims(plan.dims):
            raise ValueError("every player must read the env's rows at one level (the plan's dims, no critic block, "
                             "not recurrent)")
    seats = int(plan.dims["max_players"])
    vec = cpp.VecEnv(list(plan.scenarios), num_envs=envs, dims=encoder_dims(plan.dims), level=info_level(level),
                     seed=seed, num_threads=threads, vp_reward=0.0, trackers=False, dice_state=True)
    b = vec.allocate()
    vec.reset(seed)
    offset, (size,) = row_layout(plan.dims, level, dice_state=True)["action_mask"]
    plain = plain_columns(plan.dims, level)
    plain_index = torch.from_numpy(plain)
    plain_encoder = make_encoder(plan.dims, level)
    takes_dice = [dice_state_of(model.levels) for model in players]
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    owner = np.arange(envs) % count
    done = np.zeros(envs, np.int64)
    ply = np.zeros(envs, np.int64)
    kept: list[list[tuple]] = [[] for _ in range(envs)]  # per env, its game's reservoir
    rows_out, meta_out, games_out = [], [], []
    progress = Progress(games, "games", every=every, log=log)
    while (done < per_env).any():
        acting = np.flatnonzero(b["acting"])
        envs_of = b["env_index"][acting].astype(np.int64)
        if len(np.unique(envs_of)) != len(envs_of):
            raise RuntimeError("two seats of one game act at once: the decision count would skip")
        live = done[envs_of] < per_env
        for p, model in enumerate(players):
            rows = acting[live & (owner[envs_of] == p)]
            if not len(rows):
                continue
            obs = torch.from_numpy(np.ascontiguousarray(b["observations"][rows], dtype=np.float32))
            if not takes_dice[p]:
                obs = obs[:, plain_index]
            with torch.inference_mode():
                probs = torch.softmax(model.policy_logits(obs.to(device)).double(), dim=-1).cpu()
            b["actions"][rows] = torch.multinomial(probs, 1, generator=gen).squeeze(1).numpy()
        rows, row_envs = acting[live], envs_of[live]
        if len(rows):  # the reservoir: decision t of a game replaces a kept one with probability per_game / (t + 1)
            t = ply[row_envs]
            slot = np.where(t < per_game, t, rng.integers(0, t + 1))
            for i in np.flatnonzero(slot < per_game):
                row, env = int(rows[i]), int(row_envs[i])
                game, seat = vec.game(env), int(b["seat"][row])
                obs = b["observations"][row].copy()
                if not np.array_equal(plain_encoder.encode(game, seat), obs[plain]):
                    raise RuntimeError("a row without its dice columns is not the encoder's row without the dice input")
                entry = (int(t[i]), int(game.num_turns), seat, obs)
                if slot[i] < len(kept[env]):
                    kept[env][slot[i]] = entry
                else:
                    kept[env].append(entry)
            ply[row_envs] += 1
        for row in acting[~live]:  # past the quota: any legal move, unrecorded
            b["actions"][row] = int(rng.choice(np.flatnonzero(b["observations"][row, offset:offset + size] > 0)))
        vec.step()
        for env in np.flatnonzero(b["terminals"].reshape(-1, seats)[:, 0]):
            if done[env] < per_env:
                rewards = b["rewards"][env * seats:(env + 1) * seats]
                winner = int(np.argmax(rewards)) if rewards.max() > 0 else -1
                truncated = int(bool(b["truncations"][env * seats]))
                gid = int(env * per_env + done[env])
                games_out.append((gid, int(owner[env]), int(ply[env]), winner, truncated))
                for t, turn, seat, obs in kept[env]:
                    meta_out.append((gid, t, turn, seat, int(owner[env]), int(winner == seat), truncated))
                    rows_out.append(obs)
                done[env] += 1
            kept[env], ply[env] = [], 0
        progress.update(int(np.minimum(done, per_env).sum()), f", positions {len(rows_out)}")
    progress.update(games, f", positions {len(rows_out)}", force=True)
    m = np.array(meta_out, dtype=np.int64)
    g = np.array(sorted(games_out), dtype=np.int64)
    order = np.lexsort((m[:, 1], m[:, 0]))  # by game, then by decision
    m, rows = m[order], np.stack(rows_out)[order]
    return {"rows": rows, "plain_columns": plain.astype(np.int32), "legal": rows[:, offset:offset + size] > 0,
            "game": m[:, 0].astype(np.int32), "ply": m[:, 1].astype(np.int32), "turn": m[:, 2].astype(np.int32),
            "seat": m[:, 3].astype(np.int8), "player": m[:, 4].astype(np.int8), "y": m[:, 5].astype(np.int8),
            "truncated": m[:, 6].astype(bool), "g_id": g[:, 0].astype(np.int32), "g_player": g[:, 1].astype(np.int8),
            "g_T": g[:, 2].astype(np.int32), "g_winner": g[:, 3].astype(np.int8), "g_trunc": g[:, 4].astype(bool)}


def _run_config(ckpt) -> dict:
    config = ckpt.header.get("config") or {}
    if "env" not in config:
        raise SystemExit(f"{ckpt.ref} carries no run config: its env and training pool are unknown")
    return config


def play(args) -> dict:
    import torch

    from catanrl.runs import load_checkpoint

    torch.set_num_threads(args.threads)
    ckpts = [load_checkpoint(ref) for ref in args.players]
    plan, pools = heldout_plan([_run_config(c) for c in ckpts], args.pool_seed)
    players = [c.model("all", device=args.device) for c in ckpts]
    start = time.time()
    data = play_positions(players, plan, games=args.games, envs=args.envs, per_game=args.per_game, seed=args.seed,
                          threads=args.env_threads, device=args.device, every=args.progress)
    manifest = {"players": [str(c.ref) for c in ckpts], "dice_state": [bool(c.dice_state) for c in ckpts],
                "level": players[0].levels["level"], "dims": dict(plan.dims), "seed": args.seed, "envs": args.envs,
                "per_game": args.per_game, **pools, "games": len(data["g_id"]),
                "positions": len(data["game"]), "truncated_games": int(data["g_trunc"].sum()),
                "decisions": int(data["g_T"].sum()), "row_width": [len(data["plain_columns"]),
                                                                   int(data["rows"].shape[1])],
                "device": args.device, "wall_s": time.time() - start}
    np.savez_compressed(args.out, manifest=np.array(json.dumps(manifest)), **data)
    return manifest


# ---------------------------------------------------------------- score


def legal_index(legal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``legal`` [N, A] as one flat list: (``start`` [N + 1], ``action`` [L]); position i's legal
    actions are ``action[start[i]:start[i + 1]]``, ascending."""
    counts = legal.sum(1)
    if not counts.all():
        raise ValueError("a position without a legal action")
    start = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return start, np.nonzero(legal)[1].astype(np.int16)


def score_network(model, positions: Mapping[str, np.ndarray], *, device: str = "cpu", batch: int = 4096
                  ) -> dict[str, np.ndarray]:
    """One network on every position, at its own row (with the dice input or without):
    ``value`` [N] (the seat to move's win probability, the sigmoid of the value head's slot 0),
    ``logp`` [L] (the policy's log-probabilities on the legal actions, :func:`legal_index`'s
    order), ``top`` [N] (its most likely action) and ``top_p`` [N]."""
    import torch

    from catanrl.envs.encoding import check_dims, dice_state_of

    manifest = json.loads(str(positions["manifest"]))
    rows, legal = positions["rows"], positions["legal"]
    dice = dice_state_of(model.levels)
    width = rows.shape[1] if dice else len(positions["plain_columns"])
    if model.levels["level"] != manifest["level"] or check_dims(model.dims) != check_dims(manifest["dims"]) \
            or model.has_critic_block or getattr(model, "recurrent", False) or model.obs_size != width:
        raise ValueError(f"the network does not read the positions' rows (level {manifest['level']}, "
                         f"dims {manifest['dims']}, {width} floats, no critic block, not recurrent)")
    columns = None if dice else torch.from_numpy(positions["plain_columns"].astype(np.int64))
    value, logp, top, top_p = [], [], [], []
    for s in range(0, len(rows), batch):
        obs = torch.from_numpy(np.ascontiguousarray(rows[s:s + batch], dtype=np.float32))
        if columns is not None:
            obs = obs[:, columns]
        with torch.inference_mode():
            logits, values = model(obs.to(device))
            lp = torch.log_softmax(logits.double(), dim=-1).cpu().numpy()
        mask = legal[s:s + batch]
        best = lp.argmax(1)
        if not mask[np.arange(len(best)), best].all():
            raise RuntimeError("a network's top action is not a legal one: its mask is not the positions'")
        value.append(values[:, 0].float().cpu().numpy())
        logp.append(lp[mask].astype(np.float32))
        top.append(best.astype(np.int16))
        top_p.append(np.exp(lp.max(1)).astype(np.float32))
    return {"value": np.concatenate(value), "logp": np.concatenate(logp), "top": np.concatenate(top),
            "top_p": np.concatenate(top_p)}


#: The per-position arrays of a positions file that a scores file carries on (the report's inputs).
CARRIED = ("game", "ply", "turn", "seat", "player", "y", "truncated")


def assemble(positions: Mapping[str, np.ndarray], scores: Sequence[Mapping[str, np.ndarray]],
             checkpoints: Sequence[Mapping[str, Any]], **manifest: Any) -> dict[str, np.ndarray]:
    """The scores file: per checkpoint ``value``, ``top``, ``top_p`` [K, N] and ``logp`` [K, L],
    the legal actions' flat list, the positions' own arrays and the manifest."""
    start, action = legal_index(positions["legal"])
    manifest = {"positions": json.loads(str(positions["manifest"])), "checkpoints": list(checkpoints), **manifest}
    return {"manifest": np.array(json.dumps(manifest)), "legal_start": start, "legal_action": action,
            **{name: np.stack([s[name] for s in scores]) for name in ("value", "top", "top_p", "logp")},
            **{name: positions[name] for name in CARRIED}}


def checkpoint_list(ckpts: Sequence[str], runs: Sequence[str], every: int) -> list[dict]:
    """The checkpoints to score, ``ckpts`` first and then each run's committed ones at a multiple
    of ``every`` and its last, by epoch: ``ref`` (pinned), ``run``, ``epoch`` and ``iteration``.
    A run with an e0 that ``init_from`` took from a checkpoint of a run already listed continues
    its count."""
    from catanrl.runs import load_checkpoint, load_experiment, parse_ref

    offsets: dict[str, int] = {}

    def offset(name: str | None) -> int:
        if name is None:
            return 0
        if name not in offsets:
            run = load_experiment(name)
            init = (run.config.get("init_from") or {}).get("ckpt")
            source, which, _ = parse_ref(init) if isinstance(init, str) and not init.endswith(".pt") else (None,) * 3
            continues = source in offsets and re.fullmatch(r"e?\d+", which or "") \
                and any(c.epoch == 0 for c in run.checkpoints())
            offsets[name] = offsets[source] + int(which.lstrip("e")) if continues else 0
        return offsets[name]

    refs = [load_checkpoint(ref).ref for ref in ckpts]
    for name in runs:
        committed = load_experiment(name).checkpoints()
        refs += [c for c in committed if c.epoch % every == 0 or c is committed[-1]]
    if not refs:
        raise SystemExit("name checkpoints (--ckpts) or runs (--run)")
    return [{"ref": str(ref), "run": ref.run, "epoch": int(ref.epoch), "iteration": offset(ref.run) + int(ref.epoch)}
            for ref in refs]


def score(args) -> dict:
    import torch

    from catanrl.canon import digest
    from catanrl.runs import load_checkpoint

    torch.set_num_threads(args.threads)
    positions = dict(np.load(args.positions))
    manifest = json.loads(str(positions["manifest"]))
    checkpoints = checkpoint_list(args.ckpts or [], args.run or [], args.every)
    seen: set[str] = set()
    scores = []
    start = time.time()
    progress = Progress(len(checkpoints), "checkpoints", every=args.progress)
    for i, entry in enumerate(checkpoints):
        ckpt = load_checkpoint(entry["ref"])
        config = _run_config(ckpt)
        key = digest([config["env"], int(config["seed"])])
        if key not in seen:  # held out from this network's training pool too, in the positions' env
            seen.add(key)
            _, pools = heldout_plan([config], manifest["pool_seed"])
            if pools["pool"] != manifest["pool"]:
                raise SystemExit(f"{ckpt.ref} trained in another env than the positions' (other rules or maps)")
        entry["dice_state"] = bool(ckpt.dice_state)
        scores.append(score_network(ckpt.model("all", device=args.device), positions, device=args.device,
                                    batch=args.batch))
        progress.update(i + 1, f", {entry['ref']}", force=i + 1 == len(checkpoints))
    data = assemble(positions, scores, checkpoints, device=args.device, wall_s=time.time() - start)
    np.savez_compressed(args.out, **data)
    return {"checkpoints": len(checkpoints), "positions": len(positions["game"]),
            "legal_actions": len(data["legal_action"]), "first": checkpoints[0], "last": checkpoints[-1],
            "wall_s": time.time() - start}


# ---------------------------------------------------------------- report


def entropy(p: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The entropy in bits of each position's distribution: ``p`` [..., L] over the flat legal
    actions (:func:`legal_index`'s ``start``) -> [..., N]."""
    plogp = np.where(p > 0, p * np.log2(np.maximum(p, np.finfo(np.float64).tiny)), 0.0)
    return -np.add.reduceat(plogp, start[:-1], axis=-1)


def js_divergence(p: np.ndarray, q: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The Jensen-Shannon divergence in bits of each position's two distributions (``p`` and
    ``q`` [..., L], broadcast) -> [..., N]: 0 for the same one, 1 for disjoint ones."""
    js = entropy((p + q) / 2, start) - (entropy(p, start) + entropy(q, start)) / 2
    return np.clip(js, 0.0, 1.0)


def pair_matrices(value: np.ndarray, top: np.ndarray, p: np.ndarray, start: np.ndarray, weights: np.ndarray,
                  *, chunk: int = 16, progress: Progress | None = None) -> np.ndarray:
    """[3, S, K, K]: for every pair of checkpoints and each of the S columns of ``weights``
    [N, S] (a subset of the positions each, summing to 1), the weighted mean over the positions
    of |the two values' difference|, of the top actions' agreement and of the policies'
    Jensen-Shannon divergence (:data:`MEASURES`' order). ``value`` and ``top`` [K, N], ``p`` [K, L]."""
    k = len(value)
    out = np.zeros((len(MEASURES), weights.shape[1], k, k))
    own = entropy(p, start)
    for i in range(k):
        for j in range(i, k, chunk):
            rest = slice(j, min(j + chunk, k))
            mixed = entropy((p[i] + p[rest]) / 2, start)
            cells = (np.abs(value[i] - value[rest]), (top[i] == top[rest]).astype(np.float64),
                     np.clip(mixed - (own[i] + own[rest]) / 2, 0.0, 1.0))
            for measure, cell in enumerate(cells):
                out[measure, :, i, rest] = (cell @ weights).T
                out[measure, :, rest, i] = out[measure, :, i, rest]
        if progress is not None:
            progress.update(i + 1, force=i + 1 == k and time.time() - progress.start > progress.every)
    return out


def losses(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[N, 2K]: per position each checkpoint's Brier score, then each one's log loss (``p`` [K, N])."""
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.concatenate(((p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log1p(-p)))).T


def _clean(values: np.ndarray) -> Any:
    """Nested lists of floats rounded to 6 places, NaN as None (JSON has none)."""
    values = np.round(np.asarray(values, dtype=np.float64), 6)
    return np.where(np.isnan(values), None, values.astype(object)).tolist()


def report(data: Mapping[str, np.ndarray], *, boot: int = 2000, every: float = 5.0) -> dict:
    manifest = json.loads(str(data["manifest"]))
    checkpoints, players = manifest["checkpoints"], manifest["positions"]["players"]
    value = data["value"].astype(np.float64)
    top, start, game, player = data["top"], data["legal_start"], data["game"], data["player"]
    y, scored = data["y"].astype(np.float64), ~data["truncated"].astype(bool)
    k, n = value.shape
    subsets = {"all": np.ones(n, dtype=bool),
               **{f"player:{ref}": player == i for i, ref in enumerate(players) if (player == i).any()}}
    weights = np.stack([mask / mask.sum() for mask in subsets.values()], 1)
    pairs = pair_matrices(value, top, np.exp(data["logp"].astype(np.float64)), start, weights,
                          progress=Progress(k, "matrix rows", every=every))
    matrices = {}
    with np.errstate(invalid="ignore", divide="ignore"):  # a constant value has no correlation: None
        for s, (name, mask) in enumerate(subsets.items()):
            matrices[name] = {**{measure: pairs[m, s] for m, measure in enumerate(MEASURES)},
                              "value_corr": np.corrcoef(value[:, mask]).reshape(k, k)}
    cluster_ci = _critic_heldout().cluster_ci
    scores: dict[str, list[dict]] = {}
    for name, mask in subsets.items():  # Brier and log loss against who won, truncated games left out
        mask = mask & scored
        if not mask.any():
            continue
        cells = losses(value[:, mask], y[mask])
        lo, hi = cluster_ci(game[mask], cells, n=boot)
        mean = cells.mean(0)
        scores[name] = [{"brier": _clean([mean[i], lo[i], hi[i]]), "log_loss": _clean([mean[k + i], lo[k + i], hi[k + i]])}
                        for i in range(k)]
    entries = []
    for i, c in enumerate(checkpoints):
        entries.append({**c, **scores.get("all", [{}] * k)[i],
                        "by_player": {name.partition(":")[2]: rows[i] for name, rows in scores.items() if name != "all"}})
    whole = matrices["all"]
    measures = (*MEASURES, "value_corr")
    series = {"iteration": [c["iteration"] for c in checkpoints],
              "to_first": {m: _clean(whole[m][:, 0]) for m in measures},
              # the first checkpoint has no previous one: None
              "to_previous": {m: [None, *_clean(np.diagonal(whole[m], -1))] for m in measures}}
    runs: list[dict] = []
    for i, c in enumerate(checkpoints):  # the stretches of one run, in order: the figures' boundaries
        if runs and runs[-1]["run"] == c["run"]:
            runs[-1]["last"] = i
        else:
            runs.append({"run": c["run"], "first": i, "last": i})
    return {"manifest": manifest, "units": {"value_mad": "win probability", "js": "bits"},
            "positions": {name: int(mask.sum()) for name, mask in subsets.items()},
            "scored_positions": {name: int((mask & scored).sum()) for name, mask in subsets.items()},
            "scored_games": len(np.unique(game[scored])), "checkpoints": entries, "runs": runs, "series": series,
            "matrices": {name: {m: _clean(matrix) for m, matrix in block.items()} for name, block in matrices.items()}}


def print_report(out: Mapping[str, Any]) -> None:
    man, positions = out["manifest"]["positions"], out["positions"]
    print(f"{positions['all']} positions of {man['games']} games ({man['truncated_games']} truncated; "
          f"{out['scored_positions']['all']} positions scored against the outcome), boards from pool {man['pool']} "
          f"(training pools {', '.join(man['train_pools'])})")
    for name, count in positions.items():
        if name != "all":
            print(f"  {count} positions from {name.partition(':')[2]}")
    print("per checkpoint: Brier and log loss [mean, 95% CI, game-clustered]; then against the first checkpoint and "
          "the previous one: mean |value difference|, top-action agreement, Jensen-Shannon divergence (bits)")
    first, previous = out["series"]["to_first"], out["series"]["to_previous"]

    def show(v, spec):
        return "-" if v is None else format(v, spec)

    width = max(len(c["ref"]) for c in out["checkpoints"])
    for i, c in enumerate(out["checkpoints"]):
        outcome = ""
        if "brier" in c:
            b, ll = c["brier"], c["log_loss"]
            outcome = f"brier {b[0]:.4f} [{b[1]:.4f}, {b[2]:.4f}]  log loss {ll[0]:.4f} [{ll[1]:.4f}, {ll[2]:.4f}]  "
        print(f"  it {c['iteration']:<5} {c['ref']:<{width}} {outcome}"
              f"vs first {show(first['value_mad'][i], '.4f')} {show(first['top1_agree'][i], '.3f')} "
              f"{show(first['js'][i], '.4f')}  vs previous {show(previous['value_mad'][i], '.4f')} "
              f"{show(previous['top1_agree'][i], '.3f')} {show(previous['js'][i], '.4f')}")


#: Per heatmap: the measure, the title, the colorbar's label and whether the scale starts at 0
#: (a distance) or at the smallest cell (the agreement, which ends at 1).
FIGURES = (
    ("value_mad", "Value: mean absolute difference of the win probabilities",
     "mean |difference| of the seat to move's win probability", True),
    ("top1_agree", "Policy: share of positions with the same top action", "share of positions", False),
    ("js", "Policy: mean Jensen-Shannon divergence over the legal actions", "bits (0 same policy, 1 disjoint)", True),
)


def figures(out: Mapping[str, Any], directory: str | Path) -> list[Path]:
    """The value matrix and the two policy matrices over all positions as heatmaps (PNG) in
    ``directory``: both axes the checkpoints in order, ticked by iteration, the runs' boundaries
    drawn and the runs named along the top. Empty, with a note, without matplotlib."""
    try:
        import matplotlib
    except ImportError:
        print("matplotlib is not installed: no figures (the report's JSON holds every matrix)", file=sys.stderr)
        return []
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("one_hue", RAMP)
    cmap.set_bad("#e1e0d9")
    iteration = out["series"]["iteration"]
    k = len(iteration)
    step = next(s for s in (25, 50, 100, 150, 300, 600, 1200, 10**9) if (max(iteration) - min(iteration)) / s <= 10)
    ticks = [i for i, it in enumerate(iteration) if it % step == 0 and it not in iteration[:i]] if k > 12 else range(k)
    paths = []
    for measure, title, label, from_zero in FIGURES:
        matrix = np.array(out["matrices"]["all"][measure], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7.4, 6.6), dpi=160, facecolor=SURFACE)
        low, high = (0.0, float(np.nanmax(matrix))) if from_zero else (float(np.nanmin(matrix)), 1.0)
        image = ax.imshow(matrix, cmap=cmap, vmin=low, vmax=max(high, low + 1e-9), interpolation="nearest")
        ax.set_xticks(list(ticks), [str(iteration[i]) for i in ticks])
        ax.set_yticks(list(ticks), [str(iteration[i]) for i in ticks])
        ax.set_xlabel("iteration", color=MUTED)
        ax.set_ylabel("iteration", color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
            spine.set_linewidth(0.6)
        for run in out["runs"][1:]:  # a hairline where one run's checkpoints give way to the next's
            for line, width in ((SURFACE, 1.2), (MUTED, 0.4)):  # readable on the light cells and on the dark
                ax.axhline(run["first"] - 0.5, color=line, linewidth=width)
                ax.axvline(run["first"] - 0.5, color=line, linewidth=width)
        names = ax.secondary_xaxis("top")
        names.set_xticks([(run["first"] + run["last"]) / 2 for run in out["runs"]],
                         [str(run["run"]) for run in out["runs"]], rotation=35, ha="left", fontsize=7, color=MUTED)
        names.tick_params(length=0)
        for spine in names.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label(label, color=MUTED, fontsize=8)
        bar.ax.tick_params(colors=MUTED, labelsize=8, length=3)
        bar.outline.set_linewidth(0.6)
        bar.outline.set_edgecolor(MUTED)
        fig.canvas.draw()  # the title goes above the runs' names, wherever their slant ends
        top = fig.transFigure.inverted().transform((0, names.get_tightbbox(fig.canvas.get_renderer()).y1))[1]
        fig.suptitle(title, x=0.02, y=top + 0.02, ha="left", va="bottom", fontsize=11, color=INK)
        fig.text(0.02, 0.005, f"{out['positions']['all']} positions of {out['manifest']['positions']['games']} "
                 f"held-out self-play games; {k} checkpoints", fontsize=7, color=MUTED)
        path = directory / f"{measure}.png"
        fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play", help="play the held-out self-play games and keep a sample of their positions")
    p.add_argument("--players", nargs="+", required=True, help="checkpoint refs (run@eN or .pt paths): the networks "
                   "whose sampled policies play, in equal shares")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--envs", type=int, default=200, help="games in flight; a multiple of the players, dividing --games")
    p.add_argument("--per-game", type=int, default=10, help="positions kept per game")
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--pool-seed", type=int, default=None,
                   help="default: derive_seed(the first player's run seed, 'checkpoint_similarity')")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4, help="torch threads")
    p.add_argument("--env-threads", type=int, default=1)
    p.add_argument("--progress", type=float, default=5.0, help="seconds between progress lines")
    p.add_argument("--out", required=True)
    s = sub.add_parser("score", help="run checkpoints on the positions")
    s.add_argument("positions", help="play's file")
    s.add_argument("--ckpts", nargs="*", help="checkpoint refs (run@eN or .pt paths), scored first")
    s.add_argument("--run", action="append", help="every committed checkpoint of this run at a multiple of --every, "
                   "and its last; repeat in training order")
    s.add_argument("--every", type=int, default=25)
    s.add_argument("--device", default="cpu")
    s.add_argument("--batch", type=int, default=4096, help="rows per forward")
    s.add_argument("--threads", type=int, default=4, help="torch threads")
    s.add_argument("--progress", type=float, default=5.0, help="seconds between progress lines")
    s.add_argument("--out", required=True)
    r = sub.add_parser("report", help="the matrices, the outcome scores and the series")
    r.add_argument("npz", help="score's file")
    r.add_argument("--boot", type=int, default=2000)
    r.add_argument("--json")
    r.add_argument("--figures", help="a directory for the heatmaps (needs matplotlib)")
    args = parser.parse_args(argv)
    if args.command == "play":
        print(json.dumps(play(args), indent=2))
        return
    if args.command == "score":
        print(json.dumps(score(args), indent=2))
        return
    out = report(dict(np.load(args.npz)), boot=args.boot)
    if args.figures:
        out["figures"] = [str(path) for path in figures(out, args.figures)]
    if args.json:
        Path(args.json).write_text(json.dumps(out))
    print_report(out)
    for path in out.get("figures", []):
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
