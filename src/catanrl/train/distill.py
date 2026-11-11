"""The offline distillation gate: search a network's own positions once, then fit copies of it
to targets rebuilt from those searches (AlphaZero's learning step, offline; az.typ's gate).

``train_distill(variant, writer)`` (``train.algo: distill``, :data:`DISTILL_KEYS`) runs one of
two phases (``train.phase``), each a variant of one experiment (configs/experiments/az-gate-2p.yaml):

1. ``collect``: the start network (``init_from``) plays ``games`` games against itself with its
   raw policy, each move a softmax sample of its masked logits (as PPO samples), ``env.num_envs``
   games at a time (a chunk) on ``env``'s scenarios (its training pool), each env's first game
   only. ``positions_per_game`` decisions of each game (rows with more than one legal action; the
   VecEnv plays forced moves itself) are kept uniformly (a reservoir sample) with the game. Every
   kept position is then searched once, ``search_rows`` searches in lockstep, on the true game
   (``dets: 0``): alphazero's SearchPool, BatchEvaluator and :class:`RootOutputs`. The chunk is
   written to ``chunks/cNNNNN.npz`` in the run folder and committed in the trainer state, so a
   resume skips it. At the end the chunks are merged into ``positions.npz`` (arrays below), its
   sha256 goes into ``positions.json``, the trainer state and the last metrics row, and the chunk
   files are removed.
2. ``fit``: the start network again (``init_from``, checked against the dataset's start) is
   trained on the dataset of the collect run ``train.dataset``. Games, not positions, are held
   out (``holdout_fraction``). The policy target, over the legal actions:

   * ``policy_target: engine``: the search's own (``SearchPool.policy_target``) as stored;
   * ``policy_target: gumbel``: Gumbel's improved policy rebuilt from the stored root statistics
     with ``target_*``'s sigma (:func:`catanrl.train.alphazero.gumbel_target`; at the engine's
     constants it is the engine's target);

   then mixed with the start's own policy at the root, ``(1 - target_anchor) * target +
   target_anchor * start`` (cross-entropy to it is the cross-entropy to the target plus a KL
   anchor to the start, scaled). The loss is that cross-entropy plus ``value_coef`` times each
   seated slot's binary cross-entropy to the start's own win probabilities at the position (the
   value anchor: the trunk is shared, so the critic must not drift). ``updates`` AdamW steps on
   uniform minibatches with replacement, a fresh optimizer; an epoch is one update, so
   ``e{updates}`` is the fitted network and ``e0`` the start's weights. Every ``log_interval``
   updates a row logs, on the held-out games (``heldout/*``) and on as many training positions
   (``probe/*``): cross-entropy to the target, KL(start || net), the net's entropy, argmax
   agreement with the start and with the target, value drift (mean |p - p_start| per seat),
   value KL to the start and the value's BCE to the real outcome; per phase of
   :data:`~catanrl.train.alphazero.PHASES` on the held-out games. Row e0 adds the targets' own
   statistics (``target/*``) and the dataset's (``data/*``).

The dataset (``positions.npz``, :data:`FORMAT`): per position (N rows) ``obs`` (float32 rows),
``start_value`` (the start's per-seat win probabilities at the root, seat-relative, as the
search read them), ``root_value`` (slot 0's, clamped: what completes an unvisited action's Q),
``search_value`` (the search's backed-up root values), ``depth`` and ``max_depth``, ``game``,
``game_seed``, ``seat``, ``turn``, ``decision`` (index among the game's decisions), ``decisions``
(the game's), ``winner`` (seat, -1 for a truncated game) and ``num_legal``; and per legal action
(``num_legal`` per position, in order) ``actions`` (codec index), ``prior_logits`` (the start's
logits at the root), ``visits``, ``q`` (float64, NaN unvisited) and ``engine`` (the engine's
target). That is everything :func:`gumbel_target` reads, so any sigma can be rebuilt offline.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from catanrl import canon
from catanrl.envs.puffer import plan_env
from catanrl.eval.results import Progress
from catanrl.models import CatanModel
from catanrl.search import BatchEvaluator, make_pool, run_pools, search_config
from catanrl.seeding import derive_seed, determinism, seed_everything
from catanrl.train.alphazero import RootOutputs, gumbel_target, make_vec, phases_of
from catanrl.train.puffer import build_policy_model, check_bindable, resolve_device, starts_fresh

#: The ``train`` keys the gate reads (configs/train/distill.yaml has the same values).
DISTILL_KEYS: dict[str, Any] = {
    "algo": "distill",
    "phase": "fit",
    "dataset": None,
    "games": 5120,
    "positions_per_game": 20,
    "root": "gumbel",
    "sims": 2048,
    "dets": 0,
    "c_puct": 0.75,
    "gumbel_m": 16,
    "kernels": "torch",
    "search_rows": 512,
    "policy_target": "engine",
    "target_c_visit": 50.0,
    "target_c_scale": 0.1,
    "target_rescale": True,
    "target_prior_temperature": 1.0,
    "target_anchor": 0.0,
    "holdout_fraction": 0.1,
    "updates": 128,
    "minibatch_size": 512,
    "learning_rate": 5.0e-5,
    "weight_decay": 1.0e-4,
    "max_grad_norm": 1.0,
    "value_coef": 1.0,
    "log_interval": 8,
    "checkpoint_interval": 64,
}
#: The collect keys that decide a dataset (with the start's weights, env and seed).
COLLECT_KEYS = ("games", "positions_per_game", "root", "sims", "dets", "c_puct", "gumbel_m", "kernels")
#: Dataset file format.
FORMAT = 1
DATASET = "positions.npz"
META = "positions.json"
CHUNKS = "chunks"
#: Per-position arrays and per-legal-action arrays of a dataset.
POSITION_ARRAYS = ("obs", "start_value", "root_value", "search_value", "depth", "max_depth", "game", "game_seed",
                   "seat", "turn", "decision", "decisions", "winner", "num_legal")
EDGE_ARRAYS = ("actions", "prior_logits", "visits", "q", "engine")
#: VecEnv steps a chunk of self-play may take before it is refused (a game is cut at its turn limit).
MAX_STEPS = 200_000
#: Rows per forward when scoring the probes.
PROBE_CHUNK = 2048
#: The start's logit at an illegal action in the dense start logits (the models' MASKED_LOGIT).
MASKED = -1e9


def check_train(train: Mapping[str, Any]) -> dict[str, Any]:
    from catanrl.config import check_distill

    missing = sorted(set(DISTILL_KEYS) - set(train))
    unknown = sorted(set(train) - set(DISTILL_KEYS))
    if missing or unknown:
        raise ValueError(f"distill train keys: missing {missing}, unknown {unknown}")
    check_distill(train, DISTILL_KEYS)
    return dict(train)


def train_distill(variant: Any, writer: Any) -> None:
    """The gate's ``train.phase`` of ``variant`` (a :class:`catanrl.config.Variant`) into ``writer``'s run."""
    cfg = variant.config
    runtime = dict(cfg.get("runtime") or {})
    strict = runtime.get("determinism", "fast") == "strict"
    train = check_train(cfg["train"])
    init = cfg.get("init_from") or {}
    from catanrl.checkpoint import init_parts

    if not init or set(init_parts(init)) != {"actor", "critic"}:
        raise ValueError("train.algo: distill starts from a network, actor and critic (init_from)")
    with determinism("strict" if strict else "fast"):
        if train["phase"] == "collect":
            _collect(cfg, writer, train, runtime, strict)
        else:
            _fit(cfg, writer, train, runtime, strict)


def _start(cfg: Mapping[str, Any], runtime: Mapping[str, Any], *, load: bool) -> tuple[Any, CatanModel, str, str]:
    """(env plan, the start network on its device, the device, the start's weights sha256)."""
    from catanrl.checkpoint import load_parts

    seed = int(cfg["seed"])
    env_cfg = cfg["env"]
    pool_seed = (env_cfg.get("map_pool") or {}).get("seed")
    plan = plan_env(env_cfg, pool_seed=derive_seed(seed, "map_pool") if pool_seed is None else int(pool_seed))
    seed_everything(derive_seed(seed, "init"))
    model = build_policy_model(cfg, plan.dims)
    check_bindable(model, plan)
    if getattr(model, "recurrent", False):
        raise ValueError("train.algo: distill evaluates single positions; a recurrent network trains with ppo")
    start = model
    if not load:  # a resumed fit: the start's weights, for the dataset check, from a second copy
        start = build_policy_model(cfg, plan.dims)
    load_parts(start, cfg["init_from"])
    sha = canon.weights_sha256(start.state_dict())
    device = resolve_device(runtime.get("device", "auto"))
    model.to(device).eval()
    return plan, model, device, sha


# ---------------------------------------------------------------- collect: self-play


@dataclass
class Kept:
    """A kept decision: a copy of the game, its row, and where it was in the game."""

    game: Any
    row: np.ndarray
    seat: int
    turn: int
    decision: int


def play_games(plan: Any, model: CatanModel, evaluator: BatchEvaluator, *, games: int, keep: int, seed: int,
               threads: int) -> tuple[list[list[Kept]], np.ndarray, np.ndarray, np.ndarray]:
    """``games`` games of raw self-play (each env's first; moves sampled from softmax(logits)), with
    ``keep`` of each game's decisions (rows with more than one legal action) kept uniformly by
    reservoir sampling. Returns (kept per game in decision order, winner seat per game or -1 for a
    truncated game, decisions per game, game seed per game)."""
    vec = make_vec(plan, model.levels, num_envs=games, seed=seed, threads=threads)
    b = vec.allocate()
    vec.reset(seed)
    players = int(vec.max_players)
    mask = slice(model.mask_offset, model.mask_offset + model.num_actions)
    moves = np.random.default_rng(derive_seed(seed, "moves"))
    picks = np.random.default_rng(derive_seed(seed, "picks"))
    kept: list[list[Kept]] = [[] for _ in range(games)]
    seen = np.zeros(games, dtype=np.int64)
    winner = np.full(games, -1, dtype=np.int64)
    seeds = np.array([vec.game_seed(env) for env in range(games)], dtype=np.uint64)
    done = np.zeros(games, dtype=bool)
    for _ in range(MAX_STEPS):
        if done.all():
            break
        acting = np.flatnonzero(b["acting"])
        rows = b["observations"][acting]
        logits, _ = evaluator(rows)
        b["actions"][acting] = np.argmax(logits + moves.gumbel(size=logits.shape), axis=1)  # a softmax sample
        decides = (rows[:, mask] > 0).sum(1) > 1
        for k, row in enumerate(acting):
            env = int(b["env_index"][row])
            if done[env] or not decides[k]:
                continue
            i = int(seen[env])
            seen[env] += 1
            slot = i if i < keep else int(picks.integers(0, i + 1))
            if slot < keep:
                game = vec.game(env)  # a copy
                item = Kept(game, rows[k].copy(), int(b["seat"][row]), int(game.num_turns), i)
                if i < keep:
                    kept[env].append(item)
                else:
                    kept[env][slot] = item
        vec.step()
        ended = (b["terminals"].reshape(-1, players)[:, 0] > 0) & ~done
        for env in np.flatnonzero(ended):
            rewards = b["rewards"][env * players : (env + 1) * players]
            winner[env] = int(np.argmax(rewards)) if rewards.max() > 0 else -1
        done |= ended
    else:
        raise RuntimeError(f"self-play did not finish {games} games in {MAX_STEPS} steps")
    return [sorted(items, key=lambda item: item.decision) for items in kept], winner, seen, seeds


# ---------------------------------------------------------------- collect: searches


def search_positions(pool: Any, evaluator: BatchEvaluator, positions: Sequence[tuple[int, Kept]], *, rows: int,
                     seed: int, mask: slice, searched: Callable[[int], None] | None = None,
                     ) -> tuple[dict[str, np.ndarray], int]:
    """Search every ``(game id, Kept)`` once, ``rows`` searches in lockstep; the dataset's arrays of
    them (all but the game outcome's) and the leaf rows evaluated. ``searched(n)`` hears of every
    finished wave of ``n`` positions."""
    out: dict[str, list] = {name: [] for name in ("obs", "start_value", "root_value", "search_value", "depth",
                                                  "max_depth", "game", "seat", "turn", "decision", "num_legal",
                                                  *EDGE_ARRAYS)}
    leaves = 0
    for first in range(0, len(positions), rows):
        wave = positions[first : first + rows]
        roots = RootOutputs(pool, mask)
        try:
            sids = []
            for game_id, item in wave:
                sids.append(pool.add(item.game, seed=derive_seed(seed, "search", game_id, item.decision)))
                roots.expect(sids[-1])
            leaves += run_pools([roots], evaluator, batch_rows=[len(wave)])
            for (game_id, item), sid in zip(wave, sids):
                legal, logits, value = roots.root(sid)
                if len(logits) != 1:
                    raise RuntimeError(f"search {sid}: {len(logits)} root worlds; the dataset keeps one (dets 0)")
                edges = np.flatnonzero(legal & (item.row[mask] > 0))
                engine = pool.policy_target(sid).astype(np.float32)
                if abs(float(engine[edges].sum()) - 1.0) > 1e-4:
                    raise RuntimeError(f"search {sid}: the engine's target has mass off the row's legal actions")
                stats = pool.stats(sid)
                for name, item_value in (
                    ("obs", item.row), ("start_value", roots.roots[sid][0][2]), ("root_value", value),
                    ("search_value", pool.value(sid)), ("depth", stats.mean_depth), ("max_depth", stats.max_depth),
                    ("game", game_id), ("seat", item.seat), ("turn", item.turn), ("decision", item.decision),
                    ("num_legal", len(edges)), ("actions", edges), ("prior_logits", logits[0][edges]),
                    ("visits", pool.visits(sid)[edges]), ("q", pool.q(sid)[edges]), ("engine", engine[edges]),
                ):
                    out[name].append(item_value)
            if searched is not None:
                searched(len(wave))
        finally:
            if pool.pending:
                pool.cancel()
            for sid in pool.ids:
                pool.remove(sid)
    dtypes = {"obs": np.float32, "start_value": np.float32, "root_value": np.float32, "search_value": np.float32,
              "depth": np.float32, "max_depth": np.int32, "game": np.int32, "seat": np.int8, "turn": np.int32,
              "decision": np.int32, "num_legal": np.int32, "actions": np.int16, "prior_logits": np.float32,
              "visits": np.uint32, "q": np.float64, "engine": np.float32}
    arrays = {}
    for name, values in out.items():
        if name in EDGE_ARRAYS:
            arrays[name] = (np.concatenate(values) if values else np.zeros(0)).astype(dtypes[name])
        else:
            arrays[name] = np.asarray(values, dtype=dtypes[name])
    return arrays, leaves


# ---------------------------------------------------------------- dataset files


def save_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """``arrays`` as an uncompressed ``.npz`` written atomically; its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return canon.sha256_file(path)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def merge_chunks(chunks: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The chunks' arrays, concatenated in order (per-position and per-action arrays alike)."""
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in (*POSITION_ARRAYS, *EDGE_ARRAYS)}


def offsets_of(num_legal: np.ndarray) -> np.ndarray:
    """Start of each position's legal actions in the per-action arrays, and the end: [N + 1]."""
    return np.concatenate([[0], np.cumsum(num_legal, dtype=np.int64)])


@dataclass
class Dataset:
    """A collect run's ``positions.npz`` and ``positions.json``."""

    arrays: dict[str, np.ndarray]
    meta: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, ref: str) -> Dataset:
        """The dataset of collect run ``ref`` (finished), its file checked against its sha256."""
        from catanrl.runs import USABLE, load_experiment

        run = load_experiment(ref)
        if run.status not in USABLE:
            raise RuntimeError(f"{ref} is {run.status}: its dataset is not complete")
        train = run.config["train"]
        if train.get("algo") != "distill" or train.get("phase") != "collect":
            raise RuntimeError(f"{ref} is not a distill collect run")
        meta = canon.read_json(run.path / META)
        path = run.path / DATASET
        sha = canon.sha256_file(path)
        if sha != meta["sha256"]:
            raise RuntimeError(f"{path} has sha256 {sha}, not the {meta['sha256']} its run recorded")
        return cls(load_arrays(path), meta, path)

    @property
    def offsets(self) -> np.ndarray:
        return offsets_of(self.arrays["num_legal"])

    def __len__(self) -> int:
        return len(self.arrays["num_legal"])


# ---------------------------------------------------------------- targets


def segment_softmax(logits: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Softmax of ``logits`` within each segment [offsets[i], offsets[i + 1]) (float64)."""
    starts = offsets[:-1]
    counts = np.diff(offsets)
    logits = logits.astype(np.float64)
    top = np.repeat(np.maximum.reduceat(logits, starts), counts)
    weights = np.exp(logits - top)
    return weights / np.repeat(np.add.reduceat(weights, starts), counts)


def policy_targets(arrays: Mapping[str, np.ndarray], train: Mapping[str, Any]) -> np.ndarray:
    """The fit's target per legal action of every position (float64, summing to 1 per position):
    the engine's (renormalized from float32) or :func:`gumbel_target` rebuilt with the train's
    sigma, then mixed with the start's policy at weight ``target_anchor``."""
    offsets = offsets_of(arrays["num_legal"])
    if train["policy_target"] == "engine":
        target = arrays["engine"].astype(np.float64)
        target /= np.repeat(np.add.reduceat(target, offsets[:-1]), np.diff(offsets))
    else:
        sigma = {"c_visit": float(train["target_c_visit"]), "c_scale": float(train["target_c_scale"]),
                 "rescale": bool(train["target_rescale"]),
                 "prior_temperature": float(train["target_prior_temperature"])}
        target = np.empty(int(offsets[-1]), dtype=np.float64)
        for i in range(len(offsets) - 1):
            span = slice(int(offsets[i]), int(offsets[i + 1]))
            k = span.stop - span.start
            target[span] = gumbel_target(arrays["prior_logits"][span], np.ones(k, dtype=bool), arrays["visits"][span],
                                         arrays["q"][span], float(arrays["root_value"][i]), **sigma)
    anchor = float(train["target_anchor"])
    if anchor:
        target = (1.0 - anchor) * target + anchor * segment_softmax(arrays["prior_logits"], offsets)
    return target


def dense(values: np.ndarray, arrays: Mapping[str, np.ndarray], num_actions: int, *, fill: float = 0.0,
          index: np.ndarray | None = None) -> np.ndarray:
    """Per-action ``values`` as float32 [n, num_actions] rows (``fill`` off the legal actions), for the
    positions ``index`` (all by default)."""
    offsets = offsets_of(arrays["num_legal"])
    index = np.arange(len(offsets) - 1) if index is None else np.asarray(index)
    counts = np.diff(offsets)[index]
    local = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    edges = np.repeat(offsets[:-1][index], counts) + local
    out = np.full((len(index), num_actions), fill, dtype=np.float32)
    out[np.repeat(np.arange(len(index)), counts), arrays["actions"][edges]] = values[edges]
    return out


def outcome_targets(winner: np.ndarray, seat: np.ndarray, players: int) -> np.ndarray:
    """Seat-relative outcomes [n, players] (slot k: the seat k places after the row's): 1 to the
    winner; a truncated game (winner -1) 1/players each, the eval convention."""
    out = np.zeros((len(winner), players), dtype=np.float32)
    won = winner >= 0
    out[np.flatnonzero(won), ((winner - seat) % players)[won]] = 1.0
    out[~won] = 1.0 / players
    return out


def target_stats(target: np.ndarray, arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """``target/*`` of the fit's targets against the start's policy at the same roots."""
    offsets = offsets_of(arrays["num_legal"])
    starts, counts = offsets[:-1], np.diff(offsets)
    start = segment_softmax(arrays["prior_logits"], offsets)
    entropy = -np.add.reduceat(np.where(target > 0, target * np.log(np.maximum(target, 1e-300)), 0.0), starts)
    top = np.maximum.reduceat(target, starts)
    start_arg = _segment_argmax(arrays["prior_logits"].astype(np.float64), offsets)
    target_arg = _segment_argmax(target, offsets)
    two = counts == 2
    return {"target/entropy": float(entropy.mean()), "target/top_ge_099": float((top >= 0.99).mean()),
            "target/overrule": float((start_arg != target_arg).mean()),
            "target/mass_on_start_argmax": float(target[start_arg].mean()),
            "target/top_ge_099_two_moves": float((top[two] >= 0.99).mean()) if two.any() else 0.0,
            "target/start_entropy": float(-np.add.reduceat(
                np.where(start > 0, start * np.log(np.maximum(start, 1e-300)), 0.0), starts).mean())}


def _segment_argmax(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Global index of each segment's first maximum."""
    starts, counts = offsets[:-1], np.diff(offsets)
    top = np.repeat(np.maximum.reduceat(values, starts), counts)
    hits = np.flatnonzero(values >= top)
    return hits[np.searchsorted(hits, starts)]


def holdout_games(games: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """The held-out game ids: a seeded ``fraction`` of the distinct games (at least one when
    fraction > 0 and there are two or more)."""
    ids = np.unique(games)
    count = round(fraction * len(ids))
    if fraction > 0 and len(ids) > 1:
        count = min(max(count, 1), len(ids) - 1)
    order = np.random.default_rng(derive_seed(seed, "holdout")).permutation(len(ids))
    return np.sort(ids[order[:count]])


# ---------------------------------------------------------------- collect


def _collect(cfg: Mapping[str, Any], writer: Any, train: Mapping[str, Any], runtime: Mapping[str, Any],
             strict: bool) -> None:
    seed = int(cfg["seed"])
    threads = int(runtime.get("threads", 1))
    plan, model, device, start_sha = _start(cfg, runtime, load=True)
    levels = model.levels
    evaluator = BatchEvaluator(model, device=device, infer_rows=int(train["search_rows"]),
                               mode="strict" if strict else "fast", kernels=train["kernels"])
    players = int(model.num_players)
    params = {"sims": int(train["sims"]), "oracle": True, "root": train["root"], "c_puct": train["c_puct"],
              "gumbel_m": train["gumbel_m"], "noise": False}
    pool = make_pool(plan.dims, levels["level"], search_config(params, max_players=players, threads=threads),
                     critic_level=levels["critic_level"])
    mask = slice(model.mask_offset, model.mask_offset + model.num_actions)
    per_chunk = int(cfg["env"]["num_envs"])
    total = int(train["games"])
    chunks = (total + per_chunk - 1) // per_chunk
    folder = writer.path / CHUNKS
    spec = {"format": FORMAT, "seed": seed, "env": canon.digest(cfg["env"]), "map_pool": plan.pool["hash"],
            "level": levels["level"], "critic_level": levels["critic_level"], "start": start_sha,
            "games_per_chunk": per_chunk, **{key: train[key] for key in COLLECT_KEYS}}

    state = writer.load_trainer_state() if writer.mode == "resume" else None
    done: dict[int, dict] = {}
    if state is not None:
        if state.get("spec") != spec:
            raise RuntimeError("the collect run's spec changed since it started; refusing to resume")
        merged = state.get("dataset")
        if merged and (writer.path / DATASET).exists() and canon.sha256_file(writer.path / DATASET) == merged["sha256"]:
            writer.finish("finished")  # merged and committed before a crash: nothing is left to do
            return
        for entry in state["chunks"]:
            path = folder / f"c{entry['index']:05d}.npz"
            if path.exists() and canon.sha256_file(path) == entry["sha256"]:
                done[int(entry["index"])] = entry
    totals = {"positions": sum(e["positions"] for e in done.values()), "leaves": sum(e["leaves"] for e in done.values())}

    def commit(extra: Mapping[str, Any] | None = None) -> None:
        writer.save_trainer_state({"epoch": len(done), "spec": spec, "chunks": [done[i] for i in sorted(done)],
                                   **(extra or {})})

    def finish_dataset() -> dict[str, Any]:
        """Merge the chunks into the dataset, commit it, and remove the chunk files."""
        merged = merge_chunks([load_arrays(folder / f"c{i:05d}.npz") for i in range(chunks)])
        path = writer.path / DATASET
        sha = save_arrays(path, merged)
        meta = {"format": FORMAT, "run": writer.ref, "spec": spec, "positions": len(merged["num_legal"]),
                "games": len(np.unique(merged["game"])), "edges": len(merged["actions"]),
                "obs_size": int(merged["obs"].shape[1]), "num_actions": int(model.num_actions),
                "max_players": players, "chunks": [done[i] for i in range(chunks)], "sha256": sha,
                "bytes": path.stat().st_size, "leaves": totals["leaves"]}
        canon.atomic_write_json(writer.path / META, meta)
        commit({"dataset": meta})
        writer.record_runtime({"dataset": {"path": str(path), "sha256": sha}})
        for i in range(chunks):  # merged, hashed and committed: the chunk files were its scratch
            (folder / f"c{i:05d}.npz").unlink(missing_ok=True)
        print(f"{writer.ref}: dataset {path} ({meta['positions']} positions of {meta['games']} games), "
              f"sha256 {sha}", flush=True)
        return {"data/positions": meta["positions"], "data/games": meta["games"], "data/bytes": meta["bytes"],
                "data/leaves": meta["leaves"], "data/sha256": sha}

    status = "finished"
    for chunk in range(chunks):
        if chunk in done:
            continue
        if writer.should_stop():
            status = writer.stop_reason or "interrupted"
            break
        games = min(per_chunk, total - chunk * per_chunk)
        first_id = chunk * per_chunk
        start = time.time()
        kept, winner, decisions, game_seeds = play_games(
            plan, model, evaluator, games=games, keep=int(train["positions_per_game"]),
            seed=derive_seed(seed, "collect", chunk), threads=threads)
        played = time.time() - start
        positions = [(first_id + env, item) for env, items in enumerate(kept) for item in items]
        # ~9 min a chunk in az-gate-2p (10,240 positions at 2,048 sims): a line every 30 s meanwhile
        with Progress(f"{writer.ref}: chunk {chunk + 1}/{chunks}: search", len(positions), unit="positions") as progress:
            arrays, leaves = search_positions(pool, evaluator, positions, rows=int(train["search_rows"]), seed=seed,
                                              mask=mask, searched=progress.add)
        searched = time.time() - start - played
        local = arrays["game"] - first_id
        arrays.update({"decisions": decisions[local].astype(np.int32), "winner": winner[local].astype(np.int8),
                       "game_seed": game_seeds[local]})
        sha = save_arrays(folder / f"c{chunk:05d}.npz", arrays)
        done[chunk] = {"index": chunk, "games": games, "positions": len(arrays["num_legal"]), "leaves": int(leaves),
                       "sha256": sha}
        totals["positions"] += len(arrays["num_legal"])
        totals["leaves"] += int(leaves)
        commit()
        print(f"{writer.ref}: chunk {chunk + 1}/{chunks}: {games} games, {len(arrays['num_legal'])} positions, "
              f"self-play {played:.0f} s, search {searched:.0f} s ({leaves / max(searched, 1e-9):.0f} leaves/s)",
              flush=True)
        engine = policy_targets(arrays, {**train, "policy_target": "engine", "target_anchor": 0.0})
        logs = {"epoch": len(done), "agent_steps": totals["positions"], "collect/chunk": chunk,
                "collect/games": games, "collect/positions": len(arrays["num_legal"]),
                "collect/decisions_per_game": float(decisions.mean()),
                "collect/truncated": float((winner < 0).mean()), "collect/seat0_wins": float((winner == 0).mean()),
                "collect/selfplay_s": played, "collect/search_s": searched,
                "collect/leaves_per_s": leaves / max(searched, 1e-9),
                "collect/search_depth": float(arrays["depth"].mean()),
                "collect/search_max_depth": float(arrays["max_depth"].mean()),
                "collect/legal_actions": float(arrays["num_legal"].mean()),
                **target_stats(engine, arrays)}
        if len(done) == chunks:
            logs.update(finish_dataset())
        writer.log(logs, totals["positions"])
    if status == "finished" and len(done) == chunks and not (writer.path / DATASET).exists():
        writer.log({"epoch": len(done), "agent_steps": totals["positions"], **finish_dataset()}, totals["positions"])
    writer.finish(status)


# ---------------------------------------------------------------- fit


@dataclass
class Probe:
    """Positions the fit is scored on, with the start's outputs there and what they are scored against."""

    index: np.ndarray
    obs: torch.Tensor
    target: torch.Tensor
    start_logp: torch.Tensor
    start_value: torch.Tensor
    outcome: torch.Tensor
    phases: dict[str, torch.Tensor]


def make_probe(arrays: Mapping[str, np.ndarray], target: np.ndarray, index: np.ndarray, model: CatanModel,
               *, phases: bool) -> Probe:
    num_actions = int(model.num_actions)
    start_logits = dense(arrays["prior_logits"], arrays, num_actions, fill=MASKED, index=index)
    obs = arrays["obs"][index]
    return Probe(
        index=index, obs=torch.from_numpy(obs),
        target=torch.from_numpy(dense(target, arrays, num_actions, index=index)),
        start_logp=F.log_softmax(torch.from_numpy(start_logits), dim=1),
        start_value=torch.from_numpy(arrays["start_value"][index]),
        outcome=torch.from_numpy(outcome_targets(arrays["winner"][index].astype(np.int64),
                                                 arrays["seat"][index].astype(np.int64), model.num_players)),
        phases={name: torch.from_numpy(rows) for name, rows in phases_of(model, obs).items()} if phases else {},
    )


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    q = p.clamp(1e-12, 1 - 1e-12)
    return -(p * q.log() + (1 - p) * (1 - q).log())


def score(model: CatanModel, probe: Probe, device: str, prefix: str) -> dict[str, float]:
    """``<prefix>/*`` of the network on ``probe`` (see the module doc)."""
    if not len(probe.index):
        return {}
    model.eval()
    per_row: dict[str, list[torch.Tensor]] = {k: [] for k in ("ce", "kl_start", "entropy", "agree_start",
                                                              "agree_target")}
    value = torch.zeros(4, dtype=torch.float64)  # drift, KL to the start, BCE to the outcome, seats
    with torch.no_grad():
        for first in range(0, len(probe.index), PROBE_CHUNK):
            part = slice(first, first + PROBE_CHUNK)
            logits, value_logits, present = model.heads(probe.obs[part].to(device))
            logp = F.log_softmax(logits.float(), dim=1).cpu()
            value_logits, present = value_logits.float().cpu(), present.float().cpu()
            start_logp, target = probe.start_logp[part], probe.target[part]
            legal = start_logp > -1e8
            safe = torch.where(legal, logp, torch.zeros_like(logp))
            per_row["ce"].append(-(target * safe).sum(1))
            per_row["kl_start"].append((start_logp.exp() * torch.where(legal, start_logp - logp,
                                                                         torch.zeros_like(logp))).sum(1))
            per_row["entropy"].append(-(logp.exp() * safe).sum(1))
            argmax = torch.where(legal, logp, torch.full_like(logp, -torch.inf)).argmax(1)
            per_row["agree_start"].append((argmax == start_logp.argmax(1)).float())
            per_row["agree_target"].append((argmax == target.argmax(1)).float())
            start_value, outcome = probe.start_value[part], probe.outcome[part]
            p = torch.sigmoid(value_logits)
            value[0] += float(((p - start_value).abs() * present).sum())
            value[1] += float(((F.binary_cross_entropy_with_logits(value_logits, start_value, reduction="none")
                                - _binary_entropy(start_value)) * present).sum())
            value[2] += float((F.binary_cross_entropy_with_logits(value_logits, outcome, reduction="none")
                               * present).sum())
            value[3] += float(present.sum())
    rows = {k: torch.cat(v) for k, v in per_row.items()}
    seats = max(float(value[3]), 1.0)
    logs = {f"{prefix}/{k}": float(v.mean()) for k, v in rows.items()}
    logs.update({f"{prefix}/value_drift": float(value[0]) / seats, f"{prefix}/value_kl": float(value[1]) / seats,
                 f"{prefix}/value_bce_outcome": float(value[2]) / seats, f"{prefix}/rows": len(probe.index)})
    for name, flags in probe.phases.items():
        if bool(flags.any()):
            logs.update({f"{prefix}/kl_start_{name}": float(rows["kl_start"][flags].mean()),
                         f"{prefix}/agree_start_{name}": float(rows["agree_start"][flags].mean()),
                         f"{prefix}/rows_{name}": int(flags.sum())})
    return logs


def fit_step(model: CatanModel, optimizer: torch.optim.Optimizer, obs: torch.Tensor, target: torch.Tensor,
             start_value: torch.Tensor, train: Mapping[str, Any]) -> dict[str, float]:
    """One AdamW step on a minibatch: cross-entropy to ``target`` plus ``value_coef`` times the value
    head's BCE to ``start_value`` per seated slot; the step's statistics."""
    model.train()
    logits, value_logits, present = model.heads(obs)
    logp = F.log_softmax(logits.float(), dim=1)
    policy_loss = -(target * logp).sum(1).mean()
    present = present.float()
    value_bce = F.binary_cross_entropy_with_logits(value_logits.float(), start_value, reduction="none")
    value_loss = (value_bce * present).sum() / present.sum().clamp_min(1.0)
    loss = policy_loss + float(train["value_coef"]) * value_loss
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["max_grad_norm"]))
    optimizer.step()
    with torch.no_grad():
        value_kl = ((value_bce - _binary_entropy(start_value)) * present).sum() / present.sum().clamp_min(1.0)
    model.eval()
    return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "value_kl": value_kl.item(),
            "grad_norm": grad_norm.item()}


def _fit(cfg: Mapping[str, Any], writer: Any, train: Mapping[str, Any], runtime: Mapping[str, Any],
         strict: bool) -> None:
    seed = int(cfg["seed"])
    data = Dataset.load(train["dataset"])
    fresh = starts_fresh(writer)
    _, model, device, start_sha = _start(cfg, runtime, load=fresh)
    meta = data.meta
    if meta["spec"]["start"] != start_sha:
        raise RuntimeError(f"{train['dataset']} searched another start ({meta['spec']['start'][:12]}); this fit "
                           f"starts from {start_sha[:12]}, and its anchors are the dataset's start")
    if meta["spec"]["level"] != model.levels["level"] or meta["obs_size"] != model.obs_size \
            or meta["num_actions"] != model.num_actions or meta["max_players"] != model.num_players:
        raise RuntimeError(f"{train['dataset']}'s rows are not this network's")
    arrays = data.arrays
    target = policy_targets(arrays, train)
    held_games = holdout_games(arrays["game"], float(train["holdout_fraction"]), seed)
    held = np.isin(arrays["game"], held_games)
    held_index, train_index = np.flatnonzero(held), np.flatnonzero(~held)
    if not len(train_index):
        raise RuntimeError("no training positions are left after the held-out games")
    probe_rng = np.random.default_rng(derive_seed(seed, "probe"))
    probe_count = min(len(train_index), max(len(held_index), 1))
    train_probe = make_probe(arrays, target, np.sort(probe_rng.choice(train_index, probe_count, replace=False)),
                             model, phases=False)
    held_probe = make_probe(arrays, target, held_index, model, phases=True)
    num_actions = int(model.num_actions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]),
                                  weight_decay=float(train["weight_decay"]))
    batch = int(train["minibatch_size"])
    updates = int(train["updates"])
    update = 0

    def save() -> None:
        writer.save_checkpoint(model, epoch=update, agent_steps=update * batch)
        writer.save_trainer_state({"epoch": update, "start": start_sha, "dataset": meta["sha256"],
                                   "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                   "optimizer": optimizer.state_dict()})

    def log(extra: Mapping[str, Any], sums: Mapping[str, float], seconds: float) -> None:
        logs = {"epoch": update, "agent_steps": update * batch, "updates": update, "samples": update * batch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{f"train/{k}": v for k, v in sums.items()}, **extra,
                **score(model, train_probe, device, "probe"), **score(model, held_probe, device, "heldout"),
                "throughput/fit_s": seconds}
        writer.log(logs, update * batch)

    state = writer.load_trainer_state() if writer.mode == "resume" else None
    if state is not None and not fresh:
        if state.get("start") != start_sha or state.get("dataset") != meta["sha256"]:
            raise RuntimeError("the fit's start or dataset changed since it started; refusing to resume")
        model.load_state_dict(state["model"])
        model.to(device).eval()
        optimizer.load_state_dict(state["optimizer"])
        update = int(state["epoch"])
    else:
        save()
        writer.record_runtime({"dataset": {"run": train["dataset"], "sha256": meta["sha256"]}})
        stats = target_stats(target, arrays)
        log({**stats, "data/positions": len(data), "data/games": meta["games"],
             "data/train_positions": len(train_index), "data/heldout_positions": len(held_index),
             "data/heldout_games": len(held_games), "data/sha256": meta["sha256"]}, {}, 0.0)
    status, saved = "finished", update
    sums: dict[str, float] = {}
    count, start = 0, time.time()
    while update < updates:
        if writer.should_stop():
            status = writer.stop_reason or "interrupted"
            break
        rng = np.random.default_rng(derive_seed(seed, "minibatch", update))
        rows = np.sort(train_index[rng.integers(0, len(train_index), batch)])
        step = fit_step(model, optimizer, torch.from_numpy(arrays["obs"][rows]).to(device),
                        torch.from_numpy(dense(target, arrays, num_actions, index=rows)).to(device),
                        torch.from_numpy(arrays["start_value"][rows]).to(device), train)
        update += 1
        count += 1
        for key, value in step.items():
            sums[key] = sums.get(key, 0.0) + value
        if update % int(train["log_interval"]) == 0 or update == updates:
            log({}, {k: v / count for k, v in sums.items()}, time.time() - start)
            sums, count, start = {}, 0, time.time()
        if update % int(train["checkpoint_interval"]) == 0:
            save()
            saved = update
    if saved != update:
        save()
    writer.finish(status)
