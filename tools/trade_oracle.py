"""The critic-based trade oracle: is domestic trading worth an action space?

Nothing in the action space changes here. The question is whether, under the
critics we already have, there are positions in which a resource trade
between the seat to move and an opponent leaves *both* of them better off;
if there hardly are, a trade action would have nothing to learn. For one
network (``--run``):

1. **Positions.** ``--positions`` self-play decisions, sampled by stepping a
   ``cppanatron.VecEnv`` with the network's own policy (sampled, as in
   training) and cloning the game at a random subset of its PLAY_TURN
   decisions after the roll, the moments a trade could be proposed.
2. **Trades.** Between the mover and each opponent, every 1-for-1, 2-for-1
   and 1-for-2 that both hands can pay (give and get never share a kind).
   Each is applied to a clone through the engine's own OFFER_TRADE,
   ACCEPT_TRADE / REJECT_TRADE and CONFIRM_TRADE, so the post-trade state is
   exactly what the engine would produce.
3. **Values.** The post-trade position is encoded from the mover's seat at
   the network's level and the value head's row gives every seat's win
   probability (slot 0 the mover, slot k the seat k places after it); the
   gains are the differences from the pre-trade row. Only the mover's row is
   read: the critic trains on acting rows alone (DESIGN 1.4: a waiting seat's
   value is the acting row's slot for it), and a waiting seat's own row, with
   ``acting`` 0 and a one-action mask, is off its distribution (such reads
   disagreed with the acting row's by 4-6 pp on average when tried). At
   private level the row shows an opponent's hand only as a size, so a 1:1
   trade changes it through the mover's hand and action mask alone and the
   partner's "gain" is how the critic redistributes win probability among
   seats it cannot tell apart by hand; 2:1 and 1:2 trades also move the
   partner's hand size. The full-level 2p network sees both hands.
4. **Noise.** The headline, positions with a trade both parties gain at
   least m pp from, is a max over tens of critic differences per position,
   so it comes with a floor: the conservation residual, a trade's change in
   the row's sum of win probabilities (zero for a consistent critic), put
   through the same max-over-trades statistic; a margin the residual reaches
   in most positions is one the critic cannot resolve. It is a pessimistic
   floor (independent per-seat errors of size s add up to a residual of
   s * sqrt(P)), but a critic that kept its sum while misjudging the split
   would be noisier than it. Two more readings: an independence reference,
   the fraction with each trade's mover gain paired with another trade's
   partner gain of the same position (what "mutual" would be if a trade's
   effect on the partner were unrelated to its effect on the mover; a real
   fraction far below it means the critic sees trades as near zero-sum
   between the two parties), and 95% bootstrap intervals that resample
   self-play games, since the positions of one game are correlated.

Output: ``report/data/trade-oracle-<run>.json`` and a printed summary. In a
2-player game the two seats' win probabilities sum to about 1, so a trade can
hardly raise both: 2p is the control, 3-4p the result. ``--game`` runs the
network zero-shot on another game; the output says so, and is not comparable
with a network trained on that game.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cppanatron as cpp
import numpy as np
import torch

from catanrl.agents.trade import (
    SHAPE_NAMES,
    Trade,
    apply_trade,
    enumerate_trades,
    trade_by_hands,  # noqa: F401  (the reference route tests/test_trade_oracle.py reads from here)
)
from catanrl.agents.trade import tradeable as _tradeable

ROOT = Path(__file__).resolve().parents[1]
#: Gains, in percentage points of win probability, at which a trade counts as beneficial.
MARGINS = (0.5, 1.0, 2.0, 5.0)
#: The margin the shape and resource breakdowns use.
SHAPE_MARGIN = 1.0


# ---------------------------------------------------------------- positions


@dataclass
class Position:
    """A self-play decision (a clone of the game) and its public phase information."""

    game: cpp.Game
    mover: int
    #: The engine's turn counters: completed regular turns, and turns including setup.
    completed_turns: int
    turn: int
    vps: list[int]
    hand_sizes: list[int]
    #: Which self-play game it came from (env and episode), to see correlated positions.
    game_id: str


def sample_positions(model: Any, dims: dict, level: str, game_cfg: Any, *, count: int, seed: int,
                     num_envs: int = 64, threads: int = 4, keep_rate: float = 0.03, map_pool: int = 256,
                     turn_limit: int = 1000, log: Callable[[str], None] | None = None) -> tuple[list[Position], dict]:
    """``count`` positions from self-play of ``model`` on ``game_cfg``: every acting
    decision is kept with probability ``keep_rate`` when it is a tradeable moment.
    Moves are sampled from the policy, as in training, so the positions are the
    critic's own distribution. Returns the positions and the sampling statistics."""
    from catanrl.envs.encoding import encoder_dims, info_level

    rng = np.random.default_rng(seed)
    maps = [cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(game_cfg.map_spec(int(s))))
            for s in rng.integers(0, 2**31 - 1, size=map_pool)]
    scenario = cpp.Scenario(num_players=game_cfg.num_players, rules=game_cfg.cppanatron_rules(), maps=maps,
                            turn_limit=turn_limit)
    vec = cpp.VecEnv([scenario], num_envs=num_envs, dims=encoder_dims(dims), level=info_level(level), seed=seed,
                     num_threads=threads)
    buffers = vec.allocate()
    vec.reset(seed)
    generator = torch.Generator().manual_seed(seed)
    episode = np.zeros(num_envs, dtype=np.int64)
    positions: list[Position] = []
    steps = decisions = 0
    started = time.time()
    while len(positions) < count:
        acting = np.flatnonzero(buffers["acting"])
        if len(acting):
            obs = torch.from_numpy(buffers["observations"][acting])
            with torch.no_grad():
                logits, _ = model(obs)
            probs = torch.softmax(logits.float(), dim=1)
            choice = torch.multinomial(probs, 1, generator=generator).squeeze(1).numpy()
            buffers["actions"][acting] = choice.astype(np.int32)
            decisions += len(acting)
            for row in acting[rng.random(len(acting)) < keep_rate]:
                env, seat = int(buffers["env_index"][row]), int(buffers["seat"][row])
                game = vec.game(env)
                if _tradeable(game, seat):
                    positions.append(Position(
                        game=game, mover=seat, completed_turns=int(game.completed_turns), turn=int(game.num_turns),
                        vps=[int(p.victory_points) for p in game.players],
                        hand_sizes=[int(sum(p.resources)) for p in game.players],
                        game_id=f"{env}.{episode[env]}"))
                    if log and len(positions) % 500 == 0:
                        log(f"  {len(positions)} positions, {steps} steps, {time.time() - started:.0f} s")
        vec.step()
        steps += 1
        ended = (buffers["terminals"] | buffers["truncations"]).astype(bool)
        episode[np.unique(buffers["env_index"][ended])] += 1
    stats = vec.pop_stats()
    return positions[:count], {
        "steps": steps, "decisions": decisions, "games_finished": int(stats["episodes"]),
        "games_started": int(num_envs + episode.sum()), "mean_turns": float(stats["length"]),
        "truncated_rate": float(stats["truncated"]), "seat_win": [float(w) for w in stats["seat_win"]],
        "seconds": time.time() - started,
    }


# ---------------------------------------------------------------- values


def make_critic(model: Any, *, batch_rows: int = 256) -> Callable[[np.ndarray], np.ndarray]:
    """Rows -> every seat's win probability [rows, P], in batches on the model's device."""
    def critic(rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), int(model.num_players)), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(rows), batch_rows):
                _, values = model(torch.from_numpy(rows[start:start + batch_rows]))
                out[start:start + batch_rows] = values.double().cpu().numpy()
        return out
    return critic


TRADE_FIELDS = ("pos", "partner_slot", "shape", "mover_gain", "partner_gain", "others_gain")
#: The gain columns: the mover's, the partner's and the remaining seats' summed, all from the mover's row.
GAIN_FIELDS = ("mover_gain", "partner_gain", "others_gain")


def evaluate_positions(positions: Sequence[Position], encoder: cpp.GraphEncoder,
                       critic: Callable[[np.ndarray], np.ndarray], *, flush_rows: int = 2048,
                       log: Callable[[str], None] | None = None) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Every trade of every position, valued. Per position the rows are the mover's
    pre-trade row and one post-trade row per trade, all from the mover's seat (the
    only row shape the critic trained on); the critic runs once per ``flush_rows``.

    Returns the trades as columns (:data:`TRADE_FIELDS` plus ``label``; gains are
    probability differences) and one metadata dict per position."""
    size = int(encoder.size)
    columns: dict[str, list] = {name: [] for name in (*TRADE_FIELDS, "label")}
    meta: list[dict] = []
    pending: list[tuple[int, list[tuple[int, int, Trade]], np.ndarray]] = []
    pending_rows = 0
    started = time.time()

    def flush() -> None:
        nonlocal pending, pending_rows
        if not pending:
            return
        values = critic(np.concatenate([rows for *_, rows in pending]))
        offset = 0
        for index, trades, rows in pending:
            _record(positions[index], index, trades, values[offset:offset + len(rows)], columns, meta)
            offset += len(rows)
        pending, pending_rows = [], 0

    for index, position in enumerate(positions):
        game, mover = position.game, position.mover
        n = game.num_players
        hands = [list(p.resources) for p in game.players]
        trades = [(seat, (seat - mover) % n, trade) for seat in range(n) if seat != mover
                  for trade in enumerate_trades(hands[mover], hands[seat])]
        rows = np.empty((1 + len(trades), size), dtype=np.float32)
        encoder.encode_into(game, mover, rows[0])
        for i, (seat, _, trade) in enumerate(trades):
            encoder.encode_into(apply_trade(game, mover, seat, trade), mover, rows[1 + i])
        pending.append((index, trades, rows))
        pending_rows += len(rows)
        if pending_rows >= flush_rows:
            flush()
            if log and (index + 1) % 500 < len(pending) + 1:
                log(f"  {index + 1}/{len(positions)} positions valued, {len(columns['pos'])} trades, "
                    f"{time.time() - started:.0f} s")
    flush()
    out = {name: np.asarray(values) for name, values in columns.items()}
    for name in ("pos", "partner_slot", "shape"):
        out[name] = out[name].astype(np.int64).reshape(-1)
    for name in GAIN_FIELDS:
        out[name] = out[name].astype(np.float64).reshape(-1)
    out["label"] = out["label"].astype(object).reshape(-1)
    return out, meta


def _record(position: Position, index: int, trades, values: np.ndarray, columns: dict, meta: list) -> None:
    n = position.game.num_players
    pre = values[0]
    for i, (_, slot, trade) in enumerate(trades):
        gain = values[1 + i][:n] - pre[:n]
        columns["pos"].append(index)
        columns["partner_slot"].append(slot)
        columns["shape"].append(SHAPE_NAMES.index(trade.shape))
        columns["label"].append(trade.label)
        columns["mover_gain"].append(gain[0])
        columns["partner_gain"].append(gain[slot])
        columns["others_gain"].append(float(gain.sum() - gain[0] - gain[slot]))
    meta.append({"mover": position.mover, "completed_turns": position.completed_turns, "turn": position.turn,
                 "vps": position.vps, "hand_sizes": position.hand_sizes, "game_id": position.game_id,
                 "n_trades": len(trades), "mover_value": float(pre[0]), "value_sum": float(pre[:n].sum())})


# ---------------------------------------------------------------- summary


def _best_per_position(pos: np.ndarray, score: np.ndarray, count: int) -> np.ndarray:
    """Index of the highest-scoring trade of each position (-1 without one)."""
    out = np.full(count, -1, dtype=np.int64)
    if len(pos):
        order = np.lexsort((-score, pos))
        first = np.ones(len(order), dtype=bool)
        first[1:] = pos[order][1:] != pos[order][:-1]
        out[pos[order][first]] = order[first]
    return out


def _quantiles(x: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        return {}
    keys = ("p05", "p25", "p50", "p75", "p95")
    out = {k: round(float(v), 3) for k, v in zip(keys, np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95]))}
    out["mean"] = round(float(x.mean()), 3)
    return out


def _fraction(mask: np.ndarray, pos: np.ndarray, count: int) -> float:
    """Fraction of the ``count`` positions with at least one trade in ``mask``."""
    return round(len(np.unique(pos[mask])) / count, 4) if count else float("nan")


def _shuffled_within_positions(pos: np.ndarray, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``values`` permuted among the trades of each position (``pos`` groups them)."""
    grouped = np.argsort(pos, kind="stable")
    shuffled = np.lexsort((rng.random(len(pos)), pos))
    out = np.empty_like(values)
    out[grouped] = values[shuffled]
    return out


def _game_bootstrap_ci(per_position: dict[float, np.ndarray], game_index: np.ndarray, rng: np.random.Generator,
                       draws: int) -> dict[str, list[float]]:
    """95% interval of each margin's fraction of positions when whole self-play games
    are resampled with replacement: positions of one game are correlated, so a
    per-position bootstrap would be too narrow."""
    n_games = int(game_index.max()) + 1
    weights = np.stack([np.bincount(d, minlength=n_games) for d in rng.integers(0, n_games, size=(draws, n_games))])
    weights = weights[:, game_index].astype(np.float64)  # [draws, positions]: times each position is drawn
    total = weights.sum(axis=1)
    return {f"{m:g}": [round(float(q), 4) for q in np.quantile((weights * has).sum(axis=1) / total, [0.025, 0.975])]
            for m, has in per_position.items()}


def summarize(trades: dict[str, np.ndarray], meta: Sequence[dict], *, num_players: int,
              margins: Sequence[float] = MARGINS, bootstrap_draws: int = 200, permutations: int = 20,
              seed: int = 0) -> dict:
    """The report's numbers: the positions with a mutually beneficial trade at each
    margin, with a game-clustered bootstrap interval and the noise floors (the
    critic's conservation residual and the independence reference at the same margins),
    the best trade's gains, the shapes and resources of beneficial trades, the phase
    breakdown and what a selfish proposer would offer. Gains are in percentage points."""
    count = len(meta)
    pos = trades["pos"]
    mover, partner, others = (trades[name] * 100 for name in GAIN_FIELDS)
    n_trades = np.array([m["n_trades"] for m in meta], dtype=np.int64)
    turns = np.array([m["completed_turns"] for m in meta], dtype=np.float64)
    edges = np.quantile(turns, [1 / 3, 2 / 3]) if count else np.zeros(2)
    tertile = np.searchsorted(edges, turns, side="right")
    games, game_index = np.unique([m["game_id"] for m in meta], return_inverse=True)
    rng = np.random.default_rng(seed)

    mutual = {m: (mover >= m) & (partner >= m) for m in margins}
    smaller = np.minimum(mover, partner)
    best = _best_per_position(pos, smaller, count)
    has = best[best >= 0]
    selfish_best = _best_per_position(pos, mover, count)
    selfish = selfish_best[selfish_best >= 0]
    at = mutual[SHAPE_MARGIN]
    labels, counts = np.unique(trades["label"][at], return_counts=True) if at.any() else ([], [])
    top = sorted(zip(counts, labels), reverse=True)[:12]
    # Noise. The residual is a trade's change in the row's sum of win probabilities,
    # zero for a consistent critic; the independence reference pairs each trade's
    # mover gain with another trade's partner gain of the same position.
    residual = np.abs(mover + partner + others)
    independent = {f"{m:g}": [] for m in margins}
    for _ in range(permutations):
        shuffled = _shuffled_within_positions(pos, partner, rng)
        for m in margins:
            independent[f"{m:g}"].append(_fraction((mover >= m) & (shuffled >= m), pos, count))
    per_position = {}
    for m in margins:
        per_position[m] = np.zeros(count, dtype=bool)
        per_position[m][pos[mutual[m]]] = True

    out: dict[str, Any] = {
        "positions": count, "trades": len(pos),
        "positions_with_a_trade": int((n_trades > 0).sum()),
        "games": {"distinct": len(games),
                  "positions_per_game": {"mean": round(count / len(games), 2) if count else None,
                                         "max": int(np.bincount(game_index).max()) if count else None}},
        "trades_per_position": {"mean": round(float(n_trades.mean()), 2) if count else None,
                                "max": int(n_trades.max()) if count else None},
        "shape_base_rate": {name: round(float((trades["shape"] == i).mean()), 4) if len(pos) else None
                            for i, name in enumerate(SHAPE_NAMES)},
        "mover_value": _quantiles(np.array([m["mover_value"] for m in meta]) * 100),
        "value_sum": _quantiles(np.array([m["value_sum"] for m in meta])),
        "phase_edges_completed_turns": [round(float(e), 1) for e in edges],
        "phase_positions": [int((tertile == t).sum()) for t in range(3)],
        "positions_with_mutual_trade": {f"{m:g}": _fraction(mutual[m], pos, count) for m in margins},
        "positions_with_mutual_trade_ci95": _game_bootstrap_ci(per_position, game_index, rng, bootstrap_draws)
        if count and bootstrap_draws else {},
        "mutual_trade_rate": {f"{m:g}": round(float(mutual[m].mean()), 4) if len(pos) else None for m in margins},
        "noise": {
            "conservation_residual_pp": _quantiles(residual),
            "positions_with_residual_at_margin": {f"{m:g}": _fraction(residual >= m, pos, count) for m in margins},
            "independence_reference_positions_with_mutual_trade": {
                k: round(float(np.mean(v)), 4) if v else None for k, v in independent.items()},
        },
        "best_trade": {"mover_gain_pp": _quantiles(mover[has]), "partner_gain_pp": _quantiles(partner[has]),
                       "min_gain_pp": _quantiles(smaller[has])},
        "shapes_of_mutual_trades": {
            sname: round(float((trades["shape"][at] == i).mean()), 4) if at.any() else None
            for i, sname in enumerate(SHAPE_NAMES)},
        "top_mutual_trades": [{"trade": str(label), "count": int(c),
                               "share_of_enumerated": round(float(c / max(1, (trades["label"] == label).sum())), 4)}
                              for c, label in top],
        "by_phase": [{
            "tertile": t, "positions": int((tertile == t).sum()),
            "positions_with_mutual_trade": {
                f"{m:g}": _fraction(mutual[m] & (tertile[pos] == t), pos, int((tertile == t).sum()))
                for m in margins},
            "best_min_gain_pp_mean": round(float(smaller[has][tertile[pos[has]] == t].mean()), 3)
            if (tertile[pos[has]] == t).any() else None,
        } for t in range(3)],
        "selfish": {
            "trade_rate_good_for_mover_bad_for_partner": {
                f"{m:g}": round(float(((mover >= m) & (partner <= -m)).mean()), 4) if len(pos) else None
                for m in margins},
            "best_for_mover": {
                "mover_gain_pp": _quantiles(mover[selfish]), "partner_gain_pp": _quantiles(partner[selfish]),
                "partner_hurt": {f"{m:g}": round(float((partner[selfish] <= -m).mean()), 4) if len(selfish) else None
                                 for m in margins},
                "mutual": {f"{m:g}": round(float((partner[selfish] >= m).mean()), 4) if len(selfish) else None
                           for m in margins},
            },
        },
    }
    if num_players > 2:
        out["others_gain_pp_of_mutual_trades"] = _quantiles(others[at])
    return out


def print_summary(result: dict, file=sys.stdout) -> None:
    s = result["summary"]
    w = result.get("wall", {})
    margins = "/".join(f"{m:g}" for m in result["margins_pp"])

    def row(values: dict) -> str:
        return " / ".join("-" if v is None else f"{v:.3f}" for v in values.values())

    print(f"{result['run']} ({result['checkpoint']}; {result['level']}, {result['num_players']}p {result['game']}): "
          f"{s['positions']} positions from {s['games']['distinct']} self-play games "
          f"({result['sampling']['games_finished']} finished), "
          f"{s['trades']} trades ({s['trades_per_position']['mean']} per position; "
          f"{s['positions_with_a_trade']} positions have one); "
          f"sample {w.get('sample_s', 0):.0f} s, value {w.get('eval_s', 0):.0f} s", file=file)
    if result.get("zero_shot"):
        print(f"  ZERO-SHOT: the network trained on {result['trained_game']}; not comparable with a network trained "
              f"on {result['game']}", file=file)
    print(f"  critic: mover {s['mover_value'].get('mean')} pp, seats sum {s['value_sum'].get('mean')}; "
          f"phase edges (completed turns) {s['phase_edges_completed_turns']}", file=file)
    print(f"  positions with a mutually beneficial trade at {margins} pp: {row(s['positions_with_mutual_trade'])}",
          file=file)
    print("    95% CI (games resampled): " + " / ".join(f"[{lo:.3f}, {hi:.3f}]"
                                                        for lo, hi in s["positions_with_mutual_trade_ci95"].values()),
          file=file)
    noise = s["noise"]
    print(f"    noise floor: the critic's own residual reaches the margin in "
          f"{row(noise['positions_with_residual_at_margin'])} of positions (|residual| p50 "
          f"{noise['conservation_residual_pp'].get('p50')} p95 {noise['conservation_residual_pp'].get('p95')} pp); "
          f"with partner gains unrelated to mover gains it would be "
          f"{row(noise['independence_reference_positions_with_mutual_trade'])}", file=file)
    best = s["best_trade"]
    print(f"    best trade (max of the smaller gain): mover p50 {best['mover_gain_pp'].get('p50')} "
          f"p95 {best['mover_gain_pp'].get('p95')}; partner p50 {best['partner_gain_pp'].get('p50')} "
          f"p95 {best['partner_gain_pp'].get('p95')}", file=file)
    print(f"    shapes of mutual trades at {SHAPE_MARGIN:g} pp: {s['shapes_of_mutual_trades']} "
          f"(base {s['shape_base_rate']})", file=file)
    print("    top: " + ", ".join(f"{t['trade']} x{t['count']}" for t in s["top_mutual_trades"][:6]), file=file)
    print("    by phase (1 pp): " + ", ".join(f"T{p['tertile']} {p['positions_with_mutual_trade']['1']:.3f}"
                                             for p in s["by_phase"]), file=file)
    selfish = s["selfish"]
    print(f"    selfish: trades good for mover and bad for partner at 1 pp "
          f"{selfish['trade_rate_good_for_mover_bad_for_partner']['1']:.3f}; the mover's best trade hurts the "
          f"partner at 1 pp in {selfish['best_for_mover']['partner_hurt']['1']:.3f} of positions, "
          f"is mutual in {selfish['best_for_mover']['mutual']['1']:.3f}", file=file)
    if "others_gain_pp_of_mutual_trades" in s:
        print(f"    the other seats' gain over mutual trades: {s['others_gain_pp_of_mutual_trades']}", file=file)


# ---------------------------------------------------------------- CLI


def run_oracle(args: argparse.Namespace, log: Callable[[str], None] = print) -> dict:
    import catanrl
    from catanrl.envs.encoding import critic_level_of, make_encoder
    from catanrl.games import load_game

    torch.set_num_threads(int(args.torch_threads))
    run = catanrl.load_experiment(args.run)
    ckpt = run.checkpoint(args.which)
    level, critic_level = ckpt.levels["level"], ckpt.levels["critic_level"]
    if critic_level_of(level, critic_level) is not None:
        raise NotImplementedError("the oracle encodes one row per position; a critic at another level needs two")
    model = ckpt.model("all", "cpu")
    scenario = run.config["env"]["scenarios"][0]
    trained = load_game(scenario["game"])
    game_cfg = load_game(args.game) if args.game else trained
    zero_shot = game_cfg.name != trained.name
    turn_limit = int(scenario.get("turn_limit", 1000))
    log(f"{ckpt.ref}: {level} level, {game_cfg.name} ({game_cfg.num_players} players), dims {ckpt.dims}"
        + (f"; ZERO-SHOT, the network trained on {trained.name}" if zero_shot else ""))
    positions, sampling = sample_positions(model, ckpt.dims, level, game_cfg, count=args.positions, seed=args.seed,
                                           num_envs=args.num_envs, threads=args.threads, keep_rate=args.keep_rate,
                                           map_pool=args.map_pool, turn_limit=turn_limit, log=log)
    log(f"sampled {len(positions)} positions from {sampling['games_finished']} finished games in "
        f"{sampling['seconds']:.0f} s")
    started = time.time()
    encoder = make_encoder(ckpt.dims, level)
    trades, meta = evaluate_positions(positions, encoder, make_critic(model, batch_rows=args.batch_rows), log=log)
    eval_s = time.time() - started
    rows = sum(1 + m["n_trades"] for m in meta)
    log(f"valued {len(trades['pos'])} trades ({rows} critic rows) in {eval_s:.0f} s")
    summary = summarize(trades, meta, num_players=game_cfg.num_players, seed=args.seed)
    return {
        "format": "catanrl.trade-oracle", "run": run.ref, "checkpoint": str(ckpt.ref), "level": level,
        "game": game_cfg.name, "num_players": game_cfg.num_players, "dims": ckpt.dims,
        "trained_game": trained.name, "zero_shot": zero_shot,
        "args": {k: v for k, v in vars(args).items() if k != "out"},
        "sampling": sampling, "wall": {"sample_s": round(sampling["seconds"], 1), "eval_s": round(eval_s, 1)},
        "critic_rows": rows, "margins_pp": list(MARGINS), "summary": summary,
        "positions": meta,
    }


def default_out(run: str, game: str | None = None) -> Path:
    """``report/data/trade-oracle-<run>.json``, naming the game too when the network
    runs zero-shot on one it did not train on, so the two never share a file."""
    stem = f"trade-oracle-{run.replace('/', '-')}" + (f"-{game}" if game else "")
    return ROOT / "report" / "data" / f"{stem}.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="the run (experiment/variant) whose network judges the trades")
    parser.add_argument("--which", default="best", help="checkpoint selector (default best)")
    parser.add_argument("--game", default=None, help="game config name (default: the run's first scenario)")
    parser.add_argument("--positions", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=64, help="self-play games stepped in lockstep")
    parser.add_argument("--threads", type=int, default=4, help="VecEnv threads")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--keep-rate", type=float, default=0.03, help="chance a decision is kept as a position")
    parser.add_argument("--map-pool", type=int, default=256, help="distinct boards the self-play draws from")
    parser.add_argument("--batch-rows", type=int, default=256, help="rows per critic forward")
    parser.add_argument("--out", type=Path, default=None, help="JSON path (default report/data/trade-oracle-<run>.json)")
    args = parser.parse_args(argv)
    started = time.time()
    result = run_oracle(args)
    result["wall"]["total_s"] = round(time.time() - started, 1)
    out = args.out or default_out(args.run, result["game"] if result["zero_shot"] else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print_summary(result)
    print(f"wrote {out} ({result['wall']['total_s']:.0f} s)")


if __name__ == "__main__":
    main()
