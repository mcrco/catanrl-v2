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
   gains are the differences from the pre-trade row (``mover_row``). A
   private-level row shows an opponent's hand only as a size, so the
   partner's gain is also read from the partner's own row (``own_row``):
   each party judging the trade with what it can see, as in a negotiation.

Output: ``report/data/trade-oracle-<run>.json`` and a printed summary. In a
2-player game the two seats' win probabilities sum to about 1, so a trade can
hardly raise both: 2p is the control, 3-4p the result.
"""

from __future__ import annotations

import argparse
import itertools
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

ROOT = Path(__file__).resolve().parents[1]
#: Resource kinds in the engine's hand order.
RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
#: (cards given, cards received) of the trades enumerated.
SHAPES = ((1, 1), (2, 1), (1, 2))
SHAPE_NAMES = tuple(f"{give}:{get}" for give, get in SHAPES)
#: Gains, in percentage points of win probability, at which a trade counts as beneficial.
MARGINS = (0.5, 1.0, 2.0, 5.0)
#: The margin the shape and resource breakdowns use.
SHAPE_MARGIN = 1.0
Counts = tuple[int, int, int, int, int]


# ---------------------------------------------------------------- trades


@dataclass(frozen=True)
class Trade:
    """What the mover gives and gets, as counts per resource kind."""

    give: Counts
    get: Counts

    @property
    def shape(self) -> str:
        return f"{sum(self.give)}:{sum(self.get)}"

    @property
    def label(self) -> str:
        return f"{_side(self.give)}>{_side(self.get)}"


def _side(counts: Sequence[int]) -> str:
    return "+".join(f"{count if count > 1 else ''}{name}" for name, count in zip(RESOURCES, counts) if count)


def _bundles(size: int) -> list[Counts]:
    """Every multiset of ``size`` cards, as counts (5 singles, 15 pairs)."""
    out = []
    for combo in itertools.combinations_with_replacement(range(len(RESOURCES)), size):
        counts = [0] * len(RESOURCES)
        for kind in combo:
            counts[kind] += 1
        out.append(tuple(counts))
    return out


def _can_pay(hand: Sequence[int], counts: Sequence[int]) -> bool:
    return all(have >= need for have, need in zip(hand, counts))


def enumerate_trades(mover_hand: Sequence[int], partner_hand: Sequence[int]) -> list[Trade]:
    """Every 1:1, 2:1 and 1:2 trade both hands can pay, give and get disjoint in kind
    (giving wood and brick for wood is a gift of brick, not a trade): at most 120."""
    out = []
    for give_size, get_size in SHAPES:
        for give in _bundles(give_size):
            if not _can_pay(mover_hand, give):
                continue
            for get in _bundles(get_size):
                if _can_pay(partner_hand, get) and not any(a and b for a, b in zip(give, get)):
                    out.append(Trade(give, get))
    return out


def _tradeable(game: cpp.Game, seat: int) -> bool:
    """The moments a domestic trade could be proposed: the seat's own PLAY_TURN
    decisions after the roll (not while placing a road building card's roads)."""
    return (game.current_player_index == seat and game.current_turn_index == seat
            and game.current_prompt == cpp.ActionPrompt.PLAY_TURN and game.player(seat).has_rolled
            and game.free_roads_available == 0 and game.winning_color is None)


def apply_trade(game: cpp.Game, mover: int, partner: int, trade: Trade) -> cpp.Game:
    """A copy of ``game`` after ``trade``, through the engine's own actions: the mover
    offers, the partner accepts, every other seat rejects, the mover confirms the
    partner. The engine checks an offer's form, not the offerer's hand, and a
    confirmation debits blindly, so both hands are checked here."""
    if not _tradeable(game, mover):
        raise ValueError("a trade is proposed by the seat to move, during its turn, after the roll")
    if mover == partner:
        raise ValueError("a seat cannot trade with itself")
    if not _can_pay(game.players[mover].resources, trade.give):
        raise ValueError(f"the mover cannot pay {trade.label}")
    if not _can_pay(game.players[partner].resources, trade.get):
        raise ValueError(f"the partner cannot pay {trade.label}")
    post = game.copy()
    colors = list(post.colors)
    post.execute(cpp.Action(colors[mover], cpp.ActionType.OFFER_TRADE, (*trade.give, *trade.get)))
    while post.current_prompt == cpp.ActionPrompt.DECIDE_TRADE:
        kind = cpp.ActionType.ACCEPT_TRADE if post.current_player_index == partner else cpp.ActionType.REJECT_TRADE
        post.execute(next(a for a in post.playable_actions if a.action_type == kind))
    confirms = [a for a in post.playable_actions if a.action_type == cpp.ActionType.CONFIRM_TRADE]
    if len(confirms) != 1 or confirms[0].value[-1] != colors[partner]:
        raise RuntimeError(f"the engine did not offer to confirm {trade.label} with seat {partner}")
    post.execute(confirms[0])
    return post


def trade_by_hands(game: cpp.Game, mover: int, partner: int, trade: Trade) -> cpp.Game:
    """The reference: the same trade as two hand overwrites (``Game.set_resources``
    leaves the bank alone, as a domestic trade does). Tests compare the routes."""
    post = game.copy()
    for seat, out, inn in ((mover, trade.give, trade.get), (partner, trade.get, trade.give)):
        hand = [have - gone + got for have, gone, got in zip(post.players[seat].resources, out, inn)]
        post.set_resources(seat, hand)
    return post


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


TRADE_FIELDS = ("pos", "partner_slot", "shape", "mover_gain", "partner_gain_mover_row", "partner_gain_own_row",
                "others_gain")


def evaluate_positions(positions: Sequence[Position], encoder: cpp.GraphEncoder,
                       critic: Callable[[np.ndarray], np.ndarray], *, flush_rows: int = 2048,
                       log: Callable[[str], None] | None = None) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Every trade of every position, valued. Per position the rows are: the mover's
    pre-trade row, each partner's own pre-trade row, then per trade the mover's and
    the partner's post-trade rows; the critic runs once per ``flush_rows``.

    Returns the trades as columns (:data:`TRADE_FIELDS` plus ``label``; gains are
    probability differences) and one metadata dict per position."""
    size = int(encoder.size)
    columns: dict[str, list] = {name: [] for name in (*TRADE_FIELDS, "label")}
    meta: list[dict] = []
    pending: list[tuple[int, list[tuple[int, int]], list[tuple[int, int, Trade]], np.ndarray]] = []
    pending_rows = 0
    started = time.time()

    def flush() -> None:
        nonlocal pending, pending_rows
        if not pending:
            return
        values = critic(np.concatenate([rows for *_, rows in pending]))
        offset = 0
        for index, partners, trades, rows in pending:
            _record(positions[index], index, partners, trades, values[offset:offset + len(rows)], columns, meta)
            offset += len(rows)
        pending, pending_rows = [], 0

    for index, position in enumerate(positions):
        game, mover = position.game, position.mover
        n = game.num_players
        partners = [(seat, (seat - mover) % n) for seat in range(n) if seat != mover]
        hands = [list(p.resources) for p in game.players]
        trades = [(seat, slot, trade) for seat, slot in partners for trade in enumerate_trades(hands[mover], hands[seat])]
        rows = np.empty((1 + len(partners) + 2 * len(trades), size), dtype=np.float32)
        encoder.encode_into(game, mover, rows[0])
        for i, (seat, _) in enumerate(partners):
            encoder.encode_into(game, seat, rows[1 + i])
        base = 1 + len(partners)
        for i, (seat, _, trade) in enumerate(trades):
            post = apply_trade(game, mover, seat, trade)
            encoder.encode_into(post, mover, rows[base + 2 * i])
            encoder.encode_into(post, seat, rows[base + 2 * i + 1])
        pending.append((index, partners, trades, rows))
        pending_rows += len(rows)
        if pending_rows >= flush_rows:
            flush()
            if log and (index + 1) % 500 < len(pending) + 1:
                log(f"  {index + 1}/{len(positions)} positions valued, {len(columns['pos'])} trades, "
                    f"{time.time() - started:.0f} s")
    flush()
    out = {name: np.asarray(values) for name, values in columns.items()}
    out["pos"] = out["pos"].astype(np.int64).reshape(-1)
    out["partner_slot"] = out["partner_slot"].astype(np.int64).reshape(-1)
    out["shape"] = out["shape"].astype(np.int64).reshape(-1)
    for name in ("mover_gain", "partner_gain_mover_row", "partner_gain_own_row", "others_gain"):
        out[name] = out[name].astype(np.float64).reshape(-1)
    out["label"] = out["label"].astype(object).reshape(-1)
    return out, meta


def _record(position: Position, index: int, partners, trades, values: np.ndarray, columns: dict, meta: list) -> None:
    n = position.game.num_players
    pre = values[0]
    pre_own = {slot: values[1 + i][0] for i, (_, slot) in enumerate(partners)}
    base = 1 + len(partners)
    others = np.ones(n, dtype=bool)
    others[0] = False
    for i, (_, slot, trade) in enumerate(trades):
        post, post_own = values[base + 2 * i], values[base + 2 * i + 1][0]
        rest = others.copy()
        rest[slot] = False
        columns["pos"].append(index)
        columns["partner_slot"].append(slot)
        columns["shape"].append(SHAPE_NAMES.index(trade.shape))
        columns["label"].append(trade.label)
        columns["mover_gain"].append(post[0] - pre[0])
        columns["partner_gain_mover_row"].append(post[slot] - pre[slot])
        columns["partner_gain_own_row"].append(post_own - pre_own[slot])
        columns["others_gain"].append(float((post[:n] - pre[:n])[rest].sum()))
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


def summarize(trades: dict[str, np.ndarray], meta: Sequence[dict], *, num_players: int,
              margins: Sequence[float] = MARGINS) -> dict:
    """The report's numbers: per measure (``mover_row``: both gains from the mover's
    row; ``own_row``: the partner's from its own row) the positions with a mutually
    beneficial trade at each margin, the best trade's gains, the shapes and resources
    of beneficial trades, the phase breakdown and what a selfish proposer would offer.
    Gains are in percentage points."""
    count = len(meta)
    pos = trades["pos"]
    mover = trades["mover_gain"] * 100
    n_trades = np.array([m["n_trades"] for m in meta], dtype=np.int64)
    turns = np.array([m["completed_turns"] for m in meta], dtype=np.float64)
    edges = np.quantile(turns, [1 / 3, 2 / 3]) if count else np.zeros(2)
    tertile = np.searchsorted(edges, turns, side="right")
    out: dict[str, Any] = {
        "positions": count, "trades": len(pos),
        "positions_with_a_trade": int((n_trades > 0).sum()),
        "trades_per_position": {"mean": round(float(n_trades.mean()), 2) if count else None,
                                "max": int(n_trades.max()) if count else None},
        "shape_base_rate": {name: round(float((trades["shape"] == i).mean()), 4) if len(pos) else None
                            for i, name in enumerate(SHAPE_NAMES)},
        "mover_value": _quantiles(np.array([m["mover_value"] for m in meta]) * 100),
        "value_sum": _quantiles(np.array([m["value_sum"] for m in meta])),
        "phase_edges_completed_turns": [round(float(e), 1) for e in edges],
        "phase_positions": [int((tertile == t).sum()) for t in range(3)],
        "measures": {},
    }
    for name, partner in (("mover_row", trades["partner_gain_mover_row"] * 100),
                          ("own_row", trades["partner_gain_own_row"] * 100)):
        mutual = {m: (mover >= m) & (partner >= m) for m in margins}
        best = _best_per_position(pos, np.minimum(mover, partner), count)
        has = best[best >= 0]
        selfish_best = _best_per_position(pos, mover, count)
        selfish = selfish_best[selfish_best >= 0]
        at = mutual[SHAPE_MARGIN]
        labels, counts = np.unique(trades["label"][at], return_counts=True) if at.any() else ([], [])
        top = sorted(zip(counts, labels), reverse=True)[:12]
        measure = {
            "positions_with_mutual_trade": {f"{m:g}": _fraction(mutual[m], pos, count) for m in margins},
            "mutual_trade_rate": {f"{m:g}": round(float(mutual[m].mean()), 4) if len(pos) else None for m in margins},
            "best_trade": {"mover_gain_pp": _quantiles(mover[has]), "partner_gain_pp": _quantiles(partner[has]),
                           "min_gain_pp": _quantiles(np.minimum(mover, partner)[has])},
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
                "best_min_gain_pp_mean": round(float(np.minimum(mover, partner)[has][tertile[pos[has]] == t].mean()), 3)
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
        if name == "mover_row" and num_players > 2:
            measure["others_gain_pp_of_mutual_trades"] = _quantiles(trades["others_gain"][at] * 100)
        out["measures"][name] = measure
    return out


def print_summary(result: dict, file=sys.stdout) -> None:
    s = result["summary"]
    w = result.get("wall", {})
    print(f"{result['run']} ({result['checkpoint']}; {result['level']}, {result['num_players']}p {result['game']}): "
          f"{s['positions']} positions from {result['sampling']['games_finished']} finished games, "
          f"{s['trades']} trades ({s['trades_per_position']['mean']} per position; "
          f"{s['positions_with_a_trade']} positions have one); "
          f"sample {w.get('sample_s', 0):.0f} s, value {w.get('eval_s', 0):.0f} s", file=file)
    print(f"  critic: mover {s['mover_value'].get('mean')} pp, seats sum {s['value_sum'].get('mean')}; "
          f"phase edges (completed turns) {s['phase_edges_completed_turns']}", file=file)
    for name, m in s["measures"].items():
        print(f"  [{name}] positions with a mutually beneficial trade at 0.5/1/2/5 pp: "
              + " / ".join(f"{v:.3f}" for v in m["positions_with_mutual_trade"].values()), file=file)
        best = m["best_trade"]
        print(f"    best trade (max of the smaller gain): mover p50 {best['mover_gain_pp'].get('p50')} "
              f"p95 {best['mover_gain_pp'].get('p95')}; partner p50 {best['partner_gain_pp'].get('p50')} "
              f"p95 {best['partner_gain_pp'].get('p95')}", file=file)
        print(f"    shapes of mutual trades at {SHAPE_MARGIN:g} pp: {m['shapes_of_mutual_trades']} "
              f"(base {s['shape_base_rate']})", file=file)
        print("    top: " + ", ".join(f"{t['trade']} x{t['count']}" for t in m["top_mutual_trades"][:6]), file=file)
        print("    by phase (1 pp): " + ", ".join(f"T{p['tertile']} {p['positions_with_mutual_trade']['1']:.3f}"
                                                 for p in m["by_phase"]), file=file)
        selfish = m["selfish"]
        print(f"    selfish: trades good for mover and bad for partner at 1 pp "
              f"{selfish['trade_rate_good_for_mover_bad_for_partner']['1']:.3f}; the mover's best trade hurts the "
              f"partner at 1 pp in {selfish['best_for_mover']['partner_hurt']['1']:.3f} of positions, "
              f"is mutual in {selfish['best_for_mover']['mutual']['1']:.3f}", file=file)
        if "others_gain_pp_of_mutual_trades" in m:
            print(f"    the other seats' gain over mutual trades: {m['others_gain_pp_of_mutual_trades']}", file=file)


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
    game_cfg = load_game(args.game) if args.game else load_game(scenario["game"])
    turn_limit = int(scenario.get("turn_limit", 1000))
    log(f"{ckpt.ref}: {level} level, {game_cfg.name} ({game_cfg.num_players} players), dims {ckpt.dims}")
    positions, sampling = sample_positions(model, ckpt.dims, level, game_cfg, count=args.positions, seed=args.seed,
                                           num_envs=args.num_envs, threads=args.threads, keep_rate=args.keep_rate,
                                           map_pool=args.map_pool, turn_limit=turn_limit, log=log)
    log(f"sampled {len(positions)} positions from {sampling['games_finished']} finished games in "
        f"{sampling['seconds']:.0f} s")
    started = time.time()
    encoder = make_encoder(ckpt.dims, level)
    trades, meta = evaluate_positions(positions, encoder, make_critic(model, batch_rows=args.batch_rows), log=log)
    eval_s = time.time() - started
    rows = sum(1 + (game_cfg.num_players - 1) + 2 * m["n_trades"] for m in meta)
    log(f"valued {len(trades['pos'])} trades ({rows} critic rows) in {eval_s:.0f} s")
    summary = summarize(trades, meta, num_players=game_cfg.num_players)
    return {
        "format": "catanrl.trade-oracle", "run": run.ref, "checkpoint": str(ckpt.ref), "level": level,
        "game": game_cfg.name, "num_players": game_cfg.num_players, "dims": ckpt.dims,
        "args": {k: v for k, v in vars(args).items() if k != "out"},
        "sampling": sampling, "wall": {"sample_s": round(sampling["seconds"], 1), "eval_s": round(eval_s, 1)},
        "critic_rows": rows, "margins_pp": list(MARGINS), "summary": summary,
        "positions": meta,
    }


def default_out(run: str) -> Path:
    return ROOT / "report" / "data" / f"trade-oracle-{run.replace('/', '-')}.json"


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
    out = args.out or default_out(args.run)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print_summary(result)
    print(f"wrote {out} ({result['wall']['total_s']:.0f} s)")


if __name__ == "__main__":
    main()
