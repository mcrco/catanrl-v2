"""How far the hand tracker's deals are from the true posterior over hidden steals.

Plays random 3- and 4-player games on the CPU, following one observer with a
pool tracker and an exact tracker (``cpp.TrackerMode``), stops at random
decisions after at least one hidden steal and compares, by total variation
distance, the distribution each deal draws the other seats' resource hands
from with the true posterior, enumerated by brute force:

* the truth: every assignment of a kind to every hidden steal the observer
  could not see, replayed from the start through the observer's events
  (``HandTracker.events``), weighted by the rules' chance (a steal takes a
  uniformly random card of the victim's hand) and dropped when a later
  event becomes impossible: what a perfect counter can know, with no model
  of what seats choose to do;
* the pool deal: every deal of the unknown pool to the unknown slots, as
  ``sample_world`` draws it in pool mode, weighted as a uniform deal;
* the exact deal: the exact tracker's hypotheses while it holds them, else
  the pool deal it falls back to.

Also times ``sample_world`` per call in both modes, and asserts at every
step of every game that the exact tracker never excludes the true state.

    uv run python tools/belief_exactness.py --games 400 --positions 300
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import Counter

import cppanatron as cpp

Hands = tuple[tuple[int, ...], ...]
MAX_POOL_DEALS = 200_000


def hands_of(game: cpp.Game) -> Hands:
    return tuple(tuple(p.resources) for p in game.players)


# ---------------------------------------------------------------- the truth, by brute force


def brute_force_posterior(events: list[cpp.TrackerEvent], num_players: int) -> dict[Hands, float]:
    """The posterior over every seat's hand from the observer's events, with
    no merging: a depth-first search over the kinds of the hidden steals,
    each branch replaying the events on its own (infeasible ones die at the
    event that contradicts them)."""
    out: Counter = Counter()

    def replay(start: int, hands: list[list[int]], weight: float) -> None:
        for i in range(start, len(events)):
            e = events[i]
            t = e.type
            if t == cpp.TrackerEventType.GAIN:
                for r, c in enumerate(e.cards):
                    hands[e.player][r] += c
            elif t == cpp.TrackerEventType.LOSE:
                for r, c in enumerate(e.cards):
                    hands[e.player][r] -= c
                    if hands[e.player][r] < 0:
                        return
            elif t == cpp.TrackerEventType.TRANSFER:
                for r, c in enumerate(e.cards):
                    hands[e.other][r] -= c
                    if hands[e.other][r] < 0:
                        return
                    hands[e.player][r] += c
            elif t == cpp.TrackerEventType.MONOPOLY:
                r = e.resource.value
                taken = 0
                for seat in range(num_players):
                    if seat == e.player:
                        continue
                    if hands[seat][r] != e.amounts[seat]:
                        return
                    taken += hands[seat][r]
                    hands[seat][r] = 0
                hands[e.player][r] += taken
            elif t == cpp.TrackerEventType.STEAL:
                victim, thief = e.other, e.player
                size = sum(hands[victim])
                if size == 0:
                    return
                if e.resource is not None:
                    r = e.resource.value
                    if hands[victim][r] == 0:
                        return
                    weight *= hands[victim][r] / size
                    hands[victim][r] -= 1
                    hands[thief][r] += 1
                else:
                    for r in range(5):
                        if hands[victim][r] > 0:
                            branch = [h[:] for h in hands]
                            branch[victim][r] -= 1
                            branch[thief][r] += 1
                            replay(i + 1, branch, weight * hands[victim][r] / size)
                    return
        out[tuple(tuple(h) for h in hands)] += weight

    replay(0, [[0] * 5 for _ in range(num_players)], 1.0)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


# ---------------------------------------------------------------- the deals, analytically


def pool_distribution(tracker: cpp.HandTracker) -> dict[Hands, float] | None:
    """Every deal of the unknown pool to the unknown slots, weighted as the
    uniform deal ``sample_world`` draws in pool mode (proportional to
    1 / prod c! over the counts c each seat gets of each kind), or None
    when there are too many deals to enumerate."""
    n = tracker.num_players
    known = [list(tracker.known_resources(s)) for s in range(n)]
    unknown = [tracker.unknown_count(s) for s in range(n)]
    pool = list(tracker.unknown_pool)
    out: dict[Hands, float] = {}

    def compositions(left: int, pool_now: list[int]):
        """Every way to take `left` cards from pool_now, as counts per kind."""
        if left == 0:
            yield (0, 0, 0, 0, 0)
            return
        for c0 in range(min(left, pool_now[0]) + 1):
            for c1 in range(min(left - c0, pool_now[1]) + 1):
                for c2 in range(min(left - c0 - c1, pool_now[2]) + 1):
                    for c3 in range(min(left - c0 - c1 - c2, pool_now[3]) + 1):
                        c4 = left - c0 - c1 - c2 - c3
                        if c4 <= pool_now[4]:
                            yield (c0, c1, c2, c3, c4)

    def deal(seat: int, pool_now: list[int], weight: float, hands: list[list[int]]) -> bool:
        if seat == n:
            out[tuple(tuple(h) for h in hands)] = out.get(tuple(tuple(h) for h in hands), 0.0) + weight
            return len(out) <= MAX_POOL_DEALS
        for c in compositions(unknown[seat], pool_now):
            w = weight
            for k in c:
                w /= math.factorial(k)
            hands[seat] = [known[seat][r] + c[r] for r in range(5)]
            if not deal(seat + 1, [pool_now[r] - c[r] for r in range(5)], w, hands):
                return False
        return True

    if not deal(0, pool, 1.0, [k[:] for k in known]):
        return None
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def exact_distribution(tracker: cpp.HandTracker) -> dict[Hands, float] | None:
    """The exact tracker's hypotheses, or the pool deal it has fallen back to."""
    if not tracker.exact:
        return pool_distribution(tracker)
    out: dict[Hands, float] = {}
    for weight, hands in tracker.hypotheses:
        key = tuple(tuple(h) for h in hands)
        out[key] = out.get(key, 0.0) + weight
    return out


def sampled_distribution(game: cpp.Game, tracker: cpp.HandTracker, samples: int, seed: int) -> dict[Hands, float]:
    counts: Counter = Counter()
    for i in range(samples):
        counts[hands_of(cpp.sample_world(game, tracker, seed=seed + i))] += 1
    return {k: v / samples for k, v in counts.items()}


def total_variation(p: dict[Hands, float], q: dict[Hands, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


# ---------------------------------------------------------------- the games


def choose(game: cpp.Game, rng: random.Random, play: str, seat_play: list[str]) -> cpp.Action:
    actions = game.playable_actions
    if len(actions) == 1:
        return actions[0]
    how = play if play != "mixed" else seat_play[game.current_player_index]
    if how == "value":
        return cpp.value_action(game)
    return rng.choice(actions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=200, help="random games to play (3p and 4p alternate)")
    parser.add_argument("--positions", type=int, default=200, help="positions to measure at most")
    parser.add_argument("--max-steals", type=int, default=7,
                        help="measure only where the observer has missed at most this many steals")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=2000, help="sample_world calls per mode per position")
    parser.add_argument("--play", choices=("random", "value", "mixed"), default="random",
                        help="random legal play, the engine's F, or each seat one of them")
    parser.add_argument("--cap", type=int, default=cpp.HandTracker(cpp.Game(2), 0).max_hypotheses,
                        help="the exact tracker's max_hypotheses")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="largest allowed TV between the exact belief and the brute force")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tv_pool: list[float] = []
    tv_exact: list[float] = []
    tv_exact_sampled: list[float] = []
    tv_pool_sampled: list[float] = []
    tv_check: list[float] = []
    us_pool: list[float] = []
    us_exact: list[float] = []
    hypotheses: list[int] = []
    fallen_back = 0
    measured = 0
    steps = 0
    soundness_checks = 0
    overflows = 0
    restarts = 0
    skipped_pool_too_big = 0
    hidden_steals_total = 0
    worst = 0.0
    for g in range(args.games):
        n = 3 + g % 2
        game = cpp.Game(n, seed=args.seed * 100_003 + g)
        observer = g % n
        exact = cpp.HandTracker(game, observer, cpp.TrackerMode.EXACT, max_hypotheses=args.cap)
        pool = cpp.HandTracker(game, observer)
        events: list[cpp.TrackerEvent] = []
        hidden = 0
        seat_play = [rng.choice(("random", "value")) for _ in range(n)]
        for step in range(args.max_steps):
            if game.winning_color is not None:
                break
            action = choose(game, rng, args.play, seat_play)
            before = game.copy()
            game.execute(action)
            new = cpp.HandTracker.events(before, action, game, observer)
            events += new
            hidden += sum(1 for e in new if e.type == cpp.TrackerEventType.STEAL and e.resource is None)
            was_exact, was_overflows = exact.exact, exact.belief_overflows
            for tracker in (exact, pool):
                tracker.observe(before, action, game)
            steps += 1
            assert exact.consistent_with(game), f"game {g} step {step}: the exact tracker excluded the truth"
            assert pool.consistent_with(game), f"game {g} step {step}: the pool tracker excluded the truth"
            soundness_checks += 1
            overflows += exact.belief_overflows - was_overflows
            restarts += int(not was_exact and exact.exact and exact.belief_overflows == was_overflows)
            if exact.exact:
                hypotheses.append(exact.num_hypotheses)
            if (hidden == 0 or hidden > args.max_steals or game.winning_color is not None
                    or len(game.playable_actions) < 2 or measured >= args.positions or rng.random() > 0.03):
                continue
            truth = brute_force_posterior(events, n)
            pool_dist = pool_distribution(pool)
            if pool_dist is None:
                skipped_pool_too_big += 1
                continue
            exact_dist = exact_distribution(exact)
            if exact_dist is None:
                skipped_pool_too_big += 1
                continue
            assert truth.get(hands_of(game), 0.0) > 0, "the brute force must keep the truth"
            measured += 1
            hidden_steals_total += hidden
            fallen_back += int(not exact.exact)
            tv_pool.append(total_variation(pool_dist, truth))
            tv_exact.append(total_variation(exact_dist, truth))
            if exact.exact:
                tv_check.append(tv_exact[-1])
                worst = max(worst, tv_exact[-1])
            seed = args.seed * 7919 + measured
            start = time.perf_counter()
            pool_samples = sampled_distribution(game, pool, args.samples, seed)
            us_pool.append((time.perf_counter() - start) / args.samples * 1e6)
            start = time.perf_counter()
            exact_samples = sampled_distribution(game, exact, args.samples, seed)
            us_exact.append((time.perf_counter() - start) / args.samples * 1e6)
            tv_pool_sampled.append(total_variation(pool_samples, truth))
            tv_exact_sampled.append(total_variation(exact_samples, truth))

    def describe(values: list[float], unit: str = "") -> str:
        if not values:
            return "n/a"
        return (f"mean {statistics.mean(values):.4f}{unit}, median {statistics.median(values):.4f}{unit}, "
                f"max {max(values):.4f}{unit}")

    print(f"games {args.games} ({args.play} play), steps {steps}, soundness checks {soundness_checks} "
          f"(both trackers always consistent with the truth)")
    print(f"exact belief: hypotheses mean {statistics.mean(hypotheses) if hypotheses else 0:.2f}, "
          f"max {max(hypotheses) if hypotheses else 0}, cap {args.cap}, overflows {overflows}, restarts {restarts}")
    print(f"positions measured {measured} (after 1..{args.max_steals} hidden steals, "
          f"{hidden_steals_total / max(1, measured):.2f} on average; {skipped_pool_too_big} skipped: "
          f"too many pool deals to enumerate); exact mode fallen back at {fallen_back}")
    print(f"TV(pool deal, truth):      {describe(tv_pool)}")
    print(f"TV(exact deal, truth):     {describe(tv_exact)}")
    print(f"TV(pool deal, truth), {args.samples} samples:  {describe(tv_pool_sampled)}")
    print(f"TV(exact deal, truth), {args.samples} samples: {describe(tv_exact_sampled)}")
    print(f"sample_world per call: pool {describe(us_pool, ' us')}; exact {describe(us_exact, ' us')}")
    if worst > args.tolerance:
        print(f"FAILED: the exact belief differs from the brute force by {worst:.3g} > {args.tolerance}")
        return 1
    print(f"the exact belief matches the brute force at {len(tv_check)} positions (worst TV {worst:.2g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
