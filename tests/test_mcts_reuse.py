"""Test 32: MCTS tree reuse (``reuse=true``): a seat's search is re-rooted at the action
it played when the game handed is exactly the previous one plus that action, dropped at
a stream's first decision, refused by the agent (before the engine) when the game
diverged and by the engine after a chance action, and the counters add up. Rows with
reuse are invariant to num_envs, threads and worker splits in test_eval_invariance."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
from eval_helpers import agent, make_suite, schedule, tiny_model

from catanrl import games as game_configs
from catanrl.eval.harness import play_schedule
from catanrl.search import make_pool, run_searches

SUITE = make_suite(({"game": "base-2p"}, {"game": "base-3p"}), turn_limit=160)
GAMES = schedule(SUITE, 3, boards=2)
MODEL = tiny_model(["base-2p", "base-3p"], level="full", seed=11)


def mcts(**params):
    return agent("MCTS", MODEL, infer_rows=8, sims=6, **params)


def play(first, games=GAMES, **kwargs):
    rows = []
    play_schedule([first, agent("F"), agent("R")], games, suite=SUITE, on_game=rows.append, **kwargs)
    return {row["game_id"]: row for row in rows}


def decide(mcts, games, streams, first, seeds=None):
    """One ``act`` on ``games`` with trackers following their movers."""
    rows = np.stack([mcts.encode(g, g.current_player_index) for g in games])
    masks = rows[:, -mcts.action_space.size:]
    seeds = np.arange(100, 100 + len(games), dtype=np.uint64) if seeds is None else np.asarray(seeds, dtype=np.uint64)
    trackers = [cpp.HandTracker(g, g.current_player_index) for g in games] if mcts.needs_trackers else None
    return mcts.act(rows, masks, seeds, games=games, trackers=trackers,
                    streams=np.asarray(streams, dtype=np.int64), first=np.asarray(first, dtype=bool))


def position(seed: int, want_roll: bool = False) -> cpp.Game:
    """A base-2p game stepped with ``seed`` to a decision: the first with at least two
    options, or (``want_roll``) the first whose only option is the mover's roll."""
    game = game_configs.load_game("base-2p").cppanatron_game(seed, seed)
    rng = np.random.default_rng(seed)
    while True:
        actions = game.playable_actions
        if want_roll:
            if len(actions) == 1 and actions[0].action_type == cpp.ActionType.ROLL:
                return game
        elif len(actions) >= 2:
            return game
        game.execute(actions[int(rng.integers(len(actions)))])


@pytest.mark.parametrize("params", [{"oracle": True}, {"dets": 2}])
def test_reuse_happens_and_the_counters_add_up(params):
    first = mcts(reuse=True, **params)
    assert first.reuse and first.needs_memory
    rows = play(first, num_envs=4)
    c = first.counters
    assert c["reused"] > 0 and c["fresh"] > 0 and c["refused_diverged"] > 0 and c["refused_engine"] > 0
    assert c["reused"] + c["fresh"] == c["decisions"]
    seated = [row for row in rows.values() if 0 in row["seats"]]
    assert c["decisions"] == sum(row["decisions"][seat] for row in seated for seat, slot in enumerate(row["seats"])
                                 if slot == 0)
    # Every decision after a stream's first pops its live search: reused or refused.
    assert c["reused"] + c["refused_diverged"] + c["refused_engine"] == c["decisions"] - len(seated)
    # One live search per stream at the end, none beyond the envs' seats.
    assert len(first.pool) == len(first._live) <= 4 * 3
    assert first.search_stats["simulations"] >= 6 * c["decisions"]
    # Reuse is keyed because it changes the rows.
    assert rows != play(mcts(**params), num_envs=4)


def test_reuse_is_deterministic():
    runs = [play(mcts(reuse=True, dets=2), num_envs=n) for n in (3, 7)]
    assert runs[0] == runs[1]
    agents = [mcts(reuse=True, dets=2) for _ in range(2)]
    for first, n in zip(agents, (3, 7)):
        play(first, num_envs=n)
    # dropped_first alone depends on num_envs: fewer envs start more games in each stream.
    for name in ("decisions", "fresh", "reused", "refused_diverged", "refused_engine"):
        assert agents[0].counters[name] == agents[1].counters[name] > 0
    assert agents[0].search_stats == agents[1].search_stats


def test_a_first_decision_drops_the_live_search():
    first = mcts(reuse=True, oracle=True)
    games = [position(1), position(2)]
    decide(first, games, streams=[5, 9], first=[True, True])
    assert len(first.pool) == 2 and set(first._live) == {5, 9}
    decide(first, games[:1], streams=[5], first=[True])
    assert first.counters["dropped_first"] == 1 and first.counters["fresh"] == 3
    assert len(first.pool) == 2 and set(first._live) == {5, 9}


def test_the_next_position_reuses_and_a_diverged_one_is_refused_before_the_engine():
    class NoAdvance:
        """The agent's pool with ``advance`` forbidden."""

        def __init__(self, pool):
            self._pool = pool

        def __getattr__(self, name):
            return getattr(self._pool, name)

        def __len__(self):
            return len(self._pool)

        def advance(self, *args):
            raise AssertionError("advance called for a diverged stream")

    first = mcts(reuse=True, oracle=True)
    game = position(3)
    chosen = int(decide(first, [game], streams=[0], first=[True])[0])
    action = first.action_space.decode(game, chosen)
    following = game.copy()
    following.execute(action)
    if action.action_type in (cpp.ActionType.ROLL, cpp.ActionType.BUY_DEVELOPMENT_CARD, cpp.ActionType.MOVE_ROBBER):
        pytest.skip("the chosen action is a chance action (the engine's refusal, tested below)")
    if following.current_player_index != game.current_player_index:
        pytest.skip("the chosen action hands the turn over")
    decide(first, [following], streams=[0], first=[False])
    assert first.counters["reused"] == 1 and first.counters["fresh"] == 1
    assert first.pool.stats(first._live[0][0]).simulations == 6  # a fresh budget on the retained tree
    assert first.pool.visits(first._live[0][0]).sum() >= 6  # the retained child's visits plus the budget
    # The same game again is not "following + the action played": refused without asking the engine.
    first.pool = NoAdvance(first.pool)
    decide(first, [following], streams=[0], first=[False])
    assert first.counters["refused_diverged"] == 1 and first.counters["fresh"] == 2 and len(first.pool) == 1


@pytest.mark.parametrize("params", [{"oracle": True}, {"dets": 2}])
def test_a_chance_action_is_refused_by_the_engine_and_the_fresh_search_is_the_plain_one(params):
    first = mcts(reuse=True, **params)
    for seed in range(20, 40):
        game = position(seed, want_roll=True)
        chosen = int(decide(first, [game], streams=[1], first=[True])[0])
        assert first.action_space.decode(game, chosen).action_type == cpp.ActionType.ROLL
        rolled = game.copy()
        rolled.execute(first.action_space.decode(game, chosen))
        if rolled.current_player_index == game.current_player_index and len(rolled.playable_actions) >= 2:
            break
    else:
        pytest.fail("no roll kept the turn with a choice to make")
    seed = np.array([77], dtype=np.uint64)
    choice = int(decide(first, [rolled], streams=[1], first=[False], seeds=seed)[0])
    assert first.counters["refused_engine"] == 1 and first.counters["reused"] == 0 and first.counters["fresh"] == 2
    # The fresh search is exactly a no-reuse search from the real position with that decision seed.
    pool = make_pool(first.dims, first.level, first.config)
    tracker = cpp.HandTracker(rolled, rolled.current_player_index) if first.needs_trackers else None
    sid = pool.add(rolled, seed=77, tracker=tracker, deal=first.tracker_mode if tracker is not None else None)
    run_searches(pool, first.evaluator, batch_rows=first.config.max_pending)
    assert pool.visits(sid).tolist() == first.pool.visits(first._live[1][0]).tolist()
    assert int(pool.select(sid, 0.0)) == choice


def test_streams_are_required_and_checked():
    first = mcts(reuse=True, oracle=True)
    game = position(4)
    rows = first.encode(game, game.current_player_index)[None]
    masks, seeds = rows[:, -first.action_space.size:], np.array([1], dtype=np.uint64)
    with pytest.raises(ValueError, match="streams"):
        first.act(rows, masks, seeds, games=[game])
    with pytest.raises(ValueError, match="no live search"):
        decide(first, [game], streams=[3], first=[False])
    with pytest.raises(ValueError, match="once per batch"):
        decide(first, [game, game], streams=[3, 3], first=[True, True])
    with pytest.raises(ValueError, match="one entry per row"):
        decide(first, [game], streams=[3, 4], first=[True, True])
    assert len(first.pool) == 0 and not first._live
    assert not mcts(oracle=True).needs_memory


def test_an_error_empties_the_pool_and_the_streams_restart():
    class Failing:
        def __init__(self, inner):
            self.inner, self.calls = inner, 0
            self.obs_size, self.num_actions, self.num_players = inner.obs_size, inner.num_actions, inner.num_players

        def __call__(self, rows):
            self.calls += 1
            if self.calls == 2:
                raise MemoryError("evaluator failed")
            return self.inner(rows)

    first = mcts(reuse=True, oracle=True)
    games = [position(5), position(6)]
    decide(first, games, streams=[0, 1], first=[True, True])
    healthy = first.evaluator
    first.evaluator = Failing(healthy)
    with pytest.raises(MemoryError, match="evaluator failed"):
        decide(first, games, streams=[0, 1], first=[True, True])
    assert len(first.pool) == 0 and first.pool.pending == 0 and not first._live
    first.evaluator = healthy
    with pytest.raises(ValueError, match="no live search"):
        decide(first, games, streams=[0, 1], first=[False, False])
    assert len(decide(first, games, streams=[0, 1], first=[True, True])) == 2 and len(first.pool) == 2
