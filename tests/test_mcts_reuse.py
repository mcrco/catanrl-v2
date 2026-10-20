"""Test 32: MCTS tree reuse (``reuse=true``): a seat's search is re-rooted at the action
it played when the game handed is exactly the previous one plus that action, dropped at
a stream's first decision or by ``forget`` (a caller that saw the game end), refused by
the agent (before the engine) when the game diverged and by the engine after a chance
action (under the view check even a replayed outcome the seat's view agrees with, which
the exact check refuses as diverged), and the counters add up; with
``transpositions`` the re-rooted search keeps its linked nodes. Rows with reuse are
invariant to num_envs, threads and worker splits in test_eval_invariance."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
from eval_helpers import agent, make_suite, schedule, tiny_model

from catanrl import games as game_configs
from catanrl.agents.native import MctsAgent
from catanrl.eval.harness import play_schedule
from catanrl.search import make_pool, run_searches

SUITE = make_suite(({"game": "base-2p"}, {"game": "base-3p"}), turn_limit=160)
GAMES = schedule(SUITE, 3, boards=2)
MODEL = tiny_model(["base-2p", "base-3p"], level="full", seed=11)
#: Actions whose root edge is a chance node: advance refuses them.
CHANCE = (cpp.ActionType.ROLL, cpp.ActionType.BUY_DEVELOPMENT_CARD, cpp.ActionType.MOVE_ROBBER)


def mcts(sims: int = 6, **params):
    return agent("MCTS", MODEL, infer_rows=8, sims=sims, **params)


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


def position(seed: int, want_roll: bool = False, steps: int = 0) -> cpp.Game:
    """A base-2p game stepped with ``seed``, past ``steps`` random actions, to a decision:
    the first with at least two options, or (``want_roll``) the first whose only option
    is the mover's roll."""
    game = game_configs.load_game("base-2p").cppanatron_game(seed, seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        actions = game.playable_actions
        game.execute(actions[int(rng.integers(len(actions)))])
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
    # At most one live search per env at the end: the slot holds one seat per game, and a
    # seat's first decision drops the env's searches from its previous game.
    assert len(first.pool) == len(first._live) <= 4
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
    # Another seat of stream 5's env starting a game (streams are env * max_players + seat)
    # drops the env's search too: its previous game is over. Stream 9 is another env.
    assert first.dims["max_players"] == 3
    decide(first, games[1:], streams=[4], first=[True])
    assert first.counters["dropped_first"] == 2 and first.counters["fresh"] == 4
    assert len(first.pool) == 2 and set(first._live) == {4, 9}
    # Stream 5 started once, so its next decision is not a harness error: a fresh search,
    # refused as diverged (stream 4's move came between its decisions).
    decide(first, games[:1], streams=[5], first=[False])
    assert first.counters["refused_diverged"] == 1 and first.counters["fresh"] == 5
    assert len(first.pool) == 3 and set(first._live) == {4, 5, 9}


def test_forget_drops_an_envs_searches_and_restarts_its_streams():
    """``forget(env)`` (the server bridge, at a game's last message): the env's live
    searches go, counted as ``dropped_over``, its streams must be marked first again,
    and another env's search stays."""
    first = mcts(reuse=True, oracle=True)
    assert first.dims["max_players"] == 3
    games = [position(7), position(8), position(9)]
    decide(first, games, streams=[3, 4, 9], first=[True, True, True])  # env 1 twice, env 3
    first.forget(1)
    assert first.counters["dropped_over"] == 2 and set(first._live) == {9} and len(first.pool) == 1
    first.forget(1)  # nothing there: nothing happens
    first.forget(2)
    assert first.counters["dropped_over"] == 2 and set(first._live) == {9}
    decide(first, games[:1], streams=[3], first=[True])
    assert set(first._live) == {3, 9} and first.counters["dropped_first"] == 0 and first.counters["fresh"] == 4
    first.forget(1)
    assert first.counters["dropped_over"] == 3 and set(first._live) == {9}
    # A decision on a forgotten stream not marked first is the caller's error (and, as any
    # error in act, empties the pool).
    with pytest.raises(ValueError, match="no live search"):
        decide(first, games[:1], streams=[3], first=[False])


def test_a_first_decision_drops_the_envs_stale_searches_and_loses_no_reuse(monkeypatch):
    """A fill MCTS agent holds two seats of every 3p game. A seat's first decision drops
    the env's searches from its previous game, so the agent holds at most its seats per
    env (not every seat it ever sat at); the other seat of the same game, if it decided
    already, loses nothing, since that first decision is a move between its own and its
    next decision would diverge anyway: rows and reuse counts equal a run dropping only
    the stream's own search."""
    games = schedule(SUITE, 1, boards=2)  # slot 0 at one seat, the fill at the others
    assert any(g.seats.count(1) == 2 for g in games)

    def run(num_envs):
        fill = mcts(reuse=True, oracle=True)
        rows = []
        play_schedule([agent("F")], games, fill=fill, suite=SUITE, on_game=rows.append, num_envs=num_envs)
        return {row["game_id"]: row for row in rows}, fill

    rows, fill = run(num_envs=1)
    assert fill.counters["reused"] > 0 and len(fill._live) == 2  # the last 3p game's two seats
    assert run(num_envs=5)[0] == rows
    monkeypatch.setattr(MctsAgent, "_stale", lambda self, stream: [stream] if stream in self._live else [])
    own_rows, own = run(num_envs=1)
    assert own_rows == rows and own.counters["reused"] == fill.counters["reused"]
    assert len(own._live) == 3  # without the env drop every seat the fill ever held lingers


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
    if action.action_type in CHANCE:
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


def test_a_replayed_chance_outcome_splits_the_refusals_by_check():
    """The seat's roll played with the dice the agent's own replay drew (a server
    replays recorded outcomes): under the exact check it is refused as diverged, since
    a replayed roll draws nothing from the RNG and ``search_equivalent`` compares it;
    under the view check the seat's private row, options and turn state agree, and the
    engine refuses the chance child. Fresh, reused and the refusals' sum agree between
    the checks, their split does not: the bridge comparison asserts just that."""
    exact, view = mcts(reuse=True, dets=1), mcts(reuse=True, dets=1)
    view.reuse_check = "view"
    for seed in range(20, 40):
        game = position(seed, want_roll=True)
        chosen = [int(decide(a, [game], streams=[0], first=[True])[0]) for a in (exact, view)]
        assert chosen[0] == chosen[1]
        roll = exact.action_space.decode(game, chosen[0])
        assert roll.action_type == cpp.ActionType.ROLL
        expected = game.copy()
        expected.execute(roll)
        if expected.current_player_index == game.current_player_index and len(expected.playable_actions) >= 2:
            break
    else:
        pytest.fail("no roll kept the turn with a choice to make")
    # The real game, rolled with the dice the replay drew: the same from the seat's view.
    for dice in ((a, b) for a in range(1, 7) for b in range(1, 7)):
        rolled = game.copy()
        rolled.execute(roll, dice=dice)
        if view._same_view(game, expected, rolled):
            break
    else:
        pytest.fail("no dice reproduce the replay's roll")
    assert not expected.search_equivalent(rolled)
    for checker in (exact, view):
        decide(checker, [rolled], streams=[0], first=[False])
    for name in ("decisions", "fresh", "reused", "dropped_first"):
        assert exact.counters[name] == view.counters[name]
    assert exact.counters["refused_diverged"] == 1 and exact.counters["refused_engine"] == 0
    assert view.counters["refused_diverged"] == 0 and view.counters["refused_engine"] == 1


def test_reuse_advances_a_search_with_transpositions():
    """Both options at once: the re-rooted search is a graph (nodes another order of
    actions reached are linked), the engine re-parents what it keeps, the link count
    carries over and the budget is fresh on top of the retained visits."""
    first = mcts(reuse=True, transpositions=True, oracle=True, sims=64)
    assert first.config.transpositions
    # Mid-game positions with a bank trade or a build to make, where orders of actions meet.
    for seed, steps in ((4, 80), (5, 140), (10, 80), (3, 80)):
        game = position(seed, steps=steps)
        chosen = int(decide(first, [game], streams=[0], first=[True])[0])
        links = first.pool.stats(first._live[0][0]).transpositions
        action = first.action_space.decode(game, chosen)
        following = game.copy()
        following.execute(action)
        if links > 0 and action.action_type not in CHANCE and following.current_player_index == game.current_player_index:
            break
    else:
        pytest.fail("no probed position links nodes and keeps the turn after the action played")
    decide(first, [following], streams=[0], first=[False])
    assert first.counters["reused"] == 1 and first.counters["refused_engine"] == 0
    stats = first.pool.stats(first._live[0][0])
    assert stats.simulations == 64 and stats.transpositions >= links
    assert first.pool.visits(first._live[0][0]).sum() > 64 and first.search_stats["transpositions"] >= 2 * links


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
