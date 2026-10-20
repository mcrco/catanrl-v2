"""ViewAgent: the same decisions as the native agent on the same positions
and seeds (POLICY, MCTS dets=1 sims=8 with and without reuse, BELIEF
worlds=2); decisions that depend on the view alone (paired games whose views
are identical but whose hidden cards differ); a reusing MCTS agent's streams
(one per game slot and seat, re-rooted along the seat's own moves, dropped
with a finished or an evicted game, restarted after an error) and the
choices without reuse pinned; fairness and fallback rules."""

from __future__ import annotations

import json
import random

import cppanatron as cpp
import numpy as np
import pytest
from bridge_games import DATA, lockstep, tiny_agent
from catanatron.models.enums import ActionRecord, ActionType
from catanatron.serialization import action_from_json, action_to_json, client_view, state_to_json
from cppanatron.catanatron import from_catanatron_action, to_catanatron_action

from catanrl.agents.native import BotAgent
from catanrl.bridge.agent import BridgeError, UnfairAgent, ViewAgent, action_key, default_choice, native_agent
from catanrl.seeding import derive_seed

#: case -> (level, params, the index of its games and seeds).
CASES = {
    "POLICY": ("private", {}, 2),
    "MCTS": ("full", {"sims": 8, "dets": 1}, 1),
    "BELIEF": ("full", {"worlds": 2}, 0),
    #: Keeps its search between a seat's decisions: the ViewAgent streams them.
    "MCTS-reuse": ("full", {"sims": 8, "dets": 1, "reuse": True}, 4),
}


def _agent(case: str, dims: dict):
    level, params, _ = CASES[case]
    return tiny_agent(case.split("-")[0], level, dims, **params)


def _native_choice(native, game, seat, seed, tracker, first=None):
    """The native agent's choice on the true game; ``first`` (a reusing
    agent) puts the decision on the seat's stream, as the harness would."""
    row = native.encode(game, seat)
    size = native.action_space.size
    memory = {} if first is None else {"streams": np.array([seat]), "first": np.array([first])}
    index = native.act(row[None], row[None, -size:], np.array([seed], dtype=np.uint64),
                       games=[game.copy()] if native.needs_games else None,  # the agent may keep it
                       trackers=[tracker] if native.needs_trackers else None, **memory)[0]
    return action_to_json(to_catanatron_action(native.action_space.decode(game, int(index))))


@pytest.mark.parametrize("key", sorted(CASES))
@pytest.mark.parametrize("name", ["base-2p", "base-3p", "base-4p-friendly"])
def test_view_agent_decides_like_the_native_agent(key, name):
    """Every decision of every seat of a random game: the native agent on the
    true game and the seat's true tracker, the ViewAgent on the seat's views
    only (decide-only, as a TUI callback), with the same decision seeds.
    Under the friendly robber the agent is also asked about lone robber
    moves, as a server asks (the options bound the others' points). A
    reusing agent reuses alike on both sides (the exact check on the true
    game, the view check on the replayed one); only the refusals' split
    differs by construction (a replayed chance outcome the seat's view
    agrees with: test_a_replayed_chance_outcome_splits_the_refusals_by_check
    in test_mcts_reuse)."""
    level, params, index = CASES[key]
    friendly = name.endswith("-friendly")
    ls = lockstep(name.removesuffix("-friendly"), 20 + index, friendly_robber=friendly or None)
    n = len(ls.colors)
    native = _agent(key, ls.dims)

    def seed(ply: int, seat: int) -> int:
        return derive_seed(index, "decide", seat, ply)

    ours = ViewAgent(_agent(key, ls.dims), strict=True,
                     seed_fn=lambda view, seat: seed(len(view["action_records"]), seat))
    trackers = [cpp.HandTracker(ls.native, s) for s in range(n)]
    decided: set[int] = set()
    compared, limit = 0, 30 if key.startswith("MCTS") else 80
    while compared < limit and not ls.over:
        seat = ls.current_seat
        playable = ls.playable_json()
        if len(playable) == 1 and friendly:
            assert ours.decide(ls.view(seat), playable) is playable[0]
        if len(playable) > 1:
            view = ls.view(seat)
            mine = ours.decide(view, playable)
            assert any(mine is offered for offered in playable)  # the offered object itself
            theirs = _native_choice(native, ls.native, seat, seed(len(view["action_records"]), seat),
                                    trackers[seat], first=seat not in decided if native.needs_memory else None)
            decided.add(seat)
            assert action_key(mine) == action_key(theirs), (key, name, len(view["action_records"]))
            compared += 1
        action = ls.choose()
        before = ls.native.copy()
        ls.step(action)
        for s in range(n):
            trackers[s].observe(before, from_catanatron_action(action), ls.native)
    assert compared == limit
    assert ours.stats["decisions"] == compared
    assert not any(k.startswith("fallback") for k in ours.stats)
    if native.needs_memory:
        mine, theirs = ours.native.counters, native.counters
        assert mine["decisions"] == compared
        for counter in ("decisions", "fresh", "reused", "dropped_first", "dropped_over"):
            assert mine[counter] == theirs[counter], counter
        assert (mine["refused_diverged"] + mine["refused_engine"]
                == theirs["refused_diverged"] + theirs["refused_engine"])


# ---------------------------------------------------------------- views alone


class PairedGames:
    """Two Catanatron games with the same public moves and the same cards
    for ``seat``, but their own hidden randomness (deck order, steals between
    other seats), so the other hands differ while ``seat``'s views agree."""

    def __init__(self, name: str, index: int, seat: int) -> None:
        self.a = lockstep(name, index)
        self.b = lockstep(name, index)
        self.b.truth.id = self.a.truth.id  # one game, as far as the seat can tell
        state = self.b.truth.state
        random.Random(index + 1).shuffle(state.development_listdeck)
        state.random.seed(index + 1)
        self.seat = seat
        self.color = self.a.colors[seat]
        self.rng = random.Random(index)
        self.hidden_differed = False

    def views(self) -> tuple[dict, dict]:
        return (client_view(state_to_json(self.a.truth), self.color),
                client_view(state_to_json(self.b.truth), self.color))

    def others_move(self):
        """A random move legal in both games, or None."""
        keys_b = {action_key(a) for a in self.b.playable_json()}
        common = [a for a in self.a.truth.playable_actions if action_key(action_to_json(a)) in keys_b]
        return self.rng.choice(common) if common else None

    def step(self, action) -> bool:
        """Play ``action`` in both; the seat's private outcomes (its steals and
        purchases) and the dice are forced alike. False if they cannot be."""
        record = self.a.truth.execute(action)
        seen = action.action_type in (ActionType.BUY_DEVELOPMENT_CARD, ActionType.MOVE_ROBBER) and (
            action.color.value == self.color or (action.action_type == ActionType.MOVE_ROBBER
                                                 and action.value[1] is not None
                                                 and action.value[1].value == self.color))
        forced = None
        if seen and record.result is not None and not self._possible(action, record.result):
            return False
        if action.action_type == ActionType.ROLL or seen:
            forced = ActionRecord(action=action, result=record.result)
        self.b.truth.execute(action, action_record=forced)
        self.hidden_differed |= self._hidden(self.a) != self._hidden(self.b)
        return True

    def _possible(self, action, result) -> bool:
        state = self.b.truth.state
        if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return result in state.development_listdeck
        victim = action.value[1]
        index = state.colors.index(victim)
        return state.player_state[f"P{index}_{result}_IN_HAND"] > 0

    def _hidden(self, ls) -> dict:
        """The other seats' hands and cards."""
        state = ls.truth.state.player_state
        return {k: v for k, v in state.items() if k.endswith("_IN_HAND") and not k.startswith(f"P{self.seat}_")}


@pytest.mark.parametrize("key", sorted(CASES))
def test_decisions_depend_on_the_view_alone(key):
    """Paired games: the seat's views are identical, the other hands are not;
    two ViewAgents, each following its own game, decide alike throughout."""
    pairs = PairedGames("base-4p", 30 + CASES[key][2], seat=1)
    dims = pairs.a.dims
    agent_a = ViewAgent(_agent(key, dims), strict=True)
    agent_b = ViewAgent(_agent(key, dims), strict=True)
    decided, limit = 0, 15 if key.startswith("MCTS") else 40
    for _ in range(3000):
        if pairs.a.over or decided >= limit:
            break
        view_a, view_b = pairs.views()
        assert view_a == view_b
        playable = pairs.a.playable_json()
        if pairs.a.current_seat == pairs.seat:
            assert playable == pairs.b.playable_json()
            choice_a = agent_a.decide(view_a, playable)
            choice_b = agent_b.decide(view_b, pairs.b.playable_json())
            assert action_key(choice_a) == action_key(choice_b)
            decided += len(playable) > 1
            action = action_from_json(choice_a)
        else:
            action = pairs.others_move()
            if action is None:
                break
        if not pairs.step(action):
            break
    assert decided >= 10
    assert pairs.hidden_differed


# ---------------------------------------------------------------- a reusing search


def _own_game(agent: ViewAgent, ls, seat: int = 0, limit: int = 60, steps: int = 1000) -> int:
    """``seat`` plays the ViewAgent's choices in a lockstep game, the other seats
    random moves; the number of decisions (several options) made."""
    decided = 0
    for _ in range(steps):
        if ls.over or decided >= limit:
            break
        if ls.current_seat == seat:
            playable = ls.playable_json()
            choice = agent.decide(ls.view(seat), playable)
            assert any(choice is offered for offered in playable)
            decided += len(playable) > 1
            ls.step(action_from_json(choice))
        else:
            ls.step(ls.choose())
    return decided


def _reusing(dims: dict, **kwargs) -> ViewAgent:
    return ViewAgent(tiny_agent("MCTS", "full", dims, sims=8, dets=1, reuse=True), **kwargs)


def _next_decision(agent: ViewAgent, ls, seat: int = 0) -> None:
    """Play random moves up to ``seat``'s next decision with a choice, and its choice."""
    while not (ls.current_seat == seat and len(ls.playable_json()) > 1):
        ls.step(ls.choose())
    ls.step(action_from_json(agent.decide(ls.view(seat), ls.playable_json())))


def test_a_reusing_search_follows_the_seats_own_moves():
    """A seat playing its own choices: the search is re-rooted at the action
    played wherever the seat decides again right after it (an initial
    settlement's road, a build or a trade before the next move, a discard
    chain); a forced move the ViewAgent answers alone and every other seat's
    move diverge it (a fresh search, counted); the counters add up and the
    agent keeps copies, not the replay's game."""
    ls = lockstep("base-2p", 50)
    agent = _reusing(ls.dims, strict=True)
    native = agent.native
    assert agent.memory and native.reuse_check == "view"
    decided = _own_game(agent, ls, seat=0, limit=60)
    c = native.counters
    assert decided >= 30 and c["decisions"] == decided == agent.stats["decisions"]
    assert c["reused"] > 0 and c["reused"] + c["fresh"] == c["decisions"]
    assert agent.stats["forced"] > 0 and c["refused_diverged"] > 0
    assert c["reused"] + c["refused_diverged"] + c["refused_engine"] == c["decisions"] - 1
    assert not any(k.startswith("fallback") or k == "inexact" for k in agent.stats)
    assert len(native.pool) == 1 and set(native._live) == {0}  # slot 0, seat 0
    assert all(entry[1] is not tracker.game for entry in native._live.values() for tracker in agent.seats.values())
    assert agent.summary()["search"] == dict(c)


def test_streams_are_per_game_and_an_evicted_game_frees_its_searches():
    """Three games through one ViewAgent following two at a time: a game's
    first decision starts a stream in its own slot (the other game's search
    stays live); the third game takes the evicted first game's slot and
    drops that game's search at its first decision."""
    games = [lockstep("base-2p", 60 + i) for i in range(3)]
    agent = _reusing(games[0].dims, strict=True, max_games=2)
    native = agent.native
    _next_decision(agent, games[0])
    _next_decision(agent, games[0])
    assert native.counters["fresh"] >= 1 and set(native._live) == {0} and len(native.pool) == 1
    _next_decision(agent, games[1])  # slot 1: stream 1 * 2 + 0
    assert set(native._live) == {0, 2} and len(native.pool) == 2 and native.counters["dropped_first"] == 0
    _next_decision(agent, games[2])  # evicts game 0, takes slot 0 and drops game 0's search there
    assert native.counters["dropped_first"] == 1 and set(native._live) == {0, 2} and len(native.pool) == 2
    assert [info["env"] for info in agent._games.values()] == [1, 0]
    assert list(agent._games) == [games[1].truth.id, games[2].truth.id]
    _next_decision(agent, games[0])  # new again: evicts game 1 and takes its slot
    assert native.counters["dropped_first"] == 2 and set(native._live) == {0, 2} and len(native.pool) == 2
    assert list(agent._games) == [games[2].truth.id, games[0].truth.id]


def test_a_finished_game_frees_its_searches_and_slot():
    """A game's last message (``after``, or ``aborted`` with its id, through
    ``observe``): at the next decision the agent drops the game's searches
    and its slot is free, so the next new game takes it without evicting any
    game and drops nothing at its first decision. The other seat's copy of
    the message, one for a game never followed and a room voided before its
    game (no id) do nothing."""
    games = [lockstep("base-2p", 80 + i) for i in range(4)]
    agent = _reusing(games[0].dims, strict=True, max_games=8)
    native = agent.native
    _next_decision(agent, games[0])
    _next_decision(agent, games[1])
    assert set(native._live) == {0, 2} and len(native.pool) == 2
    over = {"type": "after", "game_id": games[0].truth.id, "index": 0, "state": {}}
    agent.observe(over)
    agent.observe(over)  # the other bot seat's copy
    agent.observe({"type": "after", "game_id": "never followed", "index": 0, "state": {}})
    agent.observe({"type": "aborted", "game_id": None, "reason": "nobody sat"})
    assert set(native._live) == {0, 2}  # queued: the work happens at the next decision
    _next_decision(agent, games[1])
    assert set(native._live) == {2} and len(native.pool) == 1 and native.counters["dropped_over"] == 1
    assert list(agent._games) == [games[1].truth.id] and agent._free_envs == [0]
    _next_decision(agent, games[2])  # takes slot 0 without evicting game 1; nothing to drop there
    assert set(native._live) == {0, 2} and len(native.pool) == 2 and native.counters["dropped_first"] == 0
    assert [info["env"] for info in agent._games.values()] == [1, 0] and not agent._free_envs
    agent.observe({"type": "aborted", "game_id": games[2].truth.id, "reason": "a seat left"})
    _next_decision(agent, games[3])  # the aborted game's slot is freed at this decision, then taken
    assert set(native._live) == {0, 2} and len(native.pool) == 2 and native.counters["dropped_over"] == 2
    assert [info["env"] for info in agent._games.values()] == [1, 0] and not agent._free_envs
    assert list(agent._games) == [games[1].truth.id, games[3].truth.id]
    assert agent.summary()["search"]["dropped_over"] == 2 and native.counters["dropped_first"] == 0


def test_an_error_in_the_agent_restarts_its_streams():
    """An act that fails empties the agent's pool and forgets its streams: the
    bridge counts the fallback, and the seat's next decision is a first
    again (it decides; no stream error)."""
    class Failing:
        def __init__(self, inner):
            self.obs_size, self.num_actions, self.num_players = inner.obs_size, inner.num_actions, inner.num_players

        def __call__(self, rows):
            raise MemoryError("evaluator failed")

    ls = lockstep("base-2p", 70)
    agent = _reusing(ls.dims)
    native = agent.native
    assert _own_game(agent, ls, seat=0, limit=3) == 3 and native._live
    healthy = native.evaluator
    native.evaluator = Failing(healthy)
    assert _own_game(agent, ls, seat=0, limit=1) == 1
    assert agent.stats["fallback:MemoryError"] == 1 and not native._live and len(native.pool) == 0
    native.evaluator = healthy
    assert _own_game(agent, ls, seat=0, limit=2) == 2
    assert agent.stats["fallback:MemoryError"] == 1 and len(native.pool) == 1
    assert native.counters["decisions"] == 5 and native.counters["dropped_first"] == 0


def test_a_reusing_agent_needs_a_caller_that_streams():
    """Only a caller that hands an agent its seat's streams (the ViewAgent,
    OpenHexSeat) may take a reusing one: another is refused up front, not at
    every move."""
    dims = lockstep("base-2p", 0).dims
    reusing = tiny_agent("MCTS", "full", dims, sims=8, dets=1, reuse=True)
    with pytest.raises(ValueError, match="streams"):
        native_agent(reusing)
    assert native_agent(reusing, memory=True)[0] is reusing
    assert ViewAgent(reusing).memory and reusing.reuse_check == "view"
    plain = tiny_agent("MCTS", "full", dims, sims=8, dets=1)
    assert not ViewAgent(plain).memory and plain.reuse_check == "exact"


PINNED = json.loads((DATA / "bridge_mcts_choices.json").read_text())


@pytest.mark.parametrize("name", sorted(PINNED))
def test_mcts_choices_without_reuse_are_pinned(name):
    """The ViewAgent's MCTS choices (sims 8, dets 1, no reuse) on a fixed random
    game, recorded before streams reached the bridge: the plain path is what it was."""
    pinned = PINNED[name]
    ls = lockstep(name, pinned["index"])
    agent = ViewAgent(tiny_agent("MCTS", "full", ls.dims, sims=8, dets=1), strict=True,
                      seed_fn=lambda view, seat: derive_seed(7, "decide", seat, len(view["action_records"])))
    keys = []
    while len(keys) < len(pinned["choices"]) and not ls.over:
        playable = ls.playable_json()
        if len(playable) > 1:
            keys.append(action_key(agent.decide(ls.view(ls.current_seat), playable)))
        ls.step(ls.choose())
    assert keys == pinned["choices"]


# ---------------------------------------------------------------- fairness and fallbacks


def test_unfair_agents_are_refused():
    dims = lockstep("base-3p", 0).dims
    with pytest.raises(UnfairAgent):
        ViewAgent(tiny_agent("POLICY", "full", dims))
    with pytest.raises(UnfairAgent):
        ViewAgent(tiny_agent("MCTS", "full", dims, oracle=True, dets=0, sims=8))
    with pytest.raises(UnfairAgent):
        ViewAgent(BotAgent("F"))
    # A public actor is fair with one other hand (it follows from the bank) only.
    public = ViewAgent(tiny_agent("POLICY", "public", dims), strict=True)
    ls = lockstep("base-3p", 0)
    while len(ls.playable_json()) < 2:
        ls.step(ls.choose())
    with pytest.raises(UnfairAgent):
        public.decide(ls.view(ls.current_seat), ls.playable_json())


def test_a_two_player_public_policy_plays():
    ls = lockstep("base-2p", 4)
    agent = ViewAgent(tiny_agent("POLICY", "public", ls.dims), strict=True)
    decided = 0
    while decided < 20 and not ls.over:
        playable = ls.playable_json()
        if len(playable) > 1:
            agent.decide(ls.view(ls.current_seat), playable)
            decided += 1
        ls.step(ls.choose())
    assert decided == 20


def _decision_point(name="base-3p", index=7):
    ls = lockstep(name, index)
    for _ in range(60):
        ls.step(ls.choose())
    while len(ls.playable_json()) < 2:
        ls.step(ls.choose())
    return ls


def test_a_chosen_action_that_was_not_offered_is_never_mapped():
    ls = _decision_point()
    view, playable = ls.view(ls.current_seat), ls.playable_json()

    def outside(agent):
        def act(rows, masks, seeds, **kwargs):
            return np.array([int(np.flatnonzero(masks[0] == 0)[0])])
        agent.native.act = act
        return agent

    strict = outside(ViewAgent(tiny_agent("POLICY", "private", ls.dims), strict=True))
    with pytest.raises(BridgeError):
        strict.decide(view, playable)
    lenient = outside(ViewAgent(tiny_agent("POLICY", "private", ls.dims)))
    assert lenient.decide(view, playable) is default_choice(playable)
    assert lenient.stats["fallback:not_offered"] == 1


def test_a_lone_offer_is_played_without_deciding():
    ls = _decision_point()
    agent = ViewAgent(tiny_agent("POLICY", "private", ls.dims), strict=True)
    only = ls.playable_json()[:1]
    assert agent.decide(ls.view(ls.current_seat), only) is only[0]
    assert agent.stats["forced"] == 1 and agent.stats["decisions"] == 0


def test_decision_seeds_are_a_function_of_the_view():
    """A sampling POLICY: fresh agents on the same view sample alike, and the
    seed is derive_seed(game id, "decide", seat, log length)."""
    ls = _decision_point("base-4p", 2)
    view, playable = ls.view(ls.current_seat), ls.playable_json()
    choices = {action_key(ViewAgent(tiny_agent("POLICY", "private", ls.dims, sample=True, temperature=5.0))
                          .decide(view, playable)) for _ in range(3)}
    assert len(choices) == 1
    agent = ViewAgent(tiny_agent("POLICY", "private", ls.dims))
    expected = derive_seed(view["game"]["id"], "decide", ls.current_seat, len(view["action_records"]))
    assert agent._seed(view, ls.current_seat) == expected


def _edited(ls, seat):
    """The seat's view with one wood moved from the bank to its hand behind
    the log's back (a TUI ``hand`` command)."""
    view = ls.view(seat)
    edited = dict(view)
    edited["resource_freqdeck"] = list(view["resource_freqdeck"])
    edited["resource_freqdeck"][0] -= 1
    edited["player_state"] = dict(view["player_state"])
    edited["player_state"][f"P{seat}_WOOD_IN_HAND"] += 1
    return edited


def _until(ls, prompt):
    """Play on until ``prompt`` is a decision of the seat to move."""
    while not ls.over:
        view = ls.view(ls.current_seat)
        if (view["current_prompt"] == prompt and not view["is_road_building"]
                and len(ls.playable_json()) > 1 and view["resource_freqdeck"][0] > 0):
            return ls.current_seat
        ls.step(ls.choose())
    raise AssertionError(f"no {prompt} decision")


@pytest.mark.parametrize("prompt", ["PLAY_TURN", "MOVE_ROBBER", "DISCARD"])
def test_an_edited_view_reanchors_on_its_exact_snapshot(prompt):
    """A view the log does not explain (a TUI edit): at any prompt the seat's
    replay restarts from the view's exact apply_setup snapshot, and the agent
    decides on it (strict: nothing inexact, no fallback)."""
    ls = _decision_point("base-4p", 3)
    seat = _until(ls, prompt)
    agent = ViewAgent(tiny_agent("POLICY", "private", ls.dims), strict=True)
    playable = ls.playable_json()
    choice = agent.decide(_edited(ls, seat), playable)
    assert any(choice is offered for offered in playable)
    tracker = next(iter(agent.seats.values()))
    assert tracker.stats["reanchored"] == 1 and tracker.position.exact
    assert tracker.game.player(seat).resources == tuple(
        n + (k == 0) for k, n in enumerate(ls.native.player(seat).resources))
