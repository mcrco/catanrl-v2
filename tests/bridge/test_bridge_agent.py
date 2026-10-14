"""ViewAgent: the same decisions as the native agent on the same positions
and seeds (POLICY, MCTS dets=1 sims=8, BELIEF worlds=2); decisions that
depend on the view alone (paired games whose views are identical but whose
hidden cards differ); fairness and fallback rules."""

from __future__ import annotations

import random

import cppanatron as cpp
import numpy as np
import pytest
from bridge_games import lockstep, tiny_agent
from catanatron.models.enums import ActionRecord, ActionType
from catanatron.serialization import action_from_json, action_to_json, client_view, state_to_json
from cppanatron.catanatron import from_catanatron_action, to_catanatron_action

from catanrl.agents.native import BotAgent
from catanrl.bridge.agent import BridgeError, UnfairAgent, ViewAgent, action_key, default_choice
from catanrl.seeding import derive_seed

CASES = {
    "POLICY": ("private", {}),
    "MCTS": ("full", {"sims": 8, "dets": 1}),
    "BELIEF": ("full", {"worlds": 2}),
}


def _native_choice(native, game, seat, seed, tracker):
    row = native.encode(game, seat)
    size = native.action_space.size
    index = native.act(row[None], row[None, -size:], np.array([seed], dtype=np.uint64),
                       games=[game] if native.needs_games else None,
                       trackers=[tracker] if native.needs_trackers else None)[0]
    return action_to_json(to_catanatron_action(native.action_space.decode(game, int(index))))


@pytest.mark.parametrize("key", sorted(CASES))
@pytest.mark.parametrize("name", ["base-2p", "base-3p"])
def test_view_agent_decides_like_the_native_agent(key, name):
    """Every decision of every seat of a random game: the native agent on the
    true game and the seat's true tracker, the ViewAgent on the seat's views
    only (decide-only, as a TUI callback), with the same decision seeds."""
    level, params = CASES[key]
    index = sorted(CASES).index(key)
    ls = lockstep(name, 20 + index)
    n = len(ls.colors)
    native = tiny_agent(key, level, ls.dims, **params)

    def seed(ply: int, seat: int) -> int:
        return derive_seed(index, "decide", seat, ply)

    ours = ViewAgent(tiny_agent(key, level, ls.dims, **params), strict=True,
                     seed_fn=lambda view, seat: seed(len(view["action_records"]), seat))
    trackers = [cpp.HandTracker(ls.native, s) for s in range(n)]
    compared, limit = 0, 30 if key == "MCTS" else 80
    while compared < limit and not ls.over:
        seat = ls.current_seat
        playable = ls.playable_json()
        if len(playable) > 1:
            view = ls.view(seat)
            mine = ours.decide(view, playable)
            assert any(mine is offered for offered in playable)  # the offered object itself
            theirs = _native_choice(native, ls.native, seat, seed(len(view["action_records"]), seat),
                                    trackers[seat])
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
    level, params = CASES[key]
    pairs = PairedGames("base-4p", 30 + sorted(CASES).index(key), seat=1)
    dims = pairs.a.dims
    agent_a = ViewAgent(tiny_agent(key, level, dims, **params), strict=True)
    agent_b = ViewAgent(tiny_agent(key, level, dims, **params), strict=True)
    decided, limit = 0, 15 if key == "MCTS" else 40
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
    while not ls.over:
        view = ls.view(ls.current_seat)
        if (view["current_prompt"] == prompt and not view["is_road_building"]
                and len(ls.playable_json()) > 1 and view["resource_freqdeck"][0] > 0):
            return ls.current_seat
        ls.step(ls.choose())
    raise AssertionError(f"no {prompt} decision")


def test_an_edited_turn_position_reanchors_on_the_snapshot():
    """A view the log does not explain: at a turn position the seat's replay
    restarts from the apply_setup snapshot of the view."""
    ls = _decision_point("base-3p", 9)
    seat = _until(ls, "PLAY_TURN")
    agent = ViewAgent(tiny_agent("POLICY", "private", ls.dims), strict=True)
    playable = ls.playable_json()
    choice = agent.decide(_edited(ls, seat), playable)
    assert any(choice is offered for offered in playable)
    tracker = next(iter(agent.seats.values()))
    assert tracker.stats["reanchored"] == 1 and tracker.position.exact


def test_an_edited_robber_position_is_refused_or_counted():
    """Elsewhere no snapshot can express the view: strict refuses; lenient
    decides on the replay as it is, or (if the replay broke on it) plays the
    default move, and counts either."""
    ls = _decision_point("base-4p", 3)
    seat = _until(ls, "MOVE_ROBBER")
    edited, playable = _edited(ls, seat), ls.playable_json()
    with pytest.raises(BridgeError):
        ViewAgent(tiny_agent("POLICY", "private", ls.dims), strict=True).decide(edited, playable)
    lenient = ViewAgent(tiny_agent("POLICY", "private", ls.dims))
    choice = lenient.decide(edited, playable)
    assert any(choice is offered for offered in playable)
    assert lenient.stats["inexact"] + lenient.stats["fallback:no_position"] == 1
