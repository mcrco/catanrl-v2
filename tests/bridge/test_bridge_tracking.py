"""The seat's hand tracker, fed from its views, against a HandTracker fed
natively (``observe(before, action, after)`` on the true cppanatron game) by
the same transitions, over random 2-6 player games: they must be equal, i.e.
agree on everything the seat can know."""

from __future__ import annotations

import os
from collections import Counter

import cppanatron as cpp
import pytest
from bridge_games import GAMES, lockstep
from cppanatron.catanatron import from_catanatron_action

from catanrl.bridge.replay import Replay, ReplayError
from catanrl.bridge.tracking import SeatTracker, transition_events

GAMES_PER_CONFIG = int(os.environ.get("CATANRL_BRIDGE_GAMES", "1"))
MAX_STEPS = int(os.environ.get("CATANRL_BRIDGE_STEPS", "600"))


def _play(name: str, index: int, *, every: int = 1, friendly_robber: bool | None = None) -> Counter:
    """A random game with a SeatTracker and a native HandTracker per seat;
    the trackers are compared at every sync. Returns the trackers' stats."""
    ls = lockstep(name, index, friendly_robber=friendly_robber)
    n = len(ls.colors)
    bridged = [SeatTracker(view) for view in ls.views()]
    native = [cpp.HandTracker(ls.native, seat) for seat in range(n)]
    for step in range(MAX_STEPS):
        if ls.over:
            break
        action = ls.choose()
        before = ls.native.copy()
        ls.step(action)
        for seat in range(n):  # the native trackers see every transition
            native[seat].observe(before, from_catanatron_action(action), ls.native)
        over = ls.truth.winning_color() is not None
        syncing = [seat for seat in range(n) if not step % every or over or ls.current_seat == seat]
        views = ls.views() if len(syncing) > 1 else {seat: ls.view(seat) for seat in syncing}
        for seat in syncing:
            tracker = bridged[seat]
            tracker.sync(views[seat], game_over=over)
            if not over and ls.current_seat == seat:
                # What the seat is offered (natively applied at every transition to it).
                tracker.options([from_catanatron_action(a) for a in ls.truth.playable_actions])
            assert not tracker.mismatches, tracker.mismatches  # re-anchored if it had to
            tracker.tracker.check_public(tracker.game)
            if tracker.stats["tracker_resets"] or tracker.stats["reanchored"]:
                # Restarted mid-game (a decide-only stream): it knows less, but it is sound.
                if not over:
                    assert tracker.tracker.consistent_with(ls.native), (name, index, step, seat)
            else:
                assert tracker.tracker == native[seat], (name, index, step, seat)
    stats: Counter = Counter()
    for tracker in bridged:
        stats.update(tracker.stats)
        stats.update({f"repair:{k}": v for k, v in tracker.replay.repairs.items()})
    return stats


@pytest.mark.parametrize("name", GAMES)
def test_tracker_equals_the_native_tracker(name):
    stats = Counter()
    for index in range(GAMES_PER_CONFIG):
        stats += _play(name, 10 + index)
    assert stats["transitions"] > 300
    assert not stats["broken"] and not stats["tracker_resets"] and not stats["mismatched_syncs"]
    assert not stats["reanchored"]


def test_tracker_equals_the_native_tracker_with_the_friendly_robber():
    """The seat's robber options bound the others' victory point cards, and
    the replay's placeholders are redealt to offer the same options."""
    stats = Counter()
    for name, index in (("base-3p", 1), ("base-4p", 0), ("extension-5p", 1)):
        stats += _play(name, index, friendly_robber=True)
    assert stats["robber_options_redealt"] > 0
    assert not stats["broken"] and not stats["tracker_resets"]


def test_monopolies_between_sparse_views_come_from_hand_sizes():
    """Views only every few steps (the TUI path): a monopoly's takes are
    solved from the sizes at the next view; the tracker stays equal to the
    native one, or (two monopolies between views) restarts soundly."""
    stats = Counter()
    for name, index in (("base-3p", 1), ("base-4p", 1), ("extension-5p", 1), ("extension-6p", 2)):
        stats += _play(name, index, every=9)
    assert stats["monopolies_from_sizes"] > 0


def test_events_of_a_hidden_steal_and_a_purchase():
    """Steals between two other seats carry no resource; purchases carry the
    card only for the buyer; both match the engine's own derivation."""
    ls = lockstep("base-4p", 5)
    replay = Replay(ls.view(0), 0, ls.native.rules, ls.native.board.map)
    seen = Counter()
    for _ in range(MAX_STEPS):
        if ls.over:
            break
        action = ls.choose()
        before = ls.native.copy()
        ls.step(action)
        records = ls.view(0)["action_records"]
        tr = replay.step(records[-1])
        ours = transition_events(tr, 0)
        native = cpp.HandTracker.events(before, from_catanatron_action(action), ls.native, 0)
        if tr.kind == "PLAY_MONOPOLY":
            continue  # takes come from the view's sizes (see the other tests)
        assert ours == native[:len(ours)], (tr.kind, ours, native)
        if tr.kind == "MOVE_ROBBER" and tr.victim is not None:
            seen["steal_seen" if 0 in (tr.player, tr.victim) else "steal_hidden"] += 1
            assert (ours[0].resource is None) == (0 not in (tr.player, tr.victim))
        if tr.kind == "BUY_DEVELOPMENT_CARD":
            seen["mine" if tr.player == 0 else "theirs"] += 1
    assert seen["steal_hidden"] and seen["theirs"]


def test_a_log_from_another_game_does_not_replay():
    ls = lockstep("base-3p", 0)
    for _ in range(40):
        ls.step(ls.choose())
    other = lockstep("base-3p", 1)
    for _ in range(40):
        other.step(other.choose())
    replay = Replay(ls.view(0), 0, ls.native.rules, ls.native.board.map)
    with pytest.raises(ReplayError):
        for entry in other.view(0)["action_records"]:
            replay.step(entry)


def test_a_rewound_log_restarts_the_replay():
    """A TUI undo shortens the log: the tracker replays from the start."""
    ls = lockstep("base-3p", 3)
    for _ in range(150):
        ls.step(ls.choose())
    view = ls.view(0)
    tracker = SeatTracker(view)
    tracker.sync(view)
    earlier = dict(view)
    earlier["action_records"] = view["action_records"][:-5]
    tracker.sync(earlier)
    assert tracker.stats["restarts"] == 1 and tracker.replay.applied == len(earlier["action_records"])
