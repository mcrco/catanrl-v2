"""The seat's hand tracker, driven by what its views show.

:class:`SeatTracker` follows one seat of one game. Each new log entry of a
view is replayed (:class:`catanrl.bridge.replay.Replay`) and turned into the
``cpp.TrackerEvent``\\ s the seat saw, exactly as ``HandTracker.events`` derives
them from true states (:func:`transition_events`):

* resources moving between hands and the bank in the open (production from
  the dice and the board, building costs, purchases, bank and port trades,
  discards, year of plenty, the second settlement's yield): ``lose`` then
  ``gain`` per seat, from the replay's per-seat differences, which for these
  moves are public;
* the robber: ``steal`` with the resource when the seat was thief or victim
  (the log shows it), without it between two other seats;
* monopoly: ``play_development_card`` and ``monopoly`` with every seat's
  take, from the public hand sizes after the move;
* purchases (``buy_development_card``, the card only for the seat's own) and
  plays of development cards, ``end_turn``;
* after every move that does not end the game, ``public_victory_points``;
* at a decision, ``observe_options`` with the actions the seat was offered
  (under the friendly robber they bound the others' victory point cards).

Every step is checked: ``tracker.check_public(game)`` (hand sizes, bank,
deck, played cards, the seat's own cards) after each transition, the replay's
hand sizes against the view's when the view after a step is known (a server
``step`` message, :meth:`SeatTracker.record`), and at each sync every public
field of the view (:func:`catanrl.bridge.position.public_mismatches`).

A monopoly's takes are public, but a log entry does not carry them. With the
view after every step (the server path) they are exact. With views only at
decisions (TUI ``decide_fns``) one monopoly between two decisions is solved
from the hand sizes at the next view; for two, the tracker restarts from the
replay (sound: it forgets the others' cards, nothing more), counted in
``stats["tracker_resets"]``.

A view the replay does not explain (a TUI edit, a log the replay cannot
follow) re-anchors it on the view's own position, which ``apply_setup``
declares exactly at every prompt (:func:`catanrl.bridge.position.snapshot`).

``mode`` is the tracker's ``cpp.TrackerMode``: exact keeps the posterior over
hidden steals from where the tracker (re)started, which a mid-game start
only enumerates when the unknown cards' deals are few.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import cppanatron as cpp

from catanrl.bridge.position import (
    Inexpressible,
    Position,
    SeatView,
    ViewError,
    game_rules,
    public_mismatches,
    seat_of,
    shared_map,
    snapshot,
)
from catanrl.bridge.replay import PLAYED_CARD, Replay, ReplayError, Transition

Event = cpp.TrackerEvent
logger = logging.getLogger(__name__)


def transition_events(tr: Transition, observer: int, *, amounts: Sequence[int] | None = None) -> list[Event]:
    """The events ``observer`` sees in one replayed transition (without the
    public victory points that follow every move).

    The order is ``HandTracker.events``'. ``amounts`` overrides a
    monopoly's takes per seat (when they were not known at the step).
    """
    p = tr.player
    kind = tr.kind
    if kind == "MOVE_ROBBER":
        if tr.victim is None:
            return []
        seen = observer in (p, tr.victim)
        resource = cpp.Resource[tr.stolen] if seen and tr.stolen is not None else None
        return [Event.steal(p, tr.victim, resource)]
    if kind == "PLAY_MONOPOLY":
        if amounts is None:  # every other seat lost exactly its take: public hand sizes
            amounts = [0 if s == p else tr.before.sizes[s] - tr.after.sizes[s]
                       for s in range(len(tr.before.resources))]
        return [Event.play_development_card(p, cpp.DevelopmentCard.MONOPOLY),
                Event.monopoly(p, cpp.Resource[tr.resource], [int(a) for a in amounts])]
    if kind == "CONFIRM_TRADE":
        value = tr.entry[0][2]
        if tr.partner is None or tr.partner == p:
            return []
        offering, asking = [int(x) for x in value[:5]], [int(x) for x in value[5:10]]
        events = []
        if sum(offering):
            events.append(Event.transfer(p, tr.partner, offering))
        if sum(asking):
            events.append(Event.transfer(tr.partner, p, asking))
        return events
    events: list[Event] = []
    if kind in PLAYED_CARD and kind != "PLAY_MONOPOLY":
        events.append(Event.play_development_card(p, cpp.DevelopmentCard[PLAYED_CARD[kind]]))
    n = len(tr.before.resources)
    deltas = [[tr.after.resources[s][k] - tr.before.resources[s][k] for k in range(5)] for s in range(n)]
    for s in range(n):
        lost = [max(0, -d) for d in deltas[s]]
        if sum(lost):
            events.append(Event.lose(s, lost))
    for s in range(n):
        gained = [max(0, d) for d in deltas[s]]
        if sum(gained):
            events.append(Event.gain(s, gained))
    if kind == "BUY_DEVELOPMENT_CARD":
        card = cpp.DevelopmentCard[tr.card] if p == observer else None
        events.append(Event.buy_development_card(p, card))
    elif kind == "END_TURN":
        events.append(Event.end_turn(p))
    return events


def points_event(tr: Transition) -> Event:
    return Event.public_victory_points([int(x) for x in tr.after.points])


class SeatTracker:
    """One seat of one game: the replayed position and the seat's tracker.

    ``sync(view)`` replays the view's new log entries; ``position`` is then
    the seat's game (``exact`` when every public field matches the view),
    ``tracker`` its ``cpp.HandTracker``. Feed ``record(index, state)`` with
    every server ``step`` (or any later view) to make monopolies exact.
    A log that no longer extends the replayed one (a TUI undo, another game)
    restarts the replay. A log the replay cannot follow, or a view it does
    not explain (a TUI edit, or the rare monopoly whose takes a decide-only
    stream cannot recover before a discard depends on them), re-anchors the
    replay on the view's exact ``apply_setup`` snapshot.
    """

    def __init__(self, view: Mapping[str, Any], color: Any = None, *, supply: Any = None,
                 rules: cpp.GameRules | None = None, mode: cpp.TrackerMode = cpp.TrackerMode.POOL) -> None:
        self.seat = seat_of(view, color)
        #: The ``cpp.HandTracker`` mode (exact keeps the posterior over hidden steals).
        self.mode = mode
        self.color = view["colors"][self.seat]
        self.colors = tuple(view["colors"])
        self.game_id = (view.get("game") or {}).get("id")
        self.rules = rules if rules is not None else game_rules(view, supply)
        self.catan_map = shared_map(view["map"])
        #: Public hand sizes after ``index`` log entries, from views seen in between.
        self.sizes: dict[int, tuple[int, ...]] = {}
        self.stats: Counter[str] = Counter()
        self.broken: str | None = None
        self.mismatches: tuple[str, ...] = ()
        self._start(view)

    def _start(self, view: Mapping[str, Any]) -> None:
        self.replay = Replay(view, self.seat, self.rules, self.catan_map)
        self.tracker = cpp.HandTracker(self.replay.game, self.seat, self.mode)
        self.broken = None
        self.mismatches = ()

    @property
    def game(self) -> cpp.Game:
        return self.replay.game

    @property
    def position(self) -> Position:
        """The seat's game (method ``"broken"`` when the replay is stuck)."""
        method = "broken" if self.broken is not None else "replay"
        return Position(self.replay.game, self.seat, method, self.mismatches)

    def record(self, index: int, state: Mapping[str, Any]) -> None:
        """The view (or server message ``state``) after ``index`` log entries."""
        try:
            self.sizes[int(index)] = SeatView(state, self.seat).hand_sizes()
        except (KeyError, ViewError):
            pass

    def sync(self, view: Mapping[str, Any], *, game_over: bool = False) -> Position:
        """Replay the view's new log entries and check the result against it.

        ``game_over``: the view's last entry ended the game (no public
        victory points follow it, as in ``HandTracker.events``). A replay
        that breaks or differs from the view re-anchors on the view's exact
        ``apply_setup`` snapshot (``stats["reanchored"]``; the tracker starts
        afresh there, which is sound).
        """
        records = view["action_records"]
        self.record(len(records), view)
        if not self.replay.follows(records):
            self.stats["restarts"] += 1
            self._start(view)
        if self.broken is None:
            try:
                self._advance(records, view, game_over)
            except (ReplayError, RuntimeError, ValueError) as error:
                self.broken = str(error)
                self.stats["broken"] += 1
                logger.warning("seat %s: the replay broke: %s", self.color, error)
            else:
                self.mismatches = tuple(public_mismatches(self.replay.game, view, self.seat))
                if self.mismatches:
                    self.stats["mismatched_syncs"] += 1
        if self.broken is not None or self.mismatches:
            self._reanchor(view, records)
        return self.position

    def _reanchor(self, view: Mapping[str, Any], records: Sequence[Any]) -> None:
        try:
            anchor = snapshot(view, color=self.color, rules=self.rules, catan_map=self.catan_map)
        except (Inexpressible, ViewError, ValueError) as error:
            self.mismatches = (f"no snapshot: {error}",) + self.mismatches
            return
        if anchor.mismatches:
            self.stats["inexact_snapshots"] += 1
            if self.broken is None:
                return  # keep the replay; its own mismatches stand
        self.replay = Replay(view, self.seat, self.rules, self.catan_map, game=anchor.game, entries=records)
        self.tracker = cpp.HandTracker(anchor.game, self.seat, self.mode)
        self.broken = None
        self.mismatches = anchor.mismatches
        self.stats["reanchored"] += 1

    def _advance(self, records: Sequence[Any], view: Mapping[str, Any], game_over: bool) -> None:
        pending: list[tuple[Transition, list[Event]]] | None = None
        unsettled: list[int] = []
        start = self.replay.applied
        for k in range(start, len(records)):
            last = k + 1 == len(records)
            # After a monopoly with unknown takes the replay's sizes are off
            # until _settle moves the placeholders: check them only then.
            known = self.sizes.get(k + 1) if pending is None else None
            tr = self.replay.step(records[k], sizes_after=known)
            events = transition_events(tr, self.seat)
            if not (game_over and last):
                events.append(points_event(tr))
            if not tr.exact_amounts:
                unsettled.append(0 if pending is None else len(pending))
                if pending is None:
                    pending = []
            if pending is None:
                for event in events:
                    self.tracker.apply(event)
                self.tracker.check_public(self.replay.game)
                self.stats["transitions"] += 1
            else:
                pending.append((tr, events))
        if pending is not None:
            self._settle(pending, unsettled, view)

    def _settle(self, pending: list[tuple[Transition, list[Event]]], unsettled: list[int],
                view: Mapping[str, Any]) -> None:
        """Monopolies whose takes were unknown: solve one from the view's hand
        sizes, or restart the tracker."""
        sizes = SeatView(view, self.seat).hand_sizes()
        replayed = tuple(sum(p.resources) for p in self.replay.game.players)
        delta = [int(a) - int(b) for a, b in zip(sizes, replayed)]
        self.replay.resize(sizes)
        if len(unsettled) == 1:
            i = unsettled[0]
            tr, events = pending[i]
            guessed = [0 if s == tr.player else tr.before.sizes[s] - tr.after.sizes[s]
                       for s in range(len(self.colors))]
            amounts = [0 if s == tr.player else guessed[s] - delta[s] for s in range(len(self.colors))]
            if min(amounts) >= 0:
                fixed = transition_events(tr, self.seat, amounts=amounts) + events[2:]
                pending[i] = (tr, fixed)
                try:
                    trial = self.tracker.copy()
                    for _, evs in pending:
                        for event in evs:
                            trial.apply(event)
                    trial.check_public(self.replay.game)
                except (RuntimeError, ValueError):
                    pass
                else:
                    self.tracker = trial
                    self.stats["monopolies_from_sizes"] += 1
                    self.stats["transitions"] += len(pending)
                    return
        self.tracker = cpp.HandTracker(self.replay.game, self.seat, self.mode)
        self.stats["tracker_resets"] += 1

    def options(self, legal: Sequence[cpp.Action]) -> None:
        """The seat is to move with these actions (what it was offered).

        Under the friendly robber they bound the others' victory point cards
        (``observe_options``), and the replay must offer the same robber
        moves: its placeholders then get victory point cards within the
        tracker's bounds.
        """
        if self.broken is not None:
            return
        legal = list(legal)
        self.tracker.observe_options(self.replay.game, legal)
        wanted = sorted(map(repr, legal))
        if not self.rules.friendly_robber or sorted(map(repr, self.replay.game.playable_actions)) == wanted:
            return
        others = [s for s in range(len(self.colors)) if s != self.seat]
        try:
            self.replay.match_points({s: self.tracker.victory_point_card_floor(s) for s in others},
                                     {s: self.tracker.victory_point_card_cap(s) for s in others})
        except ReplayError:
            pass
        matched = sorted(map(repr, self.replay.game.playable_actions)) == wanted
        self.stats["robber_options_redealt" if matched else "robber_options_differ"] += 1
