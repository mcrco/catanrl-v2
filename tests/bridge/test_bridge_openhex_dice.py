"""OpenHex's balanced dice through the bridge (its 1v1 rules, ``diceMode:
balanced``): the replayed game rolls them, every observed roll moves its public
dice state as OpenHex's draw did, and a roll that state cannot give sends the
seat back over the game on fair dice.

* A balanced game drawn by the engine on a recorded board, written as OpenHex
  events (rolls, production, discards, the robber, turn ends) for 80 rolls,
  three decks: each seat's replay holds the generator's dice state at every
  roll, as does the exact reference (tests/dice_reference.py) fed the same
  cards, and never desyncs; a view's deck size (``balancedDice.remaining``,
  all OpenHex shows of it) is compared.
* The two recorded 1v1 games (``diceMode: balanced``, fair dice in fact, see
  tests/test_dice_reference.py): the replay follows their rolls with the
  reference's state up to the first roll the reference refuses (roll 10 and
  roll 14), refuses exactly that one, and goes on over the game on fair dice;
  a reusing agent's stream restarts there. Neither record carries OpenHex's
  deck size or draw count, so only the cards themselves check the refills.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import cppanatron as cpp
import pytest

pytest.importorskip("catanatron")

from dice_reference import BalancedDice

from catanrl.bridge.openhex import RES_NAME, OpenHexError, OpenHexSeat, color_of, openhex_rules

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openhex"
RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")
#: The first roll each record's dice cannot have given (the reference agrees).
FIRST_IMPOSSIBLE = {"game-004a8250.json": 10, "game-881a71bc.json": 14}


def _record(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _redacted(events, color):
    me = color_of(color)
    out = []
    for item in events:
        event = dict(item["event"])
        if event["type"] == "devCardBought" and color_of(event["player"]) != me:
            event.pop("card", None)
        if event["type"] == "resourceStolen" and me not in (color_of(event["player"]), color_of(event["target"])):
            event.pop("resource", None)
        out.append({"seq": item["seq"], "event": event})
    return out


def _as_reference(dice) -> BalancedDice:
    return BalancedDice(sevens=dice.sevens, deck=frozenset(dice.deck), recent=dice.recent,
                        streak_player=dice.streak_player, streak_count=dice.streak_count)


def _rolls_applied(seat) -> int:
    return sum(1 for e in seat.entries if e.record[0][1] == "ROLL")


class _Never:
    """An agent that must not be asked: the replay alone is under test."""

    dims: ClassVar[dict] = {}
    action_space = None


class _Forgetful(_Never):
    """A reusing agent that is never asked to decide: records the streams it is told to drop."""

    needs_memory = True

    def __init__(self) -> None:
        self.forgotten: list[int] = []

    def forget(self, env: int) -> None:
        self.forgotten.append(env)


def test_the_rules_follow_dice_mode():
    for config, mode in [({"diceMode": "balanced"}, cpp.DiceMode.BALANCED), ({"diceMode": "random"}, cpp.DiceMode.RANDOM),
                         ({"diceMode": "BALANCED"}, cpp.DiceMode.BALANCED), ({}, cpp.DiceMode.RANDOM)]:
        assert openhex_rules(config).dice_mode == mode
    assert openhex_rules({"diceMode": "balanced"}, fair_dice=True).dice_mode == cpp.DiceMode.RANDOM
    with pytest.raises(OpenHexError, match="diceMode"):
        openhex_rules({"diceMode": "loaded"})
    record = _record("game-004a8250.json")
    assert record["config"]["diceMode"] == "balanced"
    seat = OpenHexSeat(_Never(), "fixture", record, "blue")
    assert seat.rules.dice_mode == cpp.DiceMode.BALANCED and seat.replay.game.balanced_dice == cpp.BalancedDice.fresh(2)


# ---------------------------------------------------------------- a balanced game drawn by the engine


def _balanced_game(rolls: int = 80):
    """A balanced game on 004a8250's board after its recorded setup, drawn by the
    engine (seat order blue, red; each seat rolls, discards the first cards
    offered, moves the robber to the first tile offered and ends its turn), as
    OpenHex events; and the dice state, card and roller after every roll."""
    record = _record("game-004a8250.json")
    setup = [e for e in record["events"] if e["seq"] < 10]
    assert all(e["event"]["type"].startswith("setup") for e in setup)
    builder = OpenHexSeat(_Never(), "generator", record, "blue")
    builder.feed(setup)
    game = builder.replay.game.copy()
    assert not game.is_initial_build_phase and game.dice_mode == cpp.DiceMode.BALANCED
    colors = [c.name.lower() for c in game.colors]
    events: list[dict] = list(setup)
    after: list[tuple] = []
    discards: dict[str, dict[str, int]] = {}

    def emit(event):
        events.append({"seq": len(events), "event": event})

    def flush_discards():
        for player, cards in discards.items():
            emit({"type": "resourcesDiscarded", "player": player, "resources": cards})
        discards.clear()

    while True:
        actions = game.playable_actions
        kinds = {a.action_type for a in actions}
        player = colors[game.current_player_index]
        if cpp.ActionType.ROLL in kinds:
            hands = [tuple(p.resources) for p in game.players]
            drawable = frozenset(game.balanced_dice.drawable)
            game.execute(next(a for a in actions if a.action_type == cpp.ActionType.ROLL))
            (card,) = drawable - frozenset(game.balanced_dice.deck)
            emit({"type": "diceRolled", "player": player, "die1": card[0], "die2": card[1]})
            production = {colors[s]: {r: n - h for r, n, h in zip(RESOURCE_NAMES, p.resources, hands[s]) if n > h}
                          for s, p in enumerate(game.players)}
            production = {c: got for c, got in production.items() if got}
            if production:
                emit({"type": "resourcesProduced", "production": production})
            after.append((game.balanced_dice, card, colors.index(player)))
        elif game.current_prompt == cpp.ActionPrompt.DISCARD:
            action = actions[0]
            cards = discards.setdefault(player, {})
            cards[RES_NAME[action.value.name]] = cards.get(RES_NAME[action.value.name], 0) + 1
            game.execute(action)
        elif cpp.ActionType.MOVE_ROBBER in kinds:
            flush_discards()
            action = next(a for a in actions if a.action_type == cpp.ActionType.MOVE_ROBBER)
            hands = [tuple(p.resources) for p in game.players]
            game.execute(action)
            x, y, z = action.value[0]
            emit({"type": "robberMoved", "player": player, "hex": {"x": x, "y": y, "z": z}})
            if action.value[1] is not None:
                victim = colors.index(action.value[1].name.lower())
                (resource,) = [r for r, n, h in zip(RESOURCE_NAMES, game.player(victim).resources, hands[victim])
                               if n < h]
                emit({"type": "resourceStolen", "player": player, "target": colors[victim], "resource": resource})
        else:
            flush_discards()
            game.execute(next(a for a in actions if a.action_type == cpp.ActionType.END_TURN))
            emit({"type": "turnEnded", "player": player})
            if len(after) >= rolls:
                return record, events, after, game


@pytest.mark.parametrize("color", ["blue", "red"])
def test_a_balanced_game_replays_with_its_dice_state(color):
    record, events, after, generator = _balanced_game()
    assert sum(1 for s, _c, _r in after if s.remaining == 13) == 3  # three decks drawn down to the refill
    assert any(card[0] + card[1] == 7 for _s, card, _r in after)
    assert any(e["event"]["type"] == "resourcesDiscarded" for e in events)
    seat = OpenHexSeat(_Never(), "balanced", record, color)
    reference = BalancedDice.new(2)
    checked = 0
    for item in _redacted(events, color):
        seat.feed([item])
        while checked < _rolls_applied(seat):
            state, card, roller = after[checked]
            reference = reference.apply(card, roller)
            checked += 1
            assert seat.replay.game.balanced_dice == state, f"roll {checked - 1}"
            assert _as_reference(state) == reference
            assert seat.replay.game.roll_probabilities() == state.sum_probabilities(seat.replay.game.next_roller)
    assert checked == len(after) and "dice_desync" not in seat.stats
    game = seat.replay.game
    assert game.balanced_dice == generator.balanced_dice
    assert tuple(game.resource_bank) == tuple(generator.resource_bank)
    assert [sum(p.resources) for p in game.players] == [sum(p.resources) for p in generator.players]
    # A view shows the deck size only: compared, and a different one reported.
    view = {"players": [], "board": {}, "balancedDice": {"remaining": generator.balanced_dice.remaining}}
    assert seat.mismatches(view) == []
    view["balancedDice"]["remaining"] += 1
    assert [p.split(" ")[0] for p in seat.mismatches(view)] == ["dice"]


# ---------------------------------------------------------------- the recorded games, as recorded


def _openhex_rolls(record):
    seats = list(record["turnOrder"])
    return [(seats.index(e["event"]["player"]), (e["event"]["die1"], e["event"]["die2"]))
            for e in sorted(record["events"], key=lambda e: e["seq"]) if e["event"]["type"] == "diceRolled"]


@pytest.mark.parametrize("name", sorted(FIRST_IMPOSSIBLE))
@pytest.mark.parametrize("color", ["blue", "red"])
def test_a_recorded_game_follows_its_dice_until_a_roll_they_cannot_give(name, color):
    record = _record(name)
    rolls = _openhex_rolls(record)
    first = FIRST_IMPOSSIBLE[name]
    reference = BalancedDice.new(2, record["config"]["numPlayers"])
    for roller, card in rolls[:first]:
        assert reference.card_probabilities(roller)[card] > 0
        reference = reference.apply(card, roller)
    assert reference.card_probabilities(rolls[first][0])[rolls[first][1]] == 0
    # 881a71bc's replay stops at seq 380 (a roll OpenHex withheld, test_bridge_openhex.py): to its checkpoint.
    end = record["checkpoints"][0]["seq"] if "checkpoints" in record else max(e["seq"] for e in record["events"])
    events = [e for e in _redacted(record["events"], color) if e["seq"] <= end]
    native = _Forgetful()
    seat = OpenHexSeat(native, "fixture", record, color)
    env = seat.env
    state = BalancedDice.new(2)
    checked = 0
    for item in events:
        seat.feed([item])
        if "dice_desync" in seat.stats:
            break
        while checked < _rolls_applied(seat):
            state = state.apply(rolls[checked][1], rolls[checked][0])
            checked += 1
        assert seat.replay.game.balanced_dice is not None and _as_reference(seat.replay.game.balanced_dice) == state
    assert checked == first and seat.stats["dice_desync"] == first
    assert seat.fair_dice and seat.rules.dice_mode == cpp.DiceMode.RANDOM and seat.replay.game.balanced_dice is None
    assert native.forgotten == [env]  # the stream restarts with the game replayed again
    seat.feed(events)
    assert seat.stats["dice_desync"] == first  # on fair dice from there: no other refusal
    if "checkpoints" in record:
        checkpoint = record["checkpoints"][0]
        assert seat.bank == [checkpoint["bank"][r] for r in RESOURCE_NAMES] and seat.stats["bank_short"] == 1
    else:
        final = record["final"]
        assert seat.over and seat.log.won == color_of(final["winner"])
        assert seat.bank == [final["bank"][r] for r in RESOURCE_NAMES]
        for player in final["players"]:
            i = seat.colors.index(color_of(player["color"]))
            assert sum(seat.replay.game.player(i).resources) == sum(player["hand"].values())
    seat.close()  # (a game replayed again on a larger bank restarts the stream too)
    assert native.forgotten == [env] * (2 + seat.stats.get("bank_short", 0))
