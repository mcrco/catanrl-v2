"""The LLM agent: specs, registry, stub games through the Catanatron harness and
``catanrl eval``, the seat's view of the game (the state, the history, the actions),
this game's rules in every prompt, answer parsing, the request shape, token, cost and
budget accounting, and the llm-eval experiment's stub cells. No network: every model
here is a stub or a fake client, and a real model refuses to build without
OPENROUTER_API_KEY."""

from __future__ import annotations

import json
import re
import threading
from types import SimpleNamespace

import pytest
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
    Action,
    ActionRecord,
    ActionType,
)
from catanatron.models.player import Color
from catanatron.registry import REGISTRY, SpecError
from catanatron.state_functions import get_player_freqdeck

from catanrl.agents import check_spec_syntax, resolve_agent
from catanrl.agents.llm import (
    LLMError,
    LlmPlayer,
    StubModel,
    parse_answer,
    render_state,
    system_prompt,
)
from catanrl.agents.llm import budget as llm_budget
from catanrl.agents.llm.model import OpenRouterModel, Reply, estimate_tokens, make_model
from catanrl.agents.llm.player import NOTE_CHARS, AnswerError
from catanrl.agents.llm.render import (
    CARD_LABELS,
    describe_action,
    event_text,
    node_text,
    player_status,
    render_actions,
    render_events,
    road_distances,
)
from catanrl.eval.catanatron import play_catanatron, reseat
from catanrl.eval.suites import load_suite
from catanrl.games import load_game

SPEC = "LLM:model=stub/random:strategy=plan"
SUITE = {"name": "llm-test", "purpose": "dev", "seed": 11, "boards": 1, "scenarios": [{"game": "base-2p"}],
         "turn_limit": 200}
CUBE = re.compile(r"\(-?\d+, -?\d+, -?\d+\)")


@pytest.fixture(autouse=True)
def _fresh_budget(monkeypatch):
    """No cap, no ledger and a fresh process budget per test."""
    for name in (llm_budget.BUDGET_ENV, llm_budget.LEDGER_ENV, llm_budget.FALLBACK_ENV):
        monkeypatch.delenv(name, raising=False)
    llm_budget.reset()
    yield
    llm_budget.reset()


def two_player_game(red, blue, *, map_seed=3, seed=5, game="base-2p"):
    game = load_game(game).catanatron_game([red, blue], map_seed=map_seed, seed=seed)
    reseat(game, [red, blue])
    return game


def random_game(*, seed=5, map_seed=3, game="base-2p", colors=(Color.RED, Color.BLUE)):
    players = [REGISTRY.build("R", c) for c in colors]
    g = load_game(game).catanatron_game(players, map_seed=map_seed, seed=seed)
    reseat(g, players)
    return g


def play_until(game, done, limit=5000):
    for _ in range(limit):
        if done(game) or game.winning_color() is not None:
            return game
        game.play_tick()
    return game


# ---------------------------------------------------------------- specs and registry


def test_spec_canonical_form_is_a_fixed_point():
    agent = resolve_agent(SPEC, harness="catanatron")
    assert agent.canonical == ("LLM:model=stub/random:strategy=plan:thinking=true:max_tokens=16384:temperature=0.1"
                               ":seed=0:retries=1:notes=8:history=30")
    assert resolve_agent(agent.canonical, harness="catanatron").canonical == agent.canonical
    assert agent.fair and agent.checkpoint is None and agent.label == "LLM:stub/random:plan"
    assert agent.key_dict() == {"key": "LLM", "harness": "catanatron", "params": {
        "model": "stub/random", "strategy": "plan", "thinking": True, "max_tokens": 16384, "temperature": 0.1,
        "seed": 0, "retries": 1, "notes": 8, "history": 30}}
    other = resolve_agent("LLM:model=stub/random:strategy=basic", harness="catanatron")
    assert other.key_dict() != agent.key_dict()


def test_bad_specs_raise():
    for spec in ("LLM", "LLM:model=stub/nope", "LLM:model=stub/first:strategy=zen", "LLM:model=stub/first:foo=1",
                 "LLM:model=stub/first:thinking=maybe", "LLM:model=stub/first:max_tokens=0",
                 "LLM:model=stub/first:retries=-1", "LLM:model=stub/first:notes=-1",
                 "LLM:model=stub/first:history=-1",
                 "LLM:model=stub/first:max_tokens=1000"):  # with thinking, half is a reasoning budget under 1024
        with pytest.raises(SpecError):
            check_spec_syntax(spec)
        with pytest.raises(SpecError):
            resolve_agent(spec, harness="catanatron")
    with pytest.raises(SpecError, match="catanatron harness"):
        resolve_agent(SPEC)  # the native harness has no LLM
    check_spec_syntax("LLM:model=${params.model}:strategy=${params.strategy}:thinking=true")  # a plan, unresolved
    check_spec_syntax("LLM:model=stub/first:thinking=false:max_tokens=1000")


def test_registry_builds_the_player():
    player = REGISTRY.build("LLM:model=stub/first:strategy=basic", Color.RED)
    assert isinstance(player, LlmPlayer) and player.params.model == "stub/first"
    assert REGISTRY.spec_of(player)["params"]["strategy"] == "basic"
    assert isinstance(resolve_agent(SPEC, harness="catanatron").player(Color.BLUE), LlmPlayer)


def test_a_real_model_needs_the_key_from_the_environment(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        make_model("openai/gpt-5")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        OpenRouterModel("openai/gpt-5")
    # Resolution needs no key (records are found without one); playing does (preflight).
    agent = resolve_agent("LLM:model=openai/gpt-5:strategy=basic", harness="catanatron")
    player = agent.player(Color.RED)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        player.before(two_player_game(player, REGISTRY.build("R", Color.BLUE)))


# ---------------------------------------------------------------- stub games


def test_a_stub_game_through_the_harness_vs_f():
    suite = load_suite(SUITE)
    lineup = [resolve_agent(SPEC, harness="catanatron", games=suite.games()),
              resolve_agent("F", harness="catanatron", games=suite.games())]
    games = suite.schedule(2)
    rows = []
    play_catanatron(lineup, games, suite=suite, threads=2, on_game=rows.append)
    rows.sort(key=lambda r: r["game_id"])
    assert [r["game_id"] for r in rows] == [g.game_id for g in games]
    for row, scheduled in zip(rows, games):
        assert row["seats"] == list(scheduled.seats) and row["forfeits"] == [0, 0] and not row["stuck"]
        assert sum(row["scores"]) == pytest.approx(1.0)
        if not row["truncated"]:
            assert row["seats"][row["winner_seat"]] == row["winner"]
        stats, other = row["llm"]
        assert other is None and stats["fallbacks"] == 0 and stats["retries"] == 0 and stats["estimated"]
        assert stats["decisions"] > 0 and stats["calls"] == stats["decisions"]
        assert stats["prompt_tokens"] > stats["decisions"] * 1000 and stats["completion_tokens"] > 0
        assert stats["cost"] == 0.0 and not stats["cost_estimated"] and stats["truncated"] == 0  # a stub is free
        assert row["decisions"][row["seats"].index(0)] == stats["decisions"]
    # The stub draws from a seed derived from the spec seed and the game's: games replay.
    again = []
    play_catanatron(lineup, games, suite=suite, threads=1, on_game=again.append)
    strip = lambda r: {k: v for k, v in r.items() if k != "llm"}
    assert sorted(map(strip, again), key=lambda r: r["game_id"]) == [strip(r) for r in rows]


def test_catanrl_eval_records_a_catanatron_harness_game(capsys):
    from catanrl.cli import main
    from catanrl.eval.results import find

    suite = {**SUITE, "name": "llm-cli", "purpose": "heldout"}
    (path := __import__("pathlib").Path(__import__("os").environ["CATANRL_RUNS_DIR"]) / "llm-cli.yaml").write_text(
        json.dumps({k: v for k, v in suite.items() if k != "name"}))
    assert main([
        "eval", "LLM:model=stub/first:strategy=basic", "F", "--suite", str(path), "--harness", "catanatron",
        "--threads", "1"]) == 0
    out = capsys.readouterr().out
    assert "LLM:stub/first:basic" in out and "per game:" in out and "(estimated)" in out and "$0.0000" in out
    record = find(["LLM:model=stub/first:strategy=basic", "F"], str(path), harness="catanatron")
    assert record is not None and record.key["harness"] == "catanatron" and "catanatron" in record.key["code"]
    assert record.data["fair"] == [True, False] and record.data["valid"]
    rows = list(record.games())
    assert len(rows) == 2 and all(r["llm"][0]["decisions"] > 0 for r in rows)
    llm = record.summary["slots"][0]["llm"]
    assert llm["games"] == 2 and llm["decisions"] == sum(r["llm"][0]["decisions"] for r in rows)
    assert llm["fallbacks"] == 0 and llm["estimated"] and llm["per_game"]["prompt_tokens"] > 1000
    assert llm["cost"] == 0.0 and llm["truncated"] == 0 and not llm["cost_estimated"]
    with pytest.raises(SpecError, match="catanatron harness"):  # no native record can hold an LLM
        find(["LLM:model=stub/first:strategy=basic", "F"], str(path))
    assert main(["eval", "LLM:model=stub/first:strategy=basic", "F", "--suite", str(path), "--harness",
                 "catanatron"]) == 0  # cached
    assert record.eval_id in capsys.readouterr().out


# ---------------------------------------------------------------- the seat's view


def test_the_state_text_never_shows_an_opponents_hidden_cards():
    """Unplayed development cards (so hidden VP) stay hidden; in 2p the resource hand is
    exact (it is public: test_the_opponents_hand_is_public_in_two_player_games), with three
    or more players it is a count."""
    red = resolve_agent(SPEC, harness="catanatron").player(Color.RED)
    blue = REGISTRY.build("R", Color.BLUE)
    game = two_player_game(red, blue)
    for _ in range(12):  # past the initial placements
        game.play_tick()
    state = game.state
    key = f"P{state.color_to_index[Color.BLUE]}"
    for resource in RESOURCES:
        state.player_state[f"{key}_{resource}_IN_HAND"] = 7 if resource == "ORE" else 1
    for card in DEVELOPMENT_CARDS:
        state.player_state[f"{key}_{card}_IN_HAND"] = 2 if card == "KNIGHT" else 0
    state.player_state[f"{key}_VICTORY_POINT_IN_HAND"] = 1
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = state.player_state[f"{key}_VICTORY_POINTS"] + 1
    state.player_state[f"{key}_PLAYED_KNIGHT"] = 3
    mine = f"P{state.color_to_index[Color.RED]}"
    state.player_state[f"{mine}_WHEAT_IN_HAND"] = 4
    state.player_state[f"{mine}_MONOPOLY_IN_HAND"] = 1

    text = render_state(game, Color.RED)
    blue_block = player_status(state, Color.BLUE, Color.RED)
    red_block = player_status(state, Color.RED, Color.RED)
    assert blue_block in text and red_block in text
    # Public: the hand (exact in 2p), the number of dev cards, what was played, the visible points.
    assert "hand (11 cards): Wood 1, Brick 1, Sheep 1, Wheat 1, Ore 7" in blue_block
    assert "3 unplayed (hidden)" in blue_block and "Knight 3" in blue_block
    assert f"{state.player_state[f'{key}_VICTORY_POINTS']} VP visible" in blue_block
    # Hidden: no unplayed card's kind, no victory-point card, no actual VP.
    assert "Knight 2" not in text and "Victory Point 1" not in text and "Monopoly 0" not in text
    assert f"{state.player_state[f'{key}_ACTUAL_VICTORY_POINTS']} VP visible" not in text
    # The observer's own hand and cards are shown.
    assert "Wheat 4" in red_block and "Monopoly 1" in red_block and "(you)" in red_block
    red.notes = []
    prompt = red.user_message(game, game.playable_actions)
    assert blue_block in prompt and "YOUR OWN NOTES" not in prompt


def test_three_player_hands_are_counts():
    game = random_game(game="base-3p", colors=(Color.RED, Color.BLUE, Color.WHITE))
    play_until(game, lambda g: not g.state.is_initial_build_phase and sum(get_player_freqdeck(g.state, Color.BLUE)) > 1)
    state = game.state
    block = player_status(state, Color.BLUE, Color.RED)
    n = sum(get_player_freqdeck(state, Color.BLUE))
    assert f"hand: {n} cards (which kinds is hidden from you)" in block
    assert not any(f"{lab} " in block.split("hand:")[1].split("\n")[0] for lab in ("Wood", "Brick", "Sheep", "Ore"))
    rules = system_prompt("basic", game, Color.RED)
    assert "Players: 3" in rules and "not which ones" in rules and "exact hand" not in rules


class PublicHandTracker:
    """The opponent's hand as ``observer`` can derive it in a 2-player game, from nothing but
    public facts and the observer's own hand: the dice and the board (production, computed
    against the bank, itself 19 minus both known hands), the announced actions and their
    public costs, discards, and steals, which the observer sees through its own hand."""

    def __init__(self, game, observer):
        self.game, self.observer = game, observer
        self.other = next(c for c in game.state.colors if c != observer)
        self.hand = [0] * 5
        self.bank = list(game.state.resource_freqdeck)  # the bank as dealt: public

    def mine(self):
        return get_player_freqdeck(self.game.state, self.observer)

    def step(self, action):
        from catanatron.apply_action import yield_resources
        from catanatron.models.decks import (
            CITY_COST_FREQDECK,
            DEVELOPMENT_CARD_COST_FREQDECK,
            ROAD_COST_FREQDECK,
            SETTLEMENT_COST_FREQDECK,
        )

        state = self.game.state
        initial, free_road = state.is_initial_build_phase, state.is_road_building
        mine_before = self.mine()
        bank = [t - a - b for t, a, b in zip(self.bank, mine_before, self.hand)]
        record = self.game.execute(action)
        mine_after = self.mine()
        a, kind = record.action, record.action.action_type
        index = {r: i for i, r in enumerate(RESOURCES)}
        delta = [0] * 5
        cost = {ActionType.BUILD_CITY: CITY_COST_FREQDECK, ActionType.BUY_DEVELOPMENT_CARD:
                DEVELOPMENT_CARD_COST_FREQDECK}
        if kind == ActionType.ROLL and sum(record.result) != 7:
            payout, _ = yield_resources(state.board, bank, sum(record.result))
            delta = list(payout.get(self.other, delta))
        elif a.color == self.other:
            if kind == ActionType.BUILD_SETTLEMENT and initial:
                if len(state.buildings_by_color[self.other][SETTLEMENT]) == 2:
                    for tile in state.board.map.adjacent_tiles[a.value]:
                        if tile.resource is not None:
                            delta[index[tile.resource]] += 1
            elif kind == ActionType.BUILD_SETTLEMENT:
                delta = [-x for x in SETTLEMENT_COST_FREQDECK]
            elif kind == ActionType.BUILD_ROAD and not initial and not free_road:
                delta = [-x for x in ROAD_COST_FREQDECK]
            elif kind in cost:
                delta = [-x for x in cost[kind]]
            elif kind == ActionType.DISCARD_RESOURCE:
                delta[index[a.value]] -= 1
            elif kind == ActionType.MOVE_ROBBER and a.value[1] == self.observer:
                delta = [b - c for b, c in zip(mine_before, mine_after)]  # what the observer lost
            elif kind == ActionType.PLAY_YEAR_OF_PLENTY:
                for r in a.value:
                    delta[index[r]] += 1
            elif kind == ActionType.PLAY_MONOPOLY:
                delta[index[a.value]] = mine_before[index[a.value]]
            elif kind == ActionType.MARITIME_TRADE:
                for r in a.value[:-1]:
                    if r is not None:
                        delta[index[r]] -= 1
                delta[index[a.value[-1]]] += 1
        elif kind == ActionType.MOVE_ROBBER and a.value[1] == self.other:
            delta = [b - c for b, c in zip(mine_before, mine_after)]  # minus what the observer gained
        elif kind == ActionType.PLAY_MONOPOLY:
            delta[index[a.value]] = -self.hand[index[a.value]]
        self.hand = [h + d for h, d in zip(self.hand, delta)]
        return record


@pytest.mark.parametrize("observer", [Color.RED, Color.BLUE])
def test_the_opponents_hand_is_public_in_two_player_games(observer):
    """Issue: the state hid the opponent's resource kinds although in 2p every resource
    movement is public under the fork's rules. A tracker that sees only public facts and its
    own hand matches the true hand after every action of random games (dev-card-heavy decks,
    so steals, discards, Monopoly and Year of Plenty all happen), and the state text shows it."""
    seen = set()
    for seed in range(3):
        game = random_game(seed=seed, map_seed=seed + 1)
        game.state.development_listdeck = ["MONOPOLY", "YEAR_OF_PLENTY", "KNIGHT", "ROAD_BUILDING"] * 6
        tracker = PublicHandTracker(game, observer)
        for _ in range(3000):
            if game.winning_color() is not None:
                break
            record = tracker.step(game.state.current_player().decide(game, game.playable_actions))
            seen.add(record.action.action_type)
            assert tracker.hand == get_player_freqdeck(game.state, tracker.other), (seed, record)
        hand = ", ".join(f"{lab} {n}" for lab, n in zip(("Wood", "Brick", "Sheep", "Wheat", "Ore"), tracker.hand))
        assert f"hand ({sum(tracker.hand)} cards): {hand}" in player_status(game.state, tracker.other, observer)
    assert {ActionType.MOVE_ROBBER, ActionType.DISCARD_RESOURCE, ActionType.PLAY_MONOPOLY,
            ActionType.PLAY_YEAR_OF_PLENTY, ActionType.MARITIME_TRADE, ActionType.BUY_DEVELOPMENT_CARD,
            ActionType.BUILD_CITY} <= seen


def test_tiles_are_named_by_id_everywhere():
    """Issue: tiles were listed by id but the robber's position and the robber actions by cube
    coordinate. Now every text names tiles by id, and a robber action says what the tile is,
    whose buildings sit on it and whom it steals from."""
    game = random_game(seed=2)
    play_until(game, lambda g: g.playable_actions[0].action_type == ActionType.MOVE_ROBBER)
    assert game.playable_actions[0].action_type == ActionType.MOVE_ROBBER
    color = game.state.current_color()
    board = game.state.board
    texts = [render_state(game, color), render_actions(game, game.playable_actions),
             system_prompt("plan", game, color), render_events(game, color, 1000)]
    assert not any(CUBE.search(t) for t in texts)
    robber = board.map.land_tiles[board.robber_coordinate]
    assert f"Robber: on Tile {robber.id} (" in texts[0]
    for action in game.playable_actions:
        coordinate, victim = action.value
        tile = board.map.land_tiles[coordinate]
        text = describe_action(game, action)
        assert text.startswith(f"Move the robber to Tile {tile.id} (")
        assert ("no buildings" in text) == (not any(n in board.buildings for n in tile.nodes.values()))
        for node in tile.nodes.values():
            if node in board.buildings:
                assert f"at {node}" in text
        assert (f"steal 1 random card from {victim.name}" in text) if victim else "steal nothing" in text


def test_building_actions_say_what_the_node_produces():
    """Issue: "Build settlement at node 23" said nothing about the node. Settlement, city and
    road actions (setup included) now carry the node's tiles, numbers, pips and port."""
    game = random_game(seed=4)
    catan_map = game.state.board.map
    for action in game.playable_actions:  # the first setup settlement
        assert describe_action(game, action) == f"Place your settlement at node {action.value} " \
                                                f"({node_text(catan_map, action.value)})"
    game.play_tick()  # the settlement; now its road
    for action in game.playable_actions:
        a, b = action.value
        text = describe_action(game, action)
        far = b if a in game.state.board.buildings else a
        assert text.startswith(f"Place a road {a}-{b} (to node {far}: {node_text(catan_map, far)}")
    play_until(game, lambda g: any(x.action_type == ActionType.BUILD_CITY for x in g.playable_actions))
    for action in game.playable_actions:
        if action.action_type == ActionType.BUILD_CITY:
            assert node_text(catan_map, action.value) in describe_action(game, action)


def test_the_state_lists_only_reachable_open_spots():
    """Issue: "Empty nodes" listed every empty node (the distance rule ignored) and "Available
    road edges" every free edge. Now the state lists the open spots near the seat's network."""
    game = random_game(seed=6)
    play_until(game, lambda g: g.state.num_turns > 30)
    color = game.state.current_color()
    text = render_state(game, color)
    assert "Empty nodes" not in text and "Available road" not in text
    board = game.state.board
    dist = road_distances(board, color)
    listed = [int(m) for m in re.findall(r"^  node (\d+) \(", text, re.MULTILINE)]
    assert listed and all(n in board.board_buildable_ids and dist[n] <= 2 for n in listed)
    assert all(n not in board.buildings and not any(m in board.buildings for m in board.map.graph.neighbors(n))
               for n in listed)


def test_the_history_shows_what_happened_redacted():
    """Issue: the model never saw what happened. The recent events are in the prompt, as the
    observer saw them: an opponent's bought card is never named, its own is."""
    game = random_game(seed=1)
    game.state.development_listdeck = ["MONOPOLY", "KNIGHT", "VICTORY_POINT"] * 8
    play_until(game, lambda g: sum(1 for r in g.state.action_records
                                   if r.action.action_type == ActionType.BUY_DEVELOPMENT_CARD) >= 3)
    buys = [r for r in game.state.action_records if r.action.action_type == ActionType.BUY_DEVELOPMENT_CARD]
    assert len(buys) >= 3
    for record in buys:
        buyer = record.action.color
        other = next(c for c in game.state.colors if c != buyer)
        hidden = event_text(game, record, other)
        assert hidden == f"{buyer.name} bought a development card"
        assert not any(label in hidden for label in CARD_LABELS.values())
        assert event_text(game, record, buyer) == f"you bought a development card ({CARD_LABELS[record.result]})"
    text = render_events(game, Color.RED, 10)
    assert text.startswith("=== RECENT EVENTS (the last 10, oldest first) ===")
    whole = render_events(game, Color.RED, 10_000)
    assert len(text) < len(whole) and whole.splitlines()[-1].endswith(text.splitlines()[-1])
    assert render_events(game, Color.RED, 0) == ""
    red = LlmPlayer(Color.RED, LlmPlayer.Params(model="stub/first"))
    assert "=== RECENT EVENTS" in red.user_message(game, game.playable_actions)


def test_a_steal_is_named_only_to_the_thief_and_the_victim():
    from catanrl.agents.llm.render import tile_text

    game = random_game(game="base-3p", colors=(Color.RED, Color.BLUE, Color.WHITE))
    coordinate = next(iter(game.state.board.map.land_tiles))
    tile = game.state.board.map.land_tiles[coordinate]
    record = ActionRecord(Action(Color.BLUE, ActionType.MOVE_ROBBER, (coordinate, Color.WHITE)), "ORE")
    assert event_text(game, record, Color.RED) == \
        f"BLUE moved the robber to {tile_text(tile)} and stole a card from WHITE"
    assert event_text(game, record, Color.BLUE).endswith("and stole 1 Ore from WHITE")
    assert event_text(game, record, Color.WHITE).endswith("and stole 1 Ore from you")
    two = random_game()  # in 2p the observer is always the thief or the victim
    coordinate = next(iter(two.state.board.map.land_tiles))
    record = ActionRecord(Action(Color.BLUE, ActionType.MOVE_ROBBER, (coordinate, Color.RED)), "WHEAT")
    assert event_text(two, record, Color.RED).endswith("stole 1 Wheat from you")
    assert event_text(two, record, Color.BLUE).endswith("stole 1 Wheat from RED")


def test_every_offered_action_is_described():
    """Issue: some actions fell through to "Action: KIND value" or were mislabelled (a 4:1 bank
    trade called a port trade, robber tiles as coordinates, free roads as paid ones)."""
    seen = {}
    for seed in range(3):
        game = random_game(seed=seed + 10, map_seed=seed)
        game.state.development_listdeck = ["ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY", "KNIGHT"] * 6
        for _ in range(4000):
            if game.winning_color() is not None:
                break
            for action in game.playable_actions:
                text = describe_action(game, action)
                assert not text.startswith("Action:") and not CUBE.search(text), text
                seen.setdefault(action.action_type, text)
                if action.action_type == ActionType.MARITIME_TRADE:
                    ratio = sum(1 for r in action.value[:-1] if r is not None)
                    assert text.startswith(f"Trade {ratio} ") and ("with the bank" in text) == (ratio == 4)
                if action.action_type == ActionType.BUILD_ROAD and game.state.is_road_building:
                    assert text.startswith("Build a free road (Road Building")
                if action.action_type == ActionType.DISCARD_RESOURCE:
                    assert text.startswith("Discard 1 ")
            game.play_tick()
    offered = {ActionType.ROLL, ActionType.END_TURN, ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT,
               ActionType.BUILD_CITY, ActionType.BUY_DEVELOPMENT_CARD, ActionType.PLAY_KNIGHT_CARD,
               ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY, ActionType.PLAY_ROAD_BUILDING,
               ActionType.DISCARD_RESOURCE, ActionType.MOVE_ROBBER, ActionType.MARITIME_TRADE}
    assert offered <= set(seen), offered - set(seen)
    # Trade responses (no catanatron-harness player offers one, but the fork can prompt them).
    game = random_game()
    offer = (1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    for kind, value in ((ActionType.ACCEPT_TRADE, offer), (ActionType.REJECT_TRADE, offer),
                        (ActionType.CONFIRM_TRADE, (*offer, Color.BLUE)), (ActionType.CANCEL_TRADE, None)):
        assert not describe_action(game, Action(Color.RED, kind, value)).startswith("Action:")


def test_actions_are_numbered_as_the_model_answers_them():
    game = two_player_game(REGISTRY.build("R", Color.RED), REGISTRY.build("R", Color.BLUE))
    text = render_actions(game, game.playable_actions)
    lines = text.splitlines()
    assert lines[0] == "=== AVAILABLE ACTIONS ===" and len(lines) == len(game.playable_actions) + 1
    assert lines[1].startswith("0: Place your settlement at node ")
    assert StubModel.count_actions([{"role": "user", "content": text}]) == len(game.playable_actions)


def test_a_forced_discard_says_how_many_cards_must_go():
    from catanatron.models.actions import generate_playable_actions
    from catanatron.models.enums import ActionPrompt

    red = resolve_agent(SPEC, harness="catanatron").player(Color.RED)
    game = two_player_game(red, REGISTRY.build("R", Color.BLUE))
    for _ in range(12):
        game.play_tick()
    state = game.state
    assert "discard" not in render_state(game, Color.RED).lower()
    idx = state.color_to_index[Color.RED]
    for resource in RESOURCES:
        state.player_state[f"P{idx}_{resource}_IN_HAND"] = 2
    state.current_player_index, state.current_prompt, state.is_discarding = idx, ActionPrompt.DISCARD, True
    state.discard_counts = [0] * len(state.colors)
    state.discard_counts[idx] = 5
    game.playable_actions = generate_playable_actions(state)
    assert {a.action_type for a in game.playable_actions} == {ActionType.DISCARD_RESOURCE}
    assert "a 7 was rolled: discard 5 more of your 10 resource cards" in render_state(game, Color.RED)
    prompt = red.user_message(game, game.playable_actions)
    assert "discard 5 more" in prompt and "0: Discard 1 Wood (you hold 2)" in prompt
    state.discard_counts[idx] = 1
    assert "discard 1 more of your 10" in render_state(game, Color.RED)


# ---------------------------------------------------------------- this game's rules


@pytest.mark.parametrize("strategy", ["basic", "strategy", "search", "rules", "plan"])
def test_every_prompt_states_this_games_rules(strategy):
    """Issue: the rules and the guide said players discard above 7 cards (base-2p: above 9) and
    described trading between players, which the harness never offers."""
    game = random_game()
    text = system_prompt(strategy, game, Color.RED)
    assert "=== THIS GAME'S RULES" in text and "Players: 2, in turn order RED, BLUE. You are RED." in text
    assert "more than 9 resource cards discards half" in text and "reach 15 victory points" in text
    assert "There is no trading between players in this game." in text
    assert "The robber may be moved to any land tile" in text and "14 Knight" in text
    assert not re.search(r"more than 7\b", text) and "7 or less cards" not in text
    assert "TRADE WITH OTHER PLAYERS" not in text and "trade freely with other players" not in text
    assert text.index("=== THIS GAME'S RULES") < text.index("=== BOARD COORDINATE SYSTEM")
    assert not CUBE.search(text)
    four = random_game(game="base-4p", colors=(Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE))
    text = system_prompt(strategy, four, Color.BLUE)
    assert "more than 7 resource cards discards half" in text and "Players: 4" in text
    four.state.robber_mode = "friendly_public"
    assert "at most 2 visible victory points" in system_prompt(strategy, four, Color.BLUE)


def test_the_rulebook_and_the_guide_state_this_games_target():
    game = random_game()
    for strategy in ("rules", "strategy"):
        text = system_prompt(strategy, game, Color.RED)
        assert "reaching 15 victory points first" in text
        assert not re.search(r"\b10 (?:victory points|VPs?|or more)\b", text) and "reached 10" not in text
    rules = system_prompt("rules", game, Color.RED)
    assert "reach 15 victory points" in rules and "15 or more VPs" in rules and "reached 15 VPs" in rules
    assert "played to ten points" in rules  # the worked example keeps the base game's arithmetic
    assert "White has 11 resource cards in hand. When a 7 is rolled, they must discard 5 cards." in rules
    guide = system_prompt("strategy", game, Color.RED)
    assert "Being close to 15 VP" in guide and "9 or fewer cards" in guide and "more than 9 cards" in guide


def test_the_board_section_is_static_and_on_tile_ids():
    game = random_game()
    catan_map = game.state.board.map
    text = system_prompt("basic", game, Color.RED)
    for tile in catan_map.land_tiles.values():
        name = "Desert" if tile.resource is None else f"{tile.resource.title()} {tile.number}"
        assert f"Tile {tile.id} ({name}): corner nodes " in text
    assert f"Node 0: {node_text(catan_map, 0)} | tiles " in text
    assert len(re.findall(r"^(?:3:1|2:1 \w+) port: nodes \d+, \d+$", text, re.MULTILINE)) == 9
    play_until(game, lambda g: g.state.num_turns > 20)
    assert system_prompt("basic", game, Color.RED) == text  # the board and the rules never change
    assert "corner nodes" not in render_state(game, Color.RED)  # and is not repeated per decision


# ---------------------------------------------------------------- answers


@pytest.mark.parametrize("text, index", [
    ("Reasoning.\nnotes\n2", 2),
    ("**2**", 2),
    ("Action 2.", 2),
    ("I choose action 2 because it builds toward the port.", 2),
    ("notes line\n- 3", 3),
    ("I'll go with option #1", 1),
    ("The answer is 2", 2),
])
def test_parse_answer_reads_index_and_text_answers(text, index):
    got, notes = parse_answer(text, 4)
    assert got == index
    assert notes == ("notes" if text.startswith("Reasoning") else notes)


def test_parse_answer_rejects_garbage_and_out_of_range():
    with pytest.raises(AnswerError, match="no action number"):
        parse_answer("hmm, hard to say", 4)
    with pytest.raises(AnswerError, match="no action number"):
        parse_answer("nodes 3 and 12 look good", 4)
    with pytest.raises(AnswerError, match="does not exist"):
        parse_answer("notes\n7", 4)
    with pytest.raises(AnswerError, match="empty"):
        parse_answer("  \n", 4)


def test_parse_answer_skips_a_trailing_markup_line_and_cuts_long_notes():
    assert parse_answer("notes\n3\n---", 4) == (3, "notes")
    assert parse_answer("notes\n**2**\n```", 4) == (2, "notes")
    index, notes = parse_answer("x" * 2000 + "\n1", 4)
    assert index == 1 and len(notes) == NOTE_CHARS and notes.endswith("...")


def play_decisions(script, retries=1, decisions=1, **params):
    """``decisions`` prompted decisions of a RED stub player answering ``script`` in order."""
    params = LlmPlayer.Params(model="stub/scripted", retries=retries, **params)
    red = LlmPlayer(Color.RED, params, model=StubModel("scripted", script=script), keep_log=True)
    game = two_player_game(red, REGISTRY.build("R", Color.BLUE))
    done = 0
    while done < decisions:
        if game.state.current_color() == Color.RED and len(game.playable_actions) > 1:
            done += 1
        game.play_tick()
    return red


def test_garbage_is_retried_then_falls_back():
    red = play_decisions(["no idea", "nonsense"])
    assert red.stats["decisions"] == 1 and red.stats["calls"] == 2 and red.stats["retries"] == 1
    assert red.stats["fallbacks"] == 1 and red.log[0]["fallback"]
    assert red.log[0]["exchanges"][0]["error"]
    red = play_decisions(["999", "notes after the error\n1"])
    assert red.stats == {**red.stats, "decisions": 1, "calls": 2, "retries": 1, "fallbacks": 0}
    assert red.notes == ["notes after the error"] and red.log[0]["exchanges"][1]["index"] == 1
    red = play_decisions(["garbage"], retries=0)
    assert red.stats["calls"] == 1 and red.stats["retries"] == 0 and red.stats["fallbacks"] == 1
    red = play_decisions(["I choose action 1."])
    assert red.stats["fallbacks"] == 0 and red.stats["retries"] == 0


def test_the_default_script_covers_every_path_and_notes_carry_over():
    red = play_decisions(None, decisions=5)
    assert red.stats["decisions"] == 5 and red.stats["calls"] == 8
    assert red.stats["retries"] == 3 and red.stats["fallbacks"] == 1
    assert red.notes[0] == "Taking the last option this turn."
    assert "=== YOUR OWN NOTES FROM YOUR LAST 1 DECISIONS (oldest first) ===\n- Taking the last option this turn." \
        in red.log[1]["prompt"]
    assert "system" in red.log[0] and "system" not in red.log[1]  # the transcript keeps the system prompt once


def test_only_the_last_notes_are_shown():
    """Issue: every past note went into every later prompt (prompt tokens grew with the square of
    the decisions). Only the last ``notes`` are shown, labelled as the model's own."""
    script = [f"note {k}\n0" for k in range(12)]
    red = play_decisions(script, decisions=12, notes=3)
    last = red.log[-1]["prompt"]
    assert "=== YOUR OWN NOTES FROM YOUR LAST 3 DECISIONS (oldest first) ===\n- note 8\n- note 9\n- note 10" in last
    assert "note 7" not in last and "note 0" not in last
    red = play_decisions(script, decisions=3, notes=0)
    assert "YOUR OWN NOTES" not in red.log[-1]["prompt"]


def test_fallback_prefers_end_turn():
    from catanrl.agents.llm.player import fallback_action

    build = Action(Color.RED, ActionType.BUILD_ROAD, (0, 1))
    end = Action(Color.RED, ActionType.END_TURN, None)
    assert fallback_action([build, end]) is end and fallback_action([build]) is build


class RecordingModel:
    """Replies in order (text, or a Reply) and keeps every message list and keyword it was sent."""

    def __init__(self, replies):
        self.replies, self.sent, self.kwargs = list(replies), [], []

    def complete(self, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        self.kwargs.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Reply):
            return reply
        return Reply(text=reply, reasoning=None, prompt_tokens=1, completion_tokens=1, estimated=True)


def test_an_empty_reply_is_retried_with_a_placeholder():
    model = RecordingModel(["", "notes\n1"])
    red = LlmPlayer(Color.RED, LlmPlayer.Params(model="stub/first"), model=model)
    game = two_player_game(red, REGISTRY.build("R", Color.BLUE))
    assert red.decide(game, game.playable_actions) is game.playable_actions[1]
    assert red.stats["retries"] == 1 and red.stats["fallbacks"] == 0
    first, second = model.sent
    assert second[:2] == first and second[2] == {"role": "assistant", "content": "(no answer)"}
    assert second[3]["role"] == "user" and "empty" in second[3]["content"]


def test_a_reply_cut_off_at_max_tokens_is_never_read_as_an_answer():
    """Issue: a reply cut off by max_tokens was parsed like any other (its last line is wherever
    the cap fell, which can hold a number) or fell back to End Turn silently. It is retried
    without being sent back, and counted."""
    cut = Reply(text="notes\n2", reasoning=None, prompt_tokens=10, completion_tokens=10, cost=0.0,
                finish_reason="length")
    model = RecordingModel([cut, "notes\n1"])
    red = LlmPlayer(Color.RED, LlmPlayer.Params(model="stub/first"), model=model)
    game = two_player_game(red, REGISTRY.build("R", Color.BLUE))
    assert red.decide(game, game.playable_actions) is game.playable_actions[1]
    assert red.stats["truncated"] == 1 and red.stats["retries"] == 1 and red.stats["fallbacks"] == 0
    assert model.sent[1][2] == {"role": "assistant", "content": "(reply cut off at the token limit)"}
    assert "cut off at the token limit" in model.sent[1][3]["content"]
    model = RecordingModel([cut])
    red = LlmPlayer(Color.RED, LlmPlayer.Params(model="stub/first", retries=0), model=model)
    game = two_player_game(red, REGISTRY.build("R", Color.BLUE))
    red.decide(game, game.playable_actions)
    assert red.stats["truncated"] == 1 and red.stats["fallbacks"] == 1


def test_temperature_is_sent_only_without_thinking():
    """Issue: temperature=0.1 went with every call, which thinking Anthropic models and OpenAI
    reasoning models reject."""
    for thinking, expected in ((True, None), (False, 0.1)):
        model = RecordingModel(["notes\n0"])
        red = LlmPlayer(Color.RED, LlmPlayer.Params(model="stub/first", thinking=thinking), model=model)
        game = two_player_game(red, REGISTRY.build("R", Color.BLUE))
        red.decide(game, game.playable_actions)
        assert model.kwargs[0]["temperature"] == expected and model.kwargs[0]["thinking"] is thinking
        assert model.kwargs[0]["max_tokens"] == 16384


# ---------------------------------------------------------------- tokens and dollars


def test_token_accounting_is_characters_over_four_for_the_stub():
    red = play_decisions(["notes\n0"], decisions=2)
    assert red.stats["estimated"]
    expected = sum(estimate_tokens(" " * (len(red.log[i]["prompt"]) + len(red.system(None)))) for i in range(2))
    assert red.stats["prompt_tokens"] == expected
    assert red.stats["completion_tokens"] == 2 * estimate_tokens("notes\n0")
    assert estimate_tokens("") == 1 and estimate_tokens("abcd") == 1 and estimate_tokens("abcde") == 2


def test_cost_tool_prices_a_cell():
    from catanrl.agents.llm.cost import dollars, measure, parse_prices, table

    assert dollars(1_000_000, 2_000_000, (1.25, 10.0)) == pytest.approx(1.25 + 20.0)
    assert parse_prices(["gpt-5=1.25,10", "x = 3 , 15"]) == {"gpt-5": (1.25, 10.0), "x": (3.0, 15.0)}
    with pytest.raises(ValueError):
        parse_prices(["gpt-5=cheap"])
    measured = measure(["basic", "plan"], suite=SUITE, games=2, model="stub/first")
    for strategy in ("basic", "plan"):
        m = measured[strategy]
        assert m["games"] == 2 and m["per_game"]["decisions"] > 0 and m["estimated"]
        assert 1000 < m["system_tokens"] < m["per_game"]["prompt_tokens"] / m["per_game"]["calls"]
    assert measured["plan"]["per_game"]["prompt_tokens"] != measured["basic"]["per_game"]["prompt_tokens"]
    lines = table(measured, {"m": (1.0, 2.0)}, completion_per_decision=100, cell_games=50)
    pg = measured["plan"]["per_game"]
    expected = (pg["prompt_tokens"] * 1.0 + pg["decisions"] * 100 * 2.0) / 1e6
    assert any(line.startswith("plan") and f"{expected:8.3f}" in line and f"{expected * 50:12.2f}" in line
               for line in lines)


def test_cost_tool_grows_the_notes_and_projects_longer_games():
    from catanrl.agents.llm.cost import dollars, measure, projected, shown_notes, table

    short = measure(["basic"], suite=SUITE, games=1, model="stub/first")["basic"]
    long = measure(["basic"], suite=SUITE, games=1, model="stub/first", notes_chars=200)["basic"]
    d = short["per_game"]["decisions"]
    assert long["per_game"]["decisions"] == d and d > 10 and long["notes"] == 8
    # The last 8 notes are shown: 176 more characters per note shown, min(k, 8) at decision k.
    extra = long["per_game"]["prompt_tokens"] - short["per_game"]["prompt_tokens"]
    assert extra == pytest.approx((200 - len("Taking the first option.")) / 4 * shown_notes(d, 8), rel=0.05)
    assert shown_notes(5, 8) == 10 and shown_notes(12, 8) == 36 + 8 * 3
    assert long["note_tokens"] == pytest.approx(203 / 4) and long["base_per_decision"] > 1000
    assert projected(long, d) == pytest.approx(long["per_game"]["prompt_tokens"], rel=0.02)
    assert projected(long, 4 * d) < 4.1 * projected(long, d)  # linear: catanllm's notes grew with the square
    lines = table({"basic": long}, {"m": (1.0, 2.0)}, completion_per_decision=100, cell_games=50, decisions=60)
    assert any(line.startswith("projected at more decisions") for line in lines)
    (row,) = [line for line in lines if line.startswith("basic") and f"{60:>9} {projected(long, 60):>10.0f}" in line]
    assert f"{dollars(projected(long, 60), 6000, (1.0, 2.0)):8.3f}" in row


# ---------------------------------------------------------------- the real client, without a network


def fake_response(text="notes\n3", *, finish="stop", usage=None, reasoning=None, choices=True):
    message = SimpleNamespace(content=text, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)] if choices else [],
                           usage=usage)


def fake_client(model, responses):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return calls


def test_openrouter_client_reads_usage_and_asks_for_reasoning(monkeypatch):
    """The SDK call shape and the usage fields, against a fake client (no request is made)."""
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    model = OpenRouterModel("vendor/model", api_key="not-a-key")
    usage = SimpleNamespace(prompt_tokens=1234, completion_tokens=56, cost=0.0042,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=1000),
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=40))
    calls = fake_client(model, [fake_response("thinking aloud\nnotes\n3", usage=usage, reasoning="private chain"),
                                fake_response("2"), fake_response("2", usage=None)])
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    reply = model.complete(messages, max_tokens=1000, temperature=None, seed=7, thinking=True)
    assert (reply.text, reply.reasoning, reply.prompt_tokens, reply.completion_tokens, reply.estimated) == \
        ("thinking aloud\nnotes\n3", "private chain", 1234, 56, False)
    assert (reply.cost, reply.cached_tokens, reply.reasoning_tokens, reply.finish_reason) == (0.0042, 1000, 40, "stop")
    (call,) = calls
    assert call["model"] == "vendor/model" and call["messages"] == messages and call["seed"] == 7
    assert call["max_tokens"] == 1000 and "temperature" not in call  # thinking: the provider's default
    assert call["extra_body"] == {"reasoning": {"max_tokens": 500}, "usage": {"include": True}}
    model.complete(messages, max_tokens=1000, temperature=0.2, seed=0, thinking=False)
    assert calls[1]["extra_body"]["reasoning"] == {"enabled": False} and "seed" not in calls[1]
    assert calls[1]["temperature"] == 0.2
    reply = model.complete(messages, max_tokens=10, temperature=0.0, seed=0, thinking=False)
    assert reply.estimated and reply.prompt_tokens == estimate_tokens("su") and reply.completion_tokens == 1
    assert reply.cost is None
    fake_client(model, [fake_response("notes\n2", usage=None)])
    player = LlmPlayer(Color.RED, LlmPlayer.Params(model="vendor/model"), model=model)
    game = two_player_game(player, REGISTRY.build("R", Color.BLUE))
    assert player.decide(game, game.playable_actions) is game.playable_actions[2]
    assert player.stats["prompt_tokens"] > 1000 and player.stats["estimated"] and not player.stats["fallbacks"]
    assert player.stats["cost_estimated"] and player.stats["cost"] > 0  # priced at the fallback price


def test_anthropic_system_prompts_carry_a_cache_breakpoint():
    pytest.importorskip("openai")
    messages = [{"role": "system", "content": "the static per-game prefix"}, {"role": "user", "content": "u"}]
    for name, cached in (("anthropic/claude-sonnet-5.5", True), ("openai/gpt-6.1-sol", False)):
        model = OpenRouterModel(name, api_key="not-a-key")
        calls = fake_client(model, [fake_response()])
        model.complete(messages, max_tokens=4096, temperature=None, seed=0, thinking=True)
        system = calls[0]["messages"][0]["content"]
        if cached:
            assert system == [{"type": "text", "text": "the static per-game prefix",
                               "cache_control": {"type": "ephemeral"}}]
        else:
            assert system == "the static per-game prefix"
        assert calls[0]["messages"][1] == {"role": "user", "content": "u"}
    assert messages[0]["content"] == "the static per-game prefix"  # the caller's messages are not changed


def test_provider_failures_are_retried_then_raised():
    pytest.importorskip("openai")
    model = OpenRouterModel("vendor/model", api_key="not-a-key", backoff=0.0)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=0, cost=0.001)
    fake_client(model, [fake_response(None, finish="error", usage=usage), fake_response(choices=False),
                        fake_response("notes\n1", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                                                         cost=0.002))])
    reply = model.complete([{"role": "user", "content": "u"}], max_tokens=4096, temperature=None, seed=0,
                           thinking=True)
    assert reply.text == "notes\n1" and reply.cost == pytest.approx(0.003) and reply.prompt_tokens == 20
    fake_client(model, [fake_response(None, finish="error") for _ in range(4)])
    with pytest.raises(LLMError, match="failed 4 times"):
        model.complete([{"role": "user", "content": "u"}], max_tokens=4096, temperature=None, seed=0, thinking=True)


def test_transcripts_are_saved_when_asked(monkeypatch, tmp_path):
    suite = load_suite(SUITE)
    lineup = [resolve_agent("LLM:model=stub/first:strategy=basic", harness="catanatron"),
              resolve_agent("R", harness="catanatron")]
    games = suite.schedule(2)[:1]
    monkeypatch.setenv("CATANRL_LLM_TRANSCRIPTS", str(tmp_path / "t"))
    rows = []
    play_catanatron(lineup, games, suite=suite, on_game=rows.append)
    (path,) = list((tmp_path / "t").glob("*.jsonl"))
    assert path.name == f"{games[0].game_id}.s0.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(entries) == rows[0]["llm"][0]["decisions"]
    assert entries[0]["exchanges"][0]["index"] == 0 and "=== AVAILABLE ACTIONS ===" in entries[0]["prompt"]
    assert "=== THIS GAME'S RULES" in entries[0]["system"]


# ---------------------------------------------------------------- the spending cap


class CostlyModel:
    """A fake real model: answers 0, costs ``cost`` dollars a call; ``gate`` holds its first reply."""

    def __init__(self, cost, gate=None):
        self.cost, self.gate, self.calls = cost, gate, 0

    def complete(self, messages, **_):
        self.calls += 1
        if self.gate is not None and self.calls == 1:
            self.gate.wait(60)
        return Reply(text="notes\n0", reasoning=None, prompt_tokens=100, completion_tokens=10, cost=self.cost,
                     finish_reason="stop")


def test_a_real_model_needs_a_spending_cap(monkeypatch):
    player = LlmPlayer(Color.RED, LlmPlayer.Params(model="vendor/model"), model=CostlyModel(0.1))
    with pytest.raises(LLMError, match="CATANRL_LLM_BUDGET is not set"):
        player.preflight()
    monkeypatch.setenv("CATANRL_LLM_BUDGET", "cheap")
    with pytest.raises(LLMError, match="expected dollars"):
        player.preflight()
    for value in ("25", "none"):
        monkeypatch.setenv("CATANRL_LLM_BUDGET", value)
        player.preflight()
    LlmPlayer(Color.RED, LlmPlayer.Params(model="stub/first")).preflight()  # stubs need no cap
    monkeypatch.delenv("CATANRL_LLM_BUDGET")
    LlmPlayer(Color.RED, LlmPlayer.Params(model="stub/first")).preflight()


def test_the_cap_is_checked_before_each_call(monkeypatch):
    """Issue: nothing bounded what an eval could spend. Once the spend reaches the cap the next
    call is refused before it is sent, and each game's stats carry its cost."""
    monkeypatch.setenv("CATANRL_LLM_BUDGET", "0.25")
    model = CostlyModel(0.1)
    red = LlmPlayer(Color.RED, LlmPlayer.Params(model="vendor/model"), model=model)
    game = two_player_game(red, REGISTRY.build("R", Color.BLUE))
    with pytest.raises(llm_budget.BudgetExceeded, match="reached the cap"):
        play_until(game, lambda g: False)
    assert model.calls == 3 and red.stats["cost"] == pytest.approx(0.3) and red.stats["calls"] == 3
    assert llm_budget.budget().spent == pytest.approx(0.3)


def test_the_ledger_shares_the_cap_across_processes(tmp_path):
    ledger = tmp_path / "spend.jsonl"
    first = llm_budget.Budget(1.0, ledger)
    first.charge(0.6, model="m", estimated=False)
    first.charge(0.0, model="m", estimated=False)  # nothing to record
    second = llm_budget.Budget(1.0, ledger)  # a later process
    assert second.spent == pytest.approx(0.6) and len(ledger.read_text().splitlines()) == 1
    second.check()
    second.charge(0.5, model="m", estimated=True)
    with pytest.raises(llm_budget.BudgetExceeded, match="spend.jsonl"):
        second.check()
    assert llm_budget.Budget(None, ledger).spent == pytest.approx(1.1)


def test_a_missing_cost_is_priced_conservatively(monkeypatch):
    reply = Reply(text="", reasoning=None, prompt_tokens=1_000_000, completion_tokens=100_000)
    assert llm_budget.charge_of(reply) == (pytest.approx(15.0 + 7.5), True)
    monkeypatch.setenv("CATANRL_LLM_FALLBACK_PRICE", "1,2")
    assert llm_budget.charge_of(reply) == (pytest.approx(1.2), True)
    assert llm_budget.charge_of(Reply(text="", reasoning=None, prompt_tokens=5, completion_tokens=5, cost=0.01)) \
        == (0.01, False)


def test_the_cap_stops_the_harness_and_keeps_finished_rows(monkeypatch):
    """A cap reached in one game stops every game at its next tick; the rows of games that
    finished are delivered (a resume plays only the rest), as for an error."""
    import concurrent.futures

    from catanrl.eval import catanatron as harness

    monkeypatch.setenv("CATANRL_LLM_BUDGET", "1.0")
    suite = load_suite({**SUITE, "boards": 2})
    games = suite.schedule(2)[:2]  # one board from both seats: no third game starts on a freed worker
    paying = harness.SEAT_COLORS[games[0].seats.index(0)]  # slot 0's color in the first game
    second_done = threading.Event()
    costly = []

    class Agent:
        def player(self, color):
            if color == paying:  # the first game's LLM pays (its first reply waits for the second game)
                model = CostlyModel(1.0, gate=second_done)
                costly.append(model)
                return LlmPlayer(color, LlmPlayer.Params(model="vendor/model", strategy="basic"), model=model)
            return LlmPlayer(color, LlmPlayer.Params(model="stub/first", strategy="basic"))

    real_row = harness._row

    def row(scheduled, **kwargs):
        if scheduled.game_id == games[1].game_id:
            second_done.set()
        return real_row(scheduled, **kwargs)

    monkeypatch.setattr(harness, "_row", row)
    monkeypatch.setattr(concurrent.futures, "as_completed", lambda fs: iter(list(fs)))  # the capped game first
    rows, before = [], threading.active_count()
    with pytest.raises(llm_budget.BudgetExceeded):
        harness.play_catanatron([Agent(), resolve_agent("R", harness="catanatron")], games, suite=suite, threads=2,
                                on_game=rows.append)
    assert threading.active_count() == before
    assert [r["game_id"] for r in rows] == [games[1].game_id]
    assert sum(m.calls for m in costly) == 1  # the second call was refused before it was sent


# ---------------------------------------------------------------- the experiment


def test_llm_eval_stub_cells_smoke(monkeypatch, tmp_path):
    from catanrl.config import load_plan, validate_plan
    from catanrl.runner import _missing_finals, run_variant
    from catanrl.runs import load_experiment

    monkeypatch.chdir(tmp_path)
    assert validate_plan("llm-eval") == []
    plan = load_plan("llm-eval", smoke=True, only="stub-*")
    assert [v.name for v in plan.variants] == ["stub-plan-f", "stub-plan-net", "stub-basic-f", "stub-basic-net"]
    full = load_plan("llm-eval")
    assert len(full.variants) == 20
    net = full.variant("opus55-plan-net").config
    assert net["eval"]["final"][0]["lineup"][1].startswith("BELIEF:ckpt=ppo-2p-fr3/full@e4350:")  # fair: no POLICY
    assert net["eval"]["final"][0]["suite"] == {"name": "heldout-2p", "boards": 20}
    assert net["runtime"]["threads"] >= 8
    for v in plan.variants:
        (entry,) = v.config["eval"]["final"]
        assert entry["harness"] == "catanatron" and entry["suite"] == {"name": "heldout-2p", "boards": 2}
        assert entry["lineup"][0].startswith("LLM:model=stub/random:strategy=")
    v = plan.variant("stub-basic-f")
    assert run_variant(v, argv=["test"]) == "finished"
    run = load_experiment(v.ref)
    assert not _missing_finals(run, v)
    (record,) = [r for r in __import__("catanrl.eval.results", fromlist=["x"]).all_records("*")]
    from catanrl.eval.results import EvalRecord

    record = EvalRecord.load(record)
    assert record.key["harness"] == "catanatron" and record.summary["games"] == 4
    assert record.summary["slots"][0]["llm"]["games"] == 4
    # The net cell waits for ppo-2p-fr3/full, which this runs dir does not hold: deferred, not failed.
    v = plan.variant("stub-basic-net")
    assert run_variant(v, argv=["test"]) == "finished"
    assert not _missing_finals(load_experiment(v.ref), v)


def test_smoke_plans_play_every_llm_cell_with_a_stub(monkeypatch, tmp_path):
    from catanrl.config import load_plan
    from catanrl.eval.results import find
    from catanrl.runner import _missing_finals, run_variant
    from catanrl.runs import load_experiment

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    real = {v.name: v.config["eval"]["final"][0]["lineup"] for v in load_plan("llm-eval").variants}
    assert real["opus55-plan-f"][0].startswith("LLM:model=anthropic/claude-opus-5.5:strategy=plan:")
    smoke = load_plan("llm-eval", smoke=True)
    assert [v.name for v in smoke.variants] == list(real)
    for v in smoke.variants:
        (entry,) = v.config["eval"]["final"]
        assert entry["lineup"][0].startswith("LLM:model=stub/random:strategy="), v.name
        assert v.config["meta"]["smoke"]
    # A real-model cell plays end to end without a key (or a cap), on the stub cell's record.
    v = smoke.variant("opus55-basic-f")
    assert run_variant(v, argv=["test"]) == "finished"
    assert not _missing_finals(load_experiment(v.ref), v)
    suite = {"name": "heldout-2p", "boards": 2}
    assert find(v.config["eval"]["final"][0]["lineup"], suite, harness="catanatron")
    assert find(smoke.variant("stub-basic-f").config["eval"]["final"][0]["lineup"], suite,
                harness="catanatron").eval_id == find(v.config["eval"]["final"][0]["lineup"], suite,
                                                      harness="catanatron").eval_id


def test_the_llm_agent_is_outcome_code_of_the_catanatron_harness_only():
    from catanrl import provenance

    native = [label for label, _ in provenance._components("eval", "native")["catanrl"]["py"]]
    cat = [label for label, _ in provenance._components("eval", "catanatron")["catanrl"]["py"]]
    assert "agents/spec.py" in native and not any(label.startswith("agents/llm/") for label in native)
    assert {"agents/llm/prompts.py", "agents/llm/player.py", "agents/llm/render.py", "agents/llm/budget.py",
            "eval/catanatron.py"} <= set(cat)
    assert provenance.code_id("eval", "native")["catanrl"] != provenance.code_id("eval", "catanatron")["catanrl"]


def test_preflight_builds_the_model_client_before_any_game(monkeypatch):
    from catanrl.eval import catanatron as harness

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(harness, "play_game", lambda *a, **k: pytest.fail("a game was played without a model client"))
    suite = load_suite(SUITE)
    lineup = [resolve_agent("LLM:model=openai/gpt-5:strategy=basic", harness="catanatron"),
              resolve_agent("R", harness="catanatron")]
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        harness.play_catanatron(lineup, suite.schedule(2)[:1], suite=suite, on_game=lambda r: None)
    player = lineup[0].player(Color.RED)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        player.preflight()


def test_a_failing_game_stops_the_others_and_keeps_finished_rows(monkeypatch):
    import concurrent.futures

    from catanrl.eval import catanatron as harness

    suite = load_suite({**SUITE, "boards": 2})
    games = suite.schedule(2)  # two boards from both seats; the first two games share a seed
    first_seed, first_color = games[0].game_seed, harness.SEAT_COLORS[games[0].seats.index(0)]
    second_done = threading.Event()

    class Flaky(LlmPlayer):
        def decide(self, game, playable_actions):
            if game.seed == first_seed and self.color == first_color:
                second_done.wait(60)  # fail once the second game has finished
                raise RuntimeError("boom")
            return super().decide(game, playable_actions)

    class Agent:
        def player(self, color):
            return Flaky(color, LlmPlayer.Params(model="stub/first", strategy="basic"))

    real_row = harness._row

    def row(scheduled, **kwargs):
        if scheduled.game_id == games[1].game_id:
            second_done.set()
        return real_row(scheduled, **kwargs)

    monkeypatch.setattr(harness, "_row", row)
    monkeypatch.setattr(concurrent.futures, "as_completed", lambda fs: iter(list(fs)))  # the failing game first
    rows, before = [], threading.active_count()
    with pytest.raises(RuntimeError, match="boom"):
        harness.play_catanatron([Agent(), resolve_agent("R", harness="catanatron")], games, suite=suite, threads=2,
                                on_game=rows.append)
    assert threading.active_count() == before  # every worker joined: no game keeps playing (and paying)
    assert [r["game_id"] for r in rows] == [games[1].game_id]  # the game that finished is kept


def test_verify_refuses_a_real_models_record_unless_forced(monkeypatch):
    from catanrl.cli import main
    from catanrl.eval.results import EvalRecord, Unreplayable, evaluate, real_llm_slots, verify
    from catanrl.report import _eval_flags

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    suite = {**SUITE, "name": "llm-verify", "purpose": "heldout"}
    record = evaluate(["LLM:model=stub/first:strategy=basic", "F"], suite, harness="catanatron", threads=1)
    assert real_llm_slots(record.key) == [] and "nondeterministic" not in _eval_flags(record)
    assert verify(record, games=1) == {"identical": 1, "of": 1, "differing": []}
    assert main(["verify", record.eval_id, "--games", "1", "--force"]) == 0
    real = record.data["specs"][0].replace("stub/first", "openai/gpt-5")
    slot = record.key["lineup"][0]
    key = {**record.key, "lineup": [{**slot, "params": {**slot["params"], "model": "openai/gpt-5"}}, record.key["lineup"][1]]}
    data = {**record.data, "key": key, "specs": [real, record.data["specs"][1]]}
    doctored = EvalRecord(eval_id=record.eval_id, key=key, summary=record.summary, path=record.path, data=data)
    assert real_llm_slots(doctored.key) == [0] and "nondeterministic" in _eval_flags(doctored)
    with pytest.raises(Unreplayable, match="paid"):
        verify(doctored, games=1)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):  # forced: it goes on to play, which needs the key
        verify(doctored, games=1, force=True)


# ---------------------------------------------------------------- parsing (review fixes, wt/llm 1e61215)


@pytest.mark.parametrize("text, index", [
    ("Option 0 would build a road but I prefer the settlement.\nnotes\nFinal answer: 3", 3),
    ("I considered move 1 first.\nnotes\nAnswer: 2", 2),
    ("notes\nFinal answer: 3.", 3),
    ("notes\nI pick 3", 3),
    ("notes\nIndex: 3", 3),
    ("notes\n2 (end turn)", 2),
    ("notes\n3 (build settlement at node 12)", 3),
    ("notes\n**Action 2**", 2),
    ("notes\nMove 4", 4),
])
def test_parse_answer_reads_the_last_line_only(text, index):
    assert parse_answer(text, 20) == (index, "notes")


@pytest.mark.parametrize("text", [
    "notes\nBuild settlement at node 12",  # a node id is not an action number
    "notes\nI have 7 VP so far",
    "notes\nMove the robber to tile 3",
    "I choose action 2.\nLet's build toward the port.",  # the number is not on the last line
    "Option 1 looks fine.\nnotes\nI will go with the port.",
])
def test_parse_answer_does_not_take_a_number_from_the_reasoning(text):
    with pytest.raises(AnswerError, match="no action number"):
        parse_answer(text, 20)
