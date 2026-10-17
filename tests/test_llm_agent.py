"""The LLM agent: specs, registry, stub games through the Catanatron harness and
``catanrl eval``, the public-information render, answer parsing, token and cost
accounting, and the llm-eval experiment's stub cells. No network: every model here
is a stub, and a real model refuses to build without OPENROUTER_API_KEY."""

from __future__ import annotations

import json

import pytest
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.player import Color
from catanatron.registry import REGISTRY, SpecError

from catanrl.agents import check_spec_syntax, resolve_agent
from catanrl.agents.llm import LLMError, LlmPlayer, StubModel, parse_answer, render_state
from catanrl.agents.llm.model import OpenRouterModel, estimate_tokens, make_model
from catanrl.agents.llm.player import AnswerError
from catanrl.agents.llm.render import player_status, render_actions
from catanrl.eval.catanatron import play_catanatron, reseat
from catanrl.eval.suites import load_suite
from catanrl.games import load_game

SPEC = "LLM:model=stub/random:strategy=plan"
SUITE = {"name": "llm-test", "purpose": "dev", "seed": 11, "boards": 1, "scenarios": [{"game": "base-2p"}],
         "turn_limit": 200}


def two_player_game(red, blue, *, map_seed=3, seed=5):
    game = load_game("base-2p").catanatron_game([red, blue], map_seed=map_seed, seed=seed)
    reseat(game, [red, blue])
    return game


# ---------------------------------------------------------------- specs and registry


def test_spec_canonical_form_is_a_fixed_point():
    agent = resolve_agent(SPEC, harness="catanatron")
    assert agent.canonical == ("LLM:model=stub/random:strategy=plan:thinking=true:max_tokens=8192:temperature=0.1"
                               ":seed=0:retries=1")
    assert resolve_agent(agent.canonical, harness="catanatron").canonical == agent.canonical
    assert agent.fair and agent.checkpoint is None and agent.label == "LLM:stub/random:plan"
    assert agent.key_dict() == {"key": "LLM", "harness": "catanatron", "params": {
        "model": "stub/random", "strategy": "plan", "thinking": True, "max_tokens": 8192, "temperature": 0.1,
        "seed": 0, "retries": 1}}
    other = resolve_agent("LLM:model=stub/random:strategy=basic", harness="catanatron")
    assert other.key_dict() != agent.key_dict()


def test_bad_specs_raise():
    for spec in ("LLM", "LLM:model=stub/nope", "LLM:model=stub/first:strategy=zen", "LLM:model=stub/first:foo=1",
                 "LLM:model=stub/first:thinking=maybe", "LLM:model=stub/first:max_tokens=0",
                 "LLM:model=stub/first:retries=-1"):
        with pytest.raises(SpecError):
            check_spec_syntax(spec)
        with pytest.raises(SpecError):
            resolve_agent(spec, harness="catanatron")
    with pytest.raises(SpecError, match="catanatron harness"):
        resolve_agent(SPEC)  # the native harness has no LLM
    check_spec_syntax("LLM:model=${params.model}:strategy=${params.strategy}:thinking=true")  # a plan, unresolved


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
    assert "LLM:stub/first:basic" in out and "per game:" in out and "(estimated)" in out
    record = find(["LLM:model=stub/first:strategy=basic", "F"], str(path), harness="catanatron")
    assert record is not None and record.key["harness"] == "catanatron" and "catanatron" in record.key["code"]
    assert record.data["fair"] == [True, False] and record.data["valid"]
    rows = list(record.games())
    assert len(rows) == 2 and all(r["llm"][0]["decisions"] > 0 for r in rows)
    llm = record.summary["slots"][0]["llm"]
    assert llm["games"] == 2 and llm["decisions"] == sum(r["llm"][0]["decisions"] for r in rows)
    assert llm["fallbacks"] == 0 and llm["estimated"] and llm["per_game"]["prompt_tokens"] > 1000
    with pytest.raises(SpecError, match="catanatron harness"):  # no native record can hold an LLM
        find(["LLM:model=stub/first:strategy=basic", "F"], str(path))
    assert main(["eval", "LLM:model=stub/first:strategy=basic", "F", "--suite", str(path), "--harness",
                 "catanatron"]) == 0  # cached
    assert record.eval_id in capsys.readouterr().out


# ---------------------------------------------------------------- the public-information render


def test_the_state_text_never_shows_an_opponents_cards():
    red = resolve_agent(SPEC, harness="catanatron").player(Color.RED)
    blue = REGISTRY.build("R", Color.BLUE)
    game = two_player_game(red, blue)
    for _ in range(12):  # past the initial placements
        game.play_tick()
    state = game.state
    key = f"P{state.color_to_index[Color.BLUE]}"
    for resource in RESOURCES:  # a hand the observer could not know: one of each kind, seven ore
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
    blue_line = player_status(state, Color.BLUE, Color.RED)
    red_line = player_status(state, Color.RED, Color.RED)
    assert blue_line in text and red_line in text
    # Public: the size of the hand, the number of dev cards, what was played, the visible points.
    assert "11 resource cards" in blue_line and "3 dev cards in hand" in blue_line and "KNIGHT: 3" in blue_line
    assert f"{state.player_state[f'{key}_VICTORY_POINTS']} VP (visible)" in blue_line
    # Hidden: no resource kind, no card in hand, no victory-point card, no actual VP.
    assert not any(label in blue_line for label in ("Wood", "Brick", "Sheep", "Wheat", "Ore"))
    assert "Ore: 7" not in text and "ORE: 7" not in text and "7 ORE" not in text
    assert "KNIGHT: 2" not in text and "VICTORY_POINT" not in text and "MONOPOLY: 0" not in text
    assert f"{state.player_state[f'{key}_ACTUAL_VICTORY_POINTS']} VP (visible)" not in text
    # The observer's own hand and cards are shown.
    assert "Wheat: 4" in red_line and "MONOPOLY: 1" in red_line and "(you)" in red_line
    # Every hidden fact of the opponent, as a count, is absent from the prompt the model sees.
    red.notes = []
    prompt = red.user_message(game, game.playable_actions)
    assert blue_line in prompt and "Previous moves" not in prompt
    hidden = [f"{label}: {state.player_state[f'{key}_{r}_IN_HAND']}" for label, r in
              zip(("Wood", "Brick", "Sheep", "Wheat", "Ore"), RESOURCES)]
    assert not any(h in prompt.split("=== BOARD LAYOUT ===")[0].split("\n")[-4] for h in hidden)


def test_actions_are_numbered_as_the_model_answers_them():
    game = two_player_game(REGISTRY.build("R", Color.RED), REGISTRY.build("R", Color.BLUE))
    text = render_actions(game.playable_actions)
    lines = text.splitlines()
    assert lines[0] == "=== AVAILABLE ACTIONS ===" and len(lines) == len(game.playable_actions) + 1
    assert lines[1].startswith("0: Build settlement at node ")
    assert StubModel.count_actions([{"role": "user", "content": text}]) == len(game.playable_actions)


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


def play_decisions(script, retries=1, decisions=1):
    """``decisions`` prompted decisions of a RED stub player answering ``script`` in order."""
    params = LlmPlayer.Params(model="stub/scripted", retries=retries)
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
    assert "could not be used" in red.log[0]["exchanges"][0]["error"] or red.log[0]["exchanges"][0]["error"]
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
    assert "Previous moves: Taking the last option this turn." in red.log[1]["prompt"]


def test_fallback_prefers_end_turn():
    from catanatron.models.enums import Action, ActionType

    from catanrl.agents.llm.player import fallback_action

    build = Action(Color.RED, ActionType.BUILD_ROAD, (0, 1))
    end = Action(Color.RED, ActionType.END_TURN, None)
    assert fallback_action([build, end]) is end and fallback_action([build]) is build


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
    assert measured["plan"]["per_game"]["prompt_tokens"] != measured["basic"]["per_game"]["prompt_tokens"]
    lines = table(measured, {"m": (1.0, 2.0)}, completion_per_decision=100, cell_games=50)
    pg = measured["plan"]["per_game"]
    expected = (pg["prompt_tokens"] * 1.0 + pg["decisions"] * 100 * 2.0) / 1e6
    assert any(line.startswith("plan") and f"{expected:8.3f}" in line and f"{expected * 50:12.2f}" in line
               for line in lines)


# ---------------------------------------------------------------- the experiment


def test_llm_eval_stub_cells_smoke(monkeypatch, tmp_path):
    from catanrl.config import load_plan, validate_plan
    from catanrl.runner import _missing_finals, run_variant
    from catanrl.runs import load_experiment

    monkeypatch.chdir(tmp_path)
    assert validate_plan("llm-eval") == []
    plan = load_plan("llm-eval", smoke=True, only="stub-*")
    assert [v.name for v in plan.variants] == ["stub-plan-f", "stub-plan-policy", "stub-basic-f", "stub-basic-policy"]
    assert len(load_plan("llm-eval").variants) == 16
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
    # The POLICY cell waits for puffer-graph2/full, which this runs dir does not hold: deferred, not failed.
    v = plan.variant("stub-basic-policy")
    assert run_variant(v, argv=["test"]) == "finished"
    assert not _missing_finals(load_experiment(v.ref), v)


# ---------------------------------------------------------------- the real client, without a network


def test_openrouter_client_reads_usage_and_asks_for_reasoning(monkeypatch):
    """The SDK call shape and the usage fields, against a fake client (no request is made)."""
    from types import SimpleNamespace

    pytest.importorskip("openai")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    model = OpenRouterModel("vendor/model", api_key="not-a-key")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="thinking aloud\nnotes\n3", reasoning="private chain")
        usage = SimpleNamespace(prompt_tokens=1234, completion_tokens=56)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    reply = model.complete(messages, max_tokens=1000, temperature=0.2, seed=7, thinking=True)
    assert (reply.text, reply.reasoning, reply.prompt_tokens, reply.completion_tokens, reply.estimated) == \
        ("thinking aloud\nnotes\n3", "private chain", 1234, 56, False)
    (call,) = calls
    assert call["model"] == "vendor/model" and call["messages"] == messages and call["seed"] == 7
    assert call["max_tokens"] == 1000 and call["temperature"] == 0.2
    assert call["extra_body"] == {"reasoning": {"max_tokens": 750}}
    model.complete(messages, max_tokens=1000, temperature=0.2, seed=0, thinking=False)
    assert calls[1]["extra_body"] == {"reasoning": {"enabled": False}} and "seed" not in calls[1]

    def no_usage(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="2"))], usage=None)

    model.client.chat.completions.create = no_usage
    reply = model.complete(messages, max_tokens=10, temperature=0.0, seed=0, thinking=False)
    assert reply.estimated and reply.prompt_tokens == estimate_tokens("su") and reply.completion_tokens == 1
    player = LlmPlayer(Color.RED, LlmPlayer.Params(model="vendor/model"), model=model)
    game = two_player_game(player, REGISTRY.build("R", Color.BLUE))
    assert player.decide(game, game.playable_actions) is game.playable_actions[2]
    assert player.stats["prompt_tokens"] > 1000 and player.stats["estimated"] and not player.stats["fallbacks"]


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
