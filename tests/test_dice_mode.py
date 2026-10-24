"""``dice_mode`` in game configs: ``random`` (two fair dice, the default) or
``balanced`` (OpenHex's 1v1 dice, ``cppanatron.BalancedDice``).

The default never enters a game's YAML shape, so every game, scenario, suite and
experiment hashes as before it existed: the goldens (tests/data/dice_mode_hashes.json)
were recorded by the code before dice_mode (cfec40a) for every file under
configs/games, configs/suites and configs/experiments, and each is checked while
the files its hashes come from are unchanged (an entry whose sources changed since is
skipped; the structural check, no ``dice_mode`` in any resolved config, runs on every
file). A balanced game keys apart from its fair twin, plays in the native harness,
and is refused by the catanatron harness (and by ``validate_plan`` for a catanatron
final), whose Catanatron games roll fair dice.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import cppanatron as cpp
import pytest
import yaml
from eval_helpers import agent, make_suite, schedule

from catanrl.canon import ast_digest, digest
from catanrl.config import load_plan, validate_plan
from catanrl.eval.harness import play_schedule
from catanrl.eval.suites import Suite, load_suite
from catanrl.games import GameConfig, load_game

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PACKAGE = ROOT / "src" / "catanrl"
GOLDEN = ROOT / "tests" / "data" / "dice_mode_hashes.json"
#: Boards of a suite whose schedule the goldens hash (the scenarios decide the rest).
SUITE_BOARDS = 8


# ---------------------------------------------------------------- hashes and their sources


def _sources_digest(files, python=()) -> str:
    h = hashlib.sha256()
    for path in sorted(set(files)):
        h.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    if python:
        h.update(ast_digest([(p.name, p) for p in python]).encode())
    return h.hexdigest()[:16]


def _game_files(names) -> list[Path]:
    return [CONFIGS / "games" / f"{n}.yaml" for n in names if (CONFIGS / "games" / f"{n}.yaml").exists()]


def game_hashes(path: Path) -> dict:
    return {"sources": _sources_digest([path]), "hash": digest(load_game(path).to_dict())}


def suite_hashes(path: Path) -> dict:
    suite = load_suite(path)
    names = [s["game"] for s in suite.scenarios if isinstance(s["game"], str)]
    short = Suite(**{**suite.__dict__, "boards": min(suite.boards, SUITE_BOARDS)})
    players = max(g.num_players for g in suite.games())
    return {"sources": _sources_digest([path, *_game_files(names)], [PACKAGE / "maps.py", PACKAGE / "seeding.py"]),
            "scenarios": [suite.scenario_hash(i) for i in range(len(suite.scenarios))],
            "key": digest(short.key(players, fill=True))}


def experiment_hashes(path: Path) -> dict:
    full, smoke = load_plan(path), load_plan(path, smoke=True)
    names = {s["game"]["name"] for plan in (full, smoke) for v in plan.variants for s in v.config["env"]["scenarios"]}
    defaults = [*sorted((CONFIGS / "train").glob("*.yaml")), *sorted((CONFIGS / "models").glob("*.yaml"))]
    return {"sources": _sources_digest([path, *defaults, *_game_files(names)], [PACKAGE / "config.py"]),
            "full": {v.name: v.config_hash for v in full.variants},
            "smoke": {v.name: v.config_hash for v in smoke.variants}}


KINDS = {"games": game_hashes, "suites": suite_hashes, "experiments": experiment_hashes}


def all_hashes() -> dict:
    """Every config file's hashes (what the goldens hold)."""
    return {kind: {p.name: fn(p) for p in sorted((CONFIGS / kind).glob("*.yaml"))} for kind, fn in KINDS.items()}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_existing_hashes_are_what_they_were_before_dice_mode(kind):
    golden = json.loads(GOLDEN.read_text())[kind]
    checked = 0
    for name, want in golden.items():
        path = CONFIGS / kind / name
        if not path.exists():
            continue
        got = KINDS[kind](path)
        if got["sources"] != want["sources"]:
            continue  # its sources changed since the goldens: nothing to compare
        assert got == want, f"{kind}/{name}"
        checked += 1
    if not checked:
        pytest.skip(f"every {kind} golden's sources changed")


def _keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _keys(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _keys(v)


def test_no_existing_config_names_a_dice_mode():
    for path in sorted((CONFIGS / "games").glob("*.yaml")):
        game = load_game(path)
        assert game.dice_mode == "random" and "dice_mode" not in game.to_dict()
        assert game.cppanatron_rules().dice_mode == cpp.DiceMode.RANDOM
    for path in sorted((CONFIGS / "suites").glob("*.yaml")):
        suite = load_suite(path)
        assert "dice_mode" not in set(_keys([suite._scenario(i) for i in range(len(suite.scenarios))]))
    for path in sorted((CONFIGS / "experiments").glob("*.yaml")):
        for v in load_plan(path).variants:
            assert "dice_mode" not in set(_keys(v.config)), f"{path.name}: {v.name}"


# ---------------------------------------------------------------- a balanced game


def balanced(game: str | GameConfig = "base-2p") -> GameConfig:
    """``game`` with OpenHex's balanced dice (a test-only game: no config file names one yet)."""
    data = (game if isinstance(game, GameConfig) else load_game(game)).to_dict()
    return GameConfig.from_dict({**data, "name": f"{data['name']}-balanced", "dice_mode": "balanced"})


def test_dice_mode_reads_writes_and_keys_only_when_set():
    fair = load_game("base-2p")
    assert GameConfig.from_dict({**fair.to_dict(), "dice_mode": "random"}) == fair
    game = balanced()
    assert game.dice_mode == "balanced" and game.to_dict()["dice_mode"] == "balanced"
    assert GameConfig.from_dict(game.to_dict()) == game
    assert {k: v for k, v in game.to_dict().items() if k not in ("name", "dice_mode")} == \
        {k: v for k, v in fair.to_dict().items() if k != "name"}
    rules = game.cppanatron_rules()
    assert rules.dice_mode == cpp.DiceMode.BALANCED and rules != fair.cppanatron_rules()
    assert dataclasses.replace(game, dice_mode="random").cppanatron_rules() == fair.cppanatron_rules()
    played = game.cppanatron_game(map_seed=3, seed=4)
    assert played.balanced_dice == cpp.BalancedDice.fresh(2)
    assert fair.cppanatron_game(map_seed=3, seed=4).balanced_dice is None
    # A balanced scenario is another scenario: its own hash, so its own boards.
    suites = [make_suite(({"game": g.to_dict()},)) for g in (fair, game)]
    assert suites[0].scenario_hash(0) == make_suite(({"game": "base-2p"},)).scenario_hash(0)
    assert suites[0].scenario_hash(0) != suites[1].scenario_hash(0)
    with pytest.raises(ValueError, match="dice_mode"):
        GameConfig.from_dict({**fair.to_dict(), "dice_mode": "loaded"})
    with pytest.raises(ValueError, match="fair dice"):
        game.catanatron_game([], map_seed=3, seed=4)


def test_the_native_harness_rolls_the_games_dice():
    """The same schedule (maps, seeds, seats) under fair and balanced dice: the dice
    reach the engine (the games differ), and a balanced game replays exactly."""
    fair, game = load_game("base-2p"), balanced()
    games = schedule(make_suite(({"game": "base-2p"},)), 2, boards=3)

    def play(config):
        rows = []
        play_schedule([agent("F"), agent("F")], games, scenarios=[config], turn_limit=300, on_game=rows.append,
                      num_envs=2)
        return sorted(rows, key=lambda r: r["game_id"])

    rolled = play(game)
    assert len(rolled) == len(games) and rolled == play(game)
    assert [(r["turns"], r["winner"]) for r in rolled] != [(r["turns"], r["winner"]) for r in play(fair)]


def test_the_catanatron_harness_refuses_balanced_dice():
    from catanrl.eval.catanatron import play_catanatron

    class Unbuilt:
        def player(self, color):
            raise AssertionError("refused before any player is built")

    suite = make_suite(({"game": balanced().to_dict()},))
    with pytest.raises(ValueError, match="fair dice"):
        play_catanatron([Unbuilt(), Unbuilt()], schedule(suite, 2, boards=1), suite=suite, on_game=lambda row: None)


def test_validate_plan_flags_a_catanatron_final_on_balanced_dice(tmp_path):
    suite = {"purpose": "dev", "seed": 1, "boards": 2, "turn_limit": 200,
             "scenarios": [{"game": balanced().to_dict()}]}
    (tmp_path / "bd.yaml").write_text(yaml.safe_dump(suite))
    final = {"suite": str(tmp_path / "bd.yaml"), "lineup": ["F", "F"]}
    body = {"name": "exp", "seed": 1, "env": {"scenarios": [{"game": "base-2p"}], "num_envs": 8},
            "model": {"preset": "graph-tiny"}, "train": {"algo": "ppo", "total_timesteps": 1000}}
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({**body, "eval": {"final": [{**final, "harness": "catanatron"}]}}))
    problems = validate_plan(path)
    assert len(problems) == 1 and "fair dice" in problems[0] and "base-2p-balanced" in problems[0]
    path.write_text(yaml.safe_dump({**body, "eval": {"final": [final]}}))
    assert validate_plan(path) == []
