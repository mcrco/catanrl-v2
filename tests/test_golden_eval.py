"""Test 23: the committed tiny graph net gives pinned per-game outcomes
against F for POLICY, MCTS (sims=16) and BELIEF (worlds=4), strict on CPU.

``CATANRL_UPDATE_GOLDEN=1`` rewrites tests/data/golden_eval.json; review
the diff, since any change means outcome code changed."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from eval_helpers import agent, make_suite, schedule

from catanrl.envs.encoding import action_size, row_layout
from catanrl.eval.harness import play_schedule
from catanrl.models import build_model

DATA = Path(__file__).with_name("data")
GOLDEN = DATA / "golden_eval.json"
SUITE = make_suite(({"game": "base-2p"},), turn_limit=250)
GAMES = schedule(SUITE, 2, boards=16, seed=23)  # 32 games
CASES = {
    "POLICY": ({}, GAMES),
    "MCTS": ({"sims": 16, "dets": 1}, GAMES[:8]),
    "BELIEF": ({"worlds": 4}, GAMES),
}


def tiny_graph():
    saved = torch.load(DATA / "tiny-graph.pt", weights_only=True)
    model = build_model(saved["model"], dims=saved["dims"], levels=saved["levels"],
                        num_actions=action_size(saved["dims"]), layout=row_layout(saved["dims"], "full"))
    model.load_state_dict(saved["state_dict"])
    return model


def outcomes(key: str) -> list[list]:
    params, games = CASES[key]
    rows = []
    play_schedule([agent(key, tiny_graph(), infer_rows=8, **params), agent("F")], games, suite=SUITE,
                  on_game=rows.append, device="cpu", num_envs=8)
    return [[r["game_id"], r["winner"], r["turns"], r["truncated"]] for r in sorted(rows, key=lambda r: r["game_id"])]


@pytest.mark.parametrize("key", sorted(CASES))
def test_golden_outcomes(key):
    found = outcomes(key)
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if os.environ.get("CATANRL_UPDATE_GOLDEN"):
        golden[key] = found
        GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    assert key in golden, "no golden outcomes yet: run with CATANRL_UPDATE_GOLDEN=1"
    assert found == golden[key]
