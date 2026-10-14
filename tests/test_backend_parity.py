"""Test 24: per position, a greedy POLICY picks the same action natively (on
cppanatron) and as a Catanatron Player (the game converted through
apply_setup). Both engines play the same game in lockstep through
cppanatron.parity."""

from __future__ import annotations

import numpy as np
import pytest
from eval_standins import tiny_model

checkpoint = pytest.importorskip("catanrl.checkpoint")
spec = pytest.importorskip("catanrl.agents.spec")
provenance = pytest.importorskip("catanrl.provenance")

from cppanatron import parity  # noqa: E402
from cppanatron.catanatron import from_catanatron_action  # noqa: E402

from catanrl import games  # noqa: E402
from catanrl.agents.native import make_native_agent  # noqa: E402


def tiny_checkpoint(tmp_path):
    model = tiny_model(["base-2p"], level="private", seed=4)
    config = {"seed": 0, "env": {"scenarios": [{"game": "base-2p"}], "num_envs": 1},
              "obs": {"level": "private", "critic_level": "private"}, "model": model.spec}
    ref = checkpoint.save(tmp_path / "e000000.pt", model, run=None, epoch=0, agent_steps=0, config=config,
                          code=provenance.code_id("eval"))
    return ref.path


def test_native_and_player_agree(tmp_path):
    path = tiny_checkpoint(tmp_path)
    text = f"POLICY:ckpt={path}"
    resolved = spec.resolve_agent(text)
    native = make_native_agent(resolved, device="cpu", infer_rows=1)
    config = games.load_game("base-2p")
    rng = np.random.default_rng(0)
    checked = 0
    for seed in range(4):
        spec_dict = config.map_spec(seed)
        engines = {name: parity.new_game(name, 2, seed=seed, map_spec=spec_dict, vps_to_win=15, discard_limit=9)
                   for name in parity.ENGINES}
        native_game = engines["cppanatron"].game
        for _ in range(60):
            actions = engines["catanatron"].playable_actions
            if engines["catanatron"].winning_color is not None or not actions:
                break
            if len(actions) > 1:
                seat = native_game.current_player_index
                row = native.encode(native_game, seat)[None]
                index = native.act(row, row[:, -native.action_space.size:], np.zeros(1, np.uint64))[0]
                expected = native.action_space.decode(native_game, int(index))
                color = engines["catanatron"].current_color
                player = resolved.player(color)
                chosen = player.decide(engines["catanatron"].game, actions)
                assert tuple(from_catanatron_action(chosen)) == tuple(expected)
                checked += 1
            action = actions[int(rng.integers(len(actions)))]
            outcome = engines["catanatron"].execute(action)
            engines["cppanatron"].execute(action, outcome)
    assert checked > 20
