"""Game review: an archived site game analyzed by a search from each mover's seat."""

import json

from catanatron.models.player import Color
from catanatron.players import register_builtins
from catanatron.serialization import state_to_json
from catanatron.server.config import GameConfig, new_game

from catanrl.bridge.analysis import GRADES, Analyst, accuracy, grade, stakes

from tests.bridge.bridge_games import tiny_agent

DIMS = {"max_players": 3, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}


def _record(num_players=3):
    """An archive record as catanatron-server writes one: the room and the final document."""
    register_builtins()
    config = GameConfig.from_json({"num_players": num_players, "vps_to_win": 6})
    game = new_game(config, b"\x07" * 32, "g-1")
    from catanatron.players.weighted_random import WeightedRandomPlayer

    bots = {c: WeightedRandomPlayer(c) for c in game.state.colors}
    while game.winning_color() is None and game.state.num_turns < 400:
        color = game.state.current_color()
        game.execute(bots[color].decide(game, game.playable_actions))
    seats = [{"seat": i, "color": c.value, "kind": "remote", "name": f"p{i}"} for i, c in enumerate(Color) if i < num_players]
    return {"room": {"id": "r1", "name": "test", "config": config.to_json(), "seats": seats, "game_id": "g-1"},
            "final_state": json.loads(json.dumps(state_to_json(game)))}, None


def test_grades_and_accuracy_follow_the_loss():
    assert grade(0.0, True) == "best" and grade(0.1, False) == "excellent"
    assert grade(3.0, False) == "mistake" and grade(9.0, False) == "blunder"
    assert [name for name, _ in GRADES] == ["excellent", "good", "inaccuracy", "mistake"]
    assert accuracy([0.0, 0.0]) == 100 and accuracy([]) is None
    assert accuracy([0.0, 5.0]) < accuracy([0.0, 1.0]) < 100
    # Weighted by stakes: a blunder where nothing was at stake costs nothing.
    assert accuracy([0.0, 20.0], [1.0, 0.0]) == 100 and accuracy([0.0, 20.0], [0.0, 1.0]) == 0
    assert accuracy([0.0, 20.0], [0.0, 0.0]) == accuracy([0.0, 20.0])


def test_stakes_are_the_best_value_over_the_mean():
    alike = {"a": (None, 0.50, 10), "b": (None, 0.50, 10)}
    split = {"a": (None, 0.60, 10), "b": (None, 0.40, 10), "c": (None, 0.20, 0)}
    assert stakes(alike, 2) == 0.0 and abs(stakes(split, 2) - 10.0) < 1e-9
    assert stakes({"a": (None, 0.9, 5)}, 2) == 0.0  # one searched option: no choice to weigh


def test_an_archived_game_is_analyzed_move_by_move():
    record, _ = _record()
    analyst = Analyst(tiny_agent("MCTS", "full", DIMS, sims=16, dets=1, root="gumbel", gumbel_m=32), label="tiny",
                      recheck=64)
    result = analyst.analyze(record)
    json.dumps(result)  # integers and strings only: the lobby takes no floats
    log = record["final_state"]["action_records"]
    colors = record["final_state"]["colors"]
    assert result["summary"]["decisions"] == len(result["moves"]) > 20
    assert analyst.agent.stats.get("decisions", 0) >= len(result["moves"])
    assert not [k for k in analyst.agent.stats if k.startswith("fallback")]
    for index, values in result["evals"]:
        assert len(values) == len(colors) and abs(sum(values) - 1000) <= len(colors)
    for move in result["moves"]:
        assert move["c"] == log[move["i"]][0][0]
        assert move["b"] >= move["p"] or move["g"] == "best"
        assert move["g"] in {"best", "excellent", "good", "inaccuracy", "mistake", "blunder"}
        assert all(isinstance(v, int) for v in (move["p"], move["b"], move["n"]))
    assert set(result["summary"]["accuracy"]) <= set(colors)
    # Every move inaccuracy or worse was searched again, deeper, and kept that grade.
    rechecked = [m for m in result["moves"] if "d" in m]
    assert result["summary"]["rechecked"] == len(rechecked) > 0 and all(m["d"] == 64 for m in rechecked)
    assert all(isinstance(m["s"], int) and m["s"] >= 0 for m in result["moves"])
    assert [e[0] for e in result["evals"]] == sorted(e[0] for e in result["evals"])


def test_searching_decisions_together_reviews_the_game_as_one_at_a_time():
    record, _ = _record()
    spec = dict(sims=16, dets=1, root="gumbel", gumbel_m=32)
    one = Analyst(tiny_agent("MCTS", "full", DIMS, **spec), label="tiny", onnx=None, chunk=1, recheck=None).analyze(record)
    many = Analyst(tiny_agent("MCTS", "full", DIMS, **spec), label="tiny", onnx=None, chunk=64, recheck=None).analyze(record)
    assert [m["i"] for m in one["moves"]] == [m["i"] for m in many["moves"]]
    same = sum(a["a"] == b["a"] and a["g"] == b["g"] for a, b in zip(one["moves"], many["moves"]))
    assert same >= 0.95 * len(one["moves"])


def test_onnx_runtime_reviews_as_torch_does():
    import pytest

    pytest.importorskip("onnxruntime")
    record, _ = _record()
    spec = dict(sims=16, dets=1, root="gumbel", gumbel_m=32)
    plain = Analyst(tiny_agent("MCTS", "full", DIMS, **spec), label="tiny", onnx=None, recheck=None).analyze(record)
    analyst = Analyst(tiny_agent("MCTS", "full", DIMS, **spec), label="tiny", onnx=1, recheck=None)
    fast = analyst.analyze(record)
    inner = getattr(analyst.agent.native.evaluator, "evaluator", analyst.agent.native.evaluator)
    assert type(inner).__name__ == "OnnxEvaluator" and inner.checked and inner.rows > 0
    same = sum(a["a"] == b["a"] for a, b in zip(plain["moves"], fast["moves"]))
    assert len(fast["moves"]) == len(plain["moves"]) and same >= 0.9 * len(plain["moves"])


class _Refusing:
    """A pool whose first ``n`` adds refuse the position as likely worlds do."""

    def __init__(self, pool, n):
        self.pool, self.n = pool, n

    def add(self, *args, **kwargs):
        if self.n > 0:
            self.n -= 1
            raise ValueError("likely worlds need a position whose hidden cards have at most 65536 possible deals")
        return self.pool.add(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.pool, name)


def test_likely_worlds_and_the_sampling_fallback():
    record, _ = _record()
    spec = dict(sims=16, dets=2, root="gumbel", gumbel_m=32)
    analyst = Analyst(tiny_agent("MCTS", "full", DIMS, world_choice="likely", **spec), label="tiny", onnx=None,
                      recheck=None)
    analyst.agent.likely = True
    analyst._fallback_args = (tiny_agent("MCTS", "full", DIMS, **spec), "cpu", 64, None)
    plain = analyst.analyze(record)
    assert analyst.fallback is None  # (a 3-player position of this game always lists)
    analyst.agent.native.pool = _Refusing(analyst.agent.native.pool, 5)
    mixed = analyst.analyze(record)
    # The refused positions went to the sampling searcher: every decision is still searched.
    assert analyst.fallback is not None
    assert [e[0] for e in mixed["evals"]] == [e[0] for e in plain["evals"]]
    assert not [k for k in analyst.agent.stats if k.startswith("fallback")]
