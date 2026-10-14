"""Test 14: end to end on CPU: a tiny graph net trains 3 epochs in 2p (vs F)
and 4p (vs three R); best cites a select eval; build_policy, load_checkpoint
and POLICY: agree; evals are cached; a stopped run resumes with its LR and
optimizer step."""

from __future__ import annotations

import pytest
import torch

import catanrl
from catanrl import canon
from catanrl.agents import resolve_agent
from catanrl.checkpoint import encode_positions, select_positions
from catanrl.config import load_plan
from catanrl.eval import results
from catanrl.runner import final_evals, run_variant
from catanrl.runs import RunWriter


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("name", ["2p", "4p"])
def test_tiny_end_to_end(name, monkeypatch):
    v = load_plan("tiny", variant=name).variants[0]
    assert run_variant(v, argv=["test"]) == "finished"
    run = catanrl.load_experiment(f"tiny/{name}")
    assert [c.epoch for c in run.checkpoints()] == [0, 1, 2, 3]
    assert len(run.metrics()) == 3

    best = run.aliases()["best"]
    record = results.EvalRecord.load(next(run.path.glob(f"evals/*/{best['eval_id']}.json")))
    assert record.key["suite"]["purpose"] == "select" and record.data["refs"][0] == best["ref"]

    net = run.build_policy("best")
    same = catanrl.load_checkpoint(f"tiny/{name}@best").model()
    agent = resolve_agent(f"POLICY:tiny/{name}").native(device="cpu", infer_rows=4)
    obs = torch.from_numpy(encode_positions(select_positions(net.dims, 4), net.dims, "private"))
    with torch.no_grad():
        logits = net(obs)[0]
        assert torch.equal(logits, same(obs)[0]) and torch.equal(logits, agent.model(obs)[0])
    assert net.catanrl_ref == best["ref"]

    def no_play(*args, **kwargs):
        raise AssertionError("a cached eval replayed games")

    launch = canon.read_json(run.path / "run.json")["launches"][0]
    assert launch["runtime"]["determinism"] == "strict"
    assert launch["effective"]["torch"]["flags"]["deterministic_algorithms"] is True  # after PuffeRL's init

    if name == "2p":  # an ad-hoc eval of another checkpoint is not a final: summarize skips it
        catanrl.evaluate(["POLICY:tiny/2p@e1", "F"], {"name": "heldout-2p", "boards": 1})
        from catanrl.runner import summarize

        assert "| players=2p | 0:POLICY vs F | heldout-2p |" in summarize("tiny").read_text()
        assert summarize("tiny").read_text().rstrip().endswith("| 1 |")

    monkeypatch.setattr(results, "play_schedule", no_play)
    finals = final_evals(run, v)
    assert len(finals) == 1 and finals[0].data["refs"][0] == best["ref"]


def test_stop_and_resume_keep_lr_and_optimizer_step(monkeypatch):
    v = load_plan("tiny", variant="2p").variants[0]
    real = RunWriter.should_stop

    def stop_after_one_epoch(self):
        if self.launch == 0 and len(canon.read_jsonl(self.path / "metrics.jsonl")) >= 1:
            self.stop_reason = "interrupted"
            return True
        return real(self)

    monkeypatch.setattr(RunWriter, "should_stop", stop_after_one_epoch)
    assert run_variant(v, argv=["test"]) == "interrupted"
    run = catanrl.load_experiment("tiny/2p")
    assert [c.epoch for c in run.checkpoints()] == [0, 1]
    assert run_variant(v, argv=["test"]) == "finished"
    state = torch.load(run.path / "trainer_state.pt", weights_only=False)
    steps = {int(s["step"]) for s in state["optimizer"]["state"].values()}
    assert steps == {3 * 2}  # 3 epochs x 2 minibatches, across the resume
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-3)
    launches = canon.read_json(run.path / "run.json")["launches"]
    assert [(l["mode"], l["status"]) for l in launches] == [("new", "interrupted"), ("resume", "finished")]
    assert [r["launch"] for r in run.metrics()] == [0, 1, 1]
