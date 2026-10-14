"""Training into real runs (catanrl.runs.RunWriter) from the training tests."""

from __future__ import annotations

from catanrl.config import Variant, config_hash, resolve
from catanrl.runs import RunWriter


class TestWriter(RunWriter):
    """A run writer that can stop after ``stop_after`` epochs of its launch."""

    __test__ = False
    stop_after: int | None = None

    @property
    def rows(self) -> list[dict]:
        """This launch's metric rows."""
        return [r for r in self.metrics() if r["launch"] == self.launch]

    @property
    def saved(self):
        return self.checkpoints()

    def should_stop(self) -> bool:
        if self.stop_after is not None and len(self.rows) >= self.stop_after:
            self.stop_reason = "interrupted"
            return True
        return super().should_stop()


def train(fn, config: dict, name: str = "v", *, stop_after: int | None = None) -> TestWriter:
    """Run ``fn(variant, writer)`` (train_ppo, train_imitation) in run ``test/<name>``."""
    cfg = resolve(config, experiment="test", variant=name)
    writer = TestWriter.open(Variant("test", name, None, {}, cfg, config_hash(cfg)), argv=["pytest"])
    writer.stop_after = stop_after
    with writer:
        fn(writer.variant_for_training, writer)
    return writer
