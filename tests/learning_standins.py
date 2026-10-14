"""Small local stand-ins for the experiment infrastructure (catanrl.runs,
catanrl.config) used by the learning-side tests until, or besides, the real
modules: a run writer with the RunWriter methods train_ppo / train_imitation
call, and a variant."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def weights_digest(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].contiguous()
        h.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()



@dataclass(frozen=True)
class LocalRef:
    run: str
    epoch: int
    agent_steps: int
    path: Path
    weights_sha256: str

    def __str__(self) -> str:
        return str(self.path)


@dataclass(frozen=True)
class LocalVariant:
    config: dict
    experiment: str = "local"
    name: str = "v"
    stage: str | None = None
    labels: dict = field(default_factory=dict)
    config_hash: str = "0"
    depends_on: tuple = ()

    @property
    def ref(self) -> str:
        return f"{self.experiment}/{self.name}"


class LocalWriter:
    """Writes checkpoints, trainer state and metrics into a directory."""

    def __init__(self, path: str | Path, *, mode: str = "new", launch: int = 0,
                 stop_after_epochs: int | None = None, time_limit_s: float | None = None,
                 echo: bool = False) -> None:
        self.path = Path(path)
        (self.path / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.launch = launch
        self.stop_reason = None
        self.metrics: list[dict] = []
        self.checkpoints: list[LocalRef] = []
        self.aliases: dict[str, Any] = {}
        self.status: str | None = None
        self.stop_after_epochs = stop_after_epochs
        self.deadline = None if time_limit_s is None else time.time() + time_limit_s
        self.echo = echo

    def save_checkpoint(self, model, *, epoch: int, agent_steps: int) -> LocalRef:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        path = self.path / "checkpoints" / f"e{epoch:06d}.pt"
        torch.save({"model": getattr(model, "spec", None), "state_dict": state, "epoch": epoch,
                    "agent_steps": agent_steps}, path)
        ref = LocalRef("local", epoch, agent_steps, path, weights_digest(state))
        self.checkpoints.append(ref)
        return ref

    def save_trainer_state(self, state: dict) -> None:
        torch.save(state, self.path / "trainer_state.pt")

    def load_trainer_state(self) -> dict | None:
        path = self.path / "trainer_state.pt"
        return torch.load(path, weights_only=False) if path.exists() else None

    def log(self, logs: dict, step: int) -> None:
        row = {"step": step, **{k: v for k, v in logs.items() if isinstance(v, (int, float, str))}}
        self.metrics.append(row)
        with open(self.path / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        if self.echo:
            keys = [k for k in row if k.endswith("learner_win") or k.startswith("critic/value_sum_2p")
                    or k in ("epoch", "losses/approx_kl", "learning_rate", "throughput/agent_steps_per_s")]
            print({k: (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in keys}, flush=True)

    def should_stop(self) -> bool:
        if self.stop_after_epochs is not None and len(self.metrics) >= self.stop_after_epochs:
            self.stop_reason = "interrupted"
            return True
        if self.deadline is not None and time.time() > self.deadline:
            self.stop_reason = "time_limit"
            return True
        return False

    def set_alias(self, name, ref, *, record=None, note=""):
        self.aliases[name] = (ref, record)

    def pending_selects(self):
        return []

    def finish(self, status: str) -> None:
        self.status = status
