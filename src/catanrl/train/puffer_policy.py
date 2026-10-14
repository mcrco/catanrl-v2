"""PufferLib's policy interface over a :class:`~catanrl.models.CatanModel`.

Rows are ``env * max_players + seat`` and every PufferLib segment is one row's
trajectory (the trainer keeps ``batch_size = rows * bptt_horizon``, so
segment ``s`` holds row ``s``). In each env only the seat to move acts:

* **Rollout** (``forward_eval``): only acting rows run the network. Their
  value head predicts every seat's win probability (slot ``k`` = the seat
  ``k`` places after the actor), which becomes the value of seat
  ``(actor + k) mod n``'s row. Waiting rows get the no-op only. So GAE along
  each seat's row is exact at one network evaluation per game step.
* **Training** (``forward`` and ``value_loss``, PufferLib's patched hook):
  only acting rows run the network. Slot ``k`` of an acting row at step ``t``
  is trained (binary cross-entropy) on the return of seat ``(actor + k) mod
  n``'s row at ``t``; the predictions are written back into those rows'
  stored values, as PufferLib does for the minibatch's own rows. Waiting rows
  have one legal action (log-prob 0, no gradient) and add nothing to the
  value loss.

The adapter also tracks the critic's calibration: the sum over seats of the
acting rows' predicted win probabilities, by player count.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from catanrl.models import CatanModel


@dataclass
class ValueBatch:
    """The training forward's value outputs (consumed by ``value_loss``)."""

    index: torch.Tensor    # [M] flat minibatch rows that act
    logits: torch.Tensor   # [M, P] per-seat win logits
    present: torch.Tensor  # [M, P] seats present
    rows: int              # minibatch rows (segments x horizon)


def seat_rows(rows: torch.Tensor, present: torch.Tensor, max_players: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For acting rows ``env * P + seat``: the row of each value slot, and its validity.

    Slot ``k`` belongs to seat ``(seat + k) mod n`` of the same env.
    """
    n = present.sum(dim=1).round().long().clamp_min(1)
    env = torch.div(rows, max_players, rounding_mode="floor")
    seat = rows - env * max_players
    k = torch.arange(max_players, device=rows.device)
    valid = k.unsqueeze(0) < n.unsqueeze(1)
    target = env.unsqueeze(1) * max_players + torch.remainder(seat.unsqueeze(1) + k.unsqueeze(0), n.unsqueeze(1))
    return target, valid


class PufferPolicy(nn.Module):
    """The model behind PufferLib's ``forward_eval`` / ``forward`` / ``value_loss``."""

    def __init__(self, model: CatanModel, *, num_rows: int, compile: bool = False) -> None:
        super().__init__()
        self.model = model
        # torch.compile fuses the many small elementwise ops (about 1.45x on graph-s); dynamic
        # shapes because the number of acting rows varies.
        self._heads = torch.compile(model.heads, dynamic=True) if compile else model.heads
        self.max_players = model.num_players
        self.num_rows = int(num_rows)
        if self.num_rows % self.max_players:
            raise ValueError("rows must be whole envs of max_players rows")
        self.hidden_size = 0
        self._calibration: torch.Tensor | None = None  # [P + 1, 3]: count, sum, |sum - 1|

    # ------------------------------------------------------------ rollout

    def forward_eval(self, observations: torch.Tensor, state=None):
        rows = observations.shape[0]
        if rows != self.num_rows:
            raise ValueError(f"expected all {self.num_rows} rows in one batch, got {rows}")
        if state is not None and "env_id" in state:
            env_id = state["env_id"]
            if isinstance(env_id, slice) and (env_id.start, env_id.stop) != (0, rows):
                raise ValueError("the Serial batch must be every row in order")
        acting = torch.nonzero(self.model.acting(observations)).squeeze(1)
        logits = self.model.waiting_logits(rows, observations.device)
        values = torch.zeros(rows, device=observations.device, dtype=torch.float32)
        if acting.numel():
            act_logits, value_logits, present = self._heads(observations.index_select(0, acting))
            logits.index_copy_(0, acting, act_logits.float())
            probs = torch.sigmoid(value_logits.float()) * present
            target, valid = seat_rows(acting, present, self.max_players)
            values[target[valid]] = probs[valid]
            self._track(probs, present)
        return logits, values

    def _track(self, probs: torch.Tensor, present: torch.Tensor) -> None:
        probs, present = probs.detach(), present.detach()
        n = present.sum(1).round().long()
        total = probs.sum(1)
        one_hot = F.one_hot(n, self.max_players + 1).float()  # [M, P+1]
        stats = torch.stack((torch.ones_like(total), total, (total - 1.0).abs()), dim=1)
        update = one_hot.T @ stats
        self._calibration = update if self._calibration is None else self._calibration + update

    def pop_calibration(self) -> dict[str, float]:
        """Mean over acting rows of the per-seat win probabilities' sum, by player count."""
        if self._calibration is None:
            return {}
        stats = self._calibration.cpu()
        self._calibration = None
        out: dict[str, float] = {}
        for n in range(2, self.max_players + 1):
            count = float(stats[n, 0])
            if count:
                out[f"value_sum_{n}p"] = float(stats[n, 1]) / count
                out[f"value_sum_abs_err_{n}p"] = float(stats[n, 2]) / count
        return out

    # ------------------------------------------------------------ training

    def forward(self, observations: torch.Tensor, state=None):
        rows = observations.shape[0]
        acting = torch.nonzero(self.model.acting(observations)).squeeze(1)
        logits = self.model.waiting_logits(rows, observations.device)
        act_logits, value_logits, present = self._heads(observations.index_select(0, acting))
        logits = logits.index_copy(0, acting, act_logits.float())
        return logits, ValueBatch(acting, value_logits.float(), present, rows)

    def value_loss(self, outputs: ValueBatch, returns: torch.Tensor, *, idx: torch.Tensor,
                   values: torch.Tensor, advantages: torch.Tensor):
        """PuffeRL's hook: (each minibatch row's new value, the value loss)."""
        segments, horizon = values.shape
        if segments != self.num_rows:
            raise ValueError("value_loss needs one segment per row (batch_size = rows * bptt_horizon)")
        if outputs.rows != idx.numel() * horizon:
            raise ValueError("the minibatch does not match its segment indices")
        flat = outputs.index
        local = torch.div(flat, horizon, rounding_mode="floor")
        t = flat - local * horizon
        rows = idx.long()[local]
        target_rows, valid = seat_rows(rows, outputs.present, self.max_players)
        steps = t.unsqueeze(1).expand_as(target_rows)
        with torch.no_grad():
            targets = (advantages + values)[target_rows, steps].clamp(0.0, 1.0)
        # The last step of a segment has no advantage (its return is its own value).
        weight = (valid & (steps < horizon - 1)).float()
        loss = F.binary_cross_entropy_with_logits(outputs.logits, targets, reduction="none")
        loss = (loss * weight).sum() / weight.sum().clamp_min(1.0)
        with torch.no_grad():
            probs = torch.sigmoid(outputs.logits)
            values[target_rows[valid], steps[valid]] = probs[valid]
            newvalue = values[idx].reshape(-1)
        return newvalue, loss
