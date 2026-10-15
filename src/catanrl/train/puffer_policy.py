"""PufferLib's policy interface over a :class:`~catanrl.models.CatanModel`.

Rows are ``env * max_players + seat`` and every PufferLib segment is one row's
trajectory (the trainer keeps ``batch_size = rows * bptt_horizon``, so
segment ``s`` holds row ``s``). In each env only the seat to move acts:

* **Rollout** (``forward_eval``): only acting rows run the network. Their
  value head predicts every seat's win probability (slot ``k`` = the seat
  ``k`` places after the actor), which becomes the value of seat
  ``(actor + k) mod n``'s row. Waiting rows get the no-op only. So GAE along
  each seat's row is exact at one network evaluation per game step.
* **Training** (``forward``, ``value_loss`` and ``acting_mask``, PufferLib's patched hooks):
  only acting rows run the network. Slot ``k`` of an acting row at step ``t``
  is trained (binary cross-entropy) on the return of seat ``(actor + k) mod
  n``'s row at ``t``; the predictions are written back into those rows'
  stored values, as PufferLib does for the minibatch's own rows. Waiting rows
  have one legal action (log-prob 0, no gradient) and add nothing to the
  value loss.

The adapter also tracks the critic's calibration by player count: the sum
over seats of the predicted win probabilities, against the sum of the
targets it is trained on and, given the critic's discount, against the
outcomes of finished games (see :meth:`PufferPolicy.pop_calibration`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from catanrl.models import CatanModel
from catanrl.train.critic_metrics import OutcomeCalibration


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

    def __init__(self, model: CatanModel, *, num_rows: int, compile: bool = False,
                 critic_gamma: float | None = None) -> None:
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
        # By player count: [count, predicted sum] over rollouts, [count, predicted, target] over training.
        self._rollout_sums: torch.Tensor | None = None
        self._trained_sums: torch.Tensor | None = None
        # By player count: [count, rollout network's sum, trained network's sum] over training.
        self._gap_sums: torch.Tensor | None = None
        self._rollout_values: torch.Tensor | None = None
        # Predictions against finished games' outcomes, at the critic's discount.
        self._outcomes = None if critic_gamma is None else OutcomeCalibration(
            num_envs=self.num_rows // self.max_players, max_players=self.max_players, gamma=critic_gamma)

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
        probs = present = None
        if acting.numel():
            act_logits, value_logits, present = self._heads(observations.index_select(0, acting))
            logits.index_copy_(0, acting, act_logits.float())
            probs = torch.sigmoid(value_logits.float()) * present
            target, valid = seat_rows(acting, present, self.max_players)
            values[target[valid]] = probs[valid]
            self._track(probs, present)
        if self._outcomes is not None and state is not None and "done" in state:
            self._outcomes.observe(state["reward"], state["done"], acting, probs, present)
        return logits, values

    def _track(self, probs: torch.Tensor, present: torch.Tensor) -> None:
        probs, present = probs.detach(), present.detach()
        stats = torch.stack((torch.ones_like(probs[:, 0]), probs.sum(1)), dim=1)
        self._rollout_sums = self._add(self._rollout_sums, present.sum(1), stats)

    def _add(self, total: torch.Tensor | None, players: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """Sum ``stats`` rows by player count (a one-hot matmul: deterministic on CUDA)."""
        update = F.one_hot(players.round().long(), self.max_players + 1).float().T @ stats.float()
        return update if total is None else total + update

    def pop_calibration(self) -> dict[str, float]:
        """The critic's calibration since the last pop, by player count n.

        ``value_sum_<n>p``: mean over rollout decisions of the per-seat win
        probabilities' sum. ``target_sum_<n>p``: mean over trained decisions of
        the sum of their slots' targets (the seats' lambda-returns, which pay
        nobody for truncations and discount by gamma, so a calibrated critic's
        sum sits below 1 there). ``sum_gap_<n>p``: the mean predicted minus
        target sum on the same trained decisions; 0 when the network fits its
        targets (which bootstrap from it, so not a test of calibration).
        ``rollout_vs_trained_gap_<n>p`` (after :meth:`snapshot_values`): on the
        same trained decisions, the rollout network's predicted sum minus the
        network's in training; positive when the network that collects (and
        every checkpoint) sits above the epoch's trained mean. Given the
        critic's discount, :class:`~catanrl.train.critic_metrics.OutcomeCalibration`'s
        metrics against finished games: ``truth_sum_<n>p`` (what a calibrated
        ``value_sum_<n>p`` estimates), ``brier_slot<k>_<n>p``, ...
        """
        out: dict[str, float] = {}
        rollout, trained, gap = self._rollout_sums, self._trained_sums, self._gap_sums
        self._rollout_sums = self._trained_sums = self._gap_sums = self._rollout_values = None
        for n in range(2, self.max_players + 1):
            if rollout is not None and float(rollout[n, 0]):
                out[f"value_sum_{n}p"] = float(rollout[n, 1] / rollout[n, 0])
            if trained is not None and float(trained[n, 0]):
                count = float(trained[n, 0])
                out[f"target_sum_{n}p"] = float(trained[n, 2]) / count
                out[f"sum_gap_{n}p"] = float(trained[n, 1] - trained[n, 2]) / count
            if gap is not None and float(gap[n, 0]):
                out[f"rollout_vs_trained_gap_{n}p"] = float(gap[n, 1] - gap[n, 2]) / float(gap[n, 0])
        if self._outcomes is not None:
            out.update(self._outcomes.pop())
        return out

    def snapshot_values(self, values: torch.Tensor) -> None:
        """Keep the rollout network's stored values for ``rollout_vs_trained_gap_<n>p``:
        call it before each PuffeRL train, which overwrites them (the next pop drops it)."""
        self._rollout_values = values.detach().clone()

    # ------------------------------------------------------------ training

    def acting_mask(self, observations: torch.Tensor) -> torch.Tensor:
        """PuffeRL's hook: bool [segments, horizon], the stored steps whose row acted.

        Only those weigh in the advantage normalization, the prioritized
        sampling and the means of the policy loss, entropy, KL and clip
        fraction (waiting, bot and padded rows have one legal no-op).
        """
        return observations[..., self.model.mask_offset + self.model.noop] == 0

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
            trained = t < horizon - 1
            if trained.any():
                mask = valid[trained].float()
                stats = torch.stack((torch.ones_like(mask[:, 0]), (probs[trained] * mask).sum(1),
                                     (targets[trained] * mask).sum(1)), dim=1)
                self._trained_sums = self._add(self._trained_sums, mask.sum(1), stats)
                if self._rollout_values is not None:
                    rollout = self._rollout_values[target_rows, steps][trained]
                    stats = torch.stack((torch.ones_like(mask[:, 0]), (rollout * mask).sum(1),
                                         (probs[trained] * mask).sum(1)), dim=1)
                    self._gap_sums = self._add(self._gap_sums, mask.sum(1), stats)
        return newvalue, loss
