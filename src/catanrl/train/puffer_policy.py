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

* **Recurrent models** (``model.recurrent``, :mod:`catanrl.models.lstm`):
  a seat's memory is the sequence of its own decisions. The rollout keeps
  one state per row here, not in PufferLib's ``lstm_h`` dict (which it
  zeroes at every ``evaluate``): the cell steps for acting rows only, the
  state is zeroed where ``done`` marks a new game, and it carries across
  epochs. The training forward (``use_rnn``: observations are segments x
  horizon) replays the same recursion from each segment's stored initial
  state (the fork passes the minibatch's segment indices and terminals in
  ``state``), with gradients through the horizon, as CleanRL's PPO-LSTM.
  The state is rollout-only: it is not checkpointed, and a resume starts
  from zero like the restarted games.

* **The aux head** (``aux_coef`` > 0, a graph network with ``args.aux``): the
  training forward also returns the head's predictions for the acting rows
  with their targets, the hidden features of the other seats read from the
  row's critic block (:meth:`CatanModel.aux_targets`), and ``aux_loss``
  (the fork's third hook) adds ``aux_coef`` times their masked mean squared
  error, over present other seats and hidden features, to PPO's loss. The
  rollout never runs the head.

The adapter also tracks the critic's calibration by player count: the sum
over seats of the predicted win probabilities, against the sum of the
targets it is trained on and, given the critic's discount, against the
outcomes of finished games (see :meth:`PufferPolicy.pop_calibration`), and
the aux head's error per hidden feature (:meth:`PufferPolicy.pop_aux`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from catanrl.envs.encoding import HIDDEN_FEATURES
from catanrl.models import CatanModel
from catanrl.train.critic_metrics import OutcomeCalibration


@dataclass
class ValueBatch:
    """The training forward's value outputs (consumed by ``value_loss``), and the aux
    head's when it trains (consumed by ``aux_loss``)."""

    index: torch.Tensor    # [M] flat minibatch rows that act
    logits: torch.Tensor   # [M, P] per-seat win logits
    present: torch.Tensor  # [M, P] seats present
    rows: int              # minibatch rows (segments x horizon)
    #: (predictions [M, P-1, F], targets [M, P-1, F], present other seats [M, P-1]) or None.
    aux: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


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
                 critic_gamma: float | None = None, aux_coef: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.recurrent = bool(getattr(model, "recurrent", False))
        self.aux_coef = float(aux_coef)
        if self.aux_coef < 0.0:
            raise ValueError("aux_coef must be >= 0")
        if self.aux_coef:
            if not getattr(model, "aux", False):
                raise ValueError("train.aux_coef > 0 needs a network with an aux head (model.args.aux on a shared "
                                 "graph trunk)")
            if not model.has_critic_block:
                raise ValueError("the aux head's targets are the critic block: set obs.critic_level above obs.level")
            if self.recurrent:
                raise ValueError("the aux head trains a non-recurrent network")
            self._heads_with_aux = (torch.compile(model.heads_with_aux, dynamic=True) if compile
                                    else model.heads_with_aux)
        # Per hidden feature over the epoch's trained (row, other seat) pairs:
        # [count, sum of squared errors, sum of targets, sum of squared targets].
        self._aux_sums: torch.Tensor | None = None
        # torch.compile fuses the many small elementwise ops (about 1.45x on graph-s); dynamic
        # shapes because the number of acting rows varies. A recurrent model's cell stays eager:
        # the scan over the horizon is a loop of tiny kernels.
        if self.recurrent:
            self._embed = torch.compile(model.embed, dynamic=True) if compile else model.embed
            self._decode = torch.compile(model.decode, dynamic=True) if compile else model.decode
        else:
            self._heads = torch.compile(model.heads, dynamic=True) if compile else model.heads
        self.max_players = model.num_players
        self.num_rows = int(num_rows)
        if self.num_rows % self.max_players:
            raise ValueError("rows must be whole envs of max_players rows")
        #: PuffeRL reads it when use_rnn (the size of its own, unused, state dict).
        self.hidden_size = int(getattr(model, "hidden_size", 0)) if self.recurrent else 0
        # The rollout memory (h, c) of every row, and its snapshot at the epoch's first step.
        self._memory: tuple[torch.Tensor, torch.Tensor] | None = None
        self._memory_start: tuple[torch.Tensor, torch.Tensor] | None = None
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
        if self.recurrent:
            done = state.get("done") if state is not None else None
            self._reset_memory(rows, observations.device, done)
        if acting.numel():
            act_logits, value_logits, present = self._rollout_heads(observations.index_select(0, acting), acting)
            logits.index_copy_(0, acting, act_logits.float())
            probs = torch.sigmoid(value_logits.float()) * present
            target, valid = seat_rows(acting, present, self.max_players)
            values[target[valid]] = probs[valid]
            self._track(probs, present)
        if self._outcomes is not None and state is not None and "done" in state:
            self._outcomes.observe(state["reward"], state["done"], acting, probs, present)
        return logits, values

    def _rollout_heads(self, obs: torch.Tensor, acting: torch.Tensor):
        """The heads of the acting rows; a recurrent model also steps their memory."""
        if not self.recurrent:
            return self._heads(obs)
        h, c = self._memory
        emb, rows = self._embed(obs)
        h_new, c_new = self.model.remember(emb.pooled, h.index_select(0, acting), c.index_select(0, acting))
        # The memory stays float32: under PuffeRL's amp autocast the cell returns bf16 on CUDA
        # (lstm_cell is in autocast's lower-precision list), and index_copy_ refuses a dtype mix.
        h.index_copy_(0, acting, h_new.to(h.dtype))
        c.index_copy_(0, acting, c_new.to(c.dtype))
        return self._decode(emb, rows, h_new)

    def _reset_memory(self, rows: int, device, done: torch.Tensor | None) -> None:
        """Allocate the rollout memory on first use; zero the rows whose observation
        starts a new game (``done`` arrives with the next game's first observation)."""
        if self._memory is None:
            self._memory = self.model.initial_state(rows, device)
        if done is not None:
            keep = (~done.reshape(rows).bool()).to(self._memory[0].dtype).unsqueeze(1)
            self._memory[0].mul_(keep)
            self._memory[1].mul_(keep)

    def begin_rollout(self) -> None:
        """Snapshot every row's memory: the initial state of this epoch's segments.
        Call it before each PuffeRL evaluate (the trainer does)."""
        if self.recurrent and self._memory is not None:
            self._memory_start = (self._memory[0].clone(), self._memory[1].clone())

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
        if self.recurrent:
            return self._forward_recurrent(observations, state)
        rows = observations.shape[0]
        acting = torch.nonzero(self.model.acting(observations)).squeeze(1)
        logits = self.model.waiting_logits(rows, observations.device)
        acting_obs = observations.index_select(0, acting)
        aux = None
        if self.aux_coef:
            act_logits, value_logits, present, predictions = self._heads_with_aux(acting_obs)
            targets, others = self.model.aux_targets(acting_obs)
            aux = (predictions, targets, others)
        else:
            act_logits, value_logits, present = self._heads(acting_obs)
        logits = logits.index_copy(0, acting, act_logits.float())
        return logits, ValueBatch(acting, value_logits.float(), present, rows, aux)

    def _forward_recurrent(self, observations: torch.Tensor, state):
        """The training forward over segments x horizon: the rollout's recursion,
        from each segment's stored initial state, with gradients through the horizon.

        Every step runs the cell on all segments (zero input where the row
        waits) and keeps the new state only where the row acts; a step whose
        observation starts a new game zeroes the state first. The trunk and
        the heads run on acting rows only, as in the rollout.
        """
        if observations.dim() != 3:
            raise ValueError("a recurrent policy trains on segments x horizon (PuffeRL's use_rnn)")
        if state is None or "idx" not in state or "terminals" not in state:
            raise ValueError("the training state lacks idx and terminals: this needs the catanrl-v2 PufferLib fork")
        segments, horizon, _ = observations.shape
        rows = segments * horizon
        flat = observations.reshape(rows, -1)
        acting = torch.nonzero(self.model.acting(flat)).squeeze(1)
        logits = self.model.waiting_logits(rows, flat.device)
        emb, obs_rows = self._embed(flat.index_select(0, acting))
        inputs = emb.pooled.new_zeros(rows, emb.pooled.shape[1]).index_copy(0, acting, emb.pooled)
        inputs = inputs.reshape(segments, horizon, -1)
        act = self.model.acting(flat).reshape(segments, horizon, 1)
        keep = (1.0 - state["terminals"].reshape(segments, horizon).float()).unsqueeze(2)
        idx = state["idx"].long()
        if self._memory_start is None:  # the first epoch: every row started from zero
            h, c = self.model.initial_state(segments, flat.device)
        else:
            h, c = (m.index_select(0, idx) for m in self._memory_start)
        memories = []
        for t in range(horizon):
            h, c = h * keep[:, t], c * keep[:, t]
            h_new, c_new = self.model.remember(inputs[:, t], h, c)
            # The state stays float32 along the horizon, as the rollout's memory does (under
            # autocast the cell returns bf16 on CUDA; torch.where and the stack need one dtype).
            h = torch.where(act[:, t], h_new.to(h.dtype), h)
            c = torch.where(act[:, t], c_new.to(c.dtype), c)
            memories.append(h)
        memory = torch.stack(memories, dim=1).reshape(rows, -1).index_select(0, acting)
        act_logits, value_logits, present = self._decode(emb, obs_rows, memory)
        logits = logits.index_copy(0, acting, act_logits.float())
        return logits, ValueBatch(acting, value_logits.float(), present, rows)

    def aux_loss(self, outputs: ValueBatch):
        """PuffeRL's hook: ``(aux_coef x the aux head's loss, {"aux": the loss})`` on the
        minibatch's acting rows, or None when the head does not train (the loss is then
        untouched). The loss is the mean squared error in the encoder's feature scale over
        the present other seats and the hidden features; its per-feature terms are summed
        for :meth:`pop_aux`."""
        if outputs.aux is None:
            return None
        predictions, targets, others = outputs.aux
        weight = others.float().unsqueeze(-1)                                     # [M, P-1, 1]
        errors = ((predictions.float() - targets.float()) ** 2 * weight).sum(dim=(0, 1))  # [F]
        pairs = weight.sum()
        loss = errors.sum() / (pairs * errors.shape[0]).clamp_min(1.0)
        with torch.no_grad():
            stats = torch.stack((pairs.expand(errors.shape[0]), errors, (targets.float() * weight).sum(dim=(0, 1)),
                                 (targets.float() ** 2 * weight).sum(dim=(0, 1))), dim=1)
            self._aux_sums = stats if self._aux_sums is None else self._aux_sums + stats
        return self.aux_coef * loss, {"aux": loss.detach()}

    def pop_aux(self) -> dict[str, float]:
        """The aux head's error per hidden feature since the last pop, over the trained
        (acting row, present other seat) pairs: ``mse_<feature>``, and ``var_<feature>``,
        the variance of its target over the same pairs (the error of the best constant
        prediction, so ``1 - mse / var`` is the head's R^2 on that feature)."""
        sums, self._aux_sums = self._aux_sums, None
        if sums is None or not float(sums[0, 0]):
            return {}
        out: dict[str, float] = {}
        count = sums[:, 0]
        mean = sums[:, 2] / count
        for i, name in enumerate(HIDDEN_FEATURES):
            out[f"mse_{name}"] = float(sums[i, 1] / count[i])
            out[f"var_{name}"] = max(float(sums[i, 3] / count[i] - mean[i] ** 2), 0.0)
        return out

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
