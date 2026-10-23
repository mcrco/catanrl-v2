"""The critic against the outcomes its targets estimate (critic/* metrics).

:class:`OutcomeCalibration` follows every env's game through the rollout. At
each step the acting row's value head predicts every seat's win probability
(slot ``k`` = the seat ``k`` places after the actor). When the game ends (all
its rows terminal; 1 to the winner's row, nothing for a truncation), slot
``k``'s realized outcome at decision ``t`` of a ``T``-decision game is
``gamma^(T-1-t)`` if that seat won and 0 otherwise: what the critic's
lambda-returns estimate (every seat's row discounts at every game step, and
the terminal reward arrives with the step after the last decision). So a
calibrated critic's predicted seat sum is the mean of ``gamma^(decisions
remaining)`` over the decisions of won games, not 1.

Per env, running sums discounted once per step hold each game's sums at its
end (no per-decision storage):

* ``A[s, k]``: sum over decisions where slot ``k`` was seat ``s`` of the
  prediction times ``gamma^(steps since)``;
* ``B[s, k]`` and ``C[s, k]``: the same of ``gamma^(2 steps since)`` and
  ``gamma^(steps since)``;

so with winner ``w``, slot ``k``'s sums over the game are ``prediction x
outcome = A[w, k]``, ``outcome^2 = B[w, k]`` and ``outcome = C[w, k]``.

The log loss (the value loss's binary cross-entropy, against the realized
outcome ``y`` rather than the bootstrapped target) is ``-log(1 - p) - y x
logit(p)`` per decision, so it needs ``L[s, k]``, the sum of ``logit(p) x
gamma^(steps since)`` as ``A``, and the plain per-slot sum of ``log(1 - p)``;
``p`` is clamped to ``[EPS, 1 - EPS]`` (a float32 sigmoid reaches 1).

The predictions are the rollout network's, made before it trains on the
step, and an outcome is first seen in training only after the game ends: so
against the outcome every one of these metrics is out of sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: The probability clamp of the log loss.
EPS = 1e-7


class OutcomeCalibration:
    """Per-slot predictions against realized, discounted outcomes of finished games."""

    def __init__(self, *, num_envs: int, max_players: int, gamma: float) -> None:
        self.num_envs = int(num_envs)
        self.max_players = int(max_players)
        self.gamma = float(gamma)
        self._state: dict[str, torch.Tensor] | None = None

    def _init(self, device: torch.device) -> dict[str, torch.Tensor]:
        e, p = self.num_envs, self.max_players
        f64 = dict(device=device, dtype=torch.float64)
        return {
            "decisions": torch.zeros(e, **f64),
            "players": torch.zeros(e, device=device, dtype=torch.long),
            "pred_disc": torch.zeros(e, p, p, **f64),     # A[env, seat, slot]
            "disc2": torch.zeros(e, p, p, **f64),         # B
            "disc": torch.zeros(e, p, p, **f64),          # C
            "pred_sq": torch.zeros(e, p, **f64),          # sum of predictions^2 per slot
            "pred": torch.zeros(e, p, **f64),             # sum of predictions per slot
            "logit_disc": torch.zeros(e, p, p, **f64),    # L[env, seat, slot]
            "log1m": torch.zeros(e, p, **f64),            # sum of log(1 - prediction) per slot
            # by player count: games, decisions, then per slot sum of predictions,
            # of outcomes, of squared differences and of log losses
            "totals": torch.zeros(p + 1, 2 + 4 * p, **f64),
        }

    def observe(self, reward: torch.Tensor, done: torch.Tensor, rows: torch.Tensor,
                probs: torch.Tensor | None, present: torch.Tensor | None) -> None:
        """One rollout step: ``reward`` and ``done`` [rows] arrive with this step's
        observations (a done env's observation is already the next game's first);
        ``rows`` [M] the acting rows and ``probs``/``present`` [M, P] their slots'
        predicted win probabilities and seats present."""
        e, p = self.num_envs, self.max_players
        if self._state is None:
            self._state = self._init(reward.device)
        s = self._state
        if reward.numel() != e * p:
            raise ValueError(f"expected {e * p} rows, got {reward.numel()}")

        # Games that ended before this step.
        ended = done.reshape(e, p).bool().any(dim=1)
        ended_f = ended.to(torch.float64)
        winner = (reward.reshape(e, p) > 0.5).to(torch.float64) * ended_f[:, None]
        outcome = torch.einsum("es,esk->ek", winner, s["disc"])
        cross = torch.einsum("es,esk->ek", winner, s["pred_disc"])
        outcome_sq = torch.einsum("es,esk->ek", winner, s["disc2"])
        cross_logit = torch.einsum("es,esk->ek", winner, s["logit_disc"])
        slots = torch.arange(p, device=reward.device)
        weight = (slots[None, :] < s["players"][:, None]).to(torch.float64) * ended_f[:, None]
        stats = torch.cat((ended_f[:, None], (s["decisions"] * ended_f)[:, None], s["pred"] * weight,
                           outcome * weight, (s["pred_sq"] - 2 * cross + outcome_sq) * weight,
                           -(s["log1m"] + cross_logit) * weight), dim=1)
        s["totals"] += F.one_hot(s["players"], p + 1).to(torch.float64).T @ stats
        keep = 1.0 - ended_f
        for key in ("pred_disc", "disc2", "disc", "logit_disc"):
            s[key] *= keep[:, None, None]
        for key in ("pred_sq", "pred", "log1m"):
            s[key] *= keep[:, None]
        s["decisions"] *= keep

        # Every row steps, so every running sum discounts once.
        s["pred_disc"] *= self.gamma
        s["disc"] *= self.gamma
        s["logit_disc"] *= self.gamma
        s["disc2"] *= self.gamma ** 2
        if rows.numel() == 0:
            return
        env = torch.div(rows, p, rounding_mode="floor")
        seat = rows - env * p
        n = present.sum(dim=1).round().long()
        valid = slots[None, :] < n[:, None]                               # [M, slot]
        target = torch.remainder(seat[:, None] + slots[None, :], n.clamp_min(1)[:, None])
        onehot = (F.one_hot(target, p).to(torch.float64) * valid[..., None]).transpose(1, 2)  # [M, seat, slot]
        pred = probs.detach().to(torch.float64) * valid
        clamped = pred.clamp(EPS, 1.0 - EPS)
        log1m = torch.log1p(-clamped)
        logit = (torch.log(clamped) - log1m) * valid
        s["pred_disc"].index_add_(0, env, onehot * pred[:, None, :])
        s["logit_disc"].index_add_(0, env, onehot * logit[:, None, :])
        s["log1m"].index_add_(0, env, log1m * valid)
        s["disc2"].index_add_(0, env, onehot)
        s["disc"].index_add_(0, env, onehot)
        s["pred_sq"].index_add_(0, env, pred * pred)
        s["pred"].index_add_(0, env, pred)
        s["decisions"].index_add_(0, env, torch.ones_like(env, dtype=torch.float64))
        s["players"].index_copy_(0, env, n)

    def pop(self) -> dict[str, float]:
        """By player count ``n``, over the decisions of games finished since the last pop:

        * ``truth_sum_<n>p``: mean over decisions of the realized outcomes summed over
          seats (``gamma^(decisions remaining)`` if somebody won, 0 for a truncation);
        * ``finished_value_sum_<n>p``: the mean predicted seat sum on the same decisions;
        * ``pred_slot<k>_<n>p`` and ``real_slot<k>_<n>p``: slot ``k``'s mean prediction
          and realized outcome; ``brier_slot<k>_<n>p``: their mean squared difference;
          ``bce_slot<k>_<n>p``: the mean log loss of the prediction against the outcome
          (nats; the value loss's form, out of sample);
        * ``finished_games_<n>p``: how many games.
        """
        if self._state is None:
            return {}
        totals = self._state["totals"].clone().cpu()  # on CPU, .cpu() alone aliases
        self._state["totals"].zero_()
        p = self.max_players
        out: dict[str, float] = {}
        for n in range(2, p + 1):
            games, decisions = float(totals[n, 0]), float(totals[n, 1])
            if not decisions:
                continue
            pred, real, sq, bce = (totals[n, 2 + i * p: 2 + (i + 1) * p] / decisions for i in range(4))
            out[f"truth_sum_{n}p"] = float(real[:n].sum())
            out[f"finished_value_sum_{n}p"] = float(pred[:n].sum())
            out[f"finished_games_{n}p"] = games
            for k in range(n):
                out[f"pred_slot{k}_{n}p"] = float(pred[k])
                out[f"real_slot{k}_{n}p"] = float(real[k])
                out[f"brier_slot{k}_{n}p"] = float(sq[k])
                out[f"bce_slot{k}_{n}p"] = float(bce[k])
        return out
