"""Usage checks of the dice input (``obs.dice_odds``), logged every PPO epoch as ``dice/*``.

A network with the dice input reads each seat's next-roll odds in the global block
(:data:`catanrl.envs.encoding.DICE_ODDS`). Warm-started with ``init_from.zero_new_inputs``
its weights on those inputs start at zero, so at e0 it ignores them; two cheap numbers
show whether training puts them to use:

* ``dice/weight_norm``: the Frobenius norm of the dice inputs' weights, every graph
  trunk's ``dice_projection`` (the one layer that reads them); 0 at the start;
* the sensitivity probe, on a fixed batch of positions: the policy and value on the rows
  as they are against the same rows with every present seat's odds replaced by the fair
  2d6 odds (what a game on random dice would show). ``dice/probe_kl`` is the mean
  KL(true || fair) of the policy over the batch, ``dice/probe_kl_max`` its largest,
  ``dice/probe_flips`` the share of positions whose argmax action changes and
  ``dice/probe_value_gap`` the mean absolute change of the row's own win probability;
  ``dice/probe_rows`` the batch size. All 0 at e0.

The batch is the acting rows of epoch :data:`PROBE_EPOCH` (positions of the network's own
play, once every env's games have drifted apart: the first epochs all play the openings,
where the deck is still fresh and the odds fair), every k-th up to :data:`PROBE_ROWS`, taken
once; it rides in the trainer state, so a resumed run probes the same positions. Before
that epoch only the norm is logged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from catanrl.envs.encoding import DICE_PREFIX, FAIR_ODDS, dice_odds_of
from catanrl.models.rows import feature_names

#: Positions in the probe batch.
PROBE_ROWS = 512
#: The epoch whose rollout gives the batch (1,032 envs reset together, so the first epochs'
#: 64 steps are every game's opening; ten epochs in, games are at every stage).
PROBE_EPOCH = 10
#: The trainer-state key of the batch.
STATE_KEY = "dice_probe_rows"


def dice_weight_norm(model: torch.nn.Module) -> float | None:
    """The Frobenius norm of the dice inputs' weights, every graph trunk's ``dice_projection``
    (None without one: a network class that reads the input another way)."""
    from catanrl.models.graph import GraphTrunk

    weights = [m.dice_projection.weight for m in model.modules()
               if isinstance(m, GraphTrunk) and m.dice_projection is not None]
    if not weights:
        return None
    with torch.no_grad():
        return float(sum(w.detach().float().pow(2).sum() for w in weights) ** 0.5)


def dice_slots(model: torch.nn.Module) -> np.ndarray:
    """[P, 11] positions in the model's row of each seat slot's odds (totals 2..12)."""
    names = feature_names(model.dims, model.levels["level"], True)["global"]
    offset = int(model.layout["global"][0])
    players = int(model.dims["max_players"])
    return np.array([[offset + names.index(f"{DICE_PREFIX}{k}_{t}") for t in range(2, 13)] for k in range(players)])


def fair_rows(rows: torch.Tensor, slots: np.ndarray) -> torch.Tensor:
    """``rows`` with every present seat's odds (a slot whose odds sum to about 1; absent
    slots read 0) replaced by the fair 2d6 odds, as float32."""
    out = rows.clone()
    fair = torch.tensor(FAIR_ODDS, dtype=torch.float32, device=rows.device).to(rows.dtype)
    for columns in slots:
        index = torch.as_tensor(columns, device=rows.device)
        block = out.index_select(1, index)
        present = block.sum(dim=1) > 0.5
        block[present] = fair
        out[:, index] = block
    return out


def probe(model: torch.nn.Module, rows: torch.Tensor, slots: np.ndarray) -> dict[str, float]:
    """The sensitivity probe's metrics on ``rows`` (acting rows of the model's layout)."""
    if not len(rows):
        return {}
    with torch.no_grad(), torch.autocast(rows.device.type, enabled=False):
        logits, values = model(rows.float())
        fair_logits, fair_values = model(fair_rows(rows.float(), slots))
        logp = torch.log_softmax(logits.float(), dim=-1)
        fair_logp = torch.log_softmax(fair_logits.float(), dim=-1)
        kl = (logp.exp() * (logp - fair_logp)).sum(dim=-1).clamp_min(0.0)
        flips = (logits.argmax(dim=-1) != fair_logits.argmax(dim=-1)).float()
        gap = (values[:, 0].float() - fair_values[:, 0].float()).abs()
    return {"dice/probe_kl": float(kl.mean()), "dice/probe_kl_max": float(kl.max()),
            "dice/probe_flips": float(flips.mean()), "dice/probe_value_gap": float(gap.mean()),
            "dice/probe_rows": int(len(rows))}


class DiceProbe:
    """The ``dice/*`` metrics of one run (see the module doc)."""

    def __init__(self, model: torch.nn.Module, capture_epoch: int | None = None) -> None:
        if not dice_odds_of(getattr(model, "levels", None)):
            raise ValueError("the dice probe needs a network with the dice input (obs.dice_odds)")
        self.slots = dice_slots(model)
        self.capture_epoch = int(PROBE_EPOCH if capture_epoch is None else capture_epoch)
        self.rows: torch.Tensor | None = None

    def capture(self, model: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """The fixed batch from ``observations`` (the epoch's stored rows): its acting rows,
        every k-th up to :data:`PROBE_ROWS`, on the CPU."""
        rows = observations.reshape(-1, observations.shape[-1])
        acting = torch.nonzero(model.acting(rows)).flatten()
        if len(acting) > PROBE_ROWS:
            acting = acting[torch.arange(PROBE_ROWS, device=acting.device) * (len(acting) // PROBE_ROWS)]
        self.rows = rows[acting].detach().float().cpu().clone()
        return self.rows

    def restore(self, state: dict[str, Any]) -> None:
        if state.get(STATE_KEY) is not None:
            self.rows = state[STATE_KEY].float()

    def metrics(self, model: torch.nn.Module, observations: torch.Tensor | None = None,
                epoch: int | None = None) -> dict[str, float]:
        """This epoch's ``dice/*``; the batch is taken from ``observations`` (this epoch's
        stored rows) at the first epoch from ``capture_epoch`` on."""
        out: dict[str, float] = {}
        norm = dice_weight_norm(model)
        if norm is not None:
            out["dice/weight_norm"] = norm
        due = epoch is None or int(epoch) >= self.capture_epoch
        if self.rows is None and observations is not None and due:
            self.capture(model, observations)
        if self.rows is not None:
            device = next(model.parameters()).device
            out.update(probe(model, self.rows.to(device), self.slots))
        return out
