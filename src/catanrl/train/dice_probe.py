"""Usage checks of the dice input (``obs.dice_state``), logged as ``dice/*`` by the PPO trainer
(every epoch) and the AlphaZero trainer (every iteration).

A network with the dice input reads the balanced dice state in the global block and on each
tile (:data:`catanrl.envs.encoding.DICE_STATE`). Warm-started with ``init_from.zero_new_inputs``
its weights on those inputs start at zero, so at e0 it ignores them; a few cheap numbers show
whether training puts them to use:

* ``dice/weight_norm``: the Frobenius norm of the dice inputs' weights, every graph trunk's
  ``dice_projection`` (the global features) and ``tile_dice_projection`` (each tile's), the
  two layers that read them; ``dice/weight_norm_global`` and ``dice/weight_norm_tiles`` are
  its two parts. 0 at the start;
* the sensitivity probe, on a fixed batch of positions: the policy and value on the rows as
  they are against the same rows with every dice feature replaced by what a fresh balanced
  state shows (:func:`fair_rows`: the fair 2d6 odds for every present seat, a full deck, an
  empty window, no sevens; what a game on random dice shows at every roll).
  ``dice/probe_kl`` is the mean KL(true || fair) of the policy over the batch,
  ``dice/probe_kl_max`` its largest, ``dice/probe_flips`` the share of positions whose argmax
  action changes and ``dice/probe_value_gap`` the mean absolute change of the row's own win
  probability; ``dice/probe_rows`` the batch size. All 0 at e0.

The batch is taken once and rides in the trainer state, so a resumed run probes the same
positions. PPO takes the acting rows of epoch :data:`PROBE_EPOCH` (positions of the network's
own play, once every env's games have drifted apart: the first epochs all play the openings,
where the deck is still fresh and the odds fair), every k-th up to :data:`PROBE_ROWS`.
AlphaZero takes every k-th replay position, up to :data:`PROBE_ROWS`, the first time its
learner trains (the replay then holds ``train.min_replay`` positions of whole games; they are
the replay's float16 rows). Before that only the norms are logged.
"""

from __future__ import annotations

from typing import Any

import torch

from catanrl.envs.encoding import DICE_ODDS_SCALE, DICE_PREFIX, FAIR_ODDS, dice_state_of, fresh_dice_features
from catanrl.models.rows import feature_names

#: Positions in the probe batch.
PROBE_ROWS = 512
#: The PPO epoch whose rollout gives the batch (1,032 envs reset together, so the first epochs'
#: 64 steps are every game's opening; ten epochs in, games are at every stage).
PROBE_EPOCH = 10
#: The trainer-state key of the batch.
STATE_KEY = "dice_probe_rows"


def dice_weight_norms(model: torch.nn.Module) -> dict[str, float]:
    """The Frobenius norms of the dice inputs' weights over every graph trunk: all of them
    (``dice/weight_norm``), the global features' and the tiles'. Empty without a graph trunk
    that reads the input (a network class that reads it another way)."""
    from catanrl.models.graph import GraphTrunk

    trunks = [m for m in model.modules() if isinstance(m, GraphTrunk) and m.dice_projection is not None]
    if not trunks:
        return {}
    with torch.no_grad():
        glob = float(sum(t.dice_projection.weight.detach().float().pow(2).sum() for t in trunks))
        tiles = float(sum(t.tile_dice_projection.weight.detach().float().pow(2).sum() for t in trunks))
    return {"dice/weight_norm": (glob + tiles) ** 0.5, "dice/weight_norm_global": glob ** 0.5,
            "dice/weight_norm_tiles": tiles ** 0.5}


def dice_weight_norm(model: torch.nn.Module) -> float | None:
    """``dice/weight_norm`` (None without a graph trunk that reads the input)."""
    return dice_weight_norms(model).get("dice/weight_norm")


class FairDice:
    """Where a model's row holds its dice features, and what they read on a fresh balanced
    state: :meth:`rows` rewrites a batch of rows to it."""

    def __init__(self, model: torch.nn.Module) -> None:
        names = feature_names(model.dims, model.levels["level"], True)
        self.players = int(model.dims["max_players"])
        glob, tiles = names["global"], names["tiles"]
        g0 = int(model.layout["global"][0])
        p0, (_, player_width) = model.layout["players"]
        t0, (num_tiles, tile_width) = model.layout["tiles"]
        self.present = [int(p0) + k * int(player_width) + names["players"].index("present")
                        for k in range(self.players)]
        #: [P, 11] columns of each seat slot's odds (totals 2..12).
        self.odds = [[g0 + glob.index(f"{DICE_PREFIX}odds_{k}_{t}") for t in range(2, 13)]
                     for k in range(self.players)]
        fresh = fresh_dice_features(glob, 0)  # the seat-free features: the deck, window, sevens, streak
        self.constant = [(g0 + j, value) for j, (name, value) in enumerate(zip(glob, fresh))
                         if value is not None and not name.startswith(f"{DICE_PREFIX}odds_")]
        column = lambda name: [int(t0) + t * int(tile_width) + tiles.index(name) for t in range(int(num_tiles))]
        self.tile_pips = column("pips")
        self.tile_odds = [column(f"{DICE_PREFIX}odds_{k}") for k in range(self.players)]
        self.tile_deck, self.tile_recent = column(f"{DICE_PREFIX}deck"), column(f"{DICE_PREFIX}recent")
        #: Every dice column of the row.
        self.columns = sorted([c for seat in self.odds for c in seat] + [c for c, _ in self.constant]
                              + [c for seat in self.tile_odds for c in seat] + self.tile_deck + self.tile_recent)

    def rows(self, rows: torch.Tensor) -> torch.Tensor:
        """``rows`` (float) with every dice feature as a fresh balanced state shows it: each
        present seat's odds the fair 2d6 odds (absent slots stay 0), a full deck, an empty
        window, no sevens; a numbered tile's odds its ``pips`` feature for the present seats
        (what the encoder writes under fair dice), its deck 1, its window count 0."""
        out = rows.clone()
        index = lambda columns: torch.as_tensor(columns, device=rows.device)
        present = rows.index_select(1, index(self.present))  # [B, P], 1 for a seated slot
        fair = torch.tensor([p * DICE_ODDS_SCALE for p in FAIR_ODDS], dtype=rows.dtype, device=rows.device)
        pips = rows.index_select(1, index(self.tile_pips))   # [B, T], 0 on the desert and padded slots
        for k in range(self.players):
            out[:, index(self.odds[k])] = present[:, k: k + 1] * fair
            out[:, index(self.tile_odds[k])] = present[:, k: k + 1] * pips
        for column, value in self.constant:
            out[:, column] = value
        out[:, index(self.tile_deck)] = (pips > 0).to(rows.dtype)
        out[:, index(self.tile_recent)] = 0.0
        return out


def fair_rows(rows: torch.Tensor, fair: FairDice) -> torch.Tensor:
    """``rows`` with the dice features of a fresh balanced state (:meth:`FairDice.rows`)."""
    return fair.rows(rows)


def probe(model: torch.nn.Module, rows: torch.Tensor, fair: FairDice) -> dict[str, float]:
    """The sensitivity probe's metrics on ``rows`` (acting rows of the model's layout)."""
    if not len(rows):
        return {}
    with torch.no_grad(), torch.autocast(rows.device.type, enabled=False):
        logits, values = model(rows.float())
        fair_logits, fair_values = model(fair.rows(rows.float()))
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
        if not dice_state_of(getattr(model, "levels", None)):
            raise ValueError("the dice probe needs a network with the dice input (obs.dice_state)")
        self.fair = FairDice(model)
        self.capture_epoch = int(PROBE_EPOCH if capture_epoch is None else capture_epoch)
        self.rows: torch.Tensor | None = None

    def capture(self, model: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """The fixed batch from ``observations`` (stored rows of the model's layout): its acting
        rows, every k-th up to :data:`PROBE_ROWS`, on the CPU."""
        rows = observations.reshape(-1, observations.shape[-1])
        acting = torch.nonzero(model.acting(rows)).flatten()
        if len(acting) > PROBE_ROWS:
            acting = acting[torch.arange(PROBE_ROWS, device=acting.device) * (len(acting) // PROBE_ROWS)]
        self.rows = rows[acting].detach().float().cpu().clone()
        return self.rows

    def take(self, rows: Any) -> torch.Tensor:
        """The fixed batch given as it is: ``rows`` [n, obs] of positions to act in (an array or
        a tensor, float16 or float32), kept as float32 on the CPU."""
        self.rows = torch.as_tensor(rows).detach().float().cpu().clone()
        return self.rows

    def restore(self, state: dict[str, Any]) -> None:
        if state.get(STATE_KEY) is not None:
            self.rows = state[STATE_KEY].float()

    def state(self) -> dict[str, Any]:
        """The trainer-state entry of the batch (empty before it is taken)."""
        return {STATE_KEY: self.rows} if self.rows is not None else {}

    def metrics(self, model: torch.nn.Module, observations: torch.Tensor | None = None,
                epoch: int | None = None) -> dict[str, float]:
        """This epoch's ``dice/*``; the batch is taken from ``observations`` (this epoch's
        stored rows) at the first epoch from ``capture_epoch`` on."""
        out: dict[str, float] = dict(dice_weight_norms(model))
        due = epoch is None or int(epoch) >= self.capture_epoch
        if self.rows is None and observations is not None and due:
            self.capture(model, observations)
        if self.rows is not None:
            device = next(model.parameters()).device
            out.update(probe(model, self.rows.to(device), self.fair))
        return out
