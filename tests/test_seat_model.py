"""The seat-set graph network (model class ``seat``, :mod:`catanrl.models.seat`; presets
``graph-seat-s`` and ``graph-seat-tiny``), on the CPU:

* the presets resolve, ``graph-seat-s`` is ``graph-s``'s size, and rows of every table size
  (two to six seats, padded or not) run forward and backward: masked logits, a value per
  present seat, a gradient on every weight;
* the seats are a set: whatever an absent slot holds changes no output; with the turn order
  taken away, exchanging two other seats (features, ownership channels, dice features and the
  robber's victims) exchanges their outputs and leaves the rest; with it, the outputs differ;
* the graph network's options run on it: every information level, a shared trunk, a critic at
  another level, the aux and outcome heads, the dice input (zero at the start, warm-started
  with ``init_from.zero_new_inputs``), float16, checkpoints and ``init_from``, the search
  agents on ``kernels=torch`` (the fused kernels: ``tests/test_seat_kernels.py``);
* tiny trainings: PPO on a mixture of 2, 3 and 4 players, AlphaZero with the dice input and
  the outcome head.
"""

from __future__ import annotations

import re

import cppanatron as cpp
import numpy as np
import pytest
import torch
from test_dice_input import DIMS as FR_DIMS
from test_dice_input import GAME as FR_GAME
from test_dice_input import balanced_games, decide, native, saved
from test_models import BASE_2P, BASE_4P, WIDE_4P, _names, model_for, played_games

from catanrl import checkpoint as ck
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.encoding import (
    CRITIC_PLAYERS,
    MAX_PLAYERS,
    action_size,
    dims_for,
    make_encoder,
    row_layout,
)
from catanrl.models import build_model, resolve_spec
from catanrl.models.rows import feature_names
from catanrl.models.seat import (
    SeatAttention,
    SeatPool,
    SeatTrunk,
    relative_turn_order,
    turn_order_features,
)

TINY = "graph-seat-tiny"
BASE_6P = {"max_players": 6, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}
EXTENSION_6P = dims_for(["extension-6p"])
#: (game, the dims its rows are padded to): two to six seats, in rows of two to six slots.
TABLES = [("base-2p", BASE_2P), ("base-2p", BASE_4P), ("base-3p", BASE_4P), ("base-4p", BASE_4P),
          ("base-2p", BASE_6P), ("base-4p", BASE_6P), ("extension-5p", EXTENSION_6P), ("extension-6p", EXTENSION_6P)]
#: The networks the set properties are checked on: (model args, level, critic level, dice input).
NETWORKS = {
    "separate": ({}, "private", None, False),
    "shared": ({"shared": True}, "private", None, False),
    "pool": ({"shared": True, "attention": False}, "private", None, False),
    "critic": ({}, "private", "full", False),
    "aux": ({"shared": True, "aux": True}, "private", "full", False),
    "dice": ({"shared": True, "outcome": True}, "full", None, True),
}
#: The per-slot features outside the players block: ``<stem>_<slot>`` or ``<stem>_<slot>_<total>``.
SEAT_FEATURE = re.compile(
    r"^(settlement|city|distance|road|buildings|dice_odds|dice_sevens|dice_seven_share|dice_streak)_(\d+)(?:_\d+)?$")


def seat_model(dims=BASE_4P, *, level="private", critic=None, dice=False, seed=0, preset=TINY, **args):
    torch.manual_seed(seed)
    levels = {"level": level, **({"critic_level": critic} if critic else {}), **({"dice_state": True} if dice else {})}
    return build_model({"preset": preset, "args": args}, dims=dims, levels=levels, num_actions=action_size(dims),
                       layout=row_layout(dims, level, critic, dice)).eval()


def network(name: str, dims=BASE_4P, seed: int = 0):
    args, level, critic, dice = NETWORKS[name]
    return seat_model(dims, level=level, critic=critic, dice=dice, seed=seed, **args)


def rows_of(played, dims, *, level="private", critic=None, dice=False) -> torch.Tensor:
    """Rows of the seat to act, as the VecEnv writes them (the critic's players block last)."""
    actor = make_encoder(dims, level, dice)
    other = make_encoder(dims, critic, dice) if critic else None
    rows = []
    for game in played:
        seat = game.current_player_index
        row = actor.encode(game, seat)
        rows.append(row if other is None else np.concatenate([row, other.encode_players(game, seat)]))
    return torch.from_numpy(np.stack(rows).astype(np.float32))


def rows_for(name: str, played, dims=BASE_4P) -> torch.Tensor:
    _, level, critic, dice = NETWORKS[name]
    return rows_of(played, dims, level=level, critic=critic, dice=dice)


def trunks(model) -> list[SeatTrunk]:
    return [m for m in model.modules() if isinstance(m, SeatTrunk)]


def trained_scale(model, seed: int = 1, scale: float = 0.1):
    """Every weight moved off its initial value (the zero ones too: output layers' gains, the
    attention's turn-order bias, the dice input), so that every input reaches the outputs."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * scale)
    return model


def without_turn_order(model):
    """Test-only: every other seat reads the same turn-order features (``is_self`` stays) and
    the attention no turn-order bias, so that nothing tells two other seats apart."""
    with torch.no_grad():
        for trunk in trunks(model):
            trunk.turn_features[..., 1:] = 0.0
            for layer in trunk.layers:
                if isinstance(layer.seat_from_seats, SeatAttention):
                    layer.seat_from_seats.forward_bias.zero_()
                    layer.seat_from_seats.backward_bias.zero_()
    return model


def outputs(model, rows) -> dict[str, torch.Tensor]:
    """Every output of the network: logits, values, and the aux and outcome heads' if it has them."""
    with torch.no_grad():
        logits, values = model(rows)
        out = {"logits": logits, "values": values}
        if model.aux:
            out["aux"] = model.aux_predictions(rows)
        if model.outcome:
            out["vp"], out["turns"] = model.heads_with_outcome(rows)[3:]
    return out


class Columns:
    """The columns of a raw row that belong to one seat slot."""

    def __init__(self, dims, level="private", critic=None, dice=False) -> None:
        self.layout = row_layout(dims, level, critic, dice)
        self.names = feature_names(dims, level, dice)
        self.players = dims["max_players"]
        self.mask = self.layout["action_mask"][0]
        offset, (tiles, victims) = cpp.GraphActionSpace(make_encoder(dims, level, dice).dims).layout()["robber"]
        assert victims == self.players
        self.robber = offset + np.arange(tiles) * victims

    def feature(self, block: str, name: str) -> np.ndarray:
        offset, shape = self.layout[block]
        entities = shape[0] if len(shape) == 2 else 1
        return offset + np.arange(entities) * shape[-1] + self.names[block].index(name)

    def owned(self, k: int, blocks=("global", "tiles", "nodes", "edges"), stem: str = "") -> list[np.ndarray]:
        """Slot ``k``'s per-slot features in ``blocks`` whose names start with ``stem``."""
        found = []
        for block in blocks:
            for name in self.names[block]:
                match = SEAT_FEATURE.match(name)
                if match and int(match.group(2)) == k and name.startswith(stem):
                    found.append(self.feature(block, name))
        return found

    def seat(self, k: int, present: bool = True) -> np.ndarray:
        """Slot ``k``'s players features (the critic block's too), ownership channels and dice
        features, in the same order for every slot; without ``present`` its ``present`` flags."""
        columns = []
        for block in ("players", CRITIC_PLAYERS):
            if block in self.layout:
                offset, (_, width) = self.layout[block]
                keep = [j for j, name in enumerate(self.names["players"]) if present or name != "present"]
                columns.append(offset + k * width + np.array(keep))
        return np.concatenate(columns + self.owned(k))

    def victim(self, k: int) -> np.ndarray:
        """The actions that move the robber and rob slot ``k`` (slot 0: nobody)."""
        return self.robber + k

    def exchanged(self, rows: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """The rows with slots ``a`` and ``b`` exchanged, the robber's victims in the mask too."""
        out = rows.clone()
        for x, y in ((a, b), (b, a)):
            out[:, self.seat(x)] = rows[:, self.seat(y)]
            out[:, self.mask + self.victim(x)] = rows[:, self.mask + self.victim(y)]
        return out

    def exchanged_outputs(self, out: dict[str, torch.Tensor], a: int, b: int) -> dict[str, torch.Tensor]:
        """Outputs with slots ``a`` and ``b`` (both other seats) exchanged."""
        seats = list(range(self.players))
        seats[a], seats[b] = b, a
        actions = np.arange(out["logits"].shape[1])
        actions[self.victim(a)], actions[self.victim(b)] = self.victim(b), self.victim(a)
        others = [k - 1 for k in seats[1:]]
        pick = {"logits": lambda x: x[:, actions], "values": lambda x: x[:, seats], "aux": lambda x: x[:, others],
                "vp": lambda x: x[:, seats], "turns": lambda x: x}
        return {key: pick[key](value) for key, value in out.items()}


def columns_for(name: str, dims=BASE_4P) -> Columns:
    _, level, critic, dice = NETWORKS[name]
    return Columns(dims, level, critic, dice)


# ---------------------------------------------------------------- presets and sizes


def test_the_presets_resolve_and_graph_seat_s_is_graph_s_sized():
    spec = resolve_spec({"preset": "graph-seat-s"})
    assert spec["class"] == "seat" and resolve_spec(spec) == spec
    assert spec["args"] == {"hidden": 96, "global_hidden": 64, "layers": 3, "head_hidden": 128, "shared": False,
                            "aux": False, "seat_hidden": 48, "relation_hidden": 4, "attention": True,
                            "attention_hidden": 24, "attention_heads": 4}
    count = lambda m: sum(p.numel() for p in m.parameters())
    graph = count(model_for("graph-s", BASE_4P, shared=True))
    seat = seat_model(preset="graph-seat-s", shared=True)
    assert graph == 541_540 and count(seat) == 579_844  # + 7.1%: inside graph-s's size +- 10%
    assert count(seat_model(preset="graph-seat-s")) == 920_308 and count(seat_model(BASE_2P, preset="graph-seat-s")) == 920_308
    assert count(seat_model(preset="graph-seat-s", shared=True, attention=False)) < count(seat)
    assert seat.dims_agnostic and not seat.recurrent
    assert seat.parts == model_for("graph-s", BASE_4P, shared=True).parts
    assert seat_model().parts == {"actor": ["actor."], "critic": ["critic."]}
    with pytest.raises(ValueError, match="unknown seat args"):
        resolve_spec({"preset": TINY, "args": {"seats": 3}})
    with pytest.raises(ValueError, match="attention_heads"):
        seat_model(attention_hidden=9)


def test_turn_order_tables():
    features, order = turn_order_features(), relative_turn_order()
    assert features.shape == (MAX_PLAYERS + 1, MAX_PLAYERS, 2 * MAX_PLAYERS - 1)
    for n in range(1, MAX_PLAYERS + 1):
        assert features[n, 0].tolist() == [1.0] + [0.0] * 10 and not features[n, n:].any()
        for k in range(1, n):  # is_self, forward distance k (1..5), backward distance n - k (1..5)
            assert features[n, k].nonzero().flatten().tolist() == [k, 5 + n - k]
        assert all(order[n, i, j] == (j - i) % n for i in range(n) for j in range(n))
        assert not order[n, n:].any() and not order[n, :, n:].any()
    assert not features[0].any() and len({tuple(f.tolist()) for n in range(2, 7) for f in features[n, 1:n]}) == 15


# ---------------------------------------------------------------- table sizes


@pytest.mark.parametrize(("game", "dims"), TABLES, ids=[f"{g}-in-{d['max_players']}" for g, d in TABLES])
@pytest.mark.parametrize("shared", [True, False])
def test_every_table_size_runs_forward_and_backward(game, dims, shared):
    seats = int(game.split("-")[1][0])
    model = trained_scale(seat_model(dims, shared=shared, seed=2)).train()
    obs = rows_of(played_games(game, count=5, plies=120), dims)
    logits, value_logits, present = model.heads(obs)
    mask = obs[:, model.mask_offset : model.mask_offset + model.num_actions]
    assert logits.shape == mask.shape and value_logits.shape == present.shape == (5, dims["max_players"])
    assert torch.all((logits > -1e8) == (mask > 0)) and torch.isfinite(value_logits).all()
    assert torch.all(present[:, :seats] == 1) and not present[:, seats:].any()
    with torch.no_grad():
        played, values = model(obs)
    assert torch.equal(played, logits) and torch.all(mask.gather(1, logits.argmax(1, keepdim=True)) > 0)
    assert torch.all((values[:, :seats] > 0) & (values[:, :seats] < 1)) and not values[:, seats:].any()
    # backward, with every action open so that every head is on the loss's path
    obs[:, model.mask_offset : model.mask_offset + model.num_actions] = 1.0
    logits, value_logits, present = model.heads(obs)
    target = torch.zeros_like(value_logits)
    target[:, 0] = 1.0
    value = torch.nn.functional.binary_cross_entropy_with_logits(value_logits, target, reduction="none") * present
    (value.mean() - torch.log_softmax(logits, dim=1)[:, ::7].mean()).backward()
    grads = {name: p.grad for name, p in model.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert not {name for name, g in grads.items() if not g.any()}  # no weight is off the path


def test_fresh_graph_seat_s_predicts_one_over_n_per_present_seat():
    model = seat_model(preset="graph-seat-s", shared=True)
    for game, seats in (("base-2p", 2), ("base-3p", 3), ("base-4p", 4)):
        _, values = model(rows_of(played_games(game, count=4), BASE_4P))
        assert torch.allclose(values[:, :seats], torch.full((4, seats), 1.0 / seats), atol=0.05)
        assert not values[:, seats:].any()


def test_an_empty_row_is_finite():
    """A row of zeros (no seat, no board: an env slot that is not playing) gives finite
    outputs and gradients beside real rows, and changes none of theirs."""
    model = trained_scale(seat_model(shared=True, aux=True, critic="full", seed=2)).train()
    real = rows_of(played_games("base-3p", count=3), BASE_4P, critic="full")
    obs = torch.cat((real, torch.zeros(2, real.shape[1])))
    logits, value_logits, present, aux = model.heads_with_aux(obs)
    assert all(torch.isfinite(x).all() for x in (value_logits, aux)) and not present[3:].any()
    assert torch.all(logits[3:] < -1e8)
    alone = model.heads_with_aux(real)
    assert torch.allclose(logits[:3], alone[0], atol=1e-5) and torch.allclose(value_logits[:3], alone[1], atol=1e-5)
    (value_logits.sum() + aux.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_the_network_is_dims_agnostic():
    """The same weights give the same outputs on rows padded to more seats, tiles, nodes and edges."""
    small = trained_scale(seat_model(BASE_2P, seed=3))
    played = played_games(count=8)
    logits_s, values_s = small(rows_of(played, BASE_2P))
    for dims in (WIDE_4P, BASE_6P):
        large = seat_model(dims, seed=99)
        large.load_state_dict(small.state_dict())  # identical parameter shapes
        logits_l, values_l = large(rows_of(played, dims))
        names_l = {name: i for i, name in enumerate(_names(dims))}
        columns = [names_l[name] for name in _names(BASE_2P)]
        assert torch.allclose(logits_s, logits_l[:, columns], atol=1e-4)
        assert torch.allclose(values_s, values_l[:, :2], atol=1e-5) and torch.all(values_l[:, 2:] == 0)


@pytest.mark.parametrize("attention", [True, False])
def test_a_batch_mixes_table_sizes_and_maps(attention):
    dims = dims_for(["base-2p", "base-3p", "base-4p", "random-4p"])
    model = trained_scale(seat_model(dims, shared=True, attention=attention, seed=4))
    mixed = [g for name in ("base-2p", "base-3p", "base-4p", "random-4p") for g in played_games(name, count=3)]
    obs = rows_of(mixed, dims)
    logits, values = model(obs)
    for i in range(obs.shape[0]):
        alone_logits, alone_values = model(obs[i : i + 1])
        assert torch.allclose(logits[i], alone_logits[0], atol=1e-4)
        assert torch.allclose(values[i], alone_values[0], atol=1e-5)


# ---------------------------------------------------------------- the seats are a set


@pytest.mark.parametrize("name", list(NETWORKS))
@pytest.mark.parametrize(("game", "dims"), [("base-2p", BASE_4P), ("base-3p", BASE_4P), ("base-2p", BASE_6P),
                                            ("base-4p", BASE_6P)])
def test_what_an_absent_slot_holds_changes_no_output(name, game, dims):
    seats = int(game.split("-")[1][0])
    model = trained_scale(network(name, dims, seed=5))
    columns = columns_for(name, dims)
    rows = rows_for(name, played_games(game, count=6, plies=150), dims)
    noisy = rows.clone()
    generator = torch.Generator().manual_seed(6)
    for k in range(seats, dims["max_players"]):  # everything of the slot but its `present` flags
        held = columns.seat(k, present=False)
        noisy[:, held] = torch.rand(rows.shape[0], len(held), generator=generator) * 4.0 - 1.0
    assert (noisy != rows).any(dim=0).sum() > 100 * (dims["max_players"] - seats)
    want, got = outputs(model, rows), outputs(model, noisy)
    assert set(got) == {"logits", "values", *(["aux"] if model.aux else []), *(["vp", "turns"] if model.outcome else [])}
    for key in want:
        assert torch.equal(want[key], got[key]), key


@pytest.mark.parametrize("name", list(NETWORKS))
@pytest.mark.parametrize(("game", "dims", "a", "b"), [("base-3p", BASE_4P, 1, 2), ("base-4p", BASE_4P, 1, 3),
                                                      ("base-4p", BASE_6P, 2, 3), ("extension-6p", EXTENSION_6P, 2, 5)])
def test_without_the_turn_order_other_seats_are_exchangeable(name, game, dims, a, b):
    """Weight sharing: exchanging two other seats' features, ownership channels, dice features
    and robber victims exchanges their values (and their aux, outcome and robber outputs) and
    leaves every other output; with the turn order back, the outputs differ."""
    columns = columns_for(name, dims)
    rows = rows_for(name, played_games(game, count=6, plies=200, seed=7), dims)
    rows[:, columns.mask : columns.mask + action_size(dims)] = 1.0  # every logit, the robber's too
    if NETWORKS[name][3]:  # these games roll fair dice: give each seat dice features of its own
        generator = torch.Generator().manual_seed(8)
        for k in range(int(game.split("-")[1][0])):
            dice = np.concatenate(columns.owned(k, stem="dice_"))
            rows[:, dice] = torch.rand(rows.shape[0], len(dice), generator=generator)
    there = columns.exchanged(rows, a, b)
    assert (there != rows).any(dim=1).all() and torch.equal(columns.exchanged(there, a, b), rows)

    model = without_turn_order(trained_scale(network(name, dims, seed=9)))
    want, got = outputs(model, rows), columns.exchanged_outputs(outputs(model, there), a, b)
    for key in want:
        assert torch.allclose(want[key], got[key], atol=2e-5, rtol=1e-5), key  # the seats' sums in another order
    # the check has teeth: the two seats' outputs differ, as do the robber logits by victim
    assert (want["values"][:, a] - want["values"][:, b]).abs().max() > 1e-3
    assert (want["logits"][:, columns.victim(a)] - want["logits"][:, columns.victim(b)]).abs().max() > 1e-2

    ordered = trained_scale(network(name, dims, seed=9))
    want, got = outputs(ordered, rows), columns.exchanged_outputs(outputs(ordered, there), a, b)
    assert (want["values"] - got["values"]).abs().max() > 1e-3 and (want["logits"] - got["logits"]).abs().max() > 1e-2


def test_the_attention_bias_alone_tells_the_turn_order():
    """Turn-order features zeroed, bias kept: the attention's bias by relative turn order is
    enough to tell two other seats apart (the pooled stand-in has no such bias)."""
    columns = Columns(BASE_4P)
    rows = rows_of(played_games("base-4p", count=6, plies=200, seed=7), BASE_4P)
    there = columns.exchanged(rows, 1, 3)
    for attention in (True, False):
        model = trained_scale(seat_model(shared=True, attention=attention, seed=9))
        kinds = {type(layer.seat_from_seats) for trunk in trunks(model) for layer in trunk.layers}
        assert kinds == {SeatAttention if attention else SeatPool}
        generator = torch.Generator().manual_seed(2)
        with torch.no_grad():
            for trunk in trunks(model):
                trunk.turn_features[..., 1:] = 0.0
                for layer in trunk.layers if attention else ():
                    layer.seat_from_seats.forward_bias.normal_(generator=generator)
                    layer.seat_from_seats.backward_bias.normal_(generator=generator)
        want, got = outputs(model, rows), columns.exchanged_outputs(outputs(model, there), 1, 3)
        gap = (want["logits"] - got["logits"]).abs().max()
        assert gap > 1e-2 if attention else gap < 2e-5


# ---------------------------------------------------------------- the graph network's options


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize(("level", "critic"), [("private", None), ("public", None), ("full", None),
                                               ("private", "full"), ("public", "full"), ("private", "public")])
def test_every_information_level_and_critic_level(shared, level, critic):
    if shared and critic:  # one trunk reads one row: the block is the aux head's targets
        with pytest.raises(ValueError):
            seat_model(level=level, critic=critic, shared=True)
        model = seat_model(level=level, critic=critic, shared=True, aux=True)
    else:
        model = seat_model(level=level, critic=critic, shared=shared)
    played = played_games("base-4p", count=5, plies=150)
    rows = rows_of(played, BASE_4P, level=level, critic=critic)
    assert rows.shape[1] == model.obs_size
    out = outputs(model, rows)
    assert all(torch.isfinite(v).all() for k, v in out.items() if k != "logits")
    assert torch.equal(out["logits"] > -1e8, rows[:, model.mask_offset : model.mask_offset + model.num_actions] > 0)
    if critic and not shared:  # the critic reads the block: the actor's row alone leaves the values open
        other = rows.clone()
        offset, (players, width) = model.layout[CRITIC_PLAYERS]
        hand = offset + width * np.arange(1, players) + feature_names(BASE_4P, level)["players"].index("resource_wood")
        other[:, hand] += 0.5
        moved = outputs(trained_scale(model), other)
        still = outputs(model, rows)
        assert torch.equal(moved["logits"], still["logits"]) and not torch.equal(moved["values"], still["values"])


def test_the_aux_and_outcome_heads_read_the_seat_tokens():
    plain = seat_model(shared=True, critic="full", aux=True, seed=3)
    assert plain.parts == {"actor": ["backbone.", "policy_head."], "critic": ["value_head."], "aux": ["aux_head."]}
    rows = rows_of(played_games("base-4p", count=4), BASE_4P, critic="full")
    aux = plain.heads_with_aux(rows)[3]
    targets, others = plain.aux_targets(rows)
    assert aux.shape == targets.shape == (4, 3, plain.aux_head.num_features) and others.shape == (4, 3)
    assert plain.aux_head.mlp[0].in_features == plain.backbone.pooled_dim + plain.backbone.seat_hidden
    with_head = seat_model(shared=True, critic="full", aux=True, outcome=True, seed=3)
    theirs, mine = plain.state_dict(), with_head.state_dict()
    assert list(mine)[: len(theirs)] == list(theirs) and all(torch.equal(mine[k], theirs[k]) for k in theirs)
    assert all(k.startswith("outcome_head.") for k in list(mine)[len(theirs):])
    vp, turns = with_head.heads_with_outcome(rows)[3:]
    assert vp.shape == (4, 4) and turns.shape == (4,) and torch.isfinite(vp).all() and torch.isfinite(turns).all()
    with pytest.raises(ValueError, match="no outcome head"):
        plain.heads_with_outcome(rows)
    with pytest.raises(ValueError, match="shared"):
        seat_model(outcome=True)


# ---------------------------------------------------------------- the dice input


def fr_model(dice: bool, *, shared: bool = True, seed: int = 0, **args):
    return seat_model(FR_DIMS, level="full", dice=dice, seed=seed, shared=shared, **args)


def fr_rows(dice: bool) -> torch.Tensor:
    encoder = make_encoder(FR_DIMS, "full", dice)
    return torch.from_numpy(np.stack([encoder.encode(g, s) for g, s in balanced_games()]))


def dice_keys(shared: bool) -> list[str]:
    layers = ("dice_projection", "seat_dice_projection", "tile_dice_projection")
    return sorted(f"{trunk}.{layer}.weight" for trunk in (["backbone"] if shared else ["actor.trunk", "critic.trunk"])
                  for layer in layers)


@pytest.mark.parametrize("shared", [True, False])
def test_a_fresh_dice_network_is_the_plain_one_and_each_projection_reads_its_features(shared):
    plain, dice = fr_model(False, shared=shared, seed=3), fr_model(True, shared=shared, seed=3)
    theirs, mine = plain.state_dict(), dice.state_dict()
    assert sorted(set(mine) - set(theirs)) == sorted(ck.dice_input_keys(dice)) == dice_keys(shared)
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs) and not any(mine[k].any() for k in dice_keys(shared))
    trunk = trunks(dice)[0]
    assert (trunk.dice_inputs, trunk.seat_dice_inputs, trunk.tile_dice_inputs) == (23, 14, 4)
    assert not ck.dice_input_keys(plain) and trunks(plain)[0].seat_dice_projection is None
    rows, plain_rows = fr_rows(True), fr_rows(False)
    want = outputs(plain, plain_rows)
    assert all(torch.equal(v, want[k]) for k, v in outputs(dice, rows).items())  # the weights add exact zeros
    for key in dice_keys(shared):  # one projection at a time moves the outputs
        state = {k: v.clone() for k, v in mine.items()}
        state[key] = torch.randn(state[key].shape, generator=torch.Generator().manual_seed(1)) * 0.5
        moved = fr_model(True, shared=shared, seed=4)
        moved.load_state_dict(state)
        got = outputs(trained_scale(moved, scale=0.0), rows)
        part = "values" if key.startswith("critic.") else "logits"
        assert not torch.equal(got[part], want[part]), key


def test_dice_weight_norms_count_the_seat_projection():
    from catanrl.train.dice_probe import dice_weight_norms

    model = fr_model(True, seed=3)
    assert dice_weight_norms(fr_model(False)) == {}
    assert dice_weight_norms(model) == {"dice/weight_norm": 0.0, "dice/weight_norm_global": 0.0,
                                        "dice/weight_norm_tiles": 0.0}
    with torch.no_grad():
        model.backbone.dice_projection.weight.fill_(1.0)       # 23 x 8
        model.backbone.seat_dice_projection.weight.fill_(1.0)  # 14 x 8
        model.backbone.tile_dice_projection.weight.fill_(2.0)  # 4 x 16
    norms = dice_weight_norms(model)
    assert norms["dice/weight_norm_global"] == pytest.approx((37 * 8) ** 0.5)
    assert norms["dice/weight_norm_tiles"] == pytest.approx(16.0)
    assert norms["dice/weight_norm"] == pytest.approx((37 * 8 + 256) ** 0.5)


@pytest.mark.parametrize("shared", [True, False])
def test_zero_new_inputs_warm_starts_the_dice_input(tmp_path, shared):
    source = trained_scale(fr_model(False, shared=shared, seed=3))
    ref = saved(tmp_path, source)
    target = trained_scale(fr_model(True, shared=shared, seed=4))  # its dice weights are off zero
    with pytest.raises(ck.CheckpointError, match="init_from.zero_new_inputs"):
        ck.load_parts(target, {"ckpt": str(ref.path)})
    ck.load_parts(target, {"ckpt": str(ref.path), "parts": ["actor", "critic"], "zero_new_inputs": True})
    theirs, mine = source.state_dict(), target.state_dict()
    assert sorted(set(mine) - set(theirs)) == dice_keys(shared)
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs) and not any(mine[k].any() for k in dice_keys(shared))
    want = outputs(source, fr_rows(False))
    assert all(torch.equal(v, want[k]) for k, v in outputs(target, fr_rows(True)).items())


# ---------------------------------------------------------------- checkpoints


def test_checkpoints_round_trip_and_init_from_copies_parts(tmp_path):
    rows = fr_rows(True)
    model = trained_scale(fr_model(True, shared=False, seed=3))
    ref = saved(tmp_path, model)
    stored = ck.Checkpoint.load(ref.path)
    assert stored.header["model"] == model.spec and stored.header["model"]["class"] == "seat"
    assert stored.dice_state and stored.dims_agnostic and not stored.recurrent
    loaded = stored.model("all")
    assert all(torch.equal(v, w) for v, w in zip(outputs(loaded, rows).values(), outputs(model, rows).values()))
    actor = stored.model("actor")
    with torch.no_grad():
        assert torch.equal(actor.policy_logits(rows), model.policy_logits(rows))
    # the critic alone, into a fresh network
    fresh = fr_model(True, shared=False, seed=8)
    before = outputs(fresh, rows)
    ck.load_parts(fresh, {"ckpt": str(ref.path), "parts": ["critic"]})
    after = outputs(fresh, rows)
    assert torch.equal(after["logits"], before["logits"]) and torch.equal(after["values"], outputs(model, rows)["values"])
    # a graph checkpoint is another class: refused, not half-loaded
    graph = saved(tmp_path / "graph", model_for("graph-tiny", FR_DIMS, "full", shared=True))
    with pytest.raises(ck.CheckpointError):
        ck.load_parts(fr_model(False), {"ckpt": str(graph.path)})


def test_split_shared_copies_the_trunk_into_both_networks(tmp_path):
    source = trained_scale(fr_model(False, shared=True, seed=5))
    ref = saved(tmp_path, source)
    target = fr_model(True, shared=False, seed=6)
    ck.load_parts(target, {"ckpt": str(ref.path), "split_shared": True, "zero_new_inputs": True})
    want, got = outputs(source, fr_rows(False)), outputs(target, fr_rows(True))
    assert torch.equal(want["logits"], got["logits"]) and torch.equal(want["values"], got["values"])
    assert all(torch.equal(target.state_dict()[f"{net}.trunk.{key}"], value)
               for net in ("actor", "critic") for key, value in source.backbone.state_dict().items())


# ---------------------------------------------------------------- agents and kernels


@pytest.mark.parametrize("dice", [False, True])
def test_the_search_agents_play_a_checkpoint_on_torch_kernels(tmp_path, dice):
    ref = saved(tmp_path, trained_scale(fr_model(dice, seed=7)))
    game, seat = next((g, s) for g, s in balanced_games(3, 400, 37) if len(g.playable_actions) > 2)
    for spec in ("POLICY:ckpt={}", "BELIEF:ckpt={}:worlds=3", "MCTS:ckpt={}:sims=8:dets=1:kernels=torch",
                 "MCTS:ckpt={}:sims=8:oracle=true"):
        agent = native(spec.format(ref.path))
        assert agent.model.spec["class"] == "seat" and agent.dice_state == dice
        row, choice, spy = decide(agent, game, seat)
        legal = np.flatnonzero(row[-agent.action_space.size:])
        assert int(np.asarray(choice).reshape(-1)[0]) in legal, spec
        seen = np.concatenate(spy.rows)
        assert seen.shape[1] == agent.model.obs_size and len(seen)


def test_the_fused_kernels_are_the_class_s_own(tmp_path):
    """On the CPU: the evaluator asks for a CUDA device as for the graph class (the kernels
    themselves: tests/test_seat_kernels.py), the graph class's kernels name the seats', and
    train.kernels resolves for the class."""
    from catanrl.models.graph_kernels import FusedGraph, FusedGraph16
    from catanrl.search.evaluator import BatchEvaluator

    model = fr_model(True, seed=7)
    for kernels in ("fused", "fused16"):
        with pytest.raises(ValueError, match=f"kernels={kernels} .* CUDA"):
            BatchEvaluator(model, device="cpu", infer_rows=8, kernels=kernels)
    for fused in (FusedGraph, FusedGraph16):
        with pytest.raises(ValueError, match="seat_kernels.FusedSeat"):
            fused(model)
    assert BatchEvaluator(model, device="cpu", infer_rows=8).kernels == "torch"
    ref = saved(tmp_path, model)
    with pytest.raises(ValueError, match="kernels=fused .* CUDA"):
        native(f"MCTS:ckpt={ref.path}:sims=8:dets=1:kernels=fused")
    from test_alphazero import ENV, TRAIN

    def raw(preset, **train):
        return {"seed": 3, "env": ENV, "obs": {"level": "full"}, "model": {"preset": preset, "args": {"shared": True}},
                "train": {**TRAIN, **train}, "runtime": {"device": "cpu", "threads": 2, "determinism": "strict"},
                "eval": {"select": None, "final": []}}

    for kernels in ("torch", "fused", "fused16"):
        for preset in (TINY, "graph-seat-s", "graph-tiny"):
            assert resolve(raw(preset, kernels=kernels), experiment="seat", variant="a")["train"]["kernels"] == kernels


# ---------------------------------------------------------------- float16


def test_float16_stays_close_to_float32():
    from catanrl.train.puffer_policy import PufferPolicy

    model = trained_scale(seat_model(shared=True, seed=3), scale=0.05)
    obs = rows_of(played_games("base-4p", count=8, plies=150), BASE_4P)
    exact = PufferPolicy(model, num_rows=8)
    half = PufferPolicy(model, num_rows=8, amp_dtype=torch.float16)
    with torch.no_grad():
        want_logits, want_values = exact.forward_eval(obs)
        logits, values = half.forward_eval(obs)
    assert logits.dtype == values.dtype == torch.float32 and not torch.is_autocast_enabled("cpu")
    legal = want_logits > -1e8
    assert torch.equal(logits > -1e8, legal) and torch.isfinite(values).all()
    assert torch.allclose(logits[legal], want_logits[legal], atol=0.05, rtol=0.02)
    assert torch.allclose(values, want_values, atol=0.02)
    train_logits, batch = half(obs)
    loss = train_logits[legal].sum() + batch.logits.sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


# ---------------------------------------------------------------- tiny trainings


def mixed_ppo(**model_args) -> dict:
    """A tiny PPO run on 2, 3 and 4 players: rows padded to four slots."""
    from train_configs import tiny_ppo

    config = tiny_ppo(seed=4, epochs=3, critic_targets="lambda", critic_gamma=1.0)
    config["env"]["scenarios"] = [{"game": "base-2p", "maps": 4}, {"game": "base-3p", "maps": 4},
                                  {"game": "base-4p", "maps": 4},
                                  {"game": "base-4p", "maps": 4, "seats": ["learner", "R", "learner", "R"], "name": "vsR"}]
    config["train"]["total_timesteps"] = 8 * 4 * 8 * 3  # envs x slots x horizon x epochs
    config["model"] = {"preset": TINY, "args": model_args}
    return config


@pytest.mark.parametrize("kind", ["shared-aux-float16", "separate"])
def test_a_tiny_ppo_run_trains_on_two_three_and_four_players(kind):
    from run_helpers import train

    from catanrl.train.puffer import train_ppo

    if kind == "separate":
        config = mixed_ppo()
    else:
        config = mixed_ppo(shared=True, aux=True)
        config["obs"] = {"level": "private", "critic_level": "full"}
        config["train"].update(aux_coef=0.1, precision="float16", amp=True)
    writer = train(train_ppo, config, kind)
    assert writer.status == "finished" and len(writer.rows) == 3
    assert all(np.isfinite(v) for row in writer.rows for v in row.values() if isinstance(v, float))
    first, last = ck.Checkpoint.load(writer.saved[0].path), ck.Checkpoint.load(writer.saved[-1].path)
    assert last.header["model"]["preset"] == TINY and last.header["model"]["class"] == "seat"
    before, after = first.model("all").state_dict(), last.model("all").state_dict()
    moved = {key for key in before if not torch.equal(before[key], after[key])}
    assert any("seat_projection" in k for k in moved) and any("seat_from_seats.query" in k for k in moved)
    assert any("node_from_seat" in k for k in moved) and any("seat_from_board" in k for k in moved)
    model = last.model("all")
    dims = {k: model.dims[k] for k in BASE_4P}
    assert dims["max_players"] == 4
    for game in ("base-2p", "base-3p", "base-4p"):
        rows = rows_of(played_games(game, count=3), dims, critic="full" if kind != "separate" else None)
        assert all(torch.isfinite(v).all() for k, v in outputs(model, rows).items() if k != "logits")


def az_variant(name: str, *, game: str, dice: bool, outcome: bool, **train) -> Variant:
    from test_alphazero import TRAIN

    args = {"shared": True, **({"outcome": True} if outcome else {})}
    raw = {"seed": 3, "env": {"scenarios": [{"game": game, "maps": 4, "turn_limit": 8}], "num_envs": 4},
           "obs": {"level": "full", **({"dice_state": True} if dice else {})},
           "model": {"preset": TINY, "args": args}, "train": {**TRAIN, **train},
           "runtime": {"device": "cpu", "threads": 2, "determinism": "strict"}, "eval": {"select": None, "final": []}}
    cfg = resolve(raw, experiment="seat", variant=name)
    return Variant("seat", name, None, {}, cfg, config_hash(cfg))


@pytest.mark.parametrize(("name", "game", "extras"), [
    ("fr", FR_GAME, {"dice": True, "outcome": True, "outcome_vp_coef": 0.5, "outcome_turns_coef": 0.5}),
    ("three", "base-3p", {"dice": False, "outcome": False}),
])
def test_a_tiny_alphazero_run_trains_and_saves(tmp_path, monkeypatch, name, game, extras):
    from catanrl.runs import RunWriter, load_experiment
    from catanrl.train.alphazero import train_alphazero

    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)
    variant = az_variant(name, game=game, **extras)
    with RunWriter.open(variant, argv=["test"]) as writer:
        train_alphazero(writer.variant_for_training, writer)
    assert writer.status == "finished"
    experiment = load_experiment(f"seat/{name}")
    first, last = (ck.Checkpoint.load(experiment.resolve(ref).path) for ref in ("e0", "latest"))
    assert last.header["model"]["class"] == "seat" and last.dice_state == extras["dice"]
    before, after = first.model("all").state_dict(), last.model("all").state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before if "seat_from_seats" in k)
    model = last.model("all")
    assert model.outcome == extras["outcome"]
    dims = {k: model.dims[k] for k in BASE_4P}
    rows = rows_of(played_games(game, count=3), dims, level="full", dice=extras["dice"])
    assert all(torch.isfinite(v).all() for k, v in outputs(model, rows).items() if k != "logits")
