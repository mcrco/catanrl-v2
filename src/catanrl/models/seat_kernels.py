"""The shared-trunk seat-set network's forward on custom CUDA kernels (inference only).

:class:`FusedSeat` computes what :class:`~catanrl.models.SeatModel` computes (masked logits
and ``sigmoid(value logits) * present``) from the same weights, as
:class:`~catanrl.models.graph_kernels.FusedGraph` does for the graph network and with its
kernels for everything the two networks share: the topology as neighbour lists, the board
entities' residual + LayerNorm, their neighbour means, the pooling, the heads and the output
assembly. What the seats add runs on kernels of its own (``seat_kernels.cu``):

* one kernel reads the ownership channels once per row: each entity's weight by seat and
  relation (:func:`catanrl.models.seat.relations`, kept as the dense [entity, seat] weights:
  six seats at most), and the first projections' inputs in the module's order (the plain
  features, "mine" and "another seat's"), so those projections are the module's weights in
  one GEMM per entity type, the dice input's columns and weights (``obs.dice_state``) beside
  the plain ones;
* a layer's seat GEMM gives the tokens' self term, their messages to the board and the
  attention's queries, keys and values at once; a kernel sums each entity's seats' messages
  into the columns of its GEMM input after the board's, whose stacked weights end with
  ``node_from_seat`` / ``tile_from_seat`` / ``edge_from_seat``;
* the board's messages to a seat are the weighted mean of its entities' normalized
  embeddings, then the relation's message weights (linear: applied after the mean, one warp
  per seat and relation), in the kernel that also runs the seat-to-seat mixing (the attention
  with its turn-order bias over the present seats, or the others' mean with ``attention:
  false``); one GEMM takes both through ``seat_from_board`` and the mixing's output weights;
* the tokens' residual + LayerNorm is one kernel, as the board entities'.

The seat GEMMs are small (a row has at most six tokens). Measured on an RTX 2070 Super
beside another job, graph-seat-s through ``BatchEvaluator`` at 64 and 512 rows a forward:
2.0 to 2.2 times the module's rows per second (:class:`FusedSeat16` 2.8 to 3.4 times), 10 to
15% under graph-s on the graph kernels. The outputs differ from the module's in the last
bits, within the graph kernels' bounds (logits within 1e-3 at a scale of 1e3, values within
1e-6; on gen-base-2/main@e3800 over rows of its own two-, three- and four-seat games,
3.1e-4 at logit scales of 520 to 710 and 6.6e-7: ``tests/test_seat_kernels.py``), so an
agent using the kernels keys them. Unlike the graph kernels', the dice input's GEMMs are not
kept apart: a network warm-started with zero dice weights is within rounding of its source
here, not bit-equal to it.

:class:`FusedSeat16` runs the board's GEMMs and the heads' hidden layers in half precision,
as :class:`~catanrl.models.graph_kernels.FusedGraph16` (self-play search, ``train.kernels:
fused16``); the seats' stream and GEMMs stay float32. On the same checkpoint's rows its
logits are within 6e-4 of their scale of :class:`FusedSeat`'s and its values within 3e-4.
"""

from __future__ import annotations

from pathlib import Path

import torch

from catanrl.models.graph_kernels import FusedGraph, FusedGraph16, kernels
from catanrl.models.seat import DISTANCE_NEAR, DISTANCE_ON, DISTANCE_ONE, NODE_RELATIONS

_KERNELS = None


def seat_kernels():
    """The compiled extension of the seats' kernels (built on first use, then cached by torch)."""
    global _KERNELS
    if _KERNELS is None:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("seat_kernels.cu")
        try:
            _KERNELS = load(name="catanrl_seat_kernels", sources=[str(source)], extra_cuda_cflags=["-O3"],
                            verbose=False)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(f"could not build the seat kernels ({error}); they need nvcc on PATH "
                               "(or play with kernels=torch)") from error
    return _KERNELS


class FusedSeat(FusedGraph):
    """The forward of a shared-trunk :class:`~catanrl.models.SeatModel` (no aux head, hidden 96,
    any ``seat_hidden`` up to 256, ``relation_hidden`` and attention widths, ``attention`` on
    or off; an outcome head is left out as :class:`FusedGraph` leaves it) on a CUDA device:
    ``forward(obs)`` is (logits, values, overflow), the flag as :class:`FusedGraph`'s. The
    heads are :class:`FusedGraph`'s on the final seat tokens."""

    def __init__(self, model) -> None:  # the graph class's weights are not read: no super().__init__
        if getattr(model, "spec", {}).get("class") != "seat" or not getattr(model, "shared", False) or model.aux:
            raise ValueError("kernels=fused runs shared-trunk seat-set networks without an aux head "
                             "(model.args.aux; an outcome head is accepted: no kernel runs it)")
        K, KS = kernels(), seat_kernels()
        self.K = K  # the heads' (FusedGraph._heads): the board is hidden 96 here
        trunk = model.backbone
        if trunk.hidden != K.hidden:
            raise ValueError(f"the graph kernels are built for hidden {K.hidden}, not {trunk.hidden}")
        if trunk.seat_hidden > KS.max_seat_hidden:
            raise ValueError(f"the seat kernels are built for seat_hidden up to {KS.max_seat_hidden}, "
                             f"not {trunk.seat_hidden}")
        reader = trunk.reader
        d = lambda t: t.detach()
        c = lambda t: t.detach().contiguous()
        w = lambda m: d(m.weight)
        idx = {b: getattr(reader, f"index_{b}") for b in reader.widths}
        raw = {b: reader.spans[b][1][-1] for b in reader.widths}
        sp = self.spans = reader.spans
        P = self.P = reader.num_players
        self.H, self.S, self.Hg = trunk.hidden, trunk.seat_hidden, trunk.global_hidden
        self.N, self.T, self.E = sp["nodes"][1][0], sp["tiles"][1][0], sp["edges"][1][0]
        valid = {b: int(idx[b][trunk.valid[b]]) for b in ("nodes", "tiles", "edges")}
        self.present = int(idx["players"][trunk.present])
        self.layout = [self.N, self.T, self.E, sp["nodes"][0], sp["tiles"][0], sp["edges"][0],
                       raw["nodes"], raw["tiles"], raw["edges"], valid["nodes"], valid["tiles"], valid["edges"],
                       sp["edge_nodes"][0], sp["tile_nodes"][0]]

        def cols(block: str, canonical) -> list[int]:
            """The raw columns of canonical features (-1: a feature the rows lack, read as 0)."""
            at = idx[block][torch.as_tensor(canonical, dtype=torch.long, device=idx[block].device).flatten()]
            return torch.where(at < raw[block], at, torch.full_like(at, -1)).tolist()

        # The relations kernel's table of raw columns, in seat_kernels.cu's order: it reads the
        # per-slot channels of the row's own slots and writes the first projections' inputs in
        # the module's order, so their weights are the module's.
        self.dice = trunk.dice_projection is not None
        Fp = reader.widths["players"]
        table = [cols("nodes", trunk.node_plain), cols("nodes", trunk.node_settlement[:P]),
                 cols("nodes", trunk.node_city[:P]), cols("nodes", trunk.node_distance[:P]),
                 cols("tiles", trunk.tile_plain), cols("tiles", trunk.tile_buildings[:P]),
                 cols("tiles", trunk.tile_dice_odds[:P]) if self.dice else [],
                 cols("tiles", trunk.tile_dice_plain) if self.dice else [],
                 cols("edges", trunk.edge_plain), cols("edges", trunk.edge_road[:P]),
                 cols("players", list(range(Fp))), cols("global", list(range(trunk.plain_globals))),
                 cols("global", trunk.global_dice) if self.dice else [],
                 cols("global", trunk.global_seat_dice[:P]) if self.dice else []]
        self.slayout = [self.N, self.T, self.E, P, self.S, sp["global"][0], sp["players"][0], sp["nodes"][0],
                        sp["tiles"][0], sp["edges"][0], raw["players"], raw["nodes"], raw["tiles"], raw["edges"],
                        valid["nodes"], valid["tiles"], valid["edges"], self.present,
                        len(table[0]), len(table[4]), len(table[7]) if self.dice else -1, len(table[8]),
                        Fp, trunk.plain_globals, len(table[12]), len(table[13]) // P]
        device = idx["nodes"].device
        self.table = torch.tensor([column for part in table for column in part], dtype=torch.int32, device=device)
        self.bands = [DISTANCE_ON, DISTANCE_ONE, DISTANCE_NEAR]
        self.summary = trunk.plain_globals  # the tokens' summary: after the plain global features

        # The seat projection on [the seat's features | its dice features]; the turn-order
        # features, one-hots, are a term per (seats present, slot) with the bias in.
        ws = w(trunk.seat_projection)
        parts = [ws[:, :Fp]] + ([w(trunk.seat_dice_projection)] if self.dice else [])
        self.wsT = c(torch.cat(parts, dim=1).T)
        self.turn = c(trunk.turn_features[:, :P] @ ws[:, Fp:].T + d(trunk.seat_projection.bias))
        parts = [w(trunk.global_projection)] + ([w(trunk.dice_projection)] if self.dice else [])
        self.wgT, self.bg = c(torch.cat(parts, dim=1).T), d(trunk.global_projection.bias)
        self.proj, wig, bi = {}, [], []
        for b in ("nodes", "tiles", "edges", "seats"):
            lin, inj = getattr(trunk, f"{b[:-1]}_projection"), getattr(trunk, f"{b[:-1]}_injection")
            wi, width = d(inj[0].weight), inj[0].weight.shape[0]
            wig.append(wi[:, width:])
            bi.append(d(inj[0].bias))
            self.proj[b] = {"wiT": c(wi[:, :width].T), "ln": (d(inj[1].weight), d(inj[1].bias), inj[1].eps)}
            if b != "seats":
                parts = [w(lin)] + ([w(trunk.tile_dice_projection)] if self.dice and b == "tiles" else [])
                self.proj[b].update(wT=c(torch.cat(parts, dim=1).T), b=d(lin.bias))
        self.wigT, self.bi = c(torch.cat(wig).T), torch.cat(bi)
        self.D = trunk.layers[0].relation_hidden
        self.M = (len(NODE_RELATIONS) + 2) * self.D
        self.attention = hasattr(trunk.layers[0].seat_from_seats, "query")
        mix = trunk.layers[0].seat_from_seats
        self.heads = mix.heads if self.attention else 0
        self.scale = float(mix.scale) if self.attention else 1.0
        #: The width of what a token receives from the others before the mixing's output weights.
        self.A = mix.query.weight.shape[0] if self.attention else self.S
        self.layers = []
        for layer in trunk.layers:
            mix = layer.seat_from_seats
            L = {
                "ln": {k: (d(getattr(layer, f"{k}_norm").weight), d(getattr(layer, f"{k}_norm").bias),
                           getattr(layer, f"{k}_norm").eps) for k in ("node", "tile", "edge", "seat")},
                # node update = [n | mean of neighbour nodes | of tiles | of edges | its seats' messages] @ stacked W
                "wnT": c(torch.cat([w(layer.node_self), w(layer.node_from_node), w(layer.node_from_tile),
                                    w(layer.node_from_edge), w(layer.node_from_seat)], dim=1).T),
                "bn": d(layer.node_self.bias),
                "wtT": c(torch.cat([w(layer.tile_self), w(layer.tile_from_node), w(layer.tile_from_seat)], dim=1).T),
                "bt": d(layer.tile_self.bias),
                "weT": c(torch.cat([w(layer.edge_self), w(layer.edge_from_node), w(layer.edge_from_seat)], dim=1).T),
                "be": d(layer.edge_self.bias),
                # normalized tokens @ [seat_self | seat_messages | query | key | value]
                "w1T": c(torch.cat([w(layer.seat_self), w(layer.seat_messages)]
                                   + ([w(mix.query), w(mix.key), w(mix.value)] if self.attention else [])).T),
                "bs": d(layer.seat_self.bias),
                "wm": c(torch.cat([w(layer.node_messages), w(layer.edge_messages), w(layer.tile_messages)])),
                # [the board's messages | the seats'] @ [seat_from_board | the mixing's output]
                "w2T": c(torch.cat([w(layer.seat_from_board), w(mix.output if self.attention else mix.others)],
                                   dim=1).T),
                "bias": None,
            }
            if self.attention:  # [seats present, from, to, head] over the row's slots
                order = trunk.turn_order
                L["bias"] = c((d(mix.forward_bias)[order] + d(mix.backward_bias)[order.transpose(1, 2)])[:, :P, :P])
            self.layers.append(L)
        self._head_weights(model)

    def _first(self, obs: torch.Tensor):
        """The first projections (float32 in both precisions): the seats' incidence with the
        board (weights by entity and seat, each seat's sums), the global state after its relu,
        and each entity type's and the tokens' projection through its injection's own weights."""
        KS = seat_kernels()
        xn, xt, xe, xs, gin, present, wN, wE, wT, den = KS.relations(obs, self.slayout, self.bands, self.table)
        tokens = KS.tokens(torch.mm(xs, self.wsT), self.turn, present, gin, self.P, self.summary)
        glob = torch.relu_(torch.addmm(self.bg, gin, self.wgT))
        us = [torch.mm(torch.relu_(torch.addmm(self.proj[b]["b"], r, self.proj[b]["wT"])), self.proj[b]["wiT"])
              for b, r in (("nodes", xn), ("tiles", xt), ("edges", xe))]
        us.append(torch.mm(tokens, self.proj["seats"]["wiT"]))
        return glob, us, present, (wN, wE, wT, den)

    def _norms(self, i: int):
        ln = [self.layers[i]["ln"][k] for k in ("node", "tile", "edge")]
        return [a[0] for a in ln], [a[1] for a in ln], [a[2] for a in ln]

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K, KS = kernels(), seat_kernels()
        obs = obs.float().contiguous()
        B, H, S, N, T, E, P, M, A = obs.shape[0], self.H, self.S, self.N, self.T, self.E, self.P, self.M, self.A
        (_, _, _, vn, vt, ve, nn, nnc, ne, nec, nt, ntc, tn, tnc, en, enc, overflow) = K.topo(obs, self.layout)
        glob, us, present, (wN, wE, wT, den) = self._first(obs)
        gb = torch.addmm(self.bi, glob, self.wigT)  # each type's injection: the glob part, bias in
        types = ("nodes", "tiles", "edges")
        x = [obs.new_empty(B * n, H) for n in (N, T, E)]
        seats, ZS = obs.new_empty(B * P, S), obs.new_empty(B * P, S)
        # The layers' GEMM inputs: the board's columns as FusedGraph's, then the seats' messages.
        sn, st, se = 4 * H + M - 2 * self.D, 2 * H + self.D, 2 * H + self.D
        XN, XT, XE = obs.new_empty(B * N, sn), obs.new_empty(B * T, st), obs.new_empty(B * E, se)
        Z, zs = [XN, XT, XE], [sn, st, se]
        valid, per_row = [vn, vt, ve], [N, T, E]
        none = [None] * 3
        # Injection, then layer 0's LayerNorm into the first columns of each GEMM input.
        K.entities(x, us[:3], none, [gb.narrow(1, k * H, H) for k in range(3)], [3 * H + S] * 3,
                   [self.proj[b]["ln"][0] for b in types], [self.proj[b]["ln"][1] for b in types],
                   [self.proj[b]["ln"][2] for b in types], valid, Z, zs, *self._norms(0), per_row)
        KS.step(seats, us[3], None, None, gb.narrow(1, 3 * H, S), 3 * H + S, *self.proj["seats"]["ln"], present,
                ZS, *self.layers[0]["ln"]["seat"], P)
        for i, L in enumerate(self.layers):
            G1 = torch.mm(ZS, L["w1T"])  # [self | messages to the board | queries | keys | values]
            K.gathers([XN, XT, XE, XN, XN], [sn, st, se, sn, sn], [N, T, E, N, N],
                      [XN.narrow(1, H, H), XN.narrow(1, 2 * H, H), XN.narrow(1, 3 * H, H), XT.narrow(1, H, H),
                       XE.narrow(1, H, H)], [sn, sn, sn, st, se], [N, N, N, T, E],
                      [nn, nt, ne, tn, en], [nnc, ntc, nec, tnc, enc], [1] * 5)
            KS.scatter(G1.narrow(1, S, M), wN, wE, wT, XN, 4 * H, XE, 2 * H, XT, 2 * H, P, self.D, None)
            u = [torch.mm(XN, L["wnT"]), torch.mm(XT, L["wtT"]), torch.mm(XE, L["weT"])]
            FB = obs.new_empty(B * P, M + A)  # what a token receives: from the board, from the seats
            KS.collect(XN, XE, XT, wN, wE, wT, den, L["wm"], present,
                       G1.narrow(1, S + M, 3 * A) if self.attention else ZS, L["bias"], self.scale, self.heads, A,
                       FB, P)
            G2 = torch.mm(FB, L["w2T"])
            last = i == len(self.layers) - 1
            g, bt, eps = (none, none, [0.0] * 3) if last else self._norms(i + 1)
            K.entities(x, u, [L["bn"], L["bt"], L["be"]], none, [0] * 3, none, none, [0.0] * 3,
                       valid, none if last else Z, zs, g, bt, eps, per_row)
            norm = (None, None, None, 0.0) if last else (ZS, *self.layers[i + 1]["ln"]["seat"])
            KS.step(seats, G1, G2, L["bs"], None, 0, None, None, 0.0, present, *norm, P)
        nodes, tiles, edges = x
        players = seats.view(B, P, S)
        pooled = K.pool(nodes, tiles, edges, vn, vt, ve, torch.cat((glob, players[:, 0]), dim=1), N, T, E)
        logits, values = self._heads(obs, nodes, tiles, edges, pooled, players, en, enc)
        return logits, values, overflow


class FusedSeat16(FusedSeat, FusedGraph16):
    """:class:`FusedSeat` with the board's GEMMs and the heads' hidden layers in half
    precision, as :class:`~catanrl.models.graph_kernels.FusedGraph16` (its docstring says
    what is half and what stays float32, and what the flag's bit 2 means); float32 logits and
    values within rounding of :class:`FusedSeat`'s. The seats stay float32: their stream,
    their GEMMs (six tokens a row at most) and what they exchange with the board, which is
    rounded to half only where it enters a board GEMM (a seat's messages summed into an
    entity's input: flagged when not finite) and read from the board's half embeddings."""

    def __init__(self, model) -> None:
        FusedSeat.__init__(self, model)
        self._halves()

    def exact(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The float32 forward (:class:`FusedSeat`'s) of the same weights."""
        return FusedSeat.forward(self, obs)

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K, KS = kernels(), seat_kernels()
        obs = obs.float().contiguous()
        B, H, S, N, T, E, P, M, A = obs.shape[0], self.H, self.S, self.N, self.T, self.E, self.P, self.M, self.A
        (_, _, _, vn, vt, ve, nn, nnc, ne, nec, nt, ntc, tn, tnc, en, enc, overflow) = K.topo(obs, self.layout)
        glob, us, present, (wN, wE, wT, den) = self._first(obs)
        gb = torch.addmm(self.bi, glob, self.wigT)
        types = ("nodes", "tiles", "edges")
        x = [obs.new_empty(B * n, H) for n in (N, T, E)]
        seats, ZS = obs.new_empty(B * P, S), obs.new_empty(B * P, S)
        half = lambda *shape: torch.empty(*shape, dtype=torch.half, device=obs.device)
        sn, st, se = 4 * H + M - 2 * self.D, 2 * H + self.D, 2 * H + self.D
        XN, XT, XE = half(B * N, sn), half(B * T, st), half(B * E, se)
        Z, zs = [XN, XT, XE], [sn, st, se]
        valid, per_row = [vn, vt, ve], [N, T, E]
        none = [None] * 3
        K.entities16(x, none, us[:3], none, [gb.narrow(1, k * H, H) for k in range(3)], [3 * H + S] * 3,
                     [self.proj[b]["ln"][0] for b in types], [self.proj[b]["ln"][1] for b in types],
                     [self.proj[b]["ln"][2] for b in types], valid, Z, zs, *self._norms(0), per_row, overflow)
        KS.step(seats, us[3], None, None, gb.narrow(1, 3 * H, S), 3 * H + S, *self.proj["seats"]["ln"], present,
                ZS, *self.layers[0]["ln"]["seat"], P)
        # The heads' half inputs, written by the last layer (FusedGraph16.forward).
        nodes, tiles, XR = half(B * N, H), half(B * T, H), half(B * E, 2 * H)
        for i, (L, L16) in enumerate(zip(self.layers, self.layers16)):
            G1 = torch.mm(ZS, L["w1T"])
            K.gathers16([XN, XT, XE, XN, XN], [sn, st, se, sn, sn], [N, T, E, N, N],
                        [XN.narrow(1, H, H), XN.narrow(1, 2 * H, H), XN.narrow(1, 3 * H, H), XT.narrow(1, H, H),
                         XE.narrow(1, H, H)], [sn, sn, sn, st, se], [N, N, N, T, E],
                        [nn, nt, ne, tn, en], [nnc, ntc, nec, tnc, enc], [1] * 5, overflow)
            KS.scatter(G1.narrow(1, S, M), wN, wE, wT, XN, 4 * H, XE, 2 * H, XT, 2 * H, P, self.D, overflow)
            u = [torch.mm(XN, L16["wnT"]), torch.mm(XT, L16["wtT"]), torch.mm(XE, L16["weT"])]
            FB = obs.new_empty(B * P, M + A)
            KS.collect(XN, XE, XT, wN, wE, wT, den, L["wm"], present,
                       G1.narrow(1, S + M, 3 * A) if self.attention else ZS, L["bias"], self.scale, self.heads, A,
                       FB, P)
            G2 = torch.mm(FB, L["w2T"])
            bias = [L["bn"], L["bt"], L["be"]]
            if i < len(self.layers) - 1:
                K.entities16(x, u, none, bias, none, [0] * 3, none, none, [0.0] * 3, valid, Z, zs,
                             *self._norms(i + 1), per_row, overflow)
                KS.step(seats, G1, G2, L["bs"], None, 0, None, None, 0.0, present, ZS,
                        *self.layers[i + 1]["ln"]["seat"], P)
            else:
                K.entities16(x, u, none, bias, none, [0] * 3, none, none, [0.0] * 3, valid, [nodes, tiles, XR],
                             [H, H, 2 * H], none, none, [0.0] * 3, per_row, overflow)
                KS.step(seats, G1, G2, L["bs"], None, 0, None, None, 0.0, present, None, None, None, 0.0, P)
        players = seats.view(B, P, S)
        pooled = K.pool(*x, vn, vt, ve, torch.cat((glob, players[:, 0]), dim=1), N, T, E)
        logits, values = self._heads16(obs, nodes, tiles, XR, pooled, players, en, enc, overflow)
        return logits, values, overflow
