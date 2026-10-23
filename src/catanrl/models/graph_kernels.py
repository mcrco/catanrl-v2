"""The shared-trunk graph network's forward on custom CUDA kernels (inference only).

:class:`FusedGraph` computes what :class:`~catanrl.models.GraphModel` computes (masked
logits and ``sigmoid(value logits) * present``) from the same weights, restructured for a
GPU running small batches of whole boards:

* the row reader's feature remapping is folded into the first projections' weights, so
  the raw blocks are read once (the topology kernel copies them contiguous);
* the dense per-row incidence matrices become neighbour lists (a hex board's node has at
  most three neighbours, edges and tiles; a tile six corners) and each aggregation a
  gather; since the message weights are linear, each is applied after its mean, and a
  layer's four node inputs (self, nodes, tiles, edges) are one GEMM with stacked weights;
* LayerNorm, the residual, the padding mask and the neighbour means are one kernel each
  per layer for all three entity types; the matrix products stay in cuBLAS (``torch.mm``,
  whose bias-free GEMMs run about twice as fast here as ``addmm``'s at these shapes);
* the one- and two-output head layers (settlement, city, road, robber) are warp dot
  products in the kernel that assembles the logits.

On a GTX 1080, 64 search rows of a base board take 1.31 ms against 3.24 ms for the
PyTorch module (both CUDA graphs). The outputs differ from the module's in the last bits
(summation order: logits within 1e-3 at a scale of 1e3, values within 1e-6), so an agent
using the kernels keys them (``MctsParams.kernels``). Built on first use with
``torch.utils.cpp_extension`` (nvcc on PATH; cached under ``~/.cache/torch_extensions``).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

_KERNELS = None


def kernels():
    """The compiled extension (built on first use, then cached by torch)."""
    global _KERNELS
    if _KERNELS is None:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("graph_kernels.cu")
        try:
            _KERNELS = load(name="catanrl_graph_kernels", sources=[str(source)], extra_cuda_cflags=["-O3"],
                            verbose=False)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(f"could not build the graph kernels ({error}); they need nvcc on PATH "
                               "(or play with kernels=torch)") from error
    return _KERNELS


def _fold(weight: torch.Tensor, index: torch.Tensor, raw_width: int) -> torch.Tensor:
    """A weight on canonical features (the raw block padded with a zero column, then
    ``index_select``-ed) as a weight on the raw block: [out, raw_width]."""
    out = weight.new_zeros(weight.shape[0], raw_width + 1)
    out.index_add_(1, index, weight)
    return out[:, :raw_width]


class FusedGraph:
    """The forward of a shared-trunk :class:`~catanrl.models.GraphModel` (no aux head,
    hidden 96) on a CUDA device: ``forward(obs)`` is (logits, values, overflow), the last
    a [1] int32 tensor set when a node has more than three neighbours, edges or tiles (a
    board the neighbour lists cannot hold; :class:`~catanrl.search.evaluator.BatchEvaluator`
    raises on it)."""

    def __init__(self, model) -> None:
        if getattr(model, "spec", {}).get("class") != "graph" or not getattr(model, "shared", False) or model.aux:
            raise ValueError("kernels=fused runs shared-trunk graph networks without an aux head")
        K = kernels()
        trunk, policy, value = model.backbone, model.policy_head, model.value_head
        if trunk.hidden != K.hidden:
            raise ValueError(f"the graph kernels are built for hidden {K.hidden}, not {trunk.hidden}")
        reader = trunk.reader
        d = lambda t: t.detach()
        c = lambda t: t.detach().contiguous()
        idx = {b: getattr(reader, f"index_{b}") for b in reader.widths}
        raw = {b: reader.spans[b][1][-1] for b in reader.widths}
        canon = reader.widths
        sp = self.spans = reader.spans
        self.P, self.H = reader.num_players, trunk.hidden
        self.N, self.T, self.E = sp["nodes"][1][0], sp["tiles"][1][0], sp["edges"][1][0]
        valid = {b: int(idx[b][trunk.valid[b]]) for b in ("nodes", "tiles", "edges")}
        self.present = int(idx["players"][trunk.present])
        self.layout = [self.N, self.T, self.E, sp["nodes"][0], sp["tiles"][0], sp["edges"][0],
                       raw["nodes"], raw["tiles"], raw["edges"], valid["nodes"], valid["tiles"], valid["edges"],
                       sp["edge_nodes"][0], sp["tile_nodes"][0]]
        # The global projection reads [global | players] (adjacent in the row).
        if sp["players"][0] != sp["global"][0] + raw["global"]:
            raise ValueError("the kernels expect the players block right after the global block")
        wg = d(trunk.global_projection.weight)
        G, Fp = canon["global"], canon["players"]
        parts = [_fold(wg[:, :G], idx["global"], raw["global"])]
        parts += [_fold(wg[:, G + k * Fp: G + (k + 1) * Fp], idx["players"], raw["players"]) for k in range(self.P)]
        self.g_span = (sp["global"][0], sp["global"][0] + raw["global"] + self.P * raw["players"])
        self.wgT, self.bg = c(torch.cat(parts, dim=1).T), d(trunk.global_projection.bias)
        wp = d(trunk.player_projection.weight)
        self.wp = _fold(wp[:, :Fp], idx["players"], raw["players"])
        self.bp = d(trunk.player_projection.bias) + wp[:, Fp:].T[: self.P]  # the slot one-hot, per slot
        self.proj, wig, bi = {}, [], []
        for b in ("nodes", "tiles", "edges"):
            lin, inj = getattr(trunk, f"{b[:-1]}_projection"), getattr(trunk, f"{b[:-1]}_injection")
            wi = d(inj[0].weight)
            wig.append(wi[:, self.H:])
            bi.append(d(inj[0].bias))
            self.proj[b] = {"wT": c(_fold(d(lin.weight), idx[b], raw[b]).T), "b": d(lin.bias),
                            "wiT": c(wi[:, : self.H].T), "ln": (d(inj[1].weight), d(inj[1].bias), inj[1].eps)}
        self.wigT, self.bi = c(torch.cat(wig).T), torch.cat(bi)
        self.layers = []
        for layer in trunk.layers:
            w = lambda m: d(m.weight)
            self.layers.append({
                "ln": {k: (d(getattr(layer, f"{k}_norm").weight), d(getattr(layer, f"{k}_norm").bias),
                           getattr(layer, f"{k}_norm").eps) for k in ("node", "tile", "edge")},
                # node update = [n | mean of neighbour nodes | of tiles | of edges] @ stacked W
                "wnT": c(torch.cat([w(layer.node_self), w(layer.node_from_node), w(layer.node_from_tile),
                                    w(layer.node_from_edge)], dim=1).T), "bn": d(layer.node_self.bias),
                "wtT": c(torch.cat([w(layer.tile_self), w(layer.tile_from_node)], dim=1).T),
                "bt": d(layer.tile_self.bias),
                "weT": c(torch.cat([w(layer.edge_self), w(layer.edge_from_node)], dim=1).T),
                "be": d(layer.edge_self.bias),
            })
        lin = lambda m: (c(m.weight.T), d(m.bias))
        self.other = [lin(policy.other[i]) for i in (0, 2, 4)]
        self.sc1 = (c(torch.cat([policy.settlement[0].weight, policy.city[0].weight]).T),
                    torch.cat([policy.settlement[0].bias, policy.city[0].bias]).detach())
        self.sc_out = (c(policy.settlement[2].weight[0]), c(policy.city[2].weight[0]),
                       torch.cat([policy.settlement[2].bias, policy.city[2].bias]).detach())
        self.road = [lin(policy.road[i]) for i in (0, 2)]
        self.road_out = (c(policy.road[4].weight[0]), c(policy.road[4].bias))
        self.rt = lin(policy.robber_tile)
        self.rvT = c(policy.robber_victim.weight.T)
        self.robber_out = (c(policy.robber_output.weight[0]), c(policy.robber_output.bias))
        self.nobody, self.robber_slots = d(policy.nobody), policy.robber_slots
        self.value = [lin(value.mlp[i]) for i in (0, 2, 4)]
        self.vP = value.num_players
        self.alayout = [model.mask_offset, model.num_actions, policy.num_globals, self.N, self.T, self.E,
                        self.robber_slots, policy.robber_tile.weight.shape[0], self.vP,
                        sp["players"][0], raw["players"], self.present]

    @staticmethod
    def _mlp(x: torch.Tensor, layers) -> torch.Tensor:
        for i, (wT, b) in enumerate(layers):
            x = torch.addmm(b, x, wT)
            if i < len(layers) - 1:
                x = torch.relu_(x)
        return x

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = kernels()
        obs = obs.float().contiguous()
        B, H, N, T, E = obs.shape[0], self.H, self.N, self.T, self.E
        (xn, xt, xe, vn, vt, ve, nn, nnc, ne, nec, nt, ntc, tn, tnc, en, enc, overflow) = K.topo(obs, self.layout)
        glob = torch.relu_(torch.addmm(self.bg, obs[:, self.g_span[0]: self.g_span[1]], self.wgT))
        gb = torch.addmm(self.bi, glob, self.wigT)  # each type's injection: the glob part, bias in
        types = ("nodes", "tiles", "edges")
        us = [torch.mm(torch.relu_(torch.addmm(self.proj[b]["b"], r, self.proj[b]["wT"])), self.proj[b]["wiT"])
              for b, r in zip(types, (xn, xt, xe))]
        x = [obs.new_empty(B * n, H) for n in (N, T, E)]
        XN, XT, XE = obs.new_empty(B * N, 4 * H), obs.new_empty(B * T, 2 * H), obs.new_empty(B * E, 2 * H)
        Z, zs = [XN, XT, XE], [4 * H, 2 * H, 2 * H]
        valid, per_row = [vn, vt, ve], [N, T, E]
        names = ("node", "tile", "edge")

        def norms(i):
            ln = [self.layers[i]["ln"][k] for k in names]
            return [a[0] for a in ln], [a[1] for a in ln], [a[2] for a in ln]

        # Injection, then layer 0's LayerNorm into the first H columns of each GEMM input.
        K.entities(x, us, [None] * 3, [gb.narrow(1, k * H, H) for k in range(3)], [3 * H] * 3,
                   [self.proj[b]["ln"][0] for b in types], [self.proj[b]["ln"][1] for b in types],
                   [self.proj[b]["ln"][2] for b in types], valid, Z, zs, *norms(0), per_row)
        for i, L in enumerate(self.layers):
            # Neighbour means of the normalized embeddings into the inputs' other columns.
            K.gathers([XN, XT, XE, XN, XN], [4 * H, 2 * H, 2 * H, 4 * H, 4 * H], [N, T, E, N, N],
                      [XN.narrow(1, H, H), XN.narrow(1, 2 * H, H), XN.narrow(1, 3 * H, H), XT.narrow(1, H, H),
                       XE.narrow(1, H, H)], [4 * H, 4 * H, 4 * H, 2 * H, 2 * H], [N, N, N, T, E],
                      [nn, nt, ne, tn, en], [nnc, ntc, nec, tnc, enc], [1] * 5)
            u = [torch.mm(XN, L["wnT"]), torch.mm(XT, L["wtT"]), torch.mm(XE, L["weT"])]
            last = i == len(self.layers) - 1
            g, bt, eps = ([None] * 3, [None] * 3, [0.0] * 3) if last else norms(i + 1)
            K.entities(x, u, [L["bn"], L["bt"], L["be"]], [None] * 3, [0] * 3, [None] * 3, [None] * 3, [0.0] * 3,
                       valid, [None] * 3 if last else Z, zs, g, bt, eps, per_row)
        nodes, tiles, edges = x
        pooled = K.pool(nodes, tiles, edges, vn, vt, ve, glob, N, T, E)
        other = self._mlp(pooled, self.other)
        sch = torch.relu_(torch.addmm(self.sc1[1], nodes, self.sc1[0]))
        XR = obs.new_empty(B * E, 2 * H)  # the road head's input: [edge | sum of its two ends]
        XR[:, :H].copy_(edges)
        K.gathers([nodes], [H], [N], [XR.narrow(1, H, H)], [2 * H], [E], [en], [enc], [0])
        roadh = torch.relu_(self._mlp(XR, self.road))
        rt = torch.addmm(self.rt[1], tiles, self.rt[0])
        players_raw = obs[:, self.spans["players"][0]: self.spans["players"][0] + self.P * self.spans["players"][1][1]]
        players_raw = players_raw.reshape(B, self.P, -1)
        present = players_raw[..., self.present]
        players = torch.relu(F.linear(players_raw, self.wp) + self.bp) * present.unsqueeze(-1)
        victims = torch.cat((self.nobody.expand(B, 1, -1), players[:, 1: self.robber_slots]), 1)
        rv = torch.matmul(victims, self.rvT).contiguous()
        vin = torch.cat((pooled.unsqueeze(1).expand(-1, self.vP, -1), players[:, : self.vP]), -1)
        vl = self._mlp(vin.reshape(B * self.vP, -1), self.value)
        logits, values = K.assemble(obs, self.alayout, other.contiguous(), sch, *self.sc_out, roadh, *self.road_out,
                                    rt, rv, *self.robber_out, vl)
        return logits, values, overflow

