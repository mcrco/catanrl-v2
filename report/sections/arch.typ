#import "../macros.typ": *

= Architecture <sec:arch>

Which network family should carry the policy and critic? Marco's informal v1 ablations pointed at Canopy's graph network: the fastest, most efficient, and only one that scales to any map, since it predicts board actions per node, edge and tile. His formal MLP / LSTM / CNN / GNN ablation agrees: graph ≈ LSTM > CNN > MLP at 2p. Every other section uses `graph-s` (Canopy nexus-v3-style message passing, 541,540 parameters; @sec:recipe).

== v1, informal

=== v1's informal ablations #chip("done") <exp:arch-v1-informal>
#meta(
  dates: "in v1 (catanrl, branch arch-ablation)",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[In v1 the graph net predicted held-out outcomes better than an MLP or a CNN, and under full RL the CNN trailed by 7–12 points against F at a quarter of the speed. A graph transformer tied message passing: the structured head matters, not the mixing layer.]

*Question.* Which family predicts Catan best, and at what cost?

*Setup.* Imitation loss favors the teacher's family (search targets carry its logit shape), so v1 predicted held-out games' outcomes instead (win/draw/loss cross-entropy, nats).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    header: ([Model], [Params], [Outcome loss (nats)]),
    [`graph-s` \ #sub[message passing]], [0.5M], [*0.396*],
    [`gtx-s` \ #sub[graph transformer over the same node and tile tokens]], [—], [0.397],
    [`mlp-512`], [2.4M], [0.464],
    [xdim CNN], [16.8M], [0.83],
  ),
  caption: [v1's outcome prediction on held-out games; lower is better, 0.693 is a coin flip.],
)

Under full RL (DAgger, then MARL PPO) the CNN was a 78.5M-parameter replica; `graph-m` tied `graph-s` (50.6% head to head) at 2.3× the cost per step. Step cost at batch 512: MLP 3 ms, `graph-s` 26 ms, CNN 120 ms.

*Reading.* The CNN predicted the winner worse than a coin flip; its value head memorized within a few hundred steps. The structured head means per-node settlement and city logits, per-edge road logits and per-tile robber logits. The graph net is launch-bound (about 200 small kernels per forward pass), which CUDA graphs fix at inference.

#repro(```sh
# Not in this repo: v1 (github.com/mcrco/catanrl), branch arch-ablation
```)

== v2, formal

=== Formal ablation at 2p: `graph-s`, LSTM, CNN and MLP #chip("done") <exp:arch-formal>
#meta(
  dates: "2026-10-18 01:37–13:08 (cnn port built 2026-10-17)",
  owner: "marco",
  config: "configs/experiments/arch-ablation.yaml",
  wandb: (("arch-ablation-graph-s-7e2d7415", "graph-s"), ("arch-ablation-lstm-1c936a7f", "lstm"), ("arch-ablation-cnn-e3da640a", "cnn"), ("arch-ablation-mlp-8e9e7e63", "mlp")),
  records: ("039b60f0", "1cd2de87", "073da498", "76c88ebd", "189e5e8d", "8e2e23d8", "5abf6f0c"),
)
#tldr[One recipe and budget, 2p, against F: `graph-s` 90.7, `lstm` 90.3 (a coin flip head to head with `graph-s`), `cnn` 80.4, `mlp` 64.9. The graph net is also the smallest. One seed at 100M steps ranks the families but cannot separate graph from LSTM.]

*Question.* Does v1's informal ranking hold under one controlled v2 recipe?

*Setup.* The current recipe (@sec:recipe) at the `private` level, so the recurrent cell has hidden information to remember; one shared policy–critic trunk per cell. One seed, 100M agent steps (757 epochs): enough to separate families, as `puffer-graph2` was at 86% against F by then and v1's gaps were 7–12 points. Besides `graph-s`:
/ `lstm`: `graph-s` plus an LSTM cell over its pooled embedding, stepped at the seat's own decisions (@sec:markov).
/ `cnn`: v1's xdim CNN on the v2 row (`src/catanrl/models/cnn.py`) at `xdim-default` size: 16.8M parameters, 15.1M in the projection after the convolutions. The blocks are gathered onto v1's 11 × 21 plane, laid out from the BASE map. Changes from v1: GroupNorm, not BatchNorm, so rollouts and evals run the same network (`norm: batch` restores v1); every block feature is a channel (47 at 2p, not 20 hand-picked planes); v2's structured heads. BASE map only (`hexagon` has another tile order and is refused).
/ `mlp`: A flat network over the padded row (`configs/models/mlp.yaml`), BASE dims only.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([Cell], [Steps/s], [Wall], [Best on `select-2p`], [Against F], [Against `graph-s`]),
    [`graph-s` \ #sub[0.54M params]], [17.6k], [1.6 h], [92.5 #sub[e750]], [*90.7* #ci[89.9][91.5] \ #sub[`039b60f0`]], [—],
    [`lstm` \ #sub[0.86M params]], [14.8k], [1.9 h], [92.0 #sub[e700]], [*90.3* #ci[89.4][91.1] \ #sub[`1cd2de87`]], [*51.0* #ci[49.7][52.4] \ #sub[`073da498`]],
    [`cnn` \ #sub[16.8M params]], [5.6k], [7.5 h], [81.5 #sub[e700]], [80.4 #ci[79.3][81.5] \ #sub[`76c88ebd`]], [33.7 #ci[32.4][35.0] \ #sub[`189e5e8d`]],
    [`mlp` \ #sub[2.2M params]], [108k], [0.3 h], [68.0 #sub[e700]], [64.9 #ci[63.6][66.2] \ #sub[`8e2e23d8`]], [18.1 #ci[17.0][19.2] \ #sub[`5abf6f0c`]],
  ),
  caption: [`heldout-2p`, 5,000 games per score, record under each. "Against `graph-s`" plays that cell's best checkpoint; wall time includes select evals.],
) <tab:ablation>

#figure(
  image("../figures/arch-ablation.png", width: 86%),
  caption: [Selection curves: argmax policy against F on `select-2p`, every 50 epochs.],
) <fig:ablation>

#lead[Reading.]
- The LSTM matches the graph net from epoch 300 on, at 1.6× the parameters and 0.85× the throughput: at this budget, memory over the seat's own decisions adds nothing the row lacks (at 4p it does, @sec:markov).
- The CNN, with 31× the parameters and a third of the speed, tracks ten points lower and loses two games in three to the graph net; its cost is why v1 abandoned it.
- The MLP never catches up (26 points behind, four losses in five): a cheap baseline only.

#repro(```sh
catanrl run configs/experiments/arch-ablation.yaml
# One cell: add --only graph-s (or mlp, cnn, lstm).
# Head-to-head finals need graph-s's best checkpoint, so the runner trains graph-s first.
```)

=== Graph vs LSTM at the full budget, with more seeds #chip("todo") <exp:arch-full-budget>
#meta(
  owner: "assistant",
  config: "configs/experiments/arch-lineage.yaml",
  wandb: (),
  records: (),
)
#tldr[Separating `graph-s` from `lstm` needs the full 500M-step budget and more seeds (Claude's suggestion, not queued).]

*Status.* Nothing run. The closest config, `configs/experiments/arch-lineage.yaml`, sweeps `graph-s`, `graph-m` and `mlp` over two seeds at 400M steps (6.5 h a cell; 12 cells, about a week of GPU). It ports v1's lineage script (`full` level, γ = 0.99, then γ = 1 from epoch 3000), not the current recipe, and has no `lstm` cell.
