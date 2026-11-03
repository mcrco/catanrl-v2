#import "../macros.typ": *

= Architecture <sec:arch>

Which network family should carry the policy and the critic? Marco's informal v1 ablations pointed at Canopy's graph network: the fastest, the most efficient, and the only one that scales to any map, because it predicts board actions per node, edge and tile. He still wanted a formal MLP / LSTM / CNN / GNN ablation, and it agrees: graph ≈ LSTM > CNN > MLP at 2p, and the structured head is what matters. Every other section uses the graph net `graph-s` (Canopy nexus-v3-style message passing, 541,540 parameters; details in @sec:recipe).

== v1, informal

=== v1's informal ablations #chip("done") <exp:arch-v1-informal>
#meta(
  dates: "in v1 (catanrl, branch arch-ablation)",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[In v1 the graph net predicted held-out game outcomes better than an MLP or a CNN, and under full RL the CNN trailed it by 7–12 points against F at a quarter of the speed. A graph transformer tied message passing, so the structured head matters, not the mixing layer.]

*Question.* Which family predicts Catan best, and at what cost?

*Setup.* Imitation loss cannot compare architectures: search targets carry the teacher's logit shape, so the loss favors the teacher's family. v1 used a teacher-neutral target instead: predict the terminal outcome of held-out games (win/draw/loss cross-entropy in nats; 0.693 is a coin flip).

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
  caption: [v1's teacher-neutral outcome prediction on held-out games; lower is better, and 0.693 is a coin flip.],
)

Under the full RL recipe (DAgger, then MARL PPO), a 78.5M-parameter CNN replica trailed `graph-s` by 7–12 points against F at a quarter of the training speed. `graph-m` tied `graph-s` head to head (50.6%) at 2.3× the cost per step. Step cost at batch 512: MLP 3 ms, `graph-s` 26 ms, CNN 120 ms.

*Reading.* The CNN predicted the winner worse than a coin flip, and its value head memorized within a few hundred steps. The graph transformer tying message passing says what matters is the structured head (per-node settlement and city logits, per-edge road logits, per-tile robber logits). The graph net is launch-bound (about 200 small kernels per forward pass), which CUDA graphs fix at inference.

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
#tldr[Four architectures, one recipe, one budget, 2p: against F, `graph-s` scores 90.7, `lstm` 90.3 (and a coin flip head to head with `graph-s`), `cnn` 80.4 and `mlp` 64.9. The ranking is graph ≈ LSTM > CNN > MLP, and the graph net is also the smallest of the four. One seed at 100M steps ranks the families but cannot separate graph from LSTM.]

*Question.* Does v1's informal ranking hold under one controlled v2 recipe?

*Setup.* The current recipe (@sec:recipe) at the fair `private` level, so the recurrent cell has hidden information to remember. Every cell has one shared trunk for policy and critic.
- *Budget:* one seed, 100M agent steps per architecture (757 epochs). That is enough to separate families: `puffer-graph2` was at 86% against F by then, and v1's gaps were 7–12 points.
- *Eval:* checkpoints selected on `select-2p` every 50 epochs; reported on `heldout-2p` against F, and head to head against the `graph-s` cell's best checkpoint on the same boards.

The four cells:
/ `graph-s`: The recipe's network.
/ `lstm`: The `graph-s` trunk plus an LSTM cell over its pooled embedding, stepped at each of the seat's own decisions (@sec:markov has the build).
/ `cnn`: v1's xdim CNN ported to the v2 row (`src/catanrl/models/cnn.py`), at v1's `xdim-default` size: 16.8M parameters per network, 15.1M of them in the projection after the convolutions. The tile, node and edge blocks are gathered onto v1's 11 × 21 plane, laid out once from the BASE map. Three deviations from v1: GroupNorm instead of BatchNorm, so rollouts and evals run the same network (`norm: batch` restores v1); every block feature is a channel (47 at 2p) instead of v1's 20 hand-picked planes; and v2's structured heads. BASE map only: the `hexagon` generator has BASE's dims but another tile order, and is refused at launch.
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
  caption: [The formal ablation on `heldout-2p`, 5,000 games per score, with each score's record under it. "Against `graph-s`" is the same argmax policy against the `graph-s` cell's best checkpoint on the same boards. Wall time includes the select evals. The LSTM against `graph-s` at 4p is in @sec:markov; the `graph-s` cell with a hidden-feature head is in @sec:hidden.],
) <tab:ablation>

#figure(
  image("../figures/arch-ablation.png", width: 86%),
  caption: [The four cells' selection curves (argmax policy against F on `select-2p`, every 50 epochs). The graph net and the LSTM are indistinguishable from epoch 300 on; the CNN tracks ten points lower; the MLP never catches up.],
) <fig:ablation>

#lead[Reading.]
- The LSTM ties the graph net head to head (51.0, a coin flip) at 1.6× the parameters and 0.85× the throughput: at this budget, memory over the seat's own decisions adds nothing the current row lacks (at 4p it does, @sec:markov).
- The CNN has 31× the parameters and a third of the speed; it is ten points behind against F and loses two games in three to the graph net. Its 7.5 h is why v1 abandoned it.
- The MLP is 26 points behind against F and loses four games in five; at 0.3 h it is the cheap baseline.

#repro(```sh
catanrl run configs/experiments/arch-ablation.yaml
# One cell: add --only graph-s (or mlp, cnn, lstm).
# The head-to-head finals of mlp, cnn and lstm need graph-s's best checkpoint, so the runner trains graph-s first.
```)

=== Graph vs LSTM at the full budget, with more seeds #chip("todo") <exp:arch-full-budget>
#meta(
  owner: "assistant",
  config: "configs/experiments/arch-lineage.yaml",
  wandb: (),
  records: (),
)
#tldr[One seed at 100M steps cannot separate `graph-s` from the `lstm`; that needs the full 500M-step budget and more seeds (Claude's suggestion, not queued).]

*Status.* Nothing run. The closest config, `configs/experiments/arch-lineage.yaml`, sweeps `graph-s`, `graph-m` and `mlp` over two seeds at 400M steps: 6.5 h a cell, so its 12 cells are about a week of GPU. It ports v1's lineage launch script (`full` level, γ = 0.99, then a γ = 1 stage from epoch 3000) rather than the current recipe, and it has no `lstm` cell.
