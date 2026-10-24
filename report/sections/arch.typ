#import "../macros.typ": *

= architecture <sec:arch>

which network family should carry the policy and the critic? marco's informal v1 ablations pointed at canopy's graph network: the fastest, the most efficient, and the only one that scales to any map, because it predicts board actions per node, edge and tile. he still wanted a formal MLP / LSTM / CNN / GNN ablation, and it agrees: graph ≈ lstm > cnn > mlp at 2p, and the structured head is what matters. every other section uses the graph net `graph-s` (canopy nexus-v3-style message passing, 541,540 parameters; details in @sec:recipe).

== v1, informal

=== v1’s informal ablations #chip("done") <exp:arch-v1-informal>
#meta(
  dates: "in v1 (catanrl, branch arch-ablation)",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[in v1 the graph net predicted held-out game outcomes better than an MLP or a CNN, and under full RL the CNN trailed it by 7–12 points vs F at a quarter of the speed. a graph transformer tied message passing, so the structured head matters, not the mixing layer.]

*question.* which family predicts catan best, and at what cost?

*setup.* imitation loss can't compare architectures: search targets carry the teacher's logit shape, so the loss favours the teacher's family. v1 used a teacher-neutral target instead: predict the terminal outcome of held-out games (win/draw/loss cross-entropy in nats; 0.693 is a coin flip).

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    header: ([model], [params], [outcome loss (nats)]),
    [`graph-s` (message passing)], [0.5M], [*0.396*],
    [`gtx-s` (graph transformer over the same node and tile tokens)], [—], [0.397],
    [`mlp-512`], [2.4M], [0.464],
    [xdim CNN], [16.8M], [0.83],
  ),
  caption: [v1’s teacher-neutral outcome prediction on held-out games; lower is better.],
)
under the full RL recipe (DAgger, then MARL PPO) a 78.5M-parameter CNN replica trailed `graph-s` by 7–12 points vs F at a quarter of the training speed. `graph-m` tied `graph-s` head to head (50.6%) at 2.3× the cost per step. step cost at batch 512: MLP 3 ms, `graph-s` 26 ms, CNN 120 ms.

*reading.* the CNN predicted the winner worse than a coin flip, and its value head memorized within a few hundred steps. the graph transformer tying message passing says what matters is the structured head (per-node settlement and city logits, per-edge road logits, per-tile robber logits). the graph net is launch-bound (about 200 small kernels per forward), which CUDA graphs fix at inference.

#repro(```sh
# not in this repo: v1 (github.com/mcrco/catanrl), branch arch-ablation
```)

== v2, formal

=== formal ablation at 2p: graph-s, lstm, cnn, mlp #chip("done") <exp:arch-formal>
#meta(
  dates: "2026-10-18 01:37–13:08 (cnn port built 2026-10-17)",
  owner: "marco",
  config: "configs/experiments/arch-ablation.yaml",
  wandb: (("arch-ablation-graph-s-7e2d7415", "graph-s"), ("arch-ablation-lstm-1c936a7f", "lstm"), ("arch-ablation-cnn-e3da640a", "cnn"), ("arch-ablation-mlp-8e9e7e63", "mlp")),
  records: ("039b60f0", "1cd2de87", "073da498", "76c88ebd", "189e5e8d", "8e2e23d8", "5abf6f0c"),
)
#tldr[four architectures, one recipe, one budget, 2p: `graph-s` 90.7 vs F, `lstm` 90.3 (and a coin flip head to head with `graph-s`), `cnn` 80.4, `mlp` 64.9. the ranking is graph ≈ lstm > cnn > mlp, and the graph net is also the smallest of the four. one seed at 100M steps ranks the families but can't separate graph from lstm.]

*question.* does v1’s informal ranking hold under one controlled v2 recipe?

*setup.* the current recipe (@sec:recipe) at the fair `private` level, so the recurrent cell has hidden information to remember. one seed, 100M agent steps per architecture (757 epochs). checkpoints selected on `select-2p` every 50 epochs; reported on `heldout-2p` vs F and head to head against the `graph-s` cell's best checkpoint on the same boards. 100M steps is enough to separate families: `puffer-graph2` was at 86% vs F by then, and v1’s gaps were 7–12 points. every cell has one shared trunk for policy and critic.
- `graph-s`: the recipe's network.
- `lstm`: the `graph-s` trunk plus an LSTM cell over its pooled embedding, stepped at each of the seat's own decisions (@sec:markov has the build).
- `cnn`: v1’s xdim CNN ported to the v2 row (`src/catanrl/models/cnn.py`). v1’s `xdim-default` size: 16.8M parameters per network, 15.1M of them in the projection after the convolutions. the tile, node and edge blocks are gathered onto v1’s 11 × 21 plane, laid out once from the BASE map. three deviations from v1: GroupNorm instead of BatchNorm, so rollouts and evals run the same network (`norm: batch` restores v1); every block feature is a channel (47 at 2p) instead of v1’s 20 hand-picked planes; and v2’s structured heads. BASE map only: the `hexagon` generator has BASE's dims but another tile order, and is refused at launch.
- `mlp`: a flat network over the padded row (`configs/models/mlp.yaml`), BASE dims only.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, left),
    header: ([arch], [params], [steps/s], [wall], [select-2p (best)], [vs F], [vs graph-s]),
    [`graph-s`], [0.54M], [17.6k], [1.6 h], [92.5 (e750)], [*90.7* #ci[89.9][91.5]], [—],
    [`lstm`], [0.86M], [14.8k], [1.9 h], [92.0 (e700)], [*90.3* #ci[89.4][91.1]], [*51.0* #ci[49.7][52.4]],
    [`cnn`], [16.8M], [5.6k], [7.5 h], [81.5 (e700)], [80.4 #ci[79.3][81.5]], [33.7 #ci[32.4][35.0]],
    [`mlp`], [2.2M], [108k], [0.3 h], [68.0 (e700)], [64.9 #ci[63.6][66.2]], [18.1 #ci[17.0][19.2]],
  ),
  caption: [the formal ablation on `heldout-2p`, 5,000 games per number. "vs F" is the argmax policy against F; "vs graph-s" is the same policy against the `graph-s` cell's best checkpoint on the same boards. wall time includes the select evals. records: `graph-s` `039b60f0`; `lstm` `1cd2de87` / `073da498`; `cnn` `76c88ebd` / `189e5e8d`; `mlp` `8e2e23d8` / `5abf6f0c`. the same table at 4p, and the `graph-s` cell with a hidden-feature head, are in @sec:markov and @sec:hidden.],
) <tab:ablation>

#figure(
  image("../figures/arch-ablation.png", width: 86%),
  caption: [the four cells' selection curves (argmax policy vs F on `select-2p`, every 50 epochs). the graph net and the lstm are indistinguishable from epoch 300 on; the cnn tracks ten points lower; the mlp never catches up.],
) <fig:ablation>

*reading.* graph ≈ lstm > cnn > mlp, and the structured head is what matters. the lstm ties the graph net head to head (51.0, a coin flip) at 1.6× the parameters and 0.85× the throughput: at this budget, memory over the seat's own decisions adds nothing the current row lacks (at 4p it does, @sec:markov). the cnn has 31× the parameters and a third of the speed; it is ten points behind vs F and loses two games in three to the graph net. the mlp is 26 points behind and loses four in five. at 0.3 h the mlp is the cheap baseline; the cnn's 7.5 h is why v1 abandoned it.

#repro(```sh
catanrl run configs/experiments/arch-ablation.yaml
# one cell: add --only graph-s (or mlp, cnn, lstm).
# the head-to-head finals of mlp, cnn and lstm need graph-s's best checkpoint, so the runner trains graph-s first.
```)

=== graph vs lstm at the full budget, with more seeds #chip("todo") <exp:arch-full-budget>
#meta(
  owner: "assistant",
  config: "configs/experiments/arch-lineage.yaml",
  wandb: (),
  records: (),
)
#tldr[one seed at 100M steps can't separate `graph-s` from the `lstm`; that needs the full 500M-step budget and more seeds. (my suggestion, not queued.)]

*status.* nothing run. the closest config, `configs/experiments/arch-lineage.yaml`, sweeps `graph-s`, `graph-m` and `mlp` over two seeds at 400M steps: 6.5 h a cell, so its 12 cells are about a week of GPU. it ports v1’s lineage launch script (`full` level, γ = 0.99, then a γ = 1 stage from epoch 3000) rather than the current recipe, and it has no `lstm` cell.
