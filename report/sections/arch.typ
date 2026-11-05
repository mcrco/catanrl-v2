#import "../macros.typ": *

= Architecture <sec:arch>

Which network family should carry the policy and critic? Marco's MLP / LSTM / CNN / GNN ablation ranks graph ≈ LSTM > CNN > MLP at 2p. The graph net is also the smallest, and the only family that scales to any map, since it predicts board actions per node, edge and tile. Every other section uses `graph-s` (Canopy nexus-v3-style message passing, 541,540 parameters; @sec:recipe).

=== Ablation at 2p: `graph-s`, LSTM, CNN and MLP #chip("done") <exp:arch-formal>
#meta(
  dates: "2026-10-18 01:37–13:08 (cnn port built 2026-10-17)",
  owner: "marco",
  config: "configs/experiments/arch-ablation.yaml",
  wandb: (("arch-ablation-graph-s-7e2d7415", "graph-s"), ("arch-ablation-lstm-1c936a7f", "lstm"), ("arch-ablation-cnn-e3da640a", "cnn"), ("arch-ablation-mlp-8e9e7e63", "mlp")),
  records: ("039b60f0", "1cd2de87", "073da498", "76c88ebd", "189e5e8d", "8e2e23d8", "5abf6f0c"),
)
#tldr[One recipe and budget, 2p, against F: `graph-s` 90.7, `lstm` 90.3 (a coin flip head to head with `graph-s`), `cnn` 80.4, `mlp` 64.9. The graph net is also the smallest. One seed at 100M steps ranks the families but cannot separate graph from LSTM.]

*Question.* Which family is strongest under one recipe and budget?

*Setup.* The current recipe (@sec:recipe) at the `private` level, so the recurrent cell has hidden information to remember; one shared policy–critic trunk per cell. One seed, 100M agent steps (757 epochs): enough to separate families (`puffer-graph2` was at 86% against F by then). Besides `graph-s`:
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
- The CNN, with 31× the parameters and a third of the speed, tracks ten points lower and loses two games in three to the graph net.
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

== Network size

=== `graph-m` for 3B steps on OpenHex's 1v1 rules #chip("stopped") <exp:arch-graph-m-2p>
#meta(
  dates: "2026-11-04 02:10 – 11:10",
  owner: "marco",
  config: "configs/experiments/graph-m-2p.yaml",
  wandb: (("graph-m-2p-main-651bcbbc", "main"),),
  records: (),
)
#tldr[Is the 1v1 net too small? Every 1v1 net so far is `graph-s` (0.54M parameters), and it has leveled off: a second \~1B steps on OpenHex's rules moved it about a point (@exp:recipe-2p-openhex-3), while `hexn-4d.0` leads OpenHex 1v1 about 120 Elo above our best bot. `graph-m` from scratch at `graph-s`'s learning rate fell behind `graph-s` at the same steps and stalled (20.0% against 31.0% vs fr3 at epoch 1,400), with policy updates about 5× larger. Stopped at epoch 1,500 and restarted at half the learning rate (@exp:arch-graph-m-2p-lr).]

*Question.* Does a bigger network make a stronger 1v1 bot under the same recipe (Marco)?

#lead[Setup.]
- *Network.* `graph-m`: hidden 192, global 96, 4 layers, head 256 (`graph-s`: 96, 64, 3, 128), shared trunk.
- *Recipe.* `discount-2p`'s no-discount cell (@exp:recipe-discount) with only the model and budget changed: `base-2p-fr`, `full` level, γ = 1 for the policy and the lambda critic, seed 42, 3B agent steps (\~22,700 epochs) on one cosine from 0.005 to 0.0005.
- *Monitors.* Every 100 epochs on `select-2p-fr`, 1,000 games each: against F and against the deployed 1v1 net, `ppo-2p-fr3/full@e4350`.
- *Finals (planned, not run).* `@latest` on `heldout-2p-fr`, 5,000 games each: against F and against `ppo-2p-fr3/full@e4350`.
- *Cost.* 6.0k steps/s (`graph-s`: 17.7k) at a 4.26 GiB peak with micro-batches of 2,048 rows; 4,096 gave 6.3k at 6.89 GiB with allocator warnings. About 139 h of training plus 3 h of monitors for the full budget. The budget fixes the cosine, so a shorter run means a restart, not an early stop.

#lead[Result.] Behind `graph-s` at the same steps from epoch 600 on, and flat after that:

#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Epoch], [`graph-m` vs fr3], [`graph-s` vs fr3], [`graph-m` vs F], [`graph-s` vs F]),
    [200], [9.1], [9.1], [84.4], [83.7],
    [400], [15.9], [16.3], [87.6], [89.2],
    [600], [17.5], [21.3], [92.1], [92.5],
    [800], [19.5], [21.8], [91.1], [93.2],
    [1,000], [16.7], [*26.4*], [91.1], [*95.9*],
    [1,200], [19.9], [*26.9*], [92.9], [*94.4*],
    [1,400], [20.0], [*31.0*], [92.8], [*95.9*],
    [1,500], [18.6], [*30.4*], [90.2], [*95.4*],
  ),
  caption: [Monitor score rates (%) on `select-2p-fr`, 1,000 games each (±2.5 pp at 20%). `graph-s` is `discount-2p/g1-s42`; fr3 is `ppo-2p-fr3/full@e4350`.],
)

- *Paired* (same 1,000 games). At epoch 1,000, against fr3: −9.7 pp (106 against 203 discordant, p = 4e-8); against F: −4.8 pp (35 against 83, p = 1e-5). At epoch 1,400, against fr3: −11.0 pp (115 against 225, p = 3e-9); against F: −3.1 pp (35 against 66, p = 0.003).
- *Plateau.* Against fr3 between 17% and 20% from epoch 600 to 1,500, while `graph-s` went from 21.3% to 30.4%.
- *Update size.* The approximate KL per epoch rose from 0.03 to 0.13 by epoch 1,400. `graph-s` stayed at 0.02–0.03 over the same epochs, at the same entropy (0.17 at epoch 1,000) and about the same learning rate (0.0050 against 0.0043). So the bigger net took about 4–5× larger policy steps at the same learning rate. Its critic was slightly better (Brier 0.127 against 0.134 at epoch 1,000).

#lead[Reading.] `graph-s`'s learning rate is too high for `graph-m`: its policy changes too much per update and stops improving. That says nothing yet about capacity. Stopped at epoch 1,500 (11 h), and restarted from scratch at half the learning rate.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/graph-m-2p.yaml
```)

=== `graph-m` at half the learning rate #chip("running") <exp:arch-graph-m-2p-lr>
#meta(
  dates: "2026-11-04 11:10 –",
  owner: "marco",
  config: "configs/experiments/graph-m-2p-lr.yaml",
  wandb: (("graph-m-2p-lr-main-03f356b6", "main"),),
  records: (),
)
#tldr[@exp:arch-graph-m-2p restarted from scratch at half `graph-s`'s learning rate, to keep `graph-m`'s policy updates near `graph-s`'s size. Same 3B-step budget, about 6 days.]

*Question.* At a learning rate that suits it, does `graph-m` beat `graph-s` (Marco)?

#lead[Setup.] As @exp:arch-graph-m-2p, but with the cosine from 0.0025 to 0.00025. Monitors and finals are unchanged.

#lead[Check.] At epoch 1,000, compare against `graph-s` at the same steps (26.4% vs fr3, 95.9% vs F), along with the approximate KL (`graph-s`: about 0.02).

#lead[Progress at epoch 2,000 (264M steps, 12 h).] Level with `graph-s` at the same steps, slightly ahead on paper, and still improving:

#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    header: ([Epoch], [Half learning rate], [`graph-m` at 0.005], [`graph-s`]),
    tgroup(4)[Against fr3],
    [500], [23.1], [16.5], [19.7],
    [1,000], [26.1], [16.7], [26.4],
    [1,500], [31.2], [18.6], [30.4],
    [2,000], [35.8], [—], [33.2],
    tgroup(4)[Against F],
    [500], [93.9], [88.1], [91.8],
    [1,000], [94.5], [91.1], [95.9],
    [1,500], [96.7], [90.2], [95.4],
    [2,000], [96.3], [—], [95.2],
  ),
  caption: [Monitor score rates (%) on `select-2p-fr`, 1,000 games each. The first try stopped at epoch 1,500.],
)

- *Paired at epoch 2,000* (same 1,000 games). Against `graph-s`: +2.6 pp against fr3 (199 against 173 discordant, p = 0.19) and +1.1 pp against F (39 against 28, p = 0.22).
- *Pooled over the six monitors from epoch 1,500 to 2,000* (6,000 pairs): +0.5 pp against fr3 (p = 0.57) and +0.9 pp against F (p = 0.007). Each epoch replays the same 1,000 games, so the pooled p-values are optimistic.
- *Earlier.* At epoch 500 it led by 3.4 pp against fr3 (p = 0.06); at epoch 1,000 the two were level (−0.3 pp, p = 0.91). Against the first try at epoch 1,000: +9.4 pp against fr3 (p = 3e-8).
- *Still improving.* Against fr3 it went from 26.1% at epoch 1,000 to 35.8% at epoch 2,000 (p = 5e-7).
- *Update size.* The approximate KL per epoch stayed at 0.017–0.024. `graph-s` fell from 0.020 to 0.010 as its learning rate annealed. Entropy is 0.15, against 0.12 for `graph-s`.
- *Learning rate.* The two are now about equal (0.0025 against 0.0026 at epoch 2,000). From here `graph-s` anneals to 0.0005 by epoch 3,786, and over that stretch it went from 33% to 45% against fr3, while `graph-m` stays near 0.0024. Equal-step comparisons past this point mix network size with the schedule; the finals against fr3 decide.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/graph-m-2p-lr.yaml
```)
