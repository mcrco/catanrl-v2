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
#tldr[Is the 1v1 net too small? Every 1v1 net so far is `graph-s` (0.54M parameters), and it has leveled off: a second \~1B steps on OpenHex's rules moved it about a point (@exp:recipe-2p-openhex-3), while `hexn-4d.0` leads OpenHex 1v1 about 120 Elo above our best bot. `graph-m` from scratch at `graph-s`'s learning rate fell behind `graph-s` at the same steps and stalled (20.0% against 31.0% vs fr3 at epoch 1,400), with policy updates about 5× larger. Stopped at epoch 1,500 and restarted at half the learning rate, which beat fr3 56.5% (@exp:arch-graph-m-2p-lr).]

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

=== `graph-m` at half the learning rate #chip("done") <exp:arch-graph-m-2p-lr>
#meta(
  dates: "2026-11-04 11:10 – 2026-11-10 06:16",
  owner: "marco",
  config: "configs/experiments/graph-m-2p-lr.yaml",
  wandb: (("graph-m-2p-lr-main-03f356b6", "main"),),
  records: ("914fc0f0", "d664995b"),
)
#tldr[@exp:arch-graph-m-2p restarted from scratch at half `graph-s`'s learning rate. After 3B steps (139 h) `graph-m` beats fr3, the deployed 1v1 net, 56.5% over 5,000 held-out games (p = 3e-20), and F 99.4%. `graph-s` from scratch under the same recipe ended at 44.2% against fr3 on the same games (+12.3 pp paired). Now `2p-graph-m-1` and the 1v1 yardstick, *best-2p* (Marco, 10 November).]

*Question.* At a learning rate that suits it, does `graph-m` beat `graph-s` (Marco)?

#lead[Setup.] As @exp:arch-graph-m-2p, but with the cosine from 0.0025 to 0.00025. Monitors and finals are unchanged.

#lead[Check.] At epoch 1,000, compare against `graph-s` at the same steps (26.4% vs fr3, 95.9% vs F), along with the approximate KL (`graph-s`: about 0.02).

#lead[Result.] Finals on `heldout-2p-fr` (5,000 games, both seat orders) for the last checkpoint, epoch 22,711 (3.0B steps), now `2p-graph-m-1`:
- *Against fr3*: 56.5% #ci[55.1][57.9] (`914fc0f0`). `graph-s`'s end under the same recipe (`discount-2p/g1-s42@e3786`, `2p-graph-s-4`) took 44.2% on the same games: +12.3 pp paired (1,353 against 737 discordant, p = 2e-41).
- *Against F*: 99.4% #ci[99.2][99.6] (`d664995b`), +0.8 pp over `graph-s`'s end (68 against 27, p = 3e-5) and +0.5 pp over fr3 (51 against 26, p = 0.006).
- *Caveat.* Both rows are `POLICY` at the `full` level, so both sides read every hand. The fair raw bot (`BELIEF`) and search rows are not run yet.

#lead[Reading.] fr3 (`graph-s`, 2.02B steps over three runs) had leveled off: its second \~1B steps moved it about a point (@exp:recipe-2p-openhex-3). `graph-m` from scratch passes it by 6.5 points at 3.0B steps. At equal steps it trailed `graph-s` while `graph-s` annealed (below), so the size only pays with the longer budget.

#lead[Monitors.] The last ten (epochs 21,800 to 22,700) average 57.8% against fr3, 13.1 pp above `graph-s`'s end, and 99.2% against F:

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
    [3,000], [37.1], [—], [42.5],
    [4,000], [41.5], [—], [44.7],
    [5,000], [43.6], [—], [—],
    [6,000], [42.6], [—], [—],
    [7,000], [43.2], [—], [—],
    [8,000], [45.9], [—], [—],
    [9,000], [44.8], [—], [—],
    [10,000], [44.7], [—], [—],
    [11,000], [47.2], [—], [—],
    [12,000], [47.6], [—], [—],
    [13,000], [51.3], [—], [—],
    [14,000], [51.9], [—], [—],
    [15,000], [51.7], [—], [—],
    [16,000], [51.7], [—], [—],
    [17,000], [53.0], [—], [—],
    [18,000], [52.1], [—], [—],
    [19,000], [55.4], [—], [—],
    [20,000], [54.8], [—], [—],
    [21,000], [55.4], [—], [—],
    [22,000], [58.5], [—], [—],
    [22,700], [56.6], [—], [—],
    tgroup(4)[Against F],
    [500], [93.9], [88.1], [91.8],
    [1,000], [94.5], [91.1], [95.9],
    [1,500], [96.7], [90.2], [95.4],
    [2,000], [96.3], [—], [95.2],
    [3,000], [97.1], [—], [97.5],
    [4,000], [98.0], [—], [98.8],
    [5,000], [97.2], [—], [—],
    [6,000], [98.7], [—], [—],
    [7,000], [98.5], [—], [—],
    [8,000], [98.6], [—], [—],
    [9,000], [98.9], [—], [—],
    [10,000], [98.4], [—], [—],
    [11,000], [98.1], [—], [—],
    [12,000], [98.8], [—], [—],
    [13,000], [98.1], [—], [—],
    [14,000], [99.2], [—], [—],
    [15,000], [98.8], [—], [—],
    [16,000], [99.0], [—], [—],
    [17,000], [99.1], [—], [—],
    [18,000], [99.0], [—], [—],
    [19,000], [99.1], [—], [—],
    [20,000], [99.1], [—], [—],
    [21,000], [99.0], [—], [—],
    [22,000], [99.2], [—], [—],
    [22,700], [99.3], [—], [—],
  ),
  caption: [Monitor score rates (%) on `select-2p-fr`, 1,000 games each. The first try stopped at epoch 1,500. `graph-s` ended at epoch 3,786, so its 4,000 row is its last monitor (epoch 3,750).],
)

- *Against `graph-s`'s end* (epoch 3,750, same 1,000 games), at epoch 22,700: +11.9 pp against fr3 (254 against 135 discordant, p = 2e-9) and +0.5 pp against F (10 against 5, p = 0.30). `graph-s`'s end scored 44.7% and 98.8%.
- *Gains per 1,000 epochs against fr3*, each the last ten monitors against the ten before (10,000 pairs): +0.7 pp (epoch 7,000, p = 0.31), +1.3 (8,000, p = 0.03), +1.8 (9,000, p = 0.005), +0.4 (10,000, p = 0.51), +1.5 (11,000, p = 0.02), +0.4 (12,000, p = 0.50), +1.7 (13,000, p = 0.006), +2.3 (14,000, p = 2e-4), −1.1 (15,000, p = 0.07), +1.0 (16,000, p = 0.09), +1.6 (17,000, p = 0.007), +0.7 (18,000, p = 0.26), +0.9 (19,000, p = 0.12), +0.4 (20,000, p = 0.50), +0.0 (21,000, p = 0.93), +1.7 (22,000, p = 0.002), and +1.3 for the last ten against the ten before (22,700, p = 0.03). Against F level since epoch 8,000. Pooled p-values are optimistic, since each epoch replays the same games.
- *Earlier, paired at equal steps.* Behind from epoch 2,500 on (−3.9 pp against fr3 pooled over epochs 3,100 to 3,700, p = 1e-7), while `graph-s` annealed. Level at epochs 2,000 (+2.6 pp, p = 0.19) and 1,000 (−0.3 pp, p = 0.91); ahead at epoch 500 (+3.4 pp, p = 0.06). Against the first try at epoch 1,000: +9.4 pp against fr3 (p = 3e-8).
- *Learning rate.* The cosine ended at its floor, 0.00025. As for `graph-s` (+9 pp against fr3 from epoch 2,000 to 3,000, 0.0026 to 0.00096), gains kept coming as it fell, including at the floor.
- *Update size.* Over the last 500 epochs the approximate KL per epoch had a median of 0.001 (largest 0.004), with a clip fraction of 0.007; two one-epoch outliers (0.19 at epoch 12,887, 0.14 at 17,541) had normal clip fractions, and the next monitors held. Entropy ended at 0.04. Speed was about 6.1k steps/s, with a 5.0 GiB peak.
- *Eval cost.* The two monitors take about 36 s every 100 epochs (22 s against fr3, 14 s against F), about 1.7% of the wall time.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/graph-m-2p-lr.yaml
```)

=== `graph-m` in the search against the raw fr3 bot #chip("done") <exp:arch-graph-m-search>
#meta(
  dates: "2026-11-16 20:23 – 2026-11-17 01:23",
  owner: "marco",
  config: "configs/experiments/graph-m-search-vs-raw.yaml",
  wandb: (("graph-m-search-vs-raw-main-42f8b15b", "graph-m-search-vs-raw/main"),),
  records: ("93649969",),
)
#tldr[Marco (2026-11-16): `graph-m` in the search against raw fr3, on the games the AlphaZero checkpoints are judged on. `2p-graph-m-1` in a 300-sim search scores 58.10% over 2,000 games: 4.25 pp above fr3's net in its 1,024-sim search (p = 0.002), level with e300, 4.85 pp below e1800 (p = 0.0006) and 5.60 pp below the bundle (p = 6e-5). At equal time per move the best raw policy is not the best net for the search, so the fr3 AlphaZero line stays the 1v1 search bot.]

*Question.* Where does `2p-graph-m-1`, a PPO net with no self-play with search, sit among fr3's net (53.85%), e300 (57.70%), e900 (60.30%) and e1800 (62.95%) when each plays in the search against the fair raw fr3 bot?

#lead[Setup.]
- *Row.* `graph-m` in the search against `BELIEF` of fr3 with 16 worlds, on the first 1,000 boards of `heldout-2p-fr` in both seat orders: the same 2,000 games as the AlphaZero rows, so it pairs with each of them by game.
- *Search.* 300 simulations, the equal-time budget of @exp:deploy-graph-m-check (1,024 / 3.38; no fused kernels for `graph-m` yet). The other options are the rows' own.
- *Sample.* 2,000 games: ±2.2 pp. Against e1800's row, paired, about 3.4 pp is detectable at 80% power.
- *Prediction* (Claude, by chaining rows): about 58% ± 3 pp. `graph-m`'s search beat fr3's search 54.3% at equal time, and fr3's search scores 53.85% here.
- *Cost.* 5.0 h on the GPU (6.7 games a minute); the estimate was 3–4 h.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Net in the search], [Sims], [Score against raw fr3, %], [`graph-m` minus it, pp], [p]),
    [fr3 (`809619df`)], [1,024], [53.85], [+4.25 #ci[+1.6][+6.9]], [0.002],
    [e300 (`e917f9bb`)], [1,024], [57.70], [+0.40 #ci[−2.3][+3.1]], [0.80],
    [`2p-graph-m-1` (`93649969`)], [300], [58.10 #ci[55.9][60.3]], [], [],
    [e1800 (`ea257443`)], [1,024], [62.95], [−4.85 #ci[−7.6][−2.1]], [0.0006],
    [The bundle (`b5483151`)], [1,024], [63.70], [−5.60 #ci[−8.3][−2.9]], [6e-5],
  ),
  caption: [Each net in the deployed search against `BELIEF` of fr3 on the same 2,000 held-out games. Differences are paired by game (exact McNemar). e900 (60.30%) is on the same games and was not paired here.],
) <tab:arch-graph-m-search>

*Reading.*
- *The prediction held* (about 58% ± 3 pp by chaining rows).
- *Where it sits.* At equal time per move, `graph-m`'s stronger raw policy is worth to the search about what the first 300 iterations of self-play added to fr3's net. The 1,500 iterations after that added 5 pp more, which `graph-m` does not have.
- *For the bot.* The fr3 AlphaZero line stays the 1v1 search bot. `2p-graph-m-1` stays the best raw policy.
- *Not answered.* What `graph-m` would reach after its own self-play (since answered: 67.0% at 1,024 sims after 900 iterations, @exp:az-graph-m), and its score at 1,024 sims, which would cost 3.4× the time per move without fused kernels.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/graph-m-search-vs-raw.yaml
```)

=== `graph-m` in the AlphaZero line #chip("done") <exp:az-graph-m>
#meta(
  dates: "2026-11-15 (proposed), 2026-11-18 16:24 (launched), 18:25 (stopped), 18:44 (relaunched), 2026-11-20 06:52 (done)",
  owner: "marco",
  config: "configs/experiments/az-2p-graph-m-uniform.yaml (first launch: az-2p-graph-m.yaml)",
  wandb: (("az-2p-graph-m-uniform-main-a2ac6eca", "az-2p-graph-m-uniform/main"), ("az-2p-graph-m-main-c5ebe625", "az-2p-graph-m/main (collapsed)")),
  records: ("9c7a904b", "8504808e", "ce1a2189", "ce298288", "5824eca5", "cd8ee47e", "af97f759", "b7f7ed96", "0d49a45d"),
)
#tldr[*`2p-graph-m-2`: 67.00% in the 1,024-sim search against fr3's fair raw bot, the best row of the line: +2.35 pp over `2p-graph-s-8` (p = 0.086), +4.05 pp over the deployed net (p = 0.004), +9.75 pp over its own start (p = 1e-12), after 900 iterations (32.3 h) on the local GPU.* The first launch collapsed and was stopped after 2 h; this is the relaunch without the surprise weighting. Marco (2026-11-18): continue search training, or try it on `graph-m`? The `graph-s` line has flattened (+1.70 pp over its last 1,800 iterations, p = 0.20; @exp:az-tricks), so the recipe moves to `graph-m` (Marco: "ok yeah lets do it"). *Where it starts:* `2p-graph-m-1` in the 1,024-sim search scores 57.25% against fr3's fair raw bot, 7.40 pp behind `2p-graph-s-8` on the same games and no better than at 300 sims. *First launch:* once its learner started, the net's raw policy fell to 25% against its own start (the `graph-s` runs stay at 49 to 50%). *Cause, found offline:* the surprise weighting. On this peaked net 1% of the replay took 27% of the draws; without the weighting the same three learning iterations leave the net level with its start (50.1%). Released for OpenHex as search-v4 (@exp:deploy-openhex-search-v4).]

#lead[Earlier (2026-11-15).] Marco: can the dice state go into `graph-m`? Yes: `graph-m` is the same network class as `graph-s` at other sizes, so the inputs of @exp:dice-input and the targets of @exp:az-aux attach unchanged, and `2p-graph-m-1` can take them with zero weights and play as before. What it lacked was speed: the fused kernels were written for `graph-s`'s width. Marco (2026-11-15): build the kernels; the run itself is not queued.

- *Sizes.* `graph-s`: hidden 96, 3 layers, 0.54M parameters. `graph-m`: hidden 192, 4 layers, 2.32M parameters (4.3×).
- *Is Catan that complicated?* (Marco's doubt.) For small: HexN's 332k-parameter net with rollouts is rated above our search bot. For bigger: under one recipe `graph-m` ended 12.3 pp above `graph-s` against fr3 (@exp:arch-graph-m-2p-lr), the largest single gain measured here for a raw policy. Whether size still pays after self-play with search is not measured.
- *Why.* `graph-m` beats `graph-s` as a raw policy (@exp:arch-graph-m-2p-lr), and on OpenHex's ladder raw networks sit above our search bot (@exp:deploy-ladder).
- *Cost.* 3.4× `graph-s` per leaf on the PyTorch path (@exp:deploy-graph-m-check). Self-play runs on the fused half-precision kernels (@exp:az-speed), which are built for hidden 96; without them `graph-m` self-play is not affordable on this box. With them, an iteration was guessed at roughly 3× `graph-s`'s 82 s; measured below at 130 s.
- *On Modal* (Marco's question, 2026-11-17; projected, no `graph-m` self-play has been run). On an L40S `graph-m`'s search forward runs at 51,600 rows a second on the PyTorch path, against 520,000 for `graph-s` on the fused kernels (@exp:az-hardware), and an iteration evaluates 5.5M rows (@exp:az-modal-scale): about 110 s of card time beside 76 s of CPU work, so one process on 8 cores fills the card.
  - _Today's code_ (PyTorch path, one process, \$2.58 an hour): 20 to 30 iterations an hour, \$85 to \$130 per 1,000 iterations.
  - _With fused kernels_ (assumed 3 to 5× `graph-s`'s forward): \$55 to \$75 per 1,000 with one process; \$35 to \$70 with several processes on one card, which needs the trainer of @exp:az-modal-scale too.
  - _For scale._ `graph-s` costs \$59 per 1,000 with one process and \$19 to \$20 with several. fr3's line gained 3.85 pp in 300 iterations, 6.45 pp in 900 and 9.10 pp in 1,800; `graph-m` starts 4.85 pp behind e1800 (@exp:arch-graph-m-search).
  - A 5-iteration benchmark (about \$0.50) would replace these ranges with a number. Not run.
- *Start.* `2p-graph-m-1`, a PPO net: it has none of e1800's 329,221 games of self-play, so it starts behind in the search and has to make that up.

#lead[Kernels (built 2026-11-18, commit `73c34e9`).] The trunk's width is now a constant of the build, so one source gives one compiled extension per width: `graph-s`'s is unchanged and `graph-m`'s (hidden 192) is built on first use. Depth and head widths were already free.
- *`graph-s` unchanged.* On 3,000 positions of `2p-graph-s-8` the logits and values are bit for bit those of the code before the change, in full and half precision.
- *`graph-m` against PyTorch* (3,000 positions self-played by `2p-graph-m-1`). Full precision: logits within 6.2e-4 on a scale of 735, values within 6e-7, the same best move on every row. Half precision: logits within 0.40, values within 3.2e-4, the same best move on 99.97% of rows, policies 2e-6 nats apart on average (6.8e-4 at most). `2p-graph-m-1`'s logits are 13× as large as `2p-graph-s-8`'s, so its absolute error is larger and its error relative to scale smaller.
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, right),
    header: ([Network], [Forward], [512 rows], [Rows a second]),
    [`graph-m`], [PyTorch], [33.9 ms], [15,100],
    [`graph-m`], [fused], [20.2 ms], [25,300],
    [`graph-m`], [fused, half precision], [10.3 ms], [49,600],
    mid,
    [`graph-s`], [fused, half precision], [4.3 ms], [118,900],
  ),
  caption: [One 512-row forward on the RTX 2070 Super (`tools/kernels_check.py bench`). Self-play uses the half-precision kernels and the evals' search the full-precision ones.],
) <tab:az-graph-m-kernels>
- *Speed.* Half precision on the kernels is 3.3× the PyTorch path for `graph-m` and 2.4× slower than `graph-s` on the same kernels.
- *Tests.* 10 new cases; the kernel file's 49 tests pass on the GPU, the CPU suite passes (1,274), and the GPU tests pass (148).

#lead[Self-play, measured.] Four iterations of the run's config from `2p-graph-m-1` on this box (learner started early for the smoke; no W&B).
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    header: ([], [Seconds an iteration], [Leaves a second], [Iterations an hour]),
    [`graph-m`, smoke], [123 (130 with the learner)], [34,600], [27 to 28],
    [`graph-s`, `az-2p-tricks`], [70], [66,300], [51.5],
  ),
  caption: [Self-play with search on the RTX 2070 Super, 512 games in flight. `graph-m`: the median of iterations 2 to 4; its learner adds about 8 s (`graph-s`: about 4 s).],
) <tab:az-graph-m-speed>
- *Time.* 900 iterations at 130 s are about 32.5 h of self-play on this box (`az-2p-tricks`: 18.0 h), 1.9× `graph-s`, not the 3× guessed above.
- *Warm start.* The net takes the dice inputs with zero weights, a fresh outcome head and a soft policy head copied from the policy head (2,320,100 to 3,483,177 parameters); its policy and value at the start are `2p-graph-m-1`'s exactly. No games were cut at the turn limit.
- *A peaked start, as fr3's was.* `2p-graph-m-1`'s policy at searched roots has 0.03 to 0.08 nats of entropy, and the raw visit distribution sits 8 to 14 nats of KL from it. `az-2p-kata` began from fr3 at 0.04 to 0.05 nats and 10 to 15 nats (@exp:az-kata) and learned from there on the pruned target, which sits 0.2 to 0.5 nats from the net in both runs.
- *New at such a start:* the soft policy head and the surprise weighting. The smoke's first soft loss was 17.4 (`az-2p-tricks`: 2.44) and its first policy loss 3.79 (0.68); the gradient is clipped at norm 1.0 either way. The smoke's learner ran on at most 17,000 positions, so its drift (85% agreement with the start after two learning iterations) says little about the run, which waits for 50,000. To watch over the first 50 iterations: `drift/agree_init` (`az-2p-kata`: 0.96 at iteration 50), the policy loss and the soft loss.

#lead[The run as planned (`az-2p-graph-m`; launched 2026-11-18 16:24, stopped 18:25; relaunched below).] From `2p-graph-m-1`, 900 iterations of `az-2p-tricks`' recipe with every training key unchanged; only the network and its start differ.
- *Judged by* one final row, the lineage's: the 1,024-sim deployed search on the last checkpoint against fr3's fair raw bot on the same 2,000 games, paired with `2p-graph-s-8`'s row `cc4ccbcc` (64.65%) and with the start's row below. A paired difference is detectable from about 2.7 pp.
- *Cost.* Local GPU, \$0: about 32.5 h of self-play and 3.7 h for the final row (the start's row took 221 minutes), an eval share near 10%. On Modal with one process it was projected at about 100 s an iteration, 25 h and about \$65; not used.
- *What it cannot promise.* The first 900 iterations took the `graph-s` line from 53.85% to 60.30%. If `graph-m` gains as much from where it starts it lands near `2p-graph-s-8`; whether it goes past is the open question.

#lead[Where it starts (2026-11-18 12:43 – 16:24).] `2p-graph-m-1` in the 1,024-sim deployed search against fr3's fair raw bot, on the lineage's 2,000 games: *57.25%* #ci[55.1][59.4] (1,145 wins; 61.2% first seat, 53.3% second; record `9c7a904b`).
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Row on the same 2,000 games], [Score, %], [Start minus it, pp], [Won by only one], [p]),
    [`2p-graph-s-8` (`cc4ccbcc`)], [64.65], [−7.40], [313 vs 461], [1e-7],
    [The bundle (`b5483151`)], [63.70], [−6.45], [318 vs 447], [4e-6],
    [e1800 (`ea257443`)], [62.95], [−5.70], [332 vs 446], [5e-5],
    [e900 (`4411825a`)], [60.30], [−3.05], [362 vs 423], [0.032],
    [e300 (`e917f9bb`)], [57.70], [−0.45], [380 vs 389], [0.77],
    [fr3 (`809619df`)], [53.85], [+3.40], [407 vs 339], [0.014],
    mid,
    [`2p-graph-m-1` at 300 sims (`93649969`)], [58.10], [−0.85], [28 vs 45], [0.060],
  ),
  caption: [The start of the run, `2p-graph-m-1` at 1,024 sims, against each earlier row. "Won by only one": games the start won and the other row lost, against the reverse. Exact McNemar.],
) <tab:az-graph-m-start>
- *Level with e300.* The start sits where the `graph-s` line was after 300 iterations, 7.40 pp behind `2p-graph-s-8`.
- *More simulations do nothing for this net.* 1,024 sims and 300 sims give the same result in 1,927 of the 2,000 games, and the 1,024-sim row is 0.85 pp lower (p = 0.060). Its fair raw bot scores 55.0% against the same opponent (@exp:deploy-graph-m-check), so the search adds about 2 pp at either budget.
- *Likely why* (not tested). The prior is so peaked that the search rarely leaves its first choice: in the run's first three iterations of self-play it overrules the prior on 1 to 3% of searched moves, against 5 to 6% for `2p-graph-s-8`'s line. Search training is what widens the prior (`az-2p-kata`'s entropy at roots went from 0.05 to 0.20 nats in 300 iterations), so the search's share of the strength is still to be earned here.

#lead[The first launch collapsed (2026-11-18 16:24 – 18:25).] The learner started at iteration 10 (50,906 positions in the replay). Within three iterations the net had a different top move from its start on 24% of the probe positions, and it stayed there; the `graph-s` runs were at 4 to 5% by iteration 50.
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([], [`az-2p-graph-m`, iteration 10], [Iteration 12], [Iteration 50], [`az-2p-tricks`, iteration 50]),
    [Same top move as the start, share of probe positions], [0.871], [0.756], [0.786], [0.947],
    [Policy loss on the learner's batches], [6.92], [2.40], [1.07], [0.67],
    [Soft policy loss], [20.1], [5.51], [1.81], [1.49],
    [Value loss], [0.525], [0.520], [0.527], [0.543],
    [Net's entropy at searched roots, nats], [0.05], [0.20], [0.72], [0.39],
    [Searched moves where the search overrules the prior, %], [1.5], [5.3], [12.9], [6.3],
  ),
  caption: [Health signals of the first launch against the `graph-s` run of the same recipe. No game was cut at the turn limit, and an iteration took 123 to 135 s as measured.],
) <tab:az-graph-m-collapse>
- *The losses alone looked like learning:* both policy losses fell and the value loss was flat. The check that showed the damage was a match.
- *Raw policy against its own start* (`2p-graph-m-1`, 5,000 games on heldout-2p-fr, played on the GPU beside the run): *25.1%* #ci[23.9][26.3] at iteration 25 and *30.0%* #ci[28.7][31.2] at iteration 50 (records `8504808e`, `ce1a2189`). The same match for the `graph-s` runs: `az-2p-kata` 49.9% at iteration 25 and 50.2% at 50 against fr3 (`6f0089a4`, `89c3db96`), `az-2p-tricks` 48.8% at 50 against the bundle (`a1ff705e`).
- *Stopped* at iteration 54 (18:25, clean stop, checkpoint kept), under the condition agreed at launch: stop if the net falls apart. About 2 h of GPU lost. No row in the search was played on these checkpoints.

#lead[Why: the surprise weighting on a peaked net (2026-11-18, offline).] The learner's first three iterations were replayed offline from the run's e0 on the replay it had then (the start's own self-play), under variants of the recipe (`tools/learner_replay.py`; 22 s a variant). The replay of the recipe as run matches the run.
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Recipe], [Same top move as the start after 1, 2, 3 iterations], [Policy loss on uniform rows after 3 (start: 0.51)], [Share of the buffer the draws act like], [Raw policy vs the start, %]),
    [The run itself], [0.871, 0.798, 0.756], [—], [0.083], [25.1 (iteration 25)],
    [As run, replayed], [0.874, 0.808, 0.765], [1.07], [0.083], [22.1 #ci[20.9][23.2]],
    mid,
    [No surprise weighting], [0.958, 0.949, 0.948], [0.51], [1], [*50.1* #ci[48.7][51.5]],
    [No surprise weighting, no soft policy loss], [0.981, 0.975, 0.975], [0.45], [1], [49.3 #ci[47.9][50.7]],
    [Surprise weight 0.1 (run: 0.5)], [0.935, 0.921, 0.900], [0.70], [0.69], [—],
    mid,
    [No soft policy loss], [0.889, 0.833, 0.794], [1.04], [0.083], [—],
    [No outcome losses], [0.875, 0.812, 0.766], [1.08], [0.083], [—],
    [A fifth of the learning rate], [0.959, 0.932, 0.907], [0.94], [0.083], [—],
  ),
  caption: [Three learning iterations (192 updates of 512 rows) from `2p-graph-m-1` on 50,906 positions of its own self-play. "Share of the buffer the draws act like": the effective sample size of the replay draws (`az-2p-tricks`: 0.33). Raw policy: 5,000 games on heldout-2p-fr against `2p-graph-m-1`'s raw policy.],
) <tab:az-graph-m-variants>
- *The surprise weighting is the cause.* Without it the net stays level with its start (50.1%) and its policy loss on uniform rows no longer rises. Dropping the soft policy loss or the outcome losses instead changes nothing, and a lower learning rate only slows the same drift.
- *How.* The weighting draws half the batch in proportion to how far the search's target sits from the net's prior. `2p-graph-m-1` is so sure of itself that this distance is 0 at the median position and 4.2 nats at the 99th percentile (65 at most): 1.3% of positions sit at the cap, and 1% of the replay takes 27% of the draws. The learner trained mostly on the few positions where the search overruled a confident prior (policy loss 7.0 on its batches, 0.51 on uniform rows) and got worse everywhere else: the loss on uniform rows doubled.
- *Why `graph-s` was fine.* `az-2p-tricks` started from a net 900 iterations into search training, with a wider prior: mean surprise 0.04 nats against 0.22 here, policy loss on its batches 0.68. A PPO net meets this weighting here for the first time (`az-2p-kata`, from fr3, had none).
- *The soft policy head* moves the net more (5% of top moves changed against 2.5% without its loss) but costs no strength in this test. Kept.

#lead[The relaunch (`az-2p-graph-m-uniform`; 2026-11-18 18:44).] The same run with uniform replay draws (`surprise_weight: 0`); every other key, the start, the final row and its pairings unchanged. W&B `az-2p-graph-m-uniform-main-a2ac6eca`.
- *Healthy at iterations 25 and 50.* The raw policy against its start, `2p-graph-m-1` (5,000 games each, on the GPU beside the run): *49.2%* #ci[47.8][50.6] at iteration 25 and *49.4%* #ci[48.0][50.8] at 50 (records `ce298288`, `5824eca5`); the first launch scored 25.1% and 30.0%. At iteration 50: the same top move as the start on 92.1% of probe positions (first launch 78.6%, `az-2p-kata` 95.8%), policy loss 0.31 (from 0.50 when the learner started), soft policy loss 1.29 (from 12.0), value loss 0.49, entropy at searched roots 0.12 nats (from 0.05), the search overrules the prior on 3.3% of searched moves (from 1.5%). No game cut at the turn limit; 126 s an iteration.
- *Still level at iteration 300* (2026-11-19 05:30): *49.2%* #ci[47.8][50.6] against the start (record `cd8ee47e`). The same top move as the start on 89.0% of probe positions, policy loss 0.25, soft policy loss 1.06, value loss 0.50; on fresh self-play positions the value's cross-entropy is flat: 0.492 over iterations 10–50, 0.494 over 201–300 (one iteration's reading swings by ±0.025, so single readings mislead). The search overrules the prior on 4.1% of searched moves, and the target's KL from the prior is down to 0.055 nats (0.33 before learning). The raw match only shows the net is intact: the strength reading is the final row.
- *What is lost.* The surprise weighting was one of six pieces that together added 0.95 pp on `graph-s` (p = 0.47; @exp:az-tricks). Turning it back on once the prior has widened is possible and not planned.

#lead[Result (2026-11-20 06:52).] The last checkpoint, e900, in the 1,024-sim deployed search against fr3's fair raw bot on the lineage's 2,000 games: *67.00%* #ci[64.9][69.0] (1,340 wins; 70.2% first seat, 63.8% second; record `af97f759`). Registered as `2p-graph-m-2`; released for OpenHex as search-v4 on 2026-11-21 (@exp:deploy-openhex-search-v4).
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Row on the same 2,000 games], [Score, %], [`2p-graph-m-2` minus it, pp], [Won by only one], [p]),
    [`2p-graph-s-8` (`cc4ccbcc`)], [64.65], [+2.35], [382 vs 335], [0.086],
    [The bundle, `2p-graph-s-7` (`b5483151`)], [63.70], [+3.30], [397 vs 331], [0.016],
    [e1800, `2p-graph-s-6`, deployed (`ea257443`)], [62.95], [+4.05], [423 vs 342], [0.004],
    mid,
    [Its start, `2p-graph-m-1` (`9c7a904b`)], [57.25], [+9.75], [472 vs 277], [1e-12],
  ),
  caption: [`2p-graph-m-2` (`az-2p-graph-m-uniform/main@e900`) against the earlier rows. "Won by only one": games it won and the other row lost, against the reverse. Exact McNemar.],
) <tab:az-graph-m-result>
- *The best row of the line.* 2.35 pp above `2p-graph-s-8`, which is short of significance on 2,000 games (p = 0.086; a paired gap is detectable from about 2.7 pp), and 4.05 pp above the deployed net (p = 0.004).
- *Search training paid on `graph-m`:* +9.75 pp over its start in 900 iterations. The `graph-s` line gained 6.45 pp in its first 900 and 1.70 pp in its last 1,800 (39.2 h; @exp:az-tricks).
- *Time.* 32.3 h of self-play (2026-11-18 18:44 to 2026-11-20 03:03; 152,862 games, 5.15M recorded positions, 127 s an iteration) and 3.8 h for the final row: an eval share of 11%, health checks included.
- *The gain is in the search, not the raw policy.* The raw policy at e900 scores 51.0% #ci[49.6][52.4] against its start over 5,000 games (record `b7f7ed96`), as for the `graph-s` line through e1800 (@exp:az-raw). On the rules of 2026-11-24 the same row is 52.2%, a small real edge (@tab:az-graph-m-deeper-raw).
- *Flat from about iteration 500.* The same top move as the start on 88.3% of probe positions over iterations 401–500 and 87.1% over 801–847; the search overrules the prior on 4.5 to 4.6% of searched moves, mean depth 9.1 levels, value cross-entropy on fresh positions 0.49 throughout. The target's KL from the prior kept shrinking slowly (0.048 to 0.039 nats). Whether the last 400 iterations added strength is not measured: no row in the search was played on an earlier checkpoint.
- *More simulations help it* (Marco, 2026-11-20). The same search at 4,096 sims on the same 2,000 games: *70.75%* #ci[68.7][72.7] (1,415 wins; 73.5% first seat, 68.0% second; record `0d49a45d`), +3.75 pp on the 1,024-sim row (268 games won only at 4,096, 193 only at 1,024; paired p = 0.0006). 15.2 h with 64 games in flight; the deployed search runs up to 131,072 sims a move.
- *Not done.* No head to head with `2p-graph-s-8`, no continuation queued. *best-2p* stays `2p-graph-m-1` until Marco moves it.

#repro(```sh
# the first launch's raw policy against its start, and the offline replay of its learner
catanrl eval --suite heldout-2p-fr --boards 2500 --device cuda --num-envs 512 \
  "POLICY:ckpt=az-2p-graph-m/main@e25" "POLICY:ckpt=graph-m-2p-lr/main@e22711#673425d606b5"
python tools/learner_replay.py prep     --run runs/az-2p-graph-m/main --out /tmp/gm-diag
python tools/learner_replay.py variants --run runs/az-2p-graph-m/main --out /tmp/gm-diag
python tools/learner_replay.py ckpt     --run runs/az-2p-graph-m/main --out /tmp/gm-diag --variants run no-surprise
CATANRL_RUNS_DIR=/tmp/gm-diag/runs catanrl eval --suite heldout-2p-fr --boards 2500 --device cuda --num-envs 512 \
  "POLICY:ckpt=gmdiag/no-surprise@e3" "POLICY:ckpt=graph-m-2p-lr/main@e22711#673425d606b5"
# the relaunch (its final row is af97f759), the pairings and the raw policy against the start
catanrl run configs/experiments/az-2p-graph-m-uniform.yaml
python tools/search_probe.py --pair runs/az-2p-graph-m-uniform/main/evals/heldout-2p-fr/af97f7595f82b53d.json \
  runs/az-2p-tricks/main/evals/heldout-2p-fr/cc4ccbcc8cb418f6.json
catanrl eval --suite heldout-2p-fr --boards 2500 --device cuda --num-envs 512 \
  "POLICY:ckpt=az-2p-graph-m-uniform/main@e900" "POLICY:ckpt=graph-m-2p-lr/main@e22711#673425d606b5"
# the 4,096-sim row (0d49a45d; 15.2 h) and its pairing with the 1,024-sim row
catanrl eval --suite heldout-2p-fr --boards 1000 --device cuda --threads 12 --num-envs 64 --infer-rows 512 \
  "MCTS:ckpt=az-2p-graph-m-uniform/main@e900:sims=4096:dets=1:pending=64:reuse=true:transpositions=true:kernels=fused" \
  "BELIEF:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:worlds=16"
python tools/search_probe.py --pair runs/az-2p-graph-m-uniform/main/evals/heldout-2p-fr/0d49a45d48520aaa.json \
  runs/az-2p-graph-m-uniform/main/evals/heldout-2p-fr/af97f7595f82b53d.json
# kernels: equivalence and speed
python tools/kernels_check.py equal --ckpt runs/graph-m-2p-lr/main/checkpoints/e022711.pt --rows 3000
python tools/kernels_check.py bench --ckpt runs/graph-m-2p-lr/main/checkpoints/e022711.pt
# the start's row
catanrl eval --suite heldout-2p-fr --boards 1000 --device cuda --threads 12 --num-envs 256 --infer-rows 512 \
  "MCTS:ckpt=graph-m-2p-lr/main@e22711#673425d606b5:sims=1024:dets=1:pending=64:reuse=true:transpositions=true:kernels=fused" \
  "BELIEF:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:worlds=16"
# the first launch
catanrl run configs/experiments/az-2p-graph-m.yaml
```)

== Inputs

=== The balanced-dice state as a network input #chip("done") <exp:dice-input>
#meta(
  dates: "2026-11-10 (next-roll odds), 2026-11-15 (the whole state)",
  dates-label: "built",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Marco's idea: keep track of the shuffled cards and pass that to the model. The search already rolls from the exact odds, but the network's value and prior are the same whatever the deck holds. An opt-in input gives it the deck, the recent rolls and the seven balance, on the board and on each tile. Trained in the bundled AlphaZero run (@exp:az-bundle): the network reads it (a different top move on 3% of positions), and the search gained no measurable amount.]

#lead[Why.] Marco's reading, #link("https://benburk.ca/catan-law-of-small-numbers/")["Exploiting the law of small numbers in Catan"], reverse-engineers Colonist's balanced dice: a 36-card deck reshuffled when 13 cards are left, a penalty on totals rolled in the last five rolls, and in 1v1 a multiplier on sevens that evens out each player's share and breaks streaks. A player who tracks all of it puts about 20% more probability on the roll that comes (0.26 bits a roll).
- *Same dice.* `engine/cppanatron/include/cppanatron/dice.hpp` has the post's constants: refill at 13, a window of 5, 0.34 per recent copy, a seven multiplier clamped to 0–2 with ±0.4 per streak step.
- *Who uses them here.* The search's chance nodes and its sampled worlds. The network sees none of it.
- *How much the odds move* (our dice-only simulation, `tools/dice_odds_sim.py`): the roller's P(7) runs 4–28% (fair 16.7%), and on 58% of rolls some total cannot come.
- *The post's advice* (track the deck, the window and the seven share; place the robber and hold cards by them) is what a network with this input could learn.

#lead[Setup.]
- *Board inputs.* Each seat's odds for the next roll (11 totals per seat; all that the 2026-11-10 version had), the cards left in the deck per total, the count of each total in the last five rolls, the draws left before the refill, each seat's sevens and the current streak.
- *Tile inputs.* For the tile's own number: its odds on my roll and on the opponent's, its cards left and its count in the window. The network need not route eleven board numbers to the tiles that carry them.
- *Warm start.* Every new weight starts at zero, so a network given the input plays exactly as its parent at first. Off by default: rows, networks and config hashes stay as they are.
- *Any size.* The input attaches to the graph trunk's board and tile projections, the same in `graph-s` and `graph-m` (@exp:az-graph-m).

#lead[As built.]
- *Switch.* `obs.dice_state: true`, off by default; merged 2026-11-15.
- *Row.* A 2p `full` row grows from 2,564 to 2,691 floats: 51 on the board and 4 on each of the 19 tiles.
- *Board, 51.* Each seat's odds per total, as 6 × P(total) (22); the deck's cards per total (11); each total's count in the five-roll window, over 3 (11); the draws left before the refill; and per seat its sevens, its share of the sevens (added in the build) and the streak.
- *Tile, 4.* For the tile's number: its odds on my roll and on the opponent's (× 36/5), its cards left and its count in the window; zeros on the desert.
- *Network.* Two bias-free projections that start at zero, added to the board and tile projections: 7,616 parameters on `graph-s` (541,540 to 549,156). The fused kernels read them with two more small matrix products.
- *Check on the GPU.* A network warm-started with `init_from.zero_new_inputs` gives e1800's logits and values bit for bit on the fused kernels in full and half precision (plain PyTorch on the GPU: within 3e-6, the same best move). After 6 learning steps the input's weights have norm 0.10 and move the policy by 0.0002 nats on probe positions.
- *Logged.* `dice/weight_norm`, and on fixed probe positions `dice/probe_kl`, `dice/probe_flips` and `dice/probe_value_gap`: the network with the true dice state against the same network told the dice are fair.

*Result (@exp:az-bundle).* After 900 iterations the input's weights have norm 2.25 (2.04 on the tiles, 0.94 on the board). Against the same network told the dice are fair, on 512 probe positions: a different top move on 3.3%, policies 0.017 nats apart on average, values 0.018 apart. The search gained +0.75 pp over e1800 (interval −1.6 to +3.1), with the auxiliary targets in the same run. Its own effect is not separated. The PPO pair planned on 2026-11-10 (`dice-input-2p`) is dropped for the bundled run.
