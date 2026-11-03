#import "../macros.typ": *

= The PPO recipe <sec:recipe>

Which training recipe makes the strongest raw policy, and how strong is it? Self-play PPO from scratch on the `graph-s` net, v1's graph2 run ported to v2, wins 95% against F in 2p whether or not the net sees the opponent's hidden cards, and takes a 78.55 score share against three F in 4p. The one change since is the critic: undiscounted lambda-return targets, now the default for every run.

== The recipe

/ Network: `graph-s`: Canopy nexus-v3-style heterogeneous message passing (node↔node, node↔tile, node↔edge), hidden 96, 3 layers, 541,540 parameters. One shared trunk feeds the policy head and a per-seat value head: one sigmoid per seat, that seat's win probability. The acting row's prediction fills the waiting seats' values, so GAE on each seat's own trajectory costs one network call per game step. @sec:arch compares it with other families.
/ Training: PufferLib 3 PPO with two small patches (losses over acting rows only, a value-loss hook). Pure self-play: every seat learns. Muon at 5e-3 with a cosine to 5e-4, momentum 0.95; γ = 0.999 per decision, GAE λ = 0.95; 2 update epochs, clip 0.2, entropy 0.005; prioritized 64-step segments; fp32.
/ Batch: Exactly v1's graph2 run: 1,032 games × 2 rows × 64 steps = 132,096 agent steps per epoch, 16 optimizer steps per epoch, 3,786 epochs. An agent step is one seat's row at one game step, so a 2p decision is two agent steps and a 4p decision four. `configs/experiments/puffer-graph2.yaml` lists every deviation from v1 and why.
/ Critic: Undiscounted lambda-return targets (`critic_targets: lambda`, `critic_gamma: 1` in `configs/train/ppo.yaml`), the default since the critic fix (@exp:recipe-critic-fix). Earlier runs used V-trace targets at γ = 0.999; their YAMLs pin those values, so their config hashes and records are unchanged.
/ Budgets: 2p: 500M agent steps in 7.95 h. 4p: 1B agent steps (249M decisions) in 9.3 h. Short cells for ablations: 100M agent steps at 2p (about 1.6 h for the graph nets) and 750 epochs (49M decisions) at 4p (about 1.9 h).

#figure(
  image("../figures/select-curves.png", width: 100%),
  caption: [Training-time selection curves: the argmax policy against F on the `select-*` suites, every 50 epochs. Every run keeps improving through the second half of the cosine, so matched comparisons need the whole budget. The 2p `public` variant (orange) tracks `full` (blue) throughout. Per-epoch logs on W&B: #wb("puffer-graph2-full-2fdb6f44")[puffer-graph2/full], #wb("puffer-graph2-public-3ec072be")[puffer-graph2/public], #wb("ppo-4p-main-6349dc93")[ppo-4p/main] (no local mirror) and #wb("ppo-mixed-main-f4b771c0")[ppo-mixed/main] (the generalist, @sec:generalist).],
) <fig:curves>

== Two players

=== 2p at the `full` level: `puffer-graph2/full` #chip("done") <exp:recipe-2p-full>
#meta(
  dates: "2026-10-14 12:37–20:34 (training); 2026-10-17 – 2026-10-18 (finals)",
  owner: "marco",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-full-2fdb6f44", "puffer-graph2/full"),),
  records: ("ff8528a4", "67869e7a"),
)
#tldr[v1's graph2 run, reproduced in v2 from scratch at the `full` level, scores 95.16% against F played raw (unfair: it sees F's dev cards) and 95.06% belief-averaged (fair). Hidden information is worth under a point in 2p, and v2 lands 5 points above v1's warm-started reference.]

*Question.* Does v2's PPO stack, trained from scratch, reach v1's graph2 result?

*Setup.* The recipe above at the `full` level, graph2's own setting, so the comparison with v1 is like for like. 500M agent steps in 7.95 h; best checkpoint e3350 by `select-2p` (96.8).

*Result.* On `heldout-2p` (5,000 games each), the raw `full` policy scores *95.16* #ci[94.53][95.72] against F (`ff8528a4`, unfair). Belief-averaged over 16 sampled worlds (`BELIEF`), it scores *95.06* #ci[94.42][95.63] (`67869e7a`, fair).

*Reading.* The fair, belief-averaged policy loses nothing against the oracle: v1's "hidden information is worth under a point" again. Trained from scratch, with no DAgger, v2 lands 5 points above v1's reference (89.65, @exp:v1-reference). This net, belief-averaged, is the deployed 1v1 bot (@sec:deploy). The run's finals also hold two head-to-head search rows, `04fd2c25` and `2aaf2eb7` (@sec:search).

#repro(```sh
catanrl run configs/experiments/puffer-graph2.yaml --only full
# Its finals include the two search rows (MCTS 1024 and 4096 sims), about 11.5 h of GPU on top of training
```)

=== 2p at the `public` level: `puffer-graph2/public`, the fair raw bot #chip("done") <exp:recipe-2p-public>
#meta(
  dates: "2026-10-18 18:03 – 2026-10-19 02:06",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-public-3ec072be", "puffer-graph2/public"),),
  records: ("dc8a611a",),
)
#tldr[The same recipe at the `public` level, which in 2p is fair without belief averaging, scores 94.96% against F as a plain policy, level with the `full` net. One forward pass, no sampled worlds, no tracker: this is the bot to serve raw.]

*Question.* Can a fair network skip belief averaging and lose nothing?

*Setup.* At the `public` level the net sees the opponent's exact hand, which in 2p is a perfect card counter's knowledge; `full` adds only the opponent's dev cards in hand. Same recipe and budget as `full`: 7.95 h at 17.8k steps/s. Best checkpoint e3600 (select 96.5, against `full`'s 96.8 at e3350; mean over epochs ≥ 3000: 94.4 against 95.1). The two select curves lie on top of each other (@fig:curves).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, left, right, right, left),
    header: ([Agent], [Information], [Games], [Score vs F], [Record]),
    [`public` net, raw \ #sub[`POLICY`, e3600]], [Fair], [5,000], [*94.96* #ci[94.3][95.5]], [`dc8a611a`],
    [`full` net, belief-averaged \ #sub[`BELIEF`, 16 worlds, e3350]], [Fair], [5,000], [95.06 #ci[94.42][95.63]], [`67869e7a`],
    [`full` net, raw \ #sub[e3350]], [Sees F's dev cards], [5,000], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
  ),
  caption: [What hidden information is worth in 2p: the same recipe at the `public` and `full` levels, against F on `heldout-2p`.],
)

*Reading.* The one thing `public` can't see, the opponent's dev cards, is worth nothing measurable in 2p. The hidden-feature head reached the same conclusion from the other side: the hand is inferable, the dev cards are not, and the policy didn't need them (@sec:hidden). This run's search rows (`72819d1a`, `0ef1b81e`, `831eea84`) are in @sec:search.

#repro(```sh
catanrl run configs/experiments/puffer-graph2.yaml --only public
```)

=== The 1v1 bot on OpenHex's 1v1 rules: \~1B more steps, judged against its start #chip("done") <exp:recipe-2p-openhex>
#meta(
  dates: "2026-10-24 10:46 – 2026-10-25 03:56",
  owner: "marco",
  config: "configs/experiments/ppo-2p-fr.yaml",
  wandb: (("ppo-2p-fr-full-fef9516c", "ppo-2p-fr/full"),),
  records: ("e435b80e", "6c4f23cb", "3cd19d23", "fa618736", "36bd2384", "db519a9e", "8a137f56", "ffcc5c1a"),
)
#tldr[OpenHex's 1v1 queue plays a friendly robber and balanced dice, which our 2p bots never trained on. Continued for \~1B steps on those rules, the deployed 1v1 net beats its start 63.2% head to head (\~61.7% after correcting for the start's untrained friendly-robber input) and F 98.8% (the start: 97.3%), and was still improving at the end.]

*Question.* Marco: train on the friendly robber from now on, continue the deployed 1v1 net for \~1B more steps on it, and judge it against its start (the champion) as well as F. Does it still improve?

#lead[Setup.]
- *Rules.* From OpenHex's source: its 1v1 queue is 15 VP, discard above 9, friendly robber, balanced dice; its 4p queue is 10 VP, discard above 7, normal robber, random dice, which is already our 4p setup. `base-2p-fr` is OpenHex's 1v1 exactly.
- *Friendly robber.* OpenHex shields an opponent with *public* points ≤ 2. Our engine used *actual* points < 3, which counts hidden VP cards: a mismatch, and the legal moves leaked whether the opponent held one. The engine now uses OpenHex's rule, and judges a knight on the points before it takes largest army, as OpenHex does.
- *Balanced dice* (Marco: "do balanced dice before 1v1 retrain"), following OpenHex's code: a 36-card deck refilled when 13 or fewer are left, a recency penalty on totals seen in the last 5 rolls, and per-player seven balancing. Search chance nodes use the state's own odds. The network gets no dice-state input, so the policy plays the dice blind and only the search reads them. The two recorded OpenHex games say balanced dice but were played on fair dice (an OpenHex bug, since fixed); the bridge falls back to fair dice if a roll is impossible under the balanced deck.
- *Run.* `puffer-graph2/full@best` (e3350, the deployed 1v1 net) continued on `base-2p-fr` for 7,600 epochs (\~1B agent steps) with a fresh cosine from 0.0025 (half the original peak) to 0.00025, the recipe otherwise unchanged, at \~17.8k agent steps/s (\~17.5 h with monitors and finals). It kept the start's discounted V-trace critic (γ = 0.999) to match the start, not the undiscounted default since @exp:recipe-critic-fix. Claude made that choice without asking; the two continuations below kept it too.
- *Baselines.* The start scores *96.3* #ci[94.9][97.3] against F on `select-2p-fr` (`8a137f56`). Its friendly-robber input was never on in training: a copy with that input zeroed beats it *51.5* #ci[50.2][52.9] over 5,000 games (`ffcc5c1a`). Part of any gain over the start is therefore just learning to ignore that input, so the final head to head subtracts 1.5 pp for it.
- *Identity.* The merge moved the code ids every record is keyed under (catanrl `55ba5984`, engine `a9b40d9c`, train `55588933`), so new rows pair with older records only through `tools/search_probe.py --pair`. Old-rule games (random dice, normal robber) play bit-identically to the old engine.
- *Champion eval (Marco).* PPO stagnation had been read from the win rate against F, which sits near its 95% ceiling. Every select now also plays the current checkpoint against a champion (here the start): 1,000 games on the same boards each time, ±3.1 pp, paired game by game across checkpoints. The finals include the start under the new rules, since it never trained on them. At e0 the champion row reads 50.0 (`fa6ea360`), as it must.

*Result.* All 7,600 epochs ran (1.0B agent steps at \~17.8k/s; training ended at \~03:25, the finals at 03:56). The champion row climbed the whole way and was still rising at the end; best is the last checkpoint, e7600.
#figure(
  btable(
    columns: (auto,) + (1fr,) * 7,
    align: (left,) + (right,) * 7,
    header: ([From], [e0], [e1000], [e2000], [e3000], [e4000], [e5000], [e6000]),
    [Vs the start], [49.5], [51.2], [54.4], [56.1], [57.9], [60.5], [61.9],
    [Vs F], [96.5], [97.2], [97.3], [97.6], [98.2], [98.0], [98.2],
  ),
  caption: [Monitors during training on `select-2p-fr`. Each column is the mean of ten selects (10,000 games) from that checkpoint on: e0–450, e1000–1450 and so on. The last block, e7000–7600, reads 63.2–64.4 against the start and 98.4–98.6 against F.],
)
#figure(
  btable(
    columns: (auto, auto, auto, 1fr),
    align: (left, right, right, left),
    header: ([Agent], [e7600], [The start], [Comparison]),
    tgroup(4)[Against F],
    [`POLICY`], [*98.84* #ci[98.50][99.10] \ #sub[`e435b80e`]], [97.32 #ci[96.83][97.73] \ #sub[`3cd19d23`]], [+1.52 pp, 126 vs 50, p ≈ 1e-8],
    [`BELIEF` (16 worlds)], [*98.76* #ci[98.41][99.03] \ #sub[`6c4f23cb`]], [97.30 #ci[96.81][97.71] \ #sub[`fa618736`]], [+1.46 pp, 127 vs 54, p ≈ 6e-8],
    tgroup(4)[e7600 against the start],
    [`POLICY`], [*63.2* #ci[61.8][64.5] \ #sub[`36bd2384`]], [—], [≈ 61.7 after the 1.5 pp input correction],
    [`BELIEF`], [*62.2* #ci[60.9][63.5] \ #sub[`db519a9e`]], [—], [≈ 60.7 after the correction],
  ),
  caption: [Finals on `heldout-2p-fr`, 5,000 games each. The start is `puffer-graph2/full@e3350`, the deployed 1v1 net, on the same rules and boards. Against F the comparison is paired: the difference, the games only one of the two won (e7600's count first) and McNemar's p.],
)

*Reading.* Another billion steps on OpenHex's 1v1 rules made the deployed 1v1 net clearly stronger: about 62% head to head against its own start after correcting for the start's untrained friendly-robber input (roughly +85 Elo), and its losses to F more than halved (2.7% → 1.2%). The champion row was still rising when the learning rate reached its floor; a second billion steps at that learning rate gained little more (@exp:recipe-2p-openhex-3). None of this is measured inside the search yet. When this ran, the deployed OpenHex bots ran MCTS-1024 and BELIEF on the old engine and the old net; the next run's net has since been deployed (@exp:deploy-openhex-v2).

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/ppo-2p-fr.yaml
```)

=== The same net, another \~1B steps from e7600 at the floor learning rate #chip("done") <exp:recipe-2p-openhex-3>
#meta(
  dates: "2026-10-25 18:49 – 2026-10-26 11:59",
  owner: "marco",
  config: "configs/experiments/ppo-2p-fr3.yaml",
  wandb: (("ppo-2p-fr3-full-e95ea1db", "ppo-2p-fr3/full"),),
  records: ("507970d7", "cbba2d10", "7c0d2f67", "b38e49dc", "32d82b76", "f2f3858d"),
)
#tldr[A second billion steps at `ppo-2p-fr`'s final learning rate barely moved the 1v1 net: the curve against e7600 was flat for 4,000 epochs, then rose slowly to \~52–53. The checkpoint picked as best, e4350, is level with e7600 on held-out boards: 51.0 #ci[49.6][52.4] head to head, 98.9 vs 98.8 against F.]

*Question.* Does the 1v1 net keep improving on OpenHex's 1v1 rules if training just continues where `ppo-2p-fr` ended?

#lead[Setup.]
- *Run.* `ppo-2p-fr` exactly, initialized from `ppo-2p-fr/full@e7600` with a fresh optimizer. The learning rate is held at `ppo-2p-fr`'s floor, 0.00025, for 7,600 epochs (\~1B agent steps), with no cosine. \~17 h.
- *Monitors.* The champion row plays every 50th checkpoint against e7600 (1,000 games, ±3.1 pp, paired across checkpoints; ±1 pp per 10-checkpoint block), with the score against F alongside. At e0 it reads 50.0 against the champion and 98.8 against F, and the first updates run at KL 0.0006–0.0009, where `ppo-2p-fr` ended.
- *Finals* on `heldout-2p-fr`, 5,000 games each: `@best` against F, against e7600 and against the original deployed net `puffer-graph2/full@e3350`, each as POLICY and as BELIEF.

#lead[Result.]
#figure(
  btable(
    columns: (auto,) + (1fr,) * 8,
    align: (left,) + (right,) * 8,
    header: ([From], [e0], [e1000], [e2000], [e3000], [e4000], [e5000], [e6000], [e7000]),
    [Vs e7600], [50.1], [50.0], [49.1], [49.9], [51.2], [51.9], [52.0], [52.8],
    [Vs F], [98.5], [98.7], [98.3], [98.5], [98.7], [98.6], [98.9], [98.7],
  ),
  caption: [Monitors during training on `select-2p-fr`. Each column is the mean of ten selects (10,000 games, ±1 pp) from that checkpoint on: e0–450, e1000–1450 and so on. The blocks not shown read the same; the last three checkpoints, e7500–7600, read 51.1. As Elo against e7600, the slope is −0.4 ± 0.9 per 1,000 epochs over e0–4000 and +3.5 ± 0.4 over the whole run; `ppo-2p-fr` gained +10 to +17 per 1,000 epochs against its start throughout.],
)
#figure(
  btable(
    columns: (auto, auto, auto, 1fr),
    align: (left, right, right, left),
    header: ([Agent], [e4350 (best)], [e7600 (start)], [Comparison]),
    tgroup(4)[Against F],
    [`POLICY`], [*98.92* #ci[98.59][99.17] \ #sub[`507970d7`]], [98.84 \ #sub[`e435b80e`]], [+0.08 pp, 48 vs 44, p = 0.75],
    [`BELIEF` (16 worlds)], [*98.94* #ci[98.62][99.19] \ #sub[`cbba2d10`]], [98.76 \ #sub[`6c4f23cb`]], [+0.18 pp, 53 vs 44, p = 0.42],
    tgroup(4)[Against the deployed net, `puffer-graph2/full@e3350`],
    [`POLICY`], [*64.7* #ci[63.3][66.0] \ #sub[`32d82b76`]], [63.2 \ #sub[`36bd2384`]], [+1.5 pp, 779 vs 704, p = 0.055],
    [`BELIEF`], [*63.5* #ci[62.2][64.8] \ #sub[`f2f3858d`]], [62.2 \ #sub[`db519a9e`]], [+1.3 pp, 810 vs 744, p = 0.099],
    tgroup(4)[e4350 against e7600],
    [`POLICY`], [*51.0* #ci[49.6][52.4] \ #sub[`7c0d2f67`]], [—], [p = 0.16],
    [`BELIEF`], [*51.2* #ci[49.8][52.6] \ #sub[`b38e49dc`]], [—], [p = 0.098],
  ),
  caption: [Finals on `heldout-2p-fr`, 5,000 games each. `puffer-graph2/full@e3350` is the deployed 1v1 net that `ppo-2p-fr` started from; the rows against it pair game for game with `ppo-2p-fr`'s finals on the same boards. Comparison: the paired difference, the games only one of the two won (e4350's count first) and McNemar's p; for e4350 against e7600, the p-value against an even score.],
)

*Reading.* The second billion steps bought little. For the first 4,000 epochs the net did not move against e7600; it then rose slowly to \~52–53 by e6000–7000, which is \~15–20 Elo, about a fifth of what the first billion bought (\~+100 Elo against its start). Every final points the same way: e4350 leads e7600 by 1.0–1.5 pp head to head and against the old net, and by 0.1–0.2 pp against F, none of it significant on its own. So e7600 is roughly where this recipe levels off on OpenHex's 1v1 rules, and e4350 is at most a little better.

*A lucky pick.* e4350 was picked as best because its one select row read 55.5, the highest of 153 rows of ±3.1 pp each. Picking the maximum favours a lucky row, and on held-out boards e4350 reads 51.0, about the mean of its block. The checkpoints from e6000–7000 averaged 52–53 on select, but none of them has a held-out final. Picking best from the block means instead of the single highest row would avoid this (Claude's suggestion, not queued).

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/ppo-2p-fr3.yaml
```)

=== The first try: a fresh cosine from the peak learning rate #chip("stopped") <exp:recipe-2p-openhex-2>
#meta(
  dates: "2026-10-25 14:03 – 18:48 (stopped at e2010)",
  owner: "marco",
  config: "configs/experiments/ppo-2p-fr2.yaml",
  wandb: (("ppo-2p-fr2-full-0b305e27", "ppo-2p-fr2/full"),),
  records: (),
)
#tldr[Restarting the learning rate at its peak, 0.0025, dropped the continuation to 40–43% against its own start within 50 epochs. It stayed there through e2000, so Marco stopped it and restarted at the floor (@exp:recipe-2p-openhex-3).]

*Question.* The same as @exp:recipe-2p-openhex-3.

*Setup.* `ppo-2p-fr` exactly, initialized from `ppo-2p-fr/full@e7600` with a fresh optimizer and a fresh cosine (0.0025 → 0.00025 over 7,600 epochs), as `ppo-2p-fr` had done from its own start. Monitors and planned finals as in @exp:recipe-2p-openhex-3.

*Result.* Stopped at e2010 (26%), before any finals. The champion row fell to 43.5 by e50 and stayed at 40–43: the 10-checkpoint block means are 43.3, 41.2, 40.6, 42.1 and 40.0, each over 10,000 games (±1 pp). Against F it fell from 98.8 to \~97.7.

*Reading.* The learning-rate restart caused this, not the extra steps:
- e7600 ended at the floor learning rate (0.00025): KL 0.001 per update, entropy 0.048.
- The restart at 0.0025 pushed KL to 0.02–0.03 at once, about 20 times larger.
- `ppo-2p-fr`'s own restart from e3350 (entropy 0.084, a broader policy) ran at KL 0.015 and stayed at 48–51 early.

A sharp policy that ended at a low learning rate loses \~8–10 pp head to head when the learning rate jumps back to the peak. The cosine would have returned to the floor by e7600, so this run's end would have tested a warm restart, not plain continued training. Lesson: set a continuation's learning rate from its start's final KL and entropy.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/ppo-2p-fr2.yaml
```)

== Four players

=== 4p specialist: `ppo-4p/main` #chip("done") <exp:recipe-4p>
#meta(
  dates: "2026-10-14 20:55 – 2026-10-15 06:16",
  owner: "marco",
  config: "configs/experiments/ppo-4p.yaml",
  wandb: (("ppo-4p-main-6349dc93", "ppo-4p/main (no local mirror)"),),
  records: ("e5b302ee", "b07e5bd9", "95570c74"),
)
#tldr[The 2p recipe on 4p BASE at the `private` level, the only fair level past two players, takes a 78.55 score share against three F (parity 25) and 100.0 against three random players. Its V-trace critic was biased high, which the critic fix below corrects.]

*Question.* Does the 2p recipe carry over to four players unchanged?

*Setup.* `puffer-graph2`'s recipe on `base-4p` (BASE, 10 VP, discard limit 7), all four seats learning, at the `private` level. 1B agent steps = 249M decisions (3,800 epochs), about the decisions of the 2p run, in 9.3 h. Best checkpoint e3650 by `select-4p`.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([Lineup], [Share of the policy's slots], [Record]),
    [Policy vs F, F, F], [*78.55* #ci[77.25][79.79]], [`e5b302ee`],
    [Two copies vs F, F], [93.2 together (40.45 + 52.75)], [`b07e5bd9`],
    [Policy vs R, R, R], [100.0 #ci[99.9][100.0]], [`95570c74`],
  ),
  caption: [The 4p specialist on `heldout-4p`, 4,000 games each. Parity is 25 per slot.],
)

*Reading.* The two identical copies split 40.45 / 52.75 because of the slot order, not because they differ: the copy in slot 0 sits right after an F (@sec:front, @sec:generalist). The critic's four seat predictions summed well above the realized sum, a bias from V-trace targets on waiting rows (@exp:recipe-critic-fix).

#repro(```sh
catanrl run configs/experiments/ppo-4p.yaml
```)

== The critic

=== The 4p critic fix: lambda-return targets, undiscounted #chip("done") <exp:recipe-critic-fix>
#meta(
  dates: "2026-10-18 14:58–16:00",
  config: "configs/experiments/ppo-4p-critic-fix.yaml",
  wandb: (("ppo-4p-critic-fix-lambda-3eb6d5cf", "lambda"), ("ppo-4p-critic-fix-lambda-critic-g1-57cb4f34", "lambda-critic-g1"), ("ppo-4p-critic-fix-lambda-g1-d5d2fc7f", "lambda-g1"), ("ppo-4p-critic-fix-lambda-adamw-e6f3dfe2", "lambda-adamw")),
  records: ("60cd10b6", "7d9374ed", "f4be1c20", "1d639013", "8e93695a"),
)
#tldr[`ppo-4p`'s four seats' win probabilities summed to about 1.2 against a realized 0.87. Lambda-return critic targets put the sum on the truth within about 20 epochs (undiscounted, the sum is 1.00), and the policy doesn't move. They are now the default for every run.]

*Question.* Why is the 4p critic biased high, and what fixes it without touching the policy?

*Diagnosis.* The seat sum was about 1.2 against a realized 0.87 at epoch 300, and 0.93 against 0.90 by the end. CPU replications found two causes:
- V-trace ratios on waiting rows. Only the acting seat's row gets min(ρ, 1); the other three get 1. So each seat's target values the game as if that seat alone played the improved policy, and the four one-sided gains compound through bootstrapping.
- Muon's fixed-size update of the 1 × 128 value output layer. The predicted sum alternates high and low every optimizer step, and the rollout network lands on the high side every epoch.

*Setup.* 100-epoch fine-tunes of `ppo-4p/main@latest` at its final learning rate (5e-4, constant), with lambda-return critic targets (every ratio 1; the policy's advantages stay V-trace's). Four arms of about 15 minutes each:
- `lambda`: γ = 0.999 for policy and critic.
- `lambda-critic-g1`: the critic undiscounted, the policy at γ = 0.999.
- `lambda-g1`: policy and critic undiscounted.
- `lambda-adamw`: `lambda` with the value output layer on AdamW instead of Muon.

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr, auto, auto, auto),
    align: (left, left, right, right, left),
    header: ([Arm], [Critic targets], [Seat sum vs realized], [`select-4p`], [Record]),
    [Start (`ppo-4p/main`)], [V-trace, γ = 0.999], [0.93 vs 0.90], [79.9], [`60cd10b6`],
    mid,
    [`lambda`], [Lambda, γ = 0.999], [0.906 vs 0.899], [80.4], [`7d9374ed`],
    [`lambda-critic-g1`], [Lambda, critic γ = 1], [0.998 vs 1.000], [79.9], [`f4be1c20`],
    [`lambda-g1`], [Lambda, policy and critic γ = 1], [1.000 vs 1.000], [80.4], [`1d639013`],
    [`lambda-adamw`], [Lambda, γ = 0.999, output layer on AdamW], [0.906 vs 0.898], [79.1], [`8e93695a`],
  ),
  caption: [The critic-fix arms at the end of the 100-epoch fine-tune. Seat sum: the four seats' predicted win probabilities summed, against the realized sum. `select-4p`: the policy's share against F, F, F, n=800 per point (±2.8). The start row's record sits in all four arms' run directories, so `catanrl verify` needs its path, e.g. `runs/ppo-4p-critic-fix/lambda/evals/select-4p/60cd10b66aea063c.json`.],
)
Every arm reaches its realized sum within about 20 epochs and stays there (@fig:critic); `lambda` started at 0.920 against 0.914 in its first epochs. The rollout-vs-trained gap falls from 0.004 to 0.000 in all four arms. Per-slot calibration at the end of `lambda-critic-g1`: the seat to move predicted 0.284 against a realized 0.276, the other three within 0.004. Clip fraction, entropy and game length (79–80 turns) don't change.

#figure(
  placement: auto,
  image("../figures/critic-fix.png", width: 100%),
  caption: [Left, `ppo-4p-critic-fix`: the four seats' predicted win probabilities summed, per fine-tune epoch, against the realized sum (dashed: about 0.90 at γ = 0.999, 1 undiscounted). Lambda-return targets put the sum on the truth in every arm within about 20 epochs; the AdamW output layer changes nothing. Right, from scratch under the new default (@exp:recipe-critic-from-scratch): `aux-hidden-4p/plain` (purple) against `ppo-4p/main`'s V-trace critic over its first 750 epochs (orange; its own realized sum is about 0.90, the dashed 1.0 is the undiscounted critic's). Both overshoot early; the lambda critic is on its realized sum from epoch 220, while the V-trace critic is still 0.16 above its own at epoch 750.],
) <fig:critic>

*Reading.* The targets alone fix it. They also remove Muon's phase lock, so the AdamW arm buys nothing. The undiscounted policy plays games of the same length, so over a 100-epoch fine-tune γ = 1 changes nothing the policy optimises (from scratch it does make games longer, @exp:recipe-discount). Per-seat win probabilities that sum to one are what max#super[n] backups in the search want, so `critic_targets: lambda` and `critic_gamma: 1` became the defaults in `configs/train/ppo.yaml`.

#repro(```sh
catanrl run configs/experiments/ppo-4p-critic-fix.yaml
# Four arms, about 15 min each; each fine-tunes ppo-4p/main@latest, so ppo-4p must be trained first
```)

=== The lambda critic from scratch #chip("done") <exp:recipe-critic-from-scratch>
#meta(
  dates: "2026-10-19",
  config: "configs/experiments/aux-hidden-4p.yaml",
  wandb: (("aux-hidden-4p-plain-f07eb81c", "aux-hidden-4p/plain"), ("aux-hidden-4p-aux-4dcfd020", "aux-hidden-4p/aux"), ("lstm-4p-main-4cae1a6d", "lstm-4p/main")),
  records: (),
)
#tldr[Trained from scratch, the undiscounted lambda critic overshoots early (a seat sum of 1.83 at epoch 21) and is on the realized 1.00 by epoch 220, where it stays. `ppo-4p`'s V-trace critic overshoots too and never comes back (0.94 against 0.90 at epoch 3,800). The fine-tunes couldn't show this transient, since their critics started trained.]

*Question.* Does the new default calibrate when the critic starts from random weights?

*Setup.* The first from-scratch runs under the new default: the two `aux-hidden-4p` arms, `ppo-4p`'s recipe at 750 epochs (49.2M decisions). They are a hidden-information experiment (@sec:hidden); here only their critics are read, against `ppo-4p/main`'s first 750 epochs. `lstm-4p` (@sec:markov) repeats the check on the recurrent net. The numbers are the training metric `critic/value_sum_4p`, not eval records.

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr, auto, 1fr),
    align: (left, left, left, left),
    header: ([Run], [Early overshoot], [On the realized sum], [Later]),
    [`aux-hidden-4p/plain` \ #sub[Lambda, γ = 1]], [1.83 at e21, 1.39 at e50, 1.31 at e100, 1.18 at e200], [1.00 by e220], [Mean 1.000 over e300–750, within 0.06 at every epoch],
    [`aux-hidden-4p/aux` \ #sub[Lambda, γ = 1]], [—], [—], [Mean 0.99],
    [`lstm-4p/main` \ #sub[Lambda, γ = 1]], [1.59 at e21], [1.00 by e250], [Mean 1.000 over e300–750],
    mid,
    [`ppo-4p/main` \ #sub[V-trace, γ = 0.999]], [1.27 at e100, 1.21 at e300, 1.09 at e500, 1.06 at e750], [Never], [0.94 against a realized 0.90 at e3800],
  ),
  caption: [The four seats' predicted win probabilities summed, by epoch, from random weights; the critic's targets and discount are under each run. The undiscounted lambda critic's realized sum is 1.00; the V-trace critic's is about 0.90.],
)

*Reading.* The early overshoot is a from-scratch transient that both target rules produce; only the lambda critic recovers. It lands at epoch 220, while the learning rate is still four fifths of its peak. Explained variance is 0.82–0.85 at the end, as before. The right panel of @fig:critic shows the two curves.

#repro(```sh
catanrl run configs/experiments/aux-hidden-4p.yaml --only plain
```)

== The discount

=== Discount vs no discount, from scratch #chip("done") <exp:recipe-discount>
#meta(
  dates: "2026-10-26 13:08 – 2026-10-27 23:05 pdt",
  owner: "marco",
  config: "configs/experiments/discount-2p.yaml",
  wandb: (("discount-2p-g999-s42-5057ed5a", "discount-2p/g999-s42"), ("discount-2p-g1-s42-5f1a556d", "discount-2p/g1-s42"), ("discount-2p-g999-s43-77e69c7e", "discount-2p/g999-s43"), ("discount-2p-g1-s43-12bb8a44", "discount-2p/g1-s43"), ("discount-2p-h2h-main-e52c5b42", "discount-2p-h2h")),
  records: ("57d449b3", "31ead730", "2109921e", "312910df", "9418974e", "05cc8756", "548ee03e", "e6bf7ad4", "98e98e0d", "e0b8011e"),
)
#tldr[*The discount makes games end faster*: trained without it (γ = 1 instead of 0.999, from scratch on OpenHex's 1v1 rules, two seeds per arm), games against F run 4.6 and 2.6 turns longer on the two seeds (paired over the same 5,000 games, ±0.35) and 1.6–1.7 turns longer against the reference, while the same arm on two seeds differs by half a turn. *It doesn't change strength*: head to head the arms are dead level, 50.02% and 49.94% on the two seeds (5,000 games each), and level against F and the reference.]

*Question.* Does the discount make games end faster (Marco's main question), and does dropping it play better?

*Why it might.* Every PPO run so far discounts the policy at γ = 0.999 per decision, inherited from v1. Catan needs no discount: every game ends, the only reward is the win at the end, and GAE's λ = 0.95 already does most of the variance reduction. But γ < 1 changes the objective from p(win) to E[γ#super[T] · win]: a win is worth more the sooner it comes (\~0.85 at the opening of a 160-decision game), which may push the policy towards faster, riskier wins. The only earlier test was 100 epochs of 4p fine-tuning (`lambda-g1` in @exp:recipe-critic-fix: the same game length and the same select score), far too short to say what training from scratch does.

#lead[Setup.]
- *Arms.* `g999`: γ = 0.999 for the policy and the critic. `g1`: γ = 1 for both. Both use lambda-return critic targets (the default), so only the discount differs.
- *Recipe.* `puffer-graph2`'s, from scratch on `base-2p-fr` (15 VP, discard above 9, friendly robber, balanced dice), `full` level, 500M agent steps (3,800 epochs, cosine 0.005 → 0.0005).
- *Seeds.* 42 and 43 per arm; the seed also draws the 4,096 training boards. Two seeds of this recipe had never been compared, so the seed-to-seed spread was unknown: an arm difference counts only if both seeds agree in sign and the pooled head to head clears it.
- *Monitors* every 50 epochs on `select-2p-fr` (1,000 games, ±3.1 pp at 50%): against F, and against a fixed reference, `ppo-2p-fr3/full@e4350`, so all four curves share one yardstick.
- *Finals* on `heldout-2p-fr`, 5,000 games each, on each cell's *last* checkpoint (chosen before the run, so no winner's curse): against F (±0.3 pp near 98) and against the reference (±1.4 pp at 50%). Once all four cells were done, a separate eval played `g1` against `g999` within each seed (5,000 games, paired MDE \~2 pp at 80%) and compared game lengths.
- *Game length* (the main readout, Marco). Every cell's finals play the same 5,000 game ids (board, seats and game seed), so turns per game pair up game by game between the two arms of a seed. Against F, which the net beats 98–99% of the time, this measures how fast it closes out a won game; against the reference, the length of a close game. From `g999-s42`'s finals (sd 11.5 turns against F), the paired difference within a seed resolves to about ±0.45 turns (95%) even if games on one board are uncorrelated. The gap between the two seeds of one arm is the real yardstick. The self-play episode length logged in training (`environment/length`) gives the same comparison epoch by epoch.
- *Cost.* \~8.4 h per cell, \~34 h in all, in the order s42-`g999`, s42-`g1`, s43-`g999`, s43-`g1`.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([Arm], [Score vs F], [Score vs ref], [Turns vs F], [Turns vs ref], [Self-play length]),
    tgroup(6)[Seed 42],
    [`g999` \ #sub[γ = 0.999]], [98.56 #ci[98.19][98.85] \ #sub[`57d449b3`]], [43.38 #ci[42.01][44.76] \ #sub[`31ead730`]], [70.85], [77.42], [77.7],
    [`g1` \ #sub[γ = 1]], [98.60 #ci[98.24][98.89] \ #sub[`2109921e`]], [44.20 #ci[42.83][45.58] \ #sub[`312910df`]], [75.48], [79.07], [81.0],
    tgroup(6)[Seed 43],
    [`g999` \ #sub[γ = 0.999]], [98.56 #ci[98.19][98.85] \ #sub[`9418974e`]], [44.76 #ci[43.39][46.14] \ #sub[`05cc8756`]], [71.31], [77.24], [77.3],
    [`g1` \ #sub[γ = 1]], [98.08 #ci[97.66][98.43] \ #sub[`548ee03e`]], [42.80 #ci[41.43][44.18] \ #sub[`e6bf7ad4`]], [73.96], [78.97], [82.2],
  ),
  caption: [Finals on `heldout-2p-fr`, 5,000 games each, on each cell's last checkpoint (e3786); ref = `ppo-2p-fr3/full@e4350`. Turns: the mean per game over all 5,000 (sd 11.5 vs F, 13.3 vs ref). Self-play length: `environment/length` from training, over the last 200 epochs.],
)
#figure(
  btable(
    columns: (auto,) + (1fr,) * 8,
    align: (left,) + (right,) * 8,
    header: ([Cell], [e0], [e500], [e1000], [e1500], [e2000], [e2500], [e3000], [e3500]),
    tgroup(9)[Score vs the reference (`select-2p-fr`)],
    [`g999-s42`], [9.8], [22.4], [27.4], [31.0], [35.0], [39.2], [40.9], [42.1],
    [`g1-s42`], [9.1], [22.5], [27.9], [30.8], [35.4], [39.7], [42.0], [43.3],
    [`g1-s43`], [8.7], [20.7], [26.6], [31.1], [35.8], [37.7], [41.1], [42.2],
    tgroup(9)[Score vs F (`select-2p-fr`)],
    [`g999-s42`], [72.9], [92.6], [94.5], [96.0], [96.8], [97.2], [97.9], [98.3],
    tgroup(9)[Self-play length (`environment/length`)],
    [`g999-s42`], [105.6], [83.0], [82.2], [80.6], [79.5], [78.2], [77.3], [77.5],
    [`g999-s43`], [103.9], [83.4], [81.7], [81.4], [79.2], [78.0], [78.1], [77.7],
    [`g1-s42`], [107.4], [85.5], [84.6], [84.2], [82.6], [82.5], [82.1], [81.0],
    [`g1-s43`], [112.2], [87.0], [84.5], [83.7], [85.3], [83.9], [82.8], [82.0],
  ),
  caption: [Training curves: each column is the mean over the 500 epochs from that epoch on (the last block ends at e3786).],
)

The monitors were still climbing at the end. For scale, the reference has about 2B steps behind it (500M on the old rules, then 1.5B on these), against each cell's 500M.

*Game length.* Without the discount, games run longer on both seeds. Paired game by game on the same 5,000 finals (95% intervals):
#figure(
  btable(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    header: ([Seed], [Vs F, all games], [Vs F, both won], [Vs ref, all games], [Vs ref, both won]),
    [42], [+4.64 ± 0.35], [+4.65 ± 0.35 (4,867)], [+1.64 ± 0.43], [+2.68 ± 0.82 (1,237)],
    [43], [+2.64 ± 0.36], [+2.65 ± 0.36 (4,839)], [+1.73 ± 0.43], [+2.92 ± 0.83 (1,270)],
  ),
  caption: [Turns per game, `g1` minus `g999` within a seed, on the same game ids. "Both won" keeps the games both cells won (their count in brackets).],
) <tab:recipe-discount-length>

- *It is the arm, not the seed.* The same arm on two seeds is much closer: `g999-s43` plays 0.47 ± 0.35 turns longer than `g999-s42` against F and 0.18 ± 0.43 shorter against the reference. `g1`'s two seeds are 1.5 turns apart against F (75.5 and 74.0), still under the smaller of the two arm gaps.
- *Self-play agrees, from the start.* `g1`'s self-play episodes are longer than `g999`'s in all 16 blocks of the training curves.
- *It is in how the nets win.* Against F, which both arms beat 98–99% of the time, the gap is the same on the games both won: the discounted net closes out a won game 2.6–4.6 turns sooner, as E[γ#super[T] · win] asks.

#block(sticky: true)[*Strength.* Level. Head to head within each seed, on the cells' last checkpoints:]
#figure(
  btable(
    columns: (auto, auto, auto),
    align: (left, right, left),
    header: ([Seed], [`g1` vs `g999`], [Record]),
    [42], [50.02 #ci[48.63][51.41]], [`98e98e0d`],
    [43], [49.94 #ci[48.55][51.33]], [`e0b8011e`],
    [Pooled], [49.98 #ci[49.0][51.0]], [10,000 games],
  ),
  caption: [`g1` (no discount) against `g999` (γ = 0.999) on `heldout-2p-fr`, 5,000 games a row, each seed's two cells at e3786. No game hit the turn limit.],
) <tab:recipe-discount-h2h>

Pooled over the seeds, `g1` scores 98.34 against F and 43.5 against the reference, `g999` 98.56 and 44.1; the four cells' intervals all overlap, and the monitors climbed alike. In the head to head, too, the length follows the arm: games `g1` wins run 1.98 ± 0.75 (s42) and 1.33 ± 0.76 (s43) turns longer than the games `g999` wins.

*Reading.* The discount does what its objective says: it makes the net win sooner, 2.6–4.6 turns sooner against F and 1.6–1.7 against the reference, without costing or buying any strength at 500M steps. Either γ trains an equally strong 1v1 net, so the choice comes down to the game length Marco wants.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/discount-2p.yaml
catanrl run configs/experiments/discount-2p-h2h.yaml   # g1 vs g999 within each seed
```)
