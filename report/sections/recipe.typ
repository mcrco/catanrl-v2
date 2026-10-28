#import "../macros.typ": *

= the ppo recipe <sec:recipe>

what training recipe makes the strongest raw policy, and what does it reach? self-play PPO from scratch on the `graph-s` net, v1’s graph2 run ported to v2: 95% vs F in 2p whether or not the net sees the opponent's hidden cards, and a 78.55 score share against three F in 4p. the one change since is the critic: lambda-return targets, undiscounted, now the default for every run.

== the recipe

- *network.* `graph-s`: canopy nexus-v3-style heterogeneous message passing (node↔node, node↔tile, node↔edge), hidden 96, 3 layers. one shared trunk feeds the policy head and a per-seat value head: one sigmoid per seat, that seat's win probability. the acting row's prediction fills the waiting seats' values, so GAE on each seat's own trajectory costs one network call per game step. 541,540 parameters. @sec:arch compares it with other families.
- *training.* pufferlib 3 PPO with two small patches (losses over acting rows only, a value-loss hook). pure self-play: every seat learns. muon at 5e-3 with a cosine to 5e-4, momentum 0.95; γ = 0.999 per decision, GAE 0.95; 2 update epochs, clip 0.2, entropy 0.005; prioritized 64-step segments; fp32.
- *batch.* v1’s graph2 run exactly: 1032 games × 2 rows × 64 steps = 132,096 agent steps per epoch, 16 optimizer steps per epoch, 3786 epochs. an agent step is one seat's row at one game step, so a 2p decision is two agent steps and a 4p decision four. `configs/experiments/puffer-graph2.yaml` lists every deviation from v1 and why.
- *critic.* lambda-return targets, undiscounted (`critic_targets: lambda`, `critic_gamma: 1` in `configs/train/ppo.yaml`), the default since the critic fix (@exp:recipe-critic-fix). earlier runs used V-trace targets at γ = 0.999; their yamls pin those values, so their config hashes and records are unchanged.
- *budgets.* 2p: 500M agent steps, 7.95 h. 4p: 1B agent steps = 249M decisions, 9.3 h. short cells for ablations: 100M agent steps at 2p (about 1.6 h for the graph nets) and 750 epochs = 49M decisions at 4p (about 1.9 h).

#figure(
  image("../figures/select-curves.png", width: 100%),
  caption: [training-time selection curves (argmax policy vs F on the `select-*` suites, every 50 epochs). every run keeps improving through the second half of the cosine, so matched comparisons need the whole budget. the 2p `public` variant (orange) tracks `full` (blue) throughout. per-epoch logs on wandb: #wb("puffer-graph2-full-2fdb6f44")[puffer-graph2/full], #wb("puffer-graph2-public-3ec072be")[puffer-graph2/public], #wb("ppo-4p-main-6349dc93")[ppo-4p/main] (no local mirror) and #wb("ppo-mixed-main-f4b771c0")[ppo-mixed/main] (the generalist, @sec:generalist).],
) <fig:curves>

== two players

=== 2p at the full level: `puffer-graph2/full` #chip("done") <exp:recipe-2p-full>
#meta(
  dates: "2026-10-14 12:37–20:34 (training); 2026-10-17 – 2026-10-18 (finals)",
  owner: "marco",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-full-2fdb6f44", "puffer-graph2/full"),),
  records: ("ff8528a4", "67869e7a"),
)
#tldr[v1’s graph2 run, reproduced in v2 from scratch at the `full` level: 95.16 vs F played raw (unfair: it sees F's dev cards) and 95.06 belief-averaged (fair). hidden information is worth under a point in 2p, and v2 lands 5 points above v1’s warm-started reference.]

*question.* does v2’s PPO stack, trained from scratch, reach v1’s graph2 result?

*setup.* the recipe above at the `full` level, graph2’s own setting, so the comparison with v1 is like for like. 500M agent steps in 7.95 h; best checkpoint e3350 by `select-2p` (96.8).

*result.* on `heldout-2p` (5,000 games each), the raw `full` policy scores *95.16* #ci[94.53][95.72] vs F (`ff8528a4`, unfair). belief-averaged over 16 sampled worlds (`BELIEF`) it scores *95.06* #ci[94.42][95.63] (`67869e7a`, fair).

*reading.* the fair, belief-averaged policy loses nothing against the oracle, which is v1’s "hidden information is worth under a point" again. v2 trains from scratch, with no DAgger, and lands 5 points above v1’s reference (89.65, @exp:v1-reference). this net, belief-averaged, is the deployed 1v1 bot (@sec:deploy). the run's finals also hold two head-to-head search rows, `04fd2c25` and `2aaf2eb7` (@sec:search).

#repro(```sh
catanrl run configs/experiments/puffer-graph2.yaml --only full
# its finals include the two search rows (MCTS 1024 and 4096 sims), about 11.5 h of GPU on top of training
```)

=== 2p at the public level: `puffer-graph2/public`, the fair raw bot #chip("done") <exp:recipe-2p-public>
#meta(
  dates: "2026-10-18 18:03 – 2026-10-19 02:06",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-public-3ec072be", "puffer-graph2/public"),),
  records: ("dc8a611a",),
)
#tldr[the same recipe at the `public` level, which in 2p is fair without belief averaging: 94.96 vs F as a plain policy, level with the `full` net. the opponent's dev cards are worth nothing measurable in 2p. one forward pass, no sampled worlds, no tracker: this is the bot to serve raw.]

*question.* can a fair network skip belief averaging and lose nothing?

*setup.* at the `public` level the net sees the opponent's exact hand, which in 2p is a perfect card counter's knowledge; what `full` adds is only the opponent's dev cards in hand. same recipe and budget as `full`: 7.95 h at 17.8k steps/s. best checkpoint e3600 (select 96.5, against `full`'s 96.8 at e3350; mean over epochs ≥ 3000: 94.4 against 95.1). the two select curves lie on top of each other (@fig:curves).

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, left, right, right, left),
    header: ([agent (vs F, `heldout-2p`)], [sees], [games], [vs F], [record]),
    [`public` net, raw (`POLICY`, e3600)], [fair], [5,000], [*94.96* #ci[94.3][95.5]], [`dc8a611a`],
    [`full` net, belief-averaged (`BELIEF`, 16 worlds, e3350)], [fair], [5,000], [95.06 #ci[94.42][95.63]], [`67869e7a`],
    [`full` net, raw (e3350)], [F's dev cards], [5,000], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
  ),
  caption: [what hidden information is worth in 2p: the three observation levels of the same recipe.],
)

*reading.* the one thing `public` can't see, the opponent's dev cards, is worth nothing measurable in 2p. the hidden-feature head reached the same conclusion from the other side: the hand is inferable, the dev cards are not, and the policy didn't need them (@sec:hidden). its search rows (`72819d1a`, `0ef1b81e`, `831eea84`) are in @sec:search.

#repro(```sh
catanrl run configs/experiments/puffer-graph2.yaml --only public
```)

=== the 1v1 bot on openhex's 1v1 rules, \~1b more steps, judged against its start #chip("done") <exp:recipe-2p-openhex>
#meta(
  dates: "2026-10-24 10:46 – 2026-10-25 03:56",
  owner: "marco",
  config: "configs/experiments/ppo-2p-fr.yaml",
  wandb: (("ppo-2p-fr-full-fef9516c", "ppo-2p-fr/full"),),
  records: ("e435b80e", "6c4f23cb", "3cd19d23", "fa618736", "36bd2384", "db519a9e", "8a137f56", "ffcc5c1a"),
)
#tldr[openhex's 1v1 queue plays friendly robber and balanced dice; our 2p bots trained on neither. marco: train on friendly robber from now on, continue the deployed 1v1 net for \~1b more steps on it, and judge it against its start (the champion) as well as F. result: 63.2% head to head against its start (\~61.7% after correcting for the start's untrained friendly-robber input), 98.8% vs F against the start's 97.3%, still improving at the end.]

*setup.*
- *the rules.* from openhex's source: its 1v1 queue is 15 vp, discard above 9, friendly robber, balanced dice; its 4p queue is 10 vp, discard above 7, normal robber, random dice, which is our 4p setup already. openhex's friendly robber shields an opponent with *public* points ≤ 2. our engine used *actual* points < 3, which counts hidden victory point cards (a mismatch, and the legal moves leak whether the opponent holds one), so the engine now uses openhex's rule (a knight is judged on the points before it takes largest army, as openhex does). balanced dice are in the engine too (marco: "do balanced dice before 1v1 retrain"), following openhex's code: a 36-card deck refilled when 13 or fewer are left, a recency penalty on totals seen in the last 5 rolls, and per-player seven balancing; search chance nodes use the state's own odds. `base-2p-fr` is openhex's 1v1 exactly: 15 vp, discard above 9, friendly robber, balanced dice. random-dice, normal-robber games play bit-identically to the old engine. the network gets no dice-state input, so the policy plays the dice blind and only the search reads them. the two recorded openhex games say balanced dice but were played on fair dice (an openhex bug, since fixed); the bridge falls back to fair dice if a roll is impossible under the balanced deck.
- *the run.* `puffer-graph2/full@best` (e3350, the deployed 1v1 net) continued on `base-2p-fr` for \~1b agent steps, the recipe otherwise unchanged (including its discounted v-trace critic, γ = 0.999, kept to match the start; not the undiscounted default since @exp:recipe-critic-fix, a choice i made without asking. the two continuations below kept it too), at \~17.8k agent steps/s (\~17.5 h with monitors and finals).
- *baselines.* the start vs F on `select-2p-fr`: *96.3* #ci[94.9][97.3] (`8a137f56`). its friendly-robber input was never on in training: a copy with that input zeroed beats it *51.5* #ci[50.2][52.9] over 5,000 games (`ffcc5c1a`), so part of any gain over the start is just learning to ignore that input; the final head to head subtracts 1.5 pp for it.
- *identity.* the merge moved the code ids every record is keyed under (catanrl `55ba5984`, engine `a9b40d9c`, train `55588933`), so new rows pair with older records only through `tools/search_probe.py --pair` (old-rule games are bit-identical).
- *the champion eval (marco).* ppo stagnation has been read from the win rate vs F, which sits near its 95% ceiling. every select now also plays the current checkpoint against a champion (here the start), 1,000 games on the same boards each time: ±3.1 pp, paired game by game across checkpoints. the finals include the start under the new rules, since it never trained on them. at e0 the champion row reads 50.0 (`fa6ea360`), as it must.

*result.* trained 2026-10-24 10:46 to 2026-10-25 \~03:25 (7,600 epochs, 1.0b agent steps at \~17.8k/s), finals by 03:56. the champion row climbed the whole way and was still rising at the end; best is the last checkpoint, e7600.
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    header: ([epochs], [0–450], [1000–1450], [2000–2450], [3000–3450], [4000–4450], [5000–5450], [6000–6450]),
    [vs the start], [49.5], [51.2], [54.4], [56.1], [57.9], [60.5], [61.9],
    [vs F], [96.5], [97.2], [97.3], [97.6], [98.2], [98.0], [98.2],
  ),
  caption: [the monitors during training, `select-2p-fr`, mean of ten selects (10,000 games) per block. the last block, e7000–7600, reads 63.2–64.4 vs the start and 98.4–98.6 vs F.],
)
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([final on `heldout-2p-fr` (5,000 games each)], [e7600], [the start], [paired]),
    [POLICY vs F], [*98.84* #ci[98.50][99.10] (`e435b80e`)], [97.32 #ci[96.83][97.73] (`3cd19d23`)], [+1.52 pp, 126 vs 50, p ≈ 1e-8],
    [BELIEF (16 worlds) vs F], [*98.76* #ci[98.41][99.03] (`6c4f23cb`)], [97.30 #ci[96.81][97.71] (`fa618736`)], [+1.46 pp, 127 vs 54, p ≈ 6e-8],
    [POLICY, e7600 vs the start], [*63.2* #ci[61.8][64.5] (`36bd2384`)], [—], [≈ 61.7 after the 1.5 pp input correction],
    [BELIEF, e7600 vs the start], [*62.2* #ci[60.9][63.5] (`db519a9e`)], [—], [≈ 60.7 after the correction],
  ),
  caption: [the finals. the start is `puffer-graph2/full@e3350`, the deployed 1v1 net, on the same rules and boards.],
)

*reading.* another billion steps on openhex's 1v1 rules made the deployed 1v1 net clearly stronger: about 62% head to head against its own start after correcting for the start's never-trained friendly-robber input (roughly +85 elo), and its losses to F more than halved (2.7% → 1.2%). the champion row was still rising when the learning rate reached its floor. a second billion steps at that learning rate gained little more (@exp:recipe-2p-openhex-3). none of this is measured inside the search yet: the deployed openhex bots run MCTS-1024 and BELIEF on the old engine and the old net.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/ppo-2p-fr.yaml
```)

=== the same net, another \~1b steps from e7600 at the floor learning rate #chip("done") <exp:recipe-2p-openhex-3>
#meta(
  dates: "2026-10-25 18:49 – 2026-10-26 11:59",
  owner: "marco",
  config: "configs/experiments/ppo-2p-fr3.yaml",
  wandb: (("ppo-2p-fr3-full-e95ea1db", "ppo-2p-fr3/full"),),
  records: ("507970d7", "cbba2d10", "7c0d2f67", "b38e49dc", "32d82b76", "f2f3858d"),
)
#tldr[a second billion steps, continued at `ppo-2p-fr`'s final learning rate, barely moved the 1v1 net. the curve against e7600 was flat for 4,000 epochs, then rose slowly to \~52–53. the checkpoint picked as best, e4350, is level with e7600 on held-out boards: 51.0 #ci[49.6][52.4] head to head, 98.9 vs 98.8 against F.]

*question.* does the 1v1 net keep improving on openhex's 1v1 rules if training just continues where `ppo-2p-fr` ended?

*setup.* `ppo-2p-fr` exactly, initialized from `ppo-2p-fr/full@e7600` with a fresh optimizer. the learning rate is held at `ppo-2p-fr`'s floor, 0.00025, for 7,600 epochs (\~1b agent steps); there is no cosine. the champion row plays every 50th checkpoint against e7600 (1,000 games, ±3.1 pp, paired across checkpoints; ±1 pp per 10-checkpoint block). vs F is measured alongside. finals on `heldout-2p-fr`, 5,000 games each: `@best` vs F, `@best` vs e7600, and `@best` vs the original deployed net `puffer-graph2/full@e3350`, each as POLICY and as BELIEF. at e0 it reads 50.0 vs the champion and 98.8 vs F, and the first updates run at kl 0.0006–0.0009, where `ppo-2p-fr` ended. \~17 h.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right, right),
    header: ([epochs], [0–450], [1000–1450], [2000–2450], [3000–3450], [4000–4450], [5000–5450], [6000–6450], [7000–7450]),
    [vs e7600], [50.1], [50.0], [49.1], [49.9], [51.2], [51.9], [52.0], [52.8],
    [vs F], [98.5], [98.7], [98.3], [98.5], [98.7], [98.6], [98.9], [98.7],
  ),
  caption: [the monitors during training, `select-2p-fr`, mean of ten selects (10,000 games, ±1 pp) per block. the blocks in between read the same; the last three checkpoints, e7500–7600, read 51.1. as elo against e7600 the slope is −0.4 ± 0.9 per 1,000 epochs over e0–4000 and +3.5 ± 0.4 over the whole run. `ppo-2p-fr` gained +10 to +17 per 1,000 epochs against its start throughout.],
)
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([final on `heldout-2p-fr` (5,000 games each)], [e4350 (this run's best)], [e7600 (its start)], [paired]),
    [POLICY vs F], [*98.92* #ci[98.59][99.17] (`507970d7`)], [98.84 (`e435b80e`)], [+0.08 pp, 48 vs 44, p = 0.75],
    [BELIEF (16 worlds) vs F], [*98.94* #ci[98.62][99.19] (`cbba2d10`)], [98.76 (`6c4f23cb`)], [+0.18 pp, 53 vs 44, p = 0.42],
    [POLICY vs `puffer-graph2/full@e3350`], [*64.7* #ci[63.3][66.0] (`32d82b76`)], [63.2 (`36bd2384`)], [+1.5 pp, 779 vs 704, p = 0.055],
    [BELIEF vs `puffer-graph2/full@e3350`], [*63.5* #ci[62.2][64.8] (`f2f3858d`)], [62.2 (`db519a9e`)], [+1.3 pp, 810 vs 744, p = 0.099],
    [POLICY, e4350 vs e7600], [*51.0* #ci[49.6][52.4] (`7c0d2f67`)], [—], [p = 0.16],
    [BELIEF, e4350 vs e7600], [*51.2* #ci[49.8][52.6] (`b38e49dc`)], [—], [p = 0.098],
  ),
  caption: [the finals. `puffer-graph2/full@e3350` is the deployed 1v1 net, which `ppo-2p-fr` started from; the rows against it pair game for game with `ppo-2p-fr`'s finals on the same boards.],
)

*reading.* the second billion steps bought little. for the first 4,000 epochs the net did not move at all against e7600. it then rose slowly to \~52–53 by e6000–7000, which is \~15–20 elo, about a fifth of what the first billion bought (\~+100 elo against its start). every final points the same way: e4350 is ahead of e7600 by 1.0–1.5 pp head to head and against the old net, and by 0.1–0.2 pp against F. none of these is significant on its own. so e7600 is roughly where this recipe levels off on openhex's 1v1 rules, and e4350 is at most a little better.

e4350 was picked as best because its one select row read 55.5, the highest of 153 rows of ±3.1 pp each. picking the maximum favours a lucky row, and on held-out boards e4350 reads 51.0, about the mean of its block. the checkpoints from e6000–7000 averaged 52–53 on select, but none of them has a held-out final. picking best from the block means instead of the single highest row would avoid this (my suggestion, not queued).

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/ppo-2p-fr3.yaml
```)

=== the first try: a fresh cosine from the peak learning rate #chip("stopped") <exp:recipe-2p-openhex-2>
#meta(
  dates: "2026-10-25 14:03 – 18:48 (stopped at e2010)",
  owner: "marco",
  config: "configs/experiments/ppo-2p-fr2.yaml",
  wandb: (("ppo-2p-fr2-full-0b305e27", "ppo-2p-fr2/full"),),
  records: (),
)
#tldr[the continuation restarted the learning rate at its peak, 0.0025, and fell to 40–43% against its own start within 50 epochs. it stayed there through e2000, so marco stopped it and restarted at the floor (@exp:recipe-2p-openhex-3).]

*question.* the same as @exp:recipe-2p-openhex-3.

*setup.* `ppo-2p-fr` exactly, initialized from `ppo-2p-fr/full@e7600`, with a fresh optimizer and a fresh cosine (0.0025 → 0.00025 over 7,600 epochs), as `ppo-2p-fr` had done from its own start. the monitors and planned finals were the same as @exp:recipe-2p-openhex-3.

*result.* stopped at e2010 (26%), before any finals. the champion row fell to 43.5 by e50 and stayed at 40–43. the 10-checkpoint block means are 43.3, 41.2, 40.6, 42.1 and 40.0, each over 10,000 games (±1 pp). vs F fell from 98.8 to \~97.7.

*reading.* the learning-rate restart caused this, not the extra steps:

- e7600 ended at the floor learning rate (0.00025): kl 0.001 per update, entropy 0.048.
- the restart at 0.0025 pushed kl to 0.02–0.03 at once, about 20 times larger.
- `ppo-2p-fr`'s own restart from e3350 (entropy 0.084, a broader policy) ran at kl 0.015 and stayed at 48–51 early.

a sharp policy that ended at a low learning rate loses \~8–10 pp head to head if the learning rate jumps back to the peak. the cosine would have returned to the floor by e7600, so this run's end would have tested a warm restart, not plain continued training. lesson: set a continuation's learning rate from its start's final kl and entropy.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/ppo-2p-fr2.yaml
```)

== four players

=== 4p specialist: `ppo-4p/main` #chip("done") <exp:recipe-4p>
#meta(
  dates: "2026-10-14 20:55 – 2026-10-15 06:16",
  owner: "marco",
  config: "configs/experiments/ppo-4p.yaml",
  wandb: (("ppo-4p-main-6349dc93", "ppo-4p/main (no local mirror)"),),
  records: ("e5b302ee", "b07e5bd9", "95570c74"),
)
#tldr[the 2p recipe on 4p BASE at the `private` level, the only fair level past two players: a 78.55 score share against three F (parity 25) and 100.0 against three random players. its V-trace critic was biased high, which the critic fix below corrects.]

*question.* does the 2p recipe carry over to four players unchanged?

*setup.* `puffer-graph2`’s recipe on `base-4p` (BASE, 10 vp, discard limit 7), with all four seats learning. `private` level. 1B agent steps = 249M decisions (3800 epochs), about the decisions of the 2p run; 9.3 h. best checkpoint e3650 by `select-4p`.

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([lineup (`heldout-4p`, 4,000 games each)], [share of the policy's slots], [record]),
    [policy vs F, F, F], [*78.55* #ci[77.25][79.79]], [`e5b302ee`],
    [two copies vs F, F], [93.2 together (40.45 + 52.75)], [`b07e5bd9`],
    [policy vs R, R, R], [100.0 #ci[99.9][100.0]], [`95570c74`],
  ),
  caption: [the 4p specialist. parity is 25 per slot.],
)

*reading.* the two identical copies split 40.45 / 52.75 because of the slot order, not because they differ: the copy in slot 0 sits right after an F (@sec:front, @sec:generalist). the critic summed its four seats' win probabilities well above the realized sum, a bias from V-trace targets on waiting rows (@exp:recipe-critic-fix).

#repro(```sh
catanrl run configs/experiments/ppo-4p.yaml
```)

== the critic

=== the 4p critic fix: lambda-return targets, undiscounted #chip("done") <exp:recipe-critic-fix>
#meta(
  dates: "2026-10-18 14:58–16:00",
  config: "configs/experiments/ppo-4p-critic-fix.yaml",
  wandb: (("ppo-4p-critic-fix-lambda-3eb6d5cf", "lambda"), ("ppo-4p-critic-fix-lambda-critic-g1-57cb4f34", "lambda-critic-g1"), ("ppo-4p-critic-fix-lambda-g1-d5d2fc7f", "lambda-g1"), ("ppo-4p-critic-fix-lambda-adamw-e6f3dfe2", "lambda-adamw")),
  records: ("60cd10b6", "7d9374ed", "f4be1c20", "1d639013", "8e93695a"),
)
#tldr[`ppo-4p`'s four seats' win probabilities summed to about 1.2 against a realized 0.87. lambda-return critic targets put the sum on the truth within about 20 epochs, and undiscounted the sum is 1.00; the policy doesn't move. it is now the default for every run.]

*question.* why is the 4p critic biased high, and what fixes it without touching the policy?

*diagnosis.* the seat sum was about 1.2 against a realized 0.87 at epoch 300, and 0.93 against 0.90 by the end. cpu replications found two causes.
- V-trace ratios on waiting rows. only the acting seat's row gets min(ρ, 1); the other three get 1. so each seat's target values the game as if that seat alone played the improved policy, and the four one-sided gains compound through bootstrapping.
- muon's fixed-size update of the 1 × 128 value output layer. the predicted sum alternates high and low every optimizer step, and the rollout network lands on the high side every epoch.

*setup.* 100-epoch fine-tunes of `ppo-4p/main@latest` at its final learning rate (5e-4, constant), with lambda-return critic targets (every ratio 1; the policy's advantages stay V-trace's). four arms of about 15 minutes each:
- `lambda`: γ = 0.999 for policy and critic.
- `lambda-critic-g1`: the critic undiscounted, the policy at γ = 0.999.
- `lambda-g1`: policy and critic undiscounted.
- `lambda-adamw`: `lambda` with the value output layer on AdamW instead of muon.

*result.*
#figure(
  btable(
    columns: (auto, 1fr, auto, auto, auto),
    align: (left, left, right, right, left),
    header: ([arm], [critic targets], [seat sum vs realized], [select-4p], [record]),
    [start (`ppo-4p/main`)], [V-trace, γ = 0.999], [0.93 vs 0.90], [79.9], [`60cd10b6`],
    mid,
    [`lambda`], [lambda, γ = 0.999], [0.906 vs 0.899], [80.4], [`7d9374ed`],
    [`lambda-critic-g1`], [lambda, critic γ = 1], [0.998 vs 1.000], [79.9], [`f4be1c20`],
    [`lambda-g1`], [lambda, policy and critic γ = 1], [1.000 vs 1.000], [80.4], [`1d639013`],
    [`lambda-adamw`], [lambda, γ = 0.999, output layer on AdamW], [0.906 vs 0.898], [79.1], [`8e93695a`],
  ),
  caption: [the critic-fix arms at the end of the 100-epoch fine-tune. seat sum: the four seats' predicted win probabilities summed, against the realized sum. select-4p: the policy's share vs F, F, F, n=800 per point (±2.8). the start row's record sits in all four arms' run dirs, so `catanrl verify` needs its path, e.g. `runs/ppo-4p-critic-fix/lambda/evals/select-4p/60cd10b66aea063c.json`.],
)
every arm reaches its realized sum within about 20 epochs and stays there (@fig:critic); `lambda` started at 0.920 against 0.914 in its first epochs. the rollout-vs-trained gap falls from 0.004 to 0.000 in all four arms. per-slot calibration at the end of `lambda-critic-g1`: the seat to move predicted 0.284 against a realized 0.276, the other three within 0.004. clip fraction, entropy and game length (79–80 turns) don't change.

#figure(
  image("../figures/critic-fix.png", width: 100%),
  caption: [left, `ppo-4p-critic-fix`: the four seats' predicted win probabilities summed, per fine-tune epoch, against the realized sum (dashed: about 0.90 at γ = 0.999, 1 undiscounted). lambda-return targets put the sum on the truth in every arm within about 20 epochs; the AdamW output layer changes nothing. right, from scratch under the new default (@exp:recipe-critic-from-scratch): `aux-hidden-4p/plain` (purple) against `ppo-4p/main`'s V-trace critic over its first 750 epochs (orange; its own realized sum is about 0.90, the dashed 1.0 is the undiscounted critic's). both overshoot early; the lambda critic is on its realized sum from epoch 220, while the V-trace critic is still 0.16 above its own at epoch 750.],
) <fig:critic>

*reading.* the targets alone fix it. they also remove muon's phase lock, so the AdamW arm buys nothing. the undiscounted policy plays games of the same length, so γ = 1 changes nothing the policy optimises at this horizon. per-seat win probabilities that sum to one are what max#super[n] backups in the search want, so `critic_targets: lambda` and `critic_gamma: 1` became the defaults in `configs/train/ppo.yaml`.

#repro(```sh
catanrl run configs/experiments/ppo-4p-critic-fix.yaml
# four arms, about 15 min each; each fine-tunes ppo-4p/main@latest, so ppo-4p must be trained first
```)

=== the lambda critic from scratch #chip("done") <exp:recipe-critic-from-scratch>
#meta(
  dates: "2026-10-19",
  config: "configs/experiments/aux-hidden-4p.yaml",
  wandb: (("aux-hidden-4p-plain-f07eb81c", "aux-hidden-4p/plain"), ("aux-hidden-4p-aux-4dcfd020", "aux-hidden-4p/aux"), ("lstm-4p-main-4cae1a6d", "lstm-4p/main")),
  records: (),
)
#tldr[trained from scratch, the undiscounted lambda critic overshoots early (a seat sum of 1.83 at epoch 21) and is on the realized 1.00 by epoch 220, where it stays. `ppo-4p`'s V-trace critic overshoots too and never comes back (0.94 against 0.90 at epoch 3800). the fine-tunes couldn't show the transient, since their critics started trained.]

*question.* does the new default calibrate when the critic starts from random weights?

*setup.* the first from-scratch runs under the new default: the two `aux-hidden-4p` arms, `ppo-4p`'s recipe at 750 epochs (49.2M decisions). the arms are a hidden-information experiment (@sec:hidden); here only their critics are read, against `ppo-4p/main`'s first 750 epochs. `lstm-4p` (@sec:markov) repeats the check on the recurrent net. the numbers are the training metric `critic/value_sum_4p`, not eval records.

*result.*
#figure(
  btable(
    columns: (auto, auto, 1fr, auto, 1fr),
    align: (left, left, left, left, left),
    header: ([run], [critic], [early overshoot], [on the realized sum], [later]),
    [`aux-hidden-4p/plain`], [lambda, γ = 1], [1.83 at e21, 1.39 at e50, 1.31 at e100, 1.18 at e200], [1.00 by e220], [mean 1.000 over e300–750, within 0.06 at every epoch],
    [`aux-hidden-4p/aux`], [lambda, γ = 1], [—], [—], [mean 0.99],
    [`lstm-4p/main`], [lambda, γ = 1], [1.59 at e21], [1.00 by e250], [mean 1.000 over e300–750],
    mid,
    [`ppo-4p/main`], [V-trace, γ = 0.999], [1.27 at e100, 1.21 at e300, 1.09 at e500, 1.06 at e750], [never], [0.94 against a realized 0.90 at e3800],
  ),
  caption: [the four seats' predicted win probabilities summed, by epoch, from random weights. the undiscounted lambda critic's realized sum is 1.00; the V-trace critic's is about 0.90.],
)

*reading.* the early overshoot is a from-scratch transient, and both target rules produce it; only the lambda critic recovers. it lands at epoch 220, while the learning rate is still four fifths of its peak. explained variance is 0.82–0.85 at the end, as before. the right panel of @fig:critic shows the two curves.

#repro(```sh
catanrl run configs/experiments/aux-hidden-4p.yaml --only plain
```)

== the discount

=== discount vs no discount, from scratch #chip("done") <exp:recipe-discount>
#meta(
  dates: "2026-10-26 13:08 – 2026-10-27 23:05 pdt",
  owner: "marco",
  config: "configs/experiments/discount-2p.yaml",
  wandb: (("discount-2p-g999-s42-5057ed5a", "discount-2p/g999-s42"), ("discount-2p-g1-s42-5f1a556d", "discount-2p/g1-s42"), ("discount-2p-g999-s43-77e69c7e", "discount-2p/g999-s43"), ("discount-2p-g1-s43-12bb8a44", "discount-2p/g1-s43"), ("discount-2p-h2h-main-e52c5b42", "discount-2p-h2h")),
  records: ("57d449b3", "31ead730", "2109921e", "312910df", "9418974e", "05cc8756", "548ee03e", "e6bf7ad4", "98e98e0d", "e0b8011e"),
)
#tldr[every ppo run so far discounts the policy at γ = 0.999 per decision, inherited from v1. marco asked whether training without the discount does better. two arms, γ = 0.999 vs γ = 1 for both the policy and the critic, trained from scratch on openhex's 1v1 rules with two seeds each. *the discount makes games end faster*: without it, games against F run 4.6 and 2.6 turns longer on the two seeds (paired over the same 5,000 games, ±0.35), 1.6–1.7 turns longer against the reference, and self-play episodes are longer in every 500-epoch block of both seeds. the same arm on two seeds differs by half a turn. *it doesn't change strength*: head to head the arms are dead level, 50.02% and 49.94% on the two seeds (5,000 games each), and level against F and the reference.]

*question.* marco's main question: does the discount make games end faster? second: does dropping it play better?

*why it might.* catan needs no discount: every game ends, the only reward is the win at the end, and gae's λ = 0.95 already does most of the variance reduction. but γ < 1 changes the objective from p(win) to E[γ#super[T] · win]. a win is worth more the sooner it comes (\~0.85 at the opening of a 160-decision game), which may push the policy towards faster, riskier wins. the only earlier test was 100 epochs of 4p fine-tuning (`lambda-g1` in @exp:recipe-critic-fix: the same game length and the same select score), far too short to say what training from scratch does.

*setup.*
- *arms.* `g999`: γ = 0.999 for the policy and the critic. `g1`: γ = 1 for both. the critic targets are lambda returns in both (the default), so only the discount differs.
- *recipe.* `puffer-graph2`'s, from scratch on `base-2p-fr` (15 vp, discard above 9, friendly robber, balanced dice), `full` level, 500m agent steps (3,800 epochs, cosine 0.005 → 0.0005).
- *seeds.* 42 and 43 for each arm; the seed also draws the 4,096 training boards. two seeds of this recipe have never been compared, so the spread between seeds is unknown; an arm difference counts only if both seeds agree in sign and the pooled head to head clears it.
- *monitors* every 50 epochs on `select-2p-fr` (1,000 games, ±3.1 pp at 50%): vs F, and vs a fixed reference, `ppo-2p-fr3/full@e4350`, so all four curves share one yardstick.
- *finals* on `heldout-2p-fr`, 5,000 games each, on each cell's *last* checkpoint (chosen before the run, so no winner's curse): vs F (±0.3 pp near 98) and vs the reference (±1.4 pp at 50%). once all four cells are done, a separate eval plays `g1` against `g999` within each seed (5,000 games, paired mde \~2 pp at 80%) and compares game lengths.
- *game length* (the main readout, marco). every cell's finals play the same 5,000 game ids (board, seats and game seed), so turns per game pair up game by game between the two arms of a seed: vs F, which the net beats 98–99% of the time, this is how fast it closes out a won game; vs the reference it is the length of a close game. from `g999-s42`'s finals (sd 11.5 turns vs F), the paired difference within a seed resolves to about ±0.45 turns (95%) even if games on one board are uncorrelated; the gap between the two seeds of one arm is the real yardstick. the self-play episode length logged in training (`environment/length`) gives the same comparison epoch by epoch.
- *cost.* \~8.4 h per cell, \~34 h in all, in the order s42-`g999`, s42-`g1`, s43-`g999`, s43-`g1`.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto, 1fr),
    align: (left, left, left, left, left, left, left),
    header: ([cell], [γ], [vs F], [vs ref], [turns vs F], [turns vs ref], [self-play length, last 200 epochs]),
    [`g999-s42`], [0.999], [98.56 #ci(98.19, 98.85)], [43.38 #ci(42.01, 44.76)], [70.85], [77.42], [77.7],
    [`g1-s42`], [1], [98.60 #ci(98.24, 98.89)], [44.20 #ci(42.83, 45.58)], [75.48], [79.07], [81.0],
    [`g999-s43`], [0.999], [98.56 #ci(98.19, 98.85)], [44.76 #ci(43.39, 46.14)], [71.31], [77.24], [77.3],
    [`g1-s43`], [1], [98.08 #ci(97.66, 98.43)], [42.80 #ci(41.43, 44.18)], [73.96], [78.97], [82.2],
  ),
  caption: [finals on `heldout-2p-fr`, 5,000 games each, each cell's last checkpoint (e3786); ref = `ppo-2p-fr3/full@e4350`. turns are the mean per game over all 5,000 (sd 11.5 vs F, 13.3 vs ref). self-play length is `environment/length` from training. records `57d449b3` and `31ead730` (`g999-s42`), `2109921e` and `312910df` (`g1-s42`), `9418974e` and `05cc8756` (`g999-s43`), `548ee03e` and `e6bf7ad4` (`g1-s43`), vs F and vs the reference.],
)

the monitors for `g999-s42` (`select-2p-fr`, mean of each 500 epochs) were still climbing at the end: vs the reference 9.8, 22.4, 27.4, 31.0, 35.0, 39.2, 40.9, 42.1; vs F 72.9, 92.6, 94.5, 96.0, 96.8, 97.2, 97.9, 98.3. for scale, the reference has about 2b steps behind it (500m on the old rules, then 1.5b on these), against this cell's 500m. `g1-s42`'s monitors climbed the same way: 9.1, 22.5, 27.9, 30.8, 35.4, 39.7, 42.0, 43.3 vs the reference.

*game length.* without the discount, games run longer, on both seeds. paired game by game on the same 5,000 finals (95% intervals):
#figure(
  btable(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    header: ([seed], [vs F, all games], [vs F, both won], [vs ref, all games], [vs ref, both won]),
    [42], [+4.64 ± 0.35], [+4.65 ± 0.35 (4,867)], [+1.64 ± 0.43], [+2.68 ± 0.82 (1,237)],
    [43], [+2.64 ± 0.36], [+2.65 ± 0.36 (4,839)], [+1.73 ± 0.43], [+2.92 ± 0.83 (1,270)],
  ),
  caption: [turns per game, `g1` minus `g999` within a seed, on the same game ids; "both won" keeps the games both cells won (their count in brackets).],
) <tab:recipe-discount-length>

- *it is the arm, not the seed.* the same arm on two seeds is much closer: `g999-s43` plays 0.47 ± 0.35 turns longer than `g999-s42` against F and 0.18 ± 0.43 shorter against the reference. `g1`'s two seeds are 1.5 turns apart against F (75.5 and 74.0), still under the smaller of the two arm gaps.
- *self-play agrees, from the start.* `environment/length`, mean of each 500 epochs: `g999` 105.6, 83.0, 82.2, 80.6, 79.5, 78.2, 77.3, 77.5 (s42) and 103.9, 83.4, 81.7, 81.4, 79.2, 78.0, 78.1, 77.7 (s43); `g1` 107.4, 85.5, 84.6, 84.2, 82.6, 82.5, 82.1, 81.0 (s42) and 112.2, 87.0, 84.5, 83.7, 85.3, 83.9, 82.8, 82.0 (s43). `g1` is longer in all 16 blocks.
- *it is in how the nets win.* against F, which both arms beat 98–99% of the time, the gap is the same on the games both won: the discounted net closes out a won game about 3–5 turns sooner, as E[γ#super[T] · win] asks.

*strength.* level. head to head within each seed, on the cells' last checkpoints:
#figure(
  btable(
    columns: (auto, 1fr, auto),
    align: (left, right, left),
    header: ([seed], [`g1` vs `g999`], [record]),
    [42], [50.02 #ci(48.63, 51.41)], [`98e98e0d`],
    [43], [49.94 #ci(48.55, 51.33)], [`e0b8011e`],
    [pooled], [49.98 #ci(49.0, 51.0)], [10,000 games],
  ),
  caption: [`g1` (no discount) against `g999` (γ = 0.999) on `heldout-2p-fr`, 5,000 games a row, each seed's two cells at e3786. no game hit the turn limit.],
) <tab:recipe-discount-h2h>

pooled over the seeds, `g1` also scores 98.34 vs F and 43.5 vs the reference, `g999` 98.56 and 44.1; every cell's row sits inside the others' intervals, and the monitors climbed alike (`g1-s43` vs the reference 8.7, 20.7, 26.6, 31.1, 35.8, 37.7, 41.1, 42.2). in the head to head, too, the length follows the arm: games `g1` wins run 1.98 ± 0.75 (s42) and 1.33 ± 0.76 (s43) turns longer than the games `g999` wins.

*reading.* the discount does what its objective says: it makes the net win sooner, about 3–5 turns sooner against F and 1.5 against an equal, without costing or buying any strength at 500m steps. either γ trains an equally strong 1v1 net, so the choice comes down to the game length marco wants.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/discount-2p.yaml
catanrl run configs/experiments/discount-2p-h2h.yaml   # g1 vs g999 within each seed
```)
