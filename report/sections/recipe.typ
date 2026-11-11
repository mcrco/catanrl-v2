#import "../macros.typ": *

= The PPO recipe <sec:recipe>

Self-play PPO from scratch on the `graph-s` net wins 95% against F in 2p, with or without the opponent's hidden cards, and takes a 78.55 score share against three F in 4p. Since then, only the critic changed: undiscounted lambda-return targets are the default.

== The recipe

/ Network: `graph-s`: Canopy nexus-v3-style heterogeneous message passing (node↔node, node↔tile, node↔edge), hidden 96, 3 layers, 541,540 parameters (@sec:arch compares other families). A shared trunk feeds a policy head and a per-seat value head (one sigmoid per seat: its win probability). The acting row's prediction fills the waiting seats' values, so per-seat GAE costs one network call per game step.
/ Training: PufferLib 3 PPO with two small patches (losses over acting rows only, a value-loss hook). Pure self-play: every seat learns. Muon at 5e-3 with a cosine to 5e-4, momentum 0.95; γ = 0.999 per decision, GAE λ = 0.95; 2 update epochs, clip 0.2, entropy 0.005; prioritized 64-step segments; fp32.
/ Batch: v1's graph2 run exactly: 1,032 games × 2 rows × 64 steps = 132,096 agent steps per epoch, 16 optimizer steps per epoch, 3,786 epochs. An agent step is one seat's row at one game step: two per 2p decision, four per 4p decision. `configs/experiments/puffer-graph2.yaml` lists every deviation from v1.
/ Critic: Undiscounted lambda-return targets (`critic_targets: lambda`, `critic_gamma: 1` in `configs/train/ppo.yaml`) since @exp:recipe-critic-fix. Earlier runs pin V-trace targets at γ = 0.999 in their YAMLs, so their config hashes and records are unchanged.
/ Budgets: 2p: 500M agent steps in 7.95 h. 4p: 1B agent steps (249M decisions) in 9.3 h. Ablation cells: 100M agent steps at 2p (about 1.6 h for the graph nets) and 750 epochs (49M decisions) at 4p (about 1.9 h).

#figure(
  image("../figures/select-curves.png", width: 100%),
  caption: [The argmax policy against F on the `select-*` suites, every 50 epochs of training. Every run still improves late in the cosine, so matched comparisons need the whole budget; 2p `public` (orange) tracks `full` (blue). W&B: #wb("puffer-graph2-full-2fdb6f44")[puffer-graph2/full], #wb("puffer-graph2-public-3ec072be")[puffer-graph2/public], #wb("ppo-4p-main-6349dc93")[ppo-4p/main] (no local mirror), #wb("ppo-mixed-main-f4b771c0")[ppo-mixed/main] (the generalist, @sec:generalist).],
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
#tldr[v1's graph2 run, reproduced from scratch at the `full` level, beats F 95.16% played raw (unfair: it sees F's dev cards) and 95.06% belief-averaged (fair): 5 points above v1's warm-started reference.]

*Question.* Does v2's PPO stack, from scratch, reach v1's graph2 result?

*Setup.* The recipe and 2p budget above at the `full` level, graph2's own setting; best checkpoint e3350 by `select-2p` (96.8).

*Result.* Both rows are in the table of @exp:recipe-2p-public (`heldout-2p`, 5,000 games each): raw (`ff8528a4`) and belief-averaged over 16 worlds (`67869e7a`).

*Reading.* As in v1, the fair policy loses nothing against the oracle; without DAgger, v2 beats v1's reference (89.65, @exp:v1-reference). Belief-averaged, this net is the deployed 1v1 bot (@sec:deploy); its search rows `04fd2c25` and `2aaf2eb7` are in @sec:search.

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
#tldr[At the `public` level, fair in 2p without belief averaging, the plain policy beats F 94.96%, level with the `full` net. One forward pass, no sampled worlds, no tracker: the bot to serve raw.]

*Question.* Can a fair network skip belief averaging and lose nothing?

*Setup.* Same recipe and budget as `full`, which adds only the opponent's dev cards (@sec:front): 7.95 h at 17.8k steps/s. Best checkpoint e3600 (select 96.5, against `full`'s 96.8 at e3350; mean over epochs ≥ 3000: 94.4 against 95.1); the curves overlap (@fig:curves).

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
  caption: [The `public` and `full` nets against F on `heldout-2p`.],
)

*Reading.* The opponent's dev cards, all that `public` lacks, are worth nothing measurable in 2p. The hidden-feature head agrees: the hand is inferable, the dev cards are not, and the policy didn't need them (@sec:hidden). Search rows: `72819d1a`, `0ef1b81e`, `831eea84` (@sec:search).

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
#tldr[Continued for \~1B steps on OpenHex's 1v1 rules (friendly robber, balanced dice), the deployed 1v1 net beats its start 63.2% (\~61.7% after correcting for the start's untrained friendly-robber input) and F 98.8% (the start: 97.3%), and was still improving at the end.]

*Question.* Does the deployed 1v1 net still improve with \~1B more steps on OpenHex's rules?

#lead[Setup.]
- *Rules* (Marco: train on OpenHex's rules from now on). Only 2p moves, to `base-2p-fr` (@sec:front): OpenHex's 4p queue (10 VP, discard above 7, normal robber, random dice) is already our `base-4p`.
- *Friendly robber.* Our engine shielded opponents with *actual* points < 3, counting hidden VP cards: a mismatch with OpenHex's *public* points ≤ 2, and a leak of VP cards through the legal moves. It now uses OpenHex's rule, judging a knight on the points before largest army.
- *Balanced dice* (Marco: "do balanced dice before 1v1 retrain") follow OpenHex's code: a 36-card deck refilled at 13 or fewer, a penalty on totals from the last 5 rolls, per-player seven balancing. Only the search reads the dice state (chance nodes use the state's odds); the network has no dice input. OpenHex's two recorded games say balanced but used fair dice (an OpenHex bug, since fixed); the bridge falls back to fair dice when a roll is impossible under the deck.
- *Run.* `puffer-graph2/full@best` (e3350, the deployed 1v1 net) continued for 7,600 epochs (\~1B agent steps), cosine 0.0025 (half the original peak) → 0.00025, at \~17.8k agent steps/s (\~17.5 h with monitors and finals). It kept the start's V-trace critic at γ = 0.999 instead of the new default (@exp:recipe-critic-fix): Claude's choice, made without asking, and kept by both continuations below.
- *Baselines.* The start scores *96.3* #ci[94.9][97.3] against F on `select-2p-fr` (`8a137f56`). Its friendly-robber input was never on in training, and a copy with it zeroed beats the start *51.5* #ci[50.2][52.9] over 5,000 games (`ffcc5c1a`); the final head to head subtracts this 1.5 pp.
- *Identity.* The merge changed the code ids that key every record (catanrl `55ba5984`, engine `a9b40d9c`, train `55588933`); new rows pair with older records only through `tools/search_probe.py --pair`. Old-rule games are bit-identical.
- *Champion eval (Marco).* With the win rate against F near its 95% ceiling, every select also plays a champion (here the start): 1,000 games on fixed boards, ±3.1 pp, paired across checkpoints; 50.0 at e0 (`fa6ea360`).

*Result.* The champion row rose throughout; best is the last checkpoint, e7600.
#figure(
  btable(
    columns: (auto,) + (1fr,) * 7,
    align: (left,) + (right,) * 7,
    header: ([From], [e0], [e1000], [e2000], [e3000], [e4000], [e5000], [e6000]),
    [Vs the start], [49.5], [51.2], [54.4], [56.1], [57.9], [60.5], [61.9],
    [Vs F], [96.5], [97.2], [97.3], [97.6], [98.2], [98.0], [98.2],
  ),
  caption: [Monitors on `select-2p-fr`, each the mean of ten selects (10,000 games) from that checkpoint on (e0–450, e1000–1450, …). The last block, e7000–7600, reads 63.2–64.4 against the start and 98.4–98.6 against F.],
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
  caption: [Finals on `heldout-2p-fr`, 5,000 games each; the start is `puffer-graph2/full@e3350`. Against F: the paired difference, the games only one side won (e7600's first) and McNemar's p.],
)

*Reading.* Clearly stronger: roughly +85 Elo against its start after the input correction, and its losses to F more than halved (2.7% → 1.2%). A second billion steps at the floor learning rate gained little (@exp:recipe-2p-openhex-3). Nothing here is measured inside the search yet; the OpenHex bots then ran MCTS-1024 and BELIEF on the old engine and net (since replaced, @exp:deploy-openhex-v2).

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
#tldr[A second \~1B steps at `ppo-2p-fr`'s floor learning rate barely moved the net: flat against e7600 for 4,000 epochs, then a slow rise to \~52–53. The best checkpoint, e4350, is level with e7600 on held-out boards: 51.0 #ci[49.6][52.4] head to head, 98.9 vs 98.8 against F.]

*Question.* Does the net keep improving if training simply continues from e7600?

#lead[Setup.]
- *Run.* @exp:recipe-2p-openhex from e7600 with a fresh optimizer, the learning rate held at its floor (0.00025, no cosine): 7,600 epochs (\~1B agent steps), \~17 h.
- *Monitors.* Every 50th checkpoint plays e7600 (±3.1 pp) and F. At e0: 50.0 and 98.8; the first updates run at KL 0.0006–0.0009, where `ppo-2p-fr` ended.

#lead[Result.]
#figure(
  btable(
    columns: (auto,) + (1fr,) * 8,
    align: (left,) + (right,) * 8,
    header: ([From], [e0], [e1000], [e2000], [e3000], [e4000], [e5000], [e6000], [e7000]),
    [Vs e7600], [50.1], [50.0], [49.1], [49.9], [51.2], [51.9], [52.0], [52.8],
    [Vs F], [98.5], [98.7], [98.3], [98.5], [98.7], [98.6], [98.9], [98.7],
  ),
  caption: [Monitors on `select-2p-fr`, each the mean of ten selects (10,000 games, ±1 pp) from that checkpoint on (e0–450, e1000–1450, …). Blocks not shown read the same; the last three checkpoints, e7500–7600, read 51.1.],
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
  caption: [Finals on `heldout-2p-fr`, 5,000 games each; rows against `puffer-graph2/full@e3350` pair game for game with `ppo-2p-fr`'s finals. Comparison as in @exp:recipe-2p-openhex (e4350's count first); for e4350 against e7600, p against an even score.],
)

*Reading.* The second billion bought \~15–20 Elo by e6000–7000, a fifth of the first billion's \~+100 against its start. Against e7600 the Elo slope is −0.4 ± 0.9 per 1,000 epochs over e0–4000 and +3.5 ± 0.4 over the run; `ppo-2p-fr` gained +10 to +17 per 1,000 epochs throughout. Every final favours e4350 slightly (1.0–1.5 pp head to head and against the old net, 0.1–0.2 pp against F), none significantly. e7600 is roughly where this recipe levels off on these rules.

*A lucky pick.* e4350 won on one select row of 55.5, the highest of 153 rows of ±3.1 pp; held out, it reads about its block's mean. The e6000–7000 checkpoints averaged 52–53 on select but have no held-out finals. Picking best by block means would avoid this (Claude's suggestion, not queued).

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
#tldr[Restarting the learning rate at its peak, 0.0025, dropped the continuation to 40–43% against its start within 50 epochs, where it stayed through e2000. Marco stopped it and restarted at the floor (@exp:recipe-2p-openhex-3).]

*Question.* The same as @exp:recipe-2p-openhex-3.

*Setup.* @exp:recipe-2p-openhex-3 with a fresh cosine (0.0025 → 0.00025 over 7,600 epochs), as `ppo-2p-fr` had used from its start.

*Result.* Stopped at e2010 (26%), before finals. The champion row fell to 43.5 by e50; its 10-checkpoint block means were 43.3, 41.2, 40.6, 42.1 and 40.0 (±1 pp each). Against F it fell from 98.8 to \~97.7.

*Reading.* The restart did this, not the extra steps: e7600 ended at the floor with KL 0.001 per update and entropy 0.048, and 0.0025 pushed KL to 0.02–0.03 at once, about 20 times larger. `ppo-2p-fr`'s restart from e3350 (entropy 0.084, a broader policy) ran at KL 0.015 and held 48–51. A sharp policy at a low learning rate loses \~8–10 pp when the rate jumps back to the peak; by e7600 this run would have tested a warm restart, not continued training. Lesson: set a continuation's learning rate from its start's final KL and entropy.

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
#tldr[On 4p BASE at the `private` level, the 2p recipe takes a 78.55 score share against three F (parity 25) and 100.0 against three random players. Its V-trace critic was biased high (@exp:recipe-critic-fix).]

*Question.* Does the 2p recipe carry over to four players unchanged?

*Setup.* The recipe and 4p budget above (3,800 epochs) on `base-4p`, all four seats learning, at the `private` level, the only fair one past two players. Best checkpoint e3650 by `select-4p`.

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
  caption: [The 4p specialist on `heldout-4p`, 4,000 games each.],
)

*Reading.* The identical copies split unevenly by slot order: slot 0 moves right after an F (@sec:front, @sec:generalist).

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
#tldr[`ppo-4p`'s four seat win probabilities summed well above the realized sum. Lambda-return critic targets put the sum on the truth within about 20 epochs (1.00 undiscounted) without moving the policy; they are now the default.]

*Question.* Why is the 4p critic biased high, and what fixes it without moving the policy?

*Diagnosis.* At epoch 300 the sum was about 1.2 against 0.87; at the end, 0.93 against 0.90. CPU replications found two causes:
- V-trace ratios on waiting rows: only the acting seat's row gets min(ρ, 1), the others get 1, so each seat's target values the game as if it alone played the improved policy, and the four one-sided gains compound through bootstrapping.
- Muon's fixed-size update of the 1 × 128 value output layer: the predicted sum alternates high and low every optimizer step, and the rollout network lands high every epoch.

*Setup.* 100-epoch fine-tunes of `ppo-4p/main@latest` at its final learning rate (5e-4, constant) with lambda-return critic targets (every ratio 1; the policy's advantages stay V-trace's). Four arms of about 15 minutes each, listed in the table.

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
  caption: [The arms after the 100-epoch fine-tune: seat sum (@sec:front), predicted vs realized. `select-4p`: the policy's share against F, F, F, n=800 per point (±2.8). `catanrl verify` needs the start's record by path (@sec:front), e.g. `runs/ppo-4p-critic-fix/lambda/evals/select-4p/60cd10b66aea063c.json`.],
)
Every arm stays on its realized sum (@fig:critic); `lambda` started at 0.920 against 0.914. The rollout-vs-trained gap falls from 0.004 to 0.000 everywhere. At the end of `lambda-critic-g1` the seat to move predicts 0.284 against a realized 0.276, the others within 0.004. Clip fraction, entropy and game length (79–80 turns) don't change.

#figure(
  placement: auto,
  image("../figures/critic-fix.png", width: 100%),
  caption: [Seat sum, predicted vs realized. Left, the `ppo-4p-critic-fix` fine-tunes (dashed: realized, about 0.90 at γ = 0.999 and 1 undiscounted). Right, from scratch (@exp:recipe-critic-from-scratch): `aux-hidden-4p/plain` (purple, lambda) against `ppo-4p/main`'s first 750 epochs (orange, V-trace; its realized sum is about 0.90, the dashed 1.0 is the lambda critic's).],
) <fig:critic>

*Reading.* The targets alone fix it. They also remove Muon's phase lock, so AdamW buys nothing. Over a 100-epoch fine-tune, γ = 1 leaves game length unchanged (from scratch it does not, @exp:recipe-discount). Seat values that sum to one are what the search's max#super[n] backups want, so `critic_targets: lambda` and `critic_gamma: 1` became the defaults in `configs/train/ppo.yaml`.

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
#tldr[From scratch, the undiscounted lambda critic overshoots early (seat sum 1.83 at epoch 21) and sits on the realized 1.00 from epoch 220. `ppo-4p`'s V-trace critic overshoots too and never recovers (0.94 against 0.90 at epoch 3,800). The fine-tunes, starting from a trained critic, couldn't show this.]

*Question.* Does the new default calibrate from random weights?

*Setup.* The first from-scratch runs under the new default: the two `aux-hidden-4p` arms (a hidden-information experiment, @sec:hidden), `ppo-4p`'s recipe at 750 epochs (49.2M decisions), against `ppo-4p/main`'s first 750 epochs; `lstm-4p` (@sec:markov) repeats the check on the recurrent net. Numbers come from the training metric `critic/value_sum_4p`.

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
  caption: [Seat sum by epoch from random weights. The lambda critic's realized sum is 1.00; the V-trace critic's is about 0.90.],
)

*Reading.* The lambda critic recovers with the learning rate still four fifths of its peak. The V-trace critic is still 0.16 above its realized sum at epoch 750 (@fig:critic, right). Explained variance ends at 0.82–0.85, as before.

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
#tldr[*The discount makes games end faster*: trained without it (γ = 1 vs 0.999, from scratch on OpenHex's 1v1 rules, two seeds each), games against F run 4.6 and 2.6 turns longer (paired over 5,000 games, ±0.35) and 1.6–1.7 longer against the reference, against half a turn between two seeds of one arm. *It doesn't change strength*: head to head 50.02% and 49.94% per seed, level against F and the reference too.]

*Question.* Does the discount make games end faster (Marco's main question), and does dropping it play better?

*Why it might.* The recipe's γ = 0.999 comes from v1, but Catan needs no discount: every game ends, the only reward is the final win, and GAE's λ = 0.95 already cuts most of the variance. But γ < 1 turns the objective from p(win) into E[γ#super[T] · win], worth \~0.85 at the opening of a 160-decision game, which may favour faster, riskier wins. The only earlier test (`lambda-g1`, @exp:recipe-critic-fix: same game length and select score) was 100 fine-tune epochs, far too short.

#lead[Setup.]
- *Arms.* The recipe above from scratch on `base-2p-fr` at the `full` level, 500M agent steps (3,800 epochs, cosine 0.005 → 0.0005), with γ = 0.999 (`g999`) or 1 (`g1`) for policy and critic alike. Both use lambda-return critic targets, so only the discount differs.
- *Seeds.* 42 and 43 per arm (the seed also draws the 4,096 training boards). The seed spread was unknown, so an arm difference counts only if both seeds agree in sign and the pooled head to head clears it.
- *Monitors* every 50 epochs on `select-2p-fr` (±3.1 pp at 50%), against F and a fixed reference, `ppo-2p-fr3/full@e4350`.
- *Finals* on `heldout-2p-fr`, 5,000 games each, on each cell's *last* checkpoint (fixed in advance: no winner's curse), against F (±0.3 pp near 98) and the reference (±1.4 pp at 50%); then `g1` against `g999` within each seed (paired MDE \~2 pp at 80%).
- *Game length* (Marco's main readout). All finals share 5,000 game ids, so turns pair up game by game within a seed: against F (won 98–99%) they show how fast a net closes out a won game, against the reference the length of a close game. A paired difference resolves to about ±0.45 turns (95%; sd 11.5 vs F in `g999-s42`'s finals) even if games on one board are uncorrelated; the yardstick is the gap between two seeds of one arm.
- *Cost.* \~8.4 h per cell, \~34 h in all.

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
  caption: [Finals on `heldout-2p-fr` at each cell's last checkpoint (e3786), 5,000 games each; ref = `ppo-2p-fr3/full@e4350`. Turns: mean per game (sd 11.5 vs F, 13.3 vs ref). Self-play length: `environment/length`, last 200 epochs.],
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
  caption: [Training curves: the mean of each 500-epoch block from the epoch shown (the last ends at e3786).],
)

The monitors were still climbing at the end; the reference has about 2B steps behind it (500M on the old rules, 1.5B on these) against 500M per cell.

#block(sticky: true)[*Game length.* Longer without the discount on both seeds, paired game by game:]
#figure(
  btable(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    header: ([Seed], [Vs F, all games], [Vs F, both won], [Vs ref, all games], [Vs ref, both won]),
    [42], [+4.64 ± 0.35], [+4.65 ± 0.35 (4,867)], [+1.64 ± 0.43], [+2.68 ± 0.82 (1,237)],
    [43], [+2.64 ± 0.36], [+2.65 ± 0.36 (4,839)], [+1.73 ± 0.43], [+2.92 ± 0.83 (1,270)],
  ),
  caption: [Turns per game, `g1` minus `g999` within a seed, with 95% intervals. "Both won" keeps the games both cells won (their count in brackets).],
) <tab:recipe-discount-length>

- *The arm, not the seed.* `g999-s43` plays 0.47 ± 0.35 turns longer than `g999-s42` against F and 0.18 ± 0.43 shorter against the reference. `g1`'s two seeds are 1.5 turns apart against F (75.5 and 74.0), under the smaller arm gap.
- *Self-play agrees from the start:* `g1` is longer in all 16 training blocks.
- *It is in how the nets win:* against F the gap is the same on games both won.

#block(sticky: true)[*Strength.* Level, head to head within each seed:]
#figure(
  btable(
    columns: (auto, auto, auto),
    align: (left, right, left),
    header: ([Seed], [`g1` vs `g999`], [Record]),
    [42], [50.02 #ci[48.63][51.41]], [`98e98e0d`],
    [43], [49.94 #ci[48.55][51.33]], [`e0b8011e`],
    [Pooled], [49.98 #ci[49.0][51.0]], [10,000 games],
  ),
  caption: [`g1` against `g999` on `heldout-2p-fr` at e3786, 5,000 games per seed. No game hit the turn limit.],
) <tab:recipe-discount-h2h>

Pooled over seeds, `g1` scores 98.34 against F and 43.5 against the reference, `g999` 98.56 and 44.1; all four cells' intervals overlap, and the monitors climbed alike. Head to head, games `g1` wins run 1.98 ± 0.75 (s42) and 1.33 ± 0.76 (s43) turns longer than those `g999` wins.

*Reading.* The discount does what its objective says: the net closes out games sooner, with no change in strength at 500M steps. The choice of γ comes down to the game length Marco wants.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/discount-2p.yaml
catanrl run configs/experiments/discount-2p-h2h.yaml   # g1 vs g999 within each seed
```)

== Does training keep improving?

=== Transitivity: a round robin of one run's checkpoints #chip("done") <exp:recipe-transitivity>
#meta(
  dates: "2026-11-10 06:22 – 07:58",
  owner: "marco",
  config: "configs/experiments/transitivity-2p-belief.yaml",
  wandb: (("transitivity-2p-belief-main-5c004308", "main"), ("transitivity-2p-main-33604123", "CPU try, stopped")),
  records: ("8e14efa6", "cc266f20", "d2b1ad49", "b3c1bcaf", "621df948", "47ea4e27", "29fa4fcb", "13b5109a", "f74bb619", "2595a510", "55c023d1", "4285662e", "82126ac8", "b74388af", "083a92d1", "8f9847b1", "78bbfefb", "205d9634", "cbe77520", "00ccdeb3", "2f9b4206", "5e3e9a0a", "4934d6da", "cf845991", "0784ba13", "62e2451b", "a60863e6", "28383622", "650def09", "47385ae9", "2f5cc732", "a7ebe01b", "c5e3f944", "073b9564", "fdf88c50", "5bc5d392"),
)
#tldr[Self-play only ever plays the current net, so steady gains against F and fr3 don't rule out cycles between checkpoints. A round robin of eight checkpoints of one run plus the deployed net, 1,000 games a pair with the fair bot: training is transitive. Every later checkpoint beats or ties every earlier one, one Bradley–Terry rating fits all 36 pairs (p = 0.58), and there are no cycles. The gains shrink as training goes on.]

*Question.* Does every later checkpoint beat every earlier one, and do the results fit a single rating (Marco)?

#lead[Result.]
#figure(
  btable(
    columns: (auto,) + (1fr,) * 9,
    align: (left,) + (right,) * 9,
    header: ([], [250], [500], [1,000], [1,500], [2,000], [2,500], [3,000], [3,786], [fr3]),
    [250], [–], [33.7], [26.4], [21.8], [20.2], [19.4], [14.6], [15.4], [11.5],
    [500], [66.3], [–], [41.2], [36.0], [30.8], [28.3], [26.8], [23.5], [19.3],
    [1,000], [73.6], [58.8], [–], [43.6], [42.0], [38.4], [31.6], [31.0], [25.2],
    [1,500], [78.2], [64.0], [56.4], [–], [46.0], [42.5], [36.6], [37.0], [32.6],
    [2,000], [79.8], [69.2], [58.0], [54.0], [–], [49.8], [43.1], [41.6], [33.9],
    [2,500], [80.6], [71.7], [61.6], [57.5], [50.2], [–], [45.2], [43.2], [38.6],
    [3,000], [85.4], [73.2], [68.4], [63.4], [56.9], [54.8], [–], [50.0], [39.3],
    [3,786], [84.6], [76.5], [69.0], [63.0], [58.4], [56.8], [50.0], [–], [47.6],
    [fr3], [88.5], [80.7], [74.8], [67.4], [66.1], [61.4], [60.7], [52.4], [–],
  ),
  caption: [Row's score against column (%), 1,000 games a pair on `heldout-2p-fr`, `BELIEF:worlds=16` on both sides. Numbers are `discount-2p/g1-s42`'s epochs; fr3 is `ppo-2p-fr3/full@e4350`. ±3.1 pp at 50%.],
)
- *Later beats earlier.* The later checkpoint wins 26 of the 28 pairs within the run, each above 50% at 95%. The other two are level: 2,500 against 2,000 (50.2%, 47.1–53.3) and 3,786 against 3,000 (50.0%, 46.9–53.1). No earlier checkpoint wins a pair.
- *One rating.* A Bradley–Terry fit (8 free ratings) leaves a deviance of 26.0 on 28 df (p = 0.58). The largest of the 36 residuals is z = −2.3 (fr3 against 3,786).
- *No cycles.* No three players beat each other in a ring, even counting wins inside their intervals.
- *Shrinking gains.* Elo above epoch 250 (bootstrap sd about 6): 110 at 500, 174 at 1,000, 215 at 1,500, 245 at 2,000, 261 at 2,500, 298 at 3,000, 310 at 3,786, and 352 for fr3. Per 500 epochs that is 63, 41, 30, 16 and 36 Elo, then 13 over the last 786.
- *fr3.* It beats every checkpoint of the run, the last one 52.4% (41 Elo, 30–52). At the `full` level without belief averaging it scored 55.8% against the same checkpoint over 5,000 games (@exp:recipe-discount).

#lead[One pair off the rating.] The last 786 epochs (the end of the cosine, near 0.0005) helped against fr3 but not against epoch 3,000. On the same games fr3 scored 60.7% against 3,000 and 52.4% against 3,786 (215 against 132 games that differ, paired p = 1e-5), yet 3,786 and 3,000 tie. On one rating, 8 points gained against fr3 would mean about 58% against 3,000. This was picked after the fact as the largest residual, and the overall fit is fine.

#lead[Reading.] Within this run, self-play kept improving against all its past selves, with no sign of the cycling that self-play can fall into. One rating describes the run and fr3 well enough to rank checkpoints by a single number. The scope is one run, one seed and the fair bot.

#lead[Setup.]
- *Players.* `discount-2p/g1-s42` (`graph-s` from scratch on OpenHex's 1v1 rules, 3,786 epochs; @exp:recipe-discount) at epochs 250, 500, 1,000, 1,500, 2,000, 2,500, 3,000 and 3,786, plus the deployed `ppo-2p-fr3/full@e4350` from another lineage. All 36 pairs.
- *Agent.* The fair belief-averaged bot (`BELIEF:worlds=16`, the deployed 1v1 bot) on the GPU. A first launch on the CPU used the argmax `full`-level policy to leave the GPU to `graph-m`. It was stopped after 6 of 36 pairs (1.8 h projected) so the round robin waits for the GPU (Marco). The GPU run is a config of its own, `transitivity-2p-belief`, since the agents changed. The 36,000 games took 96 min.
- *Games.* `heldout-2p-fr`'s first 500 boards in both seat orders: 1,000 games a pair (±3.1 pp at 50%, Wilson 95%). A 4.4 pp difference has 80% power, so close late pairs may not separate on their own; the rating fit pools 8,000 games a player.
- *Analysis.* The score matrix; pairs where the earlier checkpoint wins; a Bradley–Terry fit and its goodness of fit (deviance on 36 − 8 = 28 df); any significant cycle.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/transitivity-2p-belief.yaml
```)

== Rollout segments and λ

=== 256-step segments and λ = 1, from scratch #chip("queued") <exp:recipe-horizon>
#meta(
  dates: "queued for the GPU after `graph-m` (about 10 November)",
  owner: "marco",
  config: "configs/experiments/horizon-2p.yaml",
  wandb: (),
  records: (),
)
#tldr[Every run so far cuts each env's rollout into 64-step segments with λ = 0.95, so a step's target is mostly the critic's estimate about 20 decisions later. The same run as `discount-2p/g1-s42` with 256-step segments and λ = 1, so most targets are the game's actual result.]

*Question.* Do longer segments with λ = 1 train a stronger 1v1 net than 64 steps with λ = 0.95 (Marco)?

#lead[Why.] A 1v1 game is about 209 decisions (both seats, 88 turns), so a 64-step segment holds about a third of one. At the cut the targets fall back on the critic's value, and λ = 0.95 weighs a step k decisions ahead by 0.95#super[k] (0.36 at 20, 0.04 at 64). With 256 steps and λ = 1, a step's target is the actual result whenever its game ends inside the segment (about 59% of steps) and the critic's value at the cut otherwise. The critic's λ-return targets use the same λ. The graph nets have no memory, so the segment length changes only the targets.

#lead[Setup.]
- *Baseline.* `discount-2p/g1-s42` (@exp:recipe-discount): `graph-s` from scratch on `base-2p-fr`, γ = 1, seed 42, 500M steps on a cosine from 0.005 to 0.0005.
- *Changes.* Segment length 64 → 256, λ 0.95 → 1, and 1,032 → 258 envs, so each update still sees 132,096 steps (516 rows × 256) with the same 16 optimizer steps. Fewer games run in parallel. Both knobs change together, so a difference can't be pinned on either alone.
- *Monitors.* Every 50 epochs on `select-2p-fr` against fr3, on the same games as the baseline's monitors, so every epoch pairs with it game by game. No monitor or final against F, which sits near 98% (Marco).
- *Finals.* `heldout-2p-fr`, 5,000 games each, on the last checkpoint: against fr3, and head to head against the baseline's last checkpoint (±1.4 pp at 50%).
- *Cost.* The baseline took 7.9 h; collecting 4× more, 4× smaller steps may be slower (to be measured). Needs the 8 GB card alone. A 2-epoch CPU smoke passed.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/horizon-2p.yaml
```)

== Mixed precision

=== float16 training at `graph-m-2p-lr`'s start: `fp16-2p` #chip("queued") <exp:fp16>
#meta(
  dates: "queued for the GPU after graph-m finishes",
  owner: "marco",
  config: "configs/experiments/fp16-2p.yaml",
  wandb: (),
  records: (),
)
#tldr[Claude's design, approved by Marco on 2026-11-09. Every run so far trains in float32, and this card (Turing) has fp16 tensor cores but no bf16. `graph-m-2p-lr`'s first 300 epochs twice, in float32 and in float16 with loss scaling: is float16 faster at no cost in strength?]

*Question.* Does float16 mixed precision train `graph-m` at least 15% faster without a weaker net?

#lead[Setup.]
- *Arms.* `fp32`, then `fp16`: `graph-m-2p-lr`'s run exactly (seed 42, `base-2p-fr`, `graph-m` with one shared trunk, Muon, the 3B-step cosine from 0.0025), stopped after epoch 300 (39.6M steps) by the new `train.stop_epoch`. The cosine still spans the whole budget, so both follow `graph-m-2p-lr`'s first 300 epochs (the learning rate is still 99.96% of its peak). Micro-batches stay 2,048 rows in both, so precision is the only difference.
- *float16.* The network's forwards (rollout and update) run under autocast to float16. A GradScaler scales each micro-batch's loss, the gradients are unscaled before the clip, and a step with a non-finite gradient is skipped. Weights, gradients, Muon and the losses stay float32. Logged per epoch: the scale, skipped steps, peak memory, and a probe of the rollout's log-probs against the update's forward on the same rows before any step.
- *Monitors.* Against fr3 every 100 epochs on `select-2p-fr`, 1,000 games each (±2.4 pp at 18%), as context. `graph-m-2p-lr` scored 5.1, 12.6 and 18.0% at epochs 100, 200 and 300.
- *Finals.* `heldout-2p-fr` against fr3, 5,000 games a row, the same games in every row: each arm's epoch 300, and `graph-m-2p-lr`'s own epoch 300. Two float32 runs of one seed differ on the GPU, so `fp32` against `graph-m-2p-lr` gives the run-to-run spread.

#lead[Decision rule.] Adopt float16 for new runs if all three hold:
+ the median steps/s over epochs 50–300 is at least 1.15× that of `fp32` (`graph-m-2p-lr` varied about 2% over those epochs);
+ no loss is ever non-finite, and skipped steps stay under 1% of optimizer steps after epoch 10 (at most 46 of 4,640);
+ the final of `fp16` against fr3 is not significantly below that of `fp32` (paired McNemar over the shared games, two-sided 0.05), read against the gap between `fp32` and `graph-m-2p-lr`.
At about 18% a row is ±1.1 pp. The paired SE of the difference is at most 0.77 pp (independent games), so 80% power for about 2.2 pp; a smaller deficit could go unseen.

#lead[Cost.] About 3.5 h of GPU with the card alone: 1.8 h for `fp32` (300 × 132,096 steps at about 6.1k steps/s), at most 1.6 h for `fp16` if it passes rule 1, about 1.5 minutes of monitors per arm and about 5 minutes of finals. A 15-minute GPU smoke (20 epochs per arm in a scratch directory, W&B off) comes first. A CPU check of the whole path passed.

#repro(```sh
env -u WANDB_API_KEY WANDB_MODE=disabled uv run python tools/fp16_smoke.py --runs /tmp/fp16-smoke   # GPU smoke
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/fp16-2p.yaml
uv run python tools/search_probe.py --pair <fp16's final record> <fp32's final record>
```)
