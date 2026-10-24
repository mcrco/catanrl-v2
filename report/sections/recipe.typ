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

=== the 1v1 bot on openhex's 1v1 rules, \~1b more steps, judged against its start #chip("running") <exp:recipe-2p-openhex>
#meta(
  dates: "2026-10-24 10:46 onward",
  owner: "marco",
  config: "configs/experiments/ppo-2p-fr.yaml",
  wandb: (("ppo-2p-fr-full-fef9516c", "ppo-2p-fr/full"),),
  records: ("fa6ea360", "2bfef46d", "8a137f56", "ffcc5c1a"),
)
#tldr[openhex's 1v1 queue plays friendly robber and balanced dice; our 2p bots trained on neither. marco: train on friendly robber from now on, continue the deployed 1v1 net for \~1b more steps on it, and judge it against its start (the champion) as well as F. running since 2026-10-24 10:46, \~17.5 h.]

*setup.*
- *the rules.* from openhex's source: its 1v1 queue is 15 vp, discard above 9, friendly robber, balanced dice; its 4p queue is 10 vp, discard above 7, normal robber, random dice, which is our 4p setup already. openhex's friendly robber shields an opponent with *public* points ≤ 2. our engine used *actual* points < 3, which counts hidden victory point cards (a mismatch, and the legal moves leak whether the opponent holds one), so the engine now uses openhex's rule (a knight is judged on the points before it takes largest army, as openhex does). balanced dice are in the engine too (marco: "do balanced dice before 1v1 retrain"), following openhex's code: a 36-card deck refilled when 13 or fewer are left, a recency penalty on totals seen in the last 5 rolls, and per-player seven balancing; search chance nodes use the state's own odds. `base-2p-fr` is openhex's 1v1 exactly: 15 vp, discard above 9, friendly robber, balanced dice. random-dice, normal-robber games play bit-identically to the old engine. the network gets no dice-state input, so the policy plays the dice blind and only the search reads them. the two recorded openhex games say balanced dice but were played on fair dice (an openhex bug, since fixed); the bridge falls back to fair dice if a roll is impossible under the balanced deck.
- *the run.* `puffer-graph2/full@best` (e3350, the deployed 1v1 net) continued on `base-2p-fr` for \~1b agent steps, the recipe otherwise unchanged, at \~17.8k agent steps/s (\~17.5 h with monitors and finals).
- *baselines.* the start vs F on `select-2p-fr`: *96.3* #ci[94.9][97.3] (`8a137f56`). its friendly-robber input was never on in training: a copy with that input zeroed beats it *51.5* #ci[50.2][52.9] over 5,000 games (`ffcc5c1a`), so part of any gain over the start is just learning to ignore that input; the final head to head subtracts 1.5 pp for it.
- *identity.* the merge moved the code ids every record is keyed under (catanrl `55ba5984`, engine `a9b40d9c`, train `55588933`), so new rows pair with older records only through `tools/search_probe.py --pair` (old-rule games are bit-identical).
- *the champion eval (marco).* ppo stagnation has been read from the win rate vs F, which sits near its 95% ceiling. every select now also plays the current checkpoint against a champion (here the start), 1,000 games on the same boards each time: ±3.1 pp, paired game by game across checkpoints. the finals include the start under the new rules, since it never trained on them. at e0 the champion row reads 50.0 (`fa6ea360`), as it must.

*progress (2026-10-24 13:30, e1250 of \~7,576).* champion row by blocks of five selects (5,000 games each): 48.8, 50.3, 49.9, 52.8, 50.6%; vs F 96.1 → 97.0–97.2%. the net dips below its start for the first \~200 epochs on the new rules, then edges above it. the trainer's best is e750 (54.7, `71cf79d7`), the highest of many noisy rows, so it is biased up; the heldout finals give the unbiased number.

*result.* running.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/ppo-2p-fr.yaml
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
