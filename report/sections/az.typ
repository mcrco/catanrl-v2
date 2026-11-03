#import "../macros.typ": *

= AlphaZero <sec:az>

Marco's note on the idea: "currently agreement is really high, doesn't seem to yield many gains". The question is whether training the network on its own search's choices (AlphaZero self-play) gives a better network, as a raw policy or inside the search. The loop (`train.algo: alphazero`) self-plays through the batched search with Gumbel roots, keeps a replay buffer of (position, visit-count policy, per-seat outcome) and warm-starts from a PPO checkpoint.

Marco's rule for judging a run: MCTS with the new network and MCTS with the start's network each play the raw start on the same games, and the two rows are compared game by game with McNemar's test, not by playing the networks head to head (@exp:az-e8-search). So far no run has beaten its start. The raw policy got worse every time, and inside the search the trained networks played the same (e8) or worse (the soft run). The likeliest culprit is how the targets are shaped.

== First runs and cost

=== 30-minute sanity run from a weak PPO net #chip("done") <exp:az-from-ppo>
#meta(
  dates: "2026-10-14",
  owner: "marco",
  config: "configs/experiments/az-from-ppo.yaml",
  wandb: (),
  records: (),
)
#tldr[A first end-to-end test of the loop, from a weak 40-minute PPO net at 57.8% vs F. Fitting noisy 64-sim targets, its raw policy fell steadily to 18.5% vs F. It says nothing about strong networks or bigger searches.]

*Question.* Does the loop run, and does a weak network learn from its own 64-sim search?

#lead[Setup.]
- The AlphaZero stage of `az-from-ppo`, run for 30 minutes (25 iterations) in a scratch runs directory, since deleted: no records and no W&B mirror.
- Start: a 40-minute PPO sanity net, not `puffer-graph2`.
- Oracle search (`dets: 0`: the search sees the true hidden state); Gumbel targets at 64 sims on a quarter of moves and 16 on the rest; pure self-play; learning rate 5e-5.
- Nothing was played head to head against the start.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    header: ([Player], [Score vs F], [Games]),
    [Start, raw policy], [57.8], [],
    [Start, 64-sim search], [63.5], [100],
    [Trained raw policy, three successive checkpoints], [46.6 → 24.4 → 18.5], [600 each],
  ),
  caption: [`az-from-ppo`'s AlphaZero stage against F. No records survive, so there are no intervals.],
)

#lead[Reading.]
- *The targets were noisy.* With Q min-max rescaled over the root's moves, a near-single-visit Q gap swings a target by \~6.5 logits. Two search seeds agree on the top move 87% of the time (60% on discards).
- *The network fit the noise.* Its entropy rose from 0.42 to 0.82 until its argmax drifted to end-turn, and knight-before-roll (how often the policy plays a knight before rolling) fell from 0.90 to 0.27.
- *One small bug found:* the Gumbel temperature-0 finalist, fixed in cppanatron `5d9dda9`. It does not touch the targets.
- *For context*, v1 never trained above 512 sims, and 512-sim targets did not move an 87% network (v1's EX-1d, a frozen-teacher distillation of an 8 × 64-sim search: +1.3 vs F and 50.6% head to head, both null).

#repro(```sh
# Runs the PPO stage (50M steps), then 300 AlphaZero iterations. The original ran
# only the AlphaZero stage, for 25 iterations, from a 40-minute PPO net.
catanrl run configs/experiments/az-from-ppo.yaml
```)

=== Self-play cost study (CPU) #chip("done") <exp:az-cost>
#meta(
  dates: "2026-10-21",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[Before spending GPU time, Claude measured on CPU what makes self-play expensive. Tree reuse and transpositions save nothing as built; only an eval cache across one self-play step would help, cutting network forwards by 31–46%. A config-only first run at 1024-sim targets costs \~13–22 GPU hours for 60 iterations.]

*Question.* Does the search reuse built for the deployed bot make self-play cheaper, and what does a first run cost?

*Setup.* CPU measurements of self-play searches in 2p and 4p. The 2p numbers use the engine's default rules (10 VP, discard above 7), not `base-2p`'s (15 VP, discard above 9); the target probes of @exp:az-2p-2048 used `base-2p`.

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([Lever], [Effect on self-play]),
    [Tree reuse, as built], [The tree survives 43% (2p) to 53% (4p) of moves, but `advance` resets the budget, so it saves no leaves.],
    [Engine top-up budget \ #sub[search only sims − kept visits]], [15–20% fewer leaves. Kept visits are worth about their count, so reuse cannot halve the budget.],
    [Transpositions], [Nothing. They fight reuse (they restart the new root's shared children), so they stay off.],
    [Eval cache across one \ self-play step], [31–46% fewer forwards. Hashing has to move into the engine.],
  ),
  caption: [CPU measurements of self-play search, lever by lever.],
)
A 1024-sim move costs \~1,000 leaf rows and \~80 ms of engine CPU. A first run as a config-only change to the existing trainer (1024-sim Gumbel targets on every move, 512 games, from `puffer-graph2/public`) costs \~13–22 GPU hours for 60 iterations.

*Reading.* The real lever is the eval cache, and it needs engine work. Marco chose to go straight to the loop at 2048 sims and do engine work only after a positive result (@exp:az-2p-2048). The engine levers are listed in @exp:az-followups.

#repro(```sh
# Not reproducible: one-off CPU measurements; no script for them is in the repo
```)

== The 2p specialist at 2048-sim targets

=== Self-play from the 2p specialist: `az-2p-2048` #chip("stopped") <exp:az-2p-2048>
#meta(
  dates: "2026-10-21 15:40–18:53",
  owner: "marco",
  config: "configs/experiments/az-2p-2048.yaml",
  wandb: (("az-2p-2048-main-7e569a67", "az-2p-2048/main"),),
  records: ("e34b2057", "fa218f06", "14f41aa1"),
)
#tldr[Marco's pick: the best 2p network straight into self-play, with 2048-sim targets on every move. Its raw policy lost to its start (48.0 and 48.2 head to head), so the stop rule ended the run after 8 iterations. The likely cause is the targets: Gumbel's target puts ≥ 0.99 on one move in most decisions, whatever the real gap between moves.]

*Question.* Does training on 2048-sim search targets improve an already strong network (95% vs F)?

#lead[Setup.]
- Start: `puffer-graph2/public@e3600` (actor and critic), `base-2p`, 512 games in flight.
- Search: fair Gumbel search on every move (`dets: 1`, one world dealt from the tracker per search; m = 16 root moves considered).
- Training: 30 iterations × 16,384 targets planned, learning rate 5e-5.
- Select: every 2 iterations the raw policy plays its start head to head on `select-2p-2k` (4,000 games); the search finals run only if that is positive. Stop rule: two heads to head ≤ 48.5 in a row.
- Logging-only monitors (training is bit-identical with or without them): the value head's loss on finished games before it trains on them, how often the search overrules the network's argmax, and drift from the start on a fixed probe of 4,096 positions.
- Cost: the first iteration ran at 23.6k leaves/s and took 24 min, so \~12 h for the 30.

CPU probes under `base-2p` rules came first. 2048-sim targets cut the prior's distance to an 8192-sim reference from 0.295 to 0.197 in total variation (the reference's own noise floor is 0.179) and change the top move at 20% of positions. They are barely better than 1024-sim targets (−0.016 #ci[−0.034][0.001]). Robber moves, and positions where the opponent holds a hidden VP card, stay noisy.

*Result.* Stopped at e8 (3.2 h). Best stayed e0, so no finals ran (`az-2p-2048-search.yaml` was never played).
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Raw policy vs its start], [Score], [Games], [Record]),
    [e0 (the same weights)], [50.0 #ci[48.5][51.5]], [4,000], [`e34b2057`],
    [e6], [*48.0* #ci[46.5][49.5]], [4,000], [`fa218f06`],
    [e8], [*48.2* #ci[46.7][49.7]], [4,000], [`14f41aa1`],
  ),
  caption: [`az-2p-2048`'s selects on `select-2p-2k`. e2 and e4 equal e0: the first update came in iteration 5, once enough games had finished.],
)
From the start to e8 the network's entropy rose from 0.088 to 0.125, and the search overruled its argmax more often (0.18 → 0.23), where a policy that learns from its search should make that rarer. Knight-before-roll on the drift probe fell from 0.53 to 0.43.

*Reading.* It went the way the sanity run went, only slower.
- *Not the critic.* The monitor's out-of-sample value loss rose from 0.20 to 0.54, but that is the games getting longer (the first games to finish are the short ones). On the same rows the start scores 0.375 / 0.508 / 0.387 at iterations 6–8 and the trained net 0.359 / 0.538 / 0.402, a few hundredths either way.
- *The targets, as far as one look can say.* Gumbel's σ(Q) = (50 + max visits) · 0.1 · Q, with Q min-max rescaled to [0, 1] across the root's moves, puts a \~100-logit spread between the best and the worst move at 2048 sims, whatever the gap in win probability. So all 16,153 two-move decisions in the replay (mostly knight or roll) got a target ≥ 0.99 on one move, as did 81% of the decisions with more than eight legal moves. On knight-or-roll the start was ≥ 90% sure 98% of the time; the targets went against its argmax 21% of the time and played the knight 44% of the time against the start's 53%. That is the drift the network then learned.
- *Not the moves.* The same search over the start's weights beats the raw policy 62.4% (`831eea84`, @exp:az-e8-search), far above PUCT over the same policy (54.9% at 1,024 sims, 55.2% at 4,096; @sec:search). The search had plenty to teach, and the raw policy still got worse learning it.

#repro(```sh
catanrl run configs/experiments/az-2p-2048.yaml
```)

=== e8 judged inside the search #chip("done") <exp:az-e8-search>
#meta(
  dates: "2026-10-21 18:56 – 2026-10-22 02:59",
  owner: "marco",
  config: "configs/experiments/gumbel-edge-2p-e8.yaml",
  wandb: (("gumbel-edge-2p-main-efda50ac", "gumbel-edge-2p, eval only"), ("gumbel-edge-2p-e8-main-6e1d1ef0", "gumbel-edge-2p-e8, eval only")),
  records: ("831eea84", "c4ea7b52"),
)
#tldr[Marco's point: the goal is a better network inside the search, so judge AlphaZero there, not by the raw policy. In the same 2048-sim search against the same raw start, e8 scores 61.8% and the start's own weights 62.4%: −0.6 pp paired, no difference.]

*Question.* Does e8 search better than its start, even though its raw policy got worse?

*Setup.* Two rows that differ only in the network inside the search.
- Each row: the same 2048-sim Gumbel search (`dets: 1`, m = 16, `c_puct` 0.75) against the same raw start, `puffer-graph2/public@e3600`.
- Games: 1,000, on the first 500 boards of `heldout-2p` in both seat orders. Both rows play the same game ids.
- Test: paired game by game, by McNemar's test on the games only one row won (`tools/search_probe.py --pair`).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([Network inside the search], [Score vs the raw start], [Record]),
    [Start (e0)], [62.4 #ci[59.4][65.3]], [`831eea84`],
    [e8 of `az-2p-2048`], [61.8 #ci[58.7][64.8]], [`c4ea7b52`],
    mid,
    [e8 − e0, paired over the 1,000 shared games], [*−0.6 pp* #ci[−3.7][2.5]], [McNemar p = 0.75],
  ),
  caption: [2048-sim Gumbel search against the raw start on `heldout-2p` (500 boards, both seat orders).],
)
Of the shared games, 124 were won only by e8 and 130 only by e0.

*Reading.* The raw policy lost two points and the search played the same, so stopping at e8 cost nothing, but by luck. Only a quarter of the pairs differ at all (the board, the seat and the dice decide most games), so pairing roughly halves the interval two independent rows would give. This paired test is the template: every later AlphaZero run is judged this way, and so is the critic refit (@sec:critic).

#repro(```sh
# The e8 row's YAML names runs/az-2p-2048/main/checkpoints/e000008.pt by absolute path
catanrl run configs/experiments/gumbel-edge-2p.yaml      # Start's weights in the search: 831eea84
catanrl run configs/experiments/gumbel-edge-2p-e8.yaml   # e8 in the search: c4ea7b52
uv run python tools/search_probe.py --pair c4ea7b5234eb6a50 831eea84f04c89d8
```)

== Softer targets

=== Softer targets through a tempered prior: `az-2p-2048-soft` #chip("stopped") <exp:az-soft>
#meta(
  dates: "2026-10-22 15:49 – 2026-10-23 01:57",
  owner: "marco",
  config: "configs/experiments/az-2p-2048-soft.yaml",
  wandb: (("az-2p-2048-soft-main-8105fbb1", "az-2p-2048-soft/main"),),
  records: ("856a0421", "ae5a4000", "f016fee8", "8906dcf7"),
)
#tldr[Marco's next step: the same loop with the search unchanged and softer targets. The softening that looked best on CPU, tempering the prior inside the target, flattened the network instead: its search fell from 57.4% to 46.7% against the raw start by e15, and Marco stopped the run. The prior temperature (Claude's proposal) is the likely culprit, since Canopy never tempers the policy its search reads.]

*Question.* Does a network trained on softer versions of the same search's targets search better?

*Setup.* The trainer rebuilds Gumbel's improved policy itself, from each search's visits and Q and the network's root outputs (`train.policy_target: gumbel`). At the engine's constants it matches the engine's target to \~3e-8, and the default trains bit-identically to before.

The softening was picked on CPU: 600 positions of the start's raw-policy `base-2p` self-play, two 2048-sim searches each, scored by −log of the target's mass on an 8192-sim search's move. A gentler σ alone softens nothing. The start's prior is itself one-hot (a median gap of 47 nats between its top two logits on knight-or-roll decisions), so softmax(prior + σ) stays one-hot at any scale. What does soften it is tempering the prior inside the target: softmax(logits / 10 + (50 + max visits) · 0.15 · Q), with Q unrescaled (Claude's proposal from the probe, approved by Marco).
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Target], [Score vs engine \ (nats)], [Agrees with \ 8192-sim move], [Entropy], [≥ 0.99 on \ one move]),
    [Engine \ #sub[σ on rescaled Q, `c_scale` 0.1]], [], [81.4%], [0.05], [87%],
    [Best σ-only setting], [−0.10 #ci[−0.24][0.03]], [], [], [],
    [Tempered prior \ #sub[logits / 10, `c_scale` 0.15, Q unrescaled]], [*−1.12* #ci[−1.39][−0.86]], [80.5%], [0.64], [22%],
  ),
  caption: [CPU probe, 600 positions. The score is −log of the target's mass on the 8192-sim move, relative to the engine's target; lower is better.],
)
That is calibration, not better moves: two 8192-sim searches pick the same move only 86% of the time, and the expected regret under the softer target is slightly worse (+0.05 #ci[0.01][0.08] win points).

Everything else is as in `az-2p-2048`. As Marco asked, the run is judged in the search throughout:
- Selects: every 5th iteration the checkpoint, in a 512-sim Gumbel search, plays the raw start on the same 1,000 games of `select-2p-500`, paired with e0's row.
- Planned finals: the best and the last checkpoint in the 2048-sim search against the raw start on the games of `831eea84`, each paired with it (McNemar, Holm over the two). Every iteration's weights are kept, so an e8 row can pair with `c4ea7b52` later.
- Stop rule: only a collapse, two selects in a row more than 4 pp below e0 on the same games (about 2.5 paired SDs). The raw policy is not a stop criterion this time.
- Planned cost: \~19 h of training and selects, then 4–8 h of finals.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, left),
    header: ([Checkpoint], [Score vs \ raw start], [Paired \ vs e0], [Won only by \ e0 / eN], [McNemar p], [Record]),
    [e0], [57.4 #ci[54.3][60.4]], [], [], [], [`856a0421`],
    [e5], [56.0 #ci[52.9][59.0]], [−1.4 pp], [129 / 115], [0.41], [`ae5a4000`],
    [e10], [53.8 #ci[50.7][56.9]], [−3.6 pp], [158 / 122], [0.036], [`f016fee8`],
    [e15], [*46.7* #ci[43.6][49.8]], [*−10.7 pp*], [225 / 118], [< 10⁻⁶], [`8906dcf7`],
  ),
  caption: [`az-2p-2048-soft`'s selects: each checkpoint in a 512-sim Gumbel search against the raw start, on the same 1,000 games of `select-2p-500`.],
)
The monitors say why. By e13 the network's entropy was 0.68 (0.09 at the start), it agreed with the start's argmax on 72% of the drift probe, and the search overruled its argmax 41% of the time (18% at the start). At e15 the entropy was 0.92 and the search's depth had fallen from 11 to 8.2. The collapse rule would also have needed e20; Marco stopped the run after e15 ("it is sucking balls"). Best stays e0, so there are no finals.

*Reading.* The softer target pulled the prior flat faster than the search's Q could replace what the prior knew. The Canopy comparison (@exp:az-canopy-diff) puts the blame on the prior temperature: Canopy never tempers the policy the search reads.

#repro(```sh
catanrl run configs/experiments/az-2p-2048-soft.yaml   # Marco stopped it by hand after the e15 select
uv run python tools/search_probe.py --pair 8906dcf7adfc1c96 856a0421185ce86f   # e15 vs e0, paired
```)

== What next

=== What Canopy did, and where we diverge #chip("done") <exp:az-canopy-diff>
#meta(
  dates: "2026-10-23",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Marco asked how Canopy trained its AlphaZero network. Canopy trained from scratch with a recipe close to ours but published no win rates, and nobody has shown the recipe improves an already strong network. The likeliest reasons our runs failed are our target's Q scaling and the soft run's prior temperature.]

*Question.* What did Canopy do, and which of our differences could explain our two failed runs?

*Setup.* Read from cullback/canopy's code and history, checked against source, plus one more measurement on the 600-position CPU probe of @exp:az-soft.

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    header: ([], [Canopy], [Ours (`az-2p-2048`)]),
    [Start], [A 2.4M-parameter nexus-v3 net, most likely from random weights], [`puffer-graph2/public@e3600`, 95% vs F],
    [Rules], [1v1 Colonist (balanced dice, friendly robber)], [`base-2p`],
    [Search], [Gumbel, m = 6; 1,600 sims on a quarter of moves, 64-sim value-only searches on the rest; an improved-policy interior], [Gumbel, m = 16; 2,048 sims on every move],
    [Hidden cards], [The opponent's dev cards resampled every simulation], [One tracker world per search],
    [Q in the target], [Divided by the span of values the search saw, `c_scale` 1], [Min-max rescaled over the root's moves, `c_scale` 0.1],
    [Softening], [A soft-policy side output (π^(1/4), weight 8) that the search never reads], [None (`az-2p-2048`); a tempered prior inside the target (`az-2p-2048-soft`)],
    [Value], [Win/draw/loss, blended from the outcome toward the search's Q (to 85% over 60 iterations)], [Per-seat outcome],
    [Replay, \ optimizer], [500k positions; AdamW 5e-4 → 1e-4], [400k positions; learning rate 5e-5, a fresh optimizer on a Muon-trained net],
    [Length], [About 315 iterations (\~47M positions, inferred from the checkpoint name)], [8 and 15 iterations],
  ),
  caption: [Canopy's AlphaZero recipe next to ours.],
)
Canopy's evidence of success is thin. It publishes no win rates, Elo or opponents, only a README line ("we believe ... the strongest public 1v1 catan agent"). The only outside numbers are v1's: its graph checkpoints scored 23–33% against Canopy at 64 sims (48 games each).

The divergences most likely to matter:
+ *How the target scales Q.* Our engine min-max rescales Q over the root's moves at `c_scale` 0.1, so on a two-move decision any Q lead becomes a \~107-logit lead. Canopy divides by the span of values its search saw, at `c_scale` 1, so near-ties stay near-ties. On the probe's 600 positions a Canopy-shaped target (unrescaled Q, `c_scale` 1, no temperature) overrules the start on knight-or-roll 2.2% of the time, against the engine target's 18.9%. It agrees with the 8192-sim move as often (88.1 vs 87.8%) and stays sharp (entropy 0.08). The −log score of @exp:az-soft could not see this difference.
+ *The prior temperature.* The soft run's temperature is not Canopy's (Canopy softens only through its separate output), and it is what flattened the soft run.
+ *The start.* Canopy's search targets were always far better than its prior; from a 95% start the search's gain is close to its own noise. The hand-off also switches the critic's objective and starts a fresh AdamW on a Muon-trained net.
+ *Hidden cards.* One sampled world per search, against Canopy's per-simulation resampling.

*Reading.* v1 ran Canopy's recipe (its PUCT version) on a 55%-vs-F graph net and it did improve it, 55 → 70.5% raw vs F over 10 iterations, but never from a ≥ 90% start. Divergences 1 and 4 are the candidate next runs (@exp:az-followups).

#repro(```sh
# Not reproducible: a code reading of cullback/canopy plus the CPU probe, whose script is not in the repo
```)

=== Follow-ups not run #chip("todo") <exp:az-followups>
#meta(
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[Candidate next steps from the analyses above, all Claude's suggestions. Marco moved all AlphaZero work to the bottom of the queue on 2026-10-24: Canopy loses to our PPO-plus-search bot, so PPO into search is the path.]

#lead[Plan.]
- A Canopy-shaped target: Q unrescaled, `c_scale` 1, no prior temperature (divergence 1 of @exp:az-canopy-diff). (Claude's suggestion, not queued)
- Per-simulation world resampling instead of one world per search (divergence 4). (Claude's suggestion, not queued)
- An offline distillation gate before the online loop: one fixed set of 1024-sim targets, several policy-only fits judged head to head. It would answer "is there anything to distill" more cheaply than the online loop. (Claude's suggestion, not queued)
- Cheaper self-play: an engine top-up budget (15–20% fewer leaves) and an engine-side eval cache (31–46% fewer forwards), from @exp:az-cost. (Claude's suggestion, not queued)
- The warm-start critic is a discounted V-trace critic while AlphaZero trains undiscounted win/loss, so the value head drifts during the test. The critic refit (@sec:critic) measures that discount.

The search finals of `az-2p-2048` (`configs/experiments/az-2p-2048-search.yaml`: best and init in the 2048-sim search against the raw start) wait on a positive raw result and were never played.
