#import "../macros.typ": *

= alphazero <sec:az>

marco's note on the idea: "currently agreement is really high, doesn't seem to yield many gains". the question is whether training the network on its own search's choices (alphazero self-play) gives a better network, as a raw policy or inside the search. the loop (`train.algo: alphazero`) self-plays through the batched search with gumbel roots, keeps a replay buffer of (position, visit-count policy, per-seat outcome) and warm-starts from a ppo checkpoint. so far no run beat its start: the raw policy got worse every time, inside the search the trained networks played the same (e8) or worse (the soft run), and the likeliest culprit is how the targets are shaped.

== first runs and cost

=== 30-minute sanity run from a weak ppo net #chip("done") <exp:az-from-ppo>
#meta(
  dates: "2026-10-14",
  owner: "marco",
  config: "configs/experiments/az-from-ppo.yaml",
  wandb: (),
  records: (),
)
#tldr[a first end-to-end test of the loop, from a weak 40-minute ppo net at 57.8% vs F. the raw policy got steadily worse, down to 18.5% vs F, while fitting noisy 64-sim targets. it says nothing about strong networks or bigger searches.]

*question.* does the loop run, and does a weak network learn from its own 64-sim search?

*setup.* the az stage of `az-from-ppo`, run for 30 minutes in a scratch runs dir (since deleted, so no records and no wandb mirror). the start was a 40-minute ppo sanity net, not puffer-graph2. oracle search (`dets: 0`: the search sees the true hidden state), gumbel targets at 64 sims on a quarter of moves and 16 on the rest, pure self-play, lr 5e-5, 25 iterations. nothing was played head to head against the start.

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    header: ([player], [vs F], [games]),
    [start, raw policy], [57.8], [],
    [start, 64-sim search], [63.5], [100],
    [trained raw policy, three successive checkpoints], [46.6 → 24.4 → 18.5], [600 each],
  ),
  caption: [`az-from-ppo`'s az stage. no records survive, so there are no intervals.],
)

*reading.* the targets were noisy. with q min-max rescaled over the root's moves, a near-single-visit q gap swings a target by \~6.5 logits. two search seeds agree on the top move 87% of the time (60% on discards). knight-before-roll (how often the policy plays a knight before rolling) went from 0.90 to 0.27, and fitting the noise took the network's entropy from 0.42 to 0.82 until its argmax drifted to end-turn. the run found one small bug, the gumbel temperature-0 finalist (fixed in cppanatron `5d9dda9`), which doesn't touch the targets. for context, v1 never trained above 512 sims, and 512-sim targets didn't move an 87% network (v1’s EX-1d, a frozen-teacher distillation of an 8 × 64-sim search: +1.3 vs F and 50.6% head to head, both null).

#repro(```sh
# runs the ppo stage (50M steps) and then 300 az iterations; the original ran
# only the az stage, for 25 iterations, from a 40-minute ppo net
catanrl run configs/experiments/az-from-ppo.yaml
```)

=== self-play cost study (cpu) #chip("done") <exp:az-cost>
#meta(
  dates: "2026-10-21",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[before spending gpu time i measured on cpu what makes self-play expensive. tree reuse and transpositions save nothing as built; an eval cache across one self-play step would cut network forwards by 31–46%. a config-only first run at 1024-sim targets costs \~13–22 gpu hours for 60 iterations.]

*question.* does the search reuse built for the deployed bot make self-play cheaper, and what does a first run cost?

*setup.* cpu measurements of self-play searches in 2p and 4p. the 2p numbers were measured under the engine's default rules (10 vp, discard above 7), not base-2p's (15 vp, discard above 9); the target probes of `az-2p-2048` (@exp:az-2p-2048) used base-2p's rules.

*result.*
#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([lever], [effect on self-play]),
    [tree reuse, as built], [the tree survives 43% (2p) to 53% (4p) of moves, but `advance` resets the budget, so it saves no leaves],
    [engine top-up budget (search only sims − kept visits)], [15–20% fewer leaves. kept visits are worth about their count, so reuse can't halve the budget],
    [transpositions], [nothing. they fight reuse (they restart the new root's shared children), so they stay off],
    [eval cache across one self-play step], [31–46% fewer forwards. hashing has to move into the engine],
  ),
  caption: [cpu measurements of self-play search.],
)
a 1024-sim move costs \~1,000 leaf rows and \~80 ms of engine cpu. a first run as a config-only change of the existing trainer (1024-sim gumbel targets on every move, 512 games, from `puffer-graph2/public`) is \~13–22 gpu hours for 60 iterations.

*reading.* the real lever is the eval cache, and it needs engine work. marco chose to go straight to the loop at 2048 sims and do engine work only after a positive result (@exp:az-2p-2048). the engine levers are listed in @exp:az-followups.

#repro(```sh
# not reproducible: one-off cpu measurements, no script for them is in the repo
```)

== the 2p specialist at 2048-sim targets

=== az-2p-2048: self-play from the 2p specialist #chip("stopped") <exp:az-2p-2048>
#meta(
  dates: "2026-10-21 15:40–18:53",
  owner: "marco",
  config: "configs/experiments/az-2p-2048.yaml",
  wandb: (("az-2p-2048-main-7e569a67", "az-2p-2048/main"),),
  records: ("e34b2057", "fa218f06", "14f41aa1"),
)
#tldr[marco's pick: the best 2p network straight into self-play, with 2048-sim targets on every move. its raw policy lost to its start (48.0 and 48.2 head to head), so the stop rule ended the run after 8 iterations. the cause looks like the targets: gumbel's target puts ≥ 0.99 on one move in most decisions, whatever the real gap between moves.]

*question.* does training on 2048-sim search targets improve an already strong network (95% vs F)?

*setup.* start `puffer-graph2/public@e3600` (actor and critic), base-2p, 512 games in flight. fair gumbel search on every move (`dets: 1`, one world dealt from the tracker per search; m = 16 root moves considered). 30 iterations × 16,384 targets planned, lr 5e-5. every 2 iterations the raw policy plays its start head to head on `select-2p-2k` (4,000 games); the search finals run only if that is positive. logging-only monitors (training is bit-identical with or without them): the value head's loss on finished games before it trains on them, how often the search overrules the network's argmax, and drift from the start on a fixed probe of 4,096 positions. stop rule: two heads to head ≤ 48.5 in a row.

cpu probes under base-2p rules came first. 2048-sim targets cut the prior's distance to an 8192-sim reference from 0.295 to 0.197 tv (the reference's own noise floor is 0.179) and change the top move at 20% of positions. they are barely better than 1024 sims (−0.016 #ci[−0.034][0.001]). robber moves, and positions where the opponent holds a hidden vp card, stay noisy. the first iteration ran at 23.6k leaves/s and took 24 min, so \~12 h for the 30.

*result.* stopped at e8 (3.2 h). best stayed e0, so no finals ran (`az-2p-2048-search.yaml` was never played).
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([raw policy vs its start], [score], [games], [record]),
    [e0 (the same weights)], [50.0 #ci[48.5][51.5]], [4,000], [`e34b2057`],
    [e6], [*48.0* #ci[46.5][49.5]], [4,000], [`fa218f06`],
    [e8], [*48.2* #ci[46.7][49.7]], [4,000], [`14f41aa1`],
  ),
  caption: [`az-2p-2048` selects on `select-2p-2k`. e2 and e4 equal e0: the first update came in iteration 5, once enough games had finished.],
)
from the start to e8 the network's entropy rose from 0.088 to 0.125. the search overruled its argmax more often (0.18 → 0.23), where a policy that learns from its search should make that rarer. knight-before-roll on the drift probe fell from 0.53 to 0.43.

*reading.* it went the way the sanity run went, only slower. the critic wasn't the cause. the monitor's out-of-sample value loss rose from 0.20 to 0.54, but that is the games getting longer (the first games to finish are the short ones). on the same rows the start scores 0.375 / 0.508 / 0.387 at iterations 6–8 and the trained net 0.359 / 0.538 / 0.402, a few hundredths either way.

the targets were the cause, as far as one look can say. gumbel's σ(q) = (50 + max visits) · 0.1 · q, with q min-max rescaled to [0, 1] across the root's moves, puts a \~100-logit spread between the best and the worst move at 2048 sims, whatever the gap in win probability. so all 16,153 two-move decisions in the replay (mostly knight or roll) got a target ≥ 0.99 on one move, and so did 81% of the decisions with more than eight legal moves. on knight-or-roll the start was ≥ 90% sure 98% of the time. the targets went against its argmax 21% of the time and played the knight 44% of the time against the start's 53%: the drift the network then learned.

the moves behind the targets are good: the same search over the start's weights beats the raw policy 62.4 (`831eea84`, @exp:az-e8-search), far above puct over the same policy (54.9 at 1024 sims, 55.2 at 4096, @sec:search). the search had plenty to teach, and the raw policy still got worse learning it.

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
#tldr[marco's point: the goal is a better network inside the search, so judge alphazero there, not by the raw policy. in the same 2048-sim search against the same raw start, e8 scores 61.8 and the start's own weights 62.4: −0.6 pp paired, no difference. the training didn't make a better search network either.]

*question.* does e8 search better than its start, even though its raw policy got worse?

*setup.* two rows, each the same 2048-sim gumbel search (`dets: 1`, m = 16, `c_puct` 0.75) against the same raw start (`puffer-graph2/public@e3600`), 1,000 games on the first 500 boards of `heldout-2p` in both seat orders. only the network inside the search differs. the rows are paired: they play the same game ids, so they are compared game by game, with mcnemar's test on the games only one of them won (`tools/search_probe.py --pair`).

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([network inside the search, vs the raw start], [score], [record]),
    [start (e0)], [62.4 #ci[59.4][65.3]], [`831eea84`],
    [e8 of `az-2p-2048`], [61.8 #ci[58.7][64.8]], [`c4ea7b52`],
    mid,
    [e8 − e0, paired over the 1,000 shared games], [*−0.6 pp* #ci[−3.7][2.5]], [124 games won only by e8, 130 only by e0, mcnemar p = 0.75],
  ),
  caption: [2048-sim gumbel search vs the raw start on `heldout-2p` (500 boards, both seat orders).],
)

*reading.* the raw policy lost two points and the search played the same, so stopping at e8 cost nothing, but by luck. only a quarter of the pairs differ at all (the board, the seat and the dice decide most games), so pairing roughly halves the interval two independent rows would give. this paired test is the template: from here on every alphazero run is judged this way, and so is the critic refit (@sec:critic).

#repro(```sh
# the yaml of the e8 row names runs/az-2p-2048/main/checkpoints/e000008.pt by absolute path
catanrl run configs/experiments/gumbel-edge-2p.yaml      # start's weights in the search: 831eea84
catanrl run configs/experiments/gumbel-edge-2p-e8.yaml   # e8 in the search: c4ea7b52
uv run python tools/search_probe.py --pair c4ea7b5234eb6a50 831eea84f04c89d8
```)

== softer targets

=== az-2p-2048-soft: softer targets through a tempered prior #chip("stopped") <exp:az-soft>
#meta(
  dates: "2026-10-22 15:49 – 2026-10-23 01:57",
  owner: "marco",
  config: "configs/experiments/az-2p-2048-soft.yaml",
  wandb: (("az-2p-2048-soft-main-8105fbb1", "az-2p-2048-soft/main"),),
  records: ("856a0421", "ae5a4000", "f016fee8", "8906dcf7"),
)
#tldr[marco's next step: the same loop with the search unchanged and softer targets. the softening that looked best on cpu, tempering the prior inside the target, flattened the network instead: its search fell from 57.4 to 46.7 against the raw start by e15, and marco stopped the run. the prior temperature (my proposal) is the likely culprit, since canopy never tempers the policy its search reads.]

*question.* does a network trained on softer versions of the same search's targets search better?

*setup.* the trainer rebuilds gumbel's improved policy itself, from each search's visits and q and the network's root outputs (`train.policy_target: gumbel`). at the engine's constants it matches the engine's target to \~3e-8, and the default trains bit-identically to before.

the softening was picked on cpu: 600 positions of the start's raw-policy base-2p self-play, two 2048-sim searches each, scored by −log of the target's mass on an 8192-sim search's move. a gentler σ alone softens nothing. the start's prior is itself one-hot (a median gap of 47 nats between its top two logits on knight-or-roll decisions), so softmax(prior + σ) stays one-hot at any scale. what does soften it is tempering the prior inside the target: softmax(logits / 10 + (50 + max visits) · 0.15 · q), with q unrescaled (my proposal from the probe, approved by marco).
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([target], [score vs engine (nats)], [agrees with the 8192 move], [entropy], [≥ 0.99 on one move]),
    [engine (σ on rescaled q, `c_scale` 0.1)], [], [81.4%], [0.05], [87%],
    [best σ-only setting], [−0.10 #ci[−0.24][0.03]], [], [], [],
    [tempered prior (logits / 10, `c_scale` 0.15, q unrescaled)], [*−1.12* #ci[−1.39][−0.86]], [80.5%], [0.64], [22%],
  ),
  caption: [cpu probe, 600 positions. the score is −log of the target's mass on the 8192-sim move, relative to the engine's target; lower is better.],
)
that is calibration, not better moves: two 8192-sim searches pick the same move only 86% of the time, and the expected regret under the softer target is slightly worse (+0.05 #ci[0.01][0.08] win points).

everything else is as in `az-2p-2048`. it is judged in the search throughout, as marco asked: every 5th iteration the checkpoint in a 512-sim gumbel search plays the raw start on the same 1,000 games of `select-2p-500`, paired with e0’s row. the planned finals put the best and the last checkpoint in the 2048-sim search against the raw start on the games of `831eea84`, each paired with it (mcnemar, holm over the two). every iteration's weights are kept, so an e8 row can pair with `c4ea7b52` later. stop rule: only a collapse, two selects in a row more than 4 pp below e0 on the same games (about 2.5 paired sds); the raw policy is not a stop criterion this time. planned cost: \~19 h of training and selects, then 4–8 h of finals.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, left),
    header: ([checkpoint in the search], [score vs raw start], [paired vs e0], [only e0 / only eN], [mcnemar p], [record]),
    [e0], [57.4 #ci[54.3][60.4]], [], [], [], [`856a0421`],
    [e5], [56.0 #ci[52.9][59.0]], [−1.4 pp], [129 / 115], [0.41], [`ae5a4000`],
    [e10], [53.8 #ci[50.7][56.9]], [−3.6 pp], [158 / 122], [0.036], [`f016fee8`],
    [e15], [*46.7* #ci[43.6][49.8]], [*−10.7 pp*], [225 / 118], [< 10⁻⁶], [`8906dcf7`],
  ),
  caption: [`az-2p-2048-soft` selects: 512-sim gumbel search vs the raw start, 1,000 games of `select-2p-500` each.],
)
the monitors say why. by e13 the network's entropy was 0.68 (0.09 at the start), it agreed with the start's argmax on 72% of the drift probe, and the search overruled its argmax 41% of the time (18% at the start). at e15 the entropy was 0.92 and the search's depth had fallen from 11 to 8.2. the collapse rule would have needed e20 too; marco stopped the run after e15 ("it is sucking balls"). best stays e0, so there are no finals.

*reading.* the softer target pulled the prior flat faster than the search's q could replace what the prior knew. the canopy comparison (@exp:az-canopy-diff) puts the blame on the prior temperature: canopy never tempers the policy the search reads.

#repro(```sh
catanrl run configs/experiments/az-2p-2048-soft.yaml   # marco stopped it by hand after the e15 select
uv run python tools/search_probe.py --pair 8906dcf7adfc1c96 856a0421185ce86f   # e15 vs e0, paired
```)

== what next

=== what canopy did, and where we diverge #chip("done") <exp:az-canopy-diff>
#meta(
  dates: "2026-10-23",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[marco asked how canopy trained its alphazero network. canopy trained from scratch with a recipe close to ours, but published no win rates, and nobody has shown the recipe improves an already strong network. the likeliest reasons our runs failed are our target's q scaling and the soft run's prior temperature.]

*question.* what did canopy do, and which of our differences could explain our two failed runs?

*setup.* read from cullback/canopy's code and history, checked against source, plus one more measurement on the 600-position cpu probe of @exp:az-soft.

*result.* canopy's recipe next to ours:
#figure(
  btable(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    header: ([], [canopy], [ours (`az-2p-2048`)]),
    [start], [a 2.4M-parameter nexus-v3 net, most likely from random weights], [`puffer-graph2/public@e3600`, 95% vs F],
    [rules], [1v1 colonist (balanced dice, friendly robber)], [base-2p],
    [search], [gumbel, m = 6; 1600 sims on a quarter of moves, 64-sim value-only searches on the rest; an improved-policy interior], [gumbel, m = 16; 2048 sims on every move],
    [hidden cards], [the opponent's dev cards resampled every simulation], [one tracker world per search],
    [q in the target], [divided by the span of values the search saw, `c_scale` 1], [min-max rescaled over the root's moves, `c_scale` 0.1],
    [softening], [a soft-policy side output (π^(1/4), weight 8) that the search never reads], [none (`az-2p-2048`); a tempered prior inside the target (`az-2p-2048-soft`)],
    [value], [win/draw/loss, blended from the outcome toward the search's q (to 85% over 60 iterations)], [per-seat outcome],
    [replay, optimizer], [500k positions; adamw 5e-4 → 1e-4], [400k positions; lr 5e-5, a fresh optimizer on a muon-trained net],
    [length], [about 315 iterations (\~47M positions, inferred from the checkpoint name)], [8 and 15 iterations],
  ),
  caption: [canopy's alphazero recipe against ours.],
)
canopy's evidence of success is thin. it publishes no win rates, elo or opponents, only a readme line ("we believe ... the strongest public 1v1 catan agent"). the only outside numbers are v1’s: its graph checkpoints scored 23–33% against canopy at 64 sims (48 games each).

the divergences most likely to matter:
+ *how the target scales q.* our engine min-max rescales q over the root's moves at `c_scale` 0.1. so on a two-move decision any q lead becomes a \~107-logit lead. canopy divides by the span of values its search saw, at `c_scale` 1, so near-ties stay near-ties. on the probe's 600 positions a canopy-shaped target (unrescaled q, `c_scale` 1, no temperature) overrules the start on knight-or-roll 2.2% of the time, against the engine target's 18.9%. it agrees with the 8192-sim move as often (88.1 vs 87.8%) and stays sharp (entropy 0.08). the −log score of @exp:az-soft couldn't see this difference.
+ *the prior temperature.* the soft run's temperature is not canopy's (canopy softens only through its separate output), and it is what flattened the soft run.
+ *the start.* canopy's search targets were always far better than its prior. from a 95% start the search's gain is close to its own noise. the hand-off also switches the critic's objective and starts a fresh adamw on a muon-trained net.
+ *hidden cards.* one sampled world per search, against canopy's per-simulation resampling.

*reading.* v1 ran canopy's recipe (its puct version) on a 55%-vs-F graph net and it did improve it, 55 → 70.5% raw vs F over 10 iterations, but never from a ≥ 90% start. divergences 1 and 4 are the candidate next runs (@exp:az-followups).

#repro(```sh
# not reproducible: a code reading of cullback/canopy plus the cpu probe, whose script is not in the repo
```)

=== follow-ups not run #chip("todo") <exp:az-followups>
#meta(
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[candidate next steps from the analyses above, all my suggestions. marco moved all alphazero work to the bottom of the queue on 2026-10-24: canopy loses to our ppo-plus-search bot, so ppo into search is the path.]

*plan.*
- a canopy-shaped target: q unrescaled, `c_scale` 1, no prior temperature (divergence 1 of @exp:az-canopy-diff). (my suggestion, not queued)
- per-simulation world resampling instead of one world per search (divergence 4). (my suggestion, not queued)
- an offline distillation gate before the online loop: one fixed set of 1024-sim targets, several policy-only fits judged head to head. it would answer "is there anything to distill" more cheaply than the online loop. (my suggestion, not queued)
- cheaper self-play: an engine top-up budget (15–20% fewer leaves) and an engine-side eval cache (31–46% fewer forwards), from @exp:az-cost. (my suggestion, not queued)
- the warm-start critic is a discounted v-trace critic while alphazero trains undiscounted win/loss, so the value head drifts during the test. the critic refit (@sec:critic) measures that discount.

the search finals of `az-2p-2048` (`configs/experiments/az-2p-2048-search.yaml`: best and init in the 2048-sim search vs the raw start) wait on a positive raw result and were never played.
