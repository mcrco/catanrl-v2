#import "../macros.typ": *

= AlphaZero <sec:az>

Marco's note on the idea: "currently agreement is really high, doesn't seem to yield many gains". The loop (`train.algo: alphazero`) warm-starts from a PPO checkpoint, self-plays through the batched search with Gumbel roots and trains on a replay buffer of (position, visit-count policy, per-seat outcome).

Marco's rule: judge a run inside the search, with MCTS(new) and MCTS(old) each playing the raw start on the same games, paired by McNemar's test, not by raw head to head (@exp:az-e8-search). No run has beaten its start: the raw policy always got worse, and inside the search the trained networks played the same (e8) or worse (the soft run), most likely because of the targets' shape.

== First runs and cost

=== 30-minute sanity run from a weak PPO net #chip("done") <exp:az-from-ppo>
#meta(
  dates: "2026-10-14",
  owner: "marco",
  config: "configs/experiments/az-from-ppo.yaml",
  wandb: (),
  records: (),
)
#tldr[A first end-to-end test from a weak 40-minute PPO net (57.8% vs F): fitting noisy 64-sim targets, its raw policy fell to 18.5% vs F. It says nothing about strong nets or bigger searches.]

*Question.* Does the loop run, and does a weak net learn from 64-sim search?

#lead[Setup.]
- 30 minutes (25 iterations) of `az-from-ppo`'s AlphaZero stage from a 40-minute PPO sanity net, not `puffer-graph2`, in a since-deleted scratch directory (no records, no W&B).
- Oracle search (`dets: 0`); Gumbel targets at 64 sims on a quarter of moves, 16 on the rest; pure self-play; learning rate 5e-5; no head to head against the start.

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
  caption: [`az-from-ppo`'s AlphaZero stage against F. No records survive, so no intervals.],
)

#lead[Reading.]
- *Noisy targets.* With Q min-max rescaled over the root's moves, a near-single-visit Q gap swings a target by \~6.5 logits; two search seeds agree on the top move 87% of the time (60% on discards).
- *It fit the noise:* entropy rose from 0.42 to 0.82 until the argmax drifted to end-turn; knight-before-roll fell from 0.90 to 0.27.
- A small bug (the Gumbel temperature-0 finalist, fixed in cppanatron `5d9dda9`) does not touch the targets.
- v1 never trained above 512 sims; its EX-1d, distilling an 8 × 64-sim search, did not move an 87% network (+1.3 vs F, 50.6% head to head, both null).

#repro(```sh
# PPO stage (50M steps), then 300 AlphaZero iterations; the original ran only 25,
# from a 40-minute PPO net
catanrl run configs/experiments/az-from-ppo.yaml
```)

=== Self-play cost study (CPU) #chip("done") <exp:az-cost>
#meta(
  dates: "2026-10-21",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[On CPU, only an eval cache across one self-play step would make self-play cheaper (31–46% fewer network forwards). A config-only first run at 1024-sim targets costs \~13–22 GPU hours for 60 iterations.]

*Question.* What makes self-play expensive, and what would a first run cost?

*Setup.* CPU self-play searches in 2p and 4p; 2p under the engine's default rules (10 VP, discard above 7), not `base-2p`'s (15 VP, discard above 9).

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([Lever], [Effect on self-play]),
    [Tree reuse, as built], [Survives 43% (2p) to 53% (4p) of moves, but `advance` resets the budget: no leaves saved.],
    [Engine top-up budget \ #sub[search only sims − kept visits]], [15–20% fewer leaves; kept visits are worth about their count, so reuse cannot halve the budget.],
    [Transpositions], [Nothing: they fight reuse (restarting the new root's shared children); off.],
    [Eval cache across one \ self-play step], [31–46% fewer forwards; hashing must move into the engine.],
  ),
  caption: [CPU measurements of self-play search.],
)
A 1024-sim move costs \~1,000 leaf rows and \~80 ms of engine CPU; the run estimate assumes 1024-sim Gumbel targets on every move, 512 games, from `puffer-graph2/public`.

*Reading.* The eval cache is the real lever, and it needs engine work. Marco chose to go straight to 2048 sims and do engine work only after a positive result (@exp:az-2p-2048).

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
#tldr[Marco's pick: the best 2p network into self-play with 2048-sim targets on every move. Its raw policy lost to its start (48.0 and 48.2 head to head), and the stop rule ended the run at e8. Likely cause: Gumbel's target puts ≥ 0.99 on one move in most decisions, whatever the real gap.]

*Question.* Do 2048-sim targets improve an already strong network (95% vs F)?

#lead[Setup.]
- Start `puffer-graph2/public@e3600` (actor and critic), `base-2p`, 512 games in flight; fair Gumbel search on every move (`dets: 1`, m = 16).
- 30 iterations × 16,384 targets planned, learning rate 5e-5: \~12 h (iteration 1 took 24 min at 23.6k leaves/s).
- Every 2 iterations the raw policy plays its start on `select-2p-2k`; search finals only if positive. Stop rule: two heads to head ≤ 48.5 in a row.
- Logging-only monitors: out-of-sample value loss, how often the search overrules the argmax, and drift on a fixed 4,096-position probe.
- A CPU probe (`base-2p`) came first: 2048-sim targets cut the prior's total-variation distance to an 8192-sim reference from 0.295 to 0.197 (noise floor 0.179) and change the top move at 20% of positions, barely better than 1024 sims (−0.016 #ci[−0.034][0.001]). Robber moves and hidden opponent VP cards stay noisy.

*Result.* Stopped at e8 (3.2 h); best stayed e0, so no finals.
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Raw policy vs its start], [Score], [Games], [Record]),
    [e0 (the same weights)], [50.0 #ci[48.5][51.5]], [4,000], [`e34b2057`],
    [e6], [*48.0* #ci[46.5][49.5]], [4,000], [`fa218f06`],
    [e8], [*48.2* #ci[46.7][49.7]], [4,000], [`14f41aa1`],
  ),
  caption: [`az-2p-2048`'s selects on `select-2p-2k`. e2 and e4 equal e0: the first update came in iteration 5.],
)
From e0 to e8 entropy rose from 0.088 to 0.125, the search overruled the argmax more often instead of less (0.18 → 0.23), and knight-before-roll on the probe fell from 0.53 to 0.43.

*Reading.* The sanity run's failure, slower:
- *Not the critic.* Its value loss rose from 0.20 to 0.54 only because longer games finish later; on the same rows the start scores 0.375 / 0.508 / 0.387 (iterations 6–8) and the trained net 0.359 / 0.538 / 0.402.
- *The targets, as far as one look can say.* Gumbel's σ(Q) = (50 + max visits) · 0.1 · Q, with Q min-max rescaled to [0, 1] over the root's moves, gives a \~100-logit spread at 2048 sims whatever the win-probability gap. So ≥ 0.99 landed on one move in all 16,153 two-move decisions in the replay (mostly knight or roll) and in 81% of those with more than eight legal moves. On knight-or-roll, where the start was ≥ 90% sure 98% of the time, the targets overruled its argmax 21% of the time and played the knight 44% of the time (start: 53%): the drift the network learned.
- *Not the moves.* The same search over the start's weights beats the raw policy 62.4% (`831eea84`, @exp:az-e8-search), far above PUCT (54.9% at 1,024 sims, 55.2% at 4,096; @sec:search).

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
#tldr[Marco's test, inside the search: in the same 2048-sim search against the raw start, e8 scores 61.8% and the start's weights 62.4%: −0.6 pp paired, no difference.]

*Question.* Does e8 search better than its start despite its worse raw policy?

*Setup.* A 2048-sim Gumbel search (`dets: 1`, m = 16, `c_puct` 0.75) with e0 or e8 inside, each against the raw start `puffer-graph2/public@e3600` on the same 1,000 games (first 500 boards of `heldout-2p`, both seat orders); McNemar via `tools/search_probe.py --pair`.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([Network inside the search], [Score vs the raw start], [Record]),
    [Start (e0)], [62.4 #ci[59.4][65.3]], [`831eea84`],
    [e8 of `az-2p-2048`], [61.8 #ci[58.7][64.8]], [`c4ea7b52`],
    mid,
    [e8 − e0, paired], [*−0.6 pp* #ci[−3.7][2.5]], [McNemar p = 0.75],
  ),
  caption: [2048-sim Gumbel search against the raw start on `heldout-2p` (1,000 games).],
)
124 games were won only by e8, 130 only by e0.

*Reading.* Stopping at e8 cost nothing, by luck. Only a quarter of the pairs differ, so pairing roughly halves the interval of two independent rows. This paired test is the template for every later AlphaZero run and the critic refit (@sec:critic).

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
#tldr[Marco's next step: softer targets, same loop and search. Tempering the prior inside the target, best on CPU, flattened the network: its search fell from 57.4% to 46.7% against the raw start by e15, and Marco stopped the run. The prior temperature (Claude's proposal) is the likely culprit.]

*Question.* Do softer targets give a better search network?

*Setup.* The trainer rebuilds Gumbel's improved policy from each search's visits, Q and root outputs (`train.policy_target: gumbel`); at the engine's constants it matches the engine's target to \~3e-8.

The softening was picked on a CPU probe (600 positions of the start's `base-2p` self-play, two 2048-sim searches each, scored by −log of the target's mass on an 8192-sim search's move). A gentler σ alone softens nothing: the prior is one-hot (a median gap of 47 nats between its top two logits on knight-or-roll). Tempering it does: softmax(logits / 10 + (50 + max visits) · 0.15 · Q), Q unrescaled (Claude's proposal, approved by Marco).
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Target], [Score vs engine \ (nats)], [Agrees with \ 8192-sim move], [Entropy], [≥ 0.99 on \ one move]),
    [Engine \ #sub[σ on rescaled Q, `c_scale` 0.1]], [], [81.4%], [0.05], [87%],
    [Best σ-only setting], [−0.10 #ci[−0.24][0.03]], [], [], [],
    [Tempered prior \ #sub[logits / 10, `c_scale` 0.15, Q unrescaled]], [*−1.12* #ci[−1.39][−0.86]], [80.5%], [0.64], [22%],
  ),
  caption: [CPU probe, 600 positions. Score relative to the engine's target; lower is better.],
)
That is calibration, not better moves: two 8192-sim searches agree on only 86% of moves, and the softer target's expected regret is slightly worse (+0.05 #ci[0.01][0.08] win points).

Otherwise as `az-2p-2048`, judged in the search as Marco asked:
- Every 5th iteration the checkpoint, in a 512-sim Gumbel search, plays the raw start on `select-2p-500`, paired with e0.
- Planned finals: best and last checkpoints in the 2048-sim search on `831eea84`'s games, paired with it (Holm over the two).
- Stop only on collapse: two selects in a row more than 4 pp below e0 (about 2.5 paired SDs). Planned cost \~19 h, then 4–8 h of finals.

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
  caption: [`az-2p-2048-soft`'s selects: 512-sim search against the raw start on `select-2p-500`.],
)
By e13 entropy was 0.68 (start 0.09), probe agreement with the start's argmax 72%, and search overrules 41% (start 18%); at e15 entropy was 0.92 and search depth 8.2 (start 11). The collapse rule also needed e20; Marco stopped the run after e15 ("it is sucking balls"), so no finals.

*Reading.* The softer target flattened the prior faster than the search's Q could replace what it knew. Canopy never tempers the policy its search reads (@exp:az-canopy-diff).

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
#tldr[Marco asked how Canopy trained its AlphaZero network: from scratch, with a recipe close to ours, but no published win rates and no evidence it helps a strong network. Likeliest causes of our failures: the target's Q scaling and the soft run's prior temperature.]

*Question.* Which differences from Canopy could explain our two failed runs?

*Setup.* cullback/canopy's code and history, checked against source, plus one measurement on the probe of @exp:az-soft.

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    header: ([], [Canopy], [Ours (`az-2p-2048`)]),
    [Start], [2.4M-parameter nexus-v3 net, likely from random weights], [`puffer-graph2/public@e3600`, 95% vs F],
    [Rules], [1v1 Colonist (balanced dice, friendly robber)], [`base-2p`],
    [Search], [Gumbel, m = 6; 1,600 sims on a quarter of moves, 64-sim value-only on the rest; improved-policy interior], [Gumbel, m = 16; 2,048 sims on every move],
    [Hidden cards], [Opponent's dev cards resampled every simulation], [One tracker world per search],
    [Q in the target], [Divided by the span of values the search saw, `c_scale` 1], [Min-max rescaled over the root's moves, `c_scale` 0.1],
    [Softening], [Soft-policy side output (π^(1/4), weight 8) the search never reads], [None; `az-2p-2048-soft` tempers the prior inside the target],
    [Value], [Win/draw/loss outcome blended toward the search's Q (to 85% over 60 iterations)], [Per-seat outcome],
    [Replay, \ optimizer], [500k positions; AdamW 5e-4 → 1e-4], [400k positions; 5e-5, a fresh optimizer on a Muon-trained net],
    [Length], [\~315 iterations (\~47M positions, inferred from checkpoint name)], [8 and 15 iterations],
  ),
  caption: [Canopy's AlphaZero recipe next to ours.],
)
Canopy publishes no win rates, Elo or opponents, only a README line ("we believe ... the strongest public 1v1 catan agent"); v1's graph checkpoints scored 23–33% against it at 64 sims (48 games each). The divergences that likely matter:
+ *Q scaling.* Our rescaling turns any Q lead on a two-move decision into a \~107-logit lead; Canopy's keeps near-ties near. On the probe a Canopy-shaped target (unrescaled Q, `c_scale` 1, no temperature) overrules the start on knight-or-roll 2.2% of the time (engine: 18.9%) with the same agreement with the 8192-sim move (88.1 vs 87.8%) and entropy 0.08; the −log score missed this.
+ *Prior temperature.* Canopy softens only through a separate output; our temperature flattened the soft run.
+ *The start.* Canopy's targets were always far better than its prior; from 95% the search's gain is near its noise, and the hand-off also switches the critic's objective and the optimizer.
+ *Hidden cards.* One world per search vs per-simulation resampling.

*Reading.* v1's run of Canopy's PUCT recipe raised a 55%-vs-F graph net to 70.5% raw vs F in 10 iterations, never from ≥ 90%. Divergences 1 and 4 are the candidate next runs (@exp:az-followups).

#repro(```sh
# Not reproducible: a code reading of cullback/canopy plus the CPU probe, whose script is not in the repo
```)

=== Follow-ups not run #chip("todo") <exp:az-followups>
#meta(
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[All Claude's suggestions. Marco moved AlphaZero to the bottom of the queue on 2026-10-24: Canopy loses to our PPO-plus-search bot, so PPO into search is the path.]

#lead[Plan.]
- A Canopy-shaped target (divergence 1 of @exp:az-canopy-diff). Tested offline in @exp:az-gate: no effect inside the search.
- Per-simulation world resampling (divergence 4). (Claude's suggestion, not queued)
- An offline distillation gate: fits to one fixed set of search targets, as a cheap test of whether there is anything to distill. Done as @exp:az-gate (approved by Marco on 2026-11-09): null for all three targets, the control included.
- Cheaper self-play: the engine top-up budget and eval cache (@exp:az-cost). (Claude's suggestion, not queued)
- The warm-start critic is discounted V-trace while AlphaZero trains undiscounted win/loss, so the value head drifts; the critic refit (@sec:critic) measures that.

`az-2p-2048`'s search finals (`configs/experiments/az-2p-2048-search.yaml`: best and init in the 2048-sim search) were never played.

== The offline distillation gate

=== Fit fr3 once to fixed 2048-sim searches: `az-gate-2p` #chip("done") <exp:az-gate>
#meta(
  dates: "2026-11-10 08:14 – 13:32",
  owner: "marco",
  config: "configs/experiments/az-gate-2p.yaml",
  wandb: (("az-gate-2p-positions-b6aa1044", "positions"), ("az-gate-2p-engine-f61f6a78", "engine"), ("az-gate-2p-canopy-dc666ac0", "canopy"), ("az-gate-2p-canopy-anchor-f6be5e75", "canopy-anchor")),
  records: ("3a55789c", "41389abf", "5757cfc6", "ec88ecb4", "b407cba0", "c6027c5d", "08b96b96"),
)
#tldr[Claude's design, approved by Marco on 2026-11-09. Search 102,400 of fr3’s own positions once with `az-2p-2048`’s search, fit fr3 to three target shapes at `az-2p-2048`’s e6 dose, and judge each fit inside the search against fr3’s own search on the same 2,000 games. Null: inside the search the three fits score −0.15, −0.25 and −0.65 pp against fr3 (each ±2.3 pp). The control did not lose either, so at this dose the gate cannot see what broke `az-2p-2048`. Neither could the search harness for `az-2p-2048`'s own e8 (@exp:az-e8-search). By the rule fixed before launch, no full AlphaZero loop follows.]

*Question.* Do 2048-sim search targets hold anything that makes fr3 a better search network, and which target shape keeps the network intact?

#lead[Setup.]
- *Start.* fr3 (`2p-graph-s-3`, `ppo-2p-fr3/full@e4350`): `graph-s` with one shared trunk, `full` observation level, trained on `base-2p-fr`.
- *Positions.* 5,120 games of fr3’s raw self-play (moves sampled as PPO samples them) on its own 4,096 training boards. From each game, 20 decisions drawn uniformly from those with more than one legal move: 102,400 positions. 512 games (10%) are held out for validation.
- *Search.* Each position is searched once with `az-2p-2048`’s search (Gumbel root, 2,048 sims, m = 16, `c_puct` 0.75) on the true game (`dets: 0`): fr3 reads the true state, so its targets should be about it. The dataset keeps everything needed to rebuild a target offline (fr3’s root logits and values, visits, Q, the engine's own target), with its sha256.
- *Targets.* Three fits, each from fr3:
  - `engine`: the engine's Gumbel target (Q min-max rescaled, `c_scale` 0.1), which `az-2p-2048` trained on. The control: it should reproduce that run's drop.
  - `canopy`: Q unrescaled, `c_scale` 1, no prior temperature. The Canopy-shaped target of the CPU probe (@exp:az-canopy-diff).
  - `canopy-anchor`: half `canopy`, half fr3’s own policy, i.e. a KL anchor to the start.
- *Fit.* Cross-entropy to the target, plus a BCE that holds the value head at fr3’s own win probabilities (the trunk is shared). AdamW at `az-2p-2048`’s settings (lr 5e-5, minibatches of 512). Every target gets 128 updates, `az-2p-2048`’s dose at its e6 select (learning had run twice by then, 64 updates each): 0.71 passes over the training positions. W&B logs held-out cross-entropy, KL from fr3, entropy, argmax agreement with fr3 and with the target, and value drift.
- *Finals.* `heldout-2p-fr`’s first 1,000 boards in both seat orders, 2,000 games a row, the same games in every row:
  - search harness (Marco's rule): each fit inside a 512-sim Gumbel search (`dets=1`, m = 16, fused kernels) against fr3’s fair raw bot (`BELIEF:worlds=16`), and fr3 itself in the same search as the baseline;
  - raw screen (secondary): each fit as `BELIEF:worlds=16` against fr3.

#lead[Decision rule.] Each target's harness row against the baseline, paired by McNemar's test over the shared games, Holm at 0.05 over the three targets.
- A clear loss: that target breaks the net. `engine` should lose; if it doesn't, the gate cannot see what broke `az-2p-2048`.
- A clear win: run the full AlphaZero loop with that target.
- Otherwise no effect of about 3.5 pp or more. About a quarter of paired games split (@exp:az-e8-search), so 2,000 games give a paired SE of about 1.1 pp.
The raw screen is read alongside the harness, never alone.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    header: ([Network inside the search], [Score vs fr3's fair raw bot], [Paired vs fr3 inside], [Raw: `BELIEF` vs fr3's]),
    [fr3 (the baseline) \ #sub[`3a55789c`]], [55.60 #ci[53.41][57.76]], [–], [–],
    [`engine`, the control \ #sub[`41389abf`, raw `b407cba0`]], [55.45 #ci[53.26][57.62]], [−0.15 #ci[−2.49][2.19] \ #sub[284/287, p = 0.93]], [49.45 #ci[47.26][51.64]],
    [`canopy` \ #sub[`5757cfc6`, raw `c6027c5d`]], [55.35 #ci[53.16][57.52]], [−0.25 #ci[−2.48][1.98] \ #sub[256/261, p = 0.86]], [50.80 #ci[48.61][52.99]],
    [`canopy-anchor` \ #sub[`ec88ecb4`, raw `08b96b96`]], [54.95 #ci[52.76][57.12]], [−0.65 #ci[−2.78][1.48] \ #sub[229/242, p = 0.58]], [50.90 #ci[48.71][53.09]],
  ),
  caption: [`heldout-2p-fr`'s first 1,000 boards in both seat orders, 2,000 games a row, the same games in every row. The search is 512-sim Gumbel (`dets=1`, m = 16, fused kernels). "Paired" counts games won only by the fit / only by the baseline (McNemar).],
)
- *Rule.* No target is a clear loss or a clear win (smallest p 0.58; Holm needs 0.017). Between 24% and 29% of the paired games split, so the paired 95% intervals are about ±2.3 pp.
- *Raw screen.* No fit is clearly off fr3 as a fair raw bot. `engine`'s 49.45 fits either no change or `az-2p-2048`'s 48.0 at e6. Paired with each other, `engine` trails `canopy` by 1.35 pp (p = 0.27) and `canopy-anchor` by 1.45 pp (p = 0.24).

#lead[What the fits did.] On the 512 held-out games, after the 128 updates:
- *Targets.* `engine`'s target overrules fr3's argmax at 13.8% of positions, `canopy`'s at 6.2% and `canopy-anchor`'s at 3.5%. Each puts ≥ 0.99 on one move at 92%, 90% and 85% of positions.
- *Movement.* KL from fr3 is 0.24, 0.030 and 0.015. Argmax agreement with fr3 is 94.8, 98.2 and 98.8%. `engine` moved most on dev-card decisions (KL 0.48, mostly knight or roll), least on the robber (0.03).
- *Not toward the target's moves.* Argmax agreement with the target fell slightly (86.4 → 85.1%, 93.9 → 93.6%, 96.6 → 96.5%), though the cross-entropy to it fell (3.20 → 2.48 for `engine`). The changed argmaxes mostly did not land on the moves the search preferred.
- *Value held.* Mean value drift was 0.004 or less.

#lead[Reading.]
- At `az-2p-2048`'s e6 dose, `engine`'s target changed about 1 in 20 of fr3's argmaxes, and the 512-sim search absorbed it: no change of 2.5 pp or more either way. The search harness did not see `az-2p-2048`'s damage either (e8 −0.6 pp, @exp:az-e8-search), so the gate's blind spot is the harness's, not the fits'.
- With no clear win, the rule gives no full AlphaZero loop with any of these targets.
- Cost: 5.3 h. Collection 1.5 h (204M leaf evaluations at 38k a second, 1.08 GB dataset, sha256 `95d4c7c4`); fits under a minute; each harness row with its raw row about 55 min.

#lead[Cost, as planned.] About 7–8 h of GPU with the card alone: 1.8–2.5 h of searching (210M leaf evaluations; `az-2p-2048` ran 23.6k a second), a minute per fit, about 1.15 h per harness row (from `search-depth-2p`’s cells; 4.6 h for four) and about 5 minutes per raw row. A CPU smoke of the whole pipeline passed; a 10-minute GPU smoke comes first.

#repro(```sh
catanrl run configs/experiments/az-gate-2p.yaml   # positions (then the baseline row), engine, canopy, canopy-anchor
uv run python tools/search_probe.py --pair <a fit's harness record> <the baseline record>
```)

== KataGo's way of spending compute

=== Self-play from fr3 at KataGo's visit counts: `az-2p-kata` #chip("done") <exp:az-kata>
#meta(
  dates: "2026-11-11 05:59 – 2026-11-12 15:27",
  owner: "marco",
  config: "configs/experiments/az-2p-kata.yaml",
  wandb: (("az-2p-kata-main-2e3b1866", "az-2p-kata/main"),),
  records: ("e917f9bb", "809619df", "aa1df609", "efdaab77", "8e47f012", "b7f9c1f4"),
)
#tldr[Marco's idea (2026-11-10): the earlier runs trained on 605 to 5,120 games where KataGo used 4.2M, so spend the compute KataGo's way, from fr3. Self-play now plays about 3,300 games an hour (`az-2p-2048`: 190). In a first smoke, five learning steps on the root's visit counts took the net inside the search from 52.7% to 27.0% against fr3's raw policy: fr3's policy is saturated, and the visits forced by Dirichlet noise sit 10 nats of KL from it. With those visits pruned from the target (Marco's choice) the same smoke holds, 52.7% to 55.1% on the same 256 games, at the same learning rate. The long run (300 iterations, 55,143 games) is the first AlphaZero run here that improves the search: on 2,000 held-out games against fr3's raw policy, the 1,024-sim search scores 57.7% with the trained net and 53.85% with fr3's (paired +3.85 pp, p = 0.003). On the select games the gain grew with training: +4.4, +4.8 and +6.2 pp at e100, e200 and e300. The value loss never moved, so the gain is the policy's. Not measured: deeper search, and the two searches head to head.]

*Question.* Does AlphaZero from fr3 improve it once the data scale is closer to KataGo's?

#lead[Setup.]
- *Start.* fr3 (`2p-graph-s-3`, `ppo-2p-fr3/full@e4350`), actor and critic, on `base-2p-fr` and its own 4,096 training boards.
- *Self-play.* 512 games in flight, PUCT roots (`c_puct` 0.75) on the true game (`dets: 0`). A quarter of the moves get 600 visits with Dirichlet noise at the root (α 0.3, weight 0.25) and are the only ones recorded. The rest get 100 visits without noise and are only played. Moves are sampled at temperature 1 falling to 0 over the first 30.
- *Target.* `policy_target: pruned`: the root's visit distribution without the visits the noise forced, and the game's outcome for the value head. The most visited move $c$ keeps its visits. With $S$ its PUCT score, another move $a$ keeps all its visits if $Q_a >= S$, and otherwise at most $ceil(c_"puct" P_a sqrt(T + 1) \/ (S - Q_a) - 1)$, where $P$ is the net's prior without noise and $T$ the root's visits; a move cut to one visit keeps none. The first smoke used the plain visit distribution (`engine`).
- *Learning.* An iteration is 64 steps (32,768 moves, about 8,200 recorded) and 64 updates of 512 rows: about 4 uses per position, KataGo's ratio. AdamW, learning rate 5e-5. Replay of 500,000 positions (1M did not fit this box's memory); learning starts at 50,000.
- *Trainer changes* (merged): a search sends up to 24 leaves per network round (4 in a fast search), so a step costs about 50 rounds, not 601; the forwards run on the fused kernels, rebuilt after every learning step and checked against the torch module.
- *Evals* (Marco's rule). Each checkpoint inside the deployed search (1,024 sims, `dets=1`, tree reuse, transpositions) against fr3's fair raw bot (`BELIEF:worlds=16`), 2,000 games a row (±2.2 pp). The start in the same search is the baseline.
- *Smokes* (2026-11-11, 00:18 – 01:24 and 04:41 – 05:45). The same config for 8 iterations with learning from 3,000 positions (iteration 3 on, five learning steps), a 256-game row before and after, W&B off, in a scratch directory: first with the visit-count target, then with the pruned one as the gate for the long run. Metrics and records are in `report/data/az-kata-smoke/` and `report/data/az-kata-gate/`.
- *Long run.* 300 iterations (16.7 h of self-play), a checkpoint every 25, a select at e0, e100, e200 and e300 (about 2 h each), then three final rows on `heldout-2p-fr`'s first 1,000 boards: best, last and the start, each in the search. About 31 h in all.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    header: ([Network inside the 1,024-sim search], [Score vs fr3's fair raw bot], [Wins of 256]),
    [fr3, the start (e0)], [52.7 #ci[46.6][58.8]], [135],
    [e8, visit-count target], [27.0 #ci[21.9][32.7]], [69],
    [e8, pruned target (the gate)], [55.1 #ci[49.0][61.1]], [141],
  ),
  caption: [The smokes' select rows: 128 boards in both seat orders, the same games in every row. Both smokes' e0 rows are the same 135 wins.],
)
- *Gate.* With the pruned target e8 scores 55.1% where the start scores 52.7% (6 more wins of 256, inside the noise), so the net holds and the long run follows. The pruning removes 15–18% of the root's visits. The pruned targets sit 0.2–0.7 nats of KL from the net (the raw visit counts 10–13) with 0.04–0.06 nats of entropy, and their top move differs from the net's at 0.5–1.5% of positions. After five learning steps the net agrees with fr3's top move at 97.3% of 4,096 probe positions (KL from fr3 0.04) and its entropy is still 0.04–0.05 nats.
- *Speed* (the same in both smokes). 199–203 s an iteration, 3 s of it learning and the kernel rebuild; 36,000–37,500 leaves a second (`az-2p-2048`: 23,600); about 180 games and 8,100 recorded positions an iteration once games finish, 3,300 games an hour.
- *Visit-count target: strength.* The start scores 52.7% and e8 27.0%: 26 points lost in 320 updates.
- *Visit-count target: what moved.* Against fr3's own policy, the visit targets of the first iteration have 0.59 nats of entropy where fr3 has 0.05, a KL of 10.0 nats, and a different top move at 0.9% of positions. After five learning steps the net's entropy is 0.5–0.6 nats, it agrees with fr3's top move at 65.8% of 4,096 probe positions (KL from fr3 1.47), and the search overrules it at 9.1% of positions. The policy loss fell from 12.2 to 2.8.
- *Visit-count target: why.* fr3's policy is saturated. On the probe positions with a choice, the top move has probability above 0.999997 at three quarters of them, the second move's median log-probability is −31 and the median other legal move's is −86. The noise makes the search visit such moves, the visits enter the target, and fitting them means moving logits by tens of nats through the shared trunk.
- *Eval cost.* A 256-game row took 17.0 min at 256 games in flight, so about 2 h for 2,000 games (an estimate from the row's finish times). The long run's four selects and three finals are about 14 h.
- *Memory.* The host's free memory fell from 22.2 to 14.6 GB in self-play and to 9.7 GB during the second row; the GPU held at most 2.9 of 8 GB. A full 1M replay is another 5.8 GB, copied once more at each checkpoint, so the run uses a 500,000-position replay. The gate smoke left at least 8.3 GB free; the long run is launched under a guard that stops it below 3 GB.

#lead[Reading.]
- The data scale is fixed: 17× the games per hour of `az-2p-2048`. The plain visit-count target was the problem. KataGo's nets learn from visit counts from their first game and never saturate; a PPO policy at 0.05 nats is far from any visit distribution a noisy root produces.
- Learning rate (Marco asked whether it is too high, and whether this is a collapse). It was a collapse and not a fit of a better policy: the visit targets' top move differs from fr3's at 0.9% of positions, and the net's top move changed at 34%. The gate answers the rest: at the same learning rate (5e-5, gradient norm clipped at 1.0) and the same value loss (0.32–0.38), the net holds once the target is pruned. The policy target caused the drop, not the step size or the value head.
- The first smoke learned from a small replay (3.3 uses per position by e8 against about 4 planned), but the targets in fresh self-play were as far from the net as the replayed ones (KL 1.8 and 2.1 at e8), so it was not overfitting.
- The pruned targets differ from fr3 at about 1% of positions, so the policy will learn slowly; whether 54,000 games move the net inside the search is what the long run measures (±2.2 pp a row).

#lead[Result: the long run.] 300 iterations in 24.8 h with the selects (launched from commit `c021778`): 55,143 games, 2.45M recorded positions. The finals took 6.3 h more.
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Network inside the 1,024-sim search], [Score vs fr3's fair raw bot], [Wins of 2,000], [Paired vs the start], [p]),
    tgroup(5)[Finals: `heldout-2p-fr`, first 1,000 boards],
    [fr3, the start], [53.85 #ci[51.7][56.0]], [1,077], [], [],
    [e300 (best and last)], [57.70 #ci[55.5][59.8]], [1,154], [+3.85 pp (369 vs 292)], [0.003],
    tgroup(5)[Selects: `select-2p-fr-1000`],
    [fr3, the start (e0)], [51.75 #ci[49.6][53.9]], [1,035], [], [],
    [e100 (18,400 games)], [56.10 #ci[53.9][58.3]], [1,122], [+4.35 pp (329 vs 242)], [0.0003],
    [e200 (36,700 games)], [56.55 #ci[54.4][58.7]], [1,131], [+4.80 pp (374 vs 278)], [0.0002],
    [e300 (55,100 games)], [57.90 #ci[55.7][60.0]], [1,158], [+6.15 pp (425 vs 302)], [0.000006],
  ),
  caption: [Within a suite every row plays the same 2,000 games (1,000 boards in both seat orders); McNemar's exact test on the games only one of the pair won. Between select checkpoints: e200 against e100 253 vs 244 (p = 0.72), e300 against e200 244 vs 217 (p = 0.23), e300 against e100 337 vs 301 (p = 0.17).],
)
- *Strength.* On held-out boards the trained net lifts the search by 3.85 pp over the start's net (p = 0.003). On the select boards the gain is +4.35, +4.80 and +6.15 pp at e100, e200 and e300; e300 was picked as best on those games, so its select number is the optimistic one. The steps between checkpoints are not significant on their own, so whether the net was still improving at e300 is open.
- *Net.* At iteration 300 it agrees with fr3's top move at 90.2% of the probe positions (KL from fr3 0.29) and its entropy is 0.20 nats (start 0.05). The pruned targets' KL from the net fell from 0.42 to 0.08, and the policy loss from 0.47 to 0.24. The value loss stayed flat (0.38–0.39), on fresh games too (0.35–0.43): the gain is the policy's.
- *Speed.* Iterations took 194–203 s for most of the run, but about 280 s from iteration 44 to 100 and 227–238 s for a while after e200's select, with the same leaves, rounds and depth per step. The baseline final row took 248 min where the other 2,000-game rows took 131–137. Cause unknown: nothing else used the GPU or a CPU core, and the host was not swapping.
- *Not measured.* The trained net's search against fr3's search head to head, and any depth past 1,024 sims (the OpenHex bot searches 2#super[17]).
- Marco chose the pruned target over two other options Claude put forward (training from scratch, parking). Set by Claude from the smokes' costs: `dets: 0` in self-play, 300 iterations, a select every 100.

#repro(```sh
catanrl run az-2p-kata
# the smokes (W&B off, a scratch runs dir); the first one adds --set train.policy_target=engine
CATANRL_RUNS_DIR=<scratch> WANDB_MODE=disabled catanrl run az-2p-kata \
  --set train.iterations=8 --set train.min_replay=3000 --set train.checkpoint_interval=8 --set eval.select.every=8 \
  --set 'eval.select.suite={name: smoke-select-128, purpose: select, seed: 123, boards: 128, scenarios: [{game: base-2p-fr}], turn_limit: 500}'
```)

=== 600 more iterations from e300: `az-2p-kata/main+600` #chip("running") <exp:az-kata-more>
#meta(
  dates: "2026-11-13 01:18 –",
  owner: "marco",
  config: "runs/az-2p-kata/main+600/config.json, then runs/az-2p-kata/main+600-fast/config.json (forks of configs/experiments/az-2p-kata.yaml)",
  wandb: (("az-2p-kata-main-600-6ffee9f3", "az-2p-kata/main+600"), ("az-2p-kata-main-600-fast-a21eb778", "az-2p-kata/main+600-fast")),
  records: (),
)
#tldr[Marco (2026-11-13): keep the KataGo-style training going for about two days and see how it goes. This run continues @exp:az-kata from e300 with nothing changed, to iteration 900 (three times the games). At e410 (2026-11-13 08:23) Marco had it stopped and continued as `main+600-fast` on the faster self-play of @exp:az-speed: the same visit counts, targets and learning rate, 82 s an iteration instead of 197 s. It should end on the morning of 2026-11-14.]

*Question.* Does the search keep improving with more of the same self-play, or was e300 the plateau?

#lead[Setup.]
- *Run.* A fork of `az-2p-kata/main` at e300: the weights, the optimizer state and the replay buffer carry over, and the recipe is the same (pruned targets, 600 and 100 visits, learning rate 5e-5). 600 more iterations, about 110,000 more games.
- *Eval.* The first run's, so its rows are the baseline: the checkpoint in the 1,024-sim search against fr3's fair raw bot on the same 2,000 select games at e450, e600, e750 and e900 (e300 scored 57.90%, the start 51.75%), paired by game. Finals for the best and the last checkpoint on the same 2,000 held-out games (e300 57.70%, the start 53.85%). A paired comparison on 2,000 games detects about 3.4 pp at 80% power.
- *Cost.* 600 iterations at about 200 s (33 h), four selects (9 h), finals 2–4 h: about 46 h. The time limit is a 60 h guard.
- *Switch at e410* (Marco: "kill the current run and continue from latest checkpoint possible"). `main+600` was stopped at iteration 410 (75,282 games) and forked as `main+600-fast` with `train.reuse: true`, `train.kernels: fused16` and two search groups; weights, optimizer and replay carry over again. Iterations 411–413 took 82–86 s at 69,000 leaves a second, with 46% of searches starting from a kept tree; policy loss 0.26–0.27 and value loss 0.38–0.39, as in the last iterations before the switch. The 490 iterations left take about 11 h, the selects 9 h and the finals 4–5 h.
- *What the switch changes for the question.* From e411 the games come from searches on reused trees and fp16 forwards, so a change in the trend after e410 could be either more self-play or the new self-play. The evals are the same search on the same games (unchanged bit for bit), and e450's select is the first to include the new games (40 of its 450 iterations).
- *Finals.* The fork's finals are the best and the last checkpoint only: the start's row is not played again (the merge changed the eval code id), and both pair by game with the stored row (fr3 in the search, 53.85%).
- *Fix on the way.* `catanrl fork` failed on an AlphaZero run from a network: it resolved the stored config again, where each scenario holds the game's definition instead of its name. Fixed in `50236e3`, with a test.
- Set by Claude: 900 iterations and a select every 150, to fit Marco's two days.
- *Where the time goes* (read off the live run at iteration 313, nothing profiled). Self-play is 194 s of each 197 s iteration. A step is 46.5 network rounds of about 2,430 leaves, 65 ms a round, 27 µs a leaf. The GPU is busy 64% of the time (about 40 ms a round, the 16 µs a leaf of @exp:search-depth-2p); in the other 25 ms the CPU walks the trees and moves the rows, and each side waits for the other. The process uses 3.0 of 16 hardware threads: the main thread a whole one (it spins while the GPU works), the 22 workers 6–12% each. Forwards are already 512 rows, about five to a round.

#repro(```sh
catanrl fork az-2p-kata/main --as main+600 --set train.iterations=900 --set eval.select.every=150 \
  --set runtime.time_limit_hours=60
catanrl run az-2p-kata/main+600
# at e410: stop it (SIGINT saves a checkpoint at the current iteration), then
catanrl fork az-2p-kata/main+600 --as main+600-fast --set train.reuse=true --set train.kernels=fused16 \
  --set 'eval.final=[<the best and latest rows of the parent, with the fork's name>]'
catanrl run az-2p-kata/main+600-fast
```)

=== Faster self-play: tree reuse, half-precision kernels, pipelined rounds #chip("done") <exp:az-speed>
#meta(
  dates: "2026-11-13",
  owner: "marco",
  config: "train.reuse, train.kernels: fused16, runtime.search_groups (src/catanrl/train/alphazero.py, src/catanrl/search/pipeline.py)",
  wandb: (),
  records: (),
)
#tldr[Three changes to AlphaZero self-play, proposed by Claude from the timing in @exp:az-kata-more and approved by Marco: keep a game's search tree between its decisions, run the network's large matrix products in half precision, and let two groups of games take turns on the GPU so it works while the CPU walks the other group's trees. Together an iteration of the KataGo-style run takes 82 s instead of 197 s (2.4×). Old configs, their hashes and every existing code path are bit for bit unchanged. What the changes do to learning is not measured yet: the run's next selects are the first evidence.]

*Question.* The GPU was busy 64% of each round and every decision searched from an empty tree. How much faster can the same self-play run on this box?

#lead[What changed.]
- *Tree reuse* (`train.reuse: true`; off by default). The tree of a game's last search is kept, and the next search starts from the child the played move leads to when the game is where that child says it is. The visit count becomes a cap: a search that carries _n_ visits adds 600 − _n_ (100 − _n_ in a fast search). Full searches still get fresh Dirichlet noise at the root, and the pruned target still reads the root's priors without noise. Only with `root: puct`, `dets: 0` and the `engine` or `pruned` targets. Trees are dropped at a game's end and at the end of each iteration's steps. The engine gained per-search options, `reroot` and `root_priors` (cppanatron `b926970`).
- *Half-precision kernels* (`train.kernels: fused16`). The fused forward with its large matrix products in fp16; a forward whose halves overflow is flagged and evaluated again in float32 (counted in `exact_forwards`). Evals keep `kernels=fused`.
- *Pipelined rounds with pinned buffers* (`runtime.search_groups`, default 2, not hashed). Game _e_ searches in group _e_ mod 2. While the GPU evaluates one group's leaves the CPU collects the other's, straight into page-locked memory, with non-blocking copies both ways.

#lead[Measured] (each piece alone, on e300's net, beside the live run, so indicative).
#figure(
  btable(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    header: ([Piece], [Measure], [Gain]),
    [Tree reuse], [leaves per decision 224.9 → 171.3 (32 games, 16,000 decisions an arm)], [1.31×],
    [fp16 kernels], [a 512-row forward 25.7 → 17.2 µs a row; 1.18–1.24× on a whole step, where the CPU is the limit], [1.5×],
    [Two groups], [a network round 86–88 → 55–57 ms], [1.5–1.6×],
    [All three], [an iteration of the run, GPU to itself: 197 → 82 s], [2.4×],
  ),
  caption: [The last row is the live run before and after the switch at e410 (@exp:az-kata-more): 69,000 leaves a second against 37,300, and fewer leaves per decision.],
)
- *Reuse covers under half the searches.* 43–45% of searches start from a kept tree: the environment plays the dice roll itself, so a tree only survives between decisions of the same turn and in initial placement. A kept tree brings 16% of a full search's cap and 40% of a fast one's; 15–16% of fast searches need no new visits at all. Re-rooting through the roll (a chance node) is not built.
- *fp16 agrees with float32.* On the same rows against `fused`: mean KL 1.6e-6 and 3.8e-6 on two position sets, the same top move at 99.98% and 99.97%, values within 1.4e-3.
- *Three groups, a thread per group and larger forwards* did not help.
- *Memory.* Kept trees are about 0.7 MB a game. The 512-game pre-flight peaked at 3.7 GB of RAM without the replay and 1.6 GB of GPU memory.

#lead[Checks.]
- With the new options off, nothing moves: 38,340 engine arrays, 336 arrays from the eval harness's search, self-play and short training runs, and 20 GPU evaluator outputs are bit for bit those of the build before the merge. `az-2p-kata`'s config hash is unchanged (`2e3b1866f7a37e4a`, asserted in a test).
- Reuse in one, two and three groups plays and records the same games (strict mode, CPU). On CUDA in fast mode two groups are deterministic but not bit-equal to one, because the forwards' sizes differ.
- Pre-flight before the switch: a fork of a copy of the live run at e325 with all three on ran two iterations at 122–129 s beside the live run, policy loss 0.257 and value loss 0.384 (the live run then: 0.26–0.27 and 0.38–0.39).
- Tests on the merged build: the CPU suite (999 tests) passes except one bridge test that failed once under load and passes alone; the 47 GPU tests pass; of the 81 train-marked tests five `test_smoke` cases fail, as they do on main (their scratch runs dir lacks `ppo-2p-fr3/full`). The first full run also caught a pipeline test that the merge with the fp16 kernels broke (fixed in `f0e65a7`).
- *Eval records.* The merge changes the eval code id, so cached rows are not found again: new rows are played again and old ones paired by hand. The eval search itself is unchanged bit for bit.

#lead[Caveats.]
- No evidence yet that training on reused trees and fp16 forwards learns as well. With reuse a full search's 600 visits include carried ones that were chosen without this root's noise, and the target's visit counts start from them.
- PyTorch 2.14.1 frees cuBLAS workspaces that a captured CUDA graph still uses; `fused16` clears them around each capture. `fused` is left as it was, and its evaluator's workspace grows by about 1 GiB over a run.

#repro(```sh
catanrl run az-2p-kata --set train.reuse=true --set train.kernels=fused16   # two groups is the default
catanrl run az-2p-kata --set runtime.search_groups=1                        # the old round loop
pytest tests/test_search_pipeline.py tests/test_alphazero.py tests/test_graph_kernels.py -q
```)
