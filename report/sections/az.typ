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

=== 600 more iterations from e300: `az-2p-kata/main+600` #chip("done") <exp:az-kata-more>
#meta(
  dates: "2026-11-13 01:18 – 2026-11-14 05:39",
  owner: "marco",
  config: "runs/az-2p-kata/main+600/config.json, then main+600-fast and main+600-fast-noselect (forks of configs/experiments/az-2p-kata.yaml)",
  wandb: (("az-2p-kata-main-600-6ffee9f3", "az-2p-kata/main+600"), ("az-2p-kata-main-600-fast-a21eb778", "az-2p-kata/main+600-fast"), ("az-2p-kata-main-600-fast-noselect-69e27214", "az-2p-kata/main+600-fast-noselect")),
  records: ("1ab3df39", "c7276581", "a28af73d", "4411825a"),
)
#tldr[Marco (2026-11-13): keep the KataGo-style training going for about two days and see how it goes. This run continues @exp:az-kata from e300 with nothing changed, to iteration 900 (three times the games). At e410 (2026-11-13 08:23) Marco had it stopped and continued as `main+600-fast` on the faster self-play of @exp:az-speed: the same visit counts, targets and learning rate, 82 s an iteration instead of 197 s. Selects: e450 scores 58.55%, level with e300's 57.90% on the same games (p = 0.59); e600 60.85%, +2.95 pp over e300 (p = 0.015); e750 59.60%, +1.70 pp over e300 (p = 0.18) and level with e600 (p = 0.26). Every checkpoint is well past the start (e750: +7.85 pp, p = 2e-8). At e750 Marco had the selects removed (they were half of the wall-clock) and the last 150 iterations ran as `main+600-fast-noselect`. Final: on the 2,000 held-out games e900 (165,348 games of self-play) scores 60.30%, against 57.70% for e300 (paired +2.60 pp, p = 0.041) and 53.85% for the start (+6.45 pp, p = 2e-6). More of the same self-play kept improving the search, slowly: about 2.6 pp for three times the games.]

*Question.* Does the search keep improving with more of the same self-play, or was e300 the plateau?

#lead[Setup.]
- *Run.* A fork of `az-2p-kata/main` at e300: the weights, the optimizer state and the replay buffer carry over, and the recipe is the same (pruned targets, 600 and 100 visits, learning rate 5e-5). 600 more iterations, about 110,000 more games.
- *Eval.* The first run's, so its rows are the baseline: the checkpoint in the 1,024-sim search against fr3's fair raw bot on the same 2,000 select games at e450, e600, e750 and e900 (e300 scored 57.90%, the start 51.75%), paired by game. Finals for the best and the last checkpoint on the same 2,000 held-out games (e300 57.70%, the start 53.85%). A paired comparison on 2,000 games detects about 3.4 pp at 80% power.
- *Cost.* 600 iterations at about 200 s (33 h), four selects (9 h), finals 2–4 h: about 46 h. The time limit is a 60 h guard.
- *Switch at e410* (Marco: "kill the current run and continue from latest checkpoint possible"). `main+600` was stopped at iteration 410 (75,282 games) and forked as `main+600-fast` with `train.reuse: true`, `train.kernels: fused16` and two search groups; weights, optimizer and replay carry over again. Iterations 411–413 took 82–86 s at 69,000 leaves a second, with 46% of searches starting from a kept tree; policy loss 0.26–0.27 and value loss 0.38–0.39, as in the last iterations before the switch. The 490 iterations left take about 11 h, the selects 9 h and the finals 4–5 h.
- *What the switch changes for the question.* From e411 the games come from searches on reused trees and fp16 forwards, so a change in the trend after e410 could be either more self-play or the new self-play. The evals are the same search on the same games (unchanged bit for bit), and e450's select is the first to include the new games (40 of its 450 iterations).
- *Finals.* The fork's finals are the best and the last checkpoint only: the start's row is not played again (the merge changed the eval code id), and both pair by game with the stored row (fr3 in the search, 53.85%).
- *No selects after e750* (Marco, 2026-11-13: "can we remove the evals then or at least do them more rarely? i think it's safe to say the method works."). From e410 to e750 the three selects took 7.5 h and the self-play 7.8 h. `main+600-fast` was stopped when e750's select ended (2026-11-13 23:58) and forked as `main+600-fast-noselect` with `eval.select` removed: 150 iterations (3.4 h), then one final row, e900 in the search on the 2,000 held-out games (2.5 h), paired with the stored rows of e300 and the start. No `best` is picked in this fork. Set by Claude: keeping that one row, and letting e750's select (two-thirds played when the change was made) finish.
- *Fix on the way.* `catanrl fork` failed on an AlphaZero run from a network: it resolved the stored config again, where each scenario holds the game's definition instead of its name. Fixed in `50236e3`, with a test.
- Set by Claude: 900 iterations and a select every 150, to fit Marco's two days.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Network inside the 1,024-sim search], [Score vs fr3's fair raw bot], [Wins of 2,000], [Paired vs e300], [p]),
    tgroup(5)[Final: `heldout-2p-fr`, first 1,000 boards],
    [fr3, the start], [53.85 #ci[51.7][56.0]], [1,077], [], [],
    [e300], [57.70 #ci[55.5][59.8]], [1,154], [], [],
    [e900 (165,300 games)], [60.30 #ci[58.1][62.4]], [1,206], [+2.60 pp (338 vs 286)], [0.041],
    tgroup(5)[Selects: `select-2p-fr-1000`],
    [fr3, the start (e0)], [51.75 #ci[49.6][53.9]], [1,035], [], [],
    [e300 (55,100 games)], [57.90 #ci[55.7][60.0]], [1,158], [], [],
    [e450 (82,500 games)], [58.55 #ci[56.4][60.7]], [1,171], [+0.65 pp (254 vs 241)], [0.59],
    [e600 (110,200 games)], [60.85 #ci[58.7][63.0]], [1,217], [+2.95 pp (316 vs 257)], [0.015],
    [e750 (137,900 games)], [59.60 #ci[57.4][61.7]], [1,192], [+1.70 pp (317 vs 283)], [0.18],
  ),
  caption: [Within a suite every row plays the same 2,000 games (1,000 boards in both seat orders); McNemar's exact test on the games only one of the pair won. Held-out, e900 against the start: +6.45 pp (424 vs 295, p = 2e-6). Selects against the start: e450 +6.80 pp (446 vs 310, p = 9e-7), e600 +9.10 pp (474 vs 292, p = 5e-11), e750 +7.85 pp (468 vs 311, p = 2e-8). e600 against e450: +2.30 pp (282 vs 236, p = 0.048). e750 against e600: −1.25 pp (213 vs 238, p = 0.26); against e450: +1.05 pp (308 vs 287, p = 0.41). The faster self-play played e450's last 40 iterations and everything since.],
)
- *e450.* 58.55%, level with e300 (+0.65 pp, p = 0.59): 150 more iterations (27,400 games) did not move the search by an amount this eval can see (it detects about 3.4 pp). Against e100 and e200 it is +2.45 pp (p = 0.06) and +2.00 pp (p = 0.11).
- *e600.* 60.85%, the first checkpoint past e300 on these games: +2.95 pp (p = 0.015, not corrected for the two looks so far). The search is still improving with more self-play, and 190 iterations on reused trees and fp16 forwards did not stop it. The net keeps moving away from fr3 slowly: it agrees with fr3's top move at 88.3% of the probe positions (90.2% at e300; KL 0.32 from 0.29).
- *e750.* 59.60%, between e450 and e600 and level with both; `best` stays e600. Against e300 it is +1.70 pp (p = 0.18), so e600's +2.95 pp is not repeated at the next checkpoint: the three looks since e300 are +0.65, +2.95 and +1.70 pp, all positive, one significant on its own. The search has gained since e300, by an amount at the edge of what a 2,000-game row detects (about 3.4 pp). The net agrees with fr3's top move at 87.7% of the probe positions (KL 0.33).
- *e900, held-out.* 60.30% (1,206 of 2,000; 64.3% moving first, 56.3% second): +2.60 pp over e300 on the same games (p = 0.041) and +6.45 pp over the start's net (p = 2e-6). This is the one comparison of the run on games no checkpoint was picked on, and the only look at them. It agrees with the selects: the search gained 2–3 pp from e300 to e900, against 3.85 pp from the start to e300 on a third of the games. The net agrees with fr3's top move at 86.8% of the probe positions (KL 0.35); policy loss 0.29, value loss 0.39, as throughout.
- *Time.* 28.4 h in all: `main+600` e300–e410 7.1 h of self-play (the tests of @exp:az-speed ran beside its last hours), `main+600-fast` e410–e750 7.8 h of self-play and 7.5 h of selects, `main+600-fast-noselect` 3.5 h of self-play and 2.2 h for the final row.
- *Not measured.* e900 against e300 or fr3 search against search, any depth past 1,024 sims, and whether the reused-tree, fp16 self-play learns as well per game as the old one (the switch and the extra games are confounded).
- *Where the time goes* (read off the live run at iteration 313, nothing profiled). Self-play is 194 s of each 197 s iteration. A step is 46.5 network rounds of about 2,430 leaves, 65 ms a round, 27 µs a leaf. The GPU is busy 64% of the time (about 40 ms a round, the 16 µs a leaf of @exp:search-depth-2p); in the other 25 ms the CPU walks the trees and moves the rows, and each side waits for the other. The process uses 3.0 of 16 hardware threads: the main thread a whole one (it spins while the GPU works), the 22 workers 6–12% each. Forwards are already 512 rows, about five to a round.

#repro(```sh
catanrl fork az-2p-kata/main --as main+600 --set train.iterations=900 --set eval.select.every=150 \
  --set runtime.time_limit_hours=60
catanrl run az-2p-kata/main+600
# at e410: stop it (SIGINT saves a checkpoint at the current iteration), then
catanrl fork az-2p-kata/main+600 --as main+600-fast --set train.reuse=true --set train.kernels=fused16 \
  --set 'eval.final=[<the best and latest rows of the parent, with the fork's name>]'
catanrl run az-2p-kata/main+600-fast
# at e750, after its select: stop it, then
catanrl fork az-2p-kata/main+600-fast --as main+600-fast-noselect --set eval.select=null \
  --set 'eval.final=[<the latest row only, with the fork's name>]'
catanrl run az-2p-kata/main+600-fast-noselect
```)

=== 900 more iterations from e900: `az-2p-kata/main+1500` #chip("done") <exp:az-kata-1800>
#meta(
  dates: "2026-11-14 11:55 – 2026-11-15 11:12",
  owner: "marco",
  config: "runs/az-2p-kata/main+1500/config.json (a fork of main+600-fast-noselect at e900)",
  wandb: (("az-2p-kata-main-1500-22e5f5af", "az-2p-kata/main+1500"),),
  records: ("ea257443",),
)
#tldr[Marco (2026-11-14): keep training. The run of @exp:az-kata-more continues from e900 to iteration 1,800 with nothing changed, 164,000 more games in 21 h, with no evals on the way and one held-out row at the end. e1800 (329,221 games of self-play) scores 62.95% on the 2,000 held-out games: +2.65 pp over e900 (p = 0.031), +5.25 pp over e300 (p = 7e-5) and +9.10 pp over the start's net (p = 4e-11). The search is still gaining, at about half the rate per game: 2.6 pp for the 110,000 games from e300 to e900, and 2.65 pp for the 164,000 from e900 to e1800.]

*Question.* Does the search keep gaining with twice the self-play again, and at what rate? From e300 to e900 it gained 2.60 pp on the held-out games.

#lead[Setup.]
- *Run.* A fork of `az-2p-kata/main+600-fast-noselect` at e900: weights, optimizer and replay carry over; the recipe and the faster self-play of @exp:az-speed are unchanged (82 s an iteration).
- *Eval.* One row: e1800 in the 1,024-sim search against fr3's fair raw bot on the 2,000 held-out games, paired by game with the stored rows of e900 (60.30%), e300 (57.70%) and the start (53.85%). A paired comparison on 2,000 games detects about 3.4 pp at 80% power, so a gain over e900 at the earlier rate (2.6 pp per 600 iterations) may not be significant on its own; the comparison with e300 will be.
- *Cost.* 900 iterations at 82 s (20.5 h) and the final row (2.2 h): about 23 h. No selects (Marco, @exp:az-kata-more), so no `best`: the row is for the last checkpoint. A stop at any iteration keeps a checkpoint there.
- Set by Claude: 900 more iterations and the single final row.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Network inside the 1,024-sim search], [Score vs fr3's fair raw bot], [Wins of 2,000], [e1800 paired against it], [p]),
    [fr3, the start], [53.85 #ci[51.7][56.0]], [1,077], [+9.10 pp (471 vs 289)], [4e-11],
    [e300 (55,100 games)], [57.70 #ci[55.5][59.8]], [1,154], [+5.25 pp (398 vs 293)], [0.00007],
    [e900 (165,300 games)], [60.30 #ci[58.1][62.4]], [1,206], [+2.65 pp (317 vs 264)], [0.031],
    [e1800 (329,200 games)], [62.95 #ci[60.8][65.0]], [1,259], [], [],
  ),
  caption: [`heldout-2p-fr`, first 1,000 boards in both seat orders, the same 2,000 games in every row; McNemar's exact test on the games only one of the pair won.],
)
- *Strength.* 62.95% (67.8% moving first, 58.1% second). Each doubling of the self-play has added to the search on games no checkpoint was picked on: +3.85 pp by e300, +2.60 pp more by e900, +2.65 pp more by e1800. The two later steps are each significant at 5% on their own (p = 0.041 and 0.031), neither by much; together, e1800 over e300 is +5.25 pp (p = 7e-5).
- *Rate.* Per game the gain is falling: 3.85 pp for the first 55,000 games, 2.60 pp for the next 110,000, 2.65 pp for the next 164,000. No plateau is visible at 1,800 iterations.
- *Net.* It agrees with fr3's top move at 84.3% of the probe positions (86.8% at e900; KL 0.42 from 0.35). Policy loss 0.29 and value loss 0.38, flat since e300; fresh targets are as close to the net as replayed ones (KL 0.036 and 0.037).
- *Time.* 23.3 h: 21.0 h of self-play (84 s an iteration with checkpoints) and 2.3 h for the row.
- *Not measured.* e1800 against e900 or fr3 search against search, any depth past 1,024 sims, and the raw policy without search.

#repro(```sh
catanrl fork az-2p-kata/main+600-fast-noselect --as main+1500 --set train.iterations=1800
catanrl run az-2p-kata/main+1500
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

=== Where the time goes: a faster card, more cores and the main thread #chip("done") <exp:az-hardware>
#meta(
  dates: "2026-11-15",
  owner: "marco",
  config: "tools/gpu_bench.py, tools/modal_bench.py (wt/modal, unmerged)",
  wandb: (),
  records: (),
)
#tldr[Marco: would a new GPU help, and would a GPU simulator pay on a big card? Measured on rented cards (Modal, \$1.53). AlphaZero self-play on `graph-s` is not limited by the GPU: on an L40S, which runs the network 4.3× faster than the RTX 2070 Super, an iteration takes 82.7 s against 83.3 s here, and 32 cores instead of 8 give 1.29×. A profile finds why: only a third of the time is work spread over threads, and the main thread alone spends 19% of the wall-clock freeing finished search trees. `graph-m` PPO is the opposite: 96–98% of an epoch is the network, and an L40S-speed card would give 2.7×.]

*Question.* What would a faster GPU or more cores buy for the two workloads, self-play on `graph-s` and PPO on `graph-m`?

#lead[Setup.]
- *Machines.* This box (RTX 2070 Super, Ryzen 3700X, 16 threads); Modal's A10G and L40S with 8 cores, and the L40S with 32. The same torch and CUDA; W&B off; nothing left deployed.
- *Self-play.* e1800's stored config for 5 iterations, learning from 4,096 positions so the learning step is timed; medians of the last two.
- *PPO.* `graph-m-2p-lr`'s config from scratch for 8 epochs in full and half precision; medians of the last five.
- *Prediction for a new card in this box.* Self-play: the recorded trace of GPU calls replayed with a faster device. PPO: the forwards and the update scaled by the card's measured speed, the rest unchanged.
- *Profile.* `perf` on the main thread of the live self-play process (the GPU check of @exp:az-bundle), 2,567 stack samples, and the CPU time of every thread.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    header: ([], [This box], [A10G], [L40S]),
    tgroup(4)[Network alone, rows a second],
    [`graph-s` forward, fused half precision], [120,756], [175,628 (1.45×)], [520,328 (4.31×)],
    [`graph-m` update, full precision], [13,038], [16,429 (1.26×)], [37,037 (2.84×)],
    [`graph-m` update, half precision], [19,703], [27,718 (1.41×)], [54,006 (2.74×)],
    tgroup(4)[AlphaZero self-play, `graph-s`: seconds an iteration],
    [8 cores (this box: 8 cores, 16 threads)], [83.3], [], [82.7],
    [32 cores], [], [], [64.2],
    [A card of that speed in this box (predicted)], [83.3], [69.6], [67.7],
    tgroup(4)[`graph-m` PPO: steps a second],
    [Full precision], [6,092], [], [18,470],
    [Half precision], [11,717], [], [28,356],
    [Full precision, that card in this box (predicted)], [6,092], [7,620], [16,700],
    [Half precision, that card in this box (predicted)], [11,717], [16,300], [30,600],
  ),
  caption: [The two workloads on this box and on rented cards. Ratios are against this box.],
) <tab:az-hardware>

#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([Main thread of self-play on this box], [Share of wall-clock], [Runs on]),
    [Growing the trees and applying the network's results], [34%], [all 12 threads of a group],
    [Freeing finished search trees (`SearchPool::remove`)], [19%], [the main thread],
    [Waiting for the GPU], [17%], [the main thread],
    [Python and the rest], [12%], [the main thread],
    [Off the CPU], [15%], [not broken down],
  ),
  caption: [Where a second of self-play goes. The main thread is on the CPU 85% of the time; each of about 22 worker threads 17%; the process uses 4.7 of 16 logical cores.],
) <tab:az-profile>

*Reading.*
- *Self-play is limited by one thread.* On the L40S the card is idle 80% of the time and the main thread never waits for it. Only about a third of the time is spread over threads, which is why four times the cores give 1.29×.
- *Freeing trees.* Each tree node holds a copy of the game, and the board keeps its road networks and buildable spots in `std::set<int>`: many small allocations a node. Allocating and freeing is a third of the main thread's time.
- *A new card.* For self-play on `graph-s`, about 1.2×. For `graph-m` PPO, 1.25× (A10G speed) to 2.7× (L40S speed); half precision on the present card already gives 1.9× (@exp:fp16).
- *A GPU simulator* is not supported by these numbers: the time goes to tree bookkeeping on one thread, not to stepping games.
- *Limits.* 5 iterations and 8 epochs a cell; this box was measured with other processes on the CPU (load average 3.0 before the self-play benchmark); Modal's cores are 0.83× this box's on one thread; `graph-m` in self-play (@exp:az-graph-m) was not measured.

*Next* (Claude's suggestions, not queued). Free trees on an idle worker thread, a small engine change that leaves results as they are (up to 1.2× on self-play, the profile's bound). Keep the board's sets as bit masks, a larger change that removes most allocations.

#repro(```sh
# from the wt/modal worktree
python tools/gpu_bench.py a --out OUT --runs RUNS     # then: b1 --iterations 5 | b2 --epochs 8 --precision float16
modal run tools/modal_bench.py --job l40s-b1 --out OUT   # cpu | a10g-a | l40s-a | l40s-b1-t8 | l40s-b1-t32 | l40s-b2
perf record -F 149 -t <pid of catanrl run> --call-graph dwarf,32768 -- sleep 20
```)

=== Several self-play processes on one rented card, and what \$100 buys #chip("done") <exp:az-modal-scale>
#meta(
  dates: "2026-11-17",
  owner: "marco",
  config: "tools/modal_bench.py, jobs l40s-b1-x4, -x6, -x8 (wt/modal 752d2e4, unmerged)",
  wandb: (),
  records: (),
)
#tldr[Marco (2026-11-16): how fast and how expensive would AlphaZero training be on Modal with an L40S and 64 vCPUs, what would \$100 buy, and would training from random weights make sense? Measured for \$2.70. One self-play process leaves the L40S idle 75 to 82% of the time (@exp:az-hardware), so several independent trainers share one card: 4, 6 and 8 of them make 164, 198 and 207 iterations an hour together, against this box's 42.7. The card is full from about six. At \$3.97 an hour that is \$19 to \$20 per 1,000 iterations: \$100 buys about 5,000 iterations (0.9M games, 41 to 43M recorded positions) in 25 h, and the whole fr3 line so far (2,700 iterations) would take 13 h and \$52. It needs a multi-process trainer, which is not built. From random weights \$100 is about one Canopy-sized run, which likely ends below fr3. Nothing is queued.]

*Question.* How many self-play processes does one L40S feed, and what does an iteration then cost?

#lead[Setup.] K full trainers at once on one L40S with 32 physical cores (64 vCPUs) and 64 GiB, each running e1800's stored config for 5 iterations (512 games × 64 steps, 600-visit full searches on a quarter of the moves, fused half-precision kernels) with 32 / K threads. They share nothing but the card, so this measures the hardware, not a trainer.

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right, right, right),
    header: ([Processes × threads], [s an iteration, each], [Iterations an hour, all], [× this box], [Leaves a second, all], [Card busy], [\$ per 1,000 iterations]),
    [1 × 8], [82.7], [43.5], [1.02], [69,700], [18%], [59],
    [1 × 32], [64.2], [56.1], [1.31], [87,900], [25%], [71],
    [4 × 12], [87.7], [164.2], [3.85], [259,900], [77%], [24.2],
    [6 × 8], [108.9], [198.4], [4.65], [314,300], [98%], [20.0],
    [8 × 6], [139.2], [206.9], [4.85], [328,200], [100%], [19.2],
  ),
  caption: [One L40S. The first two rows are from @exp:az-hardware (2026-11-15; the first on 8 cores at \$2.58 an hour); the others cost \$3.97 an hour (32 cores, 64 GiB). This box makes 42.7 iterations an hour. Peak GPU memory 8.0, 11.6 and 16.8 GiB; a batch's round trip takes 3.6, 7.0 and 13.6 ms, and the main threads wait for the card 2–3%, 6–7% and 19–21% of the time.],
) <tab:az-modal-scale>

*Reading.*
- *Why it stops at about six.* The processes are not batched together: each has its own CUDA context and the card runs one batch at a time. A real evaluator call, with its uploads and downloads, runs at about 302,000 rows a second, and one process asks for 70,000 to 88,000, so four or five fill the card.
- *Why one process does not use more threads.* Only about a third of its main thread's time is spread over threads (@exp:az-hardware): four times the cores give 1.29×.
- *A correction.* An estimate given to Marco before this was read, that one shared inference process could give up to 1.6× more, used the bare forward's 520,000 rows a second. The call the search makes is 302,000, so that gain is not supported. What would raise the ceiling is cheaper transfers (42% of a call) or a second card.
- *What \$100 buys.* 25.2 h at \$3.97 an hour: 5,000 to 5,200 iterations, 41 to 43M recorded positions, about 0.9M games, or five days of this box in a day. The fr3 line's 2,700 iterations: about 13 h and \$52.
- *From random weights* (Marco's thought, for a less sharp start). Canopy's run from scratch was about 47M recorded positions (@exp:az-canopy-diff), the size \$100 buys; its raw net scores 30.5% against ours and its search 33.0% (@exp:canopy-raw-vs-raw, @exp:canopy-search-gpu). fr3's PPO made 1.0B decisions in 32 h on this box for free, and PPO makes about 44× the decisions a second of searched self-play. Visit-count targets sharpen any net as it trains, and pruned targets already handle fr3's sharp start (@exp:az-kata). So \$100 from scratch likely ends below fr3.
- *Other savings, not measured.* Fewer cores and less memory per process; evals at home; freeing trees off the main thread matters less once the card is the limit. An H100 would need more than twice the L40S's throughput to pay.
- *Not built.* A trainer with several self-play processes feeding one learner and one replay, and an AlphaZero resume tested on Modal. The trainer is one process today.

*Next* (Claude's suggestion, not queued). If a softer start is the aim, start AlphaZero from an early PPO checkpoint instead of random weights: most of PPO's cheap games, a policy that is not yet saturated.

#repro(```sh
# from the wt/modal worktree
modal run tools/modal_bench.py --job l40s-b1-x6 --out OUT   # l40s-b1-x4 | l40s-b1-x8
```)

=== Auxiliary targets: each seat's final VP and the turns left #chip("done") <exp:az-aux>
#meta(
  dates: "2026-11-15",
  dates-label: "proposed",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[Claude's suggestion, approved by Marco on 2026-11-15. KataGo trains its trunk on more than win or loss: ownership and score heads give many labels per position and are dropped at play time (about 1.65× training speed in its ablation). Catan's analogs: each seat's final VP and the number of turns left. Trained in the bundled run (@exp:az-bundle): the head learned its targets, and the search gained no measurable amount.]

- *Why.* A game gives the value head one bit. Final VP and game length are known for every stored position once its game ends, at no extra self-play.
- *Heads.* Two small heads on the pooled state, trained with small weights beside the policy and value losses and never read at play time: the search, the kernels and the deployed bots use the same logits and values as without them.
- *Readouts.* Both losses on W&B, with the mean error in VP and in turns.

#lead[As built.]
- *Switch.* `model.args.outcome: true`, off by default; merged 2026-11-15.
- *Head.* One extra head beside the policy and value heads, 135,938 parameters on `graph-s`, run only by the trainer's learning step. Targets: each seat's final VP over the points to win and the turns left over 100, Huber loss.
- *Weights.* `outcome_vp_coef` and `outcome_turns_coef`, both 0.5 in the bundle: an estimate, not tuned.
- *Labels.* Stored with each replay position once its game ends (6 more bytes; 6,033 a position with the dice input); positions of games cut at the turn limit are masked.
- *Check on the GPU.* With the head in the network, self-play and the evals give the same logits and values as without it. In 6 learning steps the mean error fell from 6.9 to 2.0 VP and from 19 to 7.6 turns (2.1 and 8.0 on positions not trained on).
- *Logged.* `train/outcome_vp_loss`, `train/outcome_turns_loss`, `train/outcome_vp_mae` (in VP), `train/outcome_turns_mae` (in turns) and the same under `oos/`.

*Result (@exp:az-bundle).* The head learned its targets: 1.6 VP and 6.6 turns of error on positions not trained on, from 2.2 and 8.5 early in the run. The value head is about 1% more accurate than e1800's on shared positions (Brier 0.1148 against 0.1159, @exp:az-similarity), and the search gained +0.75 pp (interval −1.6 to +3.1), with the dice input in the same run. Its own effect is not separated.

=== 900 more iterations with the dice input and auxiliary targets: `az-2p-bundle` #chip("done") <exp:az-bundle>
#meta(
  dates: "2026-11-15 19:20 – 2026-11-16 19:24",
  owner: "marco",
  config: "configs/experiments/az-2p-bundle.yaml",
  wandb: (("az-2p-bundle-main-da6d1382", "az-2p-bundle/main"),),
  records: ("b5483151",),
)
#tldr[Marco (2026-11-15): bundle everything into the continued training first and ablate later; the goal is the best 1v1 bot. e1800 continues for 900 iterations with the dice input (@exp:dice-input) and the auxiliary targets (@exp:az-aux) on. After 160,906 more games the last checkpoint scores 63.70% on e1800's 2,000 held-out games: +0.75 pp over e1800 (interval −1.6 to +3.1, p = 0.56). No measurable gain; the 900 iterations before it gained 2.65 pp. The network does read the dice state, the new head learned its targets, the critic is about 1% more accurate (@exp:az-similarity) and the raw policy is 3.8 pp stronger (@exp:az-raw).]

*Question.* How strong is the search after 900 more iterations with both additions?

#lead[Setup.]
- *Run.* From `az-2p-kata/main+1500@e1800` (`2p-graph-s-6`) under its own recipe (@exp:az-kata-1800), with the new input weights at zero and the new heads beside the old ones, so iteration 0 plays exactly as e1800. The replay starts empty: e1800's stored positions have neither the dice state nor the new labels.
- *Eval.* No selects. One row at the end: the last checkpoint in the 1,024-sim search against `BELIEF(fr3, 16 worlds)` on e1800's 2,000 held-out games (62.95% there), paired by game. The paired difference has a standard error of about 1.2 pp, so about 3.4 pp is detectable at 80% power.
- *What the row can say.* The gain over e1800 mixes the input, the targets and 900 more iterations; the last 900 iterations alone gained 2.65 pp. The row says how good the new bot is, not what each part gave: that is for the ablations.
- *Start.* The optimizer and the replay start fresh; the first learning step comes when 50,000 positions are stored, after about 6 iterations.
- *Cost.* About 21 h of self-play at 84 s an iteration (measured in the GPU check: 68,300 leaves a second, 98.5% of e1800's pace) and 2.3 h for the row, about 10% of the wall-clock. Due about 19:00 on 2026-11-16.
- *Not in the bundle.* `graph-m` (@exp:az-graph-m: the fused kernels do not fit it yet) and the search changes of @exp:search-stratified and @exp:search-paired-root, parked after @exp:search-noise.

#lead[GPU check before the launch.]
- *Tool.* `tools/bundle_smoke.py`: 8 iterations of the real config, learning from 2,000 positions.
- *Iteration 0 is e1800.* The e0 checkpoint holds e1800's weights, zeros on the dice input and the new head; on the fused kernels (full precision, the eval's; half precision, self-play's) its logits and values equal e1800's bit for bit on 1,032 positions.
- *Kernels against plain PyTorch.* With trained dice weights the fused kernels are within 4e-6 on the values, the same as e1800 without the input (4.4e-6).
- *Learning.* All losses finite; the two new losses fall (@exp:az-aux); the dice weights leave zero (@exp:dice-input).
- *Two thresholds corrected.* The tool first printed FAIL: it asked plain PyTorch on the GPU for bit equality (off by 3e-6, the same best move) and the fused values for 1e-6 of plain PyTorch. e1800 itself misses both, so they now stand at rounding and at the kernel tests' 1e-5.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Network inside the 1,024-sim search], [Score vs fr3's fair raw bot], [Wins of 2,000], [Bundle paired against it], [p]),
    [fr3, the start], [53.85 #ci[51.7][56.0]], [1,077], [+9.85 pp (484 vs 287)], [1e-12],
    [e300 (55,100 games)], [57.70 #ci[55.5][59.8]], [1,154], [+6.00 pp (414 vs 294)], [0.000007],
    [e900 (165,300 games)], [60.30 #ci[58.1][62.4]], [1,206], [+3.40 pp (360 vs 292)], [0.009],
    [e1800 (329,200 games)], [62.95 #ci[60.8][65.0]], [1,259], [+0.75 pp (301 vs 286)], [0.56],
    mid,
    [Bundle, 900 more (490,100 games)], [63.70 #ci[61.6][65.8]], [1,274], [], [],
  ),
  caption: [`heldout-2p-fr`, first 1,000 boards in both seat orders, the same 2,000 games in every row; McNemar's exact test on the games only one of the pair won.],
) <tab:az-bundle>
- *Strength.* 63.70% (69.2% moving first, 58.2% second). Against e1800 on the same games: +0.75 pp, interval −1.6 to +3.1, 301 games won only by the bundle and 286 only by e1800.
- *Reading.* The two additions and 900 more iterations together did not move the search by an amount this row can see. The row cannot tell "nothing" from "the earlier rate": e900 to e1800 gained 2.65 pp, which is inside the interval. A gain above about 3 pp is ruled out.
- *Dice input.* The network uses it, a little. Its weights grew from zero to norm 2.25. On 512 probe positions, the network with the true dice state against the same network told the dice are fair: a different top move on 3.3% of positions, policies 0.017 nats apart on average (0.57 at most), values 0.018 apart.
- *Auxiliary targets.* On positions not trained on, the head's error fell from 2.2 to 1.6 VP and from 8.5 to 6.6 turns over the run. On each run's own fresh games the value head's Brier score is the same (0.129 against e1800's 0.128); on one shared set of positions it is slightly better, 0.1148 against 0.1159 (paired interval −0.0020 to −0.0003, @exp:az-similarity).
- *Net.* It picks e1800's top move at 94.1% of probe positions. Policy loss 0.276 (0.288 at e1800's end), value loss 0.384 (0.385).
- *Time.* 24.1 h: 21.2 h of self-play (84 s an iteration, 68,300 leaves a second, 7.36M recorded positions) and 2.4 h for the row, 10% of the wall-clock.
- *Without the search* (@exp:az-raw). The bundle's raw policy is better than e1800's: 53.4% against raw fr3 where e1800 scores 49.6%, +3.82 pp on the same 5,000 games (p = 1.5e-6). The gain shows without the search and barely inside it.
- *Caveats.* The replay and the optimizer started fresh, which the earlier 900 iterations did not have to pay for: the replay took about 60 iterations to fill. One row against one opponent at one search size; the parts are not separated.
- *Registered* as `2p-graph-s-7` (`az-2p-bundle/main@e900#ad8c2caf507b`). OpenHex keeps e1800 (`catanrl-1v1-search-v3.1`): the row gives no reason to swap.

#repro(```sh
python tools/bundle_smoke.py --runs <scratch dir>   # the GPU check, ~15 min
catanrl run configs/experiments/az-2p-bundle.yaml
```)

=== Does the better search net play better without the search? #chip("done") <exp:az-raw>
#meta(
  dates: "2026-11-16 19:27 – 20:18",
  owner: "marco",
  config: "configs/experiments/az-raw-2p.yaml",
  wandb: (("az-raw-2p-main-33c6f1fc", "az-raw-2p/main"),),
  records: ("8acb3c3d", "cc9d754a", "753ffd65", "fef53cfd"),
)
#tldr[Marco (2026-11-16): compare the raw policies of the checkpoints, to see whether a net that is better inside the search is also a better raw policy. Through e1800 it is not: without search e300, e900 and e1800 score 47.6%, 49.1% and 49.6% against raw fr3, while inside the search they gained 3.85 to 9.10 pp. The bundle's net is the exception: 53.4% raw, +3.8 pp over e1800 (p = 1.5e-6), though inside the search it gained only +0.75 pp.]

*Question.* From fr3 to e300, e900 and e1800 the search gained 3.85, 6.45 and 9.10 pp against the fair raw fr3 bot. Did the raw policy gain too?

#lead[Setup.]
- *Bots.* The fair raw bot (`BELIEF:worlds=16`) of e300, e900, e1800 and the bundled run's last checkpoint, each against the same of fr3. No search on either side.
- *Games.* All 2,500 boards of `heldout-2p-fr` in both seat orders, 5,000 games a row: ±1.4 pp at 50%, and between two checkpoints 2 to 3 pp is detectable at 80% power (2.8 pp unpaired; about 2 pp if pairing by game works as well as in the search rows, not checked for raw bots). The first 1,000 boards are the search rows' boards.
- *Reading.* Each checkpoint gets two numbers against the same opponent, with the search and without. Raw scores that rise with the search scores say self-play improves the policy itself; flat raw scores say the gain lives in what the search does with the net (its value, or its second and third choices).
- *Cost.* About 13 min a row (2.7 min per 1,000 games in @exp:recipe-transitivity), under 1 h for the four rows.
- *Not planned.* A round robin among the checkpoints (6 more pairs).
- *Config.* `configs/experiments/az-raw-2p.yaml`, launched once with the four rows; e900 is `az-2p-kata/main+600-fast-noselect@e900#713c7a0a818a`.
#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, 1fr, auto),
    align: (left, right, right, right, right),
    header: ([Network], [Raw], [Wins of 5,000], [Paired with the row above], [In the search]),
    [fr3, the start], [50 by symmetry], [], [], [53.85],
    [e300], [47.58 #ci[46.2][49.0]], [2,379], [], [57.70],
    [e900], [49.06 #ci[47.7][50.4]], [2,453], [+1.48 pp (866 vs 792), p = 0.07], [60.30],
    [e1800], [49.60 #ci[48.2][51.0]], [2,480], [+0.54 pp (800 vs 773), p = 0.51], [62.95],
    [Bundle], [53.42 #ci[52.0][54.8]], [2,671], [+3.82 pp (876 vs 685), p = 1.5e-6], [63.70],
  ),
  caption: [Score in % against fr3's fair raw bot. Raw: the network as a fair raw bot (`BELIEF`, 16 worlds, no search) on all 2,500 boards of `heldout-2p-fr` in both seat orders, the same 5,000 games in every row; McNemar's exact test. In the search: the 1,024-sim rows of @tab:az-bundle (2,000 games).],
) <tab:az-raw>
- *Through e1800: no.* The raw policy did not follow the search. e300 is below fr3 (47.58%, p = 0.0007 against 50%); e900 and e1800 are level with it. From e300 to e1800 the raw policy recovered 2.0 pp (p = 0.018) while the search gained 5.25 pp. By the reading set before the rows, the search's gain lives in what the search does with the net (its value, or its second and third choices), not in a better first choice.
- *The bundle: yes.* 53.42%, above fr3 (p = 1e-6) and +3.82 pp over e1800 (interval +2.3 to +5.4). Against e900 it is +4.36 pp and against e300 +5.84 pp.
- *Likely the dice input, not shown.* A raw bot has no chance nodes, so the dice state reaches its move only through the input; the search already rolls from the exact odds, and there the bundle gained +0.75 pp. The 900 iterations before the bundle, without the input, moved the raw policy by +0.54 pp. The outcome head and the fresh replay are in the same run, so the row does not separate them.
- *Seats.* Moving first and second: e300 50.8% and 44.4%, e900 52.6% and 45.5%, e1800 53.9% and 45.3%, the bundle 57.6% and 49.2%.
- *Cost.* 13 min a row, 51 min in all. No game hit the turn limit.
- *Not measured.* The checkpoints against each other, and the raw bot of the bundle told the dice are fair (which would test the dice reading).

#repro(```sh
catanrl run configs/experiments/az-raw-2p.yaml
```)

=== How the checkpoints differ: value and policy similarity over training #chip("done") <exp:az-similarity>
#meta(
  dates: "2026-11-16 20:19 – 20:22",
  owner: "marco",
  config: "tools/checkpoint_similarity.py; report/data/az-similarity.json",
  wandb: (),
  records: (),
)
#tldr[Marco (2026-11-16): a difference measure between the critics of the checkpoints, and a similarity matrix between checkpoints over time. 111 checkpoints from fr3 to the end of the bundled run read the same 20,000 positions. The critic changed once, in the first 25 iterations, and its accuracy has been flat from e150 to e1800; the bundle moved it a little and made it slightly more accurate (Brier 0.1148 against 0.1159, paired interval −0.0020 to −0.0003). The policy has drifted at a steady pace the whole way, with no jump where the recipe changed.]

*Question.* How far does each checkpoint's value and policy sit from every other's, and does the net drift steadily, settle, or jump where the recipe changed (e410's faster self-play, the bundle's new input and head)?

#lead[Setup.]
- *Checkpoints.* Every 25 iterations: 74 from fr3 to e1800 and 37 of the bundled run.
- *Positions.* About 20,000, a few from each of 2,000 games on boards outside the training pool, played by the raw policies of fr3, e900, e1800 and the bundle's last checkpoint in equal shares, so that no one checkpoint's style sets the sample. Each position is encoded with and without the dice state, for the nets that read it and those that do not.
- *Value matrix.* For each pair of checkpoints, the mean absolute difference between their win probabilities for the seat to move.
- *Policy matrix.* For each pair, the share of positions where both prefer the same move, and the mean divergence between the two policies.
- *Against the outcome.* Each checkpoint's value scored against who won the game (Brier score and log loss), to tell a critic that moved from one that got better.
- *Cost.* Network forwards only: minutes on the GPU.
- *As run.* 2,000 games (none cut at the turn limit), 20,000 positions, 111 checkpoints: 45 s to play and under 2 min to score on the GPU. Tool: `tools/checkpoint_similarity.py` (`play`, `score`, `report`; 12 tests).

#lead[Result.]
#figure(
  grid(columns: (1fr, 1fr), gutter: 6pt,
    image("../figures/az-similarity-value.png", width: 100%),
    image("../figures/az-similarity-policy.png", width: 100%),
  ),
  caption: [Every checkpoint against every other on the same 20,000 positions, by iteration (the bundle is 1,800 to 2,700). Left: mean absolute difference of the win probabilities; darker is further apart. Right: share of positions with the same preferred move; darker is closer. Lines mark where one run hands over to the next.],
) <fig:az-similarity>
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Checkpoint], [Brier vs who won], [Value distance to fr3], [Same top move as fr3], [Same top move as e1800]),
    [fr3 (e0)], [0.1244 #ci[0.1201][0.1288]], [0], [100%], [84.3%],
    [e150], [0.1165 #ci[0.1111][0.1218]], [0.064], [93.3%], [],
    [e300], [0.1160 #ci[0.1106][0.1214]], [0.066], [90.7%], [86.6%],
    [e900], [0.1156 #ci[0.1103][0.1210]], [0.066], [87.1%], [91.8%],
    [e1800], [0.1159 #ci[0.1103][0.1216]], [0.077], [84.3%], [100%],
    [Bundle, 900 more], [0.1148 #ci[0.1093][0.1203]], [0.072], [83.0%], [93.1%],
  ),
  caption: [Brier score of the win probability against the game's winner, with intervals from a bootstrap over games (the rows share positions, so differences are far tighter than the intervals: see the text). Value distance: mean absolute difference of the win probabilities.],
) <tab:az-similarity>
- *The critic changed once.* 25 iterations of self-play moved fr3's win probabilities by 0.069 on average, the largest step between neighbouring checkpoints anywhere (the median step is 0.011). Its Brier score fell by 0.0080 by e150 (interval −0.0093 to −0.0067). fr3's critic was the less decided one: its win probabilities sit 0.26 from one half on average, the later ones 0.32.
- *Then it wandered.* From e300 to e1800 any two critics are about 0.02 apart whether 300 or 900 iterations lie between them, not far above the 0.011 between neighbours, and the Brier score did not move (e1800 against e300: −0.0001, interval −0.0008 to +0.0006). The 5.25 pp the search gained over those iterations did not come from a more accurate critic on these positions.
- *The bundle moved it.* The bundle's critics form a block of their own: 0.027 from the earlier ones, where 900 iterations used to give 0.021. They are slightly more accurate: e900 of the bundle against e1800, −0.0011 in Brier (interval −0.0020 to −0.0003, paired over games), about 1%. Which of the dice input and the outcome head did it is not separated.
- *The policy drifts steadily.* Neighbouring checkpoints prefer the same move on 98.6% of positions, 300 iterations apart on 95.9%, 900 apart on 91.1%, and the agreement with fr3 keeps falling: 90.7% at e300, 87.1% at e900, 84.3% at e1800, 83.0% at the bundle's end. No jump at e410 (the faster self-play) or at the bundle's start; the bundle's policy moved no faster than before (93.1% agreement with e1800 after 900 iterations).
- *With @exp:az-raw.* The policy kept moving while the critic's accuracy and the raw policy's strength stood still through e1800. What the search gained from those nets is not visible in either readout.
- *Caveats.* The positions come from raw policies sampled at temperature 1, not from search self-play or from the search's own nodes, where the nets are actually read. Top-move agreement ignores how close the alternatives are.

#repro(```sh
python tools/checkpoint_similarity.py play --games 2000 --device cuda --out positions.npz \
  --players az-2p-kata/main@e0 az-2p-kata/main+600-fast-noselect@e900 az-2p-kata/main+1500@e1800 az-2p-bundle/main@e900
python tools/checkpoint_similarity.py score positions.npz --device cuda --out scores.npz \
  --run az-2p-kata/main --run az-2p-kata/main+600 --run az-2p-kata/main+600-fast \
  --run az-2p-kata/main+600-fast-noselect --run az-2p-kata/main+1500 --run az-2p-bundle/main
uv run --no-project --with numpy --with matplotlib python tools/checkpoint_similarity.py report scores.npz \
  --json similarity.json --figures figures/
```)

=== Is self-play exploring enough? The training search against a longer, wider one #chip("parked") <exp:az-explore>
#meta(
  dates: "built 2026-11-16; not run",
  dates-label: "parked",
  owner: "assistant",
  config: "tools/explore_diag.py",
  wandb: (),
  records: (),
)
#tldr[Marco (2026-11-16) asked whether the PPO policy is fuzzy enough to explore, or whether the root noise covers it. It was not fuzzy, the noise reaches the root only, and since e900 the search's target has sat almost on the net. Claude's suggestion, approved by Marco ("yeah let's try it"): on 3,000 self-play positions of the bundle's net, compare the training search with a 16× longer one and with a 16× longer one whose prior is widened at every node, and let playouts say whether the moves they find are better. Marco's addition: a sweep of budgets up to 16,384 sims at several temperatures on the logits, which Marco wants at game level like @exp:search-depth-2p (about 17 h for three temperatures up to 4,096 sims at 1,000 games a cell, about 17 h more per 16,384 cell). Parked by Marco the same evening, before anything ran: "good for research validation and formalities but not what i need since i just want a really good bot". The tool is merged; the search agent's temperature option was started in `wt/prior-temp` and left unfinished.]

*Question.* Has the net reached what a 600-visit search can teach it, or is the search too narrow to find the moves that would still help?

#lead[Why.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([In self-play], [fr3], [e300], [e1800], [Bundle's end]),
    [Net's entropy (nats)], [0.05], [0.20], [0.28], [0.27],
    [Search prefers another move than the net], [1.0%], [2.9%], [3.7%], [3.5%],
    [Target's distance from the net (KL, nats)], [0.41], [0.075], [0.036], [0.033],
  ),
  caption: [From the runs' own logs (`search/net_entropy`, `target/overrule`, `target/kl_net`; means of the last 20 iterations, fr3: iteration 9). The target is the pruned visit distribution.],
) <tab:az-explore-why>
- *The start was saturated.* fr3's top move has probability above 0.999997 at three quarters of the positions with a choice (@exp:az-kata).
- *What the noise covers.* The root of a full search (a quarter of the moves): every legal move is tried, and the raw visit counts have 0.75 nats of entropy. The pruned target keeps a noise-forced move only if its value earns it (15–18% of the root's visits are removed). Below the root there is no noise: the prior decides, at `c_puct` 0.75.
- *It stopped moving.* Entropy has been flat since about e900 and the target is 0.03 nats from the net: the search mostly confirms the net. The logs cannot say whether that is convergence or narrowness.

#lead[Setup.] `tools/explore_diag.py`, four steps; no change to the trainer or the engine (the prior is widened through the logits the engine is handed).
- *Positions.* 3,000 decisions with at least two legal moves, a few a game, from 1,000 games of self-play as training plays it (full and fast searches, noise, temperature schedule, the true game) by `az-2p-bundle/main@e900`, on boards outside the training pool. Each is stored as board, seed and moves so far and restored exactly.
- *Arms*, each on a fresh tree per position; an arm's move is the top of its pruned target:
  - A1, A2: training's full search (600 visits, root noise), two seeds. Their disagreement is the floor.
  - C, long and narrow: 9,600 visits, otherwise as A.
  - B, long and wide: 9,600 visits and, at every node, the prior mixed 0.75 to 0.25 with a uniform one over the legal moves.
- *Grid* (Marco: "a search sweep up to 16384 sims deep on networks with different temperature applied to the action logits"): visits 600, 2,400, 9,600 and 16,384 × temperatures 1, 1.5, 2 and 4 on the logits at every node, without root noise, on the same positions. The entropy of each tempered prior is recorded, so the curves can be read against nats added (on 24 test positions: 0.29 nats at 1, 0.44 at 1.5, 0.59 at 2, 1.12 at 4; the 0.25 mix 1.06).
- *Referee.* Where arms chose different moves, each distinct move is played and the game finished 32 times by the net's own sampled policy for both seats, with the same chance seed across a position's moves. The measure is the mover's win rate after the arm's move minus after A1's, with a bootstrap over positions.
- *Reading.* If B's or C's moves differ from A1's no more often than A2's do, or win no more in the playouts, exploration is not the limit. If B finds better moves that A1 gave no visits, wider self-play is worth a run; if C finds them as well, visits are the limit, not width.
- *Power.* A 5% disagreement rate is known to ±0.8 pp. The referee's mean difference is known to about ±2.6 pp over 150 differing positions and ±1.8 pp over 300 (assumed spread, 32 playouts); 128 playouts are affordable.
- *Cost (estimates, unmeasured for big trees).* Positions about 15 min, the four arms about 15 min (61M leaves), the grid about 90 min (348M leaves), the referee minutes. A 16,384-visit tree takes about 150 MB, so 78 positions are in flight at that budget.
- *Why not at game level.* A 16,384-sim cell took 16–18 h per 1,000 games in @exp:search-depth-2p; four temperatures at three budgets and 2,000 games a cell are about 200 GPU hours. Game-level rows follow only for the cells the positions flag, first at 1,024 sims (2.4 h a row, paired with the existing rows).
- *Caveats.* Fresh trees, where training re-roots the last search's tree. Playouts by the raw policy measure a move's worth under raw play, not under the search. A temperature cannot revive a move the net puts at $e^(-31)$; the uniform mix can.
- *Status.* Built and tested on the CPU only (14 tests; the four steps chained on 24 real positions). Not verified there: the half-precision kernels under the logit wrapper, speed and memory with big trees.

#repro(```sh
python tools/explore_diag.py positions --ckpt "az-2p-bundle/main@e900#ad8c2caf507b" --out pos.npz --games 1000 --envs 500
python tools/explore_diag.py search --positions pos.npz --out search.npz --tree-gb 12
python tools/explore_diag.py search --positions pos.npz --out search.npz --tree-gb 12 --no-defaults \
  --grid-visits 600,2400,9600,16384 --grid-temps 1,1.5,2,4
python tools/explore_diag.py referee --positions pos.npz --search search.npz --out referee.npz --playouts 32
python tools/explore_diag.py report --search search.npz --referee referee.npz --out report.json
```)

=== Six more KataGo pieces in one continued run: `az-2p-tricks` #chip("done") <exp:az-tricks>
#meta(
  dates: "2026-11-17 07:15 – 2026-11-18 03:37",
  owner: "marco",
  config: "configs/experiments/az-2p-tricks.yaml",
  wandb: (("az-2p-tricks-main-e2ac082a", "az-2p-tricks/main"),),
  records: ("cc4ccbcc",),
)
#tldr[Marco (2026-11-17) asked which other KataGo tricks would help. Of Claude's list Marco approved four ("let's try 1-4 since they seem like they would increase computational efficiency") and added two for exploration ("forced playouts and root temperature could also help?"). Weight averaging was declined. All six go into one run of 900 iterations continued from the bundle's end, as in @exp:az-bundle: bundle first, ablate later. All six are built and measured on the bundle's net. The measurements say 3 needed a much smaller weight, 4 will touch about 40% of the moves, and 1, 5 and 6 will change little. *Result:* in the 1,024-sim search the new net (`2p-graph-s-8`) scores 64.65% against fr3's raw bot, +0.95 pp over the bundle on the same games (p = 0.47) and +1.70 pp over e1800 (p = 0.20): no gain this row can see. The 900 iterations took 15% less time than the bundle's (18.0 h against 21.2 h).]

*Question.* Does the bundle's net gain more from 900 further iterations with these six pieces than e1800 gained from the bundle's 900 (+0.75 pp, p = 0.56)?

#lead[Setup.] From `az-2p-bundle/main@e900` (weights only, fresh optimizer, empty replay), the bundle's config plus:
#figure(
  btable(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    header: ([Piece], [What changes], [Setting]),
    [1. Lower-noise value target], [The value target is the game result mixed with the search's own value of the position (Canopy mixes to 85% over 60 iterations). The search's value is weighted by the pruned visits, so noise-forced visits do not drag it.], [50% search value, ramped in over the first 60 iterations],
    [2. Policy surprise weighting], [Replay rows are drawn more often where the search disagreed with the net's prior: half the draws uniform, half in proportion to the KL of the target from the prior (KataGo).], [share 0.5; a row's weight capped at 10× the mean],
    [3. Soft policy side head], [A second policy head trained on the target raised to the power 1/4. The search never reads it; it starts as a copy of the policy head.], [temperature 4; loss weight 0.125, not KataGo's and Canopy's 8 (measured below)],
    [4. Smaller searches in decided games], [Once one seat's search value has been at least 0.9 for three searches in a row, full searches drop from a quarter to a sixteenth of the moves and fast searches from 100 to 50 visits. Recorded targets stay 600-visit ones.], [0.9, 3 searches, 1/16, 50 visits],
    [5. Forced playouts], [At the root of a full search a move that already has a visit gets at least $sqrt(2 P N)$ visits (P its prior with noise, N the root's visits). The pruned target already removes those its value does not earn.], [k = 2 (KataGo)],
    [6. Root temperature], [The root's prior is raised to the power 1/T before the noise is mixed in, in full searches only.], [T = 1.25 (KataGo's early-game value)],
  ),
  caption: [The six pieces. Every one is an optional key that leaves older configs and their hashes unchanged.],
) <tab:az-tricks>
- *What 5 and 6 can and cannot do.* Both act at the root only. The root is already the best explored node: the noise tries every legal move, the raw root visits have 0.75 nats of entropy against the net's 0.27, and 15 to 18% of root visits are pruned as noise-forced (@exp:az-explore). Below the root nothing widens the prior. A temperature of 1.25 also barely moves a prior that puts 0.999 on one move. Measured below.
- *Not included.* Weight averaging (declined). A temperature on the prior at every node, which is the piece that would reach below the root (Claude's suggestion; the knob of the parked sweep in @exp:az-explore): declined by Marco on 2026-11-17 ("let's just go with katago").

#lead[Measured before the launch (pieces 1 to 4).] The bundle's net on the CPU, the bundle's search, 6 games in flight: 1,428 decisions, 292 recorded positions, 6 finished games. A small sample; the tails are rough.
#figure(
  btable(
    columns: (auto, 1fr, 1fr),
    align: (left, right, right),
    header: ([Loss term alone], [Loss], [Gradient norm]),
    [Policy], [0.277], [0.58],
    [Value, game result only], [0.288], [0.72],
    [Value, half search value], [0.343], [0.37],
    [Bundle's whole loss], [0.571], [0.93],
    [Soft policy at weight 1], [2.471], [4.68],
  ),
  caption: [Gradient norms over all parameters on the 292 positions. The run clips the gradient at 1.0.],
) <tab:az-tricks-grad>
- *Piece 3: weight 8 would swamp the update.* At weight 1 the soft loss's gradient is already 8× the policy's and 5× the clip; at 8 the clipped update would be almost all soft policy. The soft target has 1.00 nats of entropy against the policy target's 0.23. Weight 0.125 puts it level with the policy term at the start.
- *Piece 2: the surprise is concentrated.* Mean 0.043 nats, median 0.0025, 99th percentile 1.04; 74% of positions are below 0.01. Without a cap the weighted draws have an effective sample size of 10% of the buffer and one position is drawn 43× as often as under uniform draws; with the cap at 10× it is 31%. The cap stays (Claude's guard, not KataGo's).
- *Piece 4: a large share of play is decided.* At 0.9 for three searches, 37 to 41% of decisions fall in decided games (25 to 29% at 0.95, 56 to 61% at 0.8). Their searches shrink, so iterations get faster, and they supply a quarter as many recorded positions as before.
- *Piece 1: expect little.* The search's value is barely closer to the game's result than the net's own (mean absolute gap 0.211 against 0.219) and sits 0.031 from the net. Half of the target is then nearly the net's own output: the result's signal and its noise both halve, and what is added is the search's small correction.
- *Start.* The outcome head has to be named in `init_from` to come along (`parts: [actor, critic, outcome]`, new); before this the bundle's head would have restarted from fresh weights. Checked on the CPU: all 119 of the bundle's tensors arrive unchanged, the soft head equals the policy head, and the net's outputs match the bundle's bit for bit.

#lead[Measured before the launch (pieces 5 and 6).] The bundle's net, 120 positions with at least two legal moves from 4 self-play games on held-out boards, a new 600-visit search per position with noise on, four arms on the same positions and seeds.
#figure(
  btable(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right, right),
    header: ([Arm], [Raw visit entropy (nats)], [Visits pruned], [Target entropy (nats)], [Target's top move differs], [Visits of the top move]),
    [As today], [0.827], [21.1%], [0.196], [–], [436.8],
    [Forced playouts, k = 2], [0.852], [21.4%], [0.195], [0 of 120], [435.2],
    [Root temperature 1.25], [0.861], [22.7%], [0.194], [2 of 120], [428.4],
    [Both], [0.885], [23.1%], [0.196], [3 of 120], [426.4],
  ),
  caption: [Root searches of 600 visits with and without pieces 5 and 6. "Target" is the pruned policy target. The share of legal moves that get a visit is 81 to 82% in every arm.],
) <tab:az-tricks-root>
- *Both are small.* Together they move about 10 of 600 visits off the top move and add 0.06 nats to the raw visits. The pruned target, which is what the net learns, has the same entropy in every arm and a different top move in 3 of 120 positions.
- *KataGo's rule forces only moves that already have a visit.* The first build forced unvisited moves too (Claude's brief, not KataGo's rule). That version reached every legal move and added 0.21 nats, and its target was just as unchanged (entropy 0.193 to 0.194, top move different in 2 of 120). Switched to KataGo's rule after Marco's "let's just go with katago".
- *Where an effect can come from.* Not the target. The move played is sampled from the raw visits over a game's first 30 moves (at a temperature falling from 1 to 0), so games vary slightly more, and the extra visits give a few side moves a better value estimate.

#lead[GPU check before the launch] (2026-11-17 07:04, `tools/tricks_smoke.py`, 8 iterations in a scratch directory, W&B off). Passed.
- *Same net as the bundle.* This run's epoch 0 gives the bundle's logits and values bit for bit on 1,032 positions, on the CPU and on the GPU through the plain, `fused` and `fused16` evaluators; the search ignores the soft head.
- *Faster.* 67 s an iteration against the bundle's 84 s (53.5 iterations an hour against 42.7), at 65,600 leaves a second (the bundle: 68,300). Piece 4 does it: 17% of the moves get a full search where the bundle's 25% did.
- *Fewer recorded positions.* About 6,000 an iteration against the bundle's 8,185, with the same 64 updates of 512 rows, so a recorded position is drawn about 5.4 times where the bundle drew it 4 times.
- *Pieces as configured.* Full searches run with forced playouts 2 and root temperature 1.25, fast and decided searches without; the value blend rises to 0.5; the surprise weights stay under the cap of 10; the outcome head starts at the bundle's errors (2.0 points, 7.9 turns).

#lead[Evals.] As the bundle: no in-run evals, one final row: the last checkpoint in the deployed search (1,024 sims) against raw fr3 on heldout-2p-fr's first 1,000 boards × 2 seats, paired by game (exact McNemar) with the bundle's row `b5483151` (63.70%) and e1800's `ea257443` (62.95%). 2,000 games give #sym.plus.minus 2.2 pp on the row and detect a paired difference of about 2.7 pp.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Network inside the 1,024-sim search], [Score vs fr3's fair raw bot], [Wins of 2,000], [This run paired against it], [p]),
    [fr3, the start], [53.85 #ci[51.7][56.0]], [1,077], [+10.80 pp (488 vs 272)], [4e-15],
    [e300 (55,100 games)], [57.70 #ci[55.5][59.8]], [1,154], [+6.95 pp (409 vs 270)], [1e-7],
    [e900 (165,300 games)], [60.30 #ci[58.1][62.4]], [1,206], [+4.35 pp (385 vs 298)], [0.001],
    [e1800 (329,200 games)], [62.95 #ci[60.8][65.0]], [1,259], [+1.70 pp (351 vs 317)], [0.20],
    [Bundle, 900 more (490,100 games)], [63.70 #ci[61.6][65.8]], [1,274], [+0.95 pp (313 vs 294)], [0.47],
    mid,
    [This run, 900 more (650,000 games)], [64.65 #ci[62.5][66.7]], [1,293], [], [],
  ),
  caption: [`heldout-2p-fr`, first 1,000 boards in both seat orders, the same 2,000 games in every row; McNemar's exact test on the games only one of the pair won. Records `809619df`, `e917f9bb`, `4411825a`, `ea257443`, `b5483151`, `cc4ccbcc`.],
) <tab:az-tricks-result>
- *Strength.* 64.65% (68.1% moving first, 61.2% second), the highest row of the line. Against the bundle on the same games: +0.95 pp, interval −1.5 to +3.4. Against e1800: +1.70 pp, interval −0.8 to +4.2.
- *Reading.* The six pieces and 900 more iterations did not move the search by an amount this row can see, and the step is the size of the bundle's own (+0.75 pp). The line is flattening: +6.45 pp over the first 900 iterations, +2.65 pp over the next 900, +1.70 pp over the last 1,800 together (p = 0.20).
- *Speed.* 69.8 s an iteration (median) against the bundle's 84.4 s, at 66,300 leaves a second: 18.0 h for 900 iterations against 21.2 h. Piece 4 does it: 38% of decisions were in decided games and 18% of the moves got a full search (the bundle: 25%). The run played 159,919 games (the bundle: 160,906) and recorded 5.27M positions (the bundle: 7.36M), so each recorded position was drawn about 5.6 times.
- *Health.* No game was cut at the turn limit in any iteration. Value loss flat at 0.545 from iteration 100 on (the target is half search value, so it is not comparable with the bundle's 0.384); policy loss 0.679 at iteration 261 and 0.654 at the end; soft policy loss 1.42. The net picks the bundle's top move on 89.1% of probe positions at the end. The surprise-weighted draws kept an effective sample size of 47% of the buffer.
- *What it does not say.* Which piece did what: all six went in together (bundle first, ablate later), and pieces can cancel. A gain above about 3.4 pp over the bundle is ruled out; a loss is not.
- *Name.* `2p-graph-s-8` (`az-2p-tricks/main@e900#8f4cb767d188`). Not deployed: the bot on OpenHex is still e1800.

#lead[Cost.] Planned: the bundle ran 42.7 iterations an hour, which would make 900 iterations 21 h. With piece 4 the run makes about 53: about 17 h on the local GPU, plus 2.4 h for the final row (an eighth of the wall-clock). Launched at 07:15 on 2026-11-17 after `horizon-2p` was stopped (@exp:recipe-horizon) and the branches were merged. Actual: 18.0 h of self-play (51.5 iterations an hour) and 2.4 h for the row, 12% of the 20.4 h.

