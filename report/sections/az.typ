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
- A Canopy-shaped target (divergence 1 of @exp:az-canopy-diff). Tested offline first in @exp:az-gate.
- Per-simulation world resampling (divergence 4). (Claude's suggestion, not queued)
- An offline distillation gate: fits to one fixed set of search targets, as a cheap test of whether there is anything to distill. Queued as @exp:az-gate (approved by Marco on 2026-11-09).
- Cheaper self-play: the engine top-up budget and eval cache (@exp:az-cost). (Claude's suggestion, not queued)
- The warm-start critic is discounted V-trace while AlphaZero trains undiscounted win/loss, so the value head drifts; the critic refit (@sec:critic) measures that.

`az-2p-2048`'s search finals (`configs/experiments/az-2p-2048-search.yaml`: best and init in the 2048-sim search) were never played.

== The offline distillation gate

=== Fit fr3 once to fixed 2048-sim searches: `az-gate-2p` #chip("running") <exp:az-gate>
#meta(
  dates: "2026-11-10 08:14 –",
  owner: "marco",
  config: "configs/experiments/az-gate-2p.yaml",
  wandb: (("az-gate-2p-positions-b6aa1044", "positions"),),
  records: (),
)
#tldr[Claude's design, approved by Marco on 2026-11-09. Before another self-play loop: search 102,400 of fr3’s own positions once with `az-2p-2048`’s search, fit fr3 to three target shapes from that one dataset at `az-2p-2048`’s e6 dose, and judge each fit inside the search against fr3’s own search, paired on the same 2,000 games. Running on the GPU since 10 November; a 10-minute GPU smoke of all four variants passed.]

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

#lead[Cost.] About 7–8 h of GPU with the card alone: 1.8–2.5 h of searching (210M leaf evaluations; `az-2p-2048` ran 23.6k a second), a minute per fit, about 1.15 h per harness row (from `search-depth-2p`’s cells; 4.6 h for four) and about 5 minutes per raw row. A CPU smoke of the whole pipeline passed; a 10-minute GPU smoke comes first.

#repro(```sh
catanrl run configs/experiments/az-gate-2p.yaml   # positions (then the baseline row), engine, canopy, canopy-anchor
uv run python tools/search_probe.py --pair <a fit's harness record> <the baseline record>
```)
