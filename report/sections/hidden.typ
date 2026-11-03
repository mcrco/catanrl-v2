#import "../macros.typ": *

= Hidden information <sec:hidden>

In Catan only development cards and robber steals are hidden. Three ways to know them better leave the policy unchanged: an exact tracker of the other hands is right and free but changes no agent's play, a head that predicts the hidden features is a small regularizer, and in 2p acting on that prediction is worth sixteen sampled worlds.

== The hand tracker

=== An exact-posterior hand tracker #chip("done") <exp:hidden-exact-tracker>
#meta(
  dates: "2026-10-17",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Marco's robbery and 14th-knight examples are now engine tests; the existing pool tracker already resolves all three. A new exact tracker (`tracker=exact`) matches brute-force enumeration, where the pool tracker's deal is off by 0.12–0.18 total variation, at the same cost.]

*Question.* Marco's idea: what is the best possible tracker for any number of players? His examples:
- Robbery: B holds wood and sheep, C brick, ore and wheat. If B robs C and buys a road, B holds a sheep and C ore and wheat; if C robs B and buys a dev card, C holds a brick and B a wood.
- The 14th knight: we have played 6, B 6 and C 1. When C plays another, no hand or deck holds a knight, so contending for largest army against B is pointless. Every dev-card play should update the sampling prior.

*Setup.* Neither tracker uses evidence from what seats chose not to do.
/ Pool tracker: The C++ `HandTracker`, still the default: per seat a known lower bound per resource plus a count of unknown cards, and one global pool of unknown cards. It resolves both robbery examples; the knight example is covered by sampling (`unseen_development_cards` excludes every played card) and by the row, which shows every seat's played cards. It lacks a prior over which card a hidden steal took: a steal from brick, brick, ore is sampled 1:1, not 2:1.
/ Exact tracker: Weighted hypotheses over the hidden steals, branched by the victim's count of each kind, pruned by later payments, merged when they reach the same hands. Capped at 4,096 (at most 748 seen in 424 simulated 2–6p games, 2 on average); past the cap it falls back to the pool.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    header: ([Tracker], [Mean TV from the truth], [Worst position], [Cost per sampled world]),
    [Exact], [0.0], [0.0], [About 12 µs],
    [Pool], [0.12 / 0.18 \ #sub[random / value play]], [Up to 0.9], [About 12 µs],
  ),
  caption: [Total variation (TV) distance from the brute-force posterior of the other seats' resource hands, on 160 positions with at least one hidden steal.],
) <tab:hidden-tracker-tv>

*Reading.* The pool tracker's error is real; the exact tracker removes it for free. The next two experiments ask whether any agent plays better for it.

#repro(```sh
# The arguments of the 160-position run are not recorded; defaults otherwise
uv run python tools/belief_exactness.py --play random
uv run python tools/belief_exactness.py --play value
```)

=== Exact vs pool through `BELIEF` at 4p (void: a seating artifact) #chip("done") <exp:hidden-tracker-modes>
#meta(
  dates: "2026-10-18 14:53–14:57",
  config: "configs/experiments/tracker-modes.yaml",
  wandb: (("tracker-modes-main-4a502368", "tracker-modes, eval only"), ("ppo-4p-main-6349dc93", "ppo-4p/main (no local mirror)")),
  records: ("79147906", "b07e5bd9"),
)
#tldr[The first 4p table gave exact 40.4 against pool 52.7, but the gap is the seating. At the `private` level every sampled world encodes to the true row, so `BELIEF` with either tracker is the plain argmax policy. Slot 0 sits after an F, which robs it; every 4p head to head since plays both slot orders.]

*Question.* Does `ppo-4p`'s belief policy play better with the exact tracker?

*Setup.* `BELIEF:ckpt=ppo-4p/main@e3650:worlds=16` with `tracker=exact` in slot 0, `tracker=pool` in slot 1 and two F, on 500 boards × 4 rotations of `heldout-4p` (2,000 games, 3 min).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Lineup], [Slot 0], [Slot 1], [Record]),
    [`BELIEF` exact, `BELIEF` pool, F, F], [40.4 #ci[38.2][42.5]], [52.7 #ci[50.5][54.8]], [`79147906`],
    [`POLICY`, `POLICY`, F, F \ #sub[`ppo-4p`'s own control]], [40.45], [52.75], [`b07e5bd9`],
  ),
  caption: [`ppo-4p/main@e3650` with two F on `heldout-4p`. Paired exact − pool: −12.3 pp [−16.5, −8.1], the same at every seat.],
) <tab:hidden-tracker-modes>

- *No tracker effect is possible.* A CPU replay of 1,208 decisions found 0 of 4,832 worlds per mode differing from the true row, so the record equals the control game for game.
- *The gap is the lineup* (the slot after an F, @sec:front). In the replayed games the F in slot 3 put the robber on slot 0's tiles in 280 of 280 moves; an F stole from slot 0 263 times, from slot 1 39 times. With `F:enemies=all` the difference flips to about +7 pp: even then the learner slots are not exchangeable.
- *What survived.* The exact deal reproduces the other seats' true hands in 72.0% of worlds, the pool deal in 60.5% (no fall-back, 1–71 hypotheses); the exact-mode env is bit-identical to the pool-mode env.

*Reading.* The seating voided the first 4p head to head of `ppo-mixed` against `ppo-4p` (@sec:generalist). Only an agent that reads the sampled hands can use the tracker: in 4p, `MCTS` with sampled worlds or a `full`-level net through `BELIEF` (none trained). In 2p there are no hidden steals, so the trackers coincide.

#repro(```sh
catanrl run configs/experiments/tracker-modes.yaml
# The control is a ppo-4p final:
catanrl eval "POLICY:ckpt=ppo-4p/main@e3650" "POLICY:ckpt=ppo-4p/main@e3650" F F \
  --suite heldout-4p
```)

=== Exact vs pool inside 4p search #chip("done") <exp:hidden-tracker-search>
#meta(
  dates: "2026-10-19 07:11–07:38",
  config: "configs/experiments/tracker-search.yaml",
  wandb: (("tracker-search-main-b797a5b7", "tracker-search, eval only"),),
  records: ("89915b43", "139796c6"),
)
#tldr[Inside a search that reads the sampled hands (256 sims, 4 worlds), exact and pool tie: 193 wins each out of 400 games over both slot orders. Paired games are mostly the same game: at this budget the better deal rarely changes a move. `tracker=exact` stays default-off.]

*Question.* Does the exact deal help a search that plays with the sampled hands?

*Setup.* `MCTS:sims=256:dets=4` with `ppo-4p/main@e3650`, exact against pool with two F, 50 boards × 4 rotations of `heldout-4p` per slot order (400 games); 27 min, not the estimated 3 h, as the searchers batch leaves across the 256 games in flight.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Order], [Slot 0], [Slot 1], [Record]),
    [Exact first], [Exact 44.5 #ci[37.8][51.4]], [Pool 52.5 #ci[45.6][59.3]], [`89915b43`],
    [Pool first], [Pool 44.0 #ci[37.3][50.9]], [Exact 52.0 #ci[45.1][58.8]], [`139796c6`],
  ),
  caption: [MCTS, exact against pool with two F on `heldout-4p`. Paired over both orders: 0.0 pp [−3.2, 3.2] (bootstrap over the 200 board × rotation pairs).],
) <tab:hidden-tracker-search>

The slot after the F loses 8 points in both orders (44.3 against 52.3). The same seat won 186 of the 200 pairs, in the same number of turns in 169 and the same decisions per seat in 161. Of the 14 that changed hands, 6 went to pool, 4 to exact, and in 4 exact won one game and an F the other.

*Reading.* No agent here plays differently for the exact posterior: `private`-level policies cannot see it, and search at affordable sim counts does not act on it. What would is open (@exp:hidden-open).

#repro(```sh
catanrl run configs/experiments/tracker-search.yaml
```)

== Predicting the hidden features

=== A head that predicts the hidden features, 2p #chip("done") <exp:hidden-aux-2p>
#meta(
  dates: "2026-10-18 13:11–14:53",
  owner: "marco",
  config: "configs/experiments/aux-hidden.yaml",
  wandb: (("aux-hidden-main-943317eb", "aux-hidden/main"),),
  records: ("0e28d5cf", "c6b7f682"),
)
#tldr[Marco's idea: train the model to predict the full-information observation. A head regressing the opponent's 15 hidden features does not improve the 2p policy: 89.1 against F versus 90.7 without it, 47.7 head to head. In 2p the opponent's resources are almost fully inferable; the dev cards are not.]

*Question.* Does a trunk that must encode the hidden features play better?

*Setup.* The arch ablation's `graph-s` cell (@exp:arch-formal) plus the head, same recipe and budget.
- *Target:* the `full` players block, already in the row for a `full`-level critic (`critic_level: full`); a test pins the 15 features against the encoder.
- *Fairness:* the trunk never reads the block; a test perturbs it and checks that logits, values and predictions stay bit-identical. Evals encode only the `private` row.
- *Loss:* masked MSE over present opponent seats on acting rows, coefficient 0.3 (from the gradient ratio against PPO's loss); at 0, training is bitwise the plain recipe.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, left),
    header: ([Cell], [Steps/s], [Best on `select-2p`], [Against F], [Against `graph-s`], [Records]),
    [`graph-s` (@sec:arch) \ #sub[0.54M params]], [17.6k], [92.5 #sub[e750]], [90.7 #ci[89.9][91.5]], [—], [`039b60f0`],
    [`graph-s` + aux head \ #sub[0.62M params]], [17.1k], [90.0 #sub[e700]], [89.1 #ci[88.2][89.9]], [47.7 #ci[46.3][49.1]], [`0e28d5cf` \ `c6b7f682`],
  ),
  caption: [The aux cell and the plain `graph-s` cell on `heldout-2p`, 5,000 games a row.],
) <tab:hidden-aux-2p>

The aux cell took 1.7 h; its critic (explained variance 0.930 against 0.928) and entropy matched the plain cell's. @tab:hidden-r2 has the R² per feature.

*Reading.* At coefficient 0.3 the head is only a regularizer. In 2p the tracker's known cards, hand size and production history leave little to infer. Dev cards stay hidden, except VP cards: a card bought and never played is a VP card.

#repro(```sh
catanrl run configs/experiments/aux-hidden.yaml
# The head-to-head final needs the plain cell:
catanrl run configs/experiments/arch-ablation.yaml --only graph-s
```)

=== The same head at 4p #chip("done") <exp:hidden-aux-4p>
#meta(
  dates: "2026-10-19 10:15–14:02",
  config: "configs/experiments/aux-hidden-4p.yaml",
  wandb: (("aux-hidden-4p-plain-f07eb81c", "aux-hidden-4p/plain"), ("aux-hidden-4p-aux-4dcfd020", "aux-hidden-4p/aux")),
  records: ("d97f69b3", "da4cb5be", "27a774b6", "2eb6a290"),
)
#tldr[At 4p the other hands are genuinely hidden: resource R² is 0.48–0.59, against 0.90–0.99 in 2p. The policy still gains nothing: 63.2 with the head against 64.0 without it versus three F, and 42.5 each head to head over both slot orders.]

*Question.* Does the head help at 4p, where steals and the robber hide more?

*Setup.* `ppo-4p`'s recipe (`private`, lambda critic) at 750 epochs: 196.6M agent steps = 49.2M decisions, the 2p cells' budget per decision. Plain and aux arms (coefficient 0.3, `full`-level critic) took 1.85 h and 1.92 h at 30.5k and 29.6k steps/s; `select-4p` (800 games) every 50 epochs.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    header: ([Arm], [Best on `select-4p`], [Against three F], [Head to head], [Record]),
    [Plain], [65.8 #sub[e600]], [*64.0* #ci[62.5][65.5]], [42.5], [`d97f69b3`],
    [Aux], [68.5 #sub[e700]], [63.2 #ci[61.6][64.6]], [42.5], [`da4cb5be`],
  ),
  caption: [`aux-hidden-4p` on `heldout-4p`, parity 25. Against three F: 4,000 games per arm. Head to head: both arms at one table with two F, both slot orders (8,000 games, `27a774b6` and `2eb6a290`); paired 0.0 pp [−2.0, 1.9].],
) <tab:hidden-aux-4p>

#figure(
  btable(
    columns: (auto, auto, auto),
    align: (left, right, right),
    header: ([Hidden feature], [2p #sub[`aux-hidden`]], [4p #sub[`aux-hidden-4p/aux`]]),
    [Wheat], [0.99], [0.59],
    [Ore], [0.96], [0.57],
    [Sheep], [0.96], [0.52],
    [Brick], [0.93], [0.48],
    [Wood], [0.90], [0.55],
    [Actual VPs], [0.99], [0.95],
    [VP cards in hand], [0.85], [0.65],
    [Knights in hand], [0.60], [0.51],
    [Playable-knight flag], [0.45], [0.42],
    [Other dev cards and their playable flags], [0.10–0.22], [0.07–0.21],
  ),
  caption: [R² of the aux head per hidden feature over the last 50 epochs, from the logged `aux/mse_*` and `aux/var_*`.],
) <tab:hidden-r2>

The 4p knight R² fell from 0.70 at epoch 100 to 0.53 at the end. The selects overstate the finals by 2–5 points (best of sixteen noisy selects). The slot after the F loses 9 points in both orders (38.0 against 47.0).

*Reading.* The head learns the predictable half of the other hands, and the policy gains nothing from it: a regularizer at both player counts. Acting on the prediction paid, in compute (@exp:hidden-predict). The plain arm is also the lambda critic's from-scratch test (@sec:recipe) and the Markov net's control (@sec:markov), where the same head probes the LSTM's memory.

#repro(```sh
# Plain first: the aux arm's head-to-head finals need it
catanrl run configs/experiments/aux-hidden-4p.yaml
```)

=== `PREDICT`: one predicted row instead of sixteen sampled worlds #chip("done") <exp:hidden-predict>
#meta(
  dates: "2026-10-18 17:45–18:03",
  owner: "marco",
  config: "configs/experiments/predict-vs-belief.yaml",
  wandb: (("predict-vs-belief-main-bccb764e", "predict-vs-belief, eval only"),),
  records: ("b74d240b", "e7e34d28", "a939fa5f", "751509ea"),
)
#tldr[The `full` network, acting on its `private` row with the hidden features filled in by the aux head's prediction, plays as well as the same network averaged over sixteen sampled worlds (50.4 head to head, 94.7 against F), with two forward passes instead of sixteen.]

*Question.* Marco's idea, literally: can the agent act on the prediction instead of sampling worlds?

*Setup.* `PREDICT:ckpt=puffer-graph2/full@e3350:predictor=aux-hidden/main@e700` overwrites the opponent's 15 hidden features in the `private` row with the aux head's prediction and plays the `full` network on it. Tests pin its fairness: the `private` row plus the fill is bitwise the `full` row, and with the true features in the head's place it reproduces the `full` policy exactly. All four rows took 18 min.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, 1fr, auto, auto),
    align: (left, left, right, left),
    header: ([Agent], [Opponent], [Score], [Record]),
    [`PREDICT`], [F], [*94.7* #ci[94.1][95.3]], [`b74d240b`],
    [`PREDICT`], [`BELIEF`, 16 worlds \ #sub[same `full` checkpoint]], [*50.4* #ci[49.1][51.8]], [`e7e34d28`],
    [`PREDICT` \ #sub[predictions snapped to the encoder's grid]], [`BELIEF`, 16 worlds], [49.9 #ci[48.5][51.3]], [`a939fa5f`],
    [`PREDICT`], [`arch-ablation/graph-s` \ #sub[`private`, 100M steps]], [63.1 #ci[61.8][64.4]], [`751509ea`],
    tgroup(4)[References (@sec:recipe)],
    [`BELIEF`, 16 worlds], [F], [95.06 #ci[94.42][95.63]], [`67869e7a`],
    [`full` policy on the true row (unfair)], [F], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
  ),
  caption: [`PREDICT` (`puffer-graph2/full@e3350`, predictor `aux-hidden/main@e700`) on `heldout-2p`, 5,000 games a row.],
) <tab:hidden-predict>

*Reading.* What the predictor gets right (resources, actual VPs; @tab:hidden-r2) the tracker's worlds agree on anyway, and what it cannot know (dev cards) averages out either way. The row against `graph-s` is confounded by budget (100M against `puffer-graph2`'s 500M steps) and says nothing about the fill.

#repro(```sh
catanrl run configs/experiments/predict-vs-belief.yaml
```)

== Open

=== What would read a better belief at 4p #chip("todo") <exp:hidden-open>
#meta(
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[A `full`-level 4p net through `BELIEF`, or search with many more worlds, would read a better belief. Neither is queued.]

*Plan.* (Claude's suggestion, not queued.) Train a 4p net at `full` and play it through `BELIEF`, as 2p does (whether 4p should train at `full` is itself open). With a 4p predictor it is also what `PREDICT` needs at 4p; a predictor head on its own trunk would make `PREDICT` one network. The other reader of the exact deal is search with many more than 4 worlds per decision.
