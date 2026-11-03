#import "../macros.typ": *

= Hidden information <sec:hidden>

In Catan only development cards and robber steals are hidden. This section asks whether a bot gains from knowing the hidden cards better: an exact tracker of the other hands, a network head that predicts the hidden features, and acting on that prediction instead of on sampled worlds. So far nothing here moves the policy. The exact tracker is right and free, but no agent plays differently for it; the prediction head is a small regularizer; and in 2p one predicted row is worth sixteen sampled worlds.

*Levels* (@sec:front). The encoder hides 15 features of every other seat below the level that shows them: 5 resource ratios, actual VPs, 5 dev-card counts and 4 playable flags. `private` zeroes all 15 and is the only fair level past two players; `public` (the opponent's exact resource hand) is also fair in 2p; `full` adds the opponent's dev cards and is fair only through `BELIEF` or a prediction (@sec:search).

== The hand tracker

=== An exact-posterior hand tracker #chip("done") <exp:hidden-exact-tracker>
#meta(
  dates: "2026-10-17",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Marco's two robbery examples and his 14th-knight example are now engine tests, and the existing pool tracker already resolves all three. A new exact tracker (`tracker=exact`) matches brute-force enumeration exactly, where the pool tracker's deal is off by 0.12–0.18 total variation on average, at the same sampling cost.]

*Question.* Marco's idea: what is the best possible tracker for any number of players, one that resolves hidden information as soon as later actions reveal it? His examples:
- We are A. B holds a wood and a sheep; C holds a brick, an ore and a wheat. B robs C, then buys a road: B now holds a sheep, C an ore and a wheat. If instead C robbed B and bought a dev card, C holds a brick and B a wood.
- We have played 6 knights, B 6 and C 1. When C plays another, that is the 14th and last knight: no hand and no deck holds one, so contending for largest army against B is pointless. More generally, every dev-card play should update the prior that sampled worlds are drawn from.

*Setup.* Two trackers, compared on 160 positions with at least one hidden steal against the true posterior by brute-force enumeration. Neither uses evidence from what seats chose not to do.
/ Pool tracker: The C++ `HandTracker`, which stays the default. Per seat it keeps a known lower bound per resource plus a count of unknown cards, with one global pool of unknown cards derived from the bank. That already resolves both robbery examples. The knight example is covered on the sampling side (`unseen_development_cards` is the deck minus the seat's own hand minus every played card), and the row carries every seat's played cards, so the network can count too. What it lacks is a prior over which card a hidden steal took: a steal from brick, brick, ore is sampled 1:1 instead of 2:1.
/ Exact tracker: `tracker=exact` keeps weighted hypotheses over the hidden steals, branched by the victim's count of each kind, pruned by every later payment and merged when they reach the same hands. It is capped at 4,096 hypotheses (the most seen in 424 simulated 2–6p games is 748, the average 2) and falls back to the pool past the cap.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    header: ([Tracker], [Mean TV from the truth], [Worst position], [Cost per sampled world]),
    [Exact], [0.0], [0.0], [About 12 µs],
    [Pool], [0.12 / 0.18 \ #sub[random / value play]], [Up to 0.9], [About 12 µs],
  ),
  caption: [Total variation (TV) distance between each tracker's deal of the other seats' resource hands and the enumerated posterior, on 160 positions with at least one hidden steal.],
) <tab:hidden-tracker-tv>

*Reading.* The pool tracker's approximation is real, not just theoretical, and the exact tracker removes it for free. Whether any agent plays better for it is the next two experiments.

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
#tldr[The first 4p table gave exact 40.4 against pool 52.7, but the gap is the seating, not the tracker. At the `private` level every sampled world encodes to the true row, so `BELIEF` with either tracker is the plain argmax policy, game for game. Slot 0 always moves right after an F, and F robs the seat after it; every 4p head to head since plays both slot orders.]

*Question.* Does `ppo-4p`'s belief policy play better with the exact tracker?

*Setup.* `BELIEF:ckpt=ppo-4p/main@e3650:worlds=16` with `tracker=exact` in slot 0 and `tracker=pool` in slot 1, two F in slots 2–3, on 500 boards × 4 rotations of `heldout-4p` (2,000 games, 3 min). Every game pairs the two trackers.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Lineup], [Slot 0], [Slot 1], [Record]),
    [`BELIEF` exact, `BELIEF` pool, F, F], [40.4 #ci[38.2][42.5]], [52.7 #ci[50.5][54.8]], [`79147906`],
    [`POLICY`, `POLICY`, F, F \ #sub[`ppo-4p`'s own control]], [40.45], [52.75], [`b07e5bd9`],
  ),
  caption: [`ppo-4p/main@e3650` at one table with two F on `heldout-4p`; score shares, parity 25. Paired difference exact − pool: −12.3 pp [−16.5, −8.1], the same at every seat.],
) <tab:hidden-tracker-modes>

- *No tracker effect is possible.* A CPU replay of 1,208 decisions found that 0 of 4,832 worlds per mode differ from the true row in any feature: the encoder zeroes exactly what a world could change. So the record is game for game identical to the control.
- *The gap is the lineup.* The suite seats `lineup[(seat + r) mod 4]`, so slot 0 always moves right after the F in slot 3, and F (`enemies=next`) robs and blocks the seat after it. In the replayed games the F in slot 3 put the robber on slot 0's tiles in 280 of 280 moves; an F stole from slot 0 263 times and from slot 1 39 times. With `F:enemies=all` the difference flips to about +7 pp, so even then the two learner slots are not exchangeable.
- *What survived.* The exact deal reproduces the other seats' true resource hands in 72.0% of worlds, the pool deal in 60.5% (no fall-back, 1–71 hypotheses), and the exact-mode env is bit-identical to the pool-mode env.

*Reading.* A paired 4p table needs both slot orders, or the sum of the two learner slots. This voided the first 4p head to head of `ppo-mixed` against `ppo-4p` (@sec:generalist). The tracker can only matter to an agent that reads the sampled hands: in 4p that is `MCTS` with sampled worlds, or a `full`-level 4p net played through `BELIEF` (none trained). In 2p it never matters: there are no hidden steals, so the two trackers coincide.

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
#tldr[Inside a search that does read the sampled hands (256 sims, 4 worlds), exact and pool tie exactly: 193 wins each out of 400 games over both slot orders. The two games of each pair are mostly the same game, so at this budget the better deal rarely changes a move. `tracker=exact` stays a default-off option.]

*Question.* Does the exact deal help an agent that plays out steals and builds with the sampled hands?

*Setup.* `MCTS:sims=256:dets=4` with `ppo-4p/main@e3650`, exact against pool at one table with two F, 50 boards × 4 rotations of `heldout-4p` in each slot order (400 games). It took 27 min, not the estimated 3 h, because the searchers batch their leaves across the 256 games in flight.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Order], [Slot 0], [Slot 1], [Record]),
    [Exact first], [Exact 44.5 #ci[37.8][51.4]], [Pool 52.5 #ci[45.6][59.3]], [`89915b43`],
    [Pool first], [Pool 44.0 #ci[37.3][50.9]], [Exact 52.0 #ci[45.1][58.8]], [`139796c6`],
  ),
  caption: [MCTS (256 sims, 4 worlds), exact against pool with two F on `heldout-4p`; score shares. Paired over both orders: 0.0 pp [−3.2, 3.2] (bootstrap over the 200 board × rotation pairs).],
) <tab:hidden-tracker-search>

The slot after the F loses 8 points in both orders (44.3 against 52.3): the seating artifact again, now on the searcher. On the same dice the same seat won 186 of the 200 pairs, in the same number of turns in 169 and with the same number of decisions per seat in 161. The 14 pairs that changed hands split 6 to pool, 4 to exact, and 4 where exact won one game and an F the other.

*Reading.* The exact posterior is right and free (@exp:hidden-exact-tracker), but no agent in this repo plays differently for it. The `private`-level policies cannot see it, and search at the sim counts we can afford does not act on it. What would read it is open (@exp:hidden-open).

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
#tldr[Marco's idea: train the model to predict the full-information observation from what it sees. An auxiliary head that regresses the opponent's 15 hidden features does not improve the 2p policy: 89.1 against F versus 90.7 without it, and 47.7 head to head. It does show that in 2p the opponent's resources are almost fully inferable and the dev cards are not.]

*Question.* Does a trunk that has to encode the hidden features play better?

*Setup.* The arch ablation's `graph-s` cell plus the head (100M steps, 2p, `private`), judged against F and head to head against that cell on `heldout-2p`, 5,000 games a row.
- *Target:* the `full` row of the same position. The head regresses the 15 hidden features of each other seat (a test pins the set against the encoder) from the `full` players block, which the row already carries for a critic at another level (`critic_level: full`).
- *Fairness:* the trunk never reads that block. A test perturbs it and checks that logits, values and predictions are bit-identical, so the policy stays fair and evals encode only the `private` row.
- *Loss:* a masked MSE over present opponent seats on acting rows, with coefficient 0.3 from the measured gradient ratio against PPO's loss. At 0 the training is bitwise the plain recipe.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, left),
    header: ([Cell], [Steps/s], [Best on `select-2p`], [Against F], [Against `graph-s`], [Records]),
    [`graph-s` (@sec:arch) \ #sub[0.54M params]], [17.6k], [92.5 #sub[e750]], [90.7 #ci[89.9][91.5]], [—], [`039b60f0`],
    [`graph-s` + aux head \ #sub[0.62M params]], [17.1k], [90.0 #sub[e700]], [89.1 #ci[88.2][89.9]], [47.7 #ci[46.3][49.1]], [`0e28d5cf` \ `c6b7f682`],
  ),
  caption: [The aux cell against the plain `graph-s` cell, same recipe and budget, on `heldout-2p`.],
) <tab:hidden-aux-2p>

The aux cell took 1.7 h, and its critic (explained variance 0.930 against 0.928) and entropy ended where the plain cell's did. The head's R² per feature (@tab:hidden-r2, next to 4p): resources 0.90–0.99, actual VPs 0.99, VP cards 0.85, knights 0.60, the playable-knight flag 0.45, and 0.10–0.22 for monopoly, road building, year of plenty and their playable flags.

*Reading.* At coefficient 0.3 the head acted as a regularizer that cost about a point and a half and nothing else. In 2p the tracker's known cards, the hand size and the production history leave little to infer, so the trunk had nothing new to encode. The dev cards are the genuinely hidden part and stay hidden, except VP cards: a card bought and never played is a VP card, and the net works that out.

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
#tldr[At 4p the other hands are genuinely hidden: resource R² falls to 0.48–0.59 (from 0.90–0.99 in 2p), so the head recovers about half their variance. The policy still gains nothing: 63.2 with the head against 64.0 without it versus three F, and 42.5 each head to head over both slot orders.]

*Question.* Steals and the robber hide more at 4p; does the head help where there is more to infer?

*Setup.* `ppo-4p`'s recipe (`private` level, the lambda critic default) at 750 epochs: 196.6M agent steps = 49.2M decisions, the 2p cells' budget per decision.
- Two arms: plain, and aux (coefficient 0.3, critic at the `full` level as in 2p). They took 1.85 h and 1.92 h at 30.5k and 29.6k steps/s.
- `select-4p` every 50 epochs (800 games).
- Finals: each arm against three F, and the two arms at one table with two F in both slot orders (8,000 games).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    header: ([Arm], [Best on `select-4p`], [Against three F], [Head to head], [Record]),
    [Plain], [65.8 #sub[e600]], [*64.0* #ci[62.5][65.5]], [42.5], [`d97f69b3`],
    [Aux], [68.5 #sub[e700]], [63.2 #ci[61.6][64.6]], [42.5], [`da4cb5be`],
  ),
  caption: [`aux-hidden-4p` on `heldout-4p`; score shares, parity 25. Against three F is 4,000 games per arm. Head to head is the two arms at one table with two F, in both slot orders (8,000 games, `27a774b6` and `2eb6a290`): paired 0.0 pp [−2.0, 1.9].],
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
  caption: [R² of the aux head per hidden feature over the last 50 epochs, from the logged `aux/mse_*` and `aux/var_*`. What the public row already determines (2p resources, actual VPs) the head gets nearly right; dev cards stay hidden at both player counts.],
) <tab:hidden-r2>

At 4p the knight R² was 0.70 at epoch 100 and fell to 0.53 by the end. The selects overstate the finals by 2–5 points (best of sixteen noisy selects). The slot after the F loses 9 points in both orders (38.0 against 47.0), the seating artifact of @exp:hidden-tracker-modes.

*Reading.* At 4p steals and the robber leave only about half the variance of the other hands predictable from the public row; the head learns that half, and the policy still gains nothing from a trunk that encodes it. The head is a regularizer at both player counts. What did pay, in compute, was acting on the prediction in place of sampled worlds (@exp:hidden-predict). The plain arm doubles as the lambda critic's from-scratch test (@sec:recipe) and as the Markov net's control (@sec:markov), where the same head also probes the LSTM's memory.

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
#tldr[The full-information network, acting on its `private` row with the hidden features filled in by the aux head's prediction, plays as well as the same network averaged over sixteen sampled worlds: 50.4 head to head and 94.7 against F, with two forward passes instead of sixteen.]

*Question.* Marco's idea, read literally: can the agent act on the prediction instead of sampling worlds?

*Setup.* `PREDICT:ckpt=puffer-graph2/full@e3350:predictor=aux-hidden/main@e700` overwrites the opponent's 15 hidden features in the true game's `private` row with the aux head's prediction and plays the `full` network on that row. It is fair by construction, and tests pin it: the `private` row plus the fill is bitwise the `full` row, and with the true features in the head's place the agent reproduces the `full` policy's actions exactly. `heldout-2p`, 5,000 games a row; all four rows took 18 min.

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
  caption: [`PREDICT` with `puffer-graph2/full@e3350` and the `aux-hidden/main@e700` predictor on `heldout-2p`, 5,000 games a row.],
) <tab:hidden-predict>

*Reading.* This fits the R² table (@tab:hidden-r2): what the predictor gets right (resources, actual VPs) is what the tracker's worlds agree on anyway, and what it cannot know (dev cards) averages out either way. The row against `graph-s` is confounded by budget (100M steps against `puffer-graph2`'s 500M) and says nothing about the fill by itself.

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
#tldr[No agent in the repo plays differently for the exact tracker or for a 4p predictor. A `full`-level 4p net played through `BELIEF`, or search with many more worlds, would. None is queued.]

*Plan.* (Claude's suggestion, not queued.) Train a 4p net at the `full` level and play it through `BELIEF`, as 2p does; whether 4p should train at `full` is itself an open question. That net plus a 4p predictor is also what `PREDICT` needs at 4p, and a predictor head on the `full`-level network's own trunk would make `PREDICT` one network again. Search with many more than 4 worlds per decision is the other agent that would read the exact deal.
