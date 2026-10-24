#import "../macros.typ": *

= hidden information <sec:hidden>

in catan only development cards and robber steals are hidden. this section asks whether a bot gains from knowing the hidden cards better: an exact tracker of the other hands, a network head that predicts the hidden features, and acting on that prediction instead of on sampled worlds. so far nothing here moves the policy. the exact tracker is right and free but no agent plays differently for it, the prediction head is a small regulariser, and in 2p one predicted row is worth sixteen sampled worlds.

*levels.* the encoder hides 15 features of every other seat below the level that shows them: 5 resource ratios, actual victory points, 5 dev-card counts and 4 playable flags. `private` zeroes all 15 and is the only fair level past two players. in 2p every resource movement is seen by both seats, so `public` (the opponent's exact resource hand) is fair too. `full` also shows the opponent's dev cards and is fair only through `BELIEF` or a prediction (@sec:search).

== the hand tracker

=== an exact-posterior hand tracker #chip("done") <exp:hidden-exact-tracker>
#meta(
  dates: "2026-10-17",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[marco's two robbery examples and his 14th-knight example are now engine tests, and the existing pool tracker already resolves all three. a new exact tracker (`tracker=exact`) matches brute-force enumeration exactly, where the pool tracker's deal is off by 0.12–0.18 total variation on average. sampling costs the same.]

*question.* marco's idea: the best possible tracker for any number of players, one that resolves hidden information as soon as later actions reveal it. his examples:
- we are A. B holds a wood and a sheep; C holds a brick, an ore and a wheat. B robs C, then buys a road: B now holds a sheep, C an ore and a wheat. if instead C robbed B and bought a dev card, C holds a brick and B a wood.
- we have played 6 knights, B 6 and C 1. when C plays another, that was the 14th and last knight: no hand and no deck holds one, so contending for largest army against B is pointless. more generally, every dev-card play should update the prior that sampled worlds are drawn from.

*setup.* the pool tracker (C++ `HandTracker`) keeps, per seat, a known lower bound per resource plus a count of unknown cards, and one global pool of unknown cards derived from the bank. that already resolves both robbery examples. the knight example is covered on the sampling side (`unseen_development_cards` is the deck minus my hand minus every played card), and the row carries every seat's played cards, so the network can count too. what the pool tracker lacks is a prior over which card a hidden steal took: a steal from brick, brick, ore is sampled 1:1 instead of 2:1.

the exact mode keeps weighted hypotheses over the hidden steals, branched by the victim's count of each kind, pruned by every later payment and merged when they reach the same hands. it is capped at 4096 hypotheses (the most seen in 424 simulated 2–6p games is 748, the average 2) and falls back to the pool past the cap. neither mode uses evidence from what seats chose not to do. `pool` stays the default.

*result.* on 160 positions with at least one hidden steal, against the true posterior by brute-force enumeration:
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    header: ([tracker], [mean TV from the truth], [worst position], [cost per sampled world]),
    [exact], [0.0], [0.0], [about 12 µs],
    [pool], [0.12 (random play) to 0.18 (value play)], [up to 0.9], [about 12 µs],
  ),
  caption: [total variation distance between each tracker's deal of the other seats' resource hands and the enumerated posterior.],
) <tab:hidden-tracker-tv>

*reading.* the pool tracker's approximation is real, not just theoretical, and the exact tracker removes it for free. whether any agent plays better for it is the next two experiments.

#repro(```sh
# the arguments of the 160-position run are not recorded; defaults otherwise
uv run python tools/belief_exactness.py --play random
uv run python tools/belief_exactness.py --play value
```)

=== exact vs pool through `BELIEF` at 4p (void: a seating artifact) #chip("done") <exp:hidden-tracker-modes>
#meta(
  dates: "2026-10-18 14:53–14:57",
  config: "configs/experiments/tracker-modes.yaml",
  wandb: (("tracker-modes-main-4a502368", "tracker-modes, eval only"), ("ppo-4p-main-6349dc93", "ppo-4p/main (no local mirror)")),
  records: ("79147906", "b07e5bd9"),
)
#tldr[the first 4p table gave exact 40.4 against pool 52.7, but the gap is the seating, not the tracker. at the `private` level every sampled world encodes to the true row, so `BELIEF` with either tracker is the plain argmax policy, game for game. slot 0 always moves right after an F, and F robs the seat after it; every 4p head to head since plays both slot orders.]

*question.* does `ppo-4p`'s belief policy play better with the exact tracker?

*setup.* `BELIEF:ckpt=ppo-4p/main@e3650:worlds=16` with `tracker=exact` in slot 0 and `tracker=pool` in slot 1, two F in slots 2–3, 500 boards × 4 rotations of `heldout-4p` (2,000 games, 3 min). every game is a paired comparison.

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([lineup], [slot 0], [slot 1], [record]),
    [`BELIEF` exact, `BELIEF` pool, F, F], [40.4 #ci[38.2][42.5]], [52.7 #ci[50.5][54.8]], [`79147906`],
    [`POLICY`, `POLICY`, F, F (`ppo-4p`'s own control)], [40.45], [52.75], [`b07e5bd9`],
  ),
  caption: [`ppo-4p/main@e3650` at one table with two F on `heldout-4p`. paired difference exact − pool −12.3 pp [−16.5, −8.1], the same at every seat.],
) <tab:hidden-tracker-modes>

a cpu replay of 1,208 decisions found that 0 of 4,832 worlds per mode differ from the true row in any feature: the encoder zeroes exactly what a world could change. so the record is game for game identical to the control. the gap is the lineup. the suite seats `lineup[(seat + r) mod 4]`, so slot 0 always moves right after the F in slot 3, and F (`enemies=next`) robs and blocks the seat after it. in the replayed games the F in slot 3 put the robber on slot 0’s tiles in 280 of 280 moves; slot 0 was stolen from by an F 263 times, slot 1 39 times. with `F:enemies=all` the difference flips to about +7 pp, so even then the two learner slots are not exchangeable. what survived: the exact deal reproduces the other seats' true resource hands in 72.0% of worlds, the pool deal in 60.5% (no fall-back, 1–71 hypotheses), and the exact-mode env is bit-identical to the pool-mode env.

*reading.* a paired 4p table needs both slot orders, or the sum of the two learner slots. this voided the first 4p head to head of `ppo-mixed` against `ppo-4p` (@sec:generalist). the tracker can only matter to an agent that reads the sampled hands: in 4p that is `MCTS` with sampled worlds, or a `full`-level 4p net played through `BELIEF` (none trained). in 2p it never matters: there are no hidden steals, so the two trackers coincide.

#repro(```sh
catanrl run configs/experiments/tracker-modes.yaml
# the control is a ppo-4p final:
catanrl eval "POLICY:ckpt=ppo-4p/main@e3650" "POLICY:ckpt=ppo-4p/main@e3650" F F \
  --suite heldout-4p
```)

=== exact vs pool inside 4p search #chip("done") <exp:hidden-tracker-search>
#meta(
  dates: "2026-10-19 07:11–07:38",
  config: "configs/experiments/tracker-search.yaml",
  wandb: (("tracker-search-main-b797a5b7", "tracker-search, eval only"),),
  records: ("89915b43", "139796c6"),
)
#tldr[inside a search that does read the sampled hands (256 sims, 4 worlds), exact and pool tie exactly: 193 wins each out of 400 over both slot orders. the two games of each pair are mostly the same game, so at this budget the better deal rarely changes a move. `tracker=exact` stays as a default-off option.]

*question.* does the exact deal help an agent that plays out steals and builds with the sampled hands?

*setup.* `MCTS:sims=256:dets=4` with `ppo-4p/main@e3650`, exact against pool at one table with two F, 50 boards × 4 rotations of `heldout-4p` in each slot order (400 games). it took 27 min, not the estimated 3 h, because the searchers batch their leaves across the 256 games in flight.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, left),
    header: ([order], [slot 0], [slot 1], [record]),
    [exact first], [exact 44.5 #ci[37.8][51.4]], [pool 52.5 #ci[45.6][59.3]], [`89915b43`],
    [pool first], [pool 44.0 #ci[37.3][50.9]], [exact 52.0 #ci[45.1][58.8]], [`139796c6`],
  ),
  caption: [MCTS 256 sims, 4 worlds, exact against pool with two F on `heldout-4p`. paired over both orders 0.0 pp [−3.2, 3.2] (bootstrap over the 200 board × rotation pairs).],
) <tab:hidden-tracker-search>

the slot after the F loses 8 points in both orders (44.3 against 52.3): the seating artifact again, now on the searcher. on the same dice the same seat won 186 of the 200 pairs, in the same number of turns in 169, with the same number of decisions per seat in 161. the 14 pairs that changed hands split 6 to pool, 4 to exact, and 4 where exact won one game and an F the other.

*reading.* the exact posterior is right (identical to enumeration, where the pool deal is off by 0.12–0.18 total variation) and costs nothing, but no agent in this repo plays differently for it. the `private`-level policies cannot see it, and search at the sim counts we can afford does not act on it. what would read it is open (@exp:hidden-open).

#repro(```sh
catanrl run configs/experiments/tracker-search.yaml
```)

== predicting the hidden features

=== a head that predicts the hidden features, 2p #chip("done") <exp:hidden-aux-2p>
#meta(
  dates: "2026-10-18 13:11–14:53",
  owner: "marco",
  config: "configs/experiments/aux-hidden.yaml",
  wandb: (("aux-hidden-main-943317eb", "aux-hidden/main"),),
  records: ("0e28d5cf", "c6b7f682"),
)
#tldr[marco's idea: train the model to predict the full-information observation from what it sees. an auxiliary head that regresses the opponent's 15 hidden features does not improve the 2p policy: 89.1 against F versus 90.7 without it, and 47.7 head to head. what it does show is that in 2p the opponent's resources are almost fully inferable, and the dev cards are not.]

*question.* does a trunk that has to encode the hidden features play better?

*setup.* the target already exists: the `full` row of the same position. the head regresses the 15 hidden features of each other seat (a test pins the set against the encoder) from the `full` players block the row carries for a critic at another level (`critic_level: full`). the trunk never reads that block: a test perturbs it and checks that logits, values and predictions are bit-identical, so the policy stays fair and evals encode only the `private` row. the loss is a masked MSE over present opponent seats on acting rows, with coefficient 0.3 from the measured gradient ratio against ppo's loss; at 0 the training is bitwise the plain recipe. the run is the arch ablation's `graph-s` cell plus the head (100M steps, 2p, `private`), judged against F and head to head against that cell on `heldout-2p`, 5,000 games a row.

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([cell], [params], [steps/s], [select-2p (best)], [vs F], [vs `graph-s`]),
    [`graph-s` (@sec:arch)], [0.54M], [17.6k], [92.5 (e750)], [90.7 #ci[89.9][91.5] `039b60f0`], [—],
    [`graph-s` + aux head], [0.62M], [17.1k], [90.0 (e700)], [89.1 #ci[88.2][89.9] `0e28d5cf`], [47.7 #ci[46.3][49.1] `c6b7f682`],
  ),
  caption: [the aux cell against the plain `graph-s` cell, same recipe and budget, on `heldout-2p`.],
) <tab:hidden-aux-2p>

the aux cell took 1.7 h, and its critic (explained variance 0.930 against 0.928) and entropy ended where the plain cell's did. per-feature R² of the head over the last 50 epochs (from the logged `aux/mse_*` and `aux/var_*`): resources 0.90–0.99, actual vp 0.99, vp cards 0.85, knights 0.60, monopoly, road building and year of plenty 0.10–0.22, the playable flags the same. the per-feature table, next to 4p, is in @tab:hidden-r2.

*reading.* at coefficient 0.3 the head acted as a regulariser that cost about a point and a half and nothing else. in 2p the tracker's known cards, the hand size and the production history leave little to infer, so the trunk had nothing new to encode. the dev cards are the genuinely hidden part and stay hidden, except vp cards: a card bought and never played is a vp card, and the net works that out.

#repro(```sh
catanrl run configs/experiments/aux-hidden.yaml
# the head-to-head final needs the plain cell:
catanrl run configs/experiments/arch-ablation.yaml --only graph-s
```)

=== the same head at 4p #chip("done") <exp:hidden-aux-4p>
#meta(
  dates: "2026-10-19 10:15–14:02",
  config: "configs/experiments/aux-hidden-4p.yaml",
  wandb: (("aux-hidden-4p-plain-f07eb81c", "aux-hidden-4p/plain"), ("aux-hidden-4p-aux-4dcfd020", "aux-hidden-4p/aux")),
  records: ("d97f69b3", "da4cb5be", "27a774b6", "2eb6a290"),
)
#tldr[at 4p the other hands are genuinely hidden: resource R² falls to 0.48–0.59 from 0.90–0.99 in 2p, and the head recovers about half their variance. the policy still gains nothing: 64.0 against 63.2 versus three F, and 42.5 against 42.5 head to head over both slot orders.]

*question.* steals and the robber hide more at 4p. does the head help where there is more to infer?

*setup.* `ppo-4p`'s recipe (`private` level, the lambda critic default) at 750 epochs: 196.6M agent steps = 49.2M decisions, the 2p cells' budget per decision. two arms, plain and aux (coefficient 0.3, the critic at the `full` level as in 2p): 1.85 h and 1.92 h at 30.5k and 29.6k steps/s. select-4p every 50 epochs (800 games). finals: each arm against three F, and the two arms at one table with two F in both slot orders (8,000 games).

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    header: ([arm], [select-4p (best)], [vs three F], [head to head], [record]),
    [plain], [65.8 (e600)], [*64.0* #ci[62.5][65.5]], [42.5], [`d97f69b3`],
    [aux], [68.5 (e700)], [63.2 #ci[61.6][64.6]], [42.5], [`da4cb5be`],
  ),
  caption: [`aux-hidden-4p` on `heldout-4p`, shares, parity 25. vs three F is 4,000 games per arm. head to head is the two arms at one table with two F, in both slot orders (8,000 games, `27a774b6` and `2eb6a290`): paired 0.0 pp [−2.0, 1.9].],
) <tab:hidden-aux-4p>

#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    header: ([hidden feature], [2p (`aux-hidden`)], [4p (`aux-hidden-4p/aux`)]),
    [wheat], [0.99], [0.59],
    [ore], [0.96], [0.57],
    [sheep], [0.96], [0.52],
    [brick], [0.93], [0.48],
    [wood], [0.90], [0.55],
    [actual victory points], [0.99], [0.95],
    [vp cards in hand], [0.85], [0.65],
    [knights in hand], [0.60], [0.51],
    [other dev cards, playable flags], [0.10–0.22], [0.07–0.21],
  ),
  caption: [R² of the aux head per hidden feature over the last 50 epochs. what the public row already determines (2p resources, actual vp) the head gets nearly right; dev cards stay hidden at both player counts.],
) <tab:hidden-r2>

at 4p the knight R² was 0.70 at epoch 100 and fell to 0.53 by the end. the selects overstate the finals by 2–5 points (best of sixteen noisy selects). the slot after the F loses 9 points in both orders (38.0 against 47.0), the seating artifact of @exp:hidden-tracker-modes.

*reading.* steals and the robber hide half the variance of the other hands from the public row, the head recovers that half, and the policy still gains nothing from a trunk that encodes it. the head is a regulariser at both player counts; what paid was using the prediction in place of sampled worlds (@exp:hidden-predict). the plain arm doubles as the lambda critic's from-scratch test (@sec:recipe) and the markov net's control (@sec:markov), where the same head also probes the lstm's memory.

#repro(```sh
# plain first: the aux arm's head-to-head finals need it
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
#tldr[the full-information network, acting on its `private` row with the hidden features filled in by the aux head's prediction, plays as well as the same network averaged over sixteen sampled worlds: 50.4 head to head and 94.7 against F, with two forward passes instead of sixteen. in 2p one predicted row is worth sixteen sampled worlds.]

*question.* the literal reading of marco's idea: act on the prediction instead of sampling worlds.

*setup.* `PREDICT:ckpt=puffer-graph2/full@e3350:predictor=aux-hidden/main@e700` overwrites the opponent's 15 hidden features in the true game's `private` row with the aux head's prediction and plays the `full` network on it. it is fair by construction, and tests pin it: the `private` row plus the fill is bitwise the `full` row, and with the true features in the head's place the agent reproduces the `full` policy's actions exactly. `heldout-2p`, 5,000 games a row, 18 min for all four rows.

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([row], [score], [record]),
    [`PREDICT` vs F], [*94.7* #ci[94.1][95.3]], [`b74d240b`],
    [`PREDICT` vs `BELIEF` 16 worlds, same `full` checkpoint], [*50.4* #ci[49.1][51.8]], [`e7e34d28`],
    [`PREDICT`, predictions snapped to the encoder's grid, vs `BELIEF` 16 worlds], [49.9 #ci[48.5][51.3]], [`a939fa5f`],
    [`PREDICT` vs `arch-ablation/graph-s` (`private`, 100M steps)], [63.1 #ci[61.8][64.4]], [`751509ea`],
    mid,
    [reference: `BELIEF` 16 worlds vs F (@sec:recipe)], [95.06 #ci[94.42][95.63]], [`67869e7a`],
    [reference: `full` policy on the true row vs F, unfair], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
  ),
  caption: [`PREDICT` with `puffer-graph2/full@e3350` and the `aux-hidden/main@e700` predictor on `heldout-2p`, 5,000 games a row.],
) <tab:hidden-predict>

*reading.* this fits the R² table: what the predictor gets right (resources, actual vp) is what the tracker's worlds agree on anyway, and what it cannot know (dev cards) averages out either way. the last row is confounded by budget (100M steps against `puffer-graph2`’s 500M) and says nothing about the fill by itself.

#repro(```sh
catanrl run configs/experiments/predict-vs-belief.yaml
```)

== open

=== what would read a better belief at 4p #chip("todo") <exp:hidden-open>
#meta(
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[no agent in the repo plays differently for the exact tracker or for a 4p predictor. a `full`-level 4p net played through `BELIEF`, or search with many more worlds, would. none is queued.]

*plan.* (my suggestion, not queued) train a 4p net at the `full` level and play it through `BELIEF`, as 2p does; whether 4p should train at `full` is itself an open question. that net plus a 4p predictor is also what `PREDICT` needs at 4p. a predictor head on the `full`-level network's own trunk would make `PREDICT` one network again. search with many more than 4 worlds per decision is the other agent that would read the exact deal.
