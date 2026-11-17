#import "../macros.typ": *

= One model for any number of players and any map <sec:generalist>

Marco's idea: one network for any player count and any map, using masked numeric features and the graph network, which already handles any hex board. `ppo-mixed` nearly matches the specialists against F, even on unseen maps, but loses to them head to head. Training on blob maps too made it worse everywhere, so `ppo-mixed` is the bot on the island table (@sec:deploy).

== One network over 2–4 players

=== `ppo-mixed`: 2, 3 and 4 players on BASE and hexagon boards #chip("done") <exp:gen-ppo-mixed>
#meta(
  dates: "2026-10-17 02:14–13:54",
  owner: "marco",
  config: "configs/experiments/ppo-mixed.yaml",
  wandb: (("ppo-mixed-main-f4b771c0", "ppo-mixed/main"),),
  records: ("b2690265", "0f448f68", "7c17d0ae", "9b470a17", "340ce1db", "32aab9f3"),
)
#tldr[One network for 2–4 players on two board types comes within 2–3 points of each specialist against F and plays unseen blob maps nearly as well as BASE (73.9 vs 77.0). Head to head it scores 45.7% in 2p and trails by 2.7 points in 4p.]

*Question.* Can one network match the specialists at the same total decisions?

#lead[Setup.]
- *Scenarios.* `ppo-4p`'s recipe (@sec:recipe) over six equally weighted scenarios, drawn per game by `env.scenarios`: 2, 3 and 4 players on BASE and on `hexagon` boards.
- *Masking.* Rows are padded to the env's maxima with a validity bit per slot; absent seats are zero, the global block one-hots the seat count, and the per-seat value head predicts every present seat.
- *Budget.* `ppo-4p`'s: 3,800 epochs, 1B agent steps = 249M decisions, 11.5 h. Best `e3550` by `select-mixed` (83.6). Run without the W&B mirror, backfilled.
- *Blob maps* (`random-4p`) stay out of training: they would double the row (6,316 floats vs 3,202) and the memory.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    header: ([Setting], [Games], [`ppo-mixed`], [Specialist], [Record]),
    tgroup(5)[Against F],
    [2p (`heldout-2p`)], [5,000], [92.3 #ci[91.6][93.0]], [95.2], [`b2690265`],
    [3p \ #sub[the 3-player games of `heldout-mixed`; parity 33]], [1,500], [81.1 #ci[79.0][83.0]], [—], [`9b470a17`],
    [4p (`heldout-4p`)], [4,000], [77.0 #ci[75.7][78.3]], [78.6], [`0f448f68`],
    [Blob maps, zero-shot (`heldout-random-4p`) \ #sub[15–30 tiles]], [4,000], [73.9 #ci[72.5][75.3]], [—], [`7c17d0ae`],
    [The mixed suite as a whole (`heldout-mixed`)], [4,500], [79.8 #ci[78.6][80.9]], [—], [`9b470a17`],
    tgroup(5)[Head to head (generalist's share, specialist's share)],
    [2p, vs the specialist belief-averaged \ #sub[`BELIEF`, 16 worlds]], [5,000], [45.7 #ci[44.4][47.1]], [], [`340ce1db`],
    [4p, vs `ppo-4p` with two F \ #sub[both slot orders (@exp:gen-slot-order)]], [2 × 4,000], [45.1], [47.8], [`32aab9f3` \ `0903fae0`],
  ),
  caption: [`ppo-mixed@e3550`, score shares. Specialists: `puffer-graph2/full@e3350` in 2p, raw against F (`ff8528a4`; unfair, it sees F's dev cards), and `ppo-4p/main@e3650` in 4p. The 4p head to head is paired over both slot orders: −2.7 pp #ci[−4.7][−0.8] per game for the generalist.],
)

*Reading.* Seat count and board shape cost little, even zero-shot. But each setting got a third of a specialist's decisions, so losing the direct matches is a budget question as much as a capacity one. Next: more decisions per setting (@exp:gen-followups).

#repro(```sh
# Training, then the six finals of eval.final
catanrl run configs/experiments/ppo-mixed.yaml
```)

=== Slot order decides 4p head-to-head tables #chip("done") <exp:gen-slot-order>
#meta(
  dates: "2026-10-18 16:00",
  config: "configs/experiments/slot-order.yaml",
  wandb: (("slot-order-main-f29f8c33", "slot-order, eval only"), ("ppo-4p-main-6349dc93", "ppo-4p/main (no local mirror)"), ("ppo-mixed-main-f4b771c0", "ppo-mixed/main")),
  records: ("0903fae0", "32aab9f3", "b07e5bd9"),
)
#tldr[With two F at a 4p table, the slot right after an F loses about 12 points: identical copies of `ppo-4p` split 40.45 / 52.75. Over both slot orders the 4p specialist beats `ppo-mixed` by 2.7 points, not 14.]

*Question.* How much of `ppo-mixed`'s first 4p head-to-head deficit is seating?

*Setup.* Slot 0 moves right after the F in slot 3, which robs and blocks it (@sec:front; diagnosed in @sec:hidden). `slot-order` plays the swapped lineup on the same 1,000 boards × 4 rotations (4,000 games, one minute); averaging each agent over both orders pairs them game by game.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Lineup], [Slot 0], [Slot 1], [Record]),
    [Identical copies \ #sub[`ppo-4p`, `ppo-4p`]], [40.45], [52.75], [`b07e5bd9`],
    [First reported \ #sub[`ppo-mixed`, `ppo-4p`]], [39.3 #ci[37.8][40.9]], [53.0], [`32aab9f3`],
    [Swapped \ #sub[`ppo-4p`, `ppo-mixed`]], [42.65], [50.85], [`0903fae0`],
    mid,
    table.cell(colspan: 4)[Over both orders: `ppo-mixed` 45.1, `ppo-4p` 47.8; paired −2.7 pp #ci[−4.7][−0.8] per game],
  ),
  caption: [4p shares with the learners in slots 0 and 1 and F in slots 2 and 3; 4,000 games a record.],
)

*Reading.* Every later 4p head to head runs both slot orders or sums the two learner seats.

#repro(```sh
# The swapped order (0903fae0)
catanrl run configs/experiments/slot-order.yaml
# The identical-copies control (b07e5bd9, a ppo-4p final) and the first-reported order (32aab9f3, a ppo-mixed final)
catanrl eval "POLICY:ckpt=ppo-4p/main@e3650" "POLICY:ckpt=ppo-4p/main@e3650" F F --suite heldout-4p
catanrl eval "POLICY:ckpt=ppo-mixed/main@e3550" "POLICY:ckpt=ppo-4p/main@e3650" F F --suite heldout-4p
```)

== Adding blob maps

=== `ppo-mixed-blob`: the generalist trained on blob maps too #chip("done") <exp:gen-blob>
#meta(
  dates: "2026-10-20 14:51 – 2026-10-21 15:02",
  owner: "marco",
  config: "configs/experiments/ppo-mixed-blob.yaml",
  wandb: (("ppo-mixed-blob-main-14c55ba5", "ppo-mixed-blob/main"),),
  records: ("75cab037", "799eac40", "ac020b20", "fb5a66f8", "7c17d0ae"),
)
#tldr[Blob maps as a seventh scenario made the generalist worse everywhere, blobs included (67.4 against `ppo-mixed`'s zero-shot 73.9). Three things changed at once, none isolated, so `ppo-mixed` stays the island bot.]

*Question.* Does training on blob maps beat zero-shot?

*Setup.* `ppo-mixed` plus `random-4p` at equal weight.
- *Same steps.* 512 games in flight over 7,600 epochs instead of 1,024 over 3,800; same 32,768-row minibatch.
- *Critic.* The lambda default (`ppo-mixed` used the older V-trace critic; @sec:recipe).
- *Padding.* Blobs set the dims (30 tiles, 122 nodes, 151 edges, 611 actions vs 19 / 54 / 72 / 352), so six games in seven are 19-tile boards run at blob size.
- *Cost.* Compute-bound, not memory-bound (5.5 GiB peak): 11.5k agent steps/s vs `ppo-mixed`'s 24.7k; 24.2 h.
- *Checkpoint.* `e7600`, the last, by `select-mixed` (78.8); still climbing (76.8 at e7300).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    header: ([Setting (vs F)], [Games], [`ppo-mixed-blob`], [`ppo-mixed`], [Records \ #sub[blob / mixed]]),
    [Blob maps (`heldout-random-4p`) \ #sub[zero-shot for `ppo-mixed`]], [4,000], [67.4 #ci[65.9][68.8]], [73.9 #ci[72.5][75.3]], [`75cab037` / `7c17d0ae`],
    [2p (`heldout-2p`)], [5,000], [90.1 #ci[89.3][90.9]], [92.3 #ci[91.6][93.0]], [`799eac40` / `b2690265`],
    [4p (`heldout-4p`)], [4,000], [70.6 #ci[69.2][72.0]], [77.0 #ci[75.7][78.3]], [`ac020b20` / `0f448f68`],
    [The mixed suite (`heldout-mixed`)], [4,500], [75.0 #ci[73.7][76.2]], [79.8 #ci[78.6][80.9]], [`fb5a66f8` / `9b470a17`],
  ),
  caption: [`ppo-mixed-blob@e7600` and `ppo-mixed@e3550` against F, same suites and lineups.],
)

*Reading.* A 2–7 point drop in every setting, the added one included, is not the blob scenario taking a seventh of the decisions instead of a sixth; something changed for all of them. Three things did, none tested alone: the lambda critic, half the games in flight, and blob-size padding. The deploy gate failed.

#repro(```sh
catanrl run configs/experiments/ppo-mixed-blob.yaml
```)

== Next

=== Closing the gap to the specialists #chip("todo") <exp:gen-followups>
#meta(owner: "assistant")
#tldr[Give the generalist the specialists' per-setting budget or a per-setting fine-tune, and isolate which `ppo-mixed-blob` change cost 2–7 points. (Claude's suggestion, not queued)]

*Plan.* Either 3× the decisions (the specialists' per-setting budget, about 35 h) or a per-setting fine-tune of a few hundred epochs; if the gap closes, ship the generalist. Separately, rerun `ppo-mixed-blob` with one change at a time (@exp:gen-blob).

=== A player-agnostic network: seats as a set #chip("todo") <exp:gen-player-set>
#meta(
  dates: "2026-11-16",
  dates-label: "built",
  owner: "marco",
  config: "configs/models/graph-seat-s.yaml (wt/player-set, unmerged)",
  wandb: (),
  records: (),
)
#tldr[Marco (2026-11-16): the graph network is board-agnostic; make it player-agnostic the same way for the hands and the game state. Today each seat slot has weights of its own. Marco approved the shared weights and the seats joined to the board, left attention among seats to Claude, and asked that the turn order enter the pooling. Built as `graph-seat-s` (0.58M parameters, `graph-s` 0.54M) in `wt/player-set`; it trains at 0.81× `graph-s`'s speed on an L40S. No training run yet; the first is @exp:gen-base.]

#lead[How seats enter the network today.]
- *Slots.* Six fixed seat slots, ordered from the acting seat. A 2p game leaves four of them empty.
- *Hands and state.* The 46 features of all six slots are concatenated (276 numbers) into one linear layer that makes the global vector: separate weights for each slot.
- *Board.* Ownership is six channels per node (settlement, city, distance), per edge (road) and per tile (buildings): again separate weights for each slot.
- *Shared part.* A per-seat embedding with shared weights exists, but only the value head and the robber's victim head read it; it never enters message passing.
- *Cost of that.* What the net learns about the opponent in slot 1 does not carry to slots 2 and 3, which 2p games never fill. `ppo-mixed` had a third of a specialist's decisions per seat count (2p 78M, 3p 72M, 4p 99M of 249M; the specialists 250M and 249M), and the slots do not pool them.

#lead[The design.]
- *Seat tokens.* One embedding per present seat from shared weights. The slot number is replaced by features that mean the same at any table: "this is me", how many turns after me the seat plays, and how many turns after it I play. In 2p the opponent is both the next and the previous seat; in 4p those are two seats.
- *Seats in the graph.* Seats become a fourth entity type, joined to the nodes they built on and the edges they hold a road on, with shared message weights; the board's ownership channels shrink to "mine" and "another seat's". The incidence can be read from today's ownership channels, so the engine's rows need not change.
- *Seat to seat.* Self-attention over the 2 to 6 seat tokens in each layer (Claude's call; a switch turns it off for a control), with a learned bias by the turn order between the two seats, so that "the seat after the leader" can be read.
- *Pooling and order.* The global vector reads the own seat's token and a pooled summary of the others in place of the concatenation. The turn order enters three ways: in each token before pooling, in the attention, and by keeping the own seat out of the pool (Marco: the pooling should include the order).
- *Heads.* The value head (one output per seat token) and the robber's victim head already have this shape.
- *Kept.* The seat count stays an input: 2p is zero-sum between two seats and 4p is not, so play must still differ by table size.

#lead[Cost and test.]
- *From scratch.* The input layers change, so `ppo-mixed`'s weights do not carry over. The fused kernels do not cover it (PPO does not use them).
- *Test.* `ppo-mixed`'s recipe and budget (249M decisions, 11.5 h; more if the net is slower, not measured), then `ppo-mixed`'s finals plus the two head to head.
- *Extends to.* 5–6 players with no new weights (@exp:gen-5-6p).

#lead[As built (2026-11-16).] Model class `seat`, preset `graph-seat-s`; the engine's rows are unchanged, so the old networks and checkpoints are untouched.
- *Seat tokens.* 48 wide, from one projection of a seat's 46 features and 11 turn-order features: "this is me", the forward distance and the backward distance, each one-hot. Empty slots are zero and stay zero.
- *Seats in the graph.* Joined to nodes by settlement, by city and by road distance in three bands (on the seat's network, one road away, two to five away), to edges by road, and to tiles by buildings. A board entity receives the sum of its seats' messages and a seat the mean of its entities'.
- *Seat to seat.* Attention in each of the three layers, 24 wide with 4 heads, with learned biases by forward and by backward turn distance (zero at the start). With the switch off, a token reads the mean of the others.
- *Pooling.* The global vector reads the plain globals, the own token, and the mean and the maximum of the other tokens. The maximum is the builder's addition, so that one strong opponent is not averaged away.
- *Heads.* `graph-s`'s, on the final seat tokens.
- *Dice input* (@exp:dice-input). Supported, and zero at the start as in `graph-s`. A tile reads its odds on the own seat's roll and the mean over the other seats, not each seat's.
- *Narrower than `graph-s` in one place.* A seat's own hand enters through its 48-wide token, where `graph-s` reads it into the 64-wide global vector with weights per slot.

#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    header: ([], [`graph-s`], [`graph-seat-s`]),
    [Parameters, shared trunk], [541,540], [579,844 (+7.1%)],
    [Parameters, separate trunks], [843,652], [920,308 (+9.1%)],
    [CPU, time per forward row], [1], [1.19 to 1.25×],
    [CPU, time per update row], [1], [1.33 to 1.38×],
    [L40S, agent steps a second in training], [60,786], [49,328 (0.81×)],
    [L40S, peak GPU memory], [5.3 GiB], [5.5 GiB],
  ),
  caption: [The L40S rows are six-minute stretches on Modal with compiled float32 networks: `ppo-mixed`'s config against @exp:gen-base's, which also differs in its 2p rules and has the dice input. The first, compiling, epoch took 183 s.],
) <tab:gen-player-set>

- *Tests (CPU).* 100 cases for the class, and the whole suite (1,161 passed, golden tests included). Noise in empty slots changes no output bit, at every table size from 2 to 6. With the turn order neutralised, swapping two opponents swaps their outputs and nothing else; with it back, the outputs differ. PPO on mixed 2p to 4p games and AlphaZero self-play both run with it.
- *Search.* Works on the plain PyTorch path. The fused kernels refused it by name; they are built since (@exp:gen-seat-kernels).
- *Not measured.* Playing strength (no training run yet), speed on this box's card (the estimate from the L40S ratio is about 19,500 agent steps a second), float16 on a GPU, and the 5p and 6p games beyond the unit tests.

=== A generalist base, then AlphaZero specialists #chip("todo") <exp:gen-base>
#meta(
  dates: "2026-11-16",
  dates-label: "planned",
  owner: "marco",
  config: "configs/experiments/gen-base.yaml (wt/player-set)",
  wandb: (),
  records: (),
)
#tldr[Marco's plan (2026-11-16), for after @exp:az-bundle: train the player-agnostic network (@exp:gen-player-set) from scratch on 2p, 3p and 4p games into a strong, robust policy and critic, then continue it with AlphaZero self-play per setting into the specialists deployed on OpenHex. To run on Modal beside the local queue (@exp:recipe-modal). Marco confirmed the games and the dice input, kept the small network for speed, and set the Modal budget at the month's \$30 of free credit, then added \$25. The first stage is done (@exp:gen-base-1): level with best-4p at 4p, above `ppo-mixed` at 3p, well below best-2p at 2p. The second stage is done too (@exp:gen-base-2): 996M more steps added 2 to 3.5 pp. Fused kernels for the network are built (@exp:gen-seat-kernels). No specialist is queued: Marco (2026-11-16) is hesitant to start AlphaZero training from a base this far behind the 1v1 specialist.]

#lead[Plan.]
- *Base.* PPO self-play from scratch with `graph-seat-s` on 2p, 3p and 4p games at the `private` level, the only fair one past two players (as `ppo-mixed`).
- *Games (confirmed by Marco, 2026-11-16: "everything openhex rules, with dice input").* The 2p games on OpenHex's 1v1 rules (`base-2p-fr`: 15 VP, discard at 10, friendly robber, balanced dice); 3p and 4p on base rules, which are OpenHex's 4p rules; each on the BASE board and on shuffled hexagons, as in `ppo-mixed`.
- *First stage.* `ppo-mixed`'s recipe and budget, 996M agent steps (249M decisions), so the two compare at equal training. Besides the network and the 2p rules, two things differ: the dice input is on (@exp:dice-input; under random dice it reads as a fresh deck), and the critic is the undiscounted default where `ppo-mixed` pinned the discounted one.
- *Time and cost.* 49,328 agent steps a second on an L40S with 8 cores (@tab:gen-player-set): 5.6 h, about \$15 with the finals. On this box about 14 h by the same ratio (not measured).
- *Size.* Marco asked whether to move to `graph-m`'s size for the wider mix. Kept small: `graph-m` trains at 18,470 steps a second on an L40S against `graph-s`'s 60,786 (@tab:az-hardware), so a medium seat network would take about 18 h and \$48 for the same steps, more than the budget; and the shared seat weights are this run's answer to the wider mix. A larger one follows if this one ends below `ppo-mixed` or the specialists.
- *Not tuned for the rented card.* Full precision and gradient chunks of 4,096 rows as on this box, so the run compares with `ppo-mixed`. Half precision cost 2.3 pp at equal epochs in @exp:fp16; bfloat16 and larger chunks are not measured.
- *Before the launch.* A two-epoch run on the L40S played the three finals at 50 boards against the shipped opponents' checkpoints.
- *Evals.* None in the run. Three finals with the last checkpoint's raw policy, one per table size: 2p against best-2p made fair (2,000 games, ±2.2 pp at 50%), 3p against two copies of `ppo-mixed` (3,000 games, ±1.7 pp at 33.3%), 4p against three copies of best-4p (4,000 games, ±1.3 pp at 25%).
- *Later stages.* Continue while it gains. fr3 had about 1B decisions of 2p alone; that much for each seat count, with no transfer between them, would be 12 times the first stage. How much the shared weights save is the question.
- *Specialists.* AlphaZero self-play from the base for each setting, as @exp:az-kata. Two things differ from fr3's line: the base reads `private` rows where fr3 reads `full`, and the new network needed fused kernels of its own (unfused, `graph-m` cost 3.4× per simulation, @exp:deploy-graph-m-check); they are built (@exp:gen-seat-kernels).

=== `gen-base`: the first stage #chip("done") <exp:gen-base-1>
#meta(
  dates: "2026-11-16 03:48 – 09:47",
  owner: "marco",
  config: "configs/experiments/gen-base.yaml (wt/player-set, 4f01d31)",
  wandb: (("gen-base-main-96725ccd", "gen-base/main"),),
  records: ("47f566cf", "2bd4e3a9", "a8b0f0ee", "dd5e7fb8", "5c8dcfcb", "be7738c2", "31f7ab56", "f7fb1923", "b17a6d3e"),
)
#tldr[`graph-seat-s` from scratch for 996M agent steps (249M decisions) on a Modal L40S: 5.97 h, \$15.43. At 4p it is level with best-4p, which had 2.5 times its 4p decisions. At 3p it beats `ppo-mixed` at equal training (36.2% against two copies, parity 33.3%, p = 0.001). At 2p it takes 36.2% from best-2p made fair, where fr3 took 45.0% with 12 times its 2p decisions. It was still gaining: the halfway checkpoint is 6 to 13 pp lower on the same games, and the last 236M steps, near the floor learning rate, added 4 pp at 2p, 3 pp at 3p and 1.3 pp at 4p (the last not significant). Registered as `mix-graph-seat-s-1`.]

*Question.* How strong is the player-agnostic network after `ppo-mixed`'s budget, at each table size?

*Setup.* @exp:gen-base's first stage as planned. Its raw policy at the last checkpoint (e3800) plays one seat; the other seats are copies of one opponent; every board is played once from each seat.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, 1.3fr, auto, auto, auto, auto),
    align: (left, left, right, right, right, left),
    header: ([Table], [Other seats], [Games], [`gen-base` wins (%)], [Parity], [Record]),
    [2p, OpenHex 1v1 rules \ #sub[`heldout-2p-fr`, 1,000 boards]], [best-2p made fair \ #sub[`BELIEF` of `graph-m`, 16 worlds]], [2,000], [36.2 #ci[34.1][38.3]], [50], [`47f566cf`],
    [3p \ #sub[`heldout-3p`, 1,000 boards]], [2 × `ppo-mixed`], [3,000], [36.2 #ci[34.5][37.9]], [33.3], [`2bd4e3a9`],
    [4p \ #sub[`heldout-4p`, 1,000 boards]], [3 × best-4p (`ppo-4p`)], [4,000], [25.45 #ci[24.1][26.8]], [25], [`a8b0f0ee`],
  ),
  caption: [`gen-base/main@e3800`'s share of wins by table size. Its decisions in training: 80M at 2p, 70M at 3p, 99M at 4p.],
) <tab:gen-base-1>

#lead[Reading.]
- *4p.* Level with the 4p specialist (1,018 wins of 4,000; the three copies 1,014, 1,009 and 959; p = 0.51), on 99M 4p decisions against its 249M. `ppo-mixed`, with the same share of 4p games, was 2.7 pp below the specialist in @exp:gen-slot-order; that table had two learners and two F, so the two rows do not pair.
- *3p.* Above `ppo-mixed` at equal training: 1,085 wins of 3,000 against 1,025 and 890 for the two copies (p = 0.001 against a third).
- *2p.* Well below best-2p (`graph-m`, 1.5B decisions of 2p). fr3 (1.0B decisions of 2p) took 45.0% #ci[43.6][46.4] from the same opponent over the suite's 5,000 games (`71af32b1`, @exp:deploy-graph-m-check; not paired with this row). `gen-base` had 80M decisions of 2p, and plays from `private` rows where the other two are `full`-level networks averaged over 16 sampled worlds.
- *Not separated.* The network, the dice input, the undiscounted critic and the 2p rules changed together from `ppo-mixed` (bundle first, Marco), so the 3p and 4p gains belong to the bundle.
- *Critic.* On its own self-play games in the last 100 epochs, the critic's squared error against the outcome is 0.136 at 2p, 0.176 at 3p and 0.159 at 4p (slot 0); `ppo-mixed`'s on its own games was 0.097, 0.142 and 0.122. The games and players differ, so this does not rank the critics; scoring both on the same games would (not run).
- *The run.* One container, no preemption. 47,600 agent steps a second (median; 49,300 in the six-minute check), 5.97 h billed with the compile and the finals (4 min), \$15.43. The app stopped itself after the finals; the run and its 39 checkpoints are on the volume and in `runs/gen-base/main` here.

#lead[Slope check (Claude's suggestion; Marco, 2026-11-16: "lets do slope check").]
Was it still gaining at the end? Two earlier checkpoints played the same three tables: the same boards, seats and opponents, so the rows pair game by game with e3800's. e1900 is the halfway point (`gen-base-slope`). e2900 (`gen-base-slope-e2900`) is 236M agent steps before the end, when the learning rate was 0.0011, about twice the floor; a continuation would train near that rate. 4.5 min on an L40S each, \$0.42 with the builds.
#figure(
  btable(
    columns: (1.2fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    header: ([Table], [Games], [e1900 (%)], [e2900 (%)], [e3800 (%)], [e3800 − e1900 (pp)], [e3800 − e2900 (pp)]),
    [2p against best-2p made fair], [2,000], [23.6], [32.15], [36.2], [+12.6 #ci[+10.2][+15.0]], [+4.05 #ci[+1.6][+6.5]],
    [3p against 2 × `ppo-mixed`], [3,000], [29.2], [33.1], [36.2], [+7.0 #ci[+4.9][+9.1]], [+3.1 #ci[+0.9][+5.2]],
    [4p against 3 × best-4p], [4,000], [19.2], [24.15], [25.45], [+6.3 #ci[+4.6][+8.0]], [+1.3 #ci[−0.4][+3.0]],
    mid,
    [Agent steps], [], [498M], [760M], [996M], [], [],
    [Learning rate], [], [0.00275], [0.0011], [0.0005], [], [],
  ),
  caption: [`gen-base`'s checkpoints on the same games; differences are paired. Games won only by e3800 against only by the earlier checkpoint, e1900: 453 to 201 at 2p (p = 3e-23), 652 to 442 at 3p (p = 2e-10), 728 to 477 at 4p (p = 5e-13); e2900: 360 to 279 (p = 0.002), 578 to 486 (p = 0.005), 632 to 580 (p = 0.14). Records, e1900: `dd5e7fb8`, `5c8dcfcb`, `be7738c2`; e2900: `31f7ab56`, `f7fb1923`, `b17a6d3e`.],
) <tab:gen-base-slope>
- *Reading.* It gained through the whole second half, most at 2p. At the halfway point it was below `ppo-mixed` at 3p and below best-4p at 4p; it passed `ppo-mixed` and drew level with best-4p in the second half. The gain is slowing: over the 262M steps to e2900 and then the 236M to the end, 8.6 then 4.1 pp at 2p, 3.9 then 3.1 pp at 3p, 5.0 then 1.3 pp at 4p.
- *Caveat.* Slowing and the falling learning rate are not separated: the rate went from 0.00275 at e1900 to the floor, 0.0005, at the end. The two continuations of a 2p net are both poor precedents: at the floor rate, constant, `ppo-2p-fr` gained about 1 pp in 1B steps (@exp:recipe-2p-openhex-3), and a fresh cosine from the peak went backwards (@exp:recipe-2p-openhex-2). Both started from a net 1.4B steps old; this one still gained near the floor rate.

#repro(```sh
# from the wt/modal worktree; the finals need the three opponents' checkpoints on the volume
python tools/modal_train.py launch configs/experiments/gen-base.yaml --rev wt/player-set --gpu L40S --cpu 8 --memory 32 --max-hours 7 \
  --set runtime.threads=8 --ship graph-m-2p-lr/main@e22711 --ship ppo-mixed/main@e3550 --ship ppo-4p/main@e3650
python tools/modal_train.py watch gen-base/main --every 10   # pulls into runs/ and mirrors to W&B until the run ends
# the slope check: the records land in runs/gen-base/main/evals
# (gen-base-slope-e2900.yaml with --ship gen-base/main@e2900 for the other checkpoint)
python tools/modal_train.py launch configs/experiments/gen-base-slope.yaml --rev wt/player-set --gpu L40S --cpu 8 --memory 32 --max-hours 0.25 \
  --set runtime.threads=8 --ship gen-base/main@e1900 --ship graph-m-2p-lr/main@e22711 --ship ppo-mixed/main@e3550 --ship ppo-4p/main@e3650
python tools/modal_train.py pull gen-base/main
```)

=== `gen-base-2`: the second stage #chip("done") <exp:gen-base-2>
#meta(
  dates: "2026-11-16 11:34 – 17:28",
  owner: "marco",
  config: "configs/experiments/gen-base-2.yaml (wt/player-set, 477f2cf)",
  wandb: (("gen-base-2-main-3956bb68", "gen-base-2/main"),),
  records: ("470a83b7", "9e437636", "be447abf"),
)
#tldr[`gen-base`'s last checkpoint trained for 996M more agent steps on the same games, at its floor learning rate held constant: 5.88 h on a Modal L40S, \$15.19. Doubling the training added 2 to 3.5 pp: +3.4 pp at 3p and +2.4 pp at 4p (both significant), +2.2 pp at 2p (not significant). It is now above best-4p at 4p (27.85%, parity 25%) and takes 38.4% from best-2p made fair, where fr3 took 45.0%. The gain per step is a quarter or less of the first stage's last stretch at 2p and 3p.]

*Question.* How much stronger does the base get with as many steps again?

*Setup.* Marco (2026-11-16, after the slope checks of @exp:gen-base-1): "you can go over the budget. im approving another 25 bucks total."
- *Start.* The weights of `gen-base/main@e3800` (`mix-graph-seat-s-1`), actor and critic, with a fresh optimizer.
- *Learning rate.* 0.0005, `gen-base`'s floor, constant. It is the rate that last gave a measured gain (@tab:gen-base-slope); a fresh cosine from the peak took `ppo-2p-fr` backwards (@exp:recipe-2p-openhex-2).
- *Everything else.* `gen-base`'s: games, maps (the same seed), network, optimizer settings, 3,800 epochs of 262,144 agent steps.
- *Evals.* None in the run. The three finals are `gen-base`'s, on the same boards, seats and opponents, so each row pairs game by game with e3800's (2,000, 3,000 and 4,000 games; paired differences of about 2.5, 2 and 1.7 pp are detectable). Checkpoints every 100 epochs.
- *Check before launch.* Four small epochs on the CPU from e3800: entropy 0.135 and explained variance 0.87 at the start (`gen-base` ended at 0.129 and 0.89), learning rate 0.0005.
- *Cost.* Capped at 6.4 h, \$16.54.
- *First epochs.* 48,800 agent steps a second, `gen-base`'s speed. Epoch 1 on the real games: entropy 0.140, explained variance 0.90, learning rate 0.0005.

*Result.* The last checkpoint's raw policy (e3800 of this run, 1.99B agent steps in all) against `gen-base`'s e3800 on the same games.
#figure(
  btable(
    columns: (1.3fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([Table], [Games], [Parity (%)], [First stage (%)], [Second stage (%)], [Paired difference (pp)]),
    [2p against best-2p made fair], [2,000], [50], [36.2], [38.4 #ci[36.3][40.5]], [+2.2 #ci[−0.4][+4.8]],
    [3p against 2 × `ppo-mixed`], [3,000], [33.3], [36.2], [39.6 #ci[37.8][41.4]], [+3.4 #ci[+1.2][+5.6]],
    [4p against 3 × best-4p], [4,000], [25], [25.45], [27.85 #ci[26.5][29.2]], [+2.4 #ci[+0.6][+4.2]],
  ),
  caption: [`gen-base-2/main@e3800` and `gen-base/main@e3800` in the same seat on the same boards against the same opponents. Games won only by the second stage against only by the first: 380 to 336 at 2p (p = 0.11), 616 to 513 at 3p (p = 0.002), 711 to 615 at 4p (p = 0.009). Records: `470a83b7`, `9e437636`, `be447abf`.],
) <tab:gen-base-2>
- *4p.* Above best-4p: 1,114 of 4,000 against three copies (p = 4e-5 against parity; the copies took 940, 968 and 978). The base has had 198M 4p decisions, best-4p 249M.
- *3p.* 39.6% against two copies of `ppo-mixed` (the copies took 964 and 848).
- *2p.* 38.4% against best-2p made fair; the gain over the first stage is inside the noise. fr3 took 45.0% from the same opponent (@exp:gen-base-1) with about 1.0B 2p decisions; the base has had 160M.
- *Gain per step.* The first stage's last 236M steps added 4.05, 3.1 and 1.3 pp (@tab:gen-base-slope); these 996M added 2.2, 3.4 and 2.4 pp. Per step that is about an eighth at 2p, a quarter at 3p and under a half at 4p, with wide intervals on both sides. Two things differ and are not separated: the net is further along, and the rate here was 0.0005 throughout where the last stretch of the first stage fell from 0.0011 to 0.0005.
- *Critic.* Its Brier score on its own self-play did not move: 0.136, 0.173 and 0.156 at 2p, 3p and 4p over the last 100 epochs (first stage: 0.136, 0.176, 0.159). Entropy fell from 0.129 to 0.108.
- *The run.* One container, no preemption. 5.88 h billed with the compile and the finals, \$15.19; 47,500 agent steps a second over the wall clock. The run and its 39 checkpoints are on the Volume and in `runs/gen-base-2/main` here. Registered as `mix-graph-seat-s-2`.

#repro(```sh
# from the wt/modal worktree
python tools/modal_train.py launch configs/experiments/gen-base-2.yaml --rev wt/player-set --gpu L40S --cpu 8 --memory 32 --max-hours 6.4 \
  --set runtime.threads=8 --ship gen-base/main@e3800 --ship graph-m-2p-lr/main@e22711 --ship ppo-mixed/main@e3550 --ship ppo-4p/main@e3650
python tools/modal_train.py watch gen-base-2/main --every 10
```)

=== Fused kernels for the seat network #chip("done") <exp:gen-seat-kernels>
#meta(
  dates: "2026-11-16",
  owner: "marco",
  config: "wt/seat-kernels (14b06b7, 3de015e), unmerged",
)
#tldr[`kernels: fused` and `fused16` now run `graph-seat-s`: 2.0 to 2.2 times the PyTorch path's rows per second, and 2.9 to 3.4 times in half precision, 7 to 16% under `graph-s` on its kernels. On `mix-graph-seat-s-2` the outputs match the module within the documented tolerance at 2p, 3p and 4p. Tiny AlphaZero runs from the checkpoint work at 2p and 4p. Marco is hesitant to train specialists from this base, so none is queued.]

*Question.* Can search run the base as fast as it runs `graph-s`, and what else does an AlphaZero run from the base need?

*Setup.* Marco (2026-11-16): "yeah go for it." Built by a subagent on `wt/seat-kernels`, off `wt/player-set`.
- *Design.* A second extension (`seat_kernels.cu`, five kernels: the ownership relations, the seat tokens, their residual and LayerNorm, the seats' messages into the board, the board's messages per seat with the seat-to-seat attention). The board trunk, pooling and heads run on the graph kernels, whose source is unchanged: 90 arrays of graph-class outputs are bit-identical before and after.
- *Refused, with a message.* A separate trunk, an aux head, a hidden width other than 96 (so `graph-seat-tiny`), the CPU.
- *Tests.* 142 passed on the GPU (81 new), 376 on the CPU, none failed.

*Result.*
#figure(
  btable(
    columns: (1.2fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([Network and rows], [Rows per forward], [PyTorch], [`fused`], [`fused16`], [`fused16` / PyTorch]),
    [`graph-seat-s`, 2p], [64], [12,200], [27,100], [36,800], [3.0×],
    [`graph-seat-s`, 4p], [64], [11,900], [26,500], [33,900], [2.9×],
    [`graph-s`, 2p], [64], [16,600], [30,100], [42,400], [2.6×],
    mid,
    [`graph-seat-s`, 2p], [512], [18,600], [39,100], [62,800], [3.4×],
    [`graph-seat-s`, 4p], [512], [18,900], [37,600], [55,600], [2.9×],
    [`graph-s`, 2p], [512], [23,600], [44,000], [68,100], [2.9×],
  ),
  caption: [Rows per second through the evaluator on the RTX 2070 Super, CUDA graphs, median of 9 interleaved rounds, taken beside the bundle's final eval (so the ratios are the fair part). `graph-seat-s` with `mix-graph-seat-s-2`'s weights; `graph-s` with random weights at the private level with dice.],
) <tab:gen-seat-kernels>
- *Accuracy on `gen-base-2/main@e3800`.* 256 self-play rows per game: `fused` within 3.1e-4 of the module's logits (scale 520 to 714) and 6.6e-7 of its values at 2p, 3p, 4p and hexagon 3p; `fused16` within 0.30 and 2.7e-4 of `fused`, with no fallback to float.
- *AlphaZero from the checkpoint.* Three tiny GPU runs finished (3 iterations, 16 sims): 2p with the pruned targets and the outcome head, 4p with a Gumbel root, 4p with PUCT. `MCTS` and `BELIEF` lineups played 2p, 3p and 4p games in the eval harness. Plumbing only: no rates.
- *What a real run must set.* `obs.level: private` (a wrong level loads silently); `dets: 1` or more, since `dets: 0` shows the tree the hidden hands the network does not see, and then tree reuse is off; its own `eval.select` and `eval.final` (a mixed-size run has no default suites, and no 3p suite exists).
- *A warning.* The base's policy is very peaked: on its own games 75 to 79% of decisions with a choice have a top prior above 0.99. In the smokes a 64-sim search played the raw policy's game move for move. A sized search-against-raw check comes before any long run (not queued).
- *Not measured.* Self-play speed at real settings; the cost of 4p search evals; resuming such a run.
- *Merge.* After the code freeze: the branch changes hashed code (`models/`, `search/`), so eval code ids move though the graph-class outputs do not.

=== 5–6 players on the extension map #chip("todo") <exp:gen-5-6p>
#meta(owner: "marco")
#tldr[Blocked on the engine: 5–6 player games need the special building phase; then they are one more entry in `ppo-mixed`'s `env.scenarios`.]
