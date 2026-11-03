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

=== 5–6 players on the extension map #chip("todo") <exp:gen-5-6p>
#meta(owner: "marco")
#tldr[Blocked on the engine: 5–6 player games need the special building phase; then they are one more entry in `ppo-mixed`'s `env.scenarios`.]
