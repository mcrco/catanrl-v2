#import "../macros.typ": *

= One model for any number of players and any map <sec:generalist>

Marco's idea: train one network for any player count and any map, using masked numeric features and the graph network, which already handles any hex board. Can such a generalist match the specialists? So far `ppo-mixed` comes within 2–3 points of each specialist against F and plays unseen blob maps nearly as well as BASE, but the specialists still win head to head: the generalist scores 45.7% against the 2p specialist and trails the 4p specialist by 2.7 points. Training on blob maps too made it worse everywhere, so `ppo-mixed` is the bot on the island table (@sec:deploy).

== One network over 2–4 players

=== `ppo-mixed`: 2, 3 and 4 players on BASE and hexagon boards #chip("done") <exp:gen-ppo-mixed>
#meta(
  dates: "2026-10-17 02:14–13:54",
  owner: "marco",
  config: "configs/experiments/ppo-mixed.yaml",
  wandb: (("ppo-mixed-main-f4b771c0", "ppo-mixed/main"),),
  records: ("b2690265", "0f448f68", "7c17d0ae", "9b470a17", "340ce1db", "32aab9f3"),
)
#tldr[One network for 2, 3 and 4 players on two board types comes within 2–3 points of each specialist against F, and plays blob maps it never saw nearly as well as BASE (73.9 vs 77.0). Head to head the specialists still win: the generalist scores 45.7% in 2p and trails by 2.7 points in 4p.]

*Question.* Can one network match the specialists at the same total number of decisions?

#lead[Setup.]
- *Scenarios.* `ppo-4p`'s recipe (@sec:recipe) over six equally weighted scenarios: 2, 3 and 4 players on BASE and on `hexagon` boards. `env.scenarios` draws each game from a weighted list of (player count, rules, map generator).
- *Masking.* The harness already does it: each row is padded to the env's maxima with a validity bit per slot, absent seats are all zero, the global block one-hots the number of seats, and the per-seat value head predicts every present seat.
- *Budget.* `ppo-4p`'s: 3,800 epochs, 1B agent steps = 249M decisions, 11.5 h. Best checkpoint `e3550` by `select-mixed` (83.6). Launched without the W&B mirror and backfilled.
- *Blob maps.* Blob-shaped boards of 15–30 tiles (`random-4p`) stay out of training: they would double the row (6,316 floats vs 3,202) and the memory. The run is tested on them zero-shot (`heldout-random-4p`).

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
    [2p, vs `puffer-graph2`'s belief-averaged policy], [5,000], [45.7 #ci[44.4][47.1]], [], [`340ce1db`],
    [4p, vs `ppo-4p` with two F \ #sub[both slot orders (@exp:gen-slot-order)]], [2 × 4,000], [45.1], [47.8], [`32aab9f3` \ `0903fae0`],
  ),
  caption: [`ppo-mixed@e3550` against F and against the specialists. Scores are score shares (a win counts 1, a truncation 1/N). The 4p head to head is the corrected one: paired −2.7 pp #ci[−4.7][−0.8] per game for the generalist.],
)

*Reading.* One network handles any seat count and board shape within 2–3 points of a specialist against F, and the graph net transfers to board shapes it never trained on nearly for free (73.9 on blobs vs 77.0 on BASE). But each setting got a third of a specialist's decisions, and the specialists win the direct matches: this is a budget question as much as a capacity one. The next step is more decisions per setting (@exp:gen-followups).

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
#tldr[In a 4p table with two F, the slot right after an F loses about 12 points: two identical copies of `ppo-4p` split 40.45 / 52.75. So every 4p head to head has to run both slot orders. Corrected that way, the 4p specialist beats `ppo-mixed` by 2.7 points, not 14.]

*Question.* The first 4p head to head had `ppo-mixed` at 39.3 against `ppo-4p`'s 53.0; how much of that gap is the seating?

*Setup.* Slot 0 always moves right after the F in slot 3, which robs and blocks it (see "The slot after an F" in @sec:front; the diagnosis came from the tracker-modes table, @sec:hidden). `slot-order` plays the swapped order [`ppo-4p`, `ppo-mixed`, F, F] on the same 1,000 boards × 4 rotations (4,000 games, one minute). Averaging each agent over the two orders pairs them game by game.

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
  caption: [4p head-to-head shares with the two learners in slots 0 and 1 and F in slots 2 and 3; 4,000 games a record.],
)

*Reading.* The 4p specialist does win, by three points rather than fourteen. Every later 4p head to head runs both slot orders or reports the two learner seats summed.

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
#tldr[Adding blob maps as a seventh scenario made the generalist worse everywhere, blobs included: 67.4 against `ppo-mixed`'s zero-shot 73.9. Three things changed at once and none was isolated, so `ppo-mixed` stays the island bot.]

*Question.* Zero-shot, the architecture copes with blob maps; does training on them beat `ppo-mixed`'s 73.9 on `heldout-random-4p`?

*Setup.* `ppo-mixed` with `random-4p` as a seventh scenario of equal weight.
- *Same steps.* 512 games in flight over 7,600 epochs instead of 1,024 over 3,800, with the same 32,768-row minibatch, so agent and optimizer steps match.
- *Critic.* The lambda default (`ppo-mixed` trained with the older V-trace critic; @sec:recipe).
- *Padding.* The blobs set the dims (30 tiles, 122 nodes, 151 edges, 611 actions, against 19 / 54 / 72 / 352), so every row is padded to them, and six games in seven are 19-tile boards run at blob size.
- *Cost.* Compute is the limit, not memory (5.5 GiB peak): 11.5k agent steps/s against `ppo-mixed`'s 24.7k; 24.2 h in all.
- *Checkpoint.* `e7600`, the last, by `select-mixed` (78.8); still climbing at the end (76.8 at e7300).

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
  caption: [`ppo-mixed-blob@e7600` against `ppo-mixed@e3550`, on the same suites and lineups.],
)

*Reading.* A drop of 2–7 points in every setting, the added one included, is not the seventh scenario taking a seventh of the decisions instead of a sixth: something changed for all of them. Three things did, none tested on its own: the lambda critic default, half as many games in flight per epoch, and every row padded to blob size. The deploy gate failed: for an island table `ppo-mixed` stays the stronger bot until one of those is isolated.

#repro(```sh
catanrl run configs/experiments/ppo-mixed-blob.yaml
```)

== Next

=== Closing the gap to the specialists #chip("todo") <exp:gen-followups>
#meta(owner: "assistant")
#tldr[Give the generalist the specialists' per-setting budget or a per-setting fine-tune; separately, find which of `ppo-mixed-blob`'s three changes cost the 2–7 points. (Claude's suggestion, not queued)]

*Plan.* Each setting got a third of a specialist's decisions. Give the generalist the specialists' per-setting budget (3× the decisions, about 35 h), or fine-tune it per setting for a few hundred epochs; if the gap closes, the generalist is the model to ship. Separately, rerun `ppo-mixed-blob` with one change at a time (the lambda critic, half the games in flight, padding to blob size) to find what cost the 2–7 points.

=== 5–6 players on the extension map #chip("todo") <exp:gen-5-6p>
#meta(owner: "marco")
#tldr[Not started: blocked on the engine, since 5–6 player games need the special building phase. After that, 5–6 players are one more scenario in `ppo-mixed` (one more entry in `env.scenarios`).]
