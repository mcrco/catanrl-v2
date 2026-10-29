#import "../macros.typ": *

= search at play time <sec:search>

marco's recipe has a second step: put the trained policy and critic inside a search, sample the hidden cards from the hand tracker, and get about another 5% head to head over the raw policy. this section measures that edge in 2p and 4p, and tests two search options built for the deployed bot. in 2p a PUCT search with one sampled world wins 53–55% against the raw policy, and the 2048-sim gumbel search wins 62.4%, the strongest 2p player we have. in 4p the edge is about two points of share (8% relative), and tree reuse and transpositions change nothing measurable.

*the agents.* `MCTS:ckpt=…:sims=N:dets=K` is a batched N-player PUCT search. each seat's value is backed up separately (max#super[n]), dice, dev-card draws and steals are exact chance nodes, and the root searches K worlds sampled from the hand tracker (`dets=K`). every fair row below uses `dets=1`, one sampled world per search. `oracle=true` searches the true game, sees the hidden cards, and is only a reference. `root=gumbel` swaps the PUCT root for gumbel alphazero's root over `gumbel_m` candidate moves. `BELIEF:ckpt=…:worlds=16` is the raw policy averaged over 16 sampled worlds, the fair way to play a network trained at the `full` level. `POLICY` is the raw policy's argmax.

== two players

=== MCTS on the true game vs F #chip("done") <exp:search-oracle-vs-F>
#meta(
  dates: "2026-10-14",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-full-2fdb6f44", "puffer-graph2/full"),),
  records: ("3fdae985", "609e0aa7"),
)
#tldr[an oracle search (it sees the opponent's hidden cards) scores 94.8 against F at 64 sims and 95.2 at 256, no better than the raw policy's 95.16. against F a strong policy is already at its ceiling, so every later search row is measured head to head against a raw policy.]

*question.* does search move the number against F?

*setup.* `MCTS` with `oracle=true` and `dets=1` on `puffer-graph2/full@e3350`, against F on the first 250 boards of `heldout-2p` in both seat orders: 500 games a row. these were finals of `puffer-graph2/full` under the config's first version (`4a56469`), since dropped from the yaml; the eval lines below replay them.

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, left, right, right, left),
    header: ([agent], [sees], [games], [vs F], [record]),
    [raw policy, `full` obs (reference, @sec:recipe)], [hidden cards], [5,000], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
    [MCTS 64 sims, oracle], [hidden cards], [500], [94.8 #ci[92.5][96.4]], [`3fdae985`],
    [MCTS 256 sims, oracle], [hidden cards], [500], [95.2 #ci[93.0][96.8]], [`609e0aa7`],
  ),
  caption: [oracle search with `puffer-graph2/full@e3350` against F on `heldout-2p`.],
) <tab:search-oracle>

*reading.* all three rows sit inside each other's intervals. vs F cannot see a search gain at this strength, so the search edge is measured head to head below.

#repro(```sh
# finals dropped from puffer-graph2.yaml; the records' specs with the defaults left out
catanrl eval "MCTS:ckpt=puffer-graph2/full@e3350:sims=64:dets=1:oracle=true" F \
  --suite heldout-2p --boards 250
catanrl eval "MCTS:ckpt=puffer-graph2/full@e3350:sims=256:dets=1:oracle=true" F \
  --suite heldout-2p --boards 250
```)

=== search edge over the belief-averaged policy (`full` net) #chip("done") <exp:search-edge-2p-full>
#meta(
  dates: "2026-10-17 – 2026-10-18 01:31",
  owner: "marco",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-full-2fdb6f44", "puffer-graph2/full"),),
  records: ("04fd2c25", "2aaf2eb7"),
)
#tldr[fair search (one world sampled from the tracker) beats the same network's belief-averaged policy 53.4% at 1024 sims and 55.2% at 4096. the edge is real and grows with sims, a point or two under v1’s over a stronger raw policy. it cost 11.5 gpu hours for 1,500 games.]

*question.* marco's recipe claims about 5% head to head from search. does v2 get it?

*setup.* `MCTS:sims=N:dets=1` against `BELIEF:worlds=16` of the same checkpoint, `puffer-graph2/full@e3350`, on `heldout-2p`: 500 boards in both seat orders at 1024 sims, 250 boards at 4096. these are the last two finals of `puffer-graph2/full`, played with 128 games in flight.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (right, right, right, right, left),
    header: ([sims], [games], [MCTS vs BELIEF], [v1, same pairing], [record]),
    [256], [—], [—], [51.1], [—],
    [1024], [1,000], [*53.4* #ci[50.3][56.5]], [55.1], [`04fd2c25`],
    [4096], [500], [*55.2* #ci[50.8][59.5]], [58.1], [`2aaf2eb7`],
  ),
  caption: [MCTS against the same network's `BELIEF` policy, head to head on `heldout-2p`. v1’s column is its own network against its own belief-averaged policy on 1,000 games.],
) <tab:search-edge-full>

both seats agree: at 1024 sims the searcher wins 55.0 from the first seat and 51.8 from the second.

*reading.* the edge is there and grows with sims. it is a point or two smaller than v1’s, over a stronger raw policy. the cost, 11.5 h of gpu for 1,500 games, is what parks alphazero at these sim counts (@sec:az).

#repro(```sh
# trains (8 h), then plays every final
catanrl run configs/experiments/puffer-graph2.yaml --only full
# the two search finals alone:
catanrl eval "MCTS:ckpt=puffer-graph2/full@e3350:sims=1024:dets=1" \
  "BELIEF:ckpt=puffer-graph2/full@e3350:worlds=16" --suite heldout-2p --boards 500
catanrl eval "MCTS:ckpt=puffer-graph2/full@e3350:sims=4096:dets=1" \
  "BELIEF:ckpt=puffer-graph2/full@e3350:worlds=16" --suite heldout-2p --boards 250
```)

=== search edge over the raw `public` policy #chip("done") <exp:search-edge-2p-public>
#meta(
  dates: "2026-10-19 02:07–07:10",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-public-3ec072be", "puffer-graph2/public"),),
  records: ("72819d1a", "0ef1b81e"),
)
#tldr[the same search over the fair `public` network beats its raw policy 54.9% at 1024 sims and 55.2% at 4096: the same edge the `full` network has. it took half the gpu time, because the opponent is one forward pass instead of sixteen.]

*question.* does the edge hold for the network we would serve raw?

*setup.* in 2p every resource movement is seen by both seats, so the opponent's exact hand is public, and a `public`-level policy is fair without belief averaging. `puffer-graph2/public@e3600` is the same recipe and budget as `full` at that level (@sec:recipe). `MCTS:sims=N:dets=1` against `POLICY` of the same checkpoint on `heldout-2p`, the same boards as the `full` rows, 256 games in flight.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (right, right, right, right, left),
    header: ([sims], [games], [MCTS vs raw policy], [wall], [record]),
    [1024], [1,000], [*54.9* #ci[51.8][58.0]], [1.5 h], [`72819d1a`],
    [4096], [500], [*55.2* #ci[50.8][59.5]], [3.6 h], [`0ef1b81e`],
  ),
  caption: [MCTS with `puffer-graph2/public@e3600` against its own raw policy on `heldout-2p`.],
) <tab:search-edge-public>

both seats agree. at 1024 sims the searcher wins 60.6 in seat 0 and 49.2 in seat 1, the raw policy 50.8 and 39.4: seat 0 wins about 56% of these games whoever sits there.

*reading.* the same edge as the `full` network over its belief-averaged self (53.4 and 55.2). the 1,500 games took 5.0 h against the `full` rows' 11.5 h: a one-forward opponent, and 256 games in flight instead of 128.

#repro(```sh
# trains (8 h), then plays every final
catanrl run configs/experiments/puffer-graph2.yaml --only public
# the two search finals alone:
catanrl eval "MCTS:ckpt=puffer-graph2/public@e3600:sims=1024:dets=1" \
  "POLICY:ckpt=puffer-graph2/public@e3600" --suite heldout-2p --boards 500
catanrl eval "MCTS:ckpt=puffer-graph2/public@e3600:sims=4096:dets=1" \
  "POLICY:ckpt=puffer-graph2/public@e3600" --suite heldout-2p --boards 250
```)

=== gumbel search at 2048 sims vs the raw policy #chip("done") <exp:search-gumbel-edge-2p>
#meta(
  dates: "2026-10-21 18:56–22:50",
  config: "configs/experiments/gumbel-edge-2p.yaml",
  wandb: (("gumbel-edge-2p-main-efda50ac", "gumbel-edge-2p, eval only"),),
  records: ("831eea84",),
)
#tldr[the `public` network inside a 2048-sim gumbel search wins 62.4% of 1,000 games against its own raw policy, far above PUCT over the same policy (54.9 and 55.2). it is the strongest 2p player we have, and the baseline that every later search-harness row pairs with.]

*question.* alphazero's 2048-sim targets made the raw policy worse (@exp:az-2p-2048). are the moves behind those targets good ones?

*setup.* `MCTS:ckpt=puffer-graph2/public@e3600:sims=2048:dets=1:root=gumbel:gumbel_m=16`, the fair search alphazero trained on, against `POLICY` of the same checkpoint. the first 500 boards of `heldout-2p` in both seat orders, 1,000 games.

*result.* the search wins *62.4* #ci[59.4][65.3] (`831eea84`). PUCT over the same policy wins 54.9 at 1024 sims and 55.2 at 4096 (@exp:search-edge-2p-public).

*reading.* the search had plenty to teach, and the raw policy still got worse learning it (@sec:az). these 1,000 games became the search harness, marco's rule for judging a new network: put it in this same search against the same raw start, and pair it game by game with `831eea84` (mcnemar over the shared game ids, `tools/search_probe.py --pair`). the alphazero checkpoints (@sec:az), the critic refit (@sec:critic) and canopy's search (@sec:canopy) are all judged this way.

#repro(```sh
catanrl run configs/experiments/gumbel-edge-2p.yaml
```)

=== search depth vs strength across training #chip("running") <exp:search-depth-2p>
#meta(
  dates: "2026-10-29 00:47–",
  owner: "marco",
  config: "configs/experiments/search-depth-2p.yaml",
  wandb: (("search-depth-2p-s64-b23bbbaf", "s64"), ("search-depth-2p-s128-1cd135fb", "s128"), ("search-depth-2p-s256-28bd5783", "s256"), ("search-depth-2p-s1024-2bb0e327", "s1024"), ("search-depth-2p-s4096-510e46c4", "s4096")),
  records: ("be98751d", "2672eaad", "101296e4", "13061034", "d7fc0e57", "9004b0ac", "19048e65", "ad2a43f4", "76c3fbed", "8269c07b", "032c4368", "83b507c8", "ec991f87", "1cee7f01", "49254c47", "ce9f6daf", "13ed5ff1", "25c4c060", "e26edabf", "c91e1186"),
)
#tldr[marco's question: does more search consistently make a network stronger, and does that change as the network trains? MCTS at 64 to 16384 sims against the same checkpoint's raw policy, for five checkpoints of one run from scratch, 1,000 games a cell. so far (up to 1024 sims): yes, every step up in sims helps every checkpoint, and the weaker the network, the more search adds: from 64 to 1024 sims the e50 net gains 18 points and the final net 5.5. at 64 sims the final net's search is only level with its raw policy. 4096 is running; the 16384 column costs far more than first estimated (below).]

*question.* so far we have single points: 53.4 and 55.2 at 1024 and 4096 sims over one `full` checkpoint, 54.9 and 55.2 over one `public` checkpoint (@exp:search-edge-2p-full, @exp:search-edge-2p-public). the edge barely moved between 1024 and 4096. is that a ceiling, and is it the same for a weak network as for a strong one?

*setup.*
- *network:* `discount-2p/g1-s42`, puffer-graph2's recipe from scratch on openhex's 1v1 rules (`base-2p-fr`), $gamma = 1$, obs level `full` (@exp:recipe-discount). five checkpoints about 3× apart in epochs: e50, e150, e500, e1500 and e3786, which score 47, 80, 92, 95 and 98.5% against F on the select suite.
- *searcher:* `MCTS:sims=N:dets=1:kernels=fused`, PUCT with one world sampled from the hand tracker, the fair search of every 2p row above. `kernels=fused` is the deployed search bot's network on custom cuda kernels: the same outputs to the last bits, about 1.7× faster here.
- *raw policy:* `BELIEF:worlds=16` of the same checkpoint, the fair raw policy of a `full` network.
- *games:* the first 500 boards of `heldout-2p-fr` in both seat orders, 1,000 games a cell, ±3.1 points at 50%. every cell plays the same games, so any two cells pair by game id.

*cost.* measured on this box before launch: a game has about 82 searcher decisions, and every one is a full search (the engine plays forced moves itself). one leaf costs about 16 µs of network on the fused kernels in 512-row batches (29 µs in plain torch, at any batch size), 21 µs in 96-row batches, plus 4–9 µs of tree work. a 16384-sim tree holds about 95 MB, so that cell keeps only about 80 searches in flight. the variants run in order of sims, and each plays the last checkpoint first and the middle one second.

the launch estimate (\~65 h) was too low by about 2×, for two reasons. the suite's strict determinism pads every forward to the full batch, so the tail of a cell, when only a few long games are left, costs as much per move as a full batch. and the e50 network plays some very long games (112 turns on average at 1024 sims, six of 1,000 cut at the 500-turn limit), so its cells take 3–3.5× the others. measured: 1024 sims took 1.1 h a checkpoint (3.8 h for e50); 4096 is on course for \~4.5 h a checkpoint. at that rate the 16384 column would take \~20 h a checkpoint and much longer for e50.

*result.* up to 1024 sims, mcts win rate against the same checkpoint's raw policy (%):
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    header: ([checkpoint], [vs F], [64], [128], [256], [1024], [4096], [16384]),
    [e50], [47], [57.5], [61.3], [66.4], [*75.7*], [—], [—],
    [e150], [80], [55.6], [60.2], [62.7], [*71.0*], [—], [—],
    [e500], [92], [56.2], [57.4], [60.0], [*64.4*], [—], [—],
    [e1500], [95], [51.9], [53.1], [56.9], [*60.7*], [—], [—],
    [e3786], [98.5], [48.4], [49.0], [50.7], [*53.9*], [—], [—],
  ),
  caption: [`MCTS:sims=N:dets=1` against `BELIEF:worlds=16` of the same `discount-2p/g1-s42` checkpoint on `heldout-2p-fr`, 1,000 games a cell (±3.1 points). "vs F" is the raw policy on the select suite during training.],
) <tab:search-depth>

#figure(
  image("../figures/search-depth.png", width: 92%),
  caption: [the same cells with 95% wilson intervals; one line per checkpoint, darker is later in training.],
) <fig:search-depth>

the gains are paired over the same 1,000 games (mcnemar):
#figure(
  btable(
    columns: (auto, auto, auto),
    align: (left, right, right),
    header: ([checkpoint], [64 → 1024 sims], [256 → 1024 sims]),
    [e50], [+18.1 (p = 8e-24)], [+9.2 (p = 2e-7)],
    [e150], [+15.4 (p = 3e-20)], [+8.3 (p = 2e-7)],
    [e500], [+8.2 (p = 3e-6)], [+4.4 (p = 0.003)],
    [e1500], [+8.8 (p = 3e-8)], [+3.8 (p = 0.007)],
    [e3786], [+5.5 (p = 5e-5)], [+3.2 (p = 0.007)],
  ),
  caption: [win-rate gain in points from more sims, same games, `tools/search_probe.py --pair`.],
) <tab:search-depth-paired>

*reading so far.* more search helps every checkpoint, at every step measured, and significantly from 256 to 1024 sims even for the final net. the gain shrinks as the network trains, which fits search fixing mistakes a weaker policy and critic make and a stronger one makes less often. note that each checkpoint is measured against its own raw policy, so a column compares gains, not strength. one world sampled from the tracker costs something at small budgets: at 64 sims the final net's search only matches its raw policy (48.4, interval 45.3–51.5). whether the curves flatten by 4096 (the earlier single points, 53.4 and 55.2, barely moved from 1024 to 4096) is the next column.

#repro(```sh
catanrl run configs/experiments/search-depth-2p.yaml     # six variants (s64 ... s16384), five finals each
# one cell alone:
catanrl eval "MCTS:ckpt=discount-2p/g1-s42@e3786#aa8868d9c6da:sims=1024:dets=1:kernels=fused" \
  "BELIEF:ckpt=discount-2p/g1-s42@e3786#aa8868d9c6da:worlds=16" --suite heldout-2p-fr --boards 500
```)

== four players

=== 4p search vs three raw copies #chip("done") <exp:search-4p>
#meta(
  dates: "2026-10-18 16:33–17:45 (400 games); 2026-10-19 07:38–10:15 (2,000 games)",
  config: "configs/experiments/search-4p.yaml, configs/experiments/search-4p-2000.yaml",
  wandb: (("search-4p-main-19fbd48a", "search-4p, eval only"), ("search-4p-2000-main-ec69fe1d", "search-4p-2000, eval only")),
  records: ("90a1f06c", "657b314d"),
)
#tldr[in 4p, MCTS at 1024 sims with one world takes a 26.9 share against three copies of its own raw policy (parity 25) over 2,000 games. that is a two-point edge, 8% relative, about the same relative edge as in 2p. the searcher is ahead at every seat, but the interval just touches parity (p = 0.053).]

*question.* does the search edge carry over to 4p?

*setup.* `MCTS:sims=1024:dets=1` with the critic-fix network `ppo-4p-critic-fix/lambda-critic-g1@e100` (undiscounted per-seat values that sum to one, the same raw policy as `ppo-4p` to within noise; @sec:recipe), against three copies of that network's raw policy on `heldout-4p`. first 100 boards × 4 rotations (400 games, 71 min), then 500 boards × 4 rotations (2,000 games, 2.6 h with 512 in flight, 4.7 s a game).

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, left, left),
    header: ([table], [searcher's share], [raw copies' shares], [record]),
    [100 boards × 4 (400 games)], [27.0 #ci[22.9][31.6]], [24.3 (mean)], [`90a1f06c`],
    [500 boards × 4 (2,000 games)], [*26.9* #ci[25.0][28.9]], [25.2 / 22.7 / 25.3 (24.4 pooled)], [`657b314d`],
  ),
  caption: [MCTS 1024 sims, one world, against three copies of its own raw policy on `heldout-4p`. share of wins, parity 25.],
) <tab:search-4p>

in the 2,000-game table the searcher won 538 games, p = 0.053 against 25% by the record's own test. its first 100 boards replay the 400-game record (400 of 400 games identical), and the other 1,600 games alone give 26.9. the searcher beats the raw copies' mean at every seat: +3.4 in seat 0, +2.1, +3.3 and +1.3 in seats 1–3. the first seat wins 27.9% of these games overall against seat 2’s 23.1, so a small first-seat advantage sits in these numbers too (no F at the table, so no F targeting).

*reading.* a two-point edge, 8% relative, measured to about ±2. that is the same relative edge as in 2p (7–10% there): real but small at 1024 sims and one world, and a long way from paying for its cost (each searcher decision is 1024 forwards).

#repro(```sh
catanrl run configs/experiments/search-4p.yaml
catanrl run configs/experiments/search-4p-2000.yaml
```)

== search options

=== tree reuse, transpositions and leaf dedup #chip("done") <exp:search-reuse-dedup>
#meta(
  dates: "2026-10-20 04:24–14:33",
  owner: "marco",
  config: "configs/experiments/mcts-reuse-2p.yaml, configs/experiments/mcts-reuse-4p-both.yaml, configs/experiments/mcts-reuse-4p.yaml",
  wandb: (("mcts-reuse-2p-main-b166466f", "mcts-reuse-2p, eval only"), ("mcts-reuse-4p-both-main-c3008253", "mcts-reuse-4p-both, eval only"), ("mcts-reuse-4p-main-c5f75bb4", "mcts-reuse-4p, eval only")),
  records: ("aca296e2", "6601df9a", "6010fedf", "72819d1a", "90a1f06c", "12f704f2", "1b01f4ac"),
)
#tldr[marco asked for tree reuse and leaf deduplication for the deployed bot. both search options are correct and cost nothing measurable in strength, but they buy nothing either at 1024 sims and one world. every paired row leans one to four points negative, none significantly: a null with a slight unfavourable tilt.]

*question.* does keeping a seat's tree across its own moves, or sharing nodes between move orders, make the same search budget stronger?

*setup.* three mechanisms:
- `reuse=true` keeps a seat's search alive across its decisions and re-roots it when the game advanced by exactly its own action. the retained visits get a fresh simulation budget on top. about a third of a seat's decisions re-root in 4p and half in 2p; the rest follow a roll, another seat's move or a forced move.
- `transpositions=true` gives each world a transposition table: positions reached by different build, buy or discard orders share one node. the discard chain after a 7 collapses from a tree of 86 nodes to a lattice of at most fifty, and a search holds from a handful to over a hundred such links, by position. the simulation count is unchanged; the search gets deeper rather than smaller.
- leaf dedup evaluates byte-identical rows once. it is bitwise neutral, so it has no key. rows forwarded to the network fall to 46–91% of the total, depending on worlds and players.

both options are keyed only when set, so every stored search record rebuilds unchanged. each row replays the games of a stored search table and pairs with it on shared game ids (`tools/search_probe.py --pair`).

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, left),
    header: ([row], [games], [with options], [stored row], [paired], [won only with / without], [p]),
    [2p, both options (`aca296e2` vs `72819d1a`)], [200], [58.5 #ci[51.6][65.1]], [59.5], [−1.0 pp], [7 / 9], [0.80],
    [4p, both options (`6601df9a` vs `90a1f06c`)], [200], [25.0 #ci[19.5][31.4]], [28.5], [−3.5 pp], [8 / 15], [0.21],
    [4p, reuse only (`6010fedf` vs `90a1f06c`)], [400], [25.3 #ci[21.2][29.7]], [27.0], [−1.75 pp], [9 / 16], [0.23],
  ),
  caption: [MCTS 1024 sims, one world, with the options against the stored row without them, on the same games. 2p is win rate against the raw `public` policy; 4p is share against three raw copies of the critic-fix net.],
) <tab:search-reuse>

two engine bugs surfaced on the way, and both are fixed. re-rooting treated a steal with one possible card in the sampled world as deterministic, which crashed both 4p rows (engine `b596130`, with a regression test). with both options on, several root actions could share one child through a transposition, and the new root counted that child once per action (`e9f12af`, a no-op with transpositions off, so the reuse-only row is untouched). two records from before `e9f12af` are superseded, and they read the same. `12f704f2` (2p) had 58.5 against 59.5. `1b01f4ac` (4p, both options, after the crash fix) had 25.0 against 28.5: −3.5 pp, 7 against 14, p = 0.19, and only 6 of its 200 games changed in the rerun.

*reading.* both options are correct now and neutral in strength at this budget. what they change is the shape of the search (a reused root starts with 2–3 times the nodes; the discard chain is a lattice), not the decision at 1024 sims. the strength question is a follow-up row: equal share at half the sims (not run). in self-play they save nothing either (cpu measurements, @exp:az-cost): the tree survives 43% (2p) to 53% (4p) of moves, but `advance` resets the budget, and transpositions restart the new root's shared children, so they fight reuse. the deployed bot can take them or leave them (@exp:deploy-reuse-bridge).

#repro(```sh
catanrl run configs/experiments/mcts-reuse-2p.yaml
catanrl run configs/experiments/mcts-reuse-4p-both.yaml
catanrl run configs/experiments/mcts-reuse-4p.yaml
# pair a new row with its stored row on the shared games, e.g.
uv run python tools/search_probe.py --pair aca296e23ff93a67 72819d1a1a072278
```)
