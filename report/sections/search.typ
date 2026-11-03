#import "../macros.typ": *

= Search at play time <sec:search>

Marco's recipe has a second step: put the trained policy and critic inside a search, sample the hidden cards from the hand tracker, and gain about another 5% head to head over the raw policy. This section measures that edge in 2p and 4p and tests two search options built for the deployed bot. In 2p, PUCT search with one sampled world wins 53–55% against the raw policy at 1,024–4,096 sims, more sims keep helping up to 16,384, and the 2,048-sim Gumbel search wins 62.4%, the strongest 2p player we have. In 4p the edge is about two points of share (8% relative). Tree reuse and transpositions change nothing measurable.

*Agents* (defined in @sec:front). Every fair search row uses `dets=1`: one world sampled from the hand tracker per search. `oracle=true` searches the true game and is only a reference. `root=gumbel` replaces the PUCT root with Gumbel AlphaZero's root over `gumbel_m` candidate moves. The raw opponent is `POLICY` (argmax) for a `public` net and `BELIEF:worlds=16` for a `full` net.

== Two players

=== MCTS on the true game vs F #chip("done") <exp:search-oracle-vs-F>
#meta(
  dates: "2026-10-14",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-full-2fdb6f44", "puffer-graph2/full"),),
  records: ("3fdae985", "609e0aa7"),
)
#tldr[An oracle search, which sees the opponent's hidden cards, scores 94.8% against F at 64 sims and 95.2% at 256: no better than the raw policy's 95.16%.]

*Question.* Does search move the score against F?

*Setup.* `MCTS` with `oracle=true` and `dets=1` on `puffer-graph2/full@e3350`, against F on the first 250 boards of `heldout-2p` in both seat orders: 500 games a row. These were finals of `puffer-graph2/full` under the config's first version (`4a56469`), since dropped from the YAML; the eval lines below replay them.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Agent], [Games], [Win % vs F], [Record]),
    [Raw `full` policy (reference, @sec:recipe)], [5,000], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
    [MCTS, 64 sims, oracle], [500], [94.8 #ci[92.5][96.4]], [`3fdae985`],
    [MCTS, 256 sims, oracle], [500], [95.2 #ci[93.0][96.8]], [`609e0aa7`],
  ),
  caption: [Oracle search with `puffer-graph2/full@e3350` against F on `heldout-2p`. Every row sees hidden cards: the oracle all of them, the raw `full` policy the opponent's dev cards.],
) <tab:search-oracle>

*Reading.* All three rows sit inside each other's intervals. Against F a policy this strong is already at its ceiling, so search has no room to show; every later search row is measured head to head against a raw policy.

#repro(```sh
# Finals dropped from puffer-graph2.yaml; the records' specs, defaults left out
catanrl eval "MCTS:ckpt=puffer-graph2/full@e3350:sims=64:dets=1:oracle=true" F \
  --suite heldout-2p --boards 250
catanrl eval "MCTS:ckpt=puffer-graph2/full@e3350:sims=256:dets=1:oracle=true" F \
  --suite heldout-2p --boards 250
```)

=== Search edge over the belief-averaged policy (`full` net) #chip("done") <exp:search-edge-2p-full>
#meta(
  dates: "2026-10-17 – 2026-10-18 01:31",
  owner: "marco",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-full-2fdb6f44", "puffer-graph2/full"),),
  records: ("04fd2c25", "2aaf2eb7"),
)
#tldr[Fair search (one world sampled from the tracker) beats the same network's belief-averaged policy 53.4% at 1,024 sims and 55.2% at 4,096: a point or two under v1's edge, over a stronger raw policy.]

*Question.* Does v2 get the \~5% head-to-head edge from search that Marco's recipe claims?

*Setup.* `MCTS:sims=N:dets=1` against `BELIEF:worlds=16` of the same checkpoint, `puffer-graph2/full@e3350`, on `heldout-2p`: 500 boards in both seat orders at 1,024 sims, 250 boards at 4,096. These are the last two finals of `puffer-graph2/full`, played with 128 games in flight.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (right, right, right, right, left),
    header: ([Sims], [Games], [MCTS vs `BELIEF`], [v1, same pairing], [Record]),
    [256], [—], [—], [51.1], [—],
    [1,024], [1,000], [*53.4* #ci[50.3][56.5]], [55.1], [`04fd2c25`],
    [4,096], [500], [*55.2* #ci[50.8][59.5]], [58.1], [`2aaf2eb7`],
  ),
  caption: [MCTS win rate against the same network's `BELIEF` policy on `heldout-2p`. The v1 column is v1's network against its own belief-averaged policy, 1,000 games each.],
) <tab:search-edge-full>

Both seats agree: at 1,024 sims the searcher wins 55.0% from the first seat and 51.8% from the second.

*Reading.* The edge is real and grows with sims. Its cost, 11.5 GPU hours for 1,500 games, is what parks AlphaZero at these sim counts (@sec:az).

#repro(```sh
# Trains (8 h), then plays every final
catanrl run configs/experiments/puffer-graph2.yaml --only full
# The two search finals alone:
catanrl eval "MCTS:ckpt=puffer-graph2/full@e3350:sims=1024:dets=1" \
  "BELIEF:ckpt=puffer-graph2/full@e3350:worlds=16" --suite heldout-2p --boards 500
catanrl eval "MCTS:ckpt=puffer-graph2/full@e3350:sims=4096:dets=1" \
  "BELIEF:ckpt=puffer-graph2/full@e3350:worlds=16" --suite heldout-2p --boards 250
```)

=== Search edge over the raw `public` policy #chip("done") <exp:search-edge-2p-public>
#meta(
  dates: "2026-10-19 02:07–07:10",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-public-3ec072be", "puffer-graph2/public"),),
  records: ("72819d1a", "0ef1b81e"),
)
#tldr[The same search over the fair `public` network beats its raw policy 54.9% at 1,024 sims and 55.2% at 4,096, the same edge as the `full` network, in under half the GPU time.]

*Question.* Does the edge hold for the network we would serve raw?

*Setup.* `puffer-graph2/public@e3600`: the same recipe and budget as `full`, at the `public` level, which is fair in 2p without belief averaging (@sec:front, @sec:recipe). `MCTS:sims=N:dets=1` against `POLICY` of the same checkpoint, on the same `heldout-2p` boards as the `full` rows, with 256 games in flight.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (right, right, right, right, left),
    header: ([Sims], [Games], [MCTS vs raw policy], [Wall time], [Record]),
    [1,024], [1,000], [*54.9* #ci[51.8][58.0]], [1.5 h], [`72819d1a`],
    [4,096], [500], [*55.2* #ci[50.8][59.5]], [3.6 h], [`0ef1b81e`],
  ),
  caption: [MCTS win rate with `puffer-graph2/public@e3600` against its own raw policy on `heldout-2p`.],
) <tab:search-edge-public>

Both seats agree. At 1,024 sims the searcher wins 60.6% in seat 0 and 49.2% in seat 1; the raw policy wins 50.8% and 39.4%. Seat 0 wins about 56% of these games whoever sits there.

*Reading.* The same edge as the `full` network over its belief-averaged self (53.4 and 55.2). The 1,500 games took 5.0 h against the `full` rows' 11.5 h: the opponent is one forward pass instead of sixteen, and 256 games are in flight instead of 128.

#repro(```sh
# Trains (8 h), then plays every final
catanrl run configs/experiments/puffer-graph2.yaml --only public
# The two search finals alone:
catanrl eval "MCTS:ckpt=puffer-graph2/public@e3600:sims=1024:dets=1" \
  "POLICY:ckpt=puffer-graph2/public@e3600" --suite heldout-2p --boards 500
catanrl eval "MCTS:ckpt=puffer-graph2/public@e3600:sims=4096:dets=1" \
  "POLICY:ckpt=puffer-graph2/public@e3600" --suite heldout-2p --boards 250
```)

=== Gumbel search at 2,048 sims vs the raw policy #chip("done") <exp:search-gumbel-edge-2p>
#meta(
  dates: "2026-10-21 18:56–22:50",
  config: "configs/experiments/gumbel-edge-2p.yaml",
  wandb: (("gumbel-edge-2p-main-efda50ac", "gumbel-edge-2p, eval only"),),
  records: ("831eea84",),
)
#tldr[The `public` network inside a 2,048-sim Gumbel search wins 62.4% of 1,000 games against its own raw policy, far above PUCT over the same policy (54.9 and 55.2). It is the strongest 2p player we have, and the baseline that every later search-harness row pairs with.]

*Question.* AlphaZero's 2,048-sim targets made the raw policy worse (@exp:az-2p-2048). Are the moves behind those targets good ones?

*Setup.* `MCTS:ckpt=puffer-graph2/public@e3600:sims=2048:dets=1:root=gumbel:gumbel_m=16`, the fair search AlphaZero trained on, against `POLICY` of the same checkpoint on the first 500 boards of `heldout-2p` in both seat orders (1,000 games).

*Result.* The search wins *62.4%* #ci[59.4][65.3] (`831eea84`). PUCT over the same policy wins 54.9% at 1,024 sims and 55.2% at 4,096 (@exp:search-edge-2p-public).

*Reading.* The search had plenty to teach, and the raw policy still got worse learning it (@sec:az). These 1,000 games became the search harness, Marco's rule for judging a new network: put it in this same search against the same raw start, and pair it game by game with `831eea84` (McNemar over the shared game ids, `tools/search_probe.py --pair`). The AlphaZero checkpoints (@sec:az), the critic refit (@sec:critic) and Canopy's search (@sec:canopy) are all judged this way.

#repro(```sh
catanrl run configs/experiments/gumbel-edge-2p.yaml
```)

=== Search depth vs strength across training #chip("stopped") <exp:search-depth-2p>
#meta(
  dates: "2026-10-29 00:47 – 10-31 23:41",
  owner: "marco",
  config: "configs/experiments/search-depth-2p.yaml",
  wandb: (("search-depth-2p-s64-b23bbbaf", "s64"), ("search-depth-2p-s128-1cd135fb", "s128"), ("search-depth-2p-s256-28bd5783", "s256"), ("search-depth-2p-s1024-2bb0e327", "s1024"), ("search-depth-2p-s4096-510e46c4", "s4096"), ("search-depth-2p-s16384-152f0cab", "s16384")),
  records: ("be98751d", "2672eaad", "101296e4", "13061034", "d7fc0e57", "9004b0ac", "19048e65", "ad2a43f4", "76c3fbed", "8269c07b", "032c4368", "83b507c8", "ec991f87", "1cee7f01", "49254c47", "ce9f6daf", "13ed5ff1", "25c4c060", "e26edabf", "c91e1186", "ce60e974", "f44b638f", "1055bdc2", "d653b342", "812040d1", "e07a8a98", "1f623400"),
)
#tldr[More search helps every checkpoint and has not leveled off: against the same checkpoint's raw policy, each 4× in sims from 256 up adds 3–7 points, including 4,096 → 16,384 (final net 57.0 → 60.5%, e500 71.1 → 75.8%). The weaker the network, the more search adds: at 4,096 sims the e50 net's search wins 82.0%, the final net's 57.0%. Three 16,384-sim cells (e150, e1500, e50) are pending: the out-of-memory killer stopped the run, and they are being played again.]

*Question.* Does more search consistently make a network stronger, and does that change as the network trains (Marco's question)? The earlier rows were single points that barely moved from 1,024 to 4,096 sims (53.4 and 55.2 over one `full` checkpoint, 54.9 and 55.2 over one `public` checkpoint; @exp:search-edge-2p-full, @exp:search-edge-2p-public), which could be a ceiling.

#lead[Setup.]
/ Network: `discount-2p/g1-s42`, puffer-graph2's recipe from scratch on OpenHex's 1v1 rules (`base-2p-fr`), $gamma = 1$, `full` level (@exp:recipe-discount). Five checkpoints about 3× apart in epochs: e50, e150, e500, e1500 and e3786, which score 47, 80, 92, 95 and 98.5% against F on the select suite.
/ Searcher: `MCTS:sims=N:dets=1:kernels=fused`, PUCT with one world sampled from the hand tracker, the fair search of every 2p row above. `kernels=fused` runs the deployed search bot's network on custom CUDA kernels: the same outputs to the last bit, about 1.7× faster here.
/ Raw policy: `BELIEF:worlds=16` of the same checkpoint, the fair raw policy of a `full` network.
/ Games: The first 500 boards of `heldout-2p-fr` in both seat orders, 1,000 games a cell (±3.1 points at 50%). Every cell plays the same games, so any two cells pair by game id.

#lead[Cost.]
- Measured on this box before launch: a game has about 82 searcher decisions, each a full search (the engine plays forced moves itself). A leaf costs about 16 µs of network on the fused kernels in 512-row batches (29 µs in plain PyTorch, at any batch size) and 21 µs in 96-row batches, plus 4–9 µs of tree work.
- A 16,384-sim tree holds about 95~MB, so that cell keeps only about 80 searches (160 games) in flight. The variants run in order of sims; each plays the last checkpoint first and the middle one second.
- The launch estimate (\~65 h) was about 2× too low. Strict determinism pads every forward to the full batch, so a cell's tail, when only a few long games are left, costs as much per move as a full batch. And the e50 network plays some very long games (112 turns on average at 1,024 sims, six of 1,000 cut at the 500-turn limit), so its cells take 3–3.5× the others.
- Measured wall time per checkpoint: 1.1 h at 1,024 sims (3.8 h for e50), 4.1–4.3 h at 4,096 (8.8 h for e50), and 16.3–16.5 h at 16,384 for e3786 and e500.

*Stopped.* 15 minutes into the e150 cell at 16,384 sims, the kernel's OOM killer ended the eval: it held 16.6~GB with 160 games in flight, and the box had less free memory than at launch (swap full). The two finished 16,384-sim cells (e3786, e500) are complete records. A resume plays the three missing cells (e150, e1500, e50) at a memory-safe 96 games in flight, an estimated 25 h each for e150 and e1500 and three days or more for e50.

*Result.* MCTS win rate (%) against the same checkpoint's raw policy:
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    header: ([Checkpoint], [vs F], [64], [128], [256], [1,024], [4,096], [16,384]),
    [e50], [47], [57.5], [61.3], [66.4], [75.7], [*82.0*], [—],
    [e150], [80], [55.6], [60.2], [62.7], [71.0], [*76.4*], [—],
    [e500], [92], [56.2], [57.4], [60.0], [64.4], [71.1], [*75.8*],
    [e1500], [95], [51.9], [53.1], [56.9], [60.7], [*65.0*], [—],
    [e3786], [98.5], [48.4], [49.0], [50.7], [53.9], [57.0], [*60.5*],
  ),
  caption: [Win rate by sims: `MCTS:sims=N:dets=1` against `BELIEF:worlds=16` of the same `discount-2p/g1-s42` checkpoint on `heldout-2p-fr`, 1,000 games a cell (±3.1 points). "vs F" is the raw policy on the select suite during training. "—": pending, being played again after the out-of-memory kill.],
) <tab:search-depth>

#figure(
  image("../figures/search-depth.png", width: 92%),
  caption: [The same cells with 95% Wilson intervals; one line per checkpoint, darker is later in training.],
) <fig:search-depth>

The gains, paired over the same 1,000 games (McNemar):
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Checkpoint], [64 → 1,024], [256 → 1,024], [1,024 → 4,096], [4,096 → 16,384]),
    [e50], [+18.1 #sub[(p = 8e-24)]], [+9.2 #sub[(p = 2e-7)]], [+6.4 #sub[(p = 7e-5)]], [—],
    [e150], [+15.4 #sub[(p = 3e-20)]], [+8.3 #sub[(p = 2e-7)]], [+5.4 #sub[(p = 7e-4)]], [—],
    [e500], [+8.2 #sub[(p = 3e-6)]], [+4.4 #sub[(p = 0.003)]], [+6.7 #sub[(p = 3e-5)]], [+4.7 #sub[(p = 0.003)]],
    [e1500], [+8.8 #sub[(p = 3e-8)]], [+3.8 #sub[(p = 0.007)]], [+4.3 #sub[(p = 0.005)]], [—],
    [e3786], [+5.5 #sub[(p = 5e-5)]], [+3.2 #sub[(p = 0.007)]], [+3.1 #sub[(p = 0.02)]], [+3.5 #sub[(p = 0.01)]],
  ),
  caption: [Win-rate gain in points from more sims, on the same games (`tools/search_probe.py --pair`). "—": pending.],
) <tab:search-depth-paired>

#lead[Reading.]
- More search helps every checkpoint at every step measured, with no plateau by 16,384 sims. From 256 sims up, each 4× adds 3–7 points to every checkpoint, all significant paired. For the final net the steps are nearly constant (+3.2, +3.1, +3.5 points per 4×): its gain grows about linearly in log sims, with no ceiling in sight.
- The earlier single points (53.4 and 55.2 at 1,024 and 4,096, @exp:search-edge-2p-full) moved by about as much as these steps, within their own noise.
- The gain shrinks as the network trains, which fits search fixing mistakes that a weaker policy and critic make and a stronger one makes less often. Each checkpoint plays its own raw policy, so a column compares gains, not strength.
- One sampled world costs something at small budgets: at 64 sims the final net's search only matches its raw policy (48.4 #ci[45.3][51.5]).

#repro(```sh
catanrl run configs/experiments/search-depth-2p.yaml     # Six variants (s64 ... s16384), five finals each
# One cell alone:
catanrl eval "MCTS:ckpt=discount-2p/g1-s42@e3786#aa8868d9c6da:sims=1024:dets=1:kernels=fused" \
  "BELIEF:ckpt=discount-2p/g1-s42@e3786#aa8868d9c6da:worlds=16" --suite heldout-2p-fr --boards 500
```)

== Four players

=== 4p search vs three raw copies #chip("done") <exp:search-4p>
#meta(
  dates: "2026-10-18 16:33–17:45 (400 games); 2026-10-19 07:38–10:15 (2,000 games)",
  config: "configs/experiments/search-4p.yaml, configs/experiments/search-4p-2000.yaml",
  wandb: (("search-4p-main-19fbd48a", "search-4p, eval only"), ("search-4p-2000-main-ec69fe1d", "search-4p-2000, eval only")),
  records: ("90a1f06c", "657b314d"),
)
#tldr[In 4p, MCTS at 1,024 sims with one world takes a 26.9% share against three copies of its own raw policy (parity 25) over 2,000 games: a two-point edge, 8% relative, about the same relative edge as in 2p. The searcher is ahead at every seat, but the interval just touches parity (p = 0.053).]

*Question.* Does the search edge carry over to 4p?

*Setup.* `MCTS:sims=1024:dets=1` with the critic-fix network `ppo-4p-critic-fix/lambda-critic-g1@e100` (undiscounted per-seat values that sum to one; the same raw policy as `ppo-4p` to within noise, @sec:recipe), against three copies of that network's raw policy on `heldout-4p`. First 100 boards × 4 rotations (400 games, 71 min), then 500 boards × 4 rotations (2,000 games, 2.6 h with 512 in flight, 4.7 s a game).

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, 1fr, auto),
    align: (left, right, right, left, left),
    header: ([Boards × rotations], [Games], [Searcher's share], [Raw copies' shares], [Record]),
    [100 × 4], [400], [27.0 #ci[22.9][31.6]], [24.3 (mean)], [`90a1f06c`],
    [500 × 4], [2,000], [*26.9* #ci[25.0][28.9]], [25.2 / 22.7 / 25.3 (24.4 pooled)], [`657b314d`],
  ),
  caption: [MCTS at 1,024 sims, one world, against three copies of its own raw policy on `heldout-4p`. Share of wins; parity is 25.],
) <tab:search-4p>

In the 2,000-game table the searcher won 538 games, p = 0.053 against 25% by the record's own test. Its first 100 boards replay the 400-game record (400 of 400 games identical), and the other 1,600 games alone give 26.9. The searcher beats the raw copies' mean at every seat: +3.4 in seat 0, +2.1, +3.3 and +1.3 in seats 1–3. The first seat wins 27.9% of these games overall against seat 2's 23.1%, so a small first-seat advantage sits in these numbers too (no F at the table, so no F targeting).

*Reading.* Measured to about ±2 points, the edge is real but small at 1,024 sims and one world (7–10% relative in 2p), and a long way from paying for its cost: each searcher decision is 1,024 forwards.

#repro(```sh
catanrl run configs/experiments/search-4p.yaml
catanrl run configs/experiments/search-4p-2000.yaml
```)

== Search options

=== Tree reuse, transpositions and leaf dedup #chip("done") <exp:search-reuse-dedup>
#meta(
  dates: "2026-10-20 04:24–14:33",
  owner: "marco",
  config: "configs/experiments/mcts-reuse-2p.yaml, configs/experiments/mcts-reuse-4p-both.yaml, configs/experiments/mcts-reuse-4p.yaml",
  wandb: (("mcts-reuse-2p-main-b166466f", "mcts-reuse-2p, eval only"), ("mcts-reuse-4p-both-main-c3008253", "mcts-reuse-4p-both, eval only"), ("mcts-reuse-4p-main-c5f75bb4", "mcts-reuse-4p, eval only")),
  records: ("aca296e2", "6601df9a", "6010fedf", "72819d1a", "90a1f06c", "12f704f2", "1b01f4ac"),
)
#tldr[Marco asked for tree reuse and leaf deduplication for the deployed bot. Both search options are correct and cost nothing measurable in strength, but buy nothing either at 1,024 sims and one world: every paired row leans one to four points negative, none significantly.]

*Question.* Does keeping a seat's tree across its own moves, or sharing nodes between move orders, make the same search budget stronger?

*Setup.* Three mechanisms:
/ `reuse=true`: Keeps a seat's search alive across its decisions and re-roots it when the game advanced by exactly its own action; the retained visits get a fresh simulation budget on top. About a third of a seat's decisions re-root in 4p and half in 2p; the rest follow a roll, another seat's move or a forced move.
/ `transpositions=true`: Gives each world a transposition table, so positions reached by different build, buy or discard orders share one node. The discard chain after a 7 collapses from a tree of 86 nodes to a lattice of at most 50, and a search holds from a handful to over a hundred such links, depending on the position. The sim count is unchanged; the search gets deeper rather than smaller.
/ Leaf dedup: Evaluates byte-identical rows once. It is bitwise neutral, so it has no key. Rows forwarded to the network fall to 46–91% of the total, depending on worlds and players.

Both options are keyed only when set, so every stored search record rebuilds unchanged. Each row replays the games of a stored search table and pairs with it on the shared game ids (`tools/search_probe.py --pair`).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    header: ([Options], [Games], [With], [Stored], [Paired], [Won only \ with / without], [p]),
    [2p, both \ #sub[`aca296e2` vs `72819d1a`]], [200], [58.5 #ci[51.6][65.1]], [59.5], [−1.0 pp], [7 / 9], [0.80],
    [4p, both \ #sub[`6601df9a` vs `90a1f06c`]], [200], [25.0 #ci[19.5][31.4]], [28.5], [−3.5 pp], [8 / 15], [0.21],
    [4p, reuse only \ #sub[`6010fedf` vs `90a1f06c`]], [400], [25.3 #ci[21.2][29.7]], [27.0], [−1.75 pp], [9 / 16], [0.23],
  ),
  caption: [MCTS at 1,024 sims, one world, with the options against the stored row without them, on the same games. 2p is win rate against the raw `public` policy; 4p is share against three raw copies of the critic-fix net. "Won only" counts the games only one of the two rows won.],
) <tab:search-reuse>

Two engine bugs surfaced on the way, and both are fixed:
- Re-rooting treated a steal with one possible card in the sampled world as deterministic, which crashed both 4p rows (engine `b596130`, with a regression test).
- With both options on, several root actions could share one child through a transposition, and the new root counted that child once per action (`e9f12af`; a no-op with transpositions off, so the reuse-only row is untouched).
Two records from before `e9f12af` are superseded, and they read the same: `12f704f2` (2p) had 58.5 against 59.5; `1b01f4ac` (4p, both options, after the crash fix) had 25.0 against 28.5, −3.5 pp, 7 against 14, p = 0.19, and only 6 of its 200 games changed in the rerun.

*Reading.* Both options are correct now and neutral in strength at this budget. They change the shape of the search (a reused root starts with 2–3 times the nodes; the discard chain is a lattice), not the decision at 1,024 sims. The strength question left is a follow-up row: equal share at half the sims (not run). In self-play they save nothing either (CPU measurements, @exp:az-cost): the tree survives 43% (2p) to 53% (4p) of moves, but `advance` resets the budget, and transpositions restart the new root's shared children, so they fight reuse. The deployed bot can take them or leave them (@exp:deploy-reuse-bridge).

#repro(```sh
catanrl run configs/experiments/mcts-reuse-2p.yaml
catanrl run configs/experiments/mcts-reuse-4p-both.yaml
catanrl run configs/experiments/mcts-reuse-4p.yaml
# Pair a new row with its stored row on the shared games, e.g.
uv run python tools/search_probe.py --pair aca296e23ff93a67 72819d1a1a072278
```)
