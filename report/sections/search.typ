#import "../macros.typ": *

= Search at play time <sec:search>

Marco's recipe puts the trained policy and critic inside a search, with hidden cards sampled from the hand tracker, for about another 5% head to head over the raw policy. Measured in 2p: 53–55% with PUCT and one sampled world, rising with sims up to 16,384, and 62.4% with 2,048-sim Gumbel search. In 4p the edge is about two points of share. Tree reuse and transpositions change nothing measurable.

*Agents* (@sec:front). Fair search rows use `dets=1`, one sampled world per search; `root=gumbel` uses Gumbel AlphaZero's root over `gumbel_m` candidate moves.

== Two players

=== MCTS on the true game vs F #chip("done") <exp:search-oracle-vs-F>
#meta(
  dates: "2026-10-14",
  config: "configs/experiments/puffer-graph2.yaml",
  wandb: (("puffer-graph2-full-2fdb6f44", "puffer-graph2/full"),),
  records: ("3fdae985", "609e0aa7"),
)
#tldr[An oracle search (it sees the hidden cards) scores 94.8% against F at 64 sims and 95.2% at 256, no better than the raw policy's 95.16%.]

*Question.* Does search raise the score against F?

*Setup.* `MCTS` with `oracle=true`, `dets=1` on `puffer-graph2/full@e3350`, against F on the first 250 boards of `heldout-2p` in both seat orders (500 games a row). Finals of the config's first version (`4a56469`), since dropped; the eval lines below replay them.

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
  caption: [Oracle search with `puffer-graph2/full@e3350` against F on `heldout-2p`.],
) <tab:search-oracle>

*Reading.* The rows overlap: against F this policy is at its ceiling, so later search rows play head to head against a raw policy.

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
#tldr[Fair search beats the same network's belief-averaged policy 53.4% at 1,024 sims and 55.2% at 4,096: a point or two under v1's edge, over a stronger raw policy.]

*Question.* Does v2 get the \~5% head-to-head edge Marco's recipe claims?

*Setup.* `MCTS:sims=N:dets=1` against `BELIEF:worlds=16`, both `puffer-graph2/full@e3350`, on `heldout-2p`: 500 boards in both seat orders at 1,024 sims, 250 at 4,096. The run's last two finals, 128 games in flight.

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
  caption: [MCTS against the same network's `BELIEF` policy on `heldout-2p`; v1 is its own network against its own belief-averaged policy (1,000 games each).],
) <tab:search-edge-full>

At 1,024 sims the searcher wins 55.0% from the first seat and 51.8% from the second.

*Reading.* The edge grows with sims. Its cost, 11.5 GPU hours for 1,500 games, is what parks AlphaZero at these sim counts (@sec:az).

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
#tldr[The same search over the fair `public` network beats its raw policy 54.9% at 1,024 sims and 55.2% at 4,096: the `full` network's edge, in under half the GPU time.]

*Question.* Does the edge hold for the network we would serve raw?

*Setup.* `puffer-graph2/public@e3600`, the `full` recipe and budget at the `public` level (@sec:recipe). `MCTS:sims=N:dets=1` against `POLICY` of the same checkpoint on the `full` rows' boards, 256 games in flight.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (right, right, right, right, left),
    header: ([Sims], [Games], [MCTS vs raw policy], [Wall time], [Record]),
    [1,024], [1,000], [*54.9* #ci[51.8][58.0]], [1.5 h], [`72819d1a`],
    [4,096], [500], [*55.2* #ci[50.8][59.5]], [3.6 h], [`0ef1b81e`],
  ),
  caption: [MCTS with `puffer-graph2/public@e3600` against its own raw policy on `heldout-2p`.],
) <tab:search-edge-public>

At 1,024 sims the searcher wins 60.6% in seat 0 and 49.2% in seat 1, the raw policy 50.8% and 39.4%: seat 0 wins about 56% whoever sits there.

*Reading.* The 1,500 games took 5.0 h against the `full` rows' 11.5 h: the opponent is one forward pass instead of sixteen, with twice the games in flight.

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
#tldr[The `public` network in a 2,048-sim Gumbel search wins 62.4% of 1,000 games against its own raw policy, far above PUCT (54.9 and 55.2, @exp:search-edge-2p-public): the strongest 2p player we have, and the baseline for every later search-harness row.]

*Question.* AlphaZero's 2,048-sim targets made the raw policy worse (@exp:az-2p-2048); are the moves behind them good?

*Setup.* `MCTS:ckpt=puffer-graph2/public@e3600:sims=2048:dets=1:root=gumbel:gumbel_m=16`, the search AlphaZero trained on, against `POLICY` of the same checkpoint on the first 500 boards of `heldout-2p` in both seat orders.

*Result.* The search wins *62.4%* #ci[59.4][65.3] (`831eea84`).

*Reading.* Yes: the search had plenty to teach, yet the raw policy got worse learning it (@sec:az). These games became the search harness, Marco's rule for judging a new network: put it in the same search against the same raw start and pair it game by game with `831eea84` (McNemar, `tools/search_probe.py --pair`). AlphaZero (@sec:az), the critic refit (@sec:critic) and Canopy (@sec:canopy) are judged this way.

#repro(```sh
catanrl run configs/experiments/gumbel-edge-2p.yaml
```)

=== Search depth vs strength across training #chip("parked") <exp:search-depth-2p>
#meta(
  dates: "2026-10-29 00:47 – 11-03 06:46; paused",
  owner: "marco",
  config: "configs/experiments/search-depth-2p.yaml",
  wandb: (("search-depth-2p-s64-b23bbbaf", "s64"), ("search-depth-2p-s128-1cd135fb", "s128"), ("search-depth-2p-s256-28bd5783", "s256"), ("search-depth-2p-s1024-2bb0e327", "s1024"), ("search-depth-2p-s4096-510e46c4", "s4096"), ("search-depth-2p-s16384-152f0cab", "s16384")),
  records: ("be98751d", "2672eaad", "101296e4", "13061034", "d7fc0e57", "9004b0ac", "19048e65", "ad2a43f4", "76c3fbed", "8269c07b", "032c4368", "83b507c8", "ec991f87", "1cee7f01", "49254c47", "ce9f6daf", "13ed5ff1", "25c4c060", "e26edabf", "c91e1186", "ce60e974", "f44b638f", "1055bdc2", "d653b342", "812040d1", "e07a8a98", "1f623400", "4fae3019"),
)
#tldr[More search helps every checkpoint and has not leveled off: each 4× in sims from 256 up adds 3–7 points, including 4,096 → 16,384 (final net 57.0 → 60.5%, e500 71.1 → 75.8%, e150 76.4 → 80.7%). The weaker the net, the more search adds: at 4,096 sims e50's search wins 82.0%, the final net's 57.0%. Paused on 3 November (Marco) for the 2v2 trading test and the transformer, with the e1500 and e50 cells at 16,384 sims still to play.]

*Question.* Does more search keep making a network stronger, and does that change as it trains (Marco)?

#lead[Setup.]
/ Network: `discount-2p/g1-s42`, puffer-graph2's recipe from scratch on `base-2p-fr`, $gamma = 1$, `full` level (@exp:recipe-discount). Five checkpoints about 3× apart in epochs.
/ Searcher: `MCTS:sims=N:dets=1:kernels=fused`, on the deployed bot's fused CUDA kernels: identical outputs, about 1.7× faster.
/ Raw policy: `BELIEF:worlds=16` of the same checkpoint.
/ Games: The first 500 boards of `heldout-2p-fr` in both seat orders, 1,000 games a cell (±3.1 points). Every cell plays the same games, so any two cells pair by game id.

#lead[Cost.]
- Before launch: about 82 searcher decisions a game, each a full search. A leaf costs about 16 µs of network on the fused kernels in 512-row batches (29 µs in plain PyTorch), 21 µs in 96-row batches, plus 4–9 µs of tree work.
- A 16,384-sim tree holds about 95~MB, so that cell keeps about 80 searches (160 games) in flight. Variants run by sims; each plays the last checkpoint first, then the middle one.
- The \~65 h launch estimate was 2× too low. Strict determinism pads every forward to the full batch, so a cell's tail of a few long games costs a full batch per move. And e50 plays very long games (112 turns on average at 1,024 sims, six of 1,000 cut at 500 turns), so its cells take 3–3.5× the others.
- Wall time per checkpoint: 1.1 h at 1,024 sims (e50: 3.8 h), 4.1–4.3 h at 4,096 (e50: 8.8 h), 16.3–16.5 h at 16,384 (e3786, e500) with 160 games in flight, 17.9 h for e150 with 96.

*Stopped, resumed, paused.* 15 minutes into the e150 cell at 16,384 sims, the OOM killer ended the eval at 16.6~GB with 160 games in flight (less free memory than at launch, swap full). The resume (2 November, 96 games in flight) finished e150 in 17.9 h, then was paused at the start of e1500 for the 2v2 trading test and the transformer (Marco). Still to play at 16,384: e1500 (about 18 h) and e50 (two days or more). Resuming skips finished cells, but must run on the same code: records from different code do not pair.

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    header: ([Checkpoint], [vs F], [64], [128], [256], [1,024], [4,096], [16,384]),
    [e50], [47], [57.5], [61.3], [66.4], [75.7], [*82.0*], [—],
    [e150], [80], [55.6], [60.2], [62.7], [71.0], [76.4], [*80.7*],
    [e500], [92], [56.2], [57.4], [60.0], [64.4], [71.1], [*75.8*],
    [e1500], [95], [51.9], [53.1], [56.9], [60.7], [*65.0*], [—],
    [e3786], [98.5], [48.4], [49.0], [50.7], [53.9], [57.0], [*60.5*],
  ),
  caption: [MCTS win rate (%) by sims against the same checkpoint's raw policy on `heldout-2p-fr`, 1,000 games a cell. "vs F": the raw policy on the select suite in training. "—": not yet played (sweep paused).],
) <tab:search-depth>

#figure(
  image("../figures/search-depth.png", width: 92%),
  caption: [The same cells with 95% Wilson intervals; darker is later in training.],
) <fig:search-depth>

#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    header: ([Checkpoint], [64 → 1,024], [256 → 1,024], [1,024 → 4,096], [4,096 → 16,384]),
    [e50], [+18.1 #sub[(p = 8e-24)]], [+9.2 #sub[(p = 2e-7)]], [+6.4 #sub[(p = 7e-5)]], [—],
    [e150], [+15.4 #sub[(p = 3e-20)]], [+8.3 #sub[(p = 2e-7)]], [+5.4 #sub[(p = 7e-4)]], [+4.3 #sub[(p = 0.002)]],
    [e500], [+8.2 #sub[(p = 3e-6)]], [+4.4 #sub[(p = 0.003)]], [+6.7 #sub[(p = 3e-5)]], [+4.7 #sub[(p = 0.003)]],
    [e1500], [+8.8 #sub[(p = 3e-8)]], [+3.8 #sub[(p = 0.007)]], [+4.3 #sub[(p = 0.005)]], [—],
    [e3786], [+5.5 #sub[(p = 5e-5)]], [+3.2 #sub[(p = 0.007)]], [+3.1 #sub[(p = 0.02)]], [+3.5 #sub[(p = 0.01)]],
  ),
  caption: [Win-rate gain in points from more sims, paired on the same games (McNemar, `tools/search_probe.py --pair`). "—": not yet played.],
) <tab:search-depth-paired>

#lead[Reading.]
- Every 4× step from 256 sims up is significant paired for every checkpoint. The final net's steps are nearly constant, so its gain grows about linearly in log sims, with no ceiling in sight.
- The earlier rows, which barely moved from 1,024 to 4,096 sims (@exp:search-edge-2p-full, @exp:search-edge-2p-public), were not a ceiling: they moved about as much as these steps, within their own noise.
- The gain shrinks as the network trains, consistent with search fixing mistakes a weaker policy and critic make more often. Each checkpoint plays its own raw policy, so a column compares gains, not strength.
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
#tldr[MCTS at 1,024 sims with one world takes a 26.9% share against three copies of its own raw policy (parity 25) over 2,000 games: two points, 8% relative (7–10% in 2p). It leads at every seat, but the interval just touches parity (p = 0.053).]

*Question.* Does the search edge carry over to 4p?

*Setup.* `MCTS:sims=1024:dets=1` on the critic-fix net `ppo-4p-critic-fix/lambda-critic-g1@e100` (undiscounted per-seat values summing to one; raw policy level with `ppo-4p`, @sec:recipe), against three copies of its raw policy on `heldout-4p`: 100 boards × 4 rotations (71 min), then 500 × 4 (2.6 h at 512 in flight, 4.7 s a game).

#lead[Result.]
#figure(
  btable(
    columns: (auto, auto, auto, 1fr, auto),
    align: (left, right, right, left, left),
    header: ([Boards × rotations], [Games], [Searcher's share], [Raw copies' shares], [Record]),
    [100 × 4], [400], [27.0 #ci[22.9][31.6]], [24.3 (mean)], [`90a1f06c`],
    [500 × 4], [2,000], [*26.9* #ci[25.0][28.9]], [25.2 / 22.7 / 25.3 (24.4 pooled)], [`657b314d`],
  ),
  caption: [MCTS at 1,024 sims, one world, against three copies of its own raw policy on `heldout-4p`: share of wins, parity 25.],
) <tab:search-4p>

The searcher won 538 of 2,000 games (p = 0.053 against 25%, the record's test). The first 100 boards replay the 400-game record exactly; the other 1,600 games alone give 26.9. It beats the raw copies' mean at every seat: +3.4, +2.1, +3.3 and +1.3 in seats 0–3. Seat 0 wins 27.9% of all games against seat 2's 23.1%, a small first-seat advantage (no F at the table).

*Reading.* Real but small at 1,024 sims and one world, and far from paying for its cost: each searcher decision is 1,024 forwards.

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
#tldr[Marco asked for tree reuse and leaf dedup for the deployed bot. Both options are correct and neither helps nor hurts measurably at 1,024 sims and one world: every paired row leans one to four points negative, none significantly.]

*Question.* Do tree reuse or transpositions make the same search budget stronger?

#lead[Setup.]
/ `reuse=true`: Keeps a seat's search across its decisions, re-rooting when the game advanced by exactly its own action; retained visits get a fresh simulation budget on top. About half a seat's decisions re-root in 2p and a third in 4p.
/ `transpositions=true`: A transposition table per world, so positions reached by different build, buy or discard orders share a node. The discard chain after a 7 goes from a tree of 86 nodes to a lattice of at most 50; a search holds from a handful to over a hundred such links. Same sim count, deeper search.
/ Leaf dedup: Evaluates byte-identical rows once (bitwise neutral, so it has no key). Forwarded rows fall to 46–91% of the total.

Both options are keyed only when set, so stored records rebuild unchanged. Each row replays a stored search table's games and pairs with it by game id.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    header: ([Options], [Games], [With], [Stored], [Paired], [Won only], [p]),
    [2p, both \ #sub[`aca296e2` vs `72819d1a`]], [200], [58.5 #ci[51.6][65.1]], [59.5], [−1.0 pp], [7 / 9], [0.80],
    [4p, both \ #sub[`6601df9a` vs `90a1f06c`]], [200], [25.0 #ci[19.5][31.4]], [28.5], [−3.5 pp], [8 / 15], [0.21],
    [4p, reuse only \ #sub[`6010fedf` vs `90a1f06c`]], [400], [25.3 #ci[21.2][29.7]], [27.0], [−1.75 pp], [9 / 16], [0.23],
  ),
  caption: [MCTS at 1,024 sims, one world, with the options vs the stored row without, same games. 2p: win rate vs the raw `public` policy; 4p: share vs three raw critic-fix copies. "Won only": games won with / without the options only.],
) <tab:search-reuse>

Two engine bugs surfaced and are fixed. Re-rooting treated a steal with one possible card in the sampled world as deterministic, crashing both 4p rows (engine `b596130`, with a regression test). With both options on, the new root counted a child shared by several root actions once per action (`e9f12af`; a no-op without transpositions, so the reuse-only row is untouched). The two superseded records from before `e9f12af` read the same: `12f704f2` (2p) 58.5 against 59.5; `1b01f4ac` (4p, both options, after the crash fix) 25.0 against 28.5, −3.5 pp, 7 against 14, p = 0.19, with 6 of 200 games changed in the rerun.

*Reading.* The options change the shape of the search (a reused root starts with 2–3× the nodes; the discard chain is a lattice), not the decision at 1,024 sims. The open strength question is equal share at half the sims (not run). In self-play they save nothing either (@exp:az-cost): the tree survives 43% (2p) to 53% (4p) of moves, but `advance` resets the budget and transpositions restart the new root's shared children, so they fight reuse. The deployed bot can take them or leave them (@exp:deploy-reuse-bridge).

#repro(```sh
catanrl run configs/experiments/mcts-reuse-2p.yaml
catanrl run configs/experiments/mcts-reuse-4p-both.yaml
catanrl run configs/experiments/mcts-reuse-4p.yaml
# Pair a new row with its stored row on the shared games, e.g.
uv run python tools/search_probe.py --pair aca296e23ff93a67 72819d1a1a072278
```)
