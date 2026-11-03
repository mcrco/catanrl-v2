#import "../macros.typ": *

= Benchmark against released Canopy <sec:canopy>

Marco asked for this benchmark "before anything else". Canopy is a Rust AlphaZero that calls its Catan bot the strongest public 1v1 agent, with no published win rates (@sec:az). Our bot is stronger both raw and with search (rows 3 and 2). Three rows are parked: Marco is choosing between our rules and OpenHex's 1v1 rules (Canopy's training rules) against the retrained 1v1 bot.

#figure(
  btable(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, left, right, right, left, left),
    header: ([Row], [Canopy plays], [Games], [Canopy's share], [Record], [Status]),
    [1], [1600 sims vs our raw policy \ #sub[`puffer-graph2/public@e3600`]], [60 (CPU)], [45.0 #ci[33.1][57.5]], [`00291a24`], [Stopped \ #sub[blowout check only]],
    [1], [The same, on the GPU build], [1,000], [—], [`67e147c4` \ #sub[to be written]], [Parked],
    [2], [1600 sims vs our 2048-sim Gumbel search], [1,000], [*33.0* #ci[30.2][36.0]], [`96d94fab`], [Done],
    [3], [Raw network vs our raw policy], [1,000 (CPU)], [30.5 #ci[27.7][33.4]], [`1c916131`], [Done],
    [4], [1600 sims vs F], [1,000], [—], [`9f0bafd7` \ #sub[to be written]], [Parked],
    [5], [1600 sims vs the deployed 1v1 bot \ #sub[`BELIEF` on `puffer-graph2/full@e3350`, 16 worlds]], [1,000], [—], [`aeaab0be` \ #sub[to be written]], [Parked],
  ),
  caption: [The benchmark, numbered as in the original five-row plan. 1,000-game rows play the first 500 boards of `heldout-2p` in both seat orders (`831eea84`'s boards), pairing game by game; Canopy sits in slot 0, so a record's score rate is its share.],
) <tab:canopy-rows>

== Setup

=== The Canopy bridge: fairness, CPU and GPU builds #chip("done") <exp:canopy-build>
#meta(
  dates: "2026-10-23",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Canopy plays in our harness through a fair bridge to its Rust search. The GPU build plays like the CPU build at 5.3 searched decisions a second instead of 0.2, enough for 1,000-game rows.]

*Question.* Can Canopy play in our harness fairly and cheaply enough for 1,000-game rows?

#lead[Setup.]
/ Agent: `cullback/canopy` at `6185983` (nexus-v3 iteration 315) as a `CANOPY` agent, a bridge process to its Rust search built from v1's bridge. The GPU build is its CUDA backend on the same commit, patches and checkpoint.
/ Patches: Four under `third_party/canopy/`, built reproducibly. v1's: the per-move search budget and the bridge. Ours: the bridge sends only the opponent's hidden dev-card count (v1's sent the cards), and tree reuse is exact.
/ Rules: Ours (15 VP, discard above 9, no friendly robber, random dice); Canopy trained with a friendly robber and balanced dice. It cannot express a one-card Year of Plenty (counted when offered).

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([Check], [Result]),
    tgroup(2)[The bridge],
    [Hidden information], [Redealing the opponent's hidden cards, victory point cards included, never changed the bytes Canopy received over 8,510 audited positions (2p resource hands are public anyway).],
    [Tree reuse], [Only within its own turn and through forced moves: about half its decisions keep visits, less than natively (probably a small handicap).],
    [Smoke tests \ #sub[4 games a row, plumbing only]], [0 fallbacks, refused moves or rule mismatches; Canopy won 4 of 4 against our raw policy, twice.],
    [CPU cost], [5–7 CPU-seconds a decision at 1600 sims; the five-row CPU plan was about 14 h on 4 cores.],
    tgroup(2)[GPU build against CPU build],
    [Same positions], [Same move on 300 of 300 positions at 1600 sims; root values differ by 2.5 × 10⁻⁴ on average, 20 times less than a new search seed. Batching does not change outputs.],
    [Whole games], [Same player statistically, not bit for bit (29 of 40 smoke games identical, 2 winners changed), so GPU and CPU records key apart.],
    [Throughput], [About 5.3 searched decisions a second (4 bridges of 256 games, GPU shared with the critic refit), against 0.2 for one CPU bridge.],
    [Pinning], [Not byte-reproducible (a hash map in the kernel compiler), so the records name the installed binary; it installs only as that binary, checked by a behavioural fingerprint (64 fixed positions).],
  ),
  caption: [Checks on the Canopy bridge. The CUDA build merged on its own as `b385ea9` (code ids unchanged).],
)

*Reading.* What still leans against Canopy: our dice and robber rules, the one-card Year of Plenty, and half its native tree reuse.

#repro(```sh
# CPU build: reproducible, gives the binary_sha256 pinned in
# third_party/canopy/canopy-bridge.manifest.json. SRC is a new directory; the checkpoint
# is model_iter_315.mpk of Canopy's catan-nexus-v3 release
third_party/canopy/build_bridge.sh --backend cpu /tmp/canopy-src runs/_canopy model_iter_315.mpk
# CUDA: not bit-reproducible, so keep the installed runs/_canopy-cuda
# (binary 1f7cc246...68802689); a rebuild is a new player for the record keys
```)

== Results

=== Canopy's raw network vs ours #chip("done") <exp:canopy-raw-vs-raw>
#meta(
  dates: "2026-10-23",
  owner: "marco",
  config: "configs/experiments/canopy-bench-2p.yaml",
  wandb: (),
  records: ("1c916131",),
)
#tldr[Without search, Canopy's network scores 30.5% against our raw policy over 1,000 games, so its case rests on its search.]

*Question.* How does Canopy's network without search do against our raw policy?

*Setup.* Row 3: Canopy at `sims=0` (argmax over legal moves) against `puffer-graph2/public@e3600`; CPU build, 8 minutes, no W&B mirror.

*Result.* *30.5* #ci[27.7][33.4] (`1c916131`): 34% from the first seat, 27% from the second. 0 fallbacks; 15 inexpressible one-card Years of Plenty.

*Reading.* Against Canopy: its raw policy reads the opponent's unplayed dev cards as an expectation where self-play encoded the exact hand (its search deals hands; its root input does not), and it never saw our dice or robber rules.

#repro(```sh
OMP_NUM_THREADS=1 catanrl eval --suite heldout-2p --boards 500 --device cpu --threads 2 --num-envs 32 \
  "CANOPY:sims=0:procs=2" "POLICY:ckpt=puffer-graph2/public@e3600"
```)

=== Row 1 on CPU: Canopy's search vs our raw policy, 60 games #chip("stopped") <exp:canopy-row1-cpu>
#meta(
  dates: "2026-10-23 09:55–13:06",
  owner: "marco",
  config: "configs/experiments/canopy-bench-2p.yaml",
  wandb: (("canopy-bench-2p-main-43b0309d", "canopy-bench-2p, eval only"),),
  records: ("00291a24",),
)
#tldr[Canopy's 1600-sim search scored 45.0% against our raw policy over 60 games, ruling out only a blowout; the 1,000-game GPU row (@exp:canopy-gpu-rows) replaces it.]

*Question.* Does Canopy's full search beat our raw policy?

*Setup.* Canopy at 1600 sims (`c_puct` 2.5, `c_visit` 50, `c_scale` 1, no Dirichlet noise) against `puffer-graph2/public@e3600` on the first 30 boards of `heldout-2p`, both seat orders; CPU build, 3 h 9 min. The CPU config sized rows to Canopy's CPU cost (60 games for row 1, 40 for rows 2, 4 and 5, 1,000 for row 3). Marco, rightly: "why would we play 60 games? how are we getting stat sig results with that?" At ±12.6 pp (60 games) and ±15.5 (40) they could only detect a blowout, so the CPU run stopped after this row for the GPU build (@exp:canopy-build).

*Result.* *45.0* #ci[33.1][57.5] (`00291a24`): 40% from the first seat, 50% from the second. 0 fallbacks, 0 refused moves.

#repro(```sh
# As launched: the CPU config plays row 1 first, then goes on to its 40-game rows
OMP_NUM_THREADS=1 nice -n 15 catanrl run configs/experiments/canopy-bench-2p.yaml
# Row 1 alone
OMP_NUM_THREADS=1 catanrl eval --suite heldout-2p --boards 30 --device cpu --threads 2 --num-envs 32 \
  "CANOPY:sims=1600:c_puct=2.5:c_visit=50:c_scale=1:dirichlet_epsilon=0:procs=4" \
  "POLICY:ckpt=puffer-graph2/public@e3600"
```)

=== Row 2 on GPU: Canopy's search vs ours #chip("done") <exp:canopy-search-gpu>
#meta(
  dates: "2026-10-23 13:41 – 2026-10-24 00:54",
  owner: "marco",
  config: "configs/experiments/canopy-search-2p-gpu.yaml",
  wandb: (("canopy-search-2p-gpu-main-c0e10107", "canopy-search-2p-gpu, eval only"),),
  records: ("96d94fab",),
)
#tldr[Search against search over 1,000 games: Canopy scores 33.0%, so our 2048-sim search wins two games in three. Canopy is weakest in short games and level in the longest third.]

*Question.* How does Canopy's search fare against ours?

*Setup.* Our side is `831eea84`'s search (@sec:search); its spec resolves identically. 1,000 games: ±3.1 pp, about 80% power for a 4–5 pp gap from 50%. Run first at Marco's request; the full GPU config shares its record id and will skip it. The critic refit paused at e266 to free the GPU (@sec:critic).

*Result.* *33.0* #ci[30.2][36.0] (`96d94fab`): 36.8% from the first seat, 29.2% from the second; 18.0% in the shortest third of games (40–72 turns), 30.6% in the middle (72–85), 50.3% in the longest (85–141). 0 fallbacks or truncations; 5 inexpressible one-card Years of Plenty; one refused move (an end turn from a reused tree), replaced by a fresh search.

*Reading.* Our search beats Canopy's by more than it beats our raw policy (62.4%, `831eea84`), and our raw network beats Canopy's by a similar margin (69.5%, `1c916131`). Canopy trained on Colonist's 1v1 rules, which this harness does not play (@exp:canopy-build); losing short games and holding even in long ones fits being worst early, where those rules matter most (a reading, not a test).

#repro(```sh
CATANRL_CANOPY_DIR=$PWD/runs/_canopy-cuda OMP_NUM_THREADS=1 nice -n 5 \
  catanrl run configs/experiments/canopy-search-2p-gpu.yaml
```)

=== Rows 1, 5 and 4 on GPU, 1,000 games each #chip("parked") <exp:canopy-gpu-rows>
#meta(
  dates: "2026-10-23",
  dates-label: "queued",
  owner: "marco",
  config: "configs/experiments/canopy-bench-2p-gpu.yaml",
  wandb: (),
  records: (),
)
#tldr[The rest of the benchmark at full size: Canopy's search against our raw policy, the deployed 1v1 bot and F, detecting 2–5 pp gaps. Parked on a rules question for Marco.]

*Question.* How does Canopy's search fare against our raw policy, our deployed bot and F?

*Setup.* Rows 1, 5 and 4 of @tab:canopy-rows in that order, then row 2 (played). About 19 h with the critic refit paused (the two do not fit in 8 GB).

#figure(
  btable(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    header: ([Comparison], [What it says], [Detectable gap]),
    [Row 1 vs 50%], [Canopy's search vs our raw policy], [≈ 4.4 pp],
    [Rows 1 vs 2, paired], [What our search buys against Canopy], [≈ 5.2 pp],
    [Row 1 vs `1c916131`], [What Canopy's search buys over its raw net], [≈ 5 pp],
    [Row 4 vs our policy's 95.6 \ #sub[vs F on the same boards]], [Canopy vs our policy, both measured on F], [≈ 2.2 pp],
  ),
  caption: [The gap each comparison detects with 80% power (two-sided 5%) at 1,000 games a row.],
)

*Status.* Parked (2026-10-24). OpenHex's 1v1 queue plays Canopy's training rules (friendly robber, balanced dice), so these rows may say more on those rules, against the 1v1 bot retrained for them (@sec:recipe). Marco decides.

#repro(```sh
# runtime.canopy_dir names runs/_canopy-cuda, so no CATANRL_CANOPY_DIR is needed
OMP_NUM_THREADS=1 nice -n 5 catanrl run configs/experiments/canopy-bench-2p-gpu.yaml
```)
