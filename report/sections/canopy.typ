#import "../macros.typ": *

= Benchmark against released Canopy <sec:canopy>

Marco asked for a benchmark of the released Canopy against our 2p bot "before anything else". Canopy is a Rust AlphaZero that calls its Catan bot the strongest public 1v1 agent, with no published win rates (@sec:az). Its raw network is far weaker than ours (30.5%). On Canopy's GPU build, its 1600-sim search scores 33.0% against our 2048-sim search over 1,000 games, so our search wins two games in three. Against our raw policy its search played about even over 60 games, too few to say more. The other three rows are parked: Marco is choosing between our current rules and OpenHex's 1v1 rules (friendly robber, Canopy's own robber rule) against the retrained 1v1 bot.

#figure(
  btable(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, left, right, right, left, left),
    header: ([Row], [Canopy plays], [Games], [Canopy's share], [Record], [Status]),
    [1], [1600 sims vs our raw policy \ #sub[`puffer-graph2/public@e3600`]], [60 (CPU)], [45.0 #ci[33.1][57.5]], [`00291a24`], [Stopped \ #sub[blowout check only]],
    [1], [The same, on the GPU build], [1,000], [—], [`67e147c4` \ #sub[to be written]], [Parked],
    [2], [1600 sims vs our 2048-sim Gumbel search on that network], [1,000], [*33.0* #ci[30.2][36.0]], [`96d94fab`], [Done],
    [3], [Its raw network (no search) vs our raw policy], [1,000 (CPU)], [30.5 #ci[27.7][33.4]], [`1c916131`], [Done],
    [4], [1600 sims vs F, the common reference], [1,000], [—], [`9f0bafd7` \ #sub[to be written]], [Parked],
    [5], [1600 sims vs the deployed 1v1 bot \ #sub[`BELIEF` on `puffer-graph2/full@e3350`, 16 worlds]], [1,000], [—], [`aeaab0be` \ #sub[to be written]], [Parked],
  ),
  caption: [The Canopy benchmark at a glance; rows keep the numbers of the original five-row plan. Every 1,000-game row plays the first 500 boards of `heldout-2p` in both seat orders (the boards of `831eea84`), so the rows pair game by game. Canopy always sits in slot 0, so a record's score rate is Canopy's share.],
) <tab:canopy-rows>

== Setup

=== The Canopy bridge: fairness, CPU and GPU builds #chip("done") <exp:canopy-build>
#meta(
  dates: "2026-10-23",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Released Canopy plays in our harness through a bridge to its Rust search, which shows Canopy only what a fair player sees. The GPU build plays the same moves as the CPU build at 5.3 searched decisions a second against 0.2, which makes 1,000-game rows affordable.]

*Question.* Can released Canopy play in our harness fairly, under our rules, and cheaply enough for 1,000-game rows?

#lead[Setup.]
/ Agent: Released Canopy (`cullback/canopy` at `6185983`, nexus-v3 iteration 315) plays as a `CANOPY` agent: a bridge process to its Rust search, built from v1's bridge. The GPU build is Canopy's CUDA backend on the same commit, patches and checkpoint.
/ Patches: Four, under `third_party/canopy/`, with a reproducible build. Two are v1's: the per-move search budget and the bridge. Two are ours: the bridge sends Canopy only the count of the opponent's hidden dev cards (v1's sent the true cards), and tree reuse is exact.
/ Rules: Ours: 15 VP, discard above 9, no friendly robber, random dice. Canopy trained with a friendly robber and balanced dice, so those inputs are new to it. It cannot express a one-card Year of Plenty (counted when offered).

#lead[Result.]
#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([Check], [Result]),
    tgroup(2)[The bridge],
    [Hidden information], [On 8,510 audited positions, the bytes Canopy received did not change when the opponent's hidden cards were redealt (in 2p the resource hands are public anyway). Redealing victory point cards is audited too.],
    [Tree reuse], [Canopy re-roots only within its own turn and through forced moves; about half its decisions keep visits. That is less reuse than it gets natively, probably a small handicap.],
    [Smoke tests \ #sub[4 games a row, plumbing only]], [0 fallbacks, 0 moves our engine refused, 0 rule mismatches. Canopy won 4 of 4 against our raw policy, twice.],
    [CPU cost], [5–7 CPU-seconds a decision at 1600 sims. The five-row CPU plan was about 14 h on 4 cores.],
    tgroup(2)[GPU build against CPU build],
    [Same positions], [On 300 positions at 1600 sims, the GPU build picks the CPU build's move 300 of 300 times. Root values differ by 2.5 × 10⁻⁴ on average, about 20 times less than changing the search seed. Outputs do not depend on how many games are batched together.],
    [Whole games], [The same player statistically, not bit for bit: 29 of 40 smoke games identical, 2 winners changed. So GPU and CPU records key apart.],
    [Throughput], [4 bridges of 256 games each: about 5.3 searched decisions a second while sharing the GPU with the critic refit, against 0.2 for one CPU bridge.],
    [Pinning], [The CUDA binary cannot be rebuilt byte for byte (a hash map in the kernel compiler), so the installed copy is the one the records name. It installs only as that pinned binary, checked by a behavioural fingerprint (64 fixed positions).],
  ),
  caption: [Checks on the Canopy bridge. The CUDA build was merged on its own as `b385ea9` (code ids unchanged).],
)

*Reading.* The bridge is fair on hidden information, and the GPU build is the same player as the CPU build. Three things still lean against Canopy: our dice and robber rules, the one-card Year of Plenty, and half its native tree reuse.

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
#tldr[Without search, Canopy's network scores 30.5% against our raw policy over 1,000 games. Our network is much stronger on its own, so Canopy's case rests on its search.]

*Question.* Network against network: how does Canopy's policy without search do against our raw policy?

*Setup.* Row 3 of `canopy-bench-2p.yaml`, played as a standalone eval without the W&B mirror: Canopy at `sims=0` (argmax over its legal moves) against `puffer-graph2/public@e3600` as a plain policy. 1,000 games on the CPU build, 8 minutes.

*Result.* Canopy's share is *30.5* #ci[27.7][33.4] (`1c916131`): 34% from the first seat, 27% from the second. 0 fallbacks; 15 one-card Years of Plenty it could not express.

*Reading.* Our raw policy is the stronger network by a wide margin. Two things lean against Canopy:
- its raw policy reads the opponent's unplayed dev cards as an expectation, where self-play encoded the exact hand (its search deals hands; its root input does not);
- it never saw our dice or robber rules.
Its 1600-sim search is what the other rows test.

#repro(```sh
OMP_NUM_THREADS=1 catanrl eval --suite heldout-2p --boards 500 --device cpu --threads 2 --num-envs 32 \
  "CANOPY:sims=0:procs=2" "POLICY:ckpt=puffer-graph2/public@e3600"
```)

=== Row 1 on CPU: Canopy at 1600 sims vs our raw policy, 60 games #chip("stopped") <exp:canopy-row1-cpu>
#meta(
  dates: "2026-10-23 09:55–13:06",
  owner: "marco",
  config: "configs/experiments/canopy-bench-2p.yaml",
  wandb: (("canopy-bench-2p-main-43b0309d", "canopy-bench-2p, eval only"),),
  records: ("00291a24",),
)
#tldr[With its released 1600-sim search, Canopy scored 45.0% against our raw policy over 60 games. That only rules out a blowout either way; the 1,000-game GPU row (@exp:canopy-gpu-rows) replaces it.]

*Question.* Does Canopy's full search beat our raw policy?

*Setup.* Canopy at 1600 sims (`c_puct` 2.5, `c_visit` 50, `c_scale` 1, no Dirichlet noise) against `puffer-graph2/public@e3600`, on the first 30 boards of `heldout-2p` in both seat orders; CPU build, 3 h 9 min.
- The CPU config sized its rows to Canopy's CPU cost: 60 games for row 1, 40 for rows 2, 4 and 5, and 1,000 for row 3.
- Marco, rightly: "why would we play 60 games? how are we getting stat sig results with that?" 60 games is ±12.6 pp and 40 is ±15.5, so those rows could only detect a blowout.
- The GPU build (@exp:canopy-build) makes 1,000-game rows affordable, and the CPU run stopped after this row.

*Result.* Canopy's share is *45.0* #ci[33.1][57.5] (`00291a24`): 40% from the first seat, 50% from the second. 0 fallbacks, 0 refused moves.

*Reading.* Neither side crushes the other, and 60 games show nothing finer. The row is underpowered and kept only as a blowout check.

#repro(```sh
# As launched: the CPU config plays row 1 first, then goes on to its 40-game rows
OMP_NUM_THREADS=1 nice -n 15 catanrl run configs/experiments/canopy-bench-2p.yaml
# Row 1 alone
OMP_NUM_THREADS=1 catanrl eval --suite heldout-2p --boards 30 --device cpu --threads 2 --num-envs 32 \
  "CANOPY:sims=1600:c_puct=2.5:c_visit=50:c_scale=1:dirichlet_epsilon=0:procs=4" \
  "POLICY:ckpt=puffer-graph2/public@e3600"
```)

=== Row 2 on GPU: Canopy at 1600 sims vs our 2048-sim Gumbel search #chip("done") <exp:canopy-search-gpu>
#meta(
  dates: "2026-10-23 13:41 – 2026-10-24 00:54",
  owner: "marco",
  config: "configs/experiments/canopy-search-2p-gpu.yaml",
  wandb: (("canopy-search-2p-gpu-main-c0e10107", "canopy-search-2p-gpu, eval only"),),
  records: ("96d94fab",),
)
#tldr[Search against search over 1,000 games on the GPU build: Canopy scores 33.0%, so our 2048-sim search wins two games in three. Canopy is weakest in short games and level in the longest third.]

*Question.* How does Canopy at 1600 sims do against our raw `public` network inside the 2048-sim Gumbel search of `831eea84` (@sec:search)?

*Setup.* 1,000 games: ±3.1 pp, about 80% power for a 4–5 pp gap from 50%.
- Marco asked for this row soonest, so it ran on its own ahead of the other GPU rows. It has the same record id as row 2 of the full GPU config, which will find it and skip it.
- Our side's spec resolves identically to `831eea84`'s.
- The critic refit paused at e266 to free the GPU (@sec:critic).

*Result.* Canopy scores *33.0* #ci[30.2][36.0] (`96d94fab`, n = 1,000): 36.8% from the first seat, 29.2% from the second. By game length it scores 18.0% in the shortest third (40–72 turns), 30.6% in the middle (72–85) and 50.3% in the longest (85–141). 0 fallbacks and 0 truncated games; 5 one-card Years of Plenty it could not express. Once, our engine refused a move Canopy proposed (an end turn from a reused tree), and a fresh search gave a legal one.

*Reading.* Our 2048-sim search beats Canopy's 1600-sim search by more than it beats our own raw policy (62.4%, `831eea84`), and our raw network beats Canopy's by a similar margin (69.5%, `1c916131`). Under our rules our bot is clearly stronger at both levels. Canopy trained on Colonist's 1v1 rules (friendly robber, balanced dice), which this harness does not play. Losing the short games and holding even in long ones fits being worst early, where those rules matter most; that is a reading, not a test.

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
#tldr[The rest of the benchmark at full size: Canopy's search against our raw policy, the deployed 1v1 bot and F, about 19 h on a free GPU. Each comparison can detect a gap of 2–5 pp. Parked after row 2 on a rules question for Marco (see Status).]

*Question.* Is Canopy's full search stronger than our raw policy and our deployed bot, and how does it fare against F next to our own bots?

*Setup.* Rows 1, 5 and 4 of @tab:canopy-rows in that order, then row 2, which the config finds already played.
- 1,000 games each on `heldout-2p`'s first 500 boards, both seat orders.
- About 19 h on a free GPU, with the critic refit still paused: the two do not fit in 8 GB together.
- The config is on main since 2026-10-24. Its records will be `67e147c4` (row 1), `aeaab0be` (row 5) and `9f0bafd7` (row 4).

#figure(
  btable(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    header: ([Comparison], [What it says], [Detectable gap]),
    [Row 1 vs 50%], [Whether Canopy's search beats our raw policy], [≈ 4.4 pp],
    [Rows 1 vs 2, paired], [What our search buys against Canopy], [≈ 5.2 pp],
    [Row 1 vs `1c916131`], [What Canopy's search buys over its raw network], [≈ 5 pp],
    [Row 4 vs our policy's 95.6 \ #sub[vs F on the same boards]], [Canopy against our policy, both measured on F], [≈ 2.2 pp],
  ),
  caption: [The gap each comparison finds 80% of the time (two-sided 5%) at 1,000 games a row.],
)

*Status.* Parked (2026-10-24). OpenHex's 1v1 queue plays a friendly robber and balanced dice, Canopy's training rules, so these rows may be more useful on those rules, against the 1v1 bot retrained for them (@sec:recipe). Marco decides.

#repro(```sh
# runtime.canopy_dir names runs/_canopy-cuda, so no CATANRL_CANOPY_DIR is needed
OMP_NUM_THREADS=1 nice -n 5 catanrl run configs/experiments/canopy-bench-2p-gpu.yaml
```)
