#import "../macros.typ": *

= benchmark against released canopy <sec:canopy>

marco asked for a benchmark of the released canopy against our 2p bot "before anything else". canopy is a rust alphazero that calls its catan bot the strongest public 1v1 agent, with no published win rates (@sec:az). so far its raw network is far weaker than ours (30.5%), and its full 1600-sim search played our raw policy about even over 60 games, too few to say more. on canopy's gpu build its search scores 33.0% against our 2048-sim search over 1,000 games, so our search wins two games in three. the other three rows are held: marco is choosing between our current rules and openhex's 1v1 rules (friendly robber, canopy's own robber rule) against the retrained 1v1 bot.

#figure(
  btable(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, left, right, right, left, left),
    header: ([row], [canopy plays], [games], [canopy's share], [record], [status]),
    [1], [1600 sims vs our raw policy (`puffer-graph2/public@e3600`)], [60 (cpu)], [45.0 #ci[33.1][57.5]], [`00291a24`], [stopped: blowout check only],
    [1], [the same, on the gpu build], [1,000], [—], [`67e147c4` (to be written)], [held],
    [2], [1600 sims vs our 2048-sim gumbel search on that network], [1,000], [33.0 #ci[30.2][36.0]], [`96d94fab`], [done],
    [3], [its raw network (no search) vs our raw policy], [1,000 (cpu)], [30.5 #ci[27.7][33.4]], [`1c916131`], [done],
    [4], [1600 sims vs F, the common reference], [1,000], [—], [`9f0bafd7` (to be written)], [held],
    [5], [1600 sims vs the deployed 1v1 bot (`BELIEF` on `puffer-graph2/full@e3350`, 16 worlds)], [1,000], [—], [`aeaab0be` (to be written)], [held],
  ),
  caption: [the canopy benchmark at a glance. rows keep the numbers of the original five-row plan. every 1,000-game row plays the first 500 boards of `heldout-2p` in both seat orders (the boards of `831eea84`), so the rows pair game by game. canopy always sits in slot 0, so a record's score rate is canopy's share.],
) <tab:canopy-rows>

== setup

=== the canopy bridge: fairness, cpu and gpu builds #chip("done") <exp:canopy-build>
#meta(
  dates: "2026-10-23",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[released canopy plays in our own harness through a bridge to its rust search. the bridge shows canopy only what a fair player sees. the gpu build plays the same moves as the cpu build at 5.3 searched decisions a second against 0.2, which makes 1,000-game rows affordable.]

*question.* can released canopy play in our harness fairly, under our rules, cheaply enough for 1,000-game rows?

*setup.* released canopy (`cullback/canopy` at `6185983`, nexus-v3 iteration 315) plays as a `CANOPY` agent: a bridge process to its rust search, built from v1’s bridge. four patches sit under `third_party/canopy/` with a reproducible build. two are v1’s: the per-move search budget and the bridge. two are ours: the bridge sends canopy only the count of the opponent's hidden dev cards (v1’s sent the true cards), and tree reuse is exact. the rules are ours: 15 vp, discard above 9, no friendly robber, random dice. canopy trained with a friendly robber and balanced dice, so those inputs are new to it. it cannot express a one-card year of plenty (counted when offered). the gpu build is canopy's cuda backend on the same commit, patches and checkpoint.

*result.*
#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([check], [result]),
    [hidden information], [on 8,510 audited positions the bytes canopy received did not change when the opponent's hidden cards were redealt (in 2p the resource hands are public anyway). redealing victory point cards is audited too.],
    [tree reuse], [canopy re-roots only within its own turn and through forced moves; about half its decisions keep visits. less reuse than it gets natively, probably a small handicap.],
    [smokes (4 games a row, plumbing only)], [0 fallbacks, 0 moves our engine refused, 0 rule mismatches. canopy won 4 of 4 against our raw policy, twice.],
    [cpu cost], [5–7 cpu-seconds a decision at 1600 sims. the five-row cpu plan was about 14 h on 4 cores.],
    mid,
    [gpu vs cpu, same positions], [on 300 positions at 1600 sims the gpu build picks the cpu build's move 300 of 300 times. root values differ by 2.5 × 10⁻⁴ on average, about 20 times less than changing the search seed. outputs do not depend on how many games are batched together.],
    [gpu vs cpu, whole games], [the same player statistically, not bit for bit: 29 of 40 smoke games identical, 2 winners changed. so gpu and cpu records key apart.],
    [gpu throughput], [4 bridges of 256 games each: about 5.3 searched decisions a second while sharing the gpu with the critic refit, against 0.2 for one cpu bridge.],
    [gpu pinning], [the cuda binary cannot be rebuilt byte for byte (a hash map in the kernel compiler), so the installed copy is the one the records name. it installs only as that pinned binary, checked by a behavioural fingerprint (64 fixed positions).],
  ),
  caption: [checks on the canopy bridge. the cuda build was merged on its own as `b385ea9` (code ids unchanged).],
)

*reading.* the bridge is fair on hidden information and the gpu build is the same player as the cpu build. what still leans against canopy: our dice and robber rules, the one-card year of plenty, and half its native tree reuse.

#repro(```sh
# cpu build: reproducible, gives the binary_sha256 pinned in
# third_party/canopy/canopy-bridge.manifest.json. SRC is a new directory; the checkpoint
# is model_iter_315.mpk of canopy's catan-nexus-v3 release
third_party/canopy/build_bridge.sh --backend cpu /tmp/canopy-src runs/_canopy model_iter_315.mpk
# cuda: not bit-reproducible, so keep the installed runs/_canopy-cuda
# (binary 1f7cc246...68802689); a rebuild is a new player for the record keys
```)

== results

=== canopy's raw network vs ours #chip("done") <exp:canopy-raw-vs-raw>
#meta(
  dates: "2026-10-23",
  owner: "marco",
  config: "configs/experiments/canopy-bench-2p.yaml",
  wandb: (),
  records: ("1c916131",),
)
#tldr[without search, canopy's network scores 30.5% against our raw policy over 1,000 games. our network is much stronger on its own, so canopy's case rests on its search.]

*question.* network against network: how does canopy's policy without search do against our raw policy?

*setup.* row 3 of `canopy-bench-2p.yaml`, played as a standalone eval with no wandb mirror: canopy with `sims=0` (argmax over its legal moves) against `puffer-graph2/public@e3600` as a plain policy. 1,000 games, cpu build, 8 minutes.

*result.* canopy's share is *30.5* #ci[27.7][33.4] (`1c916131`): 34% from the first seat, 27% from the second. 0 fallbacks; 15 one-card years of plenty it couldn't express.

*reading.* our raw policy is the stronger network by a wide margin. two things lean against canopy. its raw policy reads the opponent's unplayed dev cards as an expectation, where self-play encoded the exact hand (its search deals hands; its root input doesn't). and it never saw our dice or robber rules. its 1600-sim search is what the other rows test.

#repro(```sh
OMP_NUM_THREADS=1 catanrl eval --suite heldout-2p --boards 500 --device cpu --threads 2 --num-envs 32 \
  "CANOPY:sims=0:procs=2" "POLICY:ckpt=puffer-graph2/public@e3600"
```)

=== row 1 on cpu: canopy 1600 sims vs our raw policy, 60 games #chip("stopped") <exp:canopy-row1-cpu>
#meta(
  dates: "2026-10-23 09:55–13:06",
  owner: "marco",
  config: "configs/experiments/canopy-bench-2p.yaml",
  wandb: (("canopy-bench-2p-main-43b0309d", "canopy-bench-2p, eval only"),),
  records: ("00291a24",),
)
#tldr[canopy with its released 1600-sim search scored 45.0% against our raw policy over 60 games. that only rules out a blowout either way. the 1,000-game gpu row (@exp:canopy-gpu-rows) replaces it.]

*question.* does canopy's full search beat our raw policy?

*setup.* canopy at 1600 sims (`c_puct` 2.5, `c_visit` 50, `c_scale` 1, no dirichlet noise) against `puffer-graph2/public@e3600`, on the first 30 boards of `heldout-2p` in both seat orders, cpu build, 3 h 9 min. the cpu config sized its rows to canopy's cpu cost: 60 games for row 1, 40 for rows 2, 4 and 5, 1,000 for row 3. marco, rightly: "why would we play 60 games? how are we getting stat sig results with that?" 60 games is ±12.6 pp and 40 is ±15.5, so those rows could only detect a blowout. the gpu build (@exp:canopy-build) makes 1,000-game rows affordable, and the cpu run stopped after this row.

*result.* canopy's share is *45.0* #ci[33.1][57.5] (`00291a24`): 40% from the first seat, 50% from the second. 0 fallbacks, 0 refused moves.

*reading.* neither side crushes the other, and nothing finer can be read from 60 games. this is an underpowered row, kept only as a blowout check.

#repro(```sh
# as launched: the cpu config plays row 1 first, then goes on to its 40-game rows
OMP_NUM_THREADS=1 nice -n 15 catanrl run configs/experiments/canopy-bench-2p.yaml
# row 1 alone
OMP_NUM_THREADS=1 catanrl eval --suite heldout-2p --boards 30 --device cpu --threads 2 --num-envs 32 \
  "CANOPY:sims=1600:c_puct=2.5:c_visit=50:c_scale=1:dirichlet_epsilon=0:procs=4" \
  "POLICY:ckpt=puffer-graph2/public@e3600"
```)

=== row 2 on gpu: canopy 1600 sims vs our 2048-sim gumbel search #chip("done") <exp:canopy-search-gpu>
#meta(
  dates: "2026-10-23 13:41 – 2026-10-24 00:54",
  owner: "marco",
  config: "configs/experiments/canopy-search-2p-gpu.yaml",
  wandb: (("canopy-search-2p-gpu-main-c0e10107", "canopy-search-2p-gpu, eval only"),),
  records: ("96d94fab",),
)
#tldr[canopy's full search against our full search, 1,000 games on the gpu build: canopy scores 33.0%, so our 2048-sim search wins two games in three. canopy is weakest in short games and even in the longest third.]

*question.* search against search: canopy at 1600 sims against our raw public network inside the 2048-sim gumbel search of `831eea84` (@sec:search).

*setup.* marco asked for this row soonest, so it runs on its own ahead of the other gpu rows. it has the same record id as row 2 of the full gpu config, which will find it and skip it. our side's spec resolves identically to `831eea84`’s. 1,000 games: ±3.1 pp, about 80% power for a 4–5 pp gap from 50%. the critic refit paused at e266 to free the gpu (@sec:critic).

*result.* canopy scores *33.0* #ci[30.2][36.0] (`96d94fab`, n = 1,000; 36.8% from the first seat, 29.2% from the second). by game length it scores 18.0% in the shortest third (40–72 turns), 30.6% in the middle (72–85) and 50.3% in the longest (85–141). 0 fallbacks and 0 truncated games; once our engine refused a move canopy proposed (a reused tree's end turn) and a fresh search gave a legal one; 5 one-card years of plenty it couldn't express.

*reading.* our 2048-sim search beats canopy's 1600-sim search by more than it beats our own raw policy (62.4%, `831eea84`), and our raw network beats canopy's by a similar margin (69.5%, `1c916131`). so under our rules our bot is clearly stronger at both levels. canopy trained on colonist's 1v1 rules (friendly robber, balanced dice), which this harness doesn't play; losing the short games and holding even in long ones fits being worst early, where those rules matter most. that is a reading, not a test.

#repro(```sh
CATANRL_CANOPY_DIR=$PWD/runs/_canopy-cuda OMP_NUM_THREADS=1 nice -n 5 \
  catanrl run configs/experiments/canopy-search-2p-gpu.yaml
```)

=== rows 1, 5 and 4 on gpu, 1,000 games each #chip("parked") <exp:canopy-gpu-rows>
#meta(
  dates: "2026-10-23",
  dates-label: "queued",
  owner: "marco",
  config: "configs/experiments/canopy-bench-2p-gpu.yaml",
  wandb: (),
  records: (),
)
#tldr[the rest of the benchmark at full size: canopy's search against our raw policy, the deployed 1v1 bot and F. about 19 h on a free gpu. held after row 2 (see status). each comparison can detect a gap of 2–5 pp.]

*question.* is canopy's full search stronger than our raw policy and our deployed bot, and how does it fare against F next to our own bots?

*setup.* rows 1, 5 and 4 of @tab:canopy-rows in that order, then row 2, which the config finds already played. 1,000 games each on `heldout-2p`'s first 500 boards, both seat orders. about 19 h on a free gpu, with the critic refit still paused: the two don't fit in 8 gb together. the config is on main since 2026-10-24. its records will be row 1 `67e147c4`, row 5 `aeaab0be`, row 4 `9f0bafd7`.

#figure(
  btable(
    columns: (auto, auto, auto),
    align: (left, left, right),
    header: ([comparison], [what it says], [detectable gap]),
    [row 1 vs 50%], [whether canopy's search beats our raw policy], [≈ 4.4 pp],
    [rows 1 vs 2, paired], [what our search buys against canopy], [≈ 5.2 pp],
    [row 1 vs `1c916131`], [what canopy's search buys over its raw network], [≈ 5 pp],
    [row 4 vs our policy's 95.6 vs F on the same boards], [canopy against our policy, both measured on F], [≈ 2.2 pp],
  ),
  caption: [the gap each comparison finds 80% of the time (two-sided 5%) at 1,000 games a row.],
)

*status.* held (2026-10-24). openhex's 1v1 queue plays friendly robber and balanced dice, canopy's training rules, so these rows may be more useful on those rules against the 1v1 bot retrained for them (@sec:recipe). marco decides.

#repro(```sh
# runtime.canopy_dir names runs/_canopy-cuda, so no CATANRL_CANOPY_DIR
OMP_NUM_THREADS=1 nice -n 5 catanrl run configs/experiments/canopy-bench-2p-gpu.yaml
```)
