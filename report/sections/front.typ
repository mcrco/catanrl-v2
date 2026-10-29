#import "../macros.typ": *

= goal and current best bots <sec:front>

the goal is a recipe for the best catan bot out there, for any number of players on any map. v1 (#link("https://github.com/mcrco/catanrl")[catanrl]) got a two-player bot to about 90% against catanatron's `ValueFunctionPlayer` (F). v2 rewrote the stack around a C++ engine and pufferlib, and its raw policy is past 95% in 2p from scratch. the bottom line so far: self-play PPO on a graph network gives a strong raw policy, and a search on top wins 53.4–62.4% of games against that policy. two alphazero runs on top of it made the network worse, not better (@sec:az).

most experiments are a config under `configs/experiments/`. a number without an eval record says where it comes from: a training metric, or a json under `report/data/`.

#callout[
  *marco's recipe, in two lines.* implement the engine in C++, turn it into a pufferlib env, run for about 1B steps, and get a really good shared-trunk critic + policy that beats `ValueFunctionPlayer` about 90% of the time. put the critic + policy inside an MCTS harness, with a simple probabilistic belief-sampling determinization, to get another few points head to head over the raw policy.

  *measured.* the raw policy is at 95% vs F (@sec:recipe). the search edge is 53.4–55.2% with our default PUCT search and 62.4% with gumbel alphazero's root at 2048 sims (@sec:search).
]

*latest (29 october 2026).* marco's search depth sweep (MCTS at 64 to 16384 sims against the same checkpoint's raw policy, five checkpoints of one run from scratch, 1,000 games a cell): up to 1024 sims every step up in sims helps every checkpoint, and the weaker the net the more search adds (64 → 1024 sims: +18 points for the e50 net, +5.5 for the final one); 4096 is running (@exp:search-depth-2p). the deployed 1v1 net, continued for 1b steps on openhex's 1v1 rules (friendly robber, balanced dice), beats its start 63.2% and F 98.8%; a second 1b steps at its final, lowest learning rate gained little: the best checkpoint is level with it on held-out boards (51.0% head to head, 98.9% vs F), so the net has roughly leveled off. a try that restarted the learning rate at its peak fell to \~41% against its start and was stopped (@sec:recipe). the critic refit came back null (@sec:critic). discount vs no discount, from scratch with two seeds per arm: the discount makes games end faster (without it, games against F run 4.6 and 2.6 turns longer on the two seeds, paired over 5,000 games, against half a turn between two seeds of one arm), and it doesn't change strength: head to head the arms are level, 49.98% over 10,000 games (@exp:recipe-discount). canopy's search lost to our search 33.0 to 67.0 over 1,000 games (@sec:canopy). deployment: the openhex net is released for openhex (@exp:deploy-openhex-v2); it also beats the site's 1v1 bot on the site's rules, 55.8% (@exp:deploy-site-rules-2p), and the site's server now offers openhex's friendly robber and balanced dice (@exp:deploy-site-openhex-rules).

== current best bots

#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, left, right, right, left),
    header: ([agent], [opponent], [games], [score], [record]),
    table.cell(colspan: 5)[*2 players, BASE, 15 vp*],
    [raw policy at the `public` level (`puffer-graph2/public@e3600`): the fair raw bot, @sec:recipe], [F], [5,000], [*94.96* #ci[94.3][95.5]], [`dc8a611a`],
    [`full` net, belief-averaged (`BELIEF`, 16 worlds, `puffer-graph2/full@e3350`): the deployed 1v1 bot], [F], [5,000], [*95.06* #ci[94.42][95.63]], [`67869e7a`],
    table.cell(colspan: 5)[*2 players, openhex's 1v1 rules (friendly robber, balanced dice)*],
    [`ppo-2p-fr/full@e7600`, raw (`full` net continued 1b steps, @sec:recipe)], [F], [5,000], [*98.84* #ci[98.50][99.10]], [`e435b80e`],
    [the same, raw], [its start, `puffer-graph2/full@e3350`], [5,000], [*63.2* #ci[61.8][64.5]], [`36bd2384`],
    [`full` net, raw (unfair: sees F's dev cards)], [F], [5,000], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
    [2048-sim gumbel MCTS, one world, on `puffer-graph2/public@e3600` (@sec:search)], [its own raw policy], [1,000], [*62.4* #ci[59.4][65.3]], [`831eea84`],
    [released canopy's raw network (@sec:canopy)], [our raw `public` policy], [1,000], [30.5 #ci[27.7][33.4]], [`1c916131`],
    [released canopy's 1600-sim search (@sec:canopy)], [our 2048-sim gumbel MCTS], [1,000], [33.0 #ci[30.2][36.0]], [`96d94fab`],
    mid,
    table.cell(colspan: 5)[*4 players, BASE, 10 vp* (score share, parity 25)],
    [raw policy (`ppo-4p/main@e3650`, @sec:recipe)], [F, F, F], [4,000], [*78.55* #ci[77.25][79.79]], [`e5b302ee`],
    [MCTS 1024 sims, one world (`ppo-4p-critic-fix/lambda-critic-g1@e100`, @sec:search)], [three raw copies], [2,000], [26.9 #ci[25.0][28.9]], [`657b314d`],
    [trade proposer (`TRADE`, `ppo-4p/main` in every seat; the no-trade control scores 25.0, paired +16.2 pp; @sec:trading)], [three accepting copies], [4,000], [*41.2* #ci[39.7][42.7]], [`c4900f87` \ `7164d396`],
    mid,
    table.cell(colspan: 5)[*one model for 2–4 players and any map* (`ppo-mixed/main@e3550`, @sec:generalist)],
    [2p, BASE], [F], [5,000], [92.3 #ci[91.6][93.0]], [`b2690265`],
    [4p, BASE (share)], [F, F, F], [4,000], [77.0 #ci[75.7][78.3]], [`0f448f68`],
    [4p, blob maps it never trained on (share)], [F, F, F], [4,000], [73.9 #ci[72.5][75.3]], [`7c17d0ae`],
    mid,
    table.cell(colspan: 5)[#aside[*v1 reference* (graph-s, DAgger + MARL PPO, 73.7M steps), for history; see @exp:v1-reference]],
    [#aside[raw policy, belief-averaged]], [#aside[F]], [#aside[2,000]], [#aside[89.65 #ci[88.24][90.91]]], [#aside[v1 report]],
    [#aside[MCTS 256, determinized]], [#aside[F]], [#aside[1,000]], [#aside[90.5 #ci[88.5][92.2]]], [#aside[v1 report]],
    [#aside[MCTS 1024 on the true game (unfair)]], [#aside[F]], [#aside[1,000]], [#aside[92.7 #ci[90.9][94.2]]], [#aside[v1 report]],
  ),
  caption: [the best agent per setting, on held-out boards. 2p scores are win rates; 4p scores are slot 0’s score share. v2 trains from scratch (no DAgger) and lands 5 points above v1’s warm-started reference in 2p. hidden information is worth under a point in 2p: the fair `public` and belief-averaged nets are level with the raw `full` net. 4p is trained and played at the `private` level, the only fair level past two players.],
) <tab:best>

== how to read this log

sections are ideas, not dates. each experiment is one block:
- a status chip: *done*, *running*, *queued*, *stopped* (ended early), *todo* (planned, nothing run) or *parked* (planned, set aside);
- a meta line: when it ran, whose idea it was, its config, its wandb runs and the eval records that hold its numbers;
- a short summary, then the question, the setup, the result and how to read it;
- the command that reproduces it.

dates appear only in the meta lines. marco's ideas are credited to him. ideas of mine that were never run are marked "(my suggestion, not queued)".

*records.* an 8-character hex id in backticks (`831eea84`) names an eval record: it is the start of the record's eval id. a record lives under `runs/<experiment>/<variant>/evals/<suite>/`, or under `runs/_evals/<suite>/` for an eval-only row. `catanrl verify <eval id>` replays a record's games and checks them against it. it takes any unique prefix of the id, or the record's path. a record stored in several run dirs (the critic fix's start row `60cd10b6` sits in all four arms) needs the path.

*commands.* every command runs from the repo root. `catanrl run <config>` runs an experiment's variants, in dependency order, with their finals; `--only <glob>` picks variants. an eval-only row is one `catanrl eval` line, rebuilt from the record's agent specs.

*wandb.* the meta line links each run on wandb (project link at the top). runs before 2026-10-17 were backfilled from `runs/*/metrics.jsonl` and the eval records. since 2026-10-17 a live mirror (`catanrl.wandb_mirror`) logs every run whose config sets `meta.wandb.project`, and `tools/wandb_backfill.py` re-syncs a run directory into the same wandb run. the record on disk is the source of truth; a wandb failure is only a warning. a run logs losses, entropy, KL, clip fraction, game length, seat win rates, throughput, the select curve and the critic's calibration (`critic/value_sum_*`: the seats' predicted win probabilities summed, which should match the share of games somebody wins). a link labelled "eval only" is an eval run: its wandb page shows no finals, and its results are in the records. a link tagged "no local mirror" is computed from the run's config hash; no local wandb directory confirms that it was synced.

== setup

*engine.* cppanatron, a C++20 port of catanatron kept in lockstep parity with it: 2–6 players, explicit maps of any shape. the observation encoder, the action codec, the threaded multi-game env, the hand tracker and the batched MCTS all live in C++, so training, search and deployment can't drift apart. the network reads one float32 row per (game, seat), seat-relative and padded to the env's maxima: `[global | players | tiles | nodes | edges | topology | action mask]`.

*observation levels.* how much of the other seats' hidden state a row shows.
- `private`: what the seat itself sees. the encoder zeroes the other seats' 15 hidden features (5 resource ratios, actual VPs, 5 dev-card counts, 4 playable flags). that is exactly what a sampled world could change, so averaging a `private` policy over sampled worlds gives back the plain argmax policy.
- `public`: in 2p every resource movement is seen by both seats, so the opponent's exact hand is a perfect card counter's knowledge. a `public` policy is fair in 2p without belief averaging.
- `full`: adds the opponent's dev cards in hand. played raw, it is unfair.

*agents.* each player is a spec string.
- `POLICY:ckpt=<run>@<epoch>`: the raw network, argmax.
- `BELIEF:ckpt=…:worlds=16`: the raw policy averaged over 16 worlds sampled from the hand tracker's belief. it is the fair way to play a network trained on more than its seat sees.
- `MCTS:ckpt=…:sims=N:dets=K`: a batched N-player PUCT search over K worlds sampled from the tracker at the root, with per-seat values backed up max#super[n] and exact chance nodes for dice, dev-card draws and steals. `oracle=true` searches the true game and is only a reference.
- `F` is catanatron's `ValueFunctionPlayer`; `R` plays uniformly random legal moves.
- `<run>@eN` is the run's checkpoint at epoch N; `@best` is the one its select suite chose, `@latest` the last one saved.

*the slot after an F.* F (`enemies=next`) robs and blocks the seat after it. a 4p suite seats `lineup[(seat + r) mod 4]`, so slot 0 always moves right after the F in slot 3 and pays for it. any 4p table of the shape `[A, B, F, F]` therefore needs both slot orders, or the two learner slots summed (@sec:hidden has the diagnosis).

*rules.* 2p is `base-2p`: the BASE map, 15 vp, discard above 9 cards. 4p is `base-4p`: BASE under the standard rules, 10 vp, discard limit 7, which is also the engine's default. canopy trained with a friendly robber and balanced dice; we use neither.

*boards.* every number is on boards training never saw; `catanrl validate` checks the training pools against the suites.
- `heldout-2p`: 2,500 boards (seed 67) in both seat orders, 5,000 games.
- `heldout-4p`: 1,000 boards (seed 68) in 4 rotations, 4,000 games.
- `heldout-random-4p`: 1,000 blob-shaped boards of 15–30 tiles (seed 69) in 4 rotations. `heldout-mixed` mixes 2p and 3p BASE with 4p blobs.
- the `hexagon` generator (training only) keeps the BASE shape with shuffled numbers and generated harbors.
- checkpoints are chosen every 50 epochs on separate `select-*` suites, so the held-out number never picked its own checkpoint. `select-2p` is 200 boards (seed 123, 400 games) and `select-4p` 800 games. `select-mixed` is 2, 3 and 4 players on BASE, 100 boards each in every seat rotation (900 games, seed 126).
- the alphazero and critic-refit runs select on `select-2p`'s seed (123) in both seat orders: `select-2p-2k` is 2,000 boards (4,000 games), `select-2p-500` its first 500 boards (1,000 games).

*measurement.* intervals are wilson 95%, printed small and grey after the number (#ci[lo][hi]). pp means percentage points. in N-player games the number is slot 0’s _score share_: a win counts 1 and a truncated game gives 1/N to every seat, so parity is 1/N. agents that use information a player doesn't have (a raw `full` policy, an oracle search) are marked unfair; everything else is fair.

*paired statistics.* two rows played on the same game ids pair game by game: mcnemar over the shared ids (`tools/search_probe.py --pair`), with holm when a decision rests on two tests. 4p tables played in both slot orders use a bootstrap over board × rotation pairs. power figures are for 80% power at a two-sided 5% level.

== v1, for reference

=== v1 reference rows #chip("done") <exp:v1-reference>
#meta(
  dates: "in v1 (catanrl), before v2",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[v1’s best 2p numbers, kept for history and never rerun: its belief-averaged raw policy scored 89.65 vs F, and its search beat that policy 51.1–58.1% depending on sims. v2 from scratch is 5 points higher against F.]

*question.* what did v1’s best 2p bot score, as a yardstick for v2?

*setup.* v1 (#link("https://github.com/mcrco/catanrl")[catanrl]): `graph-s`, DAgger from F, then MARL PPO, 73.7M steps. the numbers come from the v1 report and are never paired with v2’s.

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, left, right, right),
    header: ([agent], [opponent], [games], [score]),
    [raw policy, belief-averaged], [F], [2,000], [89.65 #ci[88.24][90.91]],
    [MCTS 256, determinized], [F], [1,000], [90.5 #ci[88.5][92.2]],
    [MCTS 1024 on the true game (unfair)], [F], [1,000], [92.7 #ci[90.9][94.2]],
    mid,
    [MCTS 256 / 1024 / 4096 sims], [its belief-averaged raw policy], [1,000], [51.1 / 55.1 / 58.1],
  ),
  caption: [v1’s reference rows, from the v1 report.],
)

*reading.* two more v1 results that later sections lean on. v1 never trained alphazero above 512 sims, and its frozen-teacher distillation of an 8 × 64-sim search (EX-1d) gave +1.3 vs F and 50.6% head to head, both null (@sec:az). v1 also ran canopy's puct recipe on a 55%-vs-F graph net and raised it to 70.5% raw vs F over 10 iterations, but never from a start above 90% (@sec:canopy).

#repro(```sh
# not reproducible in this repo: numbers from the v1 report (github.com/mcrco/catanrl)
```)
