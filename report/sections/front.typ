#import "../macros.typ": *

= Goal and current best bots <sec:front>

The goal is a recipe for the best Catan bot, for any number of players on any map. v1 (#link("https://github.com/mcrco/catanrl")[catanrl]) reached about 90% against Catanatron's `ValueFunctionPlayer` (F) in 2p. v2 rebuilt the stack around a C++ engine and PufferLib; its raw policy, trained from scratch, wins over 95%.

So far: self-play PPO on a graph network gives a strong raw policy, and search on top wins 53.4–62.4% of games against that same policy. Two AlphaZero runs on top made the network worse (@sec:az). Most experiments are a config under `configs/experiments/`; a number without an eval record names its source (a training metric, or a JSON file under `report/data/`).

#callout[
  *Marco's recipe.* Implement the engine in C++, turn it into a PufferLib env, run for about 1B steps, and get a really good shared-trunk critic + policy that beats `ValueFunctionPlayer` about 90% of the time. Put the critic + policy inside an MCTS harness, with a simple probabilistic belief-sampling determinization, to get another few points head to head over the raw policy.

  *Measured.* The raw policy wins 95% against F (@sec:recipe). Search beats it 53.4–55.2% with our default PUCT search and 62.4% with Gumbel AlphaZero's root at 2048 sims (@sec:search).
]

== Latest results (11 November 2026)

- *Bigger 1v1 net* (Marco; @exp:arch-graph-m-2p-lr). `graph-m` (2.32M parameters, 4.3× `graph-s`), from scratch for 3B steps on OpenHex's 1v1 rules at half `graph-s`'s learning rate, beats fr3, the deployed 1v1 net, 56.5% over 5,000 held-out games (±1.4 pp, p = 3e-20) and F 99.4%. `graph-s` from scratch under the same recipe ended at 44.2% against fr3 on the same games (+12.3 pp paired, p = 2e-41). Both sides read every hand in these rows; the fair (`BELIEF`) and search rows are next. Now `2p-graph-m-1` and *best-2p*, the yardstick new 1v1 runs play (Marco). The first try at `graph-s`'s learning rate stalled, with policy updates about 5× larger (@exp:arch-graph-m-2p).
- *Transitivity* (Marco; @exp:recipe-transitivity). A round robin of eight checkpoints of one 1v1 run plus fr3, 1,000 games a pair with the fair raw bot: every later checkpoint beats or ties every earlier one, one Bradley–Terry rating fits all 36 pairs (deviance 26.0 on 28 df, p = 0.58) and there are no cycles. Gains shrink: 110 Elo from epoch 250 to 500, 13 from 3,000 to 3,786. The last stretch gained 8 points against fr3 but tied epoch 3,000 head to head.
- *AlphaZero gate* (approved by Marco; @exp:az-gate). fr3 fitted once to three shapes of 2048-sim search targets on 102,400 of its own positions, each judged inside a 512-sim search against fr3's own search on the same 2,000 games: −0.15, −0.25 and −0.65 pp (each ±2.3 pp). Null for all three, the control included, so at this dose the gate cannot see what broke the AlphaZero runs, and no full loop follows.
- *Deploy check* (Marco; @exp:deploy-graph-m-check). `graph-m` against fr3 as fair raw bots: 55.0% over 5,000 games (±1.4 pp). Inside the deployed search at equal time per move, 300 sims against fr3's 1,024 because `graph-m` can't use the fused kernels and costs 3.4× as much per simulation: 54.3% over 2,000 games (±2.2 pp, p = 1e-4). Both pass the rule fixed before launch, so `graph-m` is a deploy candidate; deploying waits for Marco's go.
- *Queued*: 256-step rollout segments and λ = 1, `graph-s` from scratch for 500M steps against the same run at 64 steps and λ = 0.95 (Marco; @exp:recipe-horizon). About 8 h on the GPU, after the fp16 A/B.

- *Search depth sweep* (Marco; @exp:search-depth-2p). MCTS at 64 to 16,384 sims against the same checkpoint's raw policy, five checkpoints of one run, 1,000 games per cell. More search helps every checkpoint at every step, with no plateau by 16,384 sims: each 4× adds 3–7 points (final net: 53.9, 57.0, 60.5% at 1,024, 4,096, 16,384). Weaker nets gain more (82.0% for the epoch-50 net at 4,096; the epoch-150 net 76.4 → 80.7% from 4,096 to 16,384). Paused on 3 November with two 16,384-sim cells (e1500, e50) still to play.
- *Trading between equals* (Marco; @exp:trade-2v2). Two seats that trade with each other, against two copies of the same net that can't, take 61.7% (adjacent) and 62.9% (alternating) of 4p games, against exactly 50 without trading (+11.7 and +12.9 pp paired, 4,000 games a row). Both traders gain, so it isn't exploitation of a naive acceptor. Against F the pair goes from 93 to 96%.
- *Transformer over the whole history* (Marco; @exp:markov-transformer). Ties the LSTM at 4p: +1.1 points head to head (−0.9 to 3.0, 8,000 games). Like the LSTM it beats the no-memory net, by 5.8 points, and takes 65.7% against three F. It trains as fast as the LSTM (23.0k decisions/s, 2.6 h).
- *LLM eval* (@exp:llm-eval). Audited before any paid run: the prompts stated the wrong rules, named robber targets in coordinates the state never used and showed no game history, and nothing capped spending. All fixed; waiting on a model list and a budget.
- *OpenHex rules* (@sec:recipe). The deployed 1v1 net, continued for 1B steps on OpenHex's 1v1 rules (friendly robber, balanced dice), beats its start 63.2% and F 98.8%. A second 1B steps at the lowest learning rate gained little (51.0% head to head, 98.9% vs F): the net has roughly leveled off. Restarting the learning rate at its peak fell to \~41% against the start and was stopped.
- *Discount vs no discount* (@exp:recipe-discount), from scratch, two seeds per arm. The discount makes games end faster (without it, games against F run 4.6 and 2.6 turns longer on the two seeds, paired over 5,000 games, against half a turn between seeds of one arm) but does not change strength: 49.98% head to head over 10,000 games.
- *Critic refit*: null (@sec:critic).
- *Canopy*: its search lost to ours 33.0–67.0 over 1,000 games (@sec:canopy).
- *Deployment*: the OpenHex net is released for OpenHex (@exp:deploy-openhex-v2). It also beats the site's 1v1 bot on the site's rules, 55.8% (@exp:deploy-site-rules-2p), and the site's server now offers OpenHex's friendly robber and balanced dice (@exp:deploy-site-openhex-rules).

== Current best bots

#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, left, right, right, left),
    header: ([Agent], [Opponent], [Games], [Score], [Record]),
    tgroup(5)[2 players, BASE rules, 15 VP (`heldout-2p`)],
    [Raw policy, `public` level: the fair raw bot (@sec:recipe) \ #sub[`puffer-graph2/public@e3600`]], [F], [5,000], [*94.96* #ci[94.3][95.5]], [`dc8a611a`],
    [`full` net, belief-averaged over 16 worlds: the deployed 1v1 bot \ #sub[`BELIEF`, `puffer-graph2/full@e3350`]], [F], [5,000], [*95.06* #ci[94.42][95.63]], [`67869e7a`],
    [`full` net, raw (unfair: sees F's dev cards)], [F], [5,000], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
    [2048-sim Gumbel MCTS, one world (@sec:search) \ #sub[`puffer-graph2/public@e3600`]], [Its own raw policy], [1,000], [*62.4* #ci[59.4][65.3]], [`831eea84`],
    [Released Canopy's raw network (@sec:canopy)], [Our raw `public` policy], [1,000], [30.5 #ci[27.7][33.4]], [`1c916131`],
    [Released Canopy's 1600-sim search (@sec:canopy)], [Our 2048-sim Gumbel MCTS], [1,000], [33.0 #ci[30.2][36.0]], [`96d94fab`],
    tgroup(5)[2 players, OpenHex 1v1 rules: friendly robber, balanced dice (`heldout-2p-fr`)],
    [`graph-m` from scratch for 3B steps, raw `full` net (@exp:arch-graph-m-2p-lr) \ #sub[`2p-graph-m-1`, `graph-m-2p-lr/main@e22711`]], [F], [5,000], [*99.42* #ci[99.17][99.60]], [`d664995b`],
    [The same], [fr3, the deployed 1v1 net, raw], [5,000], [*56.5* #ci[55.1][57.9]], [`914fc0f0`],
    [`full` net continued for 1B steps, raw (@sec:recipe) \ #sub[`ppo-2p-fr/full@e7600`]], [F], [5,000], [98.84 #ci[98.50][99.10]], [`e435b80e`],
    [The same], [Its start, `puffer-graph2/full@e3350`], [5,000], [63.2 #ci[61.8][64.5]], [`36bd2384`],
    tgroup(5)[4 players, BASE rules, 10 VP (`heldout-4p`; score share, parity 25)],
    [Raw policy (@sec:recipe) \ #sub[`ppo-4p/main@e3650`]], [F, F, F], [4,000], [*78.55* #ci[77.25][79.79]], [`e5b302ee`],
    [MCTS, 1024 sims, one world (@sec:search) \ #sub[`ppo-4p-critic-fix/lambda-critic-g1@e100`]], [Three raw copies], [2,000], [26.9 #ci[25.0][28.9]], [`657b314d`],
    [Trade proposer (`TRADE`, @sec:trading); the no-trade control scores 25.0, paired +16.2 pp \ #sub[`ppo-4p/main` in every seat]], [Three accepting copies], [4,000], [*41.2* #ci[39.7][42.7]], [`c4900f87` \ `7164d396`],
    tgroup(5)[One model for 2–4 players and any map (@sec:generalist) #h(0.3em) #sub[`ppo-mixed/main@e3550`]],
    [2p, BASE], [F], [5,000], [92.3 #ci[91.6][93.0]], [`b2690265`],
    [4p, BASE (share)], [F, F, F], [4,000], [77.0 #ci[75.7][78.3]], [`0f448f68`],
    [4p, blob maps it never trained on (share)], [F, F, F], [4,000], [73.9 #ci[72.5][75.3]], [`7c17d0ae`],
    tgroup(5)[#aside[v1 reference: `graph-s`, DAgger + MARL PPO, 73.7M steps (@exp:v1-reference)]],
    [#aside[Raw policy, belief-averaged]], [#aside[F]], [#aside[2,000]], [#aside[89.65 #ci[88.24][90.91]]], [#aside[v1 report]],
    [#aside[MCTS 256, determinized]], [#aside[F]], [#aside[1,000]], [#aside[90.5 #ci[88.5][92.2]]], [#aside[v1 report]],
    [#aside[MCTS 1024 on the true game (unfair)]], [#aside[F]], [#aside[1,000]], [#aside[92.7 #ci[90.9][94.2]]], [#aside[v1 report]],
  ),
  caption: [The best agent per setting, on held-out boards. 2p scores are win rates; 4p scores are slot 0's score share. v2 trains from scratch (no DAgger) and lands 5 points above v1's warm-started reference in 2p. Hidden information is worth under a point in 2p: the fair `public` and belief-averaged nets are level with the raw `full` net. 4p is trained and played at the `private` level, the only fair level past two players.],
) <tab:best>

== Models <sec:models>

Every network we talk about gets a name of the form _players–network–number_, so the name always gives the player count and the size (Marco): `2p-graph-s-3` is the third 2p net of preset `graph-s`. The names and their pinned checkpoints live in `configs/registry.yaml`. Records and configs keep the full checkpoint refs.

#figure(
  btable(
    columns: (auto, auto, auto, auto, 1fr),
    align: (left, left, right, left, left),
    header: ([Name], [Rules], [Steps], [Checkpoint], [Notes]),
    tgroup(5)[`graph-s`: hidden 96, 3 layers, 541,540 parameters],
    [`2p-graph-s-1`], [`base-2p`], [0.44B], [`puffer-graph2/full@e3350`], [From scratch; the first deployed 1v1 net],
    [`2p-graph-s-2`], [`base-2p-fr`], [1.45B], [`ppo-2p-fr/full@e7600`], [`2p-graph-s-1` + 1.0B on OpenHex's rules],
    [`2p-graph-s-3`], [`base-2p-fr`], [2.02B], [`ppo-2p-fr3/full@e4350`], [fr3: `2p-graph-s-2` + 0.57B; deployed; *best-2p* until 10 November],
    [`2p-graph-s-4`], [`base-2p-fr`], [0.50B], [`discount-2p/g1-s42@e3786`], [g1: from scratch, γ = 1],
    [`4p-graph-s-1`], [`base-4p`], [0.96B], [`ppo-4p/main@e3650`], [From scratch; deployed; *best-4p*],
    [`4p-graph-s-2`], [`base-4p`], [0.16B], [`aux-hidden-4p/plain@e600`], [The calibrated critic that judges trades],
    [`mix-graph-s-1`], [2–4 players], [0.93B], [`ppo-mixed/main@e3550`], [One net for 2–4 players and any map; deployed],
    tgroup(5)[`graph-m`: hidden 192, 4 layers, 2,320,100 parameters],
    [`2p-graph-m-1`], [`base-2p-fr`], [3.0B], [`graph-m-2p-lr/main@e22711`], [From scratch (@exp:arch-graph-m-2p-lr); 56.5% against fr3; *best-2p* since 10 November (Marco)],
  ),
  caption: [Named models. Steps count every agent step behind the weights, parents included. *best-2p* and *best-4p* are roles that move to a better net; every new run plays best-2p (or best-4p) as its yardstick.],
) <tab:models>

== How to read this log

Sections are ideas, not dates. Each experiment is one block: a status chip (*done*, *running*, *queued*, *stopped* early, *todo* or *parked*); a metadata line (dates, whose idea, config, W&B runs, eval records); a summary; the question, setup, result and reading; and the command that reproduces it. Marco's ideas are credited to him; ideas from Claude (the assistant writing this log) that were never run are marked "(Claude's suggestion, not queued)".

/ Records: An 8-character hex id in backticks (`831eea84`) is the start of an eval record's id. Records live under `runs/<experiment>/<variant>/evals/<suite>/`, or `runs/_evals/<suite>/` for eval-only rows. `catanrl verify <id prefix or path>` replays a record's games and checks them; a record stored in several run directories needs the path (the critic fix's start row `60cd10b6` sits in all four arms).

/ Commands: From the repo root. `catanrl run <config>` runs an experiment's variants in dependency order, with their finals; `--only <glob>` picks variants. An eval-only row is one `catanrl eval` line, rebuilt from the record's agent specs.

/ W&B: Runs before 2026-10-17 were backfilled from `runs/*/metrics.jsonl` and the eval records; since then a live mirror (`catanrl.wandb_mirror`) logs every run whose config sets `meta.wandb.project`, and `tools/wandb_backfill.py` re-syncs a run directory. The record on disk is the source of truth. Runs log the usual PPO metrics, game length, seat win rates, throughput, the select curve and the critic's calibration (`critic/value_sum_*`, the seats' predicted win probabilities summed, which should match the share of games somebody wins). "Eval only" links are eval runs whose results are in the records; "no local mirror" links are computed from the config hash, with no local W&B directory confirming the sync.

== Setup

/ Engine: cppanatron, a C++20 port of Catanatron kept in lockstep parity with it: 2–6 players, maps of any shape. The encoder, action codec, threaded env, hand tracker and batched MCTS all live in C++, so training, search and deployment cannot drift apart. The network reads one float32 row per (game, seat), seat-relative and padded: `[global | players | tiles | nodes | edges | topology | action mask]`.

/ Observation levels: How much of the other seats' hidden state a row shows.
  - `private`: what the seat itself sees. The encoder zeroes the other seats' 15 hidden features (5 resource ratios, actual VPs, 5 dev-card counts, 4 playable flags). Those are exactly what a sampled world changes, so belief-averaging a `private` policy gives back its argmax.
  - `public`: in 2p both seats see every resource movement, so the opponent's exact hand is a card counter's knowledge: a `public` policy is fair in 2p without belief averaging.
  - `full`: adds the opponent's dev cards in hand. Played raw, it is unfair.

/ Agents: Each player is a spec string.
  - `POLICY:ckpt=<run>@<epoch>`: the raw network, argmax.
  - `BELIEF:ckpt=…:worlds=16`: the raw policy averaged over 16 worlds sampled from the hand tracker; the fair way to play a net that sees more than its seat.
  - `MCTS:ckpt=…:sims=N:dets=K`: a batched N-player PUCT search over K worlds sampled from the tracker at the root, with per-seat values backed up max#super[n] and exact chance nodes for dice, dev-card draws and steals. `oracle=true` searches the true game and is only a reference.
  - `F` is Catanatron's `ValueFunctionPlayer`; `R` plays uniformly random legal moves.
  - `<run>@eN` is the run's checkpoint at epoch N; `@best` is the one its select suite chose, `@latest` the last one saved.

/ The slot after an F: F (`enemies=next`) robs and blocks the seat after it. A 4p suite seats `lineup[(seat + r) mod 4]`, so slot 0 always moves right after the F in slot 3. A 4p table of the shape `[A, B, F, F]` needs both slot orders, or the two learner slots summed (diagnosis in @sec:hidden).

/ Rules:
  - `base-2p`: the BASE map, 15 VP, discard above 9 cards.
  - `base-2p-fr`: OpenHex's 1v1 rules, which are `base-2p` plus a friendly robber (tiles touching an opponent with at most 2 public VP are off limits) and balanced dice (a 36-card deck of dice pairs, with recent totals and 7 streaks damped). New 1v1 runs train on it.
  - `base-4p`: BASE under the standard rules, 10 VP, discard limit 7; also the engine's default.

/ Boards: Every number is on boards training never saw (`catanrl validate` checks).
  - `heldout-2p`: 2,500 boards (seed 67) in both seat orders, 5,000 games. `heldout-2p-fr` is the same seed and size on `base-2p-fr`.
  - `heldout-4p`: 1,000 boards (seed 68) in 4 rotations, 4,000 games.
  - `heldout-random-4p`: 1,000 blob-shaped boards of 15–30 tiles (seed 69) in 4 rotations. `heldout-mixed` mixes 2p and 3p BASE with 4p blobs.
  - The `hexagon` generator (training only) keeps the BASE shape with shuffled numbers and generated harbors.
  - Checkpoints are chosen every 50 epochs on separate `select-*` suites, so no held-out number picked its own checkpoint. `select-2p` is 200 boards (seed 123, 400 games); `select-2p-fr` is 500 boards on the same seed (1,000 games); `select-4p` is 800 games. `select-mixed` is 2, 3 and 4 players on BASE, 100 boards each in every seat rotation (900 games, seed 126).
  - The AlphaZero and critic-refit runs select on `select-2p`'s seed (123) in both seat orders: `select-2p-2k` is 2,000 boards (4,000 games), `select-2p-500` its first 500 boards (1,000 games).

/ Measurement: Intervals are Wilson 95%, printed small after the number: #box[94.96 #ci[94.3][95.5]]. pp means percentage points. In N-player games the number is slot 0's _score share_: a win counts 1 and a truncated game 1/N to every seat, so parity is 1/N. Agents that use information a player doesn't have (a raw `full` policy, an oracle search) are marked unfair.

/ Paired statistics: Rows played on the same game ids pair game by game: McNemar over the shared ids (`tools/search_probe.py --pair`), with Holm when a decision rests on two tests. 4p tables in both slot orders use a bootstrap over board × rotation pairs. Power figures are 80% power at a two-sided 5% level.

== v1, for reference

=== v1 reference rows #chip("done") <exp:v1-reference>
#meta(
  dates: "in v1 (catanrl), before v2",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[v1's best 2p numbers, kept for history: its belief-averaged raw policy scored 89.65% against F, and its search beat that policy 51.1–58.1% depending on sims. v2 from scratch is 5 points higher against F.]

*Question.* What did v1's best 2p bot score, as a yardstick for v2?

*Setup.* v1's `graph-s`: DAgger from F, then MARL PPO, 73.7M steps. Numbers from the v1 report, never paired with v2's.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, left, right, right),
    header: ([Agent], [Opponent], [Games], [Score]),
    [Raw policy, belief-averaged], [F], [2,000], [89.65 #ci[88.24][90.91]],
    [MCTS 256, determinized], [F], [1,000], [90.5 #ci[88.5][92.2]],
    [MCTS 1024 on the true game (unfair)], [F], [1,000], [92.7 #ci[90.9][94.2]],
    mid,
    [MCTS 256 / 1024 / 4096 sims], [Its belief-averaged raw policy], [1,000], [51.1 / 55.1 / 58.1],
  ),
  caption: [v1's reference rows, from the v1 report.],
)

*Reading.* Two more v1 results that later sections use:
- v1 never trained AlphaZero above 512 sims, and its frozen-teacher distillation of an 8 × 64-sim search (EX-1d) gave +1.3 vs F and 50.6% head to head, both null (@sec:az).
- v1 ran Canopy's PUCT recipe on a 55%-vs-F graph net and raised it to 70.5% raw vs F over 10 iterations, but never from a start above 90% (@sec:canopy).

#repro(```sh
# Not reproducible in this repo: numbers from the v1 report (github.com/mcrco/catanrl)
```)
