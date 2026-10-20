# catanrl-v2 design

Goal: the strongest Catan bot for any number of players (2–6) on any map, with
a research harness where every past experiment (architecture ablations,
information levels, determinized / information-set search, belief averaging,
reward shaping vs win/loss) is a config file, and a server where people and
bots play fair games against each other.

The lesson of catanrl v1 is that most of its code re-implemented things that
open libraries already do well. v2 keeps a lean core and leans on:

| Library | Role | Where |
|---|---|---|
| **cppanatron** (ours) | the one rules engine for training and search: rules, observation encoder, action space, vectorized env, N-player MCTS, card tracking | `engine/cppanatron` (submodule, branch `v2`) |
| **Catanatron** (our fork) | the reference rules, the game server and the neutral meeting ground for bots from anywhere | `third_party/catanatron` (submodule, branch `v2`) |
| **PufferLib 3** | PPO, vectorization, logging | `third_party/pufferlib` (submodule, branch `catanrl-v2`: two small patches) |
| **PyTorch** | the networks | pip |

`src/catanrl` is glue: configs, map generators, networks, the training and
search loops, agents, evaluation and the experiment runner.

## 1. One engine, one encoder

v1's puffer integration compiled cppanatron's sources but hard-coded two
players, the BASE map and its own feature gather, and the Python side had a
third feature implementation. Every change had to be matched by hand in three
places. In v2:

* **Features, action indices and env stepping live only in C++**
  (`cppanatron::GraphEncoder`, `cppanatron::GraphActionSpace`,
  `cppanatron::VecEnv`). Training (PufferLib), search (MCTS leaves), and
  deployment (a bot playing a Catanatron/server game) all produce features by
  calling the same C++ code, so they cannot drift.
* **One compiled module.** `cppanatron._core` (nanobind) is the only
  extension. The PufferLib env is a `pufferlib.PufferEnv` subclass that calls
  `VecEnv.step()` once per vectorized step; the C++ side releases the GIL and
  writes straight into PufferLib's numpy buffers. No `binding.c`, no second
  build of the engine sources.
* **Catanatron positions enter the engine through `Game.apply_setup`** (plus a
  map built from the Catanatron map's tiles), so a bot in a Catanatron game or
  a server game gets the same features as in training.
* Parity with Catanatron stays enforced by cppanatron's differential tests,
  extended to explicit maps and 2–6 players.

### 1.1 Maps

`MapSpec` is an explicit list of tiles (cube coordinate, land/water/port,
resource or desert, number, port direction and port resource). Node and edge
ids follow Catanatron's `CatanMap.from_tiles` assignment order, so the same
spec builds identical maps in both engines. `CatanMap` is immutable and shared
between games (`std::shared_ptr<const CatanMap>`), which also makes game copies
in search cheaper.

Map *generators* are Python (`catanrl.maps`): Catanatron's BASE / MINI /
TOURNAMENT templates, the 5–6 player extension (30 land tiles, 11 ports) and
random shapes (hexagons, rectangles, islands-free blobs of N land tiles with a
water ring and ports). A generator plus a seed yields a `MapSpec`; YAML picks
a generator or lists explicit tiles.

### 1.2 Observations: seat-relative graphs

The encoder writes one float32 row per decision. Everything is relative to the
observing seat (slot 0 = me, slot k = k seats after me), so one network plays
every seat and every player count.

```
[ global | players P×Fp | tiles T×Ft | nodes N×Fn | edges E×Fe |
  topology (edge→2 nodes, tile→6 nodes, as floats, -1 = padding) | action mask A ]
```

`P, T, N, E` are the *maxima* of the env (padding plus validity bits), so one
row shape covers mixed maps and player counts. The layout is published to
Python as a dict of `(offset, shape)` so the network slices it without
knowing the order.

Information levels (v1's experiment axis, kept):

* `private` – what a seat sees: own hand, others' public state (hand sizes,
  dev-card counts, played cards, buildings, VP visible).
* `public` – plus opponents' exact resource hands (what a perfect card
  counter knows in 1v1; an oracle in N-player).
* `full` – plus opponents' development cards and actual VP (oracle).

Fair deployment of a `public`/`full` network averages its policy over, or
searches over, worlds sampled from the card tracker (1.5).

### 1.3 Action space

Map-agnostic and fixed-size for given maxima:

```
[ globals G | settlement@node N | city@node N | road@edge E | robber@tile × victim-slot T×P ]
```

Globals: roll, end turn, buy dev card, play knight, play road building,
discard ×5, monopoly ×5, year of plenty ×15, maritime trade ×60 (give, get,
ratio). Domestic trades are out of scope for now (as in v1). The graph
network's policy head emits node, edge and tile logits from the matching
embeddings and global logits from the pooled state, which is exactly this
layout.

### 1.4 VecEnv

`VecEnv(configs, num_envs, max dims, obs level, opponents, seed)` holds
`num_envs` games. Each env draws its scenario (player count, rules, map
generator output from a pre-generated `MapSpec` pool, seat of learners) from a
weighted list, so one run can mix 2p BASE, 4p random maps and 6p extension
maps. Each env exposes `P_max` agent rows (one per seat slot). Forced moves
(one legal action) are played inside the env; seats played by built-in bots
(F, random) too.

Only the seat to move acts. The rows of waiting seats get a mask with only the
no-op. Their values are not computed separately: the network's value head
predicts every seat's win probability (independent sigmoids, not a softmax, so
"values sum to ≈1" is a real calibration check), and the acting row's
prediction fills the waiting rows' values during rollout, which keeps GAE on
each seat's own trajectory exact at one network evaluation per game step.
Rewards: 1 to the winner, 0 to everyone else (or truncation: 0); optional
shaping as in v1 behind a config flag.

### 1.5 Hidden information: the card tracker

`HandTracker` (C++) follows a game from one seat's point of view using only
what that seat sees: every public resource movement, its own hand, and
unknown cards moved by steals between other seats. Per opponent it keeps the
hand size, a known lower bound per resource and a pool of unknown cards; per
resource the total across opponents follows from the public bank. Unplayed
dev cards are drawn from the unseen pool (deck + opponents' hidden cards)
as `determinize_development_cards` already does.

`sample_world(game, observer, tracker, rng)` returns a full game consistent
with the observer's information. It powers determinized MCTS
(`dets=K`), belief-averaged raw policies, and server bots that only receive
views.

Two modes (`TrackerMode`). `pool` (the default) deals the unknown pool
uniformly: sound, but after a hidden steal it forgets which card probably
moved (a steal from {brick, brick, ore} is dealt brick or ore evenly instead
of 2:1) and what later payments proved about it. `exact` also keeps the exact
posterior over every seat's hand: a weighted set of hypotheses branched at
hidden steals (each kind by its count in the victim's hand), weighted by seen
steals, pruned by payments, trades and monopoly amounts, merged when
identical, capped at 4096 hypotheses (then it deals the pool until every
hand is known again). It is exact under the rules' chance with no model of
what seats choose to do. Agents pick it with `tracker=exact` (6.5);
`tools/belief_exactness.py` measures both deals against a brute-force
posterior.

### 1.6 Search

`SearchPool` (C++) runs many independent searches in lockstep (v1's fastest
design, now native): `collect()` advances every search to its next leaf and
writes the leaves' observations into one numpy batch; Python evaluates the
batch on the GPU (CUDA graphs at power-of-two sizes); `apply()` expands and
backs up. N-player: nodes back up a value per seat (max^n: each node picks by
the value of its own player to move); chance nodes are exact (dice, dev
draws, steals). Root options: PUCT with Dirichlet noise, or Gumbel root
(sequential halving) for low budgets; determinized roots from the tracker.
`advance()` re-roots a search at a deterministic root action's child, keeping
the subtree: the MCTS agent's `reuse` option (6.5) carries a seat's search
from one decision to the next when the real game evolved by exactly the
action played (most of the gain is the discard chain after a 7 and the bank
trades before a build). Byte-identical leaf rows of a batch (a determinized
search's worlds at `private` level, one position reached by two action
orders) are evaluated once by `DedupEvaluator`, bitwise the plain evaluation
under the strict contract.

The **AlphaZero / expert-iteration loop** (`catanrl.train.alphazero`) is:
self-play games through `SearchPool` with a network (fresh, or a PPO
checkpoint) → replay buffer of (observation, visit-count policy, per-seat
outcome) → train → new network. Because the PPO network and the search
network are the same class with the same observation, the PufferLib-trained
GNN plugs straight into search (`MCTS:ckpt=…:sims=…`).

## 2. Networks

`catanrl.models`:

* `graph` – Canopy Nexus-v3 style heterogeneous message passing (nodes ↔
  nodes, nodes ↔ tiles, nodes ↔ edges) with batched per-sample adjacency built
  from the topology block, so it runs on any map. Presets `graph-s`, `graph-m`.
* `mlp` – flat baseline over the same row (for ablations; padded maps only).
* `cnn` – v1's cross-dimensional CNN (Gendre and Kaneko, arXiv 2008.07079):
  the tile, node and edge blocks gathered onto v1's 11 x 21 plane, laid out
  once from the engine's map of the preset's `board` generator (BASE by
  default). For the architecture ablation; one board only: other dims are
  refused at build, and a map with the same dims in another tile order (the
  hexagon generator against BASE) by `CnnModel.check_maps`, which training
  (`check_bindable`), the eval harness and the bridge run over their maps
  before playing. `MODEL_CLASSES` in `catanrl.models` is the table.
* `lstm` – the graph trunk with memory: one shared trunk, an LSTM cell over
  its pooled embedding stepped at each of the seat's own decisions (waiting
  rows keep their state; a game's first decision starts from zero), the
  memory injected into both heads' inputs. The stateless `forward` is the
  zero-state function (what the checkpoint self-test pins); the state is
  rollout-only and never checkpointed. PPO trains it through PufferLib's
  `use_rnn` path (segments x horizon, each segment from its stored rollout
  state, resets at terminals, as CleanRL's PPO-LSTM); the native `POLICY`
  agent keeps one memory stream per seat of a game, so eval rows stay
  invariant to `num_envs`. MCTS, BELIEF and the Catanatron/server bridges
  refuse it (a leaf, a sampled world or a replayed view has no decision
  history of its own yet).

Heads: structured policy (above) and a per-seat value head (P sigmoids). A
checkpoint stores the network config and the observation spec it was trained
on, and refuses an env whose spec differs.

### 2.1 The hidden-information head

An alternative to sampling worlds from the belief (1.5): train the network
to predict the hidden information itself. `graph` with `args.aux` (on a
shared trunk) carries a third head, `aux_head`, from the pooled state and
each other seat's embedding (the value head's inputs) to that seat's
**hidden features**: the 15 players-block features the encoder zeroes for
every seat but the observer's below the level that shows them
(`catanrl.envs.encoding.HIDDEN_FEATURES`: the hand per resource, shown from
`public`; the development cards in hand per kind, which of them are playable
this turn, and the actual victory points, shown at `full`). Own-seat features
are identical at every level and are not targets.

* **Targets without an engine change.** With `obs.level: private` and
  `obs.critic_level: full` the VecEnv appends the full-level players block
  (`critic_players`, engine E5) after the row. A shared trunk still refuses
  to *read* a separate critic level; with the aux head the block is the
  head's target only (`aux_targets`): the trunk, the policy and the value
  head read the private row, so the policy stays fair at its level and
  `reads_critic_block` is false. Inference needs no block: the native
  POLICY, BELIEF and MCTS agents drop it and rebuild the network on
  actor-level rows (the same weights load), so evals encode what a plain
  private checkpoint needs.
* **Loss.** Masked mean squared error in the encoder's feature scale over
  the present other seats and the hidden features, on acting rows only like
  the other heads, weighted by `train.aux_coef` (default 0: the head, if
  any, is not run and training is the plain PPO, bitwise). The targets are
  the encoder's ratios in [0, 1] (count over the bank per resource, the deck
  per kind, the points to win; 0/1 playable flags), so they are bounded and
  rules-agnostic; a per-count cross-entropy would need each feature's range,
  which comes from the rules and is not in the row. The hook is the
  PufferLib fork's `aux_loss(outputs)`, next to `value_loss`.
* **Logged.** `losses/aux` (the mean over features) and, per hidden feature,
  `aux/mse_<f>` and `aux/var_<f>` (the target's variance over the same pairs,
  the best constant's error), so `1 - mse / var` is the head's R^2 per
  feature: how much of the hidden state the trunk has learned to infer.
* **Checkpoints** store the head as a third part, `aux`, with its own
  self-test outputs (`part="all"` pins them); `init_from` parts stay actor
  and critic, so a warm start leaves the head fresh.
* **On the recurrent network.** `lstm` with `args.aux` carries the same head
  on its memory-injected embeddings (the pooled state concatenated with the
  memory, each seat's embedding), so its predictions are a function of the
  row and of what the seat remembers, and the aux loss trains the cell
  through the horizon (`decode_with_aux` in the training forward; the
  rollout and inference never run it). Its R^2 per feature against the
  markov head's measures the hand information memory adds beyond the row's
  tracker. The same rules: the block is the target only, the self-test pins
  the zero-state predictions, POLICY drops the block.

The experiment is `aux-hidden` (arch-ablation's graph-s cell plus the head);
the question is whether the policy's score moves, not the aux loss.

**Acting on the prediction.** The literal reading of the idea is an eval-only
agent, `PREDICT:ckpt=<public or full policy>:predictor=<private checkpoint
with the head>` (6.5): the policy decides on *one* row, the deciding seat's
private row with, for every other present seat, the hidden features the
policy's level shows (the hand per resource at `public`, all 15 at `full`;
`encoding.shown_hidden_features`) overwritten by the predictor's aux output on
that same private row. A row at `public` or `full` is exactly the private row
plus those features (the global, tile, node, edge, topology and mask blocks,
the own slot and absent slots are bitwise the same; `tests/test_predict_agent.py`
pins it), so the policy sees the row it was trained on with predictions in
place of the truth, and with the true features in the head's place it chooses
exactly what `POLICY` chooses on the level's row (the oracle test). The
alternative to a sampled belief is one deterministic forward of each network
per decision instead of `worlds` forwards. `round=true` snaps each prediction
to the encoder's grid: a count over the game's rules (the bank per resource,
the deck per kind, the points to win, read from the game as the encoder reads
them), the playable flags to 0/1; the flags are rounded independently of the
counts. The agent is fair (it reads only the private row and a learned
predictor); both checkpoints are pinned in the spec, the eval key and the
record (`refs` holds the policy's, `predictors` the predictor's); the
predictor self-tests its `aux` part. It plays natively and as a Catanatron
player; the server bridge and the TUI (`catanrl play`, which routes every model
key through the bridge's ViewAgent) refuse it for now. `configs/experiments/predict-vs-belief.yaml`
measures it: puffer-graph2/full with aux-hidden/main's head against BELIEF
(16 worlds) of the same network, against F, rounded against BELIEF, and against
the private-level arch-ablation/graph-s, on heldout-2p.

## 3. Agents, evaluation, experiments

Agent specs use Catanatron's registry syntax (grammar in 6.5):
`POLICY:ckpt=…[:sample=true]`, `MCTS:ckpt=…:sims=256:dets=4`,
`MCTS:ckpt=…:oracle=true`, `BELIEF:ckpt=…:worlds=16`,
`PREDICT:ckpt=…:predictor=…[:round=true]` (2.1), `F`, `R`, and any
Catanatron registry bot (`AB:2`). The same spec plays natively (fast
evaluation on cppanatron), as a Catanatron `Player`, or as a server client.

**LLM players** (the report's "eval frontier models on how well they play
Catan"). `LLM:model=<OpenRouter id>:strategy=<basic|strategy|search|rules|plan>`
(`:thinking=true:max_tokens=8192:temperature=0.1:seed=0:retries=1` by
default; `catanrl.agents.llm`) is a language model deciding from the game
rendered as text: catanllm's (the CS 159 project) prompting strategies,
ported as Python strings (so `code_id` hashes a prompt edit), with the board
topology section generated per map instead of hard-coded for BASE and the
preamble stating the game's own VP target (base-2p plays to 15). The text
is public information only: an opponent's hand size, dev-card count, played
cards, buildings and visible VP, never its resource kinds, unplayed cards or
hidden VP cards (catanllm's renderer read every seat's dev cards from the
observer's slot; the port fixes it and `tests/test_llm_agent.py` pins the
redaction), so the agent is `fair` by construction. The model answers with
the action number; an unusable answer (no number, out of range) is
re-prompted with the error `retries` times, then the player falls back to
End Turn (else the first action), counted as a `fallback`, not a forfeit.
Models go through OpenRouter via the openai SDK (an optional extra, `uv
sync --extra llm`, imported lazily; `OPENROUTER_API_KEY` is read from the
environment only); `model=stub/first`, `stub/random` and `stub/scripted` play
without a network (tests, `catanrl llm-cost`). The agent exists only on the
catanatron harness (6.6). Each row of an LLM seat carries `llm` stats
(prompted decisions, forced moves, calls, prompt and completion tokens from
the SDK's usage or a characters/4 estimate, retries, fallbacks, wall time),
totalled per slot in the record's summary. `configs/experiments/llm-eval.yaml`
is the eval (models x {plan, basic} x {F, our best 2p network} on the first
20 heldout-2p boards, both seats); `catanrl llm-cost` plays stub games and
prices a cell from a $/M-token table, so a budget is approved before any
real call. `catanrl run llm-eval --smoke` plays every cell with
`stub/random`: a smoke plan replaces the model of any `LLM:` spec, so a
smoke never makes a paid call. A real model's record is `nondeterministic`
(flagged † in the report) and `verify` refuses to replay it unless forced.
Model ids cannot contain `:` (the spec grammar splits on it), so OpenRouter
variants like `:free` are not addressable yet.

Evaluation: paired seats (every board played from every seat rotation),
fixed seeds, Wilson intervals, exact McNemar for paired comparisons,
head-to-head and N-player "win share vs expected 1/N".

Experiments are YAML files under `configs/experiments/` composed from
`configs/games`, `configs/models`, `configs/train` and `configs/suites`, with
optional `stages:` and a `sweep:` grid. `catanrl run EXPERIMENT.yaml` writes
`runs/<experiment>/<variant>/` (section 6). v1's research maps onto config
axes:

| v1 experiment | v2 axis |
|---|---|
| architecture ablation (xdim/MLP/graph-s/graph-m) | `model.preset` sweep (`mlp`, `graph-s`, `graph-m`, `lstm`) |
| information levels for actor/critic | `obs.level` (`private`/`public`/`full`), `obs.critic_level` |
| belief averaging of a full-info policy | agent `BELIEF:…` |
| determinized MCTS / ISMCTS, dets × sims | agent `MCTS:…:dets=K:sims=N`, a `kind: eval` sweep over `params` |
| shaped vs win/loss reward, γ=0.99 vs 1 | `reward.*`, `train.gamma` |
| DAgger warm start from F | superseded by self-play PPO from scratch (`puffer-graph2`: 95.2% vs F held-out, as v1's graph2); still available as a `train.algo: imitation` stage, then `init_from:` |
| expert iteration / AlphaZero | `train.algo: alphazero` |
| N-player and map generalization | `env.scenarios` mixture + held-out eval maps |

## 4. Game server (Catanatron fork)

`catanatron.server` turns Catanatron into a WebSocket service: rooms, seats
for humans and bots, spectators, the existing bot wire protocol (hello /
before / decide / step / after) carrying per-seat `client_view`s only.

Fairness, as far as it can go without multi-party computation:

* **Unbiased chance.** The server commits to `H(server_seed)` before the game;
  each seated client then contributes a seed; the game's random stream is
  seeded from `H(server_seed ‖ client seeds)`. Players cannot predict chance
  (they lack `server_seed`), and the server cannot steer it (it committed
  first).
* **Committed hidden state.** Every message carries a hash-chained, signed
  (Ed25519) transcript head and a salted commitment to the full state. At the
  end the server reveals `server_seed` and the salts.
* **Signed actions.** Clients sign their actions, so the server cannot forge
  moves.
* **Audit.** `catanatron-verify` replays the game from the seeds and the
  signed actions in stock Catanatron and checks every view the client
  received was the honest redaction of the committed state.

Limitation: the server knows all hidden state during the game; a server that
colludes with a player can leak it. Removing that needs mental-poker style
dealing (future work, noted in the server README).

Clients: a terminal UI (`catanrl play` / `catan-tui`, mainly for debugging
against our bots), a minimal browser client served by the server, and bot
clients (`catanrl bot --spec MCTS:… ws://…`).

## 5. Build and layout

```
engine/cppanatron/      C++ engine + nanobind module (scikit-build-core)
third_party/catanatron/ reference engine + server
third_party/pufferlib/  PufferLib 3.0 (NO_OCEAN build)
src/catanrl/            Python glue (maps, config, envs, models, train, search, agents, eval, tui)
configs/                games/ maps/ models/ experiments/
tests/                  python tests (engine parity lives in cppanatron)
runs/                   gitignored outputs
```

`uv sync` builds everything; `uv run pytest` and `ctest` (engine) test it.

## 6. Experiments: reproducibility and testability

v1's experiment store had the right habits: one name per run,
`load_experiment(name).build_policy("best")`, selectors, pins, lineage,
derived seeds and paired seats. Where it failed was at the edges: the runs
root depended on the cwd, metadata was rewritten on resume, there were six
spec grammars and a load shim per script, models were selected on the
report's own seed, and checkpoints silently changed meaning when the model
code changed (b1b053a and 5717ab8, retracted in bcea78e). v2 keeps the habits
and enforces these rules in code:

* **Records are write-once.**
* **Names are resolved once** and stored as `run@eN#sha`.
* **A cache key holds exactly what decides the outcome**, code included.
* **A checkpoint self-tests on every load.**
* **Nothing falls back silently**: no random move for a failed bot, no
  default architecture, no `best`→`latest`, no unknown config key.

### 6.1 Runs directory

```
$CATANRL_RUNS_DIR      default <main worktree>/runs (via git-common-dir): one root for all worktrees
  <experiment>/
    summary.md         `catanrl summarize`: tables per sweep axis, mean ± sd over seeds
    <variant>/         graph-s-s42.g1 = sweep cell + "." + stage
      config.json      write-once: resolved config, config_hash, resolved refs
      run.json         trainer-owned, atomic: status, launches, checkpoint index
      aliases.jsonl    append-only: best (trainer, cites its select eval), pins (CLI)
      metrics.jsonl    one row per epoch
      provenance/      <launch>.json, plus <launch>.<repo>.patch per dirty repo
      checkpoints/     e000600.pt …; orphaned/launch<n>/ after a crash
      trainer_state.pt resume state: the commit point
      evals/<suite>/   <eval_id>.json + <eval_id>.games.jsonl
  _evals/              records whose lineup slot 0 is not a run (F vs AB)
  _wandb/              the W&B mirror's own files (offline runs, logs), unless WANDB_DIR says where
```

* **Names** match `^[a-z0-9][a-z0-9._+-]*$`, so they are safe inside agent
  specs.
* **Writes** go through tmp + fsync + `os.replace`. JSONL readers drop a torn
  last line. A live trainer holds a flock.
* **At a checkpoint** the trainer writes the `.pt`, then `trainer_state.pt`
  (the commit), then `run.json`. It then runs the select eval and moves
  `best`.
* **On resume**, any `.pt` newer than `trainer_state` moves to `orphaned/`,
  so it is never `latest` and never an epoch ref. The select evals that
  committed checkpoints lack run first.

### 6.2 Experiment YAML

`configs/defaults.yaml` is the schema: it lists every key with its default.
`configs/train/<algo>.yaml` and `configs/models/<preset>.yaml` are complete.
`ppo.yaml` is written deliberately (Adam, constant LR, `amp` set explicitly)
rather than copied from PufferLib's `default.ini`. Execution keys (`device`,
`seed`, `data_dir`, `compile`, …) live in `runtime`, and putting them under
`train` is an error.

```yaml
# configs/experiments/arch-lineage.yaml: v1's lineage launch script, swept, minus its DAgger stage
name: arch-lineage                       # = file stem
seed: 42
env: {scenarios: [{game: base-2p}], num_envs: 1024}
obs: {level: full, critic_level: full}
model: {preset: graph-s}
train: {algo: ppo, total_timesteps: 400_000_000, learning_rate: 3.0e-4, gamma: 0.99, checkpoint_interval: 50}
eval:
  select: {suite: select-2p, lineup: ["POLICY:ckpt=${self}", F], every: 50, metric: score_rate}
  final:
    - {suite: heldout-2p, lineup: ["POLICY:ckpt=${run}@best", F]}
    - {suite: heldout-2p, lineup: ["BELIEF:ckpt=${run}@best:worlds=16", F]}
stages:                                  # one variant each, run in order per sweep cell
  ppo: {}                                # from scratch, ~3050 epochs of 131,072 agent steps
  g1:  {init_from: {ckpt: "${stages.ppo}@3000"}, train.gamma: 1.0, train.total_timesteps: 200_000_000}
sweep:
  axes:
    model.preset: [graph-s, graph-m, mlp]   # list axis; a mapping is a tied axis:
    seed: [42, 43]                          #   obs: {pf: {obs.level: private, obs.critic_level: full}, …}
  name: "{model.preset}-s{seed}"            # graph-m-s43.g1; 3 × 2 × 2 stages = 12 variants
meta: {notes: "replaces launch-xdim-warm-lineage-20261009.sh, without its DAgger stage", wandb: {project: catanrl-v2}}   # not hashed
runtime: {device: cuda, threads: 16, time_limit_hours: 12}                                       # not hashed
```

**Eval-only experiments.** A `kind: eval` experiment has only `eval.final`
and sweeps `params` into its specs. Example:
`MCTS:ckpt=…@ref:sims=${params.sims}:dets=${params.dets}`, swept over
`params.sims: [64, 256, 1024]` and a tied `world` axis (oracle, d4, d16).

**Suites.** A suite (`configs/suites/*.yaml`: `purpose: select | heldout |
dev`, `seed`, `scenarios`, `boards`, `turn_limit`) holds boards, not lineups.
So one suite serves vs-F, vs-R and head-to-head. A suite that mixes player
counts takes `fill: F` for the empty seats.

| Suite | Seed | Boards |
|---|---|---|
| `heldout-2p` | 67 | 2500, each played in both seat rotations |
| `select-2p` | 123 | 200 |

**Resolution.** `config.load_plan` is pure: it reads no argv and no cwd.

* Precedence: defaults < referenced files < body < stage < sweep cell <
  `--set`. A `key!:` replaces the section instead of merging.
* Errors: an unknown key (with a did-you-mean), a stage and an axis setting
  the same key, a `params` key nothing references, two cells with the same
  name.
* Values take the default's type, because PyYAML reads `3e-4` as a string.
* `dims: auto` uses the generators' seed-independent bounds.
* `config_hash` covers everything except `meta`, `runtime` and (for
  `kind: train`) `eval.final`. Refs are hashed as written.
* Refs to other runs are resolved once, when the variant is created, and
  stored in `config.json`. If one later moves, the variant warns and keeps its
  target. A ref to another variant of the same experiment, in `eval.final`
  too, is a dependency: the runner finishes that variant first. A ref in
  `eval.final` to a run of *another* experiment (a control trained elsewhere,
  which the runner cannot order) is pinned at creation only if that run is
  already finished (`USABLE`); otherwise the variant is created, trained and
  smoked without it, the final is deferred (`Unbound`), and a later
  `catanrl run` of the experiment pins and plays it once the control is
  finished (`bindings.jsonl`).
* `${self}` and `${run}@best` resolve when the eval runs, and are pinned in
  its record.
* `${init}` is the checkpoint `init_from` loads the actor from, pinned when
  the variant is created like `init_from` itself. It is allowed only in eval
  lineups, and a variant without one is an error. (`${run}@init` is this
  run's e0: the same actor weights, but recorded under this run.)

**AlphaZero evals.** Self-play drifts: v1's AlphaZero runs beat their
previous iteration (56–60%) while flat or worse against F, and against F a
strong policy is near its ceiling. So an AlphaZero variant that starts from a
network (`init_from` with the actor) is measured against that network by
default: `config.alphazero_evals`, a layer between the defaults and the body,
so an experiment's own `eval` keys override it. Each network plays as its fair
raw policy, `RAW`: `POLICY` at a `private` actor, or a `public` one in
two-player games, else `BELIEF:…:worlds=16`. The run's network takes slot 0
and its start fills the other N − 1 seats; the suites follow the training game
(`base-2p` → `select-2p`/`heldout-2p`, `base-4p` → `*-4p`, `random-4p` →
`*-random-4p`), and a mixture of games must set its own evals.

| Eval | Lineup | Suite |
|---|---|---|
| select, every checkpoint, slot 0's score rate | `RAW ${self}` vs N − 1 × `RAW ${init}` | `select-<game>` |
| final | `RAW ${run}@best` vs N − 1 × `RAW ${init}` | `heldout-<game>` |
| final | `MCTS:ckpt=${run}@best:sims=${train.sims}:dets=1:root=${train.root}:c_puct=…:gumbel_m=…` vs the start | `heldout-<game>`, 500 boards |
| final | the same search on `${run}@init` (the run's e0: exactly its starting actor and critic) vs the start, paired with the row above | `heldout-<game>`, 500 boards |
| final | `RAW ${run}@best` vs N − 1 × F, a secondary number | `heldout-<game>` |

`best` needs no rule of its own. e0 holds the start's weights, so it scores
exactly 1/N as `POLICY` (a board's rotations are mirror games; as `BELIEF`,
1/N up to world sampling) and is `best` first. `best` then moves only to a
checkpoint above every earlier one. A run from
scratch has no start and keeps the global defaults.

### 6.3 Checkpoints

A checkpoint is a plain dict that `torch.load(path, weights_only=True)` reads,
and it is usable without its run directory:

```python
{"format": "catanrl.ckpt", "schema": 1, "run": "arch-lineage/graph-s-s42.g1", "epoch": 600, "agent_steps": 157286400,
 "weights_sha256": "41f2a9…",       # over sorted (key, dtype, shape, bytes): unchanged by re-saving
 "config": {...}, "config_hash": "c41d…", "model": {"preset": "graph-s", "class": "graph", "args": {...}},
 "parts": {"actor": ["backbone.", "policy_head.", "value_head."], "critic": ["critic."]},
 "obs": {"level": "full", "critic_level": "full", "dims": {...}, "layout_hash": {...}}, "action": {"layout_hash": "…"},
 "code": {"id": {...}, "git": "a1b2c3d", "dirty": False},
 "self_test": {"positions": "<8 apply_setup positions, JSON>", "obs": T, "mask": T, "logits": T, "value": T},
 "state_dict": {...}}
```

`Checkpoint.model()` runs these checks in order:

1. The weights hash matches.
2. `layout_hash` matches, else `SchemaMismatch`.
3. Dims match, unless the class is dims-agnostic (graph is; mlp and cnn are
   not). A class tied to one board (cnn) also checks the maps it is about to
   play, in `check_bindable`, the eval harness and the bridge (section 2).
4. Re-encoding the stored positions gives the stored obs, else
   `EncoderDrift`.
5. The rebuilt model on CPU gives the stored logits and values, else
   `ModelDrift`.

The positions are `Game.apply_setup` payloads, not action logs, so neither
the toolchain's RNG nor unrelated rules changes can move them.
`allow_drift=true` skips checks 4 and 5. It is recorded, and the report
refuses results made with it. Check 5 is the one that would have caught v1's
reshape bugs, whose encoder bytes never changed.

### 6.4 Resume, warm starts, lineage

| Relaunch finds | Action |
|---|---|
| no directory | create it; launch 0 |
| same hash, `finished` | run only the missing `eval.final` |
| same hash, lock held | refuse |
| same hash, otherwise | resume as launch n+1 if the training `code_id` (6.7) and `uv.lock` match launch 0; otherwise refuse, or record `mixed_code` with `--allow-code-change` |
| different hash | `ConfigConflict` with `config_diff`; use `--variant` or `fork` |

**Migrations.** Stored configs and checkpoint headers pass through
`config.MIGRATIONS` (key, legacy value) before they are compared or loaded.
So adding a key breaks neither resume nor old checkpoints. A test fails when
a new key has no migration.

**Resume** restores the weights (sha verified), optimizer, scheduler, epoch,
step, RNG and algorithm state (DAgger β, AlphaZero champion). Games in flight
restart, recorded as `resume_exact: false`. SIGINT, SIGTERM and the time limit
write an off-interval checkpoint. PufferLib's SIGINT handler, which calls
`os._exit(0)`, is replaced.

**Time-limited runs** follow v1's habit (budget 1e9, stop after 7 h). A
`time_limit` run can feed later stages, which pin its epoch. A relaunch
resumes it, and `catanrl finish REF` ends it.

**Budgets never grow in place.** `catanrl fork REF --as V+more --set …` makes
a new variant with `init_from: {ckpt: REF@latest, restore:
weights+optimizer}`. With the constant-LR default this is a true
continuation, and `${init}` stays the parent's start.

**Warm starts** take `init_from: {ckpt: REF, parts: [actor, critic]}`, or
one ref per part (v1's hybrid). Each part loads strictly through the checks
in 6.3. `e000000.pt` is always saved, and `catanrl ls` flags a `best` that is
still the parent's weights.

**Lineage.** `run.lineage()` follows the resolved `init_from` back to the
root. W&B tags are derived from it and never read back.

### 6.5 Agent specs: one string, three backends

Specs use Catanatron's registry syntax. It splits values on `:` and lineups
on `,`, so neither character appears in a value. This replaces §3's
`mcts:ckpt=…,sims=256` form and its `catanatron:` prefix.

```
spec := KEY (":" param)*         param := value | name "=" value   (a positional value is ckpt)
KEY  := POLICY | MCTS | BELIEF | PREDICT | F | R | any Catanatron registry key
ckpt := run ["@" which] ["#" sha] | path ending in ".pt"
run  := experiment ["/" variant]  which := best | latest | init | <alias> | N | eN   (epoch; default best)
```

Examples: `POLICY:arch-lineage/graph-s-s42.g1@ref`,
`MCTS:ckpt=…@best:sims=256:dets=4`, `MCTS:ckpt=…:oracle=true`,
`BELIEF:ckpt=…@1500:worlds=16`,
`PREDICT:ckpt=puffer-graph2/full@best:predictor=aux-hidden/main@best:round=true`
(`predictor` is a second `ckpt`, resolved and pinned the same way), `AB:2`.

* **Params** are frozen dataclasses on the registered players. Unknown or
  mistyped params raise.
* **MCTS** takes engine units: `c_puct` 0.75 equals v1's 1.5. `dets ≥ 1`
  counts sampled worlds, and `oracle=true` searches the true state. There is
  no magic 0. `reuse=true` keeps a seat's search between its decisions
  (1.6; the native harness and the server bridge, which stream a seat's
  decisions; the Catanatron player decides one position at a time);
  `transpositions=true` turns on the engine's transposition table (one node
  for a position reached by reordered actions); `dedup` (execution-only,
  default true) evaluates byte-identical leaf rows once. `reuse` and
  `transpositions` are in the canonical spec and `key_dict()` only when set,
  so adding them moved no stored record's key.
* **`tracker`** (MCTS with dets, BELIEF): `pool` (default) or `exact`, the
  hand tracker mode the worlds are dealt from (1.5). It is in the canonical
  spec and `key_dict()`, so eval records tell the two apart. The native
  harness keeps VecEnv's trackers exact when any agent asks, and each agent
  passes its own deal, so a lineup can mix both.
* **PREDICT** (2.1) takes `ckpt` (a `public` or `full` policy; a `private`
  one has nothing to fill and is refused), `predictor` (a `private`
  checkpoint whose network has the aux head; any other level, no head, or a
  recurrent network is refused), `round`, and POLICY's `sample` and
  `temperature`. Both refs are pinned (`@best`, `#sha`) and both are in
  `key_dict()` and the record; a final naming either in another experiment
  waits for that run like any foreign ref (6.2).

`resolve_agent(spec)` does four things:

1. Pins the checkpoint and self-tests it, once, in the parent process.
2. Checks that each game fits (dims, value head width).
3. Derives `fair`:

   | Agent | Fair when |
   |---|---|
   | POLICY | its actor is `private`, or `public` in 2p |
   | MCTS | not `oracle` |
   | BELIEF | always |
   | PREDICT | always: it reads the private row and a learned predictor |
   | built-in bots | never |
   | LLM | always: it reads only the public state text |

4. Writes a canonical spec that is a fixed point of resolution:
   `MCTS:ckpt=arch-lineage/graph-s-s42.g1@e600#41f2a9c07d11:sims=256:dets=4:oracle=false:root=puct:c_puct=0.75:…`.

Its `key_dict()` drops execution params (device, threads). It adds the model
args and, for MCTS, the full engine `SearchConfig` read back from the
binding, less `transpositions` at its default; `reuse` and `transpositions`
appear in `params` only when set.

| Backend | Entry point |
|---|---|
| native, batched on cppanatron (`SearchPool` for MCTS) | `resolved.native(device=…, infer_rows=…)` |
| Catanatron `Player` | `catanrl.player(spec, color)`, i.e. `REGISTRY.build` once `catanrl.agents` is imported. It converts via `Game.apply_setup` and runs the native agent at batch 1 |
| server bot | `catanrl bot --spec SPEC ws://…`. Views feed the `HandTracker`. A `public`/`full` POLICY is refused (use BELIEF); PREDICT is not played here yet |
| Catanatron harness eval | `evaluate(…, harness="catanatron")`, `catanrl eval … --harness catanatron`, `harness: catanatron` in `eval.final`: the suite's schedule as Catanatron games (`catanrl.eval.catanatron`), every seat an `agent.player(color)`; the only harness for `LLM` |

The v1 shims stay, over the same path:

```python
run = catanrl.load_experiment("arch-lineage/graph-s-s42.g1")
net = run.build_policy("best", device="cuda")      # net.catanrl_ref == "…g1@e600#41f2a9c07d11"
bot = run.agent("MCTS", which="ref", sims=256).player(Color.RED)
net = catanrl.load_checkpoint("/anywhere/e000600.pt").model()
```

### 6.6 Eval records and the report

`Suite.schedule(n)` is pure:

* Board `i` gets `derive_seed(seed, purpose, scenario_hash, i)`. The hash
  ignores the game's name.
* That seed's `map` and `game` children are shared by all rotations.
* Rotation `r` puts `lineup[(seat + r) % n]` in each seat.
* The quota is exact and does not depend on `num_envs`, threads or finishing
  order.
* `select` and `heldout` boards are disjoint by construction.

```python
key = {"schema": 1, "harness": "native", "determinism": "strict", "code": code_id("eval"),
       "suite": {seed, purpose, scenarios minus names, boards, turn_limit, schedule_hash},  # incl. map hashes
       "lineup": [a.key_dict() for a in resolved], "fill": …}
eval_id = digest(key)  # runs/<slot-0 run or _evals>/evals/<suite>/<eval_id>.{json,games.jsonl}
```

**`evaluate(lineup, suite)`** returns the existing record if there is one.
Otherwise it:

1. Locks the id.
2. Plays only the `game_id`s missing from `.games.jsonl`. Each finished game
   appends one row, so a crash resumes exactly.
3. Writes the record once.

`harness` is in the key. `native` plays the schedule on cppanatron;
`catanatron` plays the same games (map seed, game seed, seat order forced
before the first action, the suite's turn limit) with Catanatron players,
`runtime.threads` games at a time in a thread pool, with Catanatron's code
in `code`. Catanatron's bots draw from the game's RNG, so their rows depend
on the game seed only; a real LLM is not replayable at all: `verify`
refuses such a record (`Unreplayable`) unless `--force`, and `report-data`
flags it `nondeterministic` (so `--final` refuses it) without replaying it.
Rows of LLM seats carry `llm` stats, which `games_sha256` and `verify` leave
out (wall time is not an outcome).

Each start or restart appends its execution (device class, torch,
`infer_rows`). Resuming under different execution needs `--allow-mixed`,
which marks the record. Any bot forfeit (a `StdioPlayer` random fallback)
makes the record `valid: false`.

**The record** holds canonical specs, pinned refs, `fair` per slot,
`games_sha256`, execution, provenance and a summary. The summary gives the
score rate (a truncation counts 1/N), Wilson 95%, rates per seat, and for N
players the win share against 1/N with an exact binomial test. Rows can be
re-scored without replaying.

**`paired(a, b)`** joins on `game_id` and requires equal `schedule_hash` and
opponents. It returns the delta and the exact McNemar test, computed in log
space, because `stats.binomial_two_sided` overflows above about 1,030 trials.

The report cites queries, never typed numbers:

```yaml
# report/results.yaml
supersedes: [b1b053a]    # refuse checkpoints whose training lineage predates these commits
cite:
  g1-vs-F:     {lineup: ["POLICY:ckpt=arch-lineage/graph-s-s42.g1@ref", F], suite: heldout-2p, oracle_ok: true}
  mcts-vs-F:   {lineup: ["MCTS:ckpt=arch-lineage/graph-s-s42.g1@ref:sims=256:dets=4", F], suite: heldout-2p}
  mcts-vs-raw: {paired: [mcts-vs-F, g1-vs-F]}
  arch:        {lineup: ["POLICY:ckpt=arch-lineage/*.g1@best", F], suite: heldout-2p, group_by: model.preset}
```

**`catanrl report-data`** writes `report/data/results.json` (rate, interval,
n, p, `eval_id`, canonical specs, `run@eN#sha`, flags), which Typst reads with
`json()`.

* **Lockfile.** The same file pins refs. A ref or wildcard that now resolves
  differently shows as a diff and needs `--update`. Wildcards match finished
  runs only.
* **Refused:**
  * suites that are not `heldout`;
  * unfair slots under test, unless the entry has `oracle_ok`;
  * invalid or drift-allowed records;
  * checkpoints with any launch, in the run or an `init_from` ancestor, that
    does not descend from a `supersedes` commit.
* **Marked †** (`flagged: true`, the reasons in `flags`): dirty, mixed or
  `fast` rows. `--final` refuses them instead, and re-plays 16 games of each
  cited record.
* **Missing records** fail with the exact `catanrl eval …` line to run, or
  are computed with `--run-missing`.

### 6.7 Provenance and code identity

`code_id()` hashes these files when catanrl is imported:

* cppanatron's `.so` and Python files;
* catanrl's outcome code: `models`, `agents` (without `agents/llm`),
  `search`, `envs`, `eval/harness`, `maps`, `games`;
* with the Catanatron harness, also Catanatron itself, `eval/catanatron`
  and `agents/llm` (the prompts and the answer parser), so a native record
  does not move when they change;
* for training, also `catanrl/train` and PufferLib's `pufferl.py`,
  `pytorch.py` and `_C`.

Python is hashed by AST, so comments and formatting change nothing. No key
holds a git SHA, so unrelated commits keep every cache valid.

`capture()` runs at each launch and each eval, always as
`git -C <package repo>`:

| Part | Recorded |
|---|---|
| git | HEAD, branch, dirty, diff sha, untracked files, submodules recorded vs checked out. A patch per dirty repo (≤ 20 MB). A ref `refs/catanrl/keep/<sha>` in each repo, since these commits are on no remote |
| code | `code_id()`. A launch or eval is refused when the disk no longer matches what was imported (a stale notebook, a rebuild under a long runner) |
| engine | `.so` sha; after E3, `build_info()` (git sha, dirty, compiler, build type) |
| python | version, every installed distribution (list + digest), `uv.lock` sha |
| torch | versions, CUDA, cuDNN, GPUs, driver, threads. Flags read after `PuffeRL.__init__`: TF32, cudnn, autocast, heavyball compile, `ADVANTAGE_CUDA` |
| run | argv, cwd, host, pid, selected env vars, source YAML sha, map pool hash |

What is allowed, warned about or refused:

* **Allowed:** a dirty tree, because the patch is saved.
* **Warned:** a submodule mismatch (`--strict` refuses it). A stale engine
  build warns until E3 lands, then is refused unless
  `runtime.allow_stale_build`.
* **Refused outright:** a map pool that changed on resume, and a YAML from
  outside the package's repo.

### 6.8 Seeds and determinism

Every seed is `derive_seed(root, *labels)`, bit-identical to v1:

| Seed | Labels | Feeds |
|---|---|---|
| init | `seed, "init"` | Python, numpy and torch before the model is built (PufferLib's seeding is commented out) |
| sample | `seed, "sample", launch` | torch after `PuffeRL.__init__` |
| rollout | `seed, "rollout", launch` | the VecEnv (one Serial PufferEnv) |
| map pool | `map_pool.seed` or `seed, "map_pool"`; then `scenario, j` | training maps |
| collection | `seed, "dagger"/"selfplay", epoch, game` | DAgger, AlphaZero |
| board | `suite.seed, purpose, scenario_hash, i` | eval maps and games |
| bots | `board, "bots", rotation` | R (native F has no RNG) |
| decision | `game_seed, "decide", rotation, seat, ply` | sampling, belief worlds, MCTS roots; never torch's global RNG |

Evals run strict, unless a `dev` suite asks for `fast` (the mode is part of
the key). Strict means:

* no cudnn benchmark, deterministic algorithms, no TF32;
* `CUBLAS_WORKSPACE_CONFIG` set in `catanrl/__init__`;
* autocast off, asserted, because PuffeRL leaks bf16 autocast;
* one padded inference shape, `infer_rows`, for policy forwards and
  SearchPool leaf chunks.

Training is `fast` by default.

| Guaranteed | Not guaranteed |
|---|---|
| maps, schedules: bitwise everywhere | GPU training: statistical only; ≥ 2 seeds per headline variant, spread in `summary.md` |
| native eval rows: bitwise across `num_envs`, threads, workers, interrupts and reruns, for one device class and `infer_rows` (needs E1, E2) | another device class or `infer_rows`: rare argmax ties may flip; `verify` reports agreement |
| SearchPool visits: bitwise across pool size, batch size, threads | another compiler or stdlib: until E4; detected, since the `.so` sha is in every key |
| strict CPU training, small config: bitwise on one machine | games in flight at a resume: restarted (`resume_exact: false`) |
| resume: weights, optimizer, scheduler, epoch, step, RNG, algorithm state | Catanatron-harness bots: they draw from the game's RNG, so only its seed applies |

### 6.9 Tests

| Tier | Guards |
|---|---|
| fast (default) | hashes and `derive_seed` pinned to v1; resolution, migrations, hashes of frozen fixture YAMLs; `validate_plan` on every experiment; `ppo.yaml` matches the keys PuffeRL reads; the runs store under a foreign cwd, with a crash injected between each write; specs; schedules golden, disjoint and exact; eval resume, forfeits, pairing, report gates; provenance; PufferLib import side effects; McNemar at n = 5000; the LLM agent: specs, registry, stub games through the catanatron harness and `catanrl eval`, the public-only render, answer parsing and retries, token and cost accounting, llm-eval's stub cells |
| end to end (default, < 30 s CPU) | a tiny graph net trains 3 epochs; `best` cites a select eval; `build_policy`, `load_checkpoint` and `POLICY:` give identical logits; a repeated `evaluate()` hits the cache; stop and resume keeps the LR and optimizer step; repeated with `[POLICY, R, R, R]` |
| engine (default) | golden maps; drift detection (byte, layout, encoder, model); eval and SearchPool invariance; spec defaults equal `SearchConfig`'s; native vs Catanatron `Player` parity per position; Catanatron action order across processes; sampled worlds give the true private row; golden outcomes of a committed 100 KB checkpoint |
| slow (`-m train`) | seeded CPU training is bitwise; straight and resumed runs match; `--smoke` on every experiment |
| opt-in | `-m gpu`: strict GPU invariance. `-m runs`: `catanrl check-runs` loads, verifies and self-tests every alias of every run |

### 6.10 Out of scope

* No database, MLflow, DVC, Hydra or sacred. W&B mirrors and never decides.
* No DAG engine or Slurm. Stages, `init_from` and `catanrl run --next` cover
  it: each terminal claims a free variant, and each variant runs in its own
  process.
* No in-place budget extension, bitwise GPU training, VecEnv snapshots, or
  checkpoint dedup or GC.
* Grid sweeps only for now; Protein can come later.
* Lineup order is not normalized: A vs B and B vs A are separate records.
* Records do not survive changes to outcome code. `catanrl results --stale`
  lists the old ones.
* From PufferLib only `PuffeRL` is used, subclassed. The fork keeps its two
  patches.
* Phase 2: `catanrl rerun --at-recorded-commit` (everything it needs is
  recorded from day 1), `import-v1`, and a stdio bridge for v1 policies.

### 6.11 From v1

v1 weights do not load: the encoder, actions, value heads and trainer all
differ. Conventions carry over, and recipes become YAML (`v1-anchor`,
`arch-lineage`, `info-levels`, `mcts-dets-sims`).

* **`v1-anchor`** is `base-2p` as is (official spiral, as the report says),
  trained with PuffeRL. PPO_Central_Critic's settings (separate LRs, target
  KL) do not carry over. So the anchor shows whether v2 reaches v1's level;
  it is not the same measurement.
* **v1 numbers** stay in a history table tagged `bcea78e` and are never
  paired with v2's.
  * Its PPO `@best` rows are selection-biased: v1 selected on seed 67, which
    is also the report's seed.
  * Its Catanatron-path `dets=1` rows mean `oracle=true`.

Needed outside `src/catanrl`:

* **cppanatron:**
  * E1: a schedule mode, `start_game(env, scenario, map_spec, game_seed,
    seats, bot_seed)`;
  * E2: per-game bot streams;
  * E3: `_core.build_info()`;
  * E4: portable replacements for every `std::` distribution (dice, steals,
    shuffles, discrete, gamma, uniform_real).
* **Catanatron fork:**
  * canonical action order in `models/actions.py`;
  * `build_all` for 2–6 players;
  * a per-seat forfeit counter in `StdioPlayer`.
