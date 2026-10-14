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

### 1.6 Search

`SearchPool` (C++) runs many independent searches in lockstep (v1's fastest
design, now native): `collect()` advances every search to its next leaf and
writes the leaves' observations into one numpy batch; Python evaluates the
batch on the GPU (CUDA graphs at power-of-two sizes); `apply()` expands and
backs up. N-player: nodes back up a value per seat (max^n: each node picks by
the value of its own player to move); chance nodes are exact (dice, dev
draws, steals). Root options: PUCT with Dirichlet noise, or Gumbel root
(sequential halving) for low budgets; determinized roots from the tracker.

The **AlphaZero / expert-iteration loop** (`catanrl.train.alphazero`) is:
self-play games through `SearchPool` with a network (fresh, or a PPO
checkpoint) → replay buffer of (observation, visit-count policy, per-seat
outcome) → train → new network. Because the PPO network and the search
network are the same class with the same observation, the PufferLib-trained
GNN plugs straight into search (`mcts:ckpt=…,sims=…`).

## 2. Networks

`catanrl.models`:

* `graph` – Canopy Nexus-v3 style heterogeneous message passing (nodes ↔
  nodes, nodes ↔ tiles, nodes ↔ edges) with batched per-sample adjacency built
  from the topology block, so it runs on any map. Presets `graph-s`, `graph-m`.
* `mlp` – flat baseline over the same row (for ablations; padded maps only).

Heads: structured policy (above) and a per-seat value head (P sigmoids). A
checkpoint stores the network config and the observation spec it was trained
on, and refuses an env whose spec differs.

## 3. Agents, evaluation, experiments

Agent specs (strings, as in Catanatron's registry):
`policy:ckpt=…[,sample]`, `mcts:ckpt=…,sims=256,dets=1`, `belief:ckpt=…,worlds=16`,
`F`, `R`, and any Catanatron registry bot (`catanatron:AB:2`). The same spec
plays natively (fast evaluation on cppanatron), as a Catanatron `Player`, or
as a server client.

Evaluation: paired seats (every board played from every seat rotation),
fixed seeds, Wilson intervals, exact McNemar for paired comparisons,
head-to-head and N-player "win share vs expected 1/N".

Experiments are YAML files under `configs/experiments/` composed from
`configs/games`, `configs/maps` and `configs/models`, with an optional
`sweep:` grid. `catanrl run EXPERIMENT.yaml` writes `runs/<experiment>/<variant>/`
(resolved config, metrics jsonl, checkpoints, eval results) and a
`summary.md`. v1's research maps onto config axes:

| v1 experiment | v2 axis |
|---|---|
| architecture ablation (xdim/MLP/graph-s/graph-m) | `model.preset` sweep |
| information levels for actor/critic | `obs.level` (`private`/`public`/`full`), `obs.critic_level` |
| belief averaging of a full-info policy | agent `belief:…` |
| determinized MCTS / ISMCTS, dets × sims | agent `mcts:…,dets=K,sims=N` sweep |
| shaped vs win/loss reward, γ=0.99 vs 1 | `reward.*`, `train.gamma` |
| DAgger warm start from F | `train.algo: imitation` then `init_from:` |
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
clients (`catanrl bot --spec mcts:… ws://…`).

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
