# Useless Overview

A continuation of [my original catanrl project](https://github.com/mcrco/catanrl).

When I first started working on this in September 2025, my goal was to make the
first superhuman (and super-heuristic) Catan bot and learn reinforcement
learning along the way because my school didn't offer a RL class and I wanted to
learn it while also getting coursework units for it by writing it off as an
"undergraduate research project". At the time, LLMs were not yet good enough to
reliably implement things like PPO or AlphaZero or port the `catanatron` engine
to another (compiled) language.

But things have changed in the past year! In terms of personal progress, I
learned about the basic RL taxonomy and the algorithms, wrote many failed
implementations, got a bot that beat Catanatron's `ValueFunctionPlayer` with a
90% winrate. And outside what I did, other people made really good bots and
servers for bots (join the Catanatron discord to see), and LLMs are now good
enough to essentially do the whole process.

So now my new goal in this repo is to create a recipe for the best bot out
there and log my process along the way. Right now, I have some raw policy bots
queueing anyone can play against [here](catan.marcoya.ng). I keep track of an
experiment log in `report/report.pdf`.

# catanrl-v2

Catan bots for any number of players (2–6) on any map, a research harness
where every experiment is a reproducible config, and a server where people
and bots play verifiably fair games.

v2 keeps a lean core and leans on open libraries:

| Piece                 | What it does                                                                                                                                                                                                 | Where                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| **cppanatron**        | C++20 rules engine (parity-tested against Catanatron), explicit maps of any shape, seat-relative graph observations, map-agnostic action codec, threaded multi-game env, hand tracker, batched N-player MCTS | `engine/cppanatron` (submodule, branch `v2`)             |
| **Catanatron** (fork) | reference rules, per-map board graphs, the fair game server (protocol 4), the terminal UI                                                                                                                    | `third_party/catanatron` (submodule, branch `v2`)        |
| **PufferLib 3**       | PPO (two small patches: acting-row losses and a value-loss hook)                                                                                                                                             | `third_party/pufferlib` (submodule, branch `catanrl-v2`) |
| **catanrl**           | maps and game configs, graph networks, PPO / DAgger / AlphaZero trainers, native agents and evaluation, the experiment store, the view bridge to the server and TUI                                          | `src/catanrl`                                            |

The architecture and its reasons are in [DESIGN.md](DESIGN.md); the
experiment system's module plan and test list in
[docs/experiments-implementation.md](docs/experiments-implementation.md);
the server protocol in
`third_party/catanatron/catanatron/catanatron/server/PROTOCOL.md`.

The submodule branches exist only locally until they are pushed.

## Setup

```bash
git clone --recurse-submodules <this repo> && cd catanrl-v2
uv sync --extra cu130         # builds cppanatron and PufferLib (NO_OCEAN) into .venv; torch for
                              # CUDA 13 (Turing and newer), or --extra cu126 for Pascal (GTX 1080)
uv run pytest -q              # fast tier (~6 min); -m train / -m gpu / -m runs for the slow tiers
```

Pick the torch extra once; `uv run` keeps it. Where nvcc's CUDA is not the
wheel's (CUDA 12.9 on a machine with the cu130 torch, say), build PufferLib
without it: `env -u CUDA_HOME PATH=$(echo "$PATH" | tr : '\n' | grep -vi cuda | paste -sd:) uv sync --extra ...`
(its CUDA extension is only used by training).

`uv sync` rebuilds the engine whenever its sources or commit change. Runs and
evals refuse an engine built from another commit (`CATANRL_ALLOW_STALE_BUILD=1`
overrides, and is recorded).

## Day to day

```bash
# experiments: YAML under configs/experiments, outputs under runs/<experiment>/<variant>/
uv run catanrl validate puffer-graph2
uv run catanrl run puffer-graph2                    # every variant, one process each; relaunch resumes
uv run catanrl run arch-lineage --only 'graph-s-*' --next   # several terminals share a sweep
uv run catanrl ls 'puffer-graph2/*'
uv run catanrl pin puffer-graph2/full@e1500 ref --note "report"

# W&B mirrors a run (DESIGN 6.10: it never decides anything): set meta.wandb.project in the
# experiment YAML or CATANRL_WANDB_PROJECT, and a relaunch resumes the same wandb run. Smoke
# runs and tests stay out (CATANRL_WANDB_SMOKE=1 opts a smoke run in). Runs made without it
# get the identical wandb run after the fact (idempotent; a re-sync keeps the run's group and tags):
uv run python tools/wandb_backfill.py ppo-mixed/main --project catanrl-v2

# evaluation (cached, seat-balanced, Wilson intervals; records keyed by everything that decides them)
uv run catanrl eval POLICY:puffer-graph2/full@ref F --suite heldout-2p
uv run catanrl eval MCTS:puffer-graph2/public@ref:sims=1024:dets=1 POLICY:puffer-graph2/public@ref --suite heldout-2p

# play: a local terminal game, or a seat on a catanatron-server
uv run catanrl play --seats human,MCTS:puffer-graph2/public@ref:sims=1024:dets=1,F,F
uv run catanatron-server --port 8765 &              # browser client at http://localhost:8765/
uv run catanrl bot --spec MCTS:puffer-graph2/public@ref:sims=1024:dets=1 \
    --new-room '{"config": {"num_players": 2}, "seats": [null, "F"]}' ws://localhost:8765
```

```python
import catanrl
run = catanrl.load_experiment("puffer-graph2/full")
net = run.build_policy("best", device="cuda")
bot = run.agent("MCTS", which="ref", sims=1024, dets=1).player(Color.RED)   # a Catanatron Player
rec = catanrl.evaluate(["POLICY:puffer-graph2/full@ref", "F"], "heldout-2p")
```

One agent spec string (`POLICY:…`, `MCTS:…:sims=N:dets=K`, `BELIEF:…:worlds=K`,
`TRADE:…:critic=…`, or any Catanatron registry bot) plays natively, as a
Catanatron `Player`, as a TUI seat and as a server bot (`TRADE`, which offers,
answers and confirms domestic trades by its critic's win probabilities, plays
natively only).
`MCTS:…:pending=P` sends P leaves to the network
per forward (virtual visits while they wait): at the default 1 a decision is
`sims` forwards of one row each, which no GPU speeds up; at 32 a 16k-simulation
decision takes about a second on a GPU. Fair agents use only their seat's information:
`POLICY` at the `private` level (or `public` in two-player games, where every
resource movement is seen by both seats), `MCTS` with `dets ≥ 1`, `BELIEF`,
and `TRADE` with a `private` critic unless `oracle=true` (true hands). The
worlds those three sample come from the seat's hand tracker:
`tracker=pool` (default) deals the unknown cards uniformly, `tracker=exact`
draws from the exact posterior over hidden steals (DESIGN 1.5). `MCTS` also
takes `reuse=true` (a seat's search is re-rooted at the action it played
when the game followed exactly; on the native harness and the server bridge), `transpositions=true`
(the engine's transposition table) and `dedup=false`
(execution-only: byte-identical leaf rows are otherwise evaluated once);
`reuse` and `transpositions` enter the record key only when set.

## Configs

| Directory             | Holds                                                                                                             |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `configs/games`       | player count, map generator and rules (`base-2p`, `base-4p`, `random-4p`, `hexagon-3p`, `extension-6p`, …)        |
| `configs/models`      | network presets (`graph-s`, `graph-m`, `mlp`, `cnn`, `graph-tiny`, `cnn-tiny`)                                    |
| `configs/train`       | trainer defaults (`ppo`, `imitation`, `alphazero`)                                                                |
| `configs/suites`      | evaluation boards: `select-*` (checkpoint selection) and `heldout-*` (reported numbers), disjoint by construction |
| `configs/experiments` | experiments: stages, sweeps, select and final evals                                                               |

v1's research maps onto config axes: architecture (`model.preset`),
information levels (`obs.level`, `obs.critic_level`), belief averaging
(`BELIEF:`), determinized search (`MCTS:…:dets=K:sims=N`), reward and γ
(`train.*`), staged runs (`stages` + `init_from`; DAgger warm starts are
superseded by from-scratch self-play PPO but still run as `train.algo:
imitation`), AlphaZero
(`train.algo: alphazero`; from a network, its evals default to head to head
against that network, `${init}`), and player-count and map generalization
(`env.scenarios` mixtures with held-out maps). See `DESIGN.md` §3 and §6.

## Status

- **Engine:** explicit maps of any shape and 2–6 players in lockstep parity
  with Catanatron (including Longest Road tie-breaks after cuts); positions at
  every prompt via `apply_setup`; seeded randomness portable across compilers.
- **PPO:** `puffer-graph2` replicates v1's PufferLib graph2 recipe (shared
  trunk, Muon with a cosine LR, pure self-play). Against native F, 400 games
  each: 62.3% / 74.0% / 80.5% at 13 / 26 / 40M agent steps, against v1's
  56% / 70.5% / 78%.
- **Search:** batched N-player MCTS (PUCT or Gumbel root, exact chance,
  determinized roots from the hand tracker); about 4–5 MB per 256-simulation
  tree.
- **AlphaZero:** the loop is built, deterministic and checked end to end; a
  first run from a PPO network with 64/16-simulation Gumbel targets degraded
  the policy (no bug found: shallow-search targets cannot improve a strong
  policy). Next attempt: targets from ≥ 1024 simulations. A run from a
  network is now judged head to head against that network, raw policy and
  search, with vs F as a secondary number (DESIGN 6.2).
- **Server:** protocol 5, specified in `PROTOCOL.md` as invariants, each with a
  client check, a verifier check and tests: commit–reveal seeds bound to the
  room and server key, signed hash-chained messages, signed moves with
  receipts, and a total verifier whose verdicts separate provable server
  misbehaviour from the client's word. Six adversarial review rounds, then a
  transcript-mutation fuzzer in the default test tier. The server still sees
  hidden state during a game (see the threat model).
- **TUI:** `catanatron-tui` / `catanrl play`, local games with debug controls
  and remote server play.
