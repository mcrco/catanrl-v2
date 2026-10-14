# catanrl-v2

Catan bots for any number of players (2–6) on any map, a research harness
where every experiment is a reproducible config, and a server where people
and bots play verifiably fair games.

v2 keeps a lean core and leans on open libraries:

| Piece | What it does | Where |
|---|---|---|
| **cppanatron** | C++20 rules engine (parity-tested against Catanatron), explicit maps of any shape, seat-relative graph observations, map-agnostic action codec, threaded multi-game env, hand tracker, batched N-player MCTS | `engine/cppanatron` (submodule, branch `v2`) |
| **Catanatron** (fork) | reference rules, per-map board graphs, the fair game server (protocol 4), the terminal UI | `third_party/catanatron` (submodule, branch `v2`) |
| **PufferLib 3** | PPO (two small patches: acting-row losses and a value-loss hook) | `third_party/pufferlib` (submodule, branch `catanrl-v2`) |
| **catanrl** | maps and game configs, graph networks, PPO / DAgger / AlphaZero trainers, native agents and evaluation, the experiment store, the view bridge to the server and TUI | `src/catanrl` |

The architecture and its reasons are in [DESIGN.md](DESIGN.md); the
experiment system's module plan and test list in
[docs/experiments-implementation.md](docs/experiments-implementation.md);
the server protocol in
`third_party/catanatron/catanatron/catanatron/server/PROTOCOL.md`.

The submodule branches exist only locally until they are pushed.

## Setup

```bash
git clone --recurse-submodules <this repo> && cd catanrl-v2
uv sync                       # builds cppanatron and PufferLib (NO_OCEAN) into .venv
uv run pytest -q              # fast tier (~6 min); -m train / -m gpu / -m runs for the slow tiers
```

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
or any Catanatron registry bot) plays natively, as a Catanatron `Player`, as a
TUI seat and as a server bot. Fair agents use only their seat's information:
`POLICY` at the `private` level (or `public` in two-player games, where every
resource movement is seen by both seats), `MCTS` with `dets ≥ 1`, and
`BELIEF`.

## Configs

| Directory | Holds |
|---|---|
| `configs/games` | player count, map generator and rules (`base-2p`, `base-4p`, `random-4p`, `hexagon-3p`, `extension-6p`, …) |
| `configs/models` | network presets (`graph-s`, `graph-m`, `mlp`, `graph-tiny`) |
| `configs/train` | trainer defaults (`ppo`, `imitation`, `alphazero`) |
| `configs/suites` | evaluation boards: `select-*` (checkpoint selection) and `heldout-*` (reported numbers), disjoint by construction |
| `configs/experiments` | experiments: stages, sweeps, select and final evals |

v1's research maps onto config axes: architecture (`model.preset`),
information levels (`obs.level`, `obs.critic_level`), belief averaging
(`BELIEF:`), determinized search (`MCTS:…:dets=K:sims=N`), reward and γ
(`train.*`), DAgger warm starts (`stages` + `init_from`), AlphaZero
(`train.algo: alphazero`), and player-count and map generalization
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
  policy). Next attempt: targets from ≥ 1024 simulations.
- **Server:** protocol 4 with commit–reveal seeds, signed and hash-chained
  messages, signed moves, receipts and an independent verifier; hardening
  continues (see the threat model in the server README).
- **TUI:** `catanatron-tui` / `catanrl play`, local games with debug controls
  and remote server play.
