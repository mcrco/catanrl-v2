# OpenHex bots: handoff for redeploying on another machine

Written 2026-10-20 after the bots were stopped on the OCI VM. The bridge is
`src/catanrl/bridge/openhex.py` (an OpenHex game replayed on cppanatron for one
seat, see its module doc); the CLI is `catanrl openhex deploy/openhex.yaml`.

## Status

- All three bots are **stopped and disabled** on the OCI VM (`ubuntu@170.9.8.49`,
  unit `catanrl-openhex`) since 2026-10-20 07:26 UTC, and have left OpenHex's
  queues. `sudo systemctl enable --now catanrl-openhex` there brings them back.
- main carries the two bridge fixes of 2026-10-20: a bot thread survives any
  error in a game (4a1d993) and a roll OpenHex paid past its bank replays on a
  larger bank (1632935). One known gap remains (see "Known issues").

## The bots (registrations are permanent; keys cannot be re-issued)

| name | mode | spec (deploy/openhex.yaml) | bot id | key env var | record at stop |
|---|---|---|---|---|---|
| catanrl-1v1 | 1v1 | `BELIEF:ckpt=puffer-graph2/full@best:worlds=16` | 05c4a25c6c0e46e1 | OPENHEX_KEY_1V1 | 54 of 110, Elo ~1400 |
| catanrl-1v1-search | 1v1 | `MCTS:ckpt=puffer-graph2/full@best:sims=1024:dets=1` | b8334c08b8d34a19 | OPENHEX_KEY_1V1_SEARCH | 3 of 4 |
| catanrl-4p | 4p | `POLICY:ckpt=ppo-4p/main@best` | 068460ec188f4a2c | OPENHEX_KEY_4P | 0 games (see 4p note) |

The keys live only in `/etc/catanrl/openhex.env` on the OCI VM (root-readable,
`sudo cat`). Copy that file; never commit it or print it into a log.

## What the new machine needs

1. **The repo with submodules.** `git clone --recurse-submodules
   https://github.com/mcrco/catanrl-v2` (submodules: mcrco/cppanatron branch v2,
   mcrco/catanatron branch v2, mcrco/PufferLib branch catanrl-v2, pinned by the
   superproject). Then `uv sync` (builds cppanatron and PufferLib; needs cmake
   and a C++20 compiler).
2. **CPU-only torch** if the machine has no CUDA: in `pyproject.toml` change the
   torch index `url = "https://download.pytorch.org/whl/cu130"` to
   `.../whl/cpu`, re-resolve (`uv lock`), and `uv sync`. The VM did exactly
   this and marked pyproject.toml/uv.lock skip-worktree so pulls do not clobber
   it. The bots run on `--device cpu` (the default).
3. **The two checkpoint directories**, about 10 MB each, under the runs root:
   `runs/puffer-graph2/full` (keep `aliases.jsonl`, `config.json`, `run.json`,
   `checkpoints/e003350.pt`; `best` resolves to e3350, weights sha256
   3355b505...) and `runs/ppo-4p/main` (`checkpoints/e003650.pt`). The runs
   root is found through git; if the checkout is not a git repo for the service
   user (or git complains about ownership), set `CATANRL_RUNS_DIR=<checkout>/runs`.
4. **The service.** `deploy/catanrl-openhex.service` assumes the checkout at
   `/opt/catanrl-v2`, a system user `catanrl`, the env file at
   `/etc/catanrl/openhex.env`, state under `/var/lib/catanrl/openhex` (one dir
   per bot: `games.jsonl`, `boards/`). Adjust paths and user; keep
   `OMP_NUM_THREADS=1` unless you want the search to use more cores (the MCTS
   spec also takes `threads=N`).
5. **Check before enabling:** `pytest tests/bridge/test_bridge_openhex.py`, then
   run one bot in the foreground with the env loaded:
   `catanrl openhex deploy/openhex.yaml --only catanrl-1v1-search --state /tmp/openhex-test`.
   It logs `game <id> as <color> against [...]` when matched and a JSON record
   when the game ends. Stop it with Ctrl-C (the queue entry is left behind: see
   "Operating").

## OpenHex facts that matter

- Site and API: https://open-hex.web.app, `/api/...` with `Authorization: Bearer <key>`.
  Register once with `POST /api/bots` (`name`, `mode`, `coordinateFormat:
  "coordinates"`, `description`; key returned once). Queue: `POST /api/queue/join`
  `{"mode"}`, `POST /api/queue/leave`, `GET /api/queue/status`. Games:
  `GET /api/games?status=active`, `GET /api/games/{id}/state`,
  `GET /api/games/{id}/events?after=<seq>`, `POST /api/games/{id}/actions`
  `{"action": {...}, "duration_ms": n}`. A refused action can come back as HTTP 200
  with an error body; `/state` lags `/events` slightly.
- **60 seconds per action; a missed deadline aborts the game** (a loss).
- Rules: 1v1 is 15 VP, discard limit 9 (discard when holding 10+), friendly
  robber, balanced dice; 4p is 10 VP, limit 7, normal robber, random dice. The
  board is a standard 19-hex layout with cube coordinates; the bridge maps
  vertices to Catanatron nodes by geometry. No dev card may be played before the
  roll (the policy sometimes wants to; the bridge rolls first).
- Matchmaking pairs whoever is waiting, not by rating, so our own bots get
  matched against each other when both are queued. The 4p queue has returned
  HTTP 500 on join since 2026-10-19 (server side); catanrl-4p has never played.
- Public data (no key): Firestore REST
  `https://firestore.googleapis.com/v1/projects/open-hex/databases/(default)/documents/`
  `bots?pageSize=200` (every bot's `elo1v1`, `games1v1`, `wins1v1`, description),
  `games/<id>` (`boardJson`, `stateJson`, `configJson`, `turnOrderJson`,
  `players`, `durationStats`), `games/<id>/events?pageSize=500` (`eventJson`,
  `actionJson`, `createdAt`, `seq`; follow `nextPageToken`), and
  `documents:runQuery` with `ARRAY_CONTAINS` on `playerBotIds` for a bot's games.
- Opponents seen: rehex-2.11 (rules-based, beat the plain bot 6-0),
  colonist-ai-r66, Asterism, and five `ar-*` "ExIt + PPO research bots" that
  learn; the plain bot is a deterministic argmax policy, so expect learners to
  adapt to it.

## Operating

- **Restart only between games.** A restart mid-game can resume (the board is
  kept on disk and the public record is the fallback) but do not rely on it:
  wait for a journal line `catanrl-1v1: {"time"` (or `catanrl-1v1-search`) and
  restart within a second, as `deploy/` scripts here did:
  `sudo journalctl -u catanrl-openhex -f -n 0 | grep --line-buffered -m1 'catanrl-1v1: {"time"' && sudo systemctl restart catanrl-openhex`.
- **On stop, leave the queues**, or OpenHex will match the dead bot and abort
  games: `POST /api/queue/leave` with each key and mode.
- Each game appends a record to `<state>/<bot>/games.jsonl`: `decisions`,
  `fallbacks` (default moves after refusals or undecidable positions),
  `mismatches` (replay-vs-OpenHex differences that lasted three polls),
  `bank_short` (replays restarted on a larger bank), `production_mismatch`
  (rolls the engine paid differently from OpenHex; see Known issues). Nonzero
  values other than `decisions` deserve a look.
- The journal is the only log (`sudo journalctl -u catanrl-openhex`). A bot that
  hits an unexpected error logs a traceback, abandons that game (OpenHex aborts
  it at the deadline) and queues again.

## Known issues

- **Withheld rolls.** OpenHex pays a roll in full unless one player's share
  alone exceeds the bank, and its bank goes negative; cppanatron pays nobody a
  resource the bank cannot pay everyone. The bridge now follows the first
  direction exactly (`BankShort` -> replay on a larger bank). The other
  direction, OpenHex withholding where the engine (richer by that margin) pays,
  cannot be undone because the engine has no bank setter: the replay keeps
  phantom cards and the game is usually lost at the next 7. Seen in 2 of 1027
  recorded rolls (both in game 881a71bc7f53411a, the fixture
  `tests/fixtures/openhex/game-881a71bc.json`). Fix: add
  `Game::set_resource_bank` to cppanatron and undo the payout in
  `Replay._check_production`; note the engine binary is part of the eval code
  id (`src/catanrl/provenance.py`), so landing it in main invalidates existing
  eval finals for `catanrl run`.
- The reported `duration_ms` is 0 for the first action of each plan (cosmetic;
  OpenHex shows it on the bot page).
- `plan()` flushes a held roll with the engine's rule if asked to move before
  the roll's production event arrived; production always follows within a
  millisecond, so this only shows up when feeding events one at a time offline
  (a warning "production after its roll was replayed").

## Cost (measured on the OCI VM: 4-core Neoverse-N1, CPU torch, one thread)

| agent | per decision, median | max seen | game length |
|---|---|---|---|
| BELIEF worlds=16 (catanrl-1v1) | 0.11 s | 0.23 s | 4-8 min |
| MCTS sims=256 | 1.7 s | 3.5 s | |
| MCTS sims=1024 (catanrl-1v1-search) | 6.9 s | 13.9 s | 15-22 min |

The bridge itself is negligible next to the agent: on a desktop CPU the seat
(board mapping, engine game, tracker) builds in 3 ms, replaying an event costs
0.03 ms, and a decision's encoding, masking and bookkeeping about 1 ms; the
process polls `/state` and `/events` about once a second while waiting. The
process holds about 300 MB with all three agents loaded (about 4 s to load each
at start). CPU use averaged 4% of a core with only the BELIEF bot playing and
about 40% of a core while the search bot plays.

## Offline analysis

To reproduce a ladder game: fetch its record and events from Firestore (above),
build `OpenHexSeat(native, game_id, {"board": boardJson, "turnOrder":
turnOrderJson, "config": configJson}, our_color)` with
`native_for(spec)`, redact the events as `tests/bridge/test_bridge_openhex.py`
does (drop `card` from the other seat's `devCardBought`, `resource` from steals
we are not part of), feed them in seq order and call `seat.plan(...)` where the
seat moves. Decisions reproduce the live ones move for move except for the
pre-roll knights OpenHex forbids. Per-move latency is the gap between
consecutive events' `createdAt`.
