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

## Versioning

A registration cannot change (name and description are fixed at `POST
/api/bots`; there is no endpoint to rename, edit or delete one) and its Elo,
record and head-to-head belong to it. So **each model we deploy is a new
registration, a version**, and `deploy/openhex.yaml` keeps every version ever
registered. Retired versions stay in the file, marked `retired: true`, for the
record and for `stats`; the runner skips them. The code lives in
`src/catanrl/bridge/openhex_deploy.py` (the file, the pins, the git check, the
key check, the stamps) and `src/catanrl/bridge/openhex_stats.py`.

**Scheme.** Names are `catanrl-<line>-v<N>`, where the line is `1v1`,
`1v1-search` or `4p`, and the version string is `<line>-v<N>`. The three bots
registered in October are v1 and keep their names: `catanrl-1v1` is `1v1-v1`,
`catanrl-1v1-search` is `1v1-search-v1`, `catanrl-4p` is `4p-v1`. A version
is one pinned agent spec under one registration. Never change a registered
version's spec: a different checkpoint, or different agent params
(`worlds`, `sims`, ...), is a new version. Bridge fixes may land under a
running version. Each game line records the commit that played it (below).

**An entry** (unknown keys are refused, so a typo cannot fall back to a default):

```yaml
bots:
  - name: catanrl-1v1-v2          # 3-32 characters, unique in the file
    mode: 1v1                     # 1v1 or 4p
    version: 1v1-v2               # required
    spec: BELIEF:ckpt=puffer-graph2/full@e3350#3355b5059503:worlds=16   # every ckpt pinned by #sha
    key_env: OPENHEX_KEY_1V1_V2   # unique among the entries that run
    bot_id: 0123456789abcdef      # optional: OpenHex's id once registered (checked against the key)
    tag: openhex-1v1-v2           # optional: the git tag it runs from
    evidence: [4b223fd273477fcf]  # optional: eval record ids that justified it
    deployed: 2026-10-26          # optional, free-form
    retired: false                # optional, default false
    retired_at: 2026-11-02        # optional, free-form; only with retired: true
```

**What the runner checks before it queues** (`catanrl openhex deploy/openhex.yaml`):

1. The file is valid.
2. Every checkpoint of every entry that runs is pinned by `#sha`. An unpinned
   ref (`@best`, or `@e3350` without `#`) is refused with the pinned ref it
   resolves to now and the whole spec, ready to paste. `--allow-unpinned`
   turns the refusal into a warning, for emergencies only. A pin whose sha is
   not the weights on disk fails when the spec resolves
   (`catanrl.runs.Run.resolve`, `Checkpoint.load`).
3. The checkout's play source (tracked files under `src/`, `engine/` and
   `third_party/`, with a submodule that is dirty or off its recorded commit
   counting as dirty) is clean, and the cppanatron build is
   `engine/cppanatron`'s HEAD. For an entry with a `tag`, HEAD's play source
   must be the tag's. A commit on top of the tag that touches only `deploy/`,
   `report/` or other files outside the play source is fine. Changes to
   `deploy/`, `runs/`, `report/`, `transcripts/`, `README.md` and untracked
   files are ignored. `--allow-dirty` runs anyway and stamps those bots
   `dirty: true`. The service user must be able to run git in the checkout
   (`git config --global --add safe.directory /opt/catanrl-v2` for that user).
   Otherwise the check fails like a dirty tree.
4. Every key variable is set, and `GET /api/bots/me` with each key returns
   that entry's name and mode, plus its id when `bot_id` is set. A key pasted
   under the wrong variable would otherwise play one model under another bot's
   name. Keys are never printed.
5. Only then do the agents load, and each bot writes its **stamp**.

**Stamps.** At startup each bot logs one line (`<bot>: version 1v1-v2,
provenance <id>: <spec> [<ckpt refs>] at <git describe>`) and appends its full
stamp to `<state>/<bot>/deploys.jsonl`, with the time and `provenance_id`. The
stamp holds:

- the version, and the spec as configured and as resolved (`canonical`);
- each checkpoint it loads (pinned ref and full `weights_sha256`);
- git: `commit`, `describe`, `dirty`, `tag`, `tag_commit`;
- `code_id` (`provenance.code_id("eval")`: the models, agents, search and envs
  code plus the engine, the same id eval records are keyed on);
- the engine build.

The bridge itself is outside `code_id`; the commit and `dirty` cover it. Every
line of `<state>/<bot>/games.jsonl` now carries `version`, `spec` and
`provenance_id`. `provenance_id` is a 12-hex hash of the stamp, so a restart on
the same code, spec and weights keeps it, and `grep <id> deploys.jsonl` gives
the full stamp of any game.

**Deploying a new version** (example: `catanrl-1v1-v2`):

1. Pick the checkpoint and the eval records that justify it.
2. Add the entry to `deploy/openhex.yaml`: name, mode, `version`, `key_env:
   OPENHEX_KEY_1V1_V2`, `tag: openhex-1v1-v2`, `evidence`, `deployed`, and no
   `bot_id` yet. To pin the spec, write it with `@best` and run `--check`,
   which refuses it and prints the pinned spec to paste. `catanrl ls` also
   shows the epoch.
3. Commit and tag: `git tag openhex-1v1-v2`, then push the branch and the tag
   (`git push origin main openhex-1v1-v2`; submodule branches first if they moved).
4. On the bot machine: `git fetch --tags && git checkout openhex-1v1-v2`, then
   `git submodule update` and `uv sync` if the engine or PufferLib moved.
5. `catanrl openhex deploy/openhex.yaml --check`. It loads every agent that
   runs, runs the git check, says `set`/`unset` per key, prints a row per bot
   (name, version, mode, spec, weights sha, tag, retired, key) and exits 0 or 1.
   It contacts nobody. Expect one problem only: the new key is unset.
6. Register by hand:
   `catanrl openhex deploy/openhex.yaml --describe catanrl-1v1-v2 > /tmp/body.json`
   prints the `POST /api/bots` body (name, mode, `coordinateFormat:
   "coordinates"`, and a description of at most 280 characters naming the
   version, the pinned spec, the commit and the repo; it contacts no one). Then
   `curl -sS -X POST https://open-hex.web.app/api/bots -H 'Content-Type: application/json' -d @/tmp/body.json`.
   The reply's `api_key` is shown once. Put `OPENHEX_KEY_1V1_V2=<key>` in
   `/etc/catanrl/openhex.env` yourself (root, mode 600), never in the repo or a
   log. Then add the reply's `bot.id` as `bot_id` (`deploy/` changes do not
   trip the git check; commit them on main).
7. `--check` again (`ok`), then restart the service between games (see
   "Operating"). The journal shows the new bot's stamp line, then `game ...
   as ...` when it is matched.

**The overlap.** The old version is not retired when the new one starts. Both
stay queued for a few days, until the new one has enough finished games to
compare. The 95% interval of a win rate near 50% is about ±10 points at 100
games and ±7 at 200 (`stats` prints it). This is worth it for four reasons:

- **Same days, same opponents.** The two records cover the same days against
  the same pool. Matchmaking is by availability, the pool changes from day to
  day, and the `ar-*` bots learn, so the old version's record from earlier
  weeks is not a fair baseline. Compare per opponent: `stats` prints that.
- **Direct games.** With both queued, OpenHex pairs them now and then.
  `stats` counts those games apart, as head-to-head between our versions.
- **Elo needs games.** A new registration starts at Elo 1500 and needs games
  before its rating means much.
- **Easy rollback.** While the old version still plays, rolling back is
  flipping one flag.

The costs: CPU (each search bot uses about 40% of a core while it plays), and
games between our own versions, which move both ratings.

**Code during an overlap.** One process runs from one checkout. If the new tag
changed nothing in the play source (a new checkpoint only, the usual case), the
old entry's tag still holds and both run together. If the play source changed,
the runner refuses the old entry, because its tag is not the running code. Then
either:

- set the old entry's `tag` to the new tag, so it plays the new code. The
  overlap then compares the two models on the same bridge, and `deploys.jsonl`
  records the switch.
- or run it from its own checkout of its old tag, as a second service with
  `--only <old names>` (comma-separated; the same `--state` root is fine, one
  dir per bot).

`--allow-dirty` also works, but it stamps the old bot dirty.

**Retiring.** Set `retired: true` and `retired_at` on the old entry, restart
between games, and take it off OpenHex's queue, because a stopped bot that
stays queued gets matched and loses on time:
`curl -sS -X POST https://open-hex.web.app/api/queue/leave -H "Authorization: Bearer $OPENHEX_KEY_1V1" -H 'Content-Type: application/json' -d '{"mode": "1v1"}'`
(with the env file loaded). Keep its key in the env file; you need it to roll back.

**Rolling back.** The registrations and keys are permanent, so a rollback only
flips flags. Set the bad version `retired: true` and leave its queue. If the
old version was already retired, set it `retired: false` again and drop its
`retired_at`. Check its `tag` holds the running code (or move it), then
`--check` and restart. Never point an existing registration at a different
spec instead: its record would mix two models.

**Stats.** `catanrl openhex stats deploy/openhex.yaml` reads OpenHex's public
Firestore only. It needs no key, contacts no API, and includes retired
entries. It finds each bot by `bot_id`, or by name and mode in the public
`bots` collection when the entry has none. It prints a row per version:
finished games, wins, win rate with its Wilson 95% interval, aborted and
ongoing games, current Elo, and first and last game. Then it lists each
version's opponents (games, wins), then the games between our own versions.
Finished means `completed` or `aborted`. When a seat misses its 60 s deadline,
OpenHex gives the game to the best other seat and rates it like any other, so
the ladder's own record counts aborted games too; `stats` notes any
disagreement with that record. Games still `active` are only counted as
ongoing.

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
5. **Check before enabling:** `pytest tests/bridge/test_bridge_openhex.py
   tests/bridge/test_bridge_openhex_deploy.py` and `catanrl openhex
   deploy/openhex.yaml --check` (see "Versioning"), then run one bot in the
   foreground with the env loaded:
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
