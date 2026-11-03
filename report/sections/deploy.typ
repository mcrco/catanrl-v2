#import "../macros.typ": *

= Deployment <sec:deploy>

This section records which bots play people, on catan.marcoya.ng and on OpenHex, and why each was picked (@tab:deploy-bots).
- *1v1.* The site's 1v1 table serves `BELIEF` over `puffer-graph2/full@e3350` with 16 worlds until it moves to `ppo-2p-fr3` on OpenHex's rules (@exp:deploy-site-openhex-rules). On OpenHex, that net already plays as `catanrl-1v1-v2` (@exp:deploy-openhex-v2).
- *A cheaper 1v1 bot.* The fair `public` network (`puffer-graph2/public@e3600`) plays as well as the belief-averaged bot with one forward pass, no sampled worlds and no tracker, so it is the bot to serve raw (@sec:recipe).
- *Islands.* A third site table seats the generalist `ppo-mixed` on random islands.
- *Search reuse* is available to the deployed bots but neutral in strength.

#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([Table or bot], [Agent spec]),
    tgroup(2)[catan.marcoya.ng (`deploy/house.yaml`)],
    ["1v1 OpenHex rules vs ppo-2p-fr3" \ #sub[waits on the house machine]], [`BELIEF:ckpt=ppo-2p-fr3/full@e4350:worlds=16`],
    ["1v1 vs puffer-graph2" \ #sub[served until the swap]], [`BELIEF:ckpt=puffer-graph2/full@best:worlds=16`],
    ["4p vs ppo-4p"], [`POLICY:ckpt=ppo-4p/main@best`],
    ["4p island vs ppo-mixed" \ #sub[a fresh 15–30 tile island per room]], [`POLICY:ckpt=ppo-mixed/main@best`],
    tgroup(2)[OpenHex (`deploy/openhex.yaml`)],
    [`catanrl-1v1` (v1)], [`BELIEF:ckpt=puffer-graph2/full@e3350:worlds=16`],
    [`catanrl-4p` (v1)], [`POLICY:ckpt=ppo-4p/main@e3650`],
    [`catanrl-1v1-search` (v1)], [`MCTS:ckpt=puffer-graph2/full@e3350:sims=65536:dets=1:` \ `pending=64:reuse=true:transpositions=true:kernels=fused`],
    [`catanrl-1v1-v2`], [`BELIEF:ckpt=ppo-2p-fr3/full@e4350:worlds=16`],
    [`catanrl-1v1-search-v2` \ #sub[not registered yet]], [`MCTS:ckpt=ppo-2p-fr3/full@e4350:sims=65536:dets=1`],
  ),
  caption: [What is deployed, as the two deploy files in this checkout list it (each spec is pinned there by its weights' sha, left out here). `@best` is e3350 for `puffer-graph2/full`, e3650 for `ppo-4p/main` and e3550 for `ppo-mixed/main`. `deploy/house.yaml` already names fr3 for the site's 1v1 table; until the house machine runs the new Catanatron, that table still serves `puffer-graph2` (@exp:deploy-site-openhex-rules).],
) <tab:deploy-bots>

== Tables on catan.marcoya.ng

=== The 4p island table #chip("done") <exp:deploy-island-table>
#meta(
  dates: "2026-10-22 18:59 utc (live)",
  owner: "marco",
  config: "deploy/house.yaml",
  wandb: (("ppo-mixed-main-f4b771c0", "ppo-mixed"), ("ppo-mixed-blob-main-14c55ba5", "ppo-mixed-blob")),
  records: ("7c17d0ae", "75cab037"),
)
#tldr[catan.marcoya.ng has a third table, "4p island vs ppo-mixed": the generalist `ppo-mixed@best` on a fresh 15–30 tile island in every room. On random islands it beat the island-trained candidate, 73.9 to 67.4 against three F. Marco's two recorded games on it both verified, and the bots won both.]

*Question.* Which network should play a table of random island maps?

#lead[Setup.]
- Two candidates on `heldout-random-4p` (blob-shaped boards of 15–30 tiles), each against three F. `ppo-mixed` never trained on islands; `ppo-mixed-blob` did, and failed its deploy gate (@sec:generalist).
- The island table runs next to the two existing tables (@tab:deploy-bots).
- Before going live, a preflight and a loopback smoke ran on the bot VM's own stack: two island games, all verified, 11 ms a decision.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([Candidate], [Share vs three F], [Record]),
    [`ppo-mixed`, zero-shot on islands \ #sub[`ppo-mixed/main@e3550`]], [*73.9* #ci[72.5][75.3]], [`7c17d0ae`],
    [`ppo-mixed-blob`, trained on islands \ #sub[`ppo-mixed-blob/main@e7600`]], [67.4 #ci[65.9][68.8]], [`75cab037`],
  ),
  caption: [The two island candidates on `heldout-random-4p`: slot 0's score share, 4,000 games each; parity is 25.],
) <tab:deploy-island>

On the live table, Marco's two recorded games both verified with no fallbacks; the bots won both.

*Reading.* `ppo-mixed` is the stronger island bot until one of the blob run's three changes is isolated (@sec:generalist).

#repro(```sh
catanrl eval "POLICY:ckpt=ppo-mixed/main@e3550" F F F --suite heldout-random-4p
catanrl eval "POLICY:ckpt=ppo-mixed-blob/main@e7600" F F F --suite heldout-random-4p
catanrl house deploy/house.yaml   # On the bot VM: keeps the three tables open
```)

=== The friendly-robber net on the site's rules #chip("done") <exp:deploy-site-rules-2p>
#meta(
  dates: "2026-10-27 03:41–04:20 utc",
  owner: "marco",
  config: "configs/experiments/site-rules-2p.yaml",
  wandb: (("site-rules-2p-main-f1663977", "site-rules-2p"),),
  records: ("b692e3b3", "fc9ab6e1"),
)
#tldr[On the site's own 1v1 rules (the normal robber, two fair dice), `ppo-2p-fr3/full@e4350` beats the site's 1v1 bot 55.8% head to head, and beats F 96.52% to that bot's 95.16% (paired +1.36 points, p = 0.0002). The net last trained on OpenHex's rules is the stronger 1v1 bot on both rule sets.]

*Question.* Does the friendly-robber net (fr3) beat the site's 1v1 bot on the rules the site plays, which fr3's last run did not train on?

*Setup.* Marco wanted fr3 on the site's 1v1 table, measured first: the site's server offered neither the friendly robber nor balanced dice, so its 1v1 room plays `base-2p`.
- Suite: `heldout-2p` (15 VP, discard above 9, the normal robber, two fair dice), 5,000 games a row, `POLICY` on both sides, on the CPU next to discount-2p.
- The fr3 row against F pairs game for game with `puffer-graph2`'s own row against F (`ff8528a4`, same schedule; the records' code ids differ).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, left, right, left),
    header: ([Agent], [Opponent], [Score], [Record]),
    [fr3 \ #sub[`ppo-2p-fr3/full@e4350`]], [F], [*96.52* #ci[95.98][96.99]], [`b692e3b3`],
    [The site's 1v1 bot, earlier row \ #sub[`puffer-graph2/full@e3350`]], [F], [95.16 #ci[94.53][95.72]], [`ff8528a4`],
    [fr3], [The site's 1v1 bot], [*55.80* #ci[54.42][57.17]], [`fc9ab6e1`],
  ),
  caption: [The friendly-robber net against the site's 1v1 bot on the site's rules (`heldout-2p`), 5,000 games a row.],
) <tab:deploy-site-rules-2p>

Paired on the same games against F, fr3 wins 194 that `puffer-graph2` loses and loses 126 that it wins: +1.36 points, p = 0.0002. Games run 75.2 turns on average against F and 79.0 head to head.

*Reading.* The fr3 net can take the site's 1v1 table on either rule set. With OpenHex's rules now on the site's server (@exp:deploy-site-openhex-rules), the table can play the rules fr3 trained on.

#repro(```sh
catanrl run configs/experiments/site-rules-2p.yaml
python tools/search_probe.py --pair runs/ppo-2p-fr3/full/evals/heldout-2p/b692e3b3c4738378.json \
  runs/puffer-graph2/full/evals/heldout-2p/ff8528a4939c421a.json
```)

=== OpenHex's 1v1 rules on the site's server #chip("running") <exp:deploy-site-openhex-rules>
#meta(
  dates: "2026-10-27 11:52 utc (server live; the 1v1 table waits on the house machine)",
  owner: "marco",
  config: "deploy/house.yaml (draft: table ppo-2p-fr3)",
  wandb: (),
  records: (),
)
#tldr[catan.marcoya.ng's server now offers OpenHex's 1v1 rules as two room options, `robber_mode: friendly_public` and `dice_mode: balanced`. They follow cppanatron exactly, and old rooms and transcripts are unchanged. The 1v1 table moves to `ppo-2p-fr3` on these rules once the house machine runs the new Catanatron.]

*Question.* Can the site's server play OpenHex's 1v1 rules, so that its 1v1 table plays the rules new 1v1 bots train on (Marco's ask)?

*Setup.* The server refused Catanatron's own friendly robber, which counts hidden victory point cards (so a legal robber move would reveal them), and had no balanced dice. Catanatron `da239a6` (branch `v2`) adds two room keys. Both are left out of a config at their defaults, so old configs keep their canonical form.
- `robber_mode: friendly_public`: a tile with an opponent at 2 public points or fewer is off limits, unless every tile is. A knight's tile is judged on the points before its largest army award, as OpenHex does. Catanatron's own `friendly_robber` is still refused.
- `dice_mode: balanced`: OpenHex's 36-card deck with recency and seven balancing, with cppanatron's integer weights. The dice state is public in every view, and the verifier replays the rolls from the revealed seeds.
- The bridge reads both from the view, so a house seat's mirror plays the same rules.

#lead[Result.]
- *Parity.* 60 `base-2p-fr` games in lockstep with cppanatron: every step's legal actions are equal (about 30,000 rolls and 5,500 robber prompts), and so is the dice state for every roller.
- *Old games.* Games under the old modes give the same document digests as before, and a transcript recorded on the old code verifies.
- *Tests.* 222 Catanatron tests pass on the new code, as does catanrl's fast suite (873 tests).
- *Smoke.* A local server running the draft table, `BELIEF` on fr3 against F: 4 games, all 8 transcripts verify, no desync in the seat's mirror.
- *Deploy.* The wheel went onto the GCP server at 11:52 UTC with no game in progress. The server key and app.js are unchanged, and the three house tables reconnected.

*Loose ends.* Two things are left as they are, since each changes a code id (Marco's call): catanrl's Catanatron harness still builds Catanatron's hidden-VP robber for friendly-robber configs, and `CatanrlPlayer`'s mirror rolls fair dice. No record is affected, since the harness refuses balanced dice and every friendly-robber config has them.

*Next.* On the house machine: install Catanatron `da239a6` (the client verifies each transcript with its own Catanatron, and an old one refuses the new keys), the three bridge files and the fr3 checkpoint. Then swap the 1v1 table for `ppo-2p-fr3` (room "1v1 OpenHex rules vs ppo-2p-fr3"), as `deploy/house.yaml` already lists it.

#repro(```sh
cd third_party/catanatron && uv build --wheel
gcloud compute scp dist/catanatron-3.3.0-py3-none-any.whl catanatron:/tmp/ --project catanatron-261015 --zone us-west1-b
# On the server: check /api/rooms for games in progress, then
sudo /opt/catanatron/bin/pip install --force-reinstall --no-deps /tmp/catanatron-3.3.0-py3-none-any.whl
sudo systemctl restart catanatron
# Tests
pytest tests/bridge/test_bridge_site_rules.py tests/test_catanatron_site_rules.py \
  third_party/catanatron/tests/test_balanced_dice.py third_party/catanatron/tests/test_friendly_public_robber.py
```)

== OpenHex versions

=== OpenHex v2: the friendly-robber net #chip("running") <exp:deploy-openhex-v2>
#meta(
  dates: "2026-10-27 (released; catanrl-1v1-v2 live on yang; the search bot not yet registered)",
  owner: "marco",
  config: "deploy/openhex.yaml at tag openhex-1v1-v2",
  wandb: (("ppo-2p-fr3-full-e95ea1db", "ppo-2p-fr3/full"),),
  records: ("cbba2d10", "b38e49dc", "507970d7", "7c0d2f67", "32d82b76"),
)
#tldr[Two new OpenHex bots on `ppo-2p-fr3/full@e4350`, the net continued on OpenHex's own 1v1 rules: `catanrl-1v1-v2` (`BELIEF`, 16 worlds) is registered and playing, and `catanrl-1v1-search-v2` (MCTS with 2#super[16] sims, Marco's pick) is not registered yet. The checkpoints are a public GitHub release, #link("https://github.com/mcrco/catanrl-v2/releases/tag/openhex-1v1-v2")[`openhex-1v1-v2`]. Marco runs the bots on yang, his other PC, with the three v1 bots alongside for a few days of comparison.]

*Status.* Marco registered `catanrl-1v1-v2` from yang (bot `9f7c562318324853`). It serves from tag `openhex-1v1-v2.1`: the release tag merged with yang's own work (GPU search, the fused graph kernels, `deploy/openhex.sh`). `catanrl-1v1-search-v2` is in `deploy/openhex.yaml` but not registered yet.

*Why this net.* Its policy beats F, beats v1's net and is level with the end of the first friendly-robber run (@exp:recipe-2p-openhex-3); `BELIEF` reads the same.

#figure(
  btable(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, left),
    header: ([Agent], [Opponent], [Score], [Record]),
    [`POLICY`], [F], [98.92 #ci[98.59][99.17]], [`507970d7`],
    [`POLICY`], [v1's net \ #sub[`puffer-graph2/full@e3350`]], [64.66 #ci[63.32][65.97]], [`32d82b76`],
    [`POLICY`], [End of the first friendly-robber run \ #sub[`ppo-2p-fr/full@e7600`]], [51.0 #ci[49.6][52.4]], [`7c0d2f67`],
    [`BELIEF`, 16 worlds], [F], [98.94 #ci[98.62][99.19]], [`cbba2d10`],
    [`BELIEF`, 16 worlds], [`BELIEF` over e7600], [51.18 #ci[49.79][52.56]], [`b38e49dc`],
  ),
  caption: [`ppo-2p-fr3/full@e4350` on `heldout-2p-fr` (OpenHex's 1v1 rules), 5,000 games a row.],
)

*Search budget.* 2#super[16] is 64 times the 1,024 sims v1's search bot was first deployed with. On this box's CPU, with training running alongside, 2,048 sims took a median 2.8 s a move (3.9 s at most, over 25 moves), which puts 2#super[16] near 90 s a move against OpenHex's 60 s limit. So `tools/move_time.py` times it on yang before the bot is registered. `deploy/openhex.yaml` notes that v1's search bot has since moved to 2#super[16] sims on yang's GTX 1080 at about 4 s a move, with 64 leaves per forward pass and the fused kernels; the v2 spec sets neither. The search's strength at this budget has not been measured.

*Versions.* OpenHex cannot rename or edit a bot, so each model is its own registration.
- `deploy/openhex.yaml` keeps every version, each spec pinned by its weights' sha, and every game line records its version and commit (`deploy/openhex-handoff.md`, "Versioning").
- The release holds one 2 MB archive per checkpoint (three: the v2 net and v1's two), checked by sha256 when fetched.

#repro(```sh
# Here: build, tag, publish (done 2026-10-27)
catanrl openhex release deploy/openhex.yaml openhex-1v1-v2
# On yang
git fetch --tags && git checkout openhex-1v1-v2 && git submodule update --init && uv sync
catanrl openhex fetch deploy/openhex.yaml
python tools/move_time.py "MCTS:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:sims=65536:dets=1" --moves 20
catanrl openhex deploy/openhex.yaml --check
```)

== Search in the deployed bots

=== Search reuse on the server bridge and the OpenHex seat #chip("done") <exp:deploy-reuse-bridge>
#meta(
  dates: "2026-10-20",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Search reuse is merged into the server bridge: a deployed seat's search can survive its own moves. It is opt-in per spec and neutral in strength (@exp:search-reuse-dedup), so the deployed bots can take it or leave it.]

*Question.* Can a deployed seat keep its search between its own moves? This is the deployment half of Marco's ask for tree reuse and leaf deduplication.

*Setup.* The server bridge keeps an `MCTS` seat's search between its decisions when the spec sets `reuse=true`. One fix came with the OpenHex seat's wiring: its robber move after its own discards re-rooted a tree dealt before the discards were known (three or more players, normal robber). It now searches fresh there.

*Result.* Merged. Reuse is opt-in per spec; neither deploy file set it when this merged. Since then `deploy/openhex.yaml` turns it on, with transpositions, for v1's search bot (`catanrl-1v1-search`, @tab:deploy-bots). On the CPU, the transposition table adds 0.2–0.5 s to a fresh 1,024-sim search.

*Reading.* The deployed bots can turn reuse on at no measurable cost in strength, but at 1,024 sims it buys none either (@exp:search-reuse-dedup).

#repro(```sh
# No eval of its own: the strength rows are in the search-options block. To serve reuse, append
# :reuse=true to an MCTS spec in deploy/openhex.yaml, e.g.
#   MCTS:ckpt=puffer-graph2/full@best:sims=1024:dets=1:reuse=true
catanrl openhex deploy/openhex.yaml
```)
