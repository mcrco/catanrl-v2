#import "../macros.typ": *

= deployment <sec:deploy>

this section records which bots play people, on catan.marcoya.ng and on openhex, and why each was picked. the 1v1 bot is `BELIEF` on `puffer-graph2/full@e3350` with 16 worlds; the fair `public` network (`puffer-graph2/public@e3600`) plays as well with one forward pass, no sampled worlds and no tracker, so it is the bot to serve raw (@sec:recipe). a third table seats the generalist `ppo-mixed` on random islands. search reuse is available to the deployed bots but neutral in strength.

#figure(
  btable(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    header: ([where], [table or bot], [agent spec]),
    [catan.marcoya.ng], ["1v1 vs puffer-graph2"], [`BELIEF:ckpt=puffer-graph2/full@best:worlds=16`],
    [catan.marcoya.ng], ["4p vs ppo-4p"], [`POLICY:ckpt=ppo-4p/main@best`],
    [catan.marcoya.ng], ["4p island vs ppo-mixed" (15–30 tile islands)], [`POLICY:ckpt=ppo-mixed/main@best`],
    mid,
    [openhex], [1v1], [`BELIEF:ckpt=puffer-graph2/full@best:worlds=16`],
    [openhex], [4p], [`POLICY:ckpt=ppo-4p/main@best`],
    [openhex], [1v1, search], [`MCTS:ckpt=puffer-graph2/full@best:sims=1024:dets=1`],
  ),
  caption: [what is deployed, as `deploy/house.yaml` and `deploy/openhex.yaml` list it in this checkout. `@best` is e3350 for `puffer-graph2/full`, e3650 for `ppo-4p/main` and e3550 for `ppo-mixed/main`.],
) <tab:deploy-bots>

== tables on catan.marcoya.ng

=== the 4p island table #chip("done") <exp:deploy-island-table>
#meta(
  dates: "2026-10-22 18:59 utc (live)",
  owner: "marco",
  config: "deploy/house.yaml",
  wandb: (("ppo-mixed-main-f4b771c0", "ppo-mixed"), ("ppo-mixed-blob-main-14c55ba5", "ppo-mixed-blob")),
  records: ("7c17d0ae", "75cab037"),
)
#tldr[catan.marcoya.ng has a third table, "4p island vs ppo-mixed": the generalist `ppo-mixed@best` on a fresh 15–30 tile island in every room. it beat the blob-trained candidate on random islands, 73.9 against 67.4 versus F. marco played two recorded games that evening: both verified, no fallbacks, and the bots won both.]

*question.* which network should play a table of random island maps?

*setup.* two candidates on `heldout-random-4p` (blob-shaped boards of 15–30 tiles) against three F. `ppo-mixed` never trained on islands; `ppo-mixed-blob` did, and its deploy gate failed (@sec:generalist). the island table runs next to the two existing tables (@tab:deploy-bots). before going live, a preflight and a loopback smoke ran on the bot vm's own stack: two island games, all verified, 11 ms a decision.

*result.*
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([candidate], [vs three F (share)], [record]),
    [`ppo-mixed/main@e3550`, zero-shot on islands], [*73.9* #ci[72.5][75.3]], [`7c17d0ae`],
    [`ppo-mixed-blob/main@e7600`, trained on islands], [67.4 #ci[65.9][68.8]], [`75cab037`],
  ),
  caption: [the two island candidates on `heldout-random-4p`, 4,000 games each, parity 25.],
) <tab:deploy-island>

marco's two recorded games on the live table were both verified, with no fallbacks; the bots won both.

*reading.* `ppo-mixed` is the stronger island bot until one of the blob run's three changes is isolated (@sec:generalist).

#repro(```sh
catanrl eval "POLICY:ckpt=ppo-mixed/main@e3550" F F F --suite heldout-random-4p
catanrl eval "POLICY:ckpt=ppo-mixed-blob/main@e7600" F F F --suite heldout-random-4p
catanrl house deploy/house.yaml   # on the bot vm: keeps the three tables open
```)

== openhex versions

=== openhex v2: the friendly-robber net #chip("running") <exp:deploy-openhex-v2>
#meta(
  dates: "2026-10-27 (released; registration on yang pending)",
  owner: "marco",
  config: "deploy/openhex.yaml at tag openhex-1v1-v2",
  wandb: (("ppo-2p-fr3-full-e95ea1db", "ppo-2p-fr3/full"),),
  records: ("cbba2d10", "b38e49dc", "507970d7", "7c0d2f67", "32d82b76"),
)
#tldr[two new openhex registrations on `ppo-2p-fr3/full@e4350`, the net continued on openhex's own 1v1 rules: `catanrl-1v1-v2` (BELIEF, 16 worlds) and `catanrl-1v1-search-v2` (MCTS with 2#super[16] sims, marco's pick). the checkpoints are a public github release, #link("https://github.com/mcrco/catanrl-v2/releases/tag/openhex-1v1-v2")[`openhex-1v1-v2`]. marco runs the bots on yang, his other pc, with the three v1 bots alongside for a few days of comparison.]

*why this net.* on `heldout-2p-fr` (5,000 games each) its policy beats F 98.92% and v1's net 64.66%, and is level with the end of the first friendly-robber run: 51.0 #ci[49.6][52.4] against `ppo-2p-fr/full@e7600` (@exp:recipe-2p-openhex-3). BELIEF reads the same: 98.94% vs F, 51.18% vs e7600.

*search budget.* 2#super[16] is 64 times v1's 1,024 sims. on this box's cpu, with training running alongside, 2,048 sims took a median 2.8 s a move (3.9 s at most, over 25 moves), which puts 2#super[16] near 90 s a move against openhex's 60 s limit. so `tools/move_time.py` times it on yang before the bot is registered. the search's strength at this budget has not been measured.

*versions.* each model is its own registration, since openhex cannot rename or edit a bot; `deploy/openhex.yaml` keeps every version, each spec pinned by its weights' sha, and every game line records its version and commit (`deploy/openhex-handoff.md`, "versioning"). the release holds one 2 MB archive per checkpoint (three: the v2 net and v1's two), checked by sha256 when fetched.

#repro(```sh
# here: build, tag, publish (done 2026-10-27)
catanrl openhex release deploy/openhex.yaml openhex-1v1-v2
# on yang
git fetch --tags && git checkout openhex-1v1-v2 && git submodule update --init && uv sync
catanrl openhex fetch deploy/openhex.yaml
python tools/move_time.py "MCTS:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:sims=65536:dets=1" --moves 20
catanrl openhex deploy/openhex.yaml --check
```)

== search in the deployed bots

=== search reuse on the server bridge and the openhex seat #chip("done") <exp:deploy-reuse-bridge>
#meta(
  dates: "2026-10-20",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[search reuse is merged into the server bridge: a deployed seat's search can survive its own moves. it is opt-in, and neither deploy file turns it on. the options are neutral in strength (@exp:search-reuse-dedup), so the deployed bots can take them or leave them.]

*question.* the deployment half of marco's ask for tree reuse and leaf deduplication: can a deployed seat keep its search between its own moves?

*setup.* the server bridge keeps an `MCTS` seat's search between its decisions when the spec sets `reuse=true`. one fix came with the openhex seat's wiring: its robber move after its own discards re-rooted a tree dealt before the discards were known (three or more players, normal robber). it now searches fresh there.

*result.* merged. reuse is opt-in per spec, and neither `deploy/house.yaml` nor `deploy/openhex.yaml` sets it. on the cpu the transposition table adds 0.2–0.5 s to a fresh 1024-sim search.

*reading.* the deployed bots can turn reuse on at no measurable cost in strength, but at 1024 sims it buys none either (@exp:search-reuse-dedup).

#repro(```sh
# no eval of its own: the strength rows are in the search-options block. to serve reuse, append
# :reuse=true to an MCTS spec in deploy/openhex.yaml, e.g.
#   MCTS:ckpt=puffer-graph2/full@best:sims=1024:dets=1:reuse=true
catanrl openhex deploy/openhex.yaml
```)
