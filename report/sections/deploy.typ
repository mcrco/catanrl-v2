#import "../macros.typ": *

= Deployment <sec:deploy>

The bots that play people are in @tab:deploy-bots. To serve a 1v1 net raw, use the fair `public` net (`puffer-graph2/public@e3600`): it matches the belief-averaged bot with one forward pass and no tracker (@sec:recipe).

#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([Table or bot], [Agent spec]),
    tgroup(2)[catan.marcoya.ng (`deploy/house.yaml`)],
    ["1v1 OpenHex rules vs ppo-2p-fr3" \ #sub[waits on the house machine (@exp:deploy-site-openhex-rules)]], [`BELIEF:ckpt=ppo-2p-fr3/full@e4350:worlds=16`],
    ["1v1 vs puffer-graph2" \ #sub[served until the swap]], [`BELIEF:ckpt=puffer-graph2/full@best:worlds=16`],
    ["4p vs ppo-4p"], [`POLICY:ckpt=ppo-4p/main@best`],
    ["4p island vs ppo-mixed" \ #sub[15–30 tile islands]], [`POLICY:ckpt=ppo-mixed/main@best`],
    tgroup(2)[OpenHex (`deploy/openhex.yaml`); the first three are v1; `catanrl-1v1-search` and `catanrl-1v1-v2` were retired on 2026-11-04],
    [`catanrl-1v1`], [`BELIEF:ckpt=puffer-graph2/full@e3350:worlds=16`],
    [`catanrl-4p`], [`POLICY:ckpt=ppo-4p/main@e3650`],
    [`catanrl-1v1-search`], [`MCTS:ckpt=puffer-graph2/full@e3350:sims=65536:dets=1:` \ `pending=64:reuse=true:transpositions=true:kernels=fused`],
    [`catanrl-1v1-v2`], [`BELIEF:ckpt=ppo-2p-fr3/full@e4350:worlds=16`],
    [`catanrl-1v1-search-v2` \ #sub[live since 2026-11-04]], [`MCTS:ckpt=ppo-2p-fr3/full@e4350:sims=131072:dets=1:` \ `pending=64:reuse=true:transpositions=true:kernels=fused:deadline=45`],
    [`catanrl-1v1-search-v3` \ #sub[released 2026-11-13; not registered yet (@exp:deploy-openhex-search-v3)]], [`MCTS:ckpt=az-2p-kata/main@e300:` the same search],
  ),
  caption: [The deploy files in this checkout, sha pins omitted. `@best` is e3350 for `puffer-graph2/full`, e3650 for `ppo-4p/main` and e3550 for `ppo-mixed/main`.],
) <tab:deploy-bots>

== Tables on catan.marcoya.ng

=== The 4p island table #chip("done") <exp:deploy-island-table>
#meta(
  dates: "2026-10-22 18:59 UTC (live)",
  owner: "marco",
  config: "deploy/house.yaml",
  wandb: (("ppo-mixed-main-f4b771c0", "ppo-mixed"), ("ppo-mixed-blob-main-14c55ba5", "ppo-mixed-blob")),
  records: ("7c17d0ae", "75cab037"),
)
#tldr[The site's third table seats the generalist `ppo-mixed@best` on a fresh island per room; it beat the island-trained candidate 73.9 to 67.4 against three F. Marco's two recorded games verified with no fallbacks, and the bots won both.]

*Question.* Which network should play random island maps?

*Setup.* `ppo-mixed` never trained on islands; `ppo-mixed-blob` did, and failed its deploy gate (@sec:generalist). Pre-launch loopback smoke on the bot VM: two island games, all verified, 11 ms a decision.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([Candidate], [Share vs three F], [Record]),
    [`ppo-mixed`, zero-shot on islands \ #sub[`ppo-mixed/main@e3550`]], [*73.9* #ci[72.5][75.3]], [`7c17d0ae`],
    [`ppo-mixed-blob`, trained on islands \ #sub[`ppo-mixed-blob/main@e7600`]], [67.4 #ci[65.9][68.8]], [`75cab037`],
  ),
  caption: [Slot 0's score share on `heldout-random-4p`, 4,000 games each.],
) <tab:deploy-island>

*Reading.* `ppo-mixed` holds the table until one of the blob run's three changes is isolated (@sec:generalist).

#repro(```sh
catanrl eval "POLICY:ckpt=ppo-mixed/main@e3550" F F F --suite heldout-random-4p
catanrl eval "POLICY:ckpt=ppo-mixed-blob/main@e7600" F F F --suite heldout-random-4p
catanrl house deploy/house.yaml   # On the bot VM: keeps the three tables open
```)

=== The friendly-robber net on the site's rules #chip("done") <exp:deploy-site-rules-2p>
#meta(
  dates: "2026-10-27 03:41–04:20 UTC",
  owner: "marco",
  config: "configs/experiments/site-rules-2p.yaml",
  wandb: (("site-rules-2p-main-f1663977", "site-rules-2p"),),
  records: ("b692e3b3", "fc9ab6e1"),
)
#tldr[On the site's rules, `ppo-2p-fr3/full@e4350` beats the site's 1v1 bot 55.8% head to head, and F 96.52% to that bot's 95.16% (paired +1.36 points, p = 0.0002). It is the stronger 1v1 bot on both rule sets.]

*Question.* Does fr3 beat the site's bot on the site's rules?

*Setup.* The server then lacked the friendly robber and balanced dice, so the room plays `base-2p` (discard above 9), which fr3's last run did not train on. `POLICY` on both sides; fr3's row vs F pairs game for game with `ff8528a4` (same schedule, different code ids).

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
  caption: [`heldout-2p`, 5,000 games a row.],
) <tab:deploy-site-rules-2p>

Paired against F, fr3 wins 194 games `puffer-graph2` loses and loses 126 it wins. Games last 75.2 turns against F, 79.0 head to head.

*Reading.* fr3 can take the table now, and later play it on the rules it trained on (@exp:deploy-site-openhex-rules).

#repro(```sh
catanrl run configs/experiments/site-rules-2p.yaml
python tools/search_probe.py --pair runs/ppo-2p-fr3/full/evals/heldout-2p/b692e3b3c4738378.json \
  runs/puffer-graph2/full/evals/heldout-2p/ff8528a4939c421a.json
```)

=== OpenHex's 1v1 rules on the site's server #chip("running") <exp:deploy-site-openhex-rules>
#meta(
  dates: "2026-10-27 11:52 UTC (server live; the 1v1 table waits on the house machine)",
  owner: "marco",
  config: "deploy/house.yaml (draft: table ppo-2p-fr3)",
  wandb: (),
  records: (),
)
#tldr[The site's server now offers OpenHex's 1v1 rules as two room options, matching cppanatron exactly; old rooms and transcripts are unchanged. The 1v1 table moves to fr3 on these rules once the house machine runs the new Catanatron.]

*Question.* Can the site's server play OpenHex's 1v1 rules?

*Setup.* The server refused Catanatron's own friendly robber, which counts hidden VP cards (a legal robber move would reveal them), and had no balanced dice. Catanatron `da239a6` (branch `v2`) adds two room keys, omitted at their defaults so old configs keep their canonical form. The bridge reads both, so a house seat's mirror follows them.
- `robber_mode: friendly_public`: tiles touching an opponent at 2 public points or fewer are off limits unless all are; a knight is judged before its largest army award, as on OpenHex. `friendly_robber` is still refused.
- `dice_mode: balanced`: OpenHex's 36-card deck with recency and seven balancing, on cppanatron's integer weights. The dice state is public; the verifier replays rolls from the revealed seeds.

#lead[Result.]
- *Parity.* 60 `base-2p-fr` games in lockstep with cppanatron: equal legal actions at every step (about 30,000 rolls, 5,500 robber prompts) and equal dice state.
- *Checks.* Old games keep their document digests, and an old-code transcript verifies. 222 Catanatron tests and catanrl's fast suite (873) pass. A local smoke of the draft table (`BELIEF` on fr3 vs F): 4 games, all 8 transcripts verify, no mirror desync.
- *Deploy.* GCP server, 11:52 UTC, no game in progress. The server key and app.js are unchanged; the three house tables reconnected.

*Loose ends.* Unchanged, since each would change a code id (Marco's call): catanrl's harness builds Catanatron's hidden-VP robber for friendly-robber configs, and `CatanrlPlayer`'s mirror rolls fair dice. No record is affected: the harness refuses balanced dice, which every friendly-robber config uses.

*Next.* On the house machine: Catanatron `da239a6` (the client verifies transcripts with its own Catanatron; an old one refuses the new keys), the three bridge files and the fr3 checkpoint, then the fr3 table in `deploy/house.yaml`.

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
#tldr[`ppo-2p-fr3/full@e4350` is released on GitHub as #link("https://github.com/mcrco/catanrl-v2/releases/tag/openhex-1v1-v2")[`openhex-1v1-v2`] for two OpenHex bots: `catanrl-1v1-v2` (`BELIEF`, 16 worlds; bot `9f7c562318324853`, tag `openhex-1v1-v2.1`) is live, and `catanrl-1v1-search-v2` (MCTS, 2#super[16] sims, Marco's pick) is not registered yet. Marco runs them on yang, his other PC, beside the three v1 bots for comparison.]

#lead[Why this net.]
#figure(
  btable(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, left),
    header: ([Agent], [Opponent], [Score], [Record]),
    [`POLICY`], [F], [98.92 #ci[98.59][99.17]], [`507970d7`],
    [`POLICY`], [v1's net \ #sub[`puffer-graph2/full@e3350`]], [64.66 #ci[63.32][65.97]], [`32d82b76`],
    [`POLICY`], [End of the first friendly-robber run (@exp:recipe-2p-openhex-3) \ #sub[`ppo-2p-fr/full@e7600`]], [51.0 #ci[49.6][52.4]], [`7c0d2f67`],
    [`BELIEF`], [F], [98.94 #ci[98.62][99.19]], [`cbba2d10`],
    [`BELIEF`], [`BELIEF` over e7600], [51.18 #ci[49.79][52.56]], [`b38e49dc`],
  ),
  caption: [`ppo-2p-fr3/full@e4350` on `heldout-2p-fr`, 5,000 games a row.],
)

*Search budget.* On this box's CPU beside training, 2,048 sims took a median 2.8 s a move (max 3.9 s over 25 moves), so 2#super[16] would take about 90 s against OpenHex's 60 s limit; it is timed on yang before registration. v1's search bot already runs 2#super[16] sims on yang's GTX 1080 at about 4 s a move with `pending=64` and the fused kernels; the v2 spec sets neither. Strength at this budget is unmeasured.

*Versions.* OpenHex cannot rename or edit a bot, so each model is a new registration. `deploy/openhex.yaml` keeps every version, sha-pinned; game lines record version and commit (`deploy/openhex-handoff.md`). The release holds one sha256-checked 2 MB archive per checkpoint (v2's and v1's two).

#repro(```sh
# Here: build, tag, publish (done 2026-10-27)
catanrl openhex release deploy/openhex.yaml openhex-1v1-v2
# On yang
git fetch --tags && git checkout openhex-1v1-v2 && git submodule update --init && uv sync
catanrl openhex fetch deploy/openhex.yaml
python tools/move_time.py "MCTS:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:sims=65536:dets=1" --moves 20
catanrl openhex deploy/openhex.yaml --check
```)

=== OpenHex v3 candidate: `graph-m` against fr3 #chip("done") <exp:deploy-graph-m-check>
#meta(
  dates: "2026-11-10 14:14 – 23:59",
  owner: "marco",
  config: "configs/experiments/deploy-check-2p-graph-m.yaml",
  wandb: (("deploy-check-2p-graph-m-main-82cef592", "main"),),
  records: ("71af32b1", "9be79702"),
)
#tldr[`2p-graph-m-1` beats fr3 56.5% with both raw nets reading every hand (@exp:arch-graph-m-2p-lr). Before it could replace fr3 on OpenHex: the same comparison with the fair raw bot, and with search at equal time per move, since `graph-m` costs more per simulation. Approved by Marco on 10 November. Both rows pass: 55.0% as the fair raw bot and 54.3% inside the search at 300 sims against fr3's 1,024. By the rule fixed before launch `graph-m` is a deploy candidate; deploying waits for Marco's go.]

*Question.* Does `graph-m` still beat fr3 as a fair bot, and inside the deployed search when both get the same time per move?

#lead[Setup.]
- *Fair raw bot.* `BELIEF:worlds=16` of `graph-m-2p-lr/main@e22711` against the same of fr3 (`catanrl-1v1-v2`'s spec), 5,000 games on `heldout-2p-fr` (±1.4 pp at 50%).
- *Search.* `catanrl-1v1-search-v2`'s options (`dets=1:pending=64:reuse=true:transpositions=true`) for both nets, head to head, 2,000 games on `heldout-2p-fr`'s first 1,000 boards in both seat orders (±2.2 pp; 80% power at about 3.1 pp). fr3 searches 1,024 sims with the fused kernels, as deployed. The fused kernels are compiled for hidden 96 and refuse `graph-m` (hidden 192), so it searches with the torch forward. Its sims are 1,024 / _k_ = 300, where _k_ is its time per simulation over fr3's.
- *Calibration.* _k_ is measured at batch 1, the way the bridge plays: `tools/search_probe.py` with one env on the GPU, each net's search at 2#super[14] sims against fr3's raw policy, on the same boards. Time per simulation is the wall time over decisions × simulations per decision. It runs on this box's RTX 2070 SUPER; the deployed bots run on yang's GTX 1080. Measured: 352 µs a simulation for `graph-m` against 104 µs for fr3 (4 games each, 290 and 331 searched decisions), so _k_ = 3.38 (`report/data/deploy-graph-m-calibration-*.json`). At this cost `graph-m`'s 2#super[17]-sim search, the deployed budget, would take about 46 s a move here, past the 45 s deadline.
- *Decision rule* (fixed before launch). `graph-m` is a deploy candidate if its search row is above 50% at 95% and its fair raw row is not below 50% at 95%.
- *Cost.* The fair raw row about 30 min. The search row is unmeasured: the AlphaZero gate's 2,000-game rows at 512 sims with one search seat are estimated at 1.15 h, so roughly 4–5 h with two search seats at 1,024.

#lead[Result.]
- *Fair raw bot.* `graph-m` beats fr3 55.0% #ci[53.6][56.4] over 5,000 games (`71af32b1`), so this half of the rule passes. Reading every hand it scored 56.5% on the same games; the fair bot is 1.5 pp lower, paired (550 games won only as the fair bot, 626 only reading every hand; p = 0.03).
- *Search.* At equal time per move `graph-m`'s 300-sim search beats fr3's 1,024-sim search 54.3% #ci[52.2][56.5] over 2,000 games (1,087 wins to 913, p = 1e-4; `9be79702`), so this half passes too. By seat: 58.0% moving first, 50.7% second.
- *Rule.* Both halves pass: `graph-m` is a deploy candidate.
- *Not covered.* The time match was measured on this box's GPU at 2#super[14] sims and played at about 1,000; the deployed bots search 2#super[17] sims on yang's GTX 1080, where `graph-m` would need a smaller budget or fused kernels for hidden 192 to stay under the 45 s deadline.
- *Cost.* 9.1 h for the search row, about twice the estimate. The first launch ran 960 games at a time: two search seats with reuse and transpositions kept about 11 MB of trees per seat, the run reached 21 GB of host RAM, and the kernel's out-of-memory killer stopped it at 14:51, just after the raw row finished (nothing else was killed). The relaunch at 14:54 ran 256 games at a time and held 8.3 GB.

#repro(```sh
# Calibration (batch 1), then the two rows
O=sims=16384:dets=1:pending=64:reuse=true:transpositions=true
uv run python tools/search_probe.py "MCTS:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:$O:kernels=fused" "POLICY:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41" --suite heldout-2p-fr --boards 2 --device cuda --num-envs 1
uv run python tools/search_probe.py "MCTS:ckpt=graph-m-2p-lr/main@e22711#673425d606b5:$O" "POLICY:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41" --suite heldout-2p-fr --boards 2 --device cuda --num-envs 1
catanrl run configs/experiments/deploy-check-2p-graph-m.yaml
```)

=== OpenHex search-v3: the AlphaZero-trained net #chip("running") <exp:deploy-openhex-search-v3>
#meta(
  dates: "2026-11-13 (released; not registered yet)",
  owner: "marco",
  config: "deploy/openhex.yaml at tag openhex-1v1-v3",
  wandb: (("az-2p-kata-main-2e3b1866", "az-2p-kata/main"),),
  records: ("e917f9bb", "809619df", "b7f9c1f4"),
)
#tldr[`az-2p-kata/main@e300` (`2p-graph-s-5`) is released on GitHub as #link("https://github.com/mcrco/catanrl-v2/releases/tag/openhex-1v1-v3")[`openhex-1v1-v3`] for a new bot, `catanrl-1v1-search-v3` (Marco's name): `catanrl-1v1-search-v2`'s search with the net AlphaZero trained. Marco registers and runs it on yang, then looks at a day of its games. The bot it replaces has won 29 of its last 101 games, so the ladder has moved more than this net's 4 pp.]

*Question.* Does the trained net look better than fr3 on OpenHex's ladder, at the deployed 2#super[17] sims?

- *Why this net.* In the 1,024-sim search against fr3's fair raw bot it scores 57.70% on 2,000 held-out games, where fr3's own net scores 53.85% (paired +3.85 pp, p = 0.003; @exp:az-kata). Not measured: 2#super[17] sims, and the two searches head to head.
- *Spec.* search-v2's with the checkpoint swapped: `MCTS:ckpt=az-2p-kata/main@e300#513d0caac817:sims=131072:dets=1:pending=64:reuse=true:transpositions=true:kernels=fused:deadline=45`.
- *Release.* Tag `openhex-1v1-v3` at `46cf640` carries every running entry (one checkout runs them all). Since `openhex-1v1-v2.2` the catanatron and PufferLib submodules moved, so yang needs `git submodule update` and `uv sync`; the engine did not move. A fetch of the published release into an empty directory resolved all four checkpoints.
- *Overlap.* search-v2 stays on the ladder beside it (Claude's choice, the handoff's overlap step; `retired: true` takes it off). They share yang's GPU and will often be matched with each other, which is the head-to-head at 2#super[17] sims.

#lead[The baseline: search-v2's record] (OpenHex's public data, 2026-11-13 09:21 UTC).
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Games created (UTC)], [Wins], [Games], [Against]),
    [4–10 November], [181], [281], [`hexn-4d.0` 33 of 75, `katana-1v1` 52 of 54, `Breeze` 24 of 42],
    [11 November], [31], [80], [`hexn-4d.0` only],
    [12 November], [20], [72], [`hexn-4d.0` 16 of 54, `Gaz` 4 of 18],
    [13 November, to 09:21], [9], [29], [`oversized-pig` 5 of 15, `hexn-4d.0` 1 of 6, `Pattern` 1 of 4, `puma-ppo-864m` 2 of 4],
  ),
  caption: [`catanrl-1v1-search-v2` since its deploy: 241 of 462 (52.2%), Elo 1445. 45 of its 47 aborted games are from 4–10 November.],
)
- Since 11 November it has won 60 of 181 (33%), against opponents that are new or have changed (`hexn-4d.0` went from 33 of 75 to 48 of 140). Whether yang's searches are also being cut short cannot be seen from here: the `cut` counts in yang's `games.jsonl` would say.
- So a day of search-v3's games is compared with search-v2's games of the same days and opponents, not with its 4–10 November record.

#repro(```sh
# Here: build, tag, publish (done 2026-11-13)
catanrl openhex release deploy/openhex.yaml openhex-1v1-v3
# On yang
git fetch --tags && git checkout openhex-1v1-v3 && git submodule update --init && uv sync
catanrl openhex fetch deploy/openhex.yaml
catanrl openhex deploy/openhex.yaml --check --device cuda
catanrl openhex deploy/openhex.yaml --describe catanrl-1v1-search-v3 > /tmp/body.json   # then register (handoff, step 7)
# Anywhere: both bots' records by opponent over the same days
catanrl openhex stats deploy/openhex.yaml --since 2026-11-13
```)

== Search in the deployed bots

=== Search reuse on the server bridge and the OpenHex seat #chip("done") <exp:deploy-reuse-bridge>
#meta(
  dates: "2026-10-20",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Merged: with `reuse=true`, a deployed `MCTS` seat keeps its search across its own moves. It is neutral in strength at 1,024 sims (@exp:search-reuse-dedup).]

*Question.* Can a deployed seat keep its search between its own moves?

*Setup.* A fix came with it: after its own discards, the OpenHex seat's robber move re-rooted a tree dealt before them (three or more players, normal robber); it now searches fresh.

*Result.* `deploy/openhex.yaml` now sets reuse, with transpositions, for `catanrl-1v1-search` (@tab:deploy-bots). On the CPU, transpositions add 0.2–0.5 s to a fresh 1,024-sim search.

*Reading.* The deployed bots can take it or leave it.

#repro(```sh
# No eval of its own (strength rows: search-options block). To serve reuse, append
# :reuse=true to an MCTS spec in deploy/openhex.yaml, e.g.
#   MCTS:ckpt=puffer-graph2/full@best:sims=1024:dets=1:reuse=true
catanrl openhex deploy/openhex.yaml
```)
