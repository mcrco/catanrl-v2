#import "../macros.typ": *

= Trading <sec:trading>

Can a bot gain from domestic trades, and are they worth an action space? A critic-only oracle finds few trades that help both sides: 2.5–6.2% of 4p positions at ≥ 2 pp, almost none in 2p. Yet a harness that proposes every trade the critic calls good for both sides takes 45.5% of 4p games against a parity of 25: +20.5 pp paired, the largest effect in this log. The edge is in proposing, not accepting, and it holds with the strongest 4p policy. No policy trades on its own yet: trades in the action space and a learned trade head are built and queued (@exp:trade-engine, @exp:trade-learned).

== Is trading worth an action space?

=== Trades in the action space and env #chip("queued") <exp:trade-engine>
#meta(dates: "2026-11-02", dates-label: "built", owner: "marco", wandb: (), records: ())
#tldr[Built and off by default. Domestic trades are now actions (offer, accept, reject, cancel, confirm with a seat), the env offers them up to 3 times a turn, and the row carries the offer, who made it and who accepted. With trades off, every stored record replays bit for bit. The code is on the integration branch; it goes to main and the GPU after the search-depth sweep.]

#lead[What changed.]
- *Action space.* `GraphActionSpace` gains a trade block before the no-op, behind `env.trades: {offers: N}`. N = 0, the default, is today's game.
- *Legality.* An offer is legal on the mover's own turn after the roll, with a card in hand and fewer than N offers made this turn.
- *Row.* A new `trade` block: what the offer gives and asks per resource, the proposer, who accepted, offers left. The rest of the row is unchanged bit for bit.
- *Seats without a trade head* (F, older checkpoints, MCTS, `BELIEF`) reject every offer; R still accepts at random.
- *Check.* 80 games from four stored 4p records (`e5b302ee`, `c4900f87`, `7164d396`, `b07e5bd9`) replay identically with trades off.

=== Trade oracle: which trades does the critic like for both sides? #chip("done") <exp:trade-oracle>
#meta(
  dates: "2026-10-17",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Marco's first step: before any action-space work, ask the trained critic which trades help both sides. A trade worth ≥ 2 pp to both exists in 6.2% of 4p positions and 0.3% of 2p positions: worth an action space in 4p, not in 2p.]

*Question.* Under the critics we already have, how often does a trade leave both parties better off, by how much, and what does it look like? Marco suspects that a policy trained with trades from scratch won't learn to trade well, so this comes first.

*Setup.* The oracle is `tools/trade_oracle.py`.
- *Positions.* 3,000 per network from its own self-play (temperature 1), taken after the roll; about 3.4 positions per game, from 884 games in 4p and 916 in 2p.
- *Trades.* Every 1:1, 2:1 and 1:2 trade between the mover and each opponent that both hands can pay (about 73 per 4p position), applied to a clone through the engine's own offer, accept and confirm actions.
- *Values.* Both seats' win probabilities are read from the mover's row: the critic trains on acting rows only, and a partner's own row is off-distribution.
- *Terms.* A seat's _gain_ is the change in its win probability, in pp. A trade is _mutual_ at margin m if both parties gain at least m.
- *Noise floor.* The critic's per-seat win probabilities don't conserve under a trade: the median residual is 1.1 pp in 4p and 0.5 pp in 2p. The 0.5 and 1 pp margins are within noise; the 2 pp margin is the one to read.
- *Data.* No eval record and no W&B mirror; the numbers are in `report/data/trade-oracle-ppo-4p-main.json` and `report/data/trade-oracle-puffer-graph2-full.json`.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    header: ([Critic], [Level], [≥ 1 pp], [≥ 2 pp], [≥ 5 pp], [Median residual]),
    [4p \ #sub[`ppo-4p/main@best` (e3650)]], [`private`], [25.8 #ci[24.2][27.5]], [*6.2* #ci[5.4][7.2]], [0.4], [1.1 pp],
    [2p \ #sub[`puffer-graph2/full@best` (e3350)]], [`full`], [4.3], [0.3 #ci[0.1][0.5]], [—], [0.5 pp],
  ),
  caption: [Share of positions (%) with a mutual trade at each margin, 3,000 positions per critic.],
)

In 4p the best mutual trade gains the mover 1.2 pp and the partner 0.9 pp on average, and costs the two seats outside it 1.1 pp. 92% of mutual trades are 2:1 (the mover gives two); the most common is two wood for an ore. At the `private` level the mover's row sees the partner's hand only as a size, so the 2:1 skew is partly what the critic can see. The mover's selfish best trade gains it 7.8 pp and costs the partner 3.7 pp on average: trades are mostly zero-sum, as expected.

*Reading.* 4p has a modest, real pool of 2:1 trades the critic likes for both sides, each paid for by the other two seats. Two-player Catan is nearly zero-sum for this critic, so a 2p trade action would almost never be used.

#repro(```sh
# The defaults are the saved arguments; each writes report/data/trade-oracle-<run>.json
uv run python tools/trade_oracle.py --run ppo-4p/main          # 4p
uv run python tools/trade_oracle.py --run puffer-graph2/full   # 2p
```)

=== The oracle on the calibrated lambda critic #chip("done") <exp:trade-oracle-lambda>
#meta(
  dates: "2026-10-19 20:15",
  wandb: (),
  records: (),
)
#tldr[A rerun on the first critic whose four win probabilities sum to 1.00. The noise floor barely moves, and the mutual pool shrinks to 2.5% of 4p positions at ≥ 2 pp: under this critic, trades are nearly zero-sum between the parties, and the mutual gains sit at its noise floor.]

*Question.* Does a calibrated critic give a cleaner oracle?

*Setup.* The same tool on `aux-hidden-4p/plain@e600`, the first net trained with lambda critic targets (@sec:recipe); its per-seat win probabilities sum to the realized 1.00. 3,000 positions from 846 self-play games, 229k trades, 3 minutes. Data in `report/data/trade-oracle-aux-hidden-4p-plain.json`; no W&B mirror.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([4p critic], [Residual, median], [Residual, p95], [≥ 1 pp], [≥ 2 pp], [≥ 5 pp]),
    [V-trace \ #sub[`ppo-4p/main@best`]], [1.1 pp], [4.8 pp], [25.8 #ci[24.2][27.5]], [6.2 #ci[5.4][7.2]], [0.4],
    [Lambda \ #sub[`aux-hidden-4p/plain@e600`]], [1.0 pp], [3.8 pp], [22.7 #ci[21.2][24.3]], [*2.5* #ci[2.0][3.2]], [0],
  ),
  caption: [The calibrated critic against the V-trace one: the critic's residual under a trade, and the share of positions (%) with a mutual trade at each margin.],
)

Some trade's residual still reaches 2 pp in 90% of positions. The V-trace critic's sum sat at 0.93 and moved more under a hand change. Under the lambda critic the best mutual trade gains the mover 0.8 pp and the partner 0.7 pp at the median, and the other two seats lose 1.1 pp on average. 94% of mutual trades are 2:1 (two sheep for a wheat, two wood for a wheat and two wood for a sheep lead the list). The mover's selfish best trade (+7.2 pp for it, −3.3 for the partner) is mutual in 0.1% of positions.

#lead[Reading.]
- If a trade's effect on the partner were unrelated to its effect on the mover, 77% of positions would show a mutual trade. Against that, 2.5% means trades are nearly zero-sum between the two parties, and the mutual pool sits at the critic's own noise floor.
- A trade head would find a trade worth ≥ 2 pp to both in about one position in forty.
- A harness that proposes by this critic proposes gains the size of its noise; resolving a one-point edge takes about 15,000 paired games. The harness below did far better than this predicts.

#repro(```sh
uv run python tools/trade_oracle.py --run aux-hidden-4p/plain
```)

== A critic-driven trade harness

=== Marco's trade harness: the pilot #chip("done") <exp:trade-harness-pilot>
#meta(
  dates: "2026-10-20 03:44–03:49",
  owner: "marco",
  config: "configs/experiments/trade-harness-pilot.yaml",
  wandb: (("trade-harness-pilot-main-8d9a5b49", "trade-harness-pilot, eval only"),),
  records: ("0fc4352d", "1d253765", "c34e389c", "09893ca0"),
)
#tldr[Marco's idea: keep trades out of the action space and let the critic propose and answer them. In a 200-game pilot the proposing seat took 46.5% against three copies that only answer offers (parity 25), and 55.5% when it re-evaluates after every build. Reading the true hands instead of sampled ones adds nothing.]

*Question.* Marco's proposal after the oracle rerun: does a seat win more if a critic-driven harness, outside the action space, proposes its best mutual trades, with responders judging offers the same way?

*Setup.* Marco's rules: a calibrated critic, gains in each seat's own view, a Pareto front for the proposer, positive expected value for the acceptor.
- *Proposer* (the `TRADE` agent). Plays the policy everywhere else. Once per own turn, after the roll, it enumerates every 1:1, 2:1 and 1:2 bundle it can pay and reads the critic once for all of them (about 84 rows a read). It keeps the bundles with a positive gain for itself and for the partner, drops the Pareto-dominated ones (those another bundle beats for both), and offers the rest in order of its own gain, up to three a turn. It confirms the acceptee best for itself.
- *Responders.* Accept iff their mean gain over eight determinized worlds (hidden hands sampled consistent with what they have seen) is positive.
- *Plumbing.* Trades go through one engine side channel (`VecEnv.set_action`); the row layout, the checkpoints and training are untouched. Fairness is black-box: the choices are identical on any game with the same public state (`tests/test_trade_fairness.py`).
- *Table.* Native F never accepts an offer: its one-ply value sees no cards move on an accept, and a strict comparison keeps reject. So every table is one network against itself, with `aux-hidden-4p/plain@e600` as policy and critic in every seat and the trading seat the only asymmetry.
- *Games.* Four rows of 200 games (50 `heldout-4p` boards × 4 rotations), five minutes in all. Parity is 25.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    header: ([Proposer], [Share], [Offers / game], [Trades / game], [Record]),
    [Marco's rule (0 pp margin)], [*46.5* #ci[39.7][53.4]], [14], [12], [`0fc4352d`],
    [Re-evaluates after every build], [*55.5* #ci[48.6][62.2]], [26], [22], [`1d253765`],
    [1 pp margin, responders too], [31.0 #ci[25.0][37.7]], [4], [—], [`c34e389c`],
    [Reads the true hands (unfair ceiling)], [46.0 #ci[39.2][52.9]], [—], [—], [`09893ca0`],
  ),
  caption: [Slot 0's share against three responder-only copies on `heldout-4p`, 200 games a row (parity 25).],
)

Under Marco's rule 85% of offers land, cancels are under 1%, and the three responders take 17–19. The bundles are the oracle's: mostly 1:1 and 2:1, with sheep for wood and ore for brick on top. So are the gains the critic claims for them (`tools/trade_pilot.py`): +1.35 pp for the mover per offer (median 1.2), +0.4 for the partner, +0.6 on an accepted read, −0.3 on a rejected one.

*Reading.* Sampling the hidden hands costs nothing: the true hands give the same share. The claimed gains sit at the critic's noise floor, yet a dozen small trades a game compound into twenty points of share, paid for by the seats outside each trade.

#repro(```sh
catanrl run configs/experiments/trade-harness-pilot.yaml   # needs the aux-hidden-4p/plain run
# The bundles offered and the gains the critic claimed for them:
uv run python tools/trade_pilot.py --config configs/experiments/trade-harness-pilot.yaml --row 0
```)

=== The harness at 4,000 games: m1 (0 pp margin) and m2 (1 pp margin) #chip("done") <exp:trade-harness-m1m2>
#meta(
  dates: "2026-10-20 03:50–04:24",
  owner: "marco",
  config: "configs/experiments/trade-harness.yaml",
  wandb: (("trade-harness-main-a9a5c206", "trade-harness, eval only"),),
  records: ("ff0c2cb2", "7ffdd60f", "d7790a27", "7c06e560"),
)
#tldr[At 4,000 games a row, Marco's rule takes 45.5% against a parity of 25: +20.5 pp paired against the same table with nobody proposing (p < 10#super[−90]), the largest effect in this log. A 1 pp margin cuts the gain to +6.7 pp.]

*Question.* Does the pilot's edge hold at a sample size that resolves it, and does a margin above the critic's noise help?

*Setup.* Same network and critic as the pilot; 34 minutes for the 16,000 games.
- *m1* is Marco's rule; *m2* puts the proposer and the responders at a 1 pp margin.
- *Control.* Each row has one: the same games with a plain `POLICY` in slot 0, so nobody proposes and slots 1–3 are identical in both.
- *Games.* 1,000 `heldout-4p` boards × 4 rotations = 4,000 games a row.
- *Test.* The paired delta is slot 0's share against its control, with McNemar's test on the games won only with trading or only without.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, left, right, left),
    header: ([Row], [Proposer], [Control], [Paired], [Offers, trades / game], [Records]),
    [m1 \ #sub[0 pp margin]], [*45.5* #ci[43.9][47.0]], [25.0 #ci[23.7][26.4]], [*+20.5 pp* \ #sub[1,204 / 386, p < 10#super[−90]]], [14.7, 12.4], [`ff0c2cb2` \ `7ffdd60f`],
    [m2 \ #sub[1 pp margin]], [31.7 #ci[30.3][33.2]], [25.0 #ci[23.7][26.4]], [+6.7 pp \ #sub[p < 10#super[−14]]], [3.8, 3.2], [`d7790a27` \ `7c06e560`],
  ),
  caption: [Slot 0's share with trading against its no-proposer control, 4,000 games a row, `heldout-4p`. Under the paired delta: games won only with trading / only without, and McNemar's p. Records: proposer, then control.],
)

In m1, 84% of offers land and the responders take 18.0, 18.3 and 18.3.

*Reading.* Under this critic the margin throws share away: the 0 pp rule trades four times as often as the 1 pp rule and wins three times as much over parity. So the critic's trade judgement carries signal well below its own per-read noise. The oracle was right about the per-trade gains and wrong about what a dozen of them a game add up to.

#repro(```sh
catanrl run configs/experiments/trade-harness.yaml   # needs the aux-hidden-4p/plain run
```)

=== Does accepting pay? (m3) #chip("done") <exp:trade-harness-m3>
#meta(
  dates: "2026-10-20 05:31–06:38",
  config: "configs/experiments/trade-harness-m3.yaml",
  wandb: (("trade-harness-m3-main-359acf90", "trade-harness-m3, eval only"),),
  records: ("fe7baed5", "58f7c074"),
)
#tldr[No. At a table of three proposers, a seat that accepts the offers it likes does no better than one that rejects everything (12.7 against 13.2, p = 0.54). The proposer captures each trade's surplus: the edge is in proposing.]

*Question.* Is the responder's side of a trade worth anything?

*Setup.* Three proposers at Marco's rule. Slot 0 is either the responder-only seat (accepts the offers its critic likes) or a plain `POLICY` that rejects every offer. 4,000 games a row, 8,000 in all, paired as in m1 (@exp:trade-harness-m1m2).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([Slot 0, against three proposers], [Share], [Record]),
    [Accepts the offers it likes], [12.7 #ci[11.7][13.8]], [`fe7baed5`],
    [Rejects every offer], [13.2 #ci[12.1][14.2]], [`58f7c074`],
  ),
  caption: [The acceptor against the rejecter, 4,000 games a row, `heldout-4p`.],
)

Paired −0.45 pp: 371 games won only by the acceptor, 389 only by the rejecter, McNemar p = 0.54. The three proposers take 27–31.

*Reading.* Accepting buys nothing. The proposer picks the trades, so it captures each one's surplus, and the acceptor's own read of a gain is noise around zero. The passive seat is crushed whether it trades or not (13 against a parity of 25).

#repro(```sh
catanrl run configs/experiments/trade-harness-m3.yaml   # needs the aux-hidden-4p/plain run
```)

=== The 4p specialist with trading (m4) #chip("done") <exp:trade-harness-m4>
#meta(
  dates: "2026-10-20 14:35–14:51",
  config: "configs/experiments/trade-harness-m4.yaml",
  wandb: (("trade-harness-m4-main-2bc4df25", "trade-harness-m4, eval only"),),
  records: ("c4900f87", "7164d396"),
)
#tldr[The edge holds with the strongest 4p policy. `ppo-4p/main@best` in every seat, proposing by the calibrated critic, takes 41.2% (+16.2 pp paired), four points less than the weaker plain policy got in m1.]

*Question.* Run because m1 was positive: does the trading edge survive a stronger player?

*Setup.* m1's tables with `ppo-4p/main@best` (e3650) in every seat. The calibrated critic of `aux-hidden-4p/plain@e600` judges the trades, because `ppo-4p`'s own V-trace critic does not sum to one. The control puts a never-trading `POLICY` in slot 0. 4,000 games a row.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Slot 0], [Share], [Offers, trades / game], [Record]),
    [Proposes (Marco's rule)], [*41.2* #ci[39.7][42.7]], [13.8, 11.6], [`c4900f87`],
    [Never trades (control)], [25.0 #ci[23.7][26.4]], [—], [`7164d396`],
  ),
  caption: [`ppo-4p/main@best` with trading against three responder-only copies of itself, 4,000 games a row, `heldout-4p`.],
)

Paired *+16.2 pp*: 1,080 games won only with trading against 433 only without, p < 10#super[−60].

*Reading.* The edge holds, four points smaller than in m1. Either a better player gets less from the same trades, or a critic from another network reads its positions slightly worse.

#repro(```sh
catanrl run configs/experiments/trade-harness-m4.yaml   # needs the ppo-4p/main and aux-hidden-4p/plain runs
```)

== Learned trading

=== A trade head trained from the PPO checkpoint #chip("queued") <exp:trade-learned>
#meta(dates: "2026-11-02", dates-label: "built", owner: "marco", config: "configs/experiments/trade-learned-4p.yaml", wandb: (), records: ())
#tldr[Step (2) of Marco's plan: option one (an offer action plus 10 floats rounded to integers, and a yes/no for the responder) on `ppo-4p/main@best`, with a function-preserving start. About 3.5 h on the GPU; queued after the search-depth sweep and waiting on Marco's answers below.]

*Plan.* Marco wants learned trading without an explosive action space. Option one is the default.
/ Option one: A trade action plus 10 floats, one per resource for what the mover gives and one for what it gets, each rounded to an integer (11 extra outputs).
/ Option two: One output per integer amount (18 choices) per resource per side (181 extra outputs).
A policy trained with trades from scratch probably won't learn to trade well, so training starts from a competent PPO policy, with the trade actions masked in at a low entropy bonus.

#lead[Built.]
- *Head.* A trade head on the shared trunk: step logits for offer, accept, reject, cancel and confirm, and ten Gaussian means with a learned spread for the bundle.
- *Floats to offer*, in a fixed order: negatives to 0, a resource on both sides netted, gives clipped to the hand, gets clipped to the bank's 19. An empty side means no offer (it still spends one of the turn's offers).
- *Log-prob.* The action's, plus, on an offer, the Gaussian density of the unrounded floats.
- *Checks.* With trades masked out, the extended `ppo-4p/main@best` gives the checkpoint's logits and values exactly (2,400 rows), and an update at learning rate 0 changes nothing (KL 0, clip fraction 0).

*One deviation, to approve.* A zero-initialized offer logit next to a trained policy's logits has a sampling probability around 4 × 10#super[−6] (1,334 post-roll rows), so training would never try a trade. The offer logit is instead a gate added to the log-sum-exp of the other legal actions: p(offer) starts at 0.1, and the other actions keep their shares.

#lead[Run.]
- 750 epochs at a constant learning rate of 5e-4 (where `ppo-4p`'s cosine schedule had decayed to), pure self-play at 4p, the undiscounted lambda critic, `ent_coef` 0.005.
- Checkpoints are selected on the table against three of m4's critic responders, where offers can land.
- Finals, 4,000 games a row (±1.3–1.5 pp): against three m4 responders, pairing with m4's control (`7164d396`, 25.0) and m4's critic proposer (`c4900f87`, 41.2); head to head with the start in both slot orders, with the start answering by m4's rule and with it rejecting everything; and against three F.

*Open for Marco.* The config holds the choices in items 1–3 and γ = 0.999; the run waits on his answers.
+ The 0.1 offer prior (the deviation above).
+ The constant learning rate of 5e-4.
+ `main@best` as the start.
+ The policy's γ: 0.999 (`ppo-4p`'s, as in the config) or 1.0. Open since 26 October.
+ Whether to add a sampled-policy row: argmax only offers where p(offer) > 0.5.

*Open harness rows (not queued).*
+ The symmetric table, where every seat proposes.
+ A number against F, which needs an F that answers offers.
+ Whether a policy trained with trades on the table keeps the edge.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/trade-learned-4p.yaml
```)
