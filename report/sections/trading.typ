#import "../macros.typ": *

= trading <sec:trading>

can a bot gain from domestic trades, and are they worth an action space? the engine has the rules, but the policy can't trade yet. a critic-only oracle finds few trades that help both sides: 2.5–6.2% of 4p positions at ≥ 2 pp, almost none in 2p. yet a harness that proposes every trade the critic calls good for both wins 45.5% of 4p games against a parity of 25. that is +20.5 pp paired, the largest effect in this log. the edge is in proposing, not accepting, it holds with the strongest 4p policy, and learned trading has not started.

== is trading worth an action space?

=== trades in the action space and env #chip("todo") <exp:trade-engine>
#meta(owner: "marco", wandb: (), records: ())
#tldr[the engine already plays domestic trades by catanatron's rules. the policy can't: the action space, the env and the row have no trades. this blocks a learned trader, not the harness below.]

*status.* cppanatron implements domestic trades (offer, accept, reject, confirm, cancel) in parity with catanatron. what is missing is everything in front of it. the map-agnostic action space (`GraphActionSpace`) leaves trades out, the env never offers them, and the encoder has no trade prompt in the row. so "adding trading" means an action-space extension plus the env, the mask and the trade prompts. the trade harness (@exp:trade-harness-pilot) sidesteps all of it with one engine side channel (`VecEnv.set_action`).

=== trade oracle: which trades does the critic like for both sides? #chip("done") <exp:trade-oracle>
#meta(
  dates: "2026-10-17",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[marco's first step: before any action-space work, ask the trained critic which trades help both sides. in 4p a trade worth ≥ 2 pp to both exists in 6.2% of positions; in 2p in 0.3%. worth an action space in 4p, not in 2p.]

*question.* marco suspects that a policy trained with trades from scratch won't learn to trade well. so first: under the critics we already have, how often does a trade leave both parties better off, by how much, and what does it look like?

*setup.* `tools/trade_oracle.py`. 3,000 positions per network from its own self-play (temperature 1, about 3.4 positions per game, 884 games in 4p and 916 in 2p), taken after the roll. for each position, every 1:1, 2:1 and 1:2 trade between the mover and each opponent that both hands can pay (about 73 per 4p position), applied to a clone through the engine's own offer, accept and confirm actions. both seats' win probabilities are read from the mover's row: the critic trains on acting rows only, and a partner's own row is off-distribution. a seat's _gain_ is the change in its win probability, in percentage points (pp). a trade is _mutual_ at margin m if both parties gain at least m. the critic has a noise floor: its per-seat win probabilities don't conserve under a trade, with a median residual of 1.1 pp in 4p and 0.5 pp in 2p. so the 0.5 and 1 pp rows are within noise, and the 2 pp row is the one to read. there is no eval record and no wandb mirror; the numbers are in `report/data/trade-oracle-ppo-4p-main.json` and `report/data/trade-oracle-puffer-graph2-full.json`.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    header: ([critic], [game, level], [≥ 1 pp], [≥ 2 pp], [≥ 5 pp], [residual (median)]),
    [`ppo-4p/main@best` (e3650)], [4p, private], [25.8 #ci[24.2][27.5]], [*6.2* #ci[5.4][7.2]], [0.4], [1.1 pp],
    [`puffer-graph2/full@best` (e3350)], [2p, full], [4.3], [0.3 #ci[0.1][0.5]], [—], [0.5 pp],
  ),
  caption: [share of positions (%) with a mutual trade at each margin, 3,000 positions per critic.],
)

in 4p the best mutual trade gains the mover 1.2 pp and the partner 0.9 pp on average, and costs the two seats outside it 1.1 pp. 92% of mutual trades are 2:1 (the mover gives two); the most common is two wood for an ore. at the private level the mover's row sees the partner's hand only as a size, so the 2:1 skew is partly what the critic can see. the mover's selfish best trade gains it 7.8 pp and costs the partner 3.7 pp on average: trades are mostly zero-sum, as expected.

*reading.* 4p has a modest, real pool of 2:1 trades the critic likes for both sides, each paid for by the other two seats. two-player catan is nearly zero-sum for this critic, so a 2p trade action would almost never be used.

#repro(```sh
# the defaults are the saved arguments; each writes report/data/trade-oracle-<run>.json
uv run python tools/trade_oracle.py --run ppo-4p/main          # 4p
uv run python tools/trade_oracle.py --run puffer-graph2/full   # 2p
```)

=== the oracle on the calibrated lambda critic #chip("done") <exp:trade-oracle-lambda>
#meta(
  dates: "2026-10-19 20:15",
  wandb: (),
  records: (),
)
#tldr[a rerun on the first critic whose four win probabilities sum to 1.00. the noise floor barely moves and the mutual pool shrinks to 2.5% of 4p positions at ≥ 2 pp. under this critic trades are nearly zero-sum between the parties, and the mutual gains sit at its noise floor.]

*question.* does a calibrated critic give a cleaner oracle?

*setup.* the same tool on `aux-hidden-4p/plain@e600`, the first net trained with lambda critic targets (@sec:recipe), whose per-seat win probabilities sum to the realized 1.00. 3,000 positions from 846 self-play games, 229k trades, 3 minutes. data in `report/data/trade-oracle-aux-hidden-4p-plain.json`; no wandb mirror.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([4p critic], [residual median], [residual p95], [≥ 1 pp], [≥ 2 pp], [≥ 5 pp]),
    [v-trace (`ppo-4p/main@best`)], [1.1 pp], [4.8 pp], [25.8 #ci[24.2][27.5]], [6.2 #ci[5.4][7.2]], [0.4],
    [lambda (`aux-hidden-4p/plain@e600`)], [1.0 pp], [3.8 pp], [22.7 #ci[21.2][24.3]], [*2.5* #ci[2.0][3.2]], [0],
  ),
  caption: [the calibrated critic against the v-trace one: share of positions (%) with a mutual trade at each margin.],
)

the residual still reaches 2 pp in 90% of positions. the v-trace critic's sum sat at 0.93 and moved more under a hand change. under the lambda critic the best mutual trade gains the mover 0.8 pp and the partner 0.7 pp at the median. 94% of mutual trades are 2:1 (two sheep for a wheat, two wood for a wheat and two wood for a sheep lead the list), and the other two seats lose 1.1 pp on average. the mover's selfish best trade (+7.2 pp for it, −3.3 for the partner) is mutual in 0.1% of positions.

*reading.* the independence reference says what 2.5% means: if a trade's effect on the partner were unrelated to its effect on the mover, 77% of positions would show a mutual trade. so under this critic trades are nearly zero-sum between the two parties, and the mutual pool sits at the critic's own noise floor. a trade head would find a trade worth ≥ 2 pp to both in about one position in forty. a harness that proposes by this critic proposes gains the size of its noise, and resolving a one-point edge takes about 15,000 paired games. the harness below did far better than this predicts.

#repro(```sh
uv run python tools/trade_oracle.py --run aux-hidden-4p/plain
```)

== a critic-driven trade harness

=== marco's trade harness: the pilot #chip("done") <exp:trade-harness-pilot>
#meta(
  dates: "2026-10-20 03:44–03:49",
  owner: "marco",
  config: "configs/experiments/trade-harness-pilot.yaml",
  wandb: (("trade-harness-pilot-main-8d9a5b49", "trade-harness-pilot, eval only"),),
  records: ("0fc4352d", "1d253765", "c34e389c", "09893ca0"),
)
#tldr[marco's idea: keep trades out of the action space and let the critic propose and answer them. in a 200-game pilot the proposing seat won 46.5% against three copies that only answer offers (parity 25), and 55.5% when it re-evaluates after every build. reading the true hands instead of sampled ones adds nothing.]

*question.* marco's proposal after the oracle rerun: make the trade decision outside the action space, as a critic-driven harness that proposes the best mutual trades, with responders judging offers the same way. does a seat that does this win more?

*setup.* marco's rules: a calibrated critic, gains in each seat's own view, a pareto front for the proposer, positive expected value for the acceptor. the `TRADE` agent plays the policy everywhere. once per own turn, after the roll, it enumerates every 1:1, 2:1 and 1:2 bundle it can pay and reads the critic once for all of them (about 84 rows a read). it keeps the bundles whose gain is positive for it and for the partner, drops the pareto-dominated ones (those another bundle beats for both), and offers the rest in order of its own gain, up to three a turn. a responder accepts iff its mean gain over eight determinized worlds (hidden hands sampled consistent with what it has seen) is positive. the proposer confirms the acceptee best for itself.

trades go through one engine side channel (`VecEnv.set_action`); the row layout, the checkpoints and training are untouched. fairness is black-box: the choices are identical on any game with the same public state (`tests/test_trade_fairness.py`). native F never accepts an offer (its one-ply value sees no cards move on an accept, and a strict comparison keeps reject). so every table is one network against itself, `aux-hidden-4p/plain@e600` as policy and critic in every seat, with the trading seat the only asymmetry. four rows of 200 games (50 `heldout-4p` boards × 4 rotations), five minutes in all. parity is 25.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    header: ([proposer], [share], [offers / game], [trades / game], [record]),
    [marco's rule (0 pp margin)], [*46.5* #ci[39.7][53.4]], [14], [12], [`0fc4352d`],
    [re-evaluates after every build], [*55.5* #ci[48.6][62.2]], [26], [22], [`1d253765`],
    [1 pp margin, responders too], [31.0 #ci[25.0][37.7]], [4], [—], [`c34e389c`],
    [reads the true hands (unfair ceiling)], [46.0 #ci[39.2][52.9]], [—], [—], [`09893ca0`],
  ),
  caption: [slot 0’s share of wins against three responder-only copies on `heldout-4p`, 200 games a row.],
)

under marco's rule 85% of offers land, cancels are under 1%, and the three responders take 17–19. the bundles are the oracle's: mostly 1:1 and 2:1, with sheep for wood and ore for brick on top. so are the gains the critic claims for them (`tools/trade_pilot.py`): +1.35 pp for the mover per offer (median 1.2), +0.4 for the partner, +0.6 on an accepted read, −0.3 on a rejected one.

*reading.* sampling the hidden hands costs nothing: the true hands give the same share. the claimed gains sit at the critic's noise floor, yet a dozen small trades a game compound into twenty points of share, paid for by the seats outside each trade.

#repro(```sh
catanrl run configs/experiments/trade-harness-pilot.yaml   # needs the aux-hidden-4p/plain run
# the bundles offered and the gains the critic claimed for them:
uv run python tools/trade_pilot.py --config configs/experiments/trade-harness-pilot.yaml --row 0
```)

=== the harness at 4,000 games: m1 (0 pp margin) and m2 (1 pp margin) #chip("done") <exp:trade-harness-m1m2>
#meta(
  dates: "2026-10-20 03:50–04:24",
  owner: "marco",
  config: "configs/experiments/trade-harness.yaml",
  wandb: (("trade-harness-main-a9a5c206", "trade-harness, eval only"),),
  records: ("ff0c2cb2", "7ffdd60f", "d7790a27", "7c06e560"),
)
#tldr[at 4,000 games a row, marco's rule takes 45.5% against a parity of 25: +20.5 pp paired against the same table with nobody proposing (p < 10#super[−90]). a 1 pp margin cuts that to +6.7 pp. the largest effect in this log.]

*question.* does the pilot's edge hold at a sample size that resolves it, and does a margin above the critic's noise help?

*setup.* m1 is marco's rule; m2 puts the proposer and the responders at a 1 pp margin. each row has a control: the same games with a plain `POLICY` in slot 0, so nobody proposes and slots 1–3 are identical in both. 1,000 `heldout-4p` boards × 4 rotations = 4,000 games a row. the paired delta is slot 0’s share against its control, with mcnemar's test on the games won only with trading or only without. same network and critic as the pilot; 34 minutes for the 16,000 games.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, left),
    header: ([row], [proposer], [control], [paired], [offers, trades / game], [records]),
    [m1, 0 pp], [*45.5* #ci[43.9][47.0]], [25.0 #ci[23.7][26.4]], [*+20.5 pp* (1,204 / 386), p < 10#super[−90]], [14.7, 12.4], [`ff0c2cb2` / `7ffdd60f`],
    [m2, 1 pp], [31.7 #ci[30.3][33.2]], [25.0 #ci[23.7][26.4]], [+6.7 pp, p < 10#super[−14]], [3.8, 3.2], [`d7790a27` / `7c06e560`],
  ),
  caption: [slot 0’s share with trading against its no-proposer control, 4,000 games a row, `heldout-4p`. in brackets after the paired delta: games won only with trading / only without.],
)

in m1, 84% of offers land and the responders take 18.0 / 18.3 / 18.3.

*reading.* under this critic the margin throws share away. the 0 pp rule trades four times as often as the 1 pp rule and wins three times as much over parity. so the critic's trade judgement carries signal well below its own per-read noise. the oracle was right about the per-trade gains and wrong about what a dozen of them a game add up to.

#repro(```sh
catanrl run configs/experiments/trade-harness.yaml   # needs the aux-hidden-4p/plain run
```)

=== m3: does accepting pay? #chip("done") <exp:trade-harness-m3>
#meta(
  dates: "2026-10-20 05:31–06:38",
  config: "configs/experiments/trade-harness-m3.yaml",
  wandb: (("trade-harness-m3-main-359acf90", "trade-harness-m3, eval only"),),
  records: ("fe7baed5", "58f7c074"),
)
#tldr[no. at a table of three proposers, a seat that accepts the offers it likes does no better than one that rejects everything (12.7 against 13.2, p = 0.54). the proposer captures each trade's surplus: the edge is in proposing.]

*question.* is the responder's side of a trade worth anything?

*setup.* three proposers at marco's rule. slot 0 is either the responder-only seat (accepts the offers its critic likes) or a plain `POLICY` that rejects every offer. 4,000 games a row, 8,000 in all, paired as in m1 (@exp:trade-harness-m1m2).

*result.*
#figure(
  btable(
    columns: (auto, auto, auto),
    align: (left, right, left),
    header: ([slot 0, against three proposers], [share], [record]),
    [accepts the offers it likes], [12.7 #ci[11.7][13.8]], [`fe7baed5`],
    [rejects every offer], [13.2 #ci[12.1][14.2]], [`58f7c074`],
  ),
  caption: [the acceptor against the rejecter, 4,000 games a row, `heldout-4p`.],
)

paired −0.45 pp: 371 games won only by the acceptor, 389 only by the rejecter, p = 0.54. the three proposers take 27–31.

*reading.* accepting buys nothing. the proposer picks the trades, so it captures each one's surplus, and the acceptor's own read of a gain is noise around zero. the passive seat is crushed whether it trades or not (13 against a parity of 25).

#repro(```sh
catanrl run configs/experiments/trade-harness-m3.yaml   # needs the aux-hidden-4p/plain run
```)

=== m4: the 4p specialist with trading #chip("done") <exp:trade-harness-m4>
#meta(
  dates: "2026-10-20 14:35–14:51",
  config: "configs/experiments/trade-harness-m4.yaml",
  wandb: (("trade-harness-m4-main-2bc4df25", "trade-harness-m4, eval only"),),
  records: ("c4900f87", "7164d396"),
)
#tldr[the edge holds with the strongest 4p policy. `ppo-4p/main@best` in every seat, proposing by the calibrated critic, takes 41.2% (+16.2 pp paired), four points less than the weaker plain policy got in m1.]

*question.* run because m1 was positive: does the trading edge survive a stronger player?

*setup.* m1’s tables with `ppo-4p/main@best` (e3650) playing every seat. the calibrated critic of `aux-hidden-4p/plain@e600` judges the trades, because `ppo-4p`'s own v-trace critic does not sum to one. the control puts a never-trading `POLICY` in slot 0. 4,000 games a row.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, left),
    header: ([slot 0], [share], [offers, trades / game], [record]),
    [proposes (marco's rule)], [*41.2* #ci[39.7][42.7]], [13.8, 11.6], [`c4900f87`],
    [never trades (control)], [25.0 #ci[23.7][26.4]], [—], [`7164d396`],
  ),
  caption: [`ppo-4p/main@best` with trading against three responder-only copies of itself, 4,000 games a row, `heldout-4p`.],
)

paired *+16.2 pp*: 1,080 games won only with trading against 433 only without, p < 10#super[−60].

*reading.* the edge holds, four points smaller than in m1. either a better player gets less from the same trades, or a critic from another network reads its positions slightly worse.

#repro(```sh
catanrl run configs/experiments/trade-harness-m4.yaml   # needs the ppo-4p/main and aux-hidden-4p/plain runs
```)

== learned trading

=== a trade head trained from the ppo checkpoint #chip("todo") <exp:trade-learned>
#meta(owner: "marco", wandb: (), records: ())
#tldr[step (2) of marco's plan: a small trade head for the proposer and a yes/no for the responder, trained from the ppo checkpoint. not started; it needs trades in the action space first (@exp:trade-engine).]

*plan.* marco wants learned trading without an explosive action space. option one: a trade action plus 10 floats, one per resource for what the mover gives and what it gets, each rounded to an integer (11 extra outputs). option two: one output per integer amount (18 choices) per resource per side (181 extra outputs). a policy trained with trades from scratch probably won't learn to trade well. so the order is: first train a competent policy with the ppo recipe. then add option one for the proposer and a yes/no for the responder, trained from the ppo checkpoint with the trade actions masked in at a low entropy bonus. option one is the default.

*open harness rows (not queued).* the symmetric table, where every seat proposes; a number against F, which needs an F that answers offers; and whether a policy trained with trades on the table keeps the edge.
