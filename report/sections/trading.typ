#import "../macros.typ": *

= Trading <sec:trading>

Can a bot gain from domestic trades? A critic-only oracle finds few trades that help both sides, yet a harness proposing every trade the critic likes for both takes 45.5% of 4p games (parity 25), +20.5 pp paired. The edge is in proposing, not accepting, and holds with the strongest 4p policy. It also holds between equals: two seats trading with each other take 62–63% against two copies that don't (parity 50; @exp:trade-2v2). A learned trade head is built and queued (@exp:trade-engine, @exp:trade-learned).

== Is trading worth an action space?

=== Trades in the action space and env #chip("queued") <exp:trade-engine>
#meta(dates: "2026-11-02", dates-label: "built", owner: "marco", wandb: (), records: ())
#tldr[Built, off by default: domestic trades as actions, up to 3 offers a turn. On integration; to main and the GPU after the search-depth sweep.]

#lead[What changed.]
- *Actions.* Offer, accept, reject, cancel and confirm with a seat, before the no-op in `GraphActionSpace`, behind `env.trades: {offers: N}` (N = 0, the default, is today's game). An offer is legal on the mover's turn after the roll, with a card in hand and under N offers made.
- *Row.* A `trade` block: gives and asks per resource, proposer, who accepted, offers left.
- *Seats without a trade head* (F, older checkpoints, MCTS, `BELIEF`) reject every offer; R accepts at random.
- *Check.* With trades off, 80 games from four stored 4p records (`e5b302ee`, `c4900f87`, `7164d396`, `b07e5bd9`) replay bit for bit.

=== Trade oracle: which trades does the critic like for both sides? #chip("done") <exp:trade-oracle>
#meta(
  dates: "2026-10-17",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Marco's first step. A trade worth ≥ 2 pp to both sides exists in 6.2% of 4p positions and 0.3% of 2p ones: worth an action space in 4p, not 2p.]

*Question.* How often does a trade help both parties under our critics?

*Setup.* `tools/trade_oracle.py`: 3,000 post-roll positions per network, from self-play (temperature 1; about 3.4 per game, 884 games in 4p, 916 in 2p), each with every 1:1, 2:1 and 1:2 trade the mover and an opponent can pay (about 73 per 4p position), played on a clone. Both seats' values are read from the mover's row, the only kind the critic trains on. A trade is _mutual_ at margin m if both win probabilities rise by at least m pp. Probabilities don't conserve under a trade (median residual in the table), so margins under 2 pp are noise. Data: `report/data/trade-oracle-ppo-4p-main.json`, `report/data/trade-oracle-puffer-graph2-full.json`; no eval record or W&B mirror.

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

In 4p the best mutual trade gives the mover +1.2 pp and the partner +0.9 on average and costs the other two seats 1.1 pp. 92% are 2:1 (the mover gives two), most often two wood for an ore, partly because the `private` row shows the partner's hand only as a size. The mover's selfish best trade: +7.8 pp for it, −3.7 for the partner.

*Reading.* 4p has a modest, real pool of mutual trades, paid for by the other two seats; 2p is nearly zero-sum, so a 2p trade action would almost never be used.

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
#tldr[On the first critic whose four win probabilities sum to 1.00, the noise floor barely moves and the mutual pool shrinks to 2.5% of 4p positions at ≥ 2 pp, level with that noise.]

*Question.* Does a calibrated critic give a cleaner oracle?

*Setup.* The same tool on `aux-hidden-4p/plain@e600`, the first net trained with lambda critic targets (@sec:recipe): 3,000 positions from 846 games, 229k trades, 3 minutes. Data in `report/data/trade-oracle-aux-hidden-4p-plain.json`; no W&B mirror.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([4p critic], [Residual, median], [Residual, p95], [≥ 1 pp], [≥ 2 pp], [≥ 5 pp]),
    [V-trace \ #sub[`ppo-4p/main@best`]], [1.1 pp], [4.8 pp], [25.8 #ci[24.2][27.5]], [6.2 #ci[5.4][7.2]], [0.4],
    [Lambda \ #sub[`aux-hidden-4p/plain@e600`]], [1.0 pp], [3.8 pp], [22.7 #ci[21.2][24.3]], [*2.5* #ci[2.0][3.2]], [0],
  ),
  caption: [Residual and share of positions (%) with a mutual trade, 3,000 4p positions per critic.],
)

Some trade's residual still reaches 2 pp in 90% of positions (the V-trace sum sat at 0.93 and moved more under a hand change). The median best mutual trade gives the mover +0.8 pp and the partner +0.7; the other two seats lose 1.1 pp on average. 94% are 2:1, led by two sheep or two wood for a wheat and two wood for a sheep. The mover's selfish best trade (+7.2 pp, −3.3 for the partner) is mutual in 0.1% of positions.

*Reading.* Were the two parties' gains independent, 77% of positions would show a mutual trade; under this critic trades are nearly zero-sum between them. A trade head would find one about once in forty positions, and a harness proposing by this critic proposes gains the size of its noise (a one-point edge needs about 15,000 paired games). The harness did far better (@exp:trade-harness-m1m2).

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
#tldr[Marco's idea, after the oracle: let the critic propose and answer trades outside the action space. Over 200 games the proposer took 46.5% against three answer-only copies (parity 25), 55.5% re-evaluating after every build; reading the true hands adds nothing.]

*Question.* Does a seat that proposes the critic's mutual trades win more?

*Setup.* Marco's rules, with a calibrated critic and gains in each seat's own view:
- *Proposer* (`TRADE`). Once per own turn after the roll, it scores every 1:1, 2:1 and 1:2 bundle it can pay in one critic read (about 84 rows), keeps the Pareto front of those positive for both sides, and offers it by its own gain, up to three a turn; it confirms the acceptee best for itself. Otherwise it plays the policy.
- *Responders* accept iff their mean gain over eight determinized worlds is positive.
- *Plumbing.* One engine side channel (`VecEnv.set_action`), leaving rows, checkpoints and training untouched; choices depend only on public state (`tests/test_trade_fairness.py`).
- *Table.* F never accepts (its one-ply value sees no change on an accept and keeps reject), so `aux-hidden-4p/plain@e600` plays and judges in every seat; trading is the only asymmetry. 50 `heldout-4p` boards × 4 rotations a row; five minutes in all.

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
  caption: [Slot 0's share against three answer-only copies, 200 `heldout-4p` games a row.],
)

Under Marco's rule 85% of offers land, under 1% are cancelled, and the responders take 17–19. Bundles and claimed gains match the oracle (`tools/trade_pilot.py`): mostly 1:1 and 2:1, sheep for wood and ore for brick on top; +1.35 pp for the mover per offer (median 1.2), +0.4 for the partner, +0.6 on an accepted read, −0.3 on a rejected one.

*Reading.* Sampling hidden hands costs nothing. Each claimed gain sits at the critic's noise floor, yet a dozen small trades a game compound into twenty points of share, paid by the seats outside each trade: the oracle missed the sum.

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
#tldr[Marco's rule takes 45.5% (parity 25), +20.5 pp paired against the same table with nobody proposing (p < 10#super[−90]): the largest effect in this log. A 1 pp margin cuts the gain to +6.7 pp.]

*Question.* Does the edge hold at 4,000 games, and does a margin help?

*Setup.* m1 is Marco's rule; m2 gives proposer and responders a 1 pp margin. Each row's control replays the same games with a plain `POLICY` in slot 0. 1,000 `heldout-4p` boards × 4 rotations a row, paired by McNemar; the pilot's network and critic; 34 minutes for 16,000 games.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, left, right, left),
    header: ([Row], [Proposer], [Control], [Paired], [Offers, trades / game], [Records]),
    [m1 \ #sub[0 pp margin]], [*45.5* #ci[43.9][47.0]], [25.0 #ci[23.7][26.4]], [*+20.5 pp* \ #sub[1,204 / 386, p < 10#super[−90]]], [14.7, 12.4], [`ff0c2cb2` \ `7ffdd60f`],
    [m2 \ #sub[1 pp margin]], [31.7 #ci[30.3][33.2]], [25.0 #ci[23.7][26.4]], [+6.7 pp \ #sub[p < 10#super[−14]]], [3.8, 3.2], [`d7790a27` \ `7c06e560`],
  ),
  caption: [Slot 0's share against its no-proposer control, 4,000 `heldout-4p` games a row. Under each delta: games won only with / only without trading, and McNemar's p.],
)

In m1, 84% of offers land; the responders take 18.0, 18.3 and 18.3.

*Reading.* The margin throws share away: the 0 pp rule trades four times as often and gains three times as much, so the critic's judgement carries signal below its per-read noise.

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
#tldr[No. Against three proposers, accepting the offers it likes does no better than rejecting everything (12.7 against 13.2, p = 0.54): the proposer captures each trade's surplus.]

*Question.* Is the responder's side of a trade worth anything?

*Setup.* Three proposers at Marco's rule; slot 0 is the answer-only seat or a plain `POLICY` that rejects every offer. 4,000 games a row, 8,000 in all, paired as in @exp:trade-harness-m1m2.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([Slot 0, against three proposers], [Share], [Record]),
    [Accepts the offers it likes], [12.7 #ci[11.7][13.8]], [`fe7baed5`],
    [Rejects every offer], [13.2 #ci[12.1][14.2]], [`58f7c074`],
  ),
  caption: [The acceptor against the rejecter, 4,000 `heldout-4p` games a row.],
)

Paired −0.45 pp (371 games won only by the acceptor, 389 only by the rejecter). The proposers take 27–31.

*Reading.* The proposer picks the trades, so the acceptor's read of a gain is noise around zero; the passive seat is crushed either way (13 against a parity of 25).

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
#tldr[`ppo-4p/main@best` in every seat, proposing by the calibrated critic, takes 41.2% (+16.2 pp paired): the edge holds with the strongest 4p policy, four points below m1.]

*Question.* Does the trading edge survive a stronger player?

*Setup.* m1's tables with `ppo-4p/main@best` (e3650) in every seat; `aux-hidden-4p/plain@e600`'s critic judges trades, since `ppo-4p`'s V-trace critic does not sum to one. 4,000 games a row; the control's slot 0 never trades.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Slot 0], [Share], [Offers, trades / game], [Record]),
    [Proposes (Marco's rule)], [*41.2* #ci[39.7][42.7]], [13.8, 11.6], [`c4900f87`],
    [Never trades (control)], [25.0 #ci[23.7][26.4]], [—], [`7164d396`],
  ),
  caption: [`ppo-4p/main@best` with trading against three answer-only copies of itself, 4,000 `heldout-4p` games a row.],
)

Paired: 1,080 games won only with trading, 433 only without, p < 10#super[−60].

*Reading.* Either a better player gets less from the same trades, or a critic from another network reads its positions slightly worse.

#repro(```sh
catanrl run configs/experiments/trade-harness-m4.yaml   # needs the ppo-4p/main and aux-hidden-4p/plain runs
```)

=== The fair test: two traders against two non-traders #chip("done") <exp:trade-2v2>
#meta(
  dates: "2026-11-03 06:47–08:08",
  owner: "marco",
  config: "configs/experiments/trade-2v2-4p.yaml",
  wandb: (("trade-2v2-4p-main-98b87d55", "main"),),
  records: ("f97034f7", "715096b5", "317670c3", "2071fef6", "f8473069", "2adefc8e", "324d79eb"),
)
#tldr[Trading pays between equals. Two seats that trade with each other take 61.7% (adjacent) and 62.9% (alternating) of games against two copies of the same net that don't, against exactly 50 without trading: +11.7 and +12.9 pp paired. Both traders gain 5–7 points each, so neither is exploiting the other. Against F, which the pair already beats 91–93% of the time, trading adds 3 points.]

*Question.* Do two seats that trade with each other beat two seats that don't (Marco)? In m1–m4 the proposer's acceptors lost share (18 each in m1), so part of the proposer's gain could be exploitation. Here both sides of every trade use the same rule; the only asymmetry is that two seats trade at all.

#lead[Setup.]
- *Seats.* T proposes and answers by Marco's rule (m4's player and critic: `ppo-4p/main@best`, judged by `aux-hidden-4p/plain@best`); N is the same net as a plain `POLICY`, which rejects every offer.
- *Rows.* Adjacent `[T, T, N, N]` and alternating `[T, N, T, N]`; the suite's four rotations put each pattern in every seating. Then the same two patterns with F as the non-traders (F never accepts, so it needs no change).
- *Control.* Each lineup with every T replaced by N, on the same games. With N in all four seats it is exactly 50 by symmetry: each game's winning seat falls in a given pair of seats in two of the four rotations.
- *Metric.* The two traders' combined share (parity 50), paired game by game with the control; 4,000 `heldout-4p` games a row. 1 h 21 min on the GPU for all seven rows.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, left),
    header: ([Lineup], [Traders' share], [Control], [Paired gain], [Trades / game], [Records]),
    tgroup(6)[Against the same net],
    [`T T N N`], [*61.7* #ci[60.2][63.2]], [50.0], [+11.7 #sub[(p = 1e-30)]], [13.7], [`f97034f7` / `317670c3`],
    [`T N T N`], [*62.9* #ci[61.4][64.4]], [50.0], [+12.9 #sub[(p = 1e-36)]], [14.1], [`715096b5` / `317670c3`],
    tgroup(6)[Against F],
    [`T T F F`], [*96.2* #ci[95.5][96.7]], [93.2], [+3.0 #sub[(p = 2e-10)]], [11.8], [`2071fef6` / `2adefc8e`],
    [`T F T F`], [*94.8* #ci[94.0][95.4]], [91.3], [+3.4 #sub[(p = 2e-11)]], [12.2], [`f8473069` / `324d79eb`],
  ),
  caption: [The two traders' combined share (%) on `heldout-4p`, 4,000 games a row, against the same lineup with every T replaced by N. Paired gain: McNemar on games the pair wins.],
) <tab:trade-2v2>

- *Paired intervals.* +11.7 (9.7–13.7) and +12.9 (11.0–14.9) against the same net; +3.0 (2.1–3.9) and +3.4 (2.4–4.4) against F.
- *Per seat, against the same net.* The traders take 31.6 and 30.1 (adjacent), 32.3 and 30.6 (alternating), each 5–7 points over 25; each outsider loses about 6.
- *Per seat, against F.* The gain comes out of F's share, which falls from 6.8 to 3.8% (adjacent) and 8.7 to 5.2% (alternating). Within the adjacent pair it is uneven: +4.4 for the trader who plays after an F, −1.5 ±1.9 for the one who plays before.
- *Trades.* Only between the two traders: about 44 offers a game, 12–14 completed, 4% cancelled. Games end 3.3 turns sooner against the same net (76.4 vs 79.7), 5.2 sooner against F.

*Reading.* Yes for Marco's question, with the harness's rule: two seats trading with each other gain about 12 points of combined share from two equals that don't, and the outsiders pay for it. Unlike m1, both sides of the trades gain, so the edge is not one trader exploiting a naive acceptor. Seating barely matters (adjacent 61.7, alternating 62.9).

#repro(```sh
catanrl run configs/experiments/trade-2v2-4p.yaml   # needs the ppo-4p/main and aux-hidden-4p/plain runs
```)

== Learned trading

=== A trade head trained from the PPO checkpoint #chip("queued") <exp:trade-learned>
#meta(dates: "2026-11-02", dates-label: "built", owner: "marco", config: "configs/experiments/trade-learned-4p.yaml", wandb: (), records: ())
#tldr[Step (2) of Marco's plan: option one on `ppo-4p/main@best`, from a function-preserving start. About 3.5 h on the GPU, after the search-depth sweep; waits on Marco's answers below.]

*Plan.* Marco wants learned trading without an explosive action space. Option one (the default): a trade action plus 10 floats, the mover's gives and gets per resource rounded to integers (11 outputs), and a yes/no for the responder. Option two: one output per integer amount (18 choices) per resource per side (181 outputs). Marco expects a policy trained with trades from scratch not to trade well, so this starts from a competent PPO policy, trade actions masked in at a low entropy bonus.

#lead[Built.]
- *Head.* On the shared trunk: logits for offer, accept, reject, cancel and confirm, and ten Gaussian means with a learned spread for the bundle. Log-prob: the action's, plus on an offer the Gaussian density of the unrounded floats.
- *Floats to offer*, in order: negatives to 0, a resource on both sides netted, gives clipped to the hand, gets to the bank's 19. An empty side means no offer but spends one of the turn's offers.
- *Checks.* With trades masked out the net gives the checkpoint's logits and values exactly (2,400 rows); an update at learning rate 0 changes nothing (KL 0, clip fraction 0).

*One deviation, to approve.* A zero-initialized offer logit would be sampled with probability around 4 × 10#super[−6] (1,334 post-roll rows), so training would never try a trade. Instead it is a gate on the log-sum-exp of the other legal actions: p(offer) starts at 0.1 and the other actions keep their shares.

#lead[Run.]
- 750 epochs at a constant 5e-4 (`ppo-4p`'s decayed cosine), 4p self-play, the undiscounted lambda critic, `ent_coef` 0.005; checkpoints selected against three of m4's critic responders, where offers land.
- Finals, 4,000 games a row (±1.3–1.5 pp): against three m4 responders, paired with m4's control (`7164d396`, 25.0) and proposer (`c4900f87`, 41.2); against the start in both slot orders, the start answering by m4's rule or rejecting all; against three F.

#lead[Open for Marco.]
The config holds a choice for items 1–4.
+ The 0.1 offer prior (the deviation above).
+ The constant 5e-4.
+ `main@best` as the start.
+ The policy's γ: 0.999 (`ppo-4p`'s, in the config) or 1.0; open since 26 October.
+ Whether to add a sampled-policy row: argmax only offers where p(offer) > 0.5.
+ Replacing the finals against m4's naive responders with the 2v2 test (@exp:trade-2v2): two learned traders against two copies of the start.

#lead[Open harness rows (not queued).]
+ The symmetric table, where every seat proposes.
+ A number against F that answers offers (@exp:trade-2v2 gives one against F that never accepts).
+ Whether a policy trained with trades on the table keeps the edge.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/trade-learned-4p.yaml
```)
