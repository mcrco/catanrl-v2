#import "../macros.typ": *

= Refitting the critic after PPO <sec:critic>

Marco's idea: during multi-agent PPO the critic chases a policy that keeps changing, so it probably never converges on the final one. Freeze the policy, keep training the critic on many of the final policy's games, and MCTS should get better with the more accurate critic. A second reason to expect a gain: the start's critic is discounted (V-trace targets at γ = 0.999, λ = 0.95), so it predicts something closer to 0.999^(moves left) × P(win) than P(win). That is also not the undiscounted win/loss AlphaZero's value head trains on.

The result is null. The refit calibrated the critic (it no longer discounts) but did not make it rank positions better, and the search did not get stronger: +2.1 pp at 512 sims (p = 0.15) and +0.1 pp at 2,048 sims (p = 1.0). The idea fails its pre-registered rule, so the value-head-only arm was not run.

== The start's critic

=== CPU probe of the start's critic #chip("done") <exp:critic-probe>
#meta(
  dates: "2026-10-23",
  wandb: (),
  records: (),
)
#tldr[Before the GPU run, Claude scored the start's critic against real outcomes on CPU. It is discounted as expected (the two seats' values sum to 0.936) but ranks positions well. The only headroom is calibration, about 0.01 nats, which the Gumbel root mostly cancels, so on this evidence alone Claude did not expect the refit to move the search rows.]

*Question.* As the first step of Marco's refit plan: how far is the start's critic from the undiscounted outcome, and how much can a refit gain on held-out games?

*Setup.* 5,120 raw-policy games of `puffer-graph2/public@e3600` on its own training boards (827k decisions). Every value is scored against the game's undiscounted outcome.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto),
    align: (left, right),
    header: ([Measure], [Value]),
    [Log loss vs the undiscounted outcome], [0.370 (constant guess 0.682)],
    [The same, by game stage], [0.68 in the first tenth of a game → 0.11 in the last],
    [The two seats' values summed], [0.936 (1 if undiscounted)],
    [Calibration], [A predicted 0.74 wins 83% of the time],
    [Held-out gain, seat-sum fix \ #sub[each seat's value ÷ the two seats' sum, no training]], [0.010 #ci[0.008][0.011] nats],
    [Held-out gain, value-head refit \ #sub[on the frozen trunk; the best fit]], [0.012 nats],
    [AUC within the same stage of the game], [0.894 for the start and both fixes],
  ),
  caption: [The start's critic (`puffer-graph2/public@e3600`) on 5,120 raw-policy games.],
)

*Reading.* It is discounted as expected: v ≈ 0.999^(decisions left) × P(win) fits where v ≈ P(win) does not, so it is under-confident about wins. The only headroom is calibration, and neither fix ranks positions any better. The Gumbel root rescales Q to its own range, which cancels most of a calibration change. Whether a separately trained critic trunk on \~400k games finds discrimination the value head could not is what the GPU run tests (@exp:critic-refit).

#repro(```sh
# Not reproducible as run: a one-off CPU probe whose script is not in the repo.
# tools/critic_heldout.py, the GPU run's held-out scorer, computes the same measures
# (log loss against the outcome, AUC within turn bins, the seat-sum fix) for any checkpoint.
```)

== The refit

=== Frozen policy, refit critic, judged in the search: `critic-refit-2p` #chip("done") <exp:critic-refit>
#meta(
  dates: "2026-10-23 09:57 – 2026-10-24 10:45 (paused 13:40–00:54 for canopy)",
  owner: "marco",
  config: "configs/experiments/critic-refit-2p.yaml",
  wandb: (("critic-refit-2p-main-790938cd", "critic-refit-2p/main"), ("az-2p-2048-soft-main-8105fbb1", "az-2p-2048-soft (baseline 856a0421)"), ("gumbel-edge-2p-main-efda50ac", "gumbel-edge-2p, eval only (baseline 831eea84)")),
  records: ("576848d8", "f841ca86", "9641a3aa", "856a0421", "831eea84"),
)
#tldr[A separate copy of the critic, refit on \~400k of the frozen policy's games, is calibrated (the discount is gone) but ranks positions no better than the start's. Inside the search it changes nothing: +2.1 pp at 512 sims (p = 0.15), +0.1 pp at 2,048 (62.5% vs 62.4%, p = 1.0). A more accurate critic is not what limits our search.]

*Question.* With only the critic different, does MCTS with a refit critic beat the raw start more often than MCTS with the start's own critic?

#lead[Setup.]
- *Frozen policy.* `train.freeze: [actor]` takes the policy out of the optimizer. Its logits stay bit-identical through training, checked at every launch against `init_from`.
- *Split trunk.* `init_from.split_shared` copies the shared trunk into separate policy and critic networks that reproduce the start exactly: identical logits, values and 512-sim searches (merged `3dc0450`). The e0 select (`576848d8`) plays the same games as the start's `856a0421` (identical games hash).
- *Arm `main`.* Refits the whole split critic (376k parameters) on the Monte Carlo outcome (γ = 1, λ = 1) of raw-policy self-play in the start's own env: Adam 3e-5 → 3e-6 over 500 epochs, \~400k games of the frozen policy at temperature 1, of which \~63% reach the loss. Segments are 256 steps, so about two thirds of the targets are real outcomes and the rest bootstrap. Estimated at \~8–14 GPU hours before launch, with two trunks not yet timed.
- *Held-out scoring.* A fixed set of 2,048 held-out games on boards outside the training pool (`tools/critic_heldout.py`) scores every checkpoint, paired, against two calibration-only fixes of the start. The run counts as converged if e500 is within 0.002 nats of e450.
- *Decision rule, fixed before launch.* e500 inside MCTS against the raw start, in two rows, each paired with the start's weights in the same search on the same games (McNemar, Holm at 0.05 over the two): 512 sims on the 1,000 games of `select-2p-500` (baseline `856a0421`), and 2,048 sims on `heldout-2p`, 500 boards in both seat orders (baseline `831eea84`). At \~1.6–1.8 pp paired SE, 80% power needs an effect of \~5 pp, so a null reads "no effect of 5 pp or more".

*Result.* The `main` arm was paused at e266 for the Canopy search row (@sec:canopy; the two do not fit in 8 GB together). Training ended at 03:00, the 2048-sim row at 10:45.

#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([Decision row], [Start's critic], [Refit (e500)], [Paired], [Won only by \ e500 / start], [McNemar p]),
    [512 sims \ #sub[`select-2p-500`, 1,000 games]], [57.4 #ci[54.3][60.4] \ #sub[`856a0421`]], [*59.5* #ci[56.4][62.5] \ #sub[`f841ca86`]], [+2.1 pp], [105 / 84], [0.15],
    [2,048 sims \ #sub[`heldout-2p`, 500 boards, \ both seat orders]], [62.4 #ci[59.4][65.3] \ #sub[`831eea84`]], [*62.5* #ci[59.5][65.5] \ #sub[`9641a3aa`]], [+0.1 pp], [110 / 109], [1.0],
  ),
  caption: [The two decision rows: MCTS against the raw start, with the start's critic or the refit one, on the same games.],
)
Neither row passes Holm (smallest p 0.15 against a threshold of 0.025), so `main` fails and the `head` arm is not run.

#figure(
  btable(
    columns: (1fr, auto),
    align: (left, right),
    header: ([Critic], [Cross-entropy vs outcome]),
    [Start (e0)], [0.3634],
    [Start, seats normalized, v/(v0+v1)], [0.3499],
    [Start, Platt scaling by turn], [0.3470],
    [Refit (e500)], [*0.3460*],
  ),
  caption: [The held-out critic on 2,048 games on boards outside the training pool (330k decisions). Lower is better.],
)
Paired per decision with game-clustered CIs, refit minus start is −0.0174 nats #ci[−0.0212][−0.0134], but refit minus Platt scaling is only −0.0011 #ci[−0.0023][+0.0002]. Ranking within a turn did not move (AUC 0.9010 → 0.9016). The run converged: e500 is within 0.0001 of e450 (the rule was 0.002), and the curve is flat from about e100.

*Reading.* The refit fixed what the probe said was wrong, the discount, and nothing else. By e50 the two seats' values summed to 1.000 (the start's: 0.936), but the critic ranks positions no better, and almost all of the calibration gain is available for free by rescaling the start's values. The search only compares positions and moves, so calibration alone cannot help it: at 2,048 sims the two critics win and lose the same games (219 of 1,000 differ, split evenly). A better critic for the search would have to rank better, not just be calibrated.

#repro(```sh
catanrl run configs/experiments/critic-refit-2p.yaml --only main   # Resumes from e266 in the same runs dir
uv run python tools/critic_heldout.py play --run critic-refit-2p/main --every 25 \
    --games 2048 --device cuda --out heldout-main.npz
uv run python tools/critic_heldout.py report heldout-main.npz --json heldout-main.json
```)

=== Head-only arm, and main vs head #chip("parked") <exp:critic-head>
#meta(
  owner: "marco",
  config: "configs/experiments/critic-refit-2p.yaml",
  wandb: (),
  records: (),
)
#tldr[Arm `head` refits only the value head on the shared trunk (74k parameters). It was to run only if a `main` row passed; neither did, so it is parked.]

*Plan.* Same config and decision rows as @exp:critic-refit, one trunk instead of two. Launch with `catanrl run configs/experiments/critic-refit-2p.yaml --only head`, only after a positive `main` row.
