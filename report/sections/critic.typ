#import "../macros.typ": *

= Refitting the critic after PPO <sec:critic>

Marco's idea: during multi-agent PPO the critic chases a moving policy, so it probably never converges on the final one. Freeze the policy, keep training the critic on the final policy's games, and MCTS should improve with the more accurate critic. A second expected gain: the start's critic is discounted (V-trace at γ = 0.999, λ = 0.95), unlike the undiscounted win/loss AlphaZero trains on.

The result is null: the refit calibrated the critic but made neither its ranking nor the search better (@exp:critic-refit). The idea fails its pre-registered rule, so the value-head-only arm was not run.

== The start's critic

=== CPU probe of the start's critic #chip("done") <exp:critic-probe>
#meta(
  dates: "2026-10-23",
  wandb: (),
  records: (),
)
#tldr[Claude's CPU check before the GPU run: the start's critic is discounted (the two seats' values sum to 0.936) but ranks positions well. The only headroom is calibration, about 0.01 nats, which the Gumbel root mostly cancels, so Claude did not expect the refit to move the search rows.]

*Question.* Step one of Marco's plan: how far is the start's critic from the undiscounted outcome, and what can a refit gain?

*Setup.* 5,120 raw-policy games of `puffer-graph2/public@e3600` on its training boards (827k decisions), each value scored against the undiscounted outcome.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto),
    align: (left, right),
    header: ([Measure], [Value]),
    [Log loss vs the undiscounted outcome], [0.370 (constant guess 0.682)],
    [The same, by game stage], [0.68 in the first tenth → 0.11 in the last],
    [The two seats' values summed], [0.936 (1 if undiscounted)],
    [Calibration], [A predicted 0.74 wins 83% of the time],
    [Held-out gain, seat-sum fix \ #sub[each seat's value ÷ the seats' sum, untrained]], [0.010 #ci[0.008][0.011] nats],
    [Held-out gain, value-head refit \ #sub[on the frozen trunk; the best fit]], [0.012 nats],
    [AUC within a game stage], [0.894 for the start and both fixes],
  ),
  caption: [The start's critic on 5,120 raw-policy games.],
)

*Reading.* v ≈ 0.999^(decisions left) × P(win) fits where v ≈ P(win) does not, so the critic is under-confident about wins. The Gumbel root's Q rescaling cancels most calibration changes. The GPU run tests whether a separate critic trunk on \~400k games ranks better (@exp:critic-refit).

#repro(```sh
# Not reproducible as run: a one-off CPU probe. tools/critic_heldout.py computes the same
# measures (log loss, AUC within turn bins, seat-sum fix) for any checkpoint.
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
#tldr[A separate critic refit on \~400k of the frozen policy's games loses the discount but ranks positions no better. Inside the search it changes nothing: +2.1 pp at 512 sims (p = 0.15), +0.1 pp at 2,048 (p = 1.0). A more accurate critic is not what limits our search.]

*Question.* Does MCTS with a refit critic beat the raw start more often than with the start's critic?

#lead[Setup.]
- *Frozen policy.* `train.freeze: [actor]` takes the policy out of the optimizer; its logits stay bit-identical, checked against `init_from` at every launch.
- *Split trunk.* `init_from.split_shared` copies the shared trunk into separate policy and critic networks that reproduce the start exactly (logits, values and 512-sim searches; merged `3dc0450`). The e0 select (`576848d8`) plays the same games as the start's `856a0421`.
- *Arm `main`.* Refits the whole split critic (376k parameters) on the Monte Carlo outcome (γ = 1, λ = 1) of the frozen policy's self-play at temperature 1 in the start's env: Adam 3e-5 → 3e-6 over 500 epochs, \~400k games, \~63% of which reach the loss. With 256-step segments about two thirds of the targets are real outcomes; the rest bootstrap. Budget: \~8–14 GPU hours.
- *Held-out scoring.* 2,048 fixed games on boards outside the training pool (`tools/critic_heldout.py`) score every checkpoint against two calibration-only fixes of the start. Converged if e500 is within 0.002 nats of e450.
- *Decision rule, fixed before launch.* e500 in MCTS against the raw start in the two rows below, each paired with the start's weights on the same games (McNemar, Holm at 0.05). At \~1.6–1.8 pp paired SE, 80% power needs \~5 pp, so a null reads "no effect of 5 pp or more".

*Result.* Paused at e266 for the Canopy search row (@sec:canopy; 8 GB holds only one); training ended at 03:00, the 2048-sim row at 10:45.

#figure(
  btable(
    columns: (1fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    header: ([Decision row], [Start's critic], [Refit (e500)], [Paired], [Won only by \ e500 / start], [McNemar p]),
    [512 sims \ #sub[`select-2p-500`, 1,000 games]], [57.4 #ci[54.3][60.4] \ #sub[`856a0421`]], [*59.5* #ci[56.4][62.5] \ #sub[`f841ca86`]], [+2.1 pp], [105 / 84], [0.15],
    [2,048 sims \ #sub[`heldout-2p`, 500 boards, \ both seat orders]], [62.4 #ci[59.4][65.3] \ #sub[`831eea84`]], [*62.5* #ci[59.5][65.5] \ #sub[`9641a3aa`]], [+0.1 pp], [110 / 109], [1.0],
  ),
  caption: [Decision rows: MCTS against the raw start with the start's or the refit critic, same games.],
)
Neither row passes Holm (smallest p 0.15 vs 0.025): `main` fails and `head` is not run.

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
  caption: [Held-out critic, scored on 330k decisions; lower is better.],
)
Paired per decision (game-clustered CIs), refit minus start is −0.0174 nats #ci[−0.0212][−0.0134], refit minus Platt only −0.0011 #ci[−0.0023][+0.0002]. Within-turn AUC did not move (0.9010 → 0.9016). Converged: e500 within 0.0001 of e450 (rule: 0.002), flat from about e100.

*Reading.* The refit fixed the discount and nothing else: by e50 the seats' values summed to 1.000 (start: 0.936), but ranking did not improve, and rescaling the start's values gets almost all of the gain free. The search only compares positions and moves, so calibration cannot help it: at 2,048 sims the two critics win and lose the same games (219 of 1,000 differ, split evenly). A better search critic must rank better.

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
#tldr[Arm `head` refits only the value head on the shared trunk (74k parameters), to run only if a `main` row passed. Neither did, so it is parked.]

*Plan.* Same config and decision rows as @exp:critic-refit, one trunk instead of two: `catanrl run configs/experiments/critic-refit-2p.yaml --only head`, only after a positive `main` row.
