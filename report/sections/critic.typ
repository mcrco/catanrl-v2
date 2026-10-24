#import "../macros.typ": *

= refitting the critic after ppo <sec:critic>

marco's idea: during multi-agent ppo the critic chases a policy that keeps changing, so it probably never converges on the final one. freeze the policy, keep training the critic on many of the final policy's games, and mcts should get better with the more accurate critic. a second reason to expect a gain: the start's critic is discounted (v-trace targets at γ = 0.999, λ = 0.95). so it predicts something closer to 0.999^(moves left) × p(win) than p(win). that is also not the undiscounted win/loss alphazero's value head trains on. so far a cpu probe found only calibration headroom (\~0.01 nats), and the gpu refit is paused at e266 of 500 with no search result yet.

== the start's critic

=== cpu probe of the start's critic #chip("done") <exp:critic-probe>
#meta(
  dates: "2026-10-23",
  wandb: (),
  records: (),
)
#tldr[before the gpu run i scored the start's critic against real outcomes on cpu. it is discounted as expected (the two seats' values sum to 0.936) but ranks positions well. the only headroom is calibration, about 0.01 nats, which the gumbel root mostly cancels, so on this evidence alone i don't expect the refit to move the search rows.]

*question.* the first step in the plan for marco's refit: how far is the start's critic from the undiscounted outcome, and how much can a refit gain on held-out games?

*setup.* 5,120 raw-policy games of `puffer-graph2/public@e3600` on its own training boards (827k decisions). every value is scored against the game's undiscounted outcome.

*result.*
#figure(
  btable(
    columns: (1fr, auto),
    align: (left, right),
    header: ([measure], [value]),
    [log loss vs the undiscounted outcome], [0.370 (constant guess 0.682)],
    [the same by game stage], [0.68 in the first tenth of a game → 0.11 in the last],
    [the two seats' values summed], [0.936 (1 if undiscounted)],
    [calibration], [a predicted 0.74 wins 83% of the time],
    [held-out gain: each seat's value ÷ the two seats' sum, no training], [0.010 #ci[0.008][0.011] nats],
    [held-out gain: refit of the value head on the frozen trunk (the best fit)], [0.012 nats],
    [auc within the same stage of the game], [0.894 for all],
  ),
  caption: [the start's critic (`puffer-graph2/public@e3600`) on 5,120 raw-policy games.],
)

*reading.* it is discounted as expected: v ≈ 0.999^(decisions left) × p(win) fits where v ≈ p(win) does not, so it is under-confident about wins. the only headroom is calibration, and neither fix ranks positions any better. the gumbel root rescales q to its own range, which cancels most of a calibration change. whether a separately trained critic trunk on \~400k games finds discrimination the value head couldn't is what the gpu run tests (@exp:critic-refit).

#repro(```sh
# not reproducible as run: a one-off cpu probe whose script is not in the repo.
# tools/critic_heldout.py, the gpu run's held-out scorer, computes the same measures
# (log loss against the outcome, auc within turn bins, the seat-sum fix) for any checkpoint.
```)

== the refit

=== critic-refit-2p: frozen policy, refit critic, judged in the search #chip("running") <exp:critic-refit>
#meta(
  dates: "2026-10-23 09:57–13:40 (e0–e266), resumed 2026-10-24 00:54",
  owner: "marco",
  config: "configs/experiments/critic-refit-2p.yaml",
  wandb: (("critic-refit-2p-main-790938cd", "critic-refit-2p/main"), ("az-2p-2048-soft-main-8105fbb1", "az-2p-2048-soft (baseline 856a0421)"), ("gumbel-edge-2p-main-efda50ac", "gumbel-edge-2p, eval only (baseline 831eea84)")),
  records: ("576848d8", "856a0421", "831eea84"),
)
#tldr[the policy stays bit-identical while a separate copy of the critic trains on the outcomes of \~400k of the frozen policy's games. paused at e266 for the canopy search row, it resumed on 2026-10-24. the critic's seat sum went from 0.936 to 1.000 by e50; the search rows at e500 decide the result.]

*question.* with only the critic different, does mcts with a refit critic beat the raw start more often than mcts with the start's own critic?

*setup.*
- *frozen policy.* `train.freeze: [actor]` takes the policy out of the optimizer. its logits stay bit-identical through training, checked at every launch against `init_from`.
- *split trunk.* `init_from.split_shared` copies the shared trunk into separate policy and critic networks that reproduce the start exactly: identical logits, values and 512-sim searches (merged `3dc0450`). the e0 select (`576848d8`) plays the same games as the start's `856a0421` (identical games hash).
- *arm `main`.* refits the whole split critic (376k parameters) on the monte-carlo outcome (γ = 1, λ = 1) of raw-policy self-play in the start's own env. adam 3e-5 → 3e-6 over 500 epochs, \~400k games of the frozen policy at temperature 1, of which \~63% reach the loss. 256-step segments, so about two thirds of the targets are real outcomes and the rest bootstrap. \~8–14 gpu hours, not yet measured with two trunks.
- *held-out scoring.* a fixed set of 2,048 held-out games on boards outside the training pool (`tools/critic_heldout.py`) scores every checkpoint, paired, against two calibration-only fixes of the start. the run counts as converged if e500 is within 0.002 nats of e450.
- *decision rule, fixed before launch.* e500 inside mcts against the raw start, in two rows, each paired with the start's weights in the same search on the same games (mcnemar, holm at 0.05 over the two). at \~1.6–1.8 pp paired se, 80% power needs an effect of \~5 pp, so a null reads "no effect of 5 pp or more".

#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, left),
    header: ([decision row (e500 in mcts vs the raw start)], [baseline: start's weights], [baseline record]),
    [512 sims, the 1,000 games of `select-2p-500`], [57.4 #ci[54.3][60.4]], [`856a0421`],
    [2048 sims, `heldout-2p` (500 boards, both seat orders)], [62.4 #ci[59.4][65.3]], [`831eea84`],
  ),
  caption: [the two rows that decide `critic-refit-2p`, and the rows they pair with.],
)

*result.* pending. the main arm was paused at e266 of 500 to free the gpu for the canopy search row (@sec:canopy; the two don't fit in 8 gb together) and resumed on 2026-10-24 00:54. by e50 the critic's two seats' values summed to 1.000, against the start's 0.936.

#repro(```sh
catanrl run configs/experiments/critic-refit-2p.yaml --only main   # resumes from e266 in the same runs dir
uv run python tools/critic_heldout.py play --run critic-refit-2p/main --every 25 \
    --games 2048 --device cuda --out heldout-main.npz
uv run python tools/critic_heldout.py report heldout-main.npz --json heldout-main.json
```)

=== head-only arm, and main vs head #chip("todo") <exp:critic-head>
#meta(
  owner: "marco",
  config: "configs/experiments/critic-refit-2p.yaml",
  wandb: (),
  records: (),
)
#tldr[arm `head` refits only the value head on the shared trunk (74k parameters). it runs only if a main row passes; main vs head, paired, then tells a more accurate critic from a merely rescaled one.]

*plan.* same config, same decision rows as @exp:critic-refit, one trunk instead of two. launch with `catanrl run configs/experiments/critic-refit-2p.yaml --only head`, only after a positive main row.
