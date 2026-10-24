#import "../macros.typ": *

= how markov a decision is <sec:markov>

does the bot gain from remembering the game instead of seeing only the latest state? marco's scope runs from an lstm over the seat's own decisions (cheap) to a transformer over the whole history. the transformer costs orders of magnitude more compute, but in his view it is the only way to a "perfect" bot. it could learn, for example, that a seat which didn't play a knight while its best tile was blocked probably holds none. so far the lstm ties the markov graph net in 2p and beats it by 4.4 points head to head in 4p, and its memory is not reading the other hands. the transformer has not started.

== an lstm over the seat's own decisions

=== the lstm policy, and the 2p test #chip("done") <exp:markov-lstm-2p>
#meta(
  dates: "2026-10-18 (built 2026-10-17)",
  owner: "marco",
  config: "configs/experiments/arch-ablation.yaml",
  wandb: (("arch-ablation-lstm-1c936a7f", "lstm"), ("arch-ablation-graph-s-7e2d7415", "graph-s")),
  records: ("1cd2de87", "073da498"),
)
#tldr[the cheap end of marco's scope: the graph net plus an lstm over the seat's own decisions. at 100M steps in 2p it ties the markov graph net (51.0 head to head, 90.3 against its 90.7 vs F): memory buys nothing measurable there.]

*question.* does memory over the seat's own history help when the row already shows the latest state?

*setup.* preset `lstm`: the `graph-s` trunk, an lstm cell over its pooled embedding stepped at each of the seat's own decisions, and the memory fed into both heads; 0.86M parameters. the state persists across env steps (waiting rows keep it) and resets at game ends. ppo replays the recurrence over segments × horizon from the stored rollout states (cleanrl's ppo-lstm shape); a 5-epoch run at learning rate 0 reproduces the rollout's logits exactly (kl 0, clip fraction 0). the `POLICY` agent keeps one memory stream per seat, so records don't depend on `num_envs`. mcts, `BELIEF` and the server bridge refuse it, because a sampled world has no decision history. the test is the lstm cell of the formal arch ablation (@sec:arch): the same recipe as `graph-s`, 100M agent steps, one seed, `private` level, 5,000 `heldout-2p` games a row.

*result.* against F, the lstm takes 90.3 #ci[89.4][91.1] (`1cd2de87`) and `graph-s` 90.7 #ci[89.9][91.5]. head to head, the lstm takes 51.0 #ci[49.7][52.4] (`073da498`). it trains at 14.8k steps/s against `graph-s`'s 17.6k (1.9 h); its select-2p best is e700 at 92.0. the full row is in the arch table (@sec:arch).

*reading.* at 100M steps in 2p, memory over the seat's own row history buys nothing measurable. the row already carries the played cards, the hand sizes and the robber history. separating the two would take the full 500M-step budget and more seeds.

#repro(```sh
catanrl run configs/experiments/arch-ablation.yaml --only graph-s   # the head-to-head opponent
catanrl run configs/experiments/arch-ablation.yaml --only lstm
```)

=== the lstm at 4p #chip("done") <exp:markov-lstm-4p>
#meta(
  dates: "2026-10-19 14:18–16:58",
  config: "configs/experiments/lstm-4p.yaml",
  wandb: (("lstm-4p-main-4cae1a6d", "lstm"), ("aux-hidden-4p-plain-f07eb81c", "markov control")),
  records: ("9198cef1", "5efc75ba", "cf8368f4", "d97f69b3"),
)
#tldr[4p hides more than 2p, so the same test at 4p. the lstm beats the same net without memory by 4.4 points head to head over both slot orders (8,000 games, one seed). against three F the two are inside the noise.]

*question.* at 4p the row is missing a steal between two other seats, who played what when, and what a seat chose not to do. does memory pay there?

*setup.* the `lstm` preset on `ppo-4p`'s recipe at `aux-hidden-4p`'s budget: 750 epochs, 49.2M decisions, the lambda critic. the markov control is `aux-hidden-4p/plain@e600`, the `graph-s` cell with the same recipe, budget, seed and critic (@sec:hidden). the head to head is one table with two F, played in both slot orders. `[A, B, F, F]` is not exchangeable: F robs the seat after it, so slot 0 loses about 10 points (@sec:hidden). the paired reading averages the two orders. 4,000 games a row.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, left),
    header: ([table], [lstm], [markov `graph-s`], [record]),
    [three F (parity 25)], [65.5 #ci[64.0][67.0]], [64.0 #ci[62.5][65.5]], [`9198cef1` / `d97f69b3`],
    mid,
    [lstm in slot 0, markov in slot 1, two F], [39.7 #ci[38.1][41.2]], [45.3 #ci[43.8][46.9]], [`5efc75ba`],
    [markov in slot 0, lstm in slot 1, two F], [50.7 #ci[49.2][52.3]], [36.3 #ci[34.8][37.8]], [`cf8368f4`],
    [both orders], [*45.2*], [40.8], [paired *+4.4 pp* #ci[2.4][6.3]],
  ),
  caption: [the lstm against the markov net at 4p, `heldout-4p`, shares in %. the slot after the F costs about 10 points (38.0 against 48.0 by slot); the lstm pays it in one order and the markov net in the other.],
)

the select curves sit within one half-width of each other at all fifteen points. the lstm's best is its last (e750 at 65.2), and it was still climbing (62.3, 61.8, 64.4, 65.2 from e600); the markov net's curve has been flat since e550. the lambda critic calibrates on the recurrent net as on the plain one: seat sum 1.59 at e21, 1.00 by e250, mean 1.000 over e300–750. cost: 22.3k steps/s, 27% slower than `graph-s`'s 30.5k; 2.53 h; peak 5.2 GiB.

*reading.* what bought nothing in 2p (51.0) buys four points at 4p against the same net without memory. the three-F row and the select curves don't separate them: the gain shows against the markov net, not against F. one seed. this takes the transformer off the shelf. the next question is what the memory carries (@exp:markov-lstm-aux-4p).

#repro(```sh
catanrl run configs/experiments/aux-hidden-4p.yaml --only plain   # the markov control
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/lstm-4p.yaml
```)

=== does the lstm's memory read hands? #chip("done") <exp:markov-lstm-aux-4p>
#meta(
  dates: "2026-10-19 17:00–19:41",
  owner: "marco",
  config: "configs/experiments/lstm-aux-4p.yaml",
  wandb: (("lstm-aux-4p-main-cbbc6e05", "lstm + aux head"), ("aux-hidden-4p-aux-4dcfd020", "markov + aux head")),
  records: ("722b7168", "561c7ee2", "f691b7fc"),
)
#tldr[marco's test: put the hidden-feature head on the lstm's memory and see whether it predicts the other hands better than the same head on the markov net. it doesn't (resources R² 0.47–0.55 against 0.48–0.59). the memory's four points are not hand reading.]

*question.* the lstm's expected edge at 4p is inferring the other hands from activity: what a seat built, what it chose not to build, who stole from whom. the row's tracker, a card counter, can't encode that. if the memory does it, a hand-regression head on the memory should beat the same head on the markov net.

*setup.* `lstm-4p` plus the hidden-feature head of @sec:hidden (coefficient 0.3, the `full` players block as the target) on the memory-injected embeddings. the reference is `aux-hidden-4p/aux`, the same head on the markov `graph-s` trunk at the same budget. R² per feature is over the last 50 epochs, from the logged `aux/mse_*` and `aux/var_*` (R² = 1 − mse / var, on wandb). nothing but the head reads the block, so the checkpoint plays as a fair `POLICY`. the recurrent aux path was built for this run (`tests/test_lstm_aux.py`, config schema 4). 2.58 h at 21.8k steps/s, peak 5.4 GiB.

*result.*
#figure(
  btable(
    columns: (auto, auto, auto),
    align: (left, right, right),
    header: ([hidden feature (R²)], [markov head], [memory head]),
    [resources, range], [0.48–0.59], [0.47–0.55],
    [#h(0.8em) wood], [0.55], [0.47],
    [#h(0.8em) wheat], [0.59], [0.55],
    [#h(0.8em) ore], [0.57], [0.52],
    [#h(0.8em) brick, sheep], [0.48, 0.52], [equal],
    [actual vp], [0.95], [0.95],
    [vp cards], [0.65], [0.63],
    [knights], [0.51], [0.51],
    [playable knight], [0.42], [0.42],
    [road building], [0.20], [0.26],
    [playable road building], [0.19], [0.24],
    mid,
    [head loss (lower is better)], [0.017], [0.020],
  ),
  caption: [what the hidden-feature head recovers about the other seats' hands, on the markov net (`aux-hidden-4p/aux`) and on the lstm's memory (`lstm-aux-4p`), last 50 epochs.],
)

the picture is the same at matched epochs 100, 300 and 500, so it is not a matter of training time. the policy takes 66.2 #ci[64.7][67.6] against three F (`722b7168`; `lstm-4p` took 65.5), select best e750 at 68.4. head to head against `lstm-4p/main@e750`, with two F and both slot orders (`561c7ee2`, `f691b7fc`, 8,000 games): 43.8 against 42.3, paired +1.5 pp #ci[−0.4][3.5].

*reading.* the memory adds no hand information beyond the tracker row: whatever the recurrent state carries, the head can't extract more of the other seats' cards from it. the aux-shaped memory doesn't move the policy either, just as the head didn't move the markov policy. one coefficient was tried (0.3, set on the 2p graph net, not re-measured for the lstm). what the memory does carry is open: the order of events, or who did what to whom.

#repro(```sh
catanrl run configs/experiments/aux-hidden-4p.yaml --only aux   # the markov head's R² reference
# after lstm-4p, whose checkpoint the finals play against:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/lstm-aux-4p.yaml
```)

== the whole history

=== a transformer over the whole history #chip("todo") <exp:markov-transformer>
#meta(owner: "marco", wandb: (), records: ())
#tldr[the expensive end of marco's scope. it was parked after the 2p tie; the 4p lstm's +4.4 pp takes it off the shelf. not started.]

*status.* a sequential model over the entire game history costs orders of magnitude more compute than the markov net. the plan was to build it only if the lstm beat the markov net, and at 4p it did. the question to answer first is what the memory carries, since it isn't hands: the candidates are the order of events and who did what to whom.
