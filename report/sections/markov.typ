#import "../macros.typ": *

= How Markov a decision is <sec:markov>

Does the bot gain from remembering the game? Marco's scope runs from a cheap LSTM over the seat's own decisions to a transformer over the whole history: orders of magnitude more compute, but in his view the only way to a "perfect" bot, one that could learn that a seat which didn't play a knight while its best tile was blocked probably holds none. So far the LSTM ties the Markov net in 2p, beats it by 4.4 points in 4p, and doesn't read hands. The transformer is running (@exp:markov-transformer).

== An LSTM over the seat's own decisions

=== The LSTM policy, and the 2p test #chip("done") <exp:markov-lstm-2p>
#meta(
  dates: "2026-10-18 (built 2026-10-17)",
  owner: "marco",
  config: "configs/experiments/arch-ablation.yaml",
  wandb: (("arch-ablation-lstm-1c936a7f", "lstm"), ("arch-ablation-graph-s-7e2d7415", "graph-s")),
  records: ("1cd2de87", "073da498"),
)
#tldr[The cheap end of Marco's scope: an LSTM over the seat's own decisions. At 100M steps in 2p it ties the Markov graph net, 51.0% head to head: memory buys nothing measurable.]

*Question.* Does memory of the seat's own decisions help in 2p?

#lead[Setup.]
- *Preset `lstm`.* `graph-s` plus an LSTM cell over the pooled embedding, stepped at each own decision and fed into both heads; 0.86M parameters. It resets at game ends.
- *Exactness.* PPO replays the recurrence from stored rollout states (CleanRL's PPO-LSTM shape); a 5-epoch update at learning rate 0 reproduces the rollout's logits (KL 0, clip fraction 0). Memory is per seat, so records don't depend on `num_envs`. MCTS, `BELIEF` and the server bridge refuse the net: a sampled world has no decision history.
- *Test.* The LSTM cell of the architecture ablation (@sec:arch): `graph-s`'s recipe, 100M agent steps, one seed, `private` level, 5,000 `heldout-2p` games a row.

*Result.* Against F the LSTM scores 90.3 #ci[89.4][91.1] (`1cd2de87`), `graph-s` 90.7 #ci[89.9][91.5]. Head to head the LSTM takes 51.0 #ci[49.7][52.4] (`073da498`). It trains at 14.8k steps/s (1.9 h) against `graph-s`'s 17.6k; its `select-2p` best is e700 at 92.0.

*Reading.* The row already carries the played cards, hand sizes and robber history. Separating the two would take the full 500M-step budget and more seeds.

#repro(```sh
catanrl run configs/experiments/arch-ablation.yaml --only graph-s   # the head-to-head opponent
catanrl run configs/experiments/arch-ablation.yaml --only lstm
```)

=== The LSTM at 4p #chip("done") <exp:markov-lstm-4p>
#meta(
  dates: "2026-10-19 14:18–16:58",
  config: "configs/experiments/lstm-4p.yaml",
  wandb: (("lstm-4p-main-4cae1a6d", "lstm"), ("aux-hidden-4p-plain-f07eb81c", "markov control")),
  records: ("9198cef1", "5efc75ba", "cf8368f4", "d97f69b3"),
)
#tldr[At 4p the LSTM beats the same net without memory by 4.4 points head to head over both slot orders (8,000 games, one seed); against three F the two are level.]

*Question.* Does memory pay at 4p, where the row hides more?

#lead[Setup.]
- *LSTM.* The `lstm` preset on `ppo-4p`'s recipe at `aux-hidden-4p`'s budget: 750 epochs, 49.2M decisions, the lambda critic.
- *Markov control.* `aux-hidden-4p/plain@e600`: `graph-s` with the same recipe, budget, seed and critic (@sec:hidden).
- *Head to head.* Two F at the table, 4,000 games in each slot order, averaged in the paired reading: the slot after an F loses about 10 points (@sec:hidden).

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, left),
    header: ([Table], [LSTM], [Markov `graph-s`], [Record]),
    [Against three F (parity 25)], [65.5 #ci[64.0][67.0]], [64.0 #ci[62.5][65.5]], [`9198cef1` / `d97f69b3`],
    tgroup(4)[Head to head, two F at the table],
    [LSTM in slot 0, Markov net in slot 1], [39.7 #ci[38.1][41.2]], [45.3 #ci[43.8][46.9]], [`5efc75ba`],
    [Markov net in slot 0, LSTM in slot 1], [50.7 #ci[49.2][52.3]], [36.3 #ci[34.8][37.8]], [`cf8368f4`],
    [Both orders], [*45.2*], [40.8], [Paired *+4.4 pp* #ci[2.4][6.3]],
  ),
  caption: [The LSTM against the Markov net, 4,000 `heldout-4p` games a row, shares in %. By slot, the seat after an F averages 38.0 and the other 48.0.],
)

- *Select curves* are within one half-width at all fifteen points. The LSTM peaks at its last, e750 (62.3, 61.8, 64.4, 65.2 from e600); the Markov curve is flat since e550.
- *Critic.* The lambda critic calibrates on the recurrent net: seat sum 1.59 at e21, 1.00 by e250, mean 1.000 over e300–750.
- *Cost.* 22.3k steps/s, 27% below `graph-s`'s 30.5k; 2.53 h; peak 5.2 GiB.

*Reading.* Memory that bought nothing in 2p buys four points at 4p, on one seed, and only against the Markov net. That takes the transformer off the shelf; next is what the memory carries (@exp:markov-lstm-aux-4p).

#repro(```sh
catanrl run configs/experiments/aux-hidden-4p.yaml --only plain   # the Markov control
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/lstm-4p.yaml
```)

=== Does the LSTM's memory read hands? #chip("done") <exp:markov-lstm-aux-4p>
#meta(
  dates: "2026-10-19 17:00–19:41",
  owner: "marco",
  config: "configs/experiments/lstm-aux-4p.yaml",
  wandb: (("lstm-aux-4p-main-cbbc6e05", "lstm + aux head"), ("aux-hidden-4p-aux-4dcfd020", "markov + aux head")),
  records: ("722b7168", "561c7ee2", "f691b7fc"),
)
#tldr[Marco's test: does a hidden-feature head on the LSTM's memory predict the other hands better than on the Markov net? No (resources R² 0.47–0.55, against 0.48–0.59): the four points are not hand reading.]

*Question.* Does the memory infer other hands from what seats built, skipped or stole?

#lead[Setup.]
- *Arm.* `lstm-4p` plus the hidden-feature head of @sec:hidden (coefficient 0.3, target the `full` players block) on the memory-injected embeddings. Only the head reads the block, so the checkpoint plays as a fair `POLICY`.
- *Reference.* `aux-hidden-4p/aux`: the same head on the Markov trunk, same budget.
- *Metric.* R² = 1 − mse / var per feature, last 50 epochs, from the logged `aux/mse_*` and `aux/var_*` (on W&B).
- *Cost.* A new recurrent aux path (`tests/test_lstm_aux.py`, config schema 4); 2.58 h at 21.8k steps/s, peak 5.4 GiB.

#lead[Result.]
#figure(
  btable(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    header: ([Hidden feature (R²)], [Markov head], [Memory head]),
    [Resources, range], [0.48–0.59], [0.47–0.55],
    [#h(0.8em) Wood], [0.55], [0.47],
    [#h(0.8em) Wheat], [0.59], [0.55],
    [#h(0.8em) Ore], [0.57], [0.52],
    [#h(0.8em) Brick, sheep], [0.48, 0.52], [Equal],
    [Actual VP], [0.95], [0.95],
    [VP cards], [0.65], [0.63],
    [Knights], [0.51], [0.51],
    [Playable knight], [0.42], [0.42],
    [Road building], [0.20], [0.26],
    [Playable road building], [0.19], [0.24],
    mid,
    [Head loss (lower is better)], [0.017], [0.020],
  ),
  caption: [R² of the hidden-feature head on the other seats' hands, last 50 epochs.],
)

Epochs 100, 300 and 500 agree. The policy takes 66.2 #ci[64.7][67.6] against three F (`722b7168`; `lstm-4p` took 65.5), select best e750 at 68.4. Head to head against `lstm-4p/main@e750` (two F, both orders, 8,000 games; `561c7ee2`, `f691b7fc`): 43.8 against 42.3, paired +1.5 pp #ci[−0.4][3.5].

*Reading.* The recurrent state holds no more of the other seats' cards than the tracker row, and the head doesn't move the policy. Only coefficient 0.3 (set on the 2p graph net) was tried. What the memory carries is open: the order of events, or who did what to whom.

#repro(```sh
catanrl run configs/experiments/aux-hidden-4p.yaml --only aux   # the Markov head's R² reference
# After lstm-4p, whose checkpoint the finals play against:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/lstm-aux-4p.yaml
```)

== The whole history

=== A transformer over the whole history #chip("running") <exp:markov-transformer>
#meta(dates: "2026-11-03 08:24 –", owner: "marco", config: "configs/experiments/transformer-4p.yaml", wandb: (("transformer-4p-main-6feb995a", "main"),), records: ())
#tldr[The expensive end of Marco's scope: a causal transformer over all of the seat's own decisions this game, sized to the LSTM, on `lstm-4p`'s recipe. Running since 3 November, about 3 h; it trains as fast as the LSTM.]

#lead[Built.]
- *Preset `transformer`.* One token per own decision from `graph-s`'s pooled embedding; 2 pre-norm layers, width 128, 4 heads, rotary positions; the current token's output feeds both heads. Context is the whole game (a `context` cap exists as a fallback).
- *Size.* A 128-wide feed-forward gives 855,780 parameters against the LSTM's 857,956: the comparison is about memory, not capacity.
- *Exactness.* Stored tokens from earlier rollout epochs pass no gradient to the trunk, as with the LSTM; an update at learning rate 0 reproduces the rollout (clip fraction 0, KL at float rounding). Eval streams are per seat, independent of `num_envs`, threads and game order; MCTS, `BELIEF` and the server refuse it.

#lead[Run.]
- `lstm-4p` with only the model changed: 750 epochs, 49.2M decisions, seed 42, the lambda critic.
- Finals, 4,000 `heldout-4p` games a row: against three F, and head to head against the Markov control (`aux-hidden-4p/plain@best`) and `lstm-4p/main@best` in both slot orders (a paired gap of about 2.8 points is detectable).
- GPU smoke (3 November, scratch runs, W&B off): the CUDA tests pass; at learning rate 0 the KL stays below 2 × 10#super[−10] with clip fraction 0; 60 epochs from scratch run at 21.5k decisions/s (the LSTM: 22.5k) with a 5.65 GiB peak (4.9).

#lead[Open for Marco.]
+ Tokens per env step (every seat's move) would show the memory who did what to whom, but need an engine mode that hands the policy a row after every action (the engine plays F's and forced moves inside its step) and cost 2–3× more to train. Marco decides after this run.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/transformer-4p.yaml
```)
