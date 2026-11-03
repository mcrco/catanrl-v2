#import "../macros.typ": *

= How Markov a decision is <sec:markov>

Does the bot gain from remembering the game instead of seeing only the latest state? Marco's scope runs from an LSTM over the seat's own decisions (cheap) to a transformer over the whole history. The transformer costs orders of magnitude more compute, but in his view it is the only way to a "perfect" bot: it could learn, for example, that a seat which didn't play a knight while its best tile was blocked probably holds none. So far the LSTM ties the Markov graph net in 2p and beats it by 4.4 points head to head in 4p, and its memory is not reading the other hands. The transformer is built and queued (@exp:markov-transformer).

== An LSTM over the seat's own decisions

=== The LSTM policy, and the 2p test #chip("done") <exp:markov-lstm-2p>
#meta(
  dates: "2026-10-18 (built 2026-10-17)",
  owner: "marco",
  config: "configs/experiments/arch-ablation.yaml",
  wandb: (("arch-ablation-lstm-1c936a7f", "lstm"), ("arch-ablation-graph-s-7e2d7415", "graph-s")),
  records: ("1cd2de87", "073da498"),
)
#tldr[The cheap end of Marco's scope: the graph net plus an LSTM over the seat's own decisions. At 100M steps in 2p it ties the Markov graph net (51.0% head to head; 90.3% against F, vs. 90.7%): memory buys nothing measurable there.]

*Question.* Does memory over the seat's own history help when the row already shows the latest state?

#lead[Setup.]
- *Preset `lstm`.* The `graph-s` trunk, an LSTM cell over its pooled embedding stepped at each of the seat's own decisions, and the memory fed into both heads; 0.86M parameters. The state persists across env steps (waiting rows keep it) and resets at game ends.
- *Training.* PPO replays the recurrence over segments × horizon from the stored rollout states (CleanRL's PPO-LSTM shape). A 5-epoch run at learning rate 0 reproduces the rollout's logits exactly (KL 0, clip fraction 0).
- *Playing.* The `POLICY` agent keeps one memory stream per seat, so records don't depend on `num_envs`. MCTS, `BELIEF` and the server bridge refuse the net, because a sampled world has no decision history.
- *Test.* The LSTM cell of the formal architecture ablation (@sec:arch): the same recipe as `graph-s`, 100M agent steps, one seed, `private` level, 5,000 `heldout-2p` games a row.

*Result.* Against F the LSTM scores 90.3 #ci[89.4][91.1] (`1cd2de87`) and `graph-s` 90.7 #ci[89.9][91.5]. Head to head the LSTM takes 51.0 #ci[49.7][52.4] (`073da498`). It trains at 14.8k steps/s (1.9 h) against `graph-s`'s 17.6k; its `select-2p` best is e700 at 92.0. The full row is in the architecture table (@sec:arch).

*Reading.* At 100M steps in 2p, memory over the seat's own row history buys nothing measurable: the row already carries the played cards, the hand sizes and the robber history. Separating the two would take the full 500M-step budget and more seeds.

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
#tldr[4p hides more than 2p, so the same test at 4p. The LSTM beats the same net without memory by 4.4 points head to head over both slot orders (8,000 games, one seed). Against three F the two are inside the noise.]

*Question.* Does memory pay at 4p, where the row is missing a steal between two other seats, who played what when, and what a seat chose not to do?

#lead[Setup.]
- *LSTM.* The `lstm` preset on `ppo-4p`'s recipe at `aux-hidden-4p`'s budget: 750 epochs, 49.2M decisions, the lambda critic.
- *Markov control.* `aux-hidden-4p/plain@e600`, the `graph-s` cell with the same recipe, budget, seed and critic (@sec:hidden).
- *Head to head.* One table with two F, played in both slot orders. `[A, B, F, F]` is not exchangeable: F robs the seat after it, so slot 0 loses about 10 points (@sec:hidden). The paired reading averages the two orders.
- *Games.* 4,000 a row.

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
  caption: [The LSTM against the Markov net at 4p, `heldout-4p`, shares in %. The slot after the F costs about 10 points (38.0 against 48.0 by slot); the LSTM pays it in one order and the Markov net in the other.],
)

- *Select curves.* Within one half-width of each other at all fifteen points. The LSTM's best is its last (e750 at 65.2), and it was still climbing (62.3, 61.8, 64.4, 65.2 from e600); the Markov net's curve has been flat since e550.
- *Critic.* The lambda critic calibrates on the recurrent net as on the plain one: seat sum 1.59 at e21, 1.00 by e250, mean 1.000 over e300–750.
- *Cost.* 22.3k steps/s, 27% slower than `graph-s`'s 30.5k; 2.53 h; peak 5.2 GiB.

*Reading.* What bought nothing in 2p (51.0) buys four points at 4p against the same net without memory, on one seed. The three-F row and the select curves don't separate them: the gain shows against the Markov net, not against F. This takes the transformer off the shelf. The next question is what the memory carries (@exp:markov-lstm-aux-4p).

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
#tldr[Marco's test: put the hidden-feature head on the LSTM's memory and see whether it predicts the other hands better than the same head on the Markov net. It doesn't (resources R² 0.47–0.55, against 0.48–0.59): the memory's four points are not hand reading.]

*Question.* Does the LSTM's memory infer the other hands from activity (what a seat built, what it chose not to build, who stole from whom), which the row's tracker, a card counter, can't encode? That is its expected edge at 4p; if it does, a hand-regression head on the memory should beat the same head on the Markov net.

#lead[Setup.]
- *Arm.* `lstm-4p` plus the hidden-feature head of @sec:hidden (coefficient 0.3, the `full` players block as the target) on the memory-injected embeddings. Nothing but the head reads the block, so the checkpoint plays as a fair `POLICY`.
- *Reference.* `aux-hidden-4p/aux`, the same head on the Markov `graph-s` trunk at the same budget.
- *Metric.* R² per feature over the last 50 epochs, from the logged `aux/mse_*` and `aux/var_*` (R² = 1 − mse / var, on W&B).
- *Build and cost.* The recurrent aux path was built for this run (`tests/test_lstm_aux.py`, config schema 4). 2.58 h at 21.8k steps/s, peak 5.4 GiB.

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
  caption: [What the hidden-feature head recovers about the other seats' hands, on the Markov net (`aux-hidden-4p/aux`) and on the LSTM's memory (`lstm-aux-4p`), last 50 epochs.],
)

The picture is the same at matched epochs 100, 300 and 500, so it is not a matter of training time. The policy takes 66.2 #ci[64.7][67.6] against three F (`722b7168`; `lstm-4p` took 65.5), with its select best at e750 (68.4). Head to head against `lstm-4p/main@e750`, with two F and both slot orders (`561c7ee2`, `f691b7fc`, 8,000 games), it takes 43.8 against 42.3: paired +1.5 pp #ci[−0.4][3.5].

*Reading.* The memory adds no hand information beyond the tracker row: whatever the recurrent state carries, the head can't extract more of the other seats' cards from it. The aux-shaped memory doesn't move the policy either, just as the head didn't move the Markov policy. Only one coefficient was tried (0.3, set on the 2p graph net, not re-measured for the LSTM). What the memory does carry is open: the order of events, or who did what to whom.

#repro(```sh
catanrl run configs/experiments/aux-hidden-4p.yaml --only aux   # the Markov head's R² reference
# After lstm-4p, whose checkpoint the finals play against:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/lstm-aux-4p.yaml
```)

== The whole history

=== A transformer over the whole history #chip("queued") <exp:markov-transformer>
#meta(dates: "2026-11-02", dates-label: "built", owner: "marco", config: "configs/experiments/transformer-4p.yaml", wandb: (), records: ())
#tldr[The expensive end of Marco's scope: the graph trunk plus a causal transformer over every one of the seat's own decisions since the start of the game, sized to the LSTM's parameter count, on `lstm-4p`'s recipe. About 3 h on the GPU; queued after the search-depth sweep.]

#lead[Built.]
- *Preset `transformer`.* The `graph-s` trunk's pooled embedding becomes one token per own decision: 2 pre-norm layers, width 128, 4 heads, rotary positions over the seat's decision index. The output at the current token feeds both heads, as the LSTM's memory does.
- *Size.* The feed-forward is 128 wide, so the net has 855,780 parameters against the LSTM's 857,956: the comparison is about memory, not capacity.
- *Context.* The whole game (a `context` cap exists as a fallback).
- *Gradients.* Tokens from earlier rollout epochs are stored, so gradients don't reach the trunk through them: the same cut as the LSTM's stored states. An update at learning rate 0 reproduces the rollout (clip fraction 0, KL at float rounding).
- *Playing.* Eval streams are per seat, independent of `num_envs`, threads and game order. MCTS, `BELIEF` and the server refuse it, as they refuse the LSTM.

#lead[Run.]
- *Recipe.* `lstm-4p` with only the model changed: 750 epochs, 49.2M decisions, seed 42, the lambda critic.
- *Finals*, 4,000 `heldout-4p` games a row: against three F, and head to head against the Markov control (`aux-hidden-4p/plain@best`) and against `lstm-4p/main@best`, each in both slot orders (a paired gap of about 2.8 points is detectable).
- *Cost.* Estimated 3 h and 6 GB peak on the GPU, from a CPU ratio of 0.83× the LSTM; a GPU smoke comes first.

*Not built: a token per env step.* Tokens for every seat's move, not only the seat's own, would let the memory see who did what to whom, the candidate for what the LSTM's memory carries. The engine plays F's moves and forced moves inside its step, so evaluating it fairly against F needs an engine mode that hands the policy a row after every action, and training would cost 2–3× more.

#lead[Open for Marco.]
+ Whether to build the env-step tokens above. Marco decides after this decision-token run.

#repro(```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True catanrl run configs/experiments/transformer-4p.yaml
```)
