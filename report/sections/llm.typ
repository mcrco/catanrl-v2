#import "../macros.typ": *

= LLM env and eval <sec:llm>

Can Catan serve as an env and an eval for language models? Marco wants frontier models evaluated as players, and eventually a small model trained with RL on the game. The eval half is built, audited and tested with a free stub model, but no real model has played: it waits on a model list and a budget. RL is not possible on this box's 8 GB GPU.

== Frontier models as players

=== LLM players in the harness #chip("todo") <exp:llm-eval>
#meta(
  dates: "2026-10-17, fixed 2026-10-29",
  dates-label: "built",
  owner: "marco",
  config: "configs/experiments/llm-eval.yaml",
  wandb: (),
  records: (),
)
#tldr[An `LLM` agent plays through the Catanatron harness with catanllm's five prompting strategies, on public information only. An audit fixed what the model is told (wrong rules, unreadable robber targets, unannotated placements, no history) and added a hard spending cap. A 20-cell sweep is ready and waits on Marco's choice of models, games and budget.]

*Question.* How well do frontier language models play Catan from public information, against F and against our 2p network?

#lead[Setup.]
- `catanllm` (the CS 159 project) already benchmarks LLM players through OpenRouter against Catanatron bots, with a few prompting strategies. v2 ports it as an agent spec that plays through the Catanatron harness next to `POLICY` and F: `LLM:model=<openrouter id>:strategy=plan|basic|strategy|search|rules:thinking=true`.
- The prompt holds public information only. The original renderer leaked every seat's dev cards; a test pins that.
- A stub model plays the same games for free, so the whole path is tested without a key.
- Real-model records are marked non-replayable, so a report refresh can't quietly spend money.
- v2's game server (protocol 5: committed seeds, signed moves, verifiable transcripts) can also seat an LLM client against our bots fairly.

*Audit.* Marco called the port "lowk vibe coded trash from like over a year ago", so it was gone through before any paid run. What the model saw was wrong or unusable in places. Each fix below comes with a test: 73 in `tests/test_llm_agent.py`, 41 of which fail on the old code.
- *Rules.* The prompts said discard above 7 cards and described player-to-player trading; `base-2p` discards above 9 and offers only bank and port trades. A rules header is now generated from the game itself (players, target, discard limit, robber, trading, deck, bank, pieces) and overrides the strategy texts.
- *The board.* Robber moves named cube coordinates while the state named tiles by id; everything is by tile id now. A placement said "settle node 23" with nothing about what node 23 produces; every settlement, city and road action now carries its tiles, numbers, pips and port. The "empty nodes" and "free edges" lists ignored the distance rule and listed the whole board; they are replaced by the open spots within 2 roads of the model's network.
- *Information.* In 2p every resource movement is public, so the opponent's exact hand is shown (our fair nets see it too). A test tracks it from public events through random games, steals and monopolies included. Unplayed dev cards stay hidden. The model now also gets the last 30 events of the game, redacted (a bought card is named only to its buyer, a stolen card only to the thief and the victim), and a line saying what it is deciding (setup, discard, robber after a 7 or a knight, free roads).
- *Prompt size.* Every past note was resent every move, so prompts grew with the square of the game length. Now only the last 8 notes are shown, cut at 300 characters, and the board sits in the system prompt, which Anthropic models cache.
- *The API.* Temperature was always sent, which reasoning models reject with thinking on; it is now sent only with thinking off. A reply cut off at the token limit could be read as an answer (often "end turn"); it is now retried and counted. `max_tokens` rose to 16,384 with half for reasoning (the old 8,192 with three quarters for reasoning left 2,048 for the answer). Provider errors inside a 200 response are retried, then raised.
- *Cost safety.* Each call's dollar cost (OpenRouter's `usage.cost`) is recorded per game. A real model refuses to play without `CATANRL_LLM_BUDGET` (dollars), which is checked before every call; reaching it stops the eval like an error, keeping finished games. `CATANRL_LLM_LEDGER` makes the cap span the sweep's cells.
- *The opponent.* The old "policy" opponent was `POLICY` over `puffer-graph2/full`, which reads the LLM's hidden dev cards. It is now `BELIEF:ckpt=ppo-2p-fr3/full@e4350:worlds=16` on CPU (about 2.6 s a game vs F), which samples them from what its seat saw. Caveat: fr3 trained on `base-2p-fr` and plays `base-2p`'s rules here, since the Catanatron harness can't roll balanced dice; `puffer-graph2` (trained on `base-2p`) is the alternative.

None of this touches the native code id (`caaf61e29bb61098`); the Catanatron harness id moved to `ac6c0cffdf10a7b7`.

*Plan.* `llm-eval.yaml` is a 20-cell sweep: five models (the stub and four placeholders) × two prompting strategies (plan, basic) × two opponents (F, the fr3 belief net). Each cell plays the first `params.boards` `heldout-2p` boards from both seats. The placeholder models are `anthropic/claude-opus-5.5`, `anthropic/claude-sonnet-5.5`, `openai/gpt-6.1-sol` and `google/gemini-4-argon`. The number of boards (20, a placeholder) and the budget are Marco's call.

*Cost.* `catanrl llm-cost` prices the sweep from stub games.
- A prompt is now about 3.9k tokens with plan (2.6k of it the cacheable system prompt) and grows linearly.
- At 80 prompted decisions a game (what a competent player makes) and 1,500 output tokens a decision, a plan game costs about \$1.8 at \$2/\$10 prices (`sonnet-5.5`, `gpt-6.1-sol`, `gemini-4-argon`) and \$3.7 at `opus-5.5`'s \$4/\$20, before caching.
- Output dominates, and a model that uses its whole 8k reasoning budget costs 2–4× that. So 200 games of one model run roughly \$350–1,500 at \$2/\$10 prices, and twice that with Opus.

#repro(```sh
catanrl llm-cost --games 5 --strategies plan,basic --price opus-5.5=4,20 --price sonnet-5.5=2,10   # The budget, free
catanrl run configs/experiments/llm-eval.yaml --smoke     # Every cell with the stub, free
OPENROUTER_API_KEY=... CATANRL_LLM_BUDGET=<dollars> CATANRL_LLM_LEDGER=runs/llm-eval.spend.jsonl \
  catanrl run configs/experiments/llm-eval.yaml            # Every cell; real calls cost money, capped
```)

== Catan as an RL env for language models

=== Train a small language model on Catan with RL #chip("todo") <exp:llm-rl>
#meta(owner: "marco", wandb: (), records: ())
#tldr[The other half of Marco's idea: train a small language model with RL in this env (for example as a Harbor task). Deferred: it doesn't fit on this box.]

*Status.* Deferred. RL on a language model needs more than the 8 GB GPU this repo runs on; the eval above is the cheap half.

=== An LLM env for building Catan bots #chip("todo") <exp:llm-bot-builder>
#meta(owner: "marco", wandb: (), records: ())
#tldr[Turn "make a strong Catan bot" itself into an LLM RL env and eval. Marco expects the next frontier models could get very good at building better Catan bots.]

*Status.* An idea; nothing is built.
