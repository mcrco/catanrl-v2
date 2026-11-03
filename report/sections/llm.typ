#import "../macros.typ": *

= LLM env and eval <sec:llm>

Marco wants frontier language models evaluated as Catan players, and eventually a small model trained with RL on the game.

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
#tldr[An `LLM` agent with catanllm's five prompting strategies plays through the Catanatron harness on public information. An audit fixed what the model is told and added a hard spending cap. No real model has played: a 20-cell sweep waits on Marco's choice of models, games and budget.]

*Question.* How well do frontier LLMs play Catan from public information?

#lead[Setup.]
- `catanllm` (the CS 159 project) benchmarks LLM players through OpenRouter against Catanatron bots. v2 ports it as an agent spec next to `POLICY` and F: `LLM:model=<openrouter id>:strategy=plan|basic|strategy|search|rules:thinking=true`.
- Public information only; the original renderer leaked every seat's dev cards, now pinned by a test.
- A free stub model tests the whole path without a key. Real-model records are non-replayable, so a report refresh can't spend money.
- v2's game server (protocol 5, verifiable transcripts) can also seat an LLM client fairly.

*Audit.* Audited before any paid run (Marco: "lowk vibe coded trash from like over a year ago"). Each fix has a test (73 in `tests/test_llm_agent.py`; 41 fail on the old code).
- *Rules.* The prompts said discard above 7 and described player-to-player trading; `base-2p` discards above 9 and has only bank and port trades. A rules header generated from the game now overrides the strategy texts.
- *Board.* Robber targets were cube coordinates, unlike the state's tile ids; now all are tile ids. "Settle node 23" said nothing about the node; every settlement, city and road action now lists its tiles, numbers, pips and port. The open-spot lists ignored the distance rule and covered the whole board; they now hold the open spots within 2 roads of the model's network.
- *Information.* In 2p every resource movement is public, so the opponent's exact hand is shown (our fair nets see it too), tracked under test through steals and monopolies. Unplayed dev cards stay hidden. The model now gets the last 30 events, redacted (bought and stolen cards are named only to the players who saw them), and a line naming the decision at hand.
- *Prompt size.* Every past note was resent each move, so prompts grew quadratically. Now only the last 8 are shown, cut at 300 characters, and the board sits in the system prompt, which Anthropic models cache.
- *API.* Temperature, which reasoning models reject with thinking on, is now sent only with thinking off. A truncated reply could be parsed as an answer (often "end turn"); it is now retried and counted. `max_tokens` rose to 16,384, half for reasoning (the old 8,192, three quarters reasoning, left 2,048 for the answer). Provider errors inside a 200 response are retried, then raised.
- *Cost safety.* Per-game cost is recorded from OpenRouter's `usage.cost`. A real model will not play without `CATANRL_LLM_BUDGET` (dollars), checked before every call; reaching it stops the eval, keeping finished games. `CATANRL_LLM_LEDGER` makes the cap span the sweep's cells.
- *Opponent.* `POLICY` over `puffer-graph2/full` read the LLM's hidden dev cards; the opponent is now `BELIEF:ckpt=ppo-2p-fr3/full@e4350:worlds=16` on CPU (about 2.6 s a game vs F), which samples them. Caveat: fr3 trained on `base-2p-fr` but plays `base-2p` here, since the Catanatron harness can't roll balanced dice; `puffer-graph2` (trained on `base-2p`) is the alternative.

The native code id is untouched (`caaf61e29bb61098`); the Catanatron harness id moved to `ac6c0cffdf10a7b7`.

*Plan.* `llm-eval.yaml` crosses five models (the stub plus placeholders `anthropic/claude-opus-5.5`, `anthropic/claude-sonnet-5.5`, `openai/gpt-6.1-sol`, `google/gemini-4-argon`) with plan or basic prompting and F or the fr3 belief net: 20 cells, each on the first `params.boards` `heldout-2p` boards from both seats. 20 boards is a placeholder.

*Cost.* `catanrl llm-cost` prices it from stub games. A plan prompt is now about 3.9k tokens (2.6k of it the cacheable system prompt) and grows linearly. At 80 prompted decisions a game (a competent player's count) and 1,500 output tokens each, a plan game costs about \$1.8 at \$2/\$10 prices (`sonnet-5.5`, `gpt-6.1-sol`, `gemini-4-argon`) and \$3.7 at `opus-5.5`'s \$4/\$20, before caching. Output dominates: a model using its whole 8k reasoning budget costs 2–4× that, so 200 games of one model run roughly \$350–1,500 at \$2/\$10 and twice that with Opus.

#repro(```sh
catanrl llm-cost --games 5 --strategies plan,basic --price opus-5.5=4,20 --price sonnet-5.5=2,10   # The budget, free
catanrl run configs/experiments/llm-eval.yaml --smoke     # Every cell with the stub, free
OPENROUTER_API_KEY=... CATANRL_LLM_BUDGET=<dollars> CATANRL_LLM_LEDGER=runs/llm-eval.spend.jsonl \
  catanrl run configs/experiments/llm-eval.yaml            # Every cell; real calls cost money, capped
```)

== Catan as an RL env for language models

=== Train a small language model on Catan with RL #chip("todo") <exp:llm-rl>
#meta(owner: "marco", wandb: (), records: ())
#tldr[The other half of Marco's idea: RL on a small language model in this env (for example as a Harbor task).]

*Status.* Deferred: it needs more than this box's 8 GB GPU.

=== An LLM env for building Catan bots #chip("todo") <exp:llm-bot-builder>
#meta(owner: "marco", wandb: (), records: ())
#tldr[Turn "make a strong Catan bot" itself into an LLM RL env and eval. Marco expects the next frontier models to get very good at it.]

*Status.* An idea; nothing is built.
