#import "../macros.typ": *

= llm env and eval <sec:llm>

can catan serve as an env and an eval for language models? marco wants frontier models evaluated as players, and eventually a small model trained with rl on the game. the eval half is built, audited and tested with a free stub model, but no real model has played: it waits on a model list and a budget. rl is not possible on this box's 8 GB gpu.

== frontier models as players

=== llm players in the harness #chip("todo") <exp:llm-eval>
#meta(
  dates: "2026-10-17, fixed 2026-10-29",
  dates-label: "built",
  owner: "marco",
  config: "configs/experiments/llm-eval.yaml",
  wandb: (),
  records: (),
)
#tldr[an `LLM` agent plays through the catanatron harness with catanllm's five prompting strategies, on public information only. an audit on 29 october fixed what the model is told (wrong rules, unreadable robber targets, unannotated placements, no history) and added a hard spending cap. a 20-cell sweep is ready; it waits on marco's choice of models, games and budget.]

*setup.* `catanllm` (the cs 159 project) already benchmarks llm players through openrouter against catanatron bots with a few prompting strategies. v2 ports it as an agent spec, `LLM:model=<openrouter id>:strategy=plan|basic|strategy|search|rules:thinking=true`, that plays through the catanatron harness next to `POLICY` and F. the prompt holds public information only (the original renderer leaked every seat's dev cards; a test pins that). a stub model plays the same games for free, so the whole path is tested without a key. real-model records are marked non-replayable, so a report refresh can't quietly spend money. v2’s game server (protocol 5: committed seeds, signed moves, verifiable transcripts) can also seat an llm client against our bots fairly.

*audit (29 october).* marco called the port "lowk vibe coded trash from like over a year ago", so it was gone through before any paid run. what the model saw was wrong or unusable in places, each now fixed with a test (73 in `tests/test_llm_agent.py`; 41 of them fail on the old code):
- *rules.* the prompts said discard above 7 cards and described player-to-player trading. base-2p discards above 9 and only offers bank and port trades. a rules header is now generated from the game itself (players, target, discard limit, robber, trading, deck, bank, pieces) and overrides the strategy texts.
- *the board.* robber moves named cube coordinates while the state named tiles by id; everything is by tile id now. a placement said "settle node 23" with nothing about what node 23 produces; every settlement, city and road action now carries its tiles, numbers, pips and port. the "empty nodes" and "free edges" lists ignored the distance rule and listed the whole board; they are replaced by the open spots within 2 roads of the model's network.
- *information.* in 2p every resource movement is public, so the opponent's exact hand is shown (our fair nets see it too); a test tracks it from public events through random games, steals and monopolies included. unplayed dev cards stay hidden. the model now gets the last 30 events of the game, redacted (a bought card is named only to its buyer, a stolen card only to the thief and the victim), and a line saying what it is deciding (setup, discard, robber after a 7 or a knight, free roads).
- *prompt size.* every past note was resent every move, so prompts grew with the square of the game length. only the last 8 notes are shown now, cut at 300 characters, and the board moved into the system prompt, which anthropic models cache.
- *the api.* temperature was always sent, which reasoning models reject with thinking on; it is sent only with thinking off. a reply cut off at the token limit could be read as an answer (often "end turn"); it is retried and counted instead. `max_tokens` rose to 16384 with half for reasoning (the old 8192 with three quarters left 2048 for the answer). provider errors inside a 200 response are retried, then raised.
- *cost safety.* each call's dollar cost (openrouter's `usage.cost`) is recorded per game. a real model refuses to play without `CATANRL_LLM_BUDGET` (dollars), checked before every call; reaching it stops the eval like an error, keeping finished games. `CATANRL_LLM_LEDGER` makes the cap span the sweep's cells.
- *the opponent.* the old "policy" opponent was `POLICY` over `puffer-graph2/full`, which reads the llm's hidden dev cards. it is now `BELIEF:ckpt=ppo-2p-fr3/full@e4350:worlds=16` on cpu (about 2.6 s a game vs F), which samples them from what its seat saw. caveat: fr3 trained on base-2p-fr and plays base-2p's rules here, since the catanatron harness can't roll balanced dice; `puffer-graph2` (trained on base-2p) is the alternative.

none of this touches the native code id (`caaf61e29bb61098`); the catanatron harness id moved to `ac6c0cffdf10a7b7`.

*plan.* `llm-eval.yaml` is a 20-cell sweep: the stub plus four placeholder models (`anthropic/claude-opus-5.5`, `anthropic/claude-sonnet-5.5`, `openai/gpt-6.1-sol`, `google/gemini-4-argon`) × plan or basic prompting × against F or against the fr3 belief net, on the first `params.boards` `heldout-2p` boards from both seats. the boards (20, a placeholder) and the budget are marco's call. `catanrl llm-cost` prices it from stub games: a prompt is now about 3.9k tokens with plan (2.6k of it the cacheable system prompt) and grows linearly. at 80 prompted decisions a game (what a competent player makes) and 1,500 output tokens a decision, a plan game costs about \$1.8 at \$2/\$10 prices (sonnet-5.5, gpt-6.1-sol, gemini-4-argon) and \$3.7 at opus-5.5's \$4/\$20, before caching. output dominates, and a model using its whole 8k reasoning budget costs 2–4× that, so 200 games of one model run roughly \$350–1,500 at \$2/\$10 prices and twice that with opus.

#repro(```sh
catanrl llm-cost --games 5 --strategies plan,basic --price opus-5.5=4,20 --price sonnet-5.5=2,10   # the budget, free
catanrl run configs/experiments/llm-eval.yaml --smoke     # every cell with the stub, free
OPENROUTER_API_KEY=... CATANRL_LLM_BUDGET=<dollars> CATANRL_LLM_LEDGER=runs/llm-eval.spend.jsonl \
  catanrl run configs/experiments/llm-eval.yaml            # every cell, real calls cost money, capped
```)

== catan as an rl env for language models

=== rl a small language model on catan #chip("todo") <exp:llm-rl>
#meta(owner: "marco", wandb: (), records: ())
#tldr[the other half of marco's idea: train a small language model with rl in this env (for example as a harbor task). deferred: it doesn't fit on this box.]

*status.* deferred. rl on a language model needs more than the 8 GB gpu this repo runs on; the eval above is the cheap half.

=== an llm env for building catan bots #chip("todo") <exp:llm-bot-builder>
#meta(owner: "marco", wandb: (), records: ())
#tldr[turn "make a strong catan bot" itself into an llm rl env and eval. marco expects the next frontier models could get very good at building better catan bots.]

*status.* an idea; nothing is built.
