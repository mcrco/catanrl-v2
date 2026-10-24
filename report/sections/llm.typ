#import "../macros.typ": *

= llm env and eval <sec:llm>

can catan serve as an env and an eval for language models? marco wants frontier models evaluated as players, and eventually a small model trained with rl on the game. the eval half is built and tested with a free stub model, but no real model has played: it waits on an api key and a budget. rl is not possible on this box's 8 GB gpu.

== frontier models as players

=== llm players in the harness #chip("todo") <exp:llm-eval>
#meta(
  dates: "2026-10-17",
  dates-label: "built",
  owner: "marco",
  config: "configs/experiments/llm-eval.yaml",
  wandb: (),
  records: (),
)
#tldr[an `LLM` agent plays through the catanatron harness with catanllm's five prompting strategies, on public information only. a 16-cell sweep is ready. a 50-game cell costs about \$16–26 as a lower bound; it needs an openrouter key and a budget.]

*setup.* `catanllm` (the cs 159 project) already benchmarks llm players through openrouter against catanatron bots with a few prompting strategies. v2 ports it as an agent spec, `LLM:model=<openrouter id>:strategy=plan|basic|strategy|search|rules:thinking=true`, that plays through the catanatron harness next to `POLICY` and F. the prompt holds public information only: opponents appear as hand sizes, dev-card counts, played cards and visible points (the original renderer leaked every seat's dev cards; a test now pins that). a stub model plays the same games for free, so the whole path is tested without a key. real-model records are marked non-replayable, so a report refresh can't quietly spend money. v2’s game server (protocol 5: committed seeds, signed moves, verifiable transcripts) can also seat an llm client against our bots fairly.

*plan.* `llm-eval.yaml` is a 16-cell sweep: four models × plan or basic prompting × against F or against `puffer-graph2/full@best`'s policy, on the first 20 `heldout-2p` boards from both seats. the model ids are placeholders. `catanrl llm-cost` prices it from stub games: a 2p game to 15 points prompts the model about 20 times at 2–8k tokens a prompt, so a 50-game cell costs about \$16–26 at gpt-5 or sonnet-4.5 prices, as a lower bound. a model that plays well lasts longer and is prompted 70–90 times a game, so budget 3–4× that. no real-model run yet, so no records and no wandb run.

#repro(```sh
catanrl llm-cost --games 5 --strategies plan,basic --price gpt-5=1.25,10   # the budget, free
catanrl run configs/experiments/llm-eval.yaml --smoke                      # every cell with stub models, free
OPENROUTER_API_KEY=... catanrl run configs/experiments/llm-eval.yaml       # every cell, real calls cost money
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
