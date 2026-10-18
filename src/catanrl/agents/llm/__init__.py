"""The ``LLM`` agent: a language model playing through the Catanatron harness.

See :mod:`catanrl.agents.llm.player` (the player and its params),
:mod:`catanrl.agents.llm.prompts` (catanllm's strategies), :mod:`catanrl.agents.llm.render`
(the public-information state text), :mod:`catanrl.agents.llm.model` (OpenRouter and the
stubs) and :mod:`catanrl.agents.llm.cost` (the budget tool). ``catanrl.agents`` registers
``LLM`` in Catanatron's registry alongside POLICY, MCTS and BELIEF.
"""

from catanrl.agents.llm.model import (
           LLMError,
           Reply,
           StubModel,
           estimate_tokens,
           is_stub,
           make_model,
)
from catanrl.agents.llm.player import AnswerError, LlmParams, LlmPlayer, parse_answer
from catanrl.agents.llm.prompts import STRATEGIES, system_prompt
from catanrl.agents.llm.render import render_actions, render_state

__all__ = ["STRATEGIES", "AnswerError", "LLMError", "LlmParams", "LlmPlayer", "Reply", "StubModel",
           "estimate_tokens", "is_stub", "make_model", "parse_answer", "render_actions", "render_state",
           "system_prompt"]
