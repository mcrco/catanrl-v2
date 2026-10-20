"""Agents: specs, resolution, native and Catanatron backends.

Importing this package registers POLICY, MCTS, BELIEF, PREDICT, TRADE and LLM in Catanatron's
``REGISTRY``, so ``REGISTRY.build("POLICY:ckpt=...", color)`` works.
"""

from catanrl.agents.llm.player import LlmParams
from catanrl.agents.spec import (
    BeliefParams,
    MctsParams,
    PolicyParams,
    PredictParams,
    ResolvedAgent,
    TradeParams,
    check_spec_syntax,
    player,
    register,
    resolve_agent,
)

register()

__all__ = ["BeliefParams", "LlmParams", "MctsParams", "PolicyParams", "PredictParams", "ResolvedAgent", "TradeParams",
           "check_spec_syntax", "player", "register", "resolve_agent"]
