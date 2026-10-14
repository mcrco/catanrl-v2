"""Agents: specs, resolution, native and Catanatron backends.

Importing this package registers POLICY, MCTS and BELIEF in Catanatron's
``REGISTRY``, so ``REGISTRY.build("POLICY:ckpt=...", color)`` works.
"""

from catanrl.agents.spec import (
    BeliefParams,
    MctsParams,
    PolicyParams,
    ResolvedAgent,
    check_spec_syntax,
    player,
    register,
    resolve_agent,
)

register()

__all__ = ["BeliefParams", "MctsParams", "PolicyParams", "ResolvedAgent", "check_spec_syntax", "player",
           "register", "resolve_agent"]
