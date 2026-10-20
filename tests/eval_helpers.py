"""Helpers for the harness tests: real suites and native agents over
in-memory networks (no checkpoint), with the real agent param dataclasses."""

from __future__ import annotations

import dataclasses

import torch

from catanrl.agents.native import make_native_agent
from catanrl.agents.spec import (
    BeliefParams,
    MctsParams,
    PolicyParams,
    PredictParams,
    ResolvedAgent,
    TradeParams,
)
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.eval.suites import Suite
from catanrl.models import build_model

PARAMS = {"POLICY": PolicyParams, "MCTS": MctsParams, "BELIEF": BeliefParams, "PREDICT": PredictParams,
          "TRADE": TradeParams}


def make_suite(scenarios, turn_limit: int = 500) -> Suite:
    return Suite(name="test", purpose="dev", seed=7, boards=1, scenarios=tuple(dict(s) for s in scenarios),
                 turn_limit=turn_limit)


def schedule(suite: Suite, lineup_size: int, boards: int, seed: int = 7):
    """Every board in every seat rotation; seats past the lineup are the fill slot."""
    return dataclasses.replace(suite, boards=boards, seed=seed).schedule(lineup_size, fill=True)


def tiny_model(games=("base-2p",), level="private", seed=0, preset="graph-tiny"):
    torch.manual_seed(seed)
    dims = dims_for(list(games))
    return build_model({"preset": preset}, dims=dims, levels={"level": level},
                       num_actions=action_size(dims), layout=row_layout(dims, level))


def agent(key: str, model=None, infer_rows: int = 16, device: str = "cpu", predictor=None, critic=None, **params):
    """A native agent over in-memory networks: ``model`` (the policy) and, for PREDICT,
    ``predictor`` (the private aux-head network), for TRADE ``critic`` (a stateless private
    network; None reads the critic head of ``model``)."""
    if key == "PREDICT":
        params = PredictParams(ckpt="<in memory>", predictor="<in memory>", **params)
    elif key in PARAMS:
        params = PARAMS[key](ckpt="<in memory>", **params)
    resolved = ResolvedAgent(key=key, impl="test", params=params, checkpoint=None, fair=False, label=key,
                             canonical=key)
    # ``critic`` is handed over only when given: the other keys' backends take no such network.
    extra = {"critic": critic} if critic is not None else {}
    return make_native_agent(resolved, device=device, infer_rows=infer_rows, model=model, predictor=predictor, **extra)
