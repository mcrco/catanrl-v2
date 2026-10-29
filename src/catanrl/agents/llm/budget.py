"""A hard cap on what language-model calls may spend, shared by every thread of the process.

``CATANRL_LLM_BUDGET`` (dollars) caps this process's spend on real-model
calls, and a real model refuses to play without it (``none`` plays uncapped,
on purpose): ``LlmPlayer.preflight`` checks it, so the eval fails before any
game. Each call is checked before it is sent: once the spend so far reaches
the cap, :class:`BudgetExceeded` is raised, which the Catanatron harness
handles as any error (running games stop at their next tick, the rows of
finished games are recorded, a resume plays only the rest). Up to one call per
harness thread can already be in flight when the cap is reached, so the spend
may pass it by that much (threads x the costliest call).

``CATANRL_LLM_LEDGER`` (a file path) makes the cap span processes: every
charge is appended to it as a JSON line and a process starts from the file's
total, so ``catanrl run llm-eval`` (one subprocess per cell) stops at the cap
across all its cells. Without it the cap is per process (per cell under
``catanrl run``).

A charge is the reply's own cost (OpenRouter's ``usage.cost``, in dollars).
When a provider omits it, the tokens are priced at ``CATANRL_LLM_FALLBACK_PRICE``
(``IN,OUT`` dollars per million tokens; default :data:`FALLBACK_PRICE`, above
every listed frontier price), so a missing cost can stop the eval early but
never let it overspend; such charges are flagged (``cost_estimated``). Stubs
cost nothing and are never checked.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from catanrl.agents.llm.model import LLMError

BUDGET_ENV = "CATANRL_LLM_BUDGET"
LEDGER_ENV = "CATANRL_LLM_LEDGER"
FALLBACK_ENV = "CATANRL_LLM_FALLBACK_PRICE"
#: Dollars per million (input, output) tokens for a reply without a reported cost.
FALLBACK_PRICE = (15.0, 75.0)


class BudgetExceeded(LLMError):
    """The spend reached ``CATANRL_LLM_BUDGET``: no further call is made."""


def _cap(text: str | None) -> float | None:
    if text is None or text.strip().lower() in ("none", "inf", "unlimited"):
        return None
    try:
        cap = float(text)
    except ValueError:
        raise LLMError(f"{BUDGET_ENV}={text!r}: expected dollars (e.g. 25) or none") from None
    if not cap >= 0:
        raise LLMError(f"{BUDGET_ENV}={text!r}: the cap must be a non-negative number of dollars")
    return cap


def fallback_price() -> tuple[float, float]:
    text = os.environ.get(FALLBACK_ENV)
    if not text:
        return FALLBACK_PRICE
    try:
        given, got = (float(x) for x in text.split(","))
    except ValueError:
        raise LLMError(f"{FALLBACK_ENV}={text!r}: expected IN,OUT dollars per million tokens") from None
    return given, got


def charge_of(reply: Any) -> tuple[float, bool]:
    """``(dollars, estimated)`` for one reply: its reported cost, else its tokens at
    :func:`fallback_price`."""
    if reply.cost is not None:
        return float(reply.cost), False
    given, got = fallback_price()
    return (reply.prompt_tokens * given + reply.completion_tokens * got) / 1e6, True


class Budget:
    """The process's spend against ``cap`` (None: uncapped), optionally kept in ``ledger``."""

    def __init__(self, cap: float | None, ledger: Path | None = None) -> None:
        self.cap, self.ledger = cap, ledger
        self.lock = threading.Lock()
        self.spent = 0.0
        if ledger is not None and ledger.exists():
            for line in ledger.read_text().splitlines():
                if line.strip():
                    self.spent += float(json.loads(line)["cost"])

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` once the spend has reached the cap."""
        with self.lock:
            if self.cap is not None and self.spent >= self.cap:
                where = f" (ledger {self.ledger})" if self.ledger is not None else ""
                raise BudgetExceeded(
                    f"LLM spend ${self.spent:.4f} reached the cap {BUDGET_ENV}=${self.cap:g}{where}: no further "
                    "call is made; finished games are recorded, and a rerun with a higher cap resumes the rest")

    def charge(self, dollars: float, *, model: str, estimated: bool) -> None:
        if dollars <= 0:
            return
        with self.lock:
            self.spent += dollars
            if self.ledger is not None:
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                with open(self.ledger, "a") as f:
                    f.write(json.dumps({"time": time.time(), "pid": os.getpid(), "model": model, "cost": dollars,
                                        "estimated": estimated}) + "\n")


_BUDGET: tuple[tuple[str | None, str | None], Budget] | None = None
_LOCK = threading.Lock()


def budget() -> Budget:
    """This process's :class:`Budget`, built from the environment (rebuilt if it changed)."""
    global _BUDGET
    env = (os.environ.get(BUDGET_ENV), os.environ.get(LEDGER_ENV))
    with _LOCK:
        if _BUDGET is None or _BUDGET[0] != env:
            _BUDGET = (env, Budget(_cap(env[0]), Path(env[1]).expanduser() if env[1] else None))
        return _BUDGET[1]


def reset() -> None:
    """Forget the process's budget (tests; the next :func:`budget` reads the environment anew)."""
    global _BUDGET
    with _LOCK:
        _BUDGET = None


def require() -> None:
    """A real model plays only under an explicit cap (``CATANRL_LLM_BUDGET``; ``none`` to opt out)."""
    if os.environ.get(BUDGET_ENV) is None:
        raise LLMError(f"{BUDGET_ENV} is not set: a real model plays only under a spending cap in dollars "
                       f"({BUDGET_ENV}=25; {LEDGER_ENV}=<file> to share it across processes; none for no cap)")
    budget()  # a malformed value fails here, before any game


__all__ = ["BUDGET_ENV", "FALLBACK_ENV", "FALLBACK_PRICE", "LEDGER_ENV", "Budget", "BudgetExceeded", "budget",
           "charge_of", "fallback_price", "require", "reset"]
