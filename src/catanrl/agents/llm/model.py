"""Model clients: OpenRouter through the OpenAI SDK, and stubs that need no network.

``make_model("stub/first" | "stub/random" | "stub/scripted", seed=...)`` or
``make_model("openai/gpt-5")`` (any other id is an OpenRouter model id).
``complete(messages, ...)`` returns a :class:`Reply` with the text, the
reasoning when the provider returns it, and the token usage: the SDK's
``usage`` fields for a real model, a characters/4 estimate (``estimated``)
for the stubs or when a provider omits usage.

The real client reads ``OPENROUTER_API_KEY`` from the environment only, never
from a file, and imports ``openai`` lazily (``uv sync --extra llm``); the
stubs never need it. A missing key or SDK is an :class:`LLMError` at
construction, before any game is played.
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
STUB_PREFIX = "stub/"
STUB_KINDS = ("first", "random", "scripted")
#: catanllm's qwen convention: without thinking, "/no_think" closes the system prompt.
NO_THINK_MODELS = ("qwen",)


class LLMError(RuntimeError):
    """A model client that cannot be built or a call that cannot be made."""


@dataclass(frozen=True)
class Reply:
    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    estimated: bool = False


def estimate_tokens(text: str) -> int:
    """About four characters per token (what the stubs and the cost tool count)."""
    return max(1, (len(text) + 3) // 4)


def prompt_chars(messages: Sequence[Mapping[str, str]]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def is_stub(model: str) -> bool:
    return str(model).startswith(STUB_PREFIX)


# ---------------------------------------------------------------- stubs

_ACTIONS = re.compile(r"^(\d+): ", re.MULTILINE)

#: ``stub/scripted``'s default replies, cycled: a two-line answer, a text answer, an
#: out-of-range index (then a good one), garbage (then a good one), and garbage twice
#: (the fallback at retries=1). ``{last}`` is the last legal index.
DEFAULT_SCRIPT = (
    "Reasoning about the board.\nTaking the last option this turn.\n{last}",
    "I choose action 1.",
    "999",
    "0",
    "no idea",
    "0",
    "nonsense",
    "nonsense",
)


class StubModel:
    """A model that answers from the prompt's action list without a network.

    ``first`` answers 0, ``random`` a legal index from its own RNG (seeded, so
    games replay), ``scripted`` the next entry of its script (the default one
    exercises every parsing path). Each reply ends with a notes line and the
    index, as the prompt asks."""

    def __init__(self, kind: str, *, seed: int = 0, script: Sequence[str] | None = None) -> None:
        if kind not in STUB_KINDS:
            raise LLMError(f"unknown stub {kind!r}; one of {', '.join(STUB_PREFIX + k for k in STUB_KINDS)}")
        self.kind = kind
        self.rng = random.Random(seed)
        self.script = list(script if script is not None else DEFAULT_SCRIPT)
        self.cursor = 0
        self.calls = 0

    @staticmethod
    def count_actions(messages: Sequence[Mapping[str, str]]) -> int:
        """The numbered actions of the last user message's ``=== AVAILABLE ACTIONS ===``."""
        for message in reversed(messages):
            if message.get("role") == "user" and "=== AVAILABLE ACTIONS ===" in (message.get("content") or ""):
                block = message["content"].split("=== AVAILABLE ACTIONS ===", 1)[1]
                return len(_ACTIONS.findall(block))
        return 0

    def complete(self, messages: Sequence[Mapping[str, str]], **_: Any) -> Reply:
        self.calls += 1
        n = max(1, self.count_actions(messages))
        if self.kind == "first":
            text = "Taking the first option.\n0"
        elif self.kind == "random":
            text = f"Picking at random.\n{self.rng.randrange(n)}"
        else:
            text = self.script[self.cursor % len(self.script)].format(last=n - 1)
            self.cursor += 1
        return Reply(text=text, reasoning=None, prompt_tokens=estimate_tokens(" " * prompt_chars(messages)),
                     completion_tokens=estimate_tokens(text), estimated=True)


# ---------------------------------------------------------------- OpenRouter


class OpenRouterModel:
    """``model`` (an OpenRouter id) through the OpenAI SDK at OpenRouter's base URL.

    ``thinking`` asks for reasoning through OpenRouter's ``reasoning`` body field
    (as catanllm did: a reasoning budget of three quarters of ``max_tokens``), or
    turns it off; the SDK retries transport and rate-limit errors with backoff."""

    def __init__(self, model: str, *, api_key: str | None = None, max_retries: int = 5, timeout: float = 600.0,
                 base_url: str = OPENROUTER_BASE_URL) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OPENROUTER_API_KEY is not set in the environment; a real model needs it "
                           "(stub/first, stub/random and stub/scripted play without one)")
        try:
            import openai
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise LLMError("the openai SDK is not installed: uv sync --extra llm") from error
        self.model = model
        self.client = openai.OpenAI(base_url=base_url, api_key=key, max_retries=max_retries, timeout=timeout,
                                    default_headers={"HTTP-Referer": "https://github.com/mod3sty/catanrl-v2",
                                                     "X-Title": "catanrl llm eval"})

    def complete(self, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float, seed: int,
                 thinking: bool) -> Reply:
        reasoning = {"max_tokens": int(max_tokens * 0.75)} if thinking else {"enabled": False}
        kwargs: dict[str, Any] = {}
        if seed:
            kwargs["seed"] = int(seed)
        response = self.client.chat.completions.create(model=self.model, messages=list(messages),
                                                       temperature=temperature, max_tokens=max_tokens,
                                                       extra_body={"reasoning": reasoning}, **kwargs)
        if not response.choices:
            raise LLMError(f"{self.model}: the response has no choices")
        message = response.choices[0].message
        text = (message.content or "").strip()
        thought = getattr(message, "reasoning", None)
        usage = getattr(response, "usage", None)
        if usage is not None and usage.prompt_tokens is not None:
            return Reply(text=text, reasoning=thought, prompt_tokens=int(usage.prompt_tokens),
                         completion_tokens=int(usage.completion_tokens or 0))
        return Reply(text=text, reasoning=thought, prompt_tokens=estimate_tokens(" " * prompt_chars(messages)),
                     completion_tokens=estimate_tokens(text), estimated=True)


def make_model(model: str, *, seed: int = 0, script: Sequence[str] | None = None) -> Any:
    """A client for ``model``: a stub (``stub/<kind>``), else OpenRouter."""
    if is_stub(model):
        return StubModel(model[len(STUB_PREFIX):], seed=seed, script=script)
    return OpenRouterModel(model)


def wants_no_think(model: str, thinking: bool) -> bool:
    return not thinking and any(name in model.lower() for name in NO_THINK_MODELS)
