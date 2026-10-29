"""Model clients: OpenRouter through the OpenAI SDK, and stubs that need no network.

``make_model("stub/first" | "stub/random" | "stub/scripted", seed=...)`` or
``make_model("openai/gpt-5")`` (any other id is an OpenRouter model id).
``complete(messages, ...)`` returns a :class:`Reply` with the text, the
reasoning when the provider returns it, the token usage (the SDK's ``usage``
fields for a real model, a characters/4 estimate, ``estimated``, for the stubs
or when a provider omits usage), the dollars charged (OpenRouter's
``usage.cost``; None when not reported, 0 for a stub) and the
``finish_reason`` (``length``: cut off by ``max_tokens``, which the player
never reads an answer from).

The real client reads ``OPENROUTER_API_KEY`` from the environment only, never
from a file, and imports ``openai`` lazily (``uv sync --extra llm``); the
stubs never need it. A missing key or SDK is an :class:`LLMError` at
construction, before any game is played. Request shape (OpenRouter's docs):
``reasoning.max_tokens`` is half of ``max_tokens`` with thinking on (the other
half is left for the visible answer; Anthropic needs the budget below
``max_tokens``, OpenAI's effort-only models map it to an effort), or
``reasoning.enabled=false``; ``temperature`` is sent only with thinking off
(Anthropic's extended thinking accepts only 1 or unset and OpenAI's reasoning
models reject it, so with thinking on the provider's default applies);
``usage.include`` asks for the cost (OpenRouter now reports it on every
reply); for ``anthropic/*`` models the system prompt, a static per-game
prefix, carries a ``cache_control`` breakpoint (OpenAI and Gemini cache
prefixes on their own). A reply without choices or with ``finish_reason``
``error`` (a provider failure inside a 200) is retried with backoff, then an
:class:`LLMError` (the SDK already retries transport and rate-limit errors).
"""

from __future__ import annotations

import os
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
STUB_PREFIX = "stub/"
STUB_KINDS = ("first", "random", "scripted")
#: catanllm's qwen convention: without thinking, "/no_think" closes the system prompt.
NO_THINK_MODELS = ("qwen",)
#: OpenRouter model ids whose system prompt gets an explicit cache breakpoint.
CACHE_CONTROL_PREFIXES = ("anthropic/",)


class LLMError(RuntimeError):
    """A model client that cannot be built or a call that cannot be made."""


@dataclass(frozen=True)
class Reply:
    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    estimated: bool = False
    #: Dollars charged for the call (OpenRouter's usage.cost); None when the provider did not say.
    cost: float | None = None
    #: The provider's finish_reason: ``stop``, ``length`` (cut off by max_tokens), ``content_filter``, ...
    finish_reason: str | None = None
    #: Prompt tokens read from a prompt cache, and completion tokens spent on reasoning (when reported).
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def estimate_tokens(text: str) -> int:
    """About four characters per token (what the stubs and the cost tool count)."""
    return max(1, (len(text) + 3) // 4)


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, Sequence):  # content parts (a cache breakpoint)
        return sum(len(part.get("text") or "") for part in content if isinstance(part, Mapping))
    return 0


def prompt_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(_content_chars(m.get("content")) for m in messages)


def is_stub(model: str) -> bool:
    return str(model).startswith(STUB_PREFIX)


def reasoning_budget(max_tokens: int) -> int:
    """The reasoning tokens asked for with thinking on: half of ``max_tokens``, the other half
    left for the visible answer (a reply cut off at ``max_tokens`` has no usable answer)."""
    return int(max_tokens) // 2


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


#: The notes line of the ``first`` and ``random`` stubs (a real model's is longer: ``notes_chars``).
STUB_NOTES = {"first": "Taking the first option.", "random": "Picking at random."}
_FILLER = "I weigh the options and keep the one that improves my production and position the most. "


def pad_notes(line: str, chars: int) -> str:
    """``line`` lengthened with filler to exactly ``chars`` characters (a stub's note is a few
    words; a real model's one-line note, shown in later prompts, is a sentence or two)."""
    if len(line) >= chars:
        return line
    return (line + " " + _FILLER * (chars // len(_FILLER) + 1))[:chars]


class StubModel:
    """A model that answers from the prompt's action list without a network.

    ``first`` answers 0, ``random`` a legal index from its own RNG (seeded, so
    games replay), ``scripted`` the next entry of its script (the default one
    exercises every parsing path). Each reply ends with a notes line and the
    index, as the prompt asks; ``notes_chars`` pads the notes line of ``first``
    and ``random`` to a real model's length (``catanrl llm-cost``). A stub
    costs nothing (``cost`` 0)."""

    def __init__(self, kind: str, *, seed: int = 0, script: Sequence[str] | None = None,
                 notes_chars: int = 0) -> None:
        if kind not in STUB_KINDS:
            raise LLMError(f"unknown stub {kind!r}; one of {', '.join(STUB_PREFIX + k for k in STUB_KINDS)}")
        self.kind = kind
        self.rng = random.Random(seed)
        self.script = list(script if script is not None else DEFAULT_SCRIPT)
        self.notes_chars = int(notes_chars)
        self.cursor = 0
        self.calls = 0

    @staticmethod
    def count_actions(messages: Sequence[Mapping[str, Any]]) -> int:
        """The numbered actions of the last user message's ``=== AVAILABLE ACTIONS ===``."""
        for message in reversed(messages):
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, str) and "=== AVAILABLE ACTIONS ===" in content:
                block = content.split("=== AVAILABLE ACTIONS ===", 1)[1]
                return len(_ACTIONS.findall(block))
        return 0

    def complete(self, messages: Sequence[Mapping[str, Any]], **_: Any) -> Reply:
        self.calls += 1
        n = max(1, self.count_actions(messages))
        if self.kind == "first":
            text = f"{pad_notes(STUB_NOTES['first'], self.notes_chars)}\n0"
        elif self.kind == "random":
            text = f"{pad_notes(STUB_NOTES['random'], self.notes_chars)}\n{self.rng.randrange(n)}"
        else:
            text = self.script[self.cursor % len(self.script)].format(last=n - 1)
            self.cursor += 1
        return Reply(text=text, reasoning=None, prompt_tokens=estimate_tokens(" " * prompt_chars(messages)),
                     completion_tokens=estimate_tokens(text), estimated=True, cost=0.0, finish_reason="stop")


# ---------------------------------------------------------------- OpenRouter


def cache_messages(model: str, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``messages`` as sent: for :data:`CACHE_CONTROL_PREFIXES` models the system prompt becomes
    a text part with an ephemeral ``cache_control`` breakpoint (OpenRouter's prompt caching:
    the static per-game prefix is written once and read at a tenth of the price after)."""
    out = [dict(m) for m in messages]
    if not str(model).startswith(CACHE_CONTROL_PREFIXES):
        return out
    for m in out:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
    return out


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, Mapping):
        value = obj.get(name)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


class OpenRouterModel:
    """``model`` (an OpenRouter id) through the OpenAI SDK at OpenRouter's base URL (the request
    shape is in the module docstring); the SDK retries transport and rate-limit errors with
    backoff, and a reply with no choices or ``finish_reason`` ``error`` is retried here
    ``failures`` times (``backoff`` seconds, doubling) before an :class:`LLMError`."""

    def __init__(self, model: str, *, api_key: str | None = None, max_retries: int = 5, timeout: float = 600.0,
                 base_url: str = OPENROUTER_BASE_URL, failures: int = 3, backoff: float = 5.0) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OPENROUTER_API_KEY is not set in the environment; a real model needs it "
                           "(stub/first, stub/random and stub/scripted play without one)")
        try:
            import openai
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise LLMError("the openai SDK is not installed: uv sync --extra llm") from error
        self.model = model
        self.failures, self.backoff = int(failures), float(backoff)
        self.client = openai.OpenAI(base_url=base_url, api_key=key, max_retries=max_retries, timeout=timeout,
                                    default_headers={"HTTP-Referer": "https://github.com/mod3sty/catanrl-v2",
                                                     "X-Title": "catanrl llm eval"})

    def complete(self, messages: Sequence[Mapping[str, Any]], *, max_tokens: int, temperature: float | None,
                 seed: int, thinking: bool) -> Reply:
        """One reply. ``temperature`` None (the player's choice with thinking on) is not sent."""
        body = {"reasoning": {"max_tokens": reasoning_budget(max_tokens)} if thinking else {"enabled": False},
                "usage": {"include": True}}
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if seed:
            kwargs["seed"] = int(seed)
        sent = cache_messages(self.model, messages)
        spent: list[Any] = []  # usage of failed attempts is charged too
        for attempt in range(self.failures + 1):
            response = self.client.chat.completions.create(model=self.model, messages=sent, max_tokens=int(max_tokens),
                                                           extra_body=body, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                spent.append(usage)
            choice = response.choices[0] if getattr(response, "choices", None) else None
            finish = getattr(choice, "finish_reason", None) if choice is not None else None
            if choice is not None and finish != "error":
                break
            if attempt == self.failures:
                error = _field(response, "error") or (f"finish_reason {finish}" if choice is not None else "no choices")
                raise LLMError(f"{self.model}: the provider failed {attempt + 1} times in a row: {error}")
            time.sleep(self.backoff * 2 ** attempt)
        message = choice.message
        text = (message.content or "").strip()
        thought = getattr(message, "reasoning", None)
        if usage is None or usage.prompt_tokens is None:
            return Reply(text=text, reasoning=thought, prompt_tokens=estimate_tokens(" " * prompt_chars(messages)),
                         completion_tokens=estimate_tokens(text + (thought or "")), estimated=True, cost=None,
                         finish_reason=finish)
        costs = [_field(u, "cost") for u in spent]
        return Reply(text=text, reasoning=thought,
                     prompt_tokens=sum(int(u.prompt_tokens or 0) for u in spent),
                     completion_tokens=sum(int(u.completion_tokens or 0) for u in spent),
                     cost=None if any(c is None for c in costs) else sum(float(c) for c in costs),
                     finish_reason=finish,
                     cached_tokens=int(_field(_field(usage, "prompt_tokens_details"), "cached_tokens") or 0),
                     reasoning_tokens=int(_field(_field(usage, "completion_tokens_details"), "reasoning_tokens") or 0))


def make_model(model: str, *, seed: int = 0, script: Sequence[str] | None = None, notes_chars: int = 0) -> Any:
    """A client for ``model``: a stub (``stub/<kind>``; ``script`` and ``notes_chars`` are its
    options), else OpenRouter."""
    if is_stub(model):
        return StubModel(model[len(STUB_PREFIX):], seed=seed, script=script, notes_chars=notes_chars)
    if script is not None or notes_chars:
        raise LLMError(f"{model}: script and notes_chars are options of the stubs")
    return OpenRouterModel(model)


def wants_no_think(model: str, thinking: bool) -> bool:
    return not thinking and any(name in model.lower() for name in NO_THINK_MODELS)
