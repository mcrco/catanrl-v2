"""``LLM``: a Catanatron player that asks a language model for each move.

A port of catanllm's ``players/llm_player.py`` (attribution: Marco Yang's
catanllm, SP 25 CS 159) onto v2's agent specs::

    LLM:model=stub/random:strategy=plan
    LLM:model=openai/gpt-5:strategy=basic:thinking=true:max_tokens=8192:temperature=0.1:seed=0

Per decision with more than one legal action the player renders the public
state and the numbered actions (``render``), sends them under the strategy's
system prompt, and parses the answer: the action number on the last line, or
a number named in the text (``parse_answer``). An answer that cannot be used
(no number, an index out of range) is re-prompted with the error ``retries``
times, then the player falls back to End Turn if it is offered, else the
first action (catanllm's fallback), counted in ``stats["fallbacks"]``. Its
one-line notes are kept and shown as ``Previous moves`` on later turns, as in
catanllm.

``stats`` per game: ``decisions`` (prompted), ``forced`` (one legal action,
no prompt), ``prompt_tokens``, ``completion_tokens`` (the SDK's usage, or a
characters/4 estimate: ``estimated``), ``calls``, ``retries``, ``fallbacks``
and ``wall_s``. ``log`` holds every exchange when ``keep_log`` is set.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from catanatron.models.enums import ActionType
from catanatron.models.player import Color, Player

from catanrl.agents.llm.model import LLMError, Reply, make_model, wants_no_think
from catanrl.agents.llm.prompts import INSTRUCTION, RETRY, STRATEGIES, system_prompt
from catanrl.agents.llm.render import render_actions, render_state

Strategy = Literal["basic", "strategy", "search", "rules", "plan"]


@dataclass(frozen=True)
class LlmParams:
    model: str
    strategy: Strategy = "basic"
    thinking: bool = True
    max_tokens: int = 8192
    temperature: float = 0.1
    seed: int = 0
    retries: int = 1


def new_stats() -> dict[str, Any]:
    return {"decisions": 0, "forced": 0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0,
            "fallbacks": 0, "wall_s": 0.0, "estimated": False}


class AnswerError(ValueError):
    """An answer that names no usable action."""


_MARKUP = re.compile(r"^[\s>*_`#\-•.:)\[\]]+|[\s*_`.:)\]]+$")
_NAMED = re.compile(r"\b(?:action|option|choice|move)\s*(?:number|#|no\.?)?\s*[:#]?\s*(\d+)", re.IGNORECASE)
_INT = re.compile(r"(?<![\d.])\d+(?![\d.])")


def parse_answer(text: str, num_actions: int) -> tuple[int, str]:
    """``(index, notes)`` from a reply: the last non-empty line as the number (markdown
    stripped), else the last ``action N`` named in the text, else the only integer on
    the last line. ``notes`` is the line before the number (catanllm's convention)."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        raise AnswerError("the reply is empty")
    last = _MARKUP.sub("", lines[-1])
    index = None
    if re.fullmatch(r"\d+", last):
        index = int(last)
    else:
        named = _NAMED.findall(text)
        if named:
            index = int(named[-1])
        else:
            ints = _INT.findall(lines[-1])
            if len(ints) == 1:
                index = int(ints[0])
    if index is None:
        raise AnswerError("no action number found; end your reply with the action number alone on the last line")
    if not 0 <= index < num_actions:
        raise AnswerError(f"action {index} does not exist; choose a number from 0 to {num_actions - 1}")
    notes = lines[-2] if len(lines) >= 2 else ""
    return index, notes


def fallback_action(playable_actions: Any) -> Any:
    """catanllm's fallback: End Turn when offered, else the first legal action."""
    for action in playable_actions:
        if action.action_type == ActionType.END_TURN:
            return action
    return playable_actions[0]


class LlmPlayer(Player):
    """A language model (OpenRouter id, or stub/first, stub/random, stub/scripted) deciding from
    the public state text; one of catanllm's prompting strategies."""

    KEY = "LLM"
    LABEL = "LLM"
    Params = LlmParams

    def __init__(self, color: Color, params: LlmParams | None = None, *, resolved: Any = None, model: Any = None,
                 keep_log: bool = False) -> None:
        super().__init__(color, params)
        if self.params.strategy not in STRATEGIES:
            raise LLMError(f"unknown strategy {self.params.strategy!r}; one of {STRATEGIES}")
        self._resolved = resolved
        self._model = model
        self._owned = model is None  # a model the player built itself is rebuilt per game
        self._system: str | None = None
        self.keep_log = keep_log
        self.notes: list[str] = []
        self.log: list[dict[str, Any]] = []
        self.stats = new_stats()

    # ------------------------------------------------------------ game hooks

    def before(self, game: Any) -> None:
        """A new game: a fresh model (the stub's RNG from the spec seed and the game's), prompt and stats.
        A model given to the constructor (tests) is kept."""
        self.notes, self.log, self.stats = [], [], new_stats()
        self._system = None
        if self._owned:
            from catanrl.seeding import derive_seed

            seed = derive_seed(int(self.params.seed), "llm", int(game.seed), self.color.value)
            self._model = make_model(self.params.model, seed=seed)

    def model(self) -> Any:
        if self._model is None:
            self._model = make_model(self.params.model, seed=int(self.params.seed))
            self._owned = True
        return self._model

    def system(self, game: Any) -> str:
        if self._system is None:
            text = system_prompt(self.params.strategy, game.state.board.map, vps=int(game.vps_to_win))
            if wants_no_think(self.params.model, self.params.thinking):
                text += "\n\n/no_think"
            self._system = text
        return self._system

    def user_message(self, game: Any, playable_actions: Any) -> str:
        state_text = render_state(game, self.color)
        if self.notes:
            state_text += "\n\nPrevious moves: " + "\n".join(self.notes)
        return f"{state_text}\n\n{render_actions(playable_actions)}\n\n{INSTRUCTION}"

    def _complete(self, messages: list[dict[str, str]]) -> Reply:
        p = self.params
        reply = self.model().complete(messages, max_tokens=int(p.max_tokens), temperature=float(p.temperature),
                                      seed=int(p.seed), thinking=bool(p.thinking))
        s = self.stats
        s["calls"] += 1
        s["prompt_tokens"] += int(reply.prompt_tokens)
        s["completion_tokens"] += int(reply.completion_tokens)
        s["estimated"] = s["estimated"] or bool(reply.estimated)
        return reply

    def decide(self, game: Any, playable_actions: Any) -> Any:
        if len(playable_actions) == 1:
            self.stats["forced"] += 1
            return playable_actions[0]
        started = time.perf_counter()
        messages = [{"role": "system", "content": self.system(game)},
                    {"role": "user", "content": self.user_message(game, playable_actions)}]
        chosen, notes, exchanges = None, "", []
        for attempt in range(int(self.params.retries) + 1):
            reply = self._complete(messages)
            try:
                index, notes = parse_answer(reply.text, len(playable_actions))
            except AnswerError as error:
                exchanges.append({"reply": reply.text, "reasoning": reply.reasoning, "error": str(error)})
                if attempt < int(self.params.retries):
                    self.stats["retries"] += 1
                    messages = [*messages, {"role": "assistant", "content": reply.text},
                                {"role": "user", "content": RETRY.format(error=error)}]
                continue
            exchanges.append({"reply": reply.text, "reasoning": reply.reasoning, "index": index})
            chosen = playable_actions[index]
            break
        if chosen is None:
            self.stats["fallbacks"] += 1
            chosen = fallback_action(playable_actions)
        self.stats["decisions"] += 1
        self.stats["wall_s"] += time.perf_counter() - started
        if notes:
            self.notes.append(notes)
        if self.keep_log:
            self.log.append({"turn": int(game.state.num_turns), "prompt": messages[1]["content"],
                             "exchanges": exchanges, "chosen": str(chosen), "fallback": chosen not in
                             [playable_actions[e["index"]] for e in exchanges if "index" in e]})
        return chosen


__all__ = ["AnswerError", "LlmParams", "LlmPlayer", "fallback_action", "new_stats", "parse_answer"]
