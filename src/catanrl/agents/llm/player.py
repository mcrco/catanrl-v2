"""``LLM``: a Catanatron player that asks a language model for each move.

A port of catanllm's ``players/llm_player.py`` (attribution: Marco Yang's
catanllm, SP 25 CS 159) onto v2's agent specs::

    LLM:model=stub/random:strategy=plan
    LLM:model=openai/gpt-5:strategy=basic:thinking=true:max_tokens=16384:temperature=0.1:seed=0

Per decision with more than one legal action the player sends the strategy's
system prompt (static for the game: the preamble, this game's rules and the
board; ``prompts``) and a user message: the seat's view of the state, the last
``history`` events as it saw them, its own last ``notes`` one-line notes and
the numbered, annotated actions (``render``). The answer is parsed from the
last line only: the action number alone, ``action N`` named there, or the
number leading or alone on it unless it names a board feature (``node 12``);
a number in the reasoning above never counts, as catanllm read ``int(last
line)`` (``parse_answer``). An answer that cannot be used (no number, an index
out of range, a reply cut off at ``max_tokens``, whose last line is
mid-sentence) is re-prompted with the error ``retries`` times, then the player
falls back to End Turn if it is offered, else the first action (catanllm's
fallback), counted in ``stats["fallbacks"]``. catanllm showed every past note
in every later prompt, so prompt tokens grew with the square of the decisions;
here only the last ``notes`` (each cut to :data:`NOTE_CHARS`) are shown,
labelled as the model's own.

``temperature`` is sent only with ``thinking`` off (``model``'s docstring);
with thinking on, half of ``max_tokens`` is the reasoning budget. Before each
call to a real model the process's spending cap is checked and after it the
reply's cost is charged (``budget``: ``CATANRL_LLM_BUDGET``, required by
``preflight``); a stub costs nothing.

``stats`` per game: ``decisions`` (prompted), ``forced`` (one legal action,
no prompt), ``prompt_tokens``, ``completion_tokens`` (the SDK's usage, or a
characters/4 estimate: ``estimated``), ``cached_tokens`` and
``reasoning_tokens`` (when reported), ``cost`` (dollars; ``cost_estimated``
when a reply's cost was missing and priced by ``budget.fallback_price``),
``calls``, ``retries``, ``truncated`` (replies cut off at ``max_tokens``),
``fallbacks`` and ``wall_s``. ``log`` holds every exchange when ``keep_log``
is set (the system prompt in the first entry).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from catanatron.models.enums import ActionType
from catanatron.models.player import Color, Player
from catanatron.params import ParamsError

from catanrl.agents.llm import budget
from catanrl.agents.llm.model import LLMError, Reply, is_stub, make_model, wants_no_think
from catanrl.agents.llm.prompts import INSTRUCTION, RETRY, STRATEGIES, system_prompt
from catanrl.agents.llm.render import render_actions, render_events, render_state

Strategy = Literal["basic", "strategy", "search", "rules", "plan"]
#: A note longer than this is cut (a model that writes its whole reasoning on the notes line).
NOTE_CHARS = 300


@dataclass(frozen=True)
class LlmParams:
    model: str
    strategy: Strategy = "basic"
    thinking: bool = True
    #: The reply's token cap; with thinking on, half of it is the reasoning budget.
    max_tokens: int = 16384
    #: Sent only with thinking off (Anthropic's thinking needs 1 or unset; OpenAI's reasoning models reject it).
    temperature: float = 0.1
    seed: int = 0
    retries: int = 1
    #: How many of its own past notes the model is shown (the last ones).
    notes: int = 8
    #: How many recent game events the model is shown.
    history: int = 30

    def __post_init__(self) -> None:
        if self.notes < 0 or self.history < 0:
            raise ParamsError("LLM: notes and history count past entries shown (>= 0)")
        if self.thinking and self.max_tokens < 2048:
            raise ParamsError("LLM: with thinking, max_tokens must be at least 2048 (half of it is the reasoning "
                              "budget, which providers floor at 1024)")


def new_stats() -> dict[str, Any]:
    return {"decisions": 0, "forced": 0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
            "reasoning_tokens": 0, "cost": 0.0, "retries": 0, "truncated": 0, "fallbacks": 0, "wall_s": 0.0,
            "estimated": False, "cost_estimated": False}


class AnswerError(ValueError):
    """An answer that names no usable action."""


_MARKUP = re.compile(r"^[\s>*_`#\-•.:)\[\]]+|[\s*_`.:)\]]+$")
_NAMED = re.compile(r"\b(?:action|option|choice|move)\s*(?:number|#|no\.?)?\s*[:#]?\s*(\d+)", re.IGNORECASE)
_INT = re.compile(r"(?<![\d.])\d+(?![\d.])")
#: A number next to one of these words names a board feature or a count, not an action.
_BOARD_WORDS = re.compile(
    r"(?:nodes?|tiles?|edges?|hex(?:es)?|turns?|vps?|points?|cards?|resources?|roads?|settlements?|cities|city|"
    r"knights?|dice|rolls?|rolled|wood|brick|sheep|wheat|ore|lumber|wool|grain|players?)", re.IGNORECASE)


def _names_a_feature(line: str, match: re.Match) -> bool:
    """Whether the word right before or after the number is a board word (``node 12``, ``7 VP``)."""
    before = re.search(r"([A-Za-z]+)\W*$", line[: match.start()])
    after = re.match(r"\W*([A-Za-z]+)", line[match.end():])
    return any(word is not None and _BOARD_WORDS.fullmatch(word.group(1)) for word in (before, after))


def parse_answer(text: str, num_actions: int) -> tuple[int, str]:
    """``(index, notes)`` from a reply, read from its last non-empty line only (markdown
    stripped): the number alone, else ``action N`` named on it, else the number leading
    it or the only number on it, unless that number names a board feature (``node 12``,
    ``7 VP``). Numbers in the reasoning above never count (catanllm read ``int(last
    line)``); a line of markup alone is skipped. ``notes`` is the line before the number
    (catanllm's convention), cut to :data:`NOTE_CHARS`."""
    # A line of markup alone (``---``, a closing fence) is not the answer line.
    lines = [line.strip() for line in (text or "").splitlines() if re.search(r"[A-Za-z0-9]", line)]
    if not lines:
        raise AnswerError("the reply is empty")
    last = _MARKUP.sub("", lines[-1])
    index = None
    if re.fullmatch(r"\d+", last):
        index = int(last)
    elif named := _NAMED.findall(last):
        index = int(named[-1])
    else:
        ints = list(_INT.finditer(last))
        leading = ints[0] if ints and ints[0].start() == 0 else None
        candidate = leading if leading is not None else (ints[0] if len(ints) == 1 else None)
        if candidate is not None and not _names_a_feature(last, candidate):
            index = int(candidate.group())
    if index is None:
        raise AnswerError("no action number found; end your reply with the action number alone on the last line")
    if not 0 <= index < num_actions:
        raise AnswerError(f"action {index} does not exist; choose a number from 0 to {num_actions - 1}")
    notes = lines[-2] if len(lines) >= 2 else ""
    if len(notes) > NOTE_CHARS:
        notes = notes[:NOTE_CHARS - 3].rstrip() + "..."
    return index, notes


def fallback_action(playable_actions: Any) -> Any:
    """catanllm's fallback: End Turn when offered, else the first legal action."""
    for action in playable_actions:
        if action.action_type == ActionType.END_TURN:
            return action
    return playable_actions[0]


class LlmPlayer(Player):
    """A language model (OpenRouter id, or stub/first, stub/random, stub/scripted) deciding from
    the seat's view of the game in text; one of catanllm's prompting strategies."""

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
        #: Extra StubModel arguments (``catanrl llm-cost`` pads the stub's notes); ignored by a real model.
        self.stub_options: dict[str, Any] = {}
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
            self._model = make_model(self.params.model, seed=seed, **self.stub_options)

    def model(self) -> Any:
        if self._model is None:
            self._model = make_model(self.params.model, seed=int(self.params.seed), **self.stub_options)
            self._owned = True
        return self._model

    @property
    def real(self) -> bool:
        """Whether calls cost money (anything but a stub id): checked against and charged to the budget."""
        return not is_stub(self.params.model)

    def preflight(self) -> None:
        """Build the model client now (the harness's preflight) and, for a real model, require a
        spending cap: a missing key, SDK or cap fails before any game."""
        self.model()
        if self.real:
            budget.require()

    def system(self, game: Any) -> str:
        if self._system is None:
            text = system_prompt(self.params.strategy, game, self.color)
            if wants_no_think(self.params.model, self.params.thinking):
                text += "\n\n/no_think"
            self._system = text
        return self._system

    def user_message(self, game: Any, playable_actions: Any) -> str:
        """The state, the recent events, the model's own last notes, the actions and the instruction."""
        parts = [render_state(game, self.color)]
        events = render_events(game, self.color, int(self.params.history))
        if events:
            parts.append(events)
        notes = self.notes[-int(self.params.notes):] if int(self.params.notes) > 0 else []
        if notes:
            parts.append(f"=== YOUR OWN NOTES FROM YOUR LAST {len(notes)} DECISIONS (oldest first) ===\n"
                         + "\n".join(f"- {note}" for note in notes))
        parts += [render_actions(game, playable_actions, self.color), INSTRUCTION]
        return "\n\n".join(parts)

    def _complete(self, messages: list[dict[str, str]]) -> Reply:
        p = self.params
        ledger = budget.budget() if self.real else None
        if ledger is not None:
            ledger.check()  # before the call: a reached cap stops here, not after paying
        reply = self.model().complete(messages, max_tokens=int(p.max_tokens),
                                      temperature=None if p.thinking else float(p.temperature),
                                      seed=int(p.seed), thinking=bool(p.thinking))
        dollars, guessed = budget.charge_of(reply)
        if ledger is not None:
            ledger.charge(dollars, model=p.model, estimated=guessed)
        s = self.stats
        s["calls"] += 1
        s["prompt_tokens"] += int(reply.prompt_tokens)
        s["completion_tokens"] += int(reply.completion_tokens)
        s["cached_tokens"] += int(reply.cached_tokens)
        s["reasoning_tokens"] += int(reply.reasoning_tokens)
        s["cost"] += dollars
        s["truncated"] += int(reply.truncated)
        s["estimated"] = s["estimated"] or bool(reply.estimated)
        s["cost_estimated"] = s["cost_estimated"] or guessed
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
            exchange = {"reply": reply.text, "reasoning": reply.reasoning, "finish_reason": reply.finish_reason}
            try:
                if reply.truncated:  # its last line is wherever the cap fell: never an answer
                    raise AnswerError("your reply was cut off at the token limit before it ended; reason more "
                                      "briefly")
                index, notes = parse_answer(reply.text, len(playable_actions))
            except AnswerError as error:
                exchanges.append({**exchange, "error": str(error)})
                if attempt < int(self.params.retries):
                    self.stats["retries"] += 1
                    # An empty assistant message (a reasoning budget spent with nothing left to say) is a
                    # 400 at some providers, which the SDK does not retry; a placeholder keeps the error text.
                    # A cut-off reply is not sent back (it may be thousands of tokens of half an answer).
                    shown = "(reply cut off at the token limit)" if reply.truncated else (reply.text or "(no answer)")
                    messages = [*messages, {"role": "assistant", "content": shown},
                                {"role": "user", "content": RETRY.format(error=error)}]
                continue
            exchanges.append({**exchange, "index": index})
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
            entry = {"turn": int(game.state.num_turns), "prompt": messages[1]["content"], "exchanges": exchanges,
                     "chosen": str(chosen), "fallback": all("index" not in e for e in exchanges)}
            if not self.log:
                entry["system"] = messages[0]["content"]
            self.log.append(entry)
        return chosen


__all__ = ["AnswerError", "LlmParams", "LlmPlayer", "fallback_action", "new_stats", "parse_answer"]
