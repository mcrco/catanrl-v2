"""``catanrl llm-cost``: what an LLM eval would cost, from stub games.

Plays the first N scheduled games of a suite with ``LLM:model=stub/…`` in
slot 0 against an opponent, once per prompting strategy, through the
Catanatron harness (no records, no network), and reports per game: prompted
decisions, prompt tokens (characters/4 of what the stub was shown: the
system prompt, the state, the recent events, its last notes and the actions,
re-prompts included) and the stub's own short completions. A real model's
completions are unknown until it plays, so the cost column assumes
``completion_per_decision`` output tokens per prompted decision (an argument)
on top of the measured prompt tokens, at each ``--price NAME=IN,OUT`` (dollars
per million tokens), with no prompt caching (an upper bound: the system prompt
is a static per-game prefix, ``system_tokens`` of every call, which providers
cache at a fraction of the input price).

Two things make the measured figure a lower bound, and both are corrected
for: the stub's one-line note is a few words while a real model's is a
sentence or two, and the last ``notes`` notes (the spec's param, default 8)
are shown in every prompt (``notes_chars`` pads the stub's notes to a real
length, so their cost is measured at that length); and a random stub loses in
fewer turns than a competent model plays, so the table also projects the
prompt tokens at ``decisions`` prompted decisions per game (twice the measured
by default) from the measured per-decision base and per-note cost
(:func:`projected`: linear in the decisions once ``notes`` notes are shown;
catanllm showed every note, which grew with their square).

    catanrl llm-cost --games 5 --strategies plan,basic --price gpt-5=1.25,10 --price sonnet-4.5=3,15
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

#: Placeholder prices ($ per million input, output tokens) for the usage line; pass --price.
PLACEHOLDER_PRICES = {"example-model": (1.25, 10.0)}


class _StubAgent:
    """A resolved LLM agent whose players pad their stub's notes (duck-typed for the harness);
    it keeps the players it made, whose system prompts the table reports."""

    def __init__(self, agent: Any, **stub_options: Any) -> None:
        self.agent, self.stub_options = agent, stub_options
        self.players: list[Any] = []

    def player(self, color: Any) -> Any:
        player = self.agent.player(color)
        player.stub_options = dict(self.stub_options)
        self.players.append(player)
        return player


def shown_notes(decisions: float, notes: int) -> float:
    """Notes shown over ``decisions`` prompted decisions when decision k shows the last
    ``min(k, notes)`` (one note per decision): ``sum(min(k, notes) for k < decisions)``."""
    d, k = float(decisions), int(notes)
    if d <= k + 1:
        return d * (d - 1) / 2
    return k * (k + 1) / 2 + k * (d - 1 - k)


def measure(strategies: Sequence[str], *, suite: Any = "heldout-2p", games: int = 5, opponent: str = "F",
            model: str = "stub/random", thinking: bool = True, threads: int = 1, notes_chars: int = 0) -> dict[str, dict]:
    """Per strategy: the LLM slot's totals over the first ``games`` scheduled games, the stub's
    notes padded to ``notes_chars``; ``note_tokens`` (per note shown), ``notes`` (how many are
    shown, the spec's param) and ``base_per_decision`` (prompt tokens per decision without the
    notes) feed :func:`projected`; ``system_tokens`` is the system prompt's share of each call."""
    from catanrl.agents import resolve_agent
    from catanrl.agents.llm.model import STUB_NOTES, STUB_PREFIX, estimate_tokens, is_stub
    from catanrl.eval.catanatron import play_catanatron
    from catanrl.eval.results import llm_totals
    from catanrl.eval.suites import load_suite

    if not is_stub(model):
        raise ValueError(f"llm-cost plays stubs only (stub/random, stub/first), not {model!r}")
    note_chars = max(int(notes_chars), len(STUB_NOTES.get(model[len(STUB_PREFIX):], "")))
    note_tokens = (note_chars + 3) / 4  # the note, its "- " bullet and its newline, in each prompt showing it
    suite = load_suite(suite)
    out = {}
    for strategy in strategies:
        spec = f"LLM:model={model}:strategy={strategy}:thinking={'true' if thinking else 'false'}"
        llm = resolve_agent(spec, harness="catanatron", games=suite.games())
        shown = int(llm.params.notes)
        stub = _StubAgent(llm, notes_chars=int(notes_chars))
        lineup = [stub, resolve_agent(opponent, harness="catanatron", games=suite.games())]
        schedule = suite.schedule(len(lineup))[: int(games)]
        rows: list[dict] = []
        play_catanatron(lineup, schedule, suite=suite, threads=threads, on_game=rows.append)
        stats = [r["llm"][0] for r in rows]
        bases = [(s["prompt_tokens"] - note_tokens * shown_notes(s["decisions"], shown)) / s["decisions"]
                 for s in stats if s["decisions"]]
        systems = [estimate_tokens(p._system) for p in stub.players if p._system]
        out[strategy] = {"spec": llm.canonical, "games": len(rows), "turns": sum(r["turns"] for r in rows) / len(rows),
                         "wins": sum(1 for r in rows if r["winner"] == 0), **llm_totals(stats),
                         "notes_chars": note_chars, "note_tokens": note_tokens, "notes": shown,
                         "base_per_decision": sum(bases) / len(bases) if bases else 0.0,
                         "system_tokens": sum(systems) / len(systems) if systems else 0.0}
    return out


def projected(m: Mapping[str, Any], decisions: float) -> float:
    """Prompt tokens per game at ``decisions`` prompted decisions, the notes as measured: decision
    k shows ``min(k, notes)`` of them (``base * D + note_tokens * shown_notes(D, notes)``), so the
    growth is linear once ``notes`` notes are shown."""
    return decisions * m["base_per_decision"] + m["note_tokens"] * shown_notes(decisions, m.get("notes", 1 << 30))


def dollars(prompt_tokens: float, completion_tokens: float, price: tuple[float, float]) -> float:
    """Cost of one game at ``price`` ($ per million input, output tokens)."""
    return prompt_tokens * price[0] / 1e6 + completion_tokens * price[1] / 1e6


def _decisions(m: Mapping[str, Any], decisions: int | None) -> int:
    return int(decisions) if decisions else round(2 * m["per_game"]["decisions"])


def table(measured: Mapping[str, Mapping[str, Any]], prices: Mapping[str, tuple[float, float]], *,
          completion_per_decision: int, cell_games: int, decisions: int | None = None) -> list[str]:
    header = (f"{'strategy':<10} {'games':>5} {'turns':>6} {'decisions':>9} {'forced':>6} {'prompt tok':>10} "
              f"{'stub compl':>10} {'fallbacks':>9} {'wall s':>6}")
    lines = [header]
    for name, m in measured.items():
        pg = m["per_game"]
        lines.append(f"{name:<10} {m['games']:>5} {m['turns']:>6.0f} {pg['decisions']:>9.1f} {pg['forced']:>6.1f} "
                     f"{pg['prompt_tokens']:>10.0f} {pg['completion_tokens']:>10.0f} {pg['fallbacks']:>9.2f} "
                     f"{pg['wall_s']:>6.1f}")
    lines.append("")
    for name, m in measured.items():
        lines.append(f"{name}: the system prompt is {m.get('system_tokens', 0):.0f} of the "
                     f"{m['per_game']['prompt_tokens'] / max(1, m['per_game']['calls']):.0f} prompt tokens of a call "
                     f"(a static per-game prefix: cached, it costs a fraction of the input price)")
    lines.append(f"assumed completion tokens per prompted decision for a real model: {completion_per_decision}")
    cost_header = f"{'strategy':<10} {'model':<18} {'$/game':>8} {'$/' + str(cell_games) + ' games':>12}"
    lines.append("as measured (the stub's decisions per game):")
    lines.append(cost_header)
    for name, m in measured.items():
        pg = m["per_game"]
        completion = pg["decisions"] * completion_per_decision
        for label, price in prices.items():
            per_game = dollars(pg["prompt_tokens"], completion, price)
            lines.append(f"{name:<10} {label:<18} {per_game:>8.3f} {per_game * cell_games:>12.2f}")
    lines.append("")
    lines.append("projected at more decisions per game (a competent model outlasts a random one; each prompt shows "
                 "the last notes only, so prompt tokens grow linearly with the decisions; no caching):")
    lines.append(cost_header + "   decisions  prompt tok")
    for name, m in measured.items():
        n = _decisions(m, decisions)
        prompt = projected(m, n)
        for label, price in prices.items():
            per_game = dollars(prompt, n * completion_per_decision, price)
            lines.append(f"{name:<10} {label:<18} {per_game:>8.3f} {per_game * cell_games:>12.2f}   {n:>9} "
                         f"{prompt:>10.0f}")
    return lines


def parse_prices(entries: Sequence[str]) -> dict[str, tuple[float, float]]:
    prices = {}
    for entry in entries:
        name, _, rest = entry.partition("=")
        try:
            given, got = (float(x) for x in rest.split(","))
        except ValueError:
            raise ValueError(f"--price {entry!r}: expected NAME=IN,OUT in dollars per million tokens") from None
        prices[name.strip()] = (given, got)
    return prices


def main(args: Any) -> int:
    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    prices = parse_prices(args.price) if args.price else dict(PLACEHOLDER_PRICES)
    measured = measure(strategies, suite=args.suite, games=args.games, opponent=args.opponent, model=args.model,
                       thinking=args.thinking, threads=args.threads, notes_chars=args.notes_chars)
    if args.json:
        print(json.dumps({"measured": measured, "prices": prices,
                          "completion_per_decision": args.completion_per_decision, "notes_chars": args.notes_chars,
                          "projected": {name: {"decisions": _decisions(m, args.decisions),
                                               "prompt_tokens": projected(m, _decisions(m, args.decisions))}
                                        for name, m in measured.items()}}, indent=1))
        return 0
    print(f"{args.model} in slot 0 vs {args.opponent} on {args.suite}, first {args.games} scheduled games per "
          f"strategy (tokens are characters/4 estimates; the stub's notes are padded to {args.notes_chars} characters)")
    if not args.price:
        print("prices are placeholders: pass --price NAME=IN,OUT ($ per million input, output tokens)")
    print("\n".join(table(measured, prices, completion_per_decision=args.completion_per_decision,
                          cell_games=args.cell_games, decisions=args.decisions)))
    return 0
