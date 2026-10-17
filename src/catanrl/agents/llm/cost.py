"""``catanrl llm-cost``: what an LLM eval would cost, from stub games.

Plays the first N scheduled games of a suite with ``LLM:model=stub/…`` in
slot 0 against an opponent, once per prompting strategy, through the
Catanatron harness (no records, no network), and reports per game: prompted
decisions, prompt tokens (characters/4 of what the stub was shown: the
system prompt, the state text and the growing notes, re-prompts included)
and the stub's own short completions. A real model's completions are
unknown until it plays, so the cost column assumes ``completion_per_decision``
output tokens per prompted decision (an argument) on top of the measured
prompt tokens, at each ``--price NAME=IN,OUT`` (dollars per million tokens).

    catanrl llm-cost --games 5 --strategies plan,basic --price gpt-5=1.25,10 --price sonnet-4.5=3,15
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

#: Placeholder prices ($ per million input, output tokens) for the usage line; pass --price.
PLACEHOLDER_PRICES = {"example-model": (1.25, 10.0)}


def measure(strategies: Sequence[str], *, suite: Any = "heldout-2p", games: int = 5, opponent: str = "F",
            model: str = "stub/random", thinking: bool = True, threads: int = 1) -> dict[str, dict]:
    """Per strategy: the LLM slot's totals over the first ``games`` scheduled games."""
    from catanrl.agents import resolve_agent
    from catanrl.eval.catanatron import play_catanatron
    from catanrl.eval.results import llm_totals
    from catanrl.eval.suites import load_suite

    suite = load_suite(suite)
    out = {}
    for strategy in strategies:
        spec = f"LLM:model={model}:strategy={strategy}:thinking={'true' if thinking else 'false'}"
        lineup = [resolve_agent(spec, harness="catanatron", games=suite.games()),
                  resolve_agent(opponent, harness="catanatron", games=suite.games())]
        schedule = suite.schedule(len(lineup))[: int(games)]
        rows: list[dict] = []
        play_catanatron(lineup, schedule, suite=suite, threads=threads, on_game=rows.append)
        stats = [r["llm"][0] for r in rows]
        out[strategy] = {"spec": spec, "games": len(rows), "turns": sum(r["turns"] for r in rows) / len(rows),
                         "wins": sum(1 for r in rows if r["winner"] == 0), **llm_totals(stats)}
    return out


def dollars(prompt_tokens: float, completion_tokens: float, price: tuple[float, float]) -> float:
    """Cost of one game at ``price`` ($ per million input, output tokens)."""
    return prompt_tokens * price[0] / 1e6 + completion_tokens * price[1] / 1e6


def table(measured: Mapping[str, Mapping[str, Any]], prices: Mapping[str, tuple[float, float]], *,
          completion_per_decision: int, cell_games: int) -> list[str]:
    header = (f"{'strategy':<10} {'games':>5} {'turns':>6} {'decisions':>9} {'forced':>6} {'prompt tok':>10} "
              f"{'stub compl':>10} {'fallbacks':>9} {'wall s':>6}")
    lines = [header]
    for name, m in measured.items():
        pg = m["per_game"]
        lines.append(f"{name:<10} {m['games']:>5} {m['turns']:>6.0f} {pg['decisions']:>9.1f} {pg['forced']:>6.1f} "
                     f"{pg['prompt_tokens']:>10.0f} {pg['completion_tokens']:>10.0f} {pg['fallbacks']:>9.2f} "
                     f"{pg['wall_s']:>6.1f}")
    lines.append("")
    lines.append(f"assumed completion tokens per prompted decision for a real model: {completion_per_decision}")
    lines.append(f"{'strategy':<10} {'model':<18} {'$/game':>8} {'$/' + str(cell_games) + ' games':>12}")
    for name, m in measured.items():
        pg = m["per_game"]
        completion = pg["decisions"] * completion_per_decision
        for label, price in prices.items():
            per_game = dollars(pg["prompt_tokens"], completion, price)
            lines.append(f"{name:<10} {label:<18} {per_game:>8.3f} {per_game * cell_games:>12.2f}")
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
                       thinking=args.thinking, threads=args.threads)
    if args.json:
        print(json.dumps({"measured": measured, "prices": prices,
                          "completion_per_decision": args.completion_per_decision}, indent=1))
        return 0
    print(f"{args.model} in slot 0 vs {args.opponent} on {args.suite}, first {args.games} scheduled games per "
          f"strategy (tokens are characters/4 estimates)")
    if not args.price:
        print("prices are placeholders: pass --price NAME=IN,OUT ($ per million input, output tokens)")
    print("\n".join(table(measured, prices, completion_per_decision=args.completion_per_decision,
                          cell_games=args.cell_games)))
    return 0
