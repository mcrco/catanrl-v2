"""``catanrl``: run experiments, evaluate specs, and look after the runs store.

::

    catanrl run arch-lineage [--only 'graph-s-*'] [--set train.gamma=1.0] [--smoke] [--next]
    catanrl run arch-lineage/graph-s-s42.g1+more        # a forked variant, from its stored config
    catanrl eval POLICY:arch-lineage/graph-s-s42.g1 F --suite heldout-2p
    catanrl eval LLM:model=stub/random:strategy=plan F --suite heldout-2p --boards 20 --harness catanatron
    catanrl llm-cost --strategies plan,basic --games 5 --price gpt-5=1.25,10   # a budget before any real call
    catanrl ls ['arch-lineage/*'] [--status running]
    catanrl pin arch-lineage/graph-s-s42.g1@e600 ref --note "report"
    catanrl fork arch-lineage/graph-s-s42.g1 --as graph-s-s42.g1+more --set train.total_timesteps=2e9
    catanrl finish arch-lineage/graph-s-s42.g1
    catanrl results ['arch-lineage/*'] [--stale]        catanrl verify EVAL_ID [--games 16]
    catanrl check-runs ['*']    catanrl summarize arch-lineage    catanrl validate arch-lineage
    catanrl report-data [report/results.yaml] [--run-missing] [--update] [--final]
    catanrl bot --spec MCTS:arch-lineage/graph-s-s42.g1@ref:sims=256:dets=4 ws://HOST:PORT/room/ID
    catanrl play --seats human,POLICY:arch-lineage/graph-s-s42.g1@ref,F,F
    catanrl house deploy/house.yaml [--server wss://HOST]   # keep tables open for people
    catanrl openhex deploy/openhex.yaml                      # play our bots on open-hex.web.app
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catanrl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment (YAML path or name) or a stored variant (exp/variant)")
    run.add_argument("target")
    run.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--only", help="variants matching this glob")
    run.add_argument("--variant", help="exactly this variant")
    run.add_argument("--smoke", action="store_true", help="a few seconds per variant, as EXP+smoke")
    run.add_argument("--next", dest="next_", action="store_true", help="claim free variants until none is left")
    run.add_argument("--allow-code-change", action="store_true", help="resume across a training code change")
    run.add_argument("--inline", action="store_true", help=argparse.SUPPRESS)

    ev = sub.add_parser("eval", help="evaluate a lineup on a suite (cached)")
    ev.add_argument("lineup", nargs="+", help="agent specs, slot 0 first")
    ev.add_argument("--suite", required=True)
    ev.add_argument("--boards", type=int, help="only the suite's first N boards")
    ev.add_argument("--fill", help="the agent for seats past the lineup")
    ev.add_argument("--harness", default="native", choices=["native", "catanatron"],
                    help="native (cppanatron, batched) or catanatron (Catanatron players: LLM, AB, ...)")
    ev.add_argument("--device")
    ev.add_argument("--num-envs", type=int)
    ev.add_argument("--threads", type=int)
    ev.add_argument("--infer-rows", type=int)
    ev.add_argument("--allow-mixed", action="store_true")
    ev.add_argument("--allow-stale-build", action="store_true", help="play on an engine built from another tree")

    cost = sub.add_parser("llm-cost", help="stub games through the LLM player: decisions, tokens and dollars per game")
    cost.add_argument("--suite", default="heldout-2p")
    cost.add_argument("--games", type=int, default=5, help="the suite's first N scheduled games per strategy")
    cost.add_argument("--strategies", default="basic,strategy,search,rules,plan", help="comma-separated")
    cost.add_argument("--opponent", default="F", help="the other seat's spec")
    cost.add_argument("--model", default="stub/random", help="a stub: stub/random (default), stub/first")
    cost.add_argument("--no-thinking", dest="thinking", action="store_false")
    cost.add_argument("--completion-per-decision", type=int, default=1500,
                      help="assumed completion tokens per prompted decision for a real model (the stub's "
                           "replies are short; reasoning models write thousands)")
    cost.add_argument("--cell-games", type=int, default=50, help="games per cell in the cost column")
    cost.add_argument("--notes-chars", type=int, default=200,
                      help="pad the stub's one-line notes to this many characters (a real model's note; every note "
                           "is shown in every later prompt); 0 keeps the stub's own 18-24")
    cost.add_argument("--decisions", type=int,
                      help="also project the cost at this many prompted decisions per game (default: twice the "
                           "measured; a competent model outlasts a random one)")
    cost.add_argument("--price", action="append", default=[], metavar="NAME=IN,OUT",
                      help="dollars per million input and output tokens; repeatable")
    cost.add_argument("--threads", type=int, default=1)
    cost.add_argument("--json", action="store_true", help="print the measurements as JSON")

    ls = sub.add_parser("ls", help="list runs")
    ls.add_argument("pattern", nargs="?", default="*")
    ls.add_argument("--status")

    pin = sub.add_parser("pin", help="name a checkpoint")
    pin.add_argument("ref")
    pin.add_argument("name")
    pin.add_argument("--note", default="")

    fork = sub.add_parser("fork", help="a new variant continuing a run's checkpoint")
    fork.add_argument("ref")
    fork.add_argument("--as", dest="as_", required=True)
    fork.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    fin = sub.add_parser("finish", help="end a stopped run so dependents can use it")
    fin.add_argument("ref")

    res = sub.add_parser("results", help="list eval records")
    res.add_argument("pattern", nargs="?", default="*")
    res.add_argument("--stale", action="store_true", help="records made by other outcome code")

    ver = sub.add_parser("verify", help="replay games of a record and compare")
    ver.add_argument("record", help="a record path or eval id")
    ver.add_argument("--games", type=int, default=16)
    ver.add_argument("--force", action="store_true",
                     help="replay a real language model's record anyway (paid calls; its games will differ)")

    chk = sub.add_parser("check-runs", help="load and self-test every alias of every run")
    chk.add_argument("pattern", nargs="?", default="*")

    summ = sub.add_parser("summarize", help="write runs/<experiment>/summary.md")
    summ.add_argument("experiment")

    val = sub.add_parser("validate", help="check an experiment YAML")
    val.add_argument("target")

    rep = sub.add_parser("report-data", help="write report/data/results.json from report/results.yaml")
    rep.add_argument("spec", nargs="?", default="report/results.yaml")
    rep.add_argument("--run-missing", action="store_true")
    rep.add_argument("--update", action="store_true", help="accept refs that now resolve differently")
    rep.add_argument("--final", action="store_true", help="refuse flagged (†) rows and re-play games of each record")
    rep.add_argument("--verify-games", type=int, default=16)

    bot = sub.add_parser("bot", help="play a catanatron-server seat with a fair agent spec")
    bot.add_argument("url", help="ws://HOST:PORT/room/ID, or the server with --new-room")
    bot.add_argument("--spec", required=True)
    bot.add_argument("--new-room", type=json.loads, metavar="JSON", help='a room config, or {"config", "seats", "name"}')
    bot.add_argument("--seat", type=int)
    bot.add_argument("--transcripts", default="transcripts", help="where the verified transcript is saved")
    bot.add_argument("--name")
    bot.add_argument("--device", default="cpu")
    bot.add_argument("--infer-rows", type=int, default=1)
    bot.add_argument("--server-key", help="pin the server's public key (hex)")
    bot.add_argument("--key", dest="key_path", help="this bot's identity key file")
    bot.add_argument("--no-verify", dest="verify", action="store_false")

    house = sub.add_parser("house", help="keep rooms open on a catanatron-server for people to play our bots")
    house.add_argument("config", help="a YAML file: server, server_key, think_s and tables")
    house.add_argument("--server", help="override the file's server URL")
    house.add_argument("--state", default="house", help="keys, transcripts and games.jsonl go here")
    house.add_argument("--device", default="cpu")

    oh = sub.add_parser("openhex", help="keep our bots queued on OpenHex (open-hex.web.app)")
    oh.add_argument("config", help="a YAML file: bots (name, spec, mode, key_env), optional api")
    oh.add_argument("--state", default="openhex", help="games.jsonl goes here")
    oh.add_argument("--device", default="cpu")
    oh.add_argument("--only", help="run just this bot (by name)")

    play = sub.add_parser("play", help="a local TUI game against our agents and Catanatron bots")
    play.add_argument("--seats", required=True, help="comma-separated: human, Catanatron specs, POLICY/MCTS/BELIEF specs")
    play.add_argument("--config", type=json.loads, metavar="JSON", help="a game config (players, map, rules)")
    play.add_argument("--seed", type=int)
    play.add_argument("--device", default="cpu")
    play.add_argument("--infer-rows", type=int, default=1)
    return p


def _find_record(text: str):
    from catanrl.eval.results import EvalRecord, all_records

    if Path(text).exists():
        return EvalRecord.load(text)
    matches = [p for p in all_records() if p.name.startswith(text)]
    if len(matches) != 1:
        raise SystemExit(f"error: {len(matches)} records match {text!r}")
    return EvalRecord.load(matches[0])


def _cmd_run(args, argv) -> int:
    from catanrl import runner
    from catanrl.config import load_plan

    target = Path(args.target)
    if "/" in args.target and not target.exists() and not args.target.endswith(".yaml"):
        return _run_stored(args, argv)
    plan = load_plan(args.target, overrides=args.overrides, only=args.only, variant=args.variant, smoke=args.smoke)
    if args.inline:
        return runner.run_inline(plan, args.variant, argv=argv, allow_code_change=args.allow_code_change)
    runner.run_plan(plan, next_=args.next_, allow_code_change=args.allow_code_change, overrides=args.overrides)
    return 0


def _run_stored(args, argv) -> int:
    """A stored variant (e.g. a fork): --set goes through the same config check as a plan."""
    from catanrl import runner
    from catanrl.config import ConfigError, Variant, config_hash, merge, parse_overrides, resolve
    from catanrl.runs import load_experiment

    if args.smoke or args.only or args.next_ or args.variant:
        raise ConfigError("--smoke, --only, --next and --variant take an experiment, not a stored variant")
    v = load_experiment(args.target).variant()
    if args.overrides:
        cfg = resolve(merge(v.config, parse_overrides(args.overrides)), experiment=v.experiment, variant=v.name)
        v = Variant(v.experiment, v.name, v.stage, v.labels, cfg, config_hash(cfg), v.depends_on)
    print(runner.run_variant(v, argv=argv, allow_code_change=args.allow_code_change))
    return 0


def _cmd_eval(args) -> int:
    from catanrl.eval.results import TRADE_KEYS, evaluate

    if args.allow_stale_build:
        os.environ["CATANRL_ALLOW_STALE_BUILD"] = "1"
    suite = {"name": args.suite, "boards": args.boards} if args.boards else args.suite
    record = evaluate(args.lineup, suite, fill=args.fill, harness=args.harness, device=args.device,
                      num_envs=args.num_envs, threads=args.threads, infer_rows=args.infer_rows,
                      allow_mixed=args.allow_mixed)
    s = record.summary
    for slot in s["slots"]:
        lo, hi = slot["wilson"]
        print(f"{slot['label']:<40} {100 * slot['score_rate']:6.2f}% [{100 * lo:.2f}, {100 * hi:.2f}] "
              f"(n={slot['games']})")
    for slot in s["slots"]:
        if "llm" in slot:
            llm = slot["llm"]["per_game"]
            print(f"{slot['label']:<40} per game: {llm['decisions']:.1f} decisions, {llm['prompt_tokens']:.0f} prompt + "
                  f"{llm['completion_tokens']:.0f} completion tokens{' (estimated)' if slot['llm']['estimated'] else ''}, "
                  f"{llm['fallbacks']:.2f} fallbacks, {llm['wall_s']:.1f} s")
        if "trades" in slot:  # a loaded record's keys are canonical (sorted): print the steps in order
            trades = slot["trades"]
            print(f"{slot['label']:<40} per game: "
                  + ", ".join(f"{trades['per_game'][k]:.2f} {k}" for k in TRADE_KEYS)
                  + f"; cancel rate {trades['cancel_rate']:.2f}")
    flags = [f for f, on in (("INVALID", not record.data["valid"]), ("mixed", record.data["mixed"])) if on]
    print(f"{record.eval_id} {record.path}{'  ' + ' '.join(flags) if flags else ''}")
    return 0


def _cmd_ls(args) -> int:
    from catanrl.runs import list_runs

    for run in list_runs(args.pattern, status=args.status):
        ckpts = run.checkpoints()
        best = run.aliases().get("best")
        flags = []
        if best and ckpts and best["epoch"] == 0 and run.config.get("init_from"):
            flags.append("best is still the parent's weights")
        best_text = f"best e{best['epoch']} {best['metric']:.3f}" if best and best.get("metric") is not None else "-"
        print(f"{run.ref:<45} {run.status!s:<12} {f'e{ckpts[-1].epoch}' if ckpts else '-':>8}  "
              f"{best_text}  {'; '.join(flags)}")
    return 0


def _cmd_results(args) -> int:
    from catanrl.eval.results import EvalRecord, all_records, stale

    records = stale(args.pattern) if args.stale else [EvalRecord.load(p) for p in all_records(args.pattern)]
    for r in records:
        print(f"{r.eval_id}  {r.data['suite']:<16} {r.summary['score_rate']:.3f} n={r.summary['games']:<5} "
              f"{' vs '.join(r.data['labels'])}")
    return 0


def _cmd_bot(args) -> int:
    from catanrl.bridge.server_bot import run_bot

    options = {k: v for k, v in (("name", args.name), ("server_key", args.server_key),
                                 ("key_path", args.key_path)) if v is not None}
    result = run_bot(args.spec, args.url, new_room=args.new_room, seat=args.seat, transcript_dir=args.transcripts,
                     device=args.device, infer_rows=args.infer_rows, verify=args.verify, **options)
    print(json.dumps(result.summary(), default=str, indent=2))
    return 0 if result.verified or not args.verify else 1


def _cmd_house(args) -> int:
    import asyncio
    import logging

    from catanrl.bridge.house import House, load_tables

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings, tables = load_tables(args.config)
    house = House(args.server or settings["server"], tables, state_dir=args.state,
                  server_key=settings.get("server_key"), think_s=settings.get("think_s", 0.8), device=args.device)
    asyncio.run(house.run())
    return 0


def _cmd_openhex(args) -> int:
    import logging
    import threading

    import yaml

    from catanrl.bridge.openhex import API, OpenHexBot

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = yaml.safe_load(Path(args.config).read_text())
    bots = []
    for entry in settings["bots"]:
        if args.only and entry["name"] != args.only:
            continue
        key = os.environ.get(entry["key_env"], "").strip()
        if not key:
            raise SystemExit(f"error: {entry['key_env']} (the API key of {entry['name']}) is not set")
        bots.append(OpenHexBot(entry["spec"], key, entry["mode"], base=settings.get("api", API), device=args.device,
                               state_dir=Path(args.state) / entry["name"], name=entry["name"]))
    if not bots:
        raise SystemExit("error: no bots to run")
    threads = [threading.Thread(target=bot.run, name=bot.name, daemon=True) for bot in bots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 0


def main(argv: list[str] | None = None) -> int:
    from catanrl._puffer import install_signal_handlers

    install_signal_handlers()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    full_argv = ["catanrl", *argv]
    try:
        if args.command == "run":
            return _cmd_run(args, full_argv)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "llm-cost":
            from catanrl.agents.llm.cost import main as cost_main

            return cost_main(args)
        if args.command == "ls":
            return _cmd_ls(args)
        if args.command == "results":
            return _cmd_results(args)
        if args.command == "bot":
            return _cmd_bot(args)
        if args.command == "house":
            return _cmd_house(args)
        if args.command == "openhex":
            return _cmd_openhex(args)
        if args.command == "play":
            from catanrl.bridge.server_bot import play_tui

            play_tui(args.seats.split(","), device=args.device, infer_rows=args.infer_rows, config=args.config,
                     seed=args.seed)
            return 0
        if args.command == "pin":
            from catanrl.runs import pin

            print(pin(args.ref, args.name, args.note))
        elif args.command == "fork":
            from catanrl.runs import fork

            v = fork(args.ref, as_=args.as_, overrides=args.overrides)
            print(f"{v.ref}: catanrl run {v.ref}")
        elif args.command == "finish":
            from catanrl.runs import finish

            finish(args.ref)
        elif args.command == "verify":
            from catanrl.eval.results import verify

            print(json.dumps(verify(_find_record(args.record), games=args.games, force=args.force)))
        elif args.command == "check-runs":
            from catanrl.runs import check_runs

            rows = check_runs(args.pattern)
            for row in rows:
                print(f"{'ok ' if row['ok'] else 'BAD'} {row['run']}@{row['which']} {row.get('ref') or row['error']}")
            return 0 if all(r["ok"] for r in rows) else 1
        elif args.command == "summarize":
            from catanrl.runner import summarize

            print(summarize(args.experiment))
        elif args.command == "validate":
            from catanrl.config import validate_plan

            problems = validate_plan(args.target)
            print("\n".join(problems) or "ok")
            return 1 if problems else 0
        elif args.command == "report-data":
            from catanrl.report import report_data

            print(report_data(args.spec, run_missing=args.run_missing, update=args.update, final=args.final,
                              verify_games=args.verify_games))
        return 0
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # catanrl's own refusals print as one line; anything else is a bug
        from catanatron.registry import SpecError

        if type(error).__module__.startswith("catanrl") or isinstance(error, SpecError):
            print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
