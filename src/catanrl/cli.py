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
    catanrl openhex deploy/openhex.yaml --check              # check the file, pins, code and keys; plays nothing
    catanrl openhex deploy/openhex.yaml --check --keys       # ... and ask OpenHex whose each key is (before a restart)
    catanrl openhex deploy/openhex.yaml --describe NAME      # the registration body for a new version
    catanrl openhex stats deploy/openhex.yaml [--since DATE] # every version's record (OpenHex's public data)
    catanrl openhex release deploy/openhex.yaml TAG [--out DIR]  # the tag's checkpoints as release assets
    catanrl openhex fetch deploy/openhex.yaml [--runs DIR]   # ... downloaded into the runs root (deploy machine)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: ``catanrl openhex ACTION ...``: each action's operands (without one, CONFIG alone: run or check the bots).
OPENHEX_ACTIONS = {"stats": ("CONFIG",), "release": ("CONFIG", "TAG"), "fetch": ("CONFIG",)}


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

    oh = sub.add_parser("openhex", help="keep our bots queued on OpenHex (open-hex.web.app)",
                        description="catanrl openhex CONFIG [--check | --describe NAME], catanrl openhex stats "
                                    "CONFIG, catanrl openhex release CONFIG TAG, or catanrl openhex fetch CONFIG "
                                    "(see catanrl.bridge.openhex_deploy for the file, catanrl.bridge.openhex_release "
                                    "for releases)")
    oh.add_argument("config", help="a YAML file: bots (name, mode, version, spec, key_env, ...), optional api and "
                                   "releases; or 'stats', 'release' or 'fetch' and the file (release: and the tag)")
    oh.add_argument("operands", nargs="*", help=argparse.SUPPRESS)
    oh.add_argument("--state", default="openhex", help="games.jsonl and deploys.jsonl go here, one dir per bot")
    oh.add_argument("--device", default="cpu")
    oh.add_argument("--only", help="run just these bots (comma-separated names)")
    oh.add_argument("--leave", action="store_true",
                    help="leave the queue for each bot and exit (on stop, so OpenHex does not match a dead bot)")
    oh.add_argument("--check", action="store_true",
                    help="check the file, pins, agents, code and keys (set or not), print a row per bot, exit "
                         "0 or 1; contacts no one")
    oh.add_argument("--keys", action="store_true",
                    help="with --check: also ask OpenHex (GET /api/bots/me) whose each key is; queues nothing")
    oh.add_argument("--since", metavar="DATE",
                    help="stats: count games created from DATE (UTC; 2026-10-20 or an ISO time)")
    oh.add_argument("--until", metavar="DATE",
                    help="stats: count games created up to DATE (UTC; a date includes its whole day)")
    oh.add_argument("--describe", metavar="NAME", help="print the POST /api/bots body for this bot and exit")
    oh.add_argument("--out", metavar="DIR", help="release: write the assets here (default: a new temporary directory)")
    oh.add_argument("--runs", metavar="DIR",
                    help="fetch: the runs root to fill (default: the runner's, CATANRL_RUNS_DIR or the checkout's runs/)")
    oh.add_argument("--base-url", help=argparse.SUPPRESS)  # fetch, for tests: BASE/<tag>/<file> instead of GitHub
    oh.add_argument("--allow-unpinned", action="store_true",
                    help="run specs whose checkpoints are not pinned by #sha (a warning instead of a refusal)")
    oh.add_argument("--allow-dirty", action="store_true",
                    help="run play source that is not committed and clean, or not the bot's tag (stamped dirty)")

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
        if "canopy" in slot:
            c = slot["canopy"]
            print(f"{slot['label']:<40} Canopy: {c['decisions']} decisions ({c['per_game_decisions']:.1f}/game, "
                  f"{c['searched']} searched, {c['forced']} forced), {c['per_decision_s']:.2f} s/decision "
                  f"({c['bridge_per_decision_s']:.2f} in the bridge), trees reused {c['reused']}, "
                  f"{c['fallbacks']} fallbacks, {c['refreshed']} refreshed, {c['canopy_illegal']} Canopy-legal "
                  "moves the engine refused")
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
    """Keep the bots queued until told to stop. Two ways to stop: SIGUSR1
    drains (the bots leave their queues, play out the games they are in, and
    the process exits); SIGINT/SIGTERM (Ctrl-C) leaves the queues and exits
    now, forfeiting any game in play. ``--leave`` only leaves the queues."""
    import logging
    import signal
    import threading

    from catanrl.agents.spec import resolve_agent
    from catanrl.bridge import openhex_deploy as deploy
    from catanrl.bridge.openhex import API, OpenHexBot, OpenHexClient

    action = args.config if args.config in OPENHEX_ACTIONS else None
    want = OPENHEX_ACTIONS.get(action, ())
    if len(args.operands) != len(want):
        extra = args.operands[len(want):]
        raise SystemExit(f"error: catanrl openhex {' '.join([action, *want]) if action else 'CONFIG'}"
                         + (f" (unexpected {' '.join(extra)!r})" if extra else ""))
    for option, value, where in (("--out", args.out, "release"), ("--runs", args.runs, "fetch"),
                                 ("--base-url", args.base_url, "fetch")):
        if value and action != where:
            raise SystemExit(f"error: {option} is for catanrl openhex {where}")
    if action in ("release", "fetch"):
        others = [option for option, value in (
            ("--check", args.check), ("--keys", args.keys), ("--describe", args.describe), ("--since", args.since),
            ("--until", args.until), ("--leave", args.leave), ("--allow-unpinned", args.allow_unpinned), ("--allow-dirty", args.allow_dirty),
            ("--only", args.only and action == "release")) if value]
        if others:
            raise SystemExit(f"error: catanrl openhex {action} does not take {', '.join(others)}")
    if action == "release":
        from catanrl.bridge import openhex_release as release

        print(release.format_release(release.release(args.operands[0], args.operands[1], args.out)))
        return 0
    if action == "fetch":
        return _openhex_fetch(args)
    if action == "stats":
        return _openhex_stats(args.operands[0], args.since, args.until)
    if args.since or args.until:
        raise SystemExit("error: --since and --until are for catanrl openhex stats")
    if args.keys and not args.check:
        raise SystemExit("error: --keys goes with --check")
    settings, entries = deploy.load_deploy(args.config)
    if args.describe:
        return _openhex_describe(args, entries)
    if args.check:
        return _openhex_check(args, settings, entries)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("catanrl.openhex")
    running = deploy.selected(entries, args.only)
    if not running:
        raise SystemExit("error: no bots to run")
    api = settings.get("api", API)
    if args.leave:
        for entry in running:
            try:
                OpenHexClient(deploy.read_key(entry), api).leave(entry.mode)
                log.info("%s: left the %s queue", entry.name, entry.mode)
            except Exception as error:  # noqa: BLE001 - leave the others anyway
                log.warning("%s: could not leave the %s queue: %s", entry.name, entry.mode, error)
        return 0
    locks = deploy.lock_bots(args.state, running)  # held until the process ends
    for warning in deploy.check_pins(running, allow_unpinned=args.allow_unpinned):
        log.warning("%s", warning)
    gits, problems = deploy.git_stamps(deploy.checkout_root(), running, allow_dirty=args.allow_dirty,
                                       engine_build=_engine_build())
    for problem in problems:
        log.warning("running anyway (--allow-dirty): %s", problem)
    keys = {entry.name: deploy.read_key(entry) for entry in running}
    # Before any agent loads: a key that is another bot's stops everything (and takes the
    # bots whose keys passed off the queue, where the last process left them).
    deploy.verify_keys(running, {entry.name: OpenHexClient(keys[entry.name], api) for entry in running})
    resolved = {entry.name: resolve_agent(entry.spec, harness="server") for entry in running}
    stamps = deploy.build_stamps(running, resolved, gits)
    stop = threading.Event()
    bots = [OpenHexBot(resolved[entry.name], keys[entry.name], entry.mode, base=api, device=args.device,
                       state_dir=Path(args.state) / entry.name, name=entry.name, stamp=stamps[entry.name],
                       stop=stop)
            for entry in running]
    for bot in bots:
        bot.announce()

    def leave_all() -> None:
        for bot in bots:
            try:
                bot.leave()
            except Exception as error:  # noqa: BLE001 - leave the others anyway
                log.warning("%s: could not leave the %s queue: %s", bot.name, bot.mode, error)

    signal.signal(signal.SIGUSR1, lambda signum, frame: stop.set())
    log.info("pid %d: SIGUSR1 (kill -USR1) drains, Ctrl-C/SIGTERM stops now", os.getpid())
    threads = [threading.Thread(target=bot.run, name=bot.name, daemon=True) for bot in bots]
    for thread in threads:
        thread.start()
    left = False
    try:
        while any(thread.is_alive() for thread in threads):
            if stop.is_set() and not left:
                log.info("draining: leaving the queues, finishing the games in play")
                leave_all()
                left = True
            for thread in threads:
                thread.join(timeout=1.0)
        log.info("every bot stopped")
    except KeyboardInterrupt:
        log.info("stopping now")
        stop.set()
    finally:
        if not left:
            leave_all()
    del locks  # (held while the bots ran)
    return 0


def _openhex_check(args, settings, entries) -> int:
    """``--check``: everything the runner checks before it queues, except OpenHex itself
    (with ``--keys``, OpenHex's ``GET /api/bots/me`` too: never the queue)."""
    from catanrl.agents.spec import resolve_agent
    from catanrl.bridge import openhex_deploy as deploy
    from catanrl.bridge.openhex import API, OpenHexClient, native_for

    problems, warnings = [], []
    running = deploy.selected(entries, args.only)
    pins = deploy.check_pins(running, allow_unpinned=True)
    (warnings if args.allow_unpinned else problems).extend(pins)
    gits, dirt = deploy.git_stamps(deploy.checkout_root(), running, allow_dirty=True, engine_build=_engine_build())
    (warnings if args.allow_dirty else problems).extend(dirt)
    head = ("name", "version", "mode", "spec", "weights", "tag", "retired", "key")
    rows = []
    for entry in entries:
        weights = deploy.weights_of(entry.spec)
        if entry in running:
            try:
                resolved = resolve_agent(entry.spec, harness="server")
                native_for(resolved, device=args.device)  # the network loads and passes its self-test
                weights = ",".join(ref.weights_sha256[:12] for ref in resolved.checkpoints)
            except Exception as error:  # noqa: BLE001 - report every bot's failure, not the first
                problems.append(f"{entry.name}: the agent does not load: {type(error).__name__}: {error}")
                weights = "FAILED"
        key = deploy.key_state(entry)
        if key == "unset" and entry in running:
            problems.append(f"{entry.key_env} ({entry.name}'s key) is not set")
        elif key == "malformed" and entry in running:
            problems.append(f"{entry.key_env} ({entry.name}'s key) is not an OpenHex key (32 lowercase hex digits)")
        elif key == "set" and entry in running and args.keys:
            try:
                deploy.verify_key(entry, OpenHexClient(deploy.read_key(entry), settings.get("api", API)))
                key = "verified"
            except deploy.DeployError as error:
                problems.append(str(error))
                key = "REFUSED"
        rows.append((entry.name, entry.version, entry.mode, entry.spec, weights, entry.tag or "-",
                     "yes" if entry.retired else "no", key if not entry.retired else f"({key})"))
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    for row in [head, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    commit = next(iter(gits.values()), {}).get("describe") if gits else None
    print(f"checkout: {deploy.checkout_root()} at {commit or 'unknown'}; api {settings.get('api', 'default')}")
    for warning in warnings:
        print(f"warning: {warning}")
    for problem in problems:
        print(f"problem: {problem}")
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _engine_build() -> dict:
    import cppanatron

    return dict(cppanatron.build_info())


def _openhex_describe(args, entries) -> int:
    """``--describe NAME``: the registration body for ``POST /api/bots``, contacting no one."""
    from catanrl.bridge import openhex_deploy as deploy

    found = [e for e in entries if e.name == args.describe]
    if not found:
        raise SystemExit(f"error: no bot named {args.describe!r} in {args.config}")
    entry = found[0]
    for warning in deploy.check_pins([entry], allow_unpinned=args.allow_unpinned):
        print(f"warning: {warning}", file=sys.stderr)
    print(json.dumps(deploy.registration(entry, deploy.entry_commit(deploy.checkout_root(), entry)), indent=2))
    return 0


def _openhex_fetch(args) -> int:
    """``fetch``: the checkpoints of the bots that run, from their tags' GitHub releases."""
    from catanrl.bridge import openhex_release as release
    from catanrl.runs import RunError, runs_root

    result = release.fetch(args.operands[0], runs=args.runs, only=args.only, base=args.base_url)
    try:
        default = str(runs_root())
    except RunError:
        default = None
    print(release.format_fetch(result, default))
    return 0 if result["ok"] else 1


def _openhex_stats(config: str, since: str | None = None, until: str | None = None) -> int:
    from catanrl.bridge import openhex_stats as stats
    from catanrl.bridge.openhex_deploy import load_deploy

    window = {}
    for option, text, end in (("since", since, False), ("until", until, True)):
        if text:
            try:
                window[option] = stats.parse_when(text, end=end)
            except ValueError:
                raise SystemExit(f"error: --{option} {text!r}: a date (2026-10-20) or an ISO time") from None
    _, entries = load_deploy(config)
    bots, games = stats.gather(entries, stats.PublicRecords())
    print(stats.format_summary(stats.summarize(entries, bots, games, **window)))
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
