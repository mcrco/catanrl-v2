"""``catanrl``: run experiments, evaluate specs, and look after the runs store.

::

    catanrl run arch-lineage [--only 'graph-s-*'] [--set train.gamma=1.0] [--smoke] [--next]
    catanrl run arch-lineage/graph-s-s42.g1+more        # a forked variant, from its stored config
    catanrl eval POLICY:arch-lineage/graph-s-s42.g1 F --suite heldout-2p
    catanrl ls ['arch-lineage/*'] [--status running]
    catanrl pin arch-lineage/graph-s-s42.g1@e600 ref --note "report"
    catanrl fork arch-lineage/graph-s-s42.g1 --as graph-s-s42.g1+more --set train.total_timesteps=2e9
    catanrl finish arch-lineage/graph-s-s42.g1
    catanrl results ['arch-lineage/*'] [--stale]        catanrl verify EVAL_ID [--games 16]
    catanrl check-runs ['*']    catanrl summarize arch-lineage    catanrl validate arch-lineage
    catanrl report-data [report/results.yaml] [--run-missing] [--update] [--final]
"""

from __future__ import annotations

import argparse
import json
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
    ev.add_argument("--device")
    ev.add_argument("--num-envs", type=int)
    ev.add_argument("--threads", type=int)
    ev.add_argument("--infer-rows", type=int)
    ev.add_argument("--allow-mixed", action="store_true")

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
    rep.add_argument("--final", action="store_true", help="refuse dagger rows and re-play games of each record")
    rep.add_argument("--verify-games", type=int, default=16)
    return p


def _find_record(text: str):
    from catanrl.eval.results import EvalRecord
    from catanrl.runs import runs_root

    path = Path(text)
    if path.exists():
        return EvalRecord.load(path)
    matches = sorted(runs_root().glob(f"**/evals/*/{text}*.json"))
    if len(matches) != 1:
        raise SystemExit(f"error: {len(matches)} records match {text!r}")
    return EvalRecord.load(matches[0])


def _cmd_run(args, argv) -> int:
    from catanrl import runner
    from catanrl.config import load_plan
    from catanrl.runs import load_experiment

    target = Path(args.target)
    if "/" in args.target and not target.exists() and not args.target.endswith(".yaml"):
        run = load_experiment(args.target)  # a stored variant (e.g. a fork)
        print(runner.run_variant(run.variant(), argv=argv, allow_code_change=args.allow_code_change))
        return 0
    plan = load_plan(args.target, overrides=args.overrides, only=args.only, variant=args.variant, smoke=args.smoke)
    if args.inline:
        return runner.run_inline(plan, args.variant, argv=argv, allow_code_change=args.allow_code_change)
    runner.run_plan(plan, next_=args.next_, allow_code_change=args.allow_code_change, overrides=args.overrides)
    return 0


def _cmd_eval(args) -> int:
    from catanrl.eval.results import evaluate

    suite = {"name": args.suite, "boards": args.boards} if args.boards else args.suite
    record = evaluate(args.lineup, suite, fill=args.fill, device=args.device, num_envs=args.num_envs,
                      threads=args.threads, infer_rows=args.infer_rows, allow_mixed=args.allow_mixed)
    s = record.summary
    for slot in s["slots"]:
        lo, hi = slot["wilson"]
        print(f"{slot['label']:<40} {100 * slot['score_rate']:6.2f}% [{100 * lo:.2f}, {100 * hi:.2f}] "
              f"(n={slot['games']})")
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
        print(f"{run.ref:<45} {str(run.status):<12} {('e%d' % ckpts[-1].epoch) if ckpts else '-':>8}  "
              f"{best_text}  {'; '.join(flags)}")
    return 0


def _cmd_results(args) -> int:
    from catanrl.eval.results import EvalRecord, stale
    from catanrl.runs import runs_root

    if args.stale:
        records = stale(args.pattern)
    else:
        root = runs_root()
        pattern = args.pattern if "/" in args.pattern else f"{args.pattern}/*"
        records = [EvalRecord.load(p) for p in sorted(root.glob(f"{pattern}/evals/*/*.json"))]
    for r in records:
        print(f"{r.eval_id}  {r.data['suite']:<16} {r.summary['score_rate']:.3f} n={r.summary['games']:<5} "
              f"{' vs '.join(r.data['labels'])}")
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
        if args.command == "ls":
            return _cmd_ls(args)
        if args.command == "results":
            return _cmd_results(args)
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

            print(json.dumps(verify(_find_record(args.record), games=args.games)))
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
