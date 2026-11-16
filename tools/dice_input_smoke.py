"""Smoke az-2p-dice (configs/experiments/az-2p-dice.yaml) on the GPU in a scratch runs directory.

Two parts, then a summary (``summary.json`` in the runs directory too) and a ``PASS`` / ``FAIL``
line; the exit code is 1 on FAIL.

**identity** (no run; ``--skip-identity`` leaves it out). The start,
az-2p-kata/main+1500@e1800, against the dice arm's network at e0 (the config's own model and
``init_from``, so every weight on the dice input is zero), on rows of random base-2p-fr games
(balanced dice; each network reads the same positions at its own input):

* ``torch_cpu``: the two modules on the CPU;
* per device evaluator (``BatchEvaluator`` with ``kernels`` torch, fused and fused16, in the
  ``strict`` and ``fast`` modes the evals and the trainer use): ``logits_equal`` and
  ``values_equal``, bit for bit, the largest differences and ``argmax_equal``. The run needs
  all of them true: the dice arm must start as the start in the trainer's self-play
  (fused16, fast) and in the final's search (fused, strict);
* ``trained``: the same network with random weights on the dice input (std 0.05, a stand-in for
  trained ones), each kernel against ``kernels=torch`` on the dice rows: fused within 1e-3 on
  the logits and 1e-6 on the values, fused16 within 0.25 and 1e-2 (the kernels' own
  tolerances), and ``reads_dice``: the outputs moved from the zero-weight ones, so the kernels do
  read the input.

**self-play**: both arms of the real config for ``--iterations`` (6) each, control then dice,
each in its own process through ``catanrl.runner.run_variant``, with ``train.min_replay``
``--min-replay`` (2,000, so the learner, the kernels' rebuild and the dice probe run within the
smoke; the real 50,000 takes ~6 iterations), ``train.replay_capacity`` 60,000 and no finals
(``--finals N``: the final on its suite's first N boards). Per arm:

* ``leaves_per_s``: the median selfplay/leaves_per_s after the first iteration (main+1500 ran
  ~69k), ``iteration_s``, and ``dice_over_control_leaves_per_s`` (expect >= 0.95: the rows are
  5% wider and the first layer has two more small products);
* ``learned`` iterations and ``finite`` (every train/* loss finite);
* ``e0_is_start``: the arm's e0 checkpoint holds the start's weights exactly (and, dice, zeros
  on the input);
* dice: ``dice/weight_norm`` at the first and last iteration (0 until the learner trains, then
  above 0) and the probe at the last (``dice/probe_kl``, ``dice/probe_flips``,
  ``dice/probe_value_gap``).

The trainer checks its kernels against the module before every iteration
(``SelfPlay.check_kernels``), so the learned iterations also run that check with trained dice
weights.

The runs directory must not be the real one: the start's run (its config, run.json and the e1800
checkpoint, read-only) is copied into it, and fr3's with ``--finals``. W&B is off
(``WANDB_MODE=disabled``, ``meta.wandb`` dropped).

    # GPU (the card alone; ~20-25 min: identity ~1 min, then 2 x 6 iterations at ~84 s plus start-up)
    env -u WANDB_API_KEY WANDB_MODE=disabled uv run python tools/dice_input_smoke.py --runs /tmp/az-dice-smoke
    # CPU check of the tool itself (tiny; the kernels are skipped)
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 env -u WANDB_API_KEY WANDB_MODE=disabled \\
        uv run python tools/dice_input_smoke.py --runs /tmp/az-dice-cpu --iterations 3 --min-replay 1 \\
        --set runtime.device=cpu --set runtime.threads=2 --set train.kernels=torch --set env.num_envs=8 \\
        --set train.steps_per_iteration=40 --set train.sims=8 --set train.fast_sims=2 \\
        --set train.minibatch_size=32 --set train.updates_per_iteration=2 --set train.search_rows=32 \\
        --set env.scenarios.0.turn_limit=12
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

START = ("az-2p-kata", "main+1500", "e001800.pt")
OPPONENT = ("ppo-2p-fr3", "full", "e004350.pt")  # the final's (--finals)
EXPERIMENT = "az-2p-dice"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / f"{EXPERIMENT}.yaml"
ARMS = ("control", "dice")
#: (kernels, mode) of the device evaluators: the evals search with fused in strict, the trainer
#: self-plays with fused16 in fast.
EVALUATORS = (("torch", "strict"), ("torch", "fast"), ("fused", "strict"), ("fused", "fast"), ("fused16", "fast"))
#: kernels against kernels=torch with trained weights: (logits, values) tolerances.
TOLERANCE = {"fused": (1e-3, 1e-6), "fused16": (0.25, 1e-2)}


def _variant(name: str, args: argparse.Namespace):
    """The plan's variant at the smoke's size (its hash recomputed)."""
    from catanrl.config import config_hash, load_plan

    overrides = [f"train.iterations={args.iterations}", f"train.min_replay={args.min_replay}",
                 "train.replay_capacity=60000", *args.overrides]
    plan = load_plan(CONFIG, overrides=overrides)
    v = plan.variant(name)
    cfg = json.loads(json.dumps(v.config))
    cfg["meta"].pop("wandb", None)
    finals = []
    if args.finals > 0:
        for entry in cfg["eval"]["final"]:
            suite = entry["suite"] if isinstance(entry["suite"], dict) else {"name": entry["suite"]}
            finals.append({**entry, "suite": {**suite, "boards": args.finals}})
    cfg["eval"]["final"] = finals
    return plan, dataclasses.replace(v, config=cfg, config_hash=config_hash(cfg))


def _child(name: str, args: argparse.Namespace) -> None:
    from catanrl.runner import run_variant

    plan, v = _variant(name, args)
    status = run_variant(v, argv=["tools/dice_input_smoke.py", *sys.argv[1:]],
                         source={"path": str(CONFIG), "sha256": plan.source_sha256})
    print(f"== {v.ref}: {status}", flush=True)
    if status != "finished":
        raise SystemExit(1)


def real_rows(games: int = 12, every: int = 7):
    """Rows of random base-2p-fr games (balanced dice: unfair odds) at the full level, the same
    positions without and with the dice input: (plain, dice)."""
    import numpy as np

    from catanrl.envs.encoding import dims_for, make_encoder
    from catanrl.games import load_game

    config = load_game("base-2p-fr")
    dims = dims_for([config])
    encoders = make_encoder(dims, "full"), make_encoder(dims, "full", True)
    rows: tuple[list, list] = ([], [])
    for seed in range(games):
        game = config.cppanatron_game(seed, 1000 + seed)
        rng = random.Random(seed)
        for step in range(600):
            if game.winning_color is not None:
                break
            if step % every == 3:
                for kept, encoder in zip(rows, encoders):
                    kept.append(encoder.encode(game, game.current_player_index))
            game.execute(rng.choice(game.playable_actions))
    return np.stack(rows[0]), np.stack(rows[1])


def _networks(args: argparse.Namespace):
    """(the start's module, the dice arm's at e0 through the config's own init_from, dice rows'
    columns that are not dice features)."""
    import numpy as np

    from catanrl import checkpoint as ck
    from catanrl.runs import load_checkpoint
    from catanrl.train.dice_probe import FairDice
    from catanrl.train.puffer import build_policy_model

    _, v = _variant("dice", args)
    start = load_checkpoint(v.config["init_from"]["ckpt"]).model("all").eval()
    model = build_policy_model(v.config, start.dims).eval()
    ck.load_parts(model, v.config["init_from"])
    plain = np.setdiff1d(np.arange(model.obs_size), FairDice(model).columns)
    return start, model, plain


def _compare(a, b, legal) -> dict:
    """Two (logits, values) results, numpy, over the legal actions."""
    import numpy as np

    (la, va), (lb, vb) = a, b
    return {"logits_equal": bool(np.array_equal(la[legal], lb[legal])), "values_equal": bool(np.array_equal(va, vb)),
            "logit_max_diff": float(np.abs(la - lb)[legal].max()), "value_max_diff": float(np.abs(va - vb).max()),
            "argmax_equal": bool((np.where(legal, la, -np.inf).argmax(1) == np.where(legal, lb, -np.inf).argmax(1)).all())}


def _identity(args: argparse.Namespace) -> dict:
    import copy

    import numpy as np
    import torch

    from catanrl import checkpoint as ck
    from catanrl.search.evaluator import BatchEvaluator

    start, model, plain = _networks(args)
    plain_rows, dice_rows = real_rows()
    assert np.array_equal(dice_rows[:, plain], plain_rows), "the dice rows are not the plain rows plus the dice columns"
    legal = plain_rows[:, start.mask_offset : start.mask_offset + start.num_actions] > 0
    out: dict = {"rows": int(len(plain_rows)), "row_width": [int(start.obs_size), int(model.obs_size)],
                 "dice_rows_differ_from_fresh": bool((dice_rows[:, np.setdiff1d(np.arange(model.obs_size), plain)]
                                                      != dice_rows[:1, np.setdiff1d(np.arange(model.obs_size), plain)]).any())}

    def module(net, rows):
        with torch.no_grad():
            logits, values = net(torch.from_numpy(rows))
        return logits.numpy(), values.numpy()

    out["torch_cpu"] = _compare(module(start, plain_rows), module(model, dice_rows), legal)
    ok = out["torch_cpu"]["logits_equal"] and out["torch_cpu"]["values_equal"]
    if not torch.cuda.is_available():
        out["device"] = {"skipped": "no CUDA device"}
        out["ok"] = bool(ok)
        return out
    trained = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for key in ck.dice_input_keys(trained):
            weight = trained.state_dict()[key]
            weight.copy_(torch.randn(weight.shape, generator=generator) * 0.05)
    out["device"], out["trained"] = {}, {}
    reference = None
    for kernels, mode in EVALUATORS:
        name = f"{kernels}/{mode}"

        def evaluator(net):
            return BatchEvaluator(copy.deepcopy(net), device="cuda", infer_rows=512, mode=mode, kernels=kernels)

        try:
            was = evaluator(start)(plain_rows)
            now = evaluator(model)(dice_rows)
            moved = evaluator(trained)(dice_rows)
        except (RuntimeError, ValueError, OSError) as error:  # nvcc or ninja missing, a kernel refusing
            out["device"][name] = {"failed": str(error)[:300]}
            ok = False
            continue
        was, now, moved = (tuple(np.array(x, dtype=np.float32) for x in result) for result in (was, now, moved))
        out["device"][name] = _compare(was, now, legal)
        ok = ok and out["device"][name]["logits_equal"] and out["device"][name]["values_equal"]
        if (kernels, mode) == ("torch", "strict"):
            reference = moved
        entry = {"reads_dice": bool(np.abs(moved[1] - now[1]).max() > 0 and np.abs(moved[0] - now[0])[legal].max() > 0)}
        if kernels in TOLERANCE and reference is not None:
            gap = _compare(reference, moved, legal)
            logits, values = TOLERANCE[kernels]
            entry.update(logit_max_diff=gap["logit_max_diff"], value_max_diff=gap["value_max_diff"],
                         argmax_equal=gap["argmax_equal"],
                         within=bool(gap["logit_max_diff"] <= logits and gap["value_max_diff"] <= values))
            ok = ok and entry["within"]
        out["trained"][name] = entry
        ok = ok and entry["reads_dice"]
    out["ok"] = bool(ok)
    return out


def _e0_is_start(name: str) -> dict:
    """The arm's e0 checkpoint holds the start's weights (dice: and zeros on the input)."""
    import torch

    from catanrl import checkpoint as ck
    from catanrl.runs import load_checkpoint, load_experiment

    run = load_experiment(f"{EXPERIMENT}/{name}")
    e0 = ck.Checkpoint.load(run.checkpoints()[0].path)
    model = e0.model("all")
    theirs, mine = load_checkpoint(run.config["init_from"]["ckpt"]).model("all").state_dict(), model.state_dict()
    extra = sorted(set(mine) - set(theirs))
    return {"weights_equal": all(torch.equal(mine[k], theirs[k]) for k in theirs), "new_keys": extra,
            "new_keys_zero": all(not bool(mine[k].any()) for k in extra),
            "dice_state": bool(e0.dice_state), "row_width": int(model.obs_size)}


def _summary(name: str) -> dict:
    from catanrl.runs import load_experiment

    rows = load_experiment(f"{EXPERIMENT}/{name}").metrics()
    late = rows[1:] or rows
    losses = [v for r in rows for k, v in r.items() if k in ("train/policy_loss", "train/value_loss")]
    e0 = _e0_is_start(name)
    out = {"iterations": len(rows),
           "leaves_per_s": statistics.median(r["selfplay/leaves_per_s"] for r in late),
           "iteration_s": statistics.median(r["throughput/iteration_s"] for r in late),
           "games_per_hour": statistics.median(r["selfplay/games_per_hour"] for r in late),
           "replay": rows[-1]["replay"], "learned": sum("train/policy_loss" in r for r in rows),
           "finite": all(isinstance(v, (int, float)) and math.isfinite(v) for v in losses),
           "e0_is_start": e0}
    ok = out["finite"] and out["learned"] > 0 and e0["weights_equal"] and e0["new_keys_zero"]
    if name == "dice":
        probed = [r for r in rows if "dice/probe_kl" in r]
        out.update(dice_weight_norm_first=rows[0].get("dice/weight_norm"),
                   dice_weight_norm_last=rows[-1].get("dice/weight_norm"),
                   dice_weight_norm_global_last=rows[-1].get("dice/weight_norm_global"),
                   dice_weight_norm_tiles_last=rows[-1].get("dice/weight_norm_tiles"),
                   probe_kl_last=probed[-1]["dice/probe_kl"] if probed else None,
                   probe_flips_last=probed[-1]["dice/probe_flips"] if probed else None,
                   probe_value_gap_last=probed[-1]["dice/probe_value_gap"] if probed else None,
                   probe_rows=probed[-1]["dice/probe_rows"] if probed else None)
        ok = ok and e0["dice_state"] and len(e0["new_keys"]) == 2 and bool(probed) \
            and (out["dice_weight_norm_last"] or 0.0) > 0.0
    else:
        ok = ok and not e0["new_keys"] and not e0["dice_state"]
    out["ok"] = bool(ok)
    return out


def _copy_run(real: Path, scratch: Path, run: tuple[str, str, str]) -> None:
    experiment, variant, ckpt = run
    source, target = real / experiment / variant, scratch / experiment / variant
    if (target / "checkpoints" / ckpt).exists():
        return
    (target / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "run.json", "aliases.jsonl"):
        if (source / name).exists():
            shutil.copyfile(source / name, target / name)
    shutil.copyfile(source / "checkpoints" / ckpt, target / "checkpoints" / ckpt)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", required=True, help="a scratch runs directory (created; never the real one)")
    parser.add_argument("--real-runs", default=None, help="where to copy the start's run from (default: the main "
                        "worktree's runs/)")
    parser.add_argument("--iterations", type=int, default=6, help="iterations per arm (train.iterations)")
    parser.add_argument("--min-replay", type=int, default=2000, help="train.min_replay (the real 50,000 takes ~6 "
                        "iterations to reach)")
    parser.add_argument("--finals", type=int, default=0, help="the final on its suite's first N boards (0: none)")
    parser.add_argument("--skip-identity", action="store_true", help="leave out the identity part")
    parser.add_argument("--only", default=None, help="arms to run (comma-separated: control,dice; 'none': the "
                        "identity part alone)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    os.environ["WANDB_MODE"] = "disabled"
    for key in ("WANDB_API_KEY", "CATANRL_WANDB_PROJECT", "CATANRL_WANDB_ENTITY"):
        os.environ.pop(key, None)
    scratch = Path(args.runs).resolve()
    if args.child is not None:
        os.environ["CATANRL_RUNS_DIR"] = str(scratch)
        _child(args.child, args)
        return

    os.environ.pop("CATANRL_RUNS_DIR", None)
    from catanrl.runs import runs_root

    real = Path(args.real_runs).resolve() if args.real_runs else runs_root()
    if scratch == real or real in scratch.parents or scratch in real.parents:
        raise SystemExit(f"--runs {scratch} overlaps the real runs directory {real}")
    _copy_run(real, scratch, START)
    if args.finals > 0:
        _copy_run(real, scratch, OPPONENT)
    os.environ["CATANRL_RUNS_DIR"] = str(scratch)

    summary: dict = {}
    if not args.skip_identity:
        summary["identity"] = _identity(args)
        print(json.dumps({"identity": summary["identity"]}, indent=1), flush=True)
    arms = [a for a in ARMS if args.only is None or a in args.only.split(",")]
    for arm in arms:
        child = [sys.executable, "-P", __file__, *(argv if argv is not None else sys.argv[1:]), "--child", arm]
        if subprocess.run(child, env=os.environ.copy(), check=False).returncode != 0:
            raise SystemExit(f"{arm} failed")
    summary.update({arm: _summary(arm) for arm in arms})
    if set(ARMS) <= set(summary):
        summary["dice_over_control_leaves_per_s"] = summary["dice"]["leaves_per_s"] / summary["control"]["leaves_per_s"]
    ok = all(part["ok"] for part in summary.values() if isinstance(part, dict))
    summary["ok"] = bool(ok)
    (scratch / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
