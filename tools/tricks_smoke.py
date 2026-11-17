"""Smoke az-2p-tricks (configs/experiments/az-2p-tricks.yaml) on the GPU in a scratch runs directory.

Two parts, then a summary (``summary.json`` in the runs directory too), a ``warnings`` line (the
reported checks that are not part of the verdict and came out false) and a ``PASS`` / ``FAIL``
line; the exit code is 1 on FAIL.

**identity** (no run; ``--skip-identity`` leaves it out). The start, az-2p-bundle/main@e900,
against the run's network at e0 (the config's own model and ``init_from``: every weight of the
start, its outcome head included, and the soft policy head, a copy of the policy head), on rows
of random base-2p-fr games (balanced dice; both networks read the same rows):

* ``weights``: every tensor of the start is in the run's network, and the only other keys are
  the soft head's, each equal to the policy head's;
* ``torch_cpu``: the two modules on the CPU, policy logits and values bit for bit;
  ``outcome_head``: the outcome head's outputs bit for bit (``init_from.parts`` carries it);
  ``soft_head``: its logits equal the policy head's, and with random weights on it
  (``moved``: a stand-in for a trained soft head) the policy logits and values stay the same
  bits while the soft logits move;
* per device evaluator (``BatchEvaluator`` with ``kernels`` torch, fused and fused16, in the
  ``strict`` and ``fast`` modes the evals and the trainer use): ``logits_equal`` and
  ``values_equal`` against the start's, bit for bit, the largest differences and
  ``argmax_equal``; and ``soft_head_ignored``: the same against the network with the moved soft
  head. The run needs the kernels' true: they must build and run with the soft head in the
  network, start as e900 in the trainer's self-play (fused16, fast) and in the final's search
  (fused, strict), and never read the soft head. ``kernels=torch`` on the device is held to
  rounding (``within``: 1e-4 on the logits, 1e-5 on the values, the same best action), as
  tools/bundle_smoke.py holds it. Without a CUDA device only ``kernels=torch`` runs, on the CPU.

**self-play**: the real config's one variant for ``--iterations`` (8) in its own process through
``catanrl.runner.run_variant``, with ``train.min_replay`` ``--min-replay`` (2,000, so the learner
runs within the smoke; the real 50,000 takes ~6 iterations), ``train.replay_capacity`` 60,000,
``train.value_blend_iterations`` ``--blend-iterations`` (4, NOT the run's 60: the search value's
share of the value target reaches its 0.5 within the smoke) and no final (``--finals N``: the
final on its suite's first N boards). A progress line per iteration, with the time so far and an
ETA. Read (the bundle's own numbers beside them come from its metrics.jsonl in the real runs
directory, read-only):

* ``speed``: the median selfplay/leaves_per_s and throughput/iteration_s after the first
  iteration, each over the bundle's median (it ran 42.7 iterations an hour). Reported: judge it.
* ``decided`` (selfplay/decided), ``sims_per_decision``, ``positions`` (recorded positions an
  iteration; the bundle's ~8,180) and ``games`` finished an iteration;
* ``learned`` iterations and ``finite``: every train/* number of every learned iteration;
* ``soft``: train/soft_policy_loss and train/soft_target_entropy at the first and last learned
  iteration (``logged``: both are there);
* ``value_blend``: train/value_blend per learned iteration (``rises_to``: it ends at
  train.value_search_blend without falling) and train/search_value_gap;
* ``surprise``: train/surprise_ess, surprise_sampled_max and surprise_sampled_mean per learned
  iteration. ``limit`` is the largest N x p over the buffer the minibatches were drawn from and
  ``cap_limit`` what the trainer's definition allows a capped position: (1 - s) + s x cap x
  (the mean surprise / the mean after the cap), which is (1 - s) + s x cap = 5.5 only when the
  cap moves the mean by nothing. ``respects_cap``: sampled_max <= limit <= cap_limit each time;
* ``outcome``: the outcome losses and their errors in victory points and in turns at the first
  and the last learned iteration, beside the bundle's first learned iteration (a fresh head:
  7.1 points, 22 turns) and its last (1.6 points, 6.7 turns). ``starts_low``: both first errors
  are nearer the bundle's last than its first, as a carried head's are;
* ``root_exploration``: the trainer logs nothing of forced playouts or the root temperature, so
  the child reads ``SelfPlay.options``: the full search's carry train.forced_playouts and
  train.root_policy_temperature (``as_planned``: 2.0 and 1.25), the fast one's and a decided
  game's fast one's carry 0 and 1 (off). target/pruned and search/target_entropy beside the
  bundle's are indirect signs (forced visits are pruned out of the target again);
* ``e0_is_start``: the e0 checkpoint holds the start's weights exactly, outcome head included,
  and its only other keys are the soft head's, equal to the policy head's;
* ``kernel_checks``: ``SelfPlay.check_kernels`` returned once per iteration (it raises, and the
  run fails, when the kernels are off the module; ``kernels: torch`` has nothing to check).

PASS needs the identity rules, ``finite``, at least two learned iterations, ``e0_is_start`` and
``root_exploration``. The rest is reported (and listed under ``warnings`` when false).

The runs directory must be a new one, and not the real one: the start's run (its config,
run.json and the e900 checkpoint, read-only) is copied into it, and fr3's with ``--finals``. W&B
is off (``WANDB_MODE=disabled``, ``meta.wandb`` dropped). The fused kernels build with nvcc and
ninja, which must be on PATH (ninja is not in the venv).

    # GPU (the card alone; ~15 min: identity ~2 min with the kernels' build, then 8 iterations
    # at <= 84 s plus start-up)
    PATH=<a directory with ninja>:/opt/cuda/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        env -u WANDB_API_KEY WANDB_MODE=disabled uv run python tools/tricks_smoke.py --runs /tmp/az-tricks-smoke
    # CPU check of the tool itself (tiny; the fused kernels are skipped)
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 env -u WANDB_API_KEY WANDB_MODE=disabled \\
        uv run python tools/tricks_smoke.py --runs /tmp/az-tricks-cpu --iterations 6 --min-replay 1 \\
        --set runtime.device=cpu --set runtime.threads=2 --set train.kernels=torch --set env.num_envs=8 \\
        --set train.steps_per_iteration=300 --set train.sims=8 --set train.fast_sims=4 --set train.pending=4 \\
        --set train.fast_pending=2 --set train.decided_fast_sims=2 --set train.minibatch_size=32 \\
        --set train.updates_per_iteration=8 --set train.search_rows=32
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

START = ("az-2p-bundle", "main", "e000900.pt")
OPPONENT = ("ppo-2p-fr3", "full", "e004350.pt")  # the final's (--finals)
EXPERIMENT = "az-2p-tricks"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / f"{EXPERIMENT}.yaml"
VARIANT = "main"
#: What the child process saw of the trainer (:func:`_instrument`), in the runs directory.
PROBE = "tricks_probe.json"
#: (kernels, mode) of the device evaluators: the evals search with fused in strict, the trainer
#: self-plays with fused16 in fast.
EVALUATORS = (("torch", "strict"), ("torch", "fast"), ("fused", "strict"), ("fused", "fast"), ("fused16", "fast"))
#: kernels=torch on the device, the start against e0: rounding (tools/bundle_smoke.py), not bit
#: for bit; the run plays through the kernels, which are.
MODULE_TOLERANCE = (1e-4, 1e-5)
#: The run's root exploration (the config's train.forced_playouts, train.root_policy_temperature)
#: and the values that turn each off.
PLANNED = {"forced_playouts": 2.0, "root_policy_temperature": 1.25}
OFF = {"forced_playouts": 0.0, "root_policy_temperature": 1.0}
OUTCOME = ("train/outcome_vp_loss", "train/outcome_turns_loss", "train/outcome_vp_mae", "train/outcome_turns_mae")


def _overrides(args: argparse.Namespace) -> list[str]:
    """The smoke's size, then the caller's ``--set`` (which win)."""
    return [f"train.iterations={args.iterations}", f"train.min_replay={args.min_replay}",
            "train.replay_capacity=60000", f"train.value_blend_iterations={args.blend_iterations}", *args.overrides]


def _variant(args: argparse.Namespace):
    """The plan's variant at the smoke's size (its hash recomputed)."""
    from catanrl.config import config_hash, load_plan

    plan = load_plan(CONFIG, overrides=_overrides(args))
    v = plan.variant(VARIANT)
    cfg = json.loads(json.dumps(v.config))
    cfg["meta"].pop("wandb", None)
    finals = []
    if args.finals > 0:
        for entry in cfg["eval"]["final"]:
            suite = entry["suite"] if isinstance(entry["suite"], dict) else {"name": entry["suite"]}
            finals.append({**entry, "suite": {**suite, "boards": args.finals}})
    cfg["eval"]["final"] = finals
    return plan, dataclasses.replace(v, config=cfg, config_hash=config_hash(cfg))


def _instrument(path: Path) -> None:
    """In the child, before the run: write to ``path`` what the trainer's self-play was built
    with (``SelfPlay.options`` and its pools' configs, a list per group of pools: nothing logs
    the root exploration), how
    often ``SelfPlay.check_kernels`` returned, and before each ``learn()`` the largest N x p of
    the replay's sampling distribution (train/surprise_sampled_max is the largest one drawn).
    Each wrapper calls the trainer's own and changes nothing it computes."""
    import numpy as np

    from catanrl.train import alphazero as az

    seen: dict = {"kernel_checks": 0, "surprise": []}
    init, check, learn = az.SelfPlay.__init__, az.SelfPlay.check_kernels, az.learn

    def write() -> None:
        path.write_text(json.dumps(seen, indent=1) + "\n")

    def exploration(config) -> dict:
        return {"forced_playouts": float(config.forced_playouts),
                "root_policy_temperature": float(config.root_policy_temperature),
                "simulations": int(config.simulations), "dirichlet_alpha": float(config.dirichlet_alpha)}

    def __init__(self, *args, **kwargs) -> None:
        init(self, *args, **kwargs)
        seen["selfplay"] = {"reuse": bool(self.reuse), "kernels": str(self.train["kernels"]), "mode": str(self.mode),
                            "device": str(self.device), "groups": len(self.groups),
                            "full": dict(self.options[0]), "fast": dict(self.options[1]),
                            "decided": dict(self.decided_options),
                            "pools": [[exploration(pool.config) for pool in pools] for pools in self.groups]}
        write()

    def check_kernels(self, *args, **kwargs) -> None:
        check(self, *args, **kwargs)
        seen["kernel_checks"] += 1
        write()

    def learned(model, optimizer, replay, train, *args, **kwargs):
        weight, cap = float(train.get("surprise_weight", 0.0)), float(train.get("surprise_cap", 0.0))
        if weight > 0.0 and replay.size:
            surprise = np.asarray(replay.surprise[: replay.size], dtype=np.float64)
            known = ~np.isnan(surprise)
            filled = np.where(known, surprise, surprise[known].mean() if known.any() else 0.0)
            capped = np.minimum(filled, cap * filled.mean()) if cap > 0.0 else filled
            p = az.surprise_probabilities(replay.surprise[: replay.size], weight, cap)
            seen["surprise"].append({"rows": int(replay.size), "limit": float(p.max() * replay.size),
                                     "mean": float(filled.mean()), "capped_mean": float(capped.mean()),
                                     "at_cap": float((filled > capped).mean())})
            write()
        return learn(model, optimizer, replay, train, *args, **kwargs)

    az.SelfPlay.__init__, az.SelfPlay.check_kernels, az.learn = __init__, check_kernels, learned
    write()


def _child(args: argparse.Namespace, scratch: Path) -> None:
    from catanrl.runner import run_variant

    plan, v = _variant(args)
    _instrument(scratch / PROBE)
    status = run_variant(v, argv=["tools/tricks_smoke.py", *sys.argv[1:]],
                         source={"path": str(CONFIG), "sha256": plan.source_sha256})
    print(f"== {v.ref}: {status}", flush=True)
    if status != "finished":
        raise SystemExit(1)


def real_rows(games: int = 12, every: int = 7):
    """Rows of random base-2p-fr games (balanced dice: unfair odds) at the full level with the
    dice input, as the start and the run both read them."""
    import numpy as np

    from catanrl.envs.encoding import dims_for, make_encoder
    from catanrl.games import load_game

    config = load_game("base-2p-fr")
    encoder = make_encoder(dims_for([config]), "full", True)
    rows = []
    for seed in range(games):
        game = config.cppanatron_game(seed, 1000 + seed)
        rng = random.Random(seed)
        for step in range(600):
            if game.winning_color is not None:
                break
            if step % every == 3:
                rows.append(encoder.encode(game, game.current_player_index))
            game.execute(rng.choice(game.playable_actions))
    return np.stack(rows)


def _start_ref(init_from) -> str:
    """The ref ``init_from`` takes its actor from (one ref, or one per part)."""
    ckpt = init_from["ckpt"]
    return str(ckpt if isinstance(ckpt, str) else ckpt["actor"])


def _weights(theirs, mine) -> dict:
    """The start's state against the run's: every tensor of the start, the outcome head's among
    them, and no other keys than the soft head's, each the policy head's tensor."""
    import torch

    extra = sorted(set(mine) - set(theirs))
    soft = [k for k in extra if k.startswith("soft_policy_head.")]
    head = [k for k in theirs if k.startswith("outcome_head.")]
    return {"weights_equal": all(k in mine and torch.equal(mine[k], theirs[k]) for k in theirs),
            "outcome_head_keys": len(head),
            "outcome_head_equal": bool(head) and all(k in mine and torch.equal(mine[k], theirs[k]) for k in head),
            "soft_head_keys": len(soft), "other_new_keys": [k for k in extra if k not in soft],
            "soft_head_is_policy_head": bool(soft) and all(
                torch.equal(mine[k], mine[k.removeprefix("soft_")]) for k in soft)}


def _weights_ok(weights: dict) -> bool:
    return bool(weights["weights_equal"] and weights["outcome_head_equal"] and weights["soft_head_is_policy_head"]
                and not weights["other_new_keys"])


def _networks(args: argparse.Namespace):
    """(the start's module, the run's at e0 through the config's own init_from)."""
    from catanrl import checkpoint as ck
    from catanrl.runs import load_checkpoint
    from catanrl.train.puffer import build_policy_model

    _, v = _variant(args)
    start = load_checkpoint(_start_ref(v.config["init_from"])).model("all").eval()
    model = build_policy_model(v.config, start.dims).eval()
    ck.load_parts(model, v.config["init_from"])
    return start, model


def _compare(a, b, legal) -> dict:
    """Two (logits, values) results, numpy, over the legal actions."""
    import numpy as np

    (la, va), (lb, vb) = a, b
    return {"logits_equal": bool(np.array_equal(la[legal], lb[legal])), "values_equal": bool(np.array_equal(va, vb)),
            "logit_max_diff": float(np.abs(la - lb)[legal].max()), "value_max_diff": float(np.abs(va - vb).max()),
            "argmax_equal": bool((np.where(legal, la, -np.inf).argmax(1) == np.where(legal, lb, -np.inf).argmax(1)).all())}


def _within(gap: dict) -> bool:
    return bool(gap["logit_max_diff"] <= MODULE_TOLERANCE[0] and gap["value_max_diff"] <= MODULE_TOLERANCE[1]
                and gap["argmax_equal"])


def _identity(args: argparse.Namespace) -> dict:
    import copy
    import gc

    import numpy as np
    import torch

    from catanrl.search.evaluator import BatchEvaluator

    start, model = _networks(args)
    rows = real_rows()
    tensor = torch.from_numpy(rows)
    legal = rows[:, start.mask_offset : start.mask_offset + start.num_actions] > 0
    # the same network with random weights on the soft head: nothing that plays may read them
    moved = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for key, weight in moved.state_dict().items():
            if key.startswith("soft_policy_head.") and weight.is_floating_point():
                weight.add_(torch.randn(weight.shape, generator=generator) * 0.05)
    out: dict = {"rows": len(rows), "row_width": [int(start.obs_size), int(model.obs_size)],
                 "weights": _weights(start.state_dict(), model.state_dict())}

    def module(net):
        with torch.no_grad():
            logits, values = net(tensor)
        return logits.numpy(), values.numpy()

    out["torch_cpu"] = _compare(module(start), module(model), legal)
    with torch.no_grad():
        was, now = start.heads_with_outcome(tensor), model.heads_with_outcome(tensor)
        logits, _, _, extras = model.training_heads(tensor)
        moved_logits, _, _, moved_extras = moved.training_heads(tensor)
    out["torch_cpu"]["value_logits_equal"] = bool(torch.equal(was[1], now[1]))
    (vp_was, turns_was), (vp, turns) = was[3:], now[3:]
    out["outcome_head"] = {"carried": bool(start.outcome and model.outcome), "vp_shape": list(vp.shape),
                           "turns_shape": list(turns.shape),
                           "vp_equal": bool(torch.equal(vp, vp_was)), "turns_equal": bool(torch.equal(turns, turns_was)),
                           "extras_equal": bool(torch.equal(extras["final_vp"], vp_was)
                                                and torch.equal(extras["turns_left"], turns_was)),
                           "finite": bool(torch.isfinite(vp).all() and torch.isfinite(turns).all()),
                           "vp_abs_max": float(vp.abs().max()), "turns_abs_max": float(turns.abs().max())}
    mask = torch.from_numpy(legal)
    soft_gap = (moved_extras["soft_policy"] - moved_logits).abs()[mask].max()
    out["soft_head"] = {"present": bool(model.soft_policy and not start.soft_policy),
                        "equals_policy": bool(torch.equal(extras["soft_policy"], logits)),
                        "policy_equals_start": bool(torch.equal(logits, was[0])),
                        "moved": {**_compare(module(model), module(moved), legal),
                                  "soft_logit_max_diff": float(soft_gap)}}
    head, soft = out["outcome_head"], out["soft_head"]
    ok = (_weights_ok(out["weights"]) and out["torch_cpu"]["logits_equal"] and out["torch_cpu"]["values_equal"]
          and out["torch_cpu"]["value_logits_equal"] and head["carried"] and head["vp_equal"] and head["turns_equal"]
          and head["extras_equal"] and head["finite"] and soft["present"] and soft["equals_policy"]
          and soft["policy_equals_start"] and soft["moved"]["logits_equal"] and soft["moved"]["values_equal"]
          and soft["moved"]["soft_logit_max_diff"] > 0.0)

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    out["device"] = {"name": device}
    if not cuda:
        out["device"]["skipped"] = "no CUDA device: kernels=fused and fused16 did not run"

    def evaluate(net, kernels: str, mode: str):
        evaluator = BatchEvaluator(copy.deepcopy(net), device=device, infer_rows=512, mode=mode, kernels=kernels)
        return tuple(np.array(x, dtype=np.float32) for x in evaluator(rows))

    for kernels, mode in EVALUATORS:
        if not cuda and kernels != "torch":
            continue
        name = f"{kernels}/{mode}"
        try:
            was, now, other = (evaluate(net, kernels, mode) for net in (start, model, moved))
        except (RuntimeError, ValueError, OSError) as error:  # nvcc or ninja missing, a kernel refusing
            out["device"][name] = {"failed": str(error)[:300]}
            ok = False
            continue
        same, ignored = _compare(was, now, legal), _compare(now, other, legal)
        same["soft_head_ignored"] = {key: ignored[key] for key in ("logits_equal", "values_equal", "logit_max_diff",
                                                                   "value_max_diff")}
        if kernels == "torch":
            same["within"] = _within(same) and _within(ignored)
            ok = ok and same["within"]
        else:
            ok = ok and same["logits_equal"] and same["values_equal"] and ignored["logits_equal"] and ignored["values_equal"]
        out["device"][name] = same
    if cuda:  # the child's run gets the card to itself
        gc.collect()
        torch.cuda.empty_cache()
    out["ok"] = bool(ok)
    return out


def _e0_is_start() -> dict:
    """The e0 checkpoint holds the start's weights, outcome head included, and the soft head as
    the policy head's copy."""
    from catanrl import checkpoint as ck
    from catanrl.runs import load_checkpoint, load_experiment

    run = load_experiment(f"{EXPERIMENT}/{VARIANT}")
    first = run.checkpoints()[0]
    e0 = ck.Checkpoint.load(first.path)
    model = e0.model("all")
    theirs = load_checkpoint(_start_ref(run.config["init_from"])).model("all").state_dict()
    out = {"checkpoint": Path(first.path).name, **_weights(theirs, model.state_dict()),
           "outcome": bool(model.outcome), "soft_policy": bool(model.soft_policy), "dice_state": bool(e0.dice_state),
           "row_width": int(model.obs_size)}
    out["ok"] = bool(out["checkpoint"] == "e000000.pt" and _weights_ok(out) and out["outcome"] and out["soft_policy"]
                     and out["dice_state"])
    return out


def _bundle(real: Path) -> dict:
    """The bundle's own numbers (its metrics.jsonl in the real runs directory, read-only): the
    medians after its first iteration, and its first and last learned iteration's outcome
    errors (a fresh head's and a trained one's)."""
    path = real / START[0] / START[1] / "metrics.jsonl"
    if not path.exists():
        return {"missing": str(path)}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    late = rows[1:] or rows
    learned = [r for r in rows if "train/policy_loss" in r]

    def median(key: str) -> float | None:
        values = [r[key] for r in late if key in r]
        return statistics.median(values) if values else None

    out = {"iterations": len(rows), "leaves_per_s": median("selfplay/leaves_per_s"),
           "iteration_s": median("throughput/iteration_s"), "target_pruned": median("target/pruned"),
           "search_target_entropy": median("search/target_entropy"),
           "positions": statistics.median(_per_iteration(rows, "positions")),
           "games": statistics.median(_per_iteration(rows, "games"))}
    out["iterations_per_hour"] = 3600.0 / out["iteration_s"] if out["iteration_s"] else None
    for key in OUTCOME:
        out[key.removeprefix("train/outcome_")] = [learned[0].get(key), learned[-1].get(key)] if learned else None
    return out


def _per_iteration(rows: list[dict], key: str) -> list[float]:
    """A running total's gain in each iteration (the first from 0)."""
    totals = [0, *(r[key] for r in rows)]
    return [b - a for a, b in itertools.pairwise(totals)]


def _over(value, reference) -> float | None:
    return value / reference if value is not None and reference else None


def _root_exploration(train: dict, probe: dict) -> dict:
    """``SelfPlay.options`` as the child saw them, against the config's train keys."""
    want = {key: float(train.get(key, off)) for key, off in OFF.items()}
    seen = probe.get("selfplay")
    out: dict = {"logged": "nothing: the trainer logs no metric of forced playouts or the root temperature",
                 "configured": want, "as_planned": want == PLANNED}
    if not seen:
        return {**out, "ok": False, "missing": "the child wrote no SelfPlay options"}

    def pick(options: dict) -> dict:
        return {key: options.get(key) for key in OFF}

    out.update(full=pick(seen["full"]), fast=pick(seen["fast"]), decided=pick(seen["decided"]),
               simulations={name: seen[name].get("simulations") for name in ("full", "fast", "decided")},
               dirichlet_alpha={name: seen[name].get("dirichlet_alpha") for name in ("full", "fast", "decided")},
               reuse=seen["reuse"], pools=seen["pools"])
    # a group's pools: with reuse one, built with the full searches' settings (each search passes its own
    # options); without, the full pool and the fast one
    groups = [[pick(pool) for pool in pools] for pools in seen["pools"]]
    out["ok"] = bool(out["as_planned"] and out["full"] == want and out["fast"] == OFF and out["decided"] == OFF
                     and groups and all(pools[:1] == [want] and all(pool == OFF for pool in pools[1:])
                                        for pools in groups))
    return out


def _summary(args: argparse.Namespace, scratch: Path, real: Path) -> dict:
    from catanrl.runs import load_experiment

    run = load_experiment(f"{EXPERIMENT}/{VARIANT}")
    rows = run.metrics()
    train = run.config["train"]
    probe = json.loads((scratch / PROBE).read_text()) if (scratch / PROBE).exists() else {}
    bundle = _bundle(real)
    late = rows[1:] or rows
    learned = [r for r in rows if "train/policy_loss" in r]
    numbers = {k: r[k] for r in learned for k in r if k.startswith("train/")}
    finite = bool(learned) and all(isinstance(r[k], (int, float)) and math.isfinite(r[k])
                                   for r in learned for k in r if k.startswith("train/"))

    def series(key: str, over: list[dict] = rows) -> list:
        return [r.get(key) for r in over]

    def ends(key: str) -> list | None:
        return [learned[0].get(key), learned[-1].get(key)] if learned else None

    def median(key: str) -> float | None:
        values = [r[key] for r in late if key in r]
        return statistics.median(values) if values else None

    leaves, seconds = median("selfplay/leaves_per_s"), median("throughput/iteration_s")
    out: dict = {
        "smoke": {"overrides": _overrides(args), "note": (
            f"train.value_blend_iterations is {train.get('value_blend_iterations')} here, not the run's 60, so "
            "that train/value_blend reaches train.value_search_blend within the smoke; min_replay and "
            "replay_capacity are the smoke's too")},
        "iterations": len(rows), "learned": len(learned), "learned_epochs": series("epoch", learned),
        "finite": finite, "train_keys": sorted(numbers),
        "speed": {"leaves_per_s": leaves, "over_bundle_leaves_per_s": _over(leaves, bundle.get("leaves_per_s")),
                  "iteration_s": seconds, "over_bundle_iteration_s": _over(seconds, bundle.get("iteration_s")),
                  "iterations_per_hour": 3600.0 / seconds if seconds else None,
                  "selfplay_s": median("throughput/selfplay_s"), "full_fraction": median("selfplay/full_fraction")},
        "decided": series("selfplay/decided"), "sims_per_decision": series("selfplay/sims_per_decision"),
        "positions": _per_iteration(rows, "positions"), "games": _per_iteration(rows, "games"),
        "replay": rows[-1]["replay"], "bundle": bundle,
    }
    out["soft"] = {"loss": ends("train/soft_policy_loss"), "target_entropy": ends("train/soft_target_entropy"),
                   "policy_loss": ends("train/policy_loss"),
                   "logged": bool(learned) and all("train/soft_policy_loss" in r and "train/soft_target_entropy" in r
                                                   for r in learned)}
    blend = series("train/value_blend", learned)
    target = float(train.get("value_search_blend", 0.0))
    out["value_blend"] = {"per_iteration": blend, "target": target,
                          "rises_to": bool(blend) and None not in blend and blend == sorted(blend)
                          and math.isclose(blend[-1], target, abs_tol=1e-9),
                          "search_value_gap": series("train/search_value_gap", learned)}

    # surprise sampling: the largest N x p drawn against the largest in the buffer and against the cap
    weight, cap = float(train.get("surprise_weight", 0.0)), float(train.get("surprise_cap", 0.0))
    sampled = series("train/surprise_sampled_max", learned)
    seen = probe.get("surprise", [])
    limits = [entry["limit"] for entry in seen]
    cap_limits = [(1.0 - weight) + weight * cap * entry["mean"] / entry["capped_mean"] if entry["capped_mean"] > 0.0
                  else 1.0 for entry in seen] if cap > 0.0 else []
    out["surprise"] = {"weight": weight, "cap": cap, "ess": series("train/surprise_ess", learned),
                       "sampled_max": sampled, "sampled_mean": series("train/surprise_sampled_mean", learned),
                       "mean": series("train/surprise_mean", learned), "limit": limits, "cap_limit": cap_limits,
                       "cap_limit_unmoved_mean": (1.0 - weight) + weight * cap if cap > 0.0 else None,
                       "at_cap": [entry["at_cap"] for entry in seen],
                       "respects_cap": bool(sampled) and None not in sampled and len(sampled) == len(limits)
                       == len(cap_limits) and all(drawn <= limit * (1 + 1e-7) and limit <= allowed * (1 + 1e-7)
                                                  for drawn, limit, allowed in zip(sampled, limits, cap_limits))}

    # the outcome head: carried over, so its errors start at the bundle's trained level, not a fresh head's
    outcome: dict = {"known": series("train/outcome_known", learned)}
    for key in OUTCOME:
        outcome[key.removeprefix("train/outcome_")] = ends(key)
    low = []
    for name in ("vp_mae", "turns_mae"):
        first, reference = (outcome[name] or [None])[0], bundle.get(name)
        if first is not None and reference and None not in reference:
            low.append(abs(first - reference[1]) < abs(first - reference[0]))
    outcome["starts_low"] = bool(low) and all(low) if len(low) == 2 else None
    tested = [r for r in rows if "oos/outcome_vp_mae" in r]
    outcome["oos_last"] = {k.removeprefix("oos/outcome_"): tested[-1][k] for k in
                           ("oos/outcome_rows", "oos/outcome_vp_mae", "oos/outcome_turns_mae")} if tested else None
    out["outcome"] = outcome

    out["root_exploration"] = {**_root_exploration(train, probe), "target_pruned": median("target/pruned"),
                               "search_target_entropy": median("search/target_entropy")}
    out["e0_is_start"] = _e0_is_start()
    checks = probe.get("kernel_checks")
    out["kernel_checks"] = {"returned": checks, "iterations": len(rows), "kernels": train.get("kernels"),
                            "each_iteration": checks == len(rows)}
    out["ok"] = bool(finite and len(learned) >= 2 and out["e0_is_start"]["ok"] and out["root_exploration"]["ok"])
    return out


def _warnings(summary: dict) -> list[str]:
    """The reported checks, outside the verdict, that came out false."""
    run = summary.get("selfplay")
    if run is None:
        return []
    checks = {"soft.logged": run["soft"]["logged"], "value_blend.rises_to": run["value_blend"]["rises_to"],
              "surprise.respects_cap": run["surprise"]["respects_cap"], "outcome.starts_low": run["outcome"]["starts_low"],
              "kernel_checks.each_iteration": run["kernel_checks"]["each_iteration"],
              "decided (selfplay/decided logged)": None not in run["decided"],
              "bundle (its metrics.jsonl read)": "missing" not in run["bundle"]}
    return [name for name, passed in checks.items() if not passed]


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _run_child(command: list[str], metrics: Path, iterations: int) -> int:
    """Run the self-play child, with a line per finished iteration (its metrics rows as they
    appear): the time since the child started and an ETA at the iterations' own pace."""
    started, seen, spent = time.time(), 0, []
    child = subprocess.Popen(command, env=os.environ.copy())
    while True:
        code = child.poll()
        rows = [line for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
        for line in rows[seen:]:
            seen += 1
            try:
                row = json.loads(line)
            except ValueError:  # a row half written: the next read has it
                seen -= 1
                break
            elapsed = time.time() - started
            spent.append(float(row.get("throughput/iteration_s", elapsed / seen)))
            print(f"tricks_smoke: iteration {seen}/{iterations} done, {_clock(elapsed)} elapsed, "
                  f"ETA {_clock(statistics.median(spent) * max(iterations - seen, 0))} "
                  f"({spent[-1]:.0f} s, replay {row.get('replay')}, "
                  f"{'learned' if 'train/policy_loss' in row else 'self-play only'})", flush=True)
        if code is not None:
            return code
        time.sleep(1.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", required=True, help="a new scratch runs directory (created; never the real one)")
    parser.add_argument("--real-runs", default=None, help="where to copy the start's run from and read the "
                        "bundle's metrics (default: the main worktree's runs/)")
    parser.add_argument("--iterations", type=int, default=8, help="self-play iterations (train.iterations)")
    parser.add_argument("--min-replay", type=int, default=2000, help="train.min_replay (the real 50,000 takes ~6 "
                        "iterations to reach)")
    parser.add_argument("--blend-iterations", type=int, default=4, help="train.value_blend_iterations (the run's "
                        "60 would leave the search value's share near 0 within the smoke)")
    parser.add_argument("--finals", type=int, default=0, help="the final on its suite's first N boards (0: none)")
    parser.add_argument("--skip-identity", action="store_true", help="leave out the identity part")
    parser.add_argument("--skip-selfplay", action="store_true", help="the identity part alone")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    os.environ["WANDB_MODE"] = "disabled"
    for key in ("WANDB_API_KEY", "CATANRL_WANDB_PROJECT", "CATANRL_WANDB_ENTITY"):
        os.environ.pop(key, None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    scratch = Path(args.runs).resolve()
    if args.child:
        os.environ["CATANRL_RUNS_DIR"] = str(scratch)
        _child(args, scratch)
        return

    os.environ.pop("CATANRL_RUNS_DIR", None)
    from catanrl.runs import runs_root

    real = Path(args.real_runs).resolve() if args.real_runs else runs_root()
    if scratch == real or real in scratch.parents or scratch in real.parents:
        raise SystemExit(f"--runs {scratch} overlaps the real runs directory {real}")
    if not args.skip_selfplay and (scratch / EXPERIMENT / VARIANT).exists():
        raise SystemExit(f"--runs {scratch} already holds {EXPERIMENT}/{VARIANT}: give a new directory")
    _, v = _variant(args)
    if v.config["train"]["kernels"] != "torch":  # fail now, not after the identity part
        import torch

        missing = [tool for tool in ("nvcc", "ninja") if shutil.which(tool) is None]
        if missing:
            raise SystemExit(f"train.kernels: {v.config['train']['kernels']} builds the fused kernels with nvcc and "
                             f"ninja; not on PATH: {', '.join(missing)}")
        if not torch.cuda.is_available():
            raise SystemExit(f"train.kernels: {v.config['train']['kernels']} needs a CUDA device and there is none "
                             "(on the CPU: --set train.kernels=torch --set runtime.device=cpu)")
    _copy_run(real, scratch, START)
    if args.finals > 0:
        _copy_run(real, scratch, OPPONENT)
    os.environ["CATANRL_RUNS_DIR"] = str(scratch)

    summary: dict = {}
    if not args.skip_identity:
        summary["identity"] = _identity(args)
        print(json.dumps({"identity": summary["identity"]}, indent=1), flush=True)
    if not args.skip_selfplay:
        child = [sys.executable, "-P", __file__, *(argv if argv is not None else sys.argv[1:]), "--child"]
        metrics = scratch / EXPERIMENT / VARIANT / "metrics.jsonl"
        if _run_child(child, metrics, int(v.config["train"]["iterations"])) != 0:
            raise SystemExit("the self-play run failed")
        summary["selfplay"] = _summary(args, scratch, real)
    ok = all(part["ok"] for part in summary.values() if isinstance(part, dict))
    summary["ok"] = bool(ok)
    summary["warnings"] = _warnings(summary)
    (scratch / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    print(f"warnings: {', '.join(summary['warnings']) or 'none'}", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    if not ok:
        raise SystemExit(1)


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


if __name__ == "__main__":
    main()
