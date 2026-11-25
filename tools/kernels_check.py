"""The fused kernels of one network against its PyTorch module on real positions, and their speed.

The network is a checkpoint (``--ckpt``: a ``.pt`` path, read-only) or a preset with random
weights (``--preset graph-m``; ``--dice``, ``--outcome``, ``--soft-policy`` switch its optional
pieces on, ``--scale`` multiplies every parameter: random init leaves the logits near zero, where
errors hide). The positions are the network's own: ``--games`` base-2p-fr games (balanced dice)
side by side, which it plays against itself by sampling its policy (a finished game makes room
for a new one), a decision's row kept with probability ``--keep`` (0.1: a game of some 400
decisions gives about 40 rows, from its opening to its end) until there are ``--rows`` (a
progress line as they come), at the network's own level and dice input.

    equal   ``BatchEvaluator`` with ``kernels`` fused (strict and fast) and fused16 (fast) against
            the module on the device (``kernels: torch``, strict), and fused16 against fused:
            the largest difference of the legal logits (and over their scale, the largest
            ``|logit|``), of the values, the share of rows with the same best move (over all rows
            and over those with a choice), the masks, the KL of the masked policies and how many
            fused16 chunks fell back to float32 (a half past its range). ``within`` holds fused
            to tests/test_graph_kernels.py's bounds for it (logits 2e-6 x scale + 1e-5, values
            1e-5) and fused16 to its bounds against fused on the logits and the values (4e-3 x
            scale, 2e-3). The exit code is 1 when one is false. The KL and the best move are
            reported: judge them beside graph_kernels.FusedGraph16's numbers for graph-s (the
            test's 1e-4 on the largest KL is for its 48 rows of a random network; a trained
            one's few rows with two near-equal large logits pass it by).
    dump    the rows and the kernels' outputs on them into an ``.npz`` (``--rows-from`` another
            dump: its rows instead of new games), for ``same``.
    same    two dumps, bit for bit: the rows, then every output. Run ``dump`` from two checkouts
            (each with its own Python; this file needs nothing of its own checkout) to hold one's
            kernels against the other's. The exit code is 1 when they differ.
    bench   rows per second of a ``--bench-rows`` (512) forward per ``kernels``: the CUDA graph's
            replay alone (``forward``) and the evaluator's call, uploads and downloads included
            (``call``); the median of 30 timings of 20 forwards, after a warm-up.

The kernels build with nvcc and ninja on PATH (ninja is not in the venv); the card should be
free. Nothing is written but ``--out``.

    P=PATH=<a directory with ninja>:/opt/cuda/bin:$PATH
    env $P .venv/bin/python tools/kernels_check.py equal --ckpt runs/graph-m-2p-lr/main/checkpoints/e022711.pt
    env $P .venv/bin/python tools/kernels_check.py equal --preset graph-m --dice --outcome --soft-policy --scale 3
    env $P .venv/bin/python tools/kernels_check.py bench --ckpt runs/graph-m-2p-lr/main/checkpoints/e022711.pt
    env $P .venv/bin/python tools/kernels_check.py dump --ckpt runs/az-2p-tricks/main/checkpoints/e000900.pt --out here.npz
    env $P <main>/.venv/bin/python tools/kernels_check.py dump --ckpt ... --rows-from here.npz --out main.npz
    .venv/bin/python tools/kernels_check.py same main.npz here.npz
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

#: (kernels, mode) of the evaluators held against the module: the evals search with fused in
#: strict (fused16 in strict, an agent's ``kernels=fused16``, has the same per-row outputs as
#: fused16 here unless a half overflows), the trainer self-plays with fused16 (and fused) in fast.
EVALUATORS = (("fused", "strict"), ("fused", "fast"), ("fused16", "fast"))
#: tests/test_graph_kernels.py's bounds: fused against the module (x the logits' scale, plus,
#: values), fused16 against fused (x scale, values).
FUSED = (2e-6, 1e-5, 1e-5)
FUSED16 = (4e-3, 2e-3)


def network(args: argparse.Namespace):
    """(the module in eval mode on the CPU, its dims, its level, whether it reads the dice input)."""
    import torch

    if args.ckpt:
        from catanrl.checkpoint import Checkpoint

        ckpt = Checkpoint.load(args.ckpt)
        return ckpt.model("all").eval(), ckpt.dims, ckpt.levels["level"], ckpt.dice_state
    from catanrl.envs.encoding import dims_for
    from catanrl.train.puffer import build_policy_model

    dims = dims_for(["base-2p-fr"])
    extra = {key: True for key, on in (("outcome", args.outcome), ("soft_policy", args.soft_policy)) if on}
    torch.manual_seed(args.seed)
    model = build_policy_model({"model": {"preset": args.preset, "args": {"shared": True, **extra}},
                                "obs": {"level": "full", "dice_state": args.dice}}, dims).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(args.scale)
    return model, dims, "full", args.dice


def own_rows(model, dims, level, dice: bool, *, count: int, games: int, seed: int, device, keep: float = 0.1):
    """``count`` rows of base-2p-fr games the network plays against itself, sampling its policy
    (``kernels: torch``): each decision's row with probability ``keep``, ``games`` games side by
    side, a finished one replaced by a new one."""
    import numpy as np

    from catanrl.envs.encoding import make_encoder
    from catanrl.games import load_game
    from catanrl.search.evaluator import BatchEvaluator

    config = load_game("base-2p-fr")
    encoder = make_encoder(dims, level, dice)
    space = encoder.action_space
    evaluator = BatchEvaluator(model, device=device, infer_rows=games, mode="fast")
    rng = np.random.default_rng(seed)
    new = lambda i: config.cppanatron_game(map_seed=seed + i, seed=1000 + seed + i)
    live = [new(i) for i in range(games)]
    started, rows, step, played = time.perf_counter(), [], 0, games
    while len(rows) < count:
        batch = np.stack([encoder.encode(game, game.current_player_index) for game in live])
        logits, _ = evaluator(batch)
        rows.extend(batch[rng.random(len(batch)) < keep])
        for i, (game, row) in enumerate(zip(live, logits.astype(np.float64))):
            p = np.exp(row - row.max())
            game.execute(space.decode(game, int(rng.choice(len(p), p=p / p.sum()))))
            if game.winning_color is not None:
                live[i], played = new(played), played + 1
        step += 1
        if step % 500 == 0:
            done, elapsed = min(len(rows), count), time.perf_counter() - started
            print(f"  rows {done}/{count}  {elapsed:.0f}s  eta {elapsed * (count - done) / max(done, 1):.0f}s",
                  file=sys.stderr, flush=True)
    return np.stack(rows[:count]).astype(np.float32)


def _gap(got, want, legal) -> dict:
    """Two (logits, values) results over the legal actions; ``want`` is the reference."""
    import numpy as np
    import torch

    (lg, vg), (lw, vw) = got, want
    scale = max(1.0, float(np.abs(lw[legal]).max()))
    best = lambda logits: np.where(legal, logits, -np.inf).argmax(1)
    choice = legal.sum(1) > 1
    same = best(lg) == best(lw)
    log_want, log_got = (torch.log_softmax(torch.from_numpy(np.where(legal, x, -1e9)).double(), 1) for x in (lw, lg))
    kl = (log_want.exp() * (log_want - log_got)).sum(1)
    return {"logit_max_diff": float(np.abs(lg - lw)[legal].max()), "logit_scale": scale,
            "value_max_diff": float(np.abs(vg - vw).max()),
            "masks_equal": bool(np.array_equal(lg > -1e8, legal) and np.array_equal(lw > -1e8, legal)),
            "finite": bool(np.isfinite(lg).all() and np.isfinite(vg).all()),
            "best_move_equal": float(same.mean()), "rows_with_a_choice": int(choice.sum()),
            "best_move_equal_with_a_choice": float(same[choice].mean()) if choice.any() else 1.0,
            "kl_mean": float(kl.mean()), "kl_max": float(kl.max())}


def _evaluate(model, rows, kernels: str, mode: str, device, infer_rows: int):
    """((logits, values), chunks evaluated again in float32) of one evaluator, freed after."""
    import gc

    import torch

    from catanrl.search.evaluator import BatchEvaluator

    evaluator = BatchEvaluator(model, device=device, infer_rows=infer_rows, mode=mode, kernels=kernels)
    out, again = evaluator(rows), int(evaluator.exact_forwards)
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    return out, again


def cmd_equal(args: argparse.Namespace) -> int:
    import torch

    device = torch.device("cuda")
    model, dims, level, dice = network(args)
    rows = own_rows(model, dims, level, dice, count=args.rows, games=args.games, seed=args.seed, device=device,
                    keep=args.keep)
    legal = rows[:, model.mask_offset: model.mask_offset + model.num_actions] > 0
    module, _ = _evaluate(model, rows, "torch", "strict", device, args.bench_rows)
    out: dict = {"network": args.ckpt or args.preset, "hidden": int(model.backbone.hidden),
                 "layers": len(model.backbone.layers), "parameters": sum(p.numel() for p in model.parameters()),
                 "dice_state": bool(dice), "outcome": bool(getattr(model, "outcome", False)),
                 "soft_policy": bool(getattr(model, "soft_policy", False)), "rows": len(rows)}
    results = {}
    for kernels, mode in EVALUATORS:
        results[kernels, mode], again = _evaluate(model, rows, kernels, mode, device, args.bench_rows)
        gap = _gap(results[kernels, mode], module, legal)
        gap["float32_chunks"] = again
        if kernels == "fused":
            gap["within"] = bool(gap["logit_max_diff"] <= FUSED[0] * gap["logit_scale"] + FUSED[1]
                                 and gap["value_max_diff"] <= FUSED[2] and gap["masks_equal"])
        out[f"{kernels}/{mode} vs module"] = gap
    gap = _gap(results["fused16", "fast"], results["fused", "fast"], legal)
    gap["within"] = bool(gap["logit_max_diff"] <= FUSED16[0] * gap["logit_scale"] and gap["finite"]
                         and gap["value_max_diff"] <= FUSED16[1] and gap["masks_equal"])
    out["fused16/fast vs fused/fast"] = gap
    out["strict_equals_fast"] = bool(all((a == b).all() for a, b in zip(results["fused", "strict"],
                                                                         results["fused", "fast"])))
    ok = all(gap["within"] for gap in out.values() if isinstance(gap, dict) and "within" in gap)
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_dump(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    device = torch.device("cuda")
    model, dims, level, dice = network(args)
    if args.rows_from:
        rows = np.load(args.rows_from)["rows"]
    else:
        rows = own_rows(model, dims, level, dice, count=args.rows, games=args.games, seed=args.seed, device=device,
                        keep=args.keep)
    arrays = {"rows": rows}
    for kernels, mode in EVALUATORS:
        (logits, values), _ = _evaluate(model, rows, kernels, mode, device, args.bench_rows)
        arrays[f"{kernels}.{mode}.logits"], arrays[f"{kernels}.{mode}.values"] = logits, values
    np.savez(args.out, **arrays)
    print(f"{args.out}: {len(rows)} rows, {', '.join(k for k in arrays if k != 'rows')}")
    return 0


def cmd_same(args: argparse.Namespace) -> int:
    import numpy as np

    a, b = np.load(args.first), np.load(args.second)
    out = {key: bool(key in b.files and a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes())
           for key in a.files}
    out.update({key: False for key in b.files if key not in a.files})
    print(json.dumps({"rows": len(a["rows"]), "equal": out}, indent=1))
    print("SAME" if all(out.values()) else "DIFFERENT")
    return 0 if all(out.values()) else 1


def cmd_bench(args: argparse.Namespace) -> int:
    import gc
    from functools import partial

    import torch

    from catanrl.search.evaluator import BatchEvaluator

    device = torch.device("cuda")
    model, dims, level, dice = network(args)
    n = args.bench_rows
    rows = own_rows(model, dims, level, dice, count=n, games=args.games, seed=args.seed, device=device, keep=args.keep)

    def timed(fn, calls: int = 20, samples: int = 30) -> float:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize(device)
        seconds = []
        for _ in range(samples):
            start = time.perf_counter()
            for _ in range(calls):
                fn()
            torch.cuda.synchronize(device)
            seconds.append((time.perf_counter() - start) / calls)
        return statistics.median(seconds)

    out: dict = {"network": args.ckpt or args.preset, "hidden": int(model.backbone.hidden), "rows": n}
    for kernels in args.kernels.split(","):
        evaluator = BatchEvaluator(model, device=device, infer_rows=n, mode="fast", kernels=kernels)
        evaluator(rows)
        graph, static, _ = evaluator._graph(n)
        with torch.inference_mode():
            static[0].copy_(torch.from_numpy(rows))
        forward, call = timed(graph.replay), timed(partial(evaluator, rows))
        out[kernels] = {"forward_ms": 1e3 * forward, "forward_rows_per_s": n / forward,
                        "call_ms": 1e3 * call, "call_rows_per_s": n / call,
                        "float32_chunks": int(evaluator.exact_forwards)}
        print(f"{out['network']}  {kernels:8s} forward {1e3 * forward:7.3f} ms  {n / forward:9,.0f} rows/s   "
              f"call {1e3 * call:7.3f} ms  {n / call:9,.0f} rows/s", flush=True)
        del evaluator, graph, static
        gc.collect()
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", default=None, help="a checkpoint file (.pt)")
    common.add_argument("--preset", default="graph-m", help="without --ckpt: a model preset, random weights")
    common.add_argument("--dice", action="store_true", help="--preset: obs.dice_state")
    common.add_argument("--outcome", action="store_true", help="--preset: model.args.outcome")
    common.add_argument("--soft-policy", action="store_true", help="--preset: model.args.soft_policy")
    common.add_argument("--scale", type=float, default=1.0, help="--preset: every parameter times this")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--rows", type=int, default=2000, help="positions (equal, dump)")
    common.add_argument("--games", type=int, default=64, help="games played side by side for them")
    common.add_argument("--keep", type=float, default=0.1, help="the share of their decisions kept as rows")
    common.add_argument("--bench-rows", type=int, default=512, help="rows per forward (infer_rows)")
    common.add_argument("--out", default=None, help="equal, bench: a JSON of the result; dump: the .npz")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("equal", parents=[common])
    dump = sub.add_parser("dump", parents=[common])
    dump.add_argument("--rows-from", default=None, help="another dump: its rows")
    bench = sub.add_parser("bench", parents=[common])
    bench.add_argument("--kernels", default="torch,fused,fused16")
    same = sub.add_parser("same")
    same.add_argument("first")
    same.add_argument("second")
    args = parser.parse_args(argv)
    if args.command == "dump" and not args.out:
        parser.error("dump needs --out")
    raise SystemExit({"equal": cmd_equal, "dump": cmd_dump, "same": cmd_same, "bench": cmd_bench}[args.command](args))


if __name__ == "__main__":
    main()
