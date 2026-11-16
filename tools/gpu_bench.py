"""How fast our workloads run on a GPU, and how much of that is the GPU's to give.

One script for every machine (this box, a Modal container): the same code, one JSON per part and a
printed table. Nothing under ``src/`` is edited: the timers inside the trainers are wrapped around
their functions from here, in the stretch's own process.

A. GPU-only microbenchmarks: synthetic batches of real rows (random base-2p-fr games, full level),
   warmed up, ``torch.cuda.synchronize`` around the timing, the median of repeats, in rows/s:
   A1  graph-s forward on the fused16 kernels at self-play's shape (a 512-row CUDA graph replay);
       and the evaluator's call on the same rows, uploads and downloads included;
   A2  graph-m forward on the PyTorch path at the PPO rollout's shape (1,032 acting rows of the
       2,064), float32 and float16 autocast; and graph-m as a search would run it today (a 512-row
       CUDA graph of the PyTorch path);
   A3  a PPO micro-batch, forward + backward at 2,048 rows through PufferPolicy.forward (half of
       them act and reach the network, as in a rollout), float32 and float16 autocast with a
       GradScaler, graph-m and graph-s. Eager, as A2: the trainers compile the heads
       (runtime.compile); B2 times that path.
   ``launch`` is the share of a call's time before the Python call returns: near 1 means the CPU,
   not the device, sets that number.
B. End-to-end stretches through ``catanrl.runner.run_variant`` in a scratch runs directory
   (never the real one; W&B off), each in its own process, with a progress line (rate and ETA):
   B1  AlphaZero self-play from e1800 (az-2p-kata/main+1500@e1800) with that run's stored train
       config: a fresh run with an empty replay, ``--iterations`` iterations, no evals.
       ``train.min_replay`` is lowered (``--min-replay``) so the learning step and the evaluator's
       rebuild are inside the timed iterations. The games all start at once, so the first
       iterations are openings: compare the same iteration numbers between machines.
   B2  graph-m PPO from scratch with graph-m-2p-lr's config (configs/experiments/graph-m-2p-lr.yaml),
       ``--epochs`` epochs of the 3B-step schedule (``train.stop_epoch``), float32 and float16, no
       monitors, no select, no finals.
C. The CPU/GPU split of B, from the same processes:
   B1  every ``Pipeline.submit`` is bracketed by two CUDA timing events (the device's time on that
       batch: uploads, forwards, downloads) and every ``Pipeline.result`` is timed (the main
       thread's wait). The loop is pipelined (two groups take turns on the device), so the device's
       time and the CPU's overlap and do not add up to the wall-clock. The prediction for a device
       ``s`` times faster replays the recorded sequence: the CPU segments as measured, every
       batch's device time divided by ``s``, the device serving the batches in order
       (:func:`simulate`; ``s = 1`` must give back the measured time, ``s = inf`` is the CPU's
       floor). The trace is saved (``*.trace.npz``) for ``predict``.
   B2  ``PufferPolicy.forward_eval`` is timed with synchronisation (the rollout's network time);
       the rest of collect is the engine and PuffeRL's bookkeeping; train is all on the device.

    # everything on this machine (free card; ~18 min)
    env -u WANDB_API_KEY WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/python tools/gpu_bench.py all --runs /tmp/gpu-bench-runs --out /tmp/gpu-bench
    # parts
    .venv/bin/python tools/gpu_bench.py a --out /tmp/gpu-bench
    .venv/bin/python tools/gpu_bench.py b1 --runs /tmp/gpu-bench-runs --out /tmp/gpu-bench [--threads 32 --tag t32]
    .venv/bin/python tools/gpu_bench.py b2 --runs /tmp/gpu-bench-runs --out /tmp/gpu-bench
    # the table of one or more machines' results (ratios to the first)
    .venv/bin/python tools/gpu_bench.py table local=/tmp/gpu-bench l40s=/tmp/l40s
    # this machine's self-play with a device s times faster (from its trace)
    .venv/bin/python tools/gpu_bench.py predict /tmp/gpu-bench/b1.trace.npz --speedup 3.2
    # CPU check of the tool itself (tiny, a minute or two)
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python tools/gpu_bench.py all --tiny \\
        --runs /tmp/gpu-bench-cpu-runs --out /tmp/gpu-bench-cpu
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START = ("az-2p-kata", "main+1500", 1800, "7c44ec80540a")  # e1800: az-2p-kata/main+1500@e1800#7c44ec80540a
PPO_CONFIG = REPO / "configs" / "experiments" / "graph-m-2p-lr.yaml"
EXPERIMENT = "gpu-bench"
SPEEDUPS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, float("inf"))
SUBMIT, RESULT, STEP = 0, 1, 2


# ---------------------------------------------------------------- machine


def _run(*argv: str) -> str | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=20).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def load_average() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def machine() -> dict:
    """The hardware and software next to every number."""
    import torch

    cpu_model = None
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    cgroup = None
    with contextlib.suppress(OSError):
        cgroup = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_model": cpu_model, "cpu_count": os.cpu_count(), "cpu_affinity": len(os.sched_getaffinity(0)),
        "cgroup_cpu_max": cgroup, "load_average": load_average(),
        "nvcc": (_run("nvcc", "--version") or "").splitlines()[-1:] or None,
    }
    with contextlib.suppress(OSError):
        kb = int(next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")).split()[1])
        info["memory_GiB"] = round(kb / 2**20, 1)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_memory_GiB=round(props.total_memory / 2**30, 1),
                    gpu_capability=list(torch.cuda.get_device_capability(0)),
                    driver=_run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"))
    else:
        info.update(gpu=None)
    return info


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float) + "\n")


# ---------------------------------------------------------------- A


def real_rows(count: int, *, seed: int = 0):
    """``count`` acting rows and the same positions' waiting rows (the other seat's: only the
    no-op) of random base-2p-fr games, full level, float32 [count, obs]."""
    import cppanatron as cpp
    import numpy as np

    from catanrl import games as game_configs
    from catanrl.envs.encoding import dims_for

    dims = dims_for(["base-2p-fr"])
    encoder = cpp.GraphEncoder(cpp.EncoderDims(**dims), cpp.InfoLevel.FULL)
    config = game_configs.load_game("base-2p-fr")
    rng = np.random.default_rng(seed)
    acting, waiting = [], []
    for i in range(count):
        game = config.cppanatron_game(map_seed=100 + i, seed=i)
        for _ in range(int(rng.integers(0, 200))):
            if game.winning_color is not None:
                break
            actions = game.playable_actions
            game.execute(actions[int(rng.integers(len(actions)))])
        if game.winning_color is not None:  # a finished game has no acting row: its opening instead
            game = config.cppanatron_game(map_seed=100 + i, seed=i)
        seat = game.current_player_index
        acting.append(encoder.encode(game, seat))
        waiting.append(encoder.encode(game, 1 - seat))
    return dims, np.stack(acting).astype(np.float32), np.stack(waiting).astype(np.float32)


def _timed(fn, *, device, calls: int, samples: int, warmup: int) -> dict:
    """Seconds per call of ``fn``: the median over ``samples`` of ``calls`` calls and one wait."""
    import torch

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        fn()
    sync()
    seconds, launch = [], []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        returned = time.perf_counter()
        sync()
        end = time.perf_counter()
        seconds.append((end - start) / calls)
        launch.append((returned - start) / (end - start))
    return {"s": statistics.median(seconds), "s_min": min(seconds), "s_max": max(seconds),
            "launch": round(statistics.median(launch), 3), "calls": calls, "samples": samples}


def bench_a(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from catanrl.train.puffer import build_policy_model
    from catanrl.train.puffer_policy import PufferPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    tiny = args.tiny
    scale = (lambda n: max(2, n // 16)) if tiny else (lambda n: n)
    dims, acting_np, waiting_np = real_rows(16 if tiny else 256)
    torch.manual_seed(0)

    def model_of(preset: str):
        return build_policy_model({"model": {"preset": preset, "args": {"shared": True}}, "obs": {"level": "full"}},
                                  dims).to(device)

    def tile(rows: np.ndarray, n: int) -> torch.Tensor:
        return torch.from_numpy(np.resize(rows, (n, rows.shape[1]))).to(device)

    presets = ("graph-tiny",) if tiny else ("graph-s", "graph-m")
    small, big = presets[0], presets[-1]
    results: dict[str, dict] = {}
    items: list[tuple[str, str, int, object]] = []  # (key, description, rows, a function that returns _timed)

    def add(key: str, what: str, rows: int, make) -> None:
        items.append((key, what, rows, make))

    # A1: self-play's forward (the evaluator's 512-row CUDA graph on the fused16 kernels)
    search_rows = scale(512)
    if cuda:
        from catanrl.search.evaluator import BatchEvaluator

        def a1(preset: str, kernels: str, raw: bool):
            def make():
                ev = BatchEvaluator(model_of(preset).eval(), device=device, infer_rows=search_rows, mode="fast",
                                    kernels=kernels)
                rows = np.resize(acting_np, (search_rows, acting_np.shape[1]))
                if raw:
                    graph, static, _ = ev._graph(search_rows)
                    static[0].copy_(torch.from_numpy(rows))
                    return _timed(graph.replay, device=device, calls=50, samples=30, warmup=50)
                return _timed(lambda: ev(rows), device=device, calls=20, samples=20, warmup=20)
            return make
        if not tiny:
            add("A1 graph-s fused16 forward", f"{search_rows}-row CUDA graph replay", search_rows,
                a1("graph-s", "fused16", True))
            add("A1 graph-s fused16 evaluator call", f"{search_rows} rows, uploads and downloads included",
                search_rows, a1("graph-s", "fused16", False))
        add(f"A2 {big} search forward (torch path)", f"{search_rows}-row CUDA graph replay, float32", search_rows,
            a1(big, "torch", True))

    # A2: the PPO rollout's forward (the acting rows of 1,032 envs)
    rollout_rows = scale(1032)

    def a2(preset: str, half: bool):
        def make():
            model = model_of(preset)
            obs = tile(acting_np, rollout_rows)

            def call():
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=half):
                    model.heads(obs)
            return _timed(call, device=device, calls=5, samples=20, warmup=10)
        return make
    for half in (False, True):
        if half and not cuda:
            continue
        add(f"A2 {big} rollout forward {'float16' if half else 'float32'}", f"{rollout_rows} acting rows, eager",
            rollout_rows, a2(big, half))

    # A3: a PPO micro-batch (forward + backward through PufferPolicy.forward)
    micro = scale(2048)

    def a3(preset: str, half: bool):
        def make():
            model = model_of(preset)
            policy = PufferPolicy(model, num_rows=micro, compile=False, amp_dtype=torch.float16 if half else None)
            obs = torch.cat((tile(acting_np, micro // 2), tile(waiting_np, micro - micro // 2)))
            obs = obs[torch.randperm(micro, generator=torch.Generator().manual_seed(1)).to(device)]
            actions = torch.zeros(micro, dtype=torch.long, device=device)
            scaler = torch.amp.GradScaler(device.type) if half else None
            parameters = [p for p in model.parameters() if p.requires_grad]

            def call():
                logits, values = policy.forward(obs)
                loss = -torch.log_softmax(logits, dim=-1).gather(1, actions[:, None]).mean()
                loss = loss + F.binary_cross_entropy_with_logits(values.logits, values.present * 0.5,
                                                                 weight=values.present)
                for p in parameters:
                    p.grad = None
                (scaler.scale(loss) if scaler is not None else loss).backward()
            return _timed(call, device=device, calls=3, samples=15, warmup=6)
        return make
    for preset in dict.fromkeys((big, small)):
        for half in (False, True):
            if half and not cuda:
                continue
            add(f"A3 {preset} update {'float16' if half else 'float32'}",
                f"forward + backward, {micro}-row micro-batch ({micro // 2} act), eager", micro, a3(preset, half))

    began = time.time()
    for i, (key, what, rows, make) in enumerate(items, 1):
        load = load_average()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        timing = make()
        results[key] = {"what": what, "rows": rows, "rows_per_s": rows / timing["s"], **timing, "load_average": load,
                        "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None}
        elapsed = time.time() - began
        print(f"A {i}/{len(items)} {key}: {results[key]['rows_per_s']:,.0f} rows/s "
              f"({timing['s'] * 1e3:.2f} ms, launch {timing['launch']:.2f}); {elapsed:.0f} s, "
              f"ETA {elapsed / i * (len(items) - i):.0f} s", flush=True)
    return {"machine": machine(), "seconds": time.time() - began, "A": results}


# ---------------------------------------------------------------- B: variants and scratch runs


def _scratch(args: argparse.Namespace) -> Path:
    """The scratch runs directory, never the real one; with e1800's run (config, run.json and the
    one checkpoint, read-only) when ``with_start``."""
    scratch = Path(args.runs).resolve()
    os.environ.pop("CATANRL_RUNS_DIR", None)
    from catanrl.runs import runs_root

    real = Path(args.real_runs).resolve() if args.real_runs else runs_root()
    if scratch == real or real in scratch.parents or scratch in real.parents:
        raise SystemExit(f"--runs {scratch} overlaps the real runs directory {real}")
    scratch.mkdir(parents=True, exist_ok=True)
    experiment, variant, epoch, _ = START
    source, target = real / experiment / variant, scratch / experiment / variant
    ckpt = f"e{epoch:06d}.pt"
    if args.command in ("b1", "all") and not (target / "checkpoints" / ckpt).exists():
        (target / "checkpoints").mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "run.json"):
            shutil.copyfile(source / name, target / name)
        shutil.copyfile(source / "checkpoints" / ckpt, target / "checkpoints" / ckpt)
    os.environ["CATANRL_RUNS_DIR"] = str(scratch)
    return scratch


def _variant(raw: dict, name: str):
    from catanrl.config import Variant, config_hash, resolve

    cfg = resolve(raw, experiment=EXPERIMENT, variant=name)
    return Variant(EXPERIMENT, name, None, {}, cfg, config_hash(cfg))


def _quiet(cfg: dict) -> dict:
    """No W&B, no evals of any kind."""
    cfg = copy.deepcopy(cfg)
    (cfg.get("meta") or {}).pop("wandb", None)
    cfg["eval"] = {"select": None, "final": []}
    for key in ("kind", "params"):
        cfg.pop(key, None)
    return cfg


def _overrides(cfg: dict, items: list[str]) -> dict:
    from catanrl.config import merge, parse_overrides

    return merge(cfg, parse_overrides(items)) if items else cfg


def b1_variant(args: argparse.Namespace, scratch: Path):
    experiment, variant, epoch, sha = START
    stored = json.loads((scratch / experiment / variant / "config.json").read_text())["config"]
    cfg = _quiet(stored)
    cfg["init_from"] = {"ckpt": f"{experiment}/{variant}@e{epoch}#{sha}", "parts": ["actor", "critic"]}
    cfg["train"].update(iterations=args.iterations, min_replay=args.min_replay)
    cfg["runtime"].pop("time_limit_hours", None)
    if args.threads:
        cfg["runtime"]["threads"] = args.threads
    if args.tiny:
        cfg["env"].update(num_envs=8)
        cfg["env"]["scenarios"][0]["maps"] = 16
        cfg["runtime"].update(device="cpu", threads=2)
        cfg["train"].update(kernels="torch", sims=16, fast_sims=8, pending=4, fast_pending=2, steps_per_iteration=4,
                            search_rows=32, min_replay=8, minibatch_size=8, updates_per_iteration=2,
                            replay_capacity=1000)
    return _variant(_overrides(cfg, args.overrides), f"az{'-' + args.tag if args.tag else ''}")


def b2_variant(args: argparse.Namespace, precision: str):
    from catanrl.config import load_plan

    plan = load_plan(PPO_CONFIG)
    (v,) = plan.variants
    cfg = _quiet(v.config)
    cfg["eval"].pop("champion", None)
    cfg["train"].update(stop_epoch=args.epochs, precision=precision, amp=precision == "float16")
    if args.threads:
        cfg["runtime"]["threads"] = args.threads
    if args.tiny:
        cfg["env"].update(num_envs=8)
        cfg["env"]["scenarios"][0]["maps"] = 16
        cfg["model"] = {"preset": "graph-tiny", "args": {"shared": True}}
        cfg["runtime"].update(device="cpu", threads=2, compile=False)
        cfg["train"].update(bptt_horizon=8, minibatch_size=64, max_minibatch_size=32, total_timesteps=128 * 64)
    name = f"ppo-{'fp16' if precision == 'float16' else 'fp32'}{'-' + args.tag if args.tag else ''}"
    return _variant(_overrides(cfg, args.overrides), name)


def _fresh(scratch: Path, v) -> None:
    """Remove a previous bench run of this name (in the scratch directory only)."""
    path = scratch / v.experiment / v.name
    if path.exists():
        shutil.rmtree(path)


def _metrics(v) -> list[dict]:
    from catanrl.runs import load_experiment

    return load_experiment(v.ref).metrics()


# ---------------------------------------------------------------- B1 + C: AlphaZero self-play


def simulate(code, t0, t1, key, gpu, speedup: float) -> float:
    """The recorded stretch with a device ``speedup`` times faster: the seconds it takes.

    The ops are the main thread's, in order: ``SUBMIT`` (a batch queued: ``gpu`` is the device's
    seconds on it, from when it reached the batch to its last download) and ``RESULT`` (the wait
    for a batch). Everything between ops, and a submit itself, is CPU time and stays as measured;
    the device runs the batches in the order submitted, each ``gpu / speedup`` long, never
    finishing one before its submit has returned."""
    now = free = 0.0
    finish: dict[int, float] = {}
    previous = float(t0[0])
    for i in range(len(code)):
        now += float(t0[i]) - previous
        if code[i] == SUBMIT:
            start = max(now, free)
            now += float(t1[i]) - float(t0[i])
            free = max(start + float(gpu[i]) / speedup, now)
            finish[int(key[i])] = free
        elif code[i] == RESULT:
            now = max(now, finish.pop(int(key[i]), now))
        else:  # STEP: a marker
            now += float(t1[i]) - float(t0[i])
        previous = float(t1[i])
    return now


class SelfPlayTrace:
    """The timers around the search pipeline and the trainer's other network calls."""

    def __init__(self, iterations: int, steps: int) -> None:
        self.iterations, self.steps = iterations, steps
        self.ops: list[tuple] = []     # (code, t0, t1, key, rows)
        self.events: list[tuple] = []  # per op: (start, end) CUDA timing events of a submit, or None
        self.sync_gpu: list[float] = []  # per op: a submit's seconds where nothing is queued (no CUDA)
        self.done: list[dict] = []
        self.traces: dict[str, object] = {}
        self.extra = {"learn_s": 0.0, "refresh_s": 0.0, "check_kernels_s": 0.0}
        self.began = time.time()
        self.step_times: list[float] = []

    def install(self) -> None:
        import torch

        from catanrl.search import pipeline
        from catanrl.train import alphazero

        trace = self
        submit, result = pipeline.Pipeline.submit, pipeline.Pipeline.result
        step, drop = alphazero.SelfPlay.step, alphazero.SelfPlay.drop_trees
        learn, refresh, check = alphazero.learn, alphazero.SelfPlay.refresh, alphazero.SelfPlay.check_kernels

        def timed_submit(self, batch, rows):
            queued = self.queued and self.evaluator.device.type == "cuda"
            t0 = time.perf_counter()
            if queued:
                first = torch.cuda.Event(enable_timing=True)
                first.record()
            submit(self, batch, rows)
            if queued:
                last = torch.cuda.Event(enable_timing=True)
                last.record()
            t1 = time.perf_counter()
            trace.ops.append((SUBMIT, t0, t1, id(batch), rows))
            trace.events.append((first, last) if queued else None)
            trace.sync_gpu.append(0.0 if queued else t1 - t0)

        def timed_result(self, batch):
            t0 = time.perf_counter()
            out = result(self, batch)
            trace.ops.append((RESULT, t0, time.perf_counter(), id(batch), 0))
            trace.events.append(None)
            trace.sync_gpu.append(0.0)
            return out

        def timed_step(self, iteration, index, *rest, **kwargs):
            t0 = time.perf_counter()
            out = step(self, iteration, index, *rest, **kwargs)
            t1 = time.perf_counter()
            trace.ops.append((STEP, t1, t1, 0, 0))
            trace.events.append(None)
            trace.sync_gpu.append(0.0)
            trace.step_times.append(t1 - t0)
            done = len(trace.step_times)
            if done % 8 == 0 or index + 1 == trace.steps:
                total = trace.iterations * trace.steps
                elapsed = time.time() - trace.began
                recent = statistics.mean(trace.step_times[-8:])
                print(f"  B1 iteration {iteration + 1}/{trace.iterations} step {index + 1}/{trace.steps}: "
                      f"{recent:.2f} s/step, {elapsed:.0f} s elapsed, ETA {recent * (total - done):.0f} s "
                      f"(load {os.getloadavg()[0]:.1f})", flush=True)
            return out

        def timed_drop(self):
            out = drop(self)
            trace.close_iteration()
            return out

        def stopwatch(fn, name):
            def wrapped(*a, **k):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fn(*a, **k)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                trace.extra[name] += time.perf_counter() - t0
                return out
            return wrapped

        pipeline.Pipeline.submit, pipeline.Pipeline.result = timed_submit, timed_result
        alphazero.SelfPlay.step, alphazero.SelfPlay.drop_trees = timed_step, timed_drop
        alphazero.learn = stopwatch(learn, "learn_s")
        alphazero.SelfPlay.refresh = stopwatch(refresh, "refresh_s")
        alphazero.SelfPlay.check_kernels = stopwatch(check, "check_kernels_s")

    def close_iteration(self) -> None:
        """The iteration's self-play is over: read the device times, summarise, simulate."""
        import numpy as np
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ops = self.ops
        if not ops:
            return
        code = np.array([o[0] for o in ops], dtype=np.int8)
        t0 = np.array([o[1] for o in ops])
        t1 = np.array([o[2] for o in ops])
        key = np.array([o[3] for o in ops], dtype=np.int64)
        rows = np.array([o[4] for o in ops], dtype=np.int64)
        gpu = np.array([pair[0].elapsed_time(pair[1]) / 1e3 if pair is not None else sync
                        for pair, sync in zip(self.events, self.sync_gpu)])
        measured = float(t1[-1] - t0[0])
        waits = float((t1 - t0)[code == RESULT].sum())
        index = len(self.done)
        self.traces.update({f"i{index}_code": code, f"i{index}_t0": t0 - t0[0], f"i{index}_t1": t1 - t0[0],
                            f"i{index}_key": key, f"i{index}_gpu": gpu, f"i{index}_rows": rows})
        submits = int((code == SUBMIT).sum())
        self.done.append({
            "traced_s": measured, "submits": submits, "rows": int(rows.sum()),
            "gpu_busy_s": float(gpu.sum()), "wait_s": waits, "cpu_s": measured - waits,
            "gpu_busy_share": float(gpu.sum()) / measured, "wait_share": waits / measured,
            "gpu_us_per_row": float(gpu.sum()) / max(int(rows.sum()), 1) * 1e6,
            "rows_per_submit": float(rows.sum()) / max(submits, 1),
            "simulated_s": {str(s): simulate(code, t0, t1, key, gpu, s) for s in SPEEDUPS},
            "load_average": load_average(), **self.extra,
        })
        self.extra = dict.fromkeys(self.extra, 0.0)
        self.ops, self.events, self.sync_gpu = [], [], []


def child_b1(args: argparse.Namespace) -> None:
    import numpy as np

    from catanrl.runner import run_variant

    scratch = Path(os.environ["CATANRL_RUNS_DIR"])
    v = b1_variant(args, scratch)
    _fresh(scratch, v)
    trace = SelfPlayTrace(int(v.config["train"]["iterations"]), int(v.config["train"]["steps_per_iteration"]))
    trace.install()
    before = load_average()
    began = time.time()
    status = run_variant(v, argv=["tools/gpu_bench.py", *sys.argv[1:]])
    total = time.time() - began
    rows = [r for r in _metrics(v) if "throughput/iteration_s" in r]
    iterations = []
    for row, timed in zip(rows, trace.done):
        iterations.append({
            "iteration": row["epoch"], "iteration_s": row["throughput/iteration_s"],
            "selfplay_s": row["throughput/selfplay_s"], "monitor_s": row.get("throughput/monitor_s"),
            "games_per_hour": row["selfplay/games_per_hour"], "leaves_per_s": row["selfplay/leaves_per_s"],
            "rounds_per_step": row["selfplay/rounds"], "decisions_per_s": row["selfplay/decisions_per_s"],
            "replay": row["replay"], "learned": "train/policy_loss" in row, **timed})
    out = Path(args.out)
    name = f"b1{'-' + args.tag if args.tag else ''}"
    np.savez_compressed(out / f"{name}.trace.npz", **trace.traces)
    _write(out / f"{name}.json", {
        "machine": machine(), "status": status, "seconds": total, "load_before": before, "variant": v.ref,
        "threads": v.config["runtime"]["threads"], "config_hash": v.config_hash,
        "train": v.config["train"], "runtime": v.config["runtime"], "iterations": iterations,
        "summary": summarize_b1(iterations)})
    print(f"== {v.ref}: {status} in {total:.0f} s", flush=True)
    if status != "finished":
        raise SystemExit(1)


def summarize_b1(iterations: list[dict], skip: int = 3) -> dict:
    """Medians over the iterations after the first ``skip`` (the openings), or over all if few."""
    late = iterations[skip:] or iterations
    if not late:
        return {}
    keys = ("iteration_s", "selfplay_s", "games_per_hour", "leaves_per_s", "rounds_per_step", "gpu_busy_share",
            "wait_share", "gpu_us_per_row", "learn_s", "refresh_s", "check_kernels_s", "monitor_s", "traced_s")
    out = {k: statistics.median(float(i[k] or 0.0) for i in late) for k in keys}
    out["iterations"] = [i["iteration"] for i in late]
    out["simulated_s"] = {s: statistics.median(i["simulated_s"][s] for i in late)
                          for s in late[0]["simulated_s"]}
    return out


# ---------------------------------------------------------------- B2 + C: graph-m PPO


def child_b2(args: argparse.Namespace) -> None:
    import torch

    from catanrl.runner import run_variant
    from catanrl.train import puffer
    from catanrl.train.puffer_policy import PufferPolicy

    scratch = Path(os.environ["CATANRL_RUNS_DIR"])
    v = b2_variant(args, args.precision)
    _fresh(scratch, v)
    epochs_total = int(v.config["train"]["stop_epoch"])
    clock = {"forward_s": 0.0, "calls": 0}
    epochs: list[dict] = []
    began = time.time()
    forward_eval, step_epoch = PufferPolicy.forward_eval, puffer.CatanPuffeRL.step_epoch

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed_forward(self, observations, state=None):
        sync()
        t0 = time.perf_counter()
        out = forward_eval(self, observations, state)
        sync()
        clock["forward_s"] += time.perf_counter() - t0
        clock["calls"] += 1
        return out

    def timed_epoch(self):
        clock.update(forward_s=0.0, calls=0)
        t0 = time.perf_counter()
        logs = step_epoch(self)
        wall = time.perf_counter() - t0
        row = {"epoch": int(self.epoch), "wall_s": wall, "rollout_forward_s": clock["forward_s"],
               "rollout_forwards": clock["calls"], "load_average": load_average()}
        if logs is not None:
            row.update(steps_per_s=logs.get("throughput/agent_steps_per_s"), collect_s=logs.get("throughput/collect_s"),
                       train_s=logs.get("throughput/train_s"))
        epochs.append(row)
        elapsed = time.time() - began
        rate = row.get("steps_per_s") or 0.0
        print(f"  B2 {args.precision} epoch {len(epochs)}/{epochs_total}: {rate:,.0f} steps/s, {wall:.1f} s "
              f"(collect {row.get('collect_s') or 0:.1f} of which forwards {clock['forward_s']:.1f}, "
              f"train {row.get('train_s') or 0:.1f}); {elapsed:.0f} s elapsed, "
              f"ETA {wall * (epochs_total - len(epochs)):.0f} s (load {os.getloadavg()[0]:.1f})", flush=True)
        return logs

    PufferPolicy.forward_eval, puffer.CatanPuffeRL.step_epoch = timed_forward, timed_epoch
    before = load_average()
    status = run_variant(v, argv=["tools/gpu_bench.py", *sys.argv[1:]])
    total = time.time() - began
    peak = max((r.get("memory/peak_GiB", 0.0) for r in _metrics(v)), default=None)
    out = Path(args.out)
    name = f"b2-{'fp16' if args.precision == 'float16' else 'fp32'}{'-' + args.tag if args.tag else ''}"
    _write(out / f"{name}.json", {
        "machine": machine(), "status": status, "seconds": total, "load_before": before, "variant": v.ref,
        "precision": args.precision, "threads": v.config["runtime"]["threads"], "config_hash": v.config_hash,
        "compile": v.config["runtime"].get("compile"), "peak_GiB": peak, "epochs": epochs,
        "summary": summarize_b2(epochs, args.warmup)})
    print(f"== {v.ref}: {status} in {total:.0f} s", flush=True)
    if status != "finished":
        raise SystemExit(1)


def summarize_b2(epochs: list[dict], warmup: int) -> dict:
    late = [e for e in epochs[warmup:] if e.get("steps_per_s")] or [e for e in epochs if e.get("steps_per_s")]
    if not late:
        return {}
    out = {k: statistics.median(float(e[k]) for e in late)
           for k in ("steps_per_s", "wall_s", "collect_s", "train_s", "rollout_forward_s")}
    out["epochs"] = [e["epoch"] for e in late]
    out["network_share"] = (out["rollout_forward_s"] + out["train_s"]) / out["wall_s"]
    out["rollout_other_s"] = out["collect_s"] - out["rollout_forward_s"]
    return out


# ---------------------------------------------------------------- table and predict


def load_results(out: Path) -> dict:
    """Every part found in an output directory, merged."""
    merged: dict = {}
    for path in sorted(out.glob("*.json")):
        if path.name != "bench.json":
            merged[path.stem] = json.loads(path.read_text())
    return merged


def rows_of(results: dict) -> dict[str, tuple[float, str]]:
    """The table's rows of one machine: name -> (value, unit); higher is better unless the unit is s."""
    rows: dict[str, tuple[float, str]] = {}
    for key, item in (results.get("a") or {}).get("A", {}).items():
        rows[key] = (item["rows_per_s"], "rows/s")
    for name, part in results.items():
        summary = part.get("summary") or {}
        if name.startswith("b1") and summary:
            label = f"B1 self-play ({name}, {part['threads']} threads)"
            rows[f"{label} s/iteration"] = (summary["iteration_s"], "s")
            rows[f"{label} games/h"] = (summary["games_per_hour"], "games/h")
            rows[f"{label} leaves/s"] = (summary["leaves_per_s"], "leaves/s")
            rows[f"{label} device busy share"] = (summary["gpu_busy_share"], "share")
            rows[f"{label} main-thread wait share"] = (summary["wait_share"], "share")
            rows[f"{label} device us/row"] = (summary["gpu_us_per_row"], "us")
        if name.startswith("b2") and summary:
            label = f"B2 graph-m PPO ({name})"
            rows[f"{label} steps/s"] = (summary["steps_per_s"], "steps/s")
            rows[f"{label} collect s"] = (summary["collect_s"], "s")
            rows[f"{label} rollout forwards s"] = (summary["rollout_forward_s"], "s")
            rows[f"{label} train s"] = (summary["train_s"], "s")
            rows[f"{label} network share"] = (summary["network_share"], "share")
    return rows


def print_table(columns: dict[str, dict]) -> None:
    names = list(columns)
    tables = {name: rows_of(results) for name, results in columns.items()}
    for name, results in columns.items():
        m = next((part["machine"] for part in results.values() if "machine" in part), {})
        print(f"{name}: {m.get('gpu')} (driver {m.get('driver')}), torch {m.get('torch')} (CUDA {m.get('torch_cuda')}), "
              f"{m.get('cpu_model')} ({m.get('cpu_affinity')} of {m.get('cpu_count')} CPUs), load {m.get('load_average')}")
    keys = list(dict.fromkeys(k for t in tables.values() for k in t))
    width = max((len(k) for k in keys), default=10)
    header = f"{'':<{width}}  " + "  ".join(f"{n:>14}" for n in names)
    header += "".join(f"  {n + '/' + names[0]:>14}" for n in names[1:])
    print(header)
    for key in keys:
        cells = [tables[n].get(key) for n in names]
        line = f"{key:<{width}}  " + "  ".join(f"{c[0]:>14,.3f}" if c else f"{'-':>14}" for c in cells)
        for c in cells[1:]:
            line += f"  {c[0] / cells[0][0]:>13.2f}x" if c and cells[0] and cells[0][0] else f"  {'-':>14}"
        print(line)


def cmd_predict(args: argparse.Namespace) -> None:
    import numpy as np

    data = np.load(args.trace)
    count = len({k.split("_")[0] for k in data.files})
    for s in args.speedup:
        seconds = []
        for i in range(count):
            part = [data[f"i{i}_{k}"] for k in ("code", "t0", "t1", "key", "gpu")]
            seconds.append(simulate(*part, s))
        measured = [float(data[f"i{i}_t1"][-1]) for i in range(count)]
        print(f"speedup {s:g}: self-play seconds per iteration {[round(x, 1) for x in seconds]} "
              f"(measured {[round(x, 1) for x in measured]})")


# ---------------------------------------------------------------- main


def _child(args: argparse.Namespace, command: str, extra: list[str]) -> None:
    argv = [sys.executable, "-P", __file__, command, "--child", *args.passthrough, *extra]
    if subprocess.run(argv, env=os.environ.copy(), check=False).returncode != 0:
        raise SystemExit(f"{command} {' '.join(extra)} failed")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="the directory of the JSON parts (and bench.json)")
    common.add_argument("--runs", default=None, help="a scratch runs directory (created; never the real one)")
    common.add_argument("--real-runs", default=None, help="where e1800's run is copied from (default: the main "
                        "worktree's runs/)")
    common.add_argument("--iterations", type=int, default=5, help="B1: self-play iterations")
    common.add_argument("--min-replay", type=int, default=4096, help="B1: train.min_replay (the run's: 50,000)")
    common.add_argument("--epochs", type=int, default=10, help="B2: epochs per precision")
    common.add_argument("--warmup", type=int, default=3, help="B2: first epochs left out of the medians")
    common.add_argument("--precision", default=None, choices=("float32", "float16"), help="B2: one precision only")
    common.add_argument("--threads", type=int, default=0, help="runtime.threads (default: the config's 12)")
    common.add_argument("--tag", default="", help="a suffix for the run and the part's file name")
    common.add_argument("--tiny", action="store_true", help="a CPU check of the tool: tiny shapes and networks")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    common.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    for name in ("all", "a", "b1", "b2"):
        sub.add_parser(name, parents=[common])
    table = sub.add_parser("table")
    table.add_argument("columns", nargs="+", metavar="NAME=DIR")
    predict = sub.add_parser("predict")
    predict.add_argument("trace")
    predict.add_argument("--speedup", type=float, action="append", required=True)
    args = parser.parse_args(argv)

    if args.command == "table":
        print_table({c.split("=", 1)[0]: load_results(Path(c.split("=", 1)[1])) for c in args.columns})
        return
    if args.command == "predict":
        cmd_predict(args)
        return

    os.environ["WANDB_MODE"] = "disabled"
    for key in ("WANDB_API_KEY", "CATANRL_WANDB_PROJECT", "CATANRL_WANDB_ENTITY"):
        os.environ.pop(key, None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    args.passthrough = [a for a in argv[1:] if a != "--child"]
    if args.command in ("b1", "b2", "all") and not args.runs:
        raise SystemExit("--runs: a scratch runs directory")

    if args.child:
        if args.command == "a":
            _write(out / "a.json", bench_a(args))
        elif args.command == "b1":
            child_b1(args)
        else:
            child_b2(args)
        return
    if args.command != "a":
        _scratch(args)
    if args.command in ("a", "all"):
        _child(args, "a", [])
    if args.command in ("b1", "all"):
        _child(args, "b1", [])
    if args.command in ("b2", "all"):
        for precision in ([args.precision] if args.precision else ["float32", "float16"]):
            if args.tiny and precision == "float16":
                continue  # float16 autocast is the CUDA path
            _child(args, "b2", ["--precision", precision])
    results = load_results(out)
    _write(out / "bench.json", results)
    print_table({"this": results})


if __name__ == "__main__":
    main()
