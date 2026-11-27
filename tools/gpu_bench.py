"""How fast our workloads run on a GPU, and how much of that is the GPU's to give.

One script for every machine (this box, a Modal container): the same code, one JSON per part and a
printed table. Nothing under ``src/`` is edited: the timers inside the trainers and the agents are
wrapped around their functions from here, in the stretch's own process.

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
B. End-to-end stretches in a scratch runs directory (never the real one: the checkpoints they
   read are copied into it; W&B off; no eval record), each in its own process, with a progress
   line (rate and ETA):
   B1  AlphaZero self-play with a run's stored train config (``--b1-run``: az-2p-graph-m-uniform/
       main+1500, the engine's cache on) from a checkpoint (``--b1-start``: the config's own
       init_from, 2p-graph-m-3): a fresh run with an empty replay, ``--iterations`` iterations,
       no evals. ``train.min_replay`` is lowered (``--min-replay``) so the learning step and the
       evaluator's rebuild are inside the timed iterations once games have ended (the second
       iteration on with the default run's config). The games all start at once, so the first
       three iterations are openings and left out of the medians: compare the same iteration
       numbers between machines.
   B2  graph-m PPO from scratch with graph-m-2p-lr's config (configs/experiments/graph-m-2p-lr.yaml),
       ``--epochs`` epochs of the 3B-step schedule (``train.stop_epoch``), float32 and float16, no
       monitors, no select, no finals.
   B3  the standard eval row's search agent: ``--games`` games of heldout-2p-fr, ``--envs`` in
       flight, MCTS on ``--net`` (1,024 simulations, 64 pending, fused16) against ``--opponent``,
       through the eval harness's ``play_schedule`` (nothing is recorded).
   B4  one deployed-style search: a single game, the float32 kernels, 64 pending, 64-row strict
       forwards, ``--search-sims`` simulations a move for ``--moves`` moves after a warm-up one.
C. The CPU/GPU split of B, from the same processes:
   B1  every ``Pipeline.submit`` is bracketed by two CUDA timing events (the device's time on that
       batch: uploads, forwards, downloads), every graph replay inside it by two more (the
       forwards alone), and every ``Pipeline.result`` is timed (the main thread's wait). The loop
       is pipelined (two groups take turns on the device), so the device's time and the CPU's
       overlap and do not add up to the wall-clock. The prediction for a device ``s`` times faster
       replays the recorded sequence: the CPU segments as measured, every batch's device time
       divided by ``s``, the device serving the batches in order (:func:`simulate`; ``s = 1`` must
       give back the measured time, ``s = inf`` is the CPU's floor). Two predictions: the forwards
       alone faster (the transfers as measured: they are the bus's and the host memory's, not the
       card's arithmetic) and the whole bracket faster. The trace is saved (``*.trace.npz``) for
       ``predict``. The learning step is timed with synchronisation; the last iteration's runs
       under ``torch.profiler`` (CUDA activity), whose kernels' seconds are the device's share.
   B2  ``PufferPolicy.forward_eval`` is timed with synchronisation (the rollout's network time);
       the rest of collect is the engine and PuffeRL's bookkeeping; train is all on the device.
   B3, B4  the agents call the evaluator and wait: every ``BatchEvaluator._replay`` (and
       ``_exact``) is timed whole (its blocking upload, the forward, the download and the wait:
       the main thread does nothing else meanwhile) and its graph replay is bracketed by two CUDA
       events (the forward alone). The same trace and predictions as B1, a call being a batch
       submitted and waited for at once; "the whole call faster" is then an upper bound (the
       call's own work on the host is inside it), "the forwards alone" the one to read. If the
       agents queue through the ``Pipeline`` instead, B1's brackets apply. The measurement
       starts after the first act that searches (the kernels' build, the graphs' captures).
D. One inference forward of ``--net``, operation by operation: fused16 at 512 rows (self-play, the
   eval row), fused16 at 64 and the float32 kernels at 64 (the deployed search), on real rows
   through the network's own encoder. One eager forward is recorded (every ATen call on the device
   and every call of our kernels, in order); each call is then issued again, many times, while the
   device is still busy with a blocking matrix product, between two CUDA events: all its launches
   are queued before the device reaches the first event, so the events are apart by the device's
   own time, launches excluded (a sample whose launches outlasted the blocker is dropped; the
   median of the rest). Grouped: half-precision matrix products, float32 ones (shapes and achieved
   GFLOP/s), each of our kernels, other device operations; beside them the transfers alone
   (pageable and page-locked, both ways), the captured graph's replay (wall-clock and, by the same
   events, device time: the difference is the launch) and the sum of the operations against it.
   ``torch.profiler`` runs last as a cross-check (the kernels' own names and seconds).
E. Hardware beside every stretch: one ``nvidia-smi`` child sampling once a second (utilisation,
   the memory controller's, memory in use, power, temperature) and the PCIe link as it is under
   load; mean and peak per stretch in its JSON.
F. One summary (``summary.json`` and a table): per workload the wall-clock, the device's share,
   the CPU's floor, the predicted speedup with a device twice as fast and one infinitely fast,
   peak device memory, and what the device's time is made of.

    # everything on this machine (free card; ~40 min on an RTX 2070 Super)
    env -u WANDB_API_KEY WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/python tools/gpu_bench.py all --out /tmp/gpu-bench
    # the same in a few seconds a part: does everything run on this card?
    .venv/bin/python tools/gpu_bench.py all --quick --out /tmp/gpu-bench-quick
    # parts (--runs: the scratch runs directory, <out>/runs unless given)
    .venv/bin/python tools/gpu_bench.py a --out /tmp/gpu-bench
    .venv/bin/python tools/gpu_bench.py ops --out /tmp/gpu-bench [--no-profiler]
    .venv/bin/python tools/gpu_bench.py b1 --out /tmp/gpu-bench [--threads 32 --tag t32]
    .venv/bin/python tools/gpu_bench.py b1 --out /tmp/gpu-bench --b1-run az-2p-kata/main+1500 \\
        --b1-start az-2p-kata/main+1500@e1800#7c44ec80540a   # the graph-s run of the earlier tables
    .venv/bin/python tools/gpu_bench.py b2 --out /tmp/gpu-bench
    .venv/bin/python tools/gpu_bench.py eval --out /tmp/gpu-bench [--games 64 --envs 64]
    .venv/bin/python tools/gpu_bench.py search --out /tmp/gpu-bench [--search-sims 32768 --moves 6]
    .venv/bin/python tools/gpu_bench.py summary --out /tmp/gpu-bench
    # the table of one or more machines' results (ratios to the first)
    .venv/bin/python tools/gpu_bench.py table local=/tmp/gpu-bench l40s=/tmp/l40s
    # this machine's self-play (or eval row: b3.trace.npz) with a device s times faster
    .venv/bin/python tools/gpu_bench.py predict /tmp/gpu-bench/b1.trace.npz --speedup 3.2
    # CPU check of the tool itself (tiny, a few minutes)
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python tools/gpu_bench.py all --tiny \\
        --out /tmp/gpu-bench-cpu
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import copy
import json
import math
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
B1_RUN = "az-2p-graph-m-uniform/main+1500"  # the run whose stored train config B1 plays
NET = "2p-graph-m-3"  # configs/registry.yaml: the network of D, B3 and B4 (and where B1_RUN starts)
OPPONENT = "BELIEF:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:worlds=16"  # the eval row's other seat
SUITE = "heldout-2p-fr"
PPO_CONFIG = REPO / "configs" / "experiments" / "graph-m-2p-lr.yaml"
EXPERIMENT = "gpu-bench"
SPEEDUPS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, float("inf"))
SHOWN = ("1.5", "2.0", "3.0", "5.0", "inf")  # the speedups of the printed predictions
SUBMIT, RESULT, STEP = 0, 1, 2
QUICK = False  # --quick: every timing loop cut to a few seconds


def net_ref(name: str) -> str:
    """A checkpoint ref as given, or the ref of a model name of configs/registry.yaml (specs do
    not resolve the names yet)."""
    if "@" in name:
        return name
    import yaml

    models = yaml.safe_load((REPO / "configs" / "registry.yaml").read_text())["models"]
    if name not in models:
        raise SystemExit(f"{name!r} is neither a checkpoint ref (run/variant@eN#sha) nor a model of "
                         "configs/registry.yaml")
    return models[name]["ref"]


def ref_parts(ref: str) -> tuple[str, str, int]:
    """(experiment, variant, epoch) of ``experiment/variant@eN[#sha]``."""
    match = re.fullmatch(r"([^/@]+)/([^@]+)@e(\d+)(#\w+)?", ref)
    if match is None:
        raise SystemExit(f"{ref!r} is not a checkpoint ref (run/variant@eN#sha)")
    return match.group(1), match.group(2), int(match.group(3))




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

    cpu_model, cpuinfo = None, {}  # a sandbox may hide the model's name: family/model/stepping still tell
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines():
            key, _, value = (part.strip() for part in line.partition(":"))
            if key == "model name":
                cpu_model = value
            elif key in ("vendor_id", "cpu family", "model", "stepping", "cpu MHz", "cache size", "siblings",
                         "cpu cores"):
                cpuinfo[key] = value
            elif key == "flags":
                cpuinfo["avx2"], cpuinfo["avx512f"] = " avx2" in f" {value}", " avx512f" in f" {value}"
    cgroup = None
    with contextlib.suppress(OSError):
        cgroup = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_model": cpu_model, "cpuinfo": cpuinfo, "cpu_count": os.cpu_count(), "cpu_affinity": len(os.sched_getaffinity(0)),
        "cgroup_cpu_max": cgroup, "load_average": load_average(),
        "nvcc": (_run("nvcc", "--version") or "").splitlines()[-1:] or None,
        "cxx": (_run(os.environ.get("CXX", "c++"), "--version") or "").splitlines()[:1] or None,
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


# ---------------------------------------------------------------- E: hardware


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:  # [N/A], [Not Supported]
        return None


def available_gb() -> float:
    with contextlib.suppress(OSError, StopIteration, ValueError):
        line = next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
        return int(line.split()[1]) / 2**20
    return float("inf")


def rss_gb() -> float:
    with contextlib.suppress(OSError, StopIteration, ValueError):
        line = next(l for l in Path("/proc/self/status").read_text().splitlines() if l.startswith("VmRSS:"))
        return int(line.split()[1]) / 2**20
    return 0.0


class Hardware:
    """``nvidia-smi`` beside a stretch: one child that prints a sample a second (utilisation, the
    memory controller's utilisation, memory in use, power, temperature), read on a thread, and
    the PCIe link asked again at the end of a stretch (an idle link may have dropped a
    generation). The card's memory is everybody's (a desktop's included): ``baseline_MiB`` is
    what was in use before this process touched the device, ``process_MiB`` this process's own
    at the end of a stretch where the driver tells (not in a container)."""

    QUERY = ("utilization.gpu", "utilization.memory", "memory.used", "power.draw", "temperature.gpu")
    NAMES = ("util_pct", "memory_util_pct", "memory_MiB", "power_W", "temperature_C")
    CARD = ("name", "memory.total", "memory.used", "pcie.link.gen.current", "pcie.link.width.current",
            "pcie.link.gen.max", "pcie.link.width.max")

    def __init__(self, enabled: bool = True, period_ms: int = 1000) -> None:
        self.samples: list[tuple] = []  # (time.time(), *NAMES)
        self.stretches: dict[str, dict] = {}
        self.child = None
        self.card: dict = {}
        self.period_ms = period_ms
        self.began = time.time()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.index = visible if visible.isdigit() else "0"
        if not enabled or shutil.which("nvidia-smi") is None:
            return
        self.card = self._card()
        if not self.card:
            return
        try:
            self.child = subprocess.Popen(
                ["nvidia-smi", "-i", self.index, f"--query-gpu={','.join(self.QUERY)}",
                 "--format=csv,noheader,nounits", "-lms", str(period_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        threading.Thread(target=self._read, daemon=True).start()
        atexit.register(self.stop)

    def _query(self, what: str, fields: tuple[str, ...]) -> list[list[str]]:
        out = _run("nvidia-smi", "-i", self.index, f"--query-{what}={','.join(fields)}",
                   "--format=csv,noheader,nounits")
        return [[cell.strip() for cell in line.split(",")] for line in (out or "").splitlines()]

    def _card(self) -> dict:
        rows = self._query("gpu", self.CARD)
        if not rows or len(rows[0]) != len(self.CARD):
            return {}
        name, total, used, gen, width, gen_max, width_max = rows[0]
        return {"gpu": name, "memory_total_MiB": _number(total), "baseline_MiB": _number(used),
                "link": {"gen": _number(gen), "width": _number(width), "gen_max": _number(gen_max),
                         "width_max": _number(width_max)}}

    def _process(self) -> float | None:
        mine = [_number(used) for pid, used in (row for row in self._query("compute-apps", ("pid", "used_memory"))
                                                 if len(row) == 2) if pid == str(os.getpid())]
        return mine[0] if mine else None

    def _read(self) -> None:
        for line in self.child.stdout:
            cells = [cell.strip() for cell in line.split(",")]
            if len(cells) == len(self.QUERY):
                self.samples.append((time.time(), *(_number(cell) for cell in cells)))

    def window(self, windows: list[tuple[float, float]]) -> dict:
        """Mean and peak of every reading over the samples inside ``windows`` ([start, end] in time.time())."""
        rows = [s for s in list(self.samples) if any(a <= s[0] <= b for a, b in windows)]
        out: dict = {"samples": len(rows), "seconds": sum(b - a for a, b in windows)}
        for i, name in enumerate(self.NAMES, 1):
            values = [s[i] for s in rows if s[i] is not None]
            out[name] = {"mean": statistics.mean(values), "peak": max(values)} if values else None
        return out

    def stretch(self, name: str, began: float | None = None, windows: list[tuple[float, float]] | None = None) -> dict:
        """The readings of a stretch (from ``began`` to now, or ``windows``), kept and printed."""
        if self.child is None:
            return {}
        if windows is None:
            windows = [(began if began is not None else self.began, time.time())]
        out = self.window(windows)
        out["link"] = (self._card() or {}).get("link")
        out["process_MiB"] = self._process()
        self.stretches[name] = out
        print(f"  hardware, {name}: {self.line(out)}", flush=True)
        return out

    def line(self, out: dict) -> str:
        if not out.get("samples"):
            return "no sample (a stretch under a second)"

        def cell(key: str, unit: str, mean: bool = True) -> str:
            value = out.get(key)
            if value is None:
                return "n/a"
            return (f"{value['mean']:,.0f} mean, " if mean else "") + f"{value['peak']:,.0f} peak {unit}"

        link = out.get("link") or {}
        own = out.get("process_MiB")
        return (f"utilisation {cell('util_pct', '%')}; memory controller {cell('memory_util_pct', '%')}; memory "
                f"{cell('memory_MiB', 'MiB', mean=False)} ({self.card.get('baseline_MiB') or 0:,.0f} before the process"
                + (f", {own:,.0f} this process's" if own is not None else "") + f"); power {cell('power_W', 'W')}; "
                f"{cell('temperature_C', 'C', mean=False)}; PCIe gen {link.get('gen') or 0:g} "
                f"x{link.get('width') or 0:g} (of gen {link.get('gen_max') or 0:g} "
                f"x{link.get('width_max') or 0:g}); {out['samples']} samples")

    def stop(self) -> None:
        child, self.child = self.child, None
        if child is not None:
            child.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=5)

    def report(self) -> dict | None:
        """The part's ``hardware``: the card, every sample, the stretches, and the whole process's readings."""
        if not self.card:
            return None
        return {**self.card, "period_s": self.period_ms / 1e3, "fields": ["t_s", *self.NAMES],
                "samples": [[round(s[0] - self.began, 2), *s[1:]] for s in list(self.samples)],
                "stretches": self.stretches, "all": self.window([(self.began, time.time())])}


def peak_memory(hardware: Hardware | None = None) -> dict:
    """The process's peak device memory: torch's own counters, and nvidia-smi's peak over the
    card's memory before the process (the CUDA context and the libraries' workspaces included)."""
    import torch

    out: dict = {"torch_allocated_GiB": None, "torch_reserved_GiB": None, "card_MiB": None, "over_baseline_MiB": None}
    if torch.cuda.is_available():
        out.update(torch_allocated_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   torch_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 3))
    report = hardware.report() if hardware is not None else None
    if report and report["all"].get("memory_MiB"):
        peak = report["all"]["memory_MiB"]["peak"]
        out.update(card_MiB=peak, over_baseline_MiB=peak - (report.get("baseline_MiB") or 0.0))
    return out




# ---------------------------------------------------------------- A


def real_rows(count: int, *, seed: int = 0, encoder=None):
    """``count`` acting rows and the same positions' waiting rows (the other seat's: only the
    no-op) of random base-2p-fr games, float32 [count, obs]: full level without the dice input,
    or ``encoder``'s rows (a network's own: its level, its dice input)."""
    import cppanatron as cpp
    import numpy as np

    from catanrl import games as game_configs
    from catanrl.envs.encoding import dims_for

    dims = dims_for(["base-2p-fr"])
    if encoder is None:
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

    if QUICK:
        calls, samples, warmup = max(1, calls // 5), max(3, samples // 5), max(2, warmup // 5)

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
    hardware = Hardware(not args.no_hardware)
    opened = time.time()
    scale = (lambda n: max(2, n // 16)) if tiny else (lambda n: n)
    dims, acting_np, waiting_np = real_rows(16 if tiny else 64 if QUICK else 256)
    torch.manual_seed(0)
    results: dict[str, dict] = {}
    # A0: one core of this machine on the engine (the same random games again: play and encode)
    seconds = []
    for _ in range(3):
        started = time.perf_counter()
        real_rows(len(acting_np))
        seconds.append(time.perf_counter() - started)
    results["A0 engine, one thread (random games played and encoded)"] = {
        "rows": len(acting_np), "seconds": min(seconds), "rows_per_s": len(acting_np) / min(seconds), "launch": 1.0,
        "what": "positions/s of real_rows: Python-driven engine moves and two encodes a position; the best of 3"}

    def model_of(preset: str):
        return build_policy_model({"model": {"preset": preset, "args": {"shared": True}}, "obs": {"level": "full"}},
                                  dims).to(device)

    def tile(rows: np.ndarray, n: int) -> torch.Tensor:
        return torch.from_numpy(np.resize(rows, (n, rows.shape[1]))).to(device)

    presets = ("graph-tiny",) if tiny else ("graph-s", "graph-m")
    small, big = presets[0], presets[-1]
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
    hardware.stretch("A", opened)
    hardware.stop()
    return {"machine": machine(), "seconds": time.time() - began, "A": results, "peak": peak_memory(hardware),
            "hardware": hardware.report()}


# ---------------------------------------------------------------- D: one forward, operation by operation

#: A matrix product -> the position of its left factor among the arguments.
MATRIX = {"mm": 0, "addmm": 1, "bmm": 0, "baddbmm": 1, "matmul": 0, "linear": 0}
ALLOCATIONS = ("empty", "new_empty", "empty_like", "empty_strided", "new_empty_strided")
GROUPS = ("half matrix products", "float32 matrix products", "our kernels", "other device operations")
OURS = ("topo_kernel", "entity16_kernel", "gather16_kernel", "assemble16_kernel", "entity_kernel", "gather_kernel",
        "pool_kernel", "assemble_kernel")  # graph_kernels.cu's, as a profiler names them
QUEUED = ("each call of one recorded eager forward issued again (2 to 256 times, about 2 ms of device time) behind a "
          "blocking matrix product, between two CUDA events on the stream: device time, launches excluded; the median "
          "of the samples whose launches were all queued before the device reached the first event")
LOOPED = "each call of one recorded forward repeated in a loop on the CPU (the tool's own check: no device)"


def _tensors(value):
    """Every tensor of an argument (lists and tuples opened)."""
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def aten_entry(func, args, kwargs, out, phase: str, device) -> dict | None:
    """What a recorded ATen call is to the device: its group, label, shapes and (a matrix
    product's) floating-point operations; None for a call that launches nothing there (a view,
    an allocation, a call on host tensors)."""
    import torch

    name = func._schema.name.split("::")[-1]
    base = name.rstrip("_")
    inputs = [*_tensors(args), *_tensors(tuple(kwargs.values()))]
    outputs = list(_tensors(out))
    if base in ALLOCATIONS or not outputs or not any(t.device.type == device.type for t in (*inputs, *outputs)):
        return None
    if not func._schema.is_mutable:
        held = {t.untyped_storage().data_ptr() for t in inputs}
        if all(t.untyped_storage().data_ptr() in held for t in outputs):
            return None  # a view of an input (or the input itself: a cast to its own type)
    entry = {"op": name, "phase": phase, "dtype": str(outputs[0].dtype).removeprefix("torch."),
             "shapes": [list(t.shape) for t in inputs], "flops": None, "call": lambda: func(*args, **kwargs)}
    if base in MATRIX and len(args) > MATRIX[base] + 1 and all(
            isinstance(a, torch.Tensor) for a in args[MATRIX[base]:MATRIX[base] + 2]):
        left, right = args[MATRIX[base]], args[MATRIX[base] + 1]
        half = left.dtype == torch.float16
        entry.update(group=GROUPS[0] if half else GROUPS[1], flops=2.0 * left.shape[-1] * outputs[0].numel(),
                     label=f"{base} {list(left.shape)} x {list(right.shape)} ({phase})")
    else:
        entry.update(group=GROUPS[3], label=f"{name} ({entry['dtype']})")
    return entry


def record_forward(forward, fused, device) -> list[dict]:
    """One eager ``forward()``: its device calls in order (:func:`aten_entry`'s dicts, and one per
    call of our kernels), each with a ``call`` that issues it again on the same tensors. The ATen
    calls are seen by a dispatch mode; our kernels are plain functions of a compiled module, so
    for the forward a stand-in takes the module's place on ``fused`` (the fused forwards read
    ``self.K`` at every call) and notes each call by name. The phases of a fused forward: "first"
    (the rows' projections and the injection, up to the first entities call), "layers", "heads"
    (from the pooling on)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    entries: list[dict] = []
    state = {"phase": "first" if fused is not None else "forward", "depth": 0}

    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            if state["depth"] == 0:  # not one a kernel call makes inside
                entry = aten_entry(func, args, kwargs, out, state["phase"], device)
                if entry is not None:
                    entries.append(entry)
            return out

    class Kernels:
        def __init__(self, module) -> None:
            self.module = module

        def __getattr__(self, name):
            target = getattr(self.module, name)
            if not callable(target):
                return target

            def call(*args, **kwargs):
                if name == "pool":
                    state["phase"] = "heads"
                phase = state["phase"]
                state["depth"] += 1
                try:
                    out = target(*args, **kwargs)
                finally:
                    state["depth"] -= 1
                label = name if name in ("topo", "pool") or name.startswith("assemble") else f"{name} ({phase})"
                entries.append({"op": name, "phase": phase, "dtype": None, "flops": None, "group": GROUPS[2],
                                "label": label, "shapes": [list(t.shape) for t in _tensors(args)][:6],
                                "call": lambda: target(*args, **kwargs)})
                if name.startswith("entities") and phase == "first":
                    state["phase"] = "layers"
                return out
            return call

    module = fused.K if fused is not None else None
    if fused is not None:
        fused.K = Kernels(module)
    try:
        with Recorder():
            forward()
    finally:
        if fused is not None:
            fused.K = module
    return entries


class DeviceTimer:
    """Device seconds of a call, launches excluded (:data:`QUEUED`). The blocker is a float32
    matrix product repeated until it outlasts three times the launches; a sample whose first
    event had already happened when the last launch returned is dropped and the blocker
    doubled. With no sample kept the eager time (launches included: an upper bound) stands in
    and the entry says so (``queued`` false). On the CPU: a loop."""

    def __init__(self, device) -> None:
        import torch

        self.device = device
        self.cuda = device.type == "cuda"
        if self.cuda:
            self.block = torch.randn(1024, 1024, device=device)
            self.out = torch.empty_like(self.block)
            self.block_s = _timed(self.blocker, device=device, calls=5, samples=5, warmup=5)["s"]

    def blocker(self) -> None:
        import torch

        torch.mm(self.block, self.block, out=self.out)

    def measure(self, call, *, samples: int = 9, target: float = 2e-3) -> dict:
        import torch

        samples = 3 if QUICK else samples
        for _ in range(3):
            call()
        if self.cuda:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        for _ in range(8):
            call()
        launch = (time.perf_counter() - start) / 8
        if self.cuda:
            torch.cuda.synchronize(self.device)
        eager = (time.perf_counter() - start) / 8
        repeats = int(min(max(target / max(eager, 1e-7), 2), 256))
        if not self.cuda:
            start = time.perf_counter()
            for _ in range(repeats):
                call()
            return {"s": (time.perf_counter() - start) / repeats, "queued": False, "repeats": repeats,
                    "launch_s": launch, "eager_s": eager}
        blockers = max(1, math.ceil((3 * repeats * launch + 1e-3) / self.block_s))
        kept: list[float] = []
        dropped = 0
        while len(kept) < samples and dropped < 6:
            for _ in range(blockers):
                self.blocker()
            first = torch.cuda.Event(enable_timing=True)
            first.record()
            for _ in range(repeats):
                call()
            last = torch.cuda.Event(enable_timing=True)
            last.record()
            queued = not first.query()  # the device has not reached the first event: every launch waited
            last.synchronize()
            if queued:
                kept.append(first.elapsed_time(last) / 1e3 / repeats)
            else:
                dropped += 1
                blockers *= 2
        return {"s": statistics.median(kept) if kept else eager, "s_min": min(kept, default=eager),
                "s_max": max(kept, default=eager), "queued": bool(kept), "samples": len(kept), "dropped": dropped,
                "repeats": repeats, "blockers": blockers, "launch_s": launch, "eager_s": eager}


def by_operation(entries: list[dict]) -> list[dict]:
    """The operations of a forward summed by (group, label): calls, seconds, share of the sum, and
    a matrix product's achieved GFLOP/s; the groups in :data:`GROUPS`' order, the largest first."""
    table: dict[tuple, dict] = {}
    for entry in entries:
        row = table.setdefault((entry["group"], entry["label"]),
                               {"group": entry["group"], "label": entry["label"], "count": 0, "s": 0.0, "flops": 0.0})
        row["count"] += 1
        row["s"] += entry["s"]
        row["flops"] += entry["flops"] or 0.0
    total = sum(row["s"] for row in table.values()) or 1.0
    for row in table.values():
        row["share"] = row["s"] / total
        row["GFLOP_per_s"] = row["flops"] / row["s"] / 1e9 if row["flops"] and row["s"] else None
    return sorted(table.values(), key=lambda row: (GROUPS.index(row["group"]), -row["s"]))


def group_seconds(rows: list[dict]) -> dict[str, dict]:
    out = {group: {"s": 0.0, "count": 0, "flops": 0.0} for group in GROUPS}
    for row in rows:
        for key in ("s", "count", "flops"):
            out[row["group"]][key] += row[key]
    return out


def kernel_group(name: str) -> str:
    """The group of a device activity by the name the profiler gives it (a kernel's, a copy's)."""
    low = name.lower()
    if any(ours in low for ours in OURS):
        return GROUPS[2]
    if "memcpy htod" in low or "memcpy dtoh" in low:
        return "transfers"
    if "gemm" in low:
        return GROUPS[0] if re.search(r"h\d*gemm|fp16|half", low) else GROUPS[1]
    return GROUPS[3]


def device_activity(events, kind=None) -> dict:
    """A profile's device activities (CUDA kernels, copies) by name and by :func:`kernel_group`."""
    from torch.autograd import DeviceType

    kind = DeviceType.CUDA if kind is None else kind
    names: dict[str, list] = {}
    for event in events:
        if event.device_type != kind or getattr(event, "is_user_annotation", False):
            continue
        row = names.setdefault(event.name, [0.0, 0])
        row[0] += event.time_range.elapsed_us() / 1e6
        row[1] += 1
    groups = {group: 0.0 for group in (*GROUPS, "transfers")}
    for name, (seconds, _) in names.items():
        groups[kernel_group(name)] += seconds
    ranked = sorted(names.items(), key=lambda item: -item[1][0])
    return {"device_s": sum(seconds for seconds, _ in names.values()), "groups": groups,
            "kernels": [{"name": name[:160], "s": seconds, "count": count, "group": kernel_group(name)}
                        for name, (seconds, count) in ranked[:60]]}


def _profiler():
    """How ``torch.profiler`` records this device's kernels: (activities, the events' device
    type, a note), found by profiling one small operation (twice: CUPTI's first session may
    record nothing). CUDA activity alone where that records kernels, else with the CPU's. On
    the CPU (the tool's own check) the CPU's events stand in. Raises where neither records."""
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    if not torch.cuda.is_available():
        return [ProfilerActivity.CPU], DeviceType.CPU, "CPU events (the tool's own check: nested calls count twice)"
    block = torch.randn(64, 64, device="cuda")
    for activities, note in (([ProfilerActivity.CUDA], "CUDA activity alone"),
                             ([ProfilerActivity.CPU, ProfilerActivity.CUDA], "CPU and CUDA activity")):
        for _ in range(2):
            with profile(activities=activities) as session:
                torch.mm(block, block)
                torch.cuda.synchronize()
            if device_activity(session.events())["kernels"]:
                return activities, DeviceType.CUDA, note
    raise RuntimeError("torch.profiler records no CUDA kernel here (no CUPTI?)")


def profiled(fn):
    """``fn()`` under ``torch.profiler``: (its result, the device's activities by name:
    :func:`device_activity`). Where the profiler cannot run (no CUPTI in a container) ``fn``
    still runs, once, and the second is ``{"error": ...}``."""
    import torch
    from torch.profiler import profile

    try:
        activities, kind, note = _profiler()
        session = profile(activities=activities)
        session.__enter__()
    except Exception as error:  # noqa: BLE001 - whatever the profiler raises, the stretch goes on
        return fn(), {"error": f"torch.profiler did not start: {error!r}"}
    try:
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except BaseException:
        with contextlib.suppress(Exception):
            session.__exit__(None, None, None)
        raise
    try:
        session.__exit__(None, None, None)
        activity = device_activity(session.events(), kind)
    except Exception as error:  # noqa: BLE001
        return out, {"error": f"torch.profiler could not be read: {error!r}"}
    activity["profiler"] = note
    if not activity["kernels"]:
        activity["error"] = "torch.profiler recorded no device activity"
    return out, activity


def transfer_times(batch, static, outputs, device) -> dict:
    """The transfers of one forward alone, as the evaluator's call makes them (pageable host
    memory, blocking) and as the pipeline does (page-locked, queued): milliseconds with the
    wait, and GB/s. The downloads include the cast of half outputs to float32 on the device."""
    import torch

    k = len(batch)
    source = torch.from_numpy(batch)
    locked = torch.empty(batch.shape, dtype=torch.float32, pin_memory=True).copy_(source)
    targets = [torch.empty(tuple(o[:k].shape), dtype=torch.float32, pin_memory=True) for o in outputs[:2]]
    flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)

    def down_locked():
        for target, output in zip(targets, outputs):
            target.copy_(output[:k], non_blocking=True)
        if len(outputs) > 2:
            flag.copy_(outputs[2].reshape(-1)[:1], non_blocking=True)

    out_bytes = sum(t.numel() * 4 for t in targets) + (4 if len(outputs) > 2 else 0)
    items = {
        "upload, pageable (the evaluator's call)": (lambda: static[:k].copy_(source, non_blocking=False), batch.nbytes),
        "upload, page-locked (the pipeline)": (lambda: static[:k].copy_(locked, non_blocking=True), batch.nbytes),
        "download, pageable (the evaluator's call)": (lambda: tuple(o[:k].float().cpu().numpy() for o in outputs),
                                                      out_bytes),
        "download, page-locked (the pipeline)": (down_locked, out_bytes),
    }
    results = {}
    with torch.inference_mode():
        for name, (fn, size) in items.items():
            timing = _timed(fn, device=device, calls=20, samples=15, warmup=10)
            results[name] = {"s": timing["s"], "bytes": int(size), "GB_per_s": size / timing["s"] / 1e9,
                             "launch": timing["launch"]}
    return results


def profile_forward(model, rows_np, *, device, kernels: str, rows: int, label: str) -> tuple[dict, object]:
    """Part D for one forward shape: (the profile, a function that runs the eager forward once,
    for the profiler's cross-check)."""
    import numpy as np
    import torch

    from catanrl.search.evaluator import BatchEvaluator
    from catanrl.search.pipeline import Pipeline

    cuda = device.type == "cuda"
    began = time.time()
    ev = BatchEvaluator(model, device=device, infer_rows=rows, mode="fast", kernels=kernels)
    batch = np.ascontiguousarray(np.resize(rows_np, (rows, rows_np.shape[1])))
    out: dict = {"kernels": kernels, "rows": rows, "method": QUEUED if cuda else LOOPED}
    timer = DeviceTimer(device)
    # the forward as it runs: the captured graph's replay, and the calls around it
    if cuda:
        graph, static, outputs = ev._graph(rows)
        with torch.inference_mode():
            static[0].copy_(torch.from_numpy(batch))
        obs = static[0]
        out["replay"] = _timed(graph.replay, device=device, calls=50, samples=30, warmup=50)
        out["replay_device"] = timer.measure(graph.replay)
    else:
        obs = torch.from_numpy(batch)

    def eager():
        with torch.inference_mode():
            return ev._forward(obs)

    out["eager"] = _timed(eager, device=device, calls=10, samples=15, warmup=5)
    out["evaluator_call"] = _timed(lambda: ev(batch), device=device, calls=20, samples=20, warmup=10)
    pipe = Pipeline(ev)
    held = pipe.batch(0, rows)
    held.observations[:rows] = batch

    def round_trip():
        pipe.submit(held, rows)
        pipe.result(held)

    out["pipeline_round_trip"] = _timed(round_trip, device=device, calls=20, samples=20, warmup=10)
    if cuda:
        out["transfers"] = transfer_times(batch, static[0], outputs, device)
        with torch.inference_mode():  # the round trips above left other rows' padding: the real rows again
            static[0].copy_(torch.from_numpy(batch))
    # one eager forward recorded, then each of its calls on its own
    with torch.inference_mode():
        entries = record_forward(lambda: ev._forward(obs), ev.fused, device)
        last = time.time()
        for i, entry in enumerate(entries, 1):
            call = entry.pop("call")
            try:
                entry.update(timer.measure(call))
            except Exception as error:  # noqa: BLE001 - a call that cannot be issued again: named, not timed
                entry.update(s=0.0, queued=False, error=repr(error))
            del call
            if time.time() - last >= 2.0 or i == len(entries):
                last = time.time()
                elapsed = time.time() - began
                print(f"  D {label}: operation {i}/{len(entries)} ({i / elapsed:.1f}/s), {elapsed:.0f} s elapsed, "
                      f"ETA {elapsed / i * (len(entries) - i):.0f} s", flush=True)
    table = by_operation(entries)
    out.update(operations=entries, by_operation=table, groups=group_seconds(table),
               sum_s=sum(entry["s"] for entry in entries),
               not_queued=sum(1 for entry in entries if cuda and not entry["queued"]),
               not_timed=[f"{entry['label']}: {entry['error']}" for entry in entries if "error" in entry],
               peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None,
               seconds=time.time() - began)
    return out, eager


def print_profile(label: str, p: dict) -> None:
    ms = lambda timing: f"{timing['s'] * 1e3:.3f} ms"
    print(f"D {label}:")
    if "replay" in p:
        print(f"  the captured graph's replay {ms(p['replay'])} wall-clock (launch {p['replay']['launch']:.2f}), "
              f"{ms(p['replay_device'])} on the device" + ("" if p["replay_device"]["queued"] else " (NOT queued)"))
    print(f"  eager forward {ms(p['eager'])} (launch {p['eager']['launch']:.2f}); the evaluator's call "
          f"{ms(p['evaluator_call'])}; a pipeline round trip {ms(p['pipeline_round_trip'])}")
    print(f"  method: {p['method']}")
    width = max(len(row["label"]) for row in p["by_operation"]) + 4
    print(f"  {'':<{width}}  {'calls':>5}  {'ms':>8}  {'share':>6}  {'GFLOP/s':>8}")
    for group in GROUPS:
        rows = [row for row in p["by_operation"] if row["group"] == group]
        if not rows:
            continue
        g = p["groups"][group]
        rate = f"{g['flops'] / g['s'] / 1e9:>8,.0f}" if g["flops"] and g["s"] else f"{'':>8}"
        print(f"  {group:<{width}}  {g['count']:>5}  {g['s'] * 1e3:>8.3f}  {g['s'] / p['sum_s']:>6.1%}  {rate}")
        for row in rows:
            rate = f"{row['GFLOP_per_s']:>8,.0f}" if row["GFLOP_per_s"] else f"{'':>8}"
            print(f"    {row['label']:<{width - 2}}  {row['count']:>5}  {row['s'] * 1e3:>8.3f}  "
                  f"{row['share']:>6.1%}  {rate}")
    line = f"  sum of the operations {p['sum_s'] * 1e3:.3f} ms"
    if "replay" in p:
        device, wall = p["replay_device"]["s"], p["replay"]["s"]
        line += (f" = {p['sum_s'] / device:.0%} of the replay's device time ({device * 1e3:.3f} ms); the replay's "
                 f"wall-clock is {wall * 1e3:.3f} ms: {(wall - device) * 1e3:.3f} ms ({(wall - device) / wall:.0%}) is "
                 f"its launch and the wait")
    print(line + (f"; {p['not_queued']} operations NOT queued (eager times: upper bounds)" if p["not_queued"] else ""))
    for failure in p.get("not_timed") or ():
        print(f"  NOT TIMED (it could not be issued again; missing from the sum): {failure}")
    for name, t in (p.get("transfers") or {}).items():
        print(f"  {name}: {t['s'] * 1e3:.3f} ms ({t['bytes'] / 1e6:.2f} MB, {t['GB_per_s']:.2f} GB/s)")
    check = p.get("profiler")
    if check and "error" not in check:
        parts = ", ".join(f"{group} {seconds * 1e3:.3f}" for group, seconds in check["groups"].items() if seconds)
        print(f"  torch.profiler, a forward's device activity: {check['device_s'] * 1e3:.3f} ms ({parts}); "
              "the largest: "
              + "; ".join(f"{k['name'][:48]} {k['s'] * 1e3:.3f} ms x{k['count']:g}" for k in check["kernels"][:6]))
    elif check:
        print(f"  torch.profiler: {check['error']}")
    print(flush=True)


OPS_SHAPES = (("fused16", 512), ("fused16", 64), ("fused", 64))  # self-play and the eval row; the deployed search
OPS_TINY = (("torch", 32), ("torch", 8))


def ops_path(args: argparse.Namespace) -> Path:
    return Path(args.out) / f"ops{'-' + args.tag if args.tag else ''}.json"


def ops_record(args: argparse.Namespace, label: str, profile: dict, **top) -> None:
    """One profile into ops.json (each runs in a process of its own: read, merged, written)."""
    path = ops_path(args)
    results = json.loads(path.read_text()) if path.exists() else {}
    results.update(top)
    results.setdefault("profiles", {})[label] = profile
    _write(path, results)


def child_ops(args: argparse.Namespace) -> None:
    """Part D for one shape (``--shape kernels:rows``) of ``--net``, into ops.json."""
    import torch

    from catanrl.agents.spec import resolve_agent

    kernels, _, rows = args.shape.partition(":")
    rows = int(rows)
    label = f"{kernels} x {rows} rows"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.tiny:
        raise SystemExit("ops: the fused kernels need a CUDA device (--tiny checks the tool on the CPU)")
    hardware = Hardware(not args.no_hardware)
    began = time.time()
    ref = net_ref(args.net)
    try:
        # The network as a search agent holds it (its rows' encoder with it), on the PyTorch path.
        agent = resolve_agent(f"MCTS:ckpt={ref}:sims=16:dets=1").native(device=str(device), infer_rows=8, mode="fast")
        model = agent.model
        _, rows_np, _ = real_rows(16 if args.tiny else 64 if QUICK else 256, encoder=agent.encoder)
        if rows_np.shape[1] != int(agent.evaluator.obs_size):
            raise RuntimeError(f"the encoder's rows are {rows_np.shape[1]} wide, the network reads "
                               f"{agent.evaluator.obs_size}")
        profile, eager = profile_forward(model, rows_np, device=device, kernels=kernels, rows=rows, label=label)
    except Exception as error:  # noqa: BLE001 - the shape's failure is its result: the other shapes run
        import traceback

        traceback.print_exc()
        ops_record(args, label, {"kernels": kernels, "rows": rows, "error": repr(error)})
        raise SystemExit(f"D {label}: FAILED ({error!r})") from None
    hardware.stretch(f"D {label}", began)
    profile.update(peak=peak_memory(hardware), hardware=hardware.report())
    ops_record(args, label, profile, machine=machine(), net=ref, name=args.net)
    # torch.profiler last (CUPTI stays loaded once started: nothing above is timed with it)
    if not args.no_profiler:
        count = 5 if QUICK else 20
        for _ in range(3):
            eager()

        def run():
            for _ in range(count):
                eager()
        try:
            _, activity = profiled(run)
        except Exception as error:  # noqa: BLE001
            activity = {"error": repr(error)}
        if "error" not in activity:
            activity.update(forwards=count, device_s=activity["device_s"] / count,
                            groups={g: s / count for g, s in activity["groups"].items()},
                            kernels=[{**k, "s": k["s"] / count, "count": k["count"] / count}
                                     for k in activity["kernels"]])
        profile["profiler"] = activity
        ops_record(args, label, profile)
    hardware.stop()
    print_profile(label, profile)




# ---------------------------------------------------------------- B: variants and scratch runs


def _seed(real: Path, scratch: Path, ref: str) -> None:
    """A checkpoint's run in the scratch directory: its config, run.json, aliases and that one
    checkpoint, copied from the real one (once)."""
    experiment, variant, epoch = ref_parts(ref)
    source, target = real / experiment / variant, scratch / experiment / variant
    ckpt = f"e{epoch:06d}.pt"
    if (target / "checkpoints" / ckpt).exists():
        return
    if not (source / "checkpoints" / ckpt).exists():
        raise SystemExit(f"{ref}: no {source / 'checkpoints' / ckpt} (--real-runs: where the runs are)")
    (target / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "run.json", "aliases.jsonl"):
        if (source / name).exists() and not (target / name).exists():
            shutil.copyfile(source / name, target / name)
    shutil.copyfile(source / "checkpoints" / ckpt, target / "checkpoints" / ckpt)


def _stored(scratch: Path, run: str) -> Path:
    """Where B1's run's stored config is kept in the scratch directory (not a run: beside them)."""
    return scratch / ".gpu-bench" / f"{run.replace('/', '__')}.config.json"


def b1_start(args: argparse.Namespace, stored: dict) -> str:
    """The checkpoint B1 starts from: ``--b1-start``, or the stored config's own ``init_from``."""
    start = args.b1_start or (stored.get("init_from") or {}).get("ckpt")
    if not start:
        raise SystemExit(f"--b1-start: {args.b1_run}'s config has no init_from to start from")
    return start


def _scratch(args: argparse.Namespace) -> Path:
    """The scratch runs directory, never the real one, with what the command reads copied in:
    B1's run's stored config, and the runs (config, run.json, the one checkpoint) of B1's
    start, of ``--net`` and of the opponent's network."""
    scratch = Path(args.runs).resolve()
    os.environ.pop("CATANRL_RUNS_DIR", None)
    from catanrl.runs import runs_root

    real = Path(args.real_runs).resolve() if args.real_runs else runs_root()
    if scratch == real or real in scratch.parents or scratch in real.parents:
        raise SystemExit(f"--runs {scratch} overlaps the real runs directory {real}")
    scratch.mkdir(parents=True, exist_ok=True)
    refs: list[str] = []
    if args.command in ("b1", "all"):
        source, target = real / args.b1_run / "config.json", _stored(scratch, args.b1_run)
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        elif not target.exists():
            raise SystemExit(f"--b1-run {args.b1_run}: no {source} (--real-runs: where the runs are)")
        refs.append(b1_start(args, json.loads(target.read_text())["config"]))
    if args.command in ("ops", "eval", "search", "all"):
        refs.append(net_ref(args.net))
    if args.command in ("eval", "search", "all"):
        refs += [part[5:] for part in args.opponent.split(":") if part.startswith("ckpt=")]
    for ref in dict.fromkeys(refs):
        _seed(real, scratch, ref)
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
    stored = json.loads(_stored(scratch, args.b1_run).read_text())["config"]
    cfg = _quiet(stored)
    cfg["init_from"] = {"ckpt": b1_start(args, stored), "parts": ["actor", "critic"]}
    cfg["train"].update(iterations=args.iterations, min_replay=args.min_replay)
    cfg["runtime"].pop("time_limit_hours", None)
    if args.threads:
        cfg["runtime"]["threads"] = args.threads
    if QUICK or args.tiny:
        # short iterations of short searches; the games cut at the first turn after the placements,
        # so that positions reach the replay (they enter it when their game ends) and the learning
        # step runs
        cfg["env"]["scenarios"][0]["turn_limit"] = 3
        cfg["train"].update(steps_per_iteration=16, updates_per_iteration=4, min_replay=min(args.min_replay, 64),
                            sims=min(int(cfg["train"]["sims"]), 160), fast_sims=min(int(cfg["train"]["fast_sims"]), 40))
    if args.tiny:
        cfg["env"].update(num_envs=8)
        cfg["env"]["scenarios"][0]["maps"] = 16
        cfg["runtime"].update(device="cpu", threads=2)
        cfg["train"].update(kernels="torch", sims=16, fast_sims=8, pending=4, fast_pending=2, steps_per_iteration=16,
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
    elif QUICK:  # the real network and device on the check's small rollout, not compiled
        cfg["env"].update(num_envs=8)
        cfg["env"]["scenarios"][0]["maps"] = 16
        cfg["runtime"].update(threads=2, compile=False)
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




def trace_summary(code, t0, t1, key, gpu, fwd, rows) -> dict:
    """A traced stretch in numbers: the device's seconds (``gpu``: the brackets; ``fwd``: the
    forwards alone), the main thread's waits, and :func:`simulate`'s seconds with a faster
    device, the whole bracket divided (``simulated_s``) and the forwards alone, the rest of a
    bracket as measured (``simulated_forward_s``)."""
    import numpy as np

    measured = float(t1[-1] - t0[0])
    fwd = np.minimum(fwd, gpu)
    waits = float((t1 - t0)[code == RESULT].sum())
    submits, total = int((code == SUBMIT).sum()), int(rows.sum())
    busy, forward = float(gpu.sum()), float(fwd.sum())
    return {
        "traced_s": measured, "submits": submits, "rows": total,
        "gpu_busy_s": busy, "forward_s": forward, "wait_s": waits, "cpu_s": measured - waits,
        "gpu_busy_share": busy / measured, "forward_share": forward / measured, "wait_share": waits / measured,
        "gpu_us_per_row": busy / max(total, 1) * 1e6, "forward_us_per_row": forward / max(total, 1) * 1e6,
        "rows_per_submit": total / max(submits, 1),
        "simulated_s": {str(s): simulate(code, t0, t1, key, gpu, s) for s in SPEEDUPS},
        "simulated_forward_s": {str(s): simulate(code, t0, t1, key, gpu - fwd + fwd / s, 1.0) for s in SPEEDUPS},
    }


class Trace:
    """The main thread's ops around the network, whichever way a workload reaches it.

    A queued ``Pipeline.submit`` is a ``SUBMIT`` between two CUDA timing events (the device's
    time on the batch: uploads, forwards, downloads) and its ``Pipeline.result`` a ``RESULT``
    (the wait). A call of the evaluator (``BatchEvaluator._replay``, ``_exact``: the agents'
    way, and a pipeline's that does not queue) blocks from its upload to its download: it is a
    ``SUBMIT`` of the call's whole time and a ``RESULT`` at once. Under both, every graph replay
    is between two events of its own: the forward alone (with no captured graph, and on the
    CPU, the forward's time is the op's). ``slot`` names who is calling (an agent)."""

    def __init__(self) -> None:
        self.ops: list[tuple] = []  # (code, t0, t1, key, rows, slot)
        self.gpu: list = []         # per op: seconds, or a submit's (first, last) events until resolve()
        self.fwd: list = []         # per op: seconds, or its replays' event pairs until resolve()
        self.active = False         # ops are recorded
        self.inside = False         # a traced call is running: the calls it makes are its own
        self.forwards: list | None = None  # the replays of the op in progress
        self.slot = 0
        self.totals: dict[int, list[float]] = {}  # slot -> [submits, rows, device s, forward s, wait s]
        self.resolved = 0
        self.calls = 0

    @staticmethod
    def event():
        import torch

        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def add(self, code: int, t0: float, t1: float, key: int, rows: int, gpu, fwd) -> None:
        self.ops.append((code, t0, t1, key, rows, self.slot))
        self.gpu.append(gpu)
        self.fwd.append(fwd)

    def install(self) -> None:
        import torch

        from catanrl.search import evaluator, pipeline

        trace = self
        submit, result = pipeline.Pipeline.submit, pipeline.Pipeline.result

        def timed_submit(self, batch, rows):
            if not trace.active or trace.inside or not self.queued:
                return submit(self, batch, rows)  # not queued: the evaluator's call below is the op
            cuda = self.evaluator.device.type == "cuda"
            trace.inside, trace.forwards = True, []
            t0 = time.perf_counter()
            try:
                first = trace.event() if cuda else None
                submit(self, batch, rows)
                last = trace.event() if cuda else None
            finally:
                forwards, trace.forwards, trace.inside = trace.forwards, None, False
            t1 = time.perf_counter()
            trace.add(SUBMIT, t0, t1, id(batch), rows, (first, last) if cuda else t1 - t0,
                      forwards if cuda and forwards else None)

        def timed_result(self, batch):
            if not trace.active or trace.inside or getattr(batch, "_called", None) is not None:
                return result(self, batch)
            trace.inside = True
            t0 = time.perf_counter()
            try:
                out = result(self, batch)
            finally:
                trace.inside = False
            trace.add(RESULT, t0, time.perf_counter(), id(batch), 0, 0.0, 0.0)
            return out

        def timed_call(call):
            def timed(self, chunk, *rest, **kwargs):
                if not trace.active or trace.inside:
                    return call(self, chunk, *rest, **kwargs)
                trace.inside, trace.forwards = True, []
                t0 = time.perf_counter()
                try:
                    out = call(self, chunk, *rest, **kwargs)
                finally:
                    forwards, trace.forwards, trace.inside = trace.forwards, None, False
                t1 = time.perf_counter()
                trace.calls += 1
                trace.add(SUBMIT, t0, t0, -trace.calls, len(chunk), t1 - t0, forwards or None)
                trace.add(RESULT, t0, t1, -trace.calls, 0, 0.0, 0.0)
                return out
            return timed

        pipeline.Pipeline.submit, pipeline.Pipeline.result = timed_submit, timed_result
        evaluator.BatchEvaluator._replay = timed_call(evaluator.BatchEvaluator._replay)
        evaluator.BatchEvaluator._exact = timed_call(evaluator.BatchEvaluator._exact)
        if torch.cuda.is_available():
            replay = torch.cuda.CUDAGraph.replay

            def timed_replay(self):
                if trace.forwards is None:
                    return replay(self)
                first = trace.event()
                replay(self)
                trace.forwards.append((first, trace.event()))

            torch.cuda.CUDAGraph.replay = timed_replay

    def resolve(self) -> None:
        """Wait for the device and read the events of the ops since the last call."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for i in range(self.resolved, len(self.ops)):
            gpu, fwd = self.gpu[i], self.fwd[i]
            if isinstance(gpu, tuple):
                gpu = gpu[0].elapsed_time(gpu[1]) / 1e3
            fwd = gpu if fwd is None else fwd if isinstance(fwd, float) else min(
                gpu, sum(first.elapsed_time(last) for first, last in fwd) / 1e3)
            self.gpu[i], self.fwd[i] = gpu, fwd
            code, t0, t1, _, rows, slot = self.ops[i]
            total = self.totals.setdefault(slot, [0, 0, 0.0, 0.0, 0.0])
            if code == SUBMIT:
                total[0] += 1
                total[1] += rows
                total[2] += gpu
                total[3] += fwd
            elif code == RESULT:
                total[4] += t1 - t0
        self.resolved = len(self.ops)

    def arrays(self) -> dict:
        """The resolved ops as arrays: code, t0, t1, key, rows, slot, gpu, fwd."""
        import numpy as np

        self.resolve()
        column = lambda i, dtype=float: np.array([op[i] for op in self.ops], dtype=dtype)
        return {"code": column(0, np.int8), "t0": column(1), "t1": column(2), "key": column(3, np.int64),
                "rows": column(4, np.int64), "slot": column(5, np.int8), "gpu": np.array(self.gpu, dtype=float),
                "fwd": np.array(self.fwd, dtype=float)}

    def clear(self) -> None:
        self.resolve()
        self.ops, self.gpu, self.fwd, self.resolved = [], [], [], 0


def summarize(arrays: dict) -> dict:
    return trace_summary(*(arrays[k] for k in ("code", "t0", "t1", "key", "gpu", "fwd", "rows")))


def saved(arrays: dict, prefix: str) -> dict:
    """A trace's arrays under ``prefix`` for the ``.npz`` (the times from its first op's)."""
    start = arrays["t0"][0]
    return {f"{prefix}_{name}": value - start if name in ("t0", "t1") else value for name, value in arrays.items()}


def predictions(simulated: dict, measured: float) -> str:
    return ", ".join(f"{'an infinitely fast one' if s == 'inf' else s + 'x'} {simulated[s]:.1f} s "
                     f"({measured / max(simulated[s], 1e-9):.2f}x)" for s in SHOWN)


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


class SelfPlayTrace(Trace):
    """The trace of self-play's steps, and the timers around the trainer's other network calls."""

    def __init__(self, iterations: int, steps: int, *, profile: bool = True) -> None:
        super().__init__()
        self.iterations, self.steps, self.profile = iterations, steps, profile
        self.done: list[dict] = []
        self.traces: dict[str, object] = {}
        self.extra = {"check_kernels_s": 0.0}  # before an iteration's steps: its row's when it closes
        self.began = time.time()
        self.step_times: list[float] = []
        self.windows: dict[str, list[tuple[float, float]]] = {"self-play": [], "learning step": []}
        self.opened = None  # time.time() of the iteration's first step
        self.batch_s = 0.0
        self.learn_profile: dict | None = None
        self.flush = lambda: None  # the part's files as far as they go (before the profiled learning step)

    def install(self) -> None:
        import torch

        from catanrl.train import alphazero

        super().install()
        trace = self
        step, drop = alphazero.SelfPlay.step, alphazero.SelfPlay.drop_trees
        learn, refresh, check = alphazero.learn, alphazero.SelfPlay.refresh, alphazero.SelfPlay.check_kernels
        batch = alphazero.ReplayBuffer.batch

        def sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def timed_step(self, iteration, index, *rest, **kwargs):
            if trace.opened is None:
                trace.opened = time.time()
            trace.active = True
            t0 = time.perf_counter()
            try:
                out = step(self, iteration, index, *rest, **kwargs)
            finally:
                trace.active = False
            t1 = time.perf_counter()
            trace.add(STEP, t1, t1, 0, 0, 0.0, 0.0)
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
                sync()
                t0 = time.perf_counter()
                out = fn(*a, **k)
                sync()
                seconds = time.perf_counter() - t0
                if name in trace.extra:
                    trace.extra[name] += seconds
                elif trace.done:  # after the iteration's steps: its own row's
                    trace.done[-1][name] += seconds
                return out
            return wrapped

        def timed_batch(self, *a, **k):
            sync()  # the update before is the device's own; what follows is the host's and the bus's
            t0 = time.perf_counter()
            out = batch(self, *a, **k)
            sync()
            trace.batch_s += time.perf_counter() - t0
            return out

        def timed_learn(*a, **k):
            last = trace.profile and len(trace.done) >= trace.iterations
            sync()
            trace.batch_s = 0.0
            began, t0 = time.time(), time.perf_counter()
            if last:
                trace.flush()
                print("  B1 the last learning step under torch.profiler (slower: left out of the medians)", flush=True)
                out, activity = profiled(lambda: learn(*a, **k))
            else:
                out = learn(*a, **k)
            sync()
            seconds = time.perf_counter() - t0
            trace.windows["learning step"].append((began, time.time()))
            if trace.done:
                trace.done[-1].update(learn_s=seconds, learn_batch_s=trace.batch_s, learn_profiled=bool(last))
            if last:
                trace.learn_profile = {**activity, "seconds": seconds, "batch_s": trace.batch_s}
            print(f"  B1 learning step {seconds:.1f} s (replay.batch {trace.batch_s:.1f} s)", flush=True)
            return out

        alphazero.SelfPlay.step, alphazero.SelfPlay.drop_trees = timed_step, timed_drop
        alphazero.learn = timed_learn
        alphazero.ReplayBuffer.batch = timed_batch
        alphazero.SelfPlay.refresh = stopwatch(refresh, "refresh_s")
        alphazero.SelfPlay.check_kernels = stopwatch(check, "check_kernels_s")

    def close_iteration(self) -> None:
        """The iteration's self-play is over: read the device times, summarise, simulate."""
        if not self.ops:
            return
        arrays = self.arrays()
        self.windows["self-play"].append((self.opened or self.began, time.time()))
        self.opened = None
        self.traces.update(saved(arrays, f"i{len(self.done)}"))
        self.done.append({**summarize(arrays), "load_average": load_average(), "learn_s": 0.0, "learn_batch_s": 0.0,
                          "learn_profiled": False, "refresh_s": 0.0, **self.extra})
        self.extra = dict.fromkeys(self.extra, 0.0)
        self.clear()


def learn_split(profile: dict | None, learn_s: float) -> dict:
    """The learning step's device share from its profile: the kernels' seconds (the transfers
    apart) over the unprofiled step's ``learn_s``, and the step with a faster device if nothing
    of the host's overlapped the kernels (it does: the launches; so these are lower bounds)."""
    if not profile or "error" in profile or not learn_s:
        return {"error": (profile or {}).get("error", "no learning step ran under the profiler")}
    kernels = sum(seconds for group, seconds in profile["groups"].items() if group != "transfers")
    kernels = min(kernels, learn_s)
    return {"learn_s": learn_s, "kernel_s": kernels, "device_share": kernels / learn_s,
            "transfer_s": profile["groups"].get("transfers", 0.0), "groups": profile["groups"],
            "profiled_s": profile["seconds"],
            "simulated_s": {str(s): learn_s - kernels * (1.0 - 1.0 / s) for s in SPEEDUPS}}


def child_b1(args: argparse.Namespace) -> None:
    import numpy as np

    from catanrl.runner import run_variant

    scratch = Path(os.environ["CATANRL_RUNS_DIR"])
    v = b1_variant(args, scratch)
    _fresh(scratch, v)
    hardware = Hardware(not args.no_hardware)
    trace = SelfPlayTrace(int(v.config["train"]["iterations"]), int(v.config["train"]["steps_per_iteration"]),
                          profile=not args.no_profiler)
    trace.install()
    before = load_average()
    began = time.time()
    out = Path(args.out)
    name = f"b1{'-' + args.tag if args.tag else ''}"

    def write(status: str, rows: list[dict]) -> dict:
        iterations = []
        for i, timed in enumerate(trace.done):
            row = rows[i] if i < len(rows) else {}
            iterations.append({
                "iteration": row.get("epoch", i + 1), "iteration_s": row.get("throughput/iteration_s"),
                "selfplay_s": row.get("throughput/selfplay_s"), "monitor_s": row.get("throughput/monitor_s"),
                "games_per_hour": row.get("selfplay/games_per_hour"), "leaves_per_s": row.get("selfplay/leaves_per_s"),
                "rounds_per_step": row.get("selfplay/rounds"), "decisions_per_s": row.get("selfplay/decisions_per_s"),
                "cache_hits": row.get("selfplay/cache_hits"), "replay": row.get("replay"),
                "learned": "train/policy_loss" in row or timed["learn_s"] > 0.0, **timed})
        summary = summarize_b1(iterations, 0 if QUICK or args.tiny else 3)
        if summary:
            summary["learn"] = learn_split(trace.learn_profile, summary["learn_s"])
            summary["predicted"] = predict_b1(summary)
        result = {
            "machine": machine(), "status": status, "seconds": time.time() - began, "load_before": before,
            "variant": v.ref, "run": args.b1_run, "start": v.config["init_from"]["ckpt"],
            "threads": v.config["runtime"]["threads"], "config_hash": v.config_hash, "train": v.config["train"],
            "runtime": v.config["runtime"], "iterations": iterations, "summary": summary,
            "learn_profile": trace.learn_profile, "peak": peak_memory(hardware), "hardware": hardware.report()}
        np.savez_compressed(out / f"{name}.trace.npz", **trace.traces)
        _write(out / f"{name}.json", result)
        return result

    trace.flush = lambda: write("running", [])
    status = run_variant(v, argv=["tools/gpu_bench.py", *sys.argv[1:]])
    total = time.time() - began
    for stretch, windows in trace.windows.items():
        if windows:
            hardware.stretch(f"B1 {stretch}", windows=windows)
    result = write(status, [r for r in _metrics(v) if "throughput/iteration_s" in r])
    hardware.stop()
    print_b1(result)
    print(f"== {v.ref}: {status} in {total:.0f} s", flush=True)
    if status != "finished":
        raise SystemExit(1)


def summarize_b1(iterations: list[dict], skip: int = 3) -> dict:
    """Medians over the iterations after the first ``skip`` (the openings), or over all if few.
    The learning step's are over the iterations that learned, the profiled one left out (it is
    slower) unless it is the only one; ``other_s`` is an iteration's time outside its self-play,
    its learning step and the evaluator's rebuild (the monitors, the bookkeeping)."""
    late = iterations[skip:] or iterations
    if not late:
        return {}
    median = lambda rows, k: statistics.median(float(i.get(k) or 0.0) for i in rows)
    keys = ("iteration_s", "selfplay_s", "games_per_hour", "leaves_per_s", "rounds_per_step", "gpu_busy_share",
            "forward_share", "wait_share", "gpu_us_per_row", "forward_us_per_row", "rows_per_submit", "refresh_s",
            "check_kernels_s", "monitor_s", "traced_s", "gpu_busy_s", "forward_s")
    plain = [i for i in late if not i.get("learn_profiled")] or late
    out = {k: median(plain if k == "iteration_s" else late, k) for k in keys}
    out["iterations"] = [i["iteration"] for i in late]
    learned = [i for i in iterations if i.get("learned") and i.get("learn_s")]
    learned = [i for i in learned if not i.get("learn_profiled")] or learned
    out["learned_iterations"] = [i["iteration"] for i in learned]
    out["learn_s"] = median(learned, "learn_s") if learned else 0.0
    out["learn_batch_s"] = median(learned, "learn_batch_s") if learned else 0.0
    out["other_s"] = statistics.median(
        max(0.0, float(i.get("iteration_s") or i["traced_s"]) - i["traced_s"] - i["learn_s"] - i["refresh_s"])
        for i in late)
    for which in ("simulated_s", "simulated_forward_s"):
        out[which] = {s: statistics.median(i[which][s] for i in late) for s in late[0][which]}
    return out


def predict_b1(summary: dict) -> dict:
    """An iteration that learns (self-play, the learning step, the rebuild, the rest) with a
    faster device: self-play by the trace (the forwards alone faster, and the whole bracket),
    the learning step by its profile (as measured without one), the rest as measured."""
    learn = summary.get("learn") or {}
    rest = summary["refresh_s"] + summary["other_s"]
    out = {"iteration_s": summary["traced_s"] + summary["learn_s"] + rest, "learn_scaled": "simulated_s" in learn,
           "forwards": {}, "bracket": {}}
    for s in summary["simulated_s"]:
        learn_s = learn["simulated_s"][s] if "simulated_s" in learn else summary["learn_s"]
        out["forwards"][s] = summary["simulated_forward_s"][s] + learn_s + rest
        out["bracket"][s] = summary["simulated_s"][s] + learn_s + rest
    return out


def print_b1(result: dict) -> None:
    s = result["summary"]
    if not s:
        print("B1: no iteration was traced", flush=True)
        return
    p, learn = s["predicted"], s["learn"]
    print(f"B1 {result['run']} from {result['start']}, {result['threads']} threads; medians over iterations "
          f"{s['iterations']}:")
    print(f"  an iteration {s['iteration_s']:.1f} s as logged; one that learns {p['iteration_s']:.1f} s: self-play "
          f"{s['traced_s']:.1f}, the learning step {s['learn_s']:.1f} (iterations {s['learned_iterations']}), the "
          f"evaluator's rebuild {s['refresh_s']:.1f}, the rest {s['other_s']:.1f}")
    print(f"  self-play: {s['leaves_per_s']:,.0f} leaves/s, {s['games_per_hour']:,.0f} games/h, "
          f"{s['rows_per_submit']:.0f} rows a batch; the device busy {s['forward_share']:.0%} of it on the forwards "
          f"({s['forward_us_per_row']:.2f} us a row), {s['gpu_busy_share']:.0%} with the transfers "
          f"({s['gpu_us_per_row']:.2f} us a row); the main thread waits {s['wait_share']:.0%} of it")
    print(f"  self-play with a device s times as fast, the forwards alone (the transfers as measured): "
          f"{predictions(s['simulated_forward_s'], s['traced_s'])}")
    print(f"  self-play, the whole bracket faster (the transfers too): {predictions(s['simulated_s'], s['traced_s'])}")
    if not s["learned_iterations"]:
        print("  NO LEARNING STEP RAN: the replay never reached train.min_replay (more --iterations, or a lower "
              "--min-replay); the iteration below is self-play and the rest alone")
    elif "error" in learn:
        print(f"  the learning step {s['learn_s']:.1f} s (replay.batch {s['learn_batch_s']:.1f} s: the host's rows and "
              f"their upload); no device share: {learn['error']}")
    else:
        groups = ", ".join(f"{g} {seconds:.2f} s" for g, seconds in learn["groups"].items() if seconds)
        print(f"  the learning step {s['learn_s']:.1f} s: the device's kernels {learn['kernel_s']:.2f} s "
              f"({learn['device_share']:.0%}; torch.profiler on the last one: {groups}), replay.batch "
              f"{s['learn_batch_s']:.1f} s (the host's rows and their upload); with a faster device "
              f"{predictions(learn['simulated_s'], s['learn_s'])} at best (the launches overlap the kernels)")
    as_measured = "" if p["learn_scaled"] else " (the learning step as measured)"
    print(f"  an iteration that learns, the forwards alone faster{as_measured}: "
          f"{predictions(p['forwards'], p['iteration_s'])}")
    print(f"  the same, the whole bracket faster: {predictions(p['bracket'], p['iteration_s'])}")
    peak = result["peak"]
    print(f"  peak device memory: {peak['torch_reserved_GiB']} GiB reserved by torch ({peak['torch_allocated_GiB']} "
          f"allocated), {peak['over_baseline_MiB']} MiB on the card over what was there before", flush=True)




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
    hardware = Hardware(not args.no_hardware)
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
    hardware.stretch(f"B2 {args.precision}", began)
    hardware.stop()
    peak = max((r.get("memory/peak_GiB", 0.0) for r in _metrics(v)), default=None)
    out = Path(args.out)
    name = f"b2-{'fp16' if args.precision == 'float16' else 'fp32'}{'-' + args.tag if args.tag else ''}"
    _write(out / f"{name}.json", {
        "machine": machine(), "status": status, "seconds": total, "load_before": before, "variant": v.ref,
        "precision": args.precision, "threads": v.config["runtime"]["threads"], "config_hash": v.config_hash,
        "compile": v.config["runtime"].get("compile"), "peak_GiB": peak, "epochs": epochs,
        "summary": summarize_b2(epochs, args.warmup), "peak": peak_memory(hardware), "hardware": hardware.report()})
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


# ---------------------------------------------------------------- B3, B4 + C: the search agents

TREE_MB = 11.0   # tools/eval_speed_check.py's: host memory of a game's kept tree at 1,024 simulations
PROCESS_GB = 2.5  # torch, CUDA and the networks, before any tree
DECISIONS_PER_GAME = 95.0  # the search seat's, for the ETA until games have finished
LOW_MEMORY = 3  # exit code of the memory guard
CACHE_ENV = "CATANRL_EVAL_CACHE_MB"


class _Enough(Exception):
    """The measured moves are in: the game is left where it is."""


def tree_gb(sims: int, in_flight: int) -> float:
    """The kept trees of ``in_flight`` games (one search seat each), by the simulations."""
    return in_flight * TREE_MB * max(1.0, (sims / 1024) ** 1.6) / 1024


def search_plan(args: argparse.Namespace, which: str) -> dict:
    """B3's and B4's shapes (``--tiny``: the tool's CPU check; ``--quick``: a few moves)."""
    if which == "b3":
        plan = {"name": "B3", "what": "the standard eval row's search agent", "sims": args.eval_sims,
                "kernels": "fused16", "pending": 64, "games": args.games, "envs": args.envs,
                "threads": args.threads or 12, "infer_rows": 512, "moves": 4 if QUICK else 0}
        if args.tiny:
            plan.update(sims=16, kernels="torch", pending=4, games=4, envs=4, threads=2, infer_rows=32, moves=6)
    else:
        plan = {"name": "B4", "what": "one deployed-style search", "sims": 4096 if QUICK else args.search_sims,
                "kernels": "fused", "pending": 64, "games": 1, "envs": 1, "threads": 1, "infer_rows": 64,
                "moves": 3 if QUICK else args.moves}
        if args.tiny:
            plan.update(sims=64, kernels="torch", pending=4, infer_rows=32, moves=3)
    kernels = "" if plan["kernels"] == "torch" else f":kernels={plan['kernels']}"
    plan["spec"] = (f"MCTS:ckpt={net_ref(args.net)}:sims={plan['sims']}:dets=1:pending={plan['pending']}"
                    f":reuse=true:transpositions=true{kernels}")
    return plan


def child_search(args: argparse.Namespace, which: str) -> None:
    """B3 or B4: the games through the eval harness's ``play_schedule`` with the trace on (C).
    The first act that searches is the warm-up (the kernels' build, the graphs' captures, the
    half-precision check): the measurement starts when it returns."""
    import numpy as np
    import torch

    from catanrl.agents.spec import resolve_agent
    from catanrl.eval.harness import play_schedule
    from catanrl.eval.suites import Suite, load_suite
    from catanrl.search import DedupEvaluator

    plan = search_plan(args, which)
    label = plan["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.tiny:
        raise SystemExit(f"{label}: the fused kernels need a CUDA device (--tiny checks the tool on the CPU)")
    megabytes = float(os.environ.get(CACHE_ENV) or 256.0)
    base = load_suite(args.suite)
    players = max(game.num_players for game in base.games())
    suite = Suite(**{**base.__dict__, "boards": min(base.boards, -(-plan["games"] // players))})
    resolved = [resolve_agent(plan["spec"]), resolve_agent(args.opponent)]
    games = suite.schedule(len(resolved))[: plan["games"]]
    in_flight = min(plan["envs"], len(games))
    need = tree_gb(plan["sims"], in_flight) + megabytes / 1024 + max(0.0, (PROCESS_GB if device == "cuda" else 1.0)
                                                                     - rss_gb())
    free = available_gb()
    print(f"{label} {plan['what']}: {plan['spec']} against {args.opponent}; {len(games)} games of {suite.name} "
          f"({suite.determinism}), {in_flight} in flight, {plan['threads']} threads, {plan['infer_rows']}-row forwards"
          + (f", {plan['moves']} searched acts after the warm-up" if plan["moves"] else "")
          + f"; about {need:.1f} GB more of host memory, {free:.1f} GB available", flush=True)
    if free - need < args.min_free_gb:
        raise SystemExit(f"{label}: that would leave {free - need:.1f} GB of host memory, under --min-free-gb "
                         f"{args.min_free_gb}: fewer --envs or simulations, or free memory")

    hardware = Hardware(not args.no_hardware)
    trace = Trace()
    trace.install()  # on the classes, before the agents are built
    agents = [agent.native(device=device, infer_rows=plan["infer_rows"], mode=suite.determinism) for agent in resolved]
    search, opponent = agents
    if not hasattr(search, "search_stats"):
        raise SystemExit(f"{label}: not a search agent")
    inner = search.evaluator.evaluator if isinstance(search.evaluator, DedupEvaluator) else search.evaluator
    slots = {1: "search", 2: "opponent"}
    seconds = {1: 0.0, 2: 0.0}
    acts: list[dict] = []
    state = {"warm": None, "end": None, "stop": False, "start": {}}

    def leaves() -> int:
        return int(search.cache_stats["leaves"]) if search.cache is not None else int(inner.rows)

    def counts() -> dict:
        return {"decisions": int(search.counters["decisions"]), "simulations": int(search.search_stats["simulations"]),
                "leaf_rows": leaves(), "forwarded_rows": int(inner.rows), "forwards": int(inner.forwards),
                "float32_rows": int(getattr(inner, "exact_rows", 0))}

    def acting(slot: int, act):
        def wrapped(rows, *a, **k):
            if state["stop"]:
                raise _Enough
            before = int(search.search_stats["simulations"])
            total = list(trace.totals.get(slot, (0, 0, 0.0, 0.0, 0.0)))
            trace.slot = slot
            t0 = time.perf_counter()
            try:
                out = act(rows, *a, **k)
            finally:
                trace.slot = 0
            trace.resolve()
            t1 = time.perf_counter()
            seconds[slot] += t1 - t0
            done = int(search.search_stats["simulations"]) - before
            if slot != 1 or done <= 0:
                return out
            if state["warm"] is None:  # the warm-up is over: everything from here is measured
                trace.clear()
                trace.totals.clear()
                seconds.update({1: 0.0, 2: 0.0})
                state.update(warm=time.time(), start=counts())
                trace.add(STEP, t1, t1, 0, 0, 0.0, 0.0)
                trace.active = True
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                print(f"{label}: warm-up act {t1 - t0:.1f} s ({done:,} simulations); measuring", flush=True)
                return out
            now = trace.totals.get(slot, (0, 0, 0.0, 0.0, 0.0))
            acts.append({"s": t1 - t0, "rows": len(rows), "simulations": done, "submits": now[0] - total[0],
                         "forwarded_rows": now[1] - total[1], "device_s": now[2] - total[2],
                         "forward_s": now[3] - total[3]})
            if which == "b4":
                a_ = acts[-1]
                elapsed = time.time() - state["warm"]
                left = (plan["moves"] - len(acts)) * elapsed / len(acts)
                print(f"{label} move {len(acts)}/{plan['moves']}: {done:,} simulations in {a_['s']:.2f} s "
                      f"({done / a_['s']:,.0f}/s), the forwards {a_['forward_s'] / a_['s']:.0%} of it, the evaluator's "
                      f"calls {a_['device_s'] / a_['s']:.0%}; {elapsed:.0f} s elapsed, ETA {left:.0f} s", flush=True)
            if plan["moves"] and len(acts) >= plan["moves"]:
                trace.add(STEP, t1, t1, 0, 0, 0.0, 0.0)
                trace.active = False
                state.update(stop=True, end=time.time())
            return out
        return wrapped

    search.act, opponent.act = acting(1, search.act), acting(2, opponent.act)
    rows: list[dict] = []
    stop = threading.Event()
    began = time.time()

    def line() -> str:
        elapsed = time.time() - began
        now = counts()
        mine = [row["decisions"][row["seats"].index(0)] for row in rows]
        per_game = sum(mine) / len(mine) if len(mine) >= 8 else DECISIONS_PER_GAME
        if plan["moves"]:
            left = (plan["moves"] - len(acts)) * elapsed / max(len(acts), 1) if acts else None
        else:
            left = max(0.0, len(games) * per_game - now["decisions"]) * elapsed / now["decisions"] if now[
                "decisions"] else None
        return (f"{label}: {len(rows)}/{len(games)} games, {now['decisions']} decisions "
                f"({now['decisions'] / max(elapsed, 1e-9):.1f}/s), {now['simulations'] / max(elapsed, 1e-9):,.0f} "
                f"simulations/s, {now['forwarded_rows'] / max(elapsed, 1e-9):,.0f} rows forwarded/s, RSS "
                f"{rss_gb():.1f} GB, {available_gb():.1f} GB available, {elapsed:.0f} s elapsed, ETA "
                + (f"{left:.0f} s" if left is not None else "unknown"))

    def tick() -> None:
        last = time.time()
        while not stop.wait(0.5):
            if available_gb() < args.min_free_gb:  # at once: nothing is recorded, so nothing is lost
                print(f"{label}: under {args.min_free_gb} GB of host memory available, stopped ({line()})", flush=True)
                os._exit(LOW_MEMORY)
            if time.time() - last >= args.every:
                last = time.time()
                print(line(), flush=True)

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        play_schedule(agents, games, suite=suite, device=device, num_envs=plan["envs"], threads=plan["threads"],
                      infer_rows=plan["infer_rows"], on_game=rows.append)
    except _Enough:
        pass
    finally:
        stop.set()
        ticker.join()
    wall = time.time() - began
    print(line(), flush=True)
    if state["warm"] is None:
        raise SystemExit(f"{label}: no act searched")
    if state["end"] is None:  # the games ran out: the measurement ends here
        trace.add(STEP, time.perf_counter(), time.perf_counter(), 0, 0, 0.0, 0.0)
        trace.active = False
        state["end"] = time.time()
    arrays = trace.arrays()
    summary = summarize(arrays)
    measured = summary["traced_s"]
    end, start = counts(), state["start"]
    delta = {name: end[name] - start[name] for name in end}
    by_slot = {slots[slot]: {"submits": int(t[0]), "rows": int(t[1]), "device_s": t[2], "forward_s": t[3],
                             "wait_s": t[4], "act_s": seconds[slot]}
               for slot, t in sorted(trace.totals.items()) if slot in slots}
    name = f"{which}{'-' + args.tag if args.tag else ''}"
    out = Path(args.out)
    np.savez_compressed(out / f"{name}.trace.npz", **saved(arrays, "i0"))
    stretch = hardware.stretch(label, windows=[(state["warm"], state["end"])])
    result = {
        "machine": machine(), "what": plan["what"], "spec": plan["spec"], "canonical": resolved[0].canonical,
        "opponent": args.opponent, "sims": plan["sims"], "pending": plan["pending"], "moves": plan["moves"],
        "suite": suite.name, "mode": suite.determinism, "kernels": inner.kernels,
        "games": len(rows), "scheduled": len(games), "in_flight": in_flight, "threads": plan["threads"],
        "infer_rows": plan["infer_rows"], "cache_mb": megabytes if search.cache is not None else 0.0,
        "wall_s": wall, "measured_s": measured, "games_per_min": len(rows) * 60 / wall,
        **delta, **{f"{key}_per_s": value / measured for key, value in delta.items() if key != "float32_rows"},
        "cache": search.cache_stats or None,
        "counters": {key: int(value) for key, value in sorted(search.counters.items())},
        "search_stats": {key: int(value) for key, value in sorted(search.search_stats.items())},
        "seconds": {"search_act": seconds[1], "opponent_act": seconds[2],
                    "harness_and_env": measured - seconds[1] - seconds[2]},
        "by_slot": by_slot, "summary": summary, "acts": acts, "peak": peak_memory(hardware),
        "hardware_stretch": stretch, "hardware": hardware.report(),
        "rss_peak_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "load_average": load_average(),
    }
    hardware.stop()
    _write(out / f"{name}.json", result)
    print_search(label, result)


def print_search(label: str, r: dict) -> None:
    s, measured = r["summary"], r["measured_s"]
    print(f"{label} {r['what']}: {r['spec']}")
    print(f"  {r['games']} of {r['scheduled']} games finished in {r['wall_s']:.0f} s "
          f"({r['games_per_min']:.2f} a minute, the warm-up included); "
          f"measured after the warm-up {measured:.1f} s: {r['decisions_per_s']:.1f} decisions/s, "
          f"{r['simulations_per_s']:,.0f} simulations/s, {r['leaf_rows_per_s']:,.0f} leaf rows/s "
          f"({r['forwarded_rows_per_s']:,.0f} forwarded in {r['forwards']:,} forwards, {r['float32_rows']:,} rows "
          f"again in float32)")
    sec = r["seconds"]
    print(f"  the search agent's acts {sec['search_act']:.1f} s ({sec['search_act'] / measured:.0%}), the opponent's "
          f"{sec['opponent_act']:.1f} s ({sec['opponent_act'] / measured:.0%}), the harness and the environment "
          f"{sec['harness_and_env']:.1f} s ({sec['harness_and_env'] / measured:.0%})")
    for slot, t in r["by_slot"].items():
        print(f"  {slot}: {t['submits']:,} batches, {t['rows']:,} rows; the forwards {t['forward_s']:.1f} s "
              f"({t['forward_s'] / measured:.0%} of the stretch), the evaluator's calls around them "
              f"{t['device_s']:.1f} s ({t['device_s'] / measured:.0%})")
    print(f"  the device busy on the forwards {s['forward_share']:.0%} of the stretch "
          f"({s['forward_us_per_row']:.2f} us a row); the calls with their transfers "
          f"{s['gpu_busy_share']:.0%}; the main thread waits "
          f"{s['wait_share']:.0%}")
    print(f"  with a device s times as fast, the forwards alone: {predictions(s['simulated_forward_s'], measured)}")
    print(f"  the whole call faster (the transfers and the call's own work too: an upper bound): "
          f"{predictions(s['simulated_s'], measured)}")
    peak = r["peak"]
    print(f"  peak device memory: {peak['torch_reserved_GiB']} GiB reserved by torch ({peak['torch_allocated_GiB']} "
          f"allocated), {peak['over_baseline_MiB']} MiB on the card over what was there before; host RSS "
          f"{r['rss_peak_GiB']:.1f} GiB", flush=True)




# ---------------------------------------------------------------- table, predict and the summary (F)


def load_results(out: Path) -> dict:
    """Every part found in an output directory, merged."""
    merged: dict = {}
    for path in sorted(out.glob("*.json")):
        if path.name not in ("bench.json", "summary.json"):
            merged[path.stem] = json.loads(path.read_text())
    return merged


def rows_of(results: dict) -> dict[str, tuple[float, str]]:
    """The table's rows of one machine: name -> (value, unit); higher is better unless the unit is s."""
    rows: dict[str, tuple[float, str]] = {}
    for key, item in (results.get("a") or {}).get("A", {}).items():
        rows[key] = (item["rows_per_s"], "rows/s")
    for name, part in results.items():
        summary = part.get("summary") or {}
        if name.startswith("ops"):
            for shape, p in (part.get("profiles") or {}).items():
                if "error" in p:
                    continue
                label = f"D forward, {shape}"
                if "replay" in p:
                    rows[f"{label}: replay rows/s"] = (p["rows"] / p["replay"]["s"], "rows/s")
                    rows[f"{label}: device ms"] = (p["replay_device"]["s"] * 1e3, "ms")
                for group, g in p["groups"].items():
                    if g["flops"] and g["s"]:
                        rows[f"{label}: {group} GFLOP/s"] = (g["flops"] / g["s"] / 1e9, "GFLOP/s")
                    elif g["s"]:
                        rows[f"{label}: {group} ms"] = (g["s"] * 1e3, "ms")
                for transfer, t in (p.get("transfers") or {}).items():
                    rows[f"{label}: {transfer} GB/s"] = (t["GB_per_s"], "GB/s")
        if name.startswith("b1") and summary:
            label = f"B1 self-play ({name}, {part['threads']} threads)"
            rows[f"{label} s/iteration"] = (summary["iteration_s"], "s")
            rows[f"{label} games/h"] = (summary["games_per_hour"], "games/h")
            rows[f"{label} leaves/s"] = (summary["leaves_per_s"], "leaves/s")
            rows[f"{label} device busy share"] = (summary["gpu_busy_share"], "share")
            if "forward_share" in summary:
                rows[f"{label} device busy share, forwards alone"] = (summary["forward_share"], "share")
            rows[f"{label} main-thread wait share"] = (summary["wait_share"], "share")
            rows[f"{label} device us/row"] = (summary["gpu_us_per_row"], "us")
            if summary.get("learn_s"):
                rows[f"{label} learning step s"] = (summary["learn_s"], "s")
        if name.startswith("b2") and summary:
            label = f"B2 graph-m PPO ({name})"
            rows[f"{label} steps/s"] = (summary["steps_per_s"], "steps/s")
            rows[f"{label} collect s"] = (summary["collect_s"], "s")
            rows[f"{label} rollout forwards s"] = (summary["rollout_forward_s"], "s")
            rows[f"{label} train s"] = (summary["train_s"], "s")
            rows[f"{label} network share"] = (summary["network_share"], "share")
        if name.startswith(("b3", "b4")) and summary:
            label = f"{name[:2].upper()} {part['what']} ({name})"
            if name.startswith("b3"):
                rows[f"{label} games/min"] = (part["games_per_min"], "games/min")
            rows[f"{label} simulations/s"] = (part["simulations_per_s"], "simulations/s")
            rows[f"{label} rows forwarded/s"] = (part["forwarded_rows_per_s"], "rows/s")
            rows[f"{label} device busy share, forwards alone"] = (summary["forward_share"], "share")
            rows[f"{label} forward us/row"] = (summary["forward_us_per_row"], "us")
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
        seconds, forwards = [], []
        for i in range(count):
            code, t0, t1, key, gpu = (data[f"i{i}_{k}"] for k in ("code", "t0", "t1", "key", "gpu"))
            seconds.append(simulate(code, t0, t1, key, gpu, s))
            if f"i{i}_fwd" in data.files:  # the forwards alone faster, the rest of a bracket as measured
                fwd = np.minimum(data[f"i{i}_fwd"], gpu)
                forwards.append(simulate(code, t0, t1, key, gpu - fwd + fwd / s, 1.0))
        measured = [float(data[f"i{i}_t1"][-1]) for i in range(count)]
        print(f"speedup {s:g}: seconds per traced stretch (an iteration's self-play) {[round(x, 1) for x in seconds]} "
              f"(measured {[round(x, 1) for x in measured]})"
              + (f"; the forwards alone faster {[round(x, 1) for x in forwards]}" if forwards else ""))


def _part(results: dict, prefix: str) -> dict | None:
    """The part of that name, or the first tagged one (``b1-t32``)."""
    return results.get(prefix) or next((part for name, part in sorted(results.items())
                                        if name.startswith(prefix + "-")), None)


def _made_of(busy: float, items: list[tuple[str, float, dict | None]]) -> dict[str, float]:
    """Shares of the device's ``busy`` seconds: each item (name, seconds, a forward's profile or
    None) split by its profile's groups, and what is left of ``busy`` (the transfers, and in an
    evaluator's blocking call the call's own work on the host)."""
    out: dict[str, float] = {}
    left = busy
    for name, seconds, profile in items:
        left -= seconds
        if profile and "error" not in profile and profile.get("sum_s"):
            for group, g in profile["groups"].items():
                if g["s"]:
                    out[group] = out.get(group, 0.0) + seconds * g["s"] / profile["sum_s"] / busy
        elif seconds:
            out[name] = out.get(name, 0.0) + seconds / busy
    if left > 1e-9:
        out["transfers and the call around the forward"] = left / busy
    return out


def _peak(part: dict) -> dict:
    peak = part.get("peak") or {}
    return {"peak_MiB": peak.get("over_baseline_MiB"), "peak_torch_GiB": peak.get("torch_reserved_GiB")}


def _traced(workload: str, part: dict, summary: dict, items: list, note: str = "") -> dict:
    """A summary row of a traced stretch (:func:`trace_summary`'s numbers)."""
    wall, forward = summary["traced_s"], summary["simulated_forward_s"]
    return {"workload": workload, "wall_s": wall, "device_share": summary["forward_share"],
            "device_share_with_transfers": summary["gpu_busy_share"], "cpu_floor_s": forward["inf"],
            "speedup_2x": wall / forward["2.0"], "speedup_inf": wall / forward["inf"],
            "speedup_2x_with_transfers": wall / summary["simulated_s"]["2.0"],
            "speedup_inf_with_transfers": wall / summary["simulated_s"]["inf"],
            **_peak(part), "made_of": _made_of(summary["gpu_busy_s"], items), "note": note}


def summary_rows(results: dict) -> list[dict]:
    """Part F: a row per workload from the parts found. Device share: the device's busy seconds
    over the wall-clock (the forwards alone; and with the transfers, the brackets). CPU floor:
    the wall-clock with the forwards free. Speedups: the wall-clock over the predicted one with
    the forwards twice as fast and free (``_with_transfers``: the whole bracket). ``made_of``:
    shares of the device's busy time."""
    rows: list[dict] = []
    profiles = (_part(results, "ops") or {}).get("profiles") or {}
    of = lambda shape: profiles.get(shape) if "error" not in (profiles.get(shape) or {"error": 1}) else None
    for shape, p in profiles.items():
        if "error" in p or "replay" not in p:
            continue
        wall, device = p["replay"]["s"], p["replay_device"]["s"]
        floor = wall * p["replay"]["launch"]  # the replay's own launch: what a free device leaves
        rows.append({"workload": f"one forward, {shape} (a graph replay)", "wall_s": wall,
                     "device_share": min(1.0, device / wall), "device_share_with_transfers": None,
                     "cpu_floor_s": floor, "speedup_2x": wall / max(floor, device / 2.0),
                     "speedup_inf": wall / floor if floor else None, "speedup_2x_with_transfers": None,
                     "speedup_inf_with_transfers": None, **_peak(p),
                     "made_of": _made_of(p["sum_s"], [("forward", p["sum_s"], p)]),
                     "note": "queued replays: the wall-clock is the larger of the launch and the device's time"})
    b1 = _part(results, "b1")
    if b1 and b1.get("summary"):
        s = b1["summary"]
        profile = of(f"{b1['train'].get('kernels')} x {b1['runtime'].get('infer_rows')} rows")
        rows.append(_traced("AlphaZero self-play, an iteration's", b1, s, [("forwards", s["forward_s"], profile)],
                            f"{b1.get('run', '')}, {b1['threads']} threads; iterations {s['iterations']}"))
        learn, p = s.get("learn") or {}, s.get("predicted") or {}
        if s.get("learn_s") and "error" not in learn:
            sim = learn["simulated_s"]
            busy = learn["kernel_s"] + learn["transfer_s"]
            rows.append({"workload": "AlphaZero learning step", "wall_s": s["learn_s"],
                         "device_share": learn["device_share"],
                         "device_share_with_transfers": min(1.0, busy / s["learn_s"]),
                         "cpu_floor_s": sim["inf"], "speedup_2x": s["learn_s"] / sim["2.0"],
                         "speedup_inf": s["learn_s"] / sim["inf"] if sim["inf"] else None,
                         "speedup_2x_with_transfers": None, "speedup_inf_with_transfers": None, **_peak(b1),
                         "made_of": {g: seconds / sum(learn["groups"].values())
                                     for g, seconds in learn["groups"].items() if seconds},
                         "note": "kernels by torch.profiler; the speedups are upper bounds (launches overlap kernels)"})
        elif s.get("learn_s"):
            rows.append({"workload": "AlphaZero learning step", "wall_s": s["learn_s"], "device_share": None,
                         "device_share_with_transfers": None, "cpu_floor_s": None, "speedup_2x": None,
                         "speedup_inf": None, "speedup_2x_with_transfers": None, "speedup_inf_with_transfers": None,
                         **_peak(b1), "made_of": {}, "note": f"no device share: {learn.get('error')}"})
        if p:
            wall = p["iteration_s"]
            kernels = learn.get("kernel_s", 0.0)
            rows.append({"workload": "AlphaZero iteration (self-play, learning step, the rest)", "wall_s": wall,
                         "device_share": (s["forward_s"] + kernels) / wall,
                         "device_share_with_transfers":
                             (s["gpu_busy_s"] + kernels + learn.get("transfer_s", 0.0)) / wall,
                         "cpu_floor_s": p["forwards"]["inf"], "speedup_2x": wall / p["forwards"]["2.0"],
                         "speedup_inf": wall / p["forwards"]["inf"],
                         "speedup_2x_with_transfers": wall / p["bracket"]["2.0"],
                         "speedup_inf_with_transfers": wall / p["bracket"]["inf"], **_peak(b1),
                         "made_of": _made_of(s["gpu_busy_s"] + kernels,
                                             [("self-play's forwards", s["forward_s"], profile),
                                              ("the learning step's kernels", kernels, None)]),
                         "note": "" if s.get("learned_iterations") else "NO LEARNING STEP RAN: self-play and the rest"})
    for prefix, workload in (("b3", "eval row's search agent"), ("b4", "deployed search")):
        part = _part(results, prefix)
        if not part or not part.get("summary"):
            continue
        workload += (f" ({part['in_flight']} games in flight, {part['sims']:,} simulations, {part['kernels']}, "
                     f"{part['infer_rows']}-row forwards)")
        slots = part.get("by_slot") or {}
        items = [("the search's forwards", (slots.get("search") or {}).get("forward_s", 0.0),
                  of(f"{part['kernels']} x {part['infer_rows']} rows")),
                 ("the opponent's forwards (PyTorch path)", (slots.get("opponent") or {}).get("forward_s", 0.0), None)]
        row = _traced(workload, part, part["summary"], items,
                      f"{part['simulations_per_s']:,.0f} simulations/s"
                      + (f", {part['games_per_min']:.1f} games/min" if prefix == "b3" else ""))
        row["rss_peak_GiB"] = part.get("rss_peak_GiB")
        rows.append(row)
    for precision in ("fp32", "fp16"):
        part = _part(results, f"b2-{precision}")
        if not part or not part.get("summary"):
            continue
        s = part["summary"]
        share, wall = s["network_share"], s["wall_s"]
        row = {"workload": f"graph-m PPO epoch, {part['precision']}", "wall_s": wall, "device_share": share,
               "device_share_with_transfers": None, "cpu_floor_s": wall * (1.0 - share),
               "speedup_2x": 1.0 / (1.0 - share / 2.0), "speedup_inf": 1.0 / (1.0 - share) if share < 1.0 else None,
               "speedup_2x_with_transfers": None, "speedup_inf_with_transfers": None, **_peak(part),
               "made_of": {"the rollout's forwards (synchronised)": s["rollout_forward_s"] / (share * wall),
                           "the update (forward and backward)": s["train_s"] / (share * wall)},
               "note": "network seconds with their launches: the share and the speedups are upper bounds"}
        if row["peak_torch_GiB"] is None:
            row["peak_torch_GiB"] = part.get("peak_GiB")
        rows.append(row)
    return rows


def _seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 1e3:.2f} ms" if value < 0.1 else f"{value:.1f} s"


def print_summary(rows: list[dict], card: dict | None) -> None:
    share = lambda v: "-" if v is None else f"{v:.0%}"
    times = lambda v: "-" if v is None else f"{v:.2f}x"
    if card:
        link = card.get("link") or {}
        print(f"{card.get('gpu')}: {card.get('memory_total_MiB') or 0:,.0f} MiB, "
              f"{card.get('baseline_MiB') or 0:,.0f} MiB in use before a part; "
              f"PCIe gen {link.get('gen') or 0:g} x{link.get('width') or 0:g} at a part's start "
              f"(of gen {link.get('gen_max') or 0:g} x{link.get('width_max') or 0:g})")
    header = ("workload", "wall", "device", "+transfers", "CPU floor", "2x device", "free device", "peak VRAM")
    table = []
    for row in rows:
        peak = (f"{row['peak_MiB']:,.0f} MiB" if row.get("peak_MiB") is not None
                else f"{row['peak_torch_GiB']:.2f} GiB (torch)" if row.get("peak_torch_GiB") is not None else "-")
        table.append((row["workload"], _seconds(row["wall_s"]), share(row["device_share"]),
                      share(row["device_share_with_transfers"]), _seconds(row["cpu_floor_s"]), times(row["speedup_2x"]),
                      times(row["speedup_inf"]), peak))
    widths = [max(len(line[i]) for line in (header, *table)) for i in range(len(header))]
    for line in (header, *table):
        print("  ".join(cell.ljust(width) if i == 0 else cell.rjust(width)
                        for i, (cell, width) in enumerate(zip(line, widths))))
    print("the device's busy time is made of:")
    for row in rows:
        parts = ", ".join(f"{name} {value:.0%}" for name, value in row["made_of"].items())
        print(f"  {row['workload']}: {parts or '-'}" + (f" [{row['note']}]" if row["note"] else ""))
    print("device: its busy seconds over the wall-clock (the forwards alone; +transfers: the whole bracket or call). "
          "CPU floor, 2x and free device: the forwards alone faster, the transfers as measured. peak VRAM: "
          "nvidia-smi's peak over what was in use before the part (the CUDA context included).", flush=True)


def cmd_summary(out: Path, failed: list[str] | None = None) -> None:
    """Part F from the parts in ``out``: summary.json and the table. ``failed``: the parts of this
    run that did not finish, named under the table with the forward shapes that could not be
    profiled."""
    results = load_results(out)
    rows = summary_rows(results)
    missing = list(failed or ())
    missing += [f"D one forward, {shape}: {p['error']}"
                for shape, p in ((_part(results, "ops") or {}).get("profiles") or {}).items() if "error" in p]
    if not rows:
        print("summary: no part with a device split here yet (ops, b1, b2, eval, search)", flush=True)
        for line in missing:
            print(f"NOT MEASURED: {line}", flush=True)
        return
    cards = [part.get("hardware") for part in results.values() if isinstance(part.get("hardware"), dict)]
    cards += [p["hardware"] for p in ((_part(results, "ops") or {}).get("profiles") or {}).values()
              if isinstance(p.get("hardware"), dict)]
    card = {k: cards[0].get(k) for k in ("gpu", "memory_total_MiB", "baseline_MiB", "link")} if cards else None
    machines = [part["machine"] for part in results.values() if "machine" in part]
    _write(out / "summary.json", {"machine": machines[0] if machines else None, "card": card, "rows": rows,
                                  "not_measured": missing})
    print_summary(rows, card)
    for line in missing:
        print(f"NOT MEASURED: {line}", flush=True)


# ---------------------------------------------------------------- main


def _child(args: argparse.Namespace, command: str, extra: list[str]) -> str | None:
    """A part in a process of its own: None, or what went wrong (the caller goes on to the next)."""
    argv = [sys.executable, "-P", __file__, command, "--child", *args.passthrough, *extra]
    try:
        code = subprocess.run(argv, env=os.environ.copy(), check=False, timeout=args.timeout or None).returncode
    except subprocess.TimeoutExpired:
        return f"still running after {args.timeout:.0f} s (--timeout), stopped"
    if code == LOW_MEMORY:
        return "stopped by its memory guard (--min-free-gb)"
    return None if code == 0 else f"exit code {code}"


def _stale(args: argparse.Namespace, command: str, extra: list[str], started: float) -> None:
    """A failed part's JSON from an earlier run in the same directory is set aside (``.stale``),
    so the tables do not show it as this run's. ``ops.json`` holds the failure itself."""
    tag = f"-{args.tag}" if args.tag else ""
    stem = {"a": "a", "b1": f"b1{tag}", "eval": f"b3{tag}", "search": f"b4{tag}",
            "b2": f"b2-{'fp16' if 'float16' in extra else 'fp32'}{tag}"}.get(command)
    for path in (Path(args.out) / f"{stem}.json", Path(args.out) / f"{stem}.trace.npz") if stem else ():
        if path.exists() and path.stat().st_mtime < started:
            path.replace(path.with_name(path.name + ".stale"))
            print(f"   {path.name} is an earlier run's: set aside as {path.name}.stale", flush=True)


def _steps(args: argparse.Namespace) -> list[tuple[str, str, list[str]]]:
    """The command's parts in order: (label, the child's command, its extra arguments)."""
    steps: list[tuple[str, str, list[str]]] = []
    wanted = lambda name: args.command in (name, "all")
    if wanted("a"):
        steps.append(("A microbenchmarks", "a", []))
    if wanted("ops"):
        shapes = ([tuple(args.shape.split(":"))] if args.shape else OPS_TINY if args.tiny else OPS_SHAPES)
        steps += [(f"D one forward, {kernels} x {rows} rows", "ops", ["--shape", f"{kernels}:{rows}"])
                  for kernels, rows in shapes]
    if wanted("b1"):
        steps.append((f"B1 self-play, {args.b1_run}", "b1", []))
    if wanted("b2"):
        for precision in ([args.precision] if args.precision else ["float32", "float16"]):
            if args.tiny and precision == "float16":
                continue  # float16 autocast is the CUDA path
            steps.append((f"B2 graph-m PPO, {precision}", "b2", ["--precision", precision]))
    if wanted("eval"):
        steps.append(("B3 the eval row's search agent", "eval", []))
    if wanted("search"):
        steps.append(("B4 one deployed-style search", "search", []))
    return steps


def main(argv: list[str] | None = None) -> None:
    global QUICK
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="the directory of the JSON parts (and bench.json, summary.json)")
    common.add_argument("--runs", default=None, help="a scratch runs directory (created; never the real one; "
                        "default: <out>/runs)")
    common.add_argument("--real-runs", default=None, help="where the checkpoints' runs are copied from (default: the "
                        "main worktree's runs/)")
    common.add_argument("--quick", action="store_true", help="a few seconds a part: does everything run on this "
                        "card? (the real networks and kernels, short loops; the numbers are rough)")
    common.add_argument("--net", default=NET, help="D, B3, B4: the network, a model of configs/registry.yaml or a "
                        f"checkpoint ref (default: {NET})")
    common.add_argument("--shape", default=None, metavar="KERNELS:ROWS", help="D: one forward shape only "
                        "(fused16:512; default: fused16:512, fused16:64, fused:64)")
    common.add_argument("--no-profiler", action="store_true", help="D, B1: no torch.profiler (the cross-check of D; "
                        "the device share of B1's learning step)")
    common.add_argument("--b1-run", default=B1_RUN,
                        help=f"B1: the run whose stored config is played (default: {B1_RUN})")
    common.add_argument("--b1-start", default=None, help="B1: the checkpoint it starts from (default: that config's "
                        "init_from)")
    common.add_argument("--iterations", type=int, default=None, help="B1: self-play iterations (default: 7)")
    common.add_argument("--min-replay", type=int, default=1024, help="B1: train.min_replay (the run's: 50,000)")
    common.add_argument("--epochs", type=int, default=None, help="B2: epochs per precision (default: 10)")
    common.add_argument("--warmup", type=int, default=3, help="B2: first epochs left out of the medians")
    common.add_argument("--precision", default=None, choices=("float32", "float16"), help="B2: one precision only")
    common.add_argument("--opponent", default=OPPONENT, help="B3, B4: the other seat's agent spec")
    common.add_argument("--suite", default=SUITE, help=f"B3, B4: the suite the games are of (default: {SUITE})")
    common.add_argument("--games", type=int, default=64, help="B3: games")
    common.add_argument("--envs", type=int, default=64, help="B3: games in flight")
    common.add_argument("--eval-sims", type=int, default=1024, help="B3: simulations a decision")
    common.add_argument("--search-sims", type=int, default=32768, help="B4: simulations a move")
    common.add_argument("--moves", type=int, default=6, help="B4: measured moves after the warm-up one")
    common.add_argument("--min-free-gb", type=float, default=3.0, help="B3, B4: host memory left free (a search "
                        "that would go under it does not start, or stops)")
    common.add_argument("--every", type=float, default=15.0, help="B3: seconds between progress lines")
    common.add_argument("--threads", type=int, default=0, help="runtime.threads (default: the configs' 12)")
    common.add_argument("--timeout", type=float, default=3600.0, help="seconds a part's process may run (0: no limit)")
    common.add_argument("--no-hardware", action="store_true", help="no nvidia-smi sampler beside the stretches")
    common.add_argument("--tag", default="", help="a suffix for the run and the part's file name")
    common.add_argument("--tiny", action="store_true", help="a CPU check of the tool: tiny shapes and networks")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="B1, B2: a config override")
    common.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    for name in ("all", "a", "ops", "b1", "b2", "eval", "search"):
        sub.add_parser(name, parents=[common])
    summary = sub.add_parser("summary")
    summary.add_argument("--out", required=True)
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
    if args.command == "summary":
        cmd_summary(Path(args.out).resolve())
        return

    QUICK = args.quick and not args.tiny
    if args.iterations is None:
        args.iterations = 3 if args.tiny else 2 if QUICK else 7
    if args.epochs is None:
        args.epochs = 2 if QUICK else 10
    os.environ["WANDB_MODE"] = "disabled"
    for key in ("WANDB_API_KEY", "CATANRL_WANDB_PROJECT", "CATANRL_WANDB_ENTITY"):
        os.environ.pop(key, None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    args.runs = args.runs or str(out / "runs")
    args.passthrough = [a for a in argv[1:] if a != "--child"]

    if args.child:
        if args.command == "a":
            _write(out / "a.json", bench_a(args))
        elif args.command == "ops":
            child_ops(args)
        elif args.command == "b1":
            child_b1(args)
        elif args.command == "b2":
            child_b2(args)
        else:
            child_search(args, "b3" if args.command == "eval" else "b4")
        return
    if args.command != "a":
        _scratch(args)
    steps = _steps(args)
    if args.command in ("ops", "all") and not args.shape:
        ops_path(args).unlink(missing_ok=True)  # the shapes merge into it: not with an earlier run's
    failed: list[str] = []
    began = time.time()
    for i, (label, command, extra) in enumerate(steps, 1):
        print(f"== part {i}/{len(steps)}: {label} ({(time.time() - began) / 60:.1f} min so far)", flush=True)
        started = time.time()
        error = _child(args, command, extra)
        if error:  # the part's failure is its result: the others run
            failed.append(f"{label}: {error}")
            print(f"== part {i}/{len(steps)} FAILED: {label}: {error}; going on", flush=True)
            _stale(args, command, extra, started)
        else:
            print(f"== part {i}/{len(steps)} done in {time.time() - started:.0f} s", flush=True)
    results = load_results(out)
    _write(out / "bench.json", results)
    print_table({"this": results})
    print()
    cmd_summary(out, failed)
    if failed:
        raise SystemExit("FAILED: " + "; ".join(failed))


if __name__ == "__main__":
    main()
