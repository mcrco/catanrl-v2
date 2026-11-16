"""tools/gpu_bench.py on rented GPUs (Modal): the same script, the same lock file, one job per call.

The image is NVIDIA's CUDA 13.0 devel image (nvcc for the fused kernels and PufferLib's extension),
Python 3.11 and ``uv sync --extra cu130 --frozen``: the torch build of this box (uv.lock). Modal's
hosts run a driver for CUDA 13 (610.57.04 / 13.3 on 2026-11-15; every result records what it saw).

What is shipped: a shallow clone of the committed checkout (HEAD of the worktree this file is in;
never the working tree, the venv or the runs directory), the three submodules cloned in the image
from their public forks at the pinned commits, and e1800's checkpoint with its run's config.json and
run.json (the stretch's start). No key goes up: W&B is disabled in the container.

Nothing is deployed: every job is one ``modal run`` (an ephemeral app that stops with the call),
every function has a timeout of at most 20 minutes, and no volume is made (the results come back as
the call's return value). Each call appends its seconds to ``<out>/tally.jsonl``.

    modal run tools/modal_bench.py --job cpu          # no GPU: the image and the tool (--tiny)
    modal run tools/modal_bench.py --job a10g-a       # A on an A10G
    modal run tools/modal_bench.py --job l40s-a       # A on an L40S (8 cores)
    modal run tools/modal_bench.py --job l40s-b1      # B1 + C, 8 cores, the config's 12 threads
    modal run tools/modal_bench.py --job l40s-b1-t8   # B1 + C, 8 cores, 8 threads
    modal run tools/modal_bench.py --job l40s-b1-t32  # B1 + C, 32 cores, 32 threads
    modal run tools/modal_bench.py --job l40s-b2      # B2 + C, float32 and float16
    modal app list                                    # nothing may be left running
    .venv/bin/python tools/gpu_bench.py table local=<local out> l40s=<out>/l40s a10g=<out>/a10g

``--out`` (default /tmp/modal-bench) takes one directory per machine: l40s, l40s-32c, a10g, cpu.
``modal`` is not in pyproject: ``uv pip install modal`` in the venv, or the CLI's own install.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE = "/root/catanrl-v2"
REMOTE_RUNS = "/root/real-runs"
START = "az-2p-kata/main+1500"
START_FILES = ("config.json", "run.json", "checkpoints/e001800.pt")
ITERATIONS = 6   # B1: the same on every machine (the first three are openings)
EPOCHS = 10      # B2: per precision
# $ per second (modal.com/pricing, 2026-11-15): the tally's estimate
PRICE = {"L40S": 0.000542, "A10G": 0.000306, None: 0.0, "core": 0.0000131, "GiB": 0.00000222}

app = modal.App("catanrl-gpu-bench")


def _staging() -> Path:
    """A shallow clone of HEAD and the start's three files, made on the caller's machine."""
    import shutil
    import tempfile

    stage = Path(os.environ.get("MODAL_BENCH_STAGE") or tempfile.mkdtemp(prefix="modal-bench-stage-"))
    clone, seed = stage / "src", stage / "runs"
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    marker = stage / "HEAD"
    if not marker.exists() or marker.read_text() != head:
        shutil.rmtree(clone, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--no-local", f"file://{REPO}", str(clone)],
                       check=True, env={**os.environ, "GIT_CONFIG_COUNT": "0"})
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", "--detach", head], check=True)
        subprocess.run(["git", "-C", str(clone), "remote", "remove", "origin"], check=True)
        marker.write_text(head)
    if not (seed / START / START_FILES[-1]).exists():
        from_dir = Path(os.environ.get("CATANRL_REAL_RUNS") or _main_runs()) / START
        for name in START_FILES:
            target = seed / START / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(from_dir / name, target)
    return stage


def _main_runs() -> Path:
    """The main worktree's runs/ (a linked worktree has none of its own)."""
    common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            check=True, capture_output=True, text=True).stdout.strip()
    return Path(common).parent / "runs"


image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "build-essential", "gcc-14", "g++-14", "ninja-build", "cmake")
    .pip_install("uv")
    .env({"CC": "gcc-14", "CXX": "g++-14"})  # the added Python's sysconfig names clang, which the image lacks
)
if modal.is_local():
    _stage = _staging()
    image = (
        image.add_local_dir(_stage / "src", REMOTE, copy=True)
        .run_commands(
            "git config --global --add safe.directory '*'",
            f"cd {REMOTE} && git submodule update --init --depth 1",
            f"cd {REMOTE} && git submodule status",
            # this box builds the engine with gcc 16; Ubuntu 24.04's gcc 13 and 14 report a stringop-overflow
            # in it that newer ones do not, and the engine builds with -Werror
            # and the builder has no GPU for torch to read PufferLib's CUDA architectures from
            # (8.6: A10G, 8.9: L40S)
            f"cd {REMOTE} && CXXFLAGS=-Wno-stringop-overflow TORCH_CUDA_ARCH_LIST='8.6;8.9' UV_LINK_MODE=copy uv sync --extra cu130 --frozen --no-dev --python 3.11",
        )
        .add_local_dir(_stage / "runs", REMOTE_RUNS, copy=True)
    )


def _bench(runs: list[list[str]]) -> dict:
    """Run gpu_bench.py once per argument list in the image's checkout; return the files it wrote."""
    out, started, codes = Path("/root/out"), time.time(), []
    env = {**os.environ, "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    env.pop("WANDB_API_KEY", None)
    subprocess.run("nvidia-smi; nproc; cat /sys/fs/cgroup/cpu.max; grep -m1 'model name' /proc/cpuinfo",
                   shell=True, check=False)
    for arguments in runs:
        command = [f"{REMOTE}/.venv/bin/python", "tools/gpu_bench.py", *arguments, "--out", str(out),
                   "--runs", "/root/scratch-runs", "--real-runs", REMOTE_RUNS]
        print("$", " ".join(command), flush=True)
        codes.append(subprocess.run(command, cwd=REMOTE, env=env, check=False).returncode)
    files = {p.name: p.read_bytes() for p in sorted(out.glob("*")) if p.is_file()} if out.exists() else {}
    return {"returncodes": codes, "seconds": time.time() - started, "files": files}


@app.function(image=image, cpu=(4.0, 4.0), memory=16384, timeout=900)
def on_cpu(runs: list[list[str]]) -> dict:
    return _bench(runs)


@app.function(image=image, gpu="A10G", cpu=(8.0, 8.0), memory=32768, timeout=600)
def on_a10g(runs: list[list[str]]) -> dict:
    return _bench(runs)


@app.function(image=image, gpu="L40S", cpu=(8.0, 8.0), memory=32768, timeout=1200)
def on_l40s(runs: list[list[str]]) -> dict:
    return _bench(runs)


@app.function(image=image, gpu="L40S", cpu=(32.0, 32.0), memory=32768, timeout=1200)
def on_l40s_32(runs: list[list[str]]) -> dict:
    return _bench(runs)


B1 = ["b1", "--iterations", str(ITERATIONS)]
# job -> (function, GPU, cores, GiB, the machine's directory, gpu_bench.py argument lists)
JOBS = {
    "cpu": (on_cpu, None, 4, 16, "cpu", [["all", "--tiny", "--iterations", "3", "--epochs", "4", "--warmup", "1"]]),
    "a10g-a": (on_a10g, "A10G", 8, 32, "a10g", [["a"]]),
    "l40s-a": (on_l40s, "L40S", 8, 32, "l40s", [["a"]]),
    "l40s-b1": (on_l40s, "L40S", 8, 32, "l40s", [B1]),
    "l40s-b1-t8": (on_l40s, "L40S", 8, 32, "l40s", [[*B1, "--threads", "8", "--tag", "t8"]]),
    "l40s-b1-t32": (on_l40s_32, "L40S", 32, 32, "l40s-32c", [[*B1, "--threads", "32", "--tag", "t32"]]),
    "l40s-b2": (on_l40s, "L40S", 8, 32, "l40s", [["b2", "--epochs", str(EPOCHS)]]),
}


@app.local_entrypoint()
def main(job: str, out: str = "/tmp/modal-bench"):
    function, gpu, cores, gib, machine, runs = JOBS[job]
    started = time.time()
    result = function.remote(runs)
    call_s = time.time() - started
    target = Path(out) / machine
    target.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        (target / name).write_bytes(data)
    rate = PRICE[gpu] + cores * PRICE["core"] + gib * PRICE["GiB"]
    # the container is billed from its start to its end: between the function's own seconds and the call's
    row = {"job": job, "gpu": gpu, "cores": cores, "GiB": gib, "returncodes": result["returncodes"],
           "function_s": round(result["seconds"], 1), "call_s": round(call_s, 1),
           "usd_per_h": round(rate * 3600, 3), "usd_function": round(rate * result["seconds"], 3),
           "usd_call": round(rate * call_s, 3), "files": sorted(result["files"]),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(Path(out) / "tally.jsonl", "a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))
    if any(result["returncodes"]):
        raise SystemExit(f"{job}: gpu_bench.py failed {result['returncodes']}")
