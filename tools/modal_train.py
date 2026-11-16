"""Training runs on rented GPUs (Modal): launch, watch, stop, pull back and mirror to W&B.

    P=.venv/bin/python
    $P tools/modal_train.py launch configs/experiments/ppo-mixed.yaml --gpu L40S --cpu 8 --memory 32 \\
        --max-hours 12 --set runtime.threads=8            # detached; prints $/h and the worst case
    $P tools/modal_train.py launch configs/experiments/az-2p-bundle.yaml --rev wt/player-set \\
        --gpu A10G --max-hours 30 --ship az-2p-kata/main+1500@e1800   # another branch; a start to upload
    $P tools/modal_train.py status [ppo-mixed/main]       # epoch, rate, ETA, cost so far, app alive?
    $P tools/modal_train.py list                          # every run the Volume knows
    $P tools/modal_train.py stop ppo-mixed/main           # stops the app now (the last commit stays)
    $P tools/modal_train.py stop ppo-mixed/main --graceful  # the trainer ends its epoch, commits, exits
    $P tools/modal_train.py resume ppo-mixed/main --max-hours 12   # same revision, config and shape
    $P tools/modal_train.py pull ppo-mixed/main           # Volume -> local runs dir, then W&B backfill
    $P tools/modal_train.py watch ppo-mixed/main --every 10        # pull + mirror in a loop
    $P tools/modal_train.py ship ppo-mixed/main@e3550     # only upload a ref's files
    modal app logs <app id>                               # the trainer's own progress lines
    modal app list                                        # nothing of ours may be left running

What goes up. The image is modal_bench.py's (NVIDIA CUDA 13.0 devel, Python 3.11, gcc 14,
``uv sync --extra cu130 --frozen``) with one layer more: the lock's third-party wheels are installed
from pyproject.toml and uv.lock alone, so two commits with the same lock share that layer. The code
is a shallow clone of exactly ``--rev`` (any commit of this repository's object store, default HEAD
of the worktree this file is in; never a working tree, a venv or the runs directory), kept under
``~/.cache/catanrl-modal/stage/<sha>`` ($CATANRL_MODAL_CACHE) so the same commit gives the same
image. The three submodules are cloned in the image from their public forks at the pinned commits
(a pin that is not pushed fails the build). No key goes up: no Secret is attached, W&B is disabled
in the container and WANDB_API_KEY is dropped from the trainer's environment.

How a run lives there. One detached app per run (``catanrl-train-<exp>-<variant>``), one function
call at a time. The trainer is ``catanrl run CONFIG --variant V --set runtime.time_limit_hours=..``
on the container's own disk (CATANRL_RUNS_DIR=/root/runs: the run store's lock, renames and appends
are not for a network mount). A supervisor copies the run directory to the Volume ``catanrl-runs``
(``runs/<exp>/<variant>``) and commits it whenever the trainer has a new commit point (a
checkpoint with its trainer state) and every ``--sync-minutes``; trainer_state.pt and run.json go
last, from one consistent read, so the Volume always holds a state the trainer itself could have
crashed in. A container that starts and finds the run on the Volume copies it back and
``catanrl run`` resumes it: that is the path after a preemption, a stopped container, ``resume``
and the 24-hour cap.

``--max-hours`` is a budget of supervisor seconds for one launch, kept in the Volume
(``_modal/<exp>/<variant>.state.json``). The trainer gets the budget less a grace as its own time
limit and stops at an epoch boundary with a checkpoint; the supervisor sends it SIGTERM at the
budget, and Modal's function timeout (budget + 150 s, at most 24 h) is the last line. Modal caps a
call at 24 h: a run with more budget ends its segment on the trainer's time limit and spawns the
next call in the same app (``--segment-hours`` shortens the segment), as long as the last segment
moved the run forward. ``resume`` starts a new app with a new budget.

Money. Prices are list prices per second (modal_bench.py's table for L40S, A10G, cores and memory;
the other GPUs from ``modal billing rates`` on 2026-11-16). ``launch`` prints the shape's $/h and
the worst case of the budget, ``status`` the supervisor's seconds times that price. Image builds,
cold starts and the Volume's storage ($0.09 per GiB-month) are not in these numbers.

Names. The run is the config's own ``<experiment>/<variant>``; a run of the same name in the local
runs directory would collide, so ``pull`` refuses a local run directory it did not make (``--merge``
applies the per-file rule anyway). ``pull`` never replaces a local file whose content it did not
write: new files are fetched, an append-only file the Volume has extended is extended, a file this
tool pulled earlier and nobody touched is updated, and anything else is a conflict that keeps the
local file and is listed (``--force`` overwrites and says so). While the run is live its pulled
run.json says ``running`` with no lock, which ``catanrl ls`` shows as ``dead``. The container's
engine and PufferLib binaries are its own builds (gcc 14), so its code id differs from this box's
in those two components: a local resume of a pulled run needs ``--allow-code-change``.

``modal`` is not in pyproject: ``uv pip install modal`` in the venv, or the CLI's own install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
APP_PREFIX = "catanrl-train"
VOLUME = "catanrl-runs"
REMOTE = "/root/catanrl-v2"   # the checkout in the image
VOL = "/vol"                  # the Volume in the container
LOCAL_RUNS = "/root/runs"     # the trainer's runs directory (container disk)
EXTRA = "cu130"
MARGIN_S = 150                # function timeout past the budget: start-up and the last commit
DAY_S = 24 * 3600             # Modal's cap on one call
# $ per second. L40S, A10G, core, GiB: modal_bench.py (modal.com/pricing, 2026-11-15); the other
# GPUs: `modal billing rates`, 2026-11-16 ($/h / 3600).
PRICE = {"L40S": 0.000542, "A10G": 0.000306, None: 0.0, "core": 0.0000131, "GiB": 0.00000222,
         "T4": 0.59 / 3600, "L4": 0.80 / 3600, "A100": 2.10 / 3600, "A100-40GB": 2.10 / 3600,
         "A100-80GB": 2.50 / 3600, "H100": 3.95 / 3600, "H200": 4.54 / 3600, "B200": 6.25 / 3600}
RUN_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]*/[a-z0-9][a-z0-9._+-]*$")
REF_RE = re.compile(r"^(?P<run>[a-z0-9][a-z0-9._+-]*/[a-z0-9][a-z0-9._+-]*)(?:@(?P<which>[^#]+))?(?:#(?P<sha>[0-9a-f]+))?$")

volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
app: modal.App | None = None


class ToolError(Exception):
    pass


# ---------------------------------------------------------------------------- money

def usd_per_s(gpu: str | None, cores: float, gib: float, gpu_usd_per_h: float | None = None) -> float:
    """List price of one second of a shape (the tally's arithmetic, as modal_bench.py)."""
    if gpu_usd_per_h is not None:
        gpu_rate = gpu_usd_per_h / 3600
    elif gpu in PRICE:
        gpu_rate = PRICE[gpu]
    else:
        raise ToolError(f"no list price for GPU {gpu!r}: pass --gpu-usd-per-hour (known: "
                        f"{', '.join(sorted(k for k in PRICE if k and k not in ('core', 'GiB')))})")
    return gpu_rate + cores * PRICE["core"] + gib * PRICE["GiB"]


def cost_line(gpu: str | None, cores: float, gib: float, max_hours: float, rate: float) -> str:
    return (f"{gpu or 'no GPU'}, {cores:g} cores, {gib:g} GiB: ${rate * 3600:.3f}/h at list prices; "
            f"worst case of --max-hours {max_hours:g}: ${rate * 3600 * max_hours:.2f}")


# ---------------------------------------------------------------------------- the revision

def _git(*args: str, cwd: Path | str = REPO, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def resolve_rev(rev: str, repo: Path = REPO) -> str:
    try:
        return _git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", cwd=repo)
    except subprocess.CalledProcessError:
        raise ToolError(f"{rev!r} is not a commit of {repo}") from None


def config_at(sha: str, config: str, repo: Path = REPO) -> tuple[str, str, bool]:
    """(repo-relative path, experiment name, has a sweep) of a config as committed at ``sha``."""
    path = Path(config)
    if path.is_absolute():
        top = Path(_git("rev-parse", "--show-toplevel", cwd=path.parent))
        path = path.relative_to(top)
    rel = path.as_posix()
    try:
        text = _git("show", f"{sha}:{rel}", cwd=repo)
    except subprocess.CalledProcessError:
        raise ToolError(f"{rel} is not in {sha[:10]} (the config must be committed in the revision)") from None
    import yaml
    raw = yaml.safe_load(text) or {}
    return rel, str(raw.get("name", path.stem)), bool(raw.get("sweep")) or bool(raw.get("stages"))


def submodule_pins(sha: str, repo: Path = REPO) -> dict[str, str]:
    out = {}
    for line in _git("ls-tree", "-r", sha, cwd=repo).splitlines():
        meta, _, path = line.partition("\t")
        mode, _, pin = meta.split()
        if mode == "160000":
            out[path] = pin
    return out


def unpushed_pins(sha: str, repo: Path = REPO) -> list[str]:
    """Pins that are neither a branch tip of their fork nor on a remote-tracking branch of a local checkout.

    The image clones the submodules from the URLs in the revision's .gitmodules, so such a pin most
    likely fails the build. A warning only: the local remote-tracking branches may be stale.
    """
    pins = submodule_pins(sha, repo)
    if not pins:
        return []
    urls = {}
    for line in _git("config", "--blob", f"{sha}:.gitmodules", "--get-regexp", r"submodule\..*\.(path|url)",
                     cwd=repo).splitlines():
        key, value = line.split(" ", 1)
        urls.setdefault(key.rsplit(".", 1)[0], {})[key.rsplit(".", 1)[1]] = value
    url_of = {v["path"]: v.get("url", "") for v in urls.values() if "path" in v}
    trees = [Path(line.split(" ", 1)[1]) for line in _git("worktree", "list", "--porcelain", cwd=repo).splitlines()
             if line.startswith("worktree ")]
    notes = []
    for path, pin in pins.items():
        tips = subprocess.run(["git", "ls-remote", "--heads", url_of.get(path, "")], capture_output=True, text=True,
                              check=False, timeout=60, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}).stdout
        if pin in tips:
            continue
        for tree in trees:
            found = subprocess.run(["git", "-C", str(tree / path), "branch", "-r", "--contains", pin],
                                   capture_output=True, text=True, check=False).stdout.strip()
            if found:
                break
        else:
            notes.append(f"{path}@{pin[:10]}")
    return notes


def stage_dir() -> Path:
    return Path(os.environ.get("CATANRL_MODAL_CACHE") or Path.home() / ".cache" / "catanrl-modal") / "stage"


def stage_revision(sha: str, cache: Path | None = None, repo: Path = REPO) -> Path:
    """``<cache>/<sha>/{src,deps}``: a depth-1 clone of exactly ``sha`` and its two lock files."""
    cache = cache or stage_dir()
    dst = cache / sha
    if (dst / "ok").exists():
        return dst
    tmp = cache / f".{sha}.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    src = tmp / "src"
    src.mkdir(parents=True)
    env = {**os.environ, "GIT_CONFIG_COUNT": "0"}
    common = Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=repo))
    _git("init", "--quiet", cwd=src, env=env)
    _git("-c", "protocol.file.allow=always", "fetch", "--quiet", "--depth", "1", f"file://{common}", sha,
         cwd=src, env=env)
    _git("-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", sha, cwd=src, env=env)
    # Modal does not upload empty directories, and git takes a .git without refs/ for no repository
    _git("update-ref", "refs/heads/staged", sha, cwd=src, env=env)
    if _git("rev-parse", "HEAD", cwd=src) != sha or _git("status", "--porcelain", cwd=src):
        raise ToolError(f"staging {sha[:10]} did not give a clean checkout of it")
    (tmp / "deps").mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(src / name, tmp / "deps" / name)
    (tmp / "ok").write_text(sha + "\n")
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return dst


def build_image(stage: Path) -> modal.Image:
    sync = (f"cd {REMOTE} && CXXFLAGS=-Wno-stringop-overflow TORCH_CUDA_ARCH_LIST='8.6;8.9' UV_LINK_MODE=copy "
            f"uv sync --extra {EXTRA} --frozen --no-dev --python 3.11")
    return (
        modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.11")
        .entrypoint([])
        .apt_install("git", "build-essential", "gcc-14", "g++-14", "ninja-build", "cmake")
        .pip_install("uv")
        .env({"CC": "gcc-14", "CXX": "g++-14"})  # the added Python's sysconfig names clang, which the image lacks
        .add_local_dir(stage / "deps", REMOTE, copy=True)
        .run_commands(sync + " --no-install-local")  # the lock's wheels: shared by commits with this lock
        .add_local_dir(stage / "src", REMOTE, copy=True)
        .run_commands(
            "git config --global --add safe.directory '*'",
            f"cd {REMOTE} && git submodule update --init --depth 1",
            f"cd {REMOTE} && git submodule status",
            sync,
        )
    )


# ---------------------------------------------------------------------------- the supervisor (in the container)

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _put_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    part.write_bytes(data)
    try:
        os.replace(part, dst)
    except OSError:  # a mount without rename
        dst.write_bytes(data)
        part.unlink(missing_ok=True)


def _put_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, part)
    try:
        os.replace(part, dst)
    except OSError:
        shutil.copyfile(src, dst)
        part.unlink(missing_ok=True)


TIME_UP, STOP = "the segment's time is up", "stop requested"
LAST = ("trainer_state.pt", "run.json")   # the trainer's commit point: copied last, from one read


def _skip(rel: Path) -> bool:
    return rel.name == ".lock" or rel.name.endswith((".tmp", ".part", ".modalpull"))


def sync_run(src: Path, dst: Path, seen: dict[str, tuple]) -> int:
    """Copy what changed in a run directory; returns the number of files written.

    trainer_state.pt and run.json are read first, together and unchanged across the read, and written
    last: everything else copied in between is at least as new as they are, which is the state a
    trainer that died right after its commit point leaves (newer checkpoints are set aside and newer
    metric rows hidden on resume). A .jsonl is cut at its last newline.
    """
    if not (src / "run.json").exists():
        return 0
    for _ in range(20):
        before = [(src / n).stat().st_mtime_ns if (src / n).exists() else None for n in LAST]
        blobs = {n: (src / n).read_bytes() for n in LAST if (src / n).exists()}
        if before == [(src / n).stat().st_mtime_ns if (src / n).exists() else None for n in LAST]:
            break
        time.sleep(0.1)
    wrote, present = 0, set()
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src)
        present.add(rel.as_posix())
        if _skip(rel) or rel.as_posix() in LAST:
            continue
        try:
            st = path.stat()
            sig = (st.st_size, st.st_mtime_ns)
            if seen.get(rel.as_posix()) == sig:
                continue
            if path.suffix == ".jsonl":
                data = path.read_bytes()
                _put_bytes(dst / rel, data[: data.rfind(b"\n") + 1])
            else:
                _put_file(path, dst / rel)
        except FileNotFoundError:  # a temporary file of the trainer, or a checkpoint set aside
            continue
        seen[rel.as_posix()] = sig
        wrote += 1
    for name in LAST:
        if name in blobs:
            sig = hashlib.sha256(blobs[name]).hexdigest()
            if seen.get(name) != sig:
                _put_bytes(dst / name, blobs[name])
                seen[name] = sig
                wrote += 1
    for path in sorted(p for p in dst.rglob("*") if p.is_file()):  # what the trainer moved or removed
        rel = path.relative_to(dst).as_posix()
        if rel not in present and not path.name.endswith(".part"):
            path.unlink()
            seen.pop(rel, None)
    return wrote


def _signatures(run_dir: Path) -> dict:
    """sync_run's record of a directory that equals its copy on the Volume."""
    seen: dict = {}
    for path in run_dir.rglob("*"):
        rel = path.relative_to(run_dir)
        if path.is_file() and not _skip(rel):
            st = path.stat()
            seen[rel.as_posix()] = (hashlib.sha256(path.read_bytes()).hexdigest() if rel.as_posix() in LAST
                                    else (st.st_size, st.st_mtime_ns))
    return seen


def restore_run(src: Path, dst: Path) -> int:
    """Copy a run directory from the Volume to the container's disk (threads: a Volume read is slow per file)."""
    from concurrent.futures import ThreadPoolExecutor

    files = [p for p in src.rglob("*") if p.is_file() and not _skip(p.relative_to(src))]
    for parent in {p.parent for p in files}:
        (dst / parent.relative_to(src)).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda p: shutil.copy2(p, dst / p.relative_to(src)), files))
    return len(files)


def progress_of(run_dir: Path) -> dict:
    """Where a run is, from run.json, config.json and the tail of metrics.jsonl."""
    out: dict = {}
    run = _read_json(run_dir / "run.json") or {}
    out["status"] = run.get("status")
    out["commit_epoch"] = (run.get("trainer_state") or {}).get("epoch")
    out["checkpoints"] = len(run.get("checkpoints") or [])
    train = ((_read_json(run_dir / "config.json") or {}).get("config") or {}).get("train") or {}
    if train.get("stop_epoch"):
        out["target"] = ["epoch", int(train["stop_epoch"])]
    elif train.get("iterations"):
        out["target"] = ["epoch", int(train["iterations"])]
    elif train.get("total_timesteps"):
        out["target"] = ["agent_steps", int(train["total_timesteps"])]
    try:
        with open(run_dir / "metrics.jsonl", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 262144))
            lines = [line for line in f.read().split(b"\n") if line.strip()]
    except OSError:
        lines = []
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a torn last line, or the cut first one
        out["epoch"] = row.get("epoch")
        out["agent_steps"] = row.get("agent_steps", row.get("step"))
        out["trainer"] = {k.split("/", 1)[1]: round(v, 3) for k, v in row.items()
                          if k.startswith("throughput/") and isinstance(v, (int, float))}
        break
    return out


def segment_plan(budget_s: float, used_s: float, timeout_s: float, segment_s: float | None) -> tuple[float, float]:
    """(seconds this call may supervise, the trainer's own time limit in hours)."""
    seg = min(budget_s - used_s, timeout_s - MARGIN_S, segment_s or DAY_S)
    grace = min(600.0, 0.1 * seg)
    return seg, max(seg - grace, 1.0) / 3600


def _supervise(spec: dict) -> dict:
    started = time.time()
    exp, variant = spec["run"].split("/")
    side = Path(VOL) / "_modal" / exp
    state_path, stop_path, log_vol = (side / f"{variant}.state.json", side / f"{variant}.stop",
                                      side / f"{variant}.log")
    vol_run, loc_run = Path(VOL) / "runs" / exp / variant, Path(LOCAL_RUNS) / exp / variant
    log_loc = Path("/root/logs") / f"{exp}--{variant}.log"
    log_loc.parent.mkdir(parents=True, exist_ok=True)

    state = _read_json(state_path) or {}
    if state.get("launch_id") != spec["launch_id"]:
        state = {"launch_id": spec["launch_id"], "used_s": 0.0, "segments": 0, "samples": [],
                 "earlier_s": float(state.get("earlier_s", 0.0)) + float(state.get("used_s", 0.0))}
    used_before = float(state["used_s"])
    state["segments"] += 1

    def save_state(**kw) -> None:
        state.update(kw, used_s=used_before + time.time() - started, heartbeat=time.time(),
                     task=os.environ.get("MODAL_TASK_ID"))
        _put_bytes(state_path, json.dumps(state, indent=1, sort_keys=True).encode())
        try:
            if log_loc.exists():
                with open(log_loc, "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 200_000))
                    _put_bytes(log_vol, f.read())
            volume.commit()
        except Exception as e:  # noqa: BLE001  a failed commit is retried by the next one
            print(f"[modal_train] commit failed: {type(e).__name__}: {e}", flush=True)

    def stop_requested() -> bool:
        return (_read_json(stop_path) or {}).get("launch_id") == spec["launch_id"]

    seg_s, limit_h = segment_plan(spec["budget_s"], used_before, spec["timeout_s"], spec.get("segment_s"))
    if stop_requested() or spec["budget_s"] - used_before < 30:
        save_state(phase="stopped" if stop_requested() else "budget_exhausted")
        print(f"[modal_train] nothing to do: {state['phase']}", flush=True)
        return state

    # the run and the shipped refs, from the Volume to this container's disk. A warm container (the next
    # segment of the same launch) keeps its own copy and its record of what the Volume already holds.
    seen: dict[str, tuple] = {}
    seen_path = log_loc.with_suffix(".seen.json")
    for name in [spec["run"], *spec.get("ships", [])]:
        src, dst = Path(VOL) / "runs" / name, Path(LOCAL_RUNS) / name
        if src.is_dir() and not (dst / "run.json").exists():
            t0 = time.time()
            count = restore_run(src, dst)
            print(f"[modal_train] restored {name} from the Volume: {count} files in {time.time() - t0:.0f} s", flush=True)
            if name == spec["run"]:  # what was just copied down need not go up again
                seen = _signatures(dst)
        elif name == spec["run"]:
            kept = _read_json(seen_path) or {}
            if kept.get("launch_id") == spec["launch_id"]:
                seen = {k: tuple(v) if isinstance(v, list) else v for k, v in kept["seen"].items()}
    start = progress_of(loc_run)
    print(f"[modal_train] {spec['run']} rev {spec['rev'][:10]} segment {state['segments']}: {seg_s:.0f} s here, "
          f"trainer limit {limit_h * 3600:.0f} s, budget {spec['budget_s']:.0f} s ({used_before:.0f} used); "
          f"on the Volume: {start.get('status') or 'no run yet'}"
          + (f" at epoch {start.get('commit_epoch')}" if start.get("status") else ""), flush=True)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; "
                   "cat /sys/fs/cgroup/cpu.max; grep -m1 'model name' /proc/cpuinfo", shell=True, check=False)

    env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
    env.update(CATANRL_RUNS_DIR=LOCAL_RUNS, WANDB_MODE="disabled", PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [f"{REMOTE}/.venv/bin/catanrl", "run", spec["config"], "--variant", variant,
           "--set", f"runtime.time_limit_hours={limit_h:.6f}"]
    for item in spec.get("sets", []):
        cmd += ["--set", item]
    if spec.get("allow_code_change"):
        cmd.append("--allow-code-change")
    print("[modal_train] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=REMOTE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def pump() -> None:
        with open(log_loc, "ab") as log:
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    last_sync, last_key, term_at, why = started, None, None, None

    def sync(phase: str) -> dict:
        now = progress_of(loc_run)
        wrote = sync_run(loc_run, vol_run, seen)
        samples = state["samples"] + [[round(time.time(), 1), now.get("epoch"), now.get("agent_steps"),
                                       state["segments"]]]
        if len(samples) > 240:
            samples = samples[::2]
        save_state(phase=phase, progress=now, samples=samples)
        _put_bytes(seen_path, json.dumps({"launch_id": spec["launch_id"], "seen": seen}).encode())
        print(f"[modal_train] committed {wrote} files to the Volume: epoch {now.get('epoch')}, "
              f"commit point e{now.get('commit_epoch')}, {state['used_s']:.0f} of {spec['budget_s']:.0f} s",
              flush=True)
        return now

    def terminate(reason: str) -> None:
        nonlocal term_at, why
        if term_at is None:
            term_at, why = time.time(), reason
            print(f"[modal_train] {reason}: SIGTERM to the trainer (it stops at its next epoch)", flush=True)
            proc.send_signal(signal.SIGTERM)

    interrupted: BaseException | None = None
    try:
        while proc.poll() is None:
            time.sleep(2)
            now = time.time()
            run = _read_json(loc_run / "run.json") or {}
            key = (json.dumps(run.get("trainer_state"), sort_keys=True), len(run.get("checkpoints") or []))
            if (key != last_key and now - last_sync >= spec["min_sync_s"]) or now - last_sync >= spec["sync_s"]:
                try:
                    volume.reload()  # the stop request is written from outside
                except Exception as e:  # noqa: BLE001
                    print(f"[modal_train] reload failed: {type(e).__name__}: {e}", flush=True)
                sync("training")
                last_sync, last_key = time.time(), key
                if stop_requested():
                    terminate(STOP)
            if now - started >= seg_s:
                terminate(TIME_UP)
            if term_at is not None and now - term_at > 60 and proc.poll() is None:
                print("[modal_train] the trainer did not stop in 60 s: killing it", flush=True)
                proc.kill()
    except BaseException as e:  # noqa: BLE001  a preemption or a stopped container cancels the call
        interrupted = e
        print(f"[modal_train] {type(e).__name__}: stopping the trainer", flush=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    proc.wait()
    reader.join(timeout=10)

    end = progress_of(loc_run)
    status, code = end.get("status"), proc.returncode
    moved = (end.get("commit_epoch") or 0) > (start.get("commit_epoch") or 0)
    left = spec["budget_s"] - (used_before + time.time() - started)
    idle = 0 if moved else int(state.get("idle_segments", 0)) + 1   # two segments in a row without a commit point
    if interrupted is not None:
        phase = "interrupted"
    elif status == "finished" and code == 0:
        phase = "finished"
    elif why == STOP:
        phase = "stopped"
    elif status == "time_limit" or (why == TIME_UP and status in ("interrupted", "finished")):
        # "finished" here: the training is done and the finals were cut; the next segment plays them
        phase = "budget_exhausted" if left < 60 else ("respawn" if idle < 2 else "stalled")
    else:
        phase = "failed"
    state["idle_segments"] = idle
    sync(phase)
    print(f"[modal_train] trainer exit {code}, run status {status}, phase {phase}, "
          f"{left:.0f} s of the budget left", flush=True)
    if interrupted is not None:
        raise interrupted
    if phase == "respawn":
        call = train_segment.spawn(spec)
        print(f"[modal_train] next segment spawned: {call.object_id}", flush=True)
    return {k: state.get(k) for k in ("phase", "used_s", "segments", "progress")}


def train_segment(spec: dict) -> dict:
    return _supervise(spec)


def _define(name: str, image: modal.Image | None, gpu: str | None, cores: float, gib: float, timeout_s: int):
    """The app and its one function, for this launch's shape (the container defines a stand-in)."""
    global app, train_segment
    app = modal.App(name)
    raw = getattr(train_segment, "get_raw_f", lambda: train_segment)()
    train_segment = app.function(
        image=image, gpu=gpu, cpu=(float(cores), float(cores)), memory=int(gib * 1024), timeout=int(timeout_s),
        volumes={VOL: volume}, max_containers=1, scaledown_window=2,
        retries=modal.Retries(max_retries=8, backoff_coefficient=1.0, initial_delay=5.0),
        env={"WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"},
    )(raw)
    return train_segment


if not modal.is_local():
    _define(APP_PREFIX, None, None, 1, 1, 300)


# ---------------------------------------------------------------------------- the Volume, from this machine

def vol_json(path: str) -> dict | None:
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except Exception as e:
        if "NotFound" in type(e).__name__ or isinstance(e, FileNotFoundError) or "No such file" in str(e):
            return None
        raise


def vol_put(files: dict[str, bytes | Path]) -> None:
    import io
    with volume.batch_upload(force=True) as batch:
        for remote, content in files.items():
            if isinstance(content, Path):
                batch.put_file(str(content), remote)
            else:
                batch.put_file(io.BytesIO(content), remote)


def vol_list(path: str) -> list[tuple[str, int, int]]:
    """(path, size, mtime) of every file under a Volume directory; empty when it is missing."""
    try:
        entries = volume.listdir(path, recursive=True)
    except Exception as e:
        if "NotFound" in type(e).__name__ or isinstance(e, FileNotFoundError) or "No such file" in str(e):
            return []
        raise
    return sorted((e.path, int(e.size), int(e.mtime)) for e in entries if getattr(e.type, "name", "") == "FILE")


def modal_cli(*args: str, check: bool = True) -> str:
    out = subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True, text=True, check=False)
    if check and out.returncode:
        raise ToolError(f"modal {' '.join(args)}: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def app_states() -> dict[str, dict]:
    return {a["app_id"]: a for a in json.loads(modal_cli("app", "list", "--json") or "[]")}


def app_alive(app_id: str | None, states: dict[str, dict]) -> bool:
    # a detached app whose client has gone shows as "ephemeral (detached)"
    return bool(app_id) and (states.get(app_id, {}).get("state") or "stopped").split()[0] in ("ephemeral", "running", "initializing", "detached")


def sidecar(run: str) -> tuple[str, str, str, str]:
    exp, variant = run.split("/")
    base = f"_modal/{exp}/{variant}"
    return f"{base}.launch.json", f"{base}.state.json", f"{base}.stop", f"{base}.log"


def check_run(run: str) -> str:
    if not RUN_RE.match(run):
        raise ToolError(f"{run!r}: expected <experiment>/<variant>")
    return run


# ---------------------------------------------------------------------------- ship

def default_runs() -> Path:
    if os.environ.get("CATANRL_RUNS_DIR"):
        return Path(os.environ["CATANRL_RUNS_DIR"])
    common = Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent / "runs"


def ref_files(ref: str, runs: Path) -> tuple[str, list[str]]:
    """(run, the files a container needs to resolve ``ref``): config, run.json, aliases, one checkpoint."""
    m = REF_RE.match(ref)
    if not m:
        raise ToolError(f"bad ref {ref!r}: expected experiment/variant[@which][#sha]")
    run, which, sha = m["run"], m["which"] or "best", m["sha"]
    path = runs / run
    record = _read_json(path / "run.json")
    if record is None or not (path / "config.json").exists():
        raise ToolError(f"{run}: no run in {runs}")
    by_epoch = {int(c["epoch"]): c for c in record.get("checkpoints") or []}
    aliases: dict[str, dict] = {}
    if (path / "aliases.jsonl").exists():
        for line in (path / "aliases.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            aliases[row["name"]] = row
    if which == "latest":
        found = by_epoch[max(by_epoch)] if by_epoch else None
    elif which == "init":
        found = by_epoch.get(0)
    elif re.fullmatch(r"e?\d+", which):
        found = by_epoch.get(int(which.lstrip("e")))
    else:
        found = by_epoch.get(int(aliases[which]["epoch"])) if which in aliases else None
    if found is None:
        raise ToolError(f"{run} has no checkpoint {which!r}")
    if sha and not found["weights_sha256"].startswith(sha):
        raise ToolError(f"{ref}: the checkpoint is #{found['weights_sha256'][:12]}")
    files = ["config.json", "run.json", f"checkpoints/{found['file']}"]
    if (path / "aliases.jsonl").exists():
        files.append("aliases.jsonl")
    for name in files:
        if not (path / name).exists():
            raise ToolError(f"{run}: {name} is missing in {runs}")
    return run, files


def ship(refs: list[str], runs: Path) -> list[str]:
    """Upload each ref's files to the Volume's runs directory; returns the runs they belong to."""
    names, upload = [], {}
    for ref in refs:
        run, files = ref_files(ref, runs)
        names.append(run)
        have = {p: size for p, size, _ in vol_list(f"runs/{run}")}
        for name in files:
            local = runs / run / name
            if name == "run.json" and f"runs/{run}/trainer_state.pt" in have:
                continue  # that run lives on the Volume: its own run.json is the truth
            if name.startswith("checkpoints/") and have.get(f"runs/{run}/{name}") == local.stat().st_size:
                continue
            upload[f"runs/{run}/{name}"] = local
    if upload:
        vol_put(upload)
    size = sum(p.stat().st_size for p in upload.values())
    print(f"shipped {len(upload)} files ({size / 1e6:.1f} MB) for {', '.join(refs) or 'nothing'}")
    return sorted(set(names))


# ---------------------------------------------------------------------------- pull

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stat(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def _is_prefix(short: Path, long: Path) -> bool:
    if short.stat().st_size >= long.stat().st_size:
        return False
    with open(short, "rb") as a, open(long, "rb") as b:
        while True:
            x = a.read(1 << 20)
            if not x:
                return True
            if b.read(len(x)) != x:
                return False


KINDS = ("new", "same", "extended", "updated", "conflict", "forced")


def pull_files(listing: list[tuple[str, int, int]], fetch, dest: Path, manifest_path: Path,
               force: bool = False, workers: int = 8, progress: bool = False) -> dict[str, list[str]]:
    """The no-overwrite rule. ``listing`` is (relative path, size, mtime); ``fetch(rel, fileobj)`` reads one.

    new: no local file. same: equal content. extended: a .jsonl or .log whose local copy is a strict
    prefix of the Volume's. updated: the local file is the one this tool pulled last time (the manifest
    has its hash), the Volume's has changed. conflict: anything else; the local file stays. forced: a
    conflict overwritten because the caller said so.
    """
    from concurrent.futures import ThreadPoolExecutor

    manifest = _read_json(manifest_path) or {}
    report: dict[str, list[str]] = {k: [] for k in KINDS}

    def save_manifest() -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        os.replace(tmp, manifest_path)

    need = []
    for rel, size, mtime in listing:
        if _skip(Path(rel)):
            continue
        local, known = dest / rel, manifest.get(rel)
        if local.exists() and known and known["remote"] == [size, mtime] and known["local"] == _stat(local):
            report["same"].append(rel)
        else:
            need.append((rel, size, mtime))

    def download(item: tuple[str, int, int]) -> Path:
        local = dest / item[0]
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(f".{local.name}.modalpull")
        with open(tmp, "wb") as f:
            fetch(item[0], f)
        return tmp

    save_manifest()  # the directory is this tool's from the first file on, also if the pull is cut short
    total, done, started, said = sum(size for _, size, _ in need), 0, time.time(), time.time()
    try:
        with ThreadPoolExecutor(workers) as pool:
            for n, ((rel, size, mtime), tmp) in enumerate(zip(need, pool.map(download, need)), 1):
                local, known = dest / rel, manifest.get(rel)
                theirs = _sha(tmp)
                if not local.exists():
                    kind = "new"
                else:
                    ours = _sha(local)
                    if ours == theirs:
                        kind = "same"
                    elif local.suffix in (".jsonl", ".log") and _is_prefix(local, tmp):
                        kind = "extended"
                    elif known and known.get("sha256") == ours:
                        kind = "updated"
                    else:
                        kind = "forced" if force else "conflict"
                if kind in ("same", "conflict"):
                    tmp.unlink()
                else:
                    os.replace(tmp, local)
                if kind != "conflict":
                    manifest[rel] = {"remote": [size, mtime], "local": _stat(local), "sha256": theirs}
                report[kind].append(rel)
                done += size
                if progress and time.time() - said >= 10:
                    said, rate = time.time(), n / (time.time() - started)
                    print(f"  pull: {n}/{len(need)} files, {done / 1e6:.0f} of {total / 1e6:.0f} MB, "
                          f"ETA {(len(need) - n) / rate / 60:.1f} min", flush=True)
    finally:
        save_manifest()
    return report


def pull(run: str, runs: Path, force: bool = False, merge: bool = False) -> dict[str, list[str]]:
    exp, variant = run.split("/")
    dest, manifest = runs / exp / variant, runs / "_modal" / exp / f"{variant}.pull.json"
    if dest.exists() and not manifest.exists() and not merge:
        raise ToolError(f"{dest} exists and was not made by this tool (a local run of the same name?): "
                        f"pull into another directory with --runs, or pass --merge for the per-file rule")
    prefix = f"runs/{run}/"
    listing = [(p[len(prefix):], size, mtime) for p, size, mtime in vol_list(prefix.rstrip("/"))]
    if not listing:
        raise ToolError(f"the Volume {VOLUME} has no runs/{run}")
    report = pull_files(listing, lambda rel, f: volume.read_file_into_fileobj(prefix + rel, f), dest, manifest, force,
                        progress=True)
    parts = [f"{len(report[k])} {k}" for k in KINDS if report[k]]
    print(f"{time.strftime('%H:%M:%S')} pull {run} -> {dest}: {', '.join(parts) or 'nothing on the Volume'}")
    for kind, note in (("extended", "appended rows"), ("updated", "replaced: pulled earlier, changed on the Volume"),
                       ("forced", "OVERWRITTEN (--force)"), ("conflict", "CONFLICT: differs locally, local file kept")):
        for rel in report[kind]:
            print(f"  {rel}: {note}")
    return report


def mirror(run: str, runs: Path) -> int:
    """W&B backfill of the pulled run, from this machine (tools/wandb_backfill.py; off under WANDB_MODE=disabled)."""
    cmd = [sys.executable, str(REPO / "tools" / "wandb_backfill.py"), str(runs / run), "--tag", "modal"]
    return subprocess.run(cmd, env={**os.environ, "CATANRL_RUNS_DIR": str(runs)}, check=False).returncode


# ---------------------------------------------------------------------------- commands

def _launch(spec: dict, image_stage: Path) -> str:
    shape = spec["shape"]
    launch_path, _, stop_path, _ = sidecar(spec["run"])
    fn = _define(f"{APP_PREFIX}-{spec['run'].replace('/', '-')}", build_image(image_stage), shape["gpu"],
                 shape["cores"], shape["GiB"], spec["timeout_s"])
    with modal.enable_output(), app.run(detach=True):
        spec["app_id"] = app.app_id
        vol_put({launch_path: json.dumps(spec, indent=1, sort_keys=True).encode(), stop_path: b"{}"})
        call = fn.spawn(spec)
        print(f"spawned {call.object_id} in {app.app_id}")
    return spec["app_id"]


def cmd_launch(args) -> int:
    sha = resolve_rev(args.rev)
    config, exp, swept = config_at(sha, args.config)
    if swept and not args.variant:
        raise ToolError(f"{config} has a sweep or stages: name one variant with --variant")
    run = check_run(f"{exp}/{args.variant or 'main'}")
    gpu = None if args.gpu.lower() in ("none", "cpu", "") else args.gpu
    rate = usd_per_s(gpu, args.cpu, args.memory, args.gpu_usd_per_hour)
    budget_s = args.max_hours * 3600
    launch_path = sidecar(run)[0]
    earlier = vol_json(launch_path)
    if earlier and app_alive(earlier.get("app_id"), app_states()):
        raise ToolError(f"{run} is running in {earlier['app_id']}: stop it first")
    if earlier and earlier.get("rev") != sha:
        print(f"note: {run} was last launched at {earlier.get('rev', '?')[:10]}; this is {sha[:10]} "
              f"(the trainer refuses a resume across a training code change without --allow-code-change)")
    pins = unpushed_pins(sha)
    if pins:
        print(f"warning: not seen on a remote-tracking branch here (the image clones them from the forks): {', '.join(pins)}")
    ships = ship(args.ship, Path(args.runs) if args.runs else default_runs()) if args.ship else []
    stage = stage_revision(sha)
    spec = {
        "run": run, "config": config, "variant": args.variant or "main", "rev": sha, "sets": args.set,
        "allow_code_change": args.allow_code_change, "ships": sorted(set(ships) | set((earlier or {}).get("ships", []))),
        "shape": {"gpu": gpu, "cores": args.cpu, "GiB": args.memory}, "usd_per_s": rate,
        "budget_s": budget_s, "timeout_s": int(min(DAY_S, budget_s + MARGIN_S)),
        "segment_s": args.segment_hours * 3600 if args.segment_hours else None,
        "sync_s": args.sync_minutes * 60, "min_sync_s": args.min_sync_seconds,
        "launch_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}", "launched": time.time(),
    }
    print(f"{run} at {sha[:10]} ({args.rev}), {config}")
    print(cost_line(gpu, args.cpu, args.memory, args.max_hours, rate))
    if args.dry_run:
        print(json.dumps(spec, indent=1, sort_keys=True))
        return 0
    app_id = _launch(spec, stage)
    print(f"detached: {app_id}. Progress: tools/modal_train.py status {run}; logs: modal app logs {app_id}; "
          f"stop: tools/modal_train.py stop {run}")
    return 0


def cmd_resume(args) -> int:
    run = check_run(args.run)
    launch_path, state_path, _, _ = sidecar(run)
    spec = vol_json(launch_path)
    if spec is None:
        raise ToolError(f"{run} was never launched (no {launch_path} on the Volume)")
    if app_alive(spec.get("app_id"), app_states()):
        raise ToolError(f"{run} is running in {spec['app_id']}: stop it first")
    state = vol_json(state_path) or {}
    left = spec["budget_s"] - float(state.get("used_s", 0.0)) if state.get("launch_id") == spec["launch_id"] else 0.0
    budget_s = args.max_hours * 3600 if args.max_hours else left
    if budget_s < 60:
        raise ToolError(f"{run}: the last budget is used up; pass --max-hours")
    shape = spec["shape"]
    rate = usd_per_s(shape["gpu"], shape["cores"], shape["GiB"], spec["usd_per_s"] * 3600 - 3600 * (
        shape["cores"] * PRICE["core"] + shape["GiB"] * PRICE["GiB"]) if shape["gpu"] not in PRICE else None)
    spec.update(budget_s=budget_s, timeout_s=int(min(DAY_S, budget_s + MARGIN_S)), usd_per_s=rate,
                launch_id=f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}", launched=time.time())
    if args.allow_code_change:
        spec["allow_code_change"] = True
    print(f"{run} at {spec['rev'][:10]}, {spec['config']}: resuming from the Volume")
    print(cost_line(shape["gpu"], shape["cores"], shape["GiB"], budget_s / 3600, rate))
    app_id = _launch(spec, stage_revision(spec["rev"]))
    print(f"detached: {app_id}")
    return 0


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h" if seconds >= 5400 else f"{seconds / 60:.1f} min"


def wall_rate(samples: list, segment: int | None = None) -> tuple[float, float] | None:
    """(epochs per second, agent steps per second) over the wall clock of the last segment's samples."""
    rows = [s for s in samples if s[1] is not None and (segment is None or len(s) < 4 or s[3] == segment)]
    if len(rows) < 2 or rows[-1][0] <= rows[0][0]:
        return None
    dt = rows[-1][0] - rows[0][0]
    return (rows[-1][1] - rows[0][1]) / dt, ((rows[-1][2] or 0) - (rows[0][2] or 0)) / dt


def eta_seconds(progress: dict, rate: tuple[float, float] | None) -> float | None:
    target = progress.get("target")
    if not target or rate is None:
        return None
    unit, total = target
    done, per_s = (progress.get("epoch"), rate[0]) if unit == "epoch" else (progress.get("agent_steps"), rate[1])
    if done is None or per_s <= 0:
        return None
    return max(total - done, 0) / per_s


def status_text(run: str, spec: dict, state: dict | None, alive: bool, app_state: str, now: float) -> str:
    shape = spec["shape"]
    head = (f"{run}  app {spec.get('app_id')} {app_state}{'' if alive else ' (not running)'}  rev {spec['rev'][:10]}  "
            f"{shape['gpu'] or 'no GPU'}, {shape['cores']:g} cores, {shape['GiB']:g} GiB")
    lines = [head]
    if not state or state.get("launch_id") != spec["launch_id"]:
        lines.append("  no heartbeat of this launch on the Volume yet (image build or cold start)")
        return "\n".join(lines)
    p = state.get("progress") or {}
    target = p.get("target") or [None, None]
    where = f"epoch {p.get('epoch')}"
    if target[0] == "epoch":
        where += f" of {target[1]}"
    if p.get("agent_steps") is not None:
        where += f", {p['agent_steps'] / 1e6:.2f}M agent steps"
        if target[0] == "agent_steps":
            where += f" of {target[1] / 1e6:.1f}M ({100 * p['agent_steps'] / target[1]:.1f}%)"
    phase = state.get("phase")
    if not alive and phase in ("training", "respawn", "interrupted"):
        phase = f"not running (it was {phase})"
    lines.append(f"  {phase}: {where}; run status {p.get('status')}, commit point e{p.get('commit_epoch')}, "
                 f"{p.get('checkpoints')} checkpoints; segment {state.get('segments')}")
    rate = wall_rate(state.get("samples") or [], state.get("segments"))
    trainer = p.get("trainer") or {}
    bits = []
    if "agent_steps_per_s" in trainer:
        bits.append(f"{trainer['agent_steps_per_s']:,.0f} agent steps/s in the last epoch (trainer)")
    if "iteration_s" in trainer:
        bits.append(f"{trainer['iteration_s']:.1f} s per iteration (trainer)")
    if rate:
        steps = f", {rate[1]:,.0f} agent steps/s" if rate[1] else ""
        bits.append(f"{rate[0] * 60:.2f} epochs/min{steps} over the wall clock of this segment")
    eta = eta_seconds(p, rate)
    left = spec["budget_s"] - state["used_s"]
    if eta is not None:
        bits.append(f"ETA {_hours(eta)}" + (" (past the budget)" if eta > left else ""))
    if bits:
        lines.append("  " + "; ".join(bits))
    spent, earlier = state["used_s"] * spec["usd_per_s"], state.get("earlier_s", 0.0) * spec["usd_per_s"]
    lines.append(f"  heartbeat {_hours(max(now - state.get('heartbeat', now), 0))} ago; used {_hours(state['used_s'])} of "
                 f"{_hours(spec['budget_s'])}; cost so far ${spent:.2f} at ${spec['usd_per_s'] * 3600:.3f}/h"
                 + (f" (+ ${earlier:.2f} in earlier launches at this price)" if earlier else ""))
    return "\n".join(lines)


def known_runs() -> list[str]:
    return sorted(p[len("_modal/"):-len(".launch.json")] for p, _, _ in vol_list("_modal") if p.endswith(".launch.json"))


def cmd_status(args) -> int:
    runs = [check_run(args.run)] if args.run else known_runs()
    states = app_states()
    for run in runs:
        launch_path, state_path, _, log_path = sidecar(run)
        spec = vol_json(launch_path)
        if spec is None:
            raise ToolError(f"{run} was never launched (no {launch_path} on the Volume)")
        alive = app_alive(spec.get("app_id"), states)
        app_state = states.get(spec.get("app_id"), {}).get("state", "gone")
        print(status_text(run, spec, vol_json(state_path), alive, app_state, time.time()))
        if args.log:
            try:
                tail = b"".join(volume.read_file(log_path)).decode(errors="replace").splitlines()[-args.log:]
                print("\n".join("  | " + line for line in tail))
            except Exception:  # noqa: BLE001
                print("  no log on the Volume yet")
    if not runs:
        print(f"the Volume {VOLUME} knows no run")
    return 0


def cmd_list(args) -> int:
    states = app_states()
    runs = known_runs()
    if not runs:
        print(f"the Volume {VOLUME} knows no run")
    for run in runs:
        launch_path, state_path, _, _ = sidecar(run)
        spec, state = vol_json(launch_path) or {}, vol_json(state_path) or {}
        p = state.get("progress") or {}
        used = state.get("used_s", 0.0) + state.get("earlier_s", 0.0)
        print(f"{run:32} {'alive' if app_alive(spec.get('app_id'), states) else 'not running':12} "
              f"{state.get('phase', '-'):17} epoch {p.get('epoch')!s:>6}  {spec.get('shape', {}).get('gpu') or 'cpu':5} "
              f"${used * spec.get('usd_per_s', 0.0):7.2f}  rev {spec.get('rev', '?')[:10]}  {spec.get('app_id')}")
    return 0


def cmd_stop(args) -> int:
    run = check_run(args.run)
    launch_path, _, stop_path, _ = sidecar(run)
    spec = vol_json(launch_path)
    if spec is None:
        raise ToolError(f"{run} was never launched")
    if not app_alive(spec.get("app_id"), app_states()):
        print(f"{run}: {spec.get('app_id')} is not running")
        return 0
    if args.graceful:
        vol_put({stop_path: json.dumps({"launch_id": spec["launch_id"], "time": time.time()}).encode()})
        print(f"{run}: stop requested; the trainer ends its epoch at the supervisor's next sync "
              f"(within {spec['sync_s'] / 60:g} min), commits and the app ends by itself")
        return 0
    modal_cli("app", "stop", spec["app_id"], "--yes")
    print(f"{run}: stopped {spec['app_id']}; the Volume keeps the last commit (tools/modal_train.py status {run})")
    return 0


def cmd_pull(args) -> int:
    run, runs = check_run(args.run), Path(args.runs) if args.runs else default_runs()
    pull(run, runs, force=args.force, merge=args.merge)
    return 0 if args.no_mirror else mirror(run, runs)


def cmd_watch(args) -> int:
    run, runs = check_run(args.run), Path(args.runs) if args.runs else default_runs()
    launch_path = sidecar(run)[0]
    while True:
        spec = vol_json(launch_path) or {}
        alive = app_alive(spec.get("app_id"), app_states())
        report = pull(run, runs, merge=args.merge)
        if not args.no_mirror:
            mirror(run, runs)
        if not alive and not any(report[k] for k in ("new", "extended", "updated")):
            print(f"{run}: the app is not running and nothing new came: done")
            return 0
        time.sleep(args.every * 60)


def cmd_ship(args) -> int:
    ship(args.refs, Path(args.runs) if args.runs else default_runs())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch", help="start a run detached")
    la.add_argument("config", help="an experiment YAML as committed in the revision (repo-relative path)")
    la.add_argument("--rev", default="HEAD", help="any commit of this repository (default: this worktree's HEAD)")
    la.add_argument("--variant", help="needed when the config has a sweep or stages")
    la.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="passed to catanrl run")
    la.add_argument("--gpu", default="L40S", help="L40S, A10G, T4, L4, A100, A100-80GB, H100, H200, B200 or none")
    la.add_argument("--gpu-usd-per-hour", type=float, help="list price of a GPU the table does not know")
    la.add_argument("--cpu", type=float, default=8, help="cores (request and limit)")
    la.add_argument("--memory", type=float, default=32, help="GiB")
    la.add_argument("--max-hours", type=float, required=True, help="budget of this launch; the run stops at a checkpoint")
    la.add_argument("--ship", action="append", default=[], metavar="REF", help="a local run ref the container must resolve")
    la.add_argument("--runs", help="the local runs directory of --ship (default: $CATANRL_RUNS_DIR or the main checkout's)")
    la.add_argument("--sync-minutes", type=float, default=5, help="commit to the Volume at least this often")
    la.add_argument("--min-sync-seconds", type=float, default=30, help="and at most this often on new checkpoints")
    la.add_argument("--segment-hours", type=float, help="end each call after this long and spawn the next (default: 24 h cap)")
    la.add_argument("--allow-code-change", action="store_true", help="passed to catanrl run")
    la.add_argument("--dry-run", action="store_true", help="resolve, stage and price, but start nothing")
    la.set_defaults(fn=cmd_launch)
    re_ = sub.add_parser("resume", help="relaunch a stopped run from the Volume: same revision, config and shape")
    re_.add_argument("run")
    re_.add_argument("--max-hours", type=float, help="a new budget (default: what the last one has left)")
    re_.add_argument("--allow-code-change", action="store_true")
    re_.set_defaults(fn=cmd_resume)
    st = sub.add_parser("status", help="epoch, rate, ETA, cost so far and whether the app is alive")
    st.add_argument("run", nargs="?")
    st.add_argument("--log", type=int, default=0, metavar="N", help="the last N lines of the trainer's output on the Volume")
    st.set_defaults(fn=cmd_status)
    sub.add_parser("list", help="every run the Volume knows").set_defaults(fn=cmd_list)
    sp = sub.add_parser("stop", help="stop the app; the Volume keeps the last commit")
    sp.add_argument("run")
    sp.add_argument("--graceful", action="store_true", help="let the trainer end its epoch and commit first")
    sp.set_defaults(fn=cmd_stop)
    for name, fn, text in (("pull", cmd_pull, "copy a run from the Volume into the local runs directory, then mirror to W&B"),
                           ("watch", cmd_watch, "pull and mirror in a loop until the app has ended")):
        pu = sub.add_parser(name, help=text)
        pu.add_argument("run")
        pu.add_argument("--runs", help="the local runs directory (default: $CATANRL_RUNS_DIR or the main checkout's)")
        pu.add_argument("--merge", action="store_true", help="pull into a run directory this tool did not make")
        pu.add_argument("--no-mirror", action="store_true", help="skip the W&B backfill")
        if name == "pull":
            pu.add_argument("--force", action="store_true", help="overwrite conflicting local files (each is named)")
        else:
            pu.add_argument("--every", type=float, default=10, metavar="MINUTES")
        pu.set_defaults(fn=fn)
    sh = sub.add_parser("ship", help="upload the files of local run refs to the Volume")
    sh.add_argument("refs", nargs="+")
    sh.add_argument("--runs")
    sh.set_defaults(fn=cmd_ship)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
