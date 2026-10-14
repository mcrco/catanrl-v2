"""Provenance and code identity (DESIGN 6.7).

:func:`code_id` hashes the code that decides outcomes (by AST, so comments
and formatting change nothing; no git SHA, so unrelated commits keep every
cache valid). Eval keys and training resumes compare it.

:func:`capture` records everything else about a launch or an eval: git
state of the superproject and every submodule (with a patch per dirty repo
and a keep ref per commit), the engine build, the Python packages, torch and
the process. Dirty trees are allowed (the patch is saved); a submodule that
differs from the superproject's record warns (``strict`` refuses).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, Literal

from catanrl.canon import ast_digest, digest, sha256_file

PACKAGE = Path(__file__).resolve().parent
#: catanrl's outcome code: what an eval's result depends on.
OUTCOME = ("models", "agents", "search", "envs", "eval/harness.py", "maps.py", "games.py")
ENV_VARS = ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "PYTHONHASHSEED", "CATANRL_RUNS_DIR", "ADVANTAGE_CUDA")
_PROCESS_START = time.time()


class ProvenanceError(RuntimeError):
    pass


class StaleProcess(ProvenanceError):
    """The code on disk is not the code this process imported."""


# ---------------------------------------------------------------- code identity


def _py_files(root: Path, label: str) -> list[tuple[str, Path]]:
    if root.is_file():
        return [(label, root)]
    return [(f"{label}/{p.relative_to(root).as_posix()}", p) for p in sorted(root.rglob("*.py"))
            if "__pycache__" not in p.parts]


def _components(kind: str, harness: str) -> dict[str, dict[str, list]]:
    """Per component: Python files (hashed by AST) and binaries (hashed by bytes)."""
    import cppanatron
    import cppanatron._core

    engine_dir = Path(cppanatron.__file__).parent
    parts: dict[str, dict[str, list]] = {
        "catanrl": {"py": [f for name in OUTCOME for f in _py_files(PACKAGE / name, name)], "bin": []},
        "engine": {"py": _py_files(engine_dir, "cppanatron"), "bin": [Path(cppanatron._core.__file__)]},
    }
    if harness == "catanatron":
        import catanatron

        parts["catanatron"] = {"py": _py_files(Path(catanatron.__file__).parent, "catanatron"), "bin": []}
    if kind == "train":
        import pufferlib

        puffer = Path(pufferlib.__file__).parent
        parts["train"] = {"py": _py_files(PACKAGE / "train", "train"), "bin": []}
        parts["pufferlib"] = {"py": [(n, puffer / n) for n in ("pufferl.py", "pytorch.py")],
                              "bin": sorted(puffer.glob("_C*.so"))}
    return parts


def _digest_component(files: dict[str, list]) -> str:
    h = hashlib.sha256(ast_digest(files["py"]).encode())
    for path in files["bin"]:
        h.update(f"|{Path(path).name}|{sha256_file(path)}".encode())
    return h.hexdigest()[:16]


_SNAPSHOTS: dict[tuple[str, str], tuple[dict[str, str], dict[str, dict[str, list]]]] = {}


def code_id(kind: Literal["eval", "train"] = "eval", harness: Literal["native", "catanatron"] = "native") -> dict:
    """``{component: digest}`` of the outcome code, frozen at the first call.

    The first call also refuses files modified after this process started:
    the process may be running older code than the disk holds.
    """
    if kind not in ("eval", "train") or harness not in ("native", "catanatron", "server"):
        raise ValueError(f"code_id({kind!r}, {harness!r})")
    harness = "catanatron" if harness == "server" else harness
    key = (kind, harness)
    if key not in _SNAPSHOTS:
        components = _components(kind, harness)
        late = [str(p) for files in components.values() for _, p in files["py"]
                if p.stat().st_mtime > _PROCESS_START + 1.0]
        if late and any(_imported(p) for p in late):
            raise StaleProcess(f"{late[0]} changed after this process started; restart it")
        _SNAPSHOTS[key] = ({name: _digest_component(files) for name, files in components.items()}, components)
    return dict(_SNAPSHOTS[key][0])


def _imported(path: str) -> bool:
    return any(getattr(m, "__file__", None) == path for m in list(sys.modules.values()))


def assert_code_current() -> None:
    """Raise :class:`StaleProcess` if any hashed file changed since the snapshot."""
    for (kind, harness), (ids, components) in _SNAPSHOTS.items():
        for name, files in components.items():
            if _digest_component(files) != ids[name]:
                raise StaleProcess(f"{name} code changed on disk since this process hashed it "
                                   f"({kind}/{harness}); restart the process")


# ---------------------------------------------------------------- git


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise ProvenanceError(f"git {' '.join(args)} in {root}: {result.stderr.strip()}")
    return result.stdout


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False).stdout


@cache
def repo_root() -> Path:
    return Path(_git(PACKAGE, "rev-parse", "--show-toplevel").strip())


def repo_state(root: Path) -> dict[str, Any]:
    """HEAD, branch, dirty, diff sha and untracked files of one repo."""
    status = _git(root, "status", "--porcelain", "--ignore-submodules=all")
    untracked = [line for line in _git(root, "ls-files", "--others", "--exclude-standard").splitlines() if line]
    diff = _git_bytes(root, "diff", "HEAD", "--binary", "--ignore-submodules=all")
    return {
        "path": str(root),
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(status.strip()),
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
        "untracked": untracked,
    }


def git_state(root: Path | None = None) -> dict[str, Any]:
    """The superproject and every submodule: recorded vs checked-out commits."""
    root = Path(root) if root is not None else repo_root()
    repos = {"catanrl-v2": repo_state(root)}
    submodules = {}
    for line in _git(root, "submodule", "status").splitlines():
        if not line.strip():
            continue
        flag, rest = line[0], line[1:].split()
        sha, path = rest[0], rest[1]
        recorded = _git(root, "ls-tree", "HEAD", path).split()
        submodules[path] = {"state": {" ": "ok", "+": "moved", "-": "missing", "U": "conflict"}.get(flag, flag),
                            "checked_out": sha, "recorded": recorded[2] if len(recorded) > 2 else None}
        if flag != "-":
            repos[path] = repo_state(root / path)
    return {"root": str(root), "repos": repos, "submodules": submodules}


def save_patches(prov: dict, dest: Path, *, cap_bytes: int = 20 << 20) -> None:
    """One ``<launch>.<repo>.patch`` per dirty repo (tracked diff plus untracked files)."""
    dest = Path(dest)
    stem = prov.get("run", {}).get("launch_name", "launch")
    for name, repo in prov["git"]["repos"].items():
        if not repo["dirty"] and not repo["untracked"]:
            continue
        root = Path(repo["path"])
        patch = _git_bytes(root, "diff", "HEAD", "--binary", "--ignore-submodules=all")
        for path in repo["untracked"]:
            patch += subprocess.run(["git", "-C", str(root), "diff", "--no-index", "--binary", "/dev/null", path],
                                    capture_output=True, check=False).stdout
        target = dest / f"{stem}.{name.replace('/', '_')}.patch"
        if len(patch) > cap_bytes:
            repo["patch"] = {"skipped": f"{len(patch)} bytes > cap {cap_bytes}"}
            continue
        dest.mkdir(parents=True, exist_ok=True)
        target.write_bytes(patch)
        repo["patch"] = {"file": target.name, "sha256": hashlib.sha256(patch).hexdigest()}


def keep_commits(prov: dict) -> None:
    """``refs/catanrl/keep/<sha>`` in each repo, so recorded commits stay reachable."""
    for repo in prov["git"]["repos"].values():
        _git(Path(repo["path"]), "update-ref", f"refs/catanrl/keep/{repo['head']}", repo["head"], check=False)


# ---------------------------------------------------------------- the rest


def engine_info() -> dict[str, Any]:
    import cppanatron
    import cppanatron._core

    return {"so_sha256": sha256_file(cppanatron._core.__file__), "build": dict(cppanatron.build_info())}


def python_info(root: Path | None = None) -> dict[str, Any]:
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                   if d.metadata["Name"])
    lock = (Path(root) if root else repo_root()) / "uv.lock"
    return {"version": sys.version, "executable": sys.executable, "packages": dists,
            "packages_digest": digest(dists), "uv_lock_sha256": sha256_file(lock) if lock.exists() else None}


def torch_runtime() -> dict[str, Any]:
    """Torch versions, devices and the flags in force (call after PuffeRL.__init__)."""
    import torch

    cuda = torch.cuda.is_available()
    try:
        autocast = torch.is_autocast_enabled("cuda")
    except TypeError:
        autocast = torch.is_autocast_enabled()
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "threads": torch.get_num_threads(),
        "flags": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                  "cudnn_benchmark": torch.backends.cudnn.benchmark,
                  "cudnn_deterministic": torch.backends.cudnn.deterministic,
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "matmul_precision": torch.get_float32_matmul_precision(), "autocast_cuda": autocast},
    }


def capture(*, argv: Sequence[str] | None = None, kind: Literal["eval", "train"] = "eval",
            harness: str = "native") -> dict:
    """Everything about this process worth keeping next to a launch or an eval."""
    assert_code_current()
    return {
        "time": time.time(),
        "git": git_state(),
        "code": code_id(kind, harness),
        "engine": engine_info(),
        "python": python_info(),
        "torch": torch_runtime(),
        "run": {"argv": list(sys.argv if argv is None else argv), "cwd": os.getcwd(), "host": socket.gethostname(),
                "pid": os.getpid(), "platform": platform.platform(),
                "env": {k: os.environ[k] for k in ENV_VARS if k in os.environ}},
    }


def check(prov: dict, *, strict: bool = False) -> list[str]:
    """Warnings about a capture; ``strict`` raises on a submodule mismatch."""
    warnings = []
    for path, sub in prov["git"]["submodules"].items():
        if sub["state"] != "ok":
            warnings.append(f"submodule {path} is {sub['state']}: checked out {sub['checked_out'][:12]}, "
                            f"recorded {str(sub['recorded'])[:12]}")
    build = prov.get("engine", {}).get("build") or {}
    engine = prov["git"]["repos"].get("engine/cppanatron")
    if build and engine and (build.get("git_sha") != engine["head"] or bool(build.get("git_dirty")) != engine["dirty"]):
        warnings.append(f"the cppanatron build ({str(build.get('git_sha'))[:12]}, dirty={build.get('git_dirty')}) "
                        f"is not the checkout ({engine['head'][:12]}, dirty={engine['dirty']}): rebuild it")
    if strict and any(w.startswith("submodule") for w in warnings):
        raise ProvenanceError("; ".join(w for w in warnings if w.startswith("submodule")))
    return warnings


def resume_diff(first: dict, now: dict) -> list[str]:
    """What a resume would change: training code components and uv.lock."""
    out = []
    a, b = first.get("code", {}), now.get("code", {})
    for name in sorted(set(a) | set(b)):
        if a.get(name) != b.get(name):
            out.append(f"code {name}: {a.get(name)} -> {b.get(name)}")
    la, lb = first.get("python", {}).get("uv_lock_sha256"), now.get("python", {}).get("uv_lock_sha256")
    if la != lb:
        out.append(f"uv.lock: {str(la)[:12]} -> {str(lb)[:12]}")
    return out
