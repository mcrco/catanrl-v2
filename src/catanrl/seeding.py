"""Seeds, RNG state and determinism flags.

Every seed is ``derive_seed(root, *labels)``, bit-identical to v1
(DESIGN 6.8). :func:`determinism` sets torch's flags for ``strict`` work
(evals, small CPU training) and restores them afterwards.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
import sys
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import torch

MAX_SEED = sys.maxsize  # 2**63 - 1, v1's modulus


def derive_seed(root: int, *labels: str | int) -> int:
    """A stable child seed: sha256 of ``"root|label|..."``, first 8 bytes, mod 2**63 - 1."""
    for label in labels:
        if "|" in str(label):
            raise ValueError(f"seed label {label!r} contains '|'")
    payload = "|".join([str(root), *(str(label) for label in labels)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % MAX_SEED


def seed_everything(seed: int) -> None:
    """Seed Python, numpy and torch (every CUDA device included)."""
    random.seed(seed)
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))


def rng_state() -> dict[str, Any]:
    """Python, numpy, torch and CUDA RNG states, loadable with ``weights_only``."""
    version, internal, gauss = random.getstate()
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": [version, list(internal), gauss],
        "numpy": {"name": name, "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)), "pos": int(pos),
                  "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: dict[str, Any]) -> None:
    if state.get("python") is not None:
        version, internal, gauss = state["python"]
        random.setstate((version, tuple(internal), gauss))
    if state.get("numpy") is not None:
        n = state["numpy"]
        np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["pos"], n["has_gauss"],
                             n["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _autocast_on() -> bool:
    try:
        return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")
    except TypeError:  # older torch: no device argument
        return torch.is_autocast_enabled()


def _flags() -> dict[str, Any]:
    return {
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "precision": torch.get_float32_matmul_precision(),
    }


def _set_flags(flags: dict[str, Any]) -> None:
    torch.backends.cudnn.benchmark = flags["benchmark"]
    torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
    torch.use_deterministic_algorithms(flags["algorithms"], warn_only=flags["warn_only"])
    torch.backends.cuda.matmul.allow_tf32 = flags["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
    torch.set_float32_matmul_precision(flags["precision"])


@contextlib.contextmanager
def determinism(mode: Literal["strict", "fast"]) -> Iterator[None]:
    """Strict: no cudnn benchmark, deterministic algorithms, no TF32, autocast
    off (asserted) and ``CUBLAS_WORKSPACE_CONFIG`` set. Fast leaves the flags
    alone. Both restore the flags on exit."""
    if mode not in ("strict", "fast"):
        raise ValueError(f"determinism must be strict or fast, not {mode!r}")
    saved = _flags()
    try:
        if mode == "strict":
            if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
                raise RuntimeError("strict determinism needs CUBLAS_WORKSPACE_CONFIG (set by importing catanrl)")
            if _autocast_on():
                raise RuntimeError("strict determinism with autocast on (PuffeRL leaks bf16 autocast)")
            _set_flags({"benchmark": False, "cudnn_deterministic": True, "algorithms": True, "warn_only": False,
                        "matmul_tf32": False, "cudnn_tf32": False, "precision": "highest"})
        yield
    finally:
        _set_flags(saved)
