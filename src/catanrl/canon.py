"""Canonical JSON, digests and the run store's file writes.

Every hash in catanrl (config hashes, eval ids, weight hashes, code ids)
goes through this module, and every record is written through it:

* :func:`canonical_json` is the one byte form hashes are taken over;
* :func:`weights_sha256` hashes a state dict by (key, dtype, shape, bytes),
  so re-saving or reordering keys leaves it unchanged;
* :func:`ast_digest` hashes Python by its AST without docstrings, so
  comments and formatting change nothing;
* writes are atomic (tmp + fsync + ``os.replace``), write-once records refuse
  different content, and JSONL appends are single ``write`` calls whose
  readers drop a torn last line.
"""

from __future__ import annotations

import ast
import enum
import fcntl
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ImmutableRecordError(RuntimeError):
    """A write-once record exists with different content."""


def _plain(obj: Any) -> Any:
    """``obj`` as JSON types: tuples and sets become lists, numpy scalars Python numbers."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                if isinstance(key, (int, float, bool)) or key is None:
                    key = json.dumps(key)
                else:
                    raise TypeError(f"canonical JSON keys must be strings, not {type(key).__name__}")
            out[key] = _plain(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_plain(x) for x in obj), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(obj, enum.Enum):
        return _plain(obj.value)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    item = getattr(obj, "item", None)  # numpy and torch scalars
    if item is not None and getattr(obj, "shape", None) == ():
        return _plain(item())
    tolist = getattr(obj, "tolist", None)  # numpy arrays
    if tolist is not None:
        return _plain(tolist())
    raise TypeError(f"{type(obj).__name__} is not canonical JSON")


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, no spaces, ``repr`` floats: the bytes every hash is over."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=True).encode()


def digest(obj: Any, n: int = 16) -> str:
    """The first ``n`` hex digits of the sha256 of ``canonical_json(obj)``."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()[:n]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_sha256(state_dict: Mapping[str, Any]) -> str:
    """sha256 over sorted (key, dtype, shape, bytes) of a state dict."""
    import torch

    h = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state dict entry {key!r} is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        h.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip


def ast_text(source: str | bytes, name: str = "<source>") -> str:
    """``ast.dump`` of the source with docstrings removed (no line numbers)."""
    tree = _StripDocstrings().visit(ast.parse(source, filename=name))
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def ast_digest(paths: Iterable[Path | tuple[str, Path]]) -> str:
    """sha256 over the AST of each file, keyed by its name.

    ``paths`` are files (keyed by their name) or ``(label, path)`` pairs, so a
    package hashes the same wherever it is checked out.
    """
    entries = []
    for entry in paths:
        label, path = (entry if isinstance(entry, tuple) else (Path(entry).name, entry))
        entries.append((str(label), Path(path)))
    h = hashlib.sha256()
    for label, path in sorted(entries):
        h.update(f"{label}\0".encode())
        h.update(ast_text(path.read_bytes(), str(path)).encode())
        h.update(b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------- file writes


def _json_text(obj: Any) -> str:
    return json.dumps(_plain(obj), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=True) + "\n"


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_tmp(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return tmp


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    tmp = _write_tmp(path, data)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, _json_text(obj).encode())


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_once_json(path: str | Path, obj: Any) -> None:
    """Create ``path`` with ``obj``; equal content is a no-op, different content raises."""
    path = Path(path)
    data = _json_text(obj).encode()
    if not path.exists():
        tmp = _write_tmp(path, data)
        try:
            os.link(tmp, path)  # fails if another writer got there first
            _fsync_dir(path.parent)
            return
        except FileExistsError:
            pass
        finally:
            tmp.unlink(missing_ok=True)
    if canonical_json(read_json(path)) != canonical_json(json.loads(data)):
        raise ImmutableRecordError(f"{path} exists with different content")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one row under an flock. A torn tail (a crash mid-append) is cut
    back to the last newline first, so it never becomes an interior line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(_plain(dict(row)), sort_keys=True, ensure_ascii=False, allow_nan=True) + "\n").encode()
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            data = os.pread(fd, size, 0)
            os.ftruncate(fd, data.rfind(b"\n") + 1)
        os.lseek(fd, 0, os.SEEK_END)
        view = memoryview(line)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)  # releases the flock


def read_jsonl(path: str | Path) -> list[dict]:
    """Every row; a torn last line (a crash mid-append) is dropped."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().split(b"\n")
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:  # no trailing newline: torn
                break
            raise
    return rows
