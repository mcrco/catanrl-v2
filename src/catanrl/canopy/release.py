"""The released Canopy build a CANOPY agent runs, pinned by hash.

A release directory (``$CATANRL_CANOPY_DIR``, else ``<runs root>/_canopy``) holds

* ``canopy-bridge``: the bridge binary, cullback/canopy's ``catan`` example built at
  :data:`CANOPY_COMMIT` (tag ``catan-nexus-v3``) with :data:`PATCHES` applied;
* ``canopy-bridge.manifest.json``: what it was built from (``write_manifest``);
* ``model_iter_315.mpk``: the released checkpoint, sha256 :data:`CHECKPOINT_SHA256`.

:func:`load_release` refuses a directory whose checkpoint, commit or patches are not the
pinned ones, or whose binary is not the one its manifest names. The agent's key carries the
binary's and the checkpoint's sha256, the commit and every patch's sha256, so a record names
exactly what played.

``python -m catanrl.canopy.release SRC PATCH_DIR BINARY OUT_DIR CHECKPOINT`` writes a manifest
for a build: it checks that ``SRC``'s HEAD tree is :data:`CANOPY_COMMIT` with the patches
applied in order (through a scratch index, nothing in the checkout changes), that ``BINARY`` is
``SRC``'s built example, and copies the binary, checkpoint and manifest into ``OUT_DIR``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from catanrl.canon import sha256_file

#: cullback/canopy at tag catan-nexus-v3.
CANOPY_COMMIT = "6185983a88ba6802e7fa9893cef5a76a15de2595"
#: Patches applied in this order: the released search did not reset its budget on tree reuse
#: (v1's pin), v1's Catanatron bridge (v1 9c66130), and the v2 benchmark extensions (hidden
#: opponent cards, per-snapshot rules and seed, raw policy, exact tree reuse, sanity checks).
PATCHES = (
    ("canopy-nexus-v3-reset-search-budget.patch", "506fb91bff6bbedb764031e8d252dd4cf6ef92ce91d1d22c9bbe353bcc4e3f67"),
    ("canopy-catanatron-bridge.patch", "6eba3e9204ee0340febb931957b6cbcc62c1dadb8c8460562cbae54715c444ac"),
    ("canopy-v2-bench-ext.patch", "60fbc1776e25ce8d6231b7fd7b905cd4e05b69fc7481c5baf24a76737300dec0"),
)
#: model_iter_315.mpk of release catan-nexus-v3 (v1's CANOPY_NEXUS_V3_CHECKPOINT_SHA256).
CHECKPOINT_NAME = "model_iter_315.mpk"
CHECKPOINT_SHA256 = "f8e4e6858930243a30243e38c1b2b96b1a8da23970f5cba69906c65b268c60cc"
BINARY_NAME = "canopy-bridge"
MANIFEST_NAME = "canopy-bridge.manifest.json"
MANIFEST_FORMAT = "catanrl.canopy_bridge_manifest/1"


class ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class Release:
    """A verified release directory."""

    directory: Path
    binary: Path
    checkpoint: Path
    binary_sha256: str
    manifest: dict

    def key(self) -> dict:
        """What of the release decides moves: the source, the patches, the binary, the weights."""
        return {"commit": CANOPY_COMMIT, "patches": [list(p) for p in PATCHES],
                "binary_sha256": self.binary_sha256, "checkpoint_sha256": CHECKPOINT_SHA256,
                "backend": self.manifest.get("backend")}


def release_dir() -> Path:
    env = os.environ.get("CATANRL_CANOPY_DIR")
    if env:
        return Path(env).expanduser().resolve()
    from catanrl.runs import runs_root

    return runs_root() / "_canopy"


_VERIFIED: dict[tuple, Release] = {}


def load_release(directory: str | Path | None = None) -> Release:
    """The release in ``directory`` (default :func:`release_dir`), verified against the pins."""
    directory = Path(directory) if directory is not None else release_dir()
    binary, checkpoint, manifest_path = (directory / BINARY_NAME, directory / CHECKPOINT_NAME,
                                         directory / MANIFEST_NAME)
    missing = [p.name for p in (binary, checkpoint, manifest_path) if not p.is_file()]
    if missing:
        raise ReleaseError(f"{directory} is not a Canopy release directory (missing {', '.join(missing)}); "
                           "set CATANRL_CANOPY_DIR (see catanrl.canopy.release)")
    stamp = tuple((p, p.stat().st_mtime_ns, p.stat().st_size) for p in (binary, checkpoint, manifest_path))
    if stamp in _VERIFIED:
        return _VERIFIED[stamp]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ReleaseError(f"{manifest_path}: not a {MANIFEST_FORMAT} manifest")
    if manifest.get("commit") != CANOPY_COMMIT:
        raise ReleaseError(f"{manifest_path}: built from {manifest.get('commit')}, not {CANOPY_COMMIT}")
    if [tuple(p) for p in manifest.get("patches", [])] != list(PATCHES):
        raise ReleaseError(f"{manifest_path}: patches {manifest.get('patches')} are not the pinned {list(PATCHES)}")
    binary_sha = sha256_file(binary)
    if manifest.get("binary_sha256") != binary_sha:
        raise ReleaseError(f"{binary} (sha256 {binary_sha[:12]}) is not the binary its manifest names "
                           f"({str(manifest.get('binary_sha256'))[:12]})")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise ReleaseError(f"{checkpoint} has sha256 {checkpoint_sha[:12]}, not the released {CHECKPOINT_SHA256[:12]}")
    release = Release(directory=directory, binary=binary, checkpoint=checkpoint, binary_sha256=binary_sha,
                      manifest=manifest)
    _VERIFIED[stamp] = release
    return release


# ---------------------------------------------------------------- writing a manifest


def _git(src: Path, *args: str, env: dict | None = None) -> str:
    out = subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True, env=env, check=False)
    if out.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} in {src}: {out.stderr.strip()}")
    return out.stdout.strip()


def patched_tree(src: Path, patch_dir: Path) -> str:
    """The tree id of :data:`CANOPY_COMMIT` with :data:`PATCHES` applied in order, built in a
    scratch index (the checkout and its index are untouched; git stores the objects)."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git(src, "read-tree", CANOPY_COMMIT, env=env)
        for name, _ in PATCHES:
            _git(src, "apply", "--cached", str(patch_dir / name), env=env)
        return _git(src, "write-tree", env=env)


def write_manifest(src: str | Path, patch_dir: str | Path, binary: str | Path, out_dir: str | Path,
                   checkpoint: str | Path, *, backend: str = "cpu-ndarray") -> Path:
    """Verify a build and install it (binary, checkpoint, manifest) into ``out_dir``."""
    src, patch_dir, binary, out_dir = Path(src), Path(patch_dir), Path(binary), Path(out_dir)
    for name, sha in PATCHES:
        if sha256_file(patch_dir / name) != sha:
            raise ReleaseError(f"{patch_dir / name} is not the pinned patch (sha256 {sha[:12]})")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ReleaseError(f"{checkpoint} is not the released checkpoint")
    head_tree = _git(src, "rev-parse", "HEAD^{tree}")
    if patched_tree(src, patch_dir) != head_tree:
        raise ReleaseError(f"{src} HEAD is not {CANOPY_COMMIT[:12]} with the pinned patches applied")
    if _git(src, "status", "--porcelain", "--untracked-files=no"):
        raise ReleaseError(f"{src} has uncommitted changes")
    built = src / "target" / "release" / "examples" / "catan"
    if not built.is_file() or sha256_file(built) != sha256_file(binary):
        raise ReleaseError(f"{binary} is not {built} (build the example first)")
    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=False).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, out_dir / BINARY_NAME)
    if Path(checkpoint).resolve() != (out_dir / CHECKPOINT_NAME).resolve():
        shutil.copy2(checkpoint, out_dir / CHECKPOINT_NAME)
    manifest = {"format": MANIFEST_FORMAT, "commit": CANOPY_COMMIT, "patches": [list(p) for p in PATCHES],
                "tree": head_tree, "source_head": _git(src, "rev-parse", "HEAD"),
                "binary_sha256": sha256_file(out_dir / BINARY_NAME), "checkpoint_sha256": CHECKPOINT_SHA256,
                "example": "catan", "profile": "release", "backend": backend, "rustc": rustc}
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: python -m catanrl.canopy.release SRC PATCH_DIR BINARY OUT_DIR CHECKPOINT")
    print(write_manifest(*sys.argv[1:]))
