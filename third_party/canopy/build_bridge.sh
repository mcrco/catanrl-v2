#!/usr/bin/env bash
# Build the Canopy bridge the CANOPY agent plays (src/catanrl/canopy) and install it as a release
# directory: canopy-bridge, canopy-bridge.manifest.json and model_iter_315.mpk.
#
#   third_party/canopy/build_bridge.sh SRC OUT CHECKPOINT
#
#   SRC         where the canopy source is built: a new directory (cloned from $CANOPY_GIT,
#               default github.com/cullback/canopy, checked out at tag catan-nexus-v3 and the
#               three patches below committed on top), or such a clone made by this script before
#   OUT         the release directory to install into (CATANRL_CANOPY_DIR, or <runs>/_canopy)
#   CHECKPOINT  model_iter_315.mpk of release catan-nexus-v3 (sha256 f8e4e685...68c60cc)
#
# The patches, in order (their sha256s are pinned in src/catanrl/canopy/release.py PATCHES):
#   canopy-nexus-v3-reset-search-budget.patch  v1's pin: a reused tree gets a full, fresh budget
#   canopy-catanatron-bridge.patch             v1's JSON-lines bridge (v1 9c66130)
#   canopy-v2-bench-ext.patch                  v2: hidden opponent cards, rules/seed per snapshot,
#                                              raw policy, exact tree reuse, sanity checks
#   canopy-v2-reproducible-build.patch         no build path in the binary (only the web UI used one)
#
# The build is reproducible: source, cargo and toolchain paths are remapped (--remap-path-prefix)
# and the last patch drops the one env!("CARGO_MANIFEST_DIR"), so the binary does not depend on
# where SRC is, and Cargo.lock is used as committed (--locked). With the
# pinned rustc it gives the binary_sha256 in canopy-bridge.manifest.json next to this script, so a
# lost release directory can be rebuilt under the same record keys. A different rustc gives a
# different binary and so different record keys: the script refuses one unless CANOPY_ANY_RUSTC=1.
#
# CPU (ndarray) backend, nice 15, $JOBS (default 4) jobs. GPU builds (not built or tested): add
# `--features cuda` (v1 set CUDA_HOME and CUDARC_CUDA_VERSION) or `--features wgpu`; the agent's
# key records the manifest's backend.
set -euo pipefail
[[ $# -eq 3 ]] || { sed -n '2,13p' "$0" >&2; exit 2; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
src=$1 out=$2 checkpoint=$3
commit=6185983a88ba6802e7fa9893cef5a76a15de2595  # tag catan-nexus-v3
rustc_pinned="rustc 1.92.0 (ded5c06cf 2025-12-08)"
patches=(canopy-nexus-v3-reset-search-budget canopy-catanatron-bridge canopy-v2-bench-ext
         canopy-v2-reproducible-build)
python=${PYTHON:-$repo/.venv/bin/python}

rustc_now=$(rustc --version)
if [[ $rustc_now != "$rustc_pinned" && ${CANOPY_ANY_RUSTC:-0} != 1 ]]; then
  echo "rustc is '$rustc_now', not the pinned '$rustc_pinned' (rustup toolchain install 1.92.0);" \
       "a different compiler gives a different binary_sha256. CANOPY_ANY_RUSTC=1 builds anyway." >&2
  exit 1
fi

if [[ ! -e $src ]]; then
  git clone -q "${CANOPY_GIT:-https://github.com/cullback/canopy}" "$src"
  git -C "$src" -c advice.detachedHead=false checkout -q "$commit"
  for p in "${patches[@]}"; do git -C "$src" apply "$here/patches/$p.patch"; done
  git -C "$src" add -A
  git -C "$src" -c user.name=catanrl -c user.email=catanrl@localhost commit -qm \
    "catanrl CANOPY bridge: ${patches[*]}"
fi
src=$(cd "$src" && pwd -P)
cargo_home=$(cd "${CARGO_HOME:-$HOME/.cargo}" && pwd -P)
sysroot=$(rustc --print sysroot)
# The release module checks the tree again (catan-nexus-v3 plus exactly these patches) when it
# writes the manifest; checking first saves a build of the wrong source.
"$python" -c 'import sys; from pathlib import Path; from catanrl.canopy.release import patched_tree, _git
src = Path(sys.argv[1]); want = patched_tree(src, Path(sys.argv[2]))
sys.exit(0 if _git(src, "rev-parse", "HEAD^{tree}") == want else f"{src}: HEAD is not catan-nexus-v3 with the pinned patches")' \
  "$src" "$here/patches"

export RUSTFLAGS="--remap-path-prefix=$src=/canopy --remap-path-prefix=$cargo_home=/cargo --remap-path-prefix=$sysroot=/rust"
(cd "$src" && env CUDA_VISIBLE_DEVICES= nice -n 15 cargo build --release --locked -j"${JOBS:-4}" --example catan)

"$python" -m catanrl.canopy.release "$src" "$here/patches" "$src/target/release/examples/catan" "$out" "$checkpoint"
built=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["binary_sha256"])' "$out/canopy-bridge.manifest.json")
if [[ ! -f $here/canopy-bridge.manifest.json ]]; then
  echo "installed in $out: binary $built (no committed manifest yet: copy $out/canopy-bridge.manifest.json here to pin it)"
  exit 0
fi
pinned=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["binary_sha256"])' "$here/canopy-bridge.manifest.json")
if [[ $built == "$pinned" ]]; then
  echo "installed in $out: binary $built, the committed manifest's"
else
  echo "installed in $out: binary $built, NOT the committed manifest's $pinned (records name the binary: a" \
       "rebuild that differs is a different player for the record keys)" >&2
  exit 3
fi
