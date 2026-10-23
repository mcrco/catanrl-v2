#!/usr/bin/env bash
# Build the Canopy bridge the CANOPY agent plays (src/catanrl/canopy) and install it as a release
# directory: canopy-bridge, canopy-bridge.manifest.json and model_iter_315.mpk.
#
#   third_party/canopy/build_bridge.sh [--backend cpu|cuda] SRC OUT CHECKPOINT
#
#   --backend   the network backend (default cpu):
#                 cpu   burn's ndarray backend, one thread per process (manifest backend
#                       "cpu-ndarray"; pinned by canopy-bridge.manifest.json here)
#                 cuda  burn's CUDA backend (cubecl), `--features cuda` (manifest backend "cuda";
#                       pinned by canopy-bridge-cuda.manifest.json here)
#   SRC         where the canopy source is built: a new directory (cloned from $CANOPY_GIT,
#               default github.com/cullback/canopy, checked out at tag catan-nexus-v3 and the
#               patches below committed on top), or such a clone made by this script before
#   OUT         the release directory to install into (CATANRL_CANOPY_DIR, or <runs>/_canopy)
#   CHECKPOINT  model_iter_315.mpk of release catan-nexus-v3 (sha256 f8e4e685...68c60cc)
#
# The patches, in order (their sha256s are pinned in src/catanrl/canopy/release.py PATCHES; the
# same for every backend):
#   canopy-nexus-v3-reset-search-budget.patch  v1's pin: a reused tree gets a full, fresh budget
#   canopy-catanatron-bridge.patch             v1's JSON-lines bridge (v1 9c66130)
#   canopy-v2-bench-ext.patch                  v2: hidden opponent cards, rules/seed per snapshot,
#                                              raw policy, exact tree reuse, sanity checks
#   canopy-v2-reproducible-build.patch         no build path in the binary (only the web UI used one)
#
# The build is reproducible: source, cargo and toolchain paths are remapped (--remap-path-prefix)
# and the last patch drops the one env!("CARGO_MANIFEST_DIR"), so the binary does not depend on
# where SRC is, and Cargo.lock is used as committed (--locked). With the
# pinned rustc it gives the binary_sha256 in the backend's committed manifest next to this script,
# so a lost release directory can be rebuilt under the same record keys. A different rustc gives a
# different binary and so different record keys: the script refuses one unless CANOPY_ANY_RUSTC=1.
# One SRC builds one backend at a time (cargo's target/release/examples/catan is the last build).
#
# cuda is NOT bit-reproducible: cubecl's #[cube] proc macro iterates a HashMap when it expands
# kernels (cubecl-macros 0.9.0 src/parse/kernel.rs:104), so the expansion order, and with it the
# codegen of cubek-*, burn-cubecl, burn-cuda, burn and canopy, changes from one compiler run to the
# next (two builds of the same tree: 1f7cc246... and 533dffaf..., same size, 12262 bytes apart).
# The two played bit-identically (1609 raw positions, 100 positions at 1600 sims), but a record
# names its binary: keep the installed cuda release; a rebuild is a new player for the keys (this
# script then exits 3), unless its sha happens to match.
#
# cuda: cudarc's bindings are pinned to CUDA 13.1 (CUDARC_CUDA_VERSION=13010, the newest cudarc
# 0.18.2 knows; v1 built the same way), so the binary does not depend on the nvcc found at build
# time. Nothing CUDA is linked: at run time the bridge loads the driver's libcuda and the
# toolkit's NVRTC, which compiles burn's kernels for the GPU it finds (headers from $CUDA_PATH,
# else /usr/local/cuda or /opt/cuda). Those, the driver and the GPU are the execution, not the
# binary (the agent records them per eval segment).
#
# Nice 15, $JOBS (default 4) jobs. The agent's key records the manifest's backend.
set -euo pipefail
backend=cpu
if [[ ${1:-} == --backend ]]; then backend=${2:-}; shift 2 || true; fi
case $backend in
  cpu)  features=() manifest_backend=cpu-ndarray pinned_manifest=canopy-bridge.manifest.json ;;
  cuda) features=(--features cuda) manifest_backend=cuda pinned_manifest=canopy-bridge-cuda.manifest.json
        export CUDARC_CUDA_VERSION=13010 ;;
  *) echo "--backend is cpu or cuda, not '$backend'" >&2; exit 2 ;;
esac
[[ $# -eq 3 ]] || { sed -n '2,16p' "$0" >&2; exit 2; }
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
# No GPU is needed to build (CUDA is loaded at run time), so none is visible to cargo.
(cd "$src" && env CUDA_VISIBLE_DEVICES= nice -n 15 cargo build --release --locked -j"${JOBS:-4}" --example catan \
   ${features[@]+"${features[@]}"})

"$python" -c 'import sys; from catanrl.canopy.release import write_manifest
print(write_manifest(*sys.argv[1:6], backend=sys.argv[6]))' \
  "$src" "$here/patches" "$src/target/release/examples/catan" "$out" "$checkpoint" "$manifest_backend"
built=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["binary_sha256"])' "$out/canopy-bridge.manifest.json")
if [[ ! -f $here/$pinned_manifest ]]; then
  echo "installed in $out: binary $built (no committed manifest yet: copy $out/canopy-bridge.manifest.json to" \
       "$here/$pinned_manifest to pin it)"
  exit 0
fi
pinned=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["binary_sha256"])' "$here/$pinned_manifest")
if [[ $built == "$pinned" ]]; then
  echo "installed in $out: binary $built, the committed manifest's"
else
  echo "installed in $out: binary $built, NOT the committed manifest's $pinned (records name the binary: a" \
       "rebuild that differs is a different player for the record keys)" >&2
  exit 3
fi
