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
# The cpu build is reproducible: source, cargo and toolchain paths are remapped
# (--remap-path-prefix) and the last patch drops the one env!("CARGO_MANIFEST_DIR"), so the binary
# does not depend on where SRC is, and Cargo.lock is used as committed (--locked). With the
# pinned rustc it gives the binary_sha256 in the backend's committed manifest next to this script,
# so a lost release directory can be rebuilt under the same record keys. A different rustc gives a
# different binary and so different record keys: the script refuses one unless CANOPY_ANY_RUSTC=1.
# Nothing outside the committed tree may steer cargo: the script refuses untracked files in SRC
# (ignored ones too) other than target/, a cargo config in SRC, its parents or CARGO_HOME, and
# cargo/rustc variables that override flags or the compiler (RUSTC, RUSTC_WRAPPER, CARGO_BUILD_*,
# CARGO_PROFILE_*, CARGO_TARGET_*, CARGO_ENCODED_RUSTFLAGS, ...). One SRC builds one backend at a
# time (cargo's target/release/examples/catan is the last build).
#
# cuda is NOT bit-reproducible: cubecl's #[cube] proc macro iterates a HashMap when it expands
# kernels (cubecl-macros 0.9.0 src/parse/kernel.rs:104), so the expansion order, and with it the
# codegen of cubek-*, burn-cubecl, burn-cuda, burn and canopy, changes from one compiler run to the
# next (three builds of the same tree: 1f7cc246..., 533dffaf... and 9bfb7370..., sizes within
# ~600 bytes). They played bit-identically on the positions tried (the build agent: 1609 raw and
# 100 at 1600 sims; review: 1639 raw and 48 at 1600 sims), but a record names its binary: the
# installed cuda release is the release, and canopy-bridge-cuda.fingerprint.json (fingerprint.py
# here) is what it answers on fixed positions, so any build or copy can be checked for the same
# player. A rebuild is a new player for the keys unless its sha happens to match.
#
# Install: the built binary is checked against the backend's committed manifest BEFORE anything
# is copied. A binary that is not the pinned one is not installed (exit 3; the build stays in
# SRC/target) unless CANOPY_INSTALL_UNPINNED=1. An OUT that already holds a release with another
# binary is never overwritten (exit 4: the records key on it; install into a new directory). The
# files are staged in OUT/.install-* and renamed into place, manifest last.
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
# Nothing but the committed tree steers the build (write_manifest checks tracked files only).
stray=$(git -C "$src" ls-files --others --directory | grep -vx 'target/' || true)
if [[ -n $stray ]]; then
  echo "$src has files outside its commit (untracked or ignored; they could steer cargo):" $stray >&2
  exit 1
fi
d=$src
while :; do
  for c in "$d/.cargo/config" "$d/.cargo/config.toml"; do
    [[ -e $c ]] && { echo "cargo config $c would steer the build: move it away" >&2; exit 1; }
  done
  [[ $d == / ]] && break
  d=$(dirname "$d")
done
for c in "$cargo_home/config" "$cargo_home/config.toml"; do
  [[ -e $c ]] && { echo "cargo config $c would steer the build: move it away" >&2; exit 1; }
done
steer=$(compgen -e | grep -E '^(RUSTC|RUSTC_WRAPPER|RUSTC_WORKSPACE_WRAPPER|RUSTC_BOOTSTRAP|RUSTDOCFLAGS|CARGO_ENCODED_RUSTFLAGS|CARGO_BUILD_.*|CARGO_PROFILE_.*|CARGO_TARGET_.*|CARGO_UNSTABLE_.*|CARGO_INCREMENTAL)$' || true)
if [[ -n $steer ]]; then
  echo "unset these first, they would steer cargo or rustc:" $steer >&2
  exit 1
fi

export RUSTFLAGS="--remap-path-prefix=$src=/canopy --remap-path-prefix=$cargo_home=/cargo --remap-path-prefix=$sysroot=/rust"
# No GPU is needed to build (CUDA is loaded at run time), so none is visible to cargo.
(cd "$src" && env CUDA_VISIBLE_DEVICES= nice -n 15 cargo build --release --locked -j"${JOBS:-4}" --example catan \
   ${features[@]+"${features[@]}"})

sha() { "$python" -c 'import sys; from catanrl.canon import sha256_file; print(sha256_file(sys.argv[1]))' "$1"; }
manifest_sha() { "$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["binary_sha256"])' "$1"; }
built_binary=$src/target/release/examples/catan
built=$(sha "$built_binary")
pinned=
[[ -f $here/$pinned_manifest ]] && pinned=$(manifest_sha "$here/$pinned_manifest")
if [[ -n $pinned && $built != "$pinned" && ${CANOPY_INSTALL_UNPINNED:-0} != 1 ]]; then
  echo "built $built_binary: binary $built, NOT the committed manifest's $pinned. Nothing installed in $out:" \
       "records name the binary, so it is a different player for the record keys" \
       "(CANOPY_INSTALL_UNPINNED=1 installs it anyway; into a new directory)" >&2
  exit 3
fi
if [[ -e $out/canopy-bridge.manifest.json || -e $out/canopy-bridge ]]; then
  have=
  [[ -f $out/canopy-bridge.manifest.json ]] && have=$(manifest_sha "$out/canopy-bridge.manifest.json")
  [[ -z $have && -f $out/canopy-bridge ]] && have=$(sha "$out/canopy-bridge")
  if [[ $have != "$built" ]]; then
    echo "$out already holds a Canopy release with binary ${have:-?}; records key on it, so it is not replaced" \
         "by $built: install into a new directory" >&2
    exit 4
  fi
fi
mkdir -p "$out"
stage=$(mktemp -d "$out/.install-XXXXXX")
trap 'rm -rf "$stage"' EXIT
"$python" -c 'import sys; from catanrl.canopy.release import write_manifest
write_manifest(*sys.argv[1:6], backend=sys.argv[6])' \
  "$src" "$here/patches" "$built_binary" "$stage" "$checkpoint" "$manifest_backend"
for f in model_iter_315.mpk canopy-bridge canopy-bridge.manifest.json; do  # each rename is atomic; manifest last
  mv -f "$stage/$f" "$out/$f"
done
if [[ -z $pinned ]]; then
  echo "installed in $out: binary $built (no committed manifest yet: copy $out/canopy-bridge.manifest.json to" \
       "$here/$pinned_manifest to pin it)"
elif [[ $built == "$pinned" ]]; then
  echo "installed in $out: binary $built, the committed manifest's"
else
  echo "installed in $out: binary $built, NOT the committed manifest's $pinned (CANOPY_INSTALL_UNPINNED=1)" >&2
fi
