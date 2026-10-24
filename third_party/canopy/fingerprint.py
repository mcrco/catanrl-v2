"""What a Canopy release answers on fixed positions: its behavioural fingerprint.

The cuda build is not bit-reproducible (build_bridge.sh), so no rebuild can give the sha256 the
records name, and the manifest written at build time is the only tie from that binary to the
pinned source. This ties it to its play instead: it runs a release's bridge on the 64 positions
in fingerprint-states.jsonl.gz (snapshots a CANOPY seat sent its bridge in real raw-Canopy vs
POLICY games on heldout-2p, forced moves left out), raw (sims 0) on all 64 and at 1600 sims (the
bench's search: c_puct 2.5, c_visit 50, c_scale 1, dirichlet epsilon 0, no tree reuse) on the
first 16, each set in one request, and digests the answers (moves, policies, values, visits).

    python third_party/canopy/fingerprint.py RELEASE_DIR                 # print it
    python third_party/canopy/fingerprint.py RELEASE_DIR --write FILE    # save it
    python third_party/canopy/fingerprint.py RELEASE_DIR --check FILE    # compare with FILE

--check exits 0 when every answer is bit-identical (expected for the same build on the same GPU,
driver and NVRTC: the cuda bridge's answers depend on neither the batch nor the run), 2 when the
moves agree but the bits differ (another GPU, driver or NVRTC, or another build of the same source;
the build agent saw CPU vs GPU and two cuda builds agree on every move tried), 1 when a move
differs. canopy-bridge-cuda.fingerprint.json next to this script is the installed cuda release's
(binary 1f7cc246...68802689) on an RTX 2070 Super, driver 595.71.05, NVRTC 13.2.78.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from catanrl import canon
from catanrl.canopy.bridge import BridgeProcess
from catanrl.canopy.release import load_release

HERE = Path(__file__).resolve().parent
STATES = HERE / "fingerprint-states.jsonl.gz"
SEARCH = {"c_puct": 2.5, "c_visit": 50.0, "c_scale": 1.0, "dirichlet_alpha": 0.05, "dirichlet_epsilon": 0.0}
PARTS = (("raw", 0, 64), ("search", 1600, 16))
FORMAT = "catanrl.canopy_fingerprint/1"


def fingerprint(release_dir: str | Path) -> dict:
    release = load_release(release_dir)
    with gzip.open(STATES, "rt") as f:
        states = [json.loads(line) for line in f]
    out = {"format": FORMAT, "states": STATES.name, "states_sha256": canon.sha256_file(STATES),
           "binary_sha256": release.binary_sha256, "backend": release.manifest.get("backend"), "params": SEARCH}
    for name, sims, n in PARTS:
        proc = BridgeProcess(release, sims=sims, reuse=False, **SEARCH)
        try:
            start = time.time()
            answer = proc.request({"active_game_ids": [], "states": states[:n], "simulations": sims, "details": True})
            seconds = time.time() - start
        finally:
            proc.close()
        out[name] = {"sims": sims, "positions": n, "actions": answer["actions"],
                     "digest": canon.digest(answer["details"], 64)}
        print(f"{name}: {n} positions at {sims} sims in {seconds:.1f} s", file=sys.stderr)
    if out["backend"] != "cpu-ndarray":
        from catanrl.eval import results

        out["measured_on"] = {**results._cuda_runtime(), "nvrtc_library": results._nvrtc_library()}
    return out


def compare(found: dict, stored: dict) -> int:
    if found["states_sha256"] != stored["states_sha256"]:
        print("the fixed positions differ from the stored fingerprint's: no comparison")
        return 1
    status = 0
    for name, _, _ in PARTS:
        a, b = found[name], stored[name]
        moves = sum(x == y for x, y in zip(a["actions"], b["actions"]))
        same_bits = a["digest"] == b["digest"]
        print(f"{name}: moves {moves}/{len(b['actions'])} the same, answers {'bit-identical' if same_bits else 'differ in their bits'}")
        if moves != len(b["actions"]):
            status = 1
        elif not same_bits and status == 0:
            status = 2
    if found["binary_sha256"] != stored["binary_sha256"]:
        print(f"binary {found['binary_sha256'][:12]}, the fingerprint's {stored['binary_sha256'][:12]}: another build")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("release_dir")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    found = fingerprint(args.release_dir)
    if args.check:
        return compare(found, json.loads(args.check.read_text()))
    text = json.dumps(found, indent=1, sort_keys=True) + "\n"
    if args.write:
        args.write.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
