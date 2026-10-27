"""Per-move wall time of an agent spec at batch 1, the way the OpenHex and house bridges play.

Plays one 1v1 game (15 VP, discard above 9) against F through the Catanatron harness and
times every decision with more than one legal action, so a search spec can be checked
against OpenHex's 60 s per-move limit on the machine that will serve it::

    .venv/bin/python tools/move_time.py "MCTS:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:sims=65536:dets=1" --moves 20
"""

from __future__ import annotations

import argparse
import statistics
import time

from catanatron import Color, Game

from catanrl.agents.spec import player

LIMIT_S = 60.0  # OpenHex's per-move limit


class _Done(Exception):
    """Enough moves timed."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec")
    parser.add_argument("--moves", type=int, default=20, help="stop after this many timed moves")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    me = player(args.spec, Color.RED)
    me.preflight()  # load the checkpoint before the clock starts
    times: list[float] = []
    decide = me.decide

    def timed(game, actions):
        start = time.perf_counter()
        action = decide(game, actions)
        if len(actions) > 1:
            times.append(time.perf_counter() - start)
            print(f"move {len(times)}: {times[-1]:.2f} s ({len(actions)} options)", flush=True)
            if len(times) >= args.moves:
                raise _Done
        return action

    me.decide = timed
    try:
        Game([me, player("F", Color.BLUE)], seed=args.seed, discard_limit=9, vps_to_win=15).play()
    except _Done:
        pass
    if not times:
        raise SystemExit("no timed move")
    print(f"{args.spec}\n  {len(times)} moves: median {statistics.median(times):.2f} s, "
          f"mean {statistics.mean(times):.2f} s, max {max(times):.2f} s; "
          f"{'within' if max(times) < LIMIT_S else 'OVER'} OpenHex's {LIMIT_S:.0f} s limit")


if __name__ == "__main__":
    main()
