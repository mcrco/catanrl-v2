"""The Canopy bridge processes: JSON lines over stdin/stdout.

One :class:`BridgeProcess` runs ``canopy-bridge catanatron-bridge`` (single-threaded, CPU)
and keeps a saved search per game id; a request searches all its states together (one
network batch per simulation round). :class:`BridgePool` runs ``procs`` of them and sends each
game to one process for its whole life (the agent assigns games by stream), writing every
request before reading any answer, so the processes search in parallel.

Protocol (the bridge's, ``rust.md`` section 3)::

    -> {"id", "active_game_ids": [...], "states": [snapshot, ...], "simulations": N, "details": true}
    <- {"id", "actions": [canopy id | null, ...], "elapsed_ms", "nn_batches", "nn_states", "details": [...]}
    <- {"id", "error"}                     # the whole request rejected

A process that dies, answers out of turn, or rejects a request raises
:class:`BridgeError`: a rejected snapshot is a serializer bug, never something to play around.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
import time
from collections.abc import Sequence
from typing import Any

from catanrl.canopy.release import Release
from catanrl.canopy.topology import topology_answer


class BridgeError(RuntimeError):
    pass


class BridgeProcess:
    """One bridge subprocess."""

    def __init__(self, release: Release, *, sims: int, c_puct: float, c_visit: float, c_scale: float,
                 dirichlet_alpha: float, dirichlet_epsilon: float, reuse: bool, timeout: float = 3600.0) -> None:
        self.timeout = float(timeout)
        self.command = [str(release.binary), "catanatron-bridge", "--checkpoint", str(release.checkpoint),
                        "--simulations", str(int(sims)), "--seed", "0", "--c-puct", repr(float(c_puct)),
                        "--c-visit", repr(float(c_visit)), "--c-scale", repr(float(c_scale)),
                        "--dirichlet-alpha", repr(float(dirichlet_alpha)),
                        "--dirichlet-epsilon", repr(float(dirichlet_epsilon))]
        if not reuse:
            self.command.append("--no-tree-reuse")
        self._stderr = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 (lives as long as the process; close())
        self._process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self._stderr, bufsize=0)
        self._buffer = b""
        self._next_id = 0
        self._pending: int | None = None
        answer = self.request({"topology": True})
        answer.pop("id", None)
        if answer != topology_answer():
            self.close()
            raise BridgeError("the bridge's board is not the one catanrl.canopy.topology pins")

    @property
    def pid(self) -> int:
        return self._process.pid

    def send(self, payload: dict[str, Any]) -> int:
        if self._pending is not None:
            raise BridgeError("a request is already in flight")
        self._check_alive()
        request_id = self._next_id
        self._next_id += 1
        line = json.dumps({"id": request_id, **payload}, separators=(",", ":")).encode() + b"\n"
        try:
            self._process.stdin.write(line)
            self._process.stdin.flush()
        except BrokenPipeError:
            raise BridgeError(f"the bridge (pid {self.pid}) closed its input: {self._tail()}") from None
        self._pending = request_id
        return request_id

    def receive(self) -> dict[str, Any]:
        if self._pending is None:
            raise BridgeError("no request in flight")
        request_id, self._pending = self._pending, None
        line = self._readline()
        try:
            answer = json.loads(line)
        except json.JSONDecodeError:
            raise BridgeError(f"the bridge answered {line[:200]!r}") from None
        if answer.get("id") != request_id:
            raise BridgeError(f"the bridge answered request {answer.get('id')} to request {request_id}: {answer}")
        if "error" in answer:
            raise BridgeError(f"the bridge rejected request {request_id}: {answer['error']}")
        return answer

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.send(payload)
        return self.receive()

    def _readline(self) -> bytes:
        deadline = time.monotonic() + self.timeout
        out = self._process.stdout
        while b"\n" not in self._buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                self.close()
                raise BridgeError(f"the bridge (pid {self.pid}) did not answer within {self.timeout:.0f} s")
            ready, _, _ = select.select([out], [], [], min(left, 5.0))
            if not ready:
                self._check_alive()
                continue
            chunk = os.read(out.fileno(), 1 << 20)
            if not chunk:
                raise BridgeError(f"the bridge (pid {self.pid}) closed its output: {self._tail()}")
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line

    def _check_alive(self) -> None:
        code = self._process.poll()
        if code is not None:
            raise BridgeError(f"the bridge (pid {self.pid}) exited with code {code}: {self._tail()}")

    def _tail(self) -> str:
        try:
            self._stderr.seek(0)
            return self._stderr.read()[-2000:].decode(errors="replace").strip() or "(no stderr)"
        except (OSError, ValueError):
            return "(stderr unavailable)"

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        stderr = getattr(self, "_stderr", None)
        if stderr is not None:
            stderr.close()
            self._stderr = None


class BridgePool:
    """``procs`` bridge processes; a game stays with one."""

    def __init__(self, release: Release, *, procs: int, **options: Any) -> None:
        if procs < 1:
            raise ValueError("procs must be positive")
        self.processes: list[BridgeProcess] = []
        try:
            for _ in range(int(procs)):
                self.processes.append(BridgeProcess(release, **options))
        except BaseException:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self.processes)

    def decide(self, batches: Sequence[Sequence[dict]], active: Sequence[Sequence[str]], *,
               sims: int, legal: Sequence[Sequence[Sequence[int]]] | None = None) -> list[dict | None]:
        """One request per process with states: its answer, or None for a process without.

        ``legal`` (the engine's Canopy ids per state) is for stand-in pools in tests, which choose
        from it; the bridge applies Canopy's own rules and is never sent it."""
        sent = []
        for process, states, live in zip(self.processes, batches, active):
            if states:
                process.send({"active_game_ids": list(live), "states": list(states), "simulations": int(sims),
                              "details": True})
                sent.append(process)
            else:
                sent.append(None)
        answers = []
        error = None
        for process in sent:  # read every answer, so no process is left with one in flight
            if process is None:
                answers.append(None)
                continue
            try:
                answers.append(process.receive())
            except BridgeError as exc:
                error = error or exc
                answers.append(None)
        if error is not None:
            raise error
        return answers

    def close(self) -> None:
        for process in self.processes:
            process.close()
        self.processes = []
