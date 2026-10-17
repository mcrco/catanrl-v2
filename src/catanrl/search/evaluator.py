"""Batched network inference for search leaves and policy decisions.

:class:`BatchEvaluator` maps float32 rows to (logits, per-seat values) on the
model's device:

* ``strict`` (evaluation): every forward has exactly ``infer_rows`` rows,
  padded with copies of the chunk's first row and chunked above that, so a
  row's outputs never depend on what else is in the batch, on the batch size
  or on the pool size. With CUDA graphs that is one captured graph.
* ``fast``: chunks are padded up to the next power of two (at most
  ``infer_rows``) and each size replays its own CUDA graph, captured on first
  use. Graph-s is launch-bound at small batches (about 200 kernels per
  forward), so graphs cut most of the per-call overhead.

On CPU there are no graphs, and strict mode evaluates rows one at a time:
CPU GEMM kernels make a row's result depend (in the last bits) on its
position in the batch, which CUDA's do not.

A recurrent model (``model.recurrent``) is evaluated through :meth:`step`
with each row's memory ``(h, c)``, padded and chunked like the rows; the
stateless call runs it at the zero state.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from catanrl.models import CatanModel

Heads = Literal["both", "policy"]


class BatchEvaluator:
    def __init__(
        self,
        model: CatanModel,
        *,
        device: str | torch.device,
        infer_rows: int = 256,
        mode: Literal["strict", "fast"] = "strict",
        cuda_graphs: bool = True,
        heads: Heads = "both",
    ) -> None:
        if infer_rows < 1:
            raise ValueError("infer_rows must be positive")
        if mode not in ("strict", "fast"):
            raise ValueError(f"mode must be strict or fast, not {mode!r}")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.infer_rows = int(infer_rows)
        self.mode = mode
        self.heads = heads
        self.obs_size = int(model.obs_size)
        self.num_actions = int(model.num_actions)
        self.num_players = int(model.num_players)
        self.use_graphs = bool(cuda_graphs) and self.device.type == "cuda"
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = {}
        self.recurrent = bool(getattr(model, "recurrent", False))
        self.hidden_size = int(getattr(model, "hidden_size", 0)) if self.recurrent else 0
        self.rows = 0
        self.forwards = 0

    # ------------------------------------------------------------ shapes

    def bucket(self, rows: int) -> int:
        if self.mode == "strict" and self.device.type == "cpu":
            return 1
        if self.mode == "strict":
            return self.infer_rows
        size = 1
        while size < rows:
            size *= 2
        return min(size, self.infer_rows)

    def _forward(self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
                 ) -> tuple[torch.Tensor, ...]:
        """(logits, values) or, with a memory, (logits, values, h', c')."""
        if state is not None:
            logits, value_logits, present, h, c = self.model.step(obs, *state)
            return logits, torch.sigmoid(value_logits) * present, h, c
        if self.heads == "policy":
            logits = self.model.policy_logits(obs)
            return logits, logits.new_zeros(obs.shape[0], self.num_players)
        return self.model(obs)

    def _graph(self, size: int):
        if size not in self._graphs:
            static: tuple[torch.Tensor, ...] = (torch.zeros(size, self.obs_size, device=self.device),)
            if self.recurrent:
                static += tuple(self.model.initial_state(size, self.device))

            def forward():
                return self._forward(static[0], static[1:] if self.recurrent else None)

            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(2):
                    forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph):
                outputs = forward()
            self._graphs[size] = (graph, static, outputs)
        return self._graphs[size]

    # ------------------------------------------------------------ evaluation

    def __call__(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(logits float32 [n, A], values float32 [n, P]) for rows [n, obs_size]."""
        observations = np.ascontiguousarray(observations, dtype=np.float32)
        n = observations.shape[0]
        if observations.ndim != 2 or observations.shape[1] != self.obs_size:
            raise ValueError(f"expected rows [n, {self.obs_size}], got {observations.shape}")
        if torch.is_autocast_enabled() or (self.device.type == "cuda" and torch.is_autocast_enabled("cuda")):
            raise RuntimeError("evaluate with autocast off (PuffeRL leaks bf16 autocast)")
        logits, values, _ = self._evaluate(observations, None)
        return logits, values

    def step(self, observations: np.ndarray, h: np.ndarray, c: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A recurrent model's decision per row from its memory: (logits, values, h', c')."""
        if not self.recurrent:
            raise ValueError("this network has no recurrent state")
        observations = np.ascontiguousarray(observations, dtype=np.float32)
        memory = tuple(np.ascontiguousarray(m, dtype=np.float32) for m in (h, c))
        if any(m.shape != (observations.shape[0], self.hidden_size) for m in memory):
            raise ValueError(f"expected memories [n, {self.hidden_size}] for {observations.shape[0]} rows")
        logits, values, (h, c) = self._evaluate(observations, memory)
        return logits, values, h, c

    def _evaluate(self, observations: np.ndarray, memory: tuple[np.ndarray, ...] | None):
        n = observations.shape[0]
        if observations.ndim != 2 or observations.shape[1] != self.obs_size:
            raise ValueError(f"expected rows [n, {self.obs_size}], got {observations.shape}")
        if torch.is_autocast_enabled() or (self.device.type == "cuda" and torch.is_autocast_enabled("cuda")):
            raise RuntimeError("evaluate with autocast off (PuffeRL leaks bf16 autocast)")
        logits = np.empty((n, self.num_actions), dtype=np.float32)
        values = np.empty((n, self.num_players), dtype=np.float32)
        out_memory = None if memory is None else tuple(np.empty_like(m) for m in memory)
        step = 1 if self.mode == "strict" and self.device.type == "cpu" else self.infer_rows
        for start in range(0, n, step):
            end = start + step
            chunk = observations[start:end]
            outputs = self._run(chunk, None if memory is None else tuple(m[start:end] for m in memory))
            logits[start : start + len(chunk)] = outputs[0]
            values[start : start + len(chunk)] = outputs[1]
            if out_memory is not None:
                for target, out in zip(out_memory, outputs[2:]):
                    target[start : start + len(chunk)] = out
        return logits, values, out_memory

    def _run(self, chunk: np.ndarray, memory: tuple[np.ndarray, ...] | None = None) -> tuple[np.ndarray, ...]:
        k = len(chunk)
        size = self.bucket(k)
        sources = (torch.from_numpy(chunk), *(torch.from_numpy(m) for m in memory or ()))
        self.rows += k
        self.forwards += 1
        with torch.inference_mode():
            if self.use_graphs:
                graph, static, outputs = self._graph(size)
                for target, source in zip(static, sources):
                    target[:k].copy_(source, non_blocking=False)
                    if size > k:
                        target[k:].copy_(target[:1].expand(size - k, -1))
                graph.replay()
            else:
                batch = []
                for source in sources:
                    tensor = source.to(self.device)
                    if size > k:
                        tensor = torch.cat((tensor, tensor[:1].expand(size - k, -1)), dim=0)
                    batch.append(tensor)
                outputs = self._forward(batch[0], tuple(batch[1:]) if memory is not None else None)
            return tuple(out[:k].float().cpu().numpy() for out in outputs)
