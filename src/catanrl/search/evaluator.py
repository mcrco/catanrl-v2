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
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = {}
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

    def _forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.heads == "policy":
            logits = self.model.policy_logits(obs)
            return logits, logits.new_zeros(obs.shape[0], self.num_players)
        return self.model(obs)

    def _graph(self, size: int):
        if size not in self._graphs:
            static = torch.zeros(size, self.obs_size, device=self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(2):
                    self._forward(static)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph):
                outputs = self._forward(static)
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
        logits = np.empty((n, self.num_actions), dtype=np.float32)
        values = np.empty((n, self.num_players), dtype=np.float32)
        step = 1 if self.mode == "strict" and self.device.type == "cpu" else self.infer_rows
        for start in range(0, n, step):
            chunk = observations[start : start + step]
            out_logits, out_values = self._run(chunk)
            logits[start : start + len(chunk)] = out_logits
            values[start : start + len(chunk)] = out_values
        return logits, values

    def _run(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = len(chunk)
        size = self.bucket(k)
        source = torch.from_numpy(chunk)
        self.rows += k
        self.forwards += 1
        with torch.inference_mode():
            if self.use_graphs:
                graph, static, (logits, values) = self._graph(size)
                static[:k].copy_(source, non_blocking=False)
                if size > k:
                    static[k:].copy_(static[:1].expand(size - k, -1))
                graph.replay()
            else:
                batch = source.to(self.device)
                if size > k:
                    batch = torch.cat((batch, batch[:1].expand(size - k, -1)), dim=0)
                logits, values = self._forward(batch)
            return logits[:k].float().cpu().numpy(), values[:k].float().cpu().numpy()
