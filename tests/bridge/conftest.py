"""The bridge tests run tiny networks one row at a time: one torch thread is
fastest, and keeps them quick on a loaded machine."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _one_torch_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
