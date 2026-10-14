"""Test 1: canonical JSON, digests, weight hashes and the store's writes."""

from __future__ import annotations

import collections

import numpy as np
import pytest
import torch

from catanrl.canon import (
    ImmutableRecordError,
    append_jsonl,
    ast_digest,
    canonical_json,
    digest,
    read_jsonl,
    weights_sha256,
    write_once_json,
)


def test_canonical_json_is_stable_and_pinned():
    obj = {"b": (1, 2.5, np.int64(3)), "a": {"z": np.float32(0.5), "y": None}, "c": [True, "x"]}
    assert canonical_json(obj) == b'{"a":{"y":null,"z":0.5},"b":[1,2.5,3],"c":[true,"x"]}'
    reordered = {"c": [True, "x"], "a": {"y": None, "z": 0.5}, "b": [1, 2.5, 3]}
    assert canonical_json(reordered) == canonical_json(obj)
    assert digest(obj) == "865a89d5f504910b"
    assert digest(obj, 8) == digest(obj)[:8]
    assert canonical_json(3e-4) == b"0.0003"


def test_weights_sha256_survives_save_load_and_key_order(tmp_path):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    state = net.state_dict()
    first = weights_sha256(state)
    torch.save(state, tmp_path / "a.pt")
    loaded = torch.load(tmp_path / "a.pt", weights_only=True)
    torch.save(loaded, tmp_path / "b.pt")
    again = torch.load(tmp_path / "b.pt", weights_only=True)
    assert weights_sha256(loaded) == weights_sha256(again) == first
    assert weights_sha256(collections.OrderedDict(reversed(list(state.items())))) == first
    state["0.bias"] = state["0.bias"].clone()
    state["0.bias"][0] += 1e-6
    assert weights_sha256(state) != first
    assert weights_sha256({"x": torch.zeros(2, dtype=torch.bfloat16)})


def test_write_once(tmp_path):
    path = tmp_path / "r" / "rec.json"
    write_once_json(path, {"a": 1, "b": [1, 2]})
    write_once_json(path, {"b": (1, 2), "a": 1})  # equal: no-op
    with pytest.raises(ImmutableRecordError):
        write_once_json(path, {"a": 2})


def test_read_jsonl_drops_a_torn_last_line(tmp_path):
    path = tmp_path / "m.jsonl"
    append_jsonl(path, {"epoch": 1})
    append_jsonl(path, {"epoch": 2})
    with open(path, "ab") as f:
        f.write(b'{"epoch": 3, "lo')
    assert read_jsonl(path) == [{"epoch": 1}, {"epoch": 2}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []
    path.write_bytes(b'{"a": 1}\n{"bad\n{"a": 2}\n')
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_ast_digest_ignores_comments_and_docstrings(tmp_path):
    a = tmp_path / "a.py"
    a.write_text('"""Doc."""\nx = 1  # one\n\ndef f():\n    """Doc."""\n    return x\n')
    first = ast_digest([a])
    a.write_text("# header\nx = 1\n\n\ndef f():\n    return x  # same\n")
    assert ast_digest([a]) == first
    a.write_text("x = 2\n\ndef f():\n    return x\n")
    assert ast_digest([a]) != first


def test_an_append_after_a_torn_tail_repairs_it(tmp_path):
    path = tmp_path / "a.jsonl"
    append_jsonl(path, {"n": 1})
    with open(path, "ab") as f:
        f.write(b'{"n": 2, "torn')  # a crash mid-append
    append_jsonl(path, {"n": 3})
    append_jsonl(path, {"n": 4})
    assert read_jsonl(path) == [{"n": 1}, {"n": 3}, {"n": 4}]
