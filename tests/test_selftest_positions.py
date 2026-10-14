"""Test 19: every packaged self-test position applies through apply_setup,
and its encodings at each level are golden (a change means the encoder
changed, so every checkpoint's self-test would report EncoderDrift)."""

from __future__ import annotations

import hashlib

from catanrl import checkpoint as ck
from catanrl.envs.encoding import dims_for, make_encoder

GOLDEN = {"private": "61fb690913bbee48", "public": "eac7ddc6211e576a", "full": "02245e853ce33443"}


def encodings(level: str) -> str:
    h = hashlib.sha256()
    for pos in ck.packaged_positions():
        game = ck._position_game(pos)
        dims = dims_for([pos["game"]])
        h.update(make_encoder(dims, level).encode(game, int(pos["seat"])).tobytes())
    return h.hexdigest()[:16]


def test_positions_cover_player_counts_and_maps():
    positions = ck.packaged_positions()
    assert len(positions) >= 24
    assert {p["game"]["num_players"] for p in positions} == {2, 3, 4, 5, 6}
    assert len({p["game"]["map"]["type"] for p in positions}) >= 3


def test_encodings_are_golden():
    assert {level: encodings(level) for level in ("private", "public", "full")} == GOLDEN
