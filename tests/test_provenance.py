"""Test 12: git state, patches, keep refs, code identity and resume diffs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from catanrl import provenance
from catanrl.provenance import ProvenanceError, StaleProcess


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "protocol.file.allow=always", "-c", "user.name=t",
                           "-c", "user.email=t@t", *args], check=True, capture_output=True, text=True).stdout


def make_repos(tmp_path: Path) -> tuple[Path, Path]:
    sub, sup = tmp_path / "sub", tmp_path / "super"
    for repo in (sub, sup):
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
    (sub / "lib.py").write_text("x = 1\n")
    git(sub, "add", "lib.py")
    git(sub, "commit", "-qm", "sub")
    (sup / "main.py").write_text("y = 2\n")
    git(sup, "add", "main.py")
    git(sup, "submodule", "-q", "add", str(sub), "engine")
    git(sup, "commit", "-qm", "super")
    return sup, sup / "engine"


def test_dirty_untracked_and_moved_submodules(tmp_path):
    sup, engine = make_repos(tmp_path)
    state = provenance.git_state(sup)
    assert not state["repos"]["catanrl-v2"]["dirty"] and state["submodules"]["engine"]["state"] == "ok"
    assert provenance.check({"git": state}, strict=True) == []
    (sup / "main.py").write_text("y = 3\n")
    (sup / "new.py").write_text("z = 4\n")
    (engine / "lib.py").write_text("x = 5\n")
    git(engine, "commit", "-qam", "moved")
    state = provenance.git_state(sup)
    repo = state["repos"]["catanrl-v2"]
    assert repo["dirty"] and repo["untracked"] == ["new.py"] and repo["diff_sha256"]
    assert state["submodules"]["engine"]["state"] == "moved"
    assert state["repos"]["engine"]["head"] == git(engine, "rev-parse", "HEAD").strip()
    warnings = provenance.check({"git": state})
    assert any("submodule engine is moved" in w for w in warnings)
    with pytest.raises(ProvenanceError):
        provenance.check({"git": state}, strict=True)

    prov = {"git": state, "run": {"launch_name": "launch0"}}
    provenance.save_patches(prov, tmp_path / "prov")
    patch = tmp_path / "prov" / "launch0.catanrl-v2.patch"
    assert patch.exists() and repo["patch"]["file"] == patch.name
    git(sup, "apply", "--check", "-R", str(patch))  # the patch is exactly the working-tree change
    provenance.keep_commits(prov)
    for name, info in state["repos"].items():
        root = sup if name == "catanrl-v2" else engine
        assert git(root, "show-ref", f"refs/catanrl/keep/{info['head']}").strip()


def test_code_id_follows_code_not_comments(tmp_path, monkeypatch):
    source = tmp_path / "mod.py"
    source.write_text("def f():\n    return 1\n")
    files = {"catanrl": {"py": [("mod.py", source)], "bin": []}}
    monkeypatch.setattr(provenance, "_components", lambda kind, harness: files)
    monkeypatch.setattr(provenance, "_SNAPSHOTS", {})
    first = provenance.code_id("eval")
    source.write_text("# a comment\ndef f():\n    \"\"\"Doc.\"\"\"\n    return 1  # same\n")
    provenance.assert_code_current()
    assert provenance._digest_component(files["catanrl"]) == first["catanrl"]
    source.write_text("def f():\n    return 2\n")
    with pytest.raises(StaleProcess):
        provenance.assert_code_current()
    assert provenance._digest_component(files["catanrl"]) != first["catanrl"]


def test_real_code_id_and_capture():
    ids = provenance.code_id("train")
    assert {"catanrl", "engine", "train", "pufferlib"} <= set(ids)
    assert provenance.code_id("eval")["catanrl"] == ids["catanrl"]
    prov = provenance.capture(argv=["test"])
    assert prov["engine"]["so_sha256"] and prov["python"]["packages_digest"]
    assert "catanrl-v2" in prov["git"]["repos"]


def test_resume_diff():
    first = {"code": {"train": "a", "engine": "e"}, "python": {"uv_lock_sha256": "l1"}}
    now = {"code": {"train": "b", "engine": "e"}, "python": {"uv_lock_sha256": "l2"}}
    assert provenance.resume_diff(first, now) == ["code train: a -> b", "uv.lock: l1 -> l2"]
    assert provenance.resume_diff(first, first) == []


def test_edits_after_process_start_are_caught_at_the_first_code_id(tmp_path, monkeypatch):
    """A module imported, then edited, then hashed: the process runs old code."""
    import sys
    import types

    original = provenance._components
    source, binary = tmp_path / "mod.py", tmp_path / "_core.so"
    source.write_text("x = 1\n")
    binary.write_bytes(b"\0")
    for path, files in ((source, {"py": [("mod.py", source)], "bin": []}),
                        (binary, {"py": [], "bin": [binary]})):
        module = types.ModuleType("fake_loaded")
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, "fake_loaded", module)
        monkeypatch.setattr(provenance, "_components", lambda kind, harness, f=files: {"x": f})
        monkeypatch.setattr(provenance, "_SNAPSHOTS", {})
        monkeypatch.setattr(provenance, "_PROCESS_START", path.stat().st_mtime - 10)
        with pytest.raises(StaleProcess):
            provenance.code_id("eval")
    assert any(label == "seeding.py" for label, _ in original("eval", "native")["catanrl"]["py"])


def test_stale_engine_builds_are_refused_unless_allowed(monkeypatch):
    prov = {"git": {"submodules": {}, "repos": {"catanrl-v2": {"head": "a", "dirty": False},
                                                 "engine/cppanatron": {"head": "b" * 40, "dirty": False}}},
            "engine": {"build": {"git_sha": "c" * 40, "git_dirty": False}}}
    monkeypatch.delenv("CATANRL_ALLOW_STALE_BUILD", raising=False)
    with pytest.raises(ProvenanceError, match="rebuild"):
        provenance.check_launch(prov, {})
    assert provenance.check_launch(prov, {"allow_stale_build": True})
    monkeypatch.setenv("CATANRL_ALLOW_STALE_BUILD", "1")
    assert provenance.check_launch(prov, {})
    assert provenance.any_dirty(prov)  # a build of another tree cannot be rebuilt from the commits
