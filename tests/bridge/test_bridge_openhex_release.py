"""GitHub-release distribution of our OpenHex bots' checkpoints (catanrl.bridge.openhex_release):
a release packs each pinned checkpoint once, deterministically, with only what resolving it
needs, and refuses secrets and refs a fetched run cannot resolve; a fetch (from a server on
127.0.0.1: these tests never reach the network) unpacks the assets into a fresh runs root
that resolves every pinned ref, refuses a wrong sha256 or tag, unsafe members, a file that
would be overwritten and a ref the release does not serve, and a second fetch downloads
nothing."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import socket
import subprocess
import tarfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from catanatron.registry import SpecError

from catanrl.bridge import openhex_deploy as deploy
from catanrl.bridge import openhex_release as release
from catanrl.bridge.openhex_deploy import DeployError
from catanrl.checkpoint import CKPT_SCHEMA, FORMAT
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.models import build_model
from catanrl.runs import Run, RunWriter, runs_root

EVIDENCE = "4b223fd273477fcf"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Only 127.0.0.1 resolves: a fetch that ignored its base URL would fail here, not reach GitHub."""
    real = socket.getaddrinfo

    def local_only(host, *args, **kwargs):
        if host not in ("127.0.0.1", "localhost"):
            raise OSError(f"test is offline: {host}")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", local_only)


def _tiny(seed=0):
    torch.manual_seed(seed)
    dims = dims_for(["base-2p"])
    return build_model({"preset": "graph-tiny"}, dims=dims, levels="full", num_actions=action_size(dims),
                       layout=row_layout(dims, "full"))


@pytest.fixture
def run():
    """A finished run exp/v in the test's runs dir: e0 and e3, ``best`` at e3."""
    raw = {"seed": 1, "env": {"scenarios": [{"game": "base-2p"}]}, "model": {"preset": "graph-tiny"}}
    cfg = resolve(raw, experiment="exp", variant="v")
    with RunWriter.open(Variant("exp", "v", None, {}, cfg, config_hash(cfg)), argv=["t"]) as w:
        first = w.save_checkpoint(_tiny(), epoch=0, agent_steps=0)
        w.save_trainer_state({"epoch": 0})
        best = w.save_checkpoint(_tiny(seed=1), epoch=3, agent_steps=30)
        w.save_trainer_state({"epoch": 3})
        w.set_alias("best", best, record=SimpleNamespace(eval_id="x", summary={"score_rate": 0.5}))
        w.finish("finished")
    return SimpleNamespace(first=first, best=best, path=runs_root() / "exp" / "v",
                           e3=f"exp/v@e3#{best.weights_sha256[:12]}", e0=f"exp/v@e0#{first.weights_sha256[:12]}")


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A throwaway git checkout with one commit, tagged ``t1``."""
    config = tmp_path / "gitconfig"
    config.write_text("[user]\n\tname = t\n\temail = t@t\n[commit]\n\tgpgsign = false\n[tag]\n\tgpgsign = false\n"
                      "[init]\n\tdefaultBranch = main\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "checkout"
    (root / "deploy").mkdir(parents=True)
    (root / "deploy" / "openhex.yaml").write_text("bots: []\n")

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "first")
    git("tag", "t1")
    return SimpleNamespace(root=root, git=git, commit=git("rev-parse", "HEAD"))


def _bot(name, spec, **fields) -> dict:
    return {"name": name, "mode": "1v1", "version": name.removeprefix("catanrl-"), "spec": spec,
            "key_env": "OPENHEX_KEY_" + name.removeprefix("catanrl-").replace("-", "_").upper(), **fields}


def _config(where: Path, *bots, name: str = "openhex.yaml", **top) -> Path:
    path = where / name
    path.write_text(json.dumps({"bots": list(bots), **top}))  # JSON is YAML
    return path


def _bots(run) -> list[dict]:
    """Two running bots of tag t1 sharing e3 (pinned to different lengths), a retired one of
    t1 on e0, and a running one with no tag."""
    return [_bot("catanrl-1v1-v2", f"BELIEF:ckpt={run.e3}:worlds=16", tag="t1", evidence=[EVIDENCE]),
            _bot("catanrl-1v1-search-v2", f"MCTS:ckpt={run.e3[:-4]}:sims=8", tag="t1"),
            _bot("catanrl-1v1", f"POLICY:ckpt={run.e0}", tag="t1", retired=True),
            _bot("catanrl-4p", f"POLICY:ckpt={run.e0}", mode="4p")]


@pytest.fixture
def released(run, checkout, tmp_path):
    config = _config(tmp_path, *_bots(run))
    out = tmp_path / "out"
    return SimpleNamespace(run=run, config=config, out=out, checkout=checkout,
                           result=release.release(config, "t1", out, root=checkout.root))


def _members(path: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(path, "r:gz") as tar:
        return {m.name: m for m in tar.getmembers()}


def _read(path: Path, name: str) -> bytes:
    with tarfile.open(path, "r:gz") as tar:
        return tar.extractfile(name).read()


# ---------------------------------------------------------------- release


def test_a_release_packs_each_pinned_checkpoint_once_with_what_resolving_it_needs(released):
    run, result = released.run, released.result
    manifest = json.loads((released.out / "manifest.json").read_text())
    assert manifest == result["manifest"]
    assert (manifest["schema"], manifest["repo"], manifest["tag"], manifest["commit"]) == (
        1, "mcrco/catanrl-v2", "t1", released.checkout.commit)
    assert result["warnings"] == []
    # the bots of the tag, retired or not; the untagged one is not in it
    assert [b["name"] for b in manifest["bots"]] == ["catanrl-1v1-v2", "catanrl-1v1-search-v2", "catanrl-1v1"]
    assert set(manifest["bots"][0]) == {"name", "mode", "version", "bot_id", "spec"}
    e0, e3 = manifest["assets"]
    assert e3 == {"file": f"exp--v--e000003-{run.best.weights_sha256[:12]}.tar.gz", "run": "exp/v", "epoch": 3,
                  "weights_sha256": run.best.weights_sha256, "refs": sorted([run.e3, run.e3[:-4]]),
                  "sha256": hashlib.sha256((released.out / e3["file"]).read_bytes()).hexdigest(),
                  "size": (released.out / e3["file"]).stat().st_size}
    assert e0["epoch"] == 0 and e0["refs"] == [run.e0]
    assert sorted(p.name for p in released.out.iterdir()) == sorted(["manifest.json", "release-notes.md",
                                                                     e0["file"], e3["file"]])
    path = released.out / e3["file"]
    members = _members(path)
    assert list(members) == ["runs/exp/v/checkpoints/e000003.pt", "runs/exp/v/config.json", "runs/exp/v/run.json"]
    assert all(m.isreg() and m.mtime == 0 and m.mode == 0o644 and m.uid == m.gid == 0 and m.uname == m.gname == ""
               for m in members.values())
    assert _read(path, "runs/exp/v/config.json") == (run.path / "config.json").read_bytes()
    assert _read(path, "runs/exp/v/checkpoints/e000003.pt") == run.best.path.read_bytes()
    assert json.loads(_read(path, "runs/exp/v/run.json")) == {
        "checkpoints": [{"agent_steps": 30, "epoch": 3, "file": "e000003.pt", "weights_sha256": run.best.weights_sha256}],
        "launches": [], "status": "released"}
    notes = (released.out / "release-notes.md").read_text()
    assert EVIDENCE in notes and "catanrl-1v1-search-v2" in notes and e3["sha256"] in notes
    assert f"catanrl openhex fetch {released.config}" in notes and "git checkout t1" in notes
    assert result["command"] == (f"cd {released.out} && gh release create t1 manifest.json {e0['file']} {e3['file']} "
                                 "--repo mcrco/catanrl-v2 --verify-tag --title t1 --notes-file release-notes.md")
    assert result["scanned"] == 6


def test_a_release_is_deterministic(released, tmp_path):
    again = release.release(released.config, "t1", tmp_path / "again", root=released.checkout.root)
    for asset in released.result["manifest"]["assets"]:
        data = (released.out / asset["file"]).read_bytes()
        assert (tmp_path / "again" / asset["file"]).read_bytes() == data
        assert data[:2] == b"\x1f\x8b" and data[4:8] == b"\0\0\0\0"  # gzip's mtime: 0, not the second it was built
    assert again["manifest"] == released.result["manifest"]
    assert (tmp_path / "again" / "manifest.json").read_bytes() == (released.out / "manifest.json").read_bytes()
    assert (tmp_path / "again" / "release-notes.md").read_bytes() == (released.out / "release-notes.md").read_bytes()


def test_the_manifest_names_the_tags_commit_else_head_with_a_warning(run, checkout, tmp_path):
    config = _config(tmp_path, _bot("catanrl-1v1-v2", f"BELIEF:ckpt={run.e3}", tag="t2"))
    (checkout.root / "README.md").write_text("later\n")
    checkout.git("add", ".")
    checkout.git("commit", "-qm", "later")
    result = release.release(config, "t2", tmp_path / "out", root=checkout.root)
    head = checkout.git("rev-parse", "HEAD")
    assert result["manifest"]["commit"] == head != checkout.commit
    assert "there is no tag t2" in result["warnings"][0] and head[:12] in result["warnings"][0]
    assert "warning: there is no tag t2" in release.format_release(result)


def test_the_deploy_file_names_the_releases_repository(run, checkout, tmp_path):
    config = _config(tmp_path, _bot("catanrl-1v1-v2", f"BELIEF:ckpt={run.e3}", tag="t1"), releases="someone/fork")
    result = release.release(config, "t1", tmp_path / "out", root=checkout.root)
    assert result["manifest"]["repo"] == "someone/fork" and "--repo someone/fork" in result["command"]
    assert deploy.parse_deploy({"bots": [_bot("catanrl-x", "F")], "releases": "a/b"})[0] == {"releases": "a/b"}
    for bad in ("not a repo", "a/b/c", 3):
        with pytest.raises(DeployError, match="releases must be a GitHub repository"):
            deploy.parse_deploy({"bots": [_bot("catanrl-x", "F")], "releases": bad})


@pytest.mark.parametrize("spec, tag, message", [
    ("BELIEF:ckpt=exp/v@e3#{sha}", "nope", "no bot in .* has tag nope \\(tags: t1\\)"),
    ("BELIEF:ckpt=exp/v@best", "t1", "not pinned by #sha: exp/v@best resolves to exp/v@e3#{sha} now"),
    ("BELIEF:ckpt=exp/v@best#{sha}", "t1", "a fetched run resolves only the epochs it holds; pin it as exp/v@e3#{sha}"),
    ("BELIEF:ckpt=exp@e3#{sha}", "t1", "pin it as exp/v@e3#{sha}"),
    ("BELIEF:ckpt=exp/v@e3#000000000000", "t1", "does not resolve"),
    ("BELIEF:ckpt={pt}#{sha}", "t1", "is a file: a release ships run checkpoints"),
    ("BELIEF:ckpt=exp/v@e3#{sha}", "t/1", "a release tag is"),
    ("F", "t1", "the bots tagged t1 \\(catanrl-1v1-v2\\) pin no checkpoint: nothing to release"),
])
def test_a_release_refuses(run, checkout, tmp_path, spec, tag, message):
    sha = run.best.weights_sha256[:12]
    spec = spec.format(sha=sha, pt=run.best.path)
    config = _config(tmp_path, _bot("catanrl-1v1-v2", spec, tag="t1"))
    with pytest.raises(DeployError, match=message.format(sha=sha)):
        release.release(config, tag, tmp_path / "out", root=checkout.root)
    assert not (tmp_path / "out").exists()


def test_a_release_refuses_an_out_dir_that_is_not_empty(released, tmp_path):
    with pytest.raises(DeployError, match="is not empty"):
        release.release(released.config, "t1", released.out, root=released.checkout.root)


def test_a_release_refuses_a_checkpoint_a_fetch_would_not_unpack(released, tmp_path, monkeypatch):
    monkeypatch.setattr(release, "ASSET_BYTES", release.TAR_SLACK + 1000)
    with pytest.raises(DeployError, match=f"{released.run.e0[:-13]}.*bytes, more than a fetch unpacks \\(1000\\)"):
        release.release(released.config, "t1", tmp_path / "out2", root=released.checkout.root)
    assert not (tmp_path / "out2").exists()


@pytest.mark.parametrize("value, what", [
    ("WANDB_API_KEY=abc", "WANDB_API_KEY"),
    ("wandb login " + "0123456789" * 4, "a W&B key"),
    ("export OPENHEX_KEY_1V1=x", "OPENHEX"),
    ("my api_key here", "api_key"),
    ("a token here", "token"),
    ("Authorization: Bearer xyz", "Bearer"),
    ({"deploy_secret": "0123456789abcdef" * 2}, "32+ hex digits in a key-like field"),
    ("-----BEGIN OPENSSH PRIVATE KEY-----", "a private key"),
])
def test_a_release_refuses_a_secret_naming_the_file_and_line_not_the_value(released, tmp_path, value, what):
    path = released.run.path / "config.json"
    data = json.loads(path.read_text())
    data["zz_note"] = value  # sorted last: a line far down the file
    path.write_text(json.dumps(data, sort_keys=True, indent=1) + "\n")
    line = next(i for i, text in enumerate(path.read_text().splitlines(), 1) if "zz_note" in text)
    with pytest.raises(DeployError) as error:
        release.release(released.config, "t1", tmp_path / "out2", root=released.checkout.root)
    text = str(error.value)
    assert f"{path}: line {line + (1 if isinstance(value, dict) else 0)}: {what}" in text
    assert "nothing was written" in text and not (tmp_path / "out2").exists()
    secret = value if isinstance(value, str) else value["deploy_secret"]
    assert secret not in text


def test_the_scan_names_a_byte_in_a_binary_file_and_passes_hashes():
    assert release.scan(b"PK\x00\x01 Bearer x", "e000003.pt") == ["e000003.pt: byte 5: Bearer"]
    clean = json.dumps({"weights_sha256": "ab" * 32, "eval_id": "05d2b1b7d753c141", "config_hash": "2fdb6f44ac56b5ff",
                        "source": {"sha256": "cd" * 32}, "game": "openhex-1v1"}).encode()
    assert release.scan(clean, "run.json") == []


def test_a_release_lists_the_home_paths_that_become_public(released, tmp_path):
    path = released.run.path / "config.json"
    data = json.loads(path.read_text())
    data["source"] = {"path": "/home/someone/dev/catanrl-v2/configs/experiments/exp.yaml", "sha256": "ab" * 32}
    path.write_text(json.dumps(data, sort_keys=True, indent=1) + "\n")
    result = release.release(released.config, "t1", tmp_path / "out2", root=released.checkout.root)
    assert result["homes"] == {"runs/exp/v/config.json": ["/home/someone/dev/catanrl-v2/configs/experiments/exp.yaml"]}
    text = release.format_release(result)
    assert "runs/exp/v/config.json: /home/someone/dev/catanrl-v2/configs/experiments/exp.yaml" in text
    assert "secret scan: 6 files, no secret found" in text


# ---------------------------------------------------------------- fetch


@pytest.fixture
def server():
    """A release host on 127.0.0.1: ``files`` by URL path, ``moved`` paths redirected (as
    GitHub sends an asset to its storage host), every GET logged."""
    files: dict[str, bytes] = {}
    moved: dict[str, str] = {}
    log: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            log.append(self.path)
            if self.path in moved:
                self.send_response(302)
                self.send_header("Location", moved[self.path])
                self.end_headers()
                return
            body = files.get(self.path)
            self.send_response(404 if body is None else 200)
            self.send_header("Content-Length", str(len(body or b"")))
            self.end_headers()
            self.wfile.write(body or b"")

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(base=f"http://127.0.0.1:{httpd.server_address[1]}", files=files, moved=moved, log=log)
    httpd.shutdown()
    httpd.server_close()


def _publish(server, out: Path, tag: str) -> None:
    """Serve ``out`` as release ``tag``: the manifest at <tag>/, each asset redirected to /storage/."""
    for path in out.iterdir():
        if path.name.endswith(".tar.gz"):
            server.files[f"/storage/{tag}/{path.name}"] = path.read_bytes()
            server.moved[f"/{tag}/{path.name}"] = f"/storage/{tag}/{path.name}"
        else:
            server.files[f"/{tag}/{path.name}"] = path.read_bytes()


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


def test_a_fetch_fills_a_fresh_runs_root_that_resolves_every_pinned_ref(released, server, tmp_path, monkeypatch):
    from catanrl.agents import resolve_agent

    run = released.run
    _publish(server, released.out, "t1")
    fresh = tmp_path / "fresh"
    e3 = released.result["manifest"]["assets"][1]
    result = release.fetch(released.config, runs=fresh, base=server.base)
    assert result["ok"] and result["root"] == str(fresh.resolve())
    # the manifest and e3's asset (redirected) only: e0 serves only a retired bot
    assert server.log == ["/t1/manifest.json", f"/t1/{e3['file']}", f"/storage/t1/{e3['file']}"]
    assert result["downloaded"] == [e3["file"]]
    assert [(r["name"], r["ref"], r["state"], r["weights"]) for r in result["rows"]] == [
        ("catanrl-1v1-v2", run.e3, f"fetched {e3['file']}", run.best.weights_sha256[:12]),
        ("catanrl-1v1-search-v2", run.e3[:-4], f"fetched {e3['file']}", run.best.weights_sha256[:12])]
    assert result["skipped"] == {"retired": ["catanrl-1v1"], "untagged": ["catanrl-4p"]}
    assert _files(fresh) == ["exp/v/checkpoints/e000003.pt", "exp/v/config.json", "exp/v/run.json"]
    assert Run(fresh / "exp" / "v").status == "released"
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(fresh))
    agent = resolve_agent(f"BELIEF:ckpt={run.e3}:worlds=16", harness="server")
    assert agent.checkpoint.weights_sha256 == run.best.weights_sha256
    with pytest.raises(SpecError, match="has no checkpoint 'best'"):  # a fetched run has no aliases
        resolve_agent("BELIEF:ckpt=exp/v@best", harness="server")
    text = release.format_fetch(result, default_root="/elsewhere")
    assert "catanrl-1v1-v2" in text and "skipped (retired): catanrl-1v1" in text and text.endswith("ok")
    assert f"CATANRL_RUNS_DIR={fresh.resolve()}" in text and "weights_only=True" in text

    server.log.clear()  # a second fetch: everything resolves here already, so no request at all
    again = release.fetch(released.config, runs=fresh, base=server.base)
    assert server.log == [] and again["downloaded"] == [] and again["written"] == [] and again["ok"]
    assert {r["state"] for r in again["rows"]} == {"already here"}


def test_a_fetch_extends_a_released_run_and_keeps_a_training_run_that_has_it(released, server, tmp_path, checkout):
    """Two releases of one run: the second fetch adds its checkpoint to the run.json the
    first wrote. A training run that already lists and holds a checkpoint needs nothing."""
    run = released.run
    bots = [_bot("catanrl-1v1-v2", f"BELIEF:ckpt={run.e3}:worlds=16", tag="t1"),
            _bot("catanrl-1v1-v3", f"POLICY:ckpt={run.e0}", tag="t2")]
    config = _config(tmp_path, *bots, name="two-tags.yaml")
    release.release(config, "t2", tmp_path / "out2", root=checkout.root)
    _publish(server, released.out, "t1")
    _publish(server, tmp_path / "out2", "t2")
    fresh = tmp_path / "fresh"
    release.fetch(config, runs=fresh, base=server.base, only="catanrl-1v1-v2")
    server.log.clear()
    result = release.fetch(config, runs=fresh, base=server.base)
    assert result["ok"] and [r["state"] for r in result["rows"]] == ["already here", f"fetched {result['downloaded'][0]}"]
    assert server.log[0] == "/t2/manifest.json" and result["downloaded"][0].startswith("exp--v--e000000-")
    rows = json.loads((fresh / "exp" / "v" / "run.json").read_text())
    assert [r["epoch"] for r in rows["checkpoints"]] == [0, 3] and rows["status"] == "released"
    server.log.clear()
    assert release.fetch(config, runs=runs_root(), base=server.base)["written"] == [] and server.log == []


def _tamper_manifest(change):
    """Change the served manifest: ``change(manifest, its entry of e3's asset)``."""
    def tamper(server, out, asset):
        manifest = json.loads(server.files["/t1/manifest.json"])
        change(manifest, next(a for a in manifest["assets"] if a["file"] == asset["file"]))
        server.files["/t1/manifest.json"] = json.dumps(manifest).encode()
    return tamper


def _flip_a_byte(server, out, asset):
    path = f"/storage/t1/{asset['file']}"
    data = bytearray(server.files[path])
    data[len(data) // 2] ^= 1
    server.files[path] = bytes(data)


def _replace(make):
    """Replace e3's asset by ``make(out, its file)`` (bytes), the manifest's sha256 and size updated to match."""
    def tamper(server, out, asset):
        data = make(out, out / asset["file"])
        server.files[f"/storage/t1/{asset['file']}"] = data
        manifest = json.loads(server.files["/t1/manifest.json"])
        for a in manifest["assets"]:
            if a["file"] == asset["file"]:
                a["sha256"], a["size"] = hashlib.sha256(data).hexdigest(), len(data)
        server.files["/t1/manifest.json"] = json.dumps(manifest).encode()
    return tamper


def _targz(members) -> bytes:
    """A tar.gz of ``members`` (TarInfo, bytes), in that order."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for info, data in members:
            info.size = len(data) if info.isreg() else 0
            tar.addfile(info, io.BytesIO(data) if info.isreg() else None)
    return buffer.getvalue()


def _evil(*members):
    """Replace e3's asset by a tar.gz of ``members`` (TarInfo, bytes), the manifest updated to match."""
    return _replace(lambda out, path: _targz(members))


def _edit(name, change):
    """Replace e3's asset by itself with member ``runs/exp/v/<name>`` as ``change(its bytes, out)``."""
    def make(out, path):
        files = {n: _read(path, n) for n in _members(path)}
        files[f"runs/exp/v/{name}"] = change(files[f"runs/exp/v/{name}"], out)
        return _targz([(tarfile.TarInfo(n), data) for n, data in sorted(files.items())])
    return _replace(make)


def _json_edit(**fields):
    return lambda data, out: json.dumps({**json.loads(data), **fields}).encode()


def _e0_weights(data, out):
    """e0's checkpoint, from the release's other asset."""
    (e0,) = out.glob("exp--v--e000000-*.tar.gz")
    return _read(e0, "runs/exp/v/checkpoints/e000000.pt")


def _headless_checkpoint(data, out):
    """A catanrl checkpoint dict that lacks its weights' sha256 and epoch."""
    buffer = io.BytesIO()
    torch.save({"format": FORMAT, "schema": CKPT_SCHEMA, "state_dict": {}}, buffer)
    return buffer.getvalue()


def _pax_chain(n=5000) -> bytes:
    """``n`` empty pax headers in a row: tarfile reads them recursively."""
    info = tarfile.TarInfo("././@PaxHeader")
    info.type = tarfile.XHDTYPE
    return gzip.compress(info.tobuf(tarfile.USTAR_FORMAT) * n + bytes(1024), mtime=0)


def _member(name, kind=tarfile.REGTYPE, link=""):
    info = tarfile.TarInfo(name)
    info.type, info.linkname = kind, link
    return info


@pytest.mark.parametrize("tamper, message", [
    (_flip_a_byte, "the manifest says .* not unpacked"),
    (_tamper_manifest(lambda m, a: m.update(tag="t0")), "it is the manifest of tag 't0', not t1"),
    (_tamper_manifest(lambda m, a: m.update(repo="someone/fork")), "not mcrco/catanrl-v2"),
    (_tamper_manifest(lambda m, a: m.update(schema=2)), "this code reads schema 1"),
    (_tamper_manifest(lambda m, a: m["assets"].remove(a)), "the release t1 serves no asset for exp/v@e3#"),
    (_tamper_manifest(lambda m, a: a.update(file="../x.tar.gz")), "is not an asset name"),
    (_evil((_member("runs/exp/v/../../evil"), b"x")), "member 'runs/exp/v/../../evil': a path with '..'"),
    (_evil((_member("/tmp/evil"), b"x")), "member '/tmp/evil': an absolute path"),
    (_evil((_member("runs/exp/v/config.json", tarfile.SYMTYPE, "/etc/passwd"), b"")), "a link"),
    (_evil((_member("runs/exp/v/config.json", tarfile.LNKTYPE, "runs/exp/v/run.json"), b"")), "a link"),
    (_evil((_member("runs/exp/v/checkpoints", tarfile.DIRTYPE), b"")), "not a regular file"),
    (_evil((_member("runs/other/v/config.json"), b"{}")), "outside runs/exp/v/"),
    (_evil((_member("runs/exp/v/aliases.jsonl"), b"{}")), "not a file an asset holds"),
    (_evil((_member("runs/exp/v/config.json"), b"{}")), "checkpoints/e000003.pt, runs/exp/v/run.json missing"),
    (_edit("checkpoints/e000003.pt", _e0_weights), "e000003.pt: weights .* differ from the pinned"),
    (_edit("checkpoints/e000003.pt", _headless_checkpoint), "KeyError"),
    (_edit("config.json", _json_edit(experiment="other")), "its config.json is not run exp/v's"),
    (_edit("config.json", lambda data, out: b"[" * 100_000), "config.json or run.json is not JSON \\(RecursionError"),
    (_edit("run.json", _json_edit(status="finished")), "its run.json is not a release's run.json listing e3"),
    (_replace(lambda out, path: _pax_chain()), "not a readable tar \\(RecursionError"),
    (_replace(lambda out, path: path.read_bytes() + gzip.compress(b"more", mtime=0)), "not one complete gzip stream"),
    (lambda server, out, asset: server.files.update({"/t1/manifest.json": b"[" * 100_000}),
     "manifest.json: not JSON \\(RecursionError"),
])
def test_a_fetch_refuses_and_writes_nothing(released, server, tmp_path, tamper, message):
    _publish(server, released.out, "t1")
    asset = released.result["manifest"]["assets"][1]
    tamper(server, released.out, asset)
    fresh = tmp_path / "fresh"
    with pytest.raises(DeployError, match=message):
        release.fetch(released.config, runs=fresh, base=server.base)
    assert _files(fresh) == [] and not (tmp_path / "evil").exists()


def test_an_asset_inflates_within_the_cap(released, monkeypatch):
    """A pax header that claims more than the cap never gets the memory: the inflate stops at the cap."""
    monkeypatch.setattr(release, "ASSET_BYTES", 1 << 20)
    info = tarfile.TarInfo("././@PaxHeader")
    info.type, info.size = tarfile.XHDTYPE, 4 << 20
    bomb = gzip.compress(info.tobuf(tarfile.USTAR_FORMAT) + bytes(4 << 20) + bytes(1024), mtime=0)
    assert len(bomb) < 10_000
    asset = {"run": "exp/v", "epoch": 3, "file": "bomb.tar.gz", "weights_sha256": "0" * 64}
    with pytest.raises(DeployError, match="bomb.tar.gz: more than 1048576 bytes unpacked"):
        release.unpack(bomb, asset)
    e3 = released.result["manifest"]["assets"][1]
    data = (released.out / e3["file"]).read_bytes()
    assert sorted(release.unpack(data, e3)) == ["exp/v/checkpoints/e000003.pt", "exp/v/config.json", "exp/v/run.json"]


def test_a_fetch_never_overwrites_a_file_with_other_content(released, server, tmp_path):
    _publish(server, released.out, "t1")
    fresh = tmp_path / "fresh"
    config = fresh / "exp" / "v" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"experiment": "exp", "variant": "v", "other": 1}\n')
    with pytest.raises(DeployError, match=f"{config} exists with different content"):
        release.fetch(released.config, runs=fresh, base=server.base)
    assert _files(fresh) == ["exp/v/config.json"]
    shutil.copyfile(released.run.path / "config.json", config)  # the same content: fine
    ckpt = fresh / "exp" / "v" / "checkpoints" / "e000003.pt"
    ckpt.parent.mkdir()
    shutil.copyfile(released.run.first.path, ckpt)  # another checkpoint under e3's name
    with pytest.raises(DeployError, match=f"{ckpt} exists with different content: not overwritten"):
        release.fetch(released.config, runs=fresh, base=server.base)
    assert ckpt.read_bytes() == released.run.first.path.read_bytes()
    ckpt.unlink()
    assert release.fetch(released.config, runs=fresh, base=server.base)["ok"]


def test_a_fetch_leaves_a_training_run_that_lacks_the_checkpoint_alone(released, server, tmp_path):
    _publish(server, released.out, "t1")
    training = tmp_path / "training"
    shutil.copytree(released.run.path, training / "exp" / "v")
    state = training / "exp" / "v" / "run.json"
    data = json.loads(state.read_text())
    data["checkpoints"] = [c for c in data["checkpoints"] if c["epoch"] != 3]
    state.write_text(json.dumps(data))
    (training / "exp" / "v" / "checkpoints" / "e000003.pt").unlink()
    before = state.read_bytes()
    with pytest.raises(DeployError, match="is a training run's \\(status finished\\) and does not list e3"):
        release.fetch(released.config, runs=training, base=server.base)
    assert state.read_bytes() == before and not (training / "exp" / "v" / "checkpoints" / "e000003.pt").exists()


def test_a_fetch_refuses_an_unpinned_or_untagged_bot_and_a_missing_release(released, server, tmp_path):
    run = released.run
    config = _config(tmp_path, _bot("catanrl-1v1-v2", "BELIEF:ckpt=exp/v@best", tag="t1"),
                     _bot("catanrl-4p", f"POLICY:ckpt={run.e0}", mode="4p"), name="unpinned.yaml")
    with pytest.raises(DeployError, match="exp/v@best is not pinned by #sha"):
        release.fetch(config, runs=tmp_path / "fresh", base=server.base, only="catanrl-1v1-v2")
    with pytest.raises(DeployError, match="catanrl-4p: no tag"):
        release.fetch(config, runs=tmp_path / "fresh", base=server.base, only="catanrl-4p")
    with pytest.raises(DeployError, match="catanrl-1v1 retired"):
        release.fetch(released.config, runs=tmp_path / "fresh", base=server.base, only="catanrl-1v1")
    with pytest.raises(DeployError, match="HTTP 404 \\(is the release published"):
        release.fetch(released.config, runs=tmp_path / "fresh", base=server.base)  # nothing published


def test_a_download_is_capped_and_never_redirected_off_https(released, server, tmp_path, monkeypatch):
    _publish(server, released.out, "t1")
    monkeypatch.setattr(release, "MANIFEST_BYTES", 100)
    with pytest.raises(DeployError, match="more than 100"):
        release.fetch(released.config, runs=tmp_path / "fresh", base=server.base)
    request = urllib.request.Request("https://github.com/mcrco/catanrl-v2/releases/download/t1/x.tar.gz")
    with pytest.raises(DeployError, match="not https"):
        release._HttpsOnly().redirect_request(request, None, 302, "Found", {}, "http://example.test/x.tar.gz")


# ---------------------------------------------------------------- the CLI


def test_the_cli_releases_and_fetches(released, server, tmp_path, monkeypatch, capsys):
    from catanrl import cli

    monkeypatch.setattr(deploy, "checkout_root", lambda: released.checkout.root)
    monkeypatch.setattr(release, "checkout_root", lambda: released.checkout.root)
    out = tmp_path / "cli-out"
    assert cli.main(["openhex", "release", str(released.config), "t1", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"cd {out} && gh release create t1 manifest.json" in text and "secret scan: 6 files" in text
    assert "--repo mcrco/catanrl-v2 --verify-tag" in text
    _publish(server, out, "t1")
    fresh = tmp_path / "fresh"
    assert cli.main(["openhex", "fetch", str(released.config), "--runs", str(fresh), "--base-url", server.base]) == 0
    text = capsys.readouterr().out
    assert text.rstrip().endswith("ok") and "catanrl-1v1-search-v2" in text
    for argv, message in [
        (["openhex", "release", str(released.config)], "catanrl openhex release CONFIG TAG"),
        (["openhex", "fetch"], "catanrl openhex fetch CONFIG"),
        (["openhex", str(released.config), "t1"], "catanrl openhex CONFIG \\(unexpected 't1'\\)"),
        (["openhex", str(released.config), "--out", "x"], "--out is for catanrl openhex release"),
        (["openhex", "stats", str(released.config), "--runs", "x"], "--runs is for catanrl openhex fetch"),
        (["openhex", "fetch", str(released.config), "--check"], "fetch does not take --check"),
        (["openhex", "release", str(released.config), "t1", "--only", "x"], "release does not take --only"),
    ]:
        with pytest.raises(SystemExit, match=message):
            cli.main(argv)
    assert cli.main(["openhex", "release", str(released.config), "nope"]) == 2  # a refusal: one line
    assert "no bot in" in capsys.readouterr().err
