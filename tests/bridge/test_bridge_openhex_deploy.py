"""Versioned OpenHex deploys (catanrl.bridge.openhex_deploy): the deploy file refuses
typos, clashes and unpinned checkpoints (naming the pinned ref to paste), the runner
skips retired versions, refuses play source that is not clean or not its tag's, and a
key that belongs to another bot; every game line names its version, spec and deploy
stamp; and the registration body fits OpenHex's 280 characters."""

from __future__ import annotations

import datetime
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from catanatron.registry import SpecError

from catanrl.bridge import openhex as openhex_module
from catanrl.bridge import openhex_deploy as deploy
from catanrl.bridge.openhex import OpenHexBot, OpenHexClient, OpenHexHTTPError
from catanrl.bridge.openhex_deploy import DeployEntry, DeployError
from catanrl.checkpoint import CheckpointRef
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.models import build_model
from catanrl.runs import RunWriter

SHA = "3355b5059503541c228d3dd863fc01f9b6f213c099a2d006f32d6bddbd4fbba4"
#: An OpenHex id (catanrl-1v1's).
PLAIN = "05c4a25c6c0e46e1"


def _bot(**fields) -> dict:
    out = {"name": "catanrl-1v1-v2", "mode": "1v1", "version": "1v1-v2",
           "spec": f"BELIEF:ckpt=puffer-graph2/full@e3350#{SHA[:12]}:worlds=16", "key_env": "OPENHEX_KEY_1V1_V2"}
    return {**out, **fields}


def _parse(*bots, **top):
    return deploy.parse_deploy({"bots": list(bots), **top})


# ---------------------------------------------------------------- the file


def test_a_deploy_file_parses_into_entries():
    settings, entries = _parse(
        _bot(bot_id="0123456789abcdef", tag="openhex-1v1-v2", evidence=["4b223fd273477fcf"],
             deployed=datetime.date(2026, 10, 26)),
        _bot(name="catanrl-1v1", version="1v1-v1", key_env="OPENHEX_KEY_1V1", retired=True,
             retired_at="2026-10-30"),
        api="https://example.test/api")
    assert settings == {"api": "https://example.test/api"}
    new, old = entries
    assert new == DeployEntry(name="catanrl-1v1-v2", mode="1v1", spec=_bot()["spec"], key_env="OPENHEX_KEY_1V1_V2",
                              version="1v1-v2", bot_id="0123456789abcdef", tag="openhex-1v1-v2",
                              evidence=("4b223fd273477fcf",), retired=False, deployed="2026-10-26")
    assert old.retired and old.retired_at == "2026-10-30" and old.tag is None


@pytest.mark.parametrize("bots, top, message", [
    ([_bot()], {"apis": "x"}, "unknown key 'apis'"),
    ([_bot(verison="x")], {}, "unknown key 'verison'"),
    ([{k: v for k, v in _bot().items() if k != "version"}], {}, "version is missing"),
    ([_bot(version=2)], {}, "version must be a string"),
    ([_bot(mode="2p")], {}, "mode '2p'"),
    ([_bot(name="ab")], {}, "3-32 characters"),
    ([_bot(name="x" * 33)], {}, "3-32 characters"),
    ([_bot(), _bot(key_env="OTHER")], {}, "name 'catanrl-1v1-v2' is used by more than one bot"),
    ([_bot(), _bot(mode="4p", key_env="OTHER")], {}, "name 'catanrl-1v1-v2' is used by more than one bot"),
    ([_bot(), _bot(name="catanrl-1v1-v3")], {}, "key_env OPENHEX_KEY_1V1_V2 is shared"),
    ([_bot(key_env="1BAD")], {}, "not an environment variable name"),
    ([_bot(bot_id="05C4A25C6C0E46E1")], {}, "16 lowercase hex digits"),
    ([_bot(bot_id=PLAIN), _bot(name="catanrl-1v1-v3", key_env="OTHER", bot_id=PLAIN)], {}, f"bot_id {PLAIN} is given"),
    ([_bot(retired="yes")], {}, "retired must be true or false"),
    ([_bot(retired_at="2026-10-30")], {}, "retired_at is set but retired is not true"),
    ([_bot(evidence="4b223fd273477fcf")], {}, "evidence must be a list"),
    (["catanrl-1v1"], {}, "expected a mapping"),
])
def test_the_deploy_file_refuses(bots, top, message):
    with pytest.raises(DeployError) as error:
        _parse(*bots, **top)
    assert message in str(error.value)


def test_a_retired_entry_may_keep_its_key_variable_and_an_empty_file_is_refused():
    _, entries = _parse(_bot(), _bot(name="catanrl-1v1", retired=True))  # the same key_env: not run, no clash
    assert [e.retired for e in entries] == [False, True]
    with pytest.raises(DeployError, match="non-empty list of bots"):
        deploy.parse_deploy({"bots": []})


def test_the_runner_skips_retired_entries():
    _, entries = _parse(_bot(), _bot(name="catanrl-1v1", key_env="OPENHEX_KEY_1V1", retired=True),
                        _bot(name="catanrl-4p", mode="4p", key_env="OPENHEX_KEY_4P"))
    assert [e.name for e in deploy.selected(entries)] == ["catanrl-1v1-v2", "catanrl-4p"]
    assert [e.name for e in deploy.selected(entries, "catanrl-4p, catanrl-1v1-v2")] == ["catanrl-1v1-v2", "catanrl-4p"]
    with pytest.raises(DeployError, match="catanrl-1v1 retired"):
        deploy.selected(entries, "catanrl-1v1")
    with pytest.raises(DeployError, match="no bot named nope"):
        deploy.selected(entries, "nope")


def test_the_cli_runner_plays_only_versions_that_are_not_retired(tmp_path, monkeypatch):
    """``catanrl openhex FILE`` end to end with OpenHex, the agents and git faked: the retired
    version is never resolved, keyed or started; each bot checks its key and gets its stamp."""
    from catanrl import cli, provenance
    from catanrl.agents import spec as spec_module

    path = tmp_path / "openhex.yaml"
    path.write_text(json.dumps({"bots": [
        _bot(bot_id="0123456789abcdef"),
        _bot(name="catanrl-1v1", version="1v1-v1", key_env="OPENHEX_KEY_1V1", retired=True,
             spec="BELIEF:ckpt=puffer-graph2/full@best:worlds=16"),  # retired: never pinned, never run
    ]}))
    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", "key-of-v2")
    monkeypatch.delenv("OPENHEX_KEY_1V1", raising=False)
    started, asked = [], []

    class Client:
        def __init__(self, key, base):
            self.key = key

        def me(self):
            asked.append(self.key)
            return {"id": "0123456789abcdef", "name": "catanrl-1v1-v2", "mode": "1v1"}

    class Bot:
        def __init__(self, spec, key, mode, *, name, stamp, **_):
            self.name, self.stamp = name, stamp
            started.append(self)

        def announce(self):
            self.announced = True

        def run(self):
            return None

    resolved = SimpleNamespace(canonical="BELIEF:ckpt=...", checkpoints=(_ref(),))
    monkeypatch.setattr(openhex_module, "OpenHexClient", Client)
    monkeypatch.setattr(openhex_module, "OpenHexBot", Bot)
    monkeypatch.setattr(spec_module, "resolve_agent", lambda spec, harness: resolved)
    monkeypatch.setattr(deploy, "git_stamps", lambda root, entries, **_: (
        {e.name: {"commit": "c" * 40, "describe": "openhex-1v1-v2", "dirty": False, "tag": None, "tag_commit": None}
         for e in entries}, []))
    monkeypatch.setattr(cli, "_engine_build", dict)
    monkeypatch.setattr(provenance, "code_id", lambda kind: {"catanrl": "a" * 16, "engine": "b" * 16})
    monkeypatch.setattr(provenance, "engine_info", lambda: {"so_sha256": "d" * 64})
    assert cli.main(["openhex", str(path), "--state", str(tmp_path / "state")]) == 0
    assert [b.name for b in started] == ["catanrl-1v1-v2"] and asked == ["key-of-v2"]
    assert started[0].announced and started[0].stamp["version"] == "1v1-v2"
    assert started[0].stamp["code_id"] == {"catanrl": "a" * 16, "engine": "b" * 16}


# ---------------------------------------------------------------- pinned checkpoints


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
    return SimpleNamespace(first=first, best=best)


def test_an_unpinned_checkpoint_is_refused_with_the_pinned_ref_it_resolves_to(run):
    pinned = f"exp/v@e3#{run.best.weights_sha256[:12]}"
    _, entries = _parse(_bot(spec="BELIEF:ckpt=exp/v@best:worlds=16"),
                        _bot(name="catanrl-1v1-mcts", key_env="OTHER", spec="MCTS:exp/v@e3:sims=8"),
                        _bot(name="catanrl-1v1-ok", key_env="OK", spec=f"POLICY:ckpt={pinned}"),
                        _bot(name="catanrl-1v1-old", key_env="OLD", spec="POLICY:ckpt=exp/v@e0", retired=True))
    with pytest.raises(DeployError) as error:
        deploy.check_pins(entries)
    text = str(error.value)
    assert f"exp/v@best resolves to {pinned} now" in text
    assert f"spec: BELIEF:ckpt={pinned}:worlds=16" in text
    assert f"spec: MCTS:{pinned}:sims=8" in text  # a positional ckpt too
    assert "catanrl-1v1-ok" not in text and "catanrl-1v1-old" not in text  # pinned; retired
    warnings = deploy.check_pins(entries, allow_unpinned=True)  # --allow-unpinned: the same, as warnings
    assert len(warnings) == 2 and pinned in warnings[0]
    _, missing = _parse(_bot(spec="POLICY:ckpt=exp/nope@best"))
    assert "does not resolve" in deploy.pin_problems(missing[0])[0]


def test_a_pin_that_is_not_the_weights_on_disk_fails_in_the_resolver(run):
    """No new check needed: catanrl.runs and catanrl.checkpoint refuse a stale pin."""
    from catanrl.agents import resolve_agent

    sha = run.best.weights_sha256[:12]
    assert resolve_agent(f"BELIEF:ckpt=exp/v@e3#{sha}", harness="server").checkpoint.weights_sha256 == run.best.weights_sha256
    with pytest.raises(SpecError, match=f"is #{sha}, not #000000000000"):
        resolve_agent("BELIEF:ckpt=exp/v@e3#000000000000", harness="server")
    shutil.copyfile(run.first.path, run.best.path)  # e3's file replaced by other (valid) weights
    with pytest.raises(SpecError, match=f"differ from the pinned {sha}"):
        resolve_agent(f"BELIEF:ckpt=exp/v@e3#{sha}", harness="server")


# ---------------------------------------------------------------- provenance


def _ref(sha=SHA):
    return CheckpointRef(run="puffer-graph2/full", which="e3350", epoch=3350, agent_steps=0,
                         path=Path("/runs/puffer-graph2/full/checkpoints/e003350.pt"), weights_sha256=sha)


def _stamp(entry=None, **git):
    entry = entry or _parse(_bot(bot_id="0123456789abcdef", tag="openhex-1v1-v2"))[1][0]
    resolved = SimpleNamespace(canonical=entry.spec + ":allow_drift=false:tracker=pool", checkpoints=(_ref(),))
    git = {"commit": "c" * 40, "describe": "openhex-1v1-v2", "dirty": False, "tag": entry.tag, "tag_commit": "c" * 40,
           **git}
    return deploy.stamp(entry, resolved, git, code={"catanrl": "a" * 16, "engine": "b" * 16},
                        engine={"so_sha256": "d" * 64, "build": {"git_sha": "e" * 40}})


def test_the_stamp_names_version_spec_weights_code_and_engine():
    s = _stamp()
    assert s["version"] == "1v1-v2" and s["spec"] == _bot()["spec"] and s["bot_id"] == "0123456789abcdef"
    assert s["canonical"].endswith(":tracker=pool")
    assert s["checkpoints"] == [{"ref": f"puffer-graph2/full@e3350#{SHA[:12]}", "weights_sha256": SHA}]
    assert s["git"] == {"commit": "c" * 40, "describe": "openhex-1v1-v2", "dirty": False, "tag": "openhex-1v1-v2",
                        "tag_commit": "c" * 40}
    assert s["code_id"] == {"catanrl": "a" * 16, "engine": "b" * 16} and s["engine"]["so_sha256"] == "d" * 64
    pid = deploy.provenance_id(s)
    assert len(pid) == 12 and pid == deploy.provenance_id(dict(reversed(list(s.items())))) == deploy.provenance_id(_stamp())
    assert pid != deploy.provenance_id(_stamp(dirty=True))
    other = _parse(_bot(bot_id="0123456789abcdef", tag="openhex-1v1-v2", version="1v1-v2b"))[1][0]
    assert pid != deploy.provenance_id(_stamp(other))


def test_build_stamps_takes_the_code_id_once(monkeypatch):
    from catanrl import provenance

    calls = []
    monkeypatch.setattr(provenance, "code_id", lambda kind: calls.append(kind) or {"catanrl": "a" * 16})
    monkeypatch.setattr(provenance, "engine_info", lambda: {"so_sha256": "d" * 64})
    _, entries = _parse(_bot(), _bot(name="catanrl-4p", mode="4p", key_env="K4"))
    resolved = {e.name: SimpleNamespace(canonical=e.spec, checkpoints=(_ref(),)) for e in entries}
    stamps = deploy.build_stamps(entries, resolved, {e.name: {"commit": None} for e in entries})
    assert calls == ["eval"] and set(stamps) == {"catanrl-1v1-v2", "catanrl-4p"}
    assert stamps["catanrl-4p"]["mode"] == "4p" and stamps["catanrl-4p"]["code_id"] == {"catanrl": "a" * 16}


def _Seat():
    """What ``OpenHexBot._record`` reads of a seat."""
    return SimpleNamespace(colors=["RED", "BLUE"], seat=1, decisions=41, fallbacks=0,
                           stats={"mismatches": 0, "bank_short": 0}, search={})


def test_every_game_line_names_its_version_spec_and_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(openhex_module, "native_for", lambda spec, device: (object(), "BELIEF:x"))
    s = _stamp()
    bot = OpenHexBot("ignored", "secret-key-123", "1v1", state_dir=tmp_path, name="catanrl-1v1-v2", stamp=s)
    bot.announce()
    line = bot._record("g1", _Seat(), {"status": "completed", "winner": "blue", "players": [{"name": "a"}, {"name": "b"}]})
    pid = deploy.provenance_id(s)
    assert (line["version"], line["spec"], line["provenance_id"]) == ("1v1-v2", _bot()["spec"], pid)
    assert line["color"] == "BLUE" and line["decisions"] == 41
    assert "git" not in line and "checkpoints" not in line  # the stamp itself stays in deploys.jsonl
    games = [json.loads(x) for x in (tmp_path / "games.jsonl").read_text().splitlines()]
    deploys = [json.loads(x) for x in (tmp_path / "deploys.jsonl").read_text().splitlines()]
    assert games == [line] and len(deploys) == 1
    assert deploys[0]["provenance_id"] == pid and {k: v for k, v in deploys[0].items() if k not in ("time", "provenance_id")} == s
    assert "secret-key-123" not in (tmp_path / "games.jsonl").read_text() + (tmp_path / "deploys.jsonl").read_text()
    plain = OpenHexBot("POLICY:x", "k", "1v1", state_dir=tmp_path / "plain")  # no stamp: the keys are still there
    assert {k: plain._record("g2", _Seat(), {})[k] for k in ("version", "spec", "provenance_id")} == {
        "version": None, "spec": "POLICY:x", "provenance_id": None}


# ---------------------------------------------------------------- the code


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway superproject: src/, deploy/, report/, README.md and a submodule
    third_party/lib, committed and tagged ``v1``."""
    config = tmp_path / "gitconfig"
    config.write_text("[user]\n\tname = t\n\temail = t@t\n[commit]\n\tgpgsign = false\n[tag]\n\tgpgsign = false\n"
                      "[protocol \"file\"]\n\tallow = always\n[init]\n\tdefaultBranch = main\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def git(where, *args):
        return subprocess.run(["git", "-C", str(where), *args], check=True, capture_output=True, text=True).stdout

    lib = tmp_path / "lib"
    (lib / "pkg").mkdir(parents=True)
    (lib / "pkg" / "x.py").write_text("X = 1\n")
    git(lib, "init", "-q")
    git(lib, "add", ".")
    git(lib, "commit", "-qm", "lib")
    root = tmp_path / "repo"
    for name, text in {"src/catanrl/a.py": "A = 1\n", "deploy/openhex.yaml": "bots: []\n",
                       "report/r.typ": "r\n", "README.md": "readme\n"}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git(root, "init", "-q")
    git(root, "submodule", "add", "-q", str(lib), "third_party/lib")
    git(root, "add", ".")
    git(root, "commit", "-qm", "first")
    git(root, "tag", "v1")
    return SimpleNamespace(root=root, git=lambda *args: git(root, *args), sub=root / "third_party" / "lib",
                           subgit=lambda *args: git(root / "third_party" / "lib", *args))


def _entries(*tags):
    return [DeployEntry(name=f"bot-{i}", mode="1v1", spec="POLICY:x", key_env=f"K{i}", version=f"v{i}", tag=tag)
            for i, tag in enumerate(tags)]


def test_a_clean_tagged_checkout_passes(repo):
    stamps, problems = deploy.git_stamps(repo.root, _entries("v1", None))
    head = repo.git("rev-parse", "HEAD").strip()
    assert problems == []
    assert stamps["bot-0"] == {"commit": head, "describe": "v1", "dirty": False, "tag": "v1", "tag_commit": head}
    assert stamps["bot-1"]["tag"] is None and stamps["bot-1"]["tag_commit"] is None


def test_changes_outside_the_play_source_are_ignored(repo):
    (repo.root / "deploy" / "openhex.yaml").write_text("bots: [changed]\n")
    (repo.root / "README.md").write_text("changed\n")
    (repo.root / "report" / "r.typ").write_text("changed\n")
    (repo.root / "src" / "catanrl" / "untracked.py").write_text("U = 1\n")
    (repo.sub / "untracked.txt").write_text("u\n")
    stamps, problems = deploy.git_stamps(repo.root, _entries("v1"))
    assert problems == [] and not stamps["bot-0"]["dirty"]
    assert stamps["bot-0"]["describe"] == "v1-dirty"  # git's own view counts every tracked file


def test_a_dirty_play_source_is_refused_unless_allowed(repo):
    (repo.root / "src" / "catanrl" / "a.py").write_text("A = 2\n")
    with pytest.raises(DeployError, match="the play source differs from HEAD .*src/catanrl/a.py"):
        deploy.git_stamps(repo.root, _entries(None))
    stamps, problems = deploy.git_stamps(repo.root, _entries(None), allow_dirty=True)
    assert stamps["bot-0"]["dirty"] and len(problems) == 1


@pytest.mark.parametrize("change", ["moved", "modified"])
def test_a_submodule_moved_or_modified_counts_as_dirty(repo, change):
    if change == "moved":
        (repo.sub / "pkg" / "x.py").write_text("X = 2\n")
        repo.subgit("commit", "-qam", "moved")
    else:
        (repo.sub / "pkg" / "x.py").write_text("X = 3\n")
    with pytest.raises(DeployError, match="third_party/lib"):
        deploy.git_stamps(repo.root, _entries(None))


def test_the_tag_must_hold_the_code_that_plays(repo):
    (repo.root / "report" / "r.typ").write_text("a report commit on top of the tag\n")
    repo.git("commit", "-qam", "report")
    stamps, problems = deploy.git_stamps(repo.root, _entries("v1"))  # the play source is still v1's
    assert problems == [] and stamps["bot-0"]["tag_commit"] != stamps["bot-0"]["commit"]
    (repo.root / "src" / "catanrl" / "a.py").write_text("A = 2\n")
    repo.git("commit", "-qam", "code")
    with pytest.raises(DeployError, match=r"bot-0: HEAD \w+ does not play the code of v1 .*src/catanrl/a.py"):
        deploy.git_stamps(repo.root, _entries("v1"))
    with pytest.raises(DeployError, match="there is no tag v9"):
        deploy.git_stamps(repo.root, _entries("v9"))
    repo.git("tag", "v2")
    stamps, problems = deploy.git_stamps(repo.root, _entries("v1", "v2"), allow_dirty=True)
    assert len(problems) == 1 and stamps["bot-0"]["dirty"] and not stamps["bot-1"]["dirty"]


def test_a_checkout_that_cannot_be_read_and_a_stale_engine_build(tmp_path, repo):
    with pytest.raises(DeployError, match="cannot read the checkout"):
        deploy.git_stamps(tmp_path / "nowhere", _entries(None))
    stamps, _ = deploy.git_stamps(tmp_path / "nowhere", _entries(None), allow_dirty=True)
    assert stamps["bot-0"] == {"commit": None, "describe": None, "dirty": True, "tag": None, "tag_commit": None}
    engine = repo.root / "engine" / "cppanatron"
    engine.mkdir(parents=True)
    subprocess.run(["git", "-C", str(engine), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(engine), "commit", "-q", "--allow-empty", "-m", "e"], check=True)
    head = subprocess.run(["git", "-C", str(engine), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    assert deploy.engine_problem(repo.root, {"git_sha": head, "git_dirty": False}) is None
    assert "rebuild it" in deploy.engine_problem(repo.root, {"git_sha": "f" * 40, "git_dirty": False})


# ---------------------------------------------------------------- the key


def test_a_key_that_is_another_bots_is_refused():
    entry = _parse(_bot(bot_id="0123456789abcdef"))[1][0]
    me = {"id": "0123456789abcdef", "name": "catanrl-1v1-v2", "mode": "1v1", "elo_1v1": 1500}
    deploy.check_identity(entry, me)
    for wrong in ({"name": "catanrl-1v1"}, {"mode": "4p"}, {"id": "fedcba9876543210"}):
        with pytest.raises(DeployError, match="OPENHEX_KEY_1V1_V2 is the key of"):
            deploy.check_identity(entry, {**me, **wrong})
    unregistered = _parse(_bot())[1][0]  # no bot_id yet: name and mode decide
    deploy.check_identity(unregistered, {**me, "id": "fedcba9876543210"})


def test_verify_key_asks_bots_me_and_never_shows_the_key(monkeypatch):
    entry = _parse(_bot())[1][0]
    client = OpenHexClient("secret-key-123", "https://example.test/api")
    asked = []

    def request(method, path, body=None):
        asked.append((method, path))
        return {"id": "fedcba9876543210", "name": "catanrl-1v1", "mode": "1v1"}

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(DeployError) as error:
        deploy.verify_key(entry, client)
    assert asked == [("GET", "/bots/me")] and "'catanrl-1v1'" in str(error.value)
    assert "secret-key-123" not in str(error.value)

    def refused(method, path, body=None):
        raise OpenHexHTTPError(401, '{"detail": "Invalid API key"}')

    monkeypatch.setattr(client, "request", refused)
    with pytest.raises(DeployError, match="OpenHex refused it .HTTP 401") as error:
        deploy.verify_key(entry, client)
    assert "secret-key-123" not in str(error.value)


# ---------------------------------------------------------------- check and describe


def test_the_registration_body_fits_280_characters():
    entry = _parse(_bot(tag="openhex-1v1-v2"))[1][0]
    body = deploy.registration(entry, "0123456789abcdef0123")
    assert set(body) == {"name", "mode", "coordinateFormat", "description"}
    assert body["name"] == "catanrl-1v1-v2" and body["mode"] == "1v1" and body["coordinateFormat"] == "coordinates"
    text = body["description"]
    assert text == (f"catanrl 1v1-v2: {entry.spec}; code 0123456789ab (openhex-1v1-v2); "
                    "https://github.com/mcrco/catanrl-v2")
    long = _parse(_bot(spec="MCTS:ckpt=exp/v@e3#0123456789ab:" + ":".join(f"p{i}=1" for i in range(80))))[1][0]
    text = deploy.registration(long, "0123456789abcdef")["description"]
    assert len(text) <= 280 and text.endswith("; code 0123456789ab; https://github.com/mcrco/catanrl-v2")
    assert "..." in text and text.startswith("catanrl 1v1-v2: MCTS:ckpt=exp/v@e3#0123456789ab:p0=1")
    huge = _parse(_bot(version="v" * 400))[1][0]
    assert len(deploy.registration(huge, None)["description"]) <= 280


def test_describe_and_check_contact_no_one(tmp_path, monkeypatch, capsys):
    from catanrl import cli
    from catanrl.agents import spec as spec_module

    def offline(*args, **kwargs):
        raise AssertionError("contacted the network")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    path = tmp_path / "openhex.yaml"
    path.write_text(json.dumps({"bots": [
        _bot(),
        _bot(name="catanrl-4p", mode="4p", key_env="OPENHEX_KEY_4P", spec="POLICY:ckpt=ppo-4p/main@e3650#813b2c61338f"),
        _bot(name="catanrl-1v1", key_env="OPENHEX_KEY_1V1", spec="BELIEF:ckpt=puffer-graph2/full@best", retired=True),
    ]}))
    assert cli.main(["openhex", str(path), "--describe", "catanrl-1v1-v2"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["name"] == "catanrl-1v1-v2" and len(body["description"]) <= 280

    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", "secret-key-123")
    monkeypatch.delenv("OPENHEX_KEY_4P", raising=False)
    resolved = SimpleNamespace(canonical="x", checkpoints=(_ref(),))
    monkeypatch.setattr(spec_module, "resolve_agent", lambda spec, harness: resolved)
    monkeypatch.setattr(openhex_module, "native_for", lambda agent, device: (object(), "x"))
    monkeypatch.setattr(deploy, "git_stamps", lambda root, entries, **_: (
        {e.name: {"commit": "c" * 40, "describe": "openhex-1v1-v2", "dirty": False} for e in entries}, []))
    monkeypatch.setattr(cli, "_engine_build", dict)
    assert cli.main(["openhex", str(path), "--check"]) == 1
    out = capsys.readouterr().out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith("catanrl-")}
    assert rows["catanrl-1v1-v2"][-1] == "set" and rows["catanrl-4p"][-1] == "unset"
    assert rows["catanrl-1v1"][-2:] == ["yes", "(unset)"] and SHA[:12] in rows["catanrl-1v1-v2"]
    assert "problem: OPENHEX_KEY_4P (catanrl-4p's key) is not set" in out and "1 problem(s)" in out
    assert "secret-key-123" not in out
    monkeypatch.setenv("OPENHEX_KEY_4P", "another-key")
    assert cli.main(["openhex", str(path), "--check"]) == 0 and capsys.readouterr().out.rstrip().endswith("ok")
