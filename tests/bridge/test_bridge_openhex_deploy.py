"""Versioned OpenHex deploys (catanrl.bridge.openhex_deploy, catanrl.bridge.openhex_stats):
the deploy file refuses typos, clashes and unpinned checkpoints (naming the pinned ref
to paste), the runner skips retired versions, refuses play source that is not clean
or not its tag's, and a key that belongs to another bot; every game line names its
version, spec and deploy stamp; the registration body fits OpenHex's 280 characters;
and the stats count a trimmed public record per version, per opponent and between
our own versions."""

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
from catanrl.bridge import openhex_stats as stats
from catanrl.bridge.openhex import OpenHexBot, OpenHexClient, OpenHexHTTPError
from catanrl.bridge.openhex_deploy import DeployEntry, DeployError
from catanrl.checkpoint import CheckpointRef
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.eval.stats import wilson_interval
from catanrl.models import build_model
from catanrl.runs import RunWriter

FIXTURE = Path(__file__).parent.parent / "fixtures" / "openhex" / "stats.json"
SHA = "3355b5059503541c228d3dd863fc01f9b6f213c099a2d006f32d6bddbd4fbba4"
#: Our three registrations in the fixture (OpenHex's public record of 2026-10-26).
PLAIN, SEARCH, FOURP = "05c4a25c6c0e46e1", "b8334c08b8d34a19", "068460ec188f4a2c"
#: Made-up API keys of OpenHex's shape (uuid4().hex).
KEY, KEY_4P = "0123456789abcdef" * 2, "fedcba9876543210" * 2


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

    path = tmp_path / "openhex.yaml"
    path.write_text(json.dumps({"bots": [
        _bot(bot_id="0123456789abcdef"),
        _bot(name="catanrl-1v1", version="1v1-v1", key_env="OPENHEX_KEY_1V1", retired=True,
             spec="BELIEF:ckpt=puffer-graph2/full@best:worlds=16"),  # retired: never pinned, never run
    ]}))
    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", KEY)
    monkeypatch.delenv("OPENHEX_KEY_1V1", raising=False)
    fake = _fake_openhex(monkeypatch)
    monkeypatch.setattr(provenance, "code_id", lambda kind: {"catanrl": "a" * 16, "engine": "b" * 16})
    monkeypatch.setattr(provenance, "engine_info", lambda: {"so_sha256": "d" * 64})
    assert cli.main(["openhex", str(path), "--state", str(tmp_path / "state")]) == 0
    assert [b.name for b in fake.started] == ["catanrl-1v1-v2"] and fake.asked == [KEY]
    assert fake.started[0].announced and fake.started[0].stamp["version"] == "1v1-v2"
    assert fake.started[0].stamp["code_id"] == {"catanrl": "a" * 16, "engine": "b" * 16}
    assert (tmp_path / "state" / "catanrl-1v1-v2" / "lock").exists() and fake.left == []


def _fake_openhex(monkeypatch, *, git_problem=None, me=None):
    """OpenHex's client, the bots, the agents and the engine build faked for ``cli.main``; git
    too, as the real ``git_stamps`` behaves: with ``git_problem``, it raises unless allowed
    dirty, and stamps dirty then. ``me``: the ``GET /api/bots/me`` reply per key."""
    from catanrl import cli
    from catanrl.agents import spec as spec_module

    fake = SimpleNamespace(started=[], asked=[], left=[])
    me = me or {KEY: {"id": "0123456789abcdef", "name": "catanrl-1v1-v2", "mode": "1v1"}}

    class Client:
        def __init__(self, key, base):
            self.key = key

        def me(self):
            fake.asked.append(self.key)
            return me[self.key]

        def leave(self, mode):
            fake.left.append((self.key, mode))

    class Bot:
        def __init__(self, spec, key, mode, *, name, stamp, **_):
            self.name, self.stamp = name, stamp
            fake.started.append(self)

        def announce(self):
            self.announced = True

        def run(self):
            return None

    def git_stamps(root, entries, *, allow_dirty=False, **_):
        if git_problem and not allow_dirty:
            raise DeployError(git_problem)
        dirty = bool(git_problem)
        return ({e.name: {"commit": "c" * 40, "describe": "openhex-1v1-v2" + "-dirty" * dirty, "dirty": dirty,
                          "tag": None, "tag_commit": None} for e in entries}, [git_problem] if git_problem else [])

    resolved = SimpleNamespace(canonical="BELIEF:ckpt=...", checkpoints=(_ref(),))
    monkeypatch.setattr(openhex_module, "OpenHexClient", Client)
    monkeypatch.setattr(openhex_module, "OpenHexBot", Bot)
    monkeypatch.setattr(openhex_module, "native_for", lambda agent, device: (object(), "x"))
    monkeypatch.setattr(spec_module, "resolve_agent", lambda spec, harness: resolved)
    monkeypatch.setattr(deploy, "git_stamps", git_stamps)
    monkeypatch.setattr(cli, "_engine_build", dict)
    return fake


def test_the_cli_runner_refuses_an_unpinned_spec_a_dirty_checkout_and_a_malformed_key(tmp_path, monkeypatch,
                                                                                     capsys):
    """The runner's own refusals, end to end: each stops it (exit 2) before OpenHex is asked
    anything; --allow-dirty runs a dirty checkout, stamped dirty; a malformed key is not shown."""
    from catanrl import cli, provenance

    monkeypatch.setattr(provenance, "code_id", lambda kind: {"catanrl": "a" * 16})
    monkeypatch.setattr(provenance, "engine_info", dict)
    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", KEY)
    path = tmp_path / "openhex.yaml"
    state = ["--state", str(tmp_path / "state")]

    path.write_text(json.dumps({"bots": [_bot(spec="BELIEF:ckpt=puffer-graph2/full@best:worlds=16")]}))
    fake = _fake_openhex(monkeypatch)
    assert cli.main(["openhex", str(path), *state]) == 2
    assert "not pinned by #sha" in capsys.readouterr().err and fake.asked == [] and fake.started == []

    path.write_text(json.dumps({"bots": [_bot()]}))
    fake = _fake_openhex(monkeypatch, git_problem="the play source differs from HEAD cccccccccccc: src/x.py")
    assert cli.main(["openhex", str(path), *state]) == 2
    assert "the play source differs" in capsys.readouterr().err and fake.asked == [] and fake.started == []
    assert cli.main(["openhex", str(path), *state, "--allow-dirty"]) == 0
    assert fake.asked == [KEY] and fake.started[0].stamp["git"]["dirty"]

    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", f"{KEY}\n{KEY_4P}")  # say, a grep that matched two variables
    fake = _fake_openhex(monkeypatch)
    assert cli.main(["openhex", str(path), *state]) == 2
    err = capsys.readouterr().err
    assert "OPENHEX_KEY_1V1_V2 (the API key of catanrl-1v1-v2) is not an OpenHex key" in err
    assert KEY not in err and KEY_4P not in err and fake.asked == []


def test_a_refused_key_stops_the_runner_and_takes_the_other_bots_off_the_queue(tmp_path, monkeypatch, capsys):
    """A restart whose key check fails would leave the last process's queue entries behind:
    the bots whose keys passed leave the queue, so OpenHex matches none of them to no one."""
    from catanrl import cli

    path = tmp_path / "openhex.yaml"
    path.write_text(json.dumps({"bots": [_bot(), _bot(name="catanrl-4p", mode="4p", key_env="OPENHEX_KEY_4P")]}))
    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", KEY)
    monkeypatch.setenv("OPENHEX_KEY_4P", KEY_4P)
    fake = _fake_openhex(monkeypatch, me={KEY: {"id": "0123456789abcdef", "name": "catanrl-1v1-v2", "mode": "1v1"},
                                          KEY_4P: {"id": PLAIN, "name": "catanrl-1v1", "mode": "1v1"}})
    assert cli.main(["openhex", str(path), "--state", str(tmp_path / "state")]) == 2
    err = capsys.readouterr().err
    assert "OPENHEX_KEY_4P is the key of 'catanrl-1v1'" in err and "took catanrl-1v1-v2 off the queue" in err
    assert fake.asked == [KEY, KEY_4P] and fake.left == [(KEY, "1v1")] and fake.started == []
    assert KEY not in err and KEY_4P not in err


def test_one_bot_runs_in_one_process(tmp_path):
    """Two processes with one key would both play its games: the second is refused."""
    _, entries = _parse(_bot(), _bot(name="catanrl-4p", mode="4p", key_env="K4"))
    held = deploy.lock_bots(tmp_path, entries[1:])
    with pytest.raises(DeployError, match="catanrl-4p already runs from"):
        deploy.lock_bots(tmp_path, entries)
    first = deploy.lock_bots(tmp_path, entries[:1])  # the refused call let go of what it had taken
    for file in held + first:
        file.close()
    assert len(deploy.lock_bots(tmp_path, entries)) == 2


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


def test_an_empty_or_short_pin_pins_nothing(run):
    """``x.pt#`` checks no sha and ``x.pt#3`` one digit (``load_checkpoint`` takes any prefix of
    a path's sha): a pin is 6-64 hex digits, the run-ref rule."""
    sha = run.best.weights_sha256
    pinned = deploy.pinned_ref(str(run.best.path))
    assert pinned.endswith(f"#{sha[:12]}")
    for ref in (f"{run.best.path}#", f"{run.best.path}#{sha[0]}", f"exp/v@e3#{sha[:5]}"):
        problems = deploy.pin_problems(_parse(_bot(spec=f"POLICY:ckpt={ref}"))[1][0])
        assert len(problems) == 1 and "not pinned by #sha" in problems[0], ref
        assert f"{ref} resolves to " in problems[0] and "does not resolve" not in problems[0]
    for ref in (f"{run.best.path}#{sha[:12]}", f"exp/v@e3#{sha[:6]}", f"exp/v@e3#{sha}"):
        assert deploy.pin_problems(_parse(_bot(spec=f"POLICY:ckpt={ref}"))[1][0]) == [], ref


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
    assert stamps["bot-0"]["describe"] == "v1"  # not git's --dirty, which counts every tracked file


def test_editing_the_deploy_file_in_place_changes_no_stamp(repo):
    """Adding a bot_id or retiring a version edits deploy/openhex.yaml on the bot machine: the
    bots that keep running keep their provenance_id across the restart."""
    entry = _parse(_bot(tag="v1"))[1][0]
    before = deploy.provenance_id(_stamp(entry, **deploy.git_stamps(repo.root, [entry])[0][entry.name]))
    (repo.root / "deploy" / "openhex.yaml").write_text("bots: [{retired: true}]\n")
    after = deploy.git_stamps(repo.root, [entry])[0][entry.name]
    assert after["describe"] == "v1" and deploy.provenance_id(_stamp(entry, **after)) == before


def test_a_dirty_play_source_is_refused_unless_allowed(repo):
    (repo.root / "src" / "catanrl" / "a.py").write_text("A = 2\n")
    with pytest.raises(DeployError, match="the play source differs from HEAD .*src/catanrl/a.py"):
        deploy.git_stamps(repo.root, _entries(None))
    stamps, problems = deploy.git_stamps(repo.root, _entries(None), allow_dirty=True)
    assert stamps["bot-0"]["dirty"] and stamps["bot-0"]["describe"] == "v1-dirty" and len(problems) == 1


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
    assert (stamps["bot-0"]["describe"], stamps["bot-1"]["describe"]) == ("v2-dirty", "v2")


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


def test_a_key_http_client_refuses_is_never_shown(monkeypatch):
    """http.client refuses a header with a newline by quoting it whole (``Invalid header value
    b'Bearer <key>'``): verify_key names the error's type only. Nothing is sent (the header is
    refused before any connection; a connection here fails the test)."""
    import socket

    def no_connection(*args, **kwargs):
        raise AssertionError("connected")

    monkeypatch.setattr(socket, "create_connection", no_connection)
    entry = _parse(_bot())[1][0]
    client = OpenHexClient(f"{KEY}\n{KEY_4P}", "http://127.0.0.1:9/api")
    with pytest.raises(ValueError, match=f"Invalid header value .*{KEY_4P}"):
        client.me()  # what verify_key's message used to quote
    with pytest.raises(DeployError, match="could not be checked: ValueError") as error:
        deploy.verify_key(entry, client)
    assert KEY not in str(error.value) and KEY_4P not in str(error.value)


def test_read_key_refuses_an_unset_or_malformed_key_naming_only_its_variable():
    entry = _parse(_bot())[1][0]
    assert deploy.read_key(entry, {"OPENHEX_KEY_1V1_V2": f" {KEY}\n"}) == KEY
    assert deploy.key_state(entry, {"OPENHEX_KEY_1V1_V2": KEY}) == "set"
    for value, says, state in (("", "is not set", "unset"), (f"{KEY}\n{KEY_4P}", "is not an OpenHex key", "malformed"),
                               (KEY.upper(), "is not an OpenHex key", "malformed")):
        with pytest.raises(DeployError, match=says) as error:
            deploy.read_key(entry, {"OPENHEX_KEY_1V1_V2": value})
        assert "OPENHEX_KEY_1V1_V2" in str(error.value) and KEY not in str(error.value).lower()
        assert deploy.key_state(entry, {"OPENHEX_KEY_1V1_V2": value}) == state


def test_verify_keys_checks_every_key_and_says_when_a_bot_could_not_leave_the_queue():
    _, entries = _parse(_bot(), _bot(name="catanrl-4p", mode="4p", key_env="K4"),
                        _bot(name="catanrl-1v1-v3", key_env="K3"))

    class Client:
        def __init__(self, me, leave_fails=False):
            self.reply, self.leave_fails, self.left = me, leave_fails, []

        def me(self):
            return self.reply

        def leave(self, mode):
            if self.leave_fails:
                raise OSError("unreachable")
            self.left.append(mode)

    good = {e.name: Client({"name": e.name, "mode": e.mode, "id": "0" * 16}) for e in entries}
    assert set(deploy.verify_keys(entries, good)) == {e.name for e in entries}
    assert all(c.left == [] for c in good.values())  # all passed: nobody leaves
    clients = {**good, "catanrl-4p": Client({"name": "catanrl-1v1", "mode": "1v1"}),
               "catanrl-1v1-v3": Client(good["catanrl-1v1-v3"].reply, leave_fails=True)}
    with pytest.raises(DeployError) as error:
        deploy.verify_keys(entries, clients)
    text = str(error.value)
    assert "K4 is the key of 'catanrl-1v1'" in text and "took catanrl-1v1-v2 off the queue" in text
    assert "catanrl-1v1-v3: could not leave the queue (OSError): leave it by hand" in text
    assert clients["catanrl-1v1-v2"].left == ["1v1"] and clients["catanrl-4p"].left == []


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

    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", KEY)
    monkeypatch.delenv("OPENHEX_KEY_4P", raising=False)
    fake = _fake_openhex(monkeypatch)
    assert cli.main(["openhex", str(path), "--check"]) == 1
    out = capsys.readouterr().out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith("catanrl-")}
    assert rows["catanrl-1v1-v2"][-1] == "set" and rows["catanrl-4p"][-1] == "unset"
    assert rows["catanrl-1v1"][-2:] == ["yes", "(unset)"] and SHA[:12] in rows["catanrl-1v1-v2"]
    assert "problem: OPENHEX_KEY_4P (catanrl-4p's key) is not set" in out and "1 problem(s)" in out
    assert KEY not in out
    monkeypatch.setenv("OPENHEX_KEY_4P", f"{KEY_4P}\n{KEY}")  # two keys a grep joined
    assert cli.main(["openhex", str(path), "--check"]) == 1
    out = capsys.readouterr().out
    assert "problem: OPENHEX_KEY_4P (catanrl-4p's key) is not an OpenHex key" in out and KEY_4P not in out
    monkeypatch.setenv("OPENHEX_KEY_4P", KEY_4P)
    assert cli.main(["openhex", str(path), "--check"]) == 0 and capsys.readouterr().out.rstrip().endswith("ok")
    assert fake.asked == [] and fake.left == []  # --check asks OpenHex nothing (without --keys)


def test_check_refuses_an_unpinned_spec_with_its_pinned_ref_unless_allowed(run, tmp_path, monkeypatch, capsys):
    from catanrl import cli

    pinned = f"exp/v@e3#{run.best.weights_sha256[:12]}"
    path = tmp_path / "openhex.yaml"
    path.write_text(json.dumps({"bots": [_bot(spec="BELIEF:ckpt=exp/v@best:worlds=16")]}))
    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", KEY)
    _fake_openhex(monkeypatch)
    assert cli.main(["openhex", str(path), "--check"]) == 1
    out = capsys.readouterr().out
    assert f"problem: catanrl-1v1-v2: not pinned by #sha: exp/v@best resolves to {pinned} now" in out
    assert f"spec: BELIEF:ckpt={pinned}:worlds=16" in out and out.rstrip().endswith("1 problem(s)")
    assert cli.main(["openhex", str(path), "--check", "--allow-unpinned"]) == 0
    out = capsys.readouterr().out
    assert f"warning: catanrl-1v1-v2: not pinned by #sha: exp/v@best resolves to {pinned} now" in out
    assert out.rstrip().endswith("ok")


def test_check_refuses_a_dirty_checkout_unless_allowed(tmp_path, monkeypatch, capsys):
    from catanrl import cli

    path = tmp_path / "openhex.yaml"
    path.write_text(json.dumps({"bots": [_bot()]}))
    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", KEY)
    _fake_openhex(monkeypatch, git_problem="the play source differs from HEAD cccccccccccc: src/x.py")
    assert cli.main(["openhex", str(path), "--check"]) == 1
    assert "problem: the play source differs" in capsys.readouterr().out
    assert cli.main(["openhex", str(path), "--check", "--allow-dirty"]) == 0
    out = capsys.readouterr().out
    assert "warning: the play source differs" in out and out.rstrip().endswith("ok")


def test_check_keys_asks_openhex_whose_each_key_is_and_queues_nothing(tmp_path, monkeypatch, capsys):
    from catanrl import cli

    path = tmp_path / "openhex.yaml"
    path.write_text(json.dumps({"bots": [_bot(), _bot(name="catanrl-4p", mode="4p", key_env="OPENHEX_KEY_4P"),
                                         _bot(name="catanrl-1v1", key_env="OPENHEX_KEY_1V1", retired=True)]}))
    monkeypatch.setenv("OPENHEX_KEY_1V1_V2", KEY)
    monkeypatch.setenv("OPENHEX_KEY_4P", KEY_4P)
    fake = _fake_openhex(monkeypatch, me={KEY: {"id": "0123456789abcdef", "name": "catanrl-1v1-v2", "mode": "1v1"},
                                          KEY_4P: {"id": PLAIN, "name": "catanrl-1v1", "mode": "1v1"}})
    assert cli.main(["openhex", str(path), "--check", "--keys"]) == 1
    out = capsys.readouterr().out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith("catanrl-")}
    assert rows["catanrl-1v1-v2"][-1] == "verified" and rows["catanrl-4p"][-1] == "REFUSED"
    assert "problem: OPENHEX_KEY_4P is the key of 'catanrl-1v1'" in out
    assert fake.asked == [KEY, KEY_4P] and fake.left == [] and fake.started == []  # the retired one is not asked
    assert KEY not in out and KEY_4P not in out
    with pytest.raises(SystemExit, match="--keys goes with --check"):
        cli.main(["openhex", str(path), "--keys"])


# ---------------------------------------------------------------- stats


def _fixture():
    return json.loads(FIXTURE.read_text())


class _Records:
    """OpenHex's public records, served from the fixture."""

    def __init__(self):
        data = _fixture()
        self.bots = {doc["name"].rsplit("/", 1)[1]: stats.document(doc) for doc in data["bots"]}
        self.rows = data["games"]

    def bot(self, bot_id):
        return self.bots.get(bot_id)

    def find_bot(self, name, mode):
        return next((b for b in self.bots.values() if b["name"] == name and b["mode"] == mode), None)

    def games(self, bot_id):
        return [stats.document(row["document"]) for row in self.rows.get(bot_id, [])]


def _v1_entries():
    return _parse(
        _bot(name="catanrl-1v1", version="1v1-v1", key_env="OPENHEX_KEY_1V1", bot_id=PLAIN),
        _bot(name="catanrl-4p", mode="4p", version="4p-v1", key_env="OPENHEX_KEY_4P", bot_id=FOURP),
        _bot(name="catanrl-1v1-search", version="1v1-search-v1", key_env="OPENHEX_KEY_1V1_SEARCH"),  # no bot_id
        _bot(name="catanrl-1v1-v9", version="1v1-v9", key_env="OPENHEX_KEY_V9", retired=True),  # never registered
    )[1]


def test_firestore_values_decode():
    doc = {"name": "projects/open-hex/databases/(default)/documents/games/abc",
           "fields": {"n": {"integerValue": "3"}, "x": {"doubleValue": 1.5}, "s": {"stringValue": "a"},
                      "z": {"nullValue": None}, "b": {"booleanValue": True},
                      "l": {"arrayValue": {"values": [{"stringValue": "p"}, {"mapValue": {"fields": {"k": {"integerValue": "1"}}}}]}},
                      "e": {"arrayValue": {}}}}
    assert stats.document(doc) == {"id": "abc", "n": 3, "x": 1.5, "s": "a", "z": None, "b": True, "l": ["p", {"k": 1}],
                                   "e": []}


def test_stats_count_versions_opponents_and_our_meetings():
    entries = _v1_entries()
    bots, games = stats.gather(entries, _Records())
    assert bots["catanrl-1v1-search"]["id"] == SEARCH and bots["catanrl-1v1-v9"] is None  # found by name and mode
    games[SEARCH].append({"id": "live", "mode": "1v1", "status": "active", "winnerBotId": None,
                          "createdAt": "2026-10-26T00:00:00+00:00", "players": [{"botId": SEARCH, "name": "x"},
                                                                                {"botId": "f" * 16, "name": "y"}]})
    summary = stats.summarize(entries, bots, games)
    plain, fourp, search, never = summary["versions"]
    assert {k: plain[k] for k in ("games", "wins", "aborted", "ongoing", "first", "last")} == {
        "games": 8, "wins": 2, "aborted": 2, "ongoing": 0,
        "first": "2026-10-17T18:49:02.863543+00:00", "last": "2026-10-20T07:02:59.377650+00:00"}
    assert plain["wilson"] == wilson_interval(2, 8) and plain["rate"] == 0.25
    assert round(plain["elo"]) == 1388 and plain["ladder"] == {"games": 109, "wins": 54}
    assert {k: search[k] for k in ("games", "wins", "aborted", "ongoing")} == {"games": 7, "wins": 4, "aborted": 1,
                                                                                 "ongoing": 1}
    assert search["first"][:10] == "2026-10-20" and search["last"][:10] == "2026-10-23"
    assert fourp["found"] and fourp["games"] == 0 and fourp["rate"] is None and fourp["elo"] == 1500
    assert not never["found"] and never["retired"]
    assert summary["opponents"]["catanrl-1v1-search"][:2] == [
        {"opponent": "hexn-4d.0", "bot_id": "aa627d8cbafb4011", "games": 3, "wins": 0},
        {"opponent": "catanrl-1v1", "bot_id": PLAIN, "games": 2, "wins": 2}]
    assert {o["opponent"]: (o["games"], o["wins"]) for o in summary["opponents"]["catanrl-1v1"]} == {
        "rehex-2.11": (2, 1), "catanrl-1v1-search": (2, 0), "colonist-ai-r66": (1, 1), "ar-PX_2": (1, 0),
        "ar-PXns-a": (1, 0), "ar-PX_b0": (1, 0)}
    assert summary["head_to_head"] == [{"a": "catanrl-1v1", "b": "catanrl-1v1-search", "games": 2, "a_wins": 0,
                                        "b_wins": 2, "other_wins": 0}]  # each meeting once, from both sides
    text = stats.format_summary(summary)
    assert "catanrl-1v1 (1v1-v1) vs catanrl-1v1-search (1v1-search-v1): 2 games, 0-2" in text
    assert "not on OpenHex" in text and "note: catanrl-1v1: OpenHex's own record says 109 games, 54 wins" in text
    low, high = wilson_interval(2, 8)
    assert f"[{100 * low:.1f}, {100 * high:.1f}]" in text


def test_public_records_page_through_a_bots_games(monkeypatch):
    rows = _fixture()["games"][SEARCH]
    asked = []

    def fetch(self, url, body=None):
        asked.append((url, body))
        if body is None:  # bots/<id>, or the bots list
            if "/bots/" in url:
                return next(b for b in _fixture()["bots"] if b["name"].endswith(url.rsplit("/", 1)[1]))
            return {"documents": [{"name": b["name"], "fields": {k: b["fields"][k] for k in ("name", "mode")}}
                                  for b in _fixture()["bots"]]}
        query = body["structuredQuery"]
        assert query["where"]["fieldFilter"]["value"] == {"stringValue": SEARCH}
        after = (query.get("startAt") or {"values": [{"referenceValue": ""}]})["values"][0]["referenceValue"]
        rest = [r for r in rows if r["document"]["name"] > after]
        return [{"readTime": "t", **r} for r in rest[: query["limit"]]] or [{"readTime": "t"}]

    monkeypatch.setattr(stats, "PAGE", 3)
    monkeypatch.setattr(stats.PublicRecords, "_fetch", fetch)
    records = stats.PublicRecords()
    games = records.games(SEARCH)
    assert [g["id"] for g in games] == [r["document"]["name"].rsplit("/", 1)[1] for r in rows]
    assert len([a for a in asked if a[1] is not None]) == 3  # 3 + 3 + 1 of 7
    assert records.find_bot("catanrl-1v1-search", "1v1")["games1v1"] == 274
    assert records.find_bot("catanrl-1v1-search", "4p") is None


def _game(game_id, me, other, name, created, winner, status="completed"):
    return {"id": game_id, "mode": "1v1", "status": status, "winnerBotId": winner, "createdAt": created,
            "completedAt": created, "players": [{"botId": me, "name": "us"}, {"botId": other, "name": name}],
            "playerBotIds": [me, other]}


def test_stats_count_a_window_of_days_only():
    entries = _v1_entries()
    bots, games = stats.gather(entries, _Records())
    since = stats.parse_when("2026-10-20")
    assert since == datetime.datetime(2026, 10, 20, tzinfo=datetime.UTC)
    assert stats.parse_when("2026-10-18", end=True) == datetime.datetime(2026, 10, 19, tzinfo=datetime.UTC)
    assert stats.parse_when("2026-10-20T06:30+02:00").utcoffset() == datetime.timedelta(hours=2)
    summary = stats.summarize(entries, bots, games, since=since)
    plain, _, search, _ = summary["versions"]
    assert (plain["games"], plain["wins"], plain["first"][:10]) == (2, 0, "2026-10-20")  # its meetings with search
    assert search["games"] == 7 and summary["opponents"]["catanrl-1v1"] == [
        {"opponent": "catanrl-1v1-search", "bot_id": SEARCH, "games": 2, "wins": 0}]
    early = stats.summarize(entries, bots, games, until=stats.parse_when("2026-10-18", end=True))
    assert early["versions"][0]["games"] == 6 and early["versions"][2]["games"] == 0 and early["head_to_head"] == []
    text = stats.format_summary(summary)
    assert text.startswith("games created from 2026-10-20 00:00 UTC only")
    assert "OpenHex's own record says" not in text  # the ladder's record is all-time: no window to compare


def test_stats_compare_overlapping_versions_per_opponent_over_the_overlap_alone():
    """The overlap is what the two versions' records share: the later one's first game to the
    earlier one's last, our own meetings left out (they are head-to-head)."""
    entries = _v1_entries()
    bots, games = stats.gather(entries, _Records())
    summary = stats.summarize(entries, bots, games)
    (overlap,) = summary["overlaps"]  # the 4p bot never played; the retired one is not on OpenHex
    assert (overlap["a"], overlap["b"], overlap["opponents"]) == ("catanrl-1v1", "catanrl-1v1-search", [])
    assert overlap["start"].startswith("2026-10-20T06:13:00") and overlap["end"].startswith("2026-10-20T06:49:01")
    assert "no games against other bots in the overlap" in stats.format_summary(summary)
    rehex, hexn = "dcc6de42b64b4b78", "aa627d8cbafb4011"
    games[PLAIN] += [_game("p1", PLAIN, rehex, "rehex-2.11", "2026-10-20T06:30:00+00:00", PLAIN),
                     _game("p2", PLAIN, rehex, "rehex-2.11", "2026-10-20T06:35:00+00:00", rehex, "aborted")]
    games[SEARCH] += [_game("s1", SEARCH, rehex, "rehex-2.11", "2026-10-20T06:40:00+00:00", SEARCH),
                      _game("s2", SEARCH, hexn, "hexn-4d.0", "2026-10-20T06:20:00+00:00", hexn),
                      _game("s3", SEARCH, rehex, "rehex-2.11", "2026-10-20T06:55:00+00:00", SEARCH)]  # after v1's last
    (overlap,) = stats.summarize(entries, bots, games)["overlaps"]
    assert overlap["opponents"] == [
        {"opponent": "rehex-2.11", "bot_id": rehex, "a_games": 2, "a_wins": 1, "b_games": 1, "b_wins": 1},
        {"opponent": "hexn-4d.0", "bot_id": hexn, "a_games": 0, "a_wins": 0, "b_games": 1, "b_wins": 0}]
    text = stats.format_summary(stats.summarize(entries, bots, games))
    assert ("overlap of catanrl-1v1 (1v1-v1) and catanrl-1v1-search (1v1-search-v1), 2026-10-20 06:13 to "
            "2026-10-20 06:49 UTC") in text
    assert "\n  opponent    catanrl-1v1  catanrl-1v1-search\n  rehex-2.11  1 of 2       1 of 1\n  hexn-4d.0   -            0 of 1" in text


def test_stats_refuse_a_bot_id_that_is_another_registration():
    entries = _parse(_bot(name="catanrl-1v1", version="1v1-v1", key_env="K1", bot_id=SEARCH, retired=True),
                     _bot(name="catanrl-4p", mode="1v1", version="4p-v1", key_env="K4", bot_id=FOURP))[1]
    bots, games = stats.gather(entries, _Records())
    summary = stats.summarize(entries, bots, games)
    assert [v["found"] for v in summary["versions"]] == [False, False] and summary["head_to_head"] == []
    text = stats.format_summary(summary)
    assert (f"note: catanrl-1v1: bot_id {SEARCH} is 'catanrl-1v1-search' (1v1) on OpenHex, not this entry: "
            "not counted") in text
    assert f"note: catanrl-4p: bot_id {FOURP} is 'catanrl-4p' (4p) on OpenHex" in text  # a mode that is not ours


def test_stats_escape_names_from_openhex():
    """Anyone may register a bot, so an opponent's name can hold a terminal escape (here OSC 52,
    a clipboard write): it is printed escaped."""
    entries = _v1_entries()
    bots, games = stats.gather(entries, _Records())
    evil = "\x1b]52;c;cm0gLXJmIH4K\x07x"
    games[PLAIN].append(_game("e1", PLAIN, "e" * 16, evil, "2026-10-19T00:00:00+00:00", PLAIN))
    text = stats.format_summary(stats.summarize(entries, bots, games))
    assert "\x1b" not in text and "\x07" not in text and "\\x1b]52;c;cm0gLXJmIH4K\\x07x" in text


def test_the_stats_cli_takes_a_window(tmp_path, monkeypatch, capsys):
    from catanrl import cli

    monkeypatch.setattr(stats, "PublicRecords", _Records)
    path = tmp_path / "openhex.yaml"
    path.write_text(json.dumps({"bots": [_bot(name="catanrl-1v1", version="1v1-v1", key_env="K1", bot_id=PLAIN)]}))
    assert cli.main(["openhex", "stats", str(path), "--since", "2026-10-20"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("games created from 2026-10-20 00:00 UTC only")
    row = next(line.split() for line in out.splitlines() if line.startswith("1v1-v1"))
    assert row[3:5] == ["2", "0"]
    with pytest.raises(SystemExit, match="--since 'soon'"):
        cli.main(["openhex", "stats", str(path), "--since", "soon"])
    with pytest.raises(SystemExit, match="--since and --until are for catanrl openhex stats"):
        cli.main(["openhex", str(path), "--check", "--until", "2026-10-20"])
