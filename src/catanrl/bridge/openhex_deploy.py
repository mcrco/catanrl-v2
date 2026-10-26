"""Versions of our OpenHex bots: which model plays under which registration, from which code.

An OpenHex registration is permanent: its name and description are fixed when it
is made, no endpoint renames, edits or deletes one, and its Elo, record and
head-to-head belong to it. So a new model (or agent) gets a new registration, a
*version* (``catanrl-1v1-v2``), and the deploy file (``deploy/openhex.yaml``)
keeps every version ever registered, the retired ones too::

    bots:
      - name: catanrl-1v1-v2
        mode: 1v1
        version: 1v1-v2
        spec: BELIEF:ckpt=puffer-graph2/full@e3350#3355b5059503:worlds=16
        key_env: OPENHEX_KEY_1V1_V2
        bot_id: 0123456789abcdef       # OpenHex's id, once registered
        tag: openhex-1v1-v2            # the git tag it runs from
        evidence: [4b223fd273477fcf]   # eval records that justified it
        deployed: 2026-10-26
        retired: false                 # true: kept for the record, never run

This module reads that file (:func:`load_deploy`), refuses a checkpoint not
pinned by its weights (:func:`check_pins`), a checkout whose play source is not
the tag's or not clean (:func:`git_stamps`) and a key that is another bot's
(:func:`verify_key`), and stamps each bot's provenance (:func:`stamp`): every
game line names its stamp by :func:`provenance_id`, and
``<state>/<bot>/deploys.jsonl`` keeps the full stamps.

Nothing here contacts OpenHex but :func:`verify_key` (``GET /api/bots/me``) and
:func:`verify_keys`, which also takes the bots whose keys passed off the queue
when another key is refused.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catanrl.canon import digest

MODES = ("1v1", "4p")
#: The keys of a bot entry; any other is refused (a typo would otherwise be a silent default).
REQUIRED = ("name", "mode", "spec", "key_env", "version")
OPTIONAL = ("bot_id", "tag", "evidence", "retired", "deployed", "retired_at")
#: The keys of the file: ``releases`` names the GitHub repository whose releases ship the
#: checkpoints (``catanrl.bridge.openhex_release``; default :data:`REPO_URL`'s).
SETTINGS = ("bots", "api", "releases")
#: Tracked source that decides play (submodules included): the checkout must match its
#: commit, and the tag's, here. Everything else (deploy/, report/, runs/, transcripts/,
#: README.md, untracked files) may differ.
PLAY_PATHS = ("src", "engine", "third_party")
REPO_URL = "https://github.com/mcrco/catanrl-v2"
#: OpenHex's limits on a registration (POST /api/bots).
NAME_LENGTH = (3, 32)
DESCRIPTION_LENGTH = 280
BOT_ID_RE = re.compile(r"[0-9a-f]{16}")
#: OpenHex's API keys are ``uuid4().hex`` (upstream main.py, POST /api/bots).
KEY_RE = re.compile(r"[0-9a-f]{32}")
#: A ``#sha`` pin: the run-ref rule (``catanrl.runs.REF_RE``), for ``.pt`` paths too.
PIN_RE = re.compile(r"[0-9a-f]{6,64}")
ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class DeployError(RuntimeError):
    """A deploy file, checkpoint pin, checkout or key that the runner refuses."""


@dataclass(frozen=True)
class DeployEntry:
    """One registration on OpenHex and what plays under it."""

    name: str
    mode: str
    #: An agent spec, every checkpoint pinned by ``#sha``.
    spec: str
    #: The environment variable holding the registration's API key.
    key_env: str
    version: str
    #: OpenHex's id (16 hex digits), once registered.
    bot_id: str | None = None
    #: The git tag the entry runs from: the checkout's play source must be the tag's.
    tag: str | None = None
    #: Eval record ids that justified deploying it.
    evidence: tuple[str, ...] = ()
    #: Kept for the record (and ``stats``); never run.
    retired: bool = False
    deployed: str | None = None
    retired_at: str | None = None


# ---------------------------------------------------------------- the file


def load_deploy(path: str | Path) -> tuple[dict, list[DeployEntry]]:
    """(settings, entries) from a deploy file: ``bots`` (see :class:`DeployEntry`), an
    optional ``api`` (OpenHex's API base) and an optional ``releases`` (``owner/name``: the
    GitHub repository whose releases ship the checkpoints)."""
    import yaml

    return parse_deploy(yaml.safe_load(Path(path).read_text()), str(path))


def parse_deploy(data: Any, source: str = "deploy file") -> tuple[dict, list[DeployEntry]]:
    """:func:`load_deploy` on the parsed YAML; raises :class:`DeployError` listing every problem."""
    if not isinstance(data, Mapping) or not isinstance(data.get("bots"), list) or not data["bots"]:
        raise DeployError(f"{source}: expected a mapping with a non-empty list of bots")
    problems = [f"unknown key {key!r} (keys: {', '.join(SETTINGS)})" for key in data if key not in SETTINGS]
    if "api" in data and not isinstance(data["api"], str):
        problems.append(f"api must be a URL, not {data['api']!r}")
    if "releases" in data and not (isinstance(data["releases"], str) and REPO_RE.fullmatch(data["releases"])):
        problems.append(f"releases must be a GitHub repository (owner/name), not {data['releases']!r}")
    entries = []
    for i, raw in enumerate(data["bots"]):
        named = isinstance(raw, Mapping) and isinstance(raw.get("name"), str)
        where = f"bots[{i}]" + (f" ({raw['name']})" if named else "")
        if not isinstance(raw, Mapping):
            problems.append(f"{where}: expected a mapping, not {raw!r}")
            continue
        entry, errors = _entry(raw)
        problems += [f"{where}: {error}" for error in errors]
        if entry is not None:
            entries.append(entry)
    problems += _clashes(entries)
    if problems:
        raise DeployError(f"{source}:\n  " + "\n  ".join(problems))
    return {key: value for key, value in data.items() if key != "bots"}, entries


def _date(value: Any) -> str:
    """A free-form date as written (YAML reads ``2026-10-26`` as a date)."""
    return value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else str(value)


def _entry(raw: Mapping[str, Any]) -> tuple[DeployEntry | None, list[str]]:
    errors = [f"unknown key {key!r} (keys: {', '.join(REQUIRED + OPTIONAL)})" for key in raw
              if key not in REQUIRED + OPTIONAL]
    errors += [f"{key} is missing" for key in REQUIRED if raw.get(key) in (None, "")]
    for key in ("name", "spec", "key_env", "version", "bot_id", "tag"):
        if raw.get(key) is not None and not isinstance(raw[key], str):
            errors.append(f"{key} must be a string (quote it), not {raw[key]!r}")
    if errors:
        return None, errors
    name, mode = raw["name"], raw["mode"]
    low, high = NAME_LENGTH
    if not low <= len(name) <= high or name != name.strip():
        errors.append(f"name {name!r}: OpenHex names are {low}-{high} characters, no surrounding spaces")
    if mode not in MODES:
        errors.append(f"mode {mode!r}: one of {', '.join(MODES)}")
    if not ENV_RE.fullmatch(raw["key_env"]):
        errors.append(f"key_env {raw['key_env']!r} is not an environment variable name")
    if raw.get("bot_id") is not None and not BOT_ID_RE.fullmatch(raw["bot_id"]):
        errors.append(f"bot_id {raw['bot_id']!r}: OpenHex ids are 16 lowercase hex digits")
    evidence = raw.get("evidence") or []
    if not isinstance(evidence, list) or not all(isinstance(e, str) and e for e in evidence):
        errors.append(f"evidence must be a list of eval record ids, not {evidence!r}")
    retired = raw.get("retired", False)
    if not isinstance(retired, bool):
        errors.append(f"retired must be true or false, not {retired!r}")
    elif raw.get("retired_at") is not None and not retired:
        errors.append("retired_at is set but retired is not true")
    if errors:
        return None, errors
    return DeployEntry(
        name=name, mode=mode, spec=raw["spec"].strip(), key_env=raw["key_env"], version=raw["version"],
        bot_id=raw.get("bot_id"), tag=raw.get("tag"), evidence=tuple(evidence), retired=retired,
        deployed=None if raw.get("deployed") is None else _date(raw["deployed"]),
        retired_at=None if raw.get("retired_at") is None else _date(raw["retired_at"]),
    ), []


def _clashes(entries: Sequence[DeployEntry]) -> list[str]:
    """Names are unique (so is each (name, mode), which OpenHex requires); so is each key
    variable among the entries that run, and each OpenHex id."""
    def repeated(values) -> list:
        return sorted(value for value, n in Counter(values).items() if n > 1)

    out = [f"name {name!r} is used by more than one bot" for name in repeated(e.name for e in entries)]
    out += [f"key_env {env} is shared by bots that run ({', '.join(e.name for e in entries if e.key_env == env and not e.retired)})"
            for env in repeated(e.key_env for e in entries if not e.retired)]
    out += [f"bot_id {bot_id} is given to more than one bot" for bot_id in repeated(e.bot_id for e in entries if e.bot_id)]
    return out


def selected(entries: Sequence[DeployEntry], only: str | None = None) -> list[DeployEntry]:
    """The entries the runner runs: the ones not retired, or those ``only`` names
    (comma-separated), which must exist and not be retired."""
    if not only:
        return [e for e in entries if not e.retired]
    names = {name.strip() for name in only.split(",") if name.strip()}
    known = {e.name: e for e in entries}
    unknown = sorted(names - set(known))
    if unknown:
        raise DeployError(f"no bot named {', '.join(unknown)} (bots: {', '.join(known)})")
    retired = sorted(name for name in names if known[name].retired)
    if retired:
        raise DeployError(f"{', '.join(retired)} retired: a retired version does not run (set retired: false)")
    return [e for e in entries if e.name in names]


# ---------------------------------------------------------------- pinned checkpoints


def unpinned_refs(spec: str) -> list[str]:
    """The checkpoint refs of ``spec`` not pinned by their weights' sha: no ``#sha`` of
    6-64 hex digits (a ``.pt`` path's ``#`` or ``#3`` would pin nothing, or a digit)."""
    from catanrl.agents.spec import spec_ckpts

    return [ref for ref in spec_ckpts(spec) if not PIN_RE.fullmatch(ref.partition("#")[2])]


def pinned_ref(ref: str) -> str:
    """``ref`` as it resolves now, pinned: ``run@eN#sha`` (an alias such as ``best``
    moves; an epoch's file could be replaced), or ``/abs/path.pt#sha``. A short or
    empty pin on ``ref`` is dropped first."""
    from catanrl.runs import load_experiment, parse_ref

    ref = ref.partition("#")[0]
    if ref.endswith(".pt"):
        from catanrl.checkpoint import Checkpoint

        return str(Checkpoint.load(ref).ref)
    run, which, _ = parse_ref(ref)
    return str(load_experiment(run).resolve(which or "best"))


def pin_problems(entry: DeployEntry) -> list[str]:
    """What is unpinned in ``entry``'s spec, each ref with the pinned ref it resolves to now
    and the whole spec pinned, ready to paste."""
    from catanatron.registry import SpecError

    try:
        refs = unpinned_refs(entry.spec)
    except (SpecError, ValueError) as error:
        return [f"{entry.name}: bad spec {entry.spec!r}: {error}"]
    if not refs:
        return []
    spec, notes = entry.spec, []
    for ref in refs:
        try:
            pinned = pinned_ref(ref)
        except (RuntimeError, OSError, ValueError) as error:  # RunError, a missing or unreadable checkpoint
            notes.append(f"{ref} does not resolve ({error})")
            continue
        spec = re.sub(rf"(?<=[:=]){re.escape(ref)}(?=:|$)", lambda _, pinned=pinned: pinned, spec)
        notes.append(f"{ref} resolves to {pinned} now")
    return [f"{entry.name}: not pinned by #sha: {'; '.join(notes)}\n    spec: {spec}"]


def check_pins(entries: Sequence[DeployEntry], *, allow_unpinned: bool = False) -> list[str]:
    """Refuse a running entry whose spec names a checkpoint not pinned by ``#sha``; with
    ``allow_unpinned`` the problems come back as warnings. (A pin whose sha is not the
    weights on disk fails when the spec resolves: ``catanrl.runs.Run.resolve`` and
    ``catanrl.checkpoint.Checkpoint.load``.)"""
    problems = [p for e in entries if not e.retired for p in pin_problems(e)]
    if problems and not allow_unpinned:
        raise DeployError("\n".join(problems) + "\n(pin every ckpt by #sha, or pass --allow-unpinned)")
    return problems


# ---------------------------------------------------------------- the code


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DeployError(f"git {' '.join(args)} in {root}: {result.stderr.strip() or f'exit {result.returncode}'}")
    return result.stdout


def code_state(root: str | Path) -> dict:
    """The checkout at ``root``: HEAD, ``git describe --tags --always`` (without git's
    ``--dirty``, which counts every tracked file: :func:`git_stamps` adds ``-dirty`` from
    the play source), and the tracked play source (:data:`PLAY_PATHS`) that differs from
    HEAD: a modified file, or a submodule moved off its recorded commit or with modified
    files (untracked files, in a submodule too, do not count)."""
    root = Path(root)
    status = _git(root, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=untracked",
                  "--", *PLAY_PATHS)
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    return {"root": str(root), "commit": _git(root, "rev-parse", "HEAD").strip(),
            "describe": _git(root, "describe", "--tags", "--always").strip(), "changed": changed}


def tag_state(root: str | Path, tag: str, commit: str) -> dict:
    """``tag``'s commit and the play source that differs between it and ``commit``: none
    when ``commit`` is the tag's, or differs from it only outside :data:`PLAY_PATHS` (a
    report or deploy-file commit on top of the tag plays the tag's code)."""
    root = Path(root)
    tagged = _tag_commit(root, tag)
    differs = [] if tagged == commit else [
        line for line in _git(root, "diff", "--name-only", tagged, commit, "--", *PLAY_PATHS).splitlines() if line]
    return {"tag_commit": tagged, "differs": differs}


def git_stamps(root: str | Path, entries: Sequence[DeployEntry], *, allow_dirty: bool = False,
               engine_build: Mapping[str, Any] | None = None) -> tuple[dict[str, dict], list[str]]:
    """Each entry's ``git`` stamp (commit, describe, dirty, tag, tag_commit) and the problems
    found: the play source differs from HEAD, or HEAD's play source is not the entry's tag's,
    or the checkout cannot be read, or (given ``engine_build``, ``cppanatron.build_info()``)
    the engine was built from another tree than the checkout's. Raises :class:`DeployError`
    on a problem unless ``allow_dirty``, in which case the entries it concerns are stamped
    ``dirty: true`` (``dirty``: the code that plays is not exactly the declared code, the
    tag's or else HEAD's; ``describe`` ends in ``-dirty`` exactly then, so an edit outside
    the play source, such as the deploy file's, changes no stamp). A problem stops every
    entry, not only the one it concerns: one process runs one checkout."""
    problems, stamps = [], {}
    try:
        state = code_state(root)
    except DeployError as error:
        state = None
        problems.append(f"cannot read the checkout: {error}")
    stale = engine_problem(root, engine_build) if engine_build is not None and state is not None else None
    tags: dict[str, dict | DeployError] = {}
    for entry in entries:
        if state is None:
            stamps[entry.name] = {"commit": None, "describe": None, "dirty": True, "tag": entry.tag, "tag_commit": None}
            continue
        dirty = bool(state["changed"]) or stale is not None
        git = {"commit": state["commit"], "describe": state["describe"], "dirty": dirty, "tag": entry.tag,
               "tag_commit": None}
        if entry.tag:
            if entry.tag not in tags:
                try:
                    tags[entry.tag] = tag_state(root, entry.tag, state["commit"])
                except DeployError as error:
                    tags[entry.tag] = error
            found = tags[entry.tag]
            if isinstance(found, DeployError):
                problems.append(f"{entry.name}: {found}")
                git["dirty"] = True
            else:
                git["tag_commit"] = found["tag_commit"]
                if found["differs"]:
                    problems.append(f"{entry.name}: HEAD {state['commit'][:12]} does not play the code of {entry.tag} "
                                    f"({found['tag_commit'][:12]}): {_paths(found['differs'])} differ")
                    git["dirty"] = True
        if git["dirty"]:
            git["describe"] += "-dirty"
        stamps[entry.name] = git
    if stale is not None:
        problems.insert(0, stale)
    if state is not None and state["changed"]:
        problems.insert(0, f"the play source differs from HEAD {state['commit'][:12]}: {_paths(state['changed'])}")
    if problems and not allow_dirty:
        raise DeployError("\n".join(problems) + "\n(commit and tag the code that plays, check out the tag, "
                          "or pass --allow-dirty to run it anyway, stamped dirty)")
    return stamps, problems


def engine_problem(root: str | Path, build: Mapping[str, Any]) -> str | None:
    """The cppanatron build (``build_info()``) not made from the checkout's engine/cppanatron
    (as ``catanrl.provenance.check`` warns), or None (also when there is no engine checkout)."""
    engine = Path(root) / "engine" / "cppanatron"
    if not (engine / ".git").exists():
        return None
    head = _git(engine, "rev-parse", "HEAD").strip()
    if build.get("git_sha") == head and not build.get("git_dirty"):
        return None
    return (f"the cppanatron build ({str(build.get('git_sha'))[:12]}, dirty={build.get('git_dirty')}) is not "
            f"engine/cppanatron ({head[:12]}): rebuild it (uv sync)")


def _paths(paths: Sequence[str], n: int = 6) -> str:
    return ", ".join(paths[:n]) + (f" and {len(paths) - n} more" if len(paths) > n else "")


# ---------------------------------------------------------------- the key


def read_key(entry: DeployEntry, environ: Mapping[str, str] = os.environ) -> str:
    """``entry``'s API key from its variable. Refuses an unset one, and one that is not an
    OpenHex key (:data:`KEY_RE`), say several keys a grep joined by newlines, which an HTTP
    library would quote whole in its error; the refusal names the variable only."""
    key = environ.get(entry.key_env, "").strip()
    if not key:
        raise DeployError(f"{entry.key_env} (the API key of {entry.name}) is not set")
    if not KEY_RE.fullmatch(key):
        raise DeployError(f"{entry.key_env} (the API key of {entry.name}) is not an OpenHex key "
                          "(32 lowercase hex digits; the value is not shown)")
    return key


def key_state(entry: DeployEntry, environ: Mapping[str, str] = os.environ) -> str:
    """``set``, ``unset`` or ``malformed`` (:func:`read_key`), for ``--check``."""
    key = environ.get(entry.key_env, "").strip()
    return "unset" if not key else "set" if KEY_RE.fullmatch(key) else "malformed"


def check_identity(entry: DeployEntry, me: Mapping[str, Any]) -> None:
    """Refuse a key (``GET /api/bots/me``'s reply ``me``) that is not ``entry``'s registration:
    a key under the wrong variable would play this entry's model under another bot's name."""
    want = {"name": entry.name, "mode": entry.mode, **({"id": entry.bot_id} if entry.bot_id else {})}
    got = {key: me.get(key) for key in want}
    if got != want:
        raise DeployError(f"{entry.key_env} is the key of {me.get('name')!r} ({me.get('mode')}, id {me.get('id')}), "
                          f"not of {entry.name!r} ({entry.mode}{', id ' + entry.bot_id if entry.bot_id else ''})")


def verify_key(entry: DeployEntry, client: Any) -> dict:
    """``GET /api/bots/me`` with ``entry``'s key (``client``: an OpenHexClient), checked by
    :func:`check_identity`; OpenHex's reply. The key itself is never in a message."""
    from catanrl.bridge.openhex import OpenHexHTTPError

    try:
        me = client.me()
    except OpenHexHTTPError as error:
        raise DeployError(f"{entry.key_env} ({entry.name}'s key): OpenHex refused it (HTTP {error.status}: "
                          f"{error.detail[:200]})") from None
    except OSError as error:  # unreachable
        raise DeployError(f"{entry.key_env} ({entry.name}'s key) could not be checked: {error}") from None
    except ValueError as error:  # not JSON; or a key http.client refuses, whose message quotes the header, key and all
        raise DeployError(f"{entry.key_env} ({entry.name}'s key) could not be checked: "
                          f"{type(error).__name__}") from None
    if not isinstance(me, Mapping):
        raise DeployError(f"{entry.key_env} ({entry.name}'s key): GET /api/bots/me sent {str(me)[:200]!r}")
    check_identity(entry, me)
    return dict(me)


def verify_keys(entries: Sequence[DeployEntry], clients: Mapping[str, Any]) -> dict[str, dict]:
    """:func:`verify_key` for every entry (``clients`` by entry name); each one's reply. On a
    refusal, the entries whose keys passed leave OpenHex's queue before this raises: a
    restart that stops here leaves the last process's queue entries behind, and OpenHex
    would match bots that nothing runs (``POST /api/queue/leave`` only deletes the entry)."""
    replies, refused = {}, []
    for entry in entries:
        try:
            replies[entry.name] = verify_key(entry, clients[entry.name])
        except DeployError as error:
            refused.append(str(error))
    if not refused:
        return replies
    left = []
    for entry in entries:
        if entry.name not in replies:
            continue
        try:
            clients[entry.name].leave(entry.mode)
            left.append(entry.name)
        except Exception as error:  # noqa: BLE001 - say so and go on: the refusal is what matters
            refused.append(f"{entry.name}: could not leave the queue ({type(error).__name__}): leave it by hand")
    raise DeployError("\n".join(refused) + (f"\n(took {', '.join(left)} off the queue; " if left else "\n(") +
                      "a bot whose key was refused may still be queued from the last run: leave its queue by hand)")


def lock_bots(state: str | Path, entries: Sequence[DeployEntry]) -> list:
    """Hold ``<state>/<bot>/lock`` for each entry while this process runs (keep the list): a
    second process running one of them from the same state root is refused. Two processes
    with one key would both play its games and both append to its games.jsonl."""
    import fcntl

    held = []
    for entry in entries:
        where = Path(state) / entry.name
        where.mkdir(parents=True, exist_ok=True)
        file = open(where / "lock", "a")  # noqa: SIM115 (held while the process runs: the caller keeps it)
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            file.close()
            for other in held:
                other.close()
            raise DeployError(f"{entry.name} already runs from {state} in another process (one bot, one process: "
                              "run each from one service, with --only)") from None
        held.append(file)
    return held


# ---------------------------------------------------------------- provenance


def stamp(entry: DeployEntry, resolved: Any, git: Mapping[str, Any], *, code: Mapping[str, str],
          engine: Mapping[str, Any]) -> dict:
    """What a bot plays with, recorded once per start: its version and spec as configured,
    the spec as resolved (``canonical``: every param, checkpoints pinned), each checkpoint
    it loads (pinned ref and full weights sha256), the checkout (:func:`git_stamps`), the
    outcome code id (``catanrl.provenance.code_id("eval")``: models, agents, search, envs
    and the engine) and the engine build. The bridge itself is not in the code id: the
    commit and ``dirty`` cover it."""
    return {
        "name": entry.name, "mode": entry.mode, "version": entry.version, "bot_id": entry.bot_id,
        "spec": entry.spec, "canonical": resolved.canonical,
        "checkpoints": [{"ref": str(ref), "weights_sha256": ref.weights_sha256} for ref in resolved.checkpoints],
        "git": dict(git), "code_id": dict(code), "engine": dict(engine),
    }


def provenance_id(stamp: Mapping[str, Any]) -> str:
    """A short id of a stamp: equal stamps (whatever their key order) share it, so a restart
    at the same commit (edits outside the play source do not count), spec and weights keeps it."""
    return digest(stamp, 12)


def build_stamps(entries: Sequence[DeployEntry], resolved: Mapping[str, Any],
                 gits: Mapping[str, Mapping[str, Any]]) -> dict[str, dict]:
    """:func:`stamp` per entry (by name), with this process's code id and engine build.
    ``code_id`` is taken once, at startup: it refuses files changed since the process
    started, and nothing in a bot process checks it again."""
    from catanrl import provenance

    code, engine = provenance.code_id("eval"), provenance.engine_info()
    return {e.name: stamp(e, resolved[e.name], gits[e.name], code=code, engine=engine) for e in entries}


# ---------------------------------------------------------------- registering


def registration(entry: DeployEntry, commit: str | None) -> dict:
    """The body of ``POST /api/bots`` for ``entry``: its name, mode and a description (at most
    280 characters, fixed for good once registered) naming the version, the pinned spec,
    the code (``commit``, the tag's when the entry has one) and the repository."""
    code = f"code {commit[:12] if commit else 'unknown'}" + (f" ({entry.tag})" if entry.tag else "")
    head, tail = f"catanrl {entry.version}: ", f"; {code}; {REPO_URL}"
    room = DESCRIPTION_LENGTH - len(head) - len(tail)
    spec = entry.spec if len(entry.spec) <= room else entry.spec[: max(0, room - 3)] + "..."
    description = head + spec + tail
    if len(description) > DESCRIPTION_LENGTH:  # a very long version: keep what fits
        description = description[: DESCRIPTION_LENGTH - 3] + "..."
    assert len(description) <= DESCRIPTION_LENGTH
    return {"name": entry.name, "mode": entry.mode, "coordinateFormat": "coordinates", "description": description}


def entry_commit(root: str | Path, entry: DeployEntry) -> str:
    """The commit ``entry`` is deployed from: its tag's, else the checkout's HEAD."""
    if entry.tag:
        return _tag_commit(Path(root), entry.tag)
    return _git(Path(root), "rev-parse", "HEAD").strip()


def _tag_commit(root: Path, tag: str) -> str:
    try:
        return _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}").strip()
    except DeployError:
        raise DeployError(f"there is no tag {tag} in {root} (git fetch --tags?)") from None


def checkout_root() -> Path:
    """The checkout whose ``src/catanrl`` this process runs (an editable install)."""
    return Path(__file__).resolve().parents[3]


def weights_of(spec: str) -> str:
    """The ``#sha`` prefixes a spec pins, as written (``-`` if none): for an entry not resolved."""
    from catanrl.agents.spec import spec_ckpts

    try:
        shas = [ref.partition("#")[2] for ref in spec_ckpts(spec)]
    except Exception:  # noqa: BLE001 - a retired entry's spec may no longer parse; it is only shown
        return "-"
    return ",".join(sha or "-" for sha in shas) or "-"
