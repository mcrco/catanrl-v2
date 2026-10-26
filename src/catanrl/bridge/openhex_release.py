"""Our OpenHex bots' checkpoints as GitHub release assets: built here, fetched on the deploy machine.

A version runs from a checkout of its git tag (:mod:`catanrl.bridge.openhex_deploy`), but
the checkpoints its spec pins are not in git. ``release`` packs them for a GitHub release
named like the tag, on the public repository (the deploy file's ``releases:``, default
``mcrco/catanrl-v2``); ``fetch`` downloads them over plain HTTPS (no ``gh``, no token)
into the deploy machine's runs root::

    catanrl openhex release deploy/openhex.yaml openhex-1v1-v2 [--out DIR]   # builds; publishes nothing
    gh release create openhex-1v1-v2 ...                                      # the command release prints
    catanrl openhex fetch deploy/openhex.yaml [--runs DIR] [--only NAMES]     # on the deploy machine

**An asset** is one checkpoint, ``<exp>--<variant>--e<NNNNNN>-<sha12>.tar.gz``, holding
``runs/<exp>/<variant>/`` with what :class:`catanrl.runs.Run` needs to resolve a pinned
``exp/variant@eN#sha`` and nothing else: ``config.json`` as it is, the checkpoint, and a
``run.json`` that lists that checkpoint alone (``"status": "released"``, no launches: their
argv, hosts and times stay home). No ``aliases.jsonl``, no metrics, evals, provenance or
trainer state. A fetched run resolves only the epochs it holds: ``@best`` is "no checkpoint
'best'" there, and ``@latest`` would be the latest fetched, so a release takes refs of the
form ``exp/variant@eN#sha`` only (it says which to paste). The tarball is deterministic
(sorted members, mtime 0, owner 0, mode 644, gzip mtime 0): one checkpoint packs to the
same bytes in every release. ``manifest.json`` lists the release's bots and its assets
with their sha256; ``release-notes.md`` is the release's text. A tag whose bots pin no
checkpoint has nothing to release and is refused. The printed ``gh release create`` passes
``--verify-tag``: gh refuses a tag that is not on GitHub instead of creating it at the
default branch's head (which the manifest's ``commit`` would not name).

Before packing, every file that goes in is scanned for secrets (:data:`SECRETS`); a hit
refuses the release, naming the file and line (byte offset in a checkpoint), never the
value. Absolute home paths are allowed and listed, in the assets, manifest.json and the
notes: a run's config.json names the YAML it was made from, and it becomes public.

**Fetching** takes the bots that run (not retired) and have a tag; a bot whose every
pinned checkpoint already resolves in the runs root needs nothing (no request at all).
For the others it reads ``manifest.json`` of their tag's release, checks it (schema, tag,
repository, every pinned ref served by an asset), downloads each asset it needs (size
capped, redirects only to https), checks its sha256, inflates it within
:data:`ASSET_BYTES` (a tar header cannot claim more memory than that), reads its members
(regular files only, exactly ``config.json``, ``run.json`` and ``checkpoints/eNNNNNN.pt``
under the asset's ``runs/<exp>/<variant>/``; nothing absolute, no ``..``, no links) and
the checkpoint's weights against the manifest, all before writing anything. Whatever a
download holds, a fetch refuses it with one message, never a traceback. Then it merges
each run into the runs root, all or nothing:

- ``config.json``: written if missing, else it must be the same JSON (another run of that
  name is refused);
- ``checkpoints/eNNNNNN.pt``: written if missing, else it must be the same bytes (a file
  is never overwritten);
- ``run.json``: written if missing; kept if it already lists the checkpoint (the same
  row); extended if it is one a fetch wrote (``"status": "released"``, no launches);
  refused if it is a training run's that lacks the checkpoint (fetch into another root).

Finally every pinned ref of every fetched bot is resolved in the runs root, weights
checked. A checkpoint loads with ``torch.load(weights_only=True)``
(:meth:`catanrl.checkpoint.Checkpoint.load`): no pickled code runs from a download.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import shlex
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from catanrl import canon
from catanrl.bridge.openhex_deploy import (
    REPO_URL,
    DeployEntry,
    DeployError,
    _git,
    _tag_commit,
    checkout_root,
    load_deploy,
    pin_problems,
    selected,
    unpinned_refs,
)

DEFAULT_REPO = REPO_URL.removeprefix("https://github.com/")
#: Where a release's files are downloaded from: ``<base>/<tag>/<file>``.
DOWNLOAD = "https://github.com/{repo}/releases/download"
SCHEMA = 1
MANIFEST, NOTES = "manifest.json", "release-notes.md"
#: The status of a run.json a release ships (and a fetch may extend).
RELEASED = "released"
TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
RUN_RE = re.compile(r"[a-z0-9][a-z0-9._+-]*/[a-z0-9][a-z0-9._+-]*")
ASSET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz")
SHA_RE = re.compile(r"[0-9a-f]{64}")
EPOCH_RE = re.compile(r"e?\d+")
#: The keys of a run.json checkpoint row that a release ships (what ``Run.checkpoints`` reads).
ROW_KEYS = ("agent_steps", "epoch", "file", "weights_sha256")
TIMEOUT = 60.0
MANIFEST_BYTES = 1 << 20
#: An asset, and the tar inside it (a 1v1 bot's checkpoint is about 2.3 MB; the largest
#: in runs/ today, arch-ablation/cnn's, is 67 MB). A release refuses a bigger one.
ASSET_BYTES = 128 << 20
#: Room for the tar's headers and padding (3 members) when a release checks its files.
TAR_SLACK = 1 << 20
USER_AGENT = "catanrl-openhex-fetch/1"

#: What refuses a release, by name; matched on every file that goes in (bytes, so a
#: checkpoint's pickled header too: torch stores it uncompressed).
SECRETS = (
    ("WANDB_API_KEY", rb"WANDB_API_KEY"),
    ("a W&B key (40 hex digits after 'wandb')", rb"(?is)wandb.{0,64}?(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])"),
    ("OPENHEX", rb"OPENHEX"),  # the key variables' prefix (a lowercase game or run name may say openhex)
    ("api_key", rb"(?i)api[_-]?key"),
    ("token", rb"(?i)token"),
    ("Bearer", rb"(?i)bearer"),
    ("32+ hex digits in a key-like field",
     rb"(?i)(?:key|secret|token|passw|auth)[a-z0-9_-]*[^a-z0-9]{1,16}[0-9a-f]{32,}(?![0-9a-f])"),
    ("a private key", rb"BEGIN [A-Z0-9 ]*PRIVATE KEY"),
)
_SECRETS = tuple((what, re.compile(pattern)) for what, pattern in SECRETS)
HOME_RE = re.compile(rb"(?:/home/|/Users/|/root/)[^\s\"'`,;:)\]}\x00-\x1f\x7f-\xff]*")


def repo_of(settings: Mapping[str, Any]) -> str:
    """The GitHub repository whose releases ship the checkpoints (``releases:``)."""
    return settings.get("releases") or DEFAULT_REPO


def _json_bytes(obj: Any) -> bytes:
    """As :mod:`catanrl.canon` writes JSON files: sorted keys, indent 1."""
    return (json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode()


def _check_tag(tag: str) -> None:
    if not TAG_RE.fullmatch(tag):
        raise DeployError(f"tag {tag!r}: a release tag is letters, digits, '.', '_' and '-'")


# ---------------------------------------------------------------- building


def checkpoints_of(entries: Sequence[DeployEntry]) -> list[dict]:
    """The checkpoints ``entries`` pin, one per run and epoch (a checkpoint two bots share
    ships once), each with the refs it serves, resolved in the runs root and its file's
    weights checked. Refuses an unpinned ref (with the pinned ref to paste), a ``.pt``
    path, and any ref not of the form ``exp/variant@eN#sha`` (a fetched run has no aliases)."""
    from catanrl.agents.spec import spec_ckpts
    from catanrl.checkpoint import Checkpoint
    from catanrl.runs import load_experiment, parse_ref

    problems, found = [], {}
    for entry in entries:
        unpinned = pin_problems(entry)
        if unpinned:
            problems += unpinned
            continue
        for ref in spec_ckpts(entry.spec):
            if ref.partition("#")[0].endswith(".pt"):
                problems.append(f"{entry.name}: {ref} is a file: a release ships run checkpoints "
                                "(exp/variant@eN#sha)")
                continue
            try:
                name, which, sha = parse_ref(ref)
                run = load_experiment(name)
                pinned = run.resolve(f"{which or 'best'}#{sha}")
                Checkpoint.load(pinned.path, ref=pinned)  # the file holds the pinned weights
            except (RuntimeError, OSError, ValueError) as error:  # RunError, CheckpointError
                problems.append(f"{entry.name}: {ref} does not resolve ({error})")
                continue
            if "/" not in name or not EPOCH_RE.fullmatch(which or ""):
                problems.append(f"{entry.name}: {ref}: a fetched run resolves only the epochs it holds; "
                                f"pin it as {pinned}")
                continue
            item = found.setdefault((run.ref, pinned.epoch), {"run": run, "ref": pinned, "refs": []})
            if ref not in item["refs"]:
                item["refs"].append(ref)
    if problems:
        raise DeployError("\n".join(problems))
    return [found[key] for key in sorted(found)]


def asset_name(run: str, epoch: int, weights_sha256: str) -> str:
    """``<exp>--<variant>--e<NNNNNN>-<sha12>.tar.gz``, with any character GitHub would rename
    in an asset name (a ``+`` in a fork's name) as ``_``."""
    experiment, variant = run.split("/")
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{experiment}--{variant}--e{epoch:06d}-{weights_sha256[:12]}.tar.gz")


def bundle(item: Mapping[str, Any]) -> dict[str, tuple[bytes, str]]:
    """The files of ``item``'s asset (:func:`checkpoints_of`): by path in the tarball, each
    with its bytes and where it comes from."""
    run, ref = item["run"], item["ref"]
    rows = [r for r in canon.read_json(run.path / "run.json")["checkpoints"] if r["epoch"] == ref.epoch]
    if len(rows) != 1 or rows[0]["file"] != f"e{ref.epoch:06d}.pt":
        raise DeployError(f"{run.path / 'run.json'}: e{ref.epoch} is not one checkpoint e{ref.epoch:06d}.pt")
    row = {key: rows[0][key] for key in ROW_KEYS}
    base = f"runs/{run.ref}"
    return {
        f"{base}/config.json": ((run.path / "config.json").read_bytes(), str(run.path / "config.json")),
        f"{base}/run.json": (_json_bytes({"checkpoints": [row], "launches": [], "status": RELEASED}),
                             f"{run.path / 'run.json'} (as shipped: this checkpoint's row only)"),
        f"{base}/checkpoints/{row['file']}": (ref.path.read_bytes(), str(ref.path)),
    }


def scan(data: bytes, source: str) -> list[str]:
    """Where ``data`` matches :data:`SECRETS`: ``source: line N: what`` (``byte N`` in a
    binary file), never the match itself."""
    binary = b"\0" in data
    hits = []
    for what, pattern in _SECRETS:
        for match in pattern.finditer(data):
            line = data.count(b"\n", 0, match.start()) + 1
            hits.append(f"{source}: {f'byte {match.start()}' if binary else f'line {line}'}: {what}")
    return hits


def home_paths(data: bytes) -> list[str]:
    """The absolute home paths in ``data`` (``/home/...``, ``/Users/...``, ``/root/...``)."""
    return sorted({m.group(0).decode(errors="replace") for m in HOME_RE.finditer(data)})


def pack(members: Mapping[str, bytes], path: str | Path) -> None:
    """``members`` as a deterministic tar.gz at ``path``: sorted, regular files, mtime 0,
    owner 0, mode 644, no gzip name or time; the same members give the same bytes."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.type = len(members[name]), 0, 0o644, tarfile.REGTYPE
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            try:
                tar.addfile(info, io.BytesIO(members[name]))
            except ValueError as error:  # a name ustar cannot hold
                raise DeployError(f"{name}: {error}") from None


def release_commit(root: str | Path, tag: str) -> tuple[str | None, list[str]]:
    """The commit ``tag`` names in the checkout, else HEAD's with a warning."""
    try:
        return _tag_commit(Path(root), tag), []
    except DeployError:
        pass
    try:
        head = _git(Path(root), "rev-parse", "HEAD").strip()
    except DeployError as error:
        return None, [f"there is no tag {tag} and no HEAD to name instead ({error})"]
    return head, [(f"there is no tag {tag} in {root}: manifest.json names HEAD {head[:12]}. Tag that commit and push "
                   "the tag before publishing (gh release create --verify-tag refuses a tag that is not on GitHub)")]


def release(config: str | Path, tag: str, out: str | Path | None = None, *,
            root: str | Path | None = None) -> dict:
    """Build the release ``tag`` of the deploy file ``config`` in ``out`` (default: a new
    temporary directory): an asset per checkpoint the bots tagged ``tag`` pin (retired or
    not), ``manifest.json`` and ``release-notes.md``. Publishes nothing: the result's
    ``command`` is the ``gh release create`` to run. Checkpoints resolve in the runs root
    (``CATANRL_RUNS_DIR``); ``root`` is the checkout whose tag (else HEAD) the manifest names."""
    _check_tag(tag)
    settings, entries = load_deploy(config)
    repo = repo_of(settings)
    chosen = [e for e in entries if e.tag == tag]
    if not chosen:
        tags = sorted({e.tag for e in entries if e.tag})
        raise DeployError(f"no bot in {config} has tag {tag} (tags: {', '.join(tags) or 'none'})")
    items = checkpoints_of(chosen)
    if not items:
        raise DeployError(f"the bots tagged {tag} ({', '.join(e.name for e in chosen)}) pin no checkpoint: "
                          "nothing to release")
    bundles, hits, homes = [], [], {}
    for item in items:
        files = bundle(item)
        size = sum(len(data) for data, _ in files.values())
        if size + TAR_SLACK > ASSET_BYTES:
            raise DeployError(f"{item['ref']}: {size} bytes, more than a fetch unpacks ({ASSET_BYTES - TAR_SLACK})")
        for name, (data, source) in files.items():
            hits += scan(data, source)
            if found := home_paths(data):
                homes[name] = found
        bundles.append((item, files))
    if hits:
        raise DeployError("secrets in the files a release would publish (nothing was written):\n  " + "\n  ".join(hits))
    out = Path(out) if out is not None else Path(tempfile.mkdtemp(prefix=f"catanrl-release-{tag}-"))
    if out.exists() and any(out.iterdir()):
        raise DeployError(f"{out} is not empty: give --out a new or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    assets = []
    for item, files in bundles:
        ref = item["ref"]
        path = out / asset_name(item["run"].ref, ref.epoch, ref.weights_sha256)
        pack({name: data for name, (data, _) in files.items()}, path)
        assets.append({"file": path.name, "sha256": canon.sha256_file(path), "size": path.stat().st_size,
                       "refs": sorted(item["refs"]), "weights_sha256": ref.weights_sha256, "run": item["run"].ref,
                       "epoch": ref.epoch})
    commit, warnings = release_commit(checkout_root() if root is None else root, tag)
    manifest = {"schema": SCHEMA, "repo": repo, "tag": tag, "commit": commit,
                "bots": [{"name": e.name, "mode": e.mode, "version": e.version, "bot_id": e.bot_id, "spec": e.spec}
                         for e in chosen],
                "assets": sorted(assets, key=lambda a: a["file"])}
    (out / MANIFEST).write_bytes(_json_bytes(manifest))
    (out / NOTES).write_text(release_notes(manifest, chosen, _shown_config(config, root)))
    for name in (MANIFEST, NOTES):  # public too: the notes name the deploy file
        if found := home_paths((out / name).read_bytes()):
            homes[name] = found
    files = [MANIFEST, *(a["file"] for a in manifest["assets"])]
    command = f"cd {shlex.quote(str(out))} && " + shlex.join(
        ["gh", "release", "create", tag, *files, "--repo", repo, "--verify-tag", "--title", tag,
         "--notes-file", NOTES])
    return {"out": str(out), "manifest": manifest, "scanned": sum(len(files) for _, files in bundles),
            "homes": homes, "warnings": warnings, "command": command}


def _shown_config(config: str | Path, root: str | Path | None) -> str:
    """``config`` as the deploy machine names it: relative to the checkout when inside it."""
    path = Path(config).resolve()
    try:
        return str(path.relative_to(Path(checkout_root() if root is None else root).resolve()))
    except ValueError:
        return str(config)


def release_notes(manifest: Mapping[str, Any], entries: Sequence[DeployEntry], config: str) -> str:
    """The release's text: its bots (with their evidence), its assets, how to fetch them."""
    tag, commit = manifest["tag"], manifest["commit"]
    lines = [f"# {tag}", "",
             (f"Checkpoints of the catanrl OpenHex bots tagged `{tag}` (commit `{(commit or 'unknown')[:12]}`), "
              "for `catanrl openhex fetch`. Each asset holds one checkpoint as `runs/<experiment>/<variant>/`: "
              "its `config.json`, a `run.json` listing that checkpoint alone, and `checkpoints/eNNNNNN.pt`. "
              "`manifest.json` lists the assets with their sha256."), "",
             "| bot | version | mode | OpenHex id | spec | evidence |", "|---|---|---|---|---|---|"]
    for e in entries:
        version = e.version + (" (retired)" if e.retired else "")
        lines.append(f"| {e.name} | {version} | {e.mode} | {e.bot_id or '-'} | `{e.spec}` | "
                     f"{', '.join(f'`{x}`' for x in e.evidence) or '-'} |")
    lines += ["", "| asset | run | epoch | weights | size | sha256 |", "|---|---|---|---|---|---|"]
    lines += [f"| {a['file']} | {a['run']} | {a['epoch']} | `{a['weights_sha256'][:12]}` | {a['size']} | "
              f"`{a['sha256']}` |" for a in manifest["assets"]]
    lines += ["", "## Fetch", "", "On the deploy machine:", "", "```sh",
              f"git fetch --tags && git checkout {tag}", f"catanrl openhex fetch {config}",
              f"catanrl openhex {config} --check", "```", "",
              ("`fetch` downloads `manifest.json` and the assets it needs over HTTPS, checks each one's sha256, "
               "unpacks them into the runs root and resolves every pinned checkpoint. A checkpoint loads with "
               "`torch.load(weights_only=True)`."), ""]
    return "\n".join(lines)


def format_release(result: Mapping[str, Any]) -> str:
    """What ``catanrl openhex release`` prints: the assets, the scan, the home paths that
    become public, and the command that publishes."""
    manifest = result["manifest"]
    commit = (manifest["commit"] or "unknown")[:12]
    lines = [f"release {manifest['tag']} of {manifest['repo']} at {commit} in {result['out']}",
             "bots: " + ", ".join(f"{b['name']} ({b['version']})" for b in manifest["bots"])]
    lines += [f"warning: {w}" for w in result["warnings"]]
    width = max(len(a["file"]) for a in manifest["assets"])
    for a in manifest["assets"]:
        lines.append(f"  {a['file'].ljust(width)}  {a['size'] / 1e6:6.2f} MB  sha256 {a['sha256'][:12]}  "
                     f"serves {', '.join(a['refs'])}")
    lines.append(f"secret scan: {result['scanned']} files, no secret found")
    if result["homes"]:
        lines.append("absolute home paths (allowed; they become public):")
        lines += [f"  {name}: {path}" for name, paths in sorted(result["homes"].items()) for path in paths]
    else:
        lines.append("absolute home paths: none")
    lines += [f"notes: {Path(result['out']) / NOTES}", "publish with (after tagging and pushing the tag):",
              f"  {result['command']}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- fetching


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    """Follow redirects (GitHub sends a release asset to its storage host), never from
    https to anything else."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if req.full_url.startswith("https:") and not newurl.startswith("https:"):
            raise DeployError(f"GET {req.full_url}: redirected to {newurl}, not https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, cap: int, timeout: float = TIMEOUT) -> bytes:
    """GET ``url`` (redirects followed, https kept), at most ``cap`` bytes."""
    opener = urllib.request.build_opener(_HttpsOnly())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    chunks, size = [], 0
    try:
        with opener.open(request, timeout=timeout) as reply:
            length = reply.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > cap:
                raise DeployError(f"GET {url}: {length} bytes, more than {cap}")
            while chunk := reply.read(1 << 20):
                chunks.append(chunk)
                size += len(chunk)
                if size > cap:
                    raise DeployError(f"GET {url}: more than {cap} bytes")
    except urllib.error.HTTPError as error:
        hint = " (is the release published, and the tag right?)" if error.code == 404 else ""
        raise DeployError(f"GET {url}: HTTP {error.code}{hint}") from None
    except (urllib.error.URLError, OSError) as error:
        raise DeployError(f"GET {url}: {getattr(error, 'reason', error)}") from None
    return b"".join(chunks)


def _asset_problems(asset: Any) -> list[str]:
    if not isinstance(asset, Mapping):
        return [f"expected a mapping, not {asset!r}"]
    problems = []
    if not isinstance(asset.get("file"), str) or not ASSET_RE.fullmatch(asset["file"]):
        problems.append(f"file {asset.get('file')!r} is not an asset name (name.tar.gz)")
    for key in ("sha256", "weights_sha256"):
        if not isinstance(asset.get(key), str) or not SHA_RE.fullmatch(asset[key]):
            problems.append(f"{key} {asset.get(key)!r} is not 64 lowercase hex digits")
    size = asset.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= ASSET_BYTES:
        problems.append(f"size {size!r} is not 1 to {ASSET_BYTES} bytes")
    if not isinstance(asset.get("run"), str) or not RUN_RE.fullmatch(asset["run"]):
        problems.append(f"run {asset.get('run')!r} is not experiment/variant")
    epoch = asset.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch < 10**6:
        problems.append(f"epoch {epoch!r} is not 0-999999")
    if not isinstance(asset.get("refs"), list) or not all(isinstance(r, str) for r in asset["refs"]):
        problems.append(f"refs {asset.get('refs')!r} is not a list of refs")
    return problems


def check_manifest(manifest: Any, tag: str, repo: str) -> list[dict]:
    """The assets of ``tag``'s ``manifest.json`` (parsed); raises :class:`DeployError`
    listing every problem: the schema, the tag and repository, each asset's fields, an
    asset name or checkpoint listed twice."""
    if not isinstance(manifest, Mapping):
        raise DeployError(f"manifest.json of {tag}: expected a JSON object")
    problems = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"schema {manifest.get('schema')!r}: this code reads schema {SCHEMA}")
    if manifest.get("tag") != tag:
        problems.append(f"it is the manifest of tag {manifest.get('tag')!r}, not {tag}")
    if manifest.get("repo") != repo:
        problems.append(f"it is the manifest of {manifest.get('repo')!r}, not {repo} (the deploy file's releases:)")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        problems.append("it lists no assets")
        assets = []
    for i, asset in enumerate(assets):
        problems += [f"assets[{i}]: {p}" for p in _asset_problems(asset)]
    if not problems:
        names = [a["file"] for a in assets]
        problems += [f"asset {n} is listed twice" for n in sorted({n for n in names if names.count(n) > 1})]
        ckpts = [(a["run"], a["epoch"]) for a in assets]
        problems += [f"{run}@e{epoch} is in more than one asset" for run, epoch in sorted(set(ckpts))
                     if ckpts.count((run, epoch)) > 1]
    if problems:
        raise DeployError(f"manifest.json of {tag}:\n  " + "\n  ".join(problems))
    return [dict(a) for a in assets]


def serves(asset: Mapping[str, Any], ref: str) -> bool:
    """``asset`` holds the checkpoint ``ref`` pins (``exp/variant@eN#sha``)."""
    from catanrl.runs import parse_ref

    try:
        run, which, sha = parse_ref(ref)
    except RuntimeError:
        return False
    return (run == asset["run"] and which is not None and EPOCH_RE.fullmatch(which) is not None
            and int(which.lstrip("e")) == asset["epoch"] and bool(sha) and asset["weights_sha256"].startswith(sha))


def resolve_in(root: str | Path, ref: str):
    """``ref`` (``exp/variant@which#sha``) resolved in the runs root ``root`` and loaded,
    weights checked (``torch.load(weights_only=True)``): its :class:`CheckpointRef`."""
    from catanrl.runs import Run, RunError, parse_ref

    run, which, sha = parse_ref(ref)
    if "/" not in run or not (Path(root) / run / "config.json").exists():
        raise RunError(f"no run {run} under {root}")
    return Run(Path(root) / run).checkpoint((which or "best") + (f"#{sha}" if sha else "")).ref


def _resolves(root: Path, ref: str):
    """:func:`resolve_in`, or None when whatever is in ``root`` does not resolve ``ref``."""
    try:
        return resolve_in(root, ref)
    except Exception:  # noqa: BLE001 - a missing run, a stray file: fetch it (the merge decides about the files)
        return None


def _member_problem(member: tarfile.TarInfo, base: str, expected: set[str], seen: Mapping[str, bytes]) -> str | None:
    name = member.name
    if name.startswith("/") or re.match(r"[A-Za-z]:", name):
        return "an absolute path"
    if ".." in PurePosixPath(name).parts or "\\" in name:
        return "a path with '..' or '\\'"
    if member.issym() or member.islnk():
        return "a link"
    if not member.isreg():
        return "not a regular file"
    if not name.startswith(base):
        return f"outside {base}"
    if name not in expected:
        return f"not a file an asset holds ({', '.join(sorted(e.removeprefix(base) for e in expected))})"
    if name in seen:
        return "listed twice"
    return None


def inflate(data: bytes, file: str) -> bytes:
    """The tar inside the gzip ``data``: one complete gzip stream of at most
    :data:`ASSET_BYTES`, inflated with that bound (so a tar header that claims gigabytes,
    as a pax header can, never gets them)."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)  # gzip
    try:
        raw = inflater.decompress(data, ASSET_BYTES + 1)
    except zlib.error as error:
        raise DeployError(f"{file}: not a readable tar.gz ({error})") from None
    if len(raw) > ASSET_BYTES or inflater.unconsumed_tail:
        raise DeployError(f"{file}: more than {ASSET_BYTES} bytes unpacked")
    if not inflater.eof or inflater.unused_data:
        raise DeployError(f"{file}: not a readable tar.gz (not one complete gzip stream)")
    return raw


def unpack(data: bytes, asset: Mapping[str, Any]) -> dict[str, bytes]:
    """The files of a downloaded asset (its sha256 already checked) by path under the runs
    root (``exp/variant/...``). The tar is :func:`inflate`'s (at most :data:`ASSET_BYTES`);
    every member must be a regular file at one of the three paths of its run and epoch
    (``runs/<exp>/<variant>/`` config.json, run.json, checkpoints/eNNNNNN.pt), all three
    there; the config must be the run's and the run.json must list the checkpoint with the
    manifest's weights. Reads only: nothing is written."""
    run, epoch, file = asset["run"], asset["epoch"], asset["file"]
    base = f"runs/{run}/"
    ckpt = f"checkpoints/e{epoch:06d}.pt"
    expected = {f"{base}config.json", f"{base}run.json", f"{base}{ckpt}"}
    files: dict[str, bytes] = {}
    raw = inflate(data, file)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            for member in tar:
                problem = _member_problem(member, base, expected, files)
                if problem:  # stop here: nothing after a bad member is read
                    raise DeployError(f"{file}: member {member.name!r}: {problem}")
                files[member.name] = tar.extractfile(member).read()
    except (tarfile.TarError, EOFError, OSError, RecursionError) as error:  # RecursionError: chained pax headers
        raise DeployError(f"{file}: not a readable tar ({type(error).__name__}: {error})") from None
    if missing := sorted(expected - set(files)):
        raise DeployError(f"{file}: {', '.join(missing)} missing")
    try:
        config = json.loads(files[f"{base}config.json"])
        shipped = json.loads(files[f"{base}run.json"])
    except (ValueError, RecursionError) as error:  # RecursionError: nested too deep
        raise DeployError(f"{file}: config.json or run.json is not JSON ({type(error).__name__}: {error})") from None
    if not isinstance(config, Mapping) or f"{config.get('experiment')}/{config.get('variant')}" != run:
        raise DeployError(f"{file}: its config.json is not run {run}'s")
    rows = shipped.get("checkpoints") if isinstance(shipped, Mapping) else None
    if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping)
            or set(rows[0]) != set(ROW_KEYS) or rows[0]["epoch"] != epoch or rows[0]["file"] != ckpt.split("/")[1]
            or rows[0]["weights_sha256"] != asset["weights_sha256"] or not isinstance(rows[0]["agent_steps"], int)
            or shipped.get("status") != RELEASED or shipped.get("launches") != []):
        raise DeployError(f"{file}: its run.json is not a release's run.json listing e{epoch} "
                          f"#{asset['weights_sha256'][:12]} alone")
    return {name.removeprefix("runs/"): content for name, content in files.items()}


def check_weights(files: Mapping[str, bytes], asset: Mapping[str, Any]) -> None:
    """The asset's checkpoint holds the manifest's weights: loaded from a temporary copy
    with ``torch.load(weights_only=True)`` (:meth:`Checkpoint.load`), before anything is
    written to the runs root."""
    from catanrl.checkpoint import Checkpoint, CheckpointRef

    epoch = asset["epoch"]
    with tempfile.TemporaryDirectory(prefix="catanrl-fetch-") as tmp:
        path = Path(tmp) / f"e{epoch:06d}.pt"
        path.write_bytes(files[f"{asset['run']}/checkpoints/e{epoch:06d}.pt"])
        ref = CheckpointRef(run=asset["run"], which=f"e{epoch}", epoch=epoch, agent_steps=0, path=path,
                            weights_sha256=asset["weights_sha256"])
        try:
            Checkpoint.load(path, ref=ref)
        except Exception as error:  # noqa: BLE001 - CheckpointError, or a header missing a key: refused all the same
            raise DeployError(f"{asset['file']}: {type(error).__name__}: {error}") from None


def _json_of(path: Path) -> bytes | None:
    """``path``'s JSON, canonical; None for a file that is not JSON (or not a file)."""
    try:
        return canon.canonical_json(canon.read_json(path)) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _row(row: Mapping[str, Any]) -> dict:
    return {key: row.get(key) for key in ROW_KEYS}


def plan_writes(root: Path, staged: Sequence[tuple[Mapping[str, Any], Mapping[str, bytes]]]
                ) -> tuple[list[tuple[Path, bytes]], list[str]]:
    """The files to write into the runs root ``root`` for the unpacked assets ``staged``
    (config.json, the checkpoints, run.json last, per run) and the problems that refuse
    them all (see the module doc for the merge rules)."""
    by_run: dict[str, list] = {}
    for asset, files in staged:
        by_run.setdefault(asset["run"], []).append((asset, files))
    writes, problems = [], []
    for run, items in sorted(by_run.items()):
        where = root / run
        configs = [files[f"{run}/config.json"] for _, files in items]
        if len({canon.canonical_json(json.loads(c)) for c in configs}) > 1:
            problems.append(f"{run}: the assets' config.json differ")
            continue
        existing = where / "config.json"
        if existing.is_symlink() or existing.exists():
            if _json_of(existing) != canon.canonical_json(json.loads(configs[0])):
                problems.append(f"{existing} exists with different content (another run of that name?): not "
                                "overwritten; fetch into an empty --runs DIR")
        else:
            writes.append((existing, configs[0]))
        rows = []
        for asset, files in items:
            name = f"checkpoints/e{asset['epoch']:06d}.pt"
            data, dest = files[f"{run}/{name}"], where / name
            if dest.is_symlink() or dest.exists():
                if not dest.is_file() or canon.sha256_file(dest) != hashlib.sha256(data).hexdigest():
                    problems.append(f"{dest} exists with different content: not overwritten")
            else:
                writes.append((dest, data))
            rows += json.loads(files[f"{run}/run.json"])["checkpoints"]
        state = where / "run.json"
        if not state.exists():
            writes.append((state, _json_bytes({"checkpoints": sorted(rows, key=lambda r: r["epoch"]),
                                               "launches": [], "status": RELEASED})))
            continue
        try:
            current = canon.read_json(state)
            have = {r["epoch"]: r for r in current.get("checkpoints", [])}
        except (OSError, ValueError, AttributeError, TypeError, KeyError) as error:
            problems.append(f"{state} cannot be read as a run.json ({type(error).__name__}): not changed")
            continue
        problems += [f"{state}: e{r['epoch']} is another checkpoint there (#{have[r['epoch']].get('weights_sha256', '')[:12]})"
                     for r in rows if r["epoch"] in have and _row(have[r["epoch"]]) != _row(r)]
        new = [r for r in rows if r["epoch"] not in have]
        if not new:
            continue  # it lists them all already: kept as it is
        if current.get("status") == RELEASED and current.get("launches") == []:
            writes.append((state, _json_bytes({**current, "checkpoints": sorted([*current["checkpoints"], *new],
                                                                                key=lambda r: r["epoch"])})))
        else:
            epochs = ", ".join(f"e{r['epoch']}" for r in new)
            problems.append(f"{state} is a training run's (status {current.get('status')}) and does not list "
                            f"{epochs}: not changed; fetch into another --runs DIR")
    return writes, problems


def fetch(config: str | Path, *, runs: str | Path | None = None, only: str | None = None,
          base: str | None = None, timeout: float = TIMEOUT) -> dict:
    """Fetch the checkpoints of the bots that run (``only``: these, comma-separated) from
    their tags' releases into the runs root ``runs`` (default: the one the resolver reads,
    ``CATANRL_RUNS_DIR`` or the checkout's ``runs/``), then resolve every pinned ref there.
    ``base`` replaces ``https://github.com/<repo>/releases/download`` (for tests). All or
    nothing: any problem refuses before a file is written."""
    from catanrl.agents.spec import spec_ckpts
    from catanrl.runs import runs_root

    settings, entries = load_deploy(config)
    repo = repo_of(settings)
    running = selected(entries, only)
    untagged = [e.name for e in running if not e.tag]
    if only and untagged:
        raise DeployError(f"{', '.join(untagged)}: no tag, so no release to fetch from (copy its run by hand)")
    bots = [e for e in running if e.tag]
    if not bots:
        raise DeployError(f"no bot that runs has a tag in {config}: nothing to fetch")
    for e in bots:
        _check_tag(e.tag)
    problems = [f"{e.name}: {ref} is not pinned by #sha (a fetch checks each checkpoint against its pin)"
                for e in bots for ref in unpinned_refs(e.spec)]
    if problems:
        raise DeployError("\n".join(problems))
    root = Path(runs).expanduser().resolve() if runs is not None else runs_root()
    wanted = [(e, ref) for e in bots for ref in spec_ckpts(e.spec)]
    here = {ref: found for _, ref in wanted if (found := _resolves(root, ref)) is not None}
    base = (base or DOWNLOAD.format(repo=repo)).rstrip("/")
    needed: dict[tuple[str, int], dict] = {}  # by checkpoint: the asset, from the first tag that serves it
    served: dict[str, dict] = {}
    for tag in sorted({e.tag for e, ref in wanted if ref not in here}):
        url = f"{base}/{urllib.parse.quote(tag)}/{MANIFEST}"
        try:
            manifest = json.loads(download(url, MANIFEST_BYTES, timeout))
        except (ValueError, RecursionError) as error:  # RecursionError: nested too deep
            raise DeployError(f"GET {url}: not JSON ({type(error).__name__}: {error})") from None
        assets = check_manifest(manifest, tag, repo)
        for e, ref in wanted:
            if e.tag != tag:
                continue
            asset = next((a for a in assets if serves(a, ref)), None)
            if asset is None:
                problems.append(f"{e.name}: the release {tag} serves no asset for {ref}")
            elif ref not in here:
                served[ref] = needed.setdefault((asset["run"], asset["epoch"]), {**asset, "tag": tag})
    if problems:
        raise DeployError("\n".join(problems))
    staged = []
    for asset in needed.values():
        url = f"{base}/{urllib.parse.quote(asset['tag'])}/{urllib.parse.quote(asset['file'])}"
        data = download(url, ASSET_BYTES, timeout)
        if len(data) != asset["size"] or hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise DeployError(f"{asset['file']}: sha256 {hashlib.sha256(data).hexdigest()[:12]} ({len(data)} bytes), "
                              f"the manifest says {asset['sha256'][:12]} ({asset['size']} bytes): not unpacked")
        files = unpack(data, asset)
        check_weights(files, asset)
        staged.append((asset, files))
    writes, problems = plan_writes(root, staged)
    if problems:
        raise DeployError("refused, nothing written:\n  " + "\n  ".join(problems))
    for path, data in writes:
        canon.atomic_write_bytes(path, data)
    rows = []
    for e, ref in wanted:
        try:
            pinned = resolve_in(root, ref)
            weights, ok = pinned.weights_sha256[:12], True
        except Exception as error:  # noqa: BLE001 - every bot's failure, not the first
            weights, ok = f"FAILED: {type(error).__name__}: {error}", False
        state = f"fetched {served[ref]['file']}" if ref in served else "already here"
        rows.append({"name": e.name, "version": e.version, "tag": e.tag, "ref": ref, "state": state,
                     "weights": weights, "ok": ok})
    return {"root": str(root), "rows": rows, "downloaded": [a["file"] for a in needed.values()],
            "written": [str(path) for path, _ in writes],
            "skipped": {"retired": [e.name for e in entries if e.retired], "untagged": untagged},
            "ok": all(r["ok"] for r in rows)}


def format_fetch(result: Mapping[str, Any], default_root: str | None = None) -> str:
    """What ``catanrl openhex fetch`` prints: a row per pinned checkpoint of each bot."""
    head = ("bot", "version", "tag", "checkpoint", "state", "weights")
    rows = [(r["name"], r["version"], r["tag"], r["ref"], r["state"], r["weights"]) for r in result["rows"]]
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    lines = [f"runs root: {result['root']}"]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip() for row in [head, *rows]]
    skipped = result["skipped"]
    if skipped["retired"]:
        lines.append(f"skipped (retired): {', '.join(skipped['retired'])}")
    if skipped["untagged"]:
        lines.append(f"skipped (no tag, no release): {', '.join(skipped['untagged'])}")
    lines.append(f"downloaded {len(result['downloaded'])} asset(s), wrote {len(result['written'])} file(s)")
    if default_root is not None and default_root != result["root"]:
        lines.append(f"note: the runner reads runs from {default_root}: run it with CATANRL_RUNS_DIR={result['root']}")
    lines.append("checkpoints load with torch.load(weights_only=True): no pickled code runs from a download")
    lines.append("ok" if result["ok"] else "FAILED: a pinned checkpoint does not resolve")
    return "\n".join(lines)
