"""Experiment plans: YAML files resolved into variants (DESIGN 6.2).

``load_plan(path)`` is pure (no argv, no cwd, no runs dir). An experiment
YAML is layered over ``configs/defaults.yaml``::

    defaults < train/<algo>.yaml, models/<preset>.yaml < body < stage < sweep cell < --set

A ``key!:`` replaces that section instead of merging, and dotted keys
(``eval.select.every: 8``) address nested ones. Unknown keys fail with a
suggestion, values take their default's type (PyYAML reads ``3e-4`` as a
string), and ``${...}`` interpolates:

* ``${seed}``, ``${model.preset}``, ...: any resolved key;
* ``${params.NAME}``: a ``params`` value (every ``params`` key must be used);
* ``${run}``: this variant's ref; ``${stages.NAME}``: an earlier stage's ref
  in the same sweep cell; ``${experiment}``;
* ``${self}``: the checkpoint under select (only inside ``eval.select``);
* ``${init}``: the checkpoint ``init_from`` loads the actor from, as bound
  when the run is created (only inside ``eval``; an error without one).

An AlphaZero variant that starts from a network (``init_from`` with the
actor) gets :func:`alphazero_evals` as a layer between the defaults and the
body: head to head against that network.

``config_hash`` covers everything but ``meta``, ``runtime`` and, for
``kind: train``, ``eval.final``. Refs are hashed as written; they are bound
to ``run@eN#sha`` once, when a variant is created (:func:`bind_refs`).
"""

from __future__ import annotations

import copy
import difflib
import fnmatch
import hashlib
import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from catanrl.canon import digest
from catanrl.games import CONFIG_DIR

#: Version of the config schema; bump it with a MIGRATIONS entry per new key.
SCHEMA = 1
#: (schema that added it, dotted key, the value configs written before it meant, and for a
#: ``train.*`` key the algo it belongs to, or None for every algo).
MIGRATIONS: list[tuple[int, str, Any, str | None]] = []
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
KINDS = ("train", "eval")
ALGOS = ("ppo", "imitation", "alphazero")
#: Sections whose keys are not checked against the defaults.
OPEN = {"meta", "params", "model.args"}
#: Keys of each entry of these lists or optional mappings.
ENTRY_KEYS = {
    "eval.select": {"suite", "lineup", "every", "metric", "fill"},
    "eval.final": {"suite", "lineup", "fill"},
    "init_from": {"ckpt", "parts", "restore"},
}
PARTS = ("actor", "critic")
RESTORE = ("weights", "weights+optimizer")
#: Execution keys: set by the runner from ``runtime``, never under ``train``.
EXECUTION_KEYS = {"device", "seed", "data_dir", "compile", "compile_mode", "cpu_offload", "torch_deterministic",
                  "use_rnn", "batch_size", "env", "threads", "num_envs", "infer_rows"}
UNHASHED = ("meta", "runtime")
INTERP = re.compile(r"\$\{([^${}]+)\}")


class ConfigError(ValueError):
    """An experiment config that cannot be resolved."""


class ConfigConflict(RuntimeError):
    """A run exists under this name with a different config hash."""


@dataclass(frozen=True)
class Variant:
    experiment: str
    name: str
    stage: str | None
    labels: dict[str, str]
    config: dict
    config_hash: str
    depends_on: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        return f"{self.experiment}/{self.name}"

    @property
    def kind(self) -> str:
        return self.config["kind"]


@dataclass(frozen=True)
class Plan:
    experiment: str
    source: Path
    source_sha256: str
    variants: list[Variant]
    smoke: bool = False

    def variant(self, name: str) -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        raise ConfigError(f"{self.experiment} has no variant {name!r}; it has {[v.name for v in self.variants]}")


# ---------------------------------------------------------------- files


def _load_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


@cache
def defaults() -> dict:
    return _load_yaml(CONFIG_DIR / "defaults.yaml")


@cache
def train_defaults(algo: str) -> dict:
    path = CONFIG_DIR / "train" / f"{algo}.yaml"
    if algo not in ALGOS:
        raise ConfigError(f"unknown train.algo {algo!r}; expected one of {ALGOS}")
    if not path.exists():
        return {"algo": algo}
    data = _load_yaml(path)
    return data.get("train", data)


def schema_keys() -> set[str]:
    """Every dotted leaf key of the schema; train keys per algo, as ``train.<algo>.<key>``."""
    keys = {k for k in _flatten(defaults()) if not k.startswith("train.")}
    for algo in ALGOS:
        keys |= {f"train.{algo}.{k}" for k in train_defaults(algo)}
    return keys


# ---------------------------------------------------------------- dict helpers


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value and path not in OPEN:
            out.update(_flatten(value, path + "."))
        else:
            out[path] = value
    return out


def _expand(d: Mapping[str, Any]) -> dict:
    """Nest dotted keys: {"a.b": 1} -> {"a": {"b": 1}} (recursively)."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, Mapping):
            value = _expand(value)
        head, _, rest = str(key).partition(".")
        if rest:
            bang = head.endswith("!")
            node = out.setdefault(head, {})
            if not isinstance(node, dict):
                raise ConfigError(f"{key!r} sets inside a non-mapping")
            _merge_into(node, _expand({rest: value}))
            if bang:
                out[head] = node
        else:
            if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
                _merge_into(out[key], value)
            else:
                out[key] = value
    return out


def _merge_into(base: dict, over: Mapping[str, Any]) -> dict:
    for key, value in over.items():
        if key.endswith("!"):
            base[key[:-1]] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge_into(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def merge(*layers: Mapping[str, Any]) -> dict:
    out: dict = {}
    for layer in layers:
        _merge_into(out, _expand(layer))
    return out


def _set_paths(layer: Mapping[str, Any]) -> set[str]:
    """Dotted paths a layer sets (a replaced section counts as its root)."""
    paths = set()

    def walk(d, prefix):
        for key, value in d.items():
            path = prefix + key.rstrip("!")
            if isinstance(value, Mapping) and value and not key.endswith("!"):
                walk(value, path + ".")
            else:
                paths.add(path)

    walk(_expand(layer), "")
    return paths


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def get_path(d: Mapping[str, Any], path: str) -> Any:
    node: Any = d
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _set_path(d: dict, path: str, value: Any) -> None:
    *heads, last = path.split(".")
    for part in heads:
        d = d.setdefault(part, {})
    d[last] = value


def parse_overrides(overrides: Sequence[str]) -> dict:
    """``["train.gamma=1.0", "seed=43"]`` as a layer (values read as YAML)."""
    layer: dict[str, Any] = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set needs KEY=VALUE, got {item!r}")
        layer[key.strip()] = yaml.safe_load(value)
    return layer


# ---------------------------------------------------------------- validation and coercion


def _suggest(key: str, known: Sequence[str]) -> str:
    close = difflib.get_close_matches(key, list(known), n=1)
    return f"; did you mean {close[0]!r}?" if close else f"; known: {sorted(known)}"


def _coerce(value: Any, default: Any, path: str) -> Any:
    """``value`` as its default's type (bool, int or float): YAML reads ``3e-4`` as a string."""
    if default is None or value is None or not isinstance(default, (bool, int, float)):
        return value
    kind = type(default)
    if isinstance(value, bool) != (kind is bool):
        raise ConfigError(f"{path}: expected a {kind.__name__}, got {value!r}")
    if kind is bool:
        return value
    try:
        number = float(value.replace("_", "")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{path}: expected a {kind.__name__}, got {value!r}") from None
    if kind is int and not number.is_integer():
        raise ConfigError(f"{path}: expected an int, got {value!r}")
    return int(number) if kind is int else number


def _check(cfg: dict, schema: Mapping[str, Any], prefix: str = "") -> dict:
    out = {}
    for key, value in cfg.items():
        path = prefix + key
        if key not in schema:
            raise ConfigError(f"unknown key {path!r}{_suggest(key, list(schema))}")
        default = schema[key]
        if path in OPEN or path in ("train", "model"):  # train and model: _check_train, _check_model
            out[key] = value
        elif path in ENTRY_KEYS:
            out[key] = _check_entries(value, path)
        elif isinstance(default, Mapping) and default:
            if not isinstance(value, Mapping):
                raise ConfigError(f"{path}: expected a mapping, got {value!r}")
            out[key] = _check(dict(value), default, path + ".")
        else:
            out[key] = _coerce(value, default, path)
    return out


def _check_entries(value: Any, path: str) -> Any:
    entries = value if isinstance(value, list) else [value]
    if path != "eval.final" and isinstance(value, list):
        raise ConfigError(f"{path}: expected a mapping or null")
    for entry in entries:
        if entry is None:
            continue
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {entry!r}")
        unknown = set(entry) - ENTRY_KEYS[path]
        if unknown:
            key = min(unknown)
            raise ConfigError(f"unknown key {path}.{key}{_suggest(key, sorted(ENTRY_KEYS[path]))}")
    return value


def _check_train(train: Any) -> dict:
    if not isinstance(train, Mapping):
        raise ConfigError("train: expected a mapping")
    algo = train.get("algo", "ppo")
    base = train_defaults(algo)
    execution = sorted(EXECUTION_KEYS & set(train))
    if execution:
        raise ConfigError(f"train.{execution[0]} is an execution key: set it under runtime (or let the runner)")
    out = dict(base)
    for key, value in train.items():
        if key not in base:
            raise ConfigError(f"unknown key 'train.{key}' for algo {algo}{_suggest(key, list(base))}")
        out[key] = _coerce(value, base[key], f"train.{key}")
    return out


def _check_model(model: Any) -> dict:
    from catanrl.models import resolve_spec

    try:
        return resolve_spec(model)
    except (ValueError, TypeError) as error:
        raise ConfigError(f"model: {error}") from None


def _check_semantics(cfg: dict) -> None:
    if cfg["kind"] not in KINDS:
        raise ConfigError(f"kind must be one of {KINDS}")
    if cfg["kind"] == "train" and not cfg["env"]["scenarios"]:
        raise ConfigError("env.scenarios is empty")
    init = cfg.get("init_from")
    if init is not None:
        if "ckpt" not in init:
            raise ConfigError("init_from needs ckpt")
        parts = init.get("parts", list(PARTS))
        if not set(parts) <= set(PARTS) or not parts:
            raise ConfigError(f"init_from.parts must be a non-empty subset of {PARTS}")
        if isinstance(init["ckpt"], Mapping) and not set(init["ckpt"]) <= set(PARTS):
            raise ConfigError(f"init_from.ckpt per part takes keys {PARTS}")
        if init.get("restore", "weights") not in RESTORE:
            raise ConfigError(f"init_from.restore must be one of {RESTORE}")
    select = cfg["eval"]["select"]
    if select is not None and ("suite" not in select or "lineup" not in select):
        raise ConfigError("eval.select needs suite and lineup")
    for entry in cfg["eval"]["final"]:
        if "suite" not in entry or "lineup" not in entry:
            raise ConfigError("every eval.final entry needs suite and lineup")


def _resolve_scenario(entry: Any) -> dict:
    """A scenario with its game's full definition, so the config hash covers the rules and map."""
    from catanrl.games import load_game

    entry = {"game": entry} if not isinstance(entry, Mapping) else dict(entry)
    if "game" not in entry:
        raise ConfigError(f"env.scenarios entry {entry} names no game")
    try:
        entry["game"] = load_game(entry["game"]).to_dict()
    except (OSError, ValueError, TypeError) as error:
        raise ConfigError(f"env.scenarios: {error}") from None
    return entry


def init_parts(init: Mapping[str, Any] | None) -> list[str]:
    """The parts ``init_from`` loads (none without it)."""
    if not init:
        return []
    ckpt = init.get("ckpt")
    return list(init.get("parts") or (ckpt.keys() if isinstance(ckpt, Mapping) else PARTS))


def alphazero_evals(level: str, players: int, suites: str) -> dict:
    """AlphaZero's default ``eval`` for a variant that starts from a network: head to
    head against that network, ``${init}``. Self-play drifts (v1's runs beat their
    previous iteration while flat or worse against F), and against F a strong policy
    is near its ceiling.

    Every network plays as its fair raw policy: POLICY at a ``private`` actor, or a
    ``public`` one in two-player games (where every resource movement is seen by both
    seats), else BELIEF. The run's network takes slot 0 and the start fills the other
    ``players - 1`` seats. Select scores each checkpoint by slot 0's score rate against
    the start on ``select-<suites>``. e0 is the start's weights, so it scores exactly
    1/players as POLICY (mirrored games; as BELIEF, up to world sampling) and is
    ``best`` first; ``best`` then moves only to a checkpoint above every earlier one.
    The finals, on ``heldout-<suites>``: the run's raw policy against the start's; its
    search (the training search's budget and root, fair with ``dets=1``) against the
    start's raw policy; the run's own e0 searching the same way there (paired with the
    previous: what training added to search, with exactly the critic the run started
    from); and the raw policy against F, kept as a secondary number.
    """
    def raw(ckpt: str) -> str:
        fair = level == "private" or (level == "public" and players == 2)
        return f"POLICY:ckpt={ckpt}" if fair else f"BELIEF:ckpt={ckpt}:worlds=16"

    search = "sims=${train.sims}:dets=1:root=${train.root}:c_puct=${train.c_puct}:gumbel_m=${train.gumbel_m}"
    start = [raw("${init}")] * (players - 1)
    heldout = f"heldout-{suites}"
    searched = {"name": heldout, "boards": 500}
    return {
        "select": {"suite": f"select-{suites}", "lineup": [raw("${self}"), *start], "every": 1,
                   "metric": "score_rate"},
        "final": [
            {"suite": heldout, "lineup": [raw("${run}@best"), *start]},
            {"suite": searched, "lineup": [f"MCTS:ckpt=${{run}}@best:{search}", *start]},
            {"suite": searched, "lineup": [f"MCTS:ckpt=${{run}}@init:{search}", *start]},
            {"suite": heldout, "lineup": [raw("${run}@best"), *["F"] * (players - 1)]},
        ],
    }


def _default_suites(scenarios: Sequence[Any]) -> tuple[int, str] | None:
    """(players, suite suffix) when every scenario plays one game that has select and
    heldout suites (``base-4p`` -> ``select-4p``, ``random-4p`` -> ``select-random-4p``)."""
    from catanrl.games import load_game

    names = {e.get("game") if isinstance(e, Mapping) else e for e in scenarios}
    if len(names) != 1 or not isinstance(next(iter(names)), str):
        return None
    name = next(iter(names))
    suffix = name.removeprefix("base-")
    if not all((CONFIG_DIR / "suites" / f"{kind}-{suffix}.yaml").exists() for kind in ("select", "heldout")):
        return None
    return load_game(name).num_players, suffix


def _algo_layer(merged: Mapping[str, Any]) -> dict:
    """What the variant's algo adds between the defaults and the body (DESIGN 6.2):
    :func:`alphazero_evals` for AlphaZero from a network; nothing otherwise."""
    train = merged.get("train") or {}
    if merged.get("kind", "train") != "train" or train.get("algo") != "alphazero":
        return {}
    if "actor" not in init_parts(merged.get("init_from")):
        return {}  # from scratch: there is no start to play
    found = _default_suites((merged.get("env") or {}).get("scenarios") or [])
    if found is None:
        if (merged.get("eval") or {}).get("select") is not None:
            return {}  # the experiment chose its own evals
        games = sorted({str(e.get("game") if isinstance(e, Mapping) else e)
                        for e in (merged.get("env") or {}).get("scenarios") or []})
        raise ConfigError(f"AlphaZero from a network on {games} has no default head-to-head suites; "
                          "set eval.select and eval.final against ${init}")
    players, suites = found
    return {"eval": alphazero_evals((merged.get("obs") or {}).get("level") or "private", players, suites)}


def migrate(cfg: dict, *, from_schema: int) -> dict:
    """A stored config of ``from_schema`` with every later key at its legacy value."""
    cfg = copy.deepcopy(cfg)
    algo = (cfg.get("train") or {}).get("algo")
    for schema, path, legacy, for_algo in MIGRATIONS:
        if schema <= from_schema or (path.startswith("train.") and for_algo not in (None, algo)):
            continue
        try:
            get_path(cfg, path)
        except KeyError:
            _set_path(cfg, path, copy.deepcopy(legacy))
    return cfg


# ---------------------------------------------------------------- interpolation


def _lookup(name: str, scope: Mapping[str, Any]) -> Any:
    if name in scope:
        return scope[name]
    try:
        return get_path(scope["__config__"], name)
    except KeyError:
        raise ConfigError(f"${{{name}}} names no key") from None


def _interpolate(value: Any, scope: Mapping[str, Any], path: str, used: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {k: _interpolate(v, scope, f"{path}.{k}" if path else k, used) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, scope, f"{path}[{i}]", used) for i, v in enumerate(value)]
    if not isinstance(value, str) or "${" not in value:
        return value

    def one(name: str) -> Any:
        name = name.strip()
        if name == "self":
            if not path.startswith("eval.select"):
                raise ConfigError(f"{path}: ${{self}} is only allowed in eval.select")
            return "${self}"
        if name == "init":  # bound when the run is created, like init_from (runs.bind_refs)
            if not path.startswith(("eval.select", "eval.final")):
                raise ConfigError(f"{path}: ${{init}} is only allowed in eval.select and eval.final")
            if "actor" not in init_parts(scope["__config__"].get("init_from")):
                raise ConfigError(f"{path}: ${{init}} is the checkpoint init_from loads the actor from, "
                                  "and this variant has none")
            return "${init}"
        if name.startswith("params."):
            used.add(name[len("params."):])
        if name.startswith("stages."):
            stage = name[len("stages."):]
            if stage not in scope["__stages__"]:
                raise ConfigError(f"{path}: ${{{name}}}: no earlier stage {stage!r}")
            return scope["__stages__"][stage]
        return _lookup(name, scope)

    whole = INTERP.fullmatch(value)
    if whole:
        return one(whole.group(1))
    return INTERP.sub(lambda m: str(one(m.group(1))), value)


# ---------------------------------------------------------------- sweeps and stages


def _cells(sweep: Mapping[str, Any]) -> list[tuple[dict[str, str], dict[str, Any], set[str]]]:
    """(labels, overrides, paths set) per sweep cell, in axis order."""
    axes = sweep.get("axes") or {}
    unknown = set(sweep) - {"axes", "name", "exclude"}
    if unknown:
        raise ConfigError(f"unknown sweep keys {sorted(unknown)}; expected axes, name, exclude")
    choices = []
    for axis, values in axes.items():
        if isinstance(values, Mapping):  # tied axis: label -> {dotted key: value}
            options = []
            for label, layer in values.items():
                if not isinstance(layer, Mapping):
                    raise ConfigError(f"sweep axis {axis}: {label} needs a mapping of keys")
                options.append((str(label), dict(layer), str(label)))
        elif isinstance(values, list):
            options = [(_label(v), {axis: v}, v) for v in values]
        else:
            raise ConfigError(f"sweep axis {axis}: expected a list or a mapping")
        if not options:
            raise ConfigError(f"sweep axis {axis} is empty")
        choices.append([(axis, label, layer, value) for label, layer, value in options])
    excludes = [dict(ex) for ex in sweep.get("exclude") or []]
    for ex in excludes:
        unknown = set(ex) - set(axes)
        if unknown:
            raise ConfigError(f"sweep.exclude names no axis {sorted(unknown)}; axes: {sorted(axes)}")
    used = [False] * len(excludes)
    cells = []
    for combo in itertools.product(*choices):
        values = {axis: value for axis, _, _, value in combo}
        hits = [all(_same(values[k], v) for k, v in ex.items()) for ex in excludes]
        used = [u or h for u, h in zip(used, hits)]
        if any(hits):
            continue
        paths: set[str] = set()
        for _, _, over, _ in combo:
            new = _set_paths(over)
            clash = [p for p in new for q in paths if _overlaps(p, q)]
            if clash:
                raise ConfigError(f"sweep axes set the same key {clash[0]!r}")
            paths |= new
        layer = merge(*(over for _, _, over, _ in combo))
        cells.append(({axis: label for axis, label, _, _ in combo}, layer, paths))
    unmatched = [ex for ex, u in zip(excludes, used) if not u]
    if unmatched:
        raise ConfigError(f"sweep.exclude {unmatched[0]} matches no cell")
    return cells


def _same(a: Any, b: Any) -> bool:
    """Axis value equality, numbers compared as numbers (YAML reads 1e-3 as a string)."""
    if a == b or _label(a) == _label(b):
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _label(value: Any) -> str:
    if isinstance(value, float):
        return repr(value).replace("e-0", "e-").replace("e+0", "e")
    return str(value).lower() if isinstance(value, bool) else str(value)


def _cell_name(template: str | None, labels: Mapping[str, str], cfg: Mapping[str, Any]) -> str:
    if template is None:
        return "-".join(labels.values()) if labels else "main"

    def field(m: re.Match) -> str:
        key = m.group(1)
        if key in labels:
            return labels[key]
        try:
            return _label(get_path(cfg, key))
        except KeyError:
            raise ConfigError(f"sweep.name {template!r}: {{{key}}} is neither an axis nor a key") from None

    return re.sub(r"\{([^{}]+)\}", field, template)


def _refs_in(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [r for v in value.values() for r in _refs_in(v)]
    if isinstance(value, list):
        return [r for v in value for r in _refs_in(v)]
    return [value] if isinstance(value, str) else []


def _smoke_refs(value: Any) -> Any:
    """Epoch refs (``@3000``, ``@e3000``) become ``@latest`` in smoke runs."""
    if isinstance(value, Mapping):
        return {k: _smoke_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_smoke_refs(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"@e?\d+(?=$|[:#,])", "@latest", value)
    return value


def _merge_existing(base: dict, over: Mapping[str, Any]) -> None:
    """Merge only keys ``base`` already has (smoke overrides fit every algo)."""
    for key, value in over.items():
        if key not in base or base[key] is None:
            continue
        if isinstance(value, Mapping) and isinstance(base[key], dict):
            _merge_existing(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


# ---------------------------------------------------------------- plans


def config_hash(cfg: Mapping[str, Any]) -> str:
    return digest(hashed_part(cfg))


def hashed_part(cfg: Mapping[str, Any]) -> dict:
    out = {k: copy.deepcopy(v) for k, v in cfg.items() if k not in UNHASHED}
    if out.get("kind", "train") == "train" and "eval" in out:
        out["eval"] = {k: v for k, v in out["eval"].items() if k != "final"}
    return out


def config_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """``["train.learning_rate: 0.0003 -> 0.0001", ...]`` over every leaf."""
    fa, fb = _flatten(a), _flatten(b)
    out = []
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key, "<unset>"), fb.get(key, "<unset>")
        if va != vb:
            out.append(f"{key}: {va!r} -> {vb!r}")
    return out


def resolve(raw: Mapping[str, Any], *, experiment: str, variant: str,
            stages: Mapping[str, str] | None = None) -> dict:
    """One variant's config from its merged layers: checked, coerced, completed, interpolated."""
    merged = merge(defaults(), _algo_layer(merge(defaults(), raw)), raw)
    merged.pop("name", None)
    cfg = _check(merged, defaults())
    cfg["train"] = _check_train(cfg["train"])
    cfg["env"]["scenarios"] = [_resolve_scenario(e) for e in cfg["env"]["scenarios"]]
    if cfg["model"] is not None or cfg["kind"] == "train":
        cfg["model"] = _check_model(cfg["model"])
    scope = {"__config__": cfg, "__stages__": dict(stages or {}), "run": f"{experiment}/{variant}",
             "experiment": experiment}
    used: set[str] = set()
    hashed = {k: v for k, v in cfg.items() if k not in ("meta",)}
    out = _interpolate(hashed, scope, "", used)
    out["meta"] = cfg["meta"]
    unused = sorted(set(cfg["params"]) - used)
    if unused:
        raise ConfigError(f"params {unused} are referenced nowhere (expected ${{params.{unused[0]}}})")
    _check_semantics(out)
    return out


def load_plan(path: str | Path, *, overrides: Sequence[str] = (), only: str | None = None,
              variant: str | None = None, smoke: bool = False) -> Plan:
    """Every variant of an experiment YAML, in dependency order."""
    path = Path(path)
    if not path.suffix and not path.exists():
        path = CONFIG_DIR / "experiments" / f"{path}.yaml"
    source = path.read_bytes()
    raw = yaml.safe_load(source) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: expected a mapping")
    raw = dict(raw)
    experiment = raw.get("name", path.stem)
    if experiment != path.stem:
        raise ConfigError(f"{path}: name {experiment!r} differs from the file stem {path.stem!r}")
    if not NAME_RE.match(experiment):
        raise ConfigError(f"experiment name {experiment!r} must match {NAME_RE.pattern}")
    stages = raw.pop("stages", None) or {}
    sweep = raw.pop("sweep", None) or {}
    if not isinstance(stages, Mapping):
        raise ConfigError("stages: expected a mapping of stage name -> overrides")
    cli = parse_overrides(overrides)
    smoke_cfg = _load_yaml(CONFIG_DIR / "smoke.yaml") if smoke else None
    if smoke:
        experiment = f"{experiment}+smoke"

    variants: list[Variant] = []
    seen: dict[str, tuple] = {}
    cli_clashes: dict[str, str] = {}
    for labels, cell_layer, cell_paths in _cells(sweep):
        base_cfg = merge(defaults(), raw, cell_layer)
        cell = _cell_name(sweep.get("name"), labels, base_cfg)
        stage_refs: dict[str, str] = {}
        for stage, stage_layer in (stages.items() if stages else [(None, {})]):
            stage_layer = dict(stage_layer or {})
            clash = [p for p in _set_paths(stage_layer) for q in cell_paths if _overlaps(p, q)]
            if clash:
                raise ConfigError(f"stage {stage} and a sweep axis both set {clash[0]!r}")
            name = cell if stage is None else f"{cell}.{stage}"
            clash = [p for p in _set_paths(cli) for q in cell_paths if _overlaps(p, q)]
            if clash:  # refused below, if this variant is kept
                cli_clashes[name] = clash[0]
            if not NAME_RE.match(name):
                raise ConfigError(f"variant name {name!r} must match {NAME_RE.pattern}")
            if name in seen:
                raise ConfigError(f"two sweep cells are both named {name!r} ({seen[name]} and {labels})")
            seen[name] = tuple(labels.items())
            layers = merge(raw, stage_layer, cell_layer, cli)
            cfg = resolve(layers, experiment=experiment, variant=name, stages=stage_refs)
            if smoke_cfg is not None:
                _merge_existing(cfg, smoke_cfg.get("set") or {})
                cfg = _smoke_refs(cfg)
                cfg["meta"] = {**cfg["meta"], "smoke": True}
                n = int(smoke_cfg.get("eval_boards", 2))
                entries = [e for e in [cfg["eval"]["select"], *cfg["eval"]["final"]] if e]
                for entry in entries:  # a suite name or {name, boards}: the first n boards
                    suite = entry["suite"] if isinstance(entry["suite"], Mapping) else {"name": entry["suite"]}
                    entry["suite"] = {**suite, "boards": min(int(suite.get("boards", n)), n)}
                if cfg["eval"]["select"]:
                    cfg["eval"]["select"]["every"] = 1
            own = re.compile(re.escape(f"{experiment}/") + r"([a-z0-9][a-z0-9._+-]*)")
            deps = tuple(sorted({m for r in _refs_in(hashed_part(cfg)) for m in own.findall(r)} - {name}))
            variants.append(Variant(experiment=experiment, name=name, stage=stage, labels=dict(labels),
                                    config=cfg, config_hash=config_hash(cfg), depends_on=deps))
            if stage is not None:
                stage_refs[stage] = f"{experiment}/{name}"
    names = {v.name for v in variants}
    for v in variants:
        missing = [d for d in v.depends_on if d not in names]
        if missing:
            raise ConfigError(f"{v.name} depends on {missing}, which are not in the plan")
    if variant is not None:
        keep = [v for v in variants if v.name == variant]
        if not keep:
            raise ConfigError(f"{experiment} has no variant {variant!r}; it has {sorted(names)}")
        variants = keep
    elif only is not None:
        variants = [v for v in variants if fnmatch.fnmatchcase(v.name, only)]
        if not variants:
            raise ConfigError(f"--only {only!r} matches no variant of {experiment}")
    for v in variants:
        if v.name in cli_clashes:
            raise ConfigError(f"--set and a sweep axis both set {cli_clashes[v.name]!r} (in {v.name})")
    return Plan(experiment=experiment, source=path.resolve(), source_sha256=hashlib.sha256(source).hexdigest(),
                variants=variants, smoke=smoke)


def validate_plan(path: str | Path) -> list[str]:
    """Problems with an experiment beyond resolution: suites exist and fit
    their use, refs are well formed, heldout maps stay out of training."""
    from catanrl.eval.suites import load_suite

    problems: list[str] = []
    try:
        plan = load_plan(path)
    except (ConfigError, OSError, ValueError) as error:
        return [str(error)]
    checked: set[tuple] = set()
    for v in plan.variants:
        cfg = v.config
        entries = ([("select", cfg["eval"]["select"])] if cfg["eval"]["select"] else []) + \
                  [("final", e) for e in cfg["eval"]["final"]]
        for where, entry in entries:
            try:
                suite = load_suite(entry["suite"])
            except (OSError, ValueError) as error:
                problems.append(f"{v.name}: eval.{where}: {error}")
                continue
            if where == "select" and suite.purpose != "select":
                problems.append(f"{v.name}: eval.select uses {suite.name}, a {suite.purpose} suite")
            if where == "final" and suite.purpose == "select":
                problems.append(f"{v.name}: eval.final uses the select suite {suite.name}")
            for spec in entry["lineup"]:
                problem = _ref_problem(str(spec))
                if problem:
                    problems.append(f"{v.name}: {spec}: {problem}")
            if cfg["kind"] == "train" and suite.purpose == "heldout":
                key = (digest([suite.seed, suite.purpose, [suite._scenario(i) for i in range(len(suite.scenarios))]]),
                       digest(cfg["env"]), cfg["seed"])
                if key not in checked:
                    checked.add(key)
                    problems += [f"{v.name}: {p}" for p in _heldout_overlap(cfg, suite)]
        init = cfg.get("init_from")
        if init:
            refs = init["ckpt"].values() if isinstance(init["ckpt"], Mapping) else [init["ckpt"]]
            for ref in refs:
                problem = _ref_problem(f"POLICY:ckpt={ref}")
                if problem:
                    problems.append(f"{v.name}: init_from {ref}: {problem}")
    return problems


def _ref_problem(spec: str) -> str | None:
    from catanrl.agents.spec import check_spec_syntax

    try:
        check_spec_syntax(spec)
    except ValueError as error:
        return str(error)
    return None


def _heldout_overlap(cfg: Mapping[str, Any], suite: Any, boards: int = 64) -> list[str]:
    """Training pool maps (generated) that the first ``boards`` boards of a heldout suite also play."""
    from catanrl.eval.suites import map_hash
    from catanrl.games import load_game
    from catanrl.seeding import derive_seed

    pool_seed = cfg["env"]["map_pool"]["seed"]
    pool_seed = derive_seed(cfg["seed"], "map_pool") if pool_seed is None else int(pool_seed)
    first = type(suite)(**{**suite.__dict__, "boards": min(suite.boards, boards)})
    held = {g.map_hash for g in first.schedule(1, fill=True)}
    problems = []
    for i, entry in enumerate(cfg["env"]["scenarios"]):
        game = load_game(entry["game"])
        if map_hash(game.map_spec(0)) == map_hash(game.map_spec(1)):
            continue  # a fixed board: every game is on it
        pool = {map_hash(game.map_spec(derive_seed(pool_seed, i, j))) for j in range(int(entry.get("maps", 1024)))}
        if pool & held:
            problems.append(f"{len(pool & held)} heldout maps of {suite.name} are in the training pool of "
                            f"scenario {i}")
    return problems
