"""Agent specs: one string, three backends (DESIGN 6.5).

Specs use Catanatron's registry syntax: ``KEY(:param)*`` with ``name=value``
params (a positional value is ``ckpt``)::

    POLICY:arch-lineage/graph-s-s42.g1@best
    MCTS:ckpt=arch-lineage/graph-s-s42.g1@e600:sims=256:dets=4
    BELIEF:ckpt=/anywhere/e000600.pt:worlds=16:tracker=exact
    F    R    AB:2

``tracker`` (MCTS with dets, BELIEF) picks how the hand tracker deals the
worlds: ``pool`` (the default) deals the unknown cards uniformly, ``exact``
draws from the exact posterior over hidden steals (``cpp.TrackerMode``).

``resolve_agent(spec)`` pins and self-tests the checkpoint once, checks the
games fit, derives ``fair`` and writes the canonical spec (a fixed point of
resolution). The resolved agent plays natively (``.native()``, batched on
cppanatron) or as a Catanatron ``Player`` (``.player(color)``; also
``REGISTRY.build("POLICY:...", color)`` once ``catanrl.agents`` is imported).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from catanatron.models.player import Color, Player
from catanatron.params import ParamsError, build_params
from catanatron.registry import REGISTRY, PlayerRegistry, SpecError, parse_spec

from catanrl.checkpoint import Checkpoint, CheckpointRef

MODEL_KEYS = ("POLICY", "MCTS", "BELIEF")
NATIVE_BOTS = ("F", "R")
HARNESSES = ("native", "catanatron", "server")


@dataclass(frozen=True)
class PolicyParams:
    ckpt: str
    sample: bool = False
    temperature: float = 1.0
    allow_drift: bool = False
    device: str = field(default="auto", metadata={"execution": True})


@dataclass(frozen=True)
class MctsParams:
    ckpt: str
    sims: int = 256
    dets: int = 1
    oracle: bool = False
    root: Literal["puct", "gumbel"] = "puct"
    c_puct: float = 0.75
    gumbel_m: int = 16
    temperature: float = 0.0
    noise: bool = False
    allow_drift: bool = False
    tracker: Literal["pool", "exact"] = "pool"
    device: str = field(default="auto", metadata={"execution": True})
    threads: int = field(default=0, metadata={"execution": True})


@dataclass(frozen=True)
class BeliefParams:
    ckpt: str
    worlds: int = 16
    aggregate: Literal["mean_prob", "mean_logit"] = "mean_prob"
    allow_drift: bool = False
    tracker: Literal["pool", "exact"] = "pool"
    device: str = field(default="auto", metadata={"execution": True})


def _execution(params_cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(params_cls) if f.metadata.get("execution")}


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value) if isinstance(value, float) else str(value)


# ---------------------------------------------------------------- resolution


@dataclass(frozen=True)
class ResolvedAgent:
    key: str
    impl: str
    params: Any
    checkpoint: CheckpointRef | None
    fair: bool
    label: str
    canonical: str
    harness: str = "native"
    details: dict = field(default_factory=dict, compare=False, repr=False)

    def __str__(self) -> str:
        return self.canonical

    def key_dict(self) -> dict:
        """What decides this agent's moves: no execution params, the model and
        (for MCTS) the full engine SearchConfig."""
        out: dict[str, Any] = {"key": self.key, "harness": self.harness}
        if self.key in MODEL_KEYS:
            skip = _execution(type(self.params)) | {"ckpt"}
            out["params"] = {k: v for k, v in dataclasses.asdict(self.params).items() if k not in skip}
            out["checkpoint"] = {"weights_sha256": self.checkpoint.weights_sha256, **self.details["model"]}
            if self.key == "MCTS":
                out["search"] = self.details["search"]
        else:
            out["params"] = dict(self.details.get("params", {}))
        return out

    def native(self, *, device: str | None = None, infer_rows: int = 256, mode: str = "strict"):
        from catanrl.agents.native import make_native_agent

        if device is None:
            device = getattr(self.params, "device", "auto")
        return make_native_agent(self, device=device, infer_rows=infer_rows, mode=mode)

    def player(self, color: Color) -> Player:
        if self.key in MODEL_KEYS:
            return _PLAYERS[self.key](color, self.params, resolved=self)
        return REGISTRY.build(self.canonical, color)


def check_spec_syntax(spec: str) -> None:
    """Grammar and param checks that need no checkpoint (``validate_plan``)."""
    if "," in spec:
        raise SpecError(f"{spec!r}: ',' separates lineup entries and cannot appear in a spec")
    key, args, named = parse_spec(spec)
    if key not in MODEL_KEYS:
        return
    params = _build_params(key, args, named)
    if "${" not in params.ckpt:
        _ckpt_syntax(params.ckpt)


def spec_ckpt(spec: str) -> str | None:
    """The checkpoint a model spec names, as written (``exp/var@which[#sha]``, a ``.pt`` path,
    or one with ``${...}`` in it), or None for a bot."""
    key, args, named = parse_spec(spec)
    if key not in MODEL_KEYS:
        return None
    return _build_params(key, args, named).ckpt


def _build_params(key: str, args: Sequence[Any], named: Mapping[str, Any]) -> Any:
    cls = _PLAYERS[key]
    try:
        params = build_params(cls, list(args), dict(named))
    except (ParamsError, TypeError) as error:
        raise SpecError(f"{key}: {error}") from None
    if not params.ckpt:
        raise SpecError(f"{key} needs a ckpt")
    if key == "MCTS":
        if params.oracle and params.dets > 1:
            raise SpecError("MCTS: oracle=true searches the true game and takes no dets")
        if params.oracle and params.tracker != "pool":
            raise SpecError("MCTS: oracle=true searches the true game and takes no tracker")
        if params.dets < 1:
            raise SpecError("MCTS: dets counts sampled worlds (>= 1); use oracle=true for the true game")
        if params.sims < 1:
            raise SpecError("MCTS: sims must be positive")
    if key == "BELIEF" and params.worlds < 1:
        raise SpecError("BELIEF: worlds must be positive")
    return params


def _ckpt_syntax(ckpt: str) -> None:
    from catanrl.runs import RunError, parse_ref

    if any(c in ckpt for c in ",:"):
        raise SpecError(f"ckpt {ckpt!r} contains ',' or ':'")
    if ckpt.partition("#")[0].endswith(".pt"):
        return
    try:
        parse_ref(ckpt)
    except RunError as error:
        raise SpecError(str(error)) from None


@lru_cache(maxsize=256)
def _tested_checkpoint(path: str, sha: str, part: str, allow_drift: bool) -> Checkpoint:
    ckpt = Checkpoint.load(path)
    if ckpt.ref.weights_sha256 != sha:
        raise SpecError(f"{path} changed since it was pinned")
    ckpt.self_test(allow_drift=allow_drift, part=part)
    return ckpt


def resolve_agent(spec: str | Mapping[str, Any] | ResolvedAgent, *, harness: str = "native",
                  games: Sequence[Any] = ()) -> ResolvedAgent:
    """Pin, self-test and check one spec; raises ``catanatron.registry.SpecError``."""
    if harness not in HARNESSES:
        raise SpecError(f"harness must be one of {HARNESSES}")
    if isinstance(spec, ResolvedAgent):
        spec = spec.canonical
    if isinstance(spec, str):
        if "," in spec:
            raise SpecError(f"{spec!r}: ',' separates lineup entries and cannot appear in a spec")
        key, args, named = parse_spec(spec)
    else:
        key, args, named = parse_spec(dict(spec))
    if key in MODEL_KEYS:
        return _resolve_model(key, args, named, harness, games)
    return _resolve_bot(key, args, named, harness)


def _resolve_model(key, args, named, harness, games) -> ResolvedAgent:
    from catanrl.envs.encoding import dims_for
    from catanrl.games import load_game
    from catanrl.runs import load_checkpoint
    from catanrl.search.pool import search_config

    params = _build_params(key, args, named)
    _ckpt_syntax(params.ckpt)
    try:
        ckpt = load_checkpoint(params.ckpt)
    except (RuntimeError, OSError, ValueError) as error:  # no such run or checkpoint, or it fails to load
        raise SpecError(f"{key}: {params.ckpt}: {error}") from None
    ref = ckpt.ref
    part = "all" if key == "MCTS" else "actor"
    try:
        ckpt = _tested_checkpoint(str(ref.path), ref.weights_sha256, part, params.allow_drift)
    except (RuntimeError, ValueError) as error:  # CheckpointError: drift, corruption, schema
        raise SpecError(f"{key}: {ref}: {error}") from None
    pinned = str(ref)  # run@eN#sha, or /abs/path.pt#sha
    params = dataclasses.replace(params, ckpt=pinned)
    levels, dims = ckpt.levels, ckpt.dims
    if ckpt.recurrent:
        # A recurrent network's memory is its seat's decision history: only the native
        # POLICY agent keeps one (MCTS leaves and BELIEF worlds have none, and the Catanatron
        # and server bridges do not replay a seat's decisions yet).
        if key != "POLICY":
            raise SpecError(f"{key}: {ref} is a recurrent ({ckpt.header['model']['class']}) network; only POLICY "
                            "keeps a seat's decision history")
        if harness != "native":
            raise SpecError(f"POLICY: {ref} is a recurrent ({ckpt.header['model']['class']}) network; only the "
                            "native harness keeps a seat's decision history")
    configs = [g if hasattr(g, "num_players") else load_game(g) for g in games]
    if configs and not ckpt.dims_agnostic and dims_for(configs) != dims:
        raise SpecError(f"{key}: {ref} is a {ckpt.header['model']['class']} network at dims {dims}; "
                        f"these games need {dims_for(configs)}")
    players = {g.num_players for g in configs} or {dims["max_players"]}
    level = levels["level"]
    if key == "POLICY":
        fair = level == "private" or (level == "public" and players == {2})
    elif key == "MCTS":
        fair = not params.oracle
    else:
        fair = True
    if harness == "server" and key == "POLICY" and level != "private":
        raise SpecError(f"POLICY with a {level} actor reads hidden state; on the server use BELIEF")
    skip = _execution(type(params))
    fields = [f.name for f in dataclasses.fields(params) if f.name not in skip]
    canonical = f"{key}:" + ":".join(f"{name}={_fmt(getattr(params, name))}" for name in fields)
    details: dict[str, Any] = {"model": {"model": ckpt.header["model"], "levels": levels, "dims": dims,
                                         "allow_drift": params.allow_drift}}
    if key == "MCTS":
        config = search_config(params, max_players=dims["max_players"], threads=0)
        details["search"] = {name: _plain(getattr(config, name)) for name in dir(config)
                             if not name.startswith("_") and name != "threads"}
    return ResolvedAgent(key=key, impl=f"catanrl.{key.lower()}", params=params, checkpoint=ref, fair=fair,
                         label=f"{key}:{ref.run or ref.path.name}@e{ref.epoch}", canonical=canonical,
                         harness=harness, details=details)


def _plain(value: Any) -> Any:
    return value.name if hasattr(value, "name") else value


def _resolve_bot(key, args, named, harness) -> ResolvedAgent:
    try:
        built = REGISTRY.build({"key": key, "params": dict(zip(_param_names(key), args)) | dict(named)}, Color.RED)
    except SpecError:
        raise
    except (ParamsError, TypeError, ValueError) as error:
        raise SpecError(f"{key}: {error}") from None
    params = REGISTRY.spec_of(built)["params"]
    if harness == "native":
        if key not in NATIVE_BOTS:
            raise SpecError(f"{key} is a Catanatron bot; the native harness plays {', '.join(MODEL_KEYS + NATIVE_BOTS)}")
        if key == "F" and (params.get("value_fn") != "base" or params.get("epsilon") is not None):
            raise SpecError("native F plays the base value function without epsilon")
    canonical = ":".join([key] + [f"{k}={_fmt(v)}" for k, v in params.items() if v is not None])
    return ResolvedAgent(key=key, impl="native" if harness == "native" else type(built).__name__,
                         params=params, checkpoint=None, fair=False, label=key, canonical=canonical,
                         harness=harness, details={"params": params})


def _param_names(key: str) -> list[str]:
    from catanatron.params import schema_of

    return [entry["name"] for entry in schema_of(REGISTRY.lookup(key))]


def player(spec: str, color: Color) -> Player:
    """A Catanatron Player for any spec (catanrl's or a registry bot)."""
    return resolve_agent(spec, harness="catanatron").player(color)


# ---------------------------------------------------------------- Catanatron players


class CatanrlPlayer(Player):
    """Base of POLICY/MCTS/BELIEF as Catanatron players.

    ``decide`` mirrors the Catanatron game on cppanatron (replaying its action
    records with their dice, card and steal outcomes, so every prompt is
    exact), then runs the native agent at batch 1.
    """

    KEY = ""

    def __init__(self, color: Color, params: Any = None, *, resolved: ResolvedAgent | None = None) -> None:
        super().__init__(color, params)
        self._resolved = resolved
        self._native = None
        self._mirror: dict[str, Any] | None = None

    def _agent(self):
        if self._resolved is None:
            self._resolved = resolve_agent({"key": self.KEY, "params": dataclasses.asdict(self.params)},
                                           harness="catanatron")
        if self._native is None:
            native = self._resolved.native(infer_rows=1)
            if getattr(native, "recurrent", False):
                raise RuntimeError(f"{self.KEY}: a recurrent network needs its seat's decision history, which "
                                   "the Catanatron player does not replay")
            self._native = native
        return self._native

    def reset_state(self) -> None:
        self._mirror = None

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        agent = self._agent()
        mirror = self._sync(game, agent)
        native_game, seat = mirror["game"], mirror["seat"]
        from cppanatron.catanatron import from_catanatron_action

        from catanrl.seeding import derive_seed

        row = agent.encode(native_game, seat)[None]
        offset, (size,) = agent.encoder.layout()["action_mask"]
        mask = row[:, offset:offset + size]
        seed = derive_seed(int(game.seed), "decide", seat, mirror["ply"])  # ply: the seat's actions so far
        index = agent.act(row, mask, np.array([seed], dtype=np.uint64),
                          games=[native_game] if agent.needs_games else None,
                          trackers=[mirror["tracker"]] if agent.needs_trackers else None)[0]
        chosen = tuple(agent.action_space.decode(native_game, int(index)))
        for action in playable_actions:
            if tuple(from_catanatron_action(action)) == chosen:
                return action
        raise RuntimeError(f"{self.KEY} chose {chosen}, which Catanatron does not offer")

    @staticmethod
    def _fit(agent, native) -> None:
        """Play at the game's dims (a dims-agnostic network rebinds, another refuses) and
        refuse a map the network would misread (the cnn's board check), as the harness and
        the bridge do, once per mirrored game."""
        from catanrl.envs.encoding import covering_dims

        need = covering_dims([native.board.map], native.num_players)
        have = agent.dims
        if any(need[k] > have[k] for k in need):
            agent.bind({k: max(need[k], have[k]) for k in need})
        agent.check_maps([native.board.map])

    def _sync(self, game, agent) -> dict[str, Any]:
        """The cppanatron mirror of ``game``, with any new actions replayed."""
        import cppanatron as cpp
        from cppanatron.parity import CHANCE_ACTIONS, CppanatronParityGame

        records = game.state.action_records
        m = self._mirror
        # Rebuild unless the game only grew since the last sync (an undo rewrites the log).
        if m is None or m["id"] != game.id or records[:len(m["records"])] != m["records"]:
            native = _native_game(game)
            self._fit(agent, native)
            seat = list(game.state.colors).index(self.color)
            tracker = cpp.HandTracker(native, seat, agent.tracker_mode) if agent.needs_trackers else None
            m = self._mirror = {"id": game.id, "game": native, "seat": seat, "tracker": tracker, "records": [],
                                "ply": 0}
        from catanatron.models.enums import Action, ActionType
        from cppanatron.catanatron import from_catanatron_action

        parity = CppanatronParityGame(m["game"])
        for record in records[len(m["records"]):]:
            action = record.action
            if action.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                action = Action(action.color, action.action_type, None)  # the record holds the outcome
            outcome = record.result if action.action_type in CHANCE_ACTIONS else None
            before = m["game"].copy() if m["tracker"] is not None else None
            parity.execute(action, outcome)
            if m["tracker"] is not None:
                m["tracker"].observe(before, from_catanatron_action(action), m["game"])
        m["records"] = list(records)
        m["ply"] = sum(1 for r in records if r.action.color == self.color)
        return m


def _native_game(game):
    """A fresh cppanatron game with ``game``'s seating, board and rules.

    The starting bank, deck and pieces follow from conservation: what the
    bank holds plus every hand (and every played card, every built piece).
    """
    import cppanatron as cpp
    from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
    from catanatron.state_functions import (
        get_dev_cards_in_hand,
        get_played_dev_cards,
        get_player_buildings,
        get_player_freqdeck,
    )
    from cppanatron.catanatron import from_catanatron_color, map_spec_from_catanatron

    state = game.state
    colors = list(state.colors)
    bank = [int(state.resource_freqdeck[i]) + sum(get_player_freqdeck(state, c)[i] for c in colors)
            for i in range(len(RESOURCES))]
    deck = [state.development_listdeck.count(card) + sum(get_dev_cards_in_hand(state, c, card)
                                                        + get_played_dev_cards(state, c, card) for c in colors)
            for card in DEVELOPMENT_CARDS]
    first = colors[0]

    def pieces(kind: str, *built: str) -> int:
        return int(state.player_state[f"P0_{kind}_AVAILABLE"]) + sum(
            len(get_player_buildings(state, first, b)) for b in built)

    rules = cpp.GameRules(vps_to_win=int(game.vps_to_win), discard_limit=int(state.discard_limit),
                          friendly_robber=bool(state.friendly_robber), resource_bank=tuple(bank),
                          development_deck=tuple(deck), roads_per_player=pieces("ROADS", "ROAD"),
                          settlements_per_player=pieces("SETTLEMENTS", "SETTLEMENT"),
                          cities_per_player=pieces("CITIES", "CITY"))
    catan_map = cpp.CatanMap.from_spec(map_spec_from_catanatron(state.board.map))
    return cpp.Game([from_catanatron_color(c) for c in colors], map=catan_map, rules=rules, seed=int(game.seed))


class PolicyPlayer(CatanrlPlayer):
    """A catanrl network's policy (argmax, or a seeded sample)."""

    KEY = "POLICY"
    Params = PolicyParams


class MctsPlayer(CatanrlPlayer):
    """Batched PUCT/Gumbel search with a catanrl network (dets: sampled worlds)."""

    KEY = "MCTS"
    Params = MctsParams


class BeliefPlayer(CatanrlPlayer):
    """A catanrl policy averaged over worlds sampled from its hand tracker."""

    KEY = "BELIEF"
    Params = BeliefParams


_PLAYERS = {"POLICY": PolicyPlayer, "MCTS": MctsPlayer, "BELIEF": BeliefPlayer}


def register(registry: PlayerRegistry = REGISTRY) -> None:
    """Make POLICY, MCTS and BELIEF buildable by key. Idempotent."""
    import catanatron.players  # noqa: F401  (registers the builtins first)

    for key, cls in _PLAYERS.items():
        if registry.get(key) is not cls:
            registry.register(key, cls, replace=True)
