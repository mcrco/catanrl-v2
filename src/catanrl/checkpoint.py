"""Checkpoints: a plain dict ``torch.load(weights_only=True)`` reads, usable
without its run directory (DESIGN 6.3).

Every checkpoint embeds a self-test: up to eight ``Game.apply_setup``
positions (packaged in ``catanrl/data/selftest_positions.json``, or
generated from the training games when none fit the dims), their encoded
rows, and the network's logits and values on them, computed on CPU in strict
mode. :meth:`Checkpoint.model` runs, in order:

1. the weights hash (:class:`WeightsCorrupt`, checked at load);
2. the encoder and action layout hash (:class:`SchemaMismatch`);
3. the dims, unless the class is dims-agnostic (:class:`DimsMismatch`);
4. re-encoding the positions gives the stored rows (:class:`EncoderDrift`);
5. the rebuilt network gives the stored outputs (:class:`ModelDrift`).

``allow_drift`` skips 4 and 5; it is recorded on the model and in every
agent spec and eval key that uses it, and the report refuses those.

A recurrent network's positions are single decisions, so its self-test pins
the zero-state function (``model(obs)``: the network at a seat's first
decision); the recurrent state itself is rollout-only and never saved.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from catanrl.canon import digest, weights_sha256

CKPT_SCHEMA = 1
FORMAT = "catanrl.ckpt"
POSITIONS = Path(__file__).with_name("data") / "selftest_positions.json"
SELF_TEST_POSITIONS = 8
#: Tolerances of check 5: CPU kernels may differ across machines in the last bits.
LOGIT_TOL = 1e-4
VALUE_TOL = 1e-5


class CheckpointError(RuntimeError):
    pass


class WeightsCorrupt(CheckpointError):
    pass


class SchemaMismatch(CheckpointError):
    pass


class DimsMismatch(CheckpointError):
    pass


class EncoderDrift(CheckpointError):
    pass


class ModelDrift(CheckpointError):
    pass


@dataclass(frozen=True)
class CheckpointRef:
    run: str | None
    which: str
    epoch: int
    agent_steps: int
    path: Path
    weights_sha256: str

    def __str__(self) -> str:
        """``run@eN#sha``, or for a checkpoint loaded by path ``/abs/path.pt#sha``."""
        if self.run is None:
            return f"{self.path}#{self.weights_sha256[:12]}"
        return f"{self.run}@e{self.epoch}#{self.weights_sha256[:12]}"


# ---------------------------------------------------------------- layouts and positions


def layout_hash(dims: Mapping[str, int], level: str) -> str:
    """The encoder's row layout, feature names and action layout at ``dims``."""
    from catanrl.envs.encoding import FEATURE_BLOCKS, make_encoder

    encoder = make_encoder(dims, level)
    return digest({
        "size": int(encoder.size),
        "layout": encoder.layout(),
        "features": {block: list(encoder.feature_names(block)) for block in FEATURE_BLOCKS},
        "actions": encoder.action_space.layout(),
        "num_actions": int(encoder.action_space.size),
    })


@cache
def packaged_positions() -> tuple[dict, ...]:
    data = json.loads(POSITIONS.read_text())
    if data.get("format") != "catanrl.selftest_positions":
        raise CheckpointError(f"{POSITIONS} is not a self-test position file")
    return tuple(data["positions"])


def _position_game(pos: Mapping[str, Any]):
    """The declared game: its own map spec (older positions: the generator at their map seed)."""
    import cppanatron as cpp

    from catanrl.games import GameConfig

    cfg = GameConfig.from_dict(pos["game"])
    spec = pos.get("map") or cfg.map_spec(int(pos["map_seed"]))
    game = cpp.Game(cfg.num_players, map=cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(spec)),
                    rules=cfg.cppanatron_rules(), seed=int(pos["seed"]))
    options = dict(pos["options"])
    if "prompt" in options:
        options["prompt"] = cpp.ActionPrompt[options["prompt"]]
    game.apply_setup([cpp.PlayerSetup(**p) for p in pos["players"]], **options)
    return game


def _fits(pos: Mapping[str, Any], dims: Mapping[str, int]) -> bool:
    import cppanatron as cpp

    from catanrl.games import GameConfig

    cfg = GameConfig.from_dict(pos["game"])
    if cfg.num_players > dims["max_players"]:
        return False
    spec = pos.get("map") or cfg.map_spec(int(pos["map_seed"]))
    need = cpp.EncoderDims.covering([cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(spec))], cfg.num_players)
    return all(int(getattr(need, k)) <= int(dims[k]) for k in ("max_tiles", "max_nodes", "max_edges"))


#: The prompts self-test positions cycle through, so every head and phase is exercised.
KINDS = ("turn", "turn+rolled", "MOVE_ROBBER", "BUILD_INITIAL_SETTLEMENT", "DISCARD", "road_building",
         "BUILD_INITIAL_ROAD", "turn+rolled")


def _kind(game) -> str:
    prompt = game.current_prompt.name
    if prompt != "PLAY_TURN":
        return prompt
    if game.free_roads_available:
        return "road_building"
    return "turn+rolled" if game.player(game.current_turn_index).has_rolled else "turn"


def generate_positions(game: Any, count: int, seed: int = 0) -> list[dict]:
    """``count`` declarable positions of random games of ``game`` (a GameConfig), cycling
    through :data:`KINDS` of prompt, each seen by the seat to act."""
    rng = random.Random(f"{seed}|{game.name}")
    out: list[dict] = []
    for _ in range(100 * count):
        if len(out) >= count:
            break
        want = KINDS[len(out) % len(KINDS)]
        map_seed, game_seed = rng.randrange(10**6), rng.randrange(10**6)
        played = game.cppanatron_game(map_seed, game_seed)
        for _ in range(rng.randrange(1, 4000)):
            if played.winning_color is not None:
                break
            if _kind(played) == want and rng.random() < 0.3:
                pos = {"game": game.to_dict(), "map_seed": map_seed, "map": game.map_spec(map_seed),
                       "seed": game_seed, "seat": played.current_player_index, **_declare(played)}
                try:
                    declared = _position_game(pos)
                except ValueError:
                    break
                if (declared.current_player_index == played.current_player_index
                        and sorted(map(str, declared.playable_actions)) == sorted(map(str, played.playable_actions))):
                    out.append(pos)
                break
            played.execute(rng.choice(played.playable_actions))
    return out


def _declare(game) -> dict:
    def holder(flag: str) -> int:
        return next((seat for seat, p in enumerate(game.players) if getattr(p, flag)), -1)

    turn = game.current_turn_index
    players = [{"settlements": list(p.settlements), "cities": list(p.cities), "roads": [list(r) for r in p.roads],
                "resources": list(p.resources), "development_cards": list(p.development_cards),
                "played_development_cards": list(p.played_development_cards),
                "discards": int(game.remaining_discards(seat))} for seat, p in enumerate(game.players)]
    options = {"robber": list(game.board.robber_coordinate), "current_player": turn,
               "acting_player": game.current_player_index, "has_rolled": game.player(turn).has_rolled,
               "has_played_development_card": game.player(turn).has_played_development_card_in_turn,
               "completed_turns": game.completed_turns, "longest_road": holder("has_road"),
               "largest_army": holder("has_army"), "prompt": game.current_prompt.name,
               "free_roads": int(game.free_roads_available)}
    return {"players": players, "options": options}


def select_positions(dims: Mapping[str, int], n: int = SELF_TEST_POSITIONS, games: Sequence[Any] = ()) -> list[dict]:
    """``n`` packaged positions that fit ``dims``, games with all ``max_players`` seats
    first, topped up from ``games``."""
    fitting = [p for p in packaged_positions() if _fits(p, dims)]
    fitting.sort(key=lambda p: p["game"]["num_players"] != dims["max_players"])
    chosen = fitting[:n]
    for game in games:
        if len(chosen) >= n:
            break
        chosen += [p for p in generate_positions(game, n - len(chosen)) if _fits(p, dims)]
    if not chosen:
        raise CheckpointError(f"no self-test position fits dims {dict(dims)}")
    return chosen[:n]


def encode_positions(positions: Sequence[Mapping[str, Any]], dims: Mapping[str, int], level: str,
                     critic_level: str | None = None) -> np.ndarray:
    """Rows as the VecEnv writes them: the actor's row, plus the critic's players block."""
    from catanrl.envs.encoding import critic_level_of, make_encoder

    actor = make_encoder(dims, level)
    critic = critic_level_of(level, critic_level)
    critic_encoder = make_encoder(dims, critic) if critic else None
    rows = []
    for pos in positions:
        game = _position_game(pos)
        row = actor.encode(game, int(pos["seat"]))
        if critic_encoder is not None:
            row = np.concatenate([row, critic_encoder.encode_players(game, int(pos["seat"]))])
        rows.append(row)
    return np.stack(rows).astype(np.float32)


def _outputs(model: nn.Module, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(logits, values, logits with every action legal): the last shows every policy head's output."""
    from catanrl.seeding import determinism

    offset, (size,) = model.layout["action_mask"]
    open_obs = obs.clone()
    open_obs[:, offset:offset + size] = 1.0
    with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False), determinism("strict"), \
            torch.no_grad():
        logits, values = model(obs)
        open_logits, _ = model(open_obs)
    return logits.float(), values.float(), open_logits.float()


def make_self_test(model: nn.Module, dims: Mapping[str, int], level: str, n: int = SELF_TEST_POSITIONS,
                   *, critic_level: str | None = None, games: Sequence[Any] = ()) -> dict:
    """Positions, rows and outputs of a CPU copy of ``model`` (strict)."""
    positions = select_positions(dims, n, games)
    obs = torch.from_numpy(encode_positions(positions, dims, level, critic_level))
    cpu = _rebuild(model.spec, model.dims, model.levels, {k: v.detach().cpu() for k, v in model.state_dict().items()})
    logits, values, open_logits = _outputs(cpu, obs)
    return {"positions": json.dumps(positions, separators=(",", ":")), "obs": obs, "logits": logits,
            "value": values, "open_logits": open_logits}


def _rebuild(spec: Mapping[str, Any], dims: Mapping[str, int], levels: Mapping[str, str],
             state: Mapping[str, torch.Tensor] | None) -> nn.Module:
    from catanrl.envs.encoding import action_size, critic_level_of, row_layout
    from catanrl.models import build_model

    critic = critic_level_of(levels["level"], levels.get("critic_level"))
    model = build_model(spec, dims=dims, levels=dict(levels), num_actions=action_size(dims),
                        layout=row_layout(dims, levels["level"], critic))
    if state is not None:
        model.load_state_dict(state)
    return model.eval()


def _layout_hashes(dims: Mapping[str, int], levels: Mapping[str, str]) -> dict[str, str]:
    return {"actor": layout_hash(dims, levels["level"]),
            "critic": layout_hash(dims, levels.get("critic_level") or levels["level"])}


# ---------------------------------------------------------------- save


def save(path: Path, model: nn.Module, *, run: str | None, epoch: int, agent_steps: int,
         config: Mapping[str, Any] | None, code: Mapping[str, Any] | None) -> CheckpointRef:
    """Write ``model`` atomically. The same epoch and weights again is a no-op;
    other weights under an existing path raise."""
    from catanrl.config import config_hash

    path = Path(path)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sha = weights_sha256(state)
    ref = CheckpointRef(run=run, which=f"e{epoch}", epoch=int(epoch), agent_steps=int(agent_steps), path=path,
                        weights_sha256=sha)
    if path.exists():
        header = Checkpoint.load(path).header
        if header["weights_sha256"] == sha and header["epoch"] == epoch:
            return ref
        raise CheckpointError(f"{path} exists with other weights or epoch")
    levels = dict(model.levels)
    games = _games_of(config)
    payload = {
        "format": FORMAT, "schema": CKPT_SCHEMA, "run": run, "epoch": int(epoch), "agent_steps": int(agent_steps),
        "weights_sha256": sha, "config": dict(config or {}), "config_hash": config_hash(config) if config else None,
        "model": dict(model.spec), "parts": {k: list(v) for k, v in model.parts.items()},
        "obs": {"level": levels["level"], "critic_level": levels.get("critic_level"), "dims": dict(model.dims),
                "layout_hash": _layout_hashes(model.dims, levels)},
        "code": dict(code or {}),
        "self_test": make_self_test(model, model.dims, levels["level"], critic_level=levels.get("critic_level"),
                                    games=games),
        "state_dict": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return ref


def _games_of(config: Mapping[str, Any] | None) -> list[Any]:
    from catanrl.games import load_game

    out = []
    for entry in ((config or {}).get("env") or {}).get("scenarios") or []:
        try:
            out.append(load_game(entry["game"] if isinstance(entry, Mapping) else entry))
        except (OSError, ValueError, KeyError):
            continue
    return out


# ---------------------------------------------------------------- load


@dataclass
class Checkpoint:
    ref: CheckpointRef
    header: dict
    state_dict: dict = field(repr=False)
    _tested: set = field(default_factory=set, repr=False)

    @classmethod
    def load(cls, path: str | Path, *, ref: CheckpointRef | None = None) -> Checkpoint:
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (RuntimeError, OSError, EOFError, ValueError, pickle.UnpicklingError) as error:  # torn or foreign
            raise WeightsCorrupt(f"{path}: cannot be read as a checkpoint ({error})") from None
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise SchemaMismatch(f"{path} is not a catanrl checkpoint")
        if payload.get("schema") != CKPT_SCHEMA:
            raise SchemaMismatch(f"{path}: checkpoint schema {payload.get('schema')}, this code reads {CKPT_SCHEMA}")
        state = payload.pop("state_dict")
        if weights_sha256(state) != payload["weights_sha256"]:
            raise WeightsCorrupt(f"{path}: the weights do not match their sha256")
        if ref is not None and ref.weights_sha256 != payload["weights_sha256"]:
            raise WeightsCorrupt(f"{path}: weights {payload['weights_sha256'][:12]} differ from the pinned "
                                 f"{ref.weights_sha256[:12]}")
        if ref is None:  # loaded by path: its identity is the file, whatever run its header names
            ref = CheckpointRef(run=None, which=f"e{payload['epoch']}", epoch=payload["epoch"],
                                agent_steps=payload["agent_steps"], path=path.resolve(),
                                weights_sha256=payload["weights_sha256"])
        return cls(ref=ref, header=payload, state_dict=state)

    @property
    def dims(self) -> dict[str, int]:
        return dict(self.header["obs"]["dims"])

    @property
    def levels(self) -> dict[str, str]:
        obs = self.header["obs"]
        return {"level": obs["level"], "critic_level": obs.get("critic_level") or obs["level"]}

    @property
    def dims_agnostic(self) -> bool:
        from catanrl.models import MODEL_CLASSES

        return MODEL_CLASSES[self.header["model"]["class"]].dims_agnostic

    @property
    def recurrent(self) -> bool:
        """A network with a per-row state between decisions (class ``lstm``); its
        self-test pins the zero-state function on single positions."""
        from catanrl.models import MODEL_CLASSES

        return MODEL_CLASSES[self.header["model"]["class"]].recurrent

    def self_test(self, *, allow_drift: bool = False, part: str = "all") -> None:
        """Checks 2, 4 and 5 (check 1 ran at load; 3 needs target dims)."""
        if (part, allow_drift) in self._tested or ("all", allow_drift) in self._tested:
            return
        levels, dims = self.levels, self.dims
        stored = self.header["obs"]["layout_hash"]
        if stored != _layout_hashes(dims, levels):
            raise SchemaMismatch(f"{self.ref}: the encoder or action layout changed since it was saved")
        if not allow_drift:
            test = self.header["self_test"]
            positions = json.loads(test["positions"])
            obs = encode_positions(positions, dims, levels["level"], levels["critic_level"])
            if obs.shape != tuple(test["obs"].shape) or not np.array_equal(obs, test["obs"].numpy()):
                raise EncoderDrift(f"{self.ref}: re-encoding its self-test positions gives other rows")
            model = _rebuild(self.header["model"], dims, levels, self.state_dict)
            logits, values, open_logits = _outputs(model, test["obs"])
            legal = test["logits"] > -1e8
            logits_ok = torch.equal(legal, logits > -1e8) and torch.allclose(
                logits[legal], test["logits"][legal], rtol=LOGIT_TOL, atol=LOGIT_TOL)
            if "open_logits" in test:  # every head, not only the legal actions' (older checkpoints lack it)
                logits_ok = logits_ok and torch.allclose(open_logits, test["open_logits"], rtol=LOGIT_TOL,
                                                         atol=LOGIT_TOL)
            if part in ("actor", "all") and not logits_ok:
                raise ModelDrift(f"{self.ref}: the rebuilt network's logits differ from its self-test")
            values_ok = torch.allclose(values, test["value"], rtol=VALUE_TOL, atol=VALUE_TOL)
            if part in ("critic", "all") and not values_ok:
                raise ModelDrift(f"{self.ref}: the rebuilt network's values differ from its self-test")
        self._tested.add((part, allow_drift))

    def model(self, part: Literal["actor", "critic", "all"] = "actor", device: str = "cpu", *,
              allow_drift: bool = False, dims: Mapping[str, int] | None = None) -> nn.Module:
        """The self-tested network (every part's weights), on ``device``, in eval mode.

        ``part`` says which outputs the self-test must reproduce. ``dims``
        rebuilds a dims-agnostic network at other dims; others raise.
        """
        if part not in ("actor", "critic", "all"):
            raise ValueError(f"part must be actor, critic or all, not {part!r}")
        self.self_test(allow_drift=allow_drift, part=part)
        target = self.dims if dims is None else {k: int(v) for k, v in dict(dims).items()}
        if target != self.dims and not self.dims_agnostic:
            raise DimsMismatch(f"{self.ref}: a {self.header['model']['class']} network trained at dims {self.dims} "
                               f"cannot run at {target}")
        model = _rebuild(self.header["model"], target, self.levels, self.state_dict)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        model.catanrl_ref = str(self.ref)
        model.catanrl_allow_drift = bool(allow_drift)
        return model


# ---------------------------------------------------------------- warm starts


def load_parts(model: nn.Module, init_from: Mapping[str, Any]) -> list[CheckpointRef]:
    """Copy the ``init_from`` parts into ``model``, each through the full checks.

    ``init_from = {"ckpt": REF or {"actor": REF, "critic": REF}, "parts": [...]}``.
    Every key of a part must exist in the source with the same shape.
    """
    def load_checkpoint(ref: Any) -> Checkpoint:
        if str(ref).endswith(".pt"):
            return Checkpoint.load(ref)
        from catanrl.runs import load_checkpoint as load_run_checkpoint

        return load_run_checkpoint(ref)

    ckpt = init_from["ckpt"]
    parts = list(init_from.get("parts") or (ckpt.keys() if isinstance(ckpt, Mapping) else ("actor", "critic")))
    target = model.state_dict()
    loaded, refs = {}, []
    for part in parts:
        ref = ckpt[part] if isinstance(ckpt, Mapping) else ckpt
        source = load_checkpoint(ref)
        source.self_test(part=part)
        refs.append(source.ref)
        prefixes = tuple(model.parts[part])
        mine = {k for k in target if k.startswith(prefixes)}
        theirs = {k for k in source.state_dict if k.startswith(prefixes)}
        if mine != theirs:
            raise CheckpointError(f"init_from {part} from {source.ref}: keys differ "
                                  f"(missing {sorted(mine - theirs)[:3]}, extra {sorted(theirs - mine)[:3]})")
        for key in mine:
            if source.state_dict[key].shape != target[key].shape:
                raise CheckpointError(f"init_from {part} from {source.ref}: {key} has shape "
                                      f"{tuple(source.state_dict[key].shape)}, the model {tuple(target[key].shape)}")
            loaded[key] = source.state_dict[key]
    model.load_state_dict({**target, **loaded})
    return refs
