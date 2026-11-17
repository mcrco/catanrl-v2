"""Does the training search miss better moves that a wider, longer search finds?

AlphaZero's self-play explores with the network's own prior at every node of the tree and Dirichlet
noise at the root only. If the prior is too peaked, a move the network gives almost no mass gets
almost no visits, its value is never measured, and the policy target can only confirm the network.
This tool measures that on fixed positions of a checkpoint's own self-play, in four steps:

  positions  self-play as training does it (the trainer's ``SelfPlay``: full/fast searches, noise,
             temperature schedule, the true game), on boards outside the training pool; a few
             decisions of every game are kept, each with what restores its game exactly (the
             board, the game seed and the actions played so far, replayed through a schedule
             VecEnv and checked with ``Game.search_equivalent``: hands, dice deck, RNG).
  search     on every position, a list of search arms. An arm is (name, visits, a temperature on
             the logits at every node, a uniform mix into the prior at every node, root noise on or
             off, seed); the prior changes through the evaluator's logits, so the search itself is
             the engine's and the trainer's settings (c_puct, pending, the true game). Defaults:
               A1, A2  training's full search, two seeds (A1 against A2 is the noise floor)
               C       the same with --wide-visits visits ("long, narrow")
               B       --wide-visits visits and the prior (1 - mix) * p + mix * uniform over the
                       legal moves at every node ("long, wide")
             --grid-visits x --grid-temps adds an arm per cell (no root noise, like the deployed
             search). Finished arms are kept: a rerun adds arms to the same file.
             An arm's move is the argmax of its pruned policy target (the trainer's definition).
  referee    for every position where the arms chose at least two different moves, each distinct
             move once: play it and finish the game --playouts times with the network's own policy
             for both seats (sampled, or the trainer's fast search with --playout-sims). Playout k
             of every move of a position has the same chance seed and the same sampling stream
             (common random numbers). The outcome is a win for the mover.
  report     tables and JSON: disagreement rates (against A1, against the network's top move, and
             against the A1-vs-A2 floor), how A1 treated the moves the other arms chose, the
             referee's paired win-rate differences (bootstrap over positions), and the curve of
             both against visits for every prior.

Each step prints a counter with rate and ETA. Nothing is written under the runs directory.

  python tools/explore_diag.py positions --ckpt az-2p-bundle/main@e900#ad8c2caf507b --out pos.npz
  python tools/explore_diag.py search --positions pos.npz --out search.npz
  python tools/explore_diag.py search --positions pos.npz --out search.npz --no-defaults \\
      --grid-visits 600,2400,9600,16384 --grid-temps 1,1.5,2,4
  python tools/explore_diag.py referee --positions pos.npz --search search.npz --out referee.npz
  python tools/explore_diag.py report --search search.npz --referee referee.npz --out report.json
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE = "A1"  # the arm every other arm's move is compared with
FLOOR = "A2"  # the reference again with another seed: the noise floor
KINDS = ("setup", "discard", "robber", "dev", "build", "other")  # a decision's type, the first that holds
SEP = "__"  # per-arm arrays of a search file: <field>__<arm>
ARM_FIELDS = ("visits", "q", "prior", "target", "chosen", "value", "depth", "max_depth", "nodes", "evaluations",
              "tree_bytes")


def _tool(name: str):
    """Another tool of this directory, as a module."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Progress:
    """A counter on stderr, at most every ``every`` seconds: done of total, the rate and the ETA."""

    def __init__(self, total: int, unit: str, *, every: float = 5.0, per: str = "min",
                 log: Callable[[str], None] | None = None) -> None:
        self.total, self.unit, self.every = int(total), unit, float(every)
        self.scale, self.per = (60.0, "min") if per == "min" else (1.0, "s")
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.start = self.last = time.time()

    def update(self, done: int, note: str = "", *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = f"{max(self.total - done, 0) / rate / 60:.1f} min" if rate > 0 else "?"
        self.log(f"{self.unit} {done:,}/{self.total:,} ({done / max(self.total, 1):.0%}), "
                 f"{rate * self.scale:,.1f}/{self.per}, {elapsed / 60:.1f} min elapsed, ETA {eta}{note}")


def _trim() -> None:
    """Hand freed tree memory back to the system (glibc keeps it otherwise)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _args(args) -> dict[str, Any]:
    """The command line, for a manifest."""
    return {key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, bool, list,
                                                                                  type(None)))}


def check_out(path: str | Path) -> Path:
    """An output path, refused under the runs directory (shared by every worktree)."""
    from catanrl.runs import runs_root

    path = Path(path).resolve()
    root = Path(runs_root()).resolve()
    if path == root or root in path.parents:
        raise SystemExit(f"{path} is under the runs directory {root}: write the diagnostic elsewhere")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_npz(path: Path, manifest: Mapping[str, Any], data: Mapping[str, np.ndarray]) -> None:
    """Write ``data`` and its manifest, whole or not at all (a temporary file, then a rename)."""
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, manifest=np.array(json.dumps(manifest)), **data)
    os.replace(tmp, path)


def load_npz(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files if key != "manifest"}
        manifest = json.loads(str(f["manifest"]))
    return manifest, data


# ---------------------------------------------------------------- arms


@dataclass(frozen=True)
class Arm:
    """A search: ``visits`` at the root, the prior at every node the softmax of the network's
    logits over ``temp``, mixed with the uniform distribution over the node's legal moves at
    weight ``mix``; Dirichlet noise at the root (training's alpha and weight) or none; ``seed``."""

    name: str
    visits: int
    temp: float = 1.0
    mix: float = 0.0
    noise: bool = True
    seed: int = 1

    def __post_init__(self) -> None:
        if not self.name or SEP in self.name or not self.name.replace("_", "").replace(".", "").isalnum():
            raise ValueError(f"arm name {self.name!r}: letters, digits, '.' and single '_' only")
        if self.visits < 1 or self.temp <= 0.0 or not 0.0 <= self.mix < 1.0:
            raise ValueError(f"arm {self.name}: visits >= 1, temp > 0 and 0 <= mix < 1")

    @property
    def plain(self) -> bool:
        """The network's own prior at every node."""
        return self.temp == 1.0 and self.mix == 0.0

    @property
    def prior_key(self) -> tuple[float, float]:
        return (float(self.temp), float(self.mix))


def _number(value: float) -> str:
    return f"{value:g}"


def parse_arm(text: str) -> Arm:
    """``name:visits=9600,temp=1.5,mix=0.25,noise=0,seed=3`` (visits is required)."""
    name, _, rest = text.partition(":")
    fields: dict[str, Any] = {}
    casts = {"visits": int, "temp": float, "mix": float, "seed": int,
             "noise": lambda v: v.lower() in ("1", "true", "yes", "on")}
    for item in filter(None, rest.split(",")):
        key, eq, value = item.partition("=")
        if key not in casts or not eq:
            raise ValueError(f"--arm {text!r}: {item!r} is not one of {sorted(casts)}=value")
        fields[key] = casts[key](value)
    if "visits" not in fields:
        raise ValueError(f"--arm {text!r}: visits=N is required")
    return Arm(name=name.strip(), **fields)


def default_arms(sims: int, *, wide_visits: int, wide_mix: float, wide_temp: float) -> list[Arm]:
    """A1 and A2 (training's search, two seeds), C (long, narrow) and B (long, wide)."""
    return [Arm(REFERENCE, sims, seed=1), Arm(FLOOR, sims, seed=2), Arm("C", wide_visits, seed=3),
            Arm("B", wide_visits, temp=wide_temp, mix=wide_mix, seed=4)]


def grid_arms(visits: Sequence[int], temps: Sequence[float], *, mix: float = 0.0, seed: int = 100) -> list[Arm]:
    """An arm per cell of visits x temperature, without root noise: ``g<visits>_t<temp>``."""
    arms = []
    for temp in temps:
        for count in visits:
            name = f"g{int(count)}_t{_number(temp)}" + (f"_m{_number(mix)}" if mix else "")
            arms.append(Arm(name, int(count), temp=float(temp), mix=float(mix), noise=False, seed=seed))
    return arms


def arm_list(args, sims: int) -> list[Arm]:
    arms = [] if args.no_defaults else default_arms(sims, wide_visits=args.wide_visits, wide_mix=args.wide_mix,
                                                    wide_temp=args.wide_temp)
    arms += [parse_arm(text) for text in args.arm or []]
    if bool(args.grid_visits) != bool(args.grid_temps):
        raise SystemExit("--grid-visits and --grid-temps go together")
    if args.grid_visits:
        arms += grid_arms([int(v) for v in args.grid_visits.split(",")], [float(t) for t in args.grid_temps.split(",")],
                          mix=args.grid_mix, seed=args.grid_seed)
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"arm names repeat: {names}")
    if not arms:
        raise SystemExit("no arms: drop --no-defaults or give --arm / --grid-visits and --grid-temps")
    return arms


# ---------------------------------------------------------------- priors


def reshape_logits(logits: np.ndarray, legal: np.ndarray, *, temp: float = 1.0, mix: float = 0.0) -> np.ndarray:
    """Logits whose softmax over each row's legal moves is ``(1 - mix) * softmax(logits / temp) +
    mix * uniform`` (float32 [R, A]; entries off ``legal`` are left as they are: the engine's
    prior is a softmax over the node's actions only). The arithmetic is float64, on the legal
    entries alone (a row has a handful of them among hundreds of actions)."""
    logits = np.asarray(logits)
    out = logits.astype(np.float32)
    rows, cols = np.nonzero(np.asarray(legal, dtype=bool))  # row-major: a row's entries are contiguous
    if not len(rows):
        return out
    start = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])  # the rows with a legal move
    size = np.diff(np.r_[start, len(rows)])
    owner = np.repeat(np.arange(len(start)), size)
    scores = logits[rows, cols].astype(np.float64) / float(temp)
    p = np.exp(scores - np.maximum.reduceat(scores, start)[owner])
    p = (1.0 - float(mix)) * p / np.add.reduceat(p, start)[owner] + float(mix) / size[owner]
    with np.errstate(divide="ignore"):
        out[rows, cols] = np.maximum(np.log(p), -60.0)  # a legal move at prior 0 stays finite
    return out


class Shaped:
    """An evaluator for ``run_pools`` whose policy logits give the engine an arm's prior at every
    node (``reshape_logits``; a row's legal moves are its own action mask), and that counts the
    rows it evaluated. The values are the network's."""

    def __init__(self, evaluator: Any, mask: slice, *, temp: float = 1.0, mix: float = 0.0,
                 tick: Callable[[int], None] | None = None) -> None:
        self.evaluator, self.mask, self.temp, self.mix = evaluator, mask, float(temp), float(mix)
        self.rows, self.tick = 0, tick

    def __getattr__(self, name: str) -> Any:
        return getattr(self.evaluator, name)

    def __call__(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, values = self.evaluator(observations)
        if self.temp != 1.0 or self.mix != 0.0:
            logits = reshape_logits(logits, observations[:, self.mask] > 0, temp=self.temp, mix=self.mix)
        self.rows += len(observations)
        if self.tick is not None:
            self.tick(self.rows)
        return logits, values


def entropy(p: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The entropy in nats of each position's distribution over its legal moves (flat ``p`` [L],
    position i's entries ``start[i]:start[i + 1]``)."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, -p * np.log(p), 0.0)
    return np.add.reduceat(terms, start[:-1]) if len(start) > 1 else np.zeros(0)


# ---------------------------------------------------------------- the network and its games


@dataclass
class Setup:
    """A checkpoint's network, its run's search settings and the held-out plan."""

    ckpt: Any
    model: Any
    train: dict[str, Any]
    plan: Any
    pools: dict[str, Any]
    device: str

    @property
    def levels(self) -> Mapping[str, Any]:
        return self.model.levels

    @property
    def mask(self) -> slice:
        return slice(self.model.mask_offset, self.model.mask_offset + self.model.num_actions)


def load_setup(ref: str, *, device: str, pool_seed: int | None, overrides: Mapping[str, Any] | None = None) -> Setup:
    """``ref``'s network on ``device`` with its run's ``train`` settings (``overrides`` replace
    keys; on the CPU the kernels are torch's: the fused ones are CUDA only) and its env on a map
    pool from ``pool_seed`` (default: derived from the run's seed), refused if it shares a board
    with the training pool."""
    from catanrl.runs import load_checkpoint
    from catanrl.seeding import derive_seed
    from catanrl.train.alphazero import check_train

    ckpt = load_checkpoint(ref)
    config = ckpt.header.get("config") or {}
    if "env" not in config or (config.get("train") or {}).get("algo") != "alphazero":
        raise SystemExit(f"{ckpt.ref} carries no alphazero run config: its search settings are unknown")
    train = check_train({**config["train"], **{k: v for k, v in (overrides or {}).items() if v is not None}})
    if train["root"] != "puct" or train["policy_target"] != "pruned" or int(train["dets"]) != 0:
        raise SystemExit("the diagnostic reads PUCT roots on the true game with the pruned target "
                         "(root: puct, policy_target: pruned, dets: 0)")
    if not str(device).startswith("cuda"):
        train["kernels"] = "torch"
    seed = int(config["seed"])
    pool_seed = derive_seed(seed, "explore_diag") if pool_seed is None else int(pool_seed)
    plan, pools = _tool("critic_heldout").heldout_plan(config["env"], seed=seed, pool_seed=pool_seed)
    model = ckpt.model("all", device=device)
    model.eval()
    return Setup(ckpt, model, train, plan, pools, device)


def schedule_env(plan: Any, levels: Mapping[str, Any]):
    """A one-game VecEnv that plays the games it is told to (``start_game``), and its buffers."""
    import cppanatron as cpp

    from catanrl.envs.encoding import dice_state_of, encoder_dims, info_level

    vec = cpp.VecEnv(list(plan.scenarios), num_envs=1, dims=encoder_dims(plan.dims),
                     level=info_level(levels["level"]), seed=0, dice_state=dice_state_of(levels), auto_reset=False)
    return vec, vec.allocate()


def restore_games(plan: Any, levels: Mapping[str, Any], data: Mapping[str, np.ndarray],
                  index: Sequence[int] | None = None) -> list[Any]:
    """The games of positions ``index`` (default: all) of a positions file, each as it stood at
    its decision: its game is started again on its board with its seed and its actions are
    replayed up to the position's ply. The engine is deterministic given the seed, so the copy
    has the original's hands, development deck, dice state and random stream."""
    vec, b = schedule_env(plan, levels)
    maps = [json.loads(str(m)) for m in data["maps"]]
    index = np.arange(len(data["game"])) if index is None else np.asarray(index, dtype=np.int64)
    out: list[Any] = [None] * len(index)
    order = np.lexsort((data["ply"][index], data["game"][index]))
    seats = ["learner"] * int(plan.dims["max_players"])
    current, played = -1, 0
    for j in order:
        i = int(index[j])
        g, ply = int(data["game"][i]), int(data["ply"][i])
        if g != current or ply < played:
            vec.reset(0)
            scenario = int(data["g_scenario"][g])
            count = int(plan.scenarios[scenario].num_players)
            vec.start_game(0, map=maps[int(data["g_map"][g])], game_seed=int(data["g_seed"][g]),
                           seats=seats[:count], scenario=scenario)
            current, played = g, 0
        actions = data["g_action"][int(data["g_start"][g]):int(data["g_start"][g + 1])]
        while played < ply:
            rows = np.flatnonzero(b["acting"])
            if len(rows) != 1:
                raise RuntimeError(f"game {g}: {len(rows)} rows act at decision {played} of its replay")
            b["actions"][rows[0]] = int(actions[played])
            try:
                vec.step()
            except ValueError as error:  # not a legal move there: the replay is another game
                raise RuntimeError(f"game {g} does not restore at decision {played}: {error}") from error
            played += 1
        out[j] = vec.game(0).copy()
    return out


# ---------------------------------------------------------------- positions


def gather(setup: Setup, *, games: int, envs: int, per_game: int, seed: int, threads: int, groups: int = 1,
           every: float = 5.0, log=None) -> tuple[dict[str, np.ndarray], list[Any]]:
    """``games`` self-play games of the trainer's ``SelfPlay`` (its searches, noise, temperature
    schedule and tree reuse), ``games / envs`` per env so that long games are not cut, and a
    uniform sample of up to ``per_game`` decisions of each (the VecEnv presents only decisions
    with at least two legal moves). Per game: its board, seed, scenario, the actions played at
    its decisions and the winner; per position: its game, ply (the decision's index), turn, the
    seat to move, whether training searched it at the full budget, the move played and whether
    the mover won. Also returned: each position's game as self-play held it (for the check that the
    file restores it)."""
    from catanrl.seeding import derive_seed
    from catanrl.train.alphazero import ReplayBuffer, SelfPlay

    if games % envs:
        raise ValueError(f"--games {games} must be a multiple of --envs {envs}")
    per_env = games // envs
    train = setup.train
    selfplay = SelfPlay(setup.plan, train, model=setup.model, num_envs=envs, threads=threads,
                        seed=derive_seed(seed, "rollout"), device=setup.device, mode="fast", groups=groups)
    vec, b, seats = selfplay.vec, selfplay.buffers, selfplay.players
    replay = ReplayBuffer(max(4 * envs, 1024), vec.observation_size, vec.action_size, seats,
                          outcome=selfplay.outcome)  # the trainer files finished games; nothing reads them here
    rng = np.random.default_rng(derive_seed(seed, "reservoir"))
    map_index: dict[str, int] = {}

    def header(env: int) -> tuple[int, int, int]:
        text = json.dumps(vec.game(env).board.map.spec().to_dict(), sort_keys=True)
        return map_index.setdefault(text, len(map_index)), int(vec.game_seed(env)), int(vec.scenario_index(env))

    headers = [header(env) for env in range(envs)]
    actions: list[list[int]] = [[] for _ in range(envs)]
    kept: list[list[tuple]] = [[] for _ in range(envs)]  # per env, its game's reservoir
    done = np.zeros(envs, np.int64)
    games_out, positions_out, originals = [], [], []
    progress = Progress(games, "games", every=every, log=log)
    iteration, leaves, decisions = 0, 0.0, 0
    while (done < per_env).any():
        iteration += 1
        draws = np.random.default_rng(derive_seed(seed, "selfplay", iteration))
        selfplay.check_kernels()
        for step in range(int(train["steps_per_iteration"])):
            acting = np.flatnonzero(b["acting"])
            row_envs = b["env_index"][acting].astype(np.int64)
            if len(np.unique(row_envs)) != len(row_envs):
                raise RuntimeError("two seats of one game act at once: the decision count would skip")
            # the full/fast draw step() is about to make, from a copy of its generator
            full = copy.deepcopy(draws).random(len(acting)) < float(train["full_search_prob"])
            for i, (row, env) in enumerate(zip(acting, row_envs)):
                if done[env] >= per_env:
                    continue
                t = len(actions[env])
                slot = t if t < per_game else int(rng.integers(0, t + 1))
                if slot < per_game:  # decision t replaces a kept one with probability per_game / (t + 1)
                    game = vec.game(int(env)).copy()
                    entry = (t, int(game.num_turns), int(b["seat"][row]), bool(full[i]),
                             int((b["observations"][row, setup.mask] > 0).sum()), game)
                    if slot < len(kept[env]):
                        kept[env][slot] = entry
                    else:
                        kept[env].append(entry)
            stats = {"leaves": 0.0, "rounds": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}
            selfplay.step(iteration, step, draws, replay, stats)
            leaves += stats["leaves"]
            decisions += len(acting)
            for row, env in zip(acting, row_envs):  # the moves step() chose: it leaves them in the buffer
                actions[env].append(int(b["actions"][row]))
            for env in np.flatnonzero(b["terminals"].reshape(-1, seats)[:, 0]):
                if done[env] < per_env:
                    rewards = b["rewards"][env * seats:(env + 1) * seats]
                    winner = int(np.argmax(rewards)) if rewards.max() > 0 else -1
                    gid = int(env * per_env + done[env])
                    games_out.append((gid, *headers[env], winner, list(actions[env])))
                    for t, turn, seat, searched, legal, game in kept[env]:
                        positions_out.append((gid, t, turn, seat, int(searched), actions[env][t],
                                              int(winner == seat), legal))
                        originals.append(game)
                    done[env] += 1
                headers[env], actions[env], kept[env] = header(int(env)), [], []
            progress.update(int(np.minimum(done, per_env).sum()),
                            f", positions {len(positions_out):,}, {decisions:,} decisions, {leaves:,.0f} leaves")
            if not (done < per_env).any():
                break
        selfplay.drop_trees()
    progress.update(games, f", positions {len(positions_out):,}, {decisions:,} decisions, {leaves:,.0f} leaves",
                    force=True)
    games_out.sort(key=lambda g: g[0])
    renumber = {g[0]: k for k, g in enumerate(games_out)}
    p = np.array([(renumber[q[0]], *q[1:]) for q in positions_out], dtype=np.int64).reshape(-1, 8)
    order = np.lexsort((p[:, 1], p[:, 0]))  # by game, then by decision
    p, originals = p[order], [originals[k] for k in order]
    lengths = np.array([len(g[5]) for g in games_out], dtype=np.int64)
    data = {
        "maps": np.array(sorted(map_index, key=map_index.get)),
        "g_map": np.array([g[1] for g in games_out], dtype=np.int32),
        "g_seed": np.array([g[2] for g in games_out], dtype=np.uint64),
        "g_scenario": np.array([g[3] for g in games_out], dtype=np.int32),
        "g_winner": np.array([g[4] for g in games_out], dtype=np.int8),
        "g_start": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "g_action": np.array([a for g in games_out for a in g[5]], dtype=np.int16),
        "game": p[:, 0].astype(np.int32), "ply": p[:, 1].astype(np.int32), "turn": p[:, 2].astype(np.int32),
        "seat": p[:, 3].astype(np.int8), "full": p[:, 4].astype(bool), "played": p[:, 5].astype(np.int16),
        "y": p[:, 6].astype(np.int8), "nlegal": p[:, 7].astype(np.int16),
        "selfplay_leaves": np.array(leaves), "selfplay_decisions": np.array(decisions),
    }
    return data, originals


def verify_positions(setup: Setup, data: Mapping[str, np.ndarray], originals: Sequence[Any] | None = None) -> None:
    """Every kept position restored (``restore_games``) and checked: the seat to move, the turn,
    the legal moves (the game's and its row's), the move played being one of them and, against
    ``originals`` (the games self-play held), ``Game.search_equivalent`` both ways."""
    from catanrl.envs.encoding import dice_state_of, make_encoder

    encoder = make_encoder(setup.plan.dims, setup.levels["level"], dice_state_of(setup.levels))
    space = encoder.action_space
    games = restore_games(setup.plan, setup.levels, data)
    for i, game in enumerate(games):
        legal = np.asarray(space.mask(game), dtype=bool)
        if int(game.current_player_index) != int(data["seat"][i]) or int(game.num_turns) != int(data["turn"][i]) \
                or int(legal.sum()) != int(data["nlegal"][i]) or not legal[int(data["played"][i])]:
            raise RuntimeError(f"position {i} (game {int(data['game'][i])}, ply {int(data['ply'][i])}) does not "
                               "restore: its replay is another game")
        if not np.array_equal(encoder.encode(game, int(data["seat"][i]))[setup.mask] > 0, legal):
            raise RuntimeError(f"position {i}: its row's action mask is not the game's legal moves")
        if originals is not None and not (game.search_equivalent(originals[i])
                                          and originals[i].search_equivalent(game)):
            raise RuntimeError(f"position {i}: the restored game is not the one self-play searched")


def positions(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    out = check_out(args.out)
    setup = load_setup(args.ckpt, device=args.device, pool_seed=args.pool_seed,
                       overrides={"sims": args.sims, "fast_sims": args.fast_sims})
    from catanrl.train.alphazero import SEARCH_GROUPS

    start = time.time()
    data, originals = gather(setup, games=args.games, envs=args.envs, per_game=args.per_game, seed=args.seed,
                             threads=args.env_threads,
                             groups=SEARCH_GROUPS if args.device.startswith("cuda") else 1, every=args.progress)
    verify_positions(setup, data, originals)
    manifest = {"step": "positions", "ckpt": str(setup.ckpt.ref), "train": setup.train,
                "env": setup.ckpt.header["config"]["env"], "run_seed": int(setup.ckpt.header["config"]["seed"]),
                "levels": dict(setup.levels),
                "dims": dict(setup.plan.dims), **setup.pools, "seed": args.seed, "envs": args.envs,
                "per_game": args.per_game, "games": len(data["g_seed"]), "positions": len(data["game"]),
                "decisions": int(data["g_start"][-1]), "truncated_games": int((data["g_winner"] < 0).sum()),
                "full_share": float(data["full"].mean()) if len(data["full"]) else 0.0,
                "selfplay_leaves": float(data["selfplay_leaves"]),
                "selfplay_decisions": int(data["selfplay_decisions"]), "restore_checked": True,
                "device": args.device, "args": _args(args), "git": git_sha(), "wall_s": time.time() - start}
    save_npz(out, manifest, data)
    print(f"{out}: {manifest['positions']:,} positions of {manifest['games']:,} games "
          f"({manifest['decisions']:,} decisions, {manifest['truncated_games']} truncated), "
          f"{manifest['wall_s'] / 60:.1f} min")
    return manifest


# ---------------------------------------------------------------- search

#: Resident bytes a tree takes per root visit while it is searched, for the positions in flight.
#: Measured on the CPU with six of the bundle's positions in flight (2026-11-16): the peak resident
#: memory grew 8.2-9.7 kB per visit at 16,384 visits and 6.2-9.1 kB at 9,600, about 2.8 times the
#: engine's own ``tree_bytes`` (2.2-5.3 kB per visit; a tree has more nodes than visits).
TREE_BYTES_PER_VISIT = 10_000


def positions_digest(data: Mapping[str, np.ndarray]) -> str:
    """What ties a search or referee file to its positions file."""
    from catanrl.canon import digest

    return digest([data[key].tolist() for key in ("g_seed", "g_action", "game", "ply")])[:16]


def legal_index(legal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat legal lists of a mask [N, A]: ``start`` [N + 1] and ``action`` [L]."""
    rows, action = np.nonzero(legal)
    start = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=len(legal)))]).astype(np.int64)
    return start, action.astype(np.int16)


def root_facts(setup: Setup, games: Sequence[Any], *, batch: int = 2048) -> dict[str, np.ndarray]:
    """What the network says at each position, without a search: its rows' legal moves (flat),
    the masked logits and policy there, the top move, the mover's value and the decision's kind
    (the first of :data:`KINDS` that holds), and the action space's names."""
    import torch

    from catanrl.envs.encoding import dice_state_of, make_encoder
    from catanrl.train.alphazero import phases_of

    encoder = make_encoder(setup.plan.dims, setup.levels["level"], dice_state_of(setup.levels))
    rows = np.stack([encoder.encode(game, int(game.current_player_index)) for game in games]).astype(np.float32)
    legal = rows[:, setup.mask] > 0
    logits, values = [], []
    with torch.inference_mode():
        for at in range(0, len(rows), batch):
            obs = torch.from_numpy(rows[at:at + batch]).to(setup.device)
            logits.append(setup.model.policy_logits(obs).double().cpu().numpy())
            values.append(setup.model(obs)[1].float().cpu().numpy()[:, 0])
    logits = np.where(legal, np.concatenate(logits), -np.inf)
    start, action = legal_index(legal)
    owner = np.repeat(np.arange(len(rows)), np.diff(start))
    flat = logits[owner, action]
    p = np.exp(flat - np.maximum.reduceat(flat, start[:-1])[owner])
    p /= np.add.reduceat(p, start[:-1])[owner]
    phases = phases_of(setup.model, rows)
    kind = np.full(len(rows), len(KINDS) - 1, dtype=np.int8)
    for k in reversed(range(len(KINDS) - 1)):
        kind[phases[KINDS[k]]] = k
    return {"action_names": np.array(encoder.action_space.names()),
            "legal_start": start, "legal_action": action, "logit": flat.astype(np.float32),
            "net_p": p.astype(np.float32), "net_top": logits.argmax(1).astype(np.int16),
            "net_value": np.concatenate(values).astype(np.float32), "kind": kind}


def inflight_for(visits: int, *, tree_gb: float, max_inflight: int) -> int:
    """Positions searched at once: as many trees of ``visits`` visits as fit ``tree_gb``."""
    return int(max(1, min(max_inflight, tree_gb * 2**30 // (visits * TREE_BYTES_PER_VISIT))))


def search_arm(setup: Setup, arm: Arm, games: Sequence[Any], start: np.ndarray, action: np.ndarray, *,
               evaluator: Any, seed: int, threads: int, inflight: int, every: float = 5.0, log=None
               ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Arm ``arm`` on every game: a new tree per position in a pool with the trainer's settings
    (PUCT root, c_puct, the leaves per round of a full search, the true game) at ``arm.visits``
    visits, root noise as the arm says, the arm's prior at every node (:class:`Shaped`). Per
    position, over its legal moves (flat): the root's visits, Q (NaN unvisited), the prior the
    tree holds (before noise) and the pruned policy target; its argmax (the arm's move), the
    mover's root value and the tree's size."""
    from catanrl.envs.encoding import dice_state_of
    from catanrl.search.pool import make_pool, run_pools, search_config
    from catanrl.seeding import derive_seed
    from catanrl.train.alphazero import pruned_visits, root_log_priors

    train, levels = setup.train, setup.levels
    pending = int(train["pending"])
    params = {"sims": arm.visits, "dets": 0, "oracle": True, "root": "puct", "c_puct": train["c_puct"],
              "gumbel_m": train["gumbel_m"], "pending": pending, "noise": arm.noise}
    config = search_config(params, max_players=int(setup.plan.dims["max_players"]), threads=threads)
    pool = make_pool(dict(setup.plan.dims), levels["level"], config, critic_level=levels["critic_level"],
                     dice_state=dice_state_of(levels))
    count, size = len(games), int(pool.action_size)
    total = len(action)
    out = {"visits": np.zeros(total, np.uint32), "q": np.full(total, np.nan, np.float32),
           "prior": np.zeros(total, np.float32), "target": np.zeros(total, np.float32),
           "chosen": np.zeros(count, np.int16), "value": np.zeros(count, np.float32),
           "depth": np.zeros(count, np.float32), "max_depth": np.zeros(count, np.int32),
           "nodes": np.zeros(count, np.int64), "evaluations": np.zeros(count, np.int64),
           "tree_bytes": np.zeros(count, np.int64)}
    progress = Progress(count * arm.visits, f"{arm.name}: visits", every=every, per="s", log=log)
    done, leaves, began = 0, 0, time.time()
    for at in range(0, count, inflight):
        chunk = range(at, min(at + inflight, count))
        base = leaves
        shaped = Shaped(evaluator, setup.mask, temp=arm.temp, mix=arm.mix,
                        tick=lambda rows, done=done, base=base: progress.update(
                            min(done * arm.visits + rows, count * arm.visits),
                            f", positions {done:,}/{count:,}, {base + rows:,} leaves"))
        try:
            ids = [pool.add(games[i], seed=derive_seed(seed, "search", arm.seed, i)) for i in chunk]
            leaves += run_pools([pool], shaped, batch_rows=[len(chunk) * pending])
            for i, sid in zip(chunk, ids):
                edges = action[start[i]:start[i + 1]].astype(np.int64)
                legal = np.zeros(size, dtype=bool)
                legal[edges] = True
                visits, q = pool.visits(sid), pool.q(sid)
                priors = pool.root_priors(sid)
                if len(priors) != 1 or (priors[0] > 0).sum() != len(edges) or not (priors[0][edges] > 0).all():
                    raise RuntimeError(f"position {i}: the root's actions are not its row's legal moves")
                kept = pruned_visits(root_log_priors(pool, sid), legal, visits, q, c_puct=pool.config.c_puct)
                target = kept / max(kept.sum(), 1.0)
                span = slice(int(start[i]), int(start[i + 1]))
                out["visits"][span], out["q"][span] = visits[edges], q[edges]
                out["prior"][span], out["target"][span] = priors[0][edges], target[edges]
                out["chosen"][i] = int(np.argmax(target))
                out["value"][i] = float(pool.value(sid)[0])
                stats = pool.stats(sid)
                out["depth"][i], out["max_depth"][i] = stats.mean_depth, stats.max_depth
                out["nodes"][i], out["evaluations"][i] = stats.nodes, stats.evaluations
                out["tree_bytes"][i] = stats.tree_bytes
        finally:
            if pool.pending:
                pool.cancel()
            for sid in list(pool.ids):
                pool.remove(sid)
            _trim()
        done = chunk[-1] + 1
    progress.update(count * arm.visits, f", positions {done:,}/{count:,}, {leaves:,} leaves", force=True)
    wall = time.time() - began
    info = {**asdict(arm), "leaves": int(leaves), "wall_s": wall, "leaves_per_s": leaves / wall if wall > 0 else 0.0,
            "inflight": int(inflight), "prior_entropy": float(entropy(out["prior"], start).mean()) if count else 0.0,
            "tree_bytes_mean": float(out["tree_bytes"].mean()) if count else 0.0,
            "tree_bytes_max": int(out["tree_bytes"].max()) if count else 0,
            "mean_depth": float(out["depth"].mean()) if count else 0.0}
    return out, info


def search(args) -> dict:
    import torch

    from catanrl.search.evaluator import BatchEvaluator

    torch.set_num_threads(args.threads)
    out = check_out(args.out)
    source, data = load_npz(args.positions)
    setup = load_setup(source["ckpt"], device=args.device, pool_seed=source["pool_seed"],
                       overrides={"sims": args.sims})
    if setup.pools["pool"] != source["pool"]:
        raise SystemExit("the positions' map pool is not the one their manifest's seed gives now")
    arms = arm_list(args, int(setup.train["sims"]))
    if args.limit:
        data = limit_positions(data, args.limit)
    key = positions_digest(data)
    manifest: dict[str, Any] = {"step": "search", "ckpt": str(setup.ckpt.ref), "positions_file": str(args.positions),
                                "positions_digest": key, "positions": len(data["game"]), "seed": args.seed,
                                "train": setup.train, "pending": int(setup.train["pending"]),
                                "c_puct": float(setup.train["c_puct"]), "device": args.device, "arms": {},
                                "runs": []}
    arrays: dict[str, np.ndarray] = {}
    if out.exists():
        manifest, arrays = load_npz(out)
        same = {"ckpt": str(setup.ckpt.ref), "positions_digest": key, "seed": args.seed,
                "c_puct": float(setup.train["c_puct"]), "pending": int(setup.train["pending"])}
        differ = {k: (manifest.get(k), v) for k, v in same.items() if manifest.get(k) != v}
        if differ:
            raise SystemExit(f"{out} holds searches of something else ({differ}): give another --out")
    for arm in arms:
        held = manifest["arms"].get(arm.name)
        if held is not None and {k: held[k] for k in asdict(arm)} != asdict(arm):
            raise SystemExit(f"{out} holds an arm {arm.name} with other settings: rename the new one")
    todo = [arm for arm in arms if arm.name not in manifest["arms"]]
    print(f"{len(data['game']):,} positions; arms to run: {[a.name for a in todo]}; "
          f"already in the file: {list(manifest['arms'])}", file=sys.stderr, flush=True)
    games = restore_games(setup.plan, setup.levels, data)
    facts = root_facts(setup, games)
    if arrays and not (np.array_equal(arrays["legal_action"], facts["legal_action"])
                       and np.allclose(arrays["logit"], facts["logit"], atol=1e-3)):
        raise SystemExit(f"{out}: the network's logits at the positions are not the ones stored there")
    for name in ("game", "ply", "turn", "seat", "full", "played", "y"):
        arrays.setdefault(name, data[name])
    for name, value in facts.items():
        arrays.setdefault(name, value)
    evaluator = BatchEvaluator(setup.model, device=setup.device, infer_rows=int(setup.train["search_rows"]),
                               mode="fast", kernels=setup.train["kernels"])
    run = {"args": _args(args), "git": git_sha(), "arms": [a.name for a in todo], "kernels": setup.train["kernels"],
           "began": time.strftime("%Y-%m-%d %H:%M:%S")}
    manifest["runs"].append(run)
    for n, arm in enumerate(todo):
        inflight = min(len(games), inflight_for(arm.visits, tree_gb=args.tree_gb, max_inflight=args.max_inflight))
        print(f"arm {n + 1}/{len(todo)} {arm}: {inflight} positions in flight", file=sys.stderr, flush=True)
        result, info = search_arm(setup, arm, games, facts["legal_start"], facts["legal_action"],
                                  evaluator=evaluator, seed=args.seed, threads=args.env_threads, inflight=inflight,
                                  every=args.progress)
        for field, value in result.items():
            arrays[f"{field}{SEP}{arm.name}"] = value
        manifest["arms"][arm.name] = info
        save_npz(out, manifest, arrays)  # arm by arm: a rerun goes on from here
        print(f"arm {arm.name}: {info['leaves']:,} leaves in {info['wall_s'] / 60:.1f} min "
              f"({info['leaves_per_s']:,.0f}/s), prior entropy {info['prior_entropy']:.3f} nats, "
              f"tree {info['tree_bytes_mean'] / 2**20:.1f} MB mean, {info['tree_bytes_max'] / 2**20:.1f} MB max",
              file=sys.stderr, flush=True)
    if not todo:
        save_npz(out, manifest, arrays)
    return manifest


def limit_positions(data: Mapping[str, np.ndarray], limit: int) -> dict[str, np.ndarray]:
    """The first ``limit`` positions of a positions file (smokes), its games untouched."""
    per_position = ("game", "ply", "turn", "seat", "full", "played", "y", "nlegal")
    return {key: (value[:limit] if key in per_position else value) for key, value in data.items()}


# ---------------------------------------------------------------- referee


def arm_names(arrays: Mapping[str, np.ndarray]) -> list[str]:
    """The arms of a search file, in the order they were run."""
    return [key.split(SEP, 1)[1] for key in arrays if key.startswith(f"chosen{SEP}")]


def moves_to_judge(arrays: Mapping[str, np.ndarray], arms: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Every distinct move the ``arms`` chose at each position where they chose at least two
    (``pos``, ``move``, sorted): a move is judged once however many arms chose it."""
    chosen = np.stack([arrays[f"chosen{SEP}{name}"] for name in arms], axis=1).astype(np.int64)
    pos, move = [], []
    for i, row in enumerate(chosen):
        distinct = np.unique(row)
        if len(distinct) > 1:
            pos += [i] * len(distinct)
            move += distinct.tolist()
    return np.array(pos, dtype=np.int32), np.array(move, dtype=np.int16)


def sample_rows(logits: np.ndarray, legal: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One legal action per row from the softmax of its masked logits, by inverse CDF at ``u``
    (uniform [0, 1) per row): the same ``u`` at the same row gives the same action."""
    scores = np.where(legal, logits.astype(np.float64), -np.inf)
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    cdf = np.cumsum(p, axis=1)
    target = np.maximum(u, 1e-12)[:, None] * cdf[:, -1:]
    return np.minimum((cdf < target).sum(axis=1), legal.shape[1] - 1 - np.argmax(legal[:, ::-1], axis=1))


@dataclass
class _Playout:
    job: int
    k: int
    game: Any
    rng: np.random.Generator
    mover: int
    ply: int
    length: int = 0


def play_out(setup: Setup, games: Sequence[Any], pos: np.ndarray, move: np.ndarray, ply: np.ndarray, *,
             playouts: int, seed: int, sims: int = 0, batch: int = 4096, threads: int = 1,
             evaluator: Any = None, tick: Callable[[int], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each job ``(pos[j], move[j])``: the position's game (``games[j]``), the move, then the game
    to its end with the network for both seats, ``playouts`` times. Playout k of a position
    reshuffles what its mover cannot know (the development deck's order: ``redeal_hidden_cards``
    with every hand as it is) and seeds the game's dice and steals from (seed, position, k), and
    samples its moves from a stream seeded the same way: common random numbers across the
    position's moves. ``sims`` 0 samples the raw policy; above 0 every decision is the trainer's
    fast search of that many visits (no noise, its temperature schedule going on from the
    position's ply). Forced moves are played without the network.

    Returns ``outcome2`` [M, K] int8 (2: the position's mover won, 0: lost, 1: cut at the turn
    limit) and ``length`` [M, K] (the playout's network decisions)."""
    import cppanatron as cpp

    from catanrl.envs.encoding import dice_state_of, make_encoder
    from catanrl.search.evaluator import BatchEvaluator
    from catanrl.search.pool import make_pool, run_pools, search_config
    from catanrl.seeding import derive_seed
    from catanrl.train.alphazero import temperature

    train, levels = setup.train, setup.levels
    encoder = make_encoder(setup.plan.dims, levels["level"], dice_state_of(levels))
    space = encoder.action_space
    if evaluator is None:
        evaluator = BatchEvaluator(setup.model, device=setup.device, infer_rows=int(train["search_rows"]),
                                   mode="fast", kernels=train["kernels"])
    pool = None
    if sims > 0:
        params = {"sims": sims, "dets": 0, "oracle": True, "root": "puct", "c_puct": train["c_puct"],
                  "gumbel_m": train["gumbel_m"], "pending": int(train["fast_pending"]), "noise": False}
        pool = make_pool(dict(setup.plan.dims), levels["level"],
                         search_config(params, max_players=int(setup.plan.dims["max_players"]), threads=threads),
                         critic_level=levels["critic_level"], dice_state=dice_state_of(levels))
    limits = [int(s.turn_limit) for s in setup.plan.scenarios]
    if len(set(limits)) != 1:
        raise ValueError("the plan's scenarios have different turn limits")
    limit = limits[0]
    outcome = np.full((len(pos), playouts), -1, dtype=np.int8)
    length = np.zeros((len(pos), playouts), dtype=np.int32)
    rows = np.zeros((batch, int(encoder.size)), dtype=np.float32)

    def begin(job: int, k: int) -> _Playout:
        game = games[job].copy()
        mover = int(game.current_player_index)
        hands = [cpp.HiddenHand(resources=list(p.resources), development_cards=list(p.development_cards),
                                development_card_owned_at_start=list(p.development_card_owned_at_start))
                 for p in game.players]
        game.redeal_hidden_cards(hands, derive_seed(seed, "chance", int(pos[job]), k))
        game.execute(space.decode(game, int(move[job])))
        return _Playout(job, k, game, np.random.default_rng(derive_seed(seed, "sample", int(pos[job]), k)), mover,
                        int(ply[job]) + 1)

    def settle(p: _Playout) -> bool:
        """Play ``p``'s forced moves; True when it is over (its outcome filed)."""
        game = p.game
        while True:
            winner = game.winning_player_index
            if winner is not None or game.num_turns >= limit:
                outcome[p.job, p.k] = 1 if winner is None else 2 * int(int(winner) == p.mover)
                length[p.job, p.k] = p.length
                return True
            actions = game.playable_actions
            if len(actions) != 1:
                return False
            game.execute(actions[0])

    queue = ((job, k) for job in range(len(pos)) for k in range(playouts))
    live: list[_Playout] = []
    finished = 0
    while True:
        while len(live) < batch:
            item = next(queue, None)
            if item is None:
                break
            p = begin(*item)
            if settle(p):
                finished += 1
            else:
                live.append(p)
        if not live:
            break
        if pool is None:
            for n, p in enumerate(live):
                encoder.encode_into(p.game, int(p.game.current_player_index), rows[n])
            obs = rows[:len(live)]
            logits, _ = evaluator(obs)
            u = np.array([p.rng.random() for p in live])
            actions = sample_rows(logits, obs[:, setup.mask] > 0, u)
        else:
            try:
                ids = [pool.add(p.game, seed=int(p.rng.integers(0, 2**62))) for p in live]
                run_pools([pool], evaluator, batch_rows=[len(live) * int(train["fast_pending"])])
                actions = [pool.select(sid, temperature(train, p.ply)) for sid, p in zip(ids, live)]
            finally:
                if pool.pending:
                    pool.cancel()
                for sid in list(pool.ids):
                    pool.remove(sid)
        still = []
        for p, a in zip(live, actions):
            p.game.execute(space.decode(p.game, int(a)))
            p.ply += 1
            p.length += 1
            if settle(p):
                finished += 1
            else:
                still.append(p)
        live = still
        if tick is not None:
            tick(finished)
    return outcome, length


def referee(args) -> dict:
    import torch

    from catanrl.search.evaluator import BatchEvaluator

    torch.set_num_threads(args.threads)
    out = check_out(args.out)
    found, arrays = load_npz(args.search)
    source, data = load_npz(args.positions)
    count = len(arrays["game"])
    if count < len(data["game"]):
        data = limit_positions(data, count)
    if positions_digest(data) != found["positions_digest"]:
        raise SystemExit(f"{args.search} searched other positions than {args.positions}")
    setup = load_setup(found["ckpt"], device=args.device, pool_seed=source["pool_seed"])
    arms = arm_names(arrays) if not args.arms else args.arms.split(",")
    missing = [name for name in arms if f"chosen{SEP}{name}" not in arrays]
    if missing:
        raise SystemExit(f"{args.search} has no arm {missing}")
    pos, move = moves_to_judge(arrays, arms)
    settings = {"ckpt": str(setup.ckpt.ref), "positions_digest": found["positions_digest"], "playouts": args.playouts,
                "playout_sims": args.playout_sims, "seed": args.seed}
    manifest: dict[str, Any] = {"step": "referee", **settings, "search_file": str(args.search), "runs": []}
    held = {"pos": np.zeros(0, np.int32), "move": np.zeros(0, np.int16),
            "outcome2": np.zeros((0, args.playouts), np.int8), "length": np.zeros((0, args.playouts), np.int32)}
    if out.exists():
        manifest, held = load_npz(out)
        differ = {k: (manifest.get(k), v) for k, v in settings.items() if manifest.get(k) != v}
        if differ:
            raise SystemExit(f"{out} holds playouts of something else ({differ}): give another --out")
    have = set(zip(held["pos"].tolist(), held["move"].tolist()))
    new = np.array([(p, m) not in have for p, m in zip(pos.tolist(), move.tolist())], dtype=bool)
    pos, move = pos[new], move[new]
    total = len(pos) * args.playouts
    print(f"{len(np.unique(pos)):,} positions with new moves to judge, {len(pos):,} moves x {args.playouts} "
          f"playouts = {total:,}; {len(have):,} moves already judged", file=sys.stderr, flush=True)
    run = {"args": _args(args), "git": git_sha(), "arms": arms, "moves": len(pos),
           "began": time.strftime("%Y-%m-%d %H:%M:%S")}
    manifest["runs"].append(run)
    evaluator = BatchEvaluator(setup.model, device=setup.device, infer_rows=int(setup.train["search_rows"]),
                               mode="fast", kernels=setup.train["kernels"])
    progress = Progress(total, "playouts", every=args.progress)
    block = max(1, args.save_every)
    began, done = time.time(), 0
    for at in range(0, len(pos), block):
        part = slice(at, min(at + block, len(pos)))
        games = restore_games(setup.plan, setup.levels, data, pos[part])
        base = done
        outcome, length = play_out(
            setup, games, pos[part], move[part], data["ply"][pos[part]], playouts=args.playouts, seed=args.seed,
            sims=args.playout_sims, batch=args.batch, threads=args.env_threads, evaluator=evaluator,
            tick=lambda finished, base=base: progress.update(base + finished))
        done += outcome.size
        held = {"pos": np.concatenate([held["pos"], pos[part]]), "move": np.concatenate([held["move"], move[part]]),
                "outcome2": np.concatenate([held["outcome2"], outcome]),
                "length": np.concatenate([held["length"], length])}
        run.update(wall_s=time.time() - began, playouts=done, decisions=int(run.get("decisions", 0) + length.sum()))
        save_npz(out, manifest, held)  # block by block: a rerun plays only what is missing
    progress.update(total, force=True)
    if not len(pos):
        save_npz(out, manifest, held)
    trunc = float((held["outcome2"] == 1).mean()) if held["outcome2"].size else 0.0
    print(f"{out}: {len(held['pos']):,} moves judged at {len(np.unique(held['pos'])):,} positions, "
          f"{trunc:.2%} of playouts cut at the turn limit, {(time.time() - began) / 60:.1f} min")
    return manifest


# ---------------------------------------------------------------- report

FAMILIES = ("DISCARD", "MONOPOLY", "YEAR_OF_PLENTY", "MARITIME")


def family(name: str) -> str:
    """An action's kind from its name: ``ROAD@12`` is ROAD, ``MARITIME_...`` is MARITIME."""
    if "@" in name:
        return name.split("@", 1)[0]
    return next((prefix for prefix in FAMILIES if name.startswith(prefix)), name)


def rate(hits: int, trials: int) -> dict[str, Any]:
    """A share with its 95% Wilson interval."""
    from catanrl.eval.stats import wilson_interval

    if not trials:
        return {"n": 0, "k": 0, "rate": None, "lo": None, "hi": None}
    lo, hi = wilson_interval(int(hits), int(trials))
    return {"n": int(trials), "k": int(hits), "rate": hits / trials, "lo": float(lo), "hi": float(hi)}


def boot_mean(values: np.ndarray, *, boot: int, seed: int) -> dict[str, Any]:
    """The mean of per-position ``values`` and its 95% bootstrap interval over positions."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": len(values), "mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def at_move(flat: np.ndarray, start: np.ndarray, action: np.ndarray, move: np.ndarray) -> np.ndarray:
    """``flat``'s entry at each position's ``move`` (flat legal lists; NaN if the move is not legal)."""
    out = np.full(len(move), np.nan)
    for i, m in enumerate(move):
        hit = np.flatnonzero(action[start[i]:start[i + 1]] == m)
        if len(hit):
            out[i] = flat[start[i] + hit[0]]
    return out


def win_table(referee_data: Mapping[str, np.ndarray] | None) -> dict[tuple[int, int], float]:
    """(position, move) -> the mover's mean playout outcome (a cut playout is half a win)."""
    if referee_data is None:
        return {}
    wins = referee_data["outcome2"].astype(np.float64).mean(axis=1) / 2.0
    return {(int(p), int(m)): float(w) for p, m, w in zip(referee_data["pos"], referee_data["move"], wins)}


def paired_gain(wins: Mapping[tuple[int, int], float], mine: np.ndarray, theirs: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per position, the referee's win rate of ``mine``'s move minus ``theirs``' (0 where they are
    the same move), and whether the position counts (the same move, or both moves judged)."""
    gain, known = np.zeros(len(mine)), np.ones(len(mine), dtype=bool)
    for i in np.flatnonzero(mine != theirs):
        a, b = wins.get((int(i), int(mine[i]))), wins.get((int(i), int(theirs[i])))
        if a is None or b is None:
            known[i] = False
        else:
            gain[i] = a - b
    return gain, known


def compare(arrays: Mapping[str, np.ndarray], wins: Mapping[tuple[int, int], float], name: str, other: str, *,
            boot: int, seed: int) -> dict[str, Any]:
    """Arm ``name`` against arm ``other``: how often their moves differ (by decision kind too),
    how ``other`` treated ``name``'s move where they differ (its visits there, the network's
    prior on it, ``name``'s own Q gap between the two moves), and the referee's paired
    difference (``name``'s move minus ``other``'s): per position (0 where the moves are the
    same) and over the positions where they differ."""
    start, action = arrays["legal_start"], arrays["legal_action"]
    mine, theirs = arrays[f"chosen{SEP}{name}"].astype(np.int64), arrays[f"chosen{SEP}{other}"].astype(np.int64)
    differ = mine != theirs
    kind = arrays["kind"]
    out: dict[str, Any] = {"arm": name, "against": other, "differ": rate(int(differ.sum()), len(differ)),
                           "by_kind": {KINDS[k]: rate(int(differ[kind == k].sum()), int((kind == k).sum()))
                                       for k in range(len(KINDS)) if (kind == k).any()}}
    if differ.any():
        their_visits = at_move(arrays[f"visits{SEP}{other}"], start, action, mine)[differ]
        prior = at_move(arrays["net_p"], start, action, mine)[differ]
        q = arrays[f"q{SEP}{name}"]
        gap = (at_move(q, start, action, mine) - at_move(q, start, action, theirs))[differ]
        names = arrays.get("action_names")
        out["where_differ"] = {
            "n": int(differ.sum()),
            "other_visits_0": float((their_visits == 0).mean()),
            "other_visits_le2": float((their_visits <= 2).mean()),
            "other_visits_median": float(np.median(their_visits)),
            "net_prior_mean": float(prior.mean()), "net_prior_median": float(np.median(prior)),
            "net_prior_below_1pct": float((prior < 0.01).mean()),
            "own_q_gap_mean": float(np.nanmean(gap)) if np.isfinite(gap).any() else None,
            "own_q_gap_unknown": float(np.isnan(gap).mean()),  # ``name`` never visited ``other``'s move
        }
        if names is not None:
            pairs: dict[str, int] = {}
            for a, b in zip(theirs[differ], mine[differ]):
                key = f"{family(str(names[a]))} -> {family(str(names[b]))}"
                pairs[key] = pairs.get(key, 0) + 1
            out["where_differ"]["moves"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:8])
    if wins:
        gain, known = paired_gain(wins, mine, theirs)
        out["referee"] = {
            "unjudged": int((~known).sum()),
            "per_position": boot_mean(gain[known], boot=boot, seed=seed),
            "where_differ": boot_mean(gain[known & differ], boot=boot, seed=seed + 1),
            "by_kind": {KINDS[k]: boot_mean(gain[known & differ & (kind == k)], boot=boot, seed=seed + 2 + k)
                        for k in range(len(KINDS)) if (known & differ & (kind == k)).any()},
        }
    return out


def report(found: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
           referee_data: Mapping[str, np.ndarray] | None = None, *, boot: int = 2000, seed: int = 0,
           reference: str = REFERENCE, floor: str = FLOOR) -> dict[str, Any]:
    """Everything the tables print, from a search file and (optionally) a referee file."""
    names = arm_names(arrays)
    if reference not in names:
        raise SystemExit(f"the search file has no reference arm {reference}")
    start = arrays["legal_start"]
    count = len(arrays["kind"])
    wins = win_table(referee_data)
    net_entropy = float(entropy(arrays["net_p"], start).mean())
    top = arrays["net_top"].astype(np.int64)
    floor_rate = None
    if floor in names:
        floor_rate = float((arrays[f"chosen{SEP}{floor}"] != arrays[f"chosen{SEP}{reference}"]).mean())
    arms = {}
    for name in names:
        info = dict(found["arms"][name])
        chosen = arrays[f"chosen{SEP}{name}"].astype(np.int64)
        versus = compare(arrays, wins, name, reference, boot=boot, seed=seed)
        prior_entropy = float(entropy(arrays[f"prior{SEP}{name}"], start).mean())
        arms[name] = {
            **{k: info[k] for k in ("visits", "temp", "mix", "noise", "seed", "leaves", "wall_s")},
            "prior_entropy": prior_entropy, "nats_added": prior_entropy - net_entropy,
            "overrule": rate(int((chosen != top).sum()), count),
            "target_entropy": float(entropy(arrays[f"target{SEP}{name}"], start).mean()),
            "root_value": float(arrays[f"value{SEP}{name}"].mean()),
            "mean_depth": float(arrays[f"depth{SEP}{name}"].mean()),
            "tree_mb": float(arrays[f"tree_bytes{SEP}{name}"].mean() / 2**20),
            "vs_reference": versus,
            "differ_minus_floor": None if floor_rate is None else versus["differ"]["rate"] - floor_rate,
        }
    pairs = [(a, b) for a, b in ((floor, reference), ("C", reference), ("B", reference), ("B", "C"))
             if a in names and b in names]
    curve: dict[str, list] = {}
    for name in names:
        arm = arms[name]
        key = f"temp {_number(arm['temp'])}, mix {_number(arm['mix'])}, noise {'on' if arm['noise'] else 'off'}"
        versus = arm["vs_reference"]
        curve.setdefault(key, []).append({
            "arm": name, "visits": arm["visits"], "prior_entropy": arm["prior_entropy"],
            "nats_added": arm["nats_added"], "differ": versus["differ"],
            "differ_minus_floor": arm["differ_minus_floor"],
            "referee_per_position": (versus.get("referee") or {}).get("per_position"),
            "referee_where_differ": (versus.get("referee") or {}).get("where_differ")})
    for points in curve.values():
        points.sort(key=lambda point: point["visits"])
    kinds = {KINDS[k]: int((arrays["kind"] == k).sum()) for k in range(len(KINDS))}
    return {"ckpt": found["ckpt"], "positions": count, "reference": reference, "floor": floor,
            "floor_rate": floor_rate, "net_entropy": net_entropy, "kinds": kinds,
            "full_share": float(arrays["full"].mean()), "judged_moves": len(wins),
            "playouts": None if referee_data is None else int(referee_data["outcome2"].shape[1]),
            "truncated_playouts": None if referee_data is None or not referee_data["outcome2"].size
            else float((referee_data["outcome2"] == 1).mean()),
            "arms": arms,
            "pairs": [compare(arrays, wins, a, b, boot=boot, seed=seed + 100 * (n + 1))
                      for n, (a, b) in enumerate(pairs)],
            "curve": curve, "boot": boot}


def _pct(r: Mapping[str, Any] | None, key: str = "rate") -> str:
    if not r or r.get(key) is None:
        return "-"
    return f"{100 * r[key]:.2f}% [{100 * r['lo']:.2f}, {100 * r['hi']:.2f}]"


def _pp(r: Mapping[str, Any] | None) -> str:
    if not r or r.get("mean") is None:
        return "-"
    return f"{100 * r['mean']:+.2f} pp [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] (n {r['n']:,})"


def print_report(out: Mapping[str, Any], *, detail: bool = False) -> None:
    """The tables. The blocks of detail are for the default arms' pairs; ``detail`` adds one for
    every other arm against the reference (a grid's arms are in the first table and the curve)."""
    print(f"{out['ckpt']}: {out['positions']:,} positions ({out['full_share']:.0%} searched at the full budget "
          f"in self-play); kinds {out['kinds']}")
    print(f"network policy entropy {out['net_entropy']:.3f} nats; reference arm {out['reference']}; noise floor "
          f"({out['floor']} vs {out['reference']}): "
          + ("-" if out["floor_rate"] is None else f"{100 * out['floor_rate']:.2f}%"))
    if out["playouts"] is not None:
        print(f"referee: {out['judged_moves']:,} moves judged, {out['playouts']} playouts each, "
              f"{100 * (out['truncated_playouts'] or 0.0):.2f}% cut at the turn limit (half a win)")
    print("\narms (overrule: the arm's move is not the network's top move; differs: from the reference's move)")
    print(f"{'arm':<16}{'visits':>7}{'temp':>6}{'mix':>6}{'noise':>6}{'prior H':>9}{'+nats':>7}  "
          f"{'overrule':<24}{'differs from ' + out['reference']:<24}{'- floor':>9}{'depth':>7}")
    for name, arm in out["arms"].items():
        minus = "-" if arm["differ_minus_floor"] is None else f"{100 * arm['differ_minus_floor']:+.2f}"
        print(f"{name:<16}{arm['visits']:>7}{arm['temp']:>6g}{arm['mix']:>6g}{'on' if arm['noise'] else 'off':>6}"
              f"{arm['prior_entropy']:>9.3f}{arm['nats_added']:>+7.3f}  {_pct(arm['overrule']):<24}"
              f"{_pct(arm['vs_reference']['differ']):<24}{minus:>9}{arm['mean_depth']:>7.1f}")
    comparisons = list(out["pairs"])
    if detail:
        comparisons += [arm["vs_reference"] for name, arm in out["arms"].items() if name != out["reference"]
                        and not any(p["arm"] == name and p["against"] == out["reference"] for p in out["pairs"])]
    for c in comparisons:
        print(f"\n{c['arm']} against {c['against']}: moves differ at {_pct(c['differ'])}")
        print("  by kind: " + ", ".join(f"{k} {_pct(v)} of {v['n']:,}" for k, v in c["by_kind"].items()))
        w = c.get("where_differ")
        if w:
            gap = "-" if w["own_q_gap_mean"] is None else f"{w['own_q_gap_mean']:+.3f}"
            print(f"  where they differ (n {w['n']:,}): {c['against']} gave {c['arm']}'s move 0 visits at "
                  f"{w['other_visits_0']:.1%}, <= 2 at {w['other_visits_le2']:.1%} (median "
                  f"{w['other_visits_median']:.0f}); the network's prior on it: mean {w['net_prior_mean']:.3f}, "
                  f"median {w['net_prior_median']:.3f}, below 1% at {w['net_prior_below_1pct']:.1%}; "
                  f"{c['arm']}'s own Q gap {gap} ({w['own_q_gap_unknown']:.1%} unknown)")
            if "moves" in w:
                print("  moves (" + c["against"] + " -> " + c["arm"] + "): "
                      + ", ".join(f"{k} x{v}" for k, v in w["moves"].items()))
        r = c.get("referee")
        if r:
            print(f"  referee, {c['arm']}'s move minus {c['against']}'s: per position {_pp(r['per_position'])}; "
                  f"where they differ {_pp(r['where_differ'])}" + (f"; {r['unjudged']} unjudged" if r["unjudged"] else ""))
            if r["by_kind"]:
                print("    by kind: " + "; ".join(f"{k} {_pp(v)}" for k, v in r["by_kind"].items()))
    print("\ncurve: against visits, for every prior (differs: from the reference's move; referee: the arm's move "
          "minus the reference's, per position)")
    for key, points in out["curve"].items():
        print(f"  {key} (prior entropy {points[0]['prior_entropy']:.3f} nats, {points[0]['nats_added']:+.3f})")
        for point in points:
            minus = "-" if point["differ_minus_floor"] is None else f"{100 * point['differ_minus_floor']:+.2f} pp"
            print(f"    {point['visits']:>6} visits ({point['arm']}): differs {_pct(point['differ'])}, minus floor "
                  f"{minus}; referee {_pp(point['referee_per_position'])}; where differ "
                  f"{_pp(point['referee_where_differ'])}")


def report_step(args) -> dict:
    found, arrays = load_npz(args.search)
    referee_data = None
    if args.referee:
        judged, referee_data = load_npz(args.referee)
        if judged["positions_digest"] != found["positions_digest"]:
            raise SystemExit(f"{args.referee} judged other positions than {args.search} searched")
    out = report(found, arrays, referee_data, boot=args.boot, seed=args.seed)
    out["manifest"] = {"search": {k: v for k, v in found.items() if k != "train"}, "git": git_sha(),
                       "args": _args(args),
                       "referee": None if not args.referee else {k: judged[k] for k in
                                                                 ("playouts", "playout_sims", "seed", "runs")}}
    print_report(out, detail=args.detail)
    if args.out:
        path = check_out(args.out)
        path.write_text(json.dumps(out, indent=1))
        print(f"\n{path}")
    return out


# ---------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)

    def common(p, *, device: bool = True) -> None:
        p.add_argument("--seed", type=int, default=20261116)
        p.add_argument("--progress", type=float, default=10.0, help="seconds between progress lines")
        if device:
            p.add_argument("--device", default="cuda")
            p.add_argument("--threads", type=int, default=2, help="torch CPU threads")
            p.add_argument("--env-threads", type=int, default=12, help="engine threads (the run's runtime.threads)")

    p = sub.add_parser("positions", help="self-play as training does it; keep a few decisions per game")
    p.add_argument("--ckpt", required=True, help="exp/var@which[#sha] or a .pt path (an alphazero run's)")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--envs", type=int, default=250, help="games in flight; --games / --envs games per env")
    p.add_argument("--per-game", type=int, default=3, help="decisions kept per game")
    p.add_argument("--pool-seed", type=int, default=None, help="the held-out map pool's seed")
    p.add_argument("--sims", type=int, default=None, help="smokes: the full search's visits")
    p.add_argument("--fast-sims", type=int, default=None, help="smokes: the fast search's visits")
    common(p)
    p.set_defaults(run=positions)

    p = sub.add_parser("search", help="search every position with a list of arms")
    p.add_argument("--positions", required=True)
    p.add_argument("--out", required=True, help="finished arms in it are kept; new ones are added")
    p.add_argument("--no-defaults", action="store_true", help="leave out A1, A2, C and B")
    p.add_argument("--wide-visits", type=int, default=9600, help="C's and B's visits")
    p.add_argument("--wide-mix", type=float, default=0.25, help="B: the uniform share of every node's prior")
    p.add_argument("--wide-temp", type=float, default=1.0, help="B: the temperature on every node's logits")
    p.add_argument("--arm", action="append", help="name:visits=N[,temp=T,mix=E,noise=0|1,seed=S] (repeatable)")
    p.add_argument("--grid-visits", default=None, help="e.g. 600,2400,9600,16384")
    p.add_argument("--grid-temps", default=None, help="e.g. 1,1.5,2,4 (grid arms have no root noise)")
    p.add_argument("--grid-mix", type=float, default=0.0)
    p.add_argument("--grid-seed", type=int, default=100)
    p.add_argument("--sims", type=int, default=None, help="smokes: A1's and A2's visits")
    p.add_argument("--limit", type=int, default=0, help="smokes: the first N positions only")
    p.add_argument("--tree-gb", type=float, default=6.0, help="RAM for the trees in flight")
    p.add_argument("--max-inflight", type=int, default=512, help="positions searched at once, at most")
    common(p)
    p.set_defaults(run=search)

    p = sub.add_parser("referee", help="play out every distinct chosen move where the arms disagree")
    p.add_argument("--positions", required=True)
    p.add_argument("--search", required=True)
    p.add_argument("--out", required=True, help="judged moves in it are kept; new ones are added")
    p.add_argument("--arms", default=None, help="comma-separated arms whose moves are judged (default: all)")
    p.add_argument("--playouts", type=int, default=32)
    p.add_argument("--playout-sims", type=int, default=0,
                   help="0: the raw policy, sampled; N: the trainer's fast search of N visits per move")
    p.add_argument("--batch", type=int, default=4096, help="playouts in flight")
    p.add_argument("--save-every", type=int, default=512, help="moves per saved block")
    common(p)
    p.set_defaults(run=referee)

    p = sub.add_parser("report", help="tables and JSON")
    p.add_argument("--search", required=True)
    p.add_argument("--referee", default=None)
    p.add_argument("--out", default=None, help="the JSON")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--detail", action="store_true", help="a block of detail for every arm, not only the default pairs")
    common(p, device=False)
    p.set_defaults(run=report_step)

    args = parser.parse_args(argv)
    run = args.run
    del args.run
    run(args)


if __name__ == "__main__":
    main()
