"""A frozen-policy critic refit scored on one fixed set of held-out games.

``play`` plays ``--games`` games of the policy against itself as PPO's rollouts
play them (every decision a temperature-1 sample of the masked logits, the
run's own env: rules, turn limit, seats) on boards of a map pool drawn from
another seed than the run's training pool (refused if any board is shared),
and on every decision records each listed checkpoint's value logits for both
slots (slot 0 the acting seat, slot 1 the other), the turn and, once the game
ends, both slots' undiscounted outcomes (1 for that seat's win; 0 for a loss
or a truncation). Each env keeps its first ``--games / --envs`` games (an
unbiased sample: long games are not cut off); envs past their quota finish on
uniform legal moves, unrecorded. Every listed checkpoint must compute the
policy's logits on the first batch of rows (a frozen actor), so the games are
every checkpoint's own. Nothing trains and nothing is written under the runs
directory.

``report`` reads that file: per checkpoint the slot-0 and the two-slot log
loss and Brier score against the outcome, with game-clustered bootstrap 95%
CIs, the same paired with the first checkpoint and with the one ``--back``
epochs earlier (the convergence check), and the slot-0 AUC within turn bins
(discrimination, which no calibration changes); and two calibration-only baselines of the
first checkpoint on the same decisions: each seat's value divided by the
two seats' sum, and per turn bin a logistic fit of the outcome on both
slots' logits, cross-fitted over two halves of the games. A refit's gain
beyond calibration is its paired loss against the better baseline.

    uv run python tools/critic_heldout.py play --run critic-refit-2p/main --every 25 \\
        --games 2048 --device cuda --out heldout-main.npz
    uv run python tools/critic_heldout.py report heldout-main.npz --json heldout-main.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-7
#: The metrics per decision, in the order of :func:`losses`' columns.
METRICS = ("bce0", "bce", "brier0", "brier")


# ---------------------------------------------------------------- play


def heldout_plan(env: Mapping[str, Any], *, seed: int, pool_seed: int | None):
    """The run's env on a map pool from ``pool_seed`` (default: derived from the run's seed),
    and the training pool's hash. Refuses a board the training pool also holds."""
    from catanrl.envs.puffer import plan_env
    from catanrl.seeding import derive_seed

    env = dict(env)
    train_seed = (env.get("map_pool") or {}).get("seed")
    train_seed = derive_seed(seed, "map_pool") if train_seed is None else int(train_seed)
    pool_seed = derive_seed(seed, "critic_heldout") if pool_seed is None else int(pool_seed)
    if pool_seed == train_seed:
        raise ValueError("the held-out pool seed is the training pool's")
    train = plan_env(env, pool_seed=train_seed)
    plan = plan_env(env, pool_seed=pool_seed)
    seen = {h for s in train.pool["scenarios"] for _, h in s["maps"]}
    shared = sum(h in seen for s in plan.pool["scenarios"] for _, h in s["maps"])
    if shared:
        raise ValueError(f"{shared} held-out boards are in the training pool (seed {train_seed})")
    return plan, {"train_pool": train.pool["hash"], "train_pool_seed": train_seed, "pool": plan.pool["hash"],
                  "pool_seed": pool_seed}


def play_games(policy, critics: Sequence[Any], plan, *, games: int, envs: int, seed: int, threads: int = 1,
               device: str = "cpu", log=print) -> dict[str, np.ndarray]:
    """``games`` self-play games of ``policy`` on ``plan`` with every critic's value logits per
    decision. Arrays: per decision ``game``, ``t``, ``turn``, ``seat``, ``z`` [N, K, 2] (critic,
    slot) and ``y`` [N, 2]; per game ``g_id``, ``g_T``, ``g_winner`` (-1 none) and ``g_trunc``."""
    import cppanatron as cpp
    import torch

    from catanrl.envs.encoding import encoder_dims, info_level

    if games % envs:
        raise ValueError(f"--games {games} is not a multiple of --envs {envs}")
    per_env = games // envs
    level = policy.levels["level"]
    for model in (policy, *critics):
        if model.levels["level"] != level or getattr(model, "has_critic_block", False) \
                or getattr(model, "recurrent", False):
            raise ValueError("every network must read the policy's rows (one level, no critic block, not recurrent)")
    players = int(plan.dims["max_players"])
    vec = cpp.VecEnv(list(plan.scenarios), num_envs=envs, dims=encoder_dims(plan.dims), level=info_level(level),
                     seed=seed, num_threads=threads, vp_reward=0.0, trackers=False)
    b = vec.allocate()
    vec.reset(seed)
    offset, (size,) = policy.layout["action_mask"]
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    done = np.zeros(envs, np.int64)
    ply = np.zeros(envs, np.int64)
    current: list[list[tuple]] = [[] for _ in range(envs)]
    current_z: list[list[np.ndarray]] = [[] for _ in range(envs)]
    rows_out, z_out, games_out = [], [], []
    checked = False
    start = last = time.time()
    while (done < per_env).any():
        acting = np.flatnonzero(b["acting"])
        envs_of = b["env_index"][acting].astype(np.int64)
        keep = done[envs_of] < per_env
        rows = acting[keep]
        if len(rows):
            obs = torch.from_numpy(np.ascontiguousarray(b["observations"][rows], dtype=np.float32)).to(device)
            with torch.inference_mode():
                logits = policy.policy_logits(obs).float()
                if not checked:  # a frozen actor: every checkpoint plays these games
                    for i, model in enumerate(critics):
                        if not torch.equal(model.policy_logits(obs).float(), logits):
                            raise ValueError(f"checkpoint {i}'s policy differs from the policy's: not a frozen actor")
                    checked = True
                z = torch.stack([model.value_logits(obs)[0].float()[:, :2] for model in critics], 1).cpu().numpy()
                probs = torch.softmax(logits.double(), dim=-1).cpu()
            b["actions"][rows] = torch.multinomial(probs, 1, generator=gen).squeeze(1).numpy()
            for i, row in enumerate(rows):
                env = int(b["env_index"][row])
                current[env].append((int(ply[env]), int(vec.game(env).num_turns), int(b["seat"][row])))
                current_z[env].append(z[i])
                ply[env] += 1
        for row in acting[~keep]:  # past the quota: any legal move, unrecorded
            legal = np.flatnonzero(b["observations"][row, offset:offset + size] > 0)
            b["actions"][row] = int(rng.choice(legal))
        vec.step()
        for env in np.flatnonzero(b["terminals"].reshape(-1, players)[:, 0]):
            if done[env] < per_env:
                rewards = b["rewards"][env * players:(env + 1) * players]
                winner = int(np.argmax(rewards)) if rewards.max() > 0 else -1
                gid = int(env * per_env + done[env])
                games_out.append((gid, int(ply[env]), winner, int(bool(b["truncations"][env * players]))))
                for (t, turn, seat), zz in zip(current[env], current_z[env]):
                    rows_out.append((gid, t, turn, seat, int(winner == seat), int(winner == 1 - seat)))
                    z_out.append(zz)
                done[env] += 1
            current[env], current_z[env], ply[env] = [], [], 0
        if time.time() - last > 60:
            last = time.time()
            log(f"{(last - start) / 60:.1f} min: games {int(np.minimum(done, per_env).sum())}/{games}, "
                f"decisions {len(rows_out)}")
    r = np.array(rows_out, dtype=np.int64)
    g = np.array(games_out, dtype=np.int64)
    return {"game": r[:, 0], "t": r[:, 1].astype(np.int32), "turn": r[:, 2].astype(np.int32),
            "seat": r[:, 3].astype(np.int8), "y": r[:, 4:6].astype(np.int8), "z": np.stack(z_out).astype(np.float32),
            "g_id": g[:, 0], "g_T": g[:, 1], "g_winner": g[:, 2], "g_trunc": g[:, 3]}


def checkpoint_refs(args) -> list[str]:
    refs = list(args.ckpts or [])
    if args.run:
        from catanrl.runs import load_experiment

        refs += [f"{args.run}@e{c.epoch}" for c in load_experiment(args.run).checkpoints() if c.epoch % args.every == 0]
    if not refs:
        raise SystemExit("name checkpoints (--ckpts) or a run (--run)")
    return refs


def play(args) -> dict:
    import torch

    from catanrl.runs import load_checkpoint

    torch.set_num_threads(args.threads)
    refs = checkpoint_refs(args)
    ckpts = [load_checkpoint(ref) for ref in refs]
    policy_ckpt = load_checkpoint(args.policy) if args.policy else ckpts[0]
    config = policy_ckpt.header.get("config") or {}
    if "env" not in config:
        raise SystemExit(f"{policy_ckpt.ref} carries no run config: its env is unknown")
    plan, pools = heldout_plan(config["env"], seed=int(config["seed"]), pool_seed=args.pool_seed)
    policy = policy_ckpt.model("all", device=args.device)
    critics = [c.model("all", device=args.device) for c in ckpts]
    seed = args.seed
    start = time.time()
    data = play_games(policy, critics, plan, games=args.games, envs=args.envs, seed=seed, threads=args.env_threads,
                      device=args.device, log=lambda m: print(m, file=sys.stderr, flush=True))
    manifest = {"policy": str(policy_ckpt.ref), "checkpoints": [str(c.ref) for c in ckpts],
                "epochs": [int(c.ref.epoch) for c in ckpts], "seed": seed, "envs": args.envs, **pools,
                "games": int(len(data["g_id"])), "decisions": int(len(data["t"])),
                "truncated": int(data["g_trunc"].sum()), "wall_s": time.time() - start}
    np.savez(args.out, manifest=np.array(json.dumps(manifest)), **data)
    return manifest


# ---------------------------------------------------------------- report


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def losses(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[N, 4]: slot-0 log loss, the two slots' mean log loss, slot-0 Brier, the two slots' mean Brier."""
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    bce = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    brier = (p - y) ** 2
    return np.stack((bce[:, 0], bce.mean(1), brier[:, 0], brier.mean(1)), 1)


def cluster_ci(game: np.ndarray, values: np.ndarray, *, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Game-clustered bootstrap 95% CI of the mean of each column of ``values`` [N, M]."""
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(game, return_inverse=True)
    groups = int(inverse.max()) + 1
    sums = np.zeros((groups, values.shape[1]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=groups).astype(np.float64)
    out = np.empty((n, values.shape[1]))
    for i in range(n):
        w = np.bincount(rng.integers(0, groups, groups), minlength=groups).astype(np.float64)
        out[i] = (w @ sums) / (w @ counts)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC, ties at half."""
    order = np.argsort(score, kind="mergesort")
    _, first, counts = np.unique(score[order], return_index=True, return_counts=True)
    ranks = np.empty(len(score))
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    pos = int(y.sum())
    neg = len(y) - pos
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def turn_bins(turn: np.ndarray, bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(turn, np.linspace(0, 1, bins + 1))[1:-1])
    return np.digitize(turn, edges)


def auc_within(score: np.ndarray, y: np.ndarray, bins: np.ndarray) -> float:
    """The decision-weighted mean of the AUC within each bin (bins with both outcomes)."""
    values, weights = [], []
    for b in np.unique(bins):
        m = bins == b
        if 0 < y[m].sum() < m.sum():
            values.append(auc(score[m], y[m]))
            weights.append(m.sum())
    return float(np.average(values, weights=weights))


def logistic(x: np.ndarray, y: np.ndarray, *, ridge: float = 1.0, iterations: int = 100) -> np.ndarray:
    """Logistic regression weights by Newton's method (``x`` carries its own bias column), with an
    L2 penalty ``ridge`` on the summed log loss: nothing next to a turn bin's ~16k decisions, but it
    keeps a separable fit on a few games finite."""
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = sigmoid(x @ w)
        grad = x.T @ (p - y) + ridge * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + ridge * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def platt_by_turn(z: np.ndarray, y: np.ndarray, game: np.ndarray, bins: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Per turn bin and slot, sigmoid(a z_slot + c z_other + b), fitted on one half of the games
    and applied to the other (both ways)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(game)
    half = np.isin(game, rng.permutation(uniq)[: len(uniq) // 2])
    p = np.empty(z.shape, dtype=np.float64)
    for b in np.unique(bins):
        for fit, apply in ((half, ~half), (~half, half)):
            f, a = fit & (bins == b), apply & (bins == b)
            if not a.any():
                continue
            for slot in (0, 1):
                features = lambda m: np.stack((z[m, slot], z[m, 1 - slot], np.ones(m.sum())), 1)  # noqa: E731
                p[a, slot] = sigmoid(features(a) @ logistic(features(f), y[f, slot].astype(np.float64)))
    return p


def report(data: Mapping[str, np.ndarray], *, boot: int = 2000, back: int = 50) -> dict:
    manifest = json.loads(str(data["manifest"]))
    z, y, game = data["z"].astype(np.float64), data["y"].astype(np.float64), data["game"]
    bins = turn_bins(data["turn"])
    k = z.shape[1]
    per = [losses(sigmoid(z[:, i]), y) for i in range(k)]
    v0 = sigmoid(z[:, 0])
    baselines = {"normalized": v0 / v0.sum(1, keepdims=True), "platt_turn": platt_by_turn(z[:, 0], y, game, bins)}
    base = {name: losses(p, y) for name, p in baselines.items()}
    best = min(base, key=lambda name: base[name][:, 0].mean())
    m = len(METRICS)
    epochs = list(manifest["epochs"])
    # per checkpoint, the latest one at least ``back`` epochs earlier (itself when there is none)
    earlier = [max((j for j in range(k) if epochs[j] <= epochs[i] - back), key=lambda j: epochs[j], default=i)
               for i in range(k)]
    # every interval in one bootstrap: the levels, the paired deltas against the first, the one ``back``
    # epochs earlier and the better baseline
    columns = [*per, *base.values(), *(per[i] - per[0] for i in range(k)),
               *(per[i] - per[earlier[i]] for i in range(k)), *(per[i] - base[best] for i in range(k))]
    values = np.concatenate(columns, 1)
    lo, hi = cluster_ci(game, values, n=boot)
    mean = values.mean(0)

    def block(j: int) -> dict:
        s = slice(j * m, (j + 1) * m)
        return {name: [float(a), float(b), float(c)] for name, a, b, c in zip(METRICS, mean[s], lo[s], hi[s])}

    out = {"manifest": manifest, "turn_bins": int(bins.max()) + 1, "best_baseline": best, "back": back,
           "checkpoints": [], "baselines": {}}
    for j, name in enumerate(base):
        out["baselines"][name] = {**block(k + j), "auc0_within_turn": auc_within(baselines[name][:, 0], y[:, 0], bins)}
    first = k + len(base)
    for i in range(k):
        out["checkpoints"].append({
            "ref": manifest["checkpoints"][i], "epoch": epochs[i], **block(i),
            "auc0_within_turn": auc_within(z[:, i, 0], y[:, 0], bins), "vs_first": block(first + i),
            "back_epoch": epochs[earlier[i]], "vs_back": block(first + k + i),
            "vs_best_baseline": block(first + 2 * k + i)})
    return out


def print_report(out: Mapping[str, Any]) -> None:
    man = out["manifest"]
    print(f"{man['games']} games, {man['decisions']} decisions ({man['truncated']} truncated), boards from pool "
          f"{man['pool']} (training pool {man['train_pool']}), policy {man['policy']}")
    print("[mean, 95% CI], game-clustered; deltas paired per decision")
    for name, b in out["baselines"].items():
        print(f"  baseline {name:<11} bce0 {b['bce0'][0]:.4f} [{b['bce0'][1]:.4f}, {b['bce0'][2]:.4f}]  "
              f"auc0 within turn {b['auc0_within_turn']:.4f}")
    for c in out["checkpoints"]:
        f, p, b = c["vs_first"]["bce0"], c["vs_back"]["bce0"], c["vs_best_baseline"]["bce0"]
        print(f"  e{c['epoch']:<5} bce0 {c['bce0'][0]:.4f} [{c['bce0'][1]:.4f}, {c['bce0'][2]:.4f}]  bce {c['bce'][0]:.4f}  "
              f"brier0 {c['brier0'][0]:.4f}  auc0|turn {c['auc0_within_turn']:.4f}  "
              f"vs first {f[0]:+.4f} [{f[1]:+.4f}, {f[2]:+.4f}]  vs e{c['back_epoch']} {p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}]  "
              f"vs {out['best_baseline']} {b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play", help="play the held-out games and record every critic's values")
    p.add_argument("--ckpts", nargs="*", help="checkpoint refs (run@eN or .pt paths) whose critics are scored")
    p.add_argument("--run", help="also every committed checkpoint of this run at a multiple of --every")
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--policy", help="the policy's checkpoint (default: the first one scored)")
    p.add_argument("--games", type=int, default=2048)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--pool-seed", type=int, default=None, help="default: derive_seed(run seed, 'critic_heldout')")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4, help="torch threads")
    p.add_argument("--env-threads", type=int, default=1)
    p.add_argument("--out", required=True)
    r = sub.add_parser("report", help="score the recorded values")
    r.add_argument("npz")
    r.add_argument("--boot", type=int, default=2000)
    r.add_argument("--back", type=int, default=50, help="epochs back for the convergence delta")
    r.add_argument("--json")
    args = parser.parse_args(argv)
    if args.command == "play":
        print(json.dumps(play(args), indent=2))
        return
    out = report(dict(np.load(args.npz)), boot=args.boot, back=args.back)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    print_report(out)


if __name__ == "__main__":
    main()
