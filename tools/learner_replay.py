"""Replay an AlphaZero run's first learning iterations offline, under variants of its recipe.

Written for az-2p-graph-m/main, whose raw policy fell to 25% against its own start as soon as the
learner ran (report, exp:az-graph-m). Three steps, each a subcommand; ``--out`` is a scratch
directory (never the runs root):

* ``prep``: cut the run's trainer state down to the replay as it was when the learner first ran
  (the start network's own self-play) and the drift probe -> ``<out>/base.pt``.
* ``variants``: from the run's e0, ``--calls`` learning iterations (:func:`catanrl.train.alphazero.learn`,
  the run's own seeds) on that replay under each variant of the train keys. After every call:
  agreement with the start's top move and KL on the run's drift probe (``drift/agree_init``), and
  on 8,192 fixed uniform rows the policy loss, agreement with the target's top move and the net's
  entropy. Writes ``<out>/variants-c<calls>.json`` and each network's ``state-<variant>-c<calls>.pt``.
* ``ckpt``: write those networks as checkpoints of scratch runs under ``<out>/runs`` (which links
  the start's run), so the eval harness can play them::

      CATANRL_RUNS_DIR=<out>/runs catanrl eval --suite heldout-2p-fr --boards 2500 --device cuda \\
        "POLICY:ckpt=gmdiag/no-surprise@e3" "POLICY:ckpt=graph-m-2p-lr/main@e22711#673425d606b5"

The static replay stands in for the run only over its first few iterations (the run's buffer
grows with the changed network's games).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VARIANTS = {
    "run": {},
    "no-surprise": {"surprise_weight": 0.0},
    "no-soft": {"soft_policy_coef": 0.0},
    "no-outcome": {"outcome_vp_coef": 0.0, "outcome_turns_coef": 0.0},
    "no-surprise-no-soft": {"surprise_weight": 0.0, "soft_policy_coef": 0.0},
    "kata": {"surprise_weight": 0.0, "soft_policy_coef": 0.0, "outcome_vp_coef": 0.0, "outcome_turns_coef": 0.0,
             "value_search_blend": 0.0},
    "surprise-only": {"soft_policy_coef": 0.0, "outcome_vp_coef": 0.0, "outcome_turns_coef": 0.0,
                      "value_search_blend": 0.0},
    "lr1e-5": {"learning_rate": 1.0e-5},
    "no-surprise-lr1e-5": {"surprise_weight": 0.0, "learning_rate": 1.0e-5},
    "surprise0.1": {"surprise_weight": 0.1},
}


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _model(cfg):
    from catanrl.envs.puffer import plan_env
    from catanrl.seeding import derive_seed
    from catanrl.train.puffer import build_policy_model

    plan = plan_env(cfg["env"], pool_seed=derive_seed(int(cfg["seed"]), "map_pool"))
    return build_policy_model(cfg, plan.dims)


def prep(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    first = next(r for r in rows if "train/policy_loss" in r)
    n0 = int(first["positions"])
    print(f"the learner first ran at iteration {first['epoch']} with {n0} positions")
    state = _load(args.run / "trainer_state.pt")
    cut = {k: (v[:n0].clone() if hasattr(v, "shape") else v) for k, v in state["replay"].items()}
    cut["size"] = cut["added"] = n0
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"replay": cut, "monitors": state["monitors"], "first": first["epoch"]}, args.out / "base.pt")


def variants(args: argparse.Namespace) -> None:
    from catanrl.seeding import derive_seed, seed_everything
    from catanrl.train import alphazero as az

    cfg = json.loads((args.run / "config.json").read_text())["config"]
    base = _load(args.out / "base.pt")
    e0 = _load(args.run / "checkpoints" / "e000000.pt")["state_dict"]
    seed = int(cfg["seed"])
    state = base["replay"]
    n0 = int(state["size"])
    replay = az.ReplayBuffer(n0, state["obs"].shape[1], state["policy"].shape[1], state["value"].shape[1],
                             outcome="final_vp" in state, search="surprise" in state)
    replay.restore(state)
    probe_obs = base["monitors"]["probe_obs"].cuda()
    init_logp = base["monitors"]["probe_init_logp"].cuda()
    has = np.flatnonzero(replay.has_policy[:n0])
    fixed = np.random.default_rng(0).choice(has, min(8192, len(has)), replace=False)
    if replay.search:
        sur = replay.surprise[:n0][has]
        p = az.surprise_probabilities(replay.surprise[:n0], 0.5, 10.0)
        print(f"rows {n0}, with a policy target {len(has)}; surprise mean {sur.mean():.3f}, median "
              f"{np.median(sur):.4f}, p90 {np.quantile(sur, .9):.3f}, p99 {np.quantile(sur, .99):.2f}, max "
              f"{sur.max():.1f}; at the cap (10x the mean) {(sur >= 10 * replay.surprise[:n0].mean()).mean():.3%}; "
              f"the top 1% of rows by sampling weight (0.5, cap 10) hold {np.sort(p)[-n0 // 100:].sum():.1%} of "
              f"the draws", flush=True)

    @torch.no_grad()
    def read(model) -> dict[str, float]:
        model.eval()
        logp = torch.cat([F.log_softmax(model.heads(probe_obs[i:i + 512].float())[0].float(), 1)
                          for i in range(0, len(probe_obs), 512)])
        kl = (init_logp.exp() * torch.where(init_logp > -1e8, init_logp - logp, torch.zeros_like(init_logp))).sum(1)
        out = {"agree_init": float((init_logp.argmax(1) == logp.argmax(1)).float().mean()), "kl_init": float(kl.mean())}
        ce, top, entropy = [], [], []
        for i in range(0, len(fixed), 512):
            b = replay.batch(fixed[i:i + 512], "cuda")
            lp = F.log_softmax(model.heads(b["obs"])[0].float(), 1)
            t = b["policy"] / b["policy"].sum(1, keepdim=True).clamp_min(1e-8)
            ce.append(-(t * lp).sum(1))
            top.append((t.argmax(1) == lp.argmax(1)).float())
            entropy.append(-(lp.exp() * torch.where(lp > -1e8, lp, torch.zeros_like(lp))).sum(1))
        out.update({"uniform_policy_loss": float(torch.cat(ce).mean()), "agree_target": float(torch.cat(top).mean()),
                    "net_entropy": float(torch.cat(entropy).mean())})
        return out

    results = {}
    for name in args.variants or list(VARIANTS):
        train = {**cfg["train"], **VARIANTS[name]}
        seed_everything(derive_seed(seed, "init"))
        model = _model(cfg)
        model.load_state_dict(e0)
        model.cuda().eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]),
                                      weight_decay=float(train["weight_decay"]))
        rows = [{"call": 0, **read(model)}]
        start = time.time()
        for k in range(args.calls):
            iteration = int(base["first"]) + k
            learned = az.learn(model, optimizer, replay, train, "cuda",
                               np.random.default_rng(derive_seed(seed, "replay", iteration)), iteration)
            rows.append({"call": k + 1, **read(model),
                         **{f"train_{key}": learned[key] for key in ("policy_loss", "soft_policy_loss", "value_loss",
                                                                     "kl_target_net", "surprise_ess") if key in learned}})
        results[name] = rows
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   args.out / f"state-{name}-c{args.calls}.pt")
        print(f"{name} ({time.time() - start:.0f} s)", flush=True)
        for row in rows:
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    (args.out / f"variants-c{args.calls}.json").write_text(json.dumps(results, indent=1))


def ckpt(args: argparse.Namespace) -> None:
    from catanrl import checkpoint

    manifest = json.loads((args.run / "config.json").read_text())
    cfg = manifest["config"]
    code = _load(args.run / "checkpoints" / "e000000.pt")["code"]
    root = args.out / "runs"
    root.mkdir(exist_ok=True)
    start = str(cfg["init_from"]["ckpt"]).split("/")[0]  # the start's run, read-only
    if not (root / start).exists():
        os.symlink(args.run.parent.parent / start, root / start)
    for name in args.variants:
        model = _model(cfg)
        model.load_state_dict(torch.load(args.out / f"state-{name}-c{args.calls}.pt"))
        model.eval()
        path = root / "gmdiag" / name
        file = f"e{args.calls:06d}.pt"
        ref = checkpoint.save(path / "checkpoints" / file, model, run=f"gmdiag/{name}", epoch=args.calls,
                              agent_steps=0, config=cfg, code=code)
        (path / "config.json").write_text(json.dumps(manifest))
        (path / "run.json").write_text(json.dumps({
            "checkpoints": [{"agent_steps": 0, "epoch": args.calls, "file": file, "launch": 0,
                             "weights_sha256": ref.weights_sha256}],
            "launches": [], "status": "finished"}))
        print(f"gmdiag/{name}@e{args.calls}", ref.weights_sha256[:12])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("prep", "variants", "ckpt"))
    parser.add_argument("--run", type=Path, required=True, help="the run's directory (runs/<experiment>/<variant>)")
    parser.add_argument("--out", type=Path, required=True, help="a scratch directory")
    parser.add_argument("--calls", type=int, default=3, help="learning iterations per variant")
    parser.add_argument("--variants", nargs="*", default=None, help=f"of: {', '.join(VARIANTS)} (default: all)")
    args = parser.parse_args()
    args.run, args.out = args.run.resolve(), args.out.resolve()
    {"prep": prep, "variants": variants, "ckpt": ckpt}[args.step](args)


if __name__ == "__main__":
    main()
