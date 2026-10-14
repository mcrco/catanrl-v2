"""Tiny CPU training configs for the training tests."""

from __future__ import annotations

import copy


def tiny_ppo(seed: int = 3, epochs: int = 4, **train) -> dict:
    num_envs, players, horizon = 8, 2, 8
    rows = num_envs * players
    config = {
        "seed": seed,
        "env": {
            "scenarios": [
                {"game": "base-2p", "maps": 4, "weight": 0.5, "name": "self"},
                {"game": "base-2p", "maps": 4, "seats": ["learner", "R"], "weight": 0.5, "name": "vsR"},
            ],
            "num_envs": num_envs,
        },
        "obs": {"level": "private"},
        "model": {"preset": "graph-tiny"},
        "train": {
            "algo": "ppo", "total_timesteps": rows * horizon * epochs, "learning_rate": 1e-3,
            "gamma": 1.0, "gae_lambda": 0.95, "update_epochs": 1, "clip_coef": 0.2, "vf_coef": 0.5,
            "vf_clip_coef": 0.2, "max_grad_norm": 0.5, "ent_coef": 0.01, "bptt_horizon": horizon,
            "minibatch_size": 64, "max_minibatch_size": 64, "optimizer": "adam", "adam_beta1": 0.9,
            "adam_beta2": 0.999, "adam_eps": 1e-8, "anneal_lr": False, "min_lr_ratio": 0.0, "amp": False,
            "precision": "float32", "prio_alpha": 0.0, "prio_beta0": 1.0, "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0, "checkpoint_interval": 2, "log_interval": 2,
        },
        "runtime": {"device": "cpu", "threads": 2, "determinism": "strict"},
    }
    config["train"].update(train)
    return copy.deepcopy(config)
