"""PPO through PufferLib: ``CatanPuffeRL`` and ``train_ppo``.

``train_ppo(variant, writer)`` trains one experiment variant: it builds the
map pools and the VecEnv (one ``Serial`` :class:`~catanrl.envs.puffer.CatanPufferEnv`),
the network, and a :class:`CatanPuffeRL`, then loops ``evaluate`` / ``train``
until the budget, a stop request or the time limit. The PufferLib pitfalls of
docs/experiments-implementation.md are handled here:

* ``pufferl`` is imported only through ``catanrl._puffer.import_pufferl``;
* logging is not throttled (``last_log_time`` is reset before every
  ``train``) and SPS is measured here;
* PufferLib's flags are overridden after ``PuffeRL.__init__``; the select
  eval runs with autocast off, a forked RNG and strict determinism;
* seeds: ``init`` before the model is built, ``sample`` after
  ``PuffeRL.__init__``, ``rollout`` for the VecEnv;
* checkpoints go through the run writer (PufferLib's ``save_checkpoint`` and
  its data_dir copy are replaced);
* rewards are win/loss in [0, 1], inside PufferLib's clamp; VP shaping, which
  can leave it, is refused;
* ``train.aux_coef`` > 0 trains a graph network's aux head (``model.args.aux``)
  on the row's critic block (``obs.critic_level`` above ``obs.level``) through
  the fork's ``aux_loss`` hook; ``losses/aux`` and ``aux/mse_<feature>`` /
  ``aux/var_<feature>`` are logged. At 0 (the default) the head, if any, is
  not run and training is the plain PPO.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from catanrl._puffer import import_pufferl
from catanrl.envs.encoding import action_size, critic_level_of, dims_for, row_layout
from catanrl.envs.puffer import make_vecenv, plan_env
from catanrl.models import CatanModel, build_model, normalize_levels
from catanrl.seeding import derive_seed as _derive_seed
from catanrl.seeding import determinism, rng_state, set_rng_state
from catanrl.seeding import seed_everything as _seed_everything
from catanrl.train.puffer_policy import PufferPolicy
from catanrl.train.select import Selector

pufferl = import_pufferl()

#: Every key PuffeRL reads from its config (tests/test_ppo_keys.py checks it by AST).
PUFFERL_KEYS = frozenset({
    "adam_beta1", "adam_beta2", "adam_eps", "amp", "anneal_lr", "batch_size", "bptt_horizon",
    "checkpoint_interval", "clip_coef", "compile", "compile_mode", "cpu_offload", "critic_gamma",
    "critic_targets", "data_dir", "device",
    "ent_coef", "env", "gae_lambda", "gamma", "learning_rate", "max_grad_norm", "max_minibatch_size",
    "min_lr_ratio", "minibatch_size", "optimizer", "precision", "prio_alpha", "prio_beta0", "seed",
    "torch_deterministic", "total_timesteps", "update_epochs", "use_rnn", "vf_clip_coef", "vf_coef",
    "vtrace_c_clip", "vtrace_rho_clip",
})
#: Keys train_ppo sets itself (from runtime, the env and the run), not from ``train``.
RUNNER_KEYS = frozenset({
    "batch_size", "compile", "compile_mode", "cpu_offload", "data_dir", "device", "env", "seed",
    "torch_deterministic", "use_rnn",
})
#: Keys of the ``train`` section that catanrl reads and PuffeRL does not.
CATANRL_TRAIN_KEYS = frozenset({"algo", "lr_schedule", "kl_target", "lr_min", "lr_max", "log_interval",
                                "muon_exclude", "muon_exclude_lr", "aux_coef"})


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def rollout_bytes(rows: int, horizon: int, obs_size: int) -> int:
    """GPU bytes of PuffeRL's experience buffers: every row's observations for
    ``bptt_horizon`` steps (float32) plus eight float32 scalars per row-step."""
    return int(rows) * int(horizon) * (int(obs_size) + 8) * 4


def puffer_config(train: Mapping[str, Any], runtime: Mapping[str, Any], *, rows: int, seed: int,
                  data_dir: str, strict: bool = False, recurrent: bool = False) -> dict[str, Any]:
    """PuffeRL's config: the ``train`` section minus catanrl's keys, plus the runner's.

    ``recurrent`` (the model's) sets ``use_rnn``: the training forward then
    sees segments x horizon, and PuffeRL sizes its (unused) state dict by
    the policy's ``hidden_size``.
    """
    config = {key: value for key, value in train.items() if key not in CATANRL_TRAIN_KEYS}
    clash = RUNNER_KEYS & set(config)
    if clash:
        raise ValueError(f"train sets runner-owned keys {sorted(clash)} (they belong in runtime)")
    config.update(
        device=resolve_device(runtime.get("device", "auto")),
        seed=int(seed),
        env="catan",
        data_dir=str(data_dir),
        use_rnn=bool(recurrent),
        torch_deterministic=bool(strict),
        cpu_offload=False,
        # runtime.compile compiles the network inside PufferPolicy; PuffeRL's own compile
        # would trace the acting-row selection too.
        compile=False,
        compile_mode="default",
        batch_size=int(rows) * int(config["bptt_horizon"]),
    )
    missing = PUFFERL_KEYS - set(config)
    unknown = set(config) - PUFFERL_KEYS
    if missing or unknown:
        raise ValueError(f"PuffeRL config: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if config["update_epochs"] * config["batch_size"] < config["minibatch_size"]:
        raise ValueError("minibatch_size exceeds the batch")
    return config


class _CaptureLogger:
    """PuffeRL's logger: keeps the last logs for train_ppo to complete and write."""

    run_id = "catanrl"

    def __init__(self) -> None:
        self.logs: dict[str, Any] | None = None

    def log(self, logs, step):
        self.logs = dict(logs)

    def close(self, model_path=None, early_stop=None):
        pass


class CatanPuffeRL(pufferl.PuffeRL):
    """PuffeRL with catanrl's checkpoints, state, select evals and quiet output."""

    def __init__(self, config: dict, vecenv: Any, policy: PufferPolicy, *, writer: Any,
                 select: Mapping[str, Any] | None = None, eval_runtime: Mapping[str, Any] | None = None,
                 lr_schedule: Mapping[str, Any] | None = None, muon_exclude: Sequence[str] = (),
                 muon_exclude_lr: float | None = None) -> None:
        self.writer = writer
        self.selector = Selector(select, writer, eval_runtime or {})
        self.lr_schedule = dict(lr_schedule or {"kind": "constant"})
        self.decisions = 0
        self.saved_epoch: int | None = None
        self._in_train = False
        self._checkpoint_due = False
        #: Recorded in every trainer state and compared on resume.
        self.extra_state: dict[str, Any] = {}
        self.capture = _CaptureLogger()
        self.msg = ""
        super().__init__(config, vecenv, policy, logger=self.capture)
        if muon_exclude:
            from catanrl.train.muon_exclude import exclude_from_muon

            exclude_from_muon(self.optimizer, self.model, muon_exclude, lr=float(muon_exclude_lr),
                              base_lr=float(config["learning_rate"]))
        # PuffeRL.__init__ set matmul precision, cudnn.benchmark and cudnn.deterministic.
        strict = bool(config["torch_deterministic"])
        torch.backends.cudnn.benchmark = not strict
        torch.backends.cudnn.deterministic = strict
        if strict:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        from catanrl.provenance import torch_runtime

        writer.record_runtime({"torch": torch_runtime(), "advantage_cuda": bool(pufferl.ADVANTAGE_CUDA),
                               "amp": config.get("amp"), "precision": config["precision"]})

    @property
    def model(self) -> CatanModel:
        return self.uncompiled_policy.model

    def print_dashboard(self, clear=False, idx=None, **kwargs):
        pass

    # ------------------------------------------------------------ state

    def state(self) -> dict[str, Any]:
        rng = rng_state()
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "decisions": self.decisions,
            # Copies: on CPU, .cpu() and state_dict() alias the live tensors.
            "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "rng": rng,
            **self.extra_state,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        for group in self.optimizer.param_groups:
            group["lr"] = state["learning_rate"]
        self.epoch = int(state["epoch"])
        self.global_step = int(state["global_step"])
        self.decisions = int(state["decisions"])
        rng = state.get("rng")
        if rng is not None:
            set_rng_state(rng)

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self) -> str:
        """Weights (the .pt), then trainer state (the commit), then the select eval.

        PuffeRL calls this from inside train(); step_epoch defers that call until
        the epoch's learning-rate adaptation is applied, so the committed state
        holds the learning rate the next epoch uses.
        """
        if self._in_train:
            self._checkpoint_due = True
            return ""
        ref = self.writer.save_checkpoint(self.model, epoch=self.epoch, agent_steps=self.global_step)
        self.writer.save_trainer_state(self.state())
        self.saved_epoch = self.epoch
        if self.selector.due(self.epoch):
            self.selector.run(ref, self.model)
        return str(getattr(ref, "path", ref))

    def close(self) -> None:
        self.vecenv.close()
        self.utilization.stop()

    # ------------------------------------------------------------ one epoch

    def step_epoch(self) -> dict[str, Any] | None:
        """Collect and train one epoch; returns the epoch's complete logs."""
        start = time.time()
        steps, env = self.global_step, self.vecenv.envs[0]
        decisions = env.decisions
        self.uncompiled_policy.begin_rollout()
        self.evaluate()
        collected = time.time()
        self.decisions += env.decisions - decisions
        self.last_log_time = 0.0  # PuffeRL logs at most every 0.25 s otherwise
        self.capture.logs = None
        self._in_train, self._checkpoint_due = True, False
        self.uncompiled_policy.snapshot_values(self.values)
        try:
            self.train()
        finally:
            self._in_train = False
        logs = self.capture.logs
        if logs is None:
            if self._checkpoint_due:
                self.save_checkpoint()
            return None
        elapsed = time.time() - start
        rows = self.global_step - steps
        logs = {key: value for key, value in logs.items() if key not in ("SPS", "uptime")}
        # PuffeRL logs before it stores this epoch's losses, so its logs carry the previous
        # epoch's; self.losses now holds this one's.
        logs.update({f"losses/{key}": value for key, value in self.losses.items()})
        logs.update({
            "decisions": self.decisions,
            "throughput/agent_steps_per_s": rows / elapsed,
            "throughput/decisions_per_s": (env.decisions - decisions) / elapsed,
            "throughput/collect_s": collected - start,
            "throughput/train_s": time.time() - collected,
        })
        logs.update({f"environment/{k}": v for k, v in env.take_stats().items()})
        logs.update({f"critic/{k}": v for k, v in self.uncompiled_policy.pop_calibration().items()})
        logs.update({f"aux/{k}": v for k, v in self.uncompiled_policy.pop_aux().items()})
        self._adapt_lr(logs)
        if self._checkpoint_due:
            self.save_checkpoint()
        return logs

    def _adapt_lr(self, logs: dict[str, Any]) -> None:
        """rl_games' KL-adaptive step size (v1's ``kl-adaptive``), when configured."""
        if self.lr_schedule.get("kind", "constant") != "kl_adaptive":
            return
        kl = logs.get("losses/approx_kl")
        if kl is None or not math.isfinite(kl):
            return
        target = float(self.lr_schedule["kl_target"])
        lr = float(self.optimizer.param_groups[0]["lr"])
        if kl > 2.0 * target:
            lr /= 1.5
        elif kl < 0.5 * target:
            lr *= 1.5
        lr = min(max(lr, float(self.lr_schedule["lr_min"])), float(self.lr_schedule["lr_max"]))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        logs["learning_rate"] = lr


def check_bindable(model: CatanModel, plan: Any) -> None:
    """Refuse a network evals could not bind, or a pool it would misread.

    Evals play at the generators' bounds (``dims_for``); a network that is not
    dims-agnostic (mlp, cnn) trained at other dims (``env.dims: pool`` or
    explicit) would train fine and then fail every select and heldout eval.
    The pool's maps then go through the network's own check: the cnn's plane
    is one board's, and a pool with the same dims in another tile order (the
    hexagon generator against BASE) would train on scrambled boards with
    nothing but weaker play to show for it.
    """
    bounds = dims_for([spec.game for spec in plan.specs])
    if not model.dims_agnostic and dict(model.dims) != bounds:
        raise ValueError(
            f"a {model.spec['class']} network is not dims-agnostic: it must train at the generators' "
            f"bounds {bounds} (env.dims: auto), not {dict(model.dims)}")
    model.check_maps(catan_map for scenario in plan.scenarios for catan_map in scenario.maps)


def build_policy_model(cfg: Mapping[str, Any], dims: Mapping[str, int]) -> CatanModel:
    obs = cfg.get("obs") or {}
    levels = normalize_levels({"level": obs.get("level", "private"), "critic_level": obs.get("critic_level")})
    critic = critic_level_of(levels["level"], levels["critic_level"])
    return build_model(cfg["model"], dims=dims, levels=levels, num_actions=action_size(dims),
                       layout=row_layout(dims, levels["level"], critic))


def _lr_schedule(train: Mapping[str, Any]) -> dict[str, Any]:
    kind = train.get("lr_schedule", "constant")
    if kind == "constant":
        return {"kind": "constant"}
    if kind != "kl_adaptive":
        raise ValueError(f"train.lr_schedule must be constant or kl_adaptive, not {kind!r}")
    if train.get("anneal_lr"):
        raise ValueError("kl_adaptive needs anneal_lr: false")
    missing = [key for key in ("kl_target", "lr_min", "lr_max") if key not in train]
    if missing:
        raise ValueError(f"train.lr_schedule kl_adaptive needs {missing}")
    return {"kind": kind, "kl_target": float(train["kl_target"]), "lr_min": float(train["lr_min"]),
            "lr_max": float(train["lr_max"])}


def _aux_coef(train: Mapping[str, Any], model: CatanModel) -> float:
    """train.aux_coef, checked against the network: a positive weight needs the aux head
    and the critic block it is trained on; 0 leaves a head, if any, untrained."""
    coef = float(train.get("aux_coef", 0.0) or 0.0)
    if coef < 0.0:
        raise ValueError(f"train.aux_coef must be >= 0, not {coef}")
    if coef and not getattr(model, "aux", False):
        raise ValueError("train.aux_coef > 0 needs a network with an aux head: model.args.aux on a shared graph "
                         "trunk (model.args.shared)")
    if coef and not model.has_critic_block:
        raise ValueError("train.aux_coef > 0 trains the aux head on the critic block: set obs.critic_level above "
                         "obs.level (its players block is the target, which no head reads)")
    return coef


def _critic_options(train: Mapping[str, Any]) -> dict[str, Any]:
    """train.critic_targets, critic_gamma, muon_exclude and muon_exclude_lr, checked."""
    # the fallbacks (vtrace, None) are the pre-2026-10-19 defaults, for callers that bypass config
    # resolution; a resolved config carries both keys from configs/train/ppo.yaml (lambda, 1.0 since then)
    targets = train.get("critic_targets", "vtrace")
    if targets not in ("vtrace", "lambda"):
        raise ValueError(f"train.critic_targets must be vtrace or lambda, not {targets!r}")
    gamma = train.get("critic_gamma")
    if gamma is not None:
        if targets != "lambda":
            raise ValueError("train.critic_gamma is the discount of critic_targets: lambda")
        gamma = float(gamma)
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"train.critic_gamma must be in (0, 1], not {gamma}")
    patterns = train.get("muon_exclude") or []
    if isinstance(patterns, str) or not all(isinstance(p, str) for p in patterns):
        raise ValueError("train.muon_exclude is a list of parameter-name patterns")
    lr = train.get("muon_exclude_lr")
    if bool(patterns) != (lr is not None):
        raise ValueError("train.muon_exclude and muon_exclude_lr (AdamW's learning rate) go together")
    return {"critic_targets": targets, "critic_gamma": gamma, "muon_exclude": list(patterns),
            "muon_exclude_lr": None if lr is None else float(lr)}


def train_ppo(variant: Any, writer: Any) -> None:
    """Train ``variant`` (a :class:`catanrl.config.Variant`) into ``writer``'s run."""
    cfg = variant.config
    seed = int(cfg["seed"])
    launch = int(getattr(writer, "launch", 0))
    env_cfg, train, runtime = cfg["env"], dict(cfg["train"]), dict(cfg.get("runtime") or {})
    reward = env_cfg.get("reward") or {}
    vp_reward = float(reward.get("vp", 0.0))
    if vp_reward:
        raise ValueError("env.reward.vp shaping can push a step outside PufferLib's reward clamp [-1, 1]")
    strict = runtime.get("determinism", "fast") == "strict"
    with determinism("strict" if strict else "fast"):
        _train_ppo(variant, writer, cfg, seed, launch, env_cfg, train, runtime, strict)


def _train_ppo(variant, writer, cfg, seed, launch, env_cfg, train, runtime, strict) -> None:
    train = {**train, **_critic_options(train)}
    pool_cfg = env_cfg.get("map_pool") or {}
    pool_seed = pool_cfg.get("seed")
    pool_seed = _derive_seed(seed, "map_pool") if pool_seed is None else int(pool_seed)
    plan = plan_env(env_cfg, pool_seed=pool_seed)

    _seed_everything(_derive_seed(seed, "init"))
    model = build_policy_model(cfg, plan.dims)
    check_bindable(model, plan)
    if cfg.get("init_from") and getattr(writer, "mode", "new") == "new":
        from catanrl.checkpoint import load_parts

        load_parts(model, cfg["init_from"])
    device = resolve_device(runtime.get("device", "auto"))
    model.to(device)

    obs = cfg.get("obs") or {}
    level = obs.get("level", "private")
    rollout_seed = _derive_seed(seed, "rollout", launch)
    vecenv = make_vecenv(
        plan, num_envs=int(env_cfg["num_envs"]), level=level, critic_level=obs.get("critic_level"),
        threads=int(runtime.get("threads", 1)), seed=rollout_seed,
        log_interval=int(train.get("log_interval", 32)),
    )
    critic_gamma = train["critic_gamma"] if train["critic_gamma"] is not None else float(train["gamma"])
    policy = PufferPolicy(model, num_rows=vecenv.num_agents, compile=bool(runtime.get("compile", False)),
                          critic_gamma=critic_gamma, aux_coef=_aux_coef(train, model))
    config = puffer_config(train, runtime, rows=vecenv.num_agents, seed=rollout_seed,
                           data_dir=str(getattr(writer, "path", ".")), strict=strict, recurrent=policy.recurrent)
    buffer = rollout_bytes(vecenv.num_agents, config["bptt_horizon"], vecenv.single_observation_space.shape[0])
    if device.startswith("cuda") and buffer > torch.cuda.get_device_properties(device).total_memory:
        raise ValueError(f"the rollout buffer needs {buffer / 2**30:.1f} GiB: lower env.num_envs or "
                         "train.bptt_horizon")
    select = (cfg.get("eval") or {}).get("select")
    eval_runtime = {key: runtime[key] for key in ("device", "num_envs", "threads", "infer_rows") if key in runtime}
    trainer = CatanPuffeRL(config, vecenv, policy, writer=writer, select=select, eval_runtime=eval_runtime,
                           lr_schedule=_lr_schedule(train), muon_exclude=train["muon_exclude"],
                           muon_exclude_lr=train["muon_exclude_lr"])
    status = "finished"
    try:  # from here on, close() stops PuffeRL's non-daemon utilization thread
        trainer.extra_state["map_pool"] = plan.pool["hash"]
        torch.manual_seed(_derive_seed(seed, "sample", launch))

        state = writer.load_trainer_state() if getattr(writer, "mode", "new") == "resume" else None
        if state is not None:
            if state.get("map_pool") not in (None, plan.pool["hash"]):
                raise RuntimeError("the map pool changed since the run started; refusing to resume")
            trainer.restore(state)
            trainer.selector.restore()
        else:
            trainer.save_checkpoint()  # e000000
        for ref in writer.pending_selects():
            trainer.selector.run(ref, model)

        while trainer.global_step < config["total_timesteps"]:
            if writer.should_stop():
                status = getattr(writer, "stop_reason", None) or "interrupted"
                break
            logs = trainer.step_epoch()
            if logs is not None:
                logs["map_pool"] = plan.pool["hash"]
                logs["memory/rollout_GiB"] = buffer / 2**30
                if device.startswith("cuda"):
                    logs["memory/peak_GiB"] = torch.cuda.max_memory_allocated() / 2**30
                writer.log(logs, trainer.global_step)
        if trainer.saved_epoch != trainer.epoch:
            trainer.save_checkpoint()  # the off-interval checkpoint
    finally:
        trainer.close()
    writer.finish(status)
