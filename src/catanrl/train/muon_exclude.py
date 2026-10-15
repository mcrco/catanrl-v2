"""Parameters taken off the main optimizer and given to AdamW (``train.muon_exclude``).

Muon (heavyball's ForeachMuon) orthogonalizes every matrix's update, so a
1 x H output layer moves the same distance at every optimizer step whatever
its gradient. On the value head's output that swings the predicted seat sum
up and down every step, and with an even number of steps per epoch the
rollout network and every checkpoint land on the same side.

:func:`exclude_from_muon` gives the parameters whose names match any of the
patterns (``fnmatch`` over the :class:`~catanrl.models.CatanModel`'s
parameter names, e.g. ``value_head.mlp.4.weight``) to AdamW (PyTorch's
defaults besides the learning rate). AdamW steps them in a pre-hook of the
main optimizer's step, then drops their gradients, so the main optimizer
(Muon, or Adam) skips them as parameters without a gradient. Its learning
rate is ``lr`` times the main learning rate's current fraction of
``base_lr`` (the configured ``learning_rate``), so it follows the cosine,
the KL schedule and a restore. Its state rides in the main optimizer's
``state_dict`` under :data:`KEY`.
"""

from __future__ import annotations

from collections.abc import Sequence
from fnmatch import fnmatchcase

import torch
from torch import nn

#: The key of AdamW's state inside the main optimizer's state_dict.
KEY = "catanrl_muon_exclude"


def excluded_parameters(model: nn.Module, patterns: Sequence[str]) -> list[tuple[str, nn.Parameter]]:
    """The model's ``(name, parameter)`` pairs matching any pattern; every pattern must match."""
    named = list(model.named_parameters())
    unmatched = [pattern for pattern in patterns if not any(fnmatchcase(name, pattern) for name, _ in named)]
    if unmatched:
        raise ValueError(f"train.muon_exclude: {unmatched} match no parameter of the model "
                         f"(its names look like {named[0][0] if named else None!r})")
    return [(name, p) for name, p in named if any(fnmatchcase(name, pattern) for pattern in patterns)]


def exclude_from_muon(optimizer: torch.optim.Optimizer, model: nn.Module, patterns: Sequence[str], *,
                      lr: float, base_lr: float) -> torch.optim.AdamW:
    """Step the parameters of ``model`` matching ``patterns`` by AdamW instead of ``optimizer``."""
    if base_lr <= 0:
        raise ValueError("train.muon_exclude needs a positive learning_rate (AdamW's follows its fraction)")
    params = [p for _, p in excluded_parameters(model, patterns)]
    managed = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if any(id(p) not in managed for p in params):
        raise ValueError("train.muon_exclude: a matched parameter is not the optimizer's")
    adamw = torch.optim.AdamW(params, lr=lr)

    def step_excluded(opt, args, kwargs):
        fraction = opt.param_groups[0]["lr"] / base_lr
        for group in adamw.param_groups:
            group["lr"] = lr * fraction
        adamw.step()
        for p in params:
            p.grad = None  # the main optimizer skips parameters without a gradient

    def save(opt, state_dict):
        state_dict[KEY] = adamw.state_dict()
        return state_dict

    def load(opt, state_dict):
        saved = state_dict.pop(KEY, None)
        if saved is not None:
            adamw.load_state_dict(saved)
        return state_dict

    optimizer.register_step_pre_hook(step_excluded)
    optimizer.register_state_dict_post_hook(save)
    optimizer.register_load_state_dict_pre_hook(load)
    return adamw
