"""Test 9: agent specs: canonical forms, param checks, key dicts, fairness, registry."""

from __future__ import annotations

from types import SimpleNamespace

import cppanatron as cpp
import pytest
import torch
from catanatron.models.player import Color
from catanatron.registry import REGISTRY, SpecError

from catanrl.agents import MctsParams, TradeParams, resolve_agent
from catanrl.agents.spec import TradePlayer, check_spec_syntax, spec_ckpts, trade_shapes
from catanrl.checkpoint import save
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.models import build_model
from catanrl.runs import RunWriter, spec_runs


def tiny(level="private", games=("base-2p",), preset="graph-tiny", seed=0):
    torch.manual_seed(seed)
    dims = dims_for(list(games))
    return build_model({"preset": preset}, dims=dims, levels=level, num_actions=action_size(dims),
                       layout=row_layout(dims, level))


@pytest.fixture
def ckpts(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    paths = {}
    for level in ("private", "public", "full"):
        model = tiny(level)
        paths[level] = save(tmp_path / f"{level}.pt", model, run=None, epoch=0, agent_steps=0,
                            config={"model": model.spec}, code={}).path
    raw = {"seed": 1, "env": {"scenarios": [{"game": "base-2p"}]}, "model": {"preset": "graph-tiny"}}
    cfg = resolve(raw, experiment="exp", variant="v")
    with RunWriter.open(Variant("exp", "v", None, {}, cfg, config_hash(cfg)), argv=["t"]) as w:
        model = tiny()
        w.save_checkpoint(model, epoch=0, agent_steps=0)
        w.save_trainer_state({"epoch": 0})
        ref = w.save_checkpoint(tiny(seed=1), epoch=3, agent_steps=30)
        w.save_trainer_state({"epoch": 3})
        w.set_alias("best", ref, record=SimpleNamespace(eval_id="x", summary={"score_rate": 0.5}))
        w.finish("finished")
    paths["run"] = ref
    return paths


def test_canonical_is_a_fixed_point_and_pins_best(ckpts):
    ref = ckpts["run"]
    agent = resolve_agent("POLICY:exp/v")  # positional ckpt, default @best
    assert agent.canonical == (f"POLICY:ckpt=exp/v@e3#{ref.weights_sha256[:12]}:sample=false:temperature=1.0"
                               ":allow_drift=false")
    assert resolve_agent(agent.canonical).canonical == agent.canonical
    assert agent.checkpoint.epoch == 3 and agent.fair
    mcts = resolve_agent("MCTS:ckpt=exp/v@latest:sims=8:dets=2:device=cpu:threads=3")
    assert resolve_agent(mcts.canonical).canonical == mcts.canonical
    assert "device" not in mcts.canonical and "threads" not in mcts.canonical
    assert mcts.params.threads == 3  # still used to execute
    path = resolve_agent(f"BELIEF:ckpt={ckpts['full']}")
    sha = path.checkpoint.weights_sha256[:12]
    assert path.canonical.startswith(f"BELIEF:ckpt={ckpts['full'].resolve()}#{sha}:worlds=16")
    assert resolve_agent(path.canonical).canonical == path.canonical


def test_bad_specs_raise(ckpts):
    for spec in ("MCTS:ckpt=exp/v:det=8", "MCTS:ckpt=exp/v:oracle=true:dets=4", "MCTS:ckpt=exp/v:dets=0",
                 "POLICY:ckpt=exp/v,F", "POLICY:ckpt=/tmp/a:b.pt", "POLICY:sample=true",
                 "POLICY:ckpt=exp/v@nope", "POLICY:ckpt=exp/missing", "NOPE",
                 "MCTS:ckpt=exp/v:oracle=true:tracker=exact", "BELIEF:ckpt=exp/v:tracker=perfect"):
        with pytest.raises(SpecError):
            resolve_agent(spec)


def test_tracker_param(ckpts):
    """``tracker`` (pool by default, or exact) is part of the canonical spec, the key dict and the
    native agent; it decides nothing for an oracle."""
    pool = resolve_agent("BELIEF:ckpt=exp/v:worlds=2")
    exact = resolve_agent("BELIEF:ckpt=exp/v:worlds=2:tracker=exact")
    assert pool.canonical.endswith(":allow_drift=false:tracker=pool") and pool.params.tracker == "pool"
    assert exact.canonical.endswith(":tracker=exact") and resolve_agent(exact.canonical).canonical == exact.canonical
    assert pool.key_dict()["params"]["tracker"] == "pool" and exact.key_dict()["params"]["tracker"] == "exact"
    assert pool.key_dict() != exact.key_dict() and exact.fair
    assert exact.native(device="cpu").tracker_mode == cpp.TrackerMode.EXACT
    assert pool.native(device="cpu").tracker_mode == cpp.TrackerMode.POOL
    mcts = resolve_agent("MCTS:ckpt=exp/v:sims=4:dets=2:tracker=exact:device=cpu")
    assert mcts.fair and mcts.native().tracker_mode == cpp.TrackerMode.EXACT and mcts.native().needs_trackers
    assert mcts.key_dict()["params"]["tracker"] == "exact"
    assert mcts.key_dict()["search"] == resolve_agent("MCTS:ckpt=exp/v:sims=4:dets=2").key_dict()["search"]


def test_key_dict(ckpts):
    a = resolve_agent("MCTS:ckpt=exp/v:sims=8:oracle=true:device=cpu:threads=2")
    b = resolve_agent("MCTS:ckpt=exp/v:sims=8:oracle=true:device=cuda:threads=4")
    assert a.key_dict() == b.key_dict()
    search = a.key_dict()["search"]
    assert search["simulations"] == 8 and search["determinizations"] == 0 and search["root"] == "PUCT"
    assert set(search) >= {"c_puct", "fpu_value", "gumbel_c_visit", "max_pending", "skip_forced"}
    assert resolve_agent("MCTS:ckpt=exp/v:sims=9").key_dict() != resolve_agent("MCTS:ckpt=exp/v:sims=8").key_dict()
    assert resolve_agent("POLICY:exp/v@e0").key_dict()["checkpoint"]["weights_sha256"] != \
        resolve_agent("POLICY:exp/v@e3").key_dict()["checkpoint"]["weights_sha256"]


def test_fairness_and_harness_gates(ckpts):
    fair = {level: resolve_agent(f"POLICY:ckpt={path}").fair for level, path in ckpts.items() if level != "run"}
    assert fair == {"private": True, "public": True, "full": False}
    assert not resolve_agent(f"POLICY:ckpt={ckpts['public']}", games=["base-4p"]).fair
    assert resolve_agent(f"BELIEF:ckpt={ckpts['full']}").fair
    assert not resolve_agent(f"MCTS:ckpt={ckpts['full']}:oracle=true").fair
    assert resolve_agent(f"MCTS:ckpt={ckpts['full']}:dets=2").fair
    assert not resolve_agent("F").fair and not resolve_agent("R").fair
    with pytest.raises(SpecError, match="BELIEF"):
        resolve_agent(f"POLICY:ckpt={ckpts['full']}", harness="server")
    resolve_agent(f"POLICY:ckpt={ckpts['private']}", harness="server")
    with pytest.raises(SpecError):
        resolve_agent("AB:2")  # Catanatron-only, natively
    assert resolve_agent("AB:2", harness="catanatron").canonical.startswith("AB:depth=2")
    with pytest.raises(SpecError):
        resolve_agent("F:value_fn=contender")


def test_registry_and_defaults(ckpts):
    for key in ("POLICY", "MCTS", "BELIEF"):
        assert key in REGISTRY
    assert MctsParams(ckpt="x").c_puct == cpp.SearchConfig().c_puct
    player = REGISTRY.build(f"POLICY:ckpt={ckpts['private']}", Color.RED)
    assert player.color == Color.RED and player.params.ckpt == str(ckpts["private"])


def test_mlp_dims_must_match(ckpts, tmp_path):
    model = tiny(preset="mlp")
    path = save(tmp_path / "mlp.pt", model, run=None, epoch=0, agent_steps=0, config={}, code={}).path
    resolve_agent(f"POLICY:ckpt={path}", games=["base-2p"])
    with pytest.raises(SpecError, match="dims"):
        resolve_agent(f"POLICY:ckpt={path}", games=["base-4p"])
    resolve_agent(f"POLICY:ckpt={ckpts['private']}", games=["base-4p"])  # graph nets run anywhere


@pytest.mark.parametrize("spec", ["POLICY:ckpt={path}:sample=true", "BELIEF:ckpt={path}:worlds=2"])
def test_whole_catanatron_games_stay_in_sync(ckpts, spec):
    from catanrl.agents import player
    from catanrl.games import load_game

    config = load_game("base-2p")
    players = [player(spec.format(path=ckpts["private"]), Color.RED), REGISTRY.build("R", Color.BLUE)]
    game = config.catanatron_game(players, map_seed=3, seed=5)
    game.play()
    assert game.state.num_turns > 10 and len(players[0]._mirror["records"]) > 50


def test_a_path_spec_pins_its_weights(ckpts, tmp_path, monkeypatch):
    agent = resolve_agent(f"POLICY:ckpt={ckpts['private']}")
    ckpts["private"].unlink()  # other weights at the same path
    save(ckpts["private"], tiny(seed=1), run=None, epoch=0, agent_steps=0, config={}, code={})
    with pytest.raises(SpecError, match="not #"):
        resolve_agent(agent.canonical)
    monkeypatch.chdir(tmp_path)
    relative = resolve_agent("POLICY:ckpt=public.pt")
    assert relative.checkpoint.path.is_absolute() and relative.checkpoint.run is None


def test_the_mirror_follows_an_undo(ckpts):
    """Undo then another action keeps the log's length; the mirror must still rebuild."""
    from cppanatron.catanatron import from_catanatron_action

    from catanrl.agents import player
    from catanrl.games import load_game

    red = player(f"POLICY:ckpt={ckpts['private']}", Color.RED)
    game = load_game("base-2p").catanatron_game([red, REGISTRY.build("R", Color.BLUE)], map_seed=3, seed=5)
    while len(game.state.action_records) < 30 or len(game.playable_actions) < 2:
        game.play_tick()
    snapshot = game.copy()
    game.execute(game.playable_actions[0])
    red._sync(game, red._agent())
    snapshot.execute(snapshot.playable_actions[-1])  # the undo, then a different action
    mirror = red._sync(snapshot, red._agent())
    native = {tuple(a) for a in mirror["game"].playable_actions}
    assert native == {tuple(from_catanatron_action(a)) for a in snapshot.playable_actions}


# ---------------------------------------------------------------- TRADE


@pytest.fixture
def trade_ckpts(ckpts, tmp_path):
    """``ckpts`` plus the critics v1 refuses: a recurrent (lstm-tiny) network and a graph
    network with a separate full critic block."""
    lstm = tiny(preset="lstm-tiny")
    ckpts["lstm"] = save(tmp_path / "lstm.pt", lstm, run=None, epoch=0, agent_steps=0,
                         config={"model": lstm.spec}, code={}).path
    torch.manual_seed(2)
    dims = dims_for(["base-2p"])
    block = build_model({"preset": "graph-tiny"}, dims=dims, levels={"level": "private", "critic_level": "full"},
                        num_actions=action_size(dims), layout=row_layout(dims, "private", "full"))
    ckpts["block"] = save(tmp_path / "block.pt", block, run=None, epoch=0, agent_steps=0,
                          config={"model": block.spec}, code={}).path
    return ckpts


def test_trade_canonical_names_the_critic_and_keys_both_networks(ckpts):
    ref = ckpts["run"]
    pinned = f"exp/v@e3#{ref.weights_sha256[:12]}"
    one = resolve_agent("TRADE:ckpt=exp/v")
    assert one.canonical == (f"TRADE:ckpt={pinned}:critic={pinned}:margin=0.0:accept_margin=0.0:dets=8"
                             ":min_feasible=0.25:tracker=pool:offers=3:when=first:shapes=11-21-12:propose=true"
                             ":oracle=false:sample=false:temperature=1.0:allow_drift=false")
    assert resolve_agent(one.canonical).canonical == one.canonical
    for same in ("TRADE:ckpt=exp/v:critic=exp/v", "TRADE:ckpt=exp/v@best:critic=exp/v@e3",
                 "TRADE:ckpt=exp/v:audit=true:device=cpu"):
        other = resolve_agent(same)
        assert other.canonical == one.canonical and other.key_dict() == one.key_dict()
    assert str(one.critic) == str(one.checkpoint) == str(ref) and one.checkpoints == (one.checkpoint,)
    assert one.fair and one.label == "TRADE:exp/v@e3" and isinstance(one.params, TradeParams)
    assert one.params.critic == pinned and one.params.audit is False
    key = one.key_dict()
    assert key["checkpoint"]["weights_sha256"] == ref.weights_sha256 == key["critic"]["weights_sha256"]
    assert key["critic"]["levels"]["level"] == "private" and key["critic"]["model"]["class"] == "graph"
    assert key["params"] == {"margin": 0.0, "accept_margin": 0.0, "dets": 8, "min_feasible": 0.25, "tracker": "pool",
                             "offers": 3, "when": "first", "shapes": "11-21-12", "propose": True, "oracle": False,
                             "sample": False, "temperature": 1.0, "allow_drift": False}
    arm = resolve_agent("TRADE:ckpt=exp/v:margin=1")
    assert arm.key_dict() != key and ":margin=1.0:" in arm.canonical and arm.key_dict()["params"]["margin"] == 1.0
    two = resolve_agent("TRADE:ckpt=exp/v:critic=exp/v@e0:propose=false")
    assert two.label == "TRADE:exp/v@e3+critic exp/v@e0" and two.critic.epoch == 0 and two.checkpoint.epoch == 3
    assert two.checkpoints == (two.checkpoint, two.critic) and not two.key_dict()["params"]["propose"]
    assert two.key_dict()["critic"]["weights_sha256"] != ref.weights_sha256
    assert resolve_agent(two.canonical).canonical == two.canonical
    assert spec_ckpts("TRADE:ckpt=exp/v@best:critic=exp/v@e0:margin=1") == ["exp/v@best", "exp/v@e0"]
    assert spec_ckpts("TRADE:ckpt=exp/v") == ["exp/v"]
    assert spec_runs("TRADE:ckpt=exp/v@best:critic=exp/w@e0") == ["exp/v", "exp/w"]
    assert spec_runs(f"TRADE:ckpt=exp/v:critic={ckpts['private']}") == ["exp/v"]
    check_spec_syntax("TRADE:ckpt=${run}@best:critic=exp/v@best:margin=1")
    check_spec_syntax("TRADE:ckpt=exp/v@best:critic=${params.critic}")
    assert trade_shapes("21") == ((2, 1),) and trade_shapes("11-21-12") == ((1, 1), (2, 1), (1, 2))


def test_trade_fairness_and_refusals(trade_ckpts):
    c = trade_ckpts
    assert resolve_agent(f"TRADE:ckpt={c['private']}").fair
    assert not resolve_agent(f"TRADE:ckpt={c['private']}:oracle=true:dets=1").fair
    public = resolve_agent(f"TRADE:ckpt={c['public']}:critic={c['private']}")
    assert public.fair and not resolve_agent(public.canonical, games=["base-4p"]).fair
    assert resolve_agent(f"TRADE:ckpt={c['lstm']}:critic={c['private']}").fair  # a recurrent policy, natively
    for spec, reason in (
        (f"TRADE:ckpt={c['private']}:critic={c['lstm']}", "recurrent"),
        (f"TRADE:ckpt={c['lstm']}", "recurrent"),  # one network: the critic is the policy's
        (f"TRADE:ckpt={c['private']}:critic={c['block']}", "critic block"),
        (f"TRADE:ckpt={c['private']}:critic={c['public']}", "private"),
        (f"TRADE:ckpt={c['full']}", "private"),
        ("TRADE:ckpt=exp/v:dets=0", "dets"),
        ("TRADE:ckpt=exp/v:oracle=true", "dets"),
        ("TRADE:ckpt=exp/v:oracle=true:dets=1:tracker=exact", "tracker"),
        ("TRADE:ckpt=exp/v:shapes=22", "shapes"),
        ("TRADE:ckpt=exp/v:shapes=11-11", "shapes"),
        ("TRADE:ckpt=exp/v:shapes=", "shapes"),
        ("TRADE:ckpt=exp/v:margin=-1", "margin"),
        ("TRADE:ckpt=exp/v:accept_margin=-0.5", "margin"),
        ("TRADE:ckpt=exp/v:offers=0", "offers"),
        ("TRADE:ckpt=exp/v:min_feasible=0", "min_feasible"),
        ("TRADE:ckpt=exp/v:min_feasible=1.5", "min_feasible"),
        ("TRADE:ckpt=exp/v:when=never", "when"),
        ("TRADE:ckpt=exp/v:critic=exp/nope", "nope"),
        ("TRADE:ckpt=exp/v:worlds=2", "worlds"),
        ("TRADE:critic=exp/v", "ckpt"),
    ):
        with pytest.raises(SpecError, match=reason):
            resolve_agent(spec)
    for harness in ("catanatron", "server"):
        with pytest.raises(SpecError, match="native"):
            resolve_agent("TRADE:ckpt=exp/v", harness=harness)
    assert "TRADE" in REGISTRY
    player = REGISTRY.build(f"TRADE:ckpt={c['private']}:margin=1", Color.RED)
    assert isinstance(player, TradePlayer) and player.params.margin == 1.0 and player.params.offers == 3
    with pytest.raises(RuntimeError, match="native"):
        player.decide(None, [1, 2])
