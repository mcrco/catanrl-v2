"""CANOPY (catanrl.canopy): the board, the action ids, the snapshot read back, fairness (the
true hidden cards never reach the bridge), the agent through the harness with a stand-in pool
(fallbacks, refreshes, determinism across envs and processes), resolution, and the code id.

No Canopy binary is needed; ``test_real_bridge_*`` run against the real release when
``CATANRL_CANOPY_DIR`` names one and are skipped otherwise.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
from collections import Counter
from pathlib import Path

import cppanatron as cpp
import pytest
from catanatron.registry import SpecError

from catanrl import canon
from catanrl.agents.native import BotAgent
from catanrl.canopy import agent as canopy_agent
from catanrl.canopy import is_canopy_spec, lineup_has_canopy, resolve
from catanrl.canopy.agent import CanopyAgent, CanopyParams, canopy_params, summary_totals
from catanrl.canopy.release import (
    CANOPY_COMMIT,
    CHECKPOINT_SHA256,
    MANIFEST_NAME,
    PATCHES,
    Release,
    ReleaseError,
    load_release,
)
from catanrl.canopy.snapshot import (
    AT,
    BASE_DECK,
    DEV_TO_CANOPY,
    NUM_ACTIONS,
    SeatHistory,
    SnapshotError,
    action_id,
    action_name,
    check_rules,
    legal_by_id,
    replay_chain,
    snapshot,
)
from catanrl.canopy.topology import PORT_NODES, RESOURCES, TILE_NODES, CanopyBoard, MapError
from catanrl.eval import results
from catanrl.eval.harness import play_schedule
from catanrl.eval.suites import Suite
from catanrl.games import load_game
from catanrl.seeding import derive_seed

BASE = load_game("base-2p")
CANOPY_TO_DEV = tuple(DEV_TO_CANOPY.index(k) for k in range(5))  # Canopy kind -> engine index
KIND_STARTS = {"settlement": 0, "road": 54, "city": 126, "roll": 180, "end_turn": 181, "buy_dev": 182,
               "knight": 183, "road_building": 184, "year_of_plenty": 185, "monopoly": 200, "robber": 205,
               "discard": 224, "maritime": 229}


def kind_of(aid: int) -> str:
    return max((start, name) for name, start in KIND_STARTS.items() if start <= aid)[1]


def _choose(game: cpp.Game, rng: random.Random) -> cpp.Action:
    """A random move that favours building and cards, so games reach every phase."""
    legal = list(game.playable_actions)
    keen = [a for a in legal if a.action_type in (AT.BUILD_CITY, AT.BUILD_SETTLEMENT, AT.BUY_DEVELOPMENT_CARD,
                                                   AT.PLAY_KNIGHT_CARD, AT.PLAY_ROAD_BUILDING,
                                                   AT.PLAY_YEAR_OF_PLENTY, AT.PLAY_MONOPOLY, AT.MARITIME_TRADE)]
    if keen and rng.random() < 0.6:
        return rng.choice(keen)
    return rng.choice(legal)


def positions(games: int = 4, seed: int = 0, max_steps: int = 900):
    """Decision positions of random base-2p games: (game, seat, tracker, history, board, previous),
    each seat's history kept as the agent keeps it; ``previous`` is the seat's last
    (game, action) for chain replays."""
    for g in range(games):
        game = BASE.cppanatron_game(map_seed=derive_seed(seed, "map", g), seed=derive_seed(seed, "game", g))
        board = CanopyBoard.from_map(game.board.map)
        trackers = [cpp.HandTracker(game, s) for s in range(2)]
        history = [SeatHistory(0), SeatHistory(1)]
        previous: list[tuple | None] = [None, None]
        rng = random.Random(g + 1000 * seed)
        for _ in range(max_steps):
            if game.winning_player_index is not None:
                break
            seat = int(game.current_player_index)
            legal = list(game.playable_actions)
            if len(legal) > 1:
                history[seat].start(game, trackers[seat])
                yield game.copy(), seat, trackers[seat].copy(), copy.deepcopy(history[seat]), board, previous[seat]
                action = _choose(game, rng)
                after = game.copy()
                after.execute(action)
                history[seat].played(game, trackers[seat], action, after)
                previous[seat] = (game.copy(), action)
            else:
                action = legal[0]
            before = game.copy()
            game.execute(action)
            for tracker in trackers:
                tracker.observe(before, action, game)


@pytest.fixture(scope="module")
def sample():
    return list(positions())


def _snap(game, seat, tracker, history, board, tree=None, seed=1):
    return snapshot(game, tracker, board, history, game_id="g", seed=seed, world_seed=derive_seed(seed, "w"),
                    tree_actions=tree)


# ---------------------------------------------------------------- board and actions


def test_suite_maps_are_canopy_boards():
    suite = Suite(name="t", purpose="dev", seed=67, boards=30, scenarios=({"game": "base-2p"},), turn_limit=500)
    for scheduled in suite.schedule(2)[::2]:
        catan_map = cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(BASE.map_spec(scheduled.map_seed)))
        board = CanopyBoard.from_map(catan_map)
        assert board.port_set == "primary"
        assert sorted(board.node_to_canopy.values()) == list(range(54))
        assert sorted(board.edge_to_canopy.values()) == list(range(72))
        land = catan_map.land_tiles
        for t, coord in enumerate(board.tile_coords):
            assert [board.node_to_canopy[int(n)] for n in land[coord].nodes] == list(TILE_NODES[t])
            name = "DESERT" if land[coord].resource is None else land[coord].resource.name
            assert board.terrains[t] == name and board.numbers[t] == land[coord].number
        ports = {int(n): r for r, nodes in catan_map.port_nodes.items() for n in nodes}
        for (a, b), name in zip(PORT_NODES["primary"], board.ports):
            r = ports[board.canopy_to_node[a]]
            assert r == ports[board.canopy_to_node[b]] and (None if r is None else r.name) == name


@pytest.mark.parametrize("map_config", [{"type": "MINI"}, {"type": "hexagon", "radius": 3},
                                        {"type": "blob", "land_tiles": 19}])
def test_other_maps_are_refused(map_config):
    game = dataclasses.replace(BASE, map=map_config).cppanatron_game(map_seed=3, seed=4)
    with pytest.raises(MapError):
        CanopyBoard.from_map(game.board.map)
    agent = CanopyAgent(CanopyParams(), None, pool_factory=lambda: FakePool())
    with pytest.raises(ValueError, match="CANOPY"):
        agent.check_maps([game.board.map])


def test_other_rules_are_refused():
    for config in (dataclasses.replace(BASE, friendly_robber=True), dataclasses.replace(BASE, resource_bank=(24,) * 5),
                   load_game("base-3p")):
        with pytest.raises(SnapshotError):
            check_rules(config.cppanatron_game(map_seed=1, seed=2))
    check_rules(BASE.cppanatron_game(map_seed=1, seed=2))


def test_action_ids_cover_every_kind_and_map_back_one_to_one(sample):
    kinds = Counter()
    for game, _seat, _tracker, _history, board, _previous in sample:
        by_id, unexpressed = legal_by_id(board, game)
        assert not unexpressed  # base-2p without trading agents: every engine action has an id
        for aid, actions in by_id.items():
            assert 0 <= aid < NUM_ACTIONS and len(actions) == 1, (aid, actions)
            assert action_id(board, actions[0]) == aid
            kinds[kind_of(aid)] += 1
    assert set(kinds) == set(KIND_STARTS), kinds
    for aid in range(NUM_ACTIONS):
        assert action_name(sample[0][4], aid)


def test_robber_and_maritime_ids_follow_canopy(sample):
    seen = Counter()
    for game, _seat, _tracker, _history, board, _previous in sample:
        for action in game.playable_actions:
            aid = action_id(board, action)
            if action.action_type == AT.MOVE_ROBBER:
                assert board.tile_coords[aid - 205] == tuple(action.value[0])
                seen["robber"] += 1
            elif action.action_type == AT.MARITIME_TRADE:
                give, get = RESOURCES.index(action.value[0]), RESOURCES.index(action.value[-1])
                assert divmod(aid - 229, 4) == (give, get if get < give else get - 1)
                seen["maritime"] += 1
            elif action.action_type == AT.PLAY_YEAR_OF_PLENTY:
                assert action_name(board, aid) == "year_of_plenty({},{})".format(
                    *sorted((r.name for r in action.value), key=lambda n: [r.name for r in RESOURCES].index(n)))
                seen["yop"] += 1
    assert min(seen.values()) > 0 and len(seen) == 3


# ---------------------------------------------------------------- the snapshot


def test_snapshot_reads_back_to_the_position(sample):
    phases = Counter()
    for game, seat, tracker, history, board, _previous in sample:
        snap = _snap(game, seat, tracker, history, board)
        opp = 1 - seat
        assert snap["current_player"] == seat and snap["port_set"] == "primary"
        assert snap["terrains"] == list(board.terrains) and snap["numbers"] == list(board.numbers)
        for s in range(2):
            p, mine = game.player(s), snap["players"][s]
            assert {board.canopy_to_node[n] for n in snap["settlements"][s]} == set(p.settlements)
            assert {board.canopy_to_node[n] for n in snap["cities"][s]} == set(p.cities)
            assert {board.canopy_to_edge[e] for e in snap["roads"][s]} == {tuple(sorted(r)) for r in p.roads}
            assert mine["hand"] == list(p.resources)  # 2p: the opponent's hand is public, the deal is the truth
            played = [mine["dev_cards_played"][DEV_TO_CANOPY[k]] for k in range(4)]
            assert played == list(p.played_development_cards[:4]) and mine["knights_played"] == played[0]
            assert mine["building_vps"] == len(p.settlements) + 2 * len(p.cities)
            assert mine["roads_left"] == p.roads_available and mine["roads_placed"] == 15 - p.roads_available
        me, them = snap["players"][seat], snap["players"][opp]
        assert [me["dev_cards"][DEV_TO_CANOPY[k]] for k in range(5)] == list(game.player(seat).development_cards)
        assert them["dev_cards"] == [0] * 5 and them["dev_cards_bought_this_turn"] == [0] * 5
        assert them["hidden_dev_cards"] == sum(game.player(opp).development_cards) and me["hidden_dev_cards"] == 0
        # Victory points: buildings + own VP cards + awards are the seat's actual points.
        awards = 2 * ((snap["longest_road"] or {}).get("player") == seat) + 2 * ((snap["largest_army"] or {}).get(
            "player") == seat)
        assert me["building_vps"] + me["dev_cards"][1] + awards == game.player(seat).actual_victory_points
        # The bridge's own checks: the bank and the deck add up, the pieces are the board's.
        for r in range(5):
            assert snap["bank"][r] + me["hand"][r] + them["hand"][r] == 19
        used = sum(sum(p["dev_cards"]) + p["hidden_dev_cards"] + sum(p["dev_cards_played"]) for p in snap["players"])
        assert used + snap["dev_deck_total"] == sum(BASE_DECK)
        assert all(b <= h for b, h in zip(me["dev_cards_bought_this_turn"], me["dev_cards"]))
        assert board.tile_coords[snap["robber"]] == tuple(game.board.robber_coordinate)
        # Trade ratios agree with the engine's maritime offers.
        for action in game.playable_actions:
            if action.action_type == AT.MARITIME_TRADE:
                give = RESOURCES.index(action.value[0])
                assert me["trade_ratios"][give] == sum(1 for v in action.value[:-1] if v is not None)
        phase = snap["phase"]
        phases[phase["kind"]] += 1
        prompt = game.current_prompt
        expected = {cpp.ActionPrompt.BUILD_INITIAL_SETTLEMENT: "place_settlement",
                    cpp.ActionPrompt.BUILD_INITIAL_ROAD: "place_road", cpp.ActionPrompt.DISCARD: "discard",
                    cpp.ActionPrompt.MOVE_ROBBER: "move_robber"}.get(prompt)
        if expected is None:
            expected = "road_building" if game.is_road_building else (
                "main" if game.player(seat).has_rolled else "pre_roll")
        assert phase["kind"] == expected
        if expected == "discard":
            assert phase["remaining"] == game.remaining_discards(seat) and phase["roller"] == game.current_turn_index
        if expected == "place_road":
            node = board.canopy_to_node[snap["last_setup_node"]]
            assert node in game.player(seat).settlements
            assert not any(node in r for r in game.player(seat).roads)  # the settlement the road must touch
        else:
            assert snap["last_setup_node"] is None
        assert snap["turn_number"] == game.completed_turns
        canon.canonical_json(snap)  # plain JSON
    assert set(phases) == {"place_settlement", "place_road", "pre_roll", "main", "discard", "move_robber",
                           "road_building"}, phases


def test_own_purchases_this_turn_are_remembered(sample):
    bought = 0
    for game, seat, tracker, history, board, _previous in sample:
        me = _snap(game, seat, tracker, history, board)["players"][seat]
        assert sum(me["dev_cards_bought_this_turn"]) == tracker.bought_this_turn(seat)
        bought += sum(me["dev_cards_bought_this_turn"])
        assert not history.mismatches
    assert bought > 0


def _redealt(game: cpp.Game, seat: int, rng: random.Random) -> cpp.Game | None:
    """``game`` with the opponent's unplayed development cards swapped for another deal of the
    unseen cards (same count, same victory point cards), or None if there is no other deal."""
    opp = 1 - seat
    held = list(game.player(opp).development_cards)
    if sum(held[:4]) == 0:
        return None
    played = [sum(game.player(s).played_development_cards[k] for s in range(2)) for k in range(5)]
    unseen = [BASE_DECK[k] - game.player(seat).development_cards[k] - played[k] for k in range(5)]
    pool = [k for k in range(4) for _ in range(unseen[k])]
    for _ in range(50):
        pick = rng.sample(pool, sum(held[:4]))
        new = [pick.count(k) for k in range(4)] + [held[4]]
        if new != held:
            break
    else:
        return None
    hands = []
    for s in range(2):
        p = game.player(s)
        devs = new if s == opp else list(p.development_cards)
        owned = tuple(p.development_card_owned_at_start) if s == seat else tuple(devs[k] > 0 for k in range(4))
        hands.append(cpp.HiddenHand(resources=list(p.resources), development_cards=devs,
                                    development_card_owned_at_start=list(owned)))
    out = game.copy()
    out.redeal_hidden_cards(hands, seed=rng.randrange(1 << 30))
    assert list(out.player(opp).development_cards) == new
    return out


def test_snapshot_never_reads_the_opponents_hidden_cards():
    """Fairness: redeal the opponent's unplayed cards (and the deck and the random stream):
    every snapshot keeps its bytes."""
    rng = random.Random(5)
    audited = 0
    for game, seat, tracker, history, board, _previous in positions(games=6, seed=3):
        other = _redealt(game, seat, rng)
        if other is None:
            continue
        a = canon.canonical_json(_snap(game, seat, tracker, history, board, seed=9))
        b = canon.canonical_json(_snap(other, seat, tracker, history, board, seed=9))
        assert a == b
        audited += 1
    assert audited >= 20, audited


def test_chains_replay_the_seats_own_turns(sample):
    """Consecutive decisions with only the seat's move (and forced moves) between them replay to
    the second position; a decision after the opponent decided does not."""
    replayed = refused = 0
    for game, seat, _tracker, _history, board, previous in sample:
        if previous is None:
            continue
        before, action = previous
        tree = replay_chain(board, before, action, game, seat)
        same_turn = before.num_turns == game.num_turns and before.current_player_index == seat
        if tree is not None:
            replayed += 1
            assert tree[0] == action_id(board, action)
        elif same_turn and action.action_type not in (AT.END_TURN,):
            refused += 1
    assert replayed > 100 and refused <= replayed // 20, (replayed, refused)


def _dealt(game: cpp.Game, seat: int, cards: list[int], seed: int) -> cpp.Game | None:
    """``game`` with the opponent's unplayed development cards dealt as ``cards`` (engine order),
    every other hidden card kept, the deck and random stream reshuffled by ``seed``; None if the
    unseen cards cannot make ``cards``."""
    hands = []
    for s in range(2):
        p = game.player(s)
        devs = cards if s != seat else list(p.development_cards)
        owned = tuple(p.development_card_owned_at_start) if s == seat else tuple(devs[k] > 0 for k in range(4))
        hands.append(cpp.HiddenHand(resources=list(p.resources), development_cards=devs,
                                    development_card_owned_at_start=list(owned)))
    out = game.copy()
    try:
        out.redeal_hidden_cards(hands, seed=seed)
    except ValueError:
        return None
    return out


def test_whether_a_chain_continues_never_depends_on_the_opponents_hidden_cards():
    """The seat ends its turn while the opponent holds old development cards; the game then goes on
    with the same public moves, forced in the world where those cards are victory points (the
    opponent's roll and end of turn), not where they are knights. Whether the seat's next search is
    re-rooted (and through what) must be the same in both worlds: the seat cannot tell them apart."""
    compared = 0
    for game, seat, _tracker, _history, board, _previous in positions(games=10, seed=6):
        end = next((a for a in game.playable_actions if a.action_type == AT.END_TURN), None)
        held = sum(game.player(1 - seat).development_cards)
        if end is None or held == 0:
            continue
        worlds = [_dealt(game, seat, [0, 0, 0, 0, held], 7), _dealt(game, seat, [held, 0, 0, 0, 0], 7)]
        if None in worlds:
            continue
        a, b = worlds[0].copy(), worlds[1].copy()
        a.execute(end)
        b.execute(end)
        for _ in range(40):  # the same public moves in both, while they are forced in the first
            legal = list(a.playable_actions)
            if a.winning_player_index is not None or (int(a.current_player_index) != seat and len(legal) != 1):
                a = None
                break
            if int(a.current_player_index) == seat and len(legal) > 1:
                break
            try:
                b.execute(legal[0])
            except ValueError:
                a = None
                break
            a.execute(legal[0])
        if a is None or int(a.current_player_index) != seat:
            continue
        # The same public position in both (the dice came from the same reseeded stream).
        assert [list(w.player(s).resources) for w in (a, b) for s in range(2)][:2] == \
            [list(b.player(s).resources) for s in range(2)] and a.num_turns == b.num_turns
        chains = [replay_chain(board, worlds[0], end, a, seat), replay_chain(board, worlds[1], end, b, seat)]
        assert chains[0] == chains[1], chains
        compared += 1
    assert compared >= 10, compared


# ---------------------------------------------------------------- the agent, with a stand-in pool


class FakePool:
    """Stands in for the bridge: a seeded random choice among the engine's ids, with injected
    choices the engine does not offer: every ``bad``-th seed (a fallback, with a legal second
    choice in the policy) and every ``stale``-th seed of a re-rooted search (refreshed)."""

    def __init__(self, procs: int = 1, bad: int = 0, stale: int = 0) -> None:
        self.procs, self.bad, self.stale = procs, bad, stale
        self.states: list[dict] = []
        self.unoffered: list[bool] = []  # per unoffered answer: whether it claimed a re-rooted search

    def __len__(self) -> int:
        return self.procs

    def decide(self, batches, active, *, sims, legal=None):
        out = []
        for p, (states, offered_ids) in enumerate(zip(batches, legal)):
            if not states:
                out.append(None)
                continue
            assert {s["game_id"] for s in states} <= set(active[p])
            actions, details = [], []
            for snap, offered in zip(states, offered_ids):
                self.states.append(snap)
                seed = snap["seed"]
                choice = offered[random.Random(seed).randrange(len(offered))]
                policy = [[a, 1.0 / len(offered)] for a in offered]
                canopy_legal = list(offered)
                outside = next(a for a in range(NUM_ACTIONS) if a not in offered)
                if self.stale and snap["reuse_tree"] and seed % self.stale == 0:
                    choice, policy, canopy_legal = outside, [[outside, 1.0]], [outside]
                elif self.bad and seed % self.bad == 0:
                    choice, policy, canopy_legal = outside, [[outside, 0.9], [offered[-1], 0.1]], [*offered, outside]
                if choice == outside:
                    self.unoffered.append(bool(snap["reuse_tree"]))
                actions.append(choice)
                details.append({"mode": "search" if len(canopy_legal) > 1 else "forced",
                                "reuse": "reused" if snap["reuse_tree"] else "fresh", "canopy_legal": canopy_legal,
                                "policy": policy, "root_visits": sims,
                                # a re-rooted search lands on a node with visits for every other seed
                                "root_visits_before": sims // 2 if snap["reuse_tree"] and seed % 2 else 0})
            out.append({"actions": actions, "details": details, "elapsed_ms": 2.0})
        return out

    def close(self) -> None:
        pass


def _play(pool: FakePool, *, num_envs: int, boards: int = 2, params: CanopyParams | None = None):
    suite = Suite(name="t", purpose="dev", seed=11, boards=boards, scenarios=({"game": "base-2p"},), turn_limit=500)
    games = suite.schedule(2)
    canopy = CanopyAgent(params or CanopyParams(sims=8, procs=pool.procs), None, pool_factory=lambda: pool)
    slots = [canopy, BotAgent("F")]
    rows: list[dict] = []
    play_schedule(slots, games, suite=suite, num_envs=num_envs, threads=1, device="cpu",
                  on_game=results._with_canopy(slots, games, rows.append))
    return sorted(rows, key=lambda r: r["game_id"]), canopy


def _outcome(row: dict) -> dict:
    timing = ("bridge_s", "wall_s")
    canopy = [None if c is None else {k: v for k, v in c.items() if k not in timing} for c in row["canopy"]]
    return {**{k: v for k, v in row.items() if k != "canopy"}, "canopy": canopy}


def test_agent_plays_the_harness_deterministically_and_counts_fallbacks():
    a_pool, b_pool = FakePool(procs=1, bad=37, stale=5), FakePool(procs=3, bad=37, stale=5)
    a_rows, a_agent = _play(a_pool, num_envs=4)
    b_rows, _ = _play(b_pool, num_envs=1)
    assert [_outcome(r) for r in a_rows] == [_outcome(r) for r in b_rows]
    stats = [r["canopy"][0] for r in a_rows]
    assert all(r["canopy"][1] is None for r in a_rows)
    assert [s["decisions"] for s in stats] == [r["decisions"][r["seats"].index(0)] for r in a_rows]
    totals = summary_totals(stats)
    assert totals["refreshed"] == sum(a_pool.unoffered) > 0
    assert totals["fallbacks"] == sum(not u for u in a_pool.unoffered) > 0
    assert totals["reused"] > 0 and totals["fallback_examples"]
    assert 0 < totals["reused_visits"] < totals["reused"] and totals["retained_visits"] == 4 * totals["reused_visits"]
    assert totals["canopy_illegal"] > 0  # the injected choices were Canopy-legal and refused
    assert a_agent.pop_game(12345) is None and not a_agent._games  # every game's stats went to its row
    # Fairness at the boundary: no snapshot the pool saw carries the opponent's card identities.
    for snap in a_pool.states:
        opp = 1 - snap["current_player"]
        assert snap["players"][opp]["dev_cards"] == [0] * 5
        assert snap["players"][opp]["dev_cards_bought_this_turn"] == [0] * 5
    assert any(s["tree_actions"] for s in a_pool.states)


def test_reuse_false_never_sends_a_tree():
    pool = FakePool()
    _play(pool, num_envs=2, boards=1, params=CanopyParams(sims=8, reuse=False))
    assert pool.states and not any(s["reuse_tree"] for s in pool.states)


def test_trade_prompts_are_rejected_and_counted():
    """Canopy has no domestic trades: an offer to its seat is rejected and counted."""
    for game, seat, _tracker, _history, _board, _previous in positions(games=1, seed=4):
        mine, theirs = game.player(seat).resources, game.player(1 - seat).resources
        if game.current_prompt == cpp.ActionPrompt.PLAY_TURN and game.player(seat).has_rolled \
                and any(mine) and any(theirs):
            give = [int(i == next(k for k in range(5) if mine[k])) for i in range(5)]
            get = [int(i == next(k for k in range(5) if theirs[k])) for i in range(5)]
            game.execute(cpp.Action(game.colors[seat], AT.OFFER_TRADE, (*give, *get)))
            break
    assert game.current_prompt == cpp.ActionPrompt.DECIDE_TRADE
    agent = CanopyAgent(CanopyParams(), None, pool_factory=lambda: FakePool())
    [answer] = agent.respond_trade([game], [77], streams=[0], first=[True])
    assert answer.action_type == AT.REJECT_TRADE
    assert agent.pop_game(77)["trade_prompts"] == 1


# ---------------------------------------------------------------- resolution and provenance


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    release = Release(directory=tmp_path, binary=tmp_path / "canopy-bridge", checkpoint=tmp_path / "m.mpk",
                      binary_sha256="ab" * 32, manifest={"backend": "cpu-ndarray"})
    monkeypatch.setattr(canopy_agent, "load_release", lambda: release)
    return release


def test_specs_resolve_to_a_canonical_fixed_point(fake_release):
    agent = resolve("CANOPY:sims=64", games=["base-2p"])
    assert agent.canonical == ("CANOPY:sims=64:c_puct=2.5:c_visit=50.0:c_scale=1.0:dirichlet_alpha=0.05:"
                               "dirichlet_epsilon=0.0:reuse=true")
    assert resolve(agent.canonical).canonical == agent.canonical and resolve(agent).canonical == agent.canonical
    key = agent.key_dict()
    assert key["canopy"] == {"commit": CANOPY_COMMIT, "patches": [list(p) for p in PATCHES],
                             "binary_sha256": "ab" * 32, "checkpoint_sha256": CHECKPOINT_SHA256,
                             "backend": "cpu-ndarray"}
    assert resolve("CANOPY:sims=64:procs=4:timeout=60").key_dict() == key  # execution params are not keyed
    assert resolve("CANOPY:sims=0").params.sims == 0 and agent.fair and agent.checkpoint is None
    assert is_canopy_spec("canopy:sims=1") and is_canopy_spec({"key": "CANOPY"}) and is_canopy_spec(agent)
    assert not is_canopy_spec("POLICY:ckpt=x") and lineup_has_canopy(["F", key])


@pytest.mark.parametrize("spec,kwargs,match", [
    ("CANOPY:sims=-1", {}, "sims"),
    ("CANOPY:dirichlet_epsilon=2", {}, "dirichlet_epsilon"),
    ("CANOPY:c_puct=0", {}, "positive"),
    ("CANOPY:nope=1", {}, "no param"),
    ("CANOPY:sims=8", {"harness": "catanatron"}, "native harness"),
    ("CANOPY:sims=8", {"games": ["base-4p"]}, "2 players"),
])
def test_bad_specs_are_refused(fake_release, spec, kwargs, match):
    with pytest.raises(SpecError, match=match):
        resolve(spec, **kwargs)


def test_params_parse_like_specs():
    assert canopy_params([], {"sims": "400", "reuse": "false"}) == CanopyParams(sims=400, reuse=False)


def test_procs_env_sets_the_bridge_processes_and_no_key(fake_release, monkeypatch):
    agent = resolve("CANOPY:sims=8:procs=2", games=["base-2p"])
    assert agent.native().params.procs == 2
    monkeypatch.setenv("CATANRL_CANOPY_PROCS", "5")
    assert agent.native().params.procs == 5
    assert resolve(agent.canonical).key_dict() == agent.key_dict() == resolve("CANOPY:sims=8").key_dict()
    monkeypatch.setenv("CATANRL_CANOPY_PROCS", "0")
    with pytest.raises(SpecError, match="CATANRL_CANOPY_PROCS"):
        agent.native()


THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party" / "canopy"
#: The committed manifest per backend of build_bridge.sh (--backend cpu | cuda).
COMMITTED = {"cpu-ndarray": MANIFEST_NAME, "cuda": "canopy-bridge-cuda.manifest.json"}


def test_the_committed_patches_and_manifests_are_the_pinned_builds():
    """third_party/canopy holds the pinned patches and, per backend, the manifest of the build
    they make (third_party/canopy/build_bridge.sh rebuilds it to the same binary): the same
    source, patches and weights; another binary."""
    for name, sha in PATCHES:
        assert canon.sha256_file(THIRD_PARTY / "patches" / name) == sha, name
    assert sorted(p.name for p in (THIRD_PARTY / "patches").iterdir()) == sorted(name for name, _ in PATCHES)
    binaries = set()
    for backend, file in COMMITTED.items():
        manifest = json.loads((THIRD_PARTY / file).read_text())
        assert manifest["commit"] == CANOPY_COMMIT and [tuple(p) for p in manifest["patches"]] == list(PATCHES)
        assert manifest["checkpoint_sha256"] == CHECKPOINT_SHA256 and len(manifest["binary_sha256"]) == 64
        assert manifest["backend"] == backend and manifest["tree"] == json.loads(
            (THIRD_PARTY / MANIFEST_NAME).read_text())["tree"]
        binaries.add(manifest["binary_sha256"])
    assert len(binaries) == len(COMMITTED)


def _release_dir(path: Path, backend: str, binary: bytes) -> Path:
    path.mkdir()
    (path / "canopy-bridge").write_bytes(binary)
    (path / "model_iter_315.mpk").write_bytes(b"weights")
    manifest = {"format": "catanrl.canopy_bridge_manifest/1", "commit": CANOPY_COMMIT, "backend": backend,
                "patches": [list(p) for p in PATCHES], "binary_sha256": canon.sha256_file(path / "canopy-bridge")}
    (path / "canopy-bridge.manifest.json").write_text(canon.canonical_json(manifest).decode())
    return path


def test_a_cuda_release_keys_apart_from_the_cpu_one(tmp_path, monkeypatch):
    """$CATANRL_CANOPY_DIR picks the build; the key names its binary and backend, so a CUDA
    build's records never mix with the CPU build's, and a CPU build keys as it always did."""
    monkeypatch.setattr("catanrl.canopy.release.CHECKPOINT_SHA256", canon.sha256_file(
        _release_dir(tmp_path / "cpu", "cpu-ndarray", b"cpu binary") / "model_iter_315.mpk"))
    _release_dir(tmp_path / "cuda", "cuda", b"cuda binary")
    keys = {}
    for backend in ("cpu", "cuda"):
        monkeypatch.setenv("CATANRL_CANOPY_DIR", str(tmp_path / backend))
        keys[backend] = load_release().key()
    differs = {k for k in keys["cpu"] if keys["cpu"][k] != keys["cuda"][k]}
    assert differs == {"binary_sha256", "backend"} and keys["cuda"]["backend"] == "cuda"
    assert set(keys["cpu"]) == {"commit", "patches", "binary_sha256", "checkpoint_sha256", "backend"}


def test_a_gpu_build_records_its_execution_and_a_cpu_build_nothing_new(monkeypatch):
    """The segment's execution gains a ``canopy`` entry (GPU, driver, NVRTC, toolkit) only for
    a GPU build: a CPU build's records resume under the execution they started with."""
    runtime = {"gpu": "G", "capability": "7.5", "driver": "1", "nvrtc": "13.2", "cuda_path": "/c", "toolkit": 13020}
    monkeypatch.setattr(results, "_cuda_runtime", lambda: dict(runtime))

    def agent(backend):
        if backend is None:
            return resolve("F")
        release = Release(directory=Path("."), binary=Path("b"), checkpoint=Path("c"), binary_sha256="ab" * 32,
                          manifest={"backend": backend})
        return canopy_agent.CanopyResolved(key="CANOPY", impl="catanrl.canopy", params=CanopyParams(),
                                           checkpoint=None, fair=True, label="CANOPY", canonical="CANOPY",
                                           release=release)

    assert results._canopy_execution([agent("cpu-ndarray"), agent(None), None]) == {}
    assert results._canopy_execution([agent(None), agent(None)]) == {}
    assert results._canopy_execution([agent("cuda"), agent(None), None]) == {"canopy": {"backend": "cuda", **runtime}}


@pytest.mark.parametrize("args,match", [(["--backend", "metal", "s", "o", "c"], "cpu or cuda"),
                                        (["--backend", "cuda", "s", "o"], "build_bridge.sh")])
def test_the_build_script_takes_a_backend(args, match):
    import subprocess

    done = subprocess.run(["bash", str(THIRD_PARTY / "build_bridge.sh"), *args], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 2 and match in done.stderr


def test_release_refuses_a_directory_that_is_not_the_pinned_build(tmp_path):
    with pytest.raises(ReleaseError, match="missing"):
        load_release(tmp_path)
    (tmp_path / "canopy-bridge").write_bytes(b"binary")
    (tmp_path / "model_iter_315.mpk").write_bytes(b"weights")
    manifest = {"format": "catanrl.canopy_bridge_manifest/1", "commit": CANOPY_COMMIT,
                "patches": [list(p) for p in PATCHES], "binary_sha256": canon.sha256_file(tmp_path / "canopy-bridge")}
    (tmp_path / "canopy-bridge.manifest.json").write_text(canon.canonical_json(manifest).decode())
    with pytest.raises(ReleaseError, match="not the released"):
        load_release(tmp_path)
    (tmp_path / "canopy-bridge.manifest.json").write_text(canon.canonical_json(
        {**manifest, "patches": manifest["patches"][:2]}).decode())
    with pytest.raises(ReleaseError, match="patches"):
        load_release(tmp_path)


def test_code_ids_without_canopy_are_unchanged(fake_release):
    from catanrl import provenance

    plain = provenance.code_id("eval", "native")
    with_canopy = provenance.code_id("eval", "native", extras=("canopy",))
    assert set(plain) == {"catanrl", "engine"} and with_canopy == {**plain, "canopy": with_canopy["canopy"]}
    labels = [label for label, _ in provenance._outcome_files("native")]
    assert not any(label.startswith("canopy") for label in labels)  # the adapter is no other record's code
    suite = Suite(name="t", purpose="dev", seed=1, boards=1, scenarios=({"game": "base-2p"},), turn_limit=500)
    canopy, bot = resolve("CANOPY:sims=8", games=["base-2p"]), resolve("F")
    assert results.eval_key([canopy, bot], suite, harness="native")["code"] == with_canopy
    key = results.eval_key([bot, bot], suite, harness="native")
    assert key["code"] == plain and results.key_code_id(key) == plain
    assert results.key_code_id(results.eval_key([bot, canopy], suite, harness="native")) == with_canopy


def _record(tmp_path: Path, name: str, lineup: list, code: dict, winners: list) -> results.EvalRecord:
    path = tmp_path / f"{name}.json"
    rows = [{"game_id": f"g{i}", "winner": w, "seats": [0, 1]} for i, w in enumerate(winners)]
    path.with_suffix(".games.jsonl").write_text("".join(canon.canonical_json(r).decode() + "\n" for r in rows))
    key = {"harness": "native", "determinism": "strict", "code": code, "lineup": lineup, "fill": None,
           "suite": {"schedule_hash": "s"}}
    return results.EvalRecord(eval_id=name, key=key, summary={}, path=path, data={})


def test_paired_ignores_the_canopy_component_of_the_slot_that_differs(tmp_path):
    base = {"catanrl": "c", "engine": "e"}
    canopy, policy, bot = {"key": "CANOPY"}, {"key": "POLICY"}, {"key": "F"}
    a = _record(tmp_path, "a", [canopy, bot], {**base, "canopy": "x"}, [0, 1, 0, 0])
    b = _record(tmp_path, "b", [policy, bot], base, [1, 1, 0, 1])
    assert results.paired(a, b)["a"] == "a"
    c = _record(tmp_path, "c", [canopy, bot], {**base, "canopy": "y"}, [0, 1, 0, 0])
    with pytest.raises(ValueError, match="code"):
        results.paired(a, c)
    d = _record(tmp_path, "d", [policy, bot], {**base, "catanrl": "other"}, [0, 1, 0, 0])
    with pytest.raises(ValueError, match="code"):
        results.paired(a, d)


# ---------------------------------------------------------------- the real bridge (optional)


def _real_release():
    if not os.environ.get("CATANRL_CANOPY_DIR"):
        pytest.skip("CATANRL_CANOPY_DIR names no Canopy release")
    return load_release()


def test_real_bridge_accepts_every_position_and_stays_inside_the_engines_rules():
    from catanrl.canopy.bridge import BridgeError, BridgeProcess

    release = _real_release()
    process = BridgeProcess(release, sims=0, c_puct=2.5, c_visit=50.0, c_scale=1.0, dirichlet_alpha=0.05,
                            dirichlet_epsilon=0.0, reuse=True)
    try:
        refused = Counter()
        count = 0
        for game, seat, tracker, history, board, _previous in positions(games=2, seed=8):
            snap = _snap(game, seat, tracker, history, board, seed=count)
            answer = process.request({"active_game_ids": ["g"], "states": [snap], "simulations": 0, "details": True})
            by_id, _ = legal_by_id(board, game)
            assert answer["actions"][0] in by_id
            refused.update(kind_of(a) for a in answer["details"][0]["canopy_legal"] if a not in by_id)
            count += 1
        assert count > 100 and not refused, refused
        # The bridge refuses a snapshot carrying the opponent's card identities.
        leaky = copy.deepcopy(snap)
        leaky["players"][1 - snap["current_player"]]["dev_cards"] = [1, 0, 0, 0, 0]
        with pytest.raises(BridgeError, match="fairness"):
            process.request({"active_game_ids": ["g"], "states": [leaky], "simulations": 0})
    finally:
        process.close()


def test_real_bridge_is_deterministic_in_a_batch():
    from catanrl.canopy.bridge import BridgeProcess

    release = _real_release()
    process = BridgeProcess(release, sims=16, c_puct=2.5, c_visit=50.0, c_scale=1.0, dirichlet_alpha=0.05,
                            dirichlet_epsilon=0.0, reuse=False)
    try:
        snaps = [{**_snap(g, s, t, h, b, seed=i), "game_id": f"g{i}"}
                 for i, (g, s, t, h, b, _) in enumerate(positions(games=1, seed=9)) if i % 25 == 0][:6]
        ids = [s["game_id"] for s in snaps]
        alone = [process.request({"active_game_ids": ids, "states": [s], "details": True}) for s in snaps]
        batch = process.request({"active_game_ids": ids, "states": snaps, "details": True})
        assert [a["actions"][0] for a in alone] == batch["actions"]
        assert [a["details"][0]["policy"] for a in alone] == [d["policy"] for d in batch["details"]]
    finally:
        process.close()
