"""Game review: a finished site game analyzed move by move by a network's search.

The site's archive keeps every finished game (catanatron-server's
``catanatron.server.games``); the house analyzes the ones people played and
sends the analysis back to the server, which shows it with the game's
replay. :class:`Analyst` replays a record (every outcome forced from its
log) and, at every decision with a choice, searches the position the way an
MCTS bot of the site would have searched it from the mover's seat: the
mover's view (its own hand, everything public), its hand tracker's worlds
for the other hands, never their true cards. The search's root gives

* the evaluation: every seat's win probability by the mover's search (the
  eval bar), in per mille, and
* the move's grade: the root value of the action played against the action
  the search prefers (its most visited), as a loss in win probability.

Domestic trades (offers, answers, confirmations) are outside the network's
action space: they are replayed but not graded. A forced move (one legal
action) is not searched.

The search is Gumbel's (``root=gumbel``) with ``gumbel_m`` covering the legal
actions, so every one of them is visited and the action played has a value
whatever its prior: a PUCT root leaves a move the network dislikes
unvisited, which is exactly the move a review is for.

The mover's information, not hindsight: the hidden hands are the worlds of the mover's
tracker, so a move is judged on what its player could know. With ``world_choice=likely``
those are the ``dets`` most likely deals, each weighted by its probability, not a couple
of random draws; a position with too many deals to list (some 3- and 4-player ones) is
searched on sampled worlds. A move graded ``inaccuracy`` or
worse is searched again with ``recheck`` simulations (2048), and that grade stands: a
single small search can call a fine move a blunder.

The result is integers only (the lobby takes no floats)::

    {"version": 2,
     "engine": {"label": ..., "spec": ..., "recheck": simulations},
     "evals": [[index, [per-mille win probability per seat, in state.colors order]], ...],
     "moves": [{"i": index, "c": color, "p": played, "b": best, "a": best action,
                "g": grade, "n": legal actions, "s": stakes, "d": recheck simulations (if any),
                "top": [[action, value, visits], ...]}, ...],
     "summary": {"label": ..., "accuracy": {color: percent}, "grades": {color: {grade: n}},
                 "decisions": n, "rechecked": n}}

``index`` is the number of log entries before the decision (the replay's
frame). A Catan move moves the odds far less than a chess move (most cost
under a point of win chance), so the scale is chess.com's on losses four
times as large: grades by the loss in percentage points, ``best`` (the
search's own choice), ``excellent`` (< 0.5), ``good`` (< 1.25),
``inaccuracy`` (< 2.5), ``mistake`` (< 5), ``blunder``; accuracy the mean
per move of chess.com's curve ``103.17 exp(-0.04354 x) - 3.17`` at
``x = 4 loss``, each move weighted by its decision's stakes (``s``: the best searched
action's win chance over the mean of the searched actions', capped at 10 points), so a
choice between equal options, or any move in a decided position, counts for next to
nothing. (Preliminary: set from the first games' losses.)

Speed: a game's decisions are independent, so they are searched together (a
chunk at a time), each forward carrying a leaf of every search; and on a CPU
the network runs in ONNX Runtime when it is installed (on the bot VM's ARM
cores 2.5 ms a row against torch's 8 ms, one row per call). The search
itself is ~1% of the time.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
from catanatron.serialization import action_to_json, client_view
from catanatron.server.games import rebuild
from cppanatron.catanatron import to_catanatron_action

from catanrl.bridge.agent import ViewAgent, action_key

logger = logging.getLogger(__name__)

VERSION = 2
TRADE_TYPES = {"OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE"}
TRADE_PROMPTS = {"DECIDE_TRADE", "DECIDE_ACCEPTEES"}
#: (grade, the loss in percentage points it stays under).
GRADES = (("excellent", 0.5), ("good", 1.25), ("inaccuracy", 2.5), ("mistake", 5.0))
#: Catan losses in chess.com's units (see the module doc).
SCALE = 4.0
TOP = 3


def grade(loss_pp: float, best: bool) -> str:
    if best:
        return "best"
    for name, limit in GRADES:
        if loss_pp < limit:
            return name
    return "blunder"


def accuracy(losses_pp: Sequence[float], weights: Sequence[float] | None = None) -> int | None:
    """The mean of chess.com's per-move score (on losses ``SCALE`` times as large), each
    move weighted by ``weights`` (its decision's stakes) when given."""
    if not losses_pp:
        return None
    scores = [min(100.0, max(0.0, 103.1668 * math.exp(-0.04354 * SCALE * loss) - 3.1669)) for loss in losses_pp]
    if weights is not None and sum(weights) > 0:
        return int(round(sum(w * x for w, x in zip(weights, scores)) / sum(weights)))
    return int(round(sum(scores) / len(scores)))


def stakes(actions: Mapping[str, Sequence[Any]], seats: int) -> float:
    """What the choice was worth, in percentage points of win chance: the best searched
    action's value over the mean of every searched action's. Near zero where the options
    are alike, or the position is decided (won or lost, nothing moves the odds)."""
    values = [_win(q, seats) for _, q, visits in actions.values() if visits > 0 and math.isfinite(q)]
    if len(values) < 2:
        return 0.0
    return max(0.0, max(values) - sum(values) / len(values)) * 100.0


#: The most a decision weighs in a seat's accuracy (percentage points of stakes).
MAX_WEIGHT = 10.0


def _key(action: Sequence[Any]) -> str:
    """The legal action a logged one is (the log names a bought card and a roll's dice)."""
    color, kind, value = action
    if kind in ("BUY_DEVELOPMENT_CARD", "ROLL"):
        value = None
    return action_key([color, kind, value])


class _Searcher(ViewAgent):
    """A ViewAgent whose MCTS decisions are queued, not searched: :meth:`_act` adds the
    position to the pool (its worlds are dealt there and then, from the seat's tracker as
    it stands) and :meth:`flush` runs every queued search at once, so each forward carries
    a leaf of every search (a game's decisions are independent: searched together, the
    network sees batches instead of one row per call)."""

    queued: dict | None = None
    _onnx: Any = None
    #: The simulations of the searches added now (None: the spec's).
    sims: int | None = None
    #: The spec takes the likely worlds: a position with too many deals to list is refused
    #: (``refused``), for the analyst's sampling searcher.
    likely: bool = False
    refused: bool = False

    def _act(self, row: np.ndarray, mask: np.ndarray, seed: int, game: Any, tracker: Any) -> int:
        native = self.native
        if self._onnx is not None:
            threads, self._onnx = self._onnx, None
            try:
                _use_onnx(native, threads)
            except Exception:
                logger.exception("ONNX Runtime could not run the network: torch runs it")
        trackers = tracker.tracker if native.needs_trackers else None
        game = game.copy()
        try:
            sid = native.pool.add(game, seed=int(seed), tracker=trackers,
                                  deal=native.tracker_mode if trackers is not None else None, simulations=self.sims)
        except ValueError as error:
            if not (self.likely and "likely worlds need" in str(error)):
                raise
            self.refused = True
            return int(np.flatnonzero(mask)[0])
        self.queued = {"sid": sid, "game": game, "mask": mask.copy(), "seat": int(tracker.seat),
                       "colors": list(tracker.colors)}
        return int(np.flatnonzero(mask)[0])  # (the replay plays the logged move whatever this is)

    def flush(self, queued: Sequence[dict]) -> list[dict]:
        """Search the queued positions together: per position, the root's statistics."""
        from catanrl.search.pool import run_searches

        native = self.native
        pool = native.pool
        try:
            cache = getattr(native, "cache", None)  # (agents from before the engine's evaluation cache have none)
            if cache is not None:
                cache.sync(native.evaluator)
            run_searches(pool, native.evaluator, batch_rows=max(1, len(queued)) * native.config.max_pending)
            results = []
            for item in queued:
                sid, game, mask = item["sid"], item["game"], item["mask"]
                q = np.asarray(pool.q(sid), dtype=np.float64)
                visits = np.asarray(pool.visits(sid), dtype=np.int64)
                value = np.asarray(pool.value(sid), dtype=np.float64)
                space = native.action_space
                legal = np.flatnonzero(mask)
                actions = {}
                for index in legal:
                    chosen = action_to_json(to_catanatron_action(space.decode(game, int(index))))
                    actions[action_key(chosen)] = (json.loads(json.dumps(chosen)), float(q[index]),
                                                   int(visits[index]))
                results.append({"actions": actions, "value": value, "seat": item["seat"],
                                "colors": item["colors"]})
            return results
        finally:
            if pool.pending:
                pool.cancel()
            for item in queued:
                pool.remove(item["sid"])


# ===== the network in ONNX Runtime =====
class OnnxEvaluator:
    """A network's (logits, values) by ONNX Runtime, for a CPU analyst: several times
    torch's speed on a small batch, the same outputs to float rounding. Exported from the
    torch evaluator's model at start (``torch.onnx.export``), and its first batch is held
    against torch's. ``model`` stays the torch module (the engine's evaluation cache keys
    on its weights)."""

    def __init__(self, evaluator: Any, *, threads: int = 2, chunk: int = 256, tolerance: float = 1e-3) -> None:
        import io

        import onnxruntime as ort
        import torch

        self.torch = evaluator
        self.model = evaluator.model
        self.obs_size = int(evaluator.obs_size)
        self.num_actions = int(evaluator.num_actions)
        self.num_players = int(evaluator.num_players)
        self.recurrent = False
        self.device = torch.device("cpu")
        self.mode = "fast"
        self.infer_rows = int(chunk)
        self.chunk = int(chunk)
        self.tolerance = float(tolerance)
        self.checked = False
        self.rows = 0
        self.forwards = 0

        class Heads(torch.nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model

            def forward(self, obs: Any) -> Any:
                logits, values = self.model(obs)
                return logits, values

        model = self.model.to("cpu").eval()
        buffer = io.BytesIO()
        with torch.inference_mode(False), torch.no_grad():
            torch.onnx.export(Heads(model), (torch.zeros(2, self.obs_size),), buffer, input_names=["obs"],
                              output_names=["logits", "values"], opset_version=18, dynamo=False,
                              dynamic_axes={"obs": {0: "n"}, "logits": {0: "n"}, "values": {0: "n"}})
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # No spinning between calls: the analyst shares its cores with the house's bots.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(buffer.getvalue(), options, providers=["CPUExecutionProvider"])

    def __call__(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observations = np.ascontiguousarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.obs_size:
            raise ValueError(f"expected rows [n, {self.obs_size}], got {observations.shape}")
        n = len(observations)
        logits = np.empty((n, self.num_actions), dtype=np.float32)
        values = np.empty((n, self.num_players), dtype=np.float32)
        for start in range(0, n, self.chunk):
            chunk = observations[start:start + self.chunk]
            out_logits, out_values = self.session.run(None, {"obs": chunk})
            logits[start:start + len(chunk)] = out_logits
            values[start:start + len(chunk)] = out_values
            self.forwards += 1
        self.rows += n
        if not self.checked and n:
            self._check(observations[:8], logits[:8], values[:8])
        return logits, values

    def _check(self, rows: np.ndarray, logits: np.ndarray, values: np.ndarray) -> None:
        """The first rows against torch's outputs: a bad export stops the analyst, not a review."""
        expected_logits, expected_values = self.torch(rows)
        finite = np.isfinite(expected_logits) & np.isfinite(logits)
        if not np.array_equal(np.isfinite(expected_logits), np.isfinite(logits)):
            raise RuntimeError("ONNX Runtime masks other actions than torch")
        gap = max(float(np.abs(np.where(finite, expected_logits - logits, 0.0)).max(initial=0.0)),
                  float(np.abs(expected_values - values).max(initial=0.0)))
        if gap > self.tolerance:
            raise RuntimeError(f"ONNX Runtime's outputs differ from torch's by {gap:.2e}")
        self.checked = True
        logger.info("ONNX Runtime matches torch to %.1e", gap)


def _use_onnx(native: Any, threads: int) -> None:
    """Swap the network under a native agent's evaluator (inside its dedup wrapper, if any)
    for ONNX Runtime."""
    outer = native.evaluator
    if hasattr(outer, "evaluator"):
        outer.evaluator = OnnxEvaluator(outer.evaluator, threads=threads)
    else:
        native.evaluator = OnnxEvaluator(outer, threads=threads)


def onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


class Analyst:
    """Analyzes archive records with one search spec (an ``MCTS:`` spec;
    ``root=gumbel`` recommended, see the module doc).

    ``onnx``: run the network in ONNX Runtime with that many threads (None: torch; the
    default when onnxruntime is installed). ``chunk``: decisions searched together.
    With ``world_choice=likely`` in the spec, a position with too many deals to list is
    searched on sampled worlds instead (the same spec without it). ``recheck``: the
    simulations of a second, deeper search of every move graded ``recheck_from`` or
    worse, whose grade replaces the first (None: no second search)."""

    def __init__(self, spec: Any, *, label: str = "", device: str = "cpu", infer_rows: int = 256,
                 onnx: int | None | str = "auto", chunk: int = 128, recheck: int | None = 2048,
                 recheck_from: str = "inaccuracy") -> None:
        if isinstance(spec, str) and not spec.startswith("MCTS:"):
            raise ValueError("an analysis searches: give an MCTS spec")
        self.spec = spec if isinstance(spec, str) else "MCTS"
        self.label = label or self.spec
        self.chunk = int(chunk)
        self.recheck = int(recheck) if recheck else None
        names = [name for name, _ in GRADES] + ["blunder"]
        if recheck_from not in names:
            raise ValueError(f"recheck_from is one of {names}")
        self.recheck_from = names.index(recheck_from)
        if onnx == "auto":
            onnx = 2 if device == "cpu" and onnx_available() else None
        # mode=fast: rows are batched (a strict CPU evaluator forwards them one at a time).
        self.agent = self._searcher(spec, device, infer_rows, onnx)
        self.fallback: _Searcher | None = None
        if isinstance(spec, str) and ":world_choice=likely" in spec:
            self.agent.likely = True
            sampled = spec.replace(":world_choice=likely", "")
            self._fallback_args = (sampled, device, infer_rows, onnx)

    @staticmethod
    def _searcher(spec: Any, device: str, infer_rows: int, onnx: Any) -> _Searcher:
        searcher = _Searcher(spec, device=device, infer_rows=infer_rows, mode="fast", max_games=8)
        searcher._onnx = onnx
        return searcher

    def _sampler(self) -> _Searcher:
        """The searcher of positions the likely worlds cannot list (built at first need)."""
        if self.fallback is None:
            self.fallback = self._searcher(*self._fallback_args)
        return self.fallback

    def analyze(self, record: Mapping[str, Any], progress: Callable[[int, int], None] | None = None) -> dict:
        playable: list = []
        docs = rebuild(dict(record), playable)
        log = record["final_state"]["action_records"]
        game_id = record["final_state"]["game"]["id"]
        supply = (record["room"].get("config") or {}).get("supply")
        for searcher in (self.agent, self.fallback):
            if searcher is not None:
                searcher.supply = supply
        self._supply = supply
        started = time.monotonic()
        decisions = [i for i, (doc, legal) in enumerate(zip(docs, playable)) if self._decision(doc, legal, log[i][0])]
        flagged_estimate = len(decisions) // 8 if self.recheck else 0
        total = len(decisions) + flagged_estimate

        def first(done: int) -> None:
            if progress is not None:
                progress(done, total)

        graded = {}
        for index, last in self._pass(game_id, docs, playable, log, decisions, None, first).items():
            graded[index] = self._grade(docs[index], log[index][0], last)
        rechecked = 0
        if self.recheck:
            worst = [i for i, (_, move) in graded.items() if move is not None and self._rank(move["g"]) >= self.recheck_from]

            def second(done: int) -> None:
                if progress is not None:
                    progress(len(decisions) + done * flagged_estimate // max(1, len(worst)), total)

            for index, last in self._pass(game_id, docs, playable, log, worst, self.recheck, second).items():
                evaluation, move = self._grade(docs[index], log[index][0], last)
                if move is not None:
                    move["d"] = self.recheck
                graded[index] = (evaluation, move)
                rechecked += 1
        evals, moves = [], []
        losses: dict[str, list] = {}
        weights: dict[str, list] = {}
        grades: dict[str, Counter] = {}
        for index in sorted(graded):
            evaluation, move = graded[index]
            evals.append(evaluation)
            if move is None:
                continue
            moves.append(move)
            losses.setdefault(move["c"], []).append((move["b"] - move["p"]) / 10)
            weights.setdefault(move["c"], []).append(min(MAX_WEIGHT, move["s"] / 10))
            grades.setdefault(move["c"], Counter())[move["g"]] += 1
        summary = {
            "label": self.label,
            "accuracy": {c: accuracy(v, weights[c]) for c, v in losses.items()},
            "grades": {c: dict(v) for c, v in grades.items()},
            "decisions": len(moves),
            "rechecked": rechecked,
            "seconds": int(time.monotonic() - started),
        }
        return {"version": VERSION, "engine": {"label": self.label, "spec": self.spec, "recheck": self.recheck},
                "evals": evals, "moves": moves, "summary": summary}

    @staticmethod
    def _rank(name: str) -> int:
        return ([n for n, _ in GRADES] + ["blunder"]).index(name) if name != "best" else -1

    def _pass(self, game_id: str, docs: list, playable: list, log: list, indices: Sequence[int],
              sims: int | None, progress: Callable[[int], None]) -> dict:
        """Search the decisions at ``indices`` (``sims`` simulations, None: the spec's), a
        chunk at a time: {index: the root's statistics}."""
        results: dict[int, dict] = {}
        pending: list[tuple[int, _Searcher, dict]] = []
        done = 0

        def search(batch: list[tuple[int, _Searcher, dict]]) -> None:
            nonlocal done
            for searcher in {id(s): s for _, s, _ in batch}.values():
                part = [(index, queued) for index, s, queued in batch if s is searcher]
                try:
                    found = searcher.flush([queued for _, queued in part])
                except Exception:
                    logger.exception("game %s: the searches of actions %d to %d failed", game_id, part[0][0],
                                     part[-1][0])
                    continue
                for (index, _), last in zip(part, found):
                    results[index] = last
            done += len(batch)
            progress(done)

        for index in indices:
            action = log[index][0]
            legal = json.loads(json.dumps(playable[index]))
            view = client_view(docs[index], action[0])
            searcher = self.agent
            searcher.queued, searcher.refused, searcher.sims = None, False, sims
            try:
                searcher.decide(view, legal)
                if searcher.queued is None and searcher.refused:
                    searcher = self._sampler()
                    searcher.supply = self._supply
                    searcher.queued, searcher.sims = None, sims
                    searcher.decide(view, legal)
            except Exception:
                logger.exception("game %s: the position at action %d failed", game_id, index)
                continue
            if searcher.queued is None:  # a fallback (no position): nothing to search
                continue
            pending.append((index, searcher, searcher.queued))
            if len(pending) >= self.chunk:
                search(pending)
                pending = []
        if pending:
            search(pending)
        return results

    @staticmethod
    def _decision(doc: Mapping[str, Any], legal: Sequence[Any], action: Sequence[Any]) -> bool:
        """Whether the logged move was a choice the network makes (not forced, not a trade)."""
        return not (len(legal) < 2 or action[1] in TRADE_TYPES or doc["current_prompt"] in TRADE_PROMPTS
                    or action[0] != doc["colors"][doc["current_player_index"]])

    @staticmethod
    def _grade(doc: Mapping[str, Any], action: Sequence[Any], last: Mapping[str, Any]) -> tuple[list, dict | None]:
        """(the eval at the decision, the graded move or None)."""
        index = len(doc["action_records"])
        color = action[0]
        seats = len(doc["colors"])
        root, colors = last["seat"], last["colors"]
        value = last["value"]
        # Seat-relative to the root player -> the document's seat order.
        absolute = [0.0] * seats
        for k in range(seats):
            absolute[doc["colors"].index(_color(colors[(root + k) % seats]))] = float(value[k])
        evaluation = [index, _per_mille(absolute)]
        actions = last["actions"]
        played = actions.get(_key(action))
        ranked = sorted(actions.values(), key=lambda a: (-a[2], -_finite(a[1])))
        best_action, best_q, _ = ranked[0]
        if played is None or not math.isfinite(played[1]):
            return evaluation, None  # not in the action space, or never visited
        is_best = _key(action) == action_key(best_action)
        loss = 0.0 if is_best else max(0.0, _win(best_q, seats) - _win(played[1], seats)) * 100.0
        return evaluation, {
            # (The most visited action's value can sit a little under the one played.)
            "i": index, "c": color, "p": _pm(_win(played[1], seats)),
            "b": max(_pm(_win(best_q, seats)), _pm(_win(played[1], seats))),
            "a": best_action, "g": grade(loss, is_best), "n": len(actions),
            # The decision's stakes, per mille of win chance: its weight in accuracy.
            "s": int(round(10 * stakes(actions, seats))),
            "top": [[a, _pm(_win(q, seats)), v] for a, q, v in ranked[:TOP]],
        }


def _color(color: Any) -> str:
    return getattr(color, "value", None) or getattr(color, "name", None) or str(color)


def _finite(x: float) -> float:
    return x if math.isfinite(x) else -1e9


def _win(value: float, seats: int) -> float:
    """A root value as a win probability (the networks' values are win probabilities)."""
    return min(1.0, max(0.0, float(value)))


def _pm(p: float) -> int:
    return int(round(1000 * p))


def _per_mille(values: Sequence[float]) -> list:
    clipped = [max(0.0, float(v)) for v in values]
    total = sum(clipped) or 1.0
    return [int(round(1000 * v / total)) for v in clipped]


# ===== the service =====
def load_engines(path: str) -> tuple[dict, list[dict]]:
    """(settings, engines) from a YAML file: ``server``, ``poll_s`` and
    ``engines`` (each ``label``, ``spec`` and optional ``when``, a room
    config subset it applies to; the first that matches analyzes a game)."""
    from pathlib import Path

    import yaml

    data = yaml.safe_load(Path(path).read_text())
    engines = list(data.pop("engines"))
    for engine in engines:
        if not str(engine.get("spec", "")).startswith("MCTS:"):
            raise ValueError(f"engine {engine.get('label')}: an analysis searches (an MCTS spec)")
    return data, engines


def engine_for(engines: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for engine in engines:
        when = engine.get("when") or {}
        if all(config.get(k) == v for k, v in when.items()):
            return engine
    return None


class Service:
    """Polls the server's analysis queue, analyzes each game with the engine
    its room's config picks, sends the result back (``submit_analysis``) and
    logs one line per game to ``STATE/analyses.jsonl``."""

    def __init__(self, server: str, engines: Sequence[Mapping[str, Any]], *, state_dir: str = "analyst",
                 device: str = "cpu", poll_s: float = 120.0, house_token: str | None = None) -> None:
        import os

        self.server = server.rstrip("/")
        self.http = self.server.replace("wss://", "https://", 1).replace("ws://", "http://", 1)
        self.engines = list(engines)
        self.state_dir = state_dir
        self.device = device
        self.poll_s = float(poll_s)
        self.token = house_token or os.environ.get("CATANATRON_HOUSE_TOKEN", "")
        if not self.token:
            raise ValueError("the analyst needs the house token (CATANATRON_HOUSE_TOKEN)")
        self._analysts: dict[str, Analyst] = {}
        os.makedirs(state_dir, exist_ok=True)

    def _get(self, path: str) -> Any:
        import urllib.request

        request = urllib.request.Request(self.http + path, headers={"X-Catanatron-House": self.token,
                                                                      "User-Agent": "catanrl-analyst"})
        with urllib.request.urlopen(request, timeout=60) as reply:
            return json.loads(reply.read())

    def _submit(self, game_id: str, analysis: dict | None = None, error: str | None = None) -> None:
        import asyncio

        from catanatron.server.client import lobby_request

        request: dict = {"type": "submit_analysis", "game_id": game_id}
        if analysis is not None:
            request["analysis"] = analysis
        else:
            request["error"] = str(error)[:500]
        asyncio.run(lobby_request(self.server, request, house_token=self.token))

    def analyst(self, engine: Mapping[str, Any]) -> Analyst:
        label = str(engine["label"])
        if label not in self._analysts:
            self._analysts[label] = Analyst(str(engine["spec"]), label=label, device=self.device,
                                            onnx=engine.get("onnx_threads", "auto"))
        return self._analysts[label]

    def redo_all(self) -> int:
        """Analyze again every game the server lists that people played (20 actions or more),
        newest first, e.g. after the analysis changed; returns how many were sent."""
        games = self._get("/api/games?limit=200").get("games", [])
        sent = 0
        for game in games:
            if game.get("actions", 0) < 20:
                continue
            try:
                sent += self.analyze(game["game_id"]) is not None
            except Exception:
                logger.exception("game %s: the analysis failed", game["game_id"])
        return sent

    def analyze(self, game_id: str) -> dict | None:
        """Analyze one game and send the result; returns the log line."""
        import os

        record = self._get(f"/api/games/{game_id}/record")
        engine = engine_for(self.engines, record["room"].get("config") or {})
        if engine is None:
            logger.warning("game %s: no engine for its config", game_id)
            self._submit(game_id, error="no engine for this config")
            return None
        started = time.monotonic()
        last = [0.0]

        def progress(index: int, total: int) -> None:
            now = time.monotonic()
            if index and now - last[0] >= 30:
                last[0] = now
                eta = (now - started) / index * (total - index)
                logger.info("game %s: %d of %d actions, %.0f s so far, ~%.0f s left", game_id, index, total,
                            now - started, eta)

        try:
            analysis = self.analyst(engine).analyze(record, progress)
        except Exception as error:
            logger.exception("game %s: the analysis failed", game_id)
            self._submit(game_id, error=f"{type(error).__name__}: {error}")
            return None
        self._submit(game_id, analysis)
        line = {"time": time.time(), "game_id": game_id, "engine": engine["label"],
                "seconds": round(time.monotonic() - started, 1), **analysis["summary"]}
        with open(os.path.join(self.state_dir, "analyses.jsonl"), "a") as file:
            file.write(json.dumps(line) + "\n")
        logger.info("game %s analyzed by %s in %.0f s: accuracy %s", game_id, engine["label"],
                    line["seconds"], analysis["summary"]["accuracy"])
        return line

    def run(self, once: bool = False) -> None:
        while True:
            try:
                queue = self._get("/api/analysis/queue")["games"]
            except Exception as error:
                logger.warning("the queue cannot be read: %s", error)
                queue = []
            if queue:
                logger.info("%d games to analyze", len(queue))
            for game_id in queue:
                try:
                    self.analyze(game_id)
                except Exception:
                    logger.exception("game %s: could not be analyzed", game_id)
            if once:
                return
            if not queue:
                time.sleep(self.poll_s)
