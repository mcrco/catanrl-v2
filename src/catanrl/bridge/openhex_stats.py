"""How our OpenHex versions do, from OpenHex's public records (no key, read-only).

OpenHex keeps its bots and games in a public Firestore (the site reads them):
``bots/<id>`` holds a bot's Elo and ladder record (``elo1v1``, ``games1v1``,
``wins1v1``, the same for ``4p``), and a ``documents:runQuery`` with
``ARRAY_CONTAINS`` on ``playerBotIds`` lists every game a bot played (``mode``,
``status``, ``winnerBotId``, ``players`` with ``botId`` and ``name``,
``createdAt``, ``completedAt``). :class:`PublicRecords` fetches them,
:func:`summarize` counts them per version (each entry of the deploy file, the
retired ones too), per opponent, and between our own versions, and
:func:`format_summary` prints that::

    catanrl openhex stats deploy/openhex.yaml

Only finished games count: ``completed`` ones and ``aborted`` ones (a seat
missed its 60 s deadline; OpenHex gives the game to the best other seat and
rates it like any other, so the ladder's own record counts it too), the latter
also shown apart. Games still ``active`` are only counted as ongoing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from catanrl.eval.stats import wilson_interval

#: OpenHex's public Firestore documents.
ROOT = "https://firestore.googleapis.com/v1/projects/open-hex/databases/(default)/documents"
#: The fields of a game the counts read (the board, state and events are left behind).
GAME_FIELDS = ("mode", "status", "winnerBotId", "createdAt", "completedAt", "players", "playerBotIds")
FINISHED = ("completed", "aborted")
PAGE = 300


def decode(value: Mapping[str, Any]) -> Any:
    """One Firestore REST value (``{"integerValue": "3"}``, ``{"mapValue": ...}``) as Python."""
    (kind, inner), = value.items()
    if kind == "integerValue":
        return int(inner)
    if kind == "doubleValue":
        return float(inner)
    if kind == "nullValue":
        return None
    if kind == "arrayValue":
        return [decode(v) for v in inner.get("values", [])]
    if kind == "mapValue":
        return {k: decode(v) for k, v in inner.get("fields", {}).items()}
    return inner  # string, boolean, timestamp, reference, bytes


def document(doc: Mapping[str, Any]) -> dict:
    """A Firestore REST document as its fields, with its ``id``."""
    return {"id": doc["name"].rsplit("/", 1)[-1], **{k: decode(v) for k, v in doc.get("fields", {}).items()}}


class PublicRecords:
    """OpenHex's public records over HTTP (urllib, as OpenHexClient): nothing here holds a key."""

    def __init__(self, root: str = ROOT, timeout: float = 60.0) -> None:
        self.root = root.rstrip("/")
        self.timeout = timeout

    def _fetch(self, url: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": "catanrl-openhex-stats/1"})
        with urllib.request.urlopen(request, timeout=self.timeout) as reply:
            return json.loads(reply.read().decode())

    def bot(self, bot_id: str) -> dict | None:
        """``bots/<id>``, None when there is no such bot."""
        try:
            return document(self._fetch(f"{self.root}/bots/{urllib.parse.quote(bot_id)}"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise

    def find_bot(self, name: str, mode: str) -> dict | None:
        """The bot registered as ``name`` in ``mode`` (unique on OpenHex), None if there is none."""
        token = ""
        while True:
            query = f"pageSize={PAGE}&mask.fieldPaths=name&mask.fieldPaths=mode"
            page = self._fetch(f"{self.root}/bots?{query}" + (f"&pageToken={urllib.parse.quote(token)}" if token else ""))
            for doc in page.get("documents", []):
                found = document(doc)
                if found.get("name") == name and found.get("mode") == mode:
                    return self.bot(found["id"])
            token = page.get("nextPageToken", "")
            if not token:
                return None

    def games(self, bot_id: str) -> list[dict]:
        """Every game ``bot_id`` played (:data:`GAME_FIELDS` only), by document id."""
        out: list[dict] = []
        after = None
        while True:
            query: dict[str, Any] = {
                "from": [{"collectionId": "games"}],
                "select": {"fields": [{"fieldPath": f} for f in GAME_FIELDS]},
                "where": {"fieldFilter": {"field": {"fieldPath": "playerBotIds"}, "op": "ARRAY_CONTAINS",
                                          "value": {"stringValue": bot_id}}},
                "orderBy": [{"field": {"fieldPath": "__name__"}, "direction": "ASCENDING"}],
                "limit": PAGE,
            }
            if after:
                query["startAt"] = {"values": [{"referenceValue": after}], "before": False}
            rows = [row["document"] for row in self._fetch(f"{self.root}:runQuery", {"structuredQuery": query})
                    if "document" in row]
            out += [document(doc) for doc in rows]
            if len(rows) < PAGE:
                return out
            after = rows[-1]["name"]


def gather(entries: Sequence[Any], records: Any) -> tuple[dict[str, dict | None], dict[str, list[dict]]]:
    """Each entry's bot (by name; found by ``bot_id``, else by name and mode) and each found
    bot's games (by bot id)."""
    bots: dict[str, dict | None] = {}
    games: dict[str, list[dict]] = {}
    for entry in entries:
        bot = records.bot(entry.bot_id) if entry.bot_id else records.find_bot(entry.name, entry.mode)
        bots[entry.name] = bot
        if bot is not None and bot["id"] not in games:
            games[bot["id"]] = records.games(bot["id"])
    return bots, games


# ---------------------------------------------------------------- counting


def summarize(entries: Sequence[Any], bots: Mapping[str, Mapping[str, Any] | None],
              games: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict:
    """Per version (entry): finished games, wins, the win rate's Wilson 95% interval, aborted
    and ongoing games, the current Elo and ladder record, the first and last game; per
    version, its opponents (games, our wins); and every pair of our versions that met
    (games, each one's wins). Pure: ``bots`` by entry name, ``games`` by bot id."""
    ours = {bot["id"]: entry for entry in entries if (bot := bots.get(entry.name)) is not None}
    versions, opponents = [], {}
    played: dict[str, Mapping[str, Any]] = {}  # every finished game of ours, once
    for entry in entries:
        bot = bots.get(entry.name)
        row: dict[str, Any] = {"name": entry.name, "version": entry.version, "mode": entry.mode,
                               "retired": entry.retired, "bot_id": bot["id"] if bot else entry.bot_id,
                               "found": bot is not None}
        versions.append(row)
        if bot is None:
            continue
        me = bot["id"]
        mine = list(games.get(me, ()))
        finished = [g for g in mine if g.get("status") in FINISHED]
        wins = sum(g.get("winnerBotId") == me for g in finished)
        row.update({
            "games": len(finished), "wins": wins, "rate": wins / len(finished) if finished else None,
            "wilson": wilson_interval(wins, len(finished)) if finished else None,
            "aborted": sum(g.get("status") == "aborted" for g in finished),
            "ongoing": sum(g.get("status") == "active" for g in mine),
            "elo": bot.get(f"elo{entry.mode}"),
            "ladder": {"games": bot.get(f"games{entry.mode}"), "wins": bot.get(f"wins{entry.mode}")},
            "first": min((g["createdAt"] for g in finished if g.get("createdAt")), default=None),
            "last": max((g.get("completedAt") or g["createdAt"] for g in finished if g.get("createdAt")), default=None),
        })
        against: dict[str, dict] = {}
        for game in finished:
            played[game["id"]] = game
            for player in game.get("players", []):
                if player.get("botId") == me:
                    continue
                o = against.setdefault(player.get("botId"), {"opponent": player.get("name"), "bot_id": player.get("botId"),
                                                             "games": 0, "wins": 0})
                o["games"] += 1
                o["wins"] += game.get("winnerBotId") == me
        opponents[entry.name] = sorted(against.values(), key=lambda o: (-o["games"], str(o["opponent"])))
    order = {e.name: i for i, e in enumerate(entries)}
    pairs: dict[tuple[str, str], dict] = {}
    for game in played.values():
        met = sorted({p.get("botId") for p in game.get("players", [])} & set(ours), key=lambda b: order[ours[b].name])
        for i, a in enumerate(met):
            for b in met[i + 1:]:
                pair = pairs.setdefault((a, b), {"a": ours[a].name, "b": ours[b].name, "games": 0,
                                                 "a_wins": 0, "b_wins": 0, "other_wins": 0})
                pair["games"] += 1
                winner = game.get("winnerBotId")
                pair["a_wins" if winner == a else "b_wins" if winner == b else "other_wins"] += 1
    return {"versions": versions, "opponents": opponents, "head_to_head": list(pairs.values())}


def format_summary(summary: Mapping[str, Any]) -> str:
    """The summary as text: a row per version, its opponents, then our versions' meetings."""
    head = ("version", "name", "mode", "games", "wins", "win%", "95% CI", "aborted", "ongoing", "elo", "first", "last")
    numeric = {3, 4, 5, 7, 8, 9}
    rows, notes = [], []
    for v in summary["versions"]:
        version = v["version"] + (" (retired)" if v["retired"] else "")
        if not v["found"]:
            rows.append((version, v["name"], v["mode"], *["-"] * 9))
            notes.append(f"note: {v['name']}: not on OpenHex " + (f"(no bot {v['bot_id']})" if v["bot_id"] else
                                                                   f"(no bot of that name in {v['mode']})"))
            continue
        lo, hi = v["wilson"] or (None, None)
        rows.append((version, v["name"], v["mode"], str(v["games"]), str(v["wins"]),
                     f"{100 * v['rate']:.1f}" if v["rate"] is not None else "-",
                     f"[{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else "-",
                     str(v["aborted"]), str(v["ongoing"]), f"{v['elo']:.0f}" if v["elo"] is not None else "-",
                     (v["first"] or "-")[:10], (v["last"] or "-")[:10]))
        ladder = v["ladder"]
        if (ladder["games"], ladder["wins"]) != (v["games"], v["wins"]):
            notes.append(f"note: {v['name']}: OpenHex's own record says {ladder['games']} games, {ladder['wins']} wins")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    lines = ["  ".join(str(c).rjust(w) if i in numeric else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
             .rstrip() for r in [head, *rows]]
    lines += notes
    lines.append("(finished games: completed or aborted; an aborted game went to the best seat that did not miss "
                 "its deadline, and OpenHex rates it; ongoing games are not counted)")
    for v in summary["versions"]:
        against = summary["opponents"].get(v["name"])
        if not against:
            continue
        lines += ["", f"{v['name']} ({v['version']}), by opponent:"]
        width = max(len(str(o["opponent"])) for o in against)
        lines += [f"  {str(o['opponent']).ljust(width)}  {o['games']:4d} games  {o['wins']:4d} wins" for o in against]
    lines += ["", "head-to-head between our versions:"]
    if not summary["head_to_head"]:
        lines.append("  none")
    version = {v["name"]: v["version"] for v in summary["versions"]}
    for pair in summary["head_to_head"]:
        other = f", {pair['other_wins']} won by others" if pair["other_wins"] else ""
        lines.append(f"  {pair['a']} ({version[pair['a']]}) vs {pair['b']} ({version[pair['b']]}): {pair['games']} "
                     f"games, {pair['a_wins']}-{pair['b_wins']}{other}")
    return "\n".join(lines)
