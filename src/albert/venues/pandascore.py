"""PandaScore esports results (CS2, LoL, Dota 2, Valorant), an independent rating source.

Kalshi only lists some matches; PandaScore has every pro match (all tiers) with map-level
results, tournament tier and exact start/end times. Auth: ``PANDASCORE_TOKEN`` in ``.env``,
sent as a bearer header (never stored). Free tier: 60 requests/minute, 1,000/hour, 100
matches per page.
"""

from datetime import datetime
from typing import Any, Iterator

import polars as pl

from albert.http import CachedClient, Fetched

# PandaScore videogame path -> the Kalshi match/map series for that game.
GAMES = {"csgo": ("KXCS2GAME", "KXCS2MAP"), "lol": ("KXLOLGAME", "KXLOLMAP"),
         "dota2": ("KXDOTA2GAME", "KXDOTA2MAP"), "valorant": ("KXVALORANTGAME", "KXVALORANTMAP")}


class PandaScore:
    def __init__(self, client: CachedClient) -> None:
        self.client = client

    def past_matches(self, game: str, start: datetime, end: datetime, *,
                     per_page: int = 100) -> Iterator[Fetched]:
        """Finished matches with ``begin_at`` in [start, end), oldest first, page by page."""
        page = 1
        while True:
            f = self.client.get(f"/{game}/matches/past", {
                "range[begin_at]": f"{start:%Y-%m-%dT%H:%M:%SZ},{end:%Y-%m-%dT%H:%M:%SZ}",
                "sort": "begin_at", "per_page": per_page, "page": page})
            yield f
            if not isinstance(f.payload, list) or len(f.payload) < per_page:
                return
            page += 1


MATCH_SCHEMA = {
    "match_id": pl.Int64, "game": pl.String, "begin_at": pl.Datetime("us", "UTC"),
    "scheduled_at": pl.Datetime("us", "UTC"), "end_at": pl.Datetime("us", "UTC"),
    "status": pl.String, "tier": pl.String, "league": pl.String, "serie": pl.String,
    "number_of_games": pl.Int64, "forfeit": pl.Boolean,
    "a_id": pl.Int64, "a_name": pl.String, "a_acronym": pl.String,
    "b_id": pl.Int64, "b_name": pl.String, "b_acronym": pl.String, "winner_id": pl.Int64,
}
GAME_SCHEMA = {"match_id": pl.Int64, "game": pl.String, "position": pl.Int64,
               "status": pl.String, "begin_at": pl.Datetime("us", "UTC"),
               "end_at": pl.Datetime("us", "UTC"), "winner_id": pl.Int64, "forfeit": pl.Boolean}


def _ts(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def match_rows(payload: Any, game: str) -> tuple[list[dict], list[dict]]:
    """(matches, map games) from a ``/{game}/matches/past`` page. Only two-team matches."""
    matches, games = [], []
    for m in payload if isinstance(payload, list) else []:
        opp = [o.get("opponent") or {} for o in m.get("opponents") or []]
        if len(opp) != 2 or any(o.get("id") is None for o in opp):
            continue
        t = m.get("tournament") or {}
        matches.append({
            "match_id": m.get("id"), "game": game, "begin_at": _ts(m.get("begin_at")),
            "scheduled_at": _ts(m.get("scheduled_at")), "end_at": _ts(m.get("end_at")),
            "status": m.get("status"), "tier": t.get("tier"),
            "league": (m.get("league") or {}).get("name"), "serie": (m.get("serie") or {}).get("full_name"),
            "number_of_games": m.get("number_of_games"), "forfeit": bool(m.get("forfeit")),
            "a_id": opp[0]["id"], "a_name": opp[0].get("name"), "a_acronym": opp[0].get("acronym"),
            "b_id": opp[1]["id"], "b_name": opp[1].get("name"), "b_acronym": opp[1].get("acronym"),
            "winner_id": m.get("winner_id"),
        })
        for g in m.get("games") or []:
            games.append({"match_id": m.get("id"), "game": game, "position": g.get("position"),
                          "status": g.get("status"), "begin_at": _ts(g.get("begin_at")),
                          "end_at": _ts(g.get("end_at")),
                          "winner_id": (g.get("winner") or {}).get("id"), "forfeit": bool(g.get("forfeit"))})
    return matches, games
