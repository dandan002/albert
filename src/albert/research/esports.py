"""Independent esports ratings (PandaScore results) vs Kalshi esports prices.

Leak-free by construction: every PandaScore map result is applied to the ratings at its own
``end_at``; each Kalshi market is predicted at its decision time (ticker start - lead) from
the ratings as of that instant. Map markets get the map-win probability; match markets get
the best-of-N series probability (N is known before the match).
"""

import math
from datetime import timedelta

import polars as pl

from albert.matching import _team_sim
from albert.venues.kalshi import ticker_start
from albert.venues.pandascore import GAMES

SERIES_GAME = {s: g for g, pair in GAMES.items() for s in pair}


def match_kalshi_pandascore(kalshi: pl.DataFrame, ps_matches: pl.DataFrame, *,
                            max_gap_hours: float = 3.0, min_sim: float = 0.8) -> pl.DataFrame:
    """One row per Kalshi two-leg esports event matched to one PandaScore match.

    Needs the same game, ``scheduled_at`` within ``max_gap_hours`` of the ticker start, and
    both teams matching by name or acronym. Ambiguous events are dropped. ``first_team_id``
    is the PandaScore team of the alphabetically first Kalshi leg (the leg we price).
    """
    k = kalshi.filter(pl.col("series_ticker").is_in(list(SERIES_GAME)) & pl.col("result").is_in(["yes", "no"]))
    ev = (k.sort("ticker").group_by("event_ticker", "series_ticker", maintain_order=True)
          .agg(pl.col("ticker"), pl.col("yes_sub_title"), pl.col("result"))
          .filter(pl.col("ticker").list.len() == 2)
          .with_columns(pl.col("event_ticker").map_elements(ticker_start, return_dtype=pl.Datetime("us", "UTC"))
                        .alias("start")).drop_nulls("start"))
    by_game = ps_matches.filter(pl.col("scheduled_at").is_not_null()).partition_by("game", as_dict=True)
    rows = []
    for e in ev.iter_rows(named=True):
        game = SERIES_GAME[e["series_ticker"]]
        cand = by_game.get((game,))
        if cand is None:
            continue
        near = cand.filter((pl.col("scheduled_at") - e["start"]).abs() <= timedelta(hours=max_gap_hours))
        hits = []
        for c in near.iter_rows(named=True):
            names = [(c["a_id"], c["a_name"], c["a_acronym"]), (c["b_id"], c["b_name"], c["b_acronym"])]
            for first, second in (names, names[::-1]):
                s0 = max(_team_sim(e["yes_sub_title"][0], first[1]), _team_sim(e["yes_sub_title"][0], first[2] or ""))
                s1 = max(_team_sim(e["yes_sub_title"][1], second[1]), _team_sim(e["yes_sub_title"][1], second[2] or ""))
                if min(s0, s1) >= min_sim:
                    hits.append((c, first[0], second[0]))
        if len({h[0]["match_id"] for h in hits}) != 1:
            continue
        c, first_id, second_id = hits[0]
        map_no = None
        if e["series_ticker"].endswith("MAP"):
            tail = e["event_ticker"].rsplit("-", 1)[-1]
            map_no = int(tail) if tail.isdigit() else None
            if map_no is None:
                continue
        rows.append({"kalshi_ticker": e["ticker"][0], "kalshi_event": e["event_ticker"],
                     "series": e["series_ticker"], "game": game, "map_no": map_no,
                     "match_id": c["match_id"], "first_team_id": first_id, "second_team_id": second_id,
                     "number_of_games": c["number_of_games"], "kalshi_first_won": e["result"][0] == "yes",
                     "start": e["start"]})
    return pl.DataFrame(rows, schema={
        "kalshi_ticker": pl.String, "kalshi_event": pl.String, "series": pl.String, "game": pl.String,
        "map_no": pl.Int64, "match_id": pl.Int64, "first_team_id": pl.Int64, "second_team_id": pl.Int64,
        "number_of_games": pl.Int64, "kalshi_first_won": pl.Boolean, "start": pl.Datetime("us", "UTC")})


def series_prob(p: float, best_of: int) -> float:
    """P(win a best-of-N) from an independent per-map win probability."""
    need = best_of // 2 + 1
    return sum(math.comb(need - 1 + j, j) * p ** need * (1 - p) ** j for j in range(need))


def predictions(mapping: pl.DataFrame, ps_matches: pl.DataFrame, ps_games: pl.DataFrame, *,
                lead_hours: float = 1.0, k: float = 24.0, init: float = 1500.0) -> pl.DataFrame:
    """Map-level Elo from all PandaScore maps, queried at each Kalshi decision time."""
    teams = ps_matches.select("match_id", "game", "a_id", "b_id", "end_at")
    maps = (ps_games.filter((pl.col("status") == "finished") & pl.col("winner_id").is_not_null() & ~pl.col("forfeit"))
            .join(teams, on=["match_id", "game"])
            .with_columns(pl.coalesce(pl.col("end_at"), pl.col("end_at_right")).alias("t"))
            .drop_nulls("t").sort("t"))
    queries = mapping.with_columns((pl.col("start") - timedelta(hours=lead_hours)).alias("t")).sort("t")
    ratings: dict[tuple[str, int], float] = {}
    games: dict[tuple[str, int], int] = {}
    mi = maps.iter_rows(named=True)
    pending = next(mi, None)
    out = []
    for q in queries.iter_rows(named=True):
        while pending is not None and pending["t"] < q["t"]:  # only maps that ended before t
            g = pending["game"]
            a, b = (g, pending["a_id"]), (g, pending["b_id"])
            ra, rb = ratings.get(a, init), ratings.get(b, init)
            pa = 1 / (1 + 10 ** ((rb - ra) / 400))
            ya = 1.0 if pending["winner_id"] == pending["a_id"] else 0.0
            ratings[a], ratings[b] = ra + k * (ya - pa), rb - k * (ya - pa)
            games[a], games[b] = games.get(a, 0) + 1, games.get(b, 0) + 1
            pending = next(mi, None)
        a, b = (q["game"], q["first_team_id"]), (q["game"], q["second_team_id"])
        p_map = 1 / (1 + 10 ** ((ratings.get(b, init) - ratings.get(a, init)) / 400))
        p = p_map if q["map_no"] is not None else series_prob(p_map, max(q["number_of_games"] or 1, 1))
        out.append({"kalshi_ticker": q["kalshi_ticker"], "elo_p": p, "p_map": p_map,
                    "n_min": min(games.get(a, 0), games.get(b, 0))})
    return pl.DataFrame(out, schema={"kalshi_ticker": pl.String, "elo_p": pl.Float64,
                                     "p_map": pl.Float64, "n_min": pl.Int64})
