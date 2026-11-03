"""Elo ratings for two-sided Kalshi matches, built only from Kalshi's own settled markets.

For each event with exactly two legs (team/player A vs B), the rating update uses the result
*after* the match, and the prediction for a match uses only matches that started earlier,
so there is no look-ahead. Ratings are kept per (sport, name): map markets (``KXCS2MAP``)
share ratings with match markets (``KXCS2GAME``). All settled events feed the ratings,
regardless of volume.

Settings were fixed before the October evaluation (2026-11-03): K = 32 (48 while a side has
fewer than 10 rated games), initial 1500, logistic scale 400.
"""

import math
import re

import polars as pl

from albert.venues.kalshi import ticker_start

ELO_SCHEMA = {"ticker": pl.String, "event_ticker": pl.String, "series": pl.String,
              "start": pl.Datetime("us", "UTC"), "elo_p": pl.Float64, "n_min": pl.Int64, "y": pl.Float64}


def _name(x: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def _sport(series: str) -> str:
    return series[:-3] + "GAME" if series.endswith("MAP") else series


# Blend fit on Aug-Sep 2026 games (research elo) and frozen for forward testing.
BLEND = (0.090, 0.978, 0.324)
ELO_SERIES = ("KXCS2GAME", "KXLOLGAME", "KXDOTA2GAME", "KXVALORANTGAME", "KXTTELITEMATCH",
              "KXTTSTARMATCH", "KXLIGAPROMATCH", "KXDARTSMATCH", "KXEBASKETBALLGAME", "KXCS2MAP",
              "KXLOLMAP", "KXDOTA2MAP", "KXVALORANTMAP", "KXR6GAME", "KXKHLGAME", "KXKBOGAME",
              "KXNPBGAME", "KXT20MATCH", "KXNCAAWVMATCH", "KXEFOOTBALLGAME")


class Ratings:
    """Current Elo state, for live predictions."""

    def __init__(self, ratings: dict, games: dict, init: float = 1500.0) -> None:
        self.ratings, self.games, self.init = ratings, games, init

    def predict(self, series: str, name_a: str, name_b: str) -> tuple[float, int]:
        """P(a beats b) and the smaller number of earlier rated games."""
        sport = _sport(series)
        a, b = (sport, _name(name_a)), (sport, _name(name_b))
        ra, rb = self.ratings.get(a, self.init), self.ratings.get(b, self.init)
        return 1 / (1 + 10 ** ((rb - ra) / 400)), min(self.games.get(a, 0), self.games.get(b, 0))


def blend(mid: float, elo_p: float, coef: tuple[float, float, float] = BLEND) -> float:
    a, b, c = coef
    return 1 / (1 + math.exp(-(a + b * logit(mid) + c * logit(elo_p))))


def elo_predictions(markets: pl.DataFrame, *, k: float = 32.0, k_new: float = 48.0,
                    new_games: int = 10, init: float = 1500.0,
                    state: dict | None = None) -> pl.DataFrame:
    """Pre-match P(first leg wins) for every settled two-leg event with a ticker start time.

    The "first leg" is the alphabetically first ticker, the same leg the volume-blind price
    samples use. ``n_min`` is the smaller number of earlier rated games of the two sides.
    """
    m = markets.filter(pl.col("result").is_in(["yes", "no"]))
    ev = (m.sort("ticker").group_by("event_ticker", "series_ticker", maintain_order=True)
          .agg(pl.col("ticker"), pl.col("yes_sub_title"), pl.col("result"))
          .filter(pl.col("ticker").list.len() == 2)
          .with_columns(pl.col("event_ticker").map_elements(
              ticker_start, return_dtype=pl.Datetime("us", "UTC")).alias("start"))
          .drop_nulls("start").sort("start", "event_ticker"))
    ratings: dict[tuple[str, str], float] = {}
    games: dict[tuple[str, str], int] = {}
    rows = []
    for e in ev.iter_rows(named=True):
        sport = _sport(e["series_ticker"])
        a, b = (sport, _name(e["yes_sub_title"][0])), (sport, _name(e["yes_sub_title"][1]))
        if not a[1] or not b[1] or a == b:
            continue
        ra, rb = ratings.get(a, init), ratings.get(b, init)
        na, nb = games.get(a, 0), games.get(b, 0)
        p = 1 / (1 + 10 ** ((rb - ra) / 400))
        ya = 1.0 if e["result"][0] == "yes" else 0.0
        rows.append({"ticker": e["ticker"][0], "event_ticker": e["event_ticker"],
                     "series": e["series_ticker"], "start": e["start"], "elo_p": p,
                     "n_min": min(na, nb), "y": ya})
        step = k if min(na, nb) >= new_games else k_new
        ratings[a] = ra + step * (ya - p)
        ratings[b] = rb - step * (ya - p)
        games[a], games[b] = na + 1, nb + 1
    if state is not None:  # hand the final ratings back for live use
        state["ratings"] = Ratings(ratings, games, init)
    return pl.DataFrame(rows, schema=ELO_SCHEMA)


def logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))
