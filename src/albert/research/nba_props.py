"""NBA player props and team totals vs Kalshi: a minutes x per-minute-rate model of player box
scores and a pace x efficiency team model, season 2025-26 (smoke test: the first weeks of 2026-27).

Markets
-------
Kalshi lists per-game player ladders (``KXNBAPTS`` "Tyrese Maxey: 20+ points", ``KXNBAREB``,
``KXNBAAST``, ``KXNBA3PT``, ``KXNBAPRA``) and team totals (``KXNBATEAMTOTAL`` "Utah over 130.5
points scored", ``KXNBATOTAL``). Other per-game player series exist (``*PROT`` "protected"
copies with ~zero volume, ``KXNBAPR/PA/RA``, ``STL``, ``BLK``, ``STOCK``, ``FTM``, ``2D``, ``3D``,
escalators); they are not modelled. Rules: the strike is ``floor_strike`` + 0.5 ("20+" = more
than 19.5). **A player who is active but never takes the court settles at the last fair price
before tip** (``rules_secondary``), so the price is (approximately) P(stat >= k | the player
plays) and the model predicts exactly that; such legs (no yes/no result) are dropped.
Every settled leg comes from ``/historical/markets?series_ticker=`` (games before Kalshi's
2026-09-06 historical cutoff) and ``/markets?series_ticker=&status=settled`` (after), raw pages
under ``data/cache/nba_props/raw`` (never ``data/raw``), rows in
``data/cache/nba_props/markets.parquet``.

Box scores (ESPN public site API, ``site.api.espn.com``; that host serves no robots.txt (403),
``www.espn.com``'s robots.txt disallows the ``anthropic-ai`` token -- flagged in the report)
-----------------------------------------------------------------------------------------------
``scoreboard?dates=YYYYMMDD`` per day (schedule, scheduled tip, final scores) and
``summary?event=<id>`` per completed game (every player's minutes, points, FG/3P/FT, rebounds,
assists, turnovers; starter flag; DNP reason; team totals for possessions), seasons 2024-25
(priors), 2025-26 (study) and 2026-27 (smoke), cached zstd under ``data/cache/nba_props/espn``
(0.6 s between requests). ESPN's ``injuries`` block is the *current* injury list (fetched months
later), not the pre-tip one, so **no injury / inactive / lineup information is used**: the model
only knows previous games (who played, started, how long). That is a real handicap against a
market that sees the official injury report (published from the afternoon; final ~30 min
before tip) and is flagged throughout.

Model (causal; a game's box enters at tip + 3 h; fitted on Dec 2024 - Jan 2026 only)
---------------------------------------------------------------------------------
* **Team** (pace x efficiency, Elo-style updates, offseason regression): team offensive /
  defensive efficiency per 100 possessions and pace; expected game possessions, team points,
  margin; Normal laws (SD and mean bias from training residuals). Team totals, game totals, and
  blowout risk for the minutes model. Also opponent-allowed rebound / assist / 3PA factors.
* **Minutes**: least-squares on last-5 / season / previous-season minutes, last game's minutes
  and starter flag, start rate, back-to-back, team games missed since the last appearance, DNP
  last game, few games this season, and expected |margin| x start rate (blowout risk); Normal law
  with SD by projected-minutes bucket, truncated to [1, 48] (the market conditions on playing).
* **Rates**: per-minute rates from decayed sums over games played (decay per game and prior
  strength in minutes picked on training NB log likelihood; previous season weighs 0.5), shrunk
  to a position (G / F / C) x role (>= 24 min) prior; 3P% shrunk to the league rate (100 att.);
  times the opponent-allowed factor and the pace factor (variant ``noadj`` drops both).
* **Counts**: negative binomial given minutes (dispersion per stat fitted with actual minutes),
  mixed over the minutes law (10 Gauss-Hermite nodes); threes = binomial(3PA, 3P%) with 3PA
  negative binomial; PRA negative binomial on the summed mean. Variant ``simple``: NB around the
  season-to-date per-game average. The variant is chosen per stat on dev log loss.

Evaluation
----------
Decision times ``t`` = 24 h and 1 h before ESPN's scheduled tip, floored to the hour (last
season's props listed ~20 h before tip, so ~20 % of legs exist at 24 h). Quote = bid / ask close
of the hourly candle ending at ``t``. Historical candles cost one request per leg (no batch
endpoint for settled markets), so legs are sampled volume-blind (:func:`sample_legs`: 450
player-game ladders per stat x split, 2 random strikes each; 150 game-total and 120 team-total
games per split, legs nearest the ladder median) and fetched in a seeded hash order, so any
prefix of the fetch is itself a random subsample. Harness ``fundamental.evaluate`` per stat x
lead (blend ``logit P = a + b logit(mid) + c logit(model)`` fitted on dev, t clustered by game),
split by quoted spread (<= 5c, 6-15c, > 15c) and player tier (star >= 20 ppg; starter; bench).
Maker (``cbb.maker_fills_all`` / ``maker_table``): bids around the dev blend vs the same bids
around the dev-recalibrated mid (control); trade-through fills from hourly-candle traded
low / high in (t, last whole hour before tip], validated against real prints on a sample.
Split: dev = games tipping before Feb 2026 (props list from Nov 2025), holdout = Feb - Jun 2026.
``KXNBATEAMTOTAL`` lists only from Feb 2026: dev Feb - Mar, holdout Apr - Jun 2026. ``KXNBAPRA``
lists only from Oct 2026 (smoke test only). Smoke: the frozen model on 2026-27 legs (live batch
candles), blend fitted on last season's dev rows.
Run: ``python -m albert.research.nba_props <stage>`` with stages ``markets``, ``espn``,
``candles``, ``smoke_candles``, ``features``, ``study`` (see :func:`main`).
"""

import hashlib
import json
import math
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.config import Settings

NAME = "nba_props"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
PROP_SERIES = {"KXNBAPTS": "pts", "KXNBAREB": "reb", "KXNBAAST": "ast", "KXNBA3PT": "fg3m", "KXNBAPRA": "pra"}
TEAM_SERIES = {"KXNBATEAMTOTAL": "team_total", "KXNBATOTAL": "total"}
SERIES = {**PROP_SERIES, **TEAM_SERIES}
SEASONS = {  # ESPN season year -> scoreboard dates (US Eastern)
    2025: (date(2024, 10, 22), date(2025, 6, 23)),
    2026: (date(2025, 10, 21), date(2026, 6, 25)),
    2027: (date(2026, 10, 20), date(2026, 11, 4)),
}
STUDY_FROM = datetime(2025, 10, 1, tzinfo=UTC)
HOLDOUT_FROM = datetime(2026, 2, 1, 5, tzinfo=UTC)   # Feb 1 2026 00:00 US Eastern
STUDY_TO = datetime(2026, 7, 1, tzinfo=UTC)
SMOKE_FROM = datetime(2026, 10, 1, tzinfo=UTC)
LEADS_H = (24, 1)
RESULT_LAG = timedelta(hours=3)   # a game's box score is used from tip + 3 h on


class NbaSettings(Settings):
    """Settings whose raw HTTP cache lives under ``data/cache/nba_props/raw`` (never ``data/raw``)."""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "cache" / NAME / "raw"


def nba_settings(s: Settings | None = None) -> NbaSettings:
    return NbaSettings(data_dir=(s.data_dir if s is not None else Settings().data_dir))


def cache_dir(s) -> Path:
    p = s.data_dir / "cache" / NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- ESPN -------------------------------------------------------------------------------------
_last_request = [0.0]


def espn_get(s, path: str, params: dict[str, Any], *, min_interval: float = 0.6) -> dict[str, Any]:
    """GET an ESPN site-API JSON once; later calls read the zstd cache under
    ``data/cache/nba_props/espn``."""
    import httpx

    from albert.http import read_raw, write_raw

    key = hashlib.sha256((path + "?" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
    file = cache_dir(s) / "espn" / f"{slug}-{key}.json.zst"
    if file.exists():
        return read_raw(file)["response"]
    r = None
    for attempt in range(6):
        wait = min_interval - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
        try:
            r = httpx.get(f"{ESPN_BASE}/{path}", params=params, timeout=30,
                          headers={"User-Agent": "albert-research (public data, cached)"})
        except httpx.HTTPError:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(3 * (attempt + 1))
            continue
        break
    if r is None:
        raise RuntimeError(f"ESPN {path} {params}: no response")
    r.raise_for_status()
    payload = r.json()
    write_raw(file, {"url": f"{ESPN_BASE}/{path}", "params": params,
                     "fetched_at": datetime.now(UTC).isoformat(), "response": payload})
    return payload


def _ts(x: str | None) -> datetime | None:
    if not x:
        return None
    x = x.replace("Z", "+00:00")
    if re.match(r".*T\d\d:\d\d\+", x):
        x = x.replace("+", ":00+", 1)
    return datetime.fromisoformat(x).astimezone(UTC)


GAME_SCHEMA: dict[str, Any] = {
    "game_id": pl.String, "season": pl.Int64, "season_type": pl.Int64, "comp_type": pl.String,
    "tip": pl.Datetime("us", "UTC"), "time_valid": pl.Boolean, "completed": pl.Boolean, "status": pl.String,
    "periods": pl.Int64, "home_id": pl.String, "home_abbr": pl.String, "home_name": pl.String,
    "home_score": pl.Float64, "away_id": pl.String, "away_abbr": pl.String, "away_name": pl.String,
    "away_score": pl.Float64,
}


def game_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per scoreboard event: scheduled tip, teams, final scores, status."""
    rows = []
    for e in payload.get("events") or []:
        comp = (e.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if set(teams) != {"home", "away"}:
            continue
        status = comp.get("status") or e.get("status") or {}
        stype = status.get("type") or {}
        row = {"game_id": str(e.get("id")), "season": (e.get("season") or {}).get("year"),
               "season_type": (e.get("season") or {}).get("type"), "comp_type": (comp.get("type") or {}).get("abbreviation"),
               "tip": _ts(comp.get("date") or e.get("date")), "time_valid": comp.get("timeValid"),
               "completed": bool(stype.get("completed")), "status": stype.get("name"), "periods": status.get("period")}
        for side in ("home", "away"):
            c = teams[side]
            t = c.get("team") or {}
            score = c.get("score")
            row.update({f"{side}_id": str(t.get("id")), f"{side}_abbr": t.get("abbreviation"),
                        f"{side}_name": t.get("displayName"),
                        f"{side}_score": float(score) if score not in (None, "") else None})
        rows.append(row)
    return rows


def fetch_games(s, seasons: dict[int, tuple[date, date]] = SEASONS) -> pl.DataFrame:
    """Every NBA game on every scoreboard date of these seasons (one request per day)."""
    rows: list[dict[str, Any]] = []
    today = datetime.now(UTC).date()
    for _season, (d0, d1) in seasons.items():
        d = d0
        while d <= min(d1, today):
            rows.extend(game_rows(espn_get(s, "scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": "100"})))
            d += timedelta(days=1)
    return pl.DataFrame(rows, schema=GAME_SCHEMA).unique("game_id", keep="last").sort("tip", "game_id")


BOX_COLS = ("min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "ast", "tov", "stl", "blk", "oreb", "dreb", "pf")
BOX_SCHEMA: dict[str, Any] = {"game_id": pl.String, "team_id": pl.String, "athlete_id": pl.String, "name": pl.String,
                              "pos": pl.String, "starter": pl.Boolean, "dnp": pl.Boolean, "reason": pl.String,
                              **{c: pl.Float64 for c in BOX_COLS}}
TEAM_BOX_SCHEMA: dict[str, Any] = {"game_id": pl.String, "team_id": pl.String, "pts": pl.Float64, "fga": pl.Float64,
                                   "fta": pl.Float64, "oreb": pl.Float64, "tov": pl.Float64, "fg3a": pl.Float64,
                                   "reb": pl.Float64, "ast": pl.Float64}


def _num(x: Any) -> float | None:
    try:
        return float(x) if x not in (None, "", "--", "-") else None
    except ValueError:
        return None


def _pair(x: Any) -> tuple[float | None, float | None]:
    if not x or "-" not in str(x):
        return None, None
    a, b = str(x).split("-", 1)
    return _num(a), _num(b)


def box_rows(game_id: str, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Player and team box rows from an ESPN ``summary`` payload. Players who did not play carry
    ``dnp`` and no stats; minutes are whole minutes as ESPN shows them."""
    players, teams = [], []
    box = payload.get("boxscore") or {}
    for t in box.get("players") or []:
        tid = str((t.get("team") or {}).get("id"))
        for st in t.get("statistics") or []:
            keys = [k.upper() for k in st.get("names") or st.get("labels") or []]
            for a in st.get("athletes") or []:
                ath = a.get("athlete") or {}
                vals = dict(zip(keys, a.get("stats") or [], strict=False))
                fgm, fga = _pair(vals.get("FG"))
                f3m, f3a = _pair(vals.get("3PT"))
                ftm, fta = _pair(vals.get("FT"))
                minutes = _num(vals.get("MIN"))
                row = {"game_id": game_id, "team_id": tid, "athlete_id": str(ath.get("id")),
                       "name": ath.get("displayName"), "pos": (ath.get("position") or {}).get("abbreviation"),
                       "starter": bool(a.get("starter")),
                       "dnp": bool(a.get("didNotPlay")) or not a.get("stats") or not minutes,
                       "reason": a.get("reason"), "min": minutes, "pts": _num(vals.get("PTS")), "fgm": fgm, "fga": fga,
                       "fg3m": f3m, "fg3a": f3a, "ftm": ftm, "fta": fta, "reb": _num(vals.get("REB")),
                       "ast": _num(vals.get("AST")), "tov": _num(vals.get("TO")), "stl": _num(vals.get("STL")),
                       "blk": _num(vals.get("BLK")), "oreb": _num(vals.get("OREB")), "dreb": _num(vals.get("DREB")),
                       "pf": _num(vals.get("PF"))}
                players.append(row)
    for t in box.get("teams") or []:
        tid = str((t.get("team") or {}).get("id"))
        st = {x.get("name"): x.get("displayValue") for x in t.get("statistics") or []}
        fgm, fga = _pair(st.get("fieldGoalsMade-fieldGoalsAttempted"))
        _, f3a = _pair(st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
        ftm, fta = _pair(st.get("freeThrowsMade-freeThrowsAttempted"))
        teams.append({"game_id": game_id, "team_id": tid, "pts": None, "fga": fga, "fta": fta,
                      "oreb": _num(st.get("offensiveRebounds")),
                      "tov": _num(st.get("totalTurnovers") or st.get("turnovers")), "fg3a": f3a,
                      "reb": _num(st.get("totalRebounds")), "ast": _num(st.get("assists"))})
    return players, teams


def fetch_boxes(s, games: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Player and team box scores of every completed game (one ``summary`` request per game),
    cached as ``data/cache/nba_props/{boxes,team_boxes}.parquet``."""
    cd = cache_dir(s)
    pp, tp = cd / "boxes.parquet", cd / "team_boxes.parquet"
    have_p = pl.read_parquet(pp) if pp.exists() else pl.DataFrame(schema=BOX_SCHEMA)
    have_t = pl.read_parquet(tp) if tp.exists() else pl.DataFrame(schema=TEAM_BOX_SCHEMA)
    done = set(have_t["game_id"])
    todo = games.filter(pl.col("completed") & ~pl.col("game_id").is_in(list(done)))["game_id"].to_list()
    prow, trow = [], []

    def save() -> tuple[pl.DataFrame, pl.DataFrame]:
        out = []
        for have, rows, schema, path in ((have_p, prow, BOX_SCHEMA, pp), (have_t, trow, TEAM_BOX_SCHEMA, tp)):
            f = pl.concat([have, pl.DataFrame(rows, schema=schema)]) if rows else have
            tmp = path.with_suffix(".tmp")
            f.write_parquet(tmp)
            tmp.replace(path)
            out.append(f)
        return out[0], out[1]

    for i, gid in enumerate(todo, 1):
        try:
            p, t = box_rows(gid, espn_get(s, "summary", {"event": gid}))
        except Exception as error:   # noqa: BLE001 - one bad game must not stop the crawl
            print(f"skip box {gid}: {str(error)[:120]}", flush=True)
            continue
        prow.extend(p)
        trow.extend(t)
        if i % 200 == 0:
            print(f"  boxes {i}/{len(todo)}", flush=True)
            have_p, have_t = save()
            prow, trow = [], []
    return save()


# ---- Kalshi markets ---------------------------------------------------------------------------
def _short_sleep(d: float) -> None:
    """429 / 5xx back-off capped at a jittered 0.3-0.8 s (the public limit is shared; the default
    doubling back-off starves a polite client)."""
    import random

    time.sleep(d if d <= 0.2 else random.uniform(0.3, 0.8))


def _api(s, *, min_interval: float = 0.25):
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    ns = nba_settings(s)
    return k.Kalshi(CachedClient("kalshi", ns.kalshi_base, ns.raw_dir, min_interval=min_interval,
                                 max_retries=60, sleep=_short_sleep))


MARKET_SCHEMA: dict[str, Any] = {
    "ticker": pl.String, "event_ticker": pl.String, "series_ticker": pl.String, "title": pl.String,
    "yes_sub_title": pl.String, "floor_strike": pl.Float64, "player_key": pl.String, "open_time": pl.Datetime("us", "UTC"),
    "close_time": pl.Datetime("us", "UTC"), "occurrence_time": pl.Datetime("us", "UTC"), "result": pl.String,
    "settlement_value": pl.Float64, "volume": pl.Float64, "fee_type": pl.String, "fee_multiplier": pl.Float64,
    "status": pl.String,
}


def market_row(m: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    from albert.venues.kalshi import _field, num, parse_ts, series_of

    cs = m.get("custom_strike") or {}
    return {"ticker": m["ticker"], "event_ticker": m.get("event_ticker"),
            "series_ticker": series_of(m.get("event_ticker") or m["ticker"]), "title": m.get("title"),
            "yes_sub_title": m.get("yes_sub_title"), "floor_strike": num(m.get("floor_strike")),
            "player_key": cs.get("basketball_player"), "open_time": parse_ts(m.get("open_time")),
            "close_time": parse_ts(m.get("close_time")), "occurrence_time": parse_ts(m.get("occurrence_datetime")),
            "result": m.get("result") or None, "settlement_value": num(_field(m, "settlement_value")),
            "volume": num(_field(m, "volume")), "fee_type": info.get("fee_type"),
            "fee_multiplier": num(info.get("fee_multiplier")), "status": m.get("status")}


def kalshi_markets(s, *, refresh: bool = False, oldest: datetime = STUDY_FROM) -> pl.DataFrame:
    """Every settled leg of :data:`SERIES` closing after ``oldest``: historical pages (newest
    first, paged until past ``oldest``) plus the live settled endpoint after the cutoff."""
    from albert.venues import kalshi as k

    path = cache_dir(s) / "markets.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    api = _api(s)
    info = {r["series_ticker"]: r for r in k.series_frame(api.series().payload).iter_rows(named=True)}
    rows: list[dict[str, Any]] = []
    for series in SERIES:
        n0 = len(rows)
        for page in api._cursor_pages("/historical/markets", {"series_ticker": series, "limit": 1000}, "markets"):
            got = [market_row(m, info.get(series, {})) for m in page.payload.get("markets") or []]
            rows.extend(got)
            closes = [r["close_time"] for r in got if r["close_time"] is not None]
            if closes and min(closes) < oldest:
                break
        for page in api.series_markets(series, "settled", min_close=max(oldest, datetime(2026, 9, 1, tzinfo=UTC)), cache=False):
            rows.extend(market_row(m, info.get(series, {})) for m in page.payload.get("markets") or [])
        print(f"  {series}: {len(rows) - n0} markets", flush=True)
    m = pl.DataFrame(rows, schema=MARKET_SCHEMA).unique("ticker", keep="last").filter(pl.col("close_time") >= oldest)
    tmp = path.with_suffix(".tmp")
    m.write_parquet(tmp)
    tmp.replace(path)
    return m


# ---- Kalshi <-> ESPN ----------------------------------------------------------------------------
KALSHI_TO_ESPN = {"GSW": "GS", "NYK": "NY", "SAS": "SA", "NOP": "NO", "UTA": "UTAH", "WAS": "WSH"}
_EV_RX = re.compile(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})$")
_MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
_NAME_RX = re.compile(r"^(.+?)(?::| records)\s+\d+\+", re.I)
_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_person(x: str | None) -> str:
    """Lower-case ASCII name without punctuation or generational suffixes."""
    x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    toks = [t for t in re.sub(r"[^a-z ]+", " ", x.replace("'", "").replace(".", "")).split() if t not in _SUFFIX]
    return " ".join(toks)


def parse_event(event_ticker: str) -> tuple[date, str, str] | None:
    """``KXNBAPTS-26JAN06LALNOP`` -> (2026-01-06, "LAL", "NOP") (scheduled US-Eastern date)."""
    m = _EV_RX.match(event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    yy, mon, dd, a, b = m.groups()
    try:
        return date(2000 + int(yy), _MONTHS[mon], int(dd)), a, b
    except ValueError:
        return None


def legs_frame(markets: pl.DataFrame, games: pl.DataFrame, boxes: pl.DataFrame | None) -> pl.DataFrame:
    """Every leg joined to its ESPN game (same Eastern date, +-1 day, same two teams), with what
    it pays on: ``kind`` (pts / reb / ast / fg3m / pra / team_total / total), ``k`` (YES iff the
    stat >= k, i.e. > floor_strike), the team (props, team totals) and, for props, the ESPN
    athlete id (name matched within the team's box scores: that game first, then the season)."""
    g = games.with_columns(pl.col("tip").dt.convert_time_zone("America/New_York").dt.date().alias("et_date"))
    by_key: dict[tuple[date, frozenset], dict[str, Any]] = {}
    for r in g.iter_rows(named=True):
        by_key[(r["et_date"], frozenset((r["home_abbr"], r["away_abbr"])))] = r
    names: dict[tuple[str, str, str], str] = {}
    season_names: dict[tuple[int, str, str], set[str]] = {}
    if boxes is not None and boxes.height:
        bx = boxes.join(games.select("game_id", "season"), on="game_id", how="left")
        for r in bx.select("game_id", "season", "team_id", "athlete_id", "name").unique().iter_rows(named=True):
            nm = norm_person(r["name"])
            names[(r["game_id"], r["team_id"], nm)] = r["athlete_id"]
            season_names.setdefault((r["season"], r["team_id"], nm), set()).add(r["athlete_id"])
    rows = []
    for r in markets.iter_rows(named=True):
        kind = SERIES.get(r["series_ticker"])
        ev = parse_event(r["event_ticker"])
        if kind is None or ev is None or r["floor_strike"] is None:
            continue
        d, a, b = ev
        ea, eb = KALSHI_TO_ESPN.get(a, a), KALSHI_TO_ESPN.get(b, b)
        gm = next((by_key.get((d + timedelta(days=dd), frozenset((ea, eb)))) for dd in (0, -1, 1)
                   if by_key.get((d + timedelta(days=dd), frozenset((ea, eb)))) is not None), None)
        if gm is None:
            continue
        leg = r["ticker"].split("-", 2)[2] if r["ticker"].count("-") >= 2 else ""
        team_side = None
        if kind != "total":
            code = leg[:3]
            ec = KALSHI_TO_ESPN.get(code, code)
            team_side = "home" if ec == gm["home_abbr"] else "away" if ec == gm["away_abbr"] else None
            if team_side is None:
                continue
        athlete, player = None, None
        if kind in PROP_SERIES.values():
            nm = _NAME_RX.match(r["title"] or "")
            player = nm.group(1).strip() if nm else None
            key = norm_person(player)
            tid = gm[f"{team_side}_id"]
            athlete = names.get((gm["game_id"], tid, key))
            if athlete is None:
                cands = season_names.get((gm["season"], tid, key), set())
                athlete = next(iter(cands)) if len(cands) == 1 else None
        y = 1 if r["result"] == "yes" else 0 if r["result"] == "no" else None
        rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "series": r["series_ticker"], "kind": kind,
                     "game_id": gm["game_id"], "season": gm["season"], "tip": gm["tip"], "time_valid": gm["time_valid"],
                     "team_side": team_side, "team_id": gm[f"{team_side}_id"] if team_side else None,
                     "opp_id": (gm["away_id"] if team_side == "home" else gm["home_id"]) if team_side else None,
                     "player": player, "player_key": r["player_key"], "athlete_id": athlete,
                     "k": float(r["floor_strike"]) + 0.5, "y": y, "result": r["result"],
                     "open_time": r["open_time"], "close_time": r["close_time"], "volume": r["volume"],
                     "fee_type": r["fee_type"], "fee_multiplier": r["fee_multiplier"]})
    str_cols = ("team_side", "team_id", "opp_id", "player", "player_key", "athlete_id", "result", "fee_type")
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        *[pl.col(c).cast(pl.String) for c in str_cols], pl.col("y").cast(pl.Int8), pl.col("fee_multiplier").cast(pl.Float64))


def floor_hour(e: pl.Expr) -> pl.Expr:
    return e.dt.truncate("1h")


def _hash_rank(cols: list[str], seed: int) -> pl.Expr:
    """Deterministic pseudo-random key from identifiers only (volume-blind, stable)."""
    return pl.concat_str([pl.lit(str(seed)), *[pl.col(c).cast(pl.String) for c in cols]], separator="|").hash(seed=seed)


# Sample sizes for historical candles (one request per leg; the public rate limit is shared).
LADDERS_PER_CELL = 450      # player-game ladders per stat x split, 2 legs each
TOTAL_GAMES_PER_CELL = 150  # game-total ladders per split, 3 legs nearest the ladder median
TEAM_GAMES_PER_CELL = 120   # team-total games per split, 2 legs per team nearest the median
TT_HOLDOUT_FROM = datetime(2026, 4, 1, 4, tzinfo=UTC)   # team totals list only from Feb 2026


def split_of(kind: pl.Expr, tip: pl.Expr) -> pl.Expr:
    return pl.when(kind == "team_total").then(tip >= TT_HOLDOUT_FROM).otherwise(tip >= HOLDOUT_FROM)


def sample_legs(legs: pl.DataFrame, *, frm: datetime = STUDY_FROM, to: datetime = STUDY_TO) -> pl.DataFrame:
    """Volume-blind seeded sample of last season's legs for historical candles. Eligible legs:
    settled yes/no, matched game with a valid tip, open by tip - 1 h (listing metadata only).
    Props: ``LADDERS_PER_CELL`` random (player, game) ladders per stat x split, 2 random strikes
    each. Game totals / team totals: random games, the legs nearest the ladder median strike."""
    el = legs.filter(pl.col("y").is_not_null() & pl.col("time_valid").fill_null(False)
                     & (pl.col("tip") >= frm) & (pl.col("tip") < to)
                     & (pl.col("open_time") <= pl.col("tip") - pl.duration(hours=1)))
    el = el.with_columns(split_of(pl.col("kind"), pl.col("tip")).alias("holdout"))
    out = []
    props = el.filter(pl.col("kind").is_in(list(PROP_SERIES.values())))
    lad = (props.select("kind", "holdout", "event", "player").unique()
           .with_columns(_hash_rank(["event", "player", "kind"], 20251101).alias("h"))
           .sort("h").group_by("kind", "holdout", maintain_order=True).head(LADDERS_PER_CELL))
    pick = (props.join(lad.select("kind", "event", "player"), on=["kind", "event", "player"])
            .with_columns(_hash_rank(["ticker"], 20251102).alias("h"))
            .sort("h").group_by("kind", "event", "player", maintain_order=True).head(2))
    out.append(pick.drop("h"))
    for kind, n_games, per in (("total", TOTAL_GAMES_PER_CELL, 3), ("team_total", TEAM_GAMES_PER_CELL, 2)):
        x = el.filter(pl.col("kind") == kind)
        gs = (x.select("holdout", "game_id").unique().with_columns(_hash_rank(["game_id"], 20251103).alias("h"))
              .sort("h").group_by("holdout", maintain_order=True).head(n_games))
        x = x.filter(pl.col("game_id").is_in(gs["game_id"].implode()))
        grp = ["game_id"] if kind == "total" else ["game_id", "team_side"]
        x = (x.with_columns((pl.col("k") - pl.col("k").median().over(grp)).abs().alias("d"))
             .sort("d", "ticker").group_by(grp, maintain_order=True).head(per))
        out.append(x.drop("d"))
    return pl.concat(out, how="diagonal_relaxed")


CANDLE_SCHEMA: dict[str, Any] = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "bid": pl.Float64,
                                 "ask": pl.Float64, "price_low": pl.Float64, "price_high": pl.Float64,
                                 "volume": pl.Float64}


def candle_rows(payload: dict[str, Any], ticker: str) -> list[dict[str, Any]]:
    """Hourly yes bid / ask close and traded-price low / high (null when nothing traded)."""
    from albert.venues.kalshi import _field, num, parse_ts

    rows = []
    for c in payload.get("candlesticks") or []:
        pr = c.get("price") or {}
        rows.append({"ticker": ticker, "end_ts": parse_ts(c.get("end_period_ts")),
                     "bid": num(_field(c.get("yes_bid") or {}, "close")), "ask": num(_field(c.get("yes_ask") or {}, "close")),
                     "price_low": num(_field(pr, "low")), "price_high": num(_field(pr, "high")),
                     "volume": num(_field(c, "volume"))})
    return rows


def candle_window(legs: pl.DataFrame) -> pl.DataFrame:
    """Per leg: candles from the earliest decision time (or listing) to the last whole hour
    before tip (in-play hours are never used)."""
    lo = floor_hour(pl.max_horizontal(pl.col("tip") - pl.duration(hours=max(LEADS_H) + 1), pl.col("open_time")))
    return legs.with_columns(lo.alias("c_from"), floor_hour(pl.col("tip")).alias("c_to")).filter(pl.col("c_from") < pl.col("c_to"))


def fetch_candles(s, legs: pl.DataFrame, *, workers: int = 2, tag: str = "hist") -> pl.DataFrame:
    """Hourly candles of each leg over :func:`candle_window` (one request per historical leg;
    batches of up to 100 legs of one game for live legs). Cached in
    ``data/cache/nba_props/candles_<tag>.parquet`` (+ ``_done``); a failed leg is retried on the
    next run."""

    path = cache_dir(s) / f"candles_{tag}.parquet"
    done_path = cache_dir(s) / f"candles_{tag}_done.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=CANDLE_SCHEMA)
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = candle_window(legs).filter(~pl.col("ticker").is_in(list(done))).unique("ticker", maintain_order=True)
    if not todo.height:
        return have
    api = _api(s, min_interval=0.15)
    cutoff = api.cutoff().get("market_settled_ts")
    jobs: list[tuple[bool, list[str], datetime, datetime]] = []
    hist = todo.filter(pl.col("close_time") < cutoff) if cutoff else todo.head(0)
    live = todo.filter(pl.col("close_time") >= cutoff) if cutoff else todo
    jobs += [(True, [r["ticker"]], r["c_from"], r["c_to"]) for r in hist.iter_rows(named=True)]
    for (_g, _s, hi), grp in live.group_by("game_id", "series", "c_to", maintain_order=True):
        tks, lo = grp["ticker"].to_list(), grp["c_from"].min()
        jobs += [(False, tks[i:i + 100], lo, hi) for i in range(0, len(tks), 100)]

    def one(job: tuple[bool, list[str], datetime, datetime]) -> tuple[list[str], list[dict[str, Any]]]:
        historical, tickers, lo, hi = job
        try:
            if historical:
                return tickers, candle_rows(api.candles(tickers[0], lo, hi, 60, True).payload, tickers[0])
            page = api.candles_batch(tickers, lo, hi, 60)
            out: list[dict[str, Any]] = []
            for mk in page.payload.get("markets") or []:
                out.extend(candle_rows(mk, mk.get("market_ticker")))
            return tickers, out
        except RuntimeError as error:
            print(f"skip candles {tickers[0]} (+{len(tickers) - 1}): {str(error)[:120]}", flush=True)
            return [], []

    rows: list[dict[str, Any]] = []
    finished: list[str] = []

    def save() -> pl.DataFrame:
        new = pl.DataFrame(rows, schema=CANDLE_SCHEMA) if rows else None
        out = pl.concat([have, new]).unique(["ticker", "end_ts"]) if new is not None else have
        for f, target in ((out, path), (pl.DataFrame({"ticker": sorted(done | set(finished))}, schema={"ticker": pl.String}), done_path)):
            tmp = target.with_suffix(".tmp")
            f.write_parquet(tmp)
            tmp.replace(target)
        return out

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (tk, r) in enumerate(pool.map(one, jobs), 1):
            rows.extend(r)
            finished.extend(tk)
            if i % 300 == 0:
                print(f"  candle jobs {i}/{len(jobs)}, {len(rows)} rows, {i / (time.monotonic() - t0):.2f} req/s", flush=True)
                save()
    return save()


# ---- model inputs -------------------------------------------------------------------------------
STATS = ("pts", "reb", "ast", "fg3a", "fg3m")


def load_box_inputs(s) -> dict[str, Any]:
    """Games (regular season, play-in, playoffs; no All-Star), player boxes, and team-games with
    possessions (FGA + 0.44 FTA - OREB + TOV, averaged over both teams) and opponent stats."""
    cd = cache_dir(s)
    games = (pl.read_parquet(cd / "games.parquet")
             .filter(pl.col("season_type").is_in([2, 3, 5]) & (pl.col("comp_type").fill_null("") != "ALLSTAR")
                     & ~pl.col("status").is_in(["STATUS_POSTPONED", "STATUS_CANCELED"])))
    boxes = pl.read_parquet(cd / "boxes.parquet").unique(["game_id", "athlete_id"], keep="last")
    tb = pl.read_parquet(cd / "team_boxes.parquet").unique(["game_id", "team_id"], keep="last")
    fin = games.filter(pl.col("completed") & (pl.col("status") == "STATUS_FINAL"))
    # team made threes from the player rows
    t3 = boxes.group_by("game_id", "team_id").agg(pl.col("fg3m").sum())
    tg = []
    for side, other in (("home", "away"), ("away", "home")):
        tg.append(fin.select("game_id", "season", "tip", "periods", pl.col(f"{side}_id").alias("team_id"),
                             pl.col(f"{other}_id").alias("opp_id"), pl.col(f"{side}_score").alias("pts"),
                             pl.col(f"{other}_score").alias("opp_pts"), pl.lit(side == "home").alias("home")))
    tg = pl.concat(tg).join(tb.drop("pts"), on=["game_id", "team_id"], how="inner").join(t3, on=["game_id", "team_id"], how="left")
    tg = tg.with_columns((pl.col("fga") + 0.44 * pl.col("fta") - pl.col("oreb") + pl.col("tov")).alias("poss_own"))
    tg = tg.with_columns(pl.col("poss_own").mean().over("game_id").alias("poss"),
                         (48 + 5 * (pl.col("periods").fill_null(4).clip(4, 9) - 4)).alias("game_min"))
    opp = tg.select("game_id", pl.col("team_id").alias("opp_id"), *[pl.col(c).alias(f"opp_{c}") for c in ("reb", "ast", "fg3a", "fg3m")])
    tg = tg.join(opp, on=["game_id", "opp_id"], how="left").filter(pl.col("poss").is_not_null())
    return {"games": games, "boxes": boxes, "team_games": tg.sort("tip", "game_id", "team_id")}


@dataclass(frozen=True)
class TeamParams:
    """Elo-style pace x efficiency model; units: points per 100 possessions, possessions per 48."""
    k_eff: float = 0.035     # step on each team's offence / defence per unit efficiency error
    k_pace: float = 0.05     # step on each team's pace per unit possession error
    k_lg: float = 0.004      # step on league means
    rho: float = 0.6         # offseason regression of team states towards 0
    home: float = 2.5        # home edge in efficiency points (split half / half)
    lam_def: float = 0.96    # per-game decay of opponent-allowed stat sums
    n0_def: float = 15.0     # shrinkage (games) of opponent factors


def team_stream(tg: pl.DataFrame, queries: pl.DataFrame, prm: TeamParams = TeamParams()) -> pl.DataFrame:
    """One causal pass over results (visible at tip + 3 h) and queries (game_id, t, home_id,
    away_id, season). Per query: expected possessions per 48 (``poss_exp``), league mean pace,
    each side's expected points, margin and total, and each side's opponent-allowed factors
    (``f_<stat>_home`` = how much the AWAY defence allows relative to the league, applied to home
    players, and vice versa), shrunk towards 1."""
    st: dict[str, dict[str, float]] = {}
    lg = {"mu": 110.0, "pace": 101.0}
    allowed: dict[str, dict[str, float]] = {}
    lg_allowed = {s_: 0.0 for s_ in ("pts", "reb", "ast", "fg3a", "fg3m", "poss")}
    season_of: dict[str, int] = {}
    stat_cols = ("pts", "reb", "ast", "fg3a", "fg3m")

    def team(tid: str, season: int) -> dict[str, float]:
        x = st.setdefault(tid, {"o": 0.0, "d": 0.0, "tau": 0.0})
        if season_of.get(tid) not in (None, season):
            for key in x:
                x[key] *= prm.rho
        season_of[tid] = season
        return x

    games = (tg.filter(pl.col("home")).select("game_id", "season", "tip", "team_id", "opp_id", "pts", "opp_pts", "poss", "game_min")
             .rename({"team_id": "home_id", "opp_id": "away_id", "pts": "home_pts", "opp_pts": "away_pts"}))
    allowed_rows = {(r["game_id"], r["team_id"]): r for r in tg.select(
        "game_id", "team_id", "pts", "reb", "ast", "fg3a", "fg3m", "poss").iter_rows(named=True)}
    ev: list[tuple] = [(r["tip"] + RESULT_LAG, 1, r) for r in games.iter_rows(named=True)]
    ev += [(r["t"], 0, r) for r in queries.iter_rows(named=True)]
    ev.sort(key=lambda e: (e[0], e[1]))
    out = []
    for _, kind, r in ev:
        h, a = team(r["home_id"], r["season"]), team(r["away_id"], r["season"])
        eff_h = lg["mu"] + h["o"] + a["d"] + prm.home / 2
        eff_a = lg["mu"] + a["o"] + h["d"] - prm.home / 2
        pace = lg["pace"] + h["tau"] + a["tau"]
        if kind == 0:
            rec = {"game_id": r["game_id"], "t": r["t"], "poss_exp": pace, "lg_pace": lg["pace"],
                   "mu_home": pace * eff_h / 100, "mu_away": pace * eff_a / 100}
            rec["mu_margin"] = rec["mu_home"] - rec["mu_away"]
            rec["mu_total"] = rec["mu_home"] + rec["mu_away"]
            lgp = max(lg_allowed["poss"], 1e-9)
            for side, defender in (("home", r["away_id"]), ("away", r["home_id"])):
                al = allowed.get(defender)
                for s_ in stat_cols:
                    if al is None or lg_allowed[s_] <= 0:
                        rec[f"f_{s_}_{side}"] = 1.0
                        continue
                    raw = (al[s_] / max(al["poss"], 1e-9)) / (lg_allowed[s_] / lgp)
                    w = al["n"] / (al["n"] + prm.n0_def)
                    rec[f"f_{s_}_{side}"] = 1 + w * (raw - 1)
            out.append(rec)
            continue
        # result: efficiency and pace updates (pace per 48 minutes, efficiency over all minutes)
        poss48 = r["poss"] * 48 / r["game_min"]
        e_h = 100 * r["home_pts"] / r["poss"] - eff_h
        e_a = 100 * r["away_pts"] / r["poss"] - eff_a
        e_p = poss48 - pace
        h["o"] += prm.k_eff * e_h
        a["d"] += prm.k_eff * e_h
        a["o"] += prm.k_eff * e_a
        h["d"] += prm.k_eff * e_a
        h["tau"] += prm.k_pace * e_p
        a["tau"] += prm.k_pace * e_p
        lg["mu"] += prm.k_lg * (e_h + e_a) / 2
        lg["pace"] += prm.k_lg * e_p
        # opponent-allowed sums (per possession, decayed per game)
        for defender, attacker in ((r["home_id"], r["away_id"]), (r["away_id"], r["home_id"])):
            row = allowed_rows.get((r["game_id"], attacker))
            if row is None or any(row[c] is None for c in stat_cols):
                continue
            al = allowed.setdefault(defender, {**{c: 0.0 for c in stat_cols}, "poss": 0.0, "n": 0.0})
            for c in stat_cols:
                al[c] = prm.lam_def * al[c] + row[c]
                lg_allowed[c] = 0.999 * lg_allowed[c] + row[c]
            al["poss"] = prm.lam_def * al["poss"] + r["poss"]
            al["n"] = prm.lam_def * al["n"] + 1
            lg_allowed["poss"] = 0.999 * lg_allowed["poss"] + r["poss"]
    return pl.DataFrame(out, infer_schema_length=None)


def game_queries(games: pl.DataFrame, leads: tuple[int, ...] = LEADS_H) -> pl.DataFrame:
    g = games.select("game_id", "season", "home_id", "away_id", "tip")
    return pl.concat([g.with_columns(floor_hour(pl.col("tip") - pl.duration(hours=h)).alias("t"), pl.lit(h).alias("lead"))
                      for h in leads])


def pos_bucket(pos: str | None) -> str:
    p = (pos or "").upper()
    if "C" in p:
        return "C"
    if p.startswith("G") or p in ("PG", "SG"):
        return "G"
    return "F"


LAMBDAS = (0.94, 0.98, 0.995)   # per-game decay of per-minute rate sums (chosen on dev)
PREV_W = 0.5             # weight of previous-season games in the rate sums


def player_features(boxes: pl.DataFrame, games: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    """As-of features for each query (athlete_id, game_id, team_id, season, tip, t, lead) from
    the athlete's box rows of games whose result was visible by ``t`` (tip + 3 h <= t) and the
    team's schedule (known in advance). Rates are decayed sums (per game played) for each decay
    in :data:`LAMBDAS`; previous-season games weigh :data:`PREV_W`."""
    b = (boxes.join(games.select("game_id", "season", "tip"), on="game_id", how="inner")
         .with_columns(pl.col("min").fill_null(0.0)).sort("athlete_id", "tip"))
    hist: dict[str, dict[str, np.ndarray]] = {}
    for (aid,), part in b.group_by("athlete_id"):
        part = part.sort("tip")
        hist[aid] = {"seen": part["tip"].dt.epoch("s").to_numpy() + RESULT_LAG.total_seconds(),
                     "tip": part["tip"].dt.epoch("s").to_numpy(), "season": part["season"].to_numpy(),
                     "min": part["min"].to_numpy(), "starter": part["starter"].fill_null(False).to_numpy(),
                     "pos": np.array(part["pos"].to_list(), dtype=object),
                     **{c: part[c].fill_null(0.0).to_numpy() for c in (*STATS, "fga", "fta")}}
    sched: dict[str, np.ndarray] = {}
    for side in ("home", "away"):
        for (tid,), part in games.group_by(f"{side}_id"):
            sched.setdefault(tid, np.array([], dtype=np.int64))
            sched[tid] = np.concatenate([sched[tid], part["tip"].dt.epoch("s").to_numpy()])
    sched = {k_: np.sort(v) for k_, v in sched.items()}
    rows = []
    for q in queries.iter_rows(named=True):
        t = q["t"].timestamp()
        tip = q["tip"].timestamp()
        rec: dict[str, Any] = {"athlete_id": q["athlete_id"], "game_id": q["game_id"], "lead": q["lead"]}
        ts = sched.get(q["team_id"], np.array([]))
        prev = ts[ts < tip - 3600]
        rec["rest_days"] = float(min((tip - prev[-1]) / 86400, 5.0)) if len(prev) else 5.0
        rec["b2b"] = bool(len(prev) and tip - prev[-1] < 30 * 3600)
        H = hist.get(q["athlete_id"])
        n = int(np.searchsorted(H["seen"], t, side="right")) if H is not None else 0
        if H is None or n == 0:
            rec.update({"n_hist": 0, "n_season": 0, "pos": None})
            rows.append(rec)
            continue
        played = H["min"][:n] > 0
        idx = np.nonzero(played)[0]
        cur = H["season"][:n] == q["season"]
        cur_idx = np.nonzero(played & cur)[0]
        prev_idx = np.nonzero(played & (H["season"][:n] == q["season"] - 1))[0]
        rec["n_hist"] = len(idx)
        rec["n_season"] = len(cur_idx)
        rec["pos"] = next((p for p in H["pos"][:n][::-1] if p), None)
        if len(idx):
            last = idx[-1]
            rec["started_last"] = bool(H["starter"][last])
            rec["start_rate5"] = float(H["starter"][idx[-5:]].mean())
            rec["m_last"] = float(H["min"][last])
            rec["m_last5"] = float(H["min"][idx[-5:]].mean())
            rec["days_since"] = float((tip - H["tip"][last]) / 86400)
            rec["missed"] = int(((ts > H["tip"][last]) & (ts < tip - 3600) & (ts + RESULT_LAG.total_seconds() <= t)).sum())
            rec["dnp_last"] = bool(not played[-1])
        if len(cur_idx):
            rec["m_season"] = float(H["min"][cur_idx].mean())
            rec["m_cur5"] = float(H["min"][cur_idx[-5:]].mean())
            rec["start_rate_season"] = float(H["starter"][cur_idx].mean())
            for c in ("pts", "reb", "ast"):
                rec[f"{c}_pg"] = float(H[c][cur_idx].mean())
        if len(prev_idx):
            rec["m_prev"] = float(H["min"][prev_idx].mean())
            for c in ("pts", "reb", "ast", "fg3m"):
                rec[f"{c}_pg_prev"] = float(H[c][prev_idx].mean())
        if len(idx):
            w_season = np.where(H["season"][idx] == q["season"], 1.0, PREV_W)
            w_season = np.where(H["season"][idx] < q["season"] - 1, 0.0, w_season)
            ranks = len(idx) - 1 - np.arange(len(idx))
            for lam in LAMBDAS:
                w = lam ** ranks * w_season
                rec[f"M_{lam}"] = float((w * H["min"][idx]).sum())
                for c in (*STATS, "fga", "fta"):
                    rec[f"S_{c}_{lam}"] = float((w * H[c][idx]).sum())
            if len(cur_idx):
                for c in ("pts", "reb", "ast", "fg3a", "fg3m"):
                    rec[f"{c}_per_min_season"] = float(H[c][cur_idx].sum() / max(H["min"][cur_idx].sum(), 1.0))
        rows.append(rec)
    out = pl.DataFrame(rows, infer_schema_length=None)
    opt = ["started_last", "start_rate5", "m_last", "m_last5", "days_since", "missed", "dnp_last", "m_season", "m_cur5",
           "start_rate_season", "m_prev", *[f"{c}_pg" for c in ("pts", "reb", "ast")],
           *[f"{c}_pg_prev" for c in ("pts", "reb", "ast", "fg3m")],
           *[f"{c}_per_min_season" for c in ("pts", "reb", "ast", "fg3a", "fg3m")],
           *[f"M_{lam}" for lam in LAMBDAS], *[f"S_{c}_{lam}" for lam in LAMBDAS for c in (*STATS, "fga", "fta")]]
    return out.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in opt if c not in out.columns])


# ---- minutes, rates, count laws -----------------------------------------------------------------
TRAIN_FROM = datetime(2024, 12, 1, tzinfo=UTC)   # model fitting: Dec 2024 .. Jan 2026 (pre-holdout)
MIN_BUCKETS = (10.0, 20.0, 28.0, 34.0)
GH_X, GH_W = np.polynomial.hermite.hermgauss(10)


def _f(df: pl.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    return df[col].cast(pl.Float64).fill_null(default).to_numpy() if col in df.columns else np.full(df.height, default)


def minutes_design(df: pl.DataFrame) -> np.ndarray:
    """Minutes regressors (missing values filled from the next-best history)."""
    m_last5 = _f(df, "m_last5")
    m_cur5 = np.where(np.isnan(_f(df, "m_cur5")), m_last5, _f(df, "m_cur5"))
    m_season = _f(df, "m_season")
    m_season = np.where(np.isnan(m_season), _f(df, "m_prev"), m_season)
    m_season = np.where(np.isnan(m_season), m_last5, m_season)
    missed = np.clip(_f(df, "missed", 0.0), 0, 5)
    sr5 = _f(df, "start_rate5", 0.0)
    return np.column_stack([
        np.ones(df.height), m_last5, m_cur5, m_season, _f(df, "m_last", 0.0), _f(df, "started_last", 0.0), sr5,
        _f(df, "b2b", 0.0), (missed > 0).astype(float), missed, _f(df, "dnp_last", 0.0),
        (_f(df, "n_season", 0.0) < 3).astype(float), np.abs(_f(df, "mu_margin_team", 0.0)) * sr5,
        np.abs(_f(df, "mu_margin_team", 0.0)) * (1 - sr5)])


@dataclass
class PropModel:
    """Everything fitted on training rows (Dec 2024 - Jan 2026, lead 1 h, the player played)."""
    beta_min: np.ndarray
    sd_min: dict[int, float]
    rookie_min: tuple[float, float]
    lam: float
    m0: float
    prior: dict[tuple[str, bool], dict[str, float]]
    p3_lg: float
    a0: float
    calib: dict[str, float]
    r: dict[str, float]
    adjust: bool = True


def _bucket(m: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(MIN_BUCKETS), m)


def per_minute_rates(df: pl.DataFrame, lam: float, m0: float, prior: dict[tuple[str, bool], dict[str, float]],
                     p3_lg: float, a0: float) -> dict[str, np.ndarray]:
    """Shrunk per-minute rates: (decayed stat sum + m0 x prior) / (decayed minutes + m0); 3P%
    shrunk to the league rate with ``a0`` attempts. Prior by position bucket x regular role."""
    pos = [pos_bucket(p) for p in df["pos"].to_list()] if "pos" in df.columns else ["F"] * df.height
    msea = _f(df, "m_season")
    msea = np.where(np.isnan(msea), _f(df, "m_prev"), msea)
    msea = np.where(np.isnan(msea), _f(df, "m_last5", 0.0), msea)
    role = msea >= 24
    M = _f(df, f"M_{lam}", 0.0)
    out = {}
    for c in ("pts", "reb", "ast", "fg3a"):
        pr = np.array([prior.get((p, bool(rl)), prior.get((p, False), {})).get(c, 0.0) for p, rl in zip(pos, role, strict=True)])
        out[c] = (_f(df, f"S_{c}_{lam}", 0.0) + m0 * pr) / (M + m0)
    out["p3"] = (_f(df, f"S_fg3m_{lam}", 0.0) + a0 * p3_lg) / (_f(df, f"S_fg3a_{lam}", 0.0) + a0)
    return out


def adj_factors(df: pl.DataFrame, adjust: bool) -> dict[str, np.ndarray]:
    """Opponent-allowed x pace factors per stat (1 when ``adjust`` is off)."""
    if not adjust:
        return {c: np.ones(df.height) for c in ("pts", "reb", "ast", "fg3a")}
    pace = _f(df, "poss_exp", 99.0) / _f(df, "lg_pace", 99.0)
    return {c: _f(df, f"f_{c}", 1.0) * pace for c in ("pts", "reb", "ast", "fg3a")}


def nb_logpmf(x: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    from scipy.special import gammaln

    mu = np.clip(mu, 1e-6, None)
    return (gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * np.log(r / (r + mu)) + x * np.log(mu / (r + mu)))


def fit_r(x: np.ndarray, mu: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda lr: -nb_logpmf(x, mu, math.exp(lr)).sum(), bounds=(-2, 6), method="bounded")
    return float(math.exp(res.x))


def fit_prop_model(train: pl.DataFrame, *, adjust: bool = True) -> tuple[PropModel, dict[str, Any]]:
    """Fit the minutes regression, priors, decay / shrinkage (grid, NB log likelihood given the
    actual minutes), mean calibration and NB dispersions on training rows."""
    tr = train.filter(pl.col("n_hist") > 0)
    X, y = minutes_design(tr), tr["min"].to_numpy()
    ok = ~np.isnan(X).any(axis=1)
    beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    pred = X @ beta
    res = y - pred
    b = _bucket(pred)
    sd_min = {int(i): float(np.nanstd(res[ok & (b == i)])) for i in range(len(MIN_BUCKETS) + 1)}
    first = train.filter(pl.col("n_hist") == 0)["min"].to_numpy()
    rookie = (float(first.mean()) if len(first) else 15.0, float(first.std()) if len(first) > 1 else 8.0)
    # priors per position x role from training box rows (role from the season minutes)
    pos = [pos_bucket(p) for p in tr["pos"].to_list()]
    msea = _f(tr, "m_season")
    msea = np.where(np.isnan(msea), _f(tr, "m_prev"), msea)
    msea = np.where(np.isnan(msea), _f(tr, "m_last5", 0.0), msea)
    role = msea >= 24
    prior: dict[tuple[str, bool], dict[str, float]] = {}
    for p_ in ("G", "F", "C"):
        for rl in (False, True):
            mask = np.array([q == p_ for q in pos]) & (role == rl)
            prior[(p_, rl)] = {c: float(tr[c].to_numpy()[mask].sum() / max(y[mask].sum(), 1.0)) for c in ("pts", "reb", "ast", "fg3a")}
    p3_lg = float(tr["fg3m"].sum() / tr["fg3a"].sum())
    adj = adj_factors(tr, adjust)
    grid = []
    for lam in LAMBDAS:
        for m0 in (10.0, 30.0, 100.0, 300.0):
            rates = per_minute_rates(tr, lam, m0, prior, p3_lg, 100.0)
            ll = 0.0
            for c in ("pts", "reb", "ast", "fg3a"):
                mu = rates[c] * y * adj[c]
                mu *= tr[c].sum() / mu.sum()
                ll += float(nb_logpmf(tr[c].to_numpy(), mu, fit_r(tr[c].to_numpy(), mu)).mean())
            grid.append({"lam": lam, "m0": m0, "ll": round(ll, 5)})
    best = max(grid, key=lambda g_: g_["ll"])
    rates = per_minute_rates(tr, best["lam"], best["m0"], prior, p3_lg, 100.0)
    calib, r = {}, {}
    for c in ("pts", "reb", "ast", "fg3a"):
        mu = rates[c] * y * adj[c]
        calib[c] = float(tr[c].sum() / mu.sum())
        r[c] = fit_r(tr[c].to_numpy(), mu * calib[c])
    mu_pra = sum(rates[c] * y * adj[c] * calib[c] for c in ("pts", "reb", "ast"))
    pra = (tr["pts"] + tr["reb"] + tr["ast"]).to_numpy()
    r["pra"] = fit_r(pra, mu_pra)
    model = PropModel(beta, sd_min, rookie, best["lam"], best["m0"], prior, p3_lg, 100.0, calib, r, adjust)
    names = ["const", "m_last5", "m_cur5", "m_season", "m_last", "started_last", "start_rate5", "b2b", "missed>0",
             "missed", "dnp_last", "n_season<3", "|margin|*start", "|margin|*bench"]
    diag = {"rows": tr.height, "min_rmse": round(float(np.sqrt(np.nanmean(res[ok] ** 2))), 3), "sd_min": sd_min,
            "beta_min": dict(zip(names, np.round(beta, 3).tolist(), strict=True)),
            "grid": grid, "chosen": best, "calib": calib, "r": r, "p3_lg": round(p3_lg, 4), "rookie_min": rookie}
    return model, diag


def minutes_law(df: pl.DataFrame, m: PropModel) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes (n x 10) and weights of the minutes law truncated to [1, 48]
    (the market conditions on the player taking the court)."""
    X = minutes_design(df)
    mu = X @ m.beta_min
    sd = np.array([m.sd_min.get(int(b_), 8.0) for b_ in _bucket(np.nan_to_num(mu, nan=15.0))])
    none = (_f(df, "n_hist", 0.0) == 0) | np.isnan(mu)
    mu = np.where(none, m.rookie_min[0], mu)
    sd = np.where(none, m.rookie_min[1], sd)
    nodes = mu[:, None] + math.sqrt(2) * sd[:, None] * GH_X[None, :]
    w = np.tile(GH_W / math.sqrt(math.pi), (df.height, 1))
    w = np.where(nodes < 1, 0.0, w)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    return np.clip(nodes, 1, 48), w


def prop_probs(df: pl.DataFrame, m: PropModel) -> np.ndarray:
    """P(stat >= k) for rows with kind in pts / reb / ast / fg3m / pra and threshold ``k``."""
    from scipy.stats import binom, nbinom

    nodes, w = minutes_law(df, m)
    rates = per_minute_rates(df, m.lam, m.m0, m.prior, m.p3_lg, m.a0)
    adj = adj_factors(df, m.adjust)
    kinds = df["kind"].to_list()
    k = df["k"].to_numpy()
    out = np.full(df.height, np.nan)
    mus = {c: rates[c][:, None] * adj[c][:, None] * m.calib[c] * nodes for c in ("pts", "reb", "ast", "fg3a")}
    for kind in ("pts", "reb", "ast", "pra"):
        idx = np.array([kk == kind for kk in kinds])
        if not idx.any():
            continue
        mu = (mus["pts"] + mus["reb"] + mus["ast"])[idx] if kind == "pra" else mus[kind][idx]
        r = m.r[kind]
        sf = nbinom.sf(k[idx, None] - 1, r, r / (r + np.clip(mu, 1e-6, None)))
        out[idx] = (sf * w[idx]).sum(axis=1)
    idx = np.array([kk == "fg3m" for kk in kinds])
    if idx.any():
        r = m.r["fg3a"]
        a = np.arange(0, 41)
        mu = np.clip(mus["fg3a"][idx], 1e-6, None)                       # (n, nodes)
        pa = nbinom.pmf(a[None, None, :], r, (r / (r + mu))[:, :, None])  # (n, nodes, a)
        p3 = rates["p3"][idx]
        made = binom.sf(k[idx, None] - 1, a[None, :], p3[:, None])         # (n, a)
        out[idx] = ((pa * made[:, None, :]).sum(axis=2) * w[idx]).sum(axis=1)
    return np.clip(out, 1e-4, 1 - 1e-4)


def simple_probs(df: pl.DataFrame, r: dict[str, float]) -> np.ndarray:
    """Baseline: negative binomial around the season-to-date per-game average (previous season
    when the current one has no games; league-ish defaults otherwise), no minutes model."""
    from scipy.stats import nbinom

    k = df["k"].to_numpy()
    kinds = df["kind"].to_list()
    out = np.full(df.height, np.nan)
    base = {"pts": 12.0, "reb": 5.0, "ast": 3.0, "fg3m": 1.3}
    mean = {}
    for c in ("pts", "reb", "ast", "fg3m"):
        cur, prev = _f(df, f"{c}_pg"), _f(df, f"{c}_pg_prev")
        if c == "fg3m":
            cur = _f(df, "fg3m_per_min_season") * _f(df, "m_season")
        mean[c] = np.where(np.isnan(cur), np.where(np.isnan(prev), base[c], prev), cur)
    mean["pra"] = mean["pts"] + mean["reb"] + mean["ast"]
    for kind in ("pts", "reb", "ast", "fg3m", "pra"):
        idx = np.array([kk == kind for kk in kinds])
        if idx.any():
            rr = r.get(kind, 5.0)
            out[idx] = nbinom.sf(k[idx] - 1, rr, rr / (rr + np.clip(mean[kind][idx], 1e-3, None)))
    return np.clip(out, 1e-4, 1 - 1e-4)


def fit_simple_r(train: pl.DataFrame) -> dict[str, float]:
    tr = train.filter(pl.col("n_season") >= 3)
    r = {}
    for c in ("pts", "reb", "ast"):
        r[c] = fit_r(tr[c].to_numpy(), np.clip(_f(tr, f"{c}_pg", 1.0), 0.1, None))
    mu3 = np.clip(_f(tr, "fg3m_per_min_season", 0.0) * _f(tr, "m_season", 0.0), 0.05, None)
    r["fg3m"] = fit_r(tr["fg3m"].to_numpy(), mu3)
    r["pra"] = fit_r((tr["pts"] + tr["reb"] + tr["ast"]).to_numpy(),
                     np.clip(_f(tr, "pts_pg", 1.0) + _f(tr, "reb_pg", 1.0) + _f(tr, "ast_pg", 1.0), 0.5, None))
    return r


# ---- assembling the study frames ----------------------------------------------------------------
TEAM_GRID = tuple(TeamParams(k_eff=ke, k_pace=kp) for ke in (0.02, 0.035, 0.05) for kp in (0.03, 0.06))


def team_scores(team_pred: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    return (team_pred.join(games.select("game_id", "tip", "home_score", "away_score"), on="game_id")
            .with_columns((pl.col("home_score") - pl.col("mu_home")).alias("e_home"),
                          (pl.col("away_score") - pl.col("mu_away")).alias("e_away"),
                          (pl.col("home_score") + pl.col("away_score") - pl.col("mu_total")).alias("e_total")))


def fit_team(inp: dict[str, Any]) -> tuple[TeamParams, pl.DataFrame, dict[str, Any]]:
    """Pick the team-model steps on training games (lead 1 h, Dec 2024 - Jan 2026) by RMSE of
    team points; Normal SDs of team points and game totals from the same residuals."""
    games = inp["games"].filter(pl.col("completed") & (pl.col("status") == "STATUS_FINAL"))
    allg = inp["games"].filter(pl.col("tip") >= datetime(2024, 10, 1, tzinfo=UTC))
    q = game_queries(allg)
    grid = []
    best = None
    for prm in TEAM_GRID:
        pred = team_stream(inp["team_games"], q.filter(pl.col("lead") == 1), prm)
        sc = team_scores(pred, games).filter((pl.col("tip") >= TRAIN_FROM) & (pl.col("tip") < HOLDOUT_FROM))
        e = np.concatenate([sc["e_home"].to_numpy(), sc["e_away"].to_numpy()])
        rmse = float(np.sqrt((e ** 2).mean()))
        grid.append({"k_eff": prm.k_eff, "k_pace": prm.k_pace, "rmse_team": round(rmse, 3)})
        if best is None or rmse < best[1]:
            best = (prm, rmse)
    prm = best[0]
    pred = team_stream(inp["team_games"], q, prm).join(q.select("game_id", "t", "lead"), on=["game_id", "t"], how="inner")
    sc = team_scores(pred, games).filter((pl.col("tip") >= TRAIN_FROM) & (pl.col("tip") < HOLDOUT_FROM) & (pl.col("lead") == 1))
    e_team = np.concatenate([sc["e_home"].to_numpy(), sc["e_away"].to_numpy()])
    laws = {"sd_team": float(e_team.std()), "bias_team": float(e_team.mean()), "sd_total": float(sc["e_total"].std()),
            "bias_total": float(sc["e_total"].mean()), "n": sc.height, "grid": grid,
            "chosen": {"k_eff": prm.k_eff, "k_pace": prm.k_pace}}
    return prm, pred, laws


def player_queries(boxes: pl.DataFrame, games: pl.DataFrame, frm: datetime) -> pl.DataFrame:
    """One query per listed player x game x lead (games tipping from ``frm``)."""
    g = games.filter(pl.col("tip") >= frm).select("game_id", "season", "tip", "home_id", "away_id")
    b = boxes.select("game_id", "athlete_id", "team_id").join(g, on="game_id", how="inner").with_columns(
        pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.lit("home")).otherwise(pl.lit("away")).alias("team_side"))
    return pl.concat([b.with_columns(floor_hour(pl.col("tip") - pl.duration(hours=h)).alias("t"), pl.lit(h).alias("lead"))
                      for h in LEADS_H])


def build_features(s, *, refresh: bool = False) -> dict[str, Any]:
    """Team predictions and player features for every listed player-game from Nov 2024 (cached)."""
    cd = cache_dir(s)
    inp = load_box_inputs(s)
    fpath, tpath, lpath = cd / "features.parquet", cd / "team_pred.parquet", cd / "team_laws.json"
    if fpath.exists() and tpath.exists() and lpath.exists() and not refresh:
        return {**inp, "feats": pl.read_parquet(fpath), "team_pred": pl.read_parquet(tpath),
                "team_laws": json.loads(lpath.read_text())}
    t0 = time.monotonic()
    _prm, team_pred, laws = fit_team(inp)
    print(f"  team model {time.monotonic() - t0:.0f}s: {laws['chosen']} sd_team {laws['sd_team']:.2f} sd_total {laws['sd_total']:.2f}", flush=True)
    q = player_queries(inp["boxes"], inp["games"], datetime(2024, 11, 1, tzinfo=UTC))
    feats = player_features(inp["boxes"], inp["games"], q)
    print(f"  player features {feats.height} rows, {time.monotonic() - t0:.0f}s", flush=True)
    side = q.select("athlete_id", "game_id", "lead", "team_id", "team_side", "tip", "t", "season")
    feats = feats.join(side, on=["athlete_id", "game_id", "lead"], how="left")
    tp = team_pred.select("game_id", "lead", "poss_exp", "lg_pace", "mu_margin", *[c for c in team_pred.columns if c.startswith("f_")])
    feats = feats.join(tp, on=["game_id", "lead"], how="left")
    home = pl.col("team_side") == "home"
    feats = feats.with_columns(
        pl.when(home).then(pl.col("mu_margin")).otherwise(-pl.col("mu_margin")).alias("mu_margin_team"),
        *[pl.when(home).then(pl.col(f"f_{c}_home")).otherwise(pl.col(f"f_{c}_away")).alias(f"f_{c}")
          for c in ("pts", "reb", "ast", "fg3a", "fg3m")])
    feats = feats.drop([c for c in feats.columns if c.endswith("_home") or c.endswith("_away")])
    actual = inp["boxes"].select("athlete_id", "game_id", pl.col("min").fill_null(0.0), *[pl.col(c).fill_null(0.0) for c in STATS], "starter", "dnp")
    feats = feats.join(actual, on=["athlete_id", "game_id"], how="left")
    for f, p in ((feats, fpath), (team_pred, tpath)):
        f.write_parquet(p)
    lpath.write_text(json.dumps(laws, default=str))
    return {**inp, "feats": feats, "team_pred": team_pred, "team_laws": laws}


def training_rows(feats: pl.DataFrame) -> pl.DataFrame:
    return feats.filter((pl.col("lead") == 1) & (pl.col("min") > 0) & (pl.col("tip") >= TRAIN_FROM) & (pl.col("tip") < HOLDOUT_FROM))


def tier_expr() -> pl.Expr:
    """Player tier from as-of season numbers (previous season when the current has < 3 games):
    star = >= 20 points per game; starter = starts most games or >= 26 minutes; else bench."""
    ppg = pl.when(pl.col("n_season").fill_null(0) >= 3).then(pl.col("pts_pg")).otherwise(pl.col("pts_pg_prev"))
    mins = pl.when(pl.col("n_season").fill_null(0) >= 3).then(pl.col("m_season")).otherwise(pl.col("m_prev"))
    return (pl.when(ppg >= 20).then(pl.lit("star"))
            .when((pl.col("start_rate_season").fill_null(0) >= 0.5) | (mins >= 26)).then(pl.lit("starter"))
            .otherwise(pl.lit("bench")))


def leg_rows(legs: pl.DataFrame, feats: pl.DataFrame, team_pred: pl.DataFrame, laws: dict[str, Any],
             models: dict[str, Any]) -> pl.DataFrame:
    """Each leg at each lead with every model variant's probability (``p_<variant>``)."""
    from scipy.stats import norm

    base = pl.concat([legs.with_columns(floor_hour(pl.col("tip") - pl.duration(hours=h)).alias("t"), pl.lit(h).alias("lead"))
                      for h in LEADS_H]).with_columns(floor_hour(pl.col("tip")).alias("end"),
                                                      pl.col("game_id").alias("event_ticker_game"))
    base = base.filter(pl.col("open_time") <= pl.col("t"))
    props = base.filter(pl.col("kind").is_in(list(PROP_SERIES.values())) & pl.col("athlete_id").is_not_null())
    props = props.join(feats.drop("team_id", "team_side", "tip", "t", "season", "min", *STATS, "starter", "dnp"),
                       on=["athlete_id", "game_id", "lead"], how="inner")
    props = props.with_columns(tier_expr().alias("tier"))
    for name, mdl in models.items():
        p = simple_probs(props, mdl) if name == "simple" else prop_probs(props, mdl)
        props = props.with_columns(pl.Series(f"p_{name}", p))
    teams = base.filter(pl.col("kind").is_in(list(TEAM_SERIES.values()))).join(
        team_pred.select("game_id", "lead", "mu_home", "mu_away", "mu_total"), on=["game_id", "lead"], how="inner")
    mu = (pl.when(pl.col("kind") == "total").then(pl.col("mu_total") + laws["bias_total"])
          .when(pl.col("team_side") == "home").then(pl.col("mu_home") + laws["bias_team"])
          .otherwise(pl.col("mu_away") + laws["bias_team"]))
    sd = pl.when(pl.col("kind") == "total").then(pl.lit(laws["sd_total"])).otherwise(pl.lit(laws["sd_team"]))
    teams = teams.with_columns(mu.alias("mu_pts"), sd.alias("sd_pts"), pl.lit("team").alias("tier"))
    pt = 1 - norm.cdf((teams["k"].to_numpy() - 0.5 - teams["mu_pts"].to_numpy()) / teams["sd_pts"].to_numpy())
    teams = teams.with_columns(*[pl.Series(f"p_{name}", np.clip(pt, 1e-4, 1 - 1e-4)) for name in models])
    return pl.concat([props, teams], how="diagonal_relaxed").with_columns(pl.col("game_id").alias("event"))


# ---- evaluation ---------------------------------------------------------------------------------
VARIANTS = ("full", "noadj", "simple")


def band3(spr: pl.Expr) -> pl.Expr:
    """Quoted spread at t: the round-3 bands."""
    return pl.when(spr <= 0.055).then(pl.lit("a <=5c")).when(spr <= 0.155).then(pl.lit("b 6-15c")).otherwise(pl.lit("c >15c"))


def with_quotes(rows: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    from albert.research.fundamental import attach_quotes

    c = candles.select("ticker", "end_ts", pl.col("bid").alias("yes_bid_close"), pl.col("ask").alias("yes_ask_close"))
    q = attach_quotes(rows, c)
    return q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
                          split_of(pl.col("kind"), pl.col("tip")).alias("holdout")).with_columns(band3(pl.col("spread")).alias("band"))


def eval_table(df: pl.DataFrame, by: list[str], *, min_dev: int = 40) -> pl.DataFrame:
    """Harness ``evaluate`` per segment (dev blend c and its t, holdout refit, taker at dev θ).
    Team totals use their own holdout start (:data:`TT_HOLDOUT_FROM`)."""
    from albert.research.fundamental import evaluate
    from albert.research.ncaaf import _summary_row

    rows = []
    for key, part in df.group_by(*by, maintain_order=True):
        if part.filter(~pl.col("holdout")).height < min_dev:
            continue
        hf = TT_HOLDOUT_FROM if set(part["kind"].unique()) == {"team_total"} else HOLDOUT_FROM
        r = _summary_row(evaluate(part, hf), **dict(zip(by, key, strict=True)))
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).sort(by) if rows else pl.DataFrame()


def variant_choice(df: pl.DataFrame) -> tuple[dict[str, str], pl.DataFrame]:
    """Per kind: the variant with the lowest dev log loss (both leads pooled)."""
    from albert.research.fundamental import _scores

    rows = []
    for (kind,), part in df.filter(~pl.col("holdout")).group_by("kind"):
        y = part["y"].to_numpy().astype(float)
        r = {"kind": kind, "n_dev": part.height, "logloss_mid": round(_scores(part["mid"].to_numpy(), y)[1], 5)}
        for v in VARIANTS:
            r[f"logloss_{v}"] = round(_scores(part[f"p_{v}"].to_numpy(), y)[1], 5)
        rows.append(r)
    tab = pl.DataFrame(rows).sort("kind")
    choice = {r["kind"]: min(VARIANTS, key=lambda v, r=r: r[f"logloss_{v}"]) for r in rows}
    return choice, tab


def fit_mid_recal(df: pl.DataFrame, cells: list[str]) -> dict[tuple, np.ndarray]:
    """Dev-only recalibration of the mid alone: logit P = a + b logit(mid) (the control fair)."""
    from albert.research.fundamental import fit_logistic, logit

    out = {}
    for key, part in df.filter(~pl.col("holdout")).group_by(*cells):
        if part.height < 100:
            continue
        X = np.column_stack([np.ones(part.height), logit(part["mid"].to_numpy())])
        out[tuple(key)] = fit_logistic(X, part["y"].to_numpy().astype(float), part["event"].to_numpy())[0]
    return out


def apply_mid_recal(df: pl.DataFrame, coefs: dict[tuple, np.ndarray], cells: list[str]) -> pl.Series:
    from albert.research.fundamental import logit

    X = np.column_stack([np.ones(df.height), logit(df["mid"].to_numpy())])
    keys = df.select(cells).rows()
    z = np.array([X[i] @ coefs[k] if k in coefs else np.nan for i, k in enumerate(keys)])
    return pl.Series("p_fair", 1 / (1 + np.exp(-z))).fill_nan(None)


def fair_values(df: pl.DataFrame) -> pl.DataFrame:
    """``p_fair_blend`` = dev blend of model and mid per kind x lead (pooled lead when a cell has
    < 150 dev rows); ``p_fair_mid`` = dev recalibrated mid per kind x lead (the control)."""
    from albert.research.cbb import apply_blend, fit_blends

    coefs = fit_blends(df, ["kind", "lead"])
    fb = fit_blends(df, ["lead"], min_rows=50)
    b = apply_blend(df, coefs, ["kind", "lead"], fb, ["lead"]).rename({"p_fair": "p_fair_blend"})
    rc = fit_mid_recal(df, ["kind", "lead"])
    return b.with_columns(apply_mid_recal(b, rc, ["kind", "lead"]).alias("p_fair_mid"))


def ohlc_of(candles: pl.DataFrame) -> pl.DataFrame:
    return candles.filter((pl.col("volume").fill_null(0) > 0) & pl.col("price_low").is_not_null()).select(
        "ticker", "end_ts", "price_low", "price_high", "volume")


def maker_runs(df: pl.DataFrame, candles: pl.DataFrame, attrs: list[str]) -> dict[str, pl.DataFrame]:
    """Trade-through maker fills (hourly-candle trade ranges in (t, tip]) around the model blend and
    around the recalibrated mid."""
    from albert.research.cbb import candle_extremes, maker_fills_all

    ext = candle_extremes(df, ohlc_of(candles))
    out = {}
    for name in ("blend", "mid"):
        d = df.with_columns(pl.col(f"p_fair_{name}").alias("p_fair"))
        out[name] = maker_fills_all(d, None, attrs, ext=ext, rules=("through",))
    return out


def validate_fill_proxy(s, df: pl.DataFrame, candles: pl.DataFrame, n: int = 150) -> dict[str, Any]:
    """Compare candle-range fills with fills from the real prints (taker side known) on a seeded
    sample of lead-1 h rows: agreement of the ``filled`` flag over margins 0-5c, both fairs."""
    from albert.research.cbb import candle_extremes, maker_fills_all, window_extremes
    from albert.research.fundamental import fetch_trades

    smp = (df.filter(pl.col("lead") == 1).with_columns(_hash_rank(["ticker"], 99).alias("h")).sort("h").head(n).drop("h"))
    trades = fetch_trades(nba_settings(s), smp.select("ticker", pl.col("t").alias("t_from"), pl.col("end").alias("t_to")),
                          name=NAME, workers=2)
    out: dict[str, Any] = {"legs": smp.height, "prints": trades.height}
    if not trades.height:
        return out
    e_c = candle_extremes(smp, ohlc_of(candles))
    e_t = window_extremes(smp, trades)
    for name in ("blend", "mid"):
        d = smp.with_columns(pl.col(f"p_fair_{name}").alias("p_fair"))
        a = maker_fills_all(d, None, [], ext=e_c, rules=("through",), margins=(0.0, 0.02, 0.05))
        b = maker_fills_all(d, None, [], ext=e_t, rules=("through",), margins=(0.0, 0.02, 0.05))
        j = a.join(b.select("ticker", "t", "side", "margin", pl.col("filled").alias("filled_prints")),
                   on=["ticker", "t", "side", "margin"])
        out[f"{name}_orders"] = j.height
        out[f"{name}_agree"] = round(float((j["filled"] == j["filled_prints"]).mean()), 4) if j.height else None
        out[f"{name}_fills_candle"] = int(j["filled"].sum())
        out[f"{name}_fills_prints"] = int(j["filled_prints"].sum())
    return out


def maker_tables(F: dict[str, pl.DataFrame], by: list[str]) -> pl.DataFrame:
    """cbb.maker_table for the model-blend fair and the plain-mid control side by side."""
    from albert.research.cbb import maker_table

    out = []
    for name, f in F.items():
        t = maker_table(f, by, gates=(0.0, 0.03))
        if t.height:
            out.append(t.with_columns(pl.lit(name).alias("fair")).select("fair", *t.columns))
    return pl.concat(out, how="diagonal_relaxed").sort(*by, "gate", "sides", "fair") if out else pl.DataFrame()


def study_frames(s) -> dict[str, Any]:
    """Features, fitted models, and every sampled leg x lead with quotes and all variants."""
    data = build_features(s)
    feats = data["feats"]
    train = training_rows(feats)
    t0 = time.monotonic()
    full, diag_full = fit_prop_model(train, adjust=True)
    noadj, diag_noadj = fit_prop_model(train, adjust=False)
    simple_r = fit_simple_r(train)
    print(f"  prop models fitted {time.monotonic() - t0:.0f}s: {diag_full['chosen']} min_rmse {diag_full['min_rmse']}", flush=True)
    models: dict[str, Any] = {"full": full, "noadj": noadj, "simple": simple_r}
    games = pl.read_parquet(cache_dir(s) / "games.parquet")
    legs = legs_frame(kalshi_markets(s), games, data["boxes"])
    out = {"data": data, "models": models, "diag": {"full": diag_full, "noadj": diag_noadj, "simple_r": simple_r},
           "legs": legs}
    for tag in ("hist", "smoke"):
        cp = cache_dir(s) / f"candles_{tag}.parquet"
        if not cp.exists():
            continue
        candles = pl.read_parquet(cp)
        done = set(pl.read_parquet(cache_dir(s) / f"candles_{tag}_done.parquet")["ticker"])
        lg = legs.filter(pl.col("ticker").is_in(list(done)) & pl.col("y").is_not_null())
        rows = leg_rows(lg, feats, data["team_pred"], data["team_laws"], models)
        # settlement check: Kalshi result vs ESPN box (props) / final score (team legs)
        act = feats.filter(pl.col("lead") == 1).select("athlete_id", "game_id", *STATS)
        rows = rows.join(act.rename({c: f"act_{c}" for c in STATS}), on=["athlete_id", "game_id"], how="left")
        rows = rows.with_columns((pl.col("act_pts") + pl.col("act_reb") + pl.col("act_ast")).alias("act_pra"))
        gs = games.select("game_id", "home_score", "away_score")
        rows = rows.join(gs, on="game_id", how="left")
        stat = (pl.when(pl.col("kind") == "pts").then(pl.col("act_pts")).when(pl.col("kind") == "reb").then(pl.col("act_reb"))
                .when(pl.col("kind") == "ast").then(pl.col("act_ast")).when(pl.col("kind") == "fg3m").then(pl.col("act_fg3m"))
                .when(pl.col("kind") == "pra").then(pl.col("act_pra"))
                .when(pl.col("kind") == "total").then(pl.col("home_score") + pl.col("away_score"))
                .when(pl.col("team_side") == "home").then(pl.col("home_score")).otherwise(pl.col("away_score")))
        rows = rows.with_columns((stat >= pl.col("k")).cast(pl.Int8).alias("y_espn"))
        out[tag] = with_quotes(rows, candles)
        out[f"{tag}_candles"] = candles
    return out


def summarize_settlement(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("lead") == 1).group_by("kind").agg(
        pl.len().alias("legs"), (pl.col("y") == pl.col("y_espn")).mean().alias("kalshi_eq_espn"),
        pl.col("y_espn").is_null().mean().alias("no_box")).sort("kind")


def run_study(s, *, validate: bool = True) -> Path:
    from albert.reports import Report

    t0 = time.monotonic()
    F = study_frames(s)
    df = F["hist"]
    # legs whose Kalshi settlement disagrees with the ESPN box are dropped (stat corrections /
    # matching errors), counted in the report
    settle = summarize_settlement(df)
    df = df.filter(pl.col("y_espn").is_null() | (pl.col("y") == pl.col("y_espn")))
    choice, vtab = variant_choice(df)
    df = df.with_columns(pl.coalesce([pl.when(pl.col("kind") == k_).then(pl.col(f"p_{v}")) for k_, v in choice.items()]).alias("p"))
    props = df.filter(pl.col("kind").is_in(list(PROP_SERIES.values())))
    pooled = props.with_columns(pl.lit("props").alias("seg"))
    main_tab = eval_table(df, ["kind", "lead"])
    pooled_tab = eval_table(pooled, ["seg", "lead"])
    band_tab = eval_table(pooled, ["lead", "band"])
    priced_tab = eval_table(pooled.with_columns(pl.when(pl.col("spread") <= 0.155).then(pl.lit("priced <=15c"))
                                                .otherwise(pl.lit("wide >15c")).alias("book")), ["lead", "book"])
    band_kind = eval_table(df.filter(pl.col("lead") == 1), ["kind", "band"])
    tier_tab = eval_table(pooled, ["lead", "tier"])
    tier_band = eval_table(pooled.filter(pl.col("lead") == 1), ["tier", "band"])
    team_band = eval_table(df.filter(pl.col("kind").is_in(list(TEAM_SERIES.values()))), ["kind", "band"])
    from albert.research.cbb import price_band

    calib = (props.filter(pl.col("lead") == 1).with_columns(price_band(pl.col("mid")).alias("price"))
             .group_by("holdout", "price").agg(pl.len().alias("n"), pl.col("mid").mean().round(4).alias("mean_mid"),
                                               pl.col("p").mean().round(4).alias("mean_model"),
                                               pl.col("y").mean().round(4).alias("mean_y"),
                                               pl.col("spread").median().alias("med_spread"))
             .sort("holdout", "price"))
    # maker: fair = dev blend vs the dev-recalibrated mid (control)
    fv = fair_values(df)
    mk = maker_runs(fv, F["hist_candles"], ["kind", "lead", "band", "tier"])
    mk_pool = {k_: f.filter(pl.col("kind").is_in(list(PROP_SERIES.values()))) for k_, f in mk.items()}
    maker_lead = maker_tables(mk_pool, ["lead"])
    maker_band = maker_tables({k_: f.filter(pl.col("lead") == 1) for k_, f in mk_pool.items()}, ["band"])
    maker_tier = maker_tables({k_: f.filter(pl.col("lead") == 1) for k_, f in mk_pool.items()}, ["tier"])
    maker_kind = maker_tables({k_: f.filter(pl.col("lead") == 1) for k_, f in mk.items()}, ["kind"])
    val = (validate_fill_proxy(s, fv.filter(pl.col("kind").is_in(list(PROP_SERIES.values()))), F["hist_candles"])
           if validate else {})
    hold = fv.filter(pl.col("holdout")).select("ticker", "t", "lead", "kind", pl.col("p_fair_blend").alias("p_fair"), "mid", "p", "y")
    hold.write_parquet(s.data_dir / "cache" / f"fundamental_{NAME}_holdout_preds.parquet")
    smoke = smoke_tables(F, df, choice) if "smoke" in F else {}
    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    lead_counts = df.group_by("kind", "lead", "holdout").len().sort("kind", "lead", "holdout")
    rep.json("Run", {"seconds": round(time.monotonic() - t0), "rows": df.height, "games": df["event"].n_unique(),
                     "variant_choice_dev": choice, "team_laws": {k_: v for k_, v in F["data"]["team_laws"].items() if k_ != "grid"},
                     "fill_proxy_validation": val})
    rep.text("Data, as-of timing, caveats", "\n".join([
        "* Markets: every settled leg of KXNBAPTS / REB / AST / 3PT / PRA / TEAMTOTAL / TOTAL (historical + live endpoints); "
        "candles for a seeded volume-blind sample (see module doc), fetched in hash order (any prefix is a random subsample): "
        f"{len(set(df['ticker']))} legs used here. Smoke: every 2026-27 leg settled by Nov 4 (live batch candles).",
        "* Box scores: ESPN site API (site.api.espn.com serves no robots.txt; www.espn.com's robots.txt disallows the "
        "`anthropic-ai` token -- review before reuse). A game's box enters the model at tip + 3 h; decision times are "
        "floor-hour(tip - 24 h / 1 h); team schedule (rest, back-to-back) is known in advance.",
        "* No injury / inactive / starting-lineup information (ESPN's injury block is today's list, not the pre-tip one): "
        "the model knows only previous games. The market sees the official injury report.",
        "* Legs settled at a fair price because the player never took the court (result 'scalar', ~3 %) are dropped; "
        "the price at t is then approximately P(stat >= k | plays), which is what the model predicts.",
        "* Quotes: hourly candle bid/ask close at t. Maker fills: hourly traded low/high in (t, last whole hour before tip] "
        "(trade-through rule; validated against real prints in 'fill_proxy_validation').",
        "* KXNBATEAMTOTAL lists from Feb 2026: its dev = Feb-Mar, holdout = Apr-Jun 2026. KXNBAPRA lists from Oct 2026 "
        "(smoke only, priced with the pts variant choice and the pooled blend).",
    ]))
    rep.table("Rows by kind x lead x split (sampled legs with a quote at t)", lead_counts, max_rows=40)
    rep.table("Settlement check: Kalshi result vs ESPN box (lead 1 h rows)", settle)
    rep.json("Prop model fit (training Dec 2024 - Jan 2026)", {k_: v for k_, v in F["diag"]["full"].items()})
    rep.table("Variant choice on dev (log loss vs outcome; lower is better)", vtab)
    rep.table("Main: per kind x lead (blend c on dev with t by game; holdout refit; taker at dev θ)", main_tab, save_as="main", max_rows=40)
    rep.table("Props pooled per lead", pooled_tab)
    rep.table("Props pooled: by quoted spread band", band_tab, save_as="band")
    rep.table("Props pooled: priced (<= 15c) vs wide books", priced_tab, save_as="priced")
    rep.table("By kind x spread band (lead 1 h)", band_kind, save_as="band_kind", max_rows=40)
    rep.table("Props pooled: by player tier", tier_tab, save_as="tier")
    rep.table("Props: tier x spread band (lead 1 h)", tier_band, max_rows=40)
    rep.table("Team / game totals by spread band", team_band)
    rep.table("Props calibration by price band (lead 1 h): mid vs model vs outcome", calib)
    rep.table("Maker, props pooled, by lead: fair = model blend vs recalibrated mid (control)", maker_lead, save_as="maker_lead", max_rows=40)
    rep.table("Maker, props, lead 1 h, by spread band", maker_band, save_as="maker_band", max_rows=40)
    rep.table("Maker, props, lead 1 h, by tier", maker_tier, save_as="maker_tier", max_rows=40)
    rep.table("Maker, lead 1 h, by kind", maker_kind, save_as="maker_kind", max_rows=60)
    for i, (name, tab) in enumerate(smoke.items()):
        rep.table(f"Smoke test 2026-27 (first weeks): {name}", tab, max_rows=40, save_as=f"smoke_{i}")
    return rep.save()


def smoke_tables(F: dict[str, Any], dev_df: pl.DataFrame, choice: dict[str, str]) -> dict[str, pl.DataFrame]:
    """Frozen model on the 2026-27 season's first weeks: dev (last season, holdout=False) fits
    the blend; the smoke rows play the holdout."""
    from albert.research.fundamental import evaluate
    from albert.research.ncaaf import _summary_row

    sm = F["smoke"].filter(pl.col("y_espn").is_null() | (pl.col("y") == pl.col("y_espn")))
    sm = sm.with_columns(pl.coalesce([pl.when(pl.col("kind") == k_).then(pl.col(f"p_{v}")) for k_, v in
                                      {**choice, "pra": choice.get("pts", "full")}.items()]).alias("p"))
    dev = dev_df.filter(~pl.col("holdout"))
    rows, bands = [], []
    for (kind, lead), part in sm.group_by("kind", "lead", maintain_order=True):
        d = dev.filter((pl.col("kind") == kind) & (pl.col("lead") == lead))
        if kind == "pra":
            d = dev.filter(pl.col("kind").is_in(["pts", "reb", "ast"]) & (pl.col("lead") == lead))
        both = pl.concat([d, part.with_columns(pl.lit(True).alias("holdout"))], how="diagonal_relaxed")
        rows.append(_summary_row(evaluate(both, SMOKE_FROM), kind=kind, lead=lead, n_smoke=part.height))
    for (band,), part in sm.filter(pl.col("lead") == 1).group_by("band", maintain_order=True):
        y = part["y"].to_numpy().astype(float)
        bands.append({"band": band, "n": part.height, "brier_model": float(((part["p"].to_numpy() - y) ** 2).mean()),
                      "brier_mid": float(((part["mid"].to_numpy() - y) ** 2).mean())})
    # blend refit on the smoke rows themselves, per spread band (props pooled, lead 1 h)
    from albert.research.fundamental import fit_logistic, logit

    refit = []
    sp = sm.filter(pl.col("kind").is_in(list(PROP_SERIES.values())) & (pl.col("lead") == 1))
    for by in (["band"], ["tier"], ["tier", "band"]):
        for key, part in sp.group_by(*by, maintain_order=True):
            if part.height < 200:
                continue
            X = np.column_stack([np.ones(part.height), logit(part["mid"].to_numpy()), logit(part["p"].to_numpy())])
            y = part["y"].to_numpy().astype(float)
            beta, tt = fit_logistic(X, y, part["event"].to_numpy())
            refit.append({"segment": " / ".join(str(k_) for k_ in key), "n": part.height, "games": part["event"].n_unique(),
                          "brier_model": round(float(((part["p"].to_numpy() - y) ** 2).mean()), 5),
                          "brier_mid": round(float(((part["mid"].to_numpy() - y) ** 2).mean()), 5),
                          "b_mid": round(float(beta[1]), 3), "c_model": round(float(beta[2]), 3), "t_c": round(float(tt[2]), 2)})
    # maker on the smoke rows: fair values fitted on last season's dev rows, margins chosen there
    both = pl.concat([dev, sm.with_columns(pl.lit(True).alias("holdout"))], how="diagonal_relaxed")
    fv = fair_values(both)
    candles = pl.concat([F["hist_candles"], F["smoke_candles"]], how="diagonal_relaxed")
    mk = maker_runs(fv, candles, ["kind", "lead", "band", "tier"])
    mk = {k_: f.filter(pl.col("kind").is_in(["pts", "reb", "ast", "fg3m"])) for k_, f in mk.items()}
    return {"per kind x lead (dev = last season Oct-Jan; smoke = holdout)": pl.DataFrame(rows, infer_schema_length=None).sort("kind", "lead"),
            "Brier by spread band (lead 1 h)": pl.DataFrame(bands).sort("band") if bands else pl.DataFrame(),
            "blend refit on smoke rows by spread band / tier (props, lead 1 h)": pl.DataFrame(refit) if refit else pl.DataFrame(),
            "maker by lead (props; holdout = smoke)": maker_tables(mk, ["lead"]),
            "maker by spread band (props, lead 1 h; holdout = smoke)": maker_tables({k_: f.filter(pl.col("lead") == 1) for k_, f in mk.items()}, ["band"]),
            "maker by tier (props, lead 1 h; holdout = smoke)": maker_tables({k_: f.filter(pl.col("lead") == 1) for k_, f in mk.items()}, ["tier"])}


def main(stage: str = "fetch", workers: str = "2") -> None:
    s = Settings()
    nw = min(int(workers), 2)
    if stage == "markets":
        m = kalshi_markets(s, refresh=True)
        print(m.group_by("series_ticker").len())
    elif stage == "espn":
        g = fetch_games(s)
        g.write_parquet(cache_dir(s) / "games.parquet")
        print(g.group_by("season", "season_type").len().sort("season", "season_type"))
        p, t = fetch_boxes(s, g.filter(pl.col("season_type").is_in([2, 3, 5]) & (pl.col("comp_type").fill_null("") != "ALLSTAR")))
        print(p.height, t.height)
    elif stage in ("candles", "smoke_candles"):
        games = pl.read_parquet(cache_dir(s) / "games.parquet")
        legs = legs_frame(kalshi_markets(s), games, None)
        print(f"legs matched to ESPN games: {legs.height}", flush=True)
        if stage == "candles":
            pick = sample_legs(legs)
            pick = pick.with_columns(_hash_rank(["ticker"], 7).alias("h")).sort("h").drop("h")   # interleave cells
            print(pick.group_by("kind", "holdout").len().sort("kind", "holdout"), flush=True)
            fetch_candles(s, pick, tag="hist", workers=nw)
        else:
            live = legs.filter((pl.col("tip") >= SMOKE_FROM) & pl.col("y").is_not_null()
                               & (pl.col("open_time") <= pl.col("tip") - pl.duration(hours=1)))
            print(live.group_by("kind").len(), flush=True)
            fetch_candles(s, live, tag="smoke", workers=nw)
    elif stage == "features":
        d = build_features(s, refresh=True)
        print(d["feats"].height, d["team_laws"])
    elif stage in ("study", "study_noval"):
        print(run_study(s, validate=stage == "study"))


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
