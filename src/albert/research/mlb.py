"""Sabermetric MLB model vs Kalshi (study ``mlb``): do fundamentals add information to Kalshi MLB
prices (moneyline, run line, game totals, first-5-innings totals, inning totals)?

Data (public MLB Stats API, statsapi.mlb.com/api/v1, cached under ``data/cache/mlb/``):
schedules with linescores (inning-by-inning runs, the settlement source for every series used
here), boxscores (team batting lines, every pitcher's line, starter = first pitcher listed),
season pitching/hitting totals (priors), people (throwing hand).

As-of rules (no look-ahead):
* Every rate used for a game on ET date ``d`` comes from games with ``officialDate < d`` (stats
  through the previous day) plus full earlier seasons. Park factors for season ``S`` come from
  seasons before ``S`` only. Bullpen workload uses the previous three days' boxscores.
* Starting pitcher: the model uses the pitcher who actually started (boxscore). Historical
  "probable pitcher as of the day before" is not served by the API (the schedule shows the
  actual starter for finished games). It is verified against Kalshi's own pitcher-strikeout
  markets (KXMLBKS), whose titles name the expected starter and which open around midnight ET
  of game day: a game is tagged ``starter_public`` when both teams' starters had a KXMLBKS
  market that opened before the decision time. Results are reported for that subset as well.
* Lineups (variant ``sp_lineup``, pre1h only): the nine starters from each player's own boxscore
  ``battingOrder`` code (100, 200, ...). Clubs post lineups roughly 2-5 h before first pitch, so
  they are public at first pitch - 1 h (rare late scratches are the residual risk); they are not
  used at 10:00 ET. The team-level ``battingOrder`` list is the *final* lineup (substitutes
  included) and leaked the game's course (blowouts bring in bench players); it is not used.
* Weather: the boxscore's weather is observed at first pitch, so it is NOT as-of. It appears
  only in the labelled diagnostic variant ``sp_lineup_wx`` (dev only, never on the holdout).
* Decision times: game-day morning (10:00 ET = 14:00 UTC, all games here are EDT) and one hour
  before the scheduled first pitch (MLB ``gameDate``). Quotes are the last hourly candle that
  closed at or before the decision time (``fundamental.attach_quotes``).

Reproduce: ``python -m albert.research.mlb fetch`` (MLB data, cached), ``python -m
albert.research.mlb quotes live`` then ``quotes hist`` (Kalshi candles), then ``run(settings())``
for dev-only variant selection and ``run(settings(), holdout=("sp_lineup", "sp"))`` for the single
holdout pass; ``write_report`` / ``save_model`` write the report and the frozen model
(``data/cache/fundamental_mlb_model.json``). Spring reuse: fetch the new season's schedule and
boxscores, ``build_features`` + ``add_park_factors`` (as-of by construction), load the JSON as the
model dict and call ``game_summaries`` / ``prob_for``.

Model: per-half-inning negative binomial run rates from a log-linear model (offense wOBA-type
rate, opposing starter FIP regressed by batters faced, bullpen FIP and recent workload, park,
home, inning) and a Monte Carlo game simulator (starter exit by inning, home team skips the
bottom of the 9th when ahead, walk-offs stop the game, extra innings with the automatic runner,
a shared game-level gamma factor fitted to the variance of game totals). One simulation prices
every market of a game consistently.
"""

import json
import math
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

API = "https://statsapi.mlb.com/api/v1"
GAME_TYPES = "R,F,D,L,W"   # regular season, wild card, division, league championship, world series
SERIES = ("KXMLBGAME", "KXMLBTOTAL", "KXMLBF5TOTAL", "KXMLBSPREAD", "KXMLBINNINGTOTAL")
STRIKEOUT_SERIES = "KXMLBKS"
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
DEV_FROM = datetime(2026, 8, 15, tzinfo=UTC)

# Kalshi team codes in tickers are the MLB Stats API abbreviations (checked on all 30 teams).
TEAM_IDS = {"ATH": 133, "ATL": 144, "AZ": 109, "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113, "CLE": 114,
            "COL": 115, "CWS": 145, "DET": 116, "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146,
            "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136,
            "SF": 137, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120}

# wOBA-type linear weights (FanGraphs-style, stable across seasons to the second decimal).
W_UBB, W_HBP, W_1B, W_2B, W_3B, W_HR = 0.69, 0.72, 0.88, 1.25, 1.58, 2.03


# ---- fetching (cached, polite) ----------------------------------------------------------------
def mlb_client(s, min_interval: float = 0.5):
    from albert.http import CachedClient
    return CachedClient("mlb", API, s.data_dir / "cache", min_interval=min_interval,
                        user_agent="albert-research/0.2 (personal research; cached)")


def _paths(s) -> dict[str, Path]:
    base = s.data_dir / "cache"
    return {k: base / f"fundamental_mlb_{k}.parquet"
            for k in ("games", "box_team", "box_pitch", "box_bat", "wx", "people", "batters", "pitch_season",
                      "bat_season", "team_season")}


def fetch_schedule(api, season: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month in range(3, 12):
        lo = date(season, month, 1)
        hi = (date(season + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
        page = api.get("/schedule", {"sportId": 1, "startDate": lo.isoformat(), "endDate": hi.isoformat(),
                                     "gameType": GAME_TYPES, "hydrate": "probablePitcher,linescore"})
        rows.extend(parse_schedule(page.payload))
    return rows


def parse_schedule(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per scheduled game (every status), with per-inning runs from the linescore."""
    rows = []
    for d in payload.get("dates") or []:
        for g in d.get("games") or []:
            st = g.get("status") or {}
            t = g.get("teams") or {}
            ls = g.get("linescore") or {}
            inn_h, inn_a = [], []
            for inn in ls.get("innings") or []:
                inn_a.append((inn.get("away") or {}).get("runs"))
                inn_h.append((inn.get("home") or {}).get("runs"))
            row = {
                "game_pk": g["gamePk"], "season": int(g.get("season") or 0), "game_type": g.get("gameType"),
                "game_date": datetime.fromisoformat(g["gameDate"]),
                "official_date": date.fromisoformat(g["officialDate"]),
                "status": st.get("detailedState"), "coded_state": st.get("codedGameState"),
                "start_tbd": bool(st.get("startTimeTBD")),
                "double_header": g.get("doubleHeader"), "game_number": g.get("gameNumber"),
                "venue_id": (g.get("venue") or {}).get("id"), "venue": (g.get("venue") or {}).get("name"),
                "scheduled_innings": g.get("scheduledInnings"),
                "day_night": g.get("dayNight"),
                "rescheduled_from": g.get("rescheduledFromDate") or g.get("resumedFromDate"),
            }
            for side in ("home", "away"):
                ts = t.get(side) or {}
                row[f"{side}_id"] = (ts.get("team") or {}).get("id")
                row[f"{side}_name"] = (ts.get("team") or {}).get("name")
                row[f"{side}_score"] = ts.get("score")
                row[f"{side}_prob_pitcher"] = (ts.get("probablePitcher") or {}).get("id")
            row["inn_home"] = inn_h
            row["inn_away"] = inn_a
            rows.append(row)
    return rows


def parse_boxscore(payload: dict[str, Any], game_pk: int) -> tuple[list[dict], list[dict]]:
    """Team batting totals (one row per team) and every pitcher's line (in order of appearance)."""
    teams, pitchers = [], []
    for side in ("home", "away"):
        t = (payload.get("teams") or {}).get(side) or {}
        bat = (t.get("teamStats") or {}).get("batting") or {}
        tid = (t.get("team") or {}).get("id")
        teams.append({"game_pk": game_pk, "side": side, "team_id": tid,
                      **{k: int(bat.get(v) or 0) for k, v in (
                          ("pa", "plateAppearances"), ("ab", "atBats"), ("h", "hits"), ("d2", "doubles"),
                          ("d3", "triples"), ("hr", "homeRuns"), ("bb", "baseOnBalls"), ("ibb", "intentionalWalks"),
                          ("hbp", "hitByPitch"), ("so", "strikeOuts"), ("sf", "sacFlies"), ("r", "runs"))}})
        players = t.get("players") or {}
        for order, pid in enumerate(t.get("pitchers") or []):
            st = ((players.get(f"ID{pid}") or {}).get("stats") or {}).get("pitching") or {}
            pitchers.append({"game_pk": game_pk, "side": side, "team_id": tid, "pitcher_id": pid, "order": order,
                             **{k: int(st.get(v) or 0) for k, v in (
                                 ("outs", "outs"), ("bf", "battersFaced"), ("h", "hits"), ("r", "runs"),
                                 ("er", "earnedRuns"), ("bb", "baseOnBalls"), ("ibb", "intentionalWalks"),
                                 ("hbp", "hitByPitch"), ("so", "strikeOuts"), ("hr", "homeRuns"),
                                 ("pitches", "numberOfPitches"), ("gs", "gamesStarted"))}})
    return teams, pitchers


def parse_batting(payload: dict[str, Any], game_pk: int) -> list[dict[str, Any]]:
    """Every batter's line with his starting lineup slot (1-9; None for substitutes).

    The slot comes from each player's own ``battingOrder`` code (100, 200, ... = starter of that
    slot; 101 = first substitute). The team-level ``battingOrder`` list is NOT used: it holds the
    last batter in each slot (substitutes included), which leaks the game's course (blowouts
    bring in bench players)."""
    rows = []
    for side in ("home", "away"):
        t = (payload.get("teams") or {}).get(side) or {}
        players = t.get("players") or {}
        slot = {}
        for pl_ in players.values():
            code = str(pl_.get("battingOrder") or "")
            if code.isdigit() and int(code) % 100 == 0:
                slot[(pl_.get("person") or {}).get("id")] = int(code) // 100
        for pid in t.get("batters") or []:
            st = ((players.get(f"ID{pid}") or {}).get("stats") or {}).get("batting") or {}
            pa = int(st.get("plateAppearances") or 0)
            if pa == 0 and pid not in slot:
                continue
            rows.append({"game_pk": game_pk, "side": side, "team_id": (t.get("team") or {}).get("id"), "player_id": pid,
                         "slot": slot.get(pid), **{k: int(st.get(v) or 0) for k, v in (
                             ("pa", "plateAppearances"), ("ab", "atBats"), ("h", "hits"), ("d2", "doubles"),
                             ("d3", "triples"), ("hr", "homeRuns"), ("bb", "baseOnBalls"), ("ibb", "intentionalWalks"),
                             ("hbp", "hitByPitch"), ("so", "strikeOuts"), ("sf", "sacFlies"))}})
    return rows


def parse_weather(payload: dict[str, Any], game_pk: int) -> dict[str, Any]:
    """Game-time weather from the boxscore notes (observed at first pitch: NOT known before ``t``;
    used only as a labelled diagnostic, never in the as-of model)."""
    import re
    info = {i.get("label"): i.get("value") or "" for i in payload.get("info") or []}
    w, wind = info.get("Weather", ""), info.get("Wind", "")
    temp = re.search(r"(-?\d+) degrees", w)
    mph = re.search(r"(\d+) mph", wind)
    speed = float(mph.group(1)) if mph else 0.0
    out_in = 1.0 if "Out To" in wind else -1.0 if "In From" in wind else 0.0
    roof = any(x in (w + wind) for x in ("Roof Closed", "Dome", "Indoors"))
    return {"game_pk": game_pk, "temp": float(temp.group(1)) if temp else None, "wind_mph": speed,
            "wind_out": 0.0 if roof else out_in * speed, "roof": roof}


def parse_season_pitching(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for st in payload.get("stats") or []:
        for sp in st.get("splits") or []:
            x = sp.get("stat") or {}
            rows.append({"season": int(sp.get("season") or 0), "pitcher_id": (sp.get("player") or {}).get("id"),
                         **{k: int(x.get(v) or 0) for k, v in (
                             ("g", "gamesPlayed"), ("gs", "gamesStarted"), ("outs", "outs"), ("bf", "battersFaced"),
                             ("so", "strikeOuts"), ("bb", "baseOnBalls"), ("ibb", "intentionalWalks"),
                             ("hbp", "hitByPitch"), ("hr", "homeRuns"), ("r", "runs"), ("er", "earnedRuns"))}})
    return rows


def parse_season_hitting(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for st in payload.get("stats") or []:
        for sp in st.get("splits") or []:
            x = sp.get("stat") or {}
            rows.append({"season": int(sp.get("season") or 0), "player_id": (sp.get("player") or {}).get("id"),
                         **{k: int(x.get(v) or 0) for k, v in (
                             ("pa", "plateAppearances"), ("ab", "atBats"), ("h", "hits"), ("d2", "doubles"),
                             ("d3", "triples"), ("hr", "homeRuns"), ("bb", "baseOnBalls"), ("ibb", "intentionalWalks"),
                             ("hbp", "hitByPitch"), ("so", "strikeOuts"), ("sf", "sacFlies"))}})
    return rows


def parse_team_hitting(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for st in payload.get("stats") or []:
        for sp in st.get("splits") or []:
            x = sp.get("stat") or {}
            rows.append({"season": int(sp.get("season") or 0), "team_id": (sp.get("team") or {}).get("id"),
                         **{k: int(x.get(v) or 0) for k, v in (
                             ("g", "gamesPlayed"), ("pa", "plateAppearances"), ("ab", "atBats"), ("h", "hits"),
                             ("d2", "doubles"), ("d3", "triples"), ("hr", "homeRuns"), ("bb", "baseOnBalls"),
                             ("ibb", "intentionalWalks"), ("hbp", "hitByPitch"), ("so", "strikeOuts"),
                             ("sf", "sacFlies"), ("r", "runs"))}})
    return rows


def parse_people(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"pitcher_id": p.get("id"), "name": p.get("fullName"),
             "throws": (p.get("pitchHand") or {}).get("code"), "bats": (p.get("batSide") or {}).get("code")}
            for p in payload.get("people") or []]


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def fetch_all(s, *, schedule_seasons=(2022, 2023, 2024, 2025, 2026), box_seasons=(2025, 2026),
              prior_seasons=(2023, 2024, 2025)) -> dict[str, pl.DataFrame]:
    """Fetch (or read from cache) everything the study needs; write parsed parquet tables."""
    api = mlb_client(s)
    paths = _paths(s)
    games = []
    for season in schedule_seasons:
        games.extend(fetch_schedule(api, season))
        print(f"schedule {season}: {len(games)} rows so far", flush=True)
    g = pl.DataFrame(games, infer_schema_length=None).unique("game_pk", keep="last")
    _write(g, paths["games"])
    final = g.filter(pl.col("season").is_in(list(box_seasons)) & (pl.col("coded_state") == "F")).sort("game_date")
    bt, bp, bb, wx = [], [], [], []
    for i, pk in enumerate(final["game_pk"].to_list(), 1):
        try:
            page = api.get(f"/game/{pk}/boxscore")
        except RuntimeError as error:
            print(f"skip boxscore {pk}: {str(error)[:100]}")
            continue
        t, p = parse_boxscore(page.payload, pk)
        bt.extend(t)
        bp.extend(p)
        bb.extend(parse_batting(page.payload, pk))
        wx.append(parse_weather(page.payload, pk))
        if i % 250 == 0:
            print(f"  boxscores {i}/{final.height}", flush=True)
    _write(pl.DataFrame(bt), paths["box_team"])
    box_bat = pl.DataFrame(bb)
    _write(box_bat, paths["box_bat"])
    _write(pl.DataFrame(wx), paths["wx"])
    box_pitch = pl.DataFrame(bp)
    _write(box_pitch, paths["box_pitch"])
    ps, th, hs = [], [], []
    for season in prior_seasons:
        hs.extend(parse_season_hitting(api.get("/stats", {"stats": "season", "group": "hitting", "season": season,
                                                          "sportId": 1, "playerPool": "ALL", "limit": 5000}).payload))
        ps.extend(parse_season_pitching(api.get("/stats", {"stats": "season", "group": "pitching", "season": season,
                                                           "sportId": 1, "playerPool": "ALL", "limit": 3000}).payload))
        th.extend(parse_team_hitting(api.get("/teams/stats", {"season": season, "group": "hitting", "stats": "season",
                                                              "sportIds": 1}).payload))
    _write(pl.DataFrame(ps), paths["pitch_season"])
    _write(pl.DataFrame(th), paths["team_season"])
    _write(pl.DataFrame(hs), paths["bat_season"])
    ids = sorted(set(box_pitch["pitcher_id"].to_list()))
    people = []
    for i in range(0, len(ids), 150):
        chunk = ",".join(str(x) for x in ids[i:i + 150])
        people.extend(parse_people(api.get("/people", {"personIds": chunk}).payload))
    _write(pl.DataFrame(people), paths["people"])
    bids = sorted(set(box_bat["player_id"].to_list()))
    batters = []
    for i in range(0, len(bids), 150):
        chunk = ",".join(str(x) for x in bids[i:i + 150])
        batters.extend(parse_people(api.get("/people", {"personIds": chunk}).payload))
    _write(pl.DataFrame(batters).rename({"pitcher_id": "player_id"}), paths["batters"])
    return load_tables(s)


def load_tables(s) -> dict[str, pl.DataFrame]:
    return {k: pl.read_parquet(p) for k, p in _paths(s).items() if p.exists()}


# ---- Kalshi quotes ------------------------------------------------------------------------------
def game_code(event_ticker: str) -> str:
    """``KXMLBINNINGTOTAL-26SEP031410CWSHOU-4`` -> ``26SEP031410CWSHOU`` (shared by a game's series)."""
    return event_ticker.split("-")[1]


def strike_of(row: dict) -> float | None:
    """Strike (runs) of a total / run-line / inning-total leg from its subtitle (``Over 8.5 runs``)."""
    import re
    m = re.search(r"over (\d+(?:\.\d+)?) run", (row.get("yes_sub_title") or "").lower())
    return float(m.group(1)) if m else None


def quote_universe(markets: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    """Markets whose pre-game candles we fetch. After the historical cutoff the batch endpoint
    makes every leg cheap, so all legs are kept; before it (per-market requests, rate-limited
    shared API) a fixed, outcome-blind subset: the home moneyline leg, totals 6.5-11.5, F5 totals
    2.5-5.5, the 1.5-run lines, no inning totals."""
    from albert.venues.kalshi import ticker_start

    m = markets.filter(pl.col("series_ticker").is_in(list(SERIES)) & pl.col("result").is_in(["yes", "no"]))
    m = m.with_columns(pl.col("event_ticker").map_elements(game_code, return_dtype=pl.String).alias("game_code"),
                       pl.col("event_ticker").map_elements(ticker_start, return_dtype=pl.Datetime("us", "UTC")).alias("start"),
                       pl.struct("yes_sub_title").map_elements(strike_of, return_dtype=pl.Float64).alias("strike"))
    home_code = pl.col("game_code").str.replace(r"G\d$", "")
    is_home_leg = home_code.str.ends_with(pl.col("ticker").str.split("-").list.last())
    hist_keep = (((pl.col("series_ticker") == "KXMLBGAME") & is_home_leg)
                 | ((pl.col("series_ticker") == "KXMLBTOTAL") & pl.col("strike").is_between(6.5, 11.5))
                 | ((pl.col("series_ticker") == "KXMLBF5TOTAL") & pl.col("strike").is_between(2.5, 5.5))
                 | ((pl.col("series_ticker") == "KXMLBSPREAD") & (pl.col("strike") == 1.5)))
    return m.filter(pl.col("start").is_not_null() & ((pl.col("close_time") >= cutoff) | hist_keep))


def fetch_quotes(s, markets: pl.DataFrame, *, workers: int = 2, max_tickers: int = 100,
                 only: str | None = None) -> pl.DataFrame:
    """Hourly candles from 12:00 UTC of game day to the scheduled start for ``quote_universe``
    markets; cached in ``data/cache/fundamental_mlb_candles.parquet`` (raw responses in the shared
    append-only Kalshi cache)."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / "fundamental_mlb_candles.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["ticker"].unique()) if have is not None and have.height else set()
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.1))
    cutoff = api.cutoff().get("market_settled_ts")
    uni = quote_universe(markets, cutoff).filter(~pl.col("ticker").is_in(list(done)))
    jobs: list[tuple[str, list[str], datetime, datetime]] = []
    for (_, start), grp in uni.group_by("game_code", "start", maintain_order=True):
        et_day = (start - timedelta(hours=4)).date()
        lo = datetime(et_day.year, et_day.month, et_day.day, 12, tzinfo=UTC)
        hi = start + timedelta(minutes=10)
        hist = grp.filter(pl.col("close_time") < cutoff)["ticker"].to_list()
        live = grp.filter(pl.col("close_time") >= cutoff)["ticker"].to_list()
        jobs += [("hist", [x], lo, hi) for x in hist]
        jobs += [("live", live[i:i + max_tickers], lo, hi) for i in range(0, len(live), max_tickers)]
    jobs = [j for j in jobs if only is None or j[0] == only]
    jobs.sort(key=lambda j: j[0] == "hist")   # cheap batched (live) jobs first
    print(f"quote jobs: {len(jobs)} for {uni.height} markets", flush=True)

    def one(job: tuple[str, list[str], datetime, datetime]) -> list[dict[str, Any]]:
        kind, tickers, lo, hi = job
        for attempt in range(5):
            try:
                if kind == "hist":
                    page = api.candles(tickers[0], lo, hi, 60, True)
                    return k.candle_rows(page.payload, tickers[0], 60)
                page = api.candles_batch(tickers, lo, hi, 60)
                out: list[dict[str, Any]] = []
                for mk in page.payload.get("markets") or []:
                    out.extend(k.candle_rows(mk, mk.get("market_ticker"), 60))
                return out
            except RuntimeError as error:
                if "429" in str(error):
                    time.sleep(5 * (attempt + 1))
                    continue
                print(f"skip candles {tickers[0]} (+{len(tickers) - 1}): {str(error)[:160]}")
                return []
        return []

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(one, jobs), 1):
            rows.extend(r)
            if i % 200 == 0:
                print(f"  quote jobs {i}/{len(jobs)}, {len(rows)} rows", flush=True)
    new = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else None
    frames = [f for f in (have, new) if f is not None and f.height]
    out = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"]) if frames else pl.DataFrame()
    _write(out, path)
    return out


# ---- as-of features ---------------------------------------------------------------------------
K_OFF = 1200.0       # team offense ballast, plate appearances of league/prior-season rate
K_SP = 300.0         # starter FIP ballast, batters faced of league rate
K_BP = 600.0         # bullpen FIP ballast, batters faced
K_START = 4.0        # starter length ballast, starts of league-average length
K_PLAT = 400.0       # platoon split ballast, plate appearances of the team's overall rate
K_BAT = 250.0        # batter wOBA ballast, plate appearances of league rate
SLOT_PA = (4.65, 4.55, 4.43, 4.33, 4.24, 4.13, 4.01, 3.90, 3.77)   # expected PA by lineup slot
PREV_W = (0.6, 0.3)  # weights of the previous one and two seasons in pitcher priors


def woba_parts(r: dict) -> tuple[float, float]:
    ubb = r["bb"] - r["ibb"]
    singles = r["h"] - r["d2"] - r["d3"] - r["hr"]
    num = W_UBB * ubb + W_HBP * r["hbp"] + W_1B * singles + W_2B * r["d2"] + W_3B * r["d3"] + W_HR * r["hr"]
    return num, float(r["ab"] + ubb + r["sf"] + r["hbp"])


def fip_parts(r: dict) -> tuple[float, float]:
    """FIP numerator per batter faced (13 HR + 3 (uBB + HBP) - 2 K) and its denominator (BF)."""
    return 13.0 * r["hr"] + 3.0 * (r["bb"] - r["ibb"] + r["hbp"]) - 2.0 * r["so"], float(r["bf"])


def kbb_parts(r: dict) -> tuple[float, float]:
    return float(r["so"] - r["bb"] + r["ibb"] - r["hbp"]), float(r["bf"])


class _Acc:
    """Running sums keyed by anything; ``rate`` regresses toward a prior with a ballast."""

    def __init__(self) -> None:
        self.num: dict = defaultdict(float)
        self.den: dict = defaultdict(float)

    def add(self, key, num: float, den: float) -> None:
        self.num[key] += num
        self.den[key] += den

    def rate(self, key, prior: float, ballast: float, extra: tuple[float, float] = (0.0, 0.0)) -> float:
        return (self.num.get(key, 0.0) + extra[0] + ballast * prior) / (self.den.get(key, 0.0) + extra[1] + ballast)

    def n(self, key) -> float:
        return self.den.get(key, 0.0)


def _innings_ok(g: dict) -> bool:
    ih, ia = g["inn_home"] or [], g["inn_away"] or []
    return len(ia) >= 9 and all(x is not None for x in ia[:9]) and all(x is not None for x in ih[:8])


def build_features(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per (final game, batting side) with as-of features: only games on earlier ET dates
    (and full earlier seasons) feed each row. Also carries the realised inning runs (targets)."""
    games = tables["games"].filter((pl.col("coded_state") == "F") & pl.col("game_type").is_in(list("RFDLW"))
                                   & pl.col("season").is_in([2025, 2026])).sort("official_date", "game_date")
    box_t = {(r["game_pk"], r["side"]): r for r in tables["box_team"].iter_rows(named=True)}
    box_p: dict = defaultdict(list)
    for r in tables["box_pitch"].sort("game_pk", "side", "order").iter_rows(named=True):
        box_p[(r["game_pk"], r["side"])].append(r)
    hand = {r["pitcher_id"]: r["throws"] for r in tables["people"].iter_rows(named=True)}
    names = {r["pitcher_id"]: r["name"] for r in tables["people"].iter_rows(named=True)}
    ps = tables["pitch_season"].group_by("season", "pitcher_id").agg(pl.all().exclude("season", "pitcher_id").sum())
    prev_p = {(r["season"], r["pitcher_id"]): r for r in ps.iter_rows(named=True)}
    lg_prev: dict[int, dict[str, float]] = {}
    for season in ps["season"].unique().to_list():
        a = ps.filter(pl.col("season") == season).sum().row(0, named=True)
        f, bf = fip_parts(a)
        k, _ = kbb_parts(a)
        lg_prev[season] = {"fip": f / bf, "kbb": k / bf, "start_outs": 15.5}
    th = tables["team_season"]
    prev_off = {}
    for r in th.iter_rows(named=True):
        num, den = woba_parts(r)
        prev_off[(r["season"], r["team_id"])] = num / den
    lg_off_prev = {season: float(sum(woba_parts(r)[0] for r in th.filter(pl.col("season") == season).iter_rows(named=True))
                                 / sum(woba_parts(r)[1] for r in th.filter(pl.col("season") == season).iter_rows(named=True)))
                   for season in th["season"].unique().to_list()}
    lg_rpg_prev = {season: float(th.filter(pl.col("season") == season)["r"].sum() / th.filter(pl.col("season") == season)["g"].sum())
                   for season in th["season"].unique().to_list()}

    off, off_r, def_r, sp_fip, sp_kbb, sp_len, bp_fip, tm_fip, plat, bat = (_Acc() for _ in range(10))
    box_b: dict = defaultdict(list)
    if "box_bat" in tables:
        for r in tables["box_bat"].iter_rows(named=True):
            box_b[(r["game_pk"], r["side"])].append(r)
    bats = {r["player_id"]: r["bats"] for r in tables["batters"].iter_rows(named=True)} if "batters" in tables else {}
    prev_b: dict = {}
    if "bat_season" in tables:
        for r in tables["bat_season"].group_by("season", "player_id").agg(pl.all().sum()).iter_rows(named=True):
            prev_b[(r["season"], r["player_id"])] = woba_parts(r)
    lg = _Acc()
    bp_pitches: dict = defaultdict(float)   # (team, date) -> reliever pitches
    rows = []

    def league(season: int, key: str, prior: float, ballast: float) -> float:
        return lg.rate((season, key), prior, ballast)

    by_date = games.partition_by("official_date", maintain_order=True)
    for day in by_date:
        d = day["official_date"][0]
        day_rows = day.to_dicts()
        # 1) features from state through the previous day
        for g in day_rows:
            season = g["season"]
            have_box = all((g["game_pk"], s) in box_t for s in ("home", "away"))
            if not have_box or not box_p.get((g["game_pk"], "home")) or not box_p.get((g["game_pk"], "away")):
                continue
            lg_woba = league(season, "woba", lg_off_prev.get(season - 1, 0.315), 20000)
            lg_fip = league(season, "fip", lg_prev.get(season - 1, {}).get("fip", 0.05), 20000)
            lg_kbb = league(season, "kbb", lg_prev.get(season - 1, {}).get("kbb", 0.14), 20000)
            lg_len = league(season, "start_outs", 15.5, 200)
            lg_rpg = league(season, "rpg", lg_rpg_prev.get(season - 1, 4.4), 300)
            for side, opp in (("home", "away"), ("away", "home")):
                team, opp_team = g[f"{side}_id"], g[f"{opp}_id"]
                starter = box_p[(g["game_pk"], opp)][0]["pitcher_id"]
                prior_off = 0.5 * prev_off.get((season - 1, team), lg_woba) / lg_off_prev.get(season - 1, lg_woba) * lg_woba + 0.5 * lg_woba
                o = off.rate((season, team), prior_off, K_OFF)
                h = hand.get(starter) or "R"
                o_hand = plat.rate((season, team, h), o, K_PLAT)
                extra_f, extra_k, extra_len = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
                for w, ps_season in zip(PREV_W, (season - 1, season - 2)):
                    pr = prev_p.get((ps_season, starter))
                    if pr and pr["bf"] > 0:
                        f, bf = fip_parts(pr)
                        k, _ = kbb_parts(pr)
                        extra_f[0] += w * f
                        extra_f[1] += w * bf
                        extra_k[0] += w * k
                        extra_k[1] += w * bf
                        if pr["gs"] >= 3 and pr["gs"] >= 0.7 * pr["g"]:
                            extra_len[0] += w * 3 * pr["outs"] / pr["g"]
                            extra_len[1] += w * 3
                lineup = sorted((r for r in box_b.get((g["game_pk"], side), []) if r["slot"]), key=lambda r: r["slot"])
                if len(lineup) == 9:
                    wsum = lsum = psum = 0.0
                    for r, w in zip(lineup, SLOT_PA):
                        ex = [0.0, 0.0]
                        for wp, bs in zip(PREV_W, (season - 1, season - 2)):
                            pb = prev_b.get((bs, r["player_id"]))
                            if pb:
                                ex[0] += wp * pb[0]
                                ex[1] += wp * pb[1]
                        lsum += w * bat.rate((season, r["player_id"]), lg_woba, K_BAT, tuple(ex))
                        b_side = bats.get(r["player_id"]) or "R"
                        psum += w * float(b_side == "S" or b_side != h)
                        wsum += w
                    off_lineup, plat_share = lsum / wsum / lg_woba, psum / wsum
                else:
                    off_lineup, plat_share = o / lg_woba, 0.55
                rows.append({
                    "off_lineup": off_lineup, "plat_share": plat_share,
                    "game_pk": g["game_pk"], "season": season, "game_type": g["game_type"], "official_date": d,
                    "game_date": g["game_date"], "venue_id": g["venue_id"], "side": side, "home": int(side == "home"),
                    "team_id": team, "opp_id": opp_team, "opp_starter": starter, "opp_starter_name": names.get(starter),
                    "opp_starter_throws": h,
                    "off_woba": o / lg_woba, "off_woba_hand": o_hand / lg_woba,
                    "off_rpg": off_r.rate((season, team), lg_rpg, 25) / lg_rpg,
                    "def_rapg": def_r.rate((season, opp_team), lg_rpg, 25) / lg_rpg,
                    "lg_rpg": lg_rpg,
                    "off_pa": off.n((season, team)),
                    "sp_fip": sp_fip.rate((season, starter), lg_fip, K_SP, tuple(extra_f)) - lg_fip,
                    "sp_kbb": sp_kbb.rate((season, starter), lg_kbb, K_SP, tuple(extra_k)) - lg_kbb,
                    "sp_outs": sp_len.rate((season, starter), lg_len, K_START, tuple(extra_len)),
                    "sp_bf": sp_fip.n((season, starter)) + extra_f[1],
                    "bp_fip": bp_fip.rate((season, opp_team), lg_fip, K_BP) - lg_fip,
                    "tm_fip": tm_fip.rate((season, opp_team), lg_fip, K_BP) - lg_fip,
                    "bp_load": (bp_pitches.get((opp_team, d - timedelta(days=1)), 0.0)
                                + 0.5 * bp_pitches.get((opp_team, d - timedelta(days=2)), 0.0)) / 100.0,
                    "runs": g[f"{side}_score"],
                    "inn": list((g["inn_home"] if side == "home" else g["inn_away"]) or []),
                    "innings_ok": _innings_ok(g),
                })
        # 2) update state with this day's results
        for g in day_rows:
            season = g["season"]
            for side, opp in (("home", "away"), ("away", "home")):
                bt = box_t.get((g["game_pk"], side))
                pitchers = box_p.get((g["game_pk"], opp)) or []
                if bt is None or not pitchers:
                    continue
                for r in box_b.get((g["game_pk"], side), []):
                    bat.add((season, r["player_id"]), *woba_parts(r))
                num, den = woba_parts(bt)
                off.add((season, g[f"{side}_id"]), num, den)
                off_r.add((season, g[f"{side}_id"]), bt["r"], 1)
                def_r.add((season, g[f"{opp}_id"]), bt["r"], 1)
                lg.add((season, "woba"), num, den)
                lg.add((season, "rpg"), bt["r"], 1)
                h = hand.get(pitchers[0]["pitcher_id"]) or "R"
                plat.add((season, g[f"{side}_id"], h), num, den)
                opp_team = g[f"{opp}_id"]
                for p in pitchers:
                    f, bf = fip_parts(p)
                    k, _ = kbb_parts(p)
                    tm_fip.add((season, opp_team), f, bf)
                    lg.add((season, "fip"), f, bf)
                    lg.add((season, "kbb"), k, bf)
                    if p["order"] == 0:
                        sp_fip.add((season, p["pitcher_id"]), f, bf)
                        sp_kbb.add((season, p["pitcher_id"]), k, bf)
                        sp_len.add((season, p["pitcher_id"]), p["outs"], 1)
                        lg.add((season, "start_outs"), p["outs"], 1)
                    else:
                        bp_fip.add((season, opp_team), f, bf)
                        bp_pitches[(opp_team, d)] += p["pitches"]
    return pl.DataFrame(rows, infer_schema_length=None)


def park_factors(games: pl.DataFrame, seasons: list[int], ridge: float = 300.0) -> dict[int, float]:
    """Multiplicative run factor per venue from earlier seasons: Poisson fit of runs scored with
    offense-season, defense-season and venue effects (ridge toward 1, in units of expected runs)."""
    g = games.filter(pl.col("season").is_in(seasons) & (pl.col("coded_state") == "F")
                     & pl.col("game_type").eq("R") & pl.col("home_score").is_not_null())
    rows = []
    for r in g.iter_rows(named=True):
        rows.append((r["home_score"], f"{r['season']}o{r['home_id']}", f"{r['season']}d{r['away_id']}", r["venue_id"]))
        rows.append((r["away_score"], f"{r['season']}o{r['away_id']}", f"{r['season']}d{r['home_id']}", r["venue_id"]))
    y = np.array([r[0] for r in rows], float)
    o_keys = {k: i for i, k in enumerate(sorted({r[1] for r in rows}))}
    d_keys = {k: i for i, k in enumerate(sorted({r[2] for r in rows}))}
    v_keys = {k: i for i, k in enumerate(sorted({r[3] for r in rows}))}
    oi = np.array([o_keys[r[1]] for r in rows])
    di = np.array([d_keys[r[2]] for r in rows])
    vi = np.array([v_keys[r[3]] for r in rows])
    o, d_, v = np.zeros(len(o_keys)), np.zeros(len(d_keys)), np.zeros(len(v_keys))
    mu0 = math.log(y.mean())
    for _ in range(60):   # block coordinate updates; each effect is a gamma(pen, pen) posterior mode
        for eff, idx, pen in ((o, oi, 5.0), (d_, di, 5.0), (v, vi, ridge)):
            lam_without = np.exp(mu0 + o[oi] + d_[di] + v[vi] - eff[idx])
            num = np.bincount(idx, weights=y, minlength=len(eff))
            den = np.bincount(idx, weights=lam_without, minlength=len(eff))
            eff[:] = np.log((num + pen) / (den + pen))
        mu0 += math.log(y.sum() / np.exp(mu0 + o[oi] + d_[di] + v[vi]).sum())
    return {k: float(math.exp(v[i])) for k, i in v_keys.items()}


# ---- half-inning run model --------------------------------------------------------------------
SD_OUTS = 4.5            # spread of a starter's outs around his regressed expectation
VARIANTS: dict[str, dict[str, Any]] = {
    "team": {"off": "off_woba", "pitch": "team"},                         # no starter identity
    "sp": {"off": "off_woba", "pitch": "sp"},                             # starter FIP + length, bullpen
    "sp_full": {"off": "off_woba_hand", "pitch": "sp", "extra": True},    # + platoon, K-BB, bullpen load
    "sp_lineup": {"off": "off_lineup", "pitch": "sp", "plat": True},      # posted lineup: pre1h only
    # DIAGNOSTIC ONLY (not as-of): + weather observed at first pitch (boxscore), standing in for a
    # 1-hour-ahead forecast; never selected, never run on the holdout
    "sp_lineup_wx": {"off": "off_lineup", "pitch": "sp", "plat": True, "wx": True},
}
DIAGNOSTIC_VARIANTS = ("sp_lineup_wx",)
LINEUP_VARIANTS = ("sp_lineup", "sp_lineup_wx")


def starter_share(e_outs: np.ndarray, sd: float = SD_OUTS) -> np.ndarray:
    """(n, 9): expected share of inning i's outs recorded by the starter, outs ~ N(e_outs, sd)."""
    grid = np.arange(0, 28, dtype=float)
    w = np.exp(-0.5 * ((grid[None, :] - np.asarray(e_outs, float)[:, None]) / sd) ** 2)
    w /= w.sum(1, keepdims=True)
    inn = np.arange(9, dtype=float)
    part = np.clip(grid[None, :] - 3 * inn[:, None], 0, 3) / 3          # (9, 28)
    return w @ part.T


def feature_names(variant: str) -> list[str]:
    v = VARIANTS[variant]
    names = [f"inn{i}" for i in range(1, 10)] + ["log_off", "log_pf", "home", "log_lg_rpg"]
    if v["pitch"] == "team":
        names += ["tm_fip"]
    else:
        names += ["sp_fip_x_share", "bp_fip_x_pen", "pen"]
        if v.get("extra"):
            names += ["sp_kbb_x_share", "bp_load_x_pen"]
    if v.get("plat"):
        names += ["plat_x_share"]
    if v.get("wx"):
        names += ["temp10", "wind_out10"]
    return names


def design(feat: pl.DataFrame, variant: str) -> np.ndarray:
    """(n_rows, 9, k) design for every (team-game, inning 1..9)."""
    v = VARIANTS[variant]
    n = feat.height
    share = starter_share(feat["sp_outs"].to_numpy())
    pen = 1 - share
    cols = [np.broadcast_to(np.eye(9)[None, :, i], (n, 9)) for i in range(9)]
    log_off = np.log(feat[v["off"]].to_numpy())
    log_pf = np.log(feat["pf"].to_numpy())
    log_lg = np.log(feat["lg_rpg"].to_numpy() / 4.4)
    for x in (log_off, log_pf, feat["home"].to_numpy().astype(float), log_lg):
        cols.append(np.repeat(x[:, None], 9, 1))
    if v["pitch"] == "team":
        cols.append(np.repeat(feat["tm_fip"].to_numpy()[:, None], 9, 1))
    else:
        cols += [share * feat["sp_fip"].to_numpy()[:, None], pen * feat["bp_fip"].to_numpy()[:, None], pen]
        if v.get("extra"):
            cols += [share * feat["sp_kbb"].to_numpy()[:, None], pen * feat["bp_load"].to_numpy()[:, None]]
    if v.get("plat"):
        cols.append(share * (feat["plat_share"].to_numpy()[:, None] - 0.55))
    if v.get("wx"):
        open_air = ~feat["roof"].fill_null(True).to_numpy() & feat["temp"].is_not_null().to_numpy()
        temp = np.where(open_air, (feat["temp"].fill_null(70.0).to_numpy() - 70.0) / 10.0, 0.0)
        wind = np.where(open_air, feat["wind_out"].fill_null(0.0).to_numpy() / 10.0, 0.0)
        cols += [np.repeat(temp[:, None], 9, 1), np.repeat(wind[:, None], 9, 1)]
    return np.stack(cols, axis=2)


def nb_negll(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    from scipy.special import digamma, gammaln

    beta, log_a = params[:-1], params[-1]
    r = math.exp(-log_a)
    mu = np.exp(X @ beta)
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
    g_beta = X.T @ ((y - mu) * r / (r + mu))
    g_r = (digamma(y + r) - digamma(r) + np.log(r / (r + mu)) + 1 - (r + y) / (r + mu)).sum()
    return -float(ll.sum()), -np.append(g_beta, -r * g_r)


def fit_nb(X: np.ndarray, y: np.ndarray, start: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Negative binomial (NB2) regression by maximum likelihood: (beta, alpha, mean log-lik)."""
    from scipy.optimize import minimize

    x0 = np.zeros(X.shape[1] + 1) if start is None else start
    if start is None:
        x0[-1] = math.log(0.5)
    res = minimize(nb_negll, x0, args=(X, y), jac=True, method="L-BFGS-B")
    return res.x[:-1], float(math.exp(res.x[-1])), -float(res.fun) / len(y)


def fit_model(feat: pl.DataFrame, variant: str) -> dict[str, Any]:
    """Fit the half-inning NB model, the extra-inning offset, and game/team gamma frailties."""
    f = feat.filter(pl.col("innings_ok"))
    X3 = design(f, variant)
    inn = np.array([(row + [None] * 9)[:9] for row in f["inn"].to_list()], dtype=object)
    Y = np.array([[np.nan if v is None else float(v) for v in row] for row in inn])
    use = ~np.isnan(Y)
    use[f["home"].to_numpy() == 1, 8] = False          # the home 9th is truncated (skipped / walk-off)
    X, y = X3[use], Y[use]
    beta, alpha, ll = fit_nb(X, y)
    # extra innings (regular season, automatic runner): away halves only (no walk-off truncation)
    xr = []
    for row, x9 in zip(f.iter_rows(named=True), X3[:, 8, :]):
        if row["home"] == 0 and row["game_type"] == "R" and len(row["inn"]) > 9:
            for v in row["inn"][9:]:
                if v is not None:
                    xr.append((x9 @ beta, float(v)))
    off_x = np.array([a for a, _ in xr])
    y_x = np.array([b for _, b in xr])
    ax = alpha
    # refit with the inning-9 linear predictor as an offset: one shift and one dispersion
    from scipy.optimize import minimize

    def negll_x(p: np.ndarray) -> float:
        return nb_negll(np.array([1.0, p[0], p[1]]), np.column_stack([off_x, np.ones_like(off_x)]), y_x)[0]
    res = minimize(negll_x, np.array([0.5, math.log(max(ax, 0.05))]), method="Nelder-Mead")
    delta_x, alpha_x = float(res.x[0]), float(math.exp(res.x[1]))
    # frailties from innings 1-8 (no truncation): Cov(H, A) and excess Var
    lam = np.exp(X3[:, :8, :] @ beta)
    mu8 = lam.sum(1)
    var8 = (lam + alpha * lam ** 2).sum(1)
    y8 = np.nansum(Y[:, :8], 1)
    ok8 = use[:, :8].all(1) | (f["home"].to_numpy() == 1) & ~np.isnan(Y[:, :8]).any(1)
    res8 = pl.DataFrame({"game_pk": f["game_pk"], "home": f["home"], "mu": mu8, "var": var8, "y": y8, "ok": ok8})
    res8 = res8.filter(pl.col("ok"))
    excess = float(((res8["y"] - res8["mu"]) ** 2 - res8["var"]).sum() / (res8["mu"] ** 2).sum())
    pair = res8.pivot(on="home", index="game_pk", values=["mu", "y"]).drop_nulls()
    cov = float(((pair["y_1"] - pair["mu_1"]) * (pair["y_0"] - pair["mu_0"])).sum() / (pair["mu_1"] * pair["mu_0"]).sum())
    # shared factor (cov > 0) or opposed factor (1 + e, 1 - e) for the negative home/away covariance
    sg2, sn2 = max(cov, 0.0), max(-cov, 0.0)
    st2 = max(excess - sg2 - sn2, 0.0)
    return {"variant": variant, "names": feature_names(variant), "beta": beta.tolist(), "alpha": alpha,
            "delta_x": delta_x, "alpha_x": alpha_x, "sg2": sg2, "sn2": sn2, "st2": st2, "loglik_per_half": ll,
            "n_half_innings": len(y), "n_extra_halves": len(y_x), "sd_outs": SD_OUTS,
            "mean_runs_half": float(y.mean()), "mean_pred_half": float(np.exp(X @ beta).mean())}


def simulate(lam_a: np.ndarray, lam_h: np.ndarray, model: dict[str, Any], *, ghost: bool, n: int,
             rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Simulate ``n`` games from per-inning NB means (innings 1..9) of the away and home offense."""
    a, ax = model["alpha"], model["alpha_x"]
    g = rng.gamma(1 / model["sg2"], model["sg2"], n) if model["sg2"] > 1e-6 else np.ones(n)
    ga = g * (rng.gamma(1 / model["st2"], model["st2"], n) if model["st2"] > 1e-6 else 1.0)
    gh = g * (rng.gamma(1 / model["st2"], model["st2"], n) if model["st2"] > 1e-6 else 1.0)
    if model.get("sn2", 0.0) > 1e-6:
        e = np.clip(rng.normal(0.0, math.sqrt(model["sn2"]), n), -0.9, 0.9)
        ga, gh = ga * (1 + e), gh * (1 - e)

    def nb(lam: float, mult: np.ndarray, alpha: float) -> np.ndarray:
        return rng.poisson(rng.gamma(1 / alpha, alpha * lam * mult))

    A = np.stack([nb(lam_a[i], ga, a) for i in range(9)], 1)
    H = np.stack([nb(lam_h[i], gh, a) for i in range(9)], 1)
    a_tot = A.sum(1)
    h8 = H[:, :8].sum(1)
    skip9 = h8 > a_tot
    need = a_tot - h8 + 1
    H[:, 8] = np.where(skip9, 0, np.minimum(H[:, 8], need))   # skipped when ahead; walk-off stops at +1
    h_tot = H[:, :9].sum(1)
    away, home = a_tot.astype(float), h_tot.astype(float)
    tied = away == home
    if ghost:
        la, lh, al = lam_a[8] * math.exp(model["delta_x"]), lam_h[8] * math.exp(model["delta_x"]), ax
    else:
        la, lh, al = lam_a[8], lam_h[8], a
    for _ in range(16):
        if not tied.any():
            break
        ta = np.where(tied, nb(la, ga, al), 0)
        th = np.where(tied, nb(lh, gh, al), 0)
        th = np.minimum(th, ta + 1)                 # walk-off stops at the winning run
        away += ta
        home += th
        tied = tied & (ta == th)
    coin = tied & (rng.random(n) < 0.5)
    home += coin
    away += tied & ~coin
    return {"away": away, "home": home, "inn_total": (A + H)[:, :9], "f5": (A[:, :5] + H[:, :5]).sum(1)}


def game_lambdas(X3: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    return np.exp(X3 @ np.asarray(model["beta"]))


def summarize(sim: dict[str, np.ndarray]) -> dict[str, Any]:
    """Probabilities for every market family from one simulated game."""
    tot = sim["away"] + sim["home"]
    margin = sim["home"] - sim["away"]
    return {"p_home": float((margin > 0).mean()),
            "total_pmf": np.bincount(np.minimum(tot, 40).astype(int), minlength=41) / len(tot),
            "f5_pmf": np.bincount(np.minimum(sim["f5"], 30).astype(int), minlength=31) / len(tot),
            "margin": margin,
            "inn_over": [(sim["inn_total"][:, i] > 0).mean() for i in range(9)],
            "inn_over1": [(sim["inn_total"][:, i] > 1).mean() for i in range(9)],
            "mean_total": float(tot.mean())}


def pythag(feat_home: dict, feat_away: dict, home_adv: float, exp: float = 1.83) -> float:
    """log5 of regressed Pythagorean records (runs scored / allowed through the previous day)."""
    def w(rs: float, ra: float) -> float:
        return rs ** exp / (rs ** exp + ra ** exp)
    ph = w(feat_home["off_rpg"], feat_away["def_rapg"])
    pa = w(feat_away["off_rpg"], feat_home["def_rapg"])
    p = (ph - ph * pa) / (ph + pa - 2 * ph * pa)
    lg = math.log(p / (1 - p)) + home_adv
    return 1 / (1 + math.exp(-lg))


# ---- Kalshi mapping, predictions -----------------------------------------------------------------
def _norm_name(x: str | None) -> str:
    import unicodedata
    x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    x = x.replace(".", " ").replace("-", " ").replace("'", "")
    return " ".join(w for w in x.split() if w not in ("jr", "sr", "ii", "iii"))


def split_codes(code: str) -> tuple[str, str] | None:
    """``TORCLE`` -> (away, home) using the 30 MLB abbreviations; ``...G2`` doubleheader suffix ok."""
    import re
    code = re.sub(r"G\d$", "", code)
    hits = [(code[:i], code[i:]) for i in range(2, len(code) - 1) if code[:i] in TEAM_IDS and code[i:] in TEAM_IDS]
    return hits[0] if len(hits) == 1 else None


def match_games(markets: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Kalshi game code -> MLB game_pk (same ET date, same away/home teams, nearest start)."""
    from albert.venues.kalshi import ticker_start

    codes = markets.select(pl.col("event_ticker").map_elements(game_code, return_dtype=pl.String).alias("game_code")).unique()
    g = games.filter(pl.col("coded_state") == "F")
    by_key: dict = defaultdict(list)
    for r in g.iter_rows(named=True):
        by_key[(r["official_date"], r["away_id"], r["home_id"])].append(r)
    rows = []
    for c in codes["game_code"].to_list():
        start = ticker_start(f"X-{c}")
        teams = split_codes(c[11:])
        if start is None or teams is None:
            continue
        et_day = (start - timedelta(hours=4)).date()
        cands = by_key.get((et_day, TEAM_IDS[teams[0]], TEAM_IDS[teams[1]]), [])
        if not cands:
            continue
        best = min(cands, key=lambda r: abs((r["game_date"] - start).total_seconds()))
        rows.append({"game_code": c, "game_pk": best["game_pk"], "ticker_start": start, "game_date": best["game_date"],
                     "start_tbd": best["start_tbd"], "game_type": best["game_type"], "away_code": teams[0],
                     "home_code": teams[1]})
    return pl.DataFrame(rows)


def market_specs(markets: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """One row per Kalshi leg with what it pays on: kind, team side, strike, inning."""
    m = markets.filter(pl.col("series_ticker").is_in(list(SERIES))).with_columns(
        pl.col("event_ticker").map_elements(game_code, return_dtype=pl.String).alias("game_code"),
        pl.struct("yes_sub_title").map_elements(strike_of, return_dtype=pl.Float64).alias("strike"))
    m = m.join(mapping, on="game_code", how="inner")
    rows = []
    for r in m.iter_rows(named=True):
        s, tail = r["series_ticker"], r["ticker"].split("-")[-1]
        spec = {"ticker": r["ticker"], "event": r["event_ticker"], "series": s, "game_pk": r["game_pk"],
                "y": int(r["result"] == "yes"), "strike": r["strike"], "side": None, "inning": None}
        if s == "KXMLBGAME":
            if tail not in (r["away_code"], r["home_code"]):
                continue
            spec["side"] = "home" if tail == r["home_code"] else "away"
        elif s == "KXMLBSPREAD":
            code = tail.rstrip("0123456789")
            if code not in (r["away_code"], r["home_code"]) or r["strike"] is None:
                continue
            spec["side"] = "home" if code == r["home_code"] else "away"
        elif s == "KXMLBINNINGTOTAL":
            spec["inning"] = int(r["event_ticker"].split("-")[-1])
            if r["strike"] is None or not 1 <= spec["inning"] <= 9:
                continue
        elif r["strike"] is None:
            continue
        rows.append(spec)
    return pl.DataFrame(rows, infer_schema_length=None)


def prob_for(spec: dict, summ: dict[str, Any]) -> float:
    s, k = spec["series"], spec["strike"]
    if s == "KXMLBGAME":
        return summ["p_home"] if spec["side"] == "home" else 1 - summ["p_home"]
    if s == "KXMLBTOTAL":
        return float(summ["total_pmf"][math.floor(k) + 1:].sum())
    if s == "KXMLBF5TOTAL":
        return float(summ["f5_pmf"][math.floor(k) + 1:].sum())
    if s == "KXMLBSPREAD":
        mg = summ["margin"] if spec["side"] == "home" else -summ["margin"]
        return float((mg > k).mean())
    if s == "KXMLBINNINGTOTAL":
        return float(summ["inn_over"][spec["inning"] - 1] if k < 1 else summ["inn_over1"][spec["inning"] - 1])
    raise ValueError(s)


def starters_public(markets_all: pl.DataFrame, mapping: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
    """Per game: the latest open time of the KXMLBKS (pitcher strikeout) markets naming each
    actual starter; null when a starter had no such market. A starter is public at ``t`` when this
    time is <= t."""
    ks = markets_all.filter(pl.col("series_ticker") == STRIKEOUT_SERIES).with_columns(
        pl.col("event_ticker").map_elements(game_code, return_dtype=pl.String).alias("game_code"),
        pl.col("title").str.split(":").list.first().map_elements(_norm_name, return_dtype=pl.String).alias("pname"))
    ks = ks.group_by("game_code", "pname").agg(pl.col("open_time").min())
    st = feat.select("game_pk", "side", pl.col("opp_starter_name").map_elements(_norm_name, return_dtype=pl.String).alias("pname"))
    st = st.join(mapping.select("game_code", "game_pk"), on="game_pk").join(ks, on=["game_code", "pname"], how="left")
    return st.group_by("game_pk").agg(pl.col("open_time").max().alias("starters_listed_at"),
                                      pl.col("open_time").null_count().alias("n_unlisted"))


# ---- study pipeline ------------------------------------------------------------------------------
FEE_MULT = {"KXMLBGAME": 0.5, "KXMLBTOTAL": 0.5, "KXMLBF5TOTAL": 0.5, "KXMLBSPREAD": 0.5, "KXMLBINNINGTOTAL": 1.0}


def add_park_factors(feat: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Park factor of each row's venue, fitted on the three seasons before the row's season."""
    pfs = {season: park_factors(games, [season - 3, season - 2, season - 1]) for season in feat["season"].unique().to_list()}
    return feat.with_columns(pl.struct("season", "venue_id").map_elements(
        lambda r: pfs[r["season"]].get(r["venue_id"], 1.0), return_dtype=pl.Float64).alias("pf"))


def features_cached(s, tables: dict[str, pl.DataFrame], refresh: bool = False) -> pl.DataFrame:
    path = s.data_dir / "cache" / "fundamental_mlb_features.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    feat = add_park_factors(build_features(tables), tables["games"])
    _write(feat, path)
    return feat


def game_summaries(feat: pl.DataFrame, model: dict[str, Any], game_pks: list[int], *, n: int = 20000) -> dict[int, dict]:
    f = feat.filter(pl.col("game_pk").is_in(game_pks)).sort("game_pk", "home")
    lam = game_lambdas(design(f, model["variant"]), model)
    out = {}
    idx: dict = defaultdict(dict)
    for i, r in enumerate(f.select("game_pk", "side", "game_type").iter_rows(named=True)):
        idx[r["game_pk"]][r["side"]] = i
        idx[r["game_pk"]]["type"] = r["game_type"]
    for pk, ix in idx.items():
        if "home" not in ix or "away" not in ix:
            continue
        rng = np.random.default_rng(pk)
        out[pk] = summarize(simulate(lam[ix["away"]], lam[ix["home"]], model, ghost=ix["type"] == "R", n=n, rng=rng))
    return out


def decision_times(mapping: pl.DataFrame) -> pl.DataFrame:
    """morning = 10:00 ET (14:00 UTC) of game day; pre1h = scheduled first pitch - 1 h (MLB
    ``gameDate``; doubleheader games with a TBD start are dropped from pre1h)."""
    rows = []
    for r in mapping.iter_rows(named=True):
        start = r["game_date"]
        d = (start - timedelta(hours=4)).date()
        morning = datetime(d.year, d.month, d.day, 14, tzinfo=UTC)
        if morning <= start - timedelta(hours=1):
            rows.append({"game_pk": r["game_pk"], "decision": "morning", "t": morning})
        if not r["start_tbd"]:
            rows.append({"game_pk": r["game_pk"], "decision": "pre1h", "t": start - timedelta(hours=1)})
    return pl.DataFrame(rows)


def predictions(specs: pl.DataFrame, summaries: dict[int, dict], times: pl.DataFrame,
                mapping: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for spec in specs.iter_rows(named=True):
        summ = summaries.get(spec["game_pk"])
        if summ is None:
            continue
        rows.append({**spec, "p": prob_for(spec, summ)})
    if not rows:
        return pl.DataFrame()
    pred = pl.DataFrame(rows, infer_schema_length=None)
    pred = pred.join(times, on="game_pk").join(mapping.select("game_pk", "game_code", "game_type"), on="game_pk")
    # cluster everything by game: a game's legs (and its inning events) share one outcome draw
    return pred.rename({"event": "kalshi_event"}).with_columns(pl.col("game_code").alias("event"))


def taker_real(df: pl.DataFrame, theta: float) -> pl.DataFrame:
    """Harness taker rule with each series' Kalshi fee multiplier (MLB: 0.5, inning totals 1.0)."""
    rows = []
    for r in df.iter_rows(named=True):
        mult = FEE_MULT[r["series"]]
        fee = lambda q: math.ceil(0.07 * mult * q * (1 - q) * 1e4 - 1e-9) / 1e4
        ey = r["p"] - r["ask"] - fee(r["ask"])
        en = (1 - r["p"]) - (1 - r["bid"]) - fee(1 - r["bid"])
        if max(ey, en) < theta:
            continue
        if ey >= en:
            rows.append({**r, "side_taken": "yes", "cost": r["ask"] + fee(r["ask"]), "pnl": r["y"] - r["ask"] - fee(r["ask"])})
        else:
            rows.append({**r, "side_taken": "no", "cost": 1 - r["bid"] + fee(1 - r["bid"]),
                         "pnl": (1 - r["y"]) - (1 - r["bid"]) - fee(1 - r["bid"])})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def taker_table(df: pl.DataFrame, holdout_from: datetime = HOLDOUT_FROM, min_trades: int = 30) -> tuple[list[dict], float | None]:
    from albert.research.fundamental import THETAS, clustered_mean

    out, best = [], None
    for theta in THETAS:
        for split, part in (("dev", df.filter(pl.col("t") < holdout_from)), ("holdout", df.filter(pl.col("t") >= holdout_from))):
            tr = taker_real(part, theta)
            if not tr.height:
                out.append({"theta": theta, "split": split, "trades": 0})
                continue
            mu, tt, k = clustered_mean(tr["pnl"].to_numpy(), tr["event"].to_numpy())
            out.append({"theta": theta, "split": split, "trades": tr.height, "events": k, "pnl_c": round(mu * 100, 3),
                        "t": None if tt is None else round(tt, 2),
                        "roi_per_$": round(float(tr["pnl"].sum() / tr["cost"].sum()), 4)})
            if split == "dev" and tr.height >= min_trades and (best is None or mu > best[1]):
                best = (theta, mu)
    return out, None if best is None else best[0]


def calibration_table(df: pl.DataFrame, col: str = "p", bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> pl.DataFrame:
    return (df.with_columns(pl.col(col).cut(list(bins[1:-1])).alias("bin"))
            .group_by("bin").agg(pl.len().alias("n"), pl.col(col).mean().round(3).alias("mean_p"),
                                 pl.col("y").mean().round(3).alias("rate"),
                                 (pl.col("mid").mean().round(3).alias("mean_mid") if "mid" in df.columns else pl.lit(None).alias("mean_mid")))
            .sort("mean_p"))


def ladder_backtest(pred: pl.DataFrame, summaries: dict[int, dict], *, cap: float = 0.25, kelly: float = 0.25) -> pl.DataFrame:
    """Per (game, decision): choose stakes across the KXMLBTOTAL ladder (YES at ask / NO at bid,
    fee multiplier 0.5) maximising expected log wealth under the model's total-runs pmf, scaled by
    ``kelly``; compare with the single best contract. Returns realised growth per game."""
    from scipy.optimize import minimize

    tot = pred.filter((pl.col("series") == "KXMLBTOTAL") & pl.col("bid").is_not_null())
    rows = []
    for (pk, dec), grp in tot.group_by("game_pk", "decision"):
        pmf = summaries[pk]["total_pmf"]
        T = np.arange(len(pmf))
        realized_T = None
        opts = []
        for r in grp.iter_rows(named=True):
            k = r["strike"]
            for side in ("yes", "no"):
                price = r["ask"] if side == "yes" else 1 - r["bid"]
                if not 0.01 <= price <= 0.99:
                    continue
                c = price + math.ceil(0.035 * price * (1 - price) * 1e4 - 1e-9) / 1e4
                pay = (T > k).astype(float) if side == "yes" else (T <= k).astype(float)
                if float(pmf @ pay) - c <= 0:
                    continue
                won = r["y"] if side == "yes" else 1 - r["y"]
                opts.append((pay / c - 1, won / c - 1))
            realized_T = realized_T or r["t"]
        if not opts:
            rows.append({"game_pk": pk, "decision": dec, "t": grp["t"][0], "n_opts": 0, "stake_ladder": 0.0,
                         "ret_ladder": 0.0, "stake_single": 0.0, "ret_single": 0.0})
            continue
        R = np.array([o[0] for o in opts])          # (m, |T|) return per $ in each outcome
        realized = np.array([o[1] for o in opts])

        def neg(f: np.ndarray) -> float:
            return -float(pmf @ np.log(np.maximum(1 + f @ R, 1e-9)))
        m = len(opts)
        res = minimize(neg, np.full(m, min(cap / m, 0.01)), method="SLSQP", bounds=[(0, cap)] * m,
                       constraints=[{"type": "ineq", "fun": lambda f: cap - f.sum()}])
        f_ladder = np.clip(res.x, 0, cap) * kelly
        single = [minimize(lambda f, j=j: neg(np.eye(m)[j] * f[0]), [0.01], bounds=[(0, cap)], method="L-BFGS-B")
                  for j in range(m)]
        j = int(np.argmin([s.fun[()] if np.ndim(s.fun) else s.fun for s in single]))
        f_single = np.zeros(m)
        f_single[j] = float(single[j].x[0]) * kelly
        rows.append({"game_pk": pk, "decision": dec, "t": grp["t"][0], "n_opts": m,
                     "stake_ladder": float(f_ladder.sum()), "ret_ladder": float(f_ladder @ realized),
                     "stake_single": float(f_single.sum()), "ret_single": float(f_single @ realized)})
    return pl.DataFrame(rows)


def ladder_summary(lad: pl.DataFrame, holdout_from: datetime = HOLDOUT_FROM) -> list[dict]:
    out = []
    for split, part in (("dev", lad.filter(pl.col("t") < holdout_from)), ("holdout", lad.filter(pl.col("t") >= holdout_from))):
        for dec in ("morning", "pre1h"):
            q = part.filter((pl.col("decision") == dec) & (pl.col("stake_ladder") > 0))
            if not q.height:
                continue
            for kind in ("ladder", "single"):
                st, rt = q[f"stake_{kind}"], q[f"ret_{kind}"]
                out.append({"split": split, "decision": dec, "policy": kind, "games_bet": int((st > 0).sum()),
                            "mean_stake_frac": round(float(st.mean()), 4),
                            "pnl_per_$": round(float(rt.sum() / max(st.sum(), 1e-9)), 4),
                            "log_growth_sum": round(float(np.log1p(rt.to_numpy()).sum()), 4),
                            "win_games_share": round(float((rt > 0).mean()), 3)})
    return out


def save_model(s, model: dict[str, Any], games: pl.DataFrame, extra: dict[str, Any], name: str = "model") -> Path:
    """Freeze the fitted model (coefficients, dispersions, frailties, ballasts, park factors for
    the next season) as JSON for reuse (e.g. spring 2027: rebuild features with
    ``build_features`` on the new season's boxscores, ``add_park_factors``, then
    ``game_summaries``)."""
    seasons = sorted(games.filter(pl.col("coded_state") == "F")["season"].unique().to_list())
    nxt = seasons[-1] + 1
    out = {**model, "frozen_at": datetime.now(UTC).isoformat(),
           "ballasts": {"K_OFF": K_OFF, "K_SP": K_SP, "K_BP": K_BP, "K_START": K_START, "K_PLAT": K_PLAT,
                        "PREV_W": list(PREV_W)},
           "woba_weights": [W_UBB, W_HBP, W_1B, W_2B, W_3B, W_HR],
           f"park_factors_{nxt}": {str(k): round(v, 4) for k, v in park_factors(games, seasons[-3:]).items()},
           **extra}
    path = s.data_dir / "cache" / f"fundamental_mlb_{name}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    return path


def _pyth_preds(feat: pl.DataFrame, home_adv: float) -> dict[int, float]:
    out = {}
    by = {(r["game_pk"], r["side"]): r for r in feat.iter_rows(named=True)}
    for (pk, side), r in by.items():
        if side == "home" and (pk, "away") in by:
            out[pk] = pythag(r, by[(pk, "away")], home_adv)
    return out


def fit_home_adv(feat: pl.DataFrame, games: pl.DataFrame) -> float:
    """Home-field logit shift for the log5/Pythagorean baseline, by maximum likelihood on ``feat``'s games."""
    won = dict(games.select("game_pk", (pl.col("home_score") > pl.col("away_score")).alias("w")).iter_rows())
    base = _pyth_preds(feat, 0.0)
    pk = [k for k in base if k in won]
    lg = np.array([math.log(base[k] / (1 - base[k])) for k in pk])
    y = np.array([float(won[k]) for k in pk])
    grid = np.linspace(-0.2, 0.4, 121)
    ll = [float((y * np.log(1 / (1 + np.exp(-(lg + h)))) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-(lg + h))))).sum()) for h in grid]
    return float(grid[int(np.argmax(ll))])


def kalshi_frame(s) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Settled legs of the studied series (Aug 15 - Nov 1 2026) and every MLB strikeout leg (any result)."""
    from albert.research.fundamental import settled_markets

    settled = settled_markets(s, list(SERIES), DEV_FROM, datetime(2026, 11, 2, tzinfo=UTC))
    ks = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").filter(pl.col("series_ticker") == STRIKEOUT_SERIES)
          .select("ticker", "event_ticker", "series_ticker", "title", "open_time", "close_time", "result").collect())
    return settled, ks


def run(s, *, n_sims: int = 20000, holdout: tuple[str, ...] | None = None,
        variants: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Stage 1 (``holdout=None``): fit every variant on games before the dev window, score dev
    only. Stage 2 (``holdout=<variants chosen on dev>``): those variants are also refit on
    everything before 2026-10-01 and scored once on October (dev rows keep the stage-1 fit)."""
    from albert.research.fundamental import attach_quotes, evaluate

    tables = load_tables(s)
    feat = features_cached(s, tables).join(tables["wx"].select("game_pk", "temp", "wind_out", "roof"), on="game_pk", how="left")
    games = tables["games"]
    settled, ks = kalshi_frame(s)
    mapping = match_games(settled, games)
    specs = market_specs(settled, mapping)
    times = decision_times(mapping)
    candles = pl.read_parquet(s.data_dir / "cache" / "fundamental_mlb_candles.parquet")
    starters = starters_public(ks, mapping, feat)
    dev_pks = mapping.filter(pl.col("game_date") < HOLDOUT_FROM)["game_pk"].to_list()
    hold_pks = mapping.filter(pl.col("game_date") >= HOLDOUT_FROM)["game_pk"].to_list()
    train1 = feat.filter(pl.col("game_date") < DEV_FROM)
    home_adv = fit_home_adv(train1, games)
    out: dict[str, Any] = {"mapping": mapping, "specs": specs, "starters": starters, "home_adv": home_adv,
                           "models": {}, "preds": {}}
    if holdout and any(v in DIAGNOSTIC_VARIANTS for v in holdout):
        raise ValueError("diagnostic (look-ahead) variants never run on the holdout")
    variants = list(holdout) if holdout else list(variants or VARIANTS)
    for variant in variants:
        m1 = fit_model(train1, variant)
        summ = game_summaries(feat, m1, dev_pks, n=n_sims)
        if holdout:
            m2 = fit_model(feat.filter(pl.col("game_date") < HOLDOUT_FROM), variant)
            summ.update(game_summaries(feat, m2, hold_pks, n=n_sims))
            out["models"][variant] = {"dev": m1, "holdout": m2}
        else:
            out["models"][variant] = {"dev": m1}
        pred = predictions(specs, summ, times, mapping)
        if variant in LINEUP_VARIANTS:   # lineups are posted ~2-5 h before first pitch: not known at 10:00 ET
            pred = pred.filter(pl.col("decision") == "pre1h")
        if not holdout:
            pred = pred.filter(pl.col("t") < HOLDOUT_FROM)
        pred = attach_quotes(pred, candles).join(starters, on="game_pk", how="left").with_columns(
            (pl.col("starters_listed_at").is_not_null() & (pl.col("starters_listed_at") <= pl.col("t"))
             & (pl.col("n_unlisted") == 0)).alias("starter_public"))
        out["preds"][variant] = pred
        out.setdefault("summaries", {})[variant] = summ
    pyth = _pyth_preds(feat.filter(pl.col("game_pk").is_in(dev_pks + (hold_pks if holdout else []))), home_adv)
    g = specs.filter(pl.col("series") == "KXMLBGAME")
    pp = predictions(g, {k: {"p_home": v} for k, v in pyth.items()}, times, mapping)
    if not holdout:
        pp = pp.filter(pl.col("t") < HOLDOUT_FROM)
    out["preds"]["pyth"] = attach_quotes(pp, candles)
    out["evaluate"] = {}
    for variant, pred in out["preds"].items():
        for (series, decision), part in pred.group_by("series", "decision"):
            part = part.filter(pl.col("side").is_null() | (pl.col("side") == "home") | (pl.col("series") != "KXMLBGAME"))
            out["evaluate"][(variant, series, decision)] = evaluate(part, HOLDOUT_FROM)
    out["line_test"] = {v: line_test(out["preds"][v], out["summaries"][v], games) for v in variants}
    if holdout:
        out["ladder"] = [{"variant": v, **r} for v in variants
                         for r in ladder_summary(ladder_backtest(out["preds"][v], out["summaries"][v]))]
    return out


def _cross_half(k: np.ndarray, q: np.ndarray) -> float | None:
    """Strike where a decreasing P(over) ladder crosses 0.5 (linear interpolation)."""
    for i in range(len(k) - 1):
        if q[i] >= 0.5 >= q[i + 1]:
            return float(k[i] + (q[i] - 0.5) / (q[i] - q[i + 1] + 1e-12) * (k[i + 1] - k[i]))
    return None


def line_test(pred: pl.DataFrame, summaries: dict[int, dict], games: pl.DataFrame) -> list[dict[str, Any]]:
    """Per game: the market's line (strike where the mid ladder crosses 0.5) vs the model's
    expected runs. OLS of realised runs on [1, line_mkt, model_mean - line_mkt] per series,
    decision and split: does the model's number move the market's?"""
    act = {r["game_pk"]: r for r in games.iter_rows(named=True)}
    rows = []
    for series, key in (("KXMLBTOTAL", "total_pmf"), ("KXMLBF5TOTAL", "f5_pmf")):
        q = _with_mid(pred.filter((pl.col("series") == series) & pl.col("bid").is_not_null()))
        for (pk, dec), g in q.group_by("game_pk", "decision"):
            g = g.sort("strike")
            lm = _cross_half(g["strike"].to_numpy(), g["mid"].to_numpy())
            a = act.get(pk)
            if lm is None or a is None or pk not in summaries:
                continue
            if series == "KXMLBTOTAL":
                y = a["home_score"] + a["away_score"]
            else:
                ih, ia = a["inn_home"] or [], a["inn_away"] or []
                if len(ih) < 5 or len(ia) < 5 or None in ih[:5] or None in ia[:5]:
                    continue
                y = sum(ih[:5]) + sum(ia[:5])
            pmf = summaries[pk][key]
            rows.append({"series": series, "decision": dec, "split": "holdout" if g["t"][0] >= HOLDOUT_FROM else "dev",
                         "line_mkt": lm, "mean_model": float(np.arange(len(pmf)) @ pmf), "y": float(y)})
    df = pl.DataFrame(rows)
    out = []
    for (series, dec, split), g in df.group_by("series", "decision", "split"):
        if g.height < 10:
            continue
        X = np.column_stack([np.ones(g.height), g["line_mkt"].to_numpy(), g["mean_model"].to_numpy() - g["line_mkt"].to_numpy()])
        y = g["y"].to_numpy()
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        XtXi = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtXi @ (X.T * e ** 2) @ X @ XtXi))     # HC0, one row per game
        out.append({"series": series, "decision": dec, "split": split, "games": g.height,
                    "coef_line_mkt": round(float(b[1]), 3), "t_line_mkt": round(float(b[1] / se[1]), 2),
                    "coef_model_minus_line": round(float(b[2]), 3), "t_model_minus_line": round(float(b[2] / se[2]), 2),
                    "corr_line_model": round(float(np.corrcoef(g["line_mkt"], g["mean_model"])[0, 1]), 3),
                    "mean_line": round(float(g["line_mkt"].mean()), 2), "mean_model": round(float(g["mean_model"].mean()), 2),
                    "mean_y": round(float(y.mean()), 2)})
    return sorted(out, key=lambda r: (r["series"], r["decision"], r["split"]))


def _with_mid(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))


def score_rows(pred: pl.DataFrame, by: list[str], holdout_from: datetime = HOLDOUT_FROM) -> pl.DataFrame:
    """Brier / log loss of model and mid by segment and split (rows with quotes only)."""
    df = _with_mid(pred.filter(pl.col("bid").is_not_null())).with_columns(
        pl.when(pl.col("t") >= holdout_from).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"),
        pl.col("p").clip(0.005, 0.995).alias("pc"), pl.col("mid").clip(0.005, 0.995).alias("mc"))
    return (df.group_by(["split", *by]).agg(
        pl.len().alias("n"), pl.col("event").n_unique().alias("games"),
        ((pl.col("p") - pl.col("y")) ** 2).mean().round(5).alias("brier_model"),
        ((pl.col("mid") - pl.col("y")) ** 2).mean().round(5).alias("brier_mid"),
        (-(pl.col("y") * pl.col("pc").log() + (1 - pl.col("y")) * (1 - pl.col("pc")).log())).mean().round(5).alias("ll_model"),
        (-(pl.col("y") * pl.col("mc").log() + (1 - pl.col("y")) * (1 - pl.col("mc")).log())).mean().round(5).alias("ll_mid"),
        pl.col("p").mean().round(4).alias("mean_p"), pl.col("mid").mean().round(4).alias("mean_mid"),
        pl.col("y").mean().round(4).alias("mean_y"), (pl.col("ask") - pl.col("bid")).mean().round(4).alias("spread"))
        .sort(["split", *by]))


def flat_eval(res: dict) -> dict[str, Any]:
    """One row from a harness ``evaluate`` result: dev/holdout Brier, blend, taker at the dev θ."""
    row: dict[str, Any] = {"rows": res.get("rows"), "games": res.get("events")}
    for sc in res.get("scores", []):
        row[f"brier_model_{sc['split']}"] = sc["brier_model"]
        row[f"brier_mid_{sc['split']}"] = sc["brier_mid"]
        row[f"n_{sc['split']}"] = sc["n"]
    b = res.get("blend") or {}
    row.update({"c_model": b.get("c_model"), "t_c": b.get("t_c"), "b_mid": b.get("b_mid"),
                "brier_blend_holdout": b.get("brier_blend_holdout"), "c_holdout_refit": b.get("c_model_holdout_refit"),
                "t_c_holdout_refit": b.get("t_c_holdout_refit")})
    th = res.get("theta_chosen_on_dev")
    row["theta_dev"] = th
    for tk in res.get("taker", []):
        if tk["theta"] == th:
            row[f"taker_{tk['split']}_trades"] = tk.get("trades")
            row[f"taker_{tk['split']}_pnl_c"] = tk.get("pnl_c")
            row[f"taker_{tk['split']}_t"] = tk.get("t")
    return row


def write_report(s, out: dict[str, Any], chosen: str, selection: pl.DataFrame, extra: dict[str, Any]) -> Path:
    from albert.reports import Report
    from albert.research.fundamental import report

    pred = out["preds"][chosen]
    main = out["evaluate"][(chosen, "KXMLBTOTAL", "pre1h")]
    report(s, "mlb", main, {"main_result": f"{chosen} / KXMLBTOTAL / pre1h", **extra})
    rep = Report(s.reports_dir, "research-fundamental-mlb-detail")
    rep.text("What this is", "Sabermetric half-inning NB model + game simulator vs Kalshi MLB prices. Dev = Aug 15 - "
             "Sep 30 2026 (model fit on 2025 + 2026 before Aug 15), holdout = October 2026 (postseason; model refit on "
             "everything before Oct 1). Clusters = games. See the module docstring of albert.research.mlb for as-of rules.")
    rep.table("Variant selection on dev (pre1h, model log loss pooled over GAME/TOTAL/F5TOTAL/SPREAD)", selection)
    rows = []
    for (variant, series, decision), res in sorted(out["evaluate"].items()):
        rows.append({"variant": variant, "series": series, "decision": decision, **flat_eval(res)})
    rep.table("Every variant x series x decision time (harness evaluate; taker fee 0.07 P(1-P))",
              pl.DataFrame(rows, infer_schema_length=None), save_as="all_results")
    model_preds = pl.concat([pr.with_columns(pl.lit(v).alias("variant")) for v, pr in out["preds"].items()
                             if v != "pyth" and v in out["models"]], how="diagonal_relaxed")
    rep.table("Holdout variants: scores by variant, series, decision, split (rows with quotes)",
              score_rows(model_preds, ["variant", "series", "decision"]), save_as="scores")
    q = _with_mid(pred.filter(pl.col("bid").is_not_null()))
    band = q.with_columns(pl.col("mid").cut([0.2, 0.4, 0.6, 0.8]).alias("mid_band"))
    rep.table("Chosen model by price band (pre1h)", score_rows(band.filter(pl.col("decision") == "pre1h"), ["series", "mid_band"]))
    rep.table("Chosen model: regular season vs postseason (pre1h)",
              score_rows(q.filter(pl.col("decision") == "pre1h").with_columns((pl.col("game_type") == "R").alias("regular")),
                         ["series", "regular"]))
    rep.table("Chosen model: starters publicly listed (KXMLBKS open before t) vs not (pre1h)",
              score_rows(q.filter(pl.col("decision") == "pre1h"), ["series", "starter_public"]))
    tot = q.filter(pl.col("series").is_in(["KXMLBTOTAL", "KXMLBF5TOTAL"]) & (pl.col("decision") == "pre1h"))
    rep.table("Totals by strike (pre1h)", score_rows(tot, ["series", "strike"]), save_as="totals_by_strike")
    rep.table("Calibration, dev, pre1h totals (model p bins)",
              calibration_table(q.filter((pl.col("series") == "KXMLBTOTAL") & (pl.col("decision") == "pre1h") & (pl.col("t") < HOLDOUT_FROM))))
    rep.table("Calibration, dev, pre1h moneyline home leg (model p bins)",
              calibration_table(q.filter((pl.col("series") == "KXMLBGAME") & (pl.col("decision") == "pre1h") & (pl.col("t") < HOLDOUT_FROM))))
    tk = []
    qa = _with_mid(model_preds.filter(pl.col("bid").is_not_null()))
    for (variant, series, decision), part in qa.group_by("variant", "series", "decision"):
        part = part.filter(pl.col("side").is_null() | (pl.col("side") == "home") | (pl.col("series") != "KXMLBGAME"))
        rows_t, th = taker_table(part)
        tk += [{"variant": variant, "series": series, "decision": decision, "theta_dev": th, **r}
               for r in rows_t if r["theta"] == th]
    if tk:
        rep.table("Taker with each series' real fee multiplier (0.5; inning totals 1.0), theta chosen on dev",
                  pl.DataFrame(tk, infer_schema_length=None).sort("variant", "series", "decision", "split"))
    lt = [{"variant": v, **r} for v, rows_ in out.get("line_test", {}).items() for r in rows_]
    if lt:
        rep.table("Totals line test: realised runs ~ market line + (model mean - market line), OLS per game, HC0 t",
                  pl.DataFrame(lt))
    if out.get("ladder") is not None:
        rep.table("Totals ladder: expected-log-growth stakes across strikes vs single best contract (quarter Kelly, cap 25%)",
                  pl.DataFrame(out["ladder"]))
    rep.json("Frozen model", {k: v for k, v in extra.items() if k.startswith("model")})
    return rep.save()


if __name__ == "__main__":
    import sys

    from albert.config import settings

    if sys.argv[1:] == ["fetch"]:
        fetch_all(settings())
    elif sys.argv[1:2] == ["quotes"]:
        from albert.research.fundamental import settled_markets
        st = settings()
        fetch_quotes(st, settled_markets(st, list(SERIES), DEV_FROM, datetime(2026, 11, 2, tzinfo=UTC)),
                     only=sys.argv[2] if len(sys.argv) > 2 else None)
