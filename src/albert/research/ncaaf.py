"""College football team-strength models vs Kalshi (``KXNCAAFGAME`` moneylines and the
``KXNCAAFSPREAD`` / ``KXNCAAFTOTAL`` / ``KXNCAAFTEAMTOTAL`` ladders).

Data
----
* **Results:** ESPN's public scoreboard JSON (``site.api.espn.com/.../college-football/
  scoreboard``), one request per (season, season type, week, division group): FBS (group 80) and
  FCS (group 81), seasons 2023-2026. Cached under ``data/cache/ncaaf/espn/`` (zstd JSON), 0.6 s
  between network requests. Each game carries its scheduled kickoff (``date``), neutral-site and
  conference flags, both scores and the final status.
* **Markets:** every settled leg of the four series (``fundamental.settled_markets``), matched
  to an ESPN game by kickoff date and team names (see :func:`match_events`).
* **Quotes:** hourly Kalshi candles (yes bid / ask at the candle close), fetched for the window
  ``[kickoff - 27 h, kickoff - 1 h]`` only; live markets in batches of one game's ladder per
  request, historical markets (week 1) one per request.

As-of discipline (no look-ahead)
--------------------------------
A prediction for a game at decision time ``t`` (24 h or 2 h before ESPN's scheduled kickoff)
uses only games whose kickoff + 4.5 h is at or before ``t`` (a college game lasts about 3.5 h,
so the result was public by then). Kickoff, venue, neutral site and conference are scheduled
facts known weeks ahead. Preseason priors come only from earlier seasons' results. All model
hyperparameters (prior regression, ridge strength, home edge, residual scale, tail shape) are
fitted on 2023-2025 and on Aug-Sep 2026 games only; the October 2026 holdout is scored once.

Models
------
* **Massey/ridge (``ridge``):** margin = h·home + r_home − r_away + ε, solved by ridge
  regression towards a preseason prior (last season's rating × ρ, by conference tier for new
  teams), with games weighted by recency. Re-solved before each decision time.
* **Offense/defense (``od``):** points_a = μ + h/2 + o_a − d_b + ε, the same for b; gives both
  team totals, the margin and the total. A Gaussian state-space (random-walk) view of strength is
  approximated by exponential recency weights (half-life chosen on dev), i.e. a discounted
  least-squares Kalman smoother with a ridge prior.
* **Score distribution:** margin and total residuals are fitted on dev games (normal vs scaled
  t, with key-number effects ignored); win probability, cover and over/under probabilities come
  from that distribution. Spread/total strikes are half points, so there are no pushes.
"""

import hashlib
import json
import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
SERIES = ("KXNCAAFGAME", "KXNCAAFSPREAD", "KXNCAAFTOTAL", "KXNCAAFTEAMTOTAL")
EASTERN = ZoneInfo("America/New_York")
GAME_LENGTH = timedelta(hours=4, minutes=30)  # kickoff -> result public (conservative)

# ESPN conference ids (competitor.team.conferenceId). Notre Dame (team 87) counts as Power 4.
POWER4 = {"1", "4", "5", "8"}                  # ACC, Big 12, Big Ten, SEC
GROUP5 = {"9", "12", "15", "17", "18", "37", "151"}  # Pac-12, C-USA, MAC, MWC, FBS Ind., Sun Belt, AAC
P4_TEAMS = {"87"}                              # Notre Dame


# ---- ESPN fetch + cache ---------------------------------------------------------------------
_last_request = [0.0]


def espn_get(s, path: str, params: dict[str, Any], *, min_interval: float = 0.6,
             cache_dir: Path | None = None) -> dict[str, Any]:
    """GET an ESPN site-API JSON once; later calls read the zstd cache under data/cache/ncaaf."""
    import httpx

    from albert.http import read_raw, write_raw

    cache_dir = cache_dir or s.data_dir / "cache" / "ncaaf" / "espn"
    key = hashlib.sha256((path + "?" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
    file = cache_dir / f"{slug}-{key}.json.zst"
    if file.exists():
        return read_raw(file)["response"]
    wait = min_interval - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        _last_request[0] = time.monotonic()
        r = httpx.get(f"{ESPN_BASE}/{path}", params=params, timeout=30,
                      headers={"User-Agent": "albert-research (public data, cached)"})
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    else:
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


GAME_SCHEMA = {
    "game_id": pl.String, "season": pl.Int64, "season_type": pl.Int64, "week": pl.Int64,
    "group": pl.String, "kickoff": pl.Datetime("us", "UTC"), "time_valid": pl.Boolean,
    "completed": pl.Boolean, "status": pl.String, "neutral": pl.Boolean, "conf_game": pl.Boolean,
    "home_id": pl.String, "home_abbr": pl.String, "home_loc": pl.String, "home_name": pl.String,
    "home_short": pl.String, "home_conf": pl.String, "home_score": pl.Float64,
    "away_id": pl.String, "away_abbr": pl.String, "away_loc": pl.String, "away_name": pl.String,
    "away_short": pl.String, "away_conf": pl.String, "away_score": pl.Float64,
}


def game_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    """One row per ESPN scoreboard event (home/away by ``homeAway``)."""
    rows = []
    for e in payload.get("events") or []:
        comp = (e.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if set(teams) != {"home", "away"}:
            continue
        status = (comp.get("status") or e.get("status") or {}).get("type") or {}
        row = {"game_id": str(e.get("id")), "season": (e.get("season") or {}).get("year"),
               "season_type": (e.get("season") or {}).get("type"), "week": (e.get("week") or {}).get("number"),
               "group": group, "kickoff": _ts(comp.get("date") or e.get("date")),
               "time_valid": comp.get("timeValid"), "completed": bool(status.get("completed")),
               "status": status.get("name"), "neutral": bool(comp.get("neutralSite")),
               "conf_game": bool(comp.get("conferenceCompetition"))}
        for side in ("home", "away"):
            c = teams[side]
            t = c.get("team") or {}
            score = c.get("score")
            row.update({f"{side}_id": str(t.get("id")), f"{side}_abbr": t.get("abbreviation"),
                        f"{side}_loc": t.get("location"), f"{side}_name": t.get("displayName"),
                        f"{side}_short": t.get("shortDisplayName"), f"{side}_conf": t.get("conferenceId"),
                        f"{side}_score": float(score) if score not in (None, "") else None})
        rows.append(row)
    return rows


def fetch_games(s, seasons: Iterable[int] = (2023, 2024, 2025, 2026), groups: Iterable[str] = ("80", "81"),
                max_week: int = 16) -> pl.DataFrame:
    """Every FBS/FCS game of these seasons (regular season weeks 1..max_week + postseason)."""
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for group in groups:
            pages = [(2, w) for w in range(1, max_week + 1)] + [(3, 1)]
            for stype, week in pages:
                payload = espn_get(s, "scoreboard", {"dates": str(season), "seasontype": str(stype),
                                                     "week": str(week), "groups": group, "limit": "500"})
                rows.extend(game_rows(payload, group))
    g = pl.DataFrame(rows, schema=GAME_SCHEMA)
    # The same game appears in both group feeds (FBS vs FCS): keep one row, FBS label first.
    return g.sort("group").unique("game_id", keep="first").sort("kickoff", "game_id")


def team_tiers(games: pl.DataFrame, *, min_fcs_games: int = 5) -> dict[tuple[int, str], str]:
    """(season, team id) -> "P4" / "G5" / "FCS" / "other". FBS tiers come from the team's ESPN
    conference id; a non-FBS team is FCS if it plays at least ``min_fcs_games`` games in the
    FBS/FCS feeds that season (Division II/III opponents appear only once or twice)."""
    out: dict[tuple[int, str], str] = {}
    n_games: dict[tuple[int, str], int] = {}
    for r in games.iter_rows(named=True):
        for side in ("home", "away"):
            key = (r["season"], r[f"{side}_id"])
            n_games[key] = n_games.get(key, 0) + 1
            conf = r[f"{side}_conf"]
            if r[f"{side}_id"] in P4_TEAMS or conf in POWER4:
                out[key] = "P4"
            elif conf in GROUP5 and out.get(key) != "P4":
                out[key] = "G5"
    for key, n in n_games.items():
        if key not in out:
            out[key] = "FCS" if n >= min_fcs_games else "other"
    return out


# ---- Kalshi <-> ESPN matching ---------------------------------------------------------------
_EVENT_RX = re.compile(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})([A-Z0-9]+)$")
_MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def event_key(event_ticker: str) -> tuple[datetime, str] | None:
    """``KXNCAAFSPREAD-26AUG27MHUYSU`` -> (2026-08-27, "MHUYSU"): the game's US-Eastern date
    and the concatenated team codes, shared by every series of one game."""
    m = _EVENT_RX.match(event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    yy, mon, dd, codes = m.groups()
    return datetime(2000 + int(yy), _MONTHS[mon], int(dd)), codes


def norm_name(x: str | None) -> str:
    x = (x or "").lower().replace("&", " and ").replace("é", "e").replace("'", "")
    x = re.sub(r"\(([a-z]+)\)", r" \1 ", x)
    x = re.sub(r"\bst\.?$", "state", x.strip())
    x = re.sub(r"\bst\.?\s", "state ", x) if not re.match(r"^st\.?\s", x) else re.sub(r"^st\.?\s", "saint ", x)
    x = re.sub(r"\buniv(ersity)?\b", "", x)
    x = re.sub(r"[^a-z0-9]+", " ", x).strip()
    return _ALIASES.get(x, x)


_ALIASES = {"north carolina state": "nc state", "nc state wolfpack": "nc state"}


def _name_score(kname: str, kcode: str, row: dict[str, Any], side: str) -> float:
    from difflib import SequenceMatcher

    k = norm_name(kname)
    best = 0.0
    for v in (row[f"{side}_loc"], row[f"{side}_short"], row[f"{side}_name"]):
        n = norm_name(v)
        if not n:
            continue
        if k == n:
            return 1.0
        r = SequenceMatcher(None, k, n).ratio()
        if n.startswith(k + " ") or k.startswith(n + " "):  # "anderson sc trojans" vs "anderson sc"
            r = max(r, 0.9)
        best = max(best, r)
    if kcode and kcode == (row[f"{side}_abbr"] or "").upper():
        best = max(best, 0.85) + 0.1
    return min(best, 1.0)


def match_events(game_legs: pl.DataFrame, games: pl.DataFrame, *, min_score: float = 0.72) -> pl.DataFrame:
    """Map each Kalshi ``KXNCAAFGAME`` event (two legs: ticker code + ``yes_sub_title``) to the
    ESPN game on the same Eastern date (±1 day) whose two teams best match both legs.

    Returns event key (codes + date), game_id, and for each leg the ESPN team id and side."""
    games = games.with_columns(pl.col("kickoff").dt.convert_time_zone("America/New_York").dt.date().alias("et_date"))
    by_date: dict[Any, list[dict[str, Any]]] = {}
    for r in games.iter_rows(named=True):
        by_date.setdefault(r["et_date"], []).append(r)
    out = []
    for ev, legs in game_legs.group_by("event_ticker"):
        legs = legs.sort("ticker").to_dicts()
        key = event_key(ev[0])
        if key is None or len(legs) != 2:
            continue
        day = key[0].date()
        cands = [g for d in (day, day + timedelta(days=1), day - timedelta(days=1)) for g in by_date.get(d, [])]
        best = None
        for g in cands:
            codes = [leg["ticker"].rsplit("-", 1)[1] for leg in legs]
            s_ha = [_name_score(legs[0]["yes_sub_title"], codes[0], g, "home"),
                    _name_score(legs[1]["yes_sub_title"], codes[1], g, "away")]
            s_ah = [_name_score(legs[0]["yes_sub_title"], codes[0], g, "away"),
                    _name_score(legs[1]["yes_sub_title"], codes[1], g, "home")]
            for sides, sc in ((("home", "away"), s_ha), (("away", "home"), s_ah)):
                tot = min(sc) + 0.01 * sum(sc) - (0.05 if g["et_date"] != day else 0.0)
                if best is None or tot > best[0]:
                    best = (tot, g, sides, sc)
        if best is None or min(best[3]) < min_score:
            out.append({"event_key": key[1] + key[0].strftime("%y%m%d"), "kalshi_event": ev[0], "game_id": None,
                        "score": None if best is None else round(min(best[3]), 3),
                        "leg0": legs[0]["ticker"], "leg0_name": legs[0]["yes_sub_title"],
                        "leg1": legs[1]["ticker"], "leg1_name": legs[1]["yes_sub_title"],
                        "espn": None if best is None else f"{best[1]['away_name']} @ {best[1]['home_name']}"})
            continue
        _, g, sides, sc = best
        out.append({"event_key": key[1] + key[0].strftime("%y%m%d"), "kalshi_event": ev[0], "game_id": g["game_id"],
                    "score": round(min(sc), 3), "leg0": legs[0]["ticker"], "leg0_name": legs[0]["yes_sub_title"],
                    "leg0_side": sides[0], "leg1": legs[1]["ticker"], "leg1_name": legs[1]["yes_sub_title"],
                    "leg1_side": sides[1], "espn": f"{g['away_name']} @ {g['home_name']}"})
    return pl.DataFrame(out, infer_schema_length=None)


_STRIKE_RX = re.compile(r"over (\d+(?:\.\d+)?) points", re.IGNORECASE)


def ladder_legs(markets: pl.DataFrame, match: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Every settled leg of the four series joined to its ESPN game, with what the leg pays on:

    * ``kind = "ml"``: team ``team_side`` wins (KXNCAAFGAME);
    * ``"spread"``: team ``team_side`` wins by more than ``strike`` (KXNCAAFSPREAD);
    * ``"total"``: combined points > ``strike`` (KXNCAAFTOTAL);
    * ``"team_total"``: team ``team_side`` scores > ``strike`` (KXNCAAFTEAMTOTAL).

    ``team_side`` is "home"/"away" in ESPN's terms (the team code in the ticker suffix is the
    same code the game legs use). ``y_espn`` recomputes the result from ESPN's final score as a
    check on the parsing (it must agree with Kalshi's ``y``)."""
    mt = match.filter(pl.col("game_id").is_not_null())
    code_side: dict[tuple[str, str], str] = {}
    for r in mt.iter_rows(named=True):
        for leg in ("leg0", "leg1"):
            code_side[(r["event_key"], r[leg].rsplit("-", 1)[1])] = r[f"{leg}_side"]
    game_of = dict(zip(mt["event_key"], mt["game_id"]))
    kinds = {"KXNCAAFGAME": "ml", "KXNCAAFSPREAD": "spread", "KXNCAAFTOTAL": "total", "KXNCAAFTEAMTOTAL": "team_total"}
    rows = []
    for r in markets.filter(pl.col("series_ticker").is_in(list(kinds))).iter_rows(named=True):
        key = event_key(r["event_ticker"])
        if key is None:
            continue
        # A ladder event can carry a date one day off its moneyline event (late kickoffs).
        eks = [key[1] + (key[0] + timedelta(days=d)).strftime("%y%m%d") for d in (0, -1, 1)]
        ek = next((e for e in eks if e in game_of), None)
        if ek is None:
            continue
        gid = game_of[ek]
        kind = kinds[r["series_ticker"]]
        suffix = r["ticker"].rsplit("-", 1)[1]
        side, strike = None, None
        if kind != "ml":
            sm = _STRIKE_RX.search(r["title"] or "")
            if sm is None:
                continue
            strike = float(sm.group(1))
        if kind in ("ml", "spread", "team_total"):
            code = suffix if kind == "ml" else re.sub(r"\d+$", "", suffix)
            side = code_side.get((ek, code))
            if side is None:
                continue
        rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "series": r["series_ticker"],
                     "event_key": ek, "game_id": gid, "kind": kind, "team_side": side, "strike": strike,
                     "y": int(r["y"]), "open_time": r["open_time"], "close_time": r["close_time"]})
    legs = pl.DataFrame(rows, infer_schema_length=None)
    g = games.select("game_id", "kickoff", "home_score", "away_score", "status")
    legs = legs.join(g, on="game_id", how="left")
    sign = pl.when(pl.col("team_side") == "home").then(1.0).otherwise(-1.0)
    margin = sign * (pl.col("home_score") - pl.col("away_score"))
    team_pts = pl.when(pl.col("team_side") == "home").then(pl.col("home_score")).otherwise(pl.col("away_score"))
    y_espn = (pl.when(pl.col("kind") == "ml").then(margin > 0)
              .when(pl.col("kind") == "spread").then(margin > pl.col("strike"))
              .when(pl.col("kind") == "total").then(pl.col("home_score") + pl.col("away_score") > pl.col("strike"))
              .otherwise(team_pts > pl.col("strike")))
    return legs.with_columns(y_espn.cast(pl.Int8).alias("y_espn"))


def kalshi_markets(s, *, refresh: bool = False) -> pl.DataFrame:
    """Settled legs of the four series: the shared markets table, completed with a direct
    ``/markets?series_ticker=..&status=settled`` pull (the table misses ~90 October games'
    moneylines). Cached in ``data/cache/fundamental_ncaaf_markets.parquet``."""
    from albert.http import CachedClient
    from albert.research.fundamental import settled_markets
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / "fundamental_ncaaf_markets.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    base = settled_markets(s, list(SERIES))
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.2))
    rows: list[dict[str, Any]] = []
    for series in SERIES:
        for attempt in range(6):  # pages are cached, so a restart after a 429 re-reads them free
            try:
                got: list[dict[str, Any]] = []
                for page in api.series_markets(series, "settled", min_close=datetime(2026, 8, 1, tzinfo=UTC),
                                               max_close=datetime(2026, 11, 1, tzinfo=UTC), cache=True):
                    got.extend(k.market_rows(page.payload, {}, "ncaaf-series-pull"))
                rows.extend(got)
                break
            except RuntimeError as error:
                print(f"retry {series}: {str(error)[:100]}")
                time.sleep(10 * (attempt + 1))
    extra = pl.DataFrame(rows, schema=k.MARKET_SCHEMA)
    extra = (extra.filter(pl.col("result").is_in(["yes", "no"]) & ~pl.col("ticker").is_in(base["ticker"].implode()))
             .with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y")))
    out = pl.concat([base, extra.select(base.columns)], how="vertical_relaxed")
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    return out


# ---- quotes -------------------------------------------------------------------------------------
QUOTE_WINDOW = (timedelta(hours=27), timedelta(hours=1))  # [kickoff - 27 h, kickoff - 1 h]


def fetch_ladder_candles(s, legs: pl.DataFrame, *, workers: int = 2, max_tickers: int = 100,
                         hist_kinds: tuple[str, ...] = ("ml",)) -> pl.DataFrame:
    """Hourly candles for every leg over ``[kickoff - 27 h, kickoff - 1 h]``. Live markets (closed
    after Kalshi's historical cutoff, 2026-09-06) go in batches of one game's legs per request;
    historical ones (week 1) need one request per leg, so only ``hist_kinds`` are fetched there
    (week-1 ladders would cost ~6,000 requests on a shared rate limit). Cached in
    ``data/cache/fundamental_ncaaf_candles.parquet``."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / "fundamental_ncaaf_candles.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["ticker"].unique()) if have is not None and have.height else set()
    todo = legs.filter(~pl.col("ticker").is_in(list(done)) & pl.col("kickoff").is_not_null())
    if not todo.height:
        return have
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.1))
    cutoff = api.cutoff().get("market_settled_ts")
    jobs: list[tuple[str, list[str], datetime, datetime]] = []
    for (gid,), grp in todo.group_by("game_id"):
        kick = grp["kickoff"][0]
        lo, hi = kick - QUOTE_WINDOW[0], kick - QUOTE_WINDOW[1]
        hist = (grp.filter((pl.col("close_time") < cutoff) & pl.col("kind").is_in(list(hist_kinds)))["ticker"].to_list()
                if cutoff else [])
        live = grp.filter(pl.col("close_time") >= cutoff)["ticker"].to_list() if cutoff else grp["ticker"].to_list()
        jobs += [("hist", [t], lo, hi) for t in hist]
        jobs += [("live", live[i:i + max_tickers], lo, hi) for i in range(0, len(live), max_tickers)]

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
                print(f"skip candles {tickers[0]} (+{len(tickers) - 1}): {str(error)[:120]}")
                return []
        return []

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(one, jobs), 1):
            rows.extend(r)
            if i % 200 == 0:
                print(f"  candle jobs {i}/{len(jobs)}, {len(rows)} rows", flush=True)
    new = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else None
    frames = [f for f in (have, new) if f is not None and f.height]
    out = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"])
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    return out


# ---- rating models ------------------------------------------------------------------------------
@dataclass(frozen=True)
class KFParams:
    """Hyperparameters of the Gaussian state-space rating model (points units).

    ``split=True``: offense/defense states per team (both team scores observed);
    ``split=False``: one strength per team (only the margin is observed; with ``q_day = 0`` this
    is exactly a ridge/Massey regression towards the preseason prior, refit on every result).
    """
    split: bool = True
    sigma: float = 13.0          # SD of one team's score (split) / of the margin (not split) around the model
    rho_e: float = 0.0           # correlation of the two teams' score errors (split only)
    q_day: float = 0.02          # random-walk variance per team-state per day in season
    rho_off: float = 0.75        # offseason regression of each state towards its tier mean
    q_off: float = 20.0          # extra offseason variance per state
    p0: float = 60.0             # prior variance of a team's state when first seen
    clip: float | None = None    # robust update: clip the standardised innovation at ±clip
    home: float = 3.0            # initial home edge (a state with small process noise)
    mu: float = 26.5             # initial points per team (split only; a state)
    tier_init: tuple[tuple[str, float], ...] = (("P4", 12.0), ("G5", 0.0), ("FCS", -14.0), ("other", -30.0))


def _event_stream(games: pl.DataFrame, queries: pl.DataFrame) -> list[tuple]:
    """Results become visible at kickoff + GAME_LENGTH (or at ``result_at`` when that column
    is present and later); queries are answered at their ``t``. At equal times a query is
    answered first (strictly-before rule for results)."""
    ev: list[tuple] = []
    done = games.filter(pl.col("completed") & pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    for r in done.iter_rows(named=True):
        seen_at = r["kickoff"] + GAME_LENGTH
        if r.get("result_at") is not None:
            seen_at = max(seen_at, r["result_at"])
        ev.append((seen_at, 1, "result", r))
    for r in queries.iter_rows(named=True):
        ev.append((r["t"], 0, "query", r))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def run_kf(games: pl.DataFrame, queries: pl.DataFrame, prm: KFParams,
           tiers: dict[tuple[int, str], str]) -> pl.DataFrame:
    """One causal pass over results and queries. ``queries``: game_id, t, season, home_id,
    away_id, neutral (one row per prediction). Returns, per query, the predictive mean and SD of
    the home score, away score, margin (home - away) and total, and the pre-game variance.

    Offseason (first event of a new season): every state is pulled towards its tier mean,
    x <- m + rho_off (x - m), P <- rho_off^2 P + q_off I. New teams start at the current mean of
    their tier (or ``tier_init`` before any team of the tier exists) with variance ``p0``."""
    ids = sorted(set(games["home_id"]) | set(games["away_id"]) | set(queries["home_id"]) | set(queries["away_id"]))
    idx = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    k = 2 if prm.split else 1
    N = k * n + 2                     # team states (+ defence block), home edge, league mean
    H_I, MU_I = k * n, k * n + 1
    x = np.zeros(N)
    P = np.zeros((N, N))
    x[H_I], P[H_I, H_I] = prm.home, 1.0
    x[MU_I], P[MU_I, MU_I] = (prm.mu, 4.0) if prm.split else (0.0, 0.0)
    seen = np.zeros(n, dtype=bool)
    tier_of: dict[str, str] = {}
    init = dict(prm.tier_init)
    season = None
    last_t = None
    R = prm.sigma ** 2 * (np.array([[1.0, prm.rho_e], [prm.rho_e, 1.0]]) if prm.split else np.eye(1))
    out: list[dict[str, Any]] = []

    def tier_mean(tier: str) -> np.ndarray:
        members = [idx[t] for t, tt in tier_of.items() if tt == tier and seen[idx[t]]]
        if len(members) >= 5:
            m = np.array([x[i] for i in members]).mean()
            return (np.array([m, np.array([x[n + i] for i in members]).mean()]) if prm.split else np.array([m]))
        r = init.get(tier, init["other"])
        return np.array([r / 2, r / 2]) if prm.split else np.array([r])

    def ensure(team: str, ssn: int) -> None:
        i = idx[team]
        tier_of[team] = tiers.get((ssn, team), tier_of.get(team, "other"))
        if not seen[i]:
            m = tier_mean(tier_of[team])
            for b in range(k):
                x[b * n + i] = m[b]
                P[b * n + i, b * n + i] = prm.p0 / k
            seen[i] = True

    def offseason() -> None:
        nonlocal P
        means = {tier: tier_mean(tier) for tier in set(tier_of.values())}
        A = np.ones(N)
        for team, tier in tier_of.items():
            i = idx[team]
            if not seen[i]:
                continue
            for b in range(k):
                j = b * n + i
                x[j] = means[tier][b] + prm.rho_off * (x[j] - means[tier][b])
                A[j] = prm.rho_off
        P = P * A[:, None] * A[None, :]
        d = np.arange(k * n)
        P[d, d] += np.where(np.tile(seen, k), prm.q_off / k, 0.0)
        P[H_I, H_I] += 0.25
        if prm.split:
            P[MU_I, MU_I] += 1.0

    def design(r: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Sparse observation matrix: the state columns it touches and their coefficients."""
        h, a = idx[r["home_id"]], idx[r["away_id"]]
        hm = 0.0 if r["neutral"] else 1.0
        if prm.split:  # rows: home points, away points; cols: o_h, o_a, d_h, d_a, home, mu
            cols = np.array([h, a, n + h, n + a, H_I, MU_I])
            return cols, np.array([[1.0, 0.0, 0.0, -1.0, hm / 2, 1.0],
                                   [0.0, 1.0, -1.0, 0.0, -hm / 2, 1.0]])
        return np.array([h, a, H_I]), np.array([[1.0, -1.0, hm]])

    team_diag = np.arange(k * n)
    for step, (t, _, kind, r) in enumerate(_event_stream(games, queries)):
        if season is not None and r["season"] != season:
            offseason()
        season = r["season"]
        if last_t is not None and prm.q_day > 0:
            days = (t - last_t).total_seconds() / 86400
            if 0 < days < 60:
                P[team_diag, team_diag] += np.where(np.tile(seen, k), prm.q_day * days / k, 0.0)
        last_t = t
        ensure(r["home_id"], season)
        ensure(r["away_id"], season)
        cols, Hc = design(r)
        mean = Hc @ x[cols]
        cov = Hc @ P[np.ix_(cols, cols)] @ Hc.T
        if kind == "query":
            if prm.split:
                vh, va, c = cov[0, 0] + R[0, 0], cov[1, 1] + R[1, 1], cov[0, 1] + R[0, 1]
                out.append({**{f: r[f] for f in ("game_id", "t")}, "mu_home": mean[0], "mu_away": mean[1],
                            "sd_home": math.sqrt(vh), "sd_away": math.sqrt(va),
                            "mu_margin": mean[0] - mean[1], "sd_margin": math.sqrt(vh + va - 2 * c),
                            "mu_total": mean[0] + mean[1], "sd_total": math.sqrt(vh + va + 2 * c),
                            "var_state_margin": cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]})
            else:
                out.append({**{f: r[f] for f in ("game_id", "t")}, "mu_margin": mean[0],
                            "sd_margin": math.sqrt(cov[0, 0] + R[0, 0]), "var_state_margin": cov[0, 0]})
            continue
        y = (np.array([r["home_score"], r["away_score"]]) if prm.split
             else np.array([r["home_score"] - r["away_score"]]))
        S = cov + R
        v = y - mean
        if prm.clip is not None:
            sd = np.sqrt(np.diag(S))
            v = np.clip(v, -prm.clip * sd, prm.clip * sd)
        PH = P[:, cols] @ Hc.T
        K = PH @ np.linalg.inv(S)
        x += K @ v
        P -= K @ PH.T
        if step % 500 == 0:
            P = (P + P.T) / 2
    return pl.DataFrame(out, infer_schema_length=None)


# ---- public sportsbook lines (ESPN core API: DraftKings open / close) -------------------------
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"


def _american(x: Any) -> float | None:
    try:
        v = str(x).replace("+", "")
        return None if v in ("", "None", "EVEN") else float(v)
    except ValueError:
        return 100.0 if str(x).upper() == "EVEN" else None


def odds_row(payload: dict[str, Any], game_id: str) -> dict[str, Any] | None:
    """DraftKings (or the first provider) open and close lines for one game: home spread (points,
    negative = home favoured), total, and home/away moneylines (American odds)."""
    items = payload.get("items") or []
    if not items:
        return None
    it = next((i for i in items if (i.get("provider") or {}).get("name") == "DraftKings"), items[0])
    row: dict[str, Any] = {"game_id": game_id, "provider": (it.get("provider") or {}).get("name")}
    for when in ("open", "close"):
        home = (it.get("homeTeamOdds") or {}).get(when) or {}
        away = (it.get("awayTeamOdds") or {}).get(when) or {}
        tot = it.get(when) or {}
        row[f"spread_{when}"] = _american((home.get("pointSpread") or {}).get("american"))
        row[f"total_{when}"] = _american((tot.get("total") or {}).get("american"))
        row[f"ml_home_{when}"] = _american((home.get("moneyLine") or {}).get("american"))
        row[f"ml_away_{when}"] = _american((away.get("moneyLine") or {}).get("american"))
    return row


def fetch_odds(s, game_ids: Iterable[str]) -> pl.DataFrame:
    """Per-game ESPN core odds (cached like the scoreboard)."""
    import httpx

    from albert.http import read_raw, write_raw

    cache_dir = s.data_dir / "cache" / "ncaaf" / "espn_odds"
    rows = []
    for gid in game_ids:
        file = cache_dir / f"odds-{gid}.json.zst"
        if file.exists():
            payload = read_raw(file)["response"]
        else:
            wait = 0.6 - (time.monotonic() - _last_request[0])
            if wait > 0:
                time.sleep(wait)
            _last_request[0] = time.monotonic()
            url = f"{ESPN_CORE}/events/{gid}/competitions/{gid}/odds"
            try:
                r = httpx.get(url, timeout=30, headers={"User-Agent": "albert-research (public data, cached)"})
            except httpx.HTTPError as error:
                print(f"odds {gid}: {error}")
                continue
            if r.status_code != 200:
                print(f"odds {gid}: {r.status_code}")
                continue
            payload = r.json()
            write_raw(file, {"url": url, "params": {}, "fetched_at": datetime.now(UTC).isoformat(), "response": payload})
        row = odds_row(payload, gid)
        if row:
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- score distribution -> contract probabilities ----------------------------------------------
@dataclass(frozen=True)
class Dist:
    """Standardised residual law: (y - mu) / (scale * sd) ~ normal (df None) or Student t(df)."""
    scale: float = 1.0
    df: float | None = None

    def sf(self, z: np.ndarray) -> np.ndarray:
        from scipy import stats

        z = np.asarray(z, dtype=float) / self.scale
        return stats.norm.sf(z) if self.df is None else stats.t.sf(z, self.df)

    def logpdf(self, z: np.ndarray) -> np.ndarray:
        from scipy import stats

        z = np.asarray(z, dtype=float)
        base = stats.norm.logpdf(z / self.scale) if self.df is None else stats.t.logpdf(z / self.scale, self.df)
        return base - math.log(self.scale)


def fit_dist(z: np.ndarray) -> dict[str, Any]:
    """MLE of the residual law on standardised residuals: normal (scale) vs Student t (df,
    scale). Returns both fits with their mean log likelihood."""
    from scipy import optimize, stats

    z = z[np.isfinite(z)]
    sn = float(np.sqrt((z ** 2).mean()))
    normal = Dist(sn, None)

    def nll(v: np.ndarray) -> float:
        return -float(stats.t.logpdf(z / math.exp(v[1]), math.exp(v[0])).mean() - v[1])

    r = optimize.minimize(nll, np.array([math.log(10.0), math.log(sn)]), method="Nelder-Mead")
    tdist = Dist(math.exp(r.x[1]), math.exp(r.x[0]))
    return {"normal": normal, "t": tdist, "n": int(z.size),
            "ll_normal": float(normal.logpdf(z).mean()), "ll_t": float(tdist.logpdf(z).mean()),
            "kurtosis": float(stats.kurtosis(z))}


def leg_probabilities(legs: pl.DataFrame, pred: pl.DataFrame, dists: dict[str, Dist]) -> pl.DataFrame:
    """``legs`` (ticker, game_id, kind, team_side, strike, t) joined to the model's predictive
    means/SDs at the same (game_id, t); returns ``p`` = P(YES). ``dists`` holds the residual law
    for "margin", "total" and "team" (team points). Kinds the model cannot price get null."""
    df = legs.join(pred, on=["game_id", "t"], how="inner")
    sign = np.where(df["team_side"].to_numpy() == "home", 1.0, -1.0)
    kind = df["kind"].to_numpy()
    strike = df["strike"].fill_null(0.0).to_numpy()
    p = np.full(df.height, np.nan)
    mu_m, sd_m = sign * df["mu_margin"].to_numpy(), df["sd_margin"].to_numpy()
    for kd in ("ml", "spread"):
        m = kind == kd
        k = 0.0 if kd == "ml" else strike[m]
        p[m] = dists["margin"].sf((k - mu_m[m]) / sd_m[m])
    if "mu_total" in df.columns:
        m = kind == "total"
        p[m] = dists["total"].sf((strike[m] - df["mu_total"].to_numpy()[m]) / df["sd_total"].to_numpy()[m])
        m = kind == "team_total"
        home = df["team_side"].to_numpy() == "home"
        mu_t = np.where(home, df["mu_home"].to_numpy(), df["mu_away"].to_numpy())
        sd_t = np.where(home, df["sd_home"].to_numpy(), df["sd_away"].to_numpy())
        p[m] = dists["team"].sf((strike[m] - mu_t[m]) / sd_t[m])
    return df.with_columns(pl.Series("p", np.clip(p, 0.001, 0.999)).fill_nan(None))


# ---- study pipeline -----------------------------------------------------------------------------
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)   # dev: games before; holdout: October 2026
LEADS_H = (24, 2)                             # decision times: hours before ESPN's scheduled kickoff


def load_inputs(s, *, fetch: bool = False) -> dict[str, Any]:
    """Games (ESPN), team tiers, Kalshi legs matched to games (FINAL games only) and candles."""
    cache = s.data_dir / "cache"
    games_path = cache / "fundamental_ncaaf_games.parquet"
    if fetch or not games_path.exists():
        fetch_games(s).write_parquet(games_path)
    games = pl.read_parquet(games_path).filter(~pl.col("status").is_in(["STATUS_CANCELED", "STATUS_POSTPONED"]))
    tiers = team_tiers(games.filter(pl.col("completed")))
    markets = kalshi_markets(s)
    match = match_events(markets.filter(pl.col("series_ticker") == "KXNCAAFGAME"), games)
    legs = ladder_legs(markets, match, games).filter(pl.col("status") == "STATUS_FINAL")
    # A Kalshi-listed game's result is used only after its moneyline market closed (the game
    # was over), or kickoff + 4.5 h, whichever is later (guards against weather delays).
    ends = legs.filter(pl.col("kind") == "ml").group_by("game_id").agg(pl.col("close_time").max().alias("result_at"))
    games = games.join(ends, on="game_id", how="left")
    candles = fetch_ladder_candles(s, legs) if fetch else pl.read_parquet(cache / "fundamental_ncaaf_candles.parquet")
    return {"games": games, "tiers": tiers, "markets": markets, "match": match, "legs": legs, "candles": candles}


def _query_cols(games: pl.DataFrame) -> pl.DataFrame:
    return games.select("game_id", "season", "home_id", "away_id", "neutral", "kickoff")


def queries_for(games: pl.DataFrame, game_ids: Iterable[str] | None, leads: Iterable[int]) -> pl.DataFrame:
    g = _query_cols(games)
    if game_ids is not None:
        g = g.filter(pl.col("game_id").is_in(list(game_ids)))
    return pl.concat([g.with_columns((pl.col("kickoff") - pl.duration(hours=h)).alias("t"), pl.lit(h).alias("lead"))
                      for h in leads]).drop("kickoff")


def game_tier(games: pl.DataFrame, tiers: dict[tuple[int, str], str]) -> pl.DataFrame:
    """Game tier for segmenting: both teams Power 4 -> "P4"; both FBS -> "G5" (any G5 team);
    any FCS / lower team -> "FCS"."""
    rows = []
    for r in games.select("game_id", "season", "home_id", "away_id").iter_rows(named=True):
        a, b = tiers.get((r["season"], r["home_id"]), "other"), tiers.get((r["season"], r["away_id"]), "other")
        tier = "P4" if a == b == "P4" else ("G5" if {a, b} <= {"P4", "G5"} else "FCS")
        rows.append({"game_id": r["game_id"], "tier": tier})
    return pl.DataFrame(rows)


def dev_residuals(pred: pl.DataFrame, games: pl.DataFrame) -> dict[str, np.ndarray]:
    """Standardised residuals of completed dev games (kickoff before DEV_END)."""
    d = pred.join(games.select("game_id", "kickoff", "home_score", "away_score"), on="game_id")
    d = d.filter(pl.col("kickoff") < DEV_END)
    out = {"margin": ((d["home_score"] - d["away_score"] - d["mu_margin"]) / d["sd_margin"]).to_numpy()}
    if "mu_total" in d.columns:
        out["total"] = ((d["home_score"] + d["away_score"] - d["mu_total"]) / d["sd_total"]).to_numpy()
        out["team"] = np.concatenate([((d["home_score"] - d["mu_home"]) / d["sd_home"]).to_numpy(),
                                      ((d["away_score"] - d["mu_away"]) / d["sd_away"]).to_numpy()])
    return out


def kalshi_frame(legs: pl.DataFrame, lead: int) -> pl.DataFrame:
    """Legs at one decision time. Moneylines keep the home leg only (the away leg is its mirror)."""
    lg = legs.filter((pl.col("kind") != "ml") | (pl.col("team_side") == "home"))
    return lg.with_columns((pl.col("kickoff") - pl.duration(hours=lead)).alias("t"), pl.lit(lead).alias("lead"))


def price_band(mid: pl.Expr) -> pl.Expr:
    return (pl.when(mid < 0.1).then(pl.lit("a <10c")).when(mid < 0.3).then(pl.lit("b 10-30c"))
            .when(mid < 0.7).then(pl.lit("c 30-70c")).when(mid < 0.9).then(pl.lit("d 70-90c"))
            .otherwise(pl.lit("e >=90c")))


def segment_scores(df: pl.DataFrame, by: str) -> pl.DataFrame:
    """Brier of model and mid per segment and split (rows already carry mid, holdout, y, p)."""
    return (df.group_by("holdout", by).agg(
        pl.len().alias("n"), pl.col("event").n_unique().alias("games"),
        ((pl.col("p") - pl.col("y")) ** 2).mean().alias("brier_model"),
        ((pl.col("mid") - pl.col("y")) ** 2).mean().alias("brier_mid"),
        pl.col("p").mean().alias("mean_p"), pl.col("mid").mean().alias("mean_mid"), pl.col("y").mean().alias("mean_y"))
        .sort("holdout", by))


def implied_location(x_strikes: np.ndarray, upper: np.ndarray, mids: np.ndarray, spreads: np.ndarray,
                     scale0: float) -> tuple[float, float] | None:
    """Market-implied (location, scale) of a score variable X from one ladder's mids.

    Each leg pays on ``X > k`` (``upper``) or ``X < k`` (``~upper``); its mid is matched to a
    normal survival function by weighted least squares (weights favour tight, mid-priced legs).
    Returns None with fewer than 2 informative legs (mid in 3-97c), when no leg brackets the
    centre (an "over" price in 15-85c: lopsided games whose ladder stops short of the line
    otherwise extrapolate to absurd locations), or when the fit is implausible (scale outside
    8-30 points, location more than 10 points beyond the strikes)."""
    from scipy import optimize, stats

    ok = (mids > 0.03) & (mids < 0.97)
    if ok.sum() < 2:
        return None
    over = np.where(upper, mids, 1 - mids)
    if not ((over > 0.15) & (over < 0.85) & ok).any():
        return None
    k, up, m = x_strikes[ok], upper[ok], mids[ok]
    w = 1.0 / np.maximum(spreads[ok], 0.01)

    def loss(v: np.ndarray) -> float:
        sf = stats.norm.sf((k - v[0]) / math.exp(v[1]))
        pr = np.where(up, sf, 1 - sf)
        return float((w * (pr - m) ** 2).sum())

    # Start at the strike whose "over" price is nearest 50c.
    p_over = np.where(up, m, 1 - m)
    start = float(k[np.argmin(np.abs(p_over - 0.5))])
    r = optimize.minimize(loss, np.array([start, math.log(scale0)]), method="Nelder-Mead",
                          options={"xatol": 1e-3, "fatol": 1e-8})
    loc, scale = float(r.x[0]), float(math.exp(r.x[1]))
    if not 8.0 <= scale <= 30.0 or not k.min() - 10 <= loc <= k.max() + 10:
        return None
    return loc, scale


def ols_hc(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC1 robust t-stats (one row per game, so rows are independent)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, kx = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - kx)
    return beta, beta / np.sqrt(np.diag(V))


def location_frame(q: pl.DataFrame) -> pl.DataFrame:
    """Per (game, t, variable): the market-implied location of the margin (home - away, from the
    spread + moneyline legs) and of the total (total ladder). ``q`` carries quotes (bid, ask)."""
    rows = []
    q = q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spr"))
    for (gid, t), grp in q.group_by("game_id", "t"):
        m = grp.filter(pl.col("kind").is_in(["ml", "spread"]))
        if m.height:
            home = m["team_side"].to_numpy() == "home"
            k = m["strike"].fill_null(0.0).to_numpy()
            fit = implied_location(np.where(home, k, -k), home, m["mid"].to_numpy(), m["spr"].to_numpy(), 16.0)
            if fit:
                rows.append({"game_id": gid, "t": t, "var": "margin", "loc_mkt": fit[0], "scale_mkt": fit[1], "legs": m.height})
        tt = grp.filter(pl.col("kind") == "total")
        if tt.height:
            fit = implied_location(tt["strike"].to_numpy(), np.ones(tt.height, dtype=bool), tt["mid"].to_numpy(),
                                   tt["spr"].to_numpy(), 16.0)
            if fit:
                rows.append({"game_id": gid, "t": t, "var": "total", "loc_mkt": fit[0], "scale_mkt": fit[1], "legs": tt.height})
    return pl.DataFrame(rows, infer_schema_length=None)


def model_run(inp: dict[str, Any], prm: KFParams, *, leads: Iterable[int] = LEADS_H) -> dict[str, Any]:
    """One causal KF pass answering (a) every completed 2024+ game at kickoff - 2 h (for the
    residual law and rating diagnostics) and (b) every Kalshi-listed game at each lead."""
    games, tiers, legs = inp["games"], inp["tiers"], inp["legs"]
    done = games.filter(pl.col("completed") & (pl.col("kickoff") >= datetime(2024, 8, 1, tzinfo=UTC)))
    q_all = queries_for(done, None, (2,))
    q_k = queries_for(games, legs["game_id"].unique().to_list(), leads)
    q = pl.concat([q_all, q_k]).unique(["game_id", "t"])
    pred = run_kf(games, q.drop("lead"), prm, tiers).join(q.select("game_id", "t", "lead"), on=["game_id", "t"])
    return {"pred": pred, "resid": dev_residuals(pred.filter(pl.col("lead") == 2), games)}


def rating_diagnostics(pred: pl.DataFrame, games: pl.DataFrame, tier: pl.DataFrame) -> pl.DataFrame:
    """RMSE / mean absolute error of the model's margin and total at kickoff - 2 h, per split
    and game tier, over all completed FBS/FCS games (not only Kalshi-listed ones)."""
    d = (pred.filter(pl.col("lead") == 2).join(games.select("game_id", "kickoff", "home_score", "away_score"), on="game_id")
         .join(tier, on="game_id", how="left")
         .with_columns((pl.col("kickoff") >= DEV_END).alias("holdout"),
                       (pl.col("home_score") - pl.col("away_score") - pl.col("mu_margin")).alias("e_m")))
    aggs = [pl.len().alias("games"), (pl.col("e_m") ** 2).mean().sqrt().alias("rmse_margin"),
            pl.col("e_m").abs().mean().alias("mae_margin"), pl.col("e_m").mean().alias("bias_margin")]
    if "mu_total" in d.columns:
        d = d.with_columns((pl.col("home_score") + pl.col("away_score") - pl.col("mu_total")).alias("e_t"))
        aggs += [(pl.col("e_t") ** 2).mean().sqrt().alias("rmse_total"), pl.col("e_t").mean().alias("bias_total")]
    d = d.with_columns(pl.col("kickoff").dt.year().alias("season_year"))
    return d.group_by("holdout", "tier").agg(aggs).sort("holdout", "tier")


def contract_frame(inp: dict[str, Any], run: dict[str, Any], dists: dict[str, Dist], lead: int,
                   tier: pl.DataFrame) -> pl.DataFrame:
    """Legs at ``lead`` hours before kickoff with model ``p``, the executable quote at ``t`` and
    ``y``; ``event`` = ESPN game id (so every ladder of one game is one cluster)."""
    from albert.research.fundamental import attach_quotes

    lg = kalshi_frame(inp["legs"], lead)
    pr = run["pred"].filter(pl.col("lead") == lead).drop("lead")
    df = leg_probabilities(lg, pr, dists).filter(pl.col("p").is_not_null())
    df = attach_quotes(df, inp["candles"]).with_columns(pl.col("game_id").alias("event"))
    df = df.join(tier, on="game_id", how="left")
    return df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("t") >= DEV_END).alias("holdout"))


# ---- ladder optimisation ------------------------------------------------------------------------
def outcome_pmf(loc: float, scale: float, dist: Dist, grid: np.ndarray) -> np.ndarray:
    """Probability of each integer score outcome in ``grid`` under loc + scale * Z, Z ~ dist."""
    upper = dist.sf((grid - 0.5 - loc) / scale)
    lower = dist.sf((grid + 0.5 - loc) / scale)
    p = np.clip(upper - lower, 0, None)
    return p / p.sum()


def ladder_options(legs: pl.DataFrame, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Every executable bet on one ladder: buy YES at the ask or NO at 1 - bid (taker fee
    included). Returns net return per $1 staked for each option and outcome (options x grid),
    the cost per contract, and labels. Leg columns: x_strike, upper (YES iff X > x_strike, else
    X < x_strike), bid, ask, ticker."""
    from albert.research.fundamental import taker_fee

    R, cost, labels = [], [], []
    for r in legs.iter_rows(named=True):
        yes_wins = grid > r["x_strike"] if r["upper"] else grid < r["x_strike"]
        for side, price, wins in (("yes", r["ask"], yes_wins), ("no", 1 - r["bid"], ~yes_wins)):
            if not 0 < price < 1:
                continue
            c = price + taker_fee(price)
            R.append(wins.astype(float) / c - 1.0)
            cost.append(c)
            labels.append(f"{r['ticker']}:{side}")
    return np.array(R).reshape(-1, grid.size), np.array(cost), labels


def kelly(R: np.ndarray, pmf: np.ndarray, cap: float) -> np.ndarray:
    """Stakes (fractions of bankroll, >= 0, summing to at most ``cap``) maximising
    E log(1 + f . R(X)). Only options with positive expected return are considered."""
    from scipy import optimize

    f = np.zeros(R.shape[0])
    ev = R @ pmf
    use = np.flatnonzero(ev > 0)
    if not use.size:
        return f
    Ru = R[use]

    def neg(v: np.ndarray) -> float:
        w = 1 + v @ Ru
        return float(-(pmf * np.log(np.maximum(w, 1e-12))).sum())

    def grad(v: np.ndarray) -> np.ndarray:
        w = np.maximum(1 + v @ Ru, 1e-12)
        return -(Ru * (pmf / w)).sum(axis=1)

    x0 = np.full(use.size, min(cap / use.size, 0.01))
    r = optimize.minimize(neg, x0, jac=grad, method="SLSQP", bounds=[(0, cap)] * use.size,
                          constraints=[{"type": "ineq", "fun": lambda v: cap - v.sum()}])
    f[use] = np.clip(r.x, 0, cap)
    return f


def single_kelly(R: np.ndarray, pmf: np.ndarray, cap: float) -> np.ndarray:
    """The best single option (highest optimal log growth on its own) at its own Kelly stake."""
    f = np.zeros(R.shape[0])
    best = (0.0, None, 0.0)
    for i in np.flatnonzero(R @ pmf > 0):
        fi = kelly(R[i:i + 1], pmf, cap)[0]
        g = float((pmf * np.log1p(fi * R[i])).sum())
        if g > best[0]:
            best = (g, i, fi)
    if best[1] is not None:
        f[best[1]] = best[2]
    return f


def line_pred(odds: pl.DataFrame, games: pl.DataFrame, which: str, leads: Iterable[int]) -> pl.DataFrame:
    """A sportsbook line as a score model, in the same columns as :func:`run_kf` output:
    margin mean = -home spread, total mean = the total, team means = (total -/+ spread) / 2.
    SDs are fitted on dev games (kickoff before DEV_END): RMS error of each mean."""
    d = (odds.select("game_id", pl.col(f"spread_{which}").alias("sp"), pl.col(f"total_{which}").alias("tot"))
         .drop_nulls().join(games.select("game_id", "kickoff", "home_score", "away_score"), on="game_id"))
    d = d.with_columns((-pl.col("sp")).alias("mu_margin"), pl.col("tot").alias("mu_total"),
                       ((pl.col("tot") - pl.col("sp")) / 2).alias("mu_home"), ((pl.col("tot") + pl.col("sp")) / 2).alias("mu_away"))
    dev = d.filter((pl.col("kickoff") < DEV_END) & pl.col("home_score").is_not_null())

    def rms(e: pl.Expr) -> float:
        return float(dev.select((e ** 2).mean().sqrt()).item())

    sd = {"sd_margin": rms(pl.col("home_score") - pl.col("away_score") - pl.col("mu_margin")),
          "sd_total": rms(pl.col("home_score") + pl.col("away_score") - pl.col("mu_total")),
          "sd_home": rms(pl.col("home_score") - pl.col("mu_home")), "sd_away": rms(pl.col("away_score") - pl.col("mu_away"))}
    d = d.with_columns(*[pl.lit(v).alias(k) for k, v in sd.items()])
    cols = ["game_id", "mu_home", "mu_away", "mu_margin", "mu_total", *sd]
    return pl.concat([d.select(cols).with_columns((d["kickoff"] - timedelta(hours=h)).alias("t"), pl.lit(h).alias("lead"))
                      for h in leads])


KINDS = ("ml", "spread", "total", "team_total")


def residual_laws(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {k: fit_dist(v) for k, v in run["resid"].items()}


def select_dev(s, inp: dict[str, Any], models: dict[str, KFParams]) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Stage 1, dev only (t before DEV_END): rating diagnostics, residual laws (normal vs t),
    and per model x law x lead x kind the harness scores and blend on dev rows only."""
    from albert.research.fundamental import evaluate

    tier = game_tier(inp["games"], inp["tiers"])
    rows, diags, runs = [], [], {}
    for name, prm in models.items():
        run = model_run(inp, prm)
        runs[name] = run
        diags.append(rating_diagnostics(run["pred"], inp["games"], tier).filter(~pl.col("holdout"))
                     .with_columns(pl.lit(name).alias("model")))
        fits = residual_laws(run)
        for law in ("normal", "t"):
            dists = {k: f[law] for k, f in fits.items()}
            for lead in LEADS_H:
                dev = contract_frame(inp, run, dists, lead, tier).filter(~pl.col("holdout"))
                for kind in KINDS:
                    part = dev.filter(pl.col("kind") == kind)
                    if part.height < 50:
                        continue
                    r = evaluate(part, DEV_END)
                    sc, b = r["scores"][0], r.get("blend", {})
                    rows.append({"model": name, "law": law, "lead": lead, "kind": kind, "n": sc["n"], "games": sc["events"],
                                 "brier_model": sc["brier_model"], "brier_mid": sc["brier_mid"],
                                 "ll_model": sc["logloss_model"], "ll_mid": sc["logloss_mid"],
                                 "c": b.get("c_model"), "t_c": b.get("t_c")})
    return pl.DataFrame(rows), pl.concat(diags, how="diagonal_relaxed"), runs


def _summary_row(r: dict[str, Any], **keys: Any) -> dict[str, Any]:
    sc = {x["split"]: x for x in r["scores"]}
    b = r.get("blend", {})
    theta = r.get("theta_chosen_on_dev")
    tk = {(x["theta"], x["split"]): x for x in r["taker"]}
    hold = tk.get((theta, "holdout"), {}) if theta is not None else {}
    devt = tk.get((theta, "dev"), {}) if theta is not None else {}
    return {**keys,
            "n_dev": sc.get("dev", {}).get("n"), "n_hold": sc.get("holdout", {}).get("n"),
            "games_hold": sc.get("holdout", {}).get("events"),
            "brier_model_dev": sc.get("dev", {}).get("brier_model"), "brier_mid_dev": sc.get("dev", {}).get("brier_mid"),
            "brier_model_hold": sc.get("holdout", {}).get("brier_model"), "brier_mid_hold": sc.get("holdout", {}).get("brier_mid"),
            "brier_blend_hold": b.get("brier_blend_holdout"),
            "c_dev": b.get("c_model"), "t_c_dev": b.get("t_c"), "c_hold_refit": b.get("c_model_holdout_refit"),
            "t_c_hold_refit": b.get("t_c_holdout_refit"), "theta": theta,
            "dev_trades": devt.get("trades"), "dev_pnl_c": devt.get("pnl_c"), "dev_t": devt.get("t"),
            "hold_trades": hold.get("trades"), "hold_pnl_c": hold.get("pnl_c"), "hold_t": hold.get("t")}


def evaluate_grid(frames: dict[int, pl.DataFrame], label: str) -> tuple[pl.DataFrame, dict[tuple, dict]]:
    """Harness evaluation per lead x kind (and all kinds pooled)."""
    from albert.research.fundamental import evaluate

    rows, full = [], {}
    for lead, df in frames.items():
        for kind in (*KINDS, "all"):
            part = df if kind == "all" else df.filter(pl.col("kind") == kind)
            if part.filter(~pl.col("holdout")).height < 50:
                continue
            r = evaluate(part, DEV_END)
            full[(label, lead, kind)] = r
            rows.append(_summary_row(r, source=label, lead=lead, kind=kind))
    return pl.DataFrame(rows, infer_schema_length=None), full


def tier_grid(frames: dict[int, pl.DataFrame]) -> pl.DataFrame:
    """Harness evaluation per lead x tier x kind, with θ chosen on that cell's dev rows."""
    from albert.research.fundamental import evaluate

    rows = []
    for lead, df in frames.items():
        for tier in ("FCS", "G5", "P4"):
            for kind in KINDS:
                part = df.filter((pl.col("kind") == kind) & (pl.col("tier") == tier))
                if part.filter(~pl.col("holdout")).height < 50:
                    continue
                rows.append(_summary_row(evaluate(part, DEV_END), tier=tier, lead=lead, kind=kind))
    return pl.DataFrame(rows, infer_schema_length=None)


def taker_segments(df: pl.DataFrame, theta: float, by: str) -> pl.DataFrame:
    """Taker PnL (cents per contract, t clustered by game) per segment at a fixed θ."""
    from albert.research.fundamental import clustered_mean, taker_pnl

    tr = taker_pnl(df, theta)
    if not tr.height:
        return pl.DataFrame()
    rows = []
    for (h, seg), part in tr.group_by("holdout", by):
        mu, t, k = clustered_mean(part["pnl"].to_numpy(), part["event"].to_numpy())
        rows.append({"holdout": h, by: seg, "trades": part.height, "games": k, "pnl_c": round(mu * 100, 2),
                     "t": None if t is None else round(t, 2)})
    return pl.DataFrame(rows).sort("holdout", by)


def margin_space(frames: dict[int, pl.DataFrame], pred: pl.DataFrame, games: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Points-space test. Per game and decision time: the market-implied location of the margin
    (spread + moneyline ladder) and of the total, the model mean, and the result. On dev, fit
    result = a + b * market + c * model (OLS, HC1 t, one row per game); report the refit on the
    holdout and the holdout RMSE of the market location vs the dev-fitted blend."""
    locs = []
    for lead, df in frames.items():
        lf = location_frame(df)
        if lf.height:
            locs.append(lf.with_columns(pl.lit(lead).alias("lead")))
    loc = pl.concat(locs)
    p = pred.select("game_id", "t", "mu_margin", *(c for c in ("mu_total",) if c in pred.columns))
    loc = loc.join(p, on=["game_id", "t"]).join(games.select("game_id", "kickoff", "home_score", "away_score"), on="game_id")
    loc = loc.with_columns(
        pl.when(pl.col("var") == "margin").then(pl.col("home_score") - pl.col("away_score"))
        .otherwise(pl.col("home_score") + pl.col("away_score")).alias("actual"),
        pl.when(pl.col("var") == "margin").then(pl.col("mu_margin")).otherwise(
            pl.col("mu_total") if "mu_total" in loc.columns else pl.lit(None)).alias("mu_model"),
        (pl.col("kickoff") >= DEV_END).alias("holdout"))
    rows = []
    for (var, lead), grp in loc.filter(pl.col("mu_model").is_not_null()).group_by("var", "lead"):
        dev, hold = grp.filter(~pl.col("holdout")), grp.filter(pl.col("holdout"))
        if dev.height < 30:
            continue

        def X(d: pl.DataFrame) -> np.ndarray:
            return np.column_stack([np.ones(d.height), d["loc_mkt"].to_numpy(), d["mu_model"].to_numpy()])

        beta, t = ols_hc(X(dev), dev["actual"].to_numpy())
        row = {"var": var, "lead": lead, "games_dev": dev.height, "games_hold": hold.height,
               "a": round(beta[0], 3), "b_mkt": round(beta[1], 3), "c_model": round(beta[2], 3),
               "t_b": round(t[1], 2), "t_c": round(t[2], 2),
               "rmse_mkt_dev": float(np.sqrt(((dev["actual"] - dev["loc_mkt"]) ** 2).mean())),
               "rmse_model_dev": float(np.sqrt(((dev["actual"] - dev["mu_model"]) ** 2).mean()))}
        if hold.height >= 30:
            bh, th = ols_hc(X(hold), hold["actual"].to_numpy())
            row.update({"c_hold_refit": round(bh[2], 3), "t_c_hold_refit": round(th[2], 2),
                        "rmse_mkt_hold": float(np.sqrt(((hold["actual"] - hold["loc_mkt"]) ** 2).mean())),
                        "rmse_model_hold": float(np.sqrt(((hold["actual"] - hold["mu_model"]) ** 2).mean())),
                        "rmse_blend_hold": float(np.sqrt(((hold["actual"].to_numpy() - X(hold) @ beta) ** 2).mean()))})
        rows.append(row)
    return pl.DataFrame(rows).sort("var", "lead"), loc


def ladder_study(ladders: pl.DataFrame, loc: pl.DataFrame, coefs: dict[tuple[str, int], np.ndarray], *,
                 mode: str = "blend", frac: float = 0.25, cap: float = 0.25) -> pl.DataFrame:
    """Kelly across a whole ladder vs the single best contract, held to settlement.

    For each (game, decision time, variable) the outcome X (margin home - away from the
    spread + both moneyline legs; total from the total ladder) gets a normal law with the
    market-implied scale and location either the dev-fitted blend a + b*market + c*model
    (``mode="blend"``) or the model mean (``mode="model"``). Stakes are ``frac`` x the
    log-optimal fractions (at most ``cap`` of bankroll per ladder), bought at the ask / sold at
    the bid with the taker fee. Returns one row per ladder: stake and realised return (fractions
    of bankroll) for the multi-contract and the single-contract choice."""
    grids = {"margin": np.arange(-120, 121), "total": np.arange(0, 171)}
    normal = Dist(1.0, None)
    lg = ladders.with_columns(
        pl.when(pl.col("kind").is_in(["ml", "spread"])).then(pl.lit("margin"))
        .when(pl.col("kind") == "total").then(pl.lit("total")).otherwise(None).alias("var"),
        pl.when(pl.col("kind") == "total").then(pl.col("strike"))
        .when(pl.col("team_side") == "home").then(pl.col("strike").fill_null(0.0))
        .otherwise(-pl.col("strike").fill_null(0.0)).alias("x_strike"),
        ((pl.col("kind") == "total") | (pl.col("team_side") == "home")).alias("upper"),
    ).filter(pl.col("var").is_not_null())
    info = {(r["game_id"], r["t"], r["var"]): r for r in loc.iter_rows(named=True)}
    rows = []
    for (gid, t, var), grp in lg.group_by("game_id", "t", "var"):
        r = info.get((gid, t, var))
        if r is None or r.get("mu_model") is None:
            continue
        beta = coefs.get((var, r["lead"]))
        if beta is None:
            continue
        center = float(beta @ np.array([1.0, r["loc_mkt"], r["mu_model"]])) if mode == "blend" else float(r["mu_model"])
        grid = grids[var]
        pmf = outcome_pmf(center, r["scale_mkt"], normal, grid)
        R, _, _ = ladder_options(grp, grid)
        if not R.size:
            continue
        j = int(np.clip(round(r["actual"]) - grid[0], 0, grid.size - 1))
        fm = frac * kelly(R, pmf, cap)
        fs = frac * single_kelly(R, pmf, cap)
        rows.append({"game_id": gid, "t": t, "lead": r["lead"], "var": var, "holdout": r["holdout"],
                     "stake_multi": float(fm.sum()), "ret_multi": float(fm @ R[:, j]), "n_multi": int((fm > 1e-6).sum()),
                     "stake_single": float(fs.sum()), "ret_single": float(fs @ R[:, j])})
    return pl.DataFrame(rows, infer_schema_length=None)


def ladder_summary(lad: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (h, var, lead), grp in lad.group_by("holdout", "var", "lead"):
        for m in ("multi", "single"):
            bets = grp.filter(pl.col(f"stake_{m}") > 1e-6)
            ret = bets[f"ret_{m}"].to_numpy()
            se = ret.std(ddof=1) / math.sqrt(ret.size) if ret.size > 1 else float("nan")
            rows.append({"holdout": h, "var": var, "lead": lead, "method": m, "ladders": grp.height, "bets": bets.height,
                         "staked": round(float(bets[f"stake_{m}"].sum()), 4), "return": round(float(ret.sum()), 4),
                         "ret_per_staked": round(float(ret.sum() / bets[f"stake_{m}"].sum()), 4) if bets.height else None,
                         "log_growth": round(float(np.log1p(ret).sum()), 4),
                         "t": round(float(ret.mean() / se), 2) if ret.size > 1 and se > 0 else None})
    return pl.DataFrame(rows).sort("holdout", "var", "lead", "method")


def holdout_study(s, inp: dict[str, Any], run: dict[str, Any], model: str, law: str,
                  odds: pl.DataFrame | None, extra: dict[str, Any]) -> Path:
    """Stage 2: the frozen model and law on dev + October holdout (scored once), with segments,
    the points-space test, the ladder optimisation, sportsbook lines as fair value, and the report."""
    from albert.reports import md_table
    from albert.research.fundamental import attach_quotes, report

    games, tier = inp["games"], game_tier(inp["games"], inp["tiers"])
    fits = residual_laws(run)
    dists = {k: f[law] for k, f in fits.items()}
    frames = {lead: contract_frame(inp, run, dists, lead, tier) for lead in LEADS_H}
    grid, full = evaluate_grid(frames, model)
    sections: list[tuple[str, pl.DataFrame]] = [(("Model vs Kalshi per decision time and contract kind "
                                                 "(dev = Aug-Sep, holdout = Oct; taker at the dev-chosen θ)"), grid)]
    # Segments (per kind, at each lead): Brier by tier and price band, taker at the dev θ.
    for lead, df in frames.items():
        df = df.with_columns(price_band(pl.col("mid")).alias("band"))
        for kind in KINDS:
            part = df.filter(pl.col("kind") == kind)
            r = full.get((model, lead, kind))
            if r is None or not part.height:
                continue
            theta = r.get("theta_chosen_on_dev")
            for by in ("tier", "band"):
                seg = segment_scores(part, by)
                if theta is not None:
                    tk = taker_segments(part, theta, by)
                    if tk.height:
                        seg = seg.join(tk.select("holdout", by, "trades", "pnl_c", "t"), on=["holdout", by], how="left")
                sections.append((f"{kind}, {lead} h before kickoff: by {by} (taker θ = {theta})", seg))
    sections.append(("Per tier (θ chosen on each cell's dev rows)", tier_grid(frames)))
    # Points space.
    ms, loc = margin_space(frames, run["pred"], games)
    sections.append(("Points-space test: result = a + b·market-implied location + c·model mean (OLS, HC1 t)", ms))
    coefs = {(r["var"], r["lead"]): np.array([r["a"], r["b_mkt"], r["c_model"]]) for r in ms.iter_rows(named=True)}
    # Ladder optimisation (both moneyline legs included; executable quotes at t).
    lad_frames = []
    for lead in LEADS_H:
        lg = inp["legs"].with_columns((pl.col("kickoff") - pl.duration(hours=lead)).alias("t"))
        lad_frames.append(attach_quotes(lg, inp["candles"]))
    ladders = pl.concat(lad_frames)
    for mode in ("blend", "model"):
        lad = ladder_study(ladders, loc, coefs, mode=mode)
        if lad.height:
            sections.append((f"Ladder Kelly (¼ Kelly, ≤25% per ladder), location = {mode}: multi-contract vs single best",
                             ladder_summary(lad)))
    # Sportsbook lines (ESPN / DraftKings) as fair value.
    if odds is not None and odds.height:
        normal = {"margin": Dist(1.0, None), "total": Dist(1.0, None), "team": Dist(1.0, None)}
        for which in ("open", "close"):
            lp = line_pred(odds, games, which, LEADS_H)
            fr = {}
            for lead in LEADS_H:
                lg = kalshi_frame(inp["legs"], lead)
                df = leg_probabilities(lg, lp.filter(pl.col("lead") == lead).drop("lead"), normal).filter(pl.col("p").is_not_null())
                df = attach_quotes(df, inp["candles"]).with_columns(pl.col("game_id").alias("event"))
                fr[lead] = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("t") >= DEV_END).alias("holdout"))
            g2, _ = evaluate_grid(fr, f"dk_{which}")
            note = " (LOOK-AHEAD: the closing line is set at kickoff, after t)" if which == "close" else ""
            sections.append((f"DraftKings {which} line (via ESPN) as fair value{note}", g2))
    # Rating diagnostics and residual laws.
    sections.append(("Rating accuracy at kickoff - 2 h, all completed FBS/FCS games", rating_diagnostics(run["pred"], games, tier)))
    laws = pl.DataFrame([{"variable": k, "n": f["n"], "ll_normal": f["ll_normal"], "ll_t": f["ll_t"], "t_df": f["t"].df,
                          "t_scale": f["t"].scale, "normal_scale": f["normal"].scale, "excess_kurtosis": f["kurtosis"]}
                         for k, f in fits.items()])
    sections.append(("Residual law on dev games (standardised by the model's predictive SD)", laws))
    pooled = full.get((model, 2, "all"))
    path = report(s, "ncaaf", pooled, {**extra, "model": model, "law": law,
                                        "primary_frame": "all contract kinds pooled, 2 h before kickoff"})
    text = path.read_text()
    for title, frame in sections:
        text += f"\n## {title}\n\n{md_table(frame, max_rows=60)}\n"
    path.write_text(text)
    grid.write_parquet(path.parent / "grid.parquet")
    for lead, df in frames.items():
        df.write_parquet(path.parent / f"contracts_{lead}h.parquet")
    return path


# Hyperparameters fitted on dev only (games of the 2024 and 2025 seasons and Aug-Sep 2026, each
# predicted 2 h before kickoff by the causal pass; 2023 is burn-in) by Nelder-Mead on the
# predictive Gaussian log likelihood of the margin (one-strength models) or of margin + total
# (offense/defense models). Frozen 2026-11-05 before any October Kalshi price was scored.
MODELS: dict[str, KFParams] = {
    # (a) ridge / Massey: no in-season drift, so each prediction is the ridge solution of all
    # earlier games towards the regressed preseason prior.
    "ridge": KFParams(split=False, sigma=13.576, rho_off=0.791, q_off=50.22, p0=537.8, q_day=0.0),
    # (a') the same with a random-walk strength (state-space margin model).
    "rw": KFParams(split=False, sigma=13.406, rho_off=0.778, q_off=40.80, p0=543.4, q_day=0.1292),
    # (b) offense/defense state-space model (prices totals and team totals as well).
    "od": KFParams(split=True, sigma=10.043, rho_off=0.762, q_off=33.76, p0=318.3, q_day=0.01513, rho_e=-0.0016),
    # (b') the same with innovations clipped at 2.5 SD (blowouts move ratings less).
    "od_robust": KFParams(split=True, clip=2.5, sigma=9.978, rho_off=0.6848, q_off=30.79, p0=398.3, q_day=0.0920,
                          rho_e=0.0011),
}


# Chosen on dev (2026-11-05, select_dev): lowest dev log loss on Kalshi legs per variable.
# Margin contracts (moneyline, spread) from the ridge model; total and team-total contracts from
# the robust offense/defense model; Student-t residual law (dev residual log likelihood and dev
# Kalshi log loss both marginally better than normal; df ~ 50-100, i.e. nearly normal).
CHOSEN = {"margin": "ridge", "totals": "od_robust", "law": "t"}


def combine_runs(margin_run: dict[str, Any], totals_run: dict[str, Any]) -> dict[str, Any]:
    """Margin columns from one model, total / team-score columns from another."""
    keep = ["game_id", "t", "lead", "mu_margin", "sd_margin", "var_state_margin"]
    tot = totals_run["pred"].select("game_id", "t", "mu_total", "sd_total", "mu_home", "mu_away", "sd_home", "sd_away")
    pred = margin_run["pred"].select(keep).join(tot, on=["game_id", "t"], how="left")
    resid = {"margin": margin_run["resid"]["margin"], "total": totals_run["resid"]["total"],
             "team": totals_run["resid"]["team"]}
    return {"pred": pred, "resid": resid}
