"""Behavioural mispricing in Kalshi sports: who trades at bad prices, and does it pay to stand opposite?

Question (user, 2026-11-18). Reason from psychology and market structure about *why* people or
algorithms would buy or sell sports contracts at inefficient prices, then test the patterns those
mechanisms predict. The hypothesis list (H1-H19, with who / other side / predicted sign / test /
already-covered) and the pre-registered rules are in ``data/research/behav_sports/hypotheses.md``.

Data (all public; Kalshi at <= 1 request/s through the cached raw store, ESPN site API through
its own zstd cache):

* **Games** (``espn``): ESPN scoreboards per league-month (NCAAF per season-week, FBS + FCS) for
  2025-08 .. 2026-11: scheduled start, home / away, final score, broadcasters (national TV),
  neutral site. ESPN is matched to Kalshi by ET date and team location / abbreviation.
* **Markets** (``markets``): Kalshi game-winner series (NFL, NCAAF, NBA, NHL, MLB, WNBA and eight
  soccer leagues), settled, closes 2025-08-01 .. 2026-11-17: the historical listing (newest
  first, by series) plus the live listing from 2026-09-15.
* **Candles** (``candles``): hourly OHLC (bid / ask / trade) over the 72 h before each market's
  close, for one leg of every two-way game (the alphabetically first ticker; the other team is its
  NO side) and the home and away legs of soccer games. Historical-era markets (settled before the
  2026-09-18 cutoff) need one request per ticker, live-era ones go 100 per batch request.

* **Cached, no new requests**: ``pre_event`` player-prop legs and hourly candles (H5 attention),
  ``vol_greeks`` football spread ladders (H15 key numbers). **Tennis** (H11): Kalshi
  ``tennis_tournament_singles`` milestones for the start, ATP / WTA / Challenger matches with exactly
  one player on the frozen US list.

Panel (``side_panel``): one row per (game, team). Quotes are the last two-sided hourly candle ending
at or before start - L (L = 24, 6, 2, 1, 0 h; carried <= 24 h because Kalshi omits unchanged hours);
the team's price is its own leg's YES or the NO side of the opponent's leg; the payoff is that
contract's. History (H3) comes from the complete schedule (results from the listing, margins from
ESPN), so a team's previous game is always its real previous game; only that game's price needs a
candle. Tests use quotes with spread <= 5c and mid in [3c, 97c]; t-stats are clustered by game and
by ET date (the larger SE). Execution: taker at the ask with Kalshi's fee rounded up per order, or a
resting bid filled only when a later candle traded strictly through it (maker fee on maker-fee
series); $100 capped book, $5 per position, $10 per game; ``vol_metrics`` on daily marks.

Splits: dev = scheduled start before 2026-10-01, holdout = 2026-10-01 .. 2026-11-17 (opened only
after the pre-registration in ``hypotheses.md``).

    python -m albert.research.behav_sports espn|markets|candles|tennis|panel|explore|holdout|all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/behav_sports")
ESPN_CACHE = Path("data/cache/behav_sports/espn")
ESPN_ROOT = "https://site.api.espn.com/apis/site/v2/sports"
ET = "America/New_York"
LIST_FROM = datetime(2025, 8, 1, tzinfo=UTC)          # earliest close listed
LIVE_FROM = datetime(2026, 9, 15, tzinfo=UTC)         # live listing overlaps the 09-18 cutoff
DATA_TO = datetime(2026, 11, 18, 5, tzinfo=UTC)       # listing end (holdout = starts through 11-17 ET)
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
WINDOW_H = 72                                          # candle window before close
KALSHI_INTERVAL = 1.0                                  # seconds between Kalshi requests (shared IP)

# league -> (Kalshi series, ESPN path, kind)
LEAGUES: dict[str, tuple[str, str, str]] = {
    "NFL": ("KXNFLGAME", "football/nfl", "us"),
    "NBA": ("KXNBAGAME", "basketball/nba", "us"),
    "NHL": ("KXNHLGAME", "hockey/nhl", "us"),
    "MLB": ("KXMLBGAME", "baseball/mlb", "us"),
    "WNBA": ("KXWNBAGAME", "basketball/wnba", "us"),
    "NCAAF": ("KXNCAAFGAME", "football/college-football", "us"),
    "EPL": ("KXEPLGAME", "soccer/eng.1", "soccer"),
    "UCL": ("KXUCLGAME", "soccer/uefa.champions", "soccer"),
    "MLS": ("KXMLSGAME", "soccer/usa.1", "soccer"),
    "LALIGA": ("KXLALIGAGAME", "soccer/esp.1", "soccer"),
    "SERIEA": ("KXSERIEAGAME", "soccer/ita.1", "soccer"),
    "BUNDESLIGA": ("KXBUNDESLIGAGAME", "soccer/ger.1", "soccer"),
    "LIGUE1": ("KXLIGUE1GAME", "soccer/fra.1", "soccer"),
    "LIGAMX": ("KXLIGAMXGAME", "soccer/mex.1", "soccer"),
}
SERIES_LEAGUE = {v[0]: k for k, v in LEAGUES.items()}
# historical candle fetch order (live-era candles are cheap and always fetched first)
HIST_PRIORITY = ["NFL", "NBA", "NHL", "WNBA", "NCAAF", "MLB", "EPL", "UCL", "LALIGA", "MLS"]
HIST_FROM = {"MLB": datetime(2026, 3, 1, tzinfo=UTC)}   # MLB history: the 2026 season only (request budget)

# national broadcasters (ESPN ``broadcasts[].names``); regional sports networks are not national
NATIONAL_TV = re.compile(
    r"^(ESPN|ESPN2|ABC|CBS|NBC|FOX|FS1|TNT|TBS|truTV|Prime Video|NFL Net|NFL Network|NBA TV|NHL Network|"
    r"MLB Network|Peacock|Apple TV\+?|Netflix|USA Net|NBCSN|ION|CBSSN|ESPNU|BTN|SECN|ACCN|Max|"
    r"Paramount\+|ESPN\+|TruTV|Telemundo|Univision|UniMas|TUDN|CBS Sports Network)$", re.IGNORECASE)
BIG_NATIONAL = re.compile(r"^(ESPN|ABC|CBS|NBC|FOX|TNT|Prime Video|Peacock|Netflix)$", re.IGNORECASE)


# ---- ESPN -----------------------------------------------------------------------------------------
_last_espn = [0.0]


def espn_get(path: str, params: dict[str, str], *, refresh_after: datetime | None = None,
             min_interval: float = 0.6, cache_dir: Path = ESPN_CACHE) -> dict[str, Any]:
    """GET an ESPN site-API JSON once (zstd cache keyed on path + params). ``refresh_after``: a
    cached copy fetched before that time is ignored (for months that were still in progress) and
    the new copy is written next to it (append-only, the newest file wins)."""
    import httpx

    from albert.http import read_raw, write_raw

    key = hashlib.sha256((path + "?" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
    files = sorted(cache_dir.glob(f"{slug}-{key}*.json.zst"))
    if files:
        rec = read_raw(files[-1])
        if refresh_after is None or datetime.fromisoformat(rec["fetched_at"]) >= refresh_after:
            return rec["response"]
    wait = min_interval - (time.monotonic() - _last_espn[0])
    if wait > 0:
        time.sleep(wait)
    r = None
    for attempt in range(5):
        _last_espn[0] = time.monotonic()
        r = httpx.get(f"{ESPN_ROOT}/{path}", params=params, timeout=30,
                      headers={"User-Agent": "albert-research (public data, cached)"})
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    assert r is not None
    r.raise_for_status()
    now = datetime.now(UTC)
    name = f"{slug}-{key}.json.zst" if not files else f"{slug}-{key}-{now:%Y%m%dT%H%M%S}.json.zst"
    write_raw(cache_dir / name, {"url": f"{ESPN_ROOT}/{path}", "params": params,
                                 "fetched_at": now.isoformat(), "response": r.json()})
    return r.json()


def _espn_ts(x: str | None) -> datetime | None:
    if not x:
        return None
    x = x.replace("Z", "+00:00")
    if re.match(r".*T\d\d:\d\d\+", x):
        x = x.replace("+", ":00+", 1)
    return datetime.fromisoformat(x).astimezone(UTC)


def espn_game_rows(payload: dict[str, Any], league: str) -> list[dict[str, Any]]:
    rows = []
    for e in payload.get("events") or []:
        for c in e.get("competitions") or []:
            sides = {t.get("homeAway"): t for t in c.get("competitors") or []}
            if set(sides) != {"home", "away"}:
                continue
            names = [n for b in c.get("broadcasts") or [] for n in (b.get("names") or [])]
            row: dict[str, Any] = {
                "league": league, "espn_id": str(c.get("id") or e.get("id")), "start": _espn_ts(c.get("date") or e.get("date")),
                "status": ((c.get("status") or {}).get("type") or {}).get("name"),
                "neutral": bool(c.get("neutralSite")), "broadcasts": "|".join(names),
                "national_tv": any(NATIONAL_TV.match(n.strip()) for n in names),
                "big_tv": any(BIG_NATIONAL.match(n.strip()) for n in names),
            }
            for side, t in sides.items():
                team = t.get("team") or {}
                sc = t.get("score")
                row.update({f"{side}_abbr": team.get("abbreviation"), f"{side}_loc": team.get("location"),
                            f"{side}_name": team.get("displayName"), f"{side}_short": team.get("shortDisplayName"),
                            f"{side}_score": float(sc) if sc not in (None, "") else None,
                            f"{side}_winner": t.get("winner")})
            rows.append(row)
    return rows


ESPN_SCHEMA = {
    "league": pl.String, "espn_id": pl.String, "start": pl.Datetime("us", "UTC"), "status": pl.String,
    "neutral": pl.Boolean, "broadcasts": pl.String, "national_tv": pl.Boolean, "big_tv": pl.Boolean,
    **{f"{s}_{k}": t for s in ("home", "away") for k, t in (
        ("abbr", pl.String), ("loc", pl.String), ("name", pl.String), ("short", pl.String),
        ("score", pl.Float64), ("winner", pl.Boolean))},
}


def _months(d0: date, d1: date) -> list[str]:
    out, y, m = [], d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append(f"{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_espn(d0: date = date(2025, 8, 1), d1: date = date(2026, 11, 18)) -> pl.DataFrame:
    """Every game of the configured leagues from ESPN (league-month pages; NCAAF by season-week)."""
    rows: list[dict[str, Any]] = []
    recent = datetime(2026, 11, 18, 6, tzinfo=UTC)     # months touching the holdout end: refetch once
    for league, (_series, path, _kind) in LEAGUES.items():
        if league == "NCAAF":
            for season in (2025, 2026):
                for stype, week in [(2, w) for w in range(1, 17)] + [(3, 1)]:
                    for group in ("80", "81"):
                        p = espn_get(f"{path}/scoreboard", {"dates": str(season), "seasontype": str(stype),
                                                            "week": str(week), "groups": group, "limit": "500"},
                                     refresh_after=recent if season == 2026 else None)
                        rows.extend(espn_game_rows(p, league))
            continue
        for ym in _months(d0, d1):
            p = espn_get(f"{path}/scoreboard", {"dates": ym, "limit": "500"},
                         refresh_after=recent if ym >= "202609" else None)
            rows.extend(espn_game_rows(p, league))
        print(f"espn {league}: {len(rows)} rows so far", flush=True)
    g = pl.DataFrame(rows, schema=ESPN_SCHEMA).unique(["league", "espn_id"], keep="last")
    g = g.filter(pl.col("start").is_not_null()).sort("league", "start")
    OUT.mkdir(parents=True, exist_ok=True)
    g.write_parquet(OUT / "espn_games.parquet")
    return g


# ---- Kalshi listings ------------------------------------------------------------------------------
def _api():
    from albert.config import settings
    from albert.research.pm_hazard import _api as mk

    return mk(settings(), min_interval=KALSHI_INTERVAL)


MARKET_COLS = ["ticker", "event_ticker", "series_ticker", "title", "yes_sub_title", "open_time", "close_time",
               "settlement_ts", "result", "volume", "fee_type", "fee_multiplier", "source"]


def _market_rows(payload: dict[str, Any], source: str, fees: dict[str, tuple[str, float]]) -> list[dict[str, Any]]:
    from albert.venues.kalshi import _field, num, parse_ts, series_of

    out = []
    for m in payload.get("markets") or []:
        if m.get("mve_collection_ticker") or m.get("mve_selected_legs"):
            continue
        series = series_of(m.get("event_ticker") or m["ticker"])
        ft, fm = fees.get(series, ("quadratic", 1.0))
        out.append({"ticker": m["ticker"], "event_ticker": m.get("event_ticker"), "series_ticker": series,
                    "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title"),
                    "open_time": parse_ts(m.get("open_time")), "close_time": parse_ts(m.get("close_time")),
                    "settlement_ts": parse_ts(m.get("settlement_ts")), "result": m.get("result") or None,
                    "volume": num(_field(m, "volume")), "fee_type": ft, "fee_multiplier": fm, "source": source})
    return out


def fetch_markets() -> pl.DataFrame:
    """Settled game-winner markets of every configured series closing LIST_FROM .. DATA_TO."""
    api = _api()
    fees = {r[0]: (r[1], r[2]) for r in pl.read_parquet("data/tables/kalshi_series.parquet")
            .select("series_ticker", "fee_type", "fee_multiplier").iter_rows()}
    rows: list[dict[str, Any]] = []
    for league, (series, _p, _k) in LEAGUES.items():
        n0 = len(rows)
        cursor = None
        while True:      # historical: newest first; stop once a page reaches LIST_FROM
            page = api.client.get("/historical/markets", {"series_ticker": series, "limit": 1000, "cursor": cursor})
            got = _market_rows(page.payload, "hist", fees)
            rows.extend(got)
            cursor = page.payload.get("cursor")
            closes = [r["close_time"] for r in got if r["close_time"] is not None]
            if not cursor or not got or (closes and min(closes) < LIST_FROM):
                break
        for page in api.series_markets(series, "settled", min_close=LIVE_FROM, max_close=DATA_TO, cache=True):
            rows.extend(_market_rows(page.payload, "live", fees))
        print(f"markets {league}: {len(rows) - n0}", flush=True)
    m = pl.DataFrame(rows, schema={c: t for c, t in zip(MARKET_COLS, [
        pl.String, pl.String, pl.String, pl.String, pl.String, pl.Datetime("us", "UTC"), pl.Datetime("us", "UTC"),
        pl.Datetime("us", "UTC"), pl.String, pl.Float64, pl.String, pl.Float64, pl.String], strict=True)})
    m = (m.filter(pl.col("close_time").is_between(LIST_FROM, DATA_TO) & pl.col("result").is_in(["yes", "no"]))
         .sort("source").unique("ticker", keep="last")
         .with_columns(pl.col("series_ticker").replace_strict(SERIES_LEAGUE, default=None).alias("league"),
                       pl.col("ticker").str.split("-").list.last().alias("code"))
         .sort("league", "close_time", "ticker"))
    m.write_parquet(OUT / "markets.parquet")
    return m


# ---- candles ---------------------------------------------------------------------------------------
def candle_targets(m: pl.DataFrame) -> pl.DataFrame:
    """Legs to fetch: one per two-way game (alphabetically first ticker), home + away for soccer (the
    two non-tie legs). Window: the WINDOW_H hours before close (never before open)."""
    kind = pl.col("league").replace_strict({k: v[2] for k, v in LEAGUES.items()}, default="us")
    m = m.with_columns(kind.alias("kind"))
    two = (m.filter(pl.col("kind") == "us").sort("ticker").group_by("event_ticker", maintain_order=True).first())
    soc = m.filter((pl.col("kind") == "soccer") & (pl.col("code") != "TIE"))
    t = pl.concat([two.select(m.columns), soc]).with_columns(
        pl.max_horizontal("open_time", pl.col("close_time") - timedelta(hours=WINDOW_H)).alias("lo"),
        pl.col("close_time").alias("hi"))
    cutoff = datetime(2026, 9, 18, tzinfo=UTC)
    return t.with_columns((pl.col("settlement_ts") < cutoff).alias("historical")).sort("league", "close_time")


def _done(parts: Path) -> set[str]:
    files = sorted(parts.glob("done-*.parquet"))
    if not files:
        return set()
    return set(pl.concat([pl.read_parquet(f) for f in files])["ticker"].to_list())


def fetch_candles(max_hist: int | None = None, leagues: list[str] | None = None) -> None:
    """Hourly OHLC for every target leg; resumable (parts + done lists under OUT/candles)."""
    from albert.http import HttpError
    from albert.research.xcat import OHLC_SCHEMA, batch_rows, ohlc_rows

    api = _api()
    parts = OUT / "candles"
    parts.mkdir(parents=True, exist_ok=True)
    t = candle_targets(pl.read_parquet(OUT / "markets.parquet"))
    if leagues:
        t = t.filter(pl.col("league").is_in(leagues))
    done = _done(parts)
    todo = t.filter(~pl.col("ticker").is_in(list(done)))
    buf: list[dict[str, Any]] = []
    fin: list[str] = []

    def flush() -> None:
        if not fin:
            return
        k = len(list(parts.glob("done-*.parquet")))
        pl.DataFrame(buf, schema=OHLC_SCHEMA).write_parquet(parts / f"part-{k:05d}.parquet")
        pl.DataFrame({"ticker": fin}).write_parquet(parts / f"done-{k:05d}.parquet")
        buf.clear()
        fin.clear()

    live = todo.filter(~pl.col("historical")).sort("hi")
    rows = live.select("ticker", "lo", "hi").rows()
    i = 0
    while i < len(rows):     # batches: <= 100 tickers whose closes span <= 12 h
        j = i
        while j < len(rows) and j - i < 100 and rows[j][2] - rows[i][2] <= timedelta(hours=12):
            j += 1
        chunk = rows[i:j]
        lo = min(r[1] for r in chunk)
        hi = max(r[2] for r in chunk)
        page = api.candles_batch([r[0] for r in chunk], lo, hi, 60)
        got = batch_rows(page.payload, 60)
        for tk, _lo, _hi in chunk:
            buf.extend(r for r in got.get(tk, []) if r["end_ts"] is not None and _lo <= r["end_ts"] <= _hi + timedelta(hours=1))
            fin.append(tk)
        i = j
        if len(fin) >= 500:
            flush()
    flush()
    print(f"live candles done: {len(rows)} legs", flush=True)

    hist = todo.filter(pl.col("historical") & pl.col("league").is_in(HIST_PRIORITY))
    for lg, d0 in HIST_FROM.items():
        hist = hist.filter((pl.col("league") != lg) | (pl.col("hi") >= d0))
    order = {lg: k for k, lg in enumerate(HIST_PRIORITY)}
    # MLB: a seeded half (event hash) goes first, the other half after the soccer leagues
    half = (pl.col("league") == "MLB") & (pl.col("event_ticker").hash(seed=11) % 2 == 1)
    hist = hist.with_columns(pl.when(half).then(pl.lit(len(HIST_PRIORITY) + 1)).otherwise(
        pl.col("league").replace_strict(order, default=99)).alias("prio")).sort("prio", "hi", descending=[False, True])
    if max_hist is not None:
        hist = hist.head(max_hist)
    n = 0
    t0 = time.monotonic()
    for tk, lo, hi in hist.select("ticker", "lo", "hi").rows():
        try:
            page = api.candles(tk, lo, hi + timedelta(hours=1), 60, historical=True)
            buf.extend(ohlc_rows(page.payload, tk, 60))
        except HttpError as exc:     # a missing market is skipped, never retried in a loop
            print(f"candles {tk}: {exc}", flush=True)
        fin.append(tk)
        n += 1
        if len(fin) >= 200:
            flush()
            print(f"hist candles {n}/{hist.height} ({(time.monotonic() - t0) / n:.2f}s each)", flush=True)
    flush()
    print(f"hist candles done: {n}", flush=True)


def load_candles() -> pl.LazyFrame:
    return pl.scan_parquet(str(OUT / "candles" / "part-*.parquet"))


# ---- frozen lists (hypotheses.md; fixed before any test) -------------------------------------------
POPULAR: dict[str, set[str]] = {    # ESPN displayName (NCAAF: ESPN location)
    "NFL": {"Dallas Cowboys", "Kansas City Chiefs", "Green Bay Packers", "Pittsburgh Steelers", "San Francisco 49ers",
            "Philadelphia Eagles", "New England Patriots", "Chicago Bears"},
    "NBA": {"Los Angeles Lakers", "Golden State Warriors", "Boston Celtics", "New York Knicks", "Chicago Bulls", "Miami Heat"},
    "MLB": {"New York Yankees", "Los Angeles Dodgers", "Boston Red Sox", "Chicago Cubs", "New York Mets", "Atlanta Braves"},
    "NHL": {"Toronto Maple Leafs", "Montreal Canadiens", "New York Rangers", "Chicago Blackhawks", "Boston Bruins",
            "Pittsburgh Penguins"},
    "WNBA": {"Indiana Fever", "New York Liberty", "Las Vegas Aces"},
    "NCAAF": {"Alabama", "Ohio State", "Michigan", "Notre Dame", "Texas", "Georgia", "USC", "LSU", "Penn State",
              "Oklahoma", "Florida", "Tennessee", "Clemson", "Nebraska", "Texas A&M", "Oregon"},
    "SOCCER": {"Manchester United", "Liverpool", "Arsenal", "Chelsea", "Manchester City", "Tottenham Hotspur",
               "Real Madrid", "Barcelona", "Bayern Munich", "Juventus", "Paris Saint-Germain", "Inter Miami CF"},
}
BLOWOUT = {"NFL": 17, "NCAAF": 21, "NBA": 15, "WNBA": 15, "NHL": 3, "MLB": 5}   # soccer: 3
LEADS_H = (24, 6, 2, 1, 0)
MAX_QUOTE_AGE_H = 24          # Kalshi omits unchanged hours; a quote is carried this long
_EV_RX = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})(\d{4})?([A-Z0-9]+?)(G\d)?$")
_MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_event(event_ticker: str, codes: list[str]) -> dict[str, Any] | None:
    """ET date and the two team codes in ticker order (US leagues: away first; soccer: home first)."""
    body = event_ticker.split("-", 1)[1] if "-" in event_ticker else ""
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})", body)
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        d = date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))
    except ValueError:
        return None
    rest = body[7:]
    teams = [c for c in codes if c != "TIE"]
    if len(teams) != 2:
        return None
    for s in (rest, rest[4:] if rest[:4].isdigit() else None):
        if s is None:
            continue
        for a, b in ((teams[0], teams[1]), (teams[1], teams[0])):
            if s == a + b or s.startswith(a + b):
                return {"et_date": d, "first": a, "second": b}
    return None


def _key(s: str | None) -> str:
    s = (s or "").lower().replace("&", "and").replace("st.", "state").replace("'", "")
    s = re.sub(r"\b(fc|cf|afc|sc|the|club|de|ac)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def team_score(code: str, sub: str | None, e: dict[str, Any], side: str) -> int:
    """Similarity of a Kalshi team (code, yes_sub_title) to an ESPN competitor."""
    a, loc, name, short = (e.get(f"{side}_{k}") for k in ("abbr", "loc", "name", "short"))
    s = 0
    if a and code.upper() == a.upper():
        s += 3
    ks = _key(sub)
    if ks and ks in {_key(loc), _key(name), _key(short)}:
        s += 3
    m = re.match(r"^(.*) ([A-Z]{1,2})$", sub or "")
    if m and _key(m.group(1)) == _key(loc):
        nick = (name or "")[len(loc or ""):].strip()
        s += 3 if nick[:len(m.group(2))].upper() == m.group(2) else 1
    if ks:
        tk, te = set(ks.split()), set(_key(name).split()) | set(_key(loc).split())
        if tk and len(tk & te) / len(tk) >= 0.5:
            s += 1
    return s


def match_espn(games: pl.DataFrame, espn: pl.DataFrame) -> pl.DataFrame:
    """One ESPN game per Kalshi event: same league, ET date within a day, best team-name score."""
    e = espn.with_columns(pl.col("start").dt.convert_time_zone(ET).dt.date().alias("et_date"))
    by: dict[tuple[str, date], list[dict[str, Any]]] = {}
    for r in e.iter_rows(named=True):
        by.setdefault((r["league"], r["et_date"]), []).append(r)
    out = []
    for g in games.iter_rows(named=True):
        best, best_s = None, 0
        for dd in (0, -1, 1):
            for r in by.get((g["league"], g["et_date"] + timedelta(days=dd)), []):
                for f, s in (("home", "away"), ("away", "home")):
                    s1 = team_score(g["first"], g["first_sub"], r, f)
                    s2 = team_score(g["second"], g["second_sub"], r, s)
                    tot = s1 + s2 - abs(dd)
                    if min(s1, s2) >= 2 and tot > best_s:
                        best, best_s = (r, f), tot
        if best is None:
            continue
        r, f = best
        other = "away" if f == "home" else "home"
        out.append({"event_ticker": g["event_ticker"], "espn_id": r["espn_id"], "match_score": best_s,
                    "start": r["start"], "first_home": f == "home", "neutral": r["neutral"],
                    "national_tv": r["national_tv"], "big_tv": r["big_tv"], "broadcasts": r["broadcasts"],
                    "first_name": r[f"{f}_name"], "second_name": r[f"{other}_name"],
                    "first_loc": r[f"{f}_loc"], "second_loc": r[f"{other}_loc"],
                    "first_score": r[f"{f}_score"], "second_score": r[f"{other}_score"], "espn_status": r["status"]})
    return pl.DataFrame(out, infer_schema_length=None)


def kalshi_games(m: pl.DataFrame) -> pl.DataFrame:
    """One row per Kalshi game event with team codes in ticker order, subtitles and leg results."""
    kind = pl.col("league").replace_strict({k: v[2] for k, v in LEAGUES.items()}, default="us")
    m = m.with_columns(kind.alias("kind"))
    rows = []
    for (ev,), g in m.group_by(["event_ticker"]):
        codes = g["code"].to_list()
        p = parse_event(ev, codes)
        if p is None:
            continue
        sub = dict(zip(g["code"], g["yes_sub_title"], strict=True))
        res = dict(zip(g["code"], g["result"], strict=True))
        tk = dict(zip(g["code"], g["ticker"], strict=True))
        vol = dict(zip(g["code"], g["volume"], strict=True))
        rows.append({"event_ticker": ev, "league": g["league"][0], "kind": g["kind"][0], "series_ticker": g["series_ticker"][0],
                     "et_date": p["et_date"], "first": p["first"], "second": p["second"],
                     "first_sub": sub.get(p["first"]), "second_sub": sub.get(p["second"]),
                     "first_ticker": tk.get(p["first"]), "second_ticker": tk.get(p["second"]), "tie_ticker": tk.get("TIE"),
                     "y_first": 1 if res.get(p["first"]) == "yes" else 0, "y_second": 1 if res.get(p["second"]) == "yes" else 0,
                     "y_tie": None if "TIE" not in res else (1 if res["TIE"] == "yes" else 0),
                     "event_volume": float(sum(v or 0 for v in vol.values())),
                     "close_time": g["close_time"].max(), "settlement_ts": g["settlement_ts"].max(),
                     "fee_type": g["fee_type"][0], "fee_multiplier": g["fee_multiplier"][0]})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- panel ----------------------------------------------------------------------------------------
def leg_quotes(candles: pl.LazyFrame, req: pl.DataFrame) -> pl.DataFrame:
    """For each request row (``ticker``, ``rid``, ``t``): the last two-sided candle quote ending at or
    before ``t`` (at most MAX_QUOTE_AGE_H old) and the leg's traded volume in the window up to ``t``."""
    tickers = req["ticker"].unique().to_list()
    c = (candles.filter(pl.col("ticker").is_in(tickers))
         .select("ticker", "end_ts", "bid_close", "ask_close", "volume").collect()
         .unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
         .with_columns(pl.col("volume").fill_null(0.0).cum_sum().over("ticker").alias("cumvol")))
    ok = c.filter((pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close")))
    r = req.sort("t")
    q = r.join_asof(ok.select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("bid_close").alias("bid"),
                              pl.col("ask_close").alias("ask")).sort("q_ts"),
                    left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                    tolerance=timedelta(hours=MAX_QUOTE_AGE_H), check_sortedness=False)
    v = r.join_asof(c.select("ticker", pl.col("end_ts").alias("v_ts"), "cumvol").sort("v_ts"),
                    left_on="t", right_on="v_ts", by="ticker", strategy="backward", check_sortedness=False)
    return q.join(v.select("rid", "cumvol"), on="rid").select("rid", "q_ts", "bid", "ask", pl.col("cumvol").fill_null(0.0))


def game_table() -> pl.DataFrame:
    """Kalshi games matched to ESPN, with the scheduled start and slot / TV / popularity features."""
    m = pl.read_parquet(OUT / "markets.parquet")
    g = kalshi_games(m)
    x = match_espn(g, pl.read_parquet(OUT / "espn_games.parquet"))
    g = g.join(x, on="event_ticker", how="inner")
    et = pl.col("start").dt.convert_time_zone(ET)
    g = g.with_columns(et.dt.hour().alias("et_hour"), et.dt.weekday().alias("dow"), et.dt.date().alias("start_date"),
                       et.dt.day().alias("dom"))
    g = g.with_columns(
        pl.col("et_hour").is_between(19, 23).alias("primetime"), (pl.col("dow") >= 6).alias("weekend"),
        pl.col("dom").is_in([1, 2, 15, 16]).alias("payday"),
        (pl.col("start") >= HOLDOUT_FROM).alias("holdout"),
        ((pl.col("close_time") >= pl.col("start")) & (pl.col("close_time") <= pl.col("start") + timedelta(hours=18))).alias("played"))
    last = g.group_by("league", "start_date").agg(pl.col("start").max().alias("last_start"))
    g = g.join(last, on=["league", "start_date"]).with_columns(
        ((pl.col("start") == pl.col("last_start")) & (pl.col("et_hour") >= 21)).alias("late_slot"))
    pop = pl.col("league").replace_strict({k: v[2] for k, v in LEAGUES.items()}, default="us")

    def is_pop(name: str, loc: str) -> pl.Expr:
        out = pl.lit(False)
        for lg, names in POPULAR.items():
            col = loc if lg == "NCAAF" else name
            sel = (pop == "soccer") if lg == "SOCCER" else (pl.col("league") == lg)
            out = out | (sel & pl.col(col).is_in(list(names)))
        return out
    return g.with_columns(is_pop("first_name", "first_loc").alias("first_pop"),
                          is_pop("second_name", "second_loc").alias("second_pop"))


def side_panel(g: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (game, team): that team's win price at each lead, its executable ask (YES on its
    own leg, or NO on the opponent's leg in two-way games), the payoff of that contract, and the
    team's attributes. US leagues: one fetched leg per game, the other team is its NO side; soccer:
    the home and away legs, each its own YES."""
    g = game_table() if g is None else g
    g = g.filter(pl.col("played"))
    cand = load_candles()
    have = set(cand.select("ticker").unique().collect()["ticker"].to_list())
    us = g.filter(pl.col("kind") == "us").with_columns(
        pl.min_horizontal("first_ticker", "second_ticker").alias("leg"))
    us = us.with_columns((pl.col("leg") == pl.col("first_ticker")).alias("leg_is_first")).filter(pl.col("leg").is_in(list(have)))
    soc = g.filter(pl.col("kind") == "soccer")
    reqs = []
    for lead in LEADS_H:
        reqs.append(us.select("event_ticker", pl.col("leg").alias("ticker"), (pl.col("start") - timedelta(hours=lead)).alias("t"))
                    .with_columns(pl.lit(lead).alias("lead")))
        for f in ("first", "second"):
            reqs.append(soc.select("event_ticker", pl.col(f"{f}_ticker").alias("ticker"),
                                   (pl.col("start") - timedelta(hours=lead)).alias("t")).with_columns(pl.lit(lead).alias("lead")))
    req = pl.concat(reqs).filter(pl.col("ticker").is_in(list(have))).with_row_index("rid")
    q = req.join(leg_quotes(cand, req), on="rid").drop("rid")
    wide = q.pivot(on="lead", index=["event_ticker", "ticker"], values=["bid", "ask", "cumvol"], aggregate_function="first")
    wide = wide.rename({c: c.replace("_", "", 1) for c in wide.columns if re.match(r"^(bid|ask|cumvol)_\d+$", c)})
    rows = []
    keep = ["event_ticker", "league", "kind", "start", "start_date", "et_hour", "dow", "dom", "primetime", "weekend", "payday",
            "late_slot", "national_tv", "big_tv", "neutral", "holdout", "settlement_ts", "fee_type", "fee_multiplier",
            "event_volume", "y_tie"]
    for f, o in (("first", "second"), ("second", "first")):
        base = {"team": pl.col(f), "opp": pl.col(o), "team_name": pl.col(f"{f}_name"), "opp_name": pl.col(f"{o}_name"),
                "is_first": pl.lit(f == "first"), "is_home": pl.col("first_home") if f == "first" else ~pl.col("first_home"),
                "pop": pl.col(f"{f}_pop"), "opp_pop": pl.col(f"{o}_pop"), "y": pl.col(f"y_{f}"),
                "pts": pl.col(f"{f}_score"), "opp_pts": pl.col(f"{o}_score")}
        # US: YES on the team's leg when it is the fetched leg, else NO on the opponent's leg
        u = us.with_columns(**{k: v.alias(k) for k, v in base.items()}).with_columns(
            (pl.col("leg_is_first") == (f == "first")).alias("own"))
        u = u.join(wide, left_on=["event_ticker", "leg"], right_on=["event_ticker", "ticker"], how="inner")
        for lead in LEADS_H:
            u = u.with_columns(
                pl.when(pl.col("own")).then((pl.col(f"bid{lead}") + pl.col(f"ask{lead}")) / 2)
                .otherwise(1 - (pl.col(f"bid{lead}") + pl.col(f"ask{lead}")) / 2).alias(f"mid{lead}"),
                pl.when(pl.col("own")).then(pl.col(f"ask{lead}")).otherwise(1 - pl.col(f"bid{lead}")).alias(f"px{lead}"),
                (pl.col(f"ask{lead}") - pl.col(f"bid{lead}")).alias(f"spr{lead}"))
        # NO on the opponent's leg pays 1 - y(opponent) (a tie also pays it; NFL only)
        u = u.with_columns(pl.when(pl.col("own")).then(pl.col("y")).otherwise(1 - pl.col(f"y_{o}")).alias("payoff"),
                           pl.col("leg").alias("ticker"), pl.when(pl.col("own")).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("side"))
        s = soc.with_columns(**{k: v.alias(k) for k, v in base.items()}).join(
            wide, left_on=["event_ticker", f"{f}_ticker"], right_on=["event_ticker", "ticker"], how="inner")
        for lead in LEADS_H:
            s = s.with_columns(((pl.col(f"bid{lead}") + pl.col(f"ask{lead}")) / 2).alias(f"mid{lead}"),
                               pl.col(f"ask{lead}").alias(f"px{lead}"), (pl.col(f"ask{lead}") - pl.col(f"bid{lead}")).alias(f"spr{lead}"))
        s = s.with_columns(pl.col("y").alias("payoff"), pl.col(f"{f}_ticker").alias("ticker"), pl.lit("yes").alias("side"))
        cols = keep + list(base) + ["ticker", "side", "payoff"] + [f"{p}{lead}" for p in ("mid", "px", "spr", "cumvol") for lead in LEADS_H]
        rows += [u.select(cols), s.select(cols)]
    p = pl.concat(rows, how="diagonal_relaxed")
    return add_history(p, g)


def team_games(g: pl.DataFrame) -> pl.DataFrame:
    """Every played game twice, once per team: result, points, final event volume (no prices)."""
    out = []
    for f, o in (("first", "second"), ("second", "first")):
        out.append(g.select("league", pl.col(f).alias("team"), "event_ticker", "start", "settlement_ts",
                            pl.col(f"y_{f}").alias("y"), pl.col(f"{f}_score").alias("pts"), pl.col(f"{o}_score").alias("opp_pts"),
                            "event_volume"))
    return pl.concat(out)


def add_history(p: pl.DataFrame, games: pl.DataFrame | None = None) -> pl.DataFrame:
    """Each team's previous game in the same league (price at T-1h, result, margin, streak, prior
    volume), as of the current game's start (the previous game has settled by then; gaps > 45 days
    reset). With ``games`` (every played game), results, margins and streaks come from the complete
    schedule and only the previous price needs a fetched quote."""
    if games is not None:
        t = team_games(games).join(p.select("event_ticker", "team", pl.col("mid1").alias("p_pre")).unique(["event_ticker", "team"]),
                                   on=["event_ticker", "team"], how="left")
    else:
        t = p.select("league", "team", "event_ticker", "start", "settlement_ts", pl.col("mid1").alias("p_pre"), "y", "pts", "opp_pts",
                     *(["event_volume"] if "event_volume" in p.columns else []))
    t = t.unique(["league", "team", "event_ticker"]).sort("league", "team", "start")
    if "event_volume" not in t.columns:
        t = t.with_columns(pl.lit(None, dtype=pl.Float64).alias("event_volume"))
    t = t.with_columns((pl.col("y") - pl.col("p_pre")).alias("surprise"), (pl.col("pts") - pl.col("opp_pts")).alias("margin"))
    grp = ["league", "team"]
    t = t.with_columns(
        pl.col("surprise").shift(1).over(grp).alias("prev_surprise"), pl.col("y").shift(1).over(grp).alias("prev_win"),
        pl.col("p_pre").shift(1).over(grp).alias("prev_p"), pl.col("margin").shift(1).over(grp).alias("prev_margin"),
        pl.col("start").shift(1).over(grp).alias("prev_start"), pl.col("settlement_ts").shift(1).over(grp).alias("prev_settle"),
        # as-of attention: mean final event volume of the team's previous 10 games (all settled before this one)
        pl.col("event_volume").log1p().shift(1).rolling_mean(10, min_samples=3).over(grp).alias("prev_logvol"))
    # streak of identical results ending with the previous game (+k wins / -k losses)
    t = t.with_columns((pl.col("y") != pl.col("y").shift(1).over(grp)).fill_null(True).cum_sum().over(grp).alias("_run"))
    t = t.with_columns(pl.col("y").cum_count().over(grp + ["_run"]).cast(pl.Int32).alias("_len"))
    t = t.with_columns(pl.when(pl.col("y") == 1).then(pl.col("_len")).otherwise(-pl.col("_len")).shift(1).over(grp).alias("prev_streak"))
    fresh = ((pl.col("start") - pl.col("prev_start")) <= timedelta(days=45)) & (pl.col("prev_settle") < pl.col("start"))
    hist = ["prev_surprise", "prev_win", "prev_p", "prev_margin", "prev_streak", "prev_logvol"]
    t = t.with_columns([pl.when(fresh).then(pl.col(c)).alias(c) for c in hist])
    t = t.select("league", "team", "event_ticker", *hist)
    p = p.join(t, on=["league", "team", "event_ticker"], how="left")
    o = t.rename({"team": "opp", **{c: f"opp_{c}" for c in hist}})
    return p.join(o, on=["league", "opp", "event_ticker"], how="left")


# ---- statistics -------------------------------------------------------------------------------------
def _cluster_se_mean(r: np.ndarray, cl: np.ndarray) -> float:
    n = len(r)
    if n < 2:
        return float("nan")
    e = r - r.mean()
    _, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=e)
    g = len(s)
    adj = g / (g - 1) if g > 1 else float("nan")
    return float(math.sqrt(adj * float((s ** 2).sum())) / n)


def cmean(df: pl.DataFrame, col: str = "r", clusters: tuple[str, ...] = ("event_ticker", "start_date")) -> dict[str, Any]:
    """Mean of ``col`` (in cents) with cluster-robust t by each clustering; ``t`` uses the larger SE."""
    d = df.filter(pl.col(col).is_not_null())
    r = d[col].to_numpy().astype(float)
    out: dict[str, Any] = {"n": len(r), "games": d["event_ticker"].n_unique() if "event_ticker" in d.columns else None}
    if len(r) < 2:
        return {**out, "mean_c": None, "t": None}
    ses = {c: _cluster_se_mean(r, d[c].cast(pl.String).to_numpy()) for c in clusters if c in d.columns}
    se = max(v for v in ses.values() if np.isfinite(v)) if any(np.isfinite(v) for v in ses.values()) else float("nan")
    out.update({"mean_c": round(100 * float(r.mean()), 2), "se_c": round(100 * se, 2),
                "t": round(float(r.mean()) / se, 2) if se > 0 else None,
                **{f"t_{c}": round(float(r.mean()) / v, 2) if v > 0 else None for c, v in ses.items()}})
    return out


def cslope(df: pl.DataFrame, y: str, x: str, clusters: tuple[str, ...] = ("event_ticker", "start_date")) -> dict[str, Any]:
    """OLS slope of y on x (with intercept), cluster-robust (sandwich) t, larger SE of the clusterings."""
    d = df.filter(pl.col(y).is_not_null() & pl.col(x).is_not_null())
    n = d.height
    if n < 10:
        return {"n": n, "slope": None, "t": None}
    Y = d[y].to_numpy().astype(float)
    X = np.column_stack([np.ones(n), d[x].to_numpy().astype(float)])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b
    ses = {}
    for c in clusters:
        if c not in d.columns:
            continue
        _, inv = np.unique(d[c].cast(pl.String).to_numpy(), return_inverse=True)
        S = np.zeros((inv.max() + 1, 2))
        np.add.at(S, inv, X * e[:, None])
        g = S.shape[0]
        V = XtX_inv @ (S.T @ S) @ XtX_inv * (g / (g - 1)) * ((n - 1) / (n - 2))
        ses[c] = math.sqrt(V[1, 1])
    se = max(ses.values())
    return {"n": n, "games": d["event_ticker"].n_unique(), "slope": round(float(b[1]), 4), "icpt_c": round(100 * float(b[0]), 2),
            "t": round(float(b[1]) / se, 2), **{f"t_{c}": round(float(b[1]) / v, 2) for c, v in ses.items()}}


class Tally:
    """Every (hypothesis, cut) cell evaluated, so the test count is exact."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, hyp: str, cut: str, res: dict[str, Any], **extra: Any) -> None:
        self.rows.append({"hyp": hyp, "cut": cut, **extra, **res})

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows, infer_schema_length=None)


# ---- H5: attention on player props ------------------------------------------------------------------
PE = Path("data/research/pre_event")


def props_panel(legs: pl.DataFrame | None = None, hourly: pl.LazyFrame | None = None, lead_h: int = 1) -> pl.DataFrame:
    """Player-prop legs (``pre_event`` sample: seeded, volume- and result-blind, <= 30 legs per game)
    with the quote at start - ``lead_h`` and the as-of attention measures: the leg's traded volume
    in the hours up to the decision (within the cached start - 30 h window), its rank and share
    among the sampled legs of the same event, and the player's share within the game (all series)."""
    legs = pl.read_parquet(PE / "legs.parquet") if legs is None else legs
    legs = legs.filter((pl.col("group") == "sports prop") & pl.col("start").is_not_null())
    hourly = pl.scan_parquet(PE / "hourly.parquet") if hourly is None else hourly
    req = legs.select("ticker", (pl.col("start") - timedelta(hours=lead_h)).alias("t")).with_row_index("rid")
    c = (hourly.filter(pl.col("ticker").is_in(req["ticker"].to_list()))
         .select("ticker", "end_ts", pl.col("bid_close"), pl.col("ask_close"), "volume").collect()
         .unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
         .with_columns(pl.col("volume").fill_null(0.0).cum_sum().over("ticker").alias("cumvol")))
    q = leg_quotes(c.lazy(), req)
    p = legs.with_row_index("rid").join(q, on="rid").drop("rid")
    p = p.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
                       pl.col("ticker").str.split("-").list.get(2, null_on_oob=True).alias("player"),
                       pl.col("start").dt.convert_time_zone(ET).dt.date().alias("start_date"))
    p = p.with_columns(
        pl.col("cumvol").rank("average").over("event_ticker").alias("att_rank"), pl.len().over("event_ticker").alias("ev_legs"),
        (pl.col("cumvol") / pl.col("cumvol").sum().over("event_ticker")).alias("att_share"))
    p = p.with_columns(((pl.col("att_rank") - 0.5) / pl.col("ev_legs")).alias("att_pct"),
                       (pl.col("y").cast(pl.Float64) - pl.col("mid")).alias("r"),
                       pl.col("series_ticker").str.replace(r"PROT$", "").alias("family"))
    return p


# ---- dev exploration ----------------------------------------------------------------------------------
SIGNAL_SPREAD = 0.05          # quotes used as prices in the tests: spread <= 5c (median is 1c)
PRICE_BAND = (0.03, 0.97)


def priced(p: pl.DataFrame, lead: int = 1) -> pl.DataFrame:
    """Rows with a usable quote at ``lead`` and the residual ``r`` = payoff - mid of that side."""
    return (p.filter(pl.col(f"mid{lead}").is_between(*PRICE_BAND) & (pl.col(f"spr{lead}") <= SIGNAL_SPREAD))
            .with_columns((pl.col("payoff") - pl.col(f"mid{lead}")).alias("r")))


def one_per_game(p: pl.DataFrame, cond: pl.Expr) -> pl.DataFrame:
    """The side(s) meeting ``cond``; in two-way games the opposite side is the mirror image, so a
    game contributes one row (soccer legs are separate contracts and both may qualify)."""
    return p.filter(cond).unique(["event_ticker", "team"]).sort("start")


def explore_games(p: pl.DataFrame, T: Tally) -> None:
    us = pl.col("kind") == "us"
    soccer = pl.col("kind") == "soccer"
    leagues_us = ["NFL", "NBA", "MLB", "NHL", "NCAAF", "WNBA"]
    d1 = priced(p, 1)
    # descriptive calibration (FLB) so later cells can be read against it
    fav = pl.col("mid1") > 0.5
    for k, cond in (("us", us), ("soccer", soccer)):
        for lo, hi in ((0.03, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.97)):
            T.add("FLB", f"{k} mid {lo:.2f}-{hi:.2f}", cmean(d1.filter(cond & pl.col("mid1").is_between(lo, hi, closed="left")).unique(["event_ticker", "team"])))
    # H1 popular side, one per game (exactly one side on the frozen list)
    pop = pl.col("pop") & ~pl.col("opp_pop")
    T.add("H1", "popular, all leagues, T-1h", cmean(one_per_game(d1, pop)))
    for lg in leagues_us:
        T.add("H1", f"popular, {lg}", cmean(one_per_game(d1, pop & (pl.col("league") == lg))))
    T.add("H1", "popular, soccer", cmean(one_per_game(d1, pop & soccer)))
    for lead in (24, 0):
        T.add("H1", f"popular, all, T-{lead}h", cmean(one_per_game(priced(p, lead), pop)))
    for name, c in (("primetime", pl.col("primetime")), ("weekend", pl.col("weekend")), ("big national TV", pl.col("big_tv")),
                    ("as favourite", fav), ("as underdog", ~fav)):
        T.add("H1", f"popular & {name}", cmean(one_per_game(d1, pop & c)))
    hv = (pl.col("prev_logvol") - pl.col("opp_prev_logvol")) >= math.log(2)      # >= 2x the opponent's prior volume
    T.add("H1", "as-of volume popularity (team's prior games >= 2x opponent's), all", cmean(one_per_game(d1, hv)))
    T.add("H1", "as-of volume popularity, US", cmean(one_per_game(d1, hv & us)))
    T.add("H1", "as-of volume popularity, soccer", cmean(one_per_game(d1, hv & soccer)))
    # H2 home side (US: first-listed = away, so 'home' is also 'second-listed'); soccer home leg YES
    home = pl.col("is_home") & ~pl.col("neutral")
    T.add("H2", "home, US pooled", cmean(one_per_game(d1, home & us)))
    for lg in leagues_us:
        T.add("H2", f"home, {lg}", cmean(one_per_game(d1, home & (pl.col("league") == lg))))
    T.add("H2", "home leg YES, soccer", cmean(one_per_game(d1, home & soccer)))
    T.add("H2", "away leg YES, soccer", cmean(one_per_game(d1, ~pl.col("is_home") & ~pl.col("neutral") & soccer)))
    # H3 recency: first team's residual on its last-game surprise minus the opponent's
    f = d1.filter(us & pl.col("is_first")).with_columns(
        (pl.col("prev_surprise") - pl.col("opp_prev_surprise")).alias("ds"))
    T.add("H3", "slope r ~ (surprise - opp surprise), US", cslope(f, "r", "ds"))
    for lg in ["NFL", "NBA", "MLB", "NHL", "NCAAF"]:
        T.add("H3", f"slope, {lg}", cslope(f.filter(pl.col("league") == lg), "r", "ds"))
    T.add("H3", "slope, soccer legs", cslope(d1.filter(soccer).with_columns(
        (pl.col("prev_surprise") - pl.col("opp_prev_surprise")).alias("ds")).unique(["event_ticker", "team"]), "r", "ds"))
    blow = pl.col("league").replace_strict(BLOWOUT, default=3, return_dtype=pl.Float64)

    def hist_cond(pre: str) -> dict[str, pl.Expr]:
        c = lambda k: pl.col(pre + k)
        return {"after upset win (prev p<=.4, won)": (c("prev_win") == 1) & (c("prev_p") <= 0.4),
                "after upset loss (prev p>=.6, lost)": (c("prev_win") == 0) & (c("prev_p") >= 0.6),
                "after blowout win": c("prev_margin") >= blow, "after blowout loss": c("prev_margin") <= -blow,
                "win streak >= 3": c("prev_streak") >= 3, "loss streak >= 3": c("prev_streak") <= -3}
    own, opp = hist_cond(""), hist_cond("opp_")
    for name, c in own.items():     # the team meets the condition and its opponent does not
        T.add("H3", name, cmean(one_per_game(d1, c & ~opp[name].fill_null(False))))
    # H4 primetime / weekend / national TV: favourite side; late drift reversal
    for name, c in (("primetime", pl.col("primetime")), ("not primetime", ~pl.col("primetime")), ("weekend", pl.col("weekend")),
                    ("weekday", ~pl.col("weekend")), ("big national TV", pl.col("big_tv")), ("not big TV", ~pl.col("big_tv"))):
        T.add("H4", f"favourite & {name}, US", cmean(one_per_game(d1, fav & us & c)))
    d0 = priced(p, 0).filter(us & pl.col("is_first") & pl.col("mid6").is_not_null() & (pl.col("spr6") <= SIGNAL_SPREAD)).with_columns(
        (pl.col("mid0") - pl.col("mid6")).alias("drift"))
    for name, c in (("primetime", pl.col("primetime")), ("not primetime", ~pl.col("primetime")),
                    ("big national TV", pl.col("big_tv")), ("not big TV", ~pl.col("big_tv"))):
        T.add("H4", f"slope r(T) ~ drift(T-6h..T), {name}", cslope(d0.filter(c), "r", "drift"))
    # H10 late slot: underdog side
    dog = pl.col("mid1") <= 0.4
    T.add("H10", "underdog, late slot, US", cmean(one_per_game(d1, dog & us & pl.col("late_slot"))))
    T.add("H10", "underdog, earlier slots, US", cmean(one_per_game(d1, dog & us & ~pl.col("late_slot"))))
    for lg in ["MLB", "NBA", "NHL", "NCAAF", "NFL"]:
        T.add("H10", f"underdog, late slot, {lg}", cmean(one_per_game(d1, dog & pl.col("late_slot") & (pl.col("league") == lg))))
    # H14 payday
    T.add("H14", "favourite, payday dates, US", cmean(one_per_game(d1, fav & us & pl.col("payday"))))
    T.add("H14", "favourite, other dates, US", cmean(one_per_game(d1, fav & us & ~pl.col("payday"))))


def explore_props(p: pl.DataFrame, T: Tally) -> None:
    p = p.filter((pl.col("spread") <= SIGNAL_SPREAD) & pl.col("mid").is_between(*PRICE_BAND) & (pl.col("ev_legs") >= 3))
    p = p.with_columns(pl.when(pl.col("att_pct") >= 2 / 3).then(pl.lit("top")).when(pl.col("att_pct") < 1 / 3).then(pl.lit("bot"))
                       .otherwise(pl.lit("mid")).alias("att"),
                       pl.col("series_ticker").str.extract(r"^KX(MLB|NFL|WNBA|NHL|NBA)").alias("sport"))
    for a in ("top", "mid", "bot"):
        T.add("H5", f"props att={a}", cmean(p.filter(pl.col("att") == a)))
    for lo, hi in ((0.03, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.97)):
        for a in ("top", "bot"):
            T.add("H5", f"props mid {lo}-{hi} att={a}", cmean(p.filter(pl.col("mid").is_between(lo, hi, closed="left") & (pl.col("att") == a))))
    for s in ("MLB", "NFL", "WNBA", "NHL"):
        for a in ("top", "bot"):
            T.add("H5", f"props {s} att={a}", cmean(p.filter((pl.col("sport") == s) & (pl.col("att") == a))))


def run_explore() -> pl.DataFrame:
    """Dev-only exploration: every cell is written to ``explore_dev.csv`` (the test count)."""
    g = pl.read_parquet(OUT / "games.parquet") if (OUT / "games.parquet").exists() else game_table()
    p = side_panel(g)
    p.write_parquet(OUT / "panel.parquet")
    dev = p.filter(~pl.col("holdout"))
    T = Tally()
    explore_games(dev, T)
    explore_props(props_panel().filter(~pl.col("holdout")), T)
    explore_keynum(keynum_rows().filter(~pl.col("holdout")), T)
    if (OUT / "tennis_markets.parquet").exists():
        explore_tennis(tennis_panel().filter(~pl.col("holdout")), T)
    out = T.frame()
    out.write_csv(OUT / "explore_dev.csv")
    return out


# ---- execution, ledgers, metrics ------------------------------------------------------------------------
STAKE = 5.0                   # $ per position
GAME_CAP = 10.0               # open cost per game in the capped book
BANKROLL = 100.0


def fee_order(n: pl.Expr, price: pl.Expr, maker: bool = False) -> pl.Expr:
    """Kalshi fee for one order of n contracts, rounded up to the cent. Taker 0.07 x mult x n P(1-P);
    maker 0.0175 x mult x n P(1-P) only on ``*_with_maker_fees`` series (``fees.kalshi_fee``)."""
    from albert.fees import KALSHI_MAKER, KALSHI_TAKER

    coef = pl.lit(KALSHI_TAKER) if not maker else pl.when(pl.col("fee_type").str.contains("with_maker_fees")).then(
        pl.lit(KALSHI_MAKER)).otherwise(pl.lit(0.0))
    raw = coef * pl.col("fee_multiplier").fill_null(1.0) * n * price * (1 - price)
    return (raw * 100).round(9).ceil() / 100


def taker_trades(rows: pl.DataFrame, lead: int, stake: float = STAKE) -> pl.DataFrame:
    """Buy each row's side at its ask at start - ``lead`` (``px{lead}``), ~$stake order, held to
    settlement. Only quotes with spread <= SIGNAL_SPREAD are crossed."""
    f = rows.filter(pl.col(f"px{lead}").is_between(0.01, 0.99) & (pl.col(f"spr{lead}") <= SIGNAL_SPREAD))
    f = f.with_columns(pl.col(f"px{lead}").alias("price"), pl.col(f"mid{lead}").alias("mid_in"),
                       (pl.col("start") - timedelta(hours=lead)).alias("t"))
    f = f.with_columns(pl.max_horizontal(pl.lit(1.0), (stake / pl.col("price")).floor()).alias("n"))
    f = f.with_columns(fee_order(pl.col("n"), pl.col("price")).alias("fee_order"))
    return f.with_columns((pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n")).alias("pnl"),
                          (pl.col("payoff") - pl.col("mid_in")).alias("gross"),
                          pl.when(pl.col("side") == "yes").then(1).otherwise(-1).alias("dir"))


def maker_trades(rows: pl.DataFrame, lead: int, improve: float = 0.0, stake: float = STAKE,
                 candles: pl.DataFrame | None = None) -> pl.DataFrame:
    """Rest a bid for each row's side at its best bid + ``improve`` (never at or above the ask) from
    start - ``lead`` to the start; filled only if a later hourly candle traded strictly through the
    bid (side YES: trade low < bid; side NO: YES trade high > 1 - bid). Maker fee on maker-fee series."""
    c = (load_candles() if candles is None else candles.lazy()).select("ticker", "end_ts", "p_low", "p_high").collect()
    f = rows.filter(pl.col(f"px{lead}").is_between(0.01, 0.99) & pl.col(f"spr{lead}").is_not_null())
    f = f.with_columns((pl.col(f"px{lead}") - pl.col(f"spr{lead}") + improve).alias("bid_side"),
                       (pl.col("start") - timedelta(hours=lead)).alias("t"), pl.col(f"mid{lead}").alias("mid_in"))
    f = f.filter(pl.col("bid_side") < pl.col(f"px{lead}") - 1e-9)
    w = f.select("event_ticker", "team", "ticker", "side", "bid_side", "t", "start").join(c, on="ticker")
    w = w.filter((pl.col("end_ts") > pl.col("t") + timedelta(hours=1) - timedelta(seconds=1)) & (pl.col("end_ts") <= pl.col("start")))
    thr = pl.when(pl.col("side") == "yes").then(pl.col("p_low") < pl.col("bid_side") - 1e-9).otherwise(
        pl.col("p_high") > 1 - pl.col("bid_side") + 1e-9)
    hit = w.filter(thr).group_by("event_ticker", "team").agg(pl.col("end_ts").min().alias("t_fill"))
    f = f.join(hit, on=["event_ticker", "team"], how="inner").with_columns(pl.col("bid_side").alias("price"),
                                                                           pl.col("t_fill").alias("t"))
    f = f.with_columns(pl.max_horizontal(pl.lit(1.0), (stake / pl.col("price")).floor()).alias("n"))
    f = f.with_columns(fee_order(pl.col("n"), pl.col("price"), maker=True).alias("fee_order"))
    return f.with_columns((pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n")).alias("pnl"),
                          (pl.col("payoff") - pl.col("mid_in")).alias("gross"),
                          pl.when(pl.col("side") == "yes").then(1).otherwise(-1).alias("dir"))


def trade_stats(tr: pl.DataFrame) -> dict[str, Any]:
    """Per-contract net PnL (c) with t clustered by game and by ET date (larger SE), plus gross vs mid."""
    if tr.is_empty():
        return {"trades": 0}
    w = tr.with_columns(pl.col("pnl").alias("r"))
    net = cmean(w)
    gross = cmean(tr.with_columns(pl.col("gross").alias("r")))
    return {"trades": tr.height, "games": tr["event_ticker"].n_unique(), "dates": tr["start_date"].n_unique(),
            "net_c": net.get("mean_c"), "t_net": net.get("t"), "gross_c": gross.get("mean_c"), "t_gross": gross.get("t"),
            "avg_price": round(float(tr["price"].mean()), 3), "fee_c": round(100 * float((tr["fee_order"] / tr["n"]).mean()), 2),
            "win": round(float(tr["payoff"].mean()), 3), "usd": round(float((tr["pnl"] * tr["n"]).sum()), 2)}


def capped_ledger(tr: pl.DataFrame, strategy: str, variant: str, split: str) -> pl.DataFrame:
    """Trades in time order on a $100 book: taken while total open cost stays <= bankroll and the
    game's open cost <= GAME_CAP; capital returns at settlement (``ta_calendar.capped_ledger``)."""
    from albert.research import ta_calendar as tc

    if tr.is_empty():
        return pl.DataFrame()
    return tc.capped_ledger(tr.select("t", "n", "price", "fee_order", "event_ticker", "settlement_ts", "ticker", "dir", "payoff"),
                            strategy, variant, split, bankroll=BANKROLL, event_cap=GAME_CAP)


def marks_for(tickers: list[str]) -> pl.DataFrame:
    """Hourly YES mids (ticker, ts, yes_mid) from this study's candles, for daily MTM."""
    c = load_candles().filter(pl.col("ticker").is_in(tickers)).select("ticker", "end_ts", "bid_close", "ask_close", "p_close").collect()
    two = (pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close"))
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid").unique(["ticker", "ts"]).sort("ticker", "ts"))


def ledger_stats(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    if led.is_empty() or led.height < 3:
        return {"positions": 0 if led.is_empty() else led.height}
    m = ledger_metrics(led, marks_for(led["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct", "skew",
            "first_day", "last_day")
    return {k: m.get(k) for k in keep}


# ---- H15: key numbers in football spread ladders -----------------------------------------------------
VG = Path("data/research/vol_greeks")
KEY_NUMBERS = (3, 7, 10, 14)


def keynum_rows(lead_h: int = 1, games: pl.DataFrame | None = None, legs: pl.DataFrame | None = None,
                quotes: pl.DataFrame | None = None) -> pl.DataFrame:
    """Implied vs realized probability that the full-game margin equals an integer m, from adjacent
    spread strikes one point apart (P(M > m - .5) - P(M > m + .5), and the mirror for the other team),
    quoted at start - ``lead_h`` (start from the matched game-winner event). Cached ``vol_greeks`` ladders."""
    if legs is None:
        legs = (pl.scan_parquet(VG / "paths_legs.parquet")
                .filter(pl.col("series_ticker").is_in(["KXNFLSPREAD", "KXNCAAFSPREAD"]) & (pl.col("op") == "spread"))
                .select("ticker", "ladder", "series_ticker", "lo", "hi", "y").collect())
    g = pl.read_parquet(OUT / "games.parquet") if games is None else games
    g = g.select(pl.col("event_ticker").str.replace(r"^KX(NFL|NCAAF)GAME", "KX${1}SPREAD").alias("ladder"), "start",
                 "start_date", "league")
    legs = legs.join(g, on="ladder", how="inner")
    q = quotes if quotes is not None else pl.scan_parquet(VG / "paths_sp_spread_quotes.parquet").filter(
        pl.col("ladder").is_in(legs["ladder"].unique().to_list())).collect()
    req = legs.select("ticker", (pl.col("start") - timedelta(hours=lead_h)).alias("t")).sort("t")
    ok = q.filter((pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))).select(
        "ticker", pl.col("t").alias("q_ts"), "bid", "ask").sort("q_ts")
    qq = req.join_asof(ok, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                       tolerance=timedelta(hours=MAX_QUOTE_AGE_H), check_sortedness=False)
    legs = legs.join(qq.select("ticker", "bid", "ask"), on="ticker", how="left").with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"))
    # strike on the margin of the team named in the leg: first team lo = m - .5 (hi = inf); second team hi = -(m - .5)
    up = legs.filter(pl.col("lo").is_finite()).with_columns(pl.col("lo").alias("k"), pl.lit("first").alias("team"))
    dn = legs.filter(~pl.col("lo").is_finite() & pl.col("hi").is_finite()).with_columns((-pl.col("hi")).alias("k"),
                                                                                      pl.lit("second").alias("team"))
    lad = pl.concat([up, dn]).sort("ladder", "team", "k")
    nxt = lad.select("ladder", "team", (pl.col("k") - 1).alias("k"), pl.col("mid").alias("mid_up"), pl.col("y").alias("y_up"),
                     pl.col("spread").alias("spread_up"))
    pr = lad.join(nxt, on=["ladder", "team", "k"], how="inner").with_columns(
        (pl.col("k") + 0.5).cast(pl.Int64).alias("m"), (pl.col("mid") - pl.col("mid_up")).alias("implied"),
        (pl.col("y") - pl.col("y_up")).cast(pl.Float64).alias("realized"))
    return pr.with_columns((pl.col("realized") - pl.col("implied")).alias("r"), pl.col("m").is_in(list(KEY_NUMBERS)).alias("key"),
                           (pl.col("start") >= HOLDOUT_FROM).alias("holdout"), pl.col("ladder").alias("event_ticker"))


def explore_keynum(k: pl.DataFrame, T: Tally) -> None:
    k = k.filter(pl.col("implied").is_not_null() & (pl.col("spread") <= 0.10) & (pl.col("spread_up") <= 0.10))
    for lg in ("NFL", "NCAAF"):
        for key in (True, False):
            T.add("H15", f"{lg} margin mass, {'key 3/7/10/14' if key else 'other margins'}",
                  cmean(k.filter((pl.col("league") == lg) & (pl.col("key") == key))))


# ---- H11: American tennis players on a US venue ----------------------------------------------------------
TENNIS_SERIES = ("KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH")
US_PLAYERS = {   # frozen in hypotheses.md before any test
    "Taylor Fritz", "Tommy Paul", "Ben Shelton", "Frances Tiafoe", "Sebastian Korda", "Brandon Nakashima", "Alex Michelsen",
    "Learner Tien", "Reilly Opelka", "Marcos Giron", "Christopher Eubanks", "Jenson Brooksby", "Aleksandar Kovacevic",
    "Ethan Quinn", "Emilio Nava", "Mackenzie McDonald", "Brandon Holt", "Nishesh Basavareddy", "Zachary Svajda",
    "Tristan Boyer", "Eliot Spizzirri", "Michael Mmoh", "Denis Kudla", "Patrick Kypson", "Martin Damm", "Darwin Blanch",
    "Coco Gauff", "Jessica Pegula", "Madison Keys", "Emma Navarro", "Danielle Collins", "Amanda Anisimova", "Sofia Kenin",
    "McCartney Kessler", "Peyton Stearns", "Hailey Baptiste", "Ashlyn Krueger", "Iva Jovic", "Caty McNally",
    "Taylor Townsend", "Bernarda Pera", "Alycia Parks", "Robin Montgomery", "Katie Volynets", "Sloane Stephens",
    "Venus Williams", "Ann Li", "Clervie Ngounoue", "Elizabeth Mandlik", "Varvara Lepchenko", "Alison Riske-Amritraj",
    "Shelby Rogers"}
TENNIS_FROM = datetime(2026, 8, 13, tzinfo=UTC)


def fetch_tennis() -> pl.DataFrame:
    """Tennis singles milestones (match start) and settled ATP / WTA / Challenger markets with one
    listed American; then hourly candles of the American's leg over the 30 h before the start."""
    from albert.http import HttpError
    from albert.research.xcat import OHLC_SCHEMA, batch_rows, ohlc_rows
    from albert.venues.kalshi import parse_ts

    api = _api()
    ms, cursor = [], None
    while True:
        page = api.client.get("/milestones", {"category": "Sports", "type": "tennis_tournament_singles", "limit": 500,
                                              "minimum_start_date": TENNIS_FROM.strftime("%Y-%m-%dT%H:%M:%SZ"), "cursor": cursor})
        got = page.payload.get("milestones") or []
        for x in got:
            d = x.get("details") or {}
            ev = d.get("main_game_event_ticker") or next(iter(x.get("primary_event_tickers") or []), None)
            ms.append({"event_ticker": ev, "start": parse_ts(x.get("start_date")), "tour": d.get("tour"),
                       "round": d.get("round"), "tournament": d.get("tournament_name")})
        cursor = page.payload.get("cursor")
        starts = [s for s in (parse_ts(x.get("start_date")) for x in got) if s is not None]
        if not cursor or not got or (starts and min(starts) > DATA_TO):
            break
    mil = pl.DataFrame(ms, infer_schema_length=None).drop_nulls(["event_ticker", "start"]).sort("start").unique("event_ticker", keep="first")
    fees = {r[0]: (r[1], r[2]) for r in pl.read_parquet("data/tables/kalshi_series.parquet")
            .select("series_ticker", "fee_type", "fee_multiplier").iter_rows()}
    rows: list[dict[str, Any]] = []
    old = (pl.scan_parquet("data/tables/kalshi_markets.parquet").filter(pl.col("series_ticker").is_in(list(TENNIS_SERIES))
           & pl.col("result").is_in(["yes", "no"])).select(MARKET_COLS[:-1]).collect().with_columns(pl.lit("table").alias("source")))
    for series in TENNIS_SERIES:
        for page in api.series_markets(series, "settled", min_close=datetime(2026, 10, 31, tzinfo=UTC), max_close=DATA_TO, cache=True):
            rows.extend(_market_rows(page.payload, "live", fees))
    new = pl.DataFrame(rows, infer_schema_length=None).select(MARKET_COLS) if rows else old.head(0)
    m = pl.concat([old, new.with_columns(pl.col(c).cast(old.schema[c]) for c in old.columns)]).unique("ticker", keep="last")
    m = m.with_columns(pl.col("yes_sub_title").is_in(list(US_PLAYERS)).alias("us"))
    ev = m.group_by("event_ticker").agg(pl.col("us").sum().alias("n_us"), pl.len().alias("legs"))
    m = m.join(ev, on="event_ticker").filter((pl.col("n_us") == 1) & (pl.col("legs") == 2) & pl.col("us"))
    m = m.join(mil, on="event_ticker", how="inner").filter(pl.col("start") >= TENNIS_FROM)
    m.write_parquet(OUT / "tennis_markets.parquet")
    parts = OUT / "tennis_candles"
    parts.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    cutoff = datetime(2026, 9, 18, tzinfo=UTC)
    tgt = m.with_columns((pl.col("start") - timedelta(hours=30)).alias("lo"), pl.col("start").alias("hi"))
    for tk, lo, hi in tgt.filter(pl.col("settlement_ts") < cutoff).select("ticker", "lo", "hi").rows():
        try:
            out.extend(ohlc_rows(api.candles(tk, lo, hi, 60, historical=True).payload, tk, 60))
        except HttpError as exc:
            print(f"tennis candles {tk}: {exc}", flush=True)
    live = tgt.filter(pl.col("settlement_ts") >= cutoff).sort("hi").select("ticker", "lo", "hi").rows()
    i = 0
    while i < len(live):     # <= 100 tickers whose starts span <= 24 h: <= 54 candles each
        j = i
        while j < len(live) and j - i < 100 and live[j][2] - live[i][2] <= timedelta(hours=24):
            j += 1
        chunk = live[i:j]
        i = j
        got = batch_rows(api.candles_batch([r[0] for r in chunk], min(r[1] for r in chunk), max(r[2] for r in chunk), 60).payload, 60)
        for tk, lo, hi in chunk:
            out.extend(r for r in got.get(tk, []) if r["end_ts"] is not None and lo <= r["end_ts"] <= hi)
    pl.DataFrame(out, schema=OHLC_SCHEMA).write_parquet(parts / "part-00000.parquet")
    print(f"tennis: {m.height} matches with one listed American, {len(out)} candles", flush=True)
    return m


def tennis_panel(lead_h: int = 2) -> pl.DataFrame:
    m = pl.read_parquet(OUT / "tennis_markets.parquet")
    c = pl.scan_parquet(str(OUT / "tennis_candles" / "part-*.parquet"))
    req = m.select("ticker", (pl.col("start") - timedelta(hours=lead_h)).alias("t")).with_row_index("rid")
    q = leg_quotes(c, req)
    p = m.with_row_index("rid").join(q, on="rid").drop("rid")
    return p.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
                          (pl.col("result") == "yes").cast(pl.Float64).alias("y"), pl.col("ticker").alias("event_key"),
                          pl.col("start").dt.convert_time_zone(ET).dt.date().alias("start_date"),
                          (pl.col("start") >= HOLDOUT_FROM).alias("holdout")).with_columns(
        (pl.col("y") - pl.col("mid")).alias("r"))


def explore_tennis(p: pl.DataFrame, T: Tally) -> None:
    p = p.filter((pl.col("spread") <= SIGNAL_SPREAD) & pl.col("mid").is_between(*PRICE_BAND))
    T.add("H11", "American's leg YES, all tours, T-2h", cmean(p))
    T.add("H11", "American, ATP/WTA main tour", cmean(p.filter(pl.col("series_ticker").is_in(["KXATPMATCH", "KXWTAMATCH"]))))
    T.add("H11", "American, Challenger", cmean(p.filter(pl.col("series_ticker").str.contains("CHALLENGER"))))
    T.add("H11", "American as favourite", cmean(p.filter(pl.col("mid") > 0.5)))
    T.add("H11", "American as underdog", cmean(p.filter(pl.col("mid") <= 0.5)))


# ---- pre-registered rules: dev vs holdout ----------------------------------------------------------------
def split_of(df: pl.DataFrame) -> pl.Expr:
    return pl.when(pl.col("holdout")).then(pl.lit("holdout")).otherwise(pl.lit("dev"))


def robustness(tr: pl.DataFrame) -> list[dict[str, Any]]:
    """Net c/contract by league, price bucket and month (each split)."""
    out = []
    tr = tr.with_columns(pl.col("price").cut([0.2, 0.4, 0.6, 0.8], labels=["<20c", "20-40c", "40-60c", "60-80c", ">=80c"])
                         .cast(pl.String).alias("pbucket"), pl.col("start").dt.strftime("%Y-%m").alias("month"))
    for col in ("league", "pbucket", "month"):
        for (sp, v), sub in tr.group_by(["split", col]):
            out.append({"by": col, "value": v, "split": sp, **trade_stats(sub)})
    return out


def evaluate(name: str, trades: pl.DataFrame) -> tuple[list[dict[str, Any]], list[pl.DataFrame]]:
    """Per-split trade statistics and capped-book ledger metrics for one rule's trades."""
    rows, leds = [], []
    for sp in ("dev", "holdout"):
        sub = trades.filter(pl.col("split") == sp)
        led = capped_ledger(sub.sort("t"), name, name, sp)
        if not led.is_empty():
            leds.append(led)
        rows.append({"rule": name, "split": sp, **trade_stats(sub), **{f"book_{k}": v for k, v in ledger_stats(led).items()}})
    return rows, leds


# ---- holdout driver --------------------------------------------------------------------------------------
def _blow() -> pl.Expr:
    return pl.col("league").replace_strict(BLOWOUT, default=3, return_dtype=pl.Float64)


def _after_blowout_loss() -> pl.Expr:
    b = _blow()
    return (pl.col("prev_margin") <= -b) & ~(pl.col("opp_prev_margin") <= -b).fill_null(False)


def _after_narrow_loss() -> pl.Expr:
    b = (_blow() / 3).ceil()
    return (pl.col("prev_margin") < 0) & (pl.col("prev_margin") >= -b) & ~(pl.col("opp_prev_margin") < 0).fill_null(False)


# pre-registered in hypotheses.md (2026-11-18 ~07:55 UTC) before the holdout was opened
RULES: list[dict[str, Any]] = [
    {"id": "R1", "name": "R1 back team after blowout loss (taker T-1h)", "hyp": "H3", "lead": 1, "exec": "taker",
     "buy": _after_blowout_loss(), "control": _after_narrow_loss()},
    {"id": "R1m", "name": "R1m back team after blowout loss (maker bid T-6h)", "hyp": "H3", "lead": 6, "exec": "maker",
     "buy": _after_blowout_loss(), "control": None},
    {"id": "R2", "name": "R2 fade popular favourite: buy the underdog (taker T-1h; falsification test)", "hyp": "H1", "lead": 1,
     "exec": "taker", "buy": pl.col("opp_pop") & ~pl.col("pop") & (pl.col("mid1") < 0.5),
     "control": ~pl.col("opp_pop") & ~pl.col("pop") & (pl.col("mid1") < 0.5)},
]


def rule_trades(p: pl.DataFrame, rule: dict[str, Any]) -> pl.DataFrame:
    rows = p.filter(rule["buy"])
    tr = taker_trades(rows, rule["lead"]) if rule["exec"] == "taker" else maker_trades(rows, rule["lead"], rule.get("improve", 0.0))
    return tr.with_columns(split_of(tr).alias("split"))


def run_holdout() -> pl.DataFrame:
    """Every pre-registered rule and its control on dev and holdout: per-trade stats, robustness,
    capped-book ledgers (``ledgers/``) and their vol_metrics."""
    if not RULES:
        raise RuntimeError("no pre-registered rules")
    p = pl.read_parquet(OUT / "panel.parquet")
    res, rob = [], []
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    for rule in RULES:
        for kind, sel in (("rule", rule["buy"]), ("control", rule.get("control"))):
            if sel is None:
                continue
            r = {**rule, "buy": sel}
            tr = rule_trades(p, r)
            name = rule["name"] if kind == "rule" else f"{rule['name']} [control]"
            if tr.is_empty():
                res.append({"rule": name, "kind": kind, "trades": 0})
                continue
            tr.write_parquet(OUT / f"trades_{rule['id']}_{kind}.parquet")
            rows, leds = evaluate(name, tr)
            res += [{**x, "kind": kind, "hyp": rule["hyp"]} for x in rows]
            if kind == "rule":
                rob += [{"rule": name, **x} for x in robustness(tr)]
                for led in leds:
                    led.write_parquet(OUT / "ledgers" / f"ledger_{rule['id']}_{led['split'][0]}.parquet")
    out = pl.DataFrame(res, infer_schema_length=None)
    out = out.with_columns(pl.col(c).map_elements(lambda v: f"[{v[0]}, {v[1]}]" if v is not None else "", return_dtype=pl.String)
                           for c in out.columns if out.schema[c] == pl.List(pl.Float64))
    out.write_csv(OUT / "holdout_results.csv")
    pl.DataFrame(rob, infer_schema_length=None).write_csv(OUT / "robustness.csv")
    return out


# ---- displayed size from the recorded L2 (Nov 5+) ----------------------------------------------------------
STREAM = Path("data/raw/stream/kalshi")


def l2_top_at_hours(wanted: dict[datetime, set[str]], max_gap_lines: int = 20000) -> pl.DataFrame:
    """Best YES bid / ask and displayed size for ``wanted`` tickers from the ``orderbook_snapshot``
    messages that open each recorded hourly file (hour start, UTC). Reads only the head of each file
    (stops after ``max_gap_lines`` lines without a snapshot), so memory stays small."""
    import io

    import zstandard

    rows = []
    for hour, tickers in sorted(wanted.items()):
        d = STREAM / hour.strftime("%Y-%m-%d")
        for f in sorted(d.glob(f"{hour:%H}-*.jsonl.zst")):
            with open(f, "rb") as fh:
                r = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
                gap = 0
                for line in r:
                    if "orderbook_snapshot" not in line:
                        gap += 1
                        if gap > max_gap_lines:
                            break
                        continue
                    gap = 0
                    frame = json.loads(json.loads(line)["frame"])
                    msg = frame.get("msg") or {}
                    tk = msg.get("market_ticker")
                    if tk not in tickers:
                        continue
                    yes = sorted(((float(a), float(b)) for a, b in msg.get("yes_dollars_fp") or []), reverse=True)
                    no = sorted(((float(a), float(b)) for a, b in msg.get("no_dollars_fp") or []), reverse=True)
                    rows.append({"hour": hour, "ticker": tk,
                                 "yes_bid": yes[0][0] if yes else None, "yes_bid_size": yes[0][1] if yes else None,
                                 "yes_ask": 1 - no[0][0] if no else None, "yes_ask_size": no[0][1] if no else None})
    return pl.DataFrame(rows, schema={"hour": pl.Datetime("us", "UTC"), "ticker": pl.String, "yes_bid": pl.Float64,
                                      "yes_bid_size": pl.Float64, "yes_ask": pl.Float64, "yes_ask_size": pl.Float64})


# ---- report helpers --------------------------------------------------------------------------------------
def md_table(df: pl.DataFrame, cols: list[str] | None = None, max_rows: int = 200) -> str:
    cols = [c for c in (cols or df.columns) if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.select(cols).head(max_rows).iter_rows():
        lines.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def sample_table(p: pl.DataFrame) -> pl.DataFrame:
    """Priced games per league and split (quote at T-1h, spread <= 5c)."""
    d = priced(p, 1).unique(["event_ticker"])
    return (d.with_columns(split_of(d).alias("split")).group_by("league", "split").agg(pl.len().alias("games"))
            .pivot(on="split", index="league", values="games").sort("league"))


def write_report(sections: list[tuple[str, str]]) -> Path:
    """``report.md`` from (heading, markdown body) pairs."""
    lines = ["# Behavioural mispricing in Kalshi sports", ""]
    for head, body in sections:
        lines += [f"## {head}", "", body.strip(), ""]
    path = OUT / "report.md"
    path.write_text("\n".join(lines))
    return path


# ---- CLI ---------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.behav_sports", description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["espn", "markets", "candles", "tennis", "panel", "explore", "holdout", "all"])
    ap.add_argument("--max-hist", type=int, default=None, help="cap on historical candle requests (candles step)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step in ("espn", "all"):
        fetch_espn()
    if a.step in ("markets", "all"):
        fetch_markets()
    if a.step in ("candles", "all"):
        fetch_candles(max_hist=a.max_hist)
    if a.step in ("tennis", "all"):
        fetch_tennis()
    if a.step in ("panel", "all"):
        g = game_table()
        g.write_parquet(OUT / "games.parquet")
        p = side_panel(g)
        p.write_parquet(OUT / "panel.parquet")
        if p.is_empty():
            raise RuntimeError("empty panel: investigate before reporting")
        print(p.group_by("league", "holdout").len().sort("league", "holdout"))
    if a.step in ("explore", "all"):
        out = run_explore()
        print(f"{out.height} dev cells -> {OUT / 'explore_dev.csv'}")
    if a.step in ("holdout", "all"):
        run_holdout()


if __name__ == "__main__":
    main()
