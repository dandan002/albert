"""Men's college basketball: adjusted-efficiency ratings vs Kalshi (``KXNCAAMBGAME`` moneylines,
``KXNCAAMBSPREAD`` / ``KXNCAAMBTOTAL`` / ``KXNCAAMBTEAMTOTAL`` ladders), season 2025-26.

Data
----
* **Results and box totals:** ESPN's public scoreboard JSON (``site.api.espn.com/.../
  mens-college-basketball/scoreboard?dates=YYYYMMDD&groups=50``), one request per day, seasons
  2023-24 .. 2026-27, cached zstd under ``data/cache/cbb/espn/`` (0.6 s between requests). Each
  game carries the scheduled tip (``date``), neutral site, conference ids, final scores, and team
  FGA / FTA / FGM / 3PA / rebounds (no turnovers or offensive rebounds; see possessions below).
* **Markets:** every settled leg of the four series from ``/historical/markets?series_ticker=``
  (games before Kalshi's 2026-09-06 historical cutoff) and ``/markets?series_ticker=`` (after),
  cached under ``data/cache/cbb/raw/kalshi`` (never ``data/raw``) and in
  ``data/cache/fundamental_cbb_markets.parquet``. Kalshi events are matched to ESPN games by
  Eastern date and team names.
* **Quotes and trades:** hourly candles over ``[tip - 27 h, tip - 1 h]``; historical markets
  need one request per leg and the public rate limit was shared with other research jobs, so
  legs are sampled at random with fixed seeds per conference tier x split (volume-blind; see
  :func:`candle_legs`): every low-major moneyline, 120 mid- and 120 high-major moneylines per
  split, and 2 mid-priced legs (nearest the ladder's median strike, from listing metadata) of
  each spread and total ladder of 120 low / 30 mid / 30 high games per split. Trades for the
  maker test (every public print from the first decision time to tip) for a further subsample
  (all low-major legs; 60 moneyline + 15 ladder games per split for mid / high).
* **Tiers:** a game is low-major when both teams play in a low-major conference (see ``HIGH`` /
  ``MID``), mid-major when the better-known team is mid-major, high-major otherwise.

Possessions
-----------
The scoreboard has no turnovers or offensive rebounds, so possessions are estimated as
``FGA + 0.475·FTA + c`` per team (``c`` = league mean of TO − OREB, ≈ 1.6, taken from public
D-I averages and only a constant: efficiencies are relative). Game possessions = mean of the two
teams' estimates. This is the usual "simple possessions" estimate with a constant correction.

Models (Kalman / ridge state-space, causal, refit after every result)
---------------------------------------------------------------------
* ``eff``: KenPom-style tempo-free model. Per game: offensive efficiency of each team
  ``100·pts/poss = μ + o_i − d_j ± h/2 + ε`` (noise variance ∝ 1/poss) and possessions
  ``poss = π + τ_i + τ_j + ε_p``. Predicted margin = E[poss]·(E[eff_h] − E[eff_a])/100, total
  likewise; predictive SD from state covariance + residual noise, scaled on dev.
* ``pts``: offense / defense per team in points (``ncaaf.run_kf``-style), no tempo.
* ``massey``: one strength per team, margin only.
Preseason prior = last season's state regressed toward the conference-tier mean (``rho_off``),
new teams start at their tier mean. Hyperparameters are fitted on 2024-25 games and Nov 2025 –
Jan 2026 (dev) only (2023-24 is burn-in); see ``MODELS``. Contract probabilities come from the
predictive mean / SD and a residual law (normal or Student t) fitted on dev residuals
(``ncaaf.fit_dist`` / ``ncaaf.leg_probabilities``).

Evaluation
----------
Harness ``fundamental.evaluate`` per lead x contract kind x tier (blend
``logit P = a + b·logit(mid) + c·logit(model)`` fitted on dev, c's t clustered by game; taker at
the ask / bid with the series fee, θ chosen on dev). Fair value for the maker test = the dev
blend per kind x lead x tier (pooled kind x lead when a cell has < 150 dev rows). Maker fills
(:func:`maker_fills_all`) under the harness trade-through rule and a lenient trade-at bound.
Fees: today's series schedules (``KXNCAAMBGAME`` / ``SPREAD`` / ``TOTAL`` carry maker fees);
last season's may have differed. Run: ``python -c "from albert.research import cbb; cbb.main()"``.

As-of discipline (no look-ahead)
--------------------------------
Decision times are 24 h and 2 h before ESPN's scheduled tip. A result enters the model at
``tip + 2.75 h`` (a college game lasts about 2 h; later when the Kalshi moneyline closed later),
strictly after any query at the same instant. Tip time, venue, neutral site and conference are
scheduled facts. Dev = games tipping Nov 2025 – Jan 2026; holdout = Feb – Apr 2026, scored once
with the frozen model, law and blend.
"""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.config import Settings

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
SERIES = ("KXNCAAMBGAME", "KXNCAAMBSPREAD", "KXNCAAMBTOTAL", "KXNCAAMBTEAMTOTAL")
KINDS = {"KXNCAAMBGAME": "ml", "KXNCAAMBSPREAD": "spread", "KXNCAAMBTOTAL": "total",
         "KXNCAAMBTEAMTOTAL": "team_total"}
GAME_LENGTH = timedelta(hours=2, minutes=45)   # tip -> result public (conservative)
SEASONS = {  # season label (ESPN year) -> first and last scoreboard date (US Eastern)
    2024: (date(2023, 11, 1), date(2024, 4, 10)),
    2025: (date(2024, 11, 1), date(2025, 4, 10)),
    2026: (date(2025, 11, 1), date(2026, 4, 10)),
    2027: (date(2026, 11, 1), date(2026, 11, 5)),
}
DEV_FROM = datetime(2025, 11, 1, tzinfo=UTC)
HOLDOUT_FROM = datetime(2026, 2, 1, 5, tzinfo=UTC)   # Feb 1 2026 00:00 US Eastern
SEASON_END = datetime(2026, 4, 15, tzinfo=UTC)
LEADS_H = (24, 2)

# Conference tiers by ESPN conference short name (competition.groups.shortName). High-major =
# the power conferences (Pac-12 only while it had 12 teams, 2023-24); mid-major = the eight next
# strongest leagues by KenPom conference rank over 2023-25; low-major = all others.
HIGH = {"ACC", "Big 12", "Big East", "Big Ten", "SEC"}
HIGH_2024_ONLY = {"Pac-12"}
MID = {"A-10", "American", "Mountain West", "WCC", "MVC", "CUSA", "MAC", "Sun Belt"}


class CbbSettings(Settings):
    """Settings whose raw HTTP cache lives under ``data/cache/cbb/raw`` (never ``data/raw``)."""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "cache" / "cbb" / "raw"


def cbb_settings(s: Settings | None = None) -> CbbSettings:
    return CbbSettings(data_dir=(s.data_dir if s is not None else Settings().data_dir))


# ---- ESPN ---------------------------------------------------------------------------------------
_last_request = [0.0]


def espn_get(s, path: str, params: dict[str, Any], *, min_interval: float = 0.6, tag: str | None = None) -> dict[str, Any]:
    """GET an ESPN site-API JSON once; later calls read the zstd cache under data/cache/cbb/espn.
    ``tag`` (e.g. an hour stamp) is mixed into the cache key only, so a live day can be
    re-read later without overwriting the earlier file."""
    import httpx

    from albert.http import read_raw, write_raw

    cache_dir = s.data_dir / "cache" / "cbb" / "espn"
    keyed = params if tag is None else {**params, "__tag": tag}
    key = hashlib.sha256((path + "?" + json.dumps(keyed, sort_keys=True)).encode()).hexdigest()[:24]
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
    file = cache_dir / f"{slug}-{key}.json.zst"
    if file.exists():
        return read_raw(file)["response"]
    r = None
    for attempt in range(5):
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


STAT_KEYS = {"fieldGoalsAttempted": "fga", "fieldGoalsMade": "fgm", "freeThrowsAttempted": "fta",
             "freeThrowsMade": "ftm", "threePointFieldGoalsAttempted": "fg3a", "rebounds": "reb"}

GAME_SCHEMA: dict[str, Any] = {
    "game_id": pl.String, "season": pl.Int64, "season_type": pl.Int64, "tip": pl.Datetime("us", "UTC"),
    "time_valid": pl.Boolean, "completed": pl.Boolean, "status": pl.String, "neutral": pl.Boolean,
    "conf_game": pl.Boolean, "group_id": pl.String, "group_name": pl.String, "periods": pl.Int64,
    "note": pl.String,
}
for _side in ("home", "away"):
    GAME_SCHEMA.update({f"{_side}_id": pl.String, f"{_side}_abbr": pl.String, f"{_side}_loc": pl.String,
                        f"{_side}_name": pl.String, f"{_side}_short": pl.String, f"{_side}_conf": pl.String,
                        f"{_side}_score": pl.Float64})
    GAME_SCHEMA.update({f"{_side}_{v}": pl.Float64 for v in STAT_KEYS.values()})


def game_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per ESPN scoreboard event (home/away by ``homeAway``) with box totals."""
    rows = []
    for e in payload.get("events") or []:
        comp = (e.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if set(teams) != {"home", "away"}:
            continue
        status = (comp.get("status") or e.get("status") or {})
        stype = status.get("type") or {}
        grp = comp.get("groups") or {}
        notes = comp.get("notes") or []
        row = {"game_id": str(e.get("id")), "season": (e.get("season") or {}).get("year"),
               "season_type": (e.get("season") or {}).get("type"), "tip": _ts(comp.get("date") or e.get("date")),
               "time_valid": comp.get("timeValid"), "completed": bool(stype.get("completed")),
               "status": stype.get("name"), "neutral": bool(comp.get("neutralSite")),
               "conf_game": bool(comp.get("conferenceCompetition")), "group_id": grp.get("id"),
               "group_name": grp.get("shortName"), "periods": status.get("period"),
               "note": notes[0].get("headline") if notes else None}
        for side in ("home", "away"):
            c = teams[side]
            t = c.get("team") or {}
            score = c.get("score")
            row.update({f"{side}_id": str(t.get("id")), f"{side}_abbr": t.get("abbreviation"),
                        f"{side}_loc": t.get("location"), f"{side}_name": t.get("displayName"),
                        f"{side}_short": t.get("shortDisplayName"), f"{side}_conf": t.get("conferenceId"),
                        f"{side}_score": float(score) if score not in (None, "") else None})
            stats = {st.get("name"): st.get("displayValue") for st in c.get("statistics") or []}
            for name, col in STAT_KEYS.items():
                v = stats.get(name)
                try:
                    row[f"{side}_{col}"] = float(v) if v not in (None, "", "-") else None
                except ValueError:
                    row[f"{side}_{col}"] = None
        rows.append(row)
    return rows


def fetch_games(s, seasons: dict[int, tuple[date, date]] = SEASONS, *, until: date | None = None) -> pl.DataFrame:
    """Every game in ESPN's Division I feed (``groups=50``) on every date of these seasons."""
    rows: list[dict[str, Any]] = []
    today = until or datetime.now(UTC).date()
    for _season, (d0, d1) in seasons.items():
        d = d0
        while d <= min(d1, today):
            payload = espn_get(s, "scoreboard", {"dates": d.strftime("%Y%m%d"), "groups": "50", "limit": "500"})
            rows.extend(game_rows(payload))
            d += timedelta(days=1)
    g = pl.DataFrame(rows, schema=GAME_SCHEMA)
    return g.unique("game_id", keep="last").sort("tip", "game_id")


def conference_names(games: pl.DataFrame) -> dict[str, str]:
    """ESPN conference id -> short name, read off conference games (competition.groups)."""
    cg = games.filter(pl.col("conf_game") & pl.col("group_id").is_not_null() & (pl.col("home_conf") == pl.col("group_id")))
    return dict(cg.group_by("group_id").agg(pl.col("group_name").mode().first()).iter_rows())


def team_tiers(games: pl.DataFrame, *, min_games: int = 10) -> dict[tuple[int, str], str]:
    """(season, team id) -> "high" / "mid" / "low" for Division I teams (a member of a conference
    seen in D-I conference games, or at least ``min_games`` games in the D-I feed that season),
    "nond1" otherwise. Tier from the team's conference."""
    names = conference_names(games)
    out: dict[tuple[int, str], str] = {}
    counts: dict[tuple[int, str], int] = {}
    conf: dict[tuple[int, str], str | None] = {}
    for r in games.select("season", "home_id", "away_id", "home_conf", "away_conf").iter_rows(named=True):
        for side in ("home", "away"):
            key = (r["season"], r[f"{side}_id"])
            counts[key] = counts.get(key, 0) + 1
            conf[key] = r[f"{side}_conf"] or conf.get(key)
    for key, n in counts.items():
        name = names.get(conf.get(key) or "", "")
        # Division I: a known D-I conference, or enough games in the D-I feed (independents).
        if n < min_games and not name:
            out[key] = "nond1"
            continue
        if name in HIGH or (key[0] == 2024 and name in HIGH_2024_ONLY):
            out[key] = "high"
        elif name in MID:
            out[key] = "mid"
        else:
            out[key] = "low"
    return out


def game_tier(games: pl.DataFrame, tiers: dict[tuple[int, str], str]) -> pl.DataFrame:
    """Game tier = the higher-profile team's tier (market attention follows the bigger name):
    "high" if either team is high-major, else "mid" if either is mid-major, else "low"."""
    rank = {"high": 3, "mid": 2, "low": 1, "nond1": 0}
    rows = []
    for r in games.select("game_id", "season", "home_id", "away_id").iter_rows(named=True):
        a, b = tiers.get((r["season"], r["home_id"]), "nond1"), tiers.get((r["season"], r["away_id"]), "nond1")
        top = max(a, b, key=lambda x: rank[x])
        rows.append({"game_id": r["game_id"], "tier": "low" if top == "nond1" else top,
                     "tier_min": min(a, b, key=lambda x: rank[x])})
    return pl.DataFrame(rows)


# ---- Kalshi markets -----------------------------------------------------------------------------
def _short_sleep(d: float) -> None:
    """Throttle waits as asked; 429 / 5xx back-offs capped at a jittered 0.25-0.6 s. Kalshi's
    public rate limit is shared by several research jobs on this machine, and the default
    doubling back-off (1, 2, 4, 8 s) starves a polite 2-thread client to ~0.6 requests/s."""
    import random

    time.sleep(d if d <= 0.2 else random.uniform(0.25, 0.6))


def _api(s, *, min_interval: float = 0.25):
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    cs = cbb_settings(s)
    return k.Kalshi(CachedClient("kalshi", cs.kalshi_base, cs.raw_dir, min_interval=min_interval,
                                 max_retries=40, sleep=_short_sleep))


def series_info(s) -> dict[str, dict[str, Any]]:
    from albert.venues import kalshi as k

    df = k.series_frame(_api(s).series().payload).filter(pl.col("series_ticker").is_in(list(SERIES)))
    return {r["series_ticker"]: r for r in df.iter_rows(named=True)}


def kalshi_markets(s, *, refresh: bool = False, oldest: datetime = datetime(2025, 10, 25, tzinfo=UTC)) -> pl.DataFrame:
    """Every settled leg of the four series closing after ``oldest``: historical pages (newest
    first, paged until past ``oldest``) plus the live endpoint for markets settled after the
    cutoff. Cached in ``data/cache/fundamental_cbb_markets.parquet``."""
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / "fundamental_cbb_markets.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    api = _api(s)
    info = series_info(s)
    rows: list[dict[str, Any]] = []
    for series in SERIES:
        n0 = len(rows)
        for page in api._cursor_pages("/historical/markets", {"series_ticker": series, "limit": 1000}, "markets"):
            got = k.market_rows(page.payload, info, "cbb-historical")
            rows.extend(got)
            closes = [r["close_time"] for r in got if r["close_time"] is not None]
            if closes and min(closes) < oldest:
                break
        for page in api.series_markets(series, "settled", min_close=oldest, cache=False):
            rows.extend(k.market_rows(page.payload, info, "cbb-live"))
        print(f"  {series}: {len(rows) - n0} markets", flush=True)
    m = pl.DataFrame(rows, schema=k.MARKET_SCHEMA).unique("ticker", keep="last")
    m = (m.filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("close_time") >= oldest))
         .with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y")))
    tmp = path.with_suffix(".tmp")
    m.write_parquet(tmp)
    tmp.replace(path)
    return m


# ---- Kalshi <-> ESPN matching -------------------------------------------------------------------
def _norm(x: str | None) -> str:
    from albert.research.ncaaf import norm_name

    return norm_name(x)


def match_events(game_legs: pl.DataFrame, games: pl.DataFrame, *, min_score: float = 0.72) -> pl.DataFrame:
    """Map each ``KXNCAAMBGAME`` event (two legs: ticker team code + ``yes_sub_title``) to the
    ESPN game on the same US-Eastern date (±1 day) whose two teams best match both legs. Exact
    normalised-name hits are tried first; otherwise fuzzy scores (``ncaaf._name_score``)."""
    from albert.research.ncaaf import _name_score, event_key

    games = games.with_columns(pl.col("tip").dt.convert_time_zone("America/New_York").dt.date().alias("et_date"))
    by_date: dict[Any, list[dict[str, Any]]] = {}
    exact: dict[tuple[Any, str], list[tuple[dict[str, Any], str]]] = {}
    for r in games.iter_rows(named=True):
        by_date.setdefault(r["et_date"], []).append(r)
        for side in ("home", "away"):
            for v in {_norm(r[f"{side}_loc"]), _norm(r[f"{side}_short"]), _norm(r[f"{side}_name"])}:
                if v:
                    exact.setdefault((r["et_date"], v), []).append((r, side))
    out = []
    for ev, legs in game_legs.group_by("event_ticker"):
        legs = legs.sort("ticker").to_dicts()
        key = event_key(ev[0])
        if key is None or len(legs) != 2:
            continue
        day = key[0].date()
        codes = [leg["ticker"].rsplit("-", 1)[1] for leg in legs]
        best = None
        # 1) exact names, same game, opposite sides
        for d in (day, day + timedelta(days=1), day - timedelta(days=1)):
            h0 = exact.get((d, _norm(legs[0]["yes_sub_title"])), [])
            h1 = exact.get((d, _norm(legs[1]["yes_sub_title"])), [])
            pairs = [(g0, s0, s1) for g0, s0 in h0 for g1, s1 in h1 if g0["game_id"] == g1["game_id"] and s0 != s1]
            if len(pairs) == 1:
                g, s0, s1 = pairs[0]
                best = (2.0, g, (s0, s1), [1.0, 1.0])
                break
        if best is None:
            cands = [g for d in (day, day + timedelta(days=1), day - timedelta(days=1)) for g in by_date.get(d, [])]
            for g in cands:
                s_ha = [_name_score(legs[0]["yes_sub_title"], codes[0], g, "home"),
                        _name_score(legs[1]["yes_sub_title"], codes[1], g, "away")]
                if max(s_ha) < 0.5:
                    s_ah = [_name_score(legs[0]["yes_sub_title"], codes[0], g, "away"),
                            _name_score(legs[1]["yes_sub_title"], codes[1], g, "home")]
                    if max(s_ah) < 0.5:
                        continue
                else:
                    s_ah = [_name_score(legs[0]["yes_sub_title"], codes[0], g, "away"),
                            _name_score(legs[1]["yes_sub_title"], codes[1], g, "home")]
                for sides, sc in ((("home", "away"), s_ha), (("away", "home"), s_ah)):
                    tot = min(sc) + 0.01 * sum(sc) - (0.05 if g["et_date"] != day else 0.0)
                    if best is None or tot > best[0]:
                        best = (tot, g, sides, sc)
        base = {"event_key": key[1] + key[0].strftime("%y%m%d"), "kalshi_event": ev[0],
                "leg0": legs[0]["ticker"], "leg0_name": legs[0]["yes_sub_title"],
                "leg1": legs[1]["ticker"], "leg1_name": legs[1]["yes_sub_title"]}
        if best is None or min(best[3]) < min_score:
            out.append({**base, "game_id": None, "score": None if best is None else round(min(best[3]), 3),
                        "espn": None if best is None else f"{best[1]['away_name']} @ {best[1]['home_name']}"})
            continue
        _, g, sides, sc = best
        out.append({**base, "game_id": g["game_id"], "score": round(min(sc), 3), "leg0_side": sides[0],
                    "leg1_side": sides[1], "espn": f"{g['away_name']} @ {g['home_name']}"})
    return pl.DataFrame(out, infer_schema_length=None)


_STRIKE_RX = re.compile(r"over (\d+(?:\.\d+)?) points", re.I)


def ladder_legs(markets: pl.DataFrame, match: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Every settled leg joined to its ESPN game, with what it pays on:

    * ``kind = "ml"``: team ``team_side`` wins (KXNCAAMBGAME);
    * ``"spread"``: team ``team_side`` wins by more than ``strike`` (KXNCAAMBSPREAD);
    * ``"total"``: combined points > ``strike`` (KXNCAAMBTOTAL);
    * ``"team_total"``: team ``team_side`` scores > ``strike`` (KXNCAAMBTEAMTOTAL).

    The strike is read from the title or ``yes_sub_title`` ("... over 20.5 Points"). ``y_espn``
    recomputes the result from ESPN's final score as a parsing check (it must agree with ``y``)."""
    from albert.research.ncaaf import event_key

    mt = match.filter(pl.col("game_id").is_not_null())
    code_side: dict[tuple[str, str], str] = {}
    for r in mt.iter_rows(named=True):
        for leg in ("leg0", "leg1"):
            code_side[(r["event_key"], r[leg].rsplit("-", 1)[1])] = r[f"{leg}_side"]
    game_of = dict(zip(mt["event_key"], mt["game_id"], strict=True))
    rows = []
    for r in markets.filter(pl.col("series_ticker").is_in(list(KINDS))).iter_rows(named=True):
        key = event_key(r["event_ticker"])
        if key is None:
            continue
        eks = [key[1] + (key[0] + timedelta(days=d)).strftime("%y%m%d") for d in (0, -1, 1)]
        ek = next((e for e in eks if e in game_of), None)
        if ek is None:
            continue
        kind = KINDS[r["series_ticker"]]
        suffix = r["ticker"].rsplit("-", 1)[1]
        side, strike = None, None
        if kind != "ml":
            sm = _STRIKE_RX.search(r["yes_sub_title"] or "") or _STRIKE_RX.search(r["title"] or "")
            if sm is None:
                continue
            strike = float(sm.group(1))
        if kind in ("ml", "spread", "team_total"):
            code = suffix if kind == "ml" else re.sub(r"\d+$", "", suffix)
            side = code_side.get((ek, code))
            if side is None:
                continue
        rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "series": r["series_ticker"],
                     "event_key": ek, "game_id": game_of[ek], "kind": kind, "team_side": side, "strike": strike,
                     "y": int(r["y"]), "open_time": r["open_time"], "close_time": r["close_time"],
                     "fee_type": r["fee_type"], "fee_multiplier": r["fee_multiplier"], "volume": r["volume"]})
    legs = pl.DataFrame(rows, infer_schema_length=None)
    g = games.select("game_id", "tip", "home_score", "away_score", "status")
    legs = legs.join(g, on="game_id", how="left")
    sign = pl.when(pl.col("team_side") == "home").then(1.0).otherwise(-1.0)
    margin = sign * (pl.col("home_score") - pl.col("away_score"))
    team_pts = pl.when(pl.col("team_side") == "home").then(pl.col("home_score")).otherwise(pl.col("away_score"))
    y_espn = (pl.when(pl.col("kind") == "ml").then(margin > 0)
              .when(pl.col("kind") == "spread").then(margin > pl.col("strike"))
              .when(pl.col("kind") == "total").then(pl.col("home_score") + pl.col("away_score") > pl.col("strike"))
              .otherwise(team_pts > pl.col("strike")))
    return legs.with_columns(y_espn.cast(pl.Int8).alias("y_espn"))


# ---- possessions and model inputs ---------------------------------------------------------------
TO_MINUS_OREB = 1.6   # D-I mean turnovers minus offensive rebounds per team-game (public averages)


def model_games(games: pl.DataFrame, tiers: dict[tuple[int, str], str]) -> pl.DataFrame:
    """Completed Division I vs Division I games with estimated possessions.

    ``poss`` = mean over both teams of FGA + 0.475·FTA + TO_MINUS_OREB (full game, overtime
    included: efficiencies use it); ``poss40`` = ``poss`` scaled to 40 minutes (tempo)."""
    g = games.filter(pl.col("completed") & pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
                     & (pl.col("status") == "STATUS_FINAL"))
    d1 = [tiers.get((r["season"], r["home_id"]), "nond1") != "nond1" and tiers.get((r["season"], r["away_id"]), "nond1") != "nond1"
          for r in g.select("season", "home_id", "away_id").iter_rows(named=True)]
    g = g.filter(pl.Series(d1, dtype=pl.Boolean))
    pe = ((pl.col("home_fga") + 0.475 * pl.col("home_fta") + pl.col("away_fga") + 0.475 * pl.col("away_fta")) / 2
          + TO_MINUS_OREB)
    minutes = 40 + 5 * (pl.col("periods").fill_null(2).clip(2, 9) - 2)
    # Seven games lack box totals: league-mean possessions (their efficiencies are still used).
    pe = pl.coalesce(pe, 69.0 * minutes / 40)
    return g.with_columns(pe.alias("poss")).with_columns((pl.col("poss") * 40 / minutes).alias("poss40"))


# ---- rating models (causal Kalman filter / ridge state-space) ----------------------------------
@dataclass(frozen=True)
class EffParams:
    """Hyperparameters. ``mode``: "eff" (offense/defense efficiency per 100 possessions + tempo),
    "pts" (offense/defense in points per game) or "massey" (one strength, margin only).

    Units: efficiency points per 100 possessions ("eff"), points ("pts", "massey"); tempo in
    possessions per 40 minutes."""
    mode: str = "eff"
    sigma: float = 11.0          # SD of one team's efficiency (per 70 possessions) / points / margin
    rho_e: float = 0.0           # correlation of the two teams' errors (eff, pts)
    sigma_p: float = 5.0         # SD of game possessions around the tempo prediction (eff)
    q_day: float = 0.02          # random-walk variance per team state per day in season
    q_day_tempo: float = 0.005   # same for tempo states
    rho_off: float = 0.6         # offseason regression of each state towards its tier mean
    q_off: float = 20.0          # extra offseason variance per (o, d) state
    q_off_tempo: float = 2.0     # extra offseason variance per tempo state
    p0: float = 100.0            # prior variance of a new team's rating (split over o and d)
    p0_tempo: float = 9.0
    home: float = 4.5            # initial home edge (eff: per 100 poss; pts/massey: points)
    mu: float = 104.0            # initial league mean efficiency (eff) / points per team (pts)
    pace: float = 68.0           # initial league mean possessions per 40 min (eff)
    clip: float | None = None    # robust update: clip standardised innovations at ±clip
    tier_init: tuple[tuple[str, float], ...] = (("high", 14.0), ("mid", 4.0), ("low", -6.0))


def _event_stream(games: pl.DataFrame, queries: pl.DataFrame) -> list[tuple]:
    """Results become visible at tip + GAME_LENGTH (or ``result_at`` if later); queries are
    answered at their ``t``. At equal times the query is answered first."""
    ev: list[tuple] = []
    has_ra = "result_at" in games.columns
    for r in games.iter_rows(named=True):
        seen = r["tip"] + GAME_LENGTH
        if has_ra and r["result_at"] is not None:
            seen = max(seen, r["result_at"])
        ev.append((seen, 1, "result", r))
    for r in queries.iter_rows(named=True):
        ev.append((r["t"], 0, "query", r))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def run_model(games: pl.DataFrame, queries: pl.DataFrame, prm: EffParams,
              tiers: dict[tuple[int, str], str]) -> pl.DataFrame:
    """One causal pass. ``games`` from :func:`model_games`; ``queries``: game_id, t, season,
    home_id, away_id, neutral. Results visible before a query are absorbed in one joint Kalman
    update (identical to sequential updates). Returns per query the predictive mean / SD of the
    home and away score, margin (home − away) and total, plus the expected possessions."""
    ids = sorted(set(games["home_id"]) | set(games["away_id"]) | set(queries["home_id"]) | set(queries["away_id"]))
    idx = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    nb = {"eff": 3, "pts": 2, "massey": 1}[prm.mode]
    G = nb * n
    MU, H, PI = G, G + 1, G + 2
    N = G + 3
    x = np.zeros(N)
    P = np.zeros((N, N))
    x[H], P[H, H] = prm.home, 1.0
    if prm.mode in ("eff", "pts"):
        x[MU], P[MU, MU] = prm.mu, 4.0
    if prm.mode == "eff":
        x[PI], P[PI, PI] = prm.pace, 4.0
    seen = np.zeros(n, dtype=bool)
    tier_of: dict[str, str] = {}
    init = dict(prm.tier_init)
    blocks_rating = [0, 1] if prm.mode in ("eff", "pts") else [0]

    def tier_mean(tier: str) -> np.ndarray:
        members = np.array([idx[t] for t, tt in tier_of.items() if tt == tier and seen[idx[t]]], dtype=int)
        out = np.zeros(nb)
        if members.size >= 5:
            for b in range(nb):
                out[b] = x[b * n + members].mean()
        else:
            r = init.get(tier, init["low"])
            if prm.mode in ("eff", "pts"):
                out[0] = out[1] = r / 2
            else:
                out[0] = r
        return out

    def p_init(b: int) -> float:
        if prm.mode == "eff" and b == 2:
            return prm.p0_tempo
        return prm.p0 / len(blocks_rating)

    def ensure(team: str, ssn: int) -> None:
        i = idx[team]
        tier_of[team] = tiers.get((ssn, team), tier_of.get(team, "low"))
        if tier_of[team] == "nond1":
            tier_of[team] = "low"
        if not seen[i]:
            m = tier_mean(tier_of[team])
            for b in range(nb):
                x[b * n + i] = m[b]
                P[b * n + i, b * n + i] = p_init(b)
            seen[i] = True

    def offseason() -> None:
        nonlocal P
        means = {tier: tier_mean(tier) for tier in set(tier_of.values())}
        A = np.ones(N)
        for team, tier in tier_of.items():
            i = idx[team]
            if not seen[i]:
                continue
            for b in range(nb):
                j = b * n + i
                x[j] = means[tier][b] + prm.rho_off * (x[j] - means[tier][b])
                A[j] = prm.rho_off
        P = P * A[:, None] * A[None, :]
        for b in range(nb):
            q = prm.q_off_tempo if (prm.mode == "eff" and b == 2) else prm.q_off / len(blocks_rating)
            d = b * n + np.arange(n)
            P[d, d] += np.where(seen, q, 0.0)
        P[H, H] += 0.25
        if prm.mode in ("eff", "pts"):
            P[MU, MU] += 1.0
        if prm.mode == "eff":
            P[PI, PI] += 1.0

    def design(r: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        h, a = idx[r["home_id"]], idx[r["away_id"]]
        hm = 0.0 if r["neutral"] else 1.0
        if prm.mode == "eff":  # rows: eff home, eff away, possessions
            cols = np.array([h, a, n + h, n + a, 2 * n + h, 2 * n + a, MU, H, PI])
            Hc = np.array([[1.0, 0, 0, -1, 0, 0, 1, hm / 2, 0],
                           [0, 1.0, -1, 0, 0, 0, 1, -hm / 2, 0],
                           [0, 0, 0, 0, 1.0, 1.0, 0, 0, 1.0]])
        elif prm.mode == "pts":  # rows: home points, away points
            cols = np.array([h, a, n + h, n + a, MU, H])
            Hc = np.array([[1.0, 0, 0, -1, 1, hm / 2], [0, 1.0, -1, 0, 1, -hm / 2]])
        else:
            cols = np.array([h, a, H])
            Hc = np.array([[1.0, -1.0, hm]])
        return cols, Hc

    def noise(poss: float | None) -> np.ndarray:
        s2 = prm.sigma ** 2
        if prm.mode == "eff":
            k = 70.0 / max(poss or 70.0, 40.0)
            return np.array([[s2 * k, prm.rho_e * s2 * k, 0], [prm.rho_e * s2 * k, s2 * k, 0], [0, 0, prm.sigma_p ** 2]])
        if prm.mode == "pts":
            return s2 * np.array([[1.0, prm.rho_e], [prm.rho_e, 1.0]])
        return np.array([[s2]])

    def observe(r: dict[str, Any]) -> np.ndarray:
        if prm.mode == "eff":
            return np.array([100 * r["home_score"] / r["poss"], 100 * r["away_score"] / r["poss"], r["poss40"]])
        if prm.mode == "pts":
            return np.array([r["home_score"], r["away_score"]])
        return np.array([r["home_score"] - r["away_score"]])

    pending: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal P
        if not pending:
            return
        mats = [design(r) for r in pending]
        cols = np.unique(np.concatenate([c for c, _ in mats]))
        pos = {c: j for j, c in enumerate(cols)}
        k = mats[0][1].shape[0]
        m = len(pending)
        Hb = np.zeros((k * m, cols.size))
        R = np.zeros((k * m, k * m))
        y = np.zeros(k * m)
        for g, (r, (c, Hc)) in enumerate(zip(pending, mats, strict=True)):
            sl = slice(g * k, (g + 1) * k)
            for j, cc in enumerate(c):
                Hb[sl, pos[cc]] += Hc[:, j]
            R[sl, sl] = noise(r.get("poss"))
            y[sl] = observe(r)
        Pc = P[np.ix_(cols, cols)]
        S = Hb @ Pc @ Hb.T + R
        v = y - Hb @ x[cols]
        if prm.clip is not None:
            sd = np.sqrt(np.diag(S))
            v = np.clip(v, -prm.clip * sd, prm.clip * sd)
        PH = P[:, cols] @ Hb.T
        K = np.linalg.solve(S, PH.T).T
        x[:] += K @ v
        P -= K @ PH.T
        P = (P + P.T) / 2 if m > 50 else P
        pending.clear()

    team_diag_blocks = [(b * n + np.arange(n), prm.q_day_tempo if (prm.mode == "eff" and b == 2) else prm.q_day / len(blocks_rating))
                        for b in range(nb)]
    season = None
    last_t = None
    out: list[dict[str, Any]] = []
    for t, _, kind, r in _event_stream(games, queries):
        if season is not None and r["season"] != season:
            flush()
            offseason()
        season = r["season"]
        if kind == "query":
            flush()
        if last_t is not None and (prm.q_day > 0 or prm.q_day_tempo > 0):
            days = (t - last_t).total_seconds() / 86400
            if 0 < days < 60:
                for d, q in team_diag_blocks:
                    P[d, d] += np.where(seen, q * days, 0.0)
        last_t = t
        ensure(r["home_id"], season)
        ensure(r["away_id"], season)
        if kind == "result":
            pending.append(r)
            continue
        cols, Hc = design(r)
        mean = Hc @ x[cols]
        C = Hc @ P[np.ix_(cols, cols)] @ Hc.T
        rec: dict[str, Any] = {"game_id": r["game_id"], "t": r["t"]}
        if prm.mode == "eff":
            eh, ea, pp = mean
            S = C + noise(pp)
            grads = {"home": np.array([pp, 0, eh]) / 100, "away": np.array([0, pp, ea]) / 100,
                     "margin": np.array([pp, -pp, eh - ea]) / 100, "total": np.array([pp, pp, eh + ea]) / 100}
            means = {"home": pp * eh / 100, "away": pp * ea / 100}
            means["margin"], means["total"] = means["home"] - means["away"], means["home"] + means["away"]
            for key, gvec in grads.items():
                rec[f"mu_{key}"] = means[key]
                rec[f"sd_{key}"] = math.sqrt(gvec @ S @ gvec)
            rec["var_state_margin"] = float(grads["margin"] @ C @ grads["margin"])
            rec["poss_exp"] = pp
            rec["eff_home"], rec["eff_away"] = eh, ea
        elif prm.mode == "pts":
            S = C + noise(None)
            rec.update({"mu_home": mean[0], "mu_away": mean[1], "sd_home": math.sqrt(S[0, 0]),
                        "sd_away": math.sqrt(S[1, 1]), "mu_margin": mean[0] - mean[1],
                        "sd_margin": math.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1]), "mu_total": mean[0] + mean[1],
                        "sd_total": math.sqrt(S[0, 0] + S[1, 1] + 2 * S[0, 1]),
                        "var_state_margin": C[0, 0] + C[1, 1] - 2 * C[0, 1]})
        else:
            rec.update({"mu_margin": mean[0], "sd_margin": math.sqrt(C[0, 0] + prm.sigma ** 2),
                        "var_state_margin": C[0, 0]})
        out.append(rec)
    return pl.DataFrame(out, infer_schema_length=None)


def queries_for(games: pl.DataFrame, leads: tuple[int, ...]) -> pl.DataFrame:
    g = games.select("game_id", "season", "home_id", "away_id", "neutral", "tip")
    return pl.concat([g.with_columns((pl.col("tip") - pl.duration(hours=h)).alias("t"), pl.lit(h).alias("lead"))
                      for h in leads]).drop("tip")


def gaussian_ll(pred: pl.DataFrame, games: pl.DataFrame, *, frm: datetime, to: datetime) -> dict[str, float]:
    """Mean Gaussian predictive log likelihood of margin (and total) for games tipping in
    [frm, to), plus RMSEs; the hyperparameter objective (dev windows only)."""
    d = (pred.join(games.select("game_id", "tip", "home_score", "away_score"), on="game_id")
         .filter((pl.col("tip") >= frm) & (pl.col("tip") < to)))
    em = (d["home_score"] - d["away_score"] - d["mu_margin"]).to_numpy()
    sm = d["sd_margin"].to_numpy()
    out = {"n": d.height, "ll_margin": float((-0.5 * (em / sm) ** 2 - np.log(sm) - 0.9189385).mean()),
           "rmse_margin": float(np.sqrt((em ** 2).mean()))}
    if "mu_total" in d.columns:
        et = (d["home_score"] + d["away_score"] - d["mu_total"]).to_numpy()
        st = d["sd_total"].to_numpy()
        out.update({"ll_total": float((-0.5 * (et / st) ** 2 - np.log(st) - 0.9189385).mean()),
                    "rmse_total": float(np.sqrt((et ** 2).mean()))})
    return out


# ---- quotes and trades --------------------------------------------------------------------------
QUOTE_WINDOW = (timedelta(hours=27), timedelta(hours=1))   # candles over [tip - 27 h, tip - 1 h]


def sample_ladder_legs(legs: pl.DataFrame, tier: pl.DataFrame, *, per_cell: int = 250, per_ladder: int = 2,
                       seed: int = 20251101) -> pl.DataFrame:
    """Volume-blind sample of spread / total legs for candle requests (historical candles cost
    one request per leg). Games are drawn at random (fixed seed) per tier x split (dev / holdout),
    ``per_cell`` each; from each sampled game's spread ladder and total ladder the ``per_ladder``
    legs open by tip − 2 h nearest the ladder's median strike are kept (the ladder is centred on
    the listing line, so these are the mid-priced legs; chosen from listing metadata only)."""
    lad = legs.filter(pl.col("kind").is_in(["spread", "total"]) & (pl.col("open_time") <= pl.col("tip") - pl.duration(hours=2)))
    g = (lad.select("game_id", "tip").unique("game_id").join(tier, on="game_id", how="left")
         .with_columns((pl.col("tip") >= HOLDOUT_FROM).alias("holdout")).sort("game_id"))
    picked = []
    for (_tr, _h), grp in g.group_by("tier", "holdout", maintain_order=True):
        picked.append(grp.sample(min(per_cell, grp.height), seed=seed, shuffle=False))
    ids = pl.concat(picked)["game_id"]
    lad = lad.filter(pl.col("game_id").is_in(ids.implode()))
    # Spread strikes are signed by team: x = strike for the home team's legs, -strike for away.
    lad = lad.with_columns(pl.when((pl.col("kind") == "spread") & (pl.col("team_side") == "away"))
                           .then(-pl.col("strike")).otherwise(pl.col("strike")).alias("x"))
    rows = []
    for (_gid, _kind), grp in lad.group_by("game_id", "kind"):
        med = float(grp["x"].median())
        rows.append(grp.with_columns((pl.col("x") - med).abs().alias("d")).sort("d", "ticker").head(per_ladder))
    return pl.concat(rows).drop("x", "d") if rows else lad.head(0)


def fetch_leg_candles(s, legs: pl.DataFrame, *, workers: int = 2) -> pl.DataFrame:
    """Hourly candles of each leg over ``[tip - 27 h, tip - 1 h]`` (one request per leg for
    historical markets; batches of 100 for live ones). Cached per leg in
    ``data/cache/fundamental_cbb_candles.parquet``; raw pages under data/cache/cbb/raw."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / "fundamental_cbb_candles.parquet"
    done_path = s.data_dir / "cache" / "fundamental_cbb_candles_done.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = legs.filter(~pl.col("ticker").is_in(list(done)) & pl.col("tip").is_not_null()).unique("ticker", maintain_order=True)
    if not todo.height:
        return have if have is not None else pl.DataFrame(schema=k.CANDLE_SCHEMA)
    api = _api(s, min_interval=0.12)
    cutoff = api.cutoff().get("market_settled_ts")
    jobs: list[tuple[bool, list[str], datetime, datetime]] = []
    for (_gid,), grp in todo.group_by("game_id", maintain_order=True):
        tip = grp["tip"][0]
        lo, hi = tip - QUOTE_WINDOW[0], tip - QUOTE_WINDOW[1]
        hist = grp.filter(pl.col("close_time") < cutoff)["ticker"].to_list() if cutoff else []
        live = grp.filter(pl.col("close_time") >= cutoff)["ticker"].to_list() if cutoff else grp["ticker"].to_list()
        jobs += [(True, [t], lo, hi) for t in hist]
        jobs += [(False, live[i:i + 100], lo, hi) for i in range(0, len(live), 100)]

    def one(job: tuple[bool, list[str], datetime, datetime]) -> tuple[list[str], list[dict[str, Any]]]:
        historical, tickers, lo, hi = job
        for attempt in range(5):
            try:
                if historical:
                    page = api.candles(tickers[0], lo, hi, 60, True)
                    return tickers, k.candle_rows(page.payload, tickers[0], 60)
                page = api.candles_batch(tickers, lo, hi, 60)
                out: list[dict[str, Any]] = []
                for mk in page.payload.get("markets") or []:
                    out.extend(k.candle_rows(mk, mk.get("market_ticker"), 60))
                return tickers, out
            except RuntimeError as error:
                if "429" in str(error):
                    time.sleep(5 * (attempt + 1))
                    continue
                print(f"skip candles {tickers[0]} (+{len(tickers) - 1}): {str(error)[:120]}", flush=True)
                return tickers, []
        return [], []

    rows: list[dict[str, Any]] = []
    finished: list[str] = []

    def save() -> pl.DataFrame:
        new = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else None
        frames = [f for f in (have, new) if f is not None and f.height]
        out = (pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"])
               if frames else pl.DataFrame(schema=k.CANDLE_SCHEMA))
        for f, target in ((out, path), (pl.DataFrame({"ticker": sorted(done | set(finished))}), done_path)):
            tmp = target.with_suffix(".tmp")
            f.write_parquet(tmp)
            tmp.replace(target)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (tk, r) in enumerate(pool.map(one, jobs), 1):
            rows.extend(r)
            finished.extend(tk)
            if i % 500 == 0:
                print(f"  candle jobs {i}/{len(jobs)}, {len(rows)} rows", flush=True)
                save()
    return save()


def trade_windows(legs: pl.DataFrame, leads: tuple[int, ...] = LEADS_H) -> pl.DataFrame:
    """One trade window per leg: from the earliest decision time at which it is open to tip."""
    first = max(leads)
    return legs.select("ticker", pl.max_horizontal(pl.col("tip") - pl.duration(hours=first), pl.col("open_time")).alias("t_from"),
                       pl.col("tip").alias("t_to")).filter(pl.col("t_from") < pl.col("t_to"))


# ---- study pipeline -----------------------------------------------------------------------------
def load_inputs(s, *, fetch_games_: bool = False) -> dict[str, Any]:
    """ESPN games, tiers, model games, Kalshi legs matched to games (final games only, results
    checked against ESPN scores, tip times sanity-checked against the moneyline close)."""
    cache = s.data_dir / "cache"
    gpath = cache / "fundamental_cbb_games.parquet"
    if fetch_games_ or not gpath.exists():
        fetch_games(s).write_parquet(gpath)
    games = pl.read_parquet(gpath).filter(~pl.col("status").is_in(["STATUS_CANCELED", "STATUS_POSTPONED"]))
    tiers = team_tiers(games.filter(pl.col("completed")))
    tier = game_tier(games, tiers)
    markets = kalshi_markets(s)
    mpath = cache / "fundamental_cbb_match.parquet"
    if mpath.exists() and pl.read_parquet(mpath).height:
        match = pl.read_parquet(mpath)
    else:
        match = match_events(markets.filter(pl.col("series_ticker") == "KXNCAAMBGAME"), games)
        match.write_parquet(mpath)
    legs = ladder_legs(markets.filter(pl.col("series_ticker") != "KXNCAAMBTEAMTOTAL"), match, games)
    # A Kalshi-listed game's result enters the model only after its moneyline closed (game
    # over) or tip + 2.75 h, whichever is later.
    ends = legs.filter(pl.col("kind") == "ml").group_by("game_id").agg(pl.col("close_time").max().alias("result_at"))
    games = games.join(ends, on="game_id", how="left")
    mg = model_games(games, tiers)
    # Tip sanity: the moneyline must close at least 1.3 h after ESPN's tip (else the tip time
    # is wrong and a "2 h before tip" quote could be in-play); ESPN time must be valid.
    ok_tip = (games.join(ends, on="game_id", how="inner", suffix="_ml")
              .filter(pl.col("time_valid").fill_null(False)
                      & ((pl.col("result_at_ml") - pl.col("tip")) >= pl.duration(minutes=78))
                      & ((pl.col("result_at_ml") - pl.col("tip")) <= pl.duration(hours=8)))["game_id"])
    legs = legs.join(tier, on="game_id", how="left").with_columns(
        pl.col("game_id").is_in(ok_tip.implode()).alias("tip_ok"),
        pl.col("game_id").is_in(mg["game_id"].implode()).alias("d1_game"))
    return {"games": games, "tiers": tiers, "tier": tier, "mgames": mg, "markets": markets, "match": match, "legs": legs}


def study_legs(inp: dict[str, Any], *, frm: datetime = DEV_FROM, to: datetime = SEASON_END) -> pl.DataFrame:
    """Legs used in the study: final D-I vs D-I games with a sane tip, results agreeing with ESPN."""
    return inp["legs"].filter(pl.col("tip_ok") & pl.col("d1_game") & (pl.col("status") == "STATUS_FINAL")
                              & (pl.col("y") == pl.col("y_espn")) & (pl.col("tip") >= frm) & (pl.col("tip") < to))


# Sample sizes (games per tier x split) for candle requests; low-major is the study's focus.
ML_PER_CELL = {"low": 10_000, "mid": 120, "high": 120}
LADDER_PER_CELL = {"low": 120, "mid": 30, "high": 30}


def sample_games(g: pl.DataFrame, per_cell: dict[str, int], seed: int) -> pl.Series:
    """Seeded, volume-blind random game ids per tier x split (dev / holdout)."""
    g = g.select("game_id", "tier", "tip").unique("game_id").with_columns(
        (pl.col("tip") >= HOLDOUT_FROM).alias("holdout")).sort("game_id")
    picked = [grp.sample(min(per_cell[tr], grp.height), seed=seed, shuffle=False)
              for (tr, _h), grp in g.group_by("tier", "holdout", maintain_order=True)]
    return pl.concat(picked)["game_id"]


def candle_legs(inp: dict[str, Any]) -> pl.DataFrame:
    """Legs whose candles are fetched (historical candles cost one request per leg and the
    public rate limit is shared): the home moneyline leg of every low-major game and of a
    seeded sample of 120 high- and 120 mid-major games per split, plus the seeded ladder sample
    (``LADDER_PER_CELL`` games per tier x split, 2 legs per spread / total ladder). Sizes were
    cut three times while fetching because Kalshi's shared public rate limit (other research
    jobs) let through only ~0.3-2 requests/s."""
    lg = study_legs(inp)
    ml = lg.filter((pl.col("kind") == "ml") & (pl.col("team_side") == "home"))
    ml = ml.filter(pl.col("game_id").is_in(sample_games(ml, ML_PER_CELL, 20251102).implode()))
    lad_all = lg.filter(pl.col("kind").is_in(["spread", "total"]) & (pl.col("open_time") <= pl.col("tip") - pl.duration(hours=2)))
    ids = sample_games(lad_all, LADDER_PER_CELL, 20251101)
    lad = sample_ladder_legs(lad_all.filter(pl.col("game_id").is_in(ids.implode())), inp["tier"], per_cell=10_000)
    return pl.concat([ml, lad], how="diagonal_relaxed")


def fetch_trades(s, windows: pl.DataFrame, *, workers: int = 2) -> pl.DataFrame:
    """Every public trade of each leg inside its window (ticker, t_from, t_to); the harness's
    ``fundamental.fetch_trades`` with this study's client (short 429 back-off, raw cache under
    data/cache/cbb/raw). A window that fails is NOT marked done, so a re-run retries it.
    Cached in ``data/cache/fundamental_cbb_trades.parquet`` (+ ``_windows``)."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / "fundamental_cbb_trades.parquet"
    done_path = path.with_name(path.stem + "_windows.parquet")
    have = pl.read_parquet(path) if path.exists() else None
    done = pl.read_parquet(done_path) if done_path.exists() else pl.DataFrame(
        schema={"ticker": pl.String, "t_from": pl.Datetime("us", "UTC"), "t_to": pl.Datetime("us", "UTC")})
    w = windows.select("ticker", "t_from", "t_to").unique(maintain_order=True)
    todo = w.join(done, on=["ticker", "t_from", "t_to"], how="anti")
    if not todo.height:
        return have if have is not None else pl.DataFrame(schema=k.TRADE_SCHEMA)
    api = _api(s, min_interval=0.12)
    cut = api.cutoff()
    cutoff = cut.get("trades_created_ts") or cut.get("market_settled_ts")

    def one(r: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        historical = bool(cutoff and r["t_to"] < cutoff)
        out: list[dict[str, Any]] = []
        try:
            for page in api.trades(r["ticker"], historical, min_ts=r["t_from"], max_ts=r["t_to"]):
                out.extend(k.trade_rows(page.payload))
        except RuntimeError as error:
            print(f"skip trades {r['ticker']}: {str(error)[:120]}", flush=True)
            return None, []
        return r, out

    rows: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []

    def save() -> pl.DataFrame:
        new = pl.DataFrame(rows, schema=k.TRADE_SCHEMA) if rows else None
        frames = [f for f in (have, new) if f is not None and f.height]
        out = pl.concat(frames, how="diagonal_relaxed").unique("trade_id") if frames else pl.DataFrame(schema=k.TRADE_SCHEMA)
        fin = pl.DataFrame(finished, schema=done.schema) if finished else done.head(0)
        for f, target in ((out, path), (pl.concat([done, fin]), done_path)):
            tmp = target.with_suffix(".tmp")
            f.write_parquet(tmp)
            tmp.replace(target)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (r, got) in enumerate(pool.map(one, todo.iter_rows(named=True)), 1):
            if r is not None:
                rows.extend(got)
                finished.append({"ticker": r["ticker"], "t_from": r["t_from"], "t_to": r["t_to"]})
            if i % 500 == 0:
                print(f"  trades {i}/{todo.height}, {len(rows)} rows", flush=True)
                save()
    return save()


# ---- evaluation ---------------------------------------------------------------------------------
# Hyperparameters fitted on dev only (2024-25 season and Nov 2025 - Jan 2026 games, each predicted
# 2 h before tip by the causal pass; 2023-24 is burn-in), objective = mean Gaussian predictive
# log likelihood of margin (+ total). ``massey``: Nelder-Mead (96 evaluations). ``pts`` / ``eff``:
# the massey drift / offseason parameters converted to their units (points split over offence
# and defence; efficiency = points x 100/69), noise from moment matching, then a few local
# variants (tempo drift, offseason regression, possession noise); the machine was too loaded
# for a full Nelder-Mead on the 1,100-state filter. Dev objective (2024-25, dev 2025-26):
# massey -3.878 (margin only); pts -8.143; eff -8.134; margin RMSE 11.54 / 11.92 points for all
# three, total RMSE 16.70 / 17.71 (pts) and 16.55 / 17.49 (eff). Frozen 2026-11-06 before any
# holdout price was scored.
MODELS: dict[str, EffParams] = {
    "massey": EffParams(mode="massey", sigma=10.856, q_day=0.0197, rho_off=0.7444, q_off=21.45, p0=966.0,
                        home=3.0, mu=0.0),
    "pts": EffParams(mode="pts", sigma=9.87, rho_e=0.396, q_day=0.03, rho_off=0.65, q_off=21.1, p0=936.0,
                     home=3.0, mu=71.0),
    "eff": EffParams(mode="eff", sigma=12.1, rho_e=0.25, sigma_p=4.5, q_day=0.036, q_day_tempo=0.04,
                     rho_off=0.65, q_off=44.0, q_off_tempo=6.0, p0=1970.0, p0_tempo=9.0, home=4.5, mu=104.0),
}
# Chosen on dev (2026-11-06, ``main("dev")``, all tiers pooled): lowest dev Kalshi log loss over
# moneyline + spread legs -> "pts" (0.57326; "eff" 0.57327, "massey" 0.57375: a tie); over total
# legs -> "eff" (0.68676 vs "pts" 0.69111: tempo matters for totals); normal law beats Student t
# in every cell.
CHOSEN: dict[str, str] = {"margin": "pts", "totals": "eff", "law": "normal"}


def run_all(inp: dict[str, Any], prm: EffParams, *, leads: tuple[int, ...] = LEADS_H) -> pl.DataFrame:
    """One causal pass answering every D-I game since Nov 2024 at tip − 2 h (residual laws,
    diagnostics) and every study game at each lead."""
    mg = inp["mgames"]
    q_all = queries_for(mg.filter(pl.col("tip") >= datetime(2024, 11, 1, tzinfo=UTC)), (2,))
    sg = study_legs(inp)["game_id"].unique()
    q_k = queries_for(inp["games"].filter(pl.col("game_id").is_in(sg.implode())), leads)
    q = pl.concat([q_all, q_k]).unique(["game_id", "t"])
    pred = run_model(mg, q.drop("lead"), prm, inp["tiers"])
    return pred.join(q.select("game_id", "t", "lead"), on=["game_id", "t"])


def cached_run(s, inp: dict[str, Any], name: str) -> pl.DataFrame:
    """:func:`run_all` for ``MODELS[name]``, cached by a hash of the parameters."""
    prm = MODELS[name]
    key = hashlib.sha256(repr(prm).encode()).hexdigest()[:10]
    path = s.data_dir / "cache" / f"fundamental_cbb_run_{name}_{key}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    pred = run_all(inp, prm)
    pred.write_parquet(path)
    return pred


def dev_residuals(pred: pl.DataFrame, mg: pl.DataFrame) -> dict[str, np.ndarray]:
    """Standardised residuals at tip − 2 h of games tipping Nov 2024 – Jan 2026 (dev only)."""
    d = (pred.filter(pl.col("lead") == 2).join(mg.select("game_id", "tip", "home_score", "away_score"), on="game_id")
         .filter((pl.col("tip") >= datetime(2024, 11, 1, tzinfo=UTC)) & (pl.col("tip") < HOLDOUT_FROM)))
    out = {"margin": ((d["home_score"] - d["away_score"] - d["mu_margin"]) / d["sd_margin"]).to_numpy()}
    if "mu_total" in d.columns:
        out["total"] = ((d["home_score"] + d["away_score"] - d["mu_total"]) / d["sd_total"]).to_numpy()
        out["team"] = np.concatenate([((d["home_score"] - d["mu_home"]) / d["sd_home"]).to_numpy(),
                                      ((d["away_score"] - d["mu_away"]) / d["sd_away"]).to_numpy()])
    return out


def contract_frame(legs: pl.DataFrame, pred: pl.DataFrame, dists: dict[str, Any], candles: pl.DataFrame,
                   lead: int) -> pl.DataFrame:
    """Legs open at ``t = tip − lead`` with model ``p``, the executable quote at ``t`` (last
    hourly candle closed by ``t``), ``y``, tier; ``event`` = ESPN game id; ``end`` = tip."""
    from albert.research.fundamental import attach_quotes
    from albert.research.ncaaf import leg_probabilities

    lg = (legs.with_columns((pl.col("tip") - pl.duration(hours=lead)).alias("t"), pl.lit(lead).alias("lead"))
          .filter(pl.col("open_time") <= pl.col("t")))
    pr = pred.filter(pl.col("lead") == lead).drop("lead")
    if "mu_total" not in pr.columns:
        lg = lg.filter(pl.col("kind") != "total")
    df = leg_probabilities(lg, pr, dists).filter(pl.col("p").is_not_null())
    df = attach_quotes(df, candles).with_columns(pl.col("game_id").alias("event"), pl.col("tip").alias("end"))
    return df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("t") >= HOLDOUT_FROM).alias("holdout"))


def kalshi_scores(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Brier / log loss of model and mid per segment and split."""
    q = pl.col("p").clip(0.005, 0.995)
    m = pl.col("mid").clip(0.005, 0.995)
    return (df.group_by("holdout", *by).agg(
        pl.len().alias("n"), pl.col("event").n_unique().alias("games"),
        ((pl.col("p") - pl.col("y")) ** 2).mean().alias("brier_model"), ((pl.col("mid") - pl.col("y")) ** 2).mean().alias("brier_mid"),
        (-(pl.col("y") * q.log() + (1 - pl.col("y")) * (1 - q).log())).mean().alias("ll_model"),
        (-(pl.col("y") * m.log() + (1 - pl.col("y")) * (1 - m).log())).mean().alias("ll_mid"),
        (pl.col("ask") - pl.col("bid")).mean().alias("spread_c")).sort("holdout", *by))


def fit_blends(df: pl.DataFrame, cells: list[str], *, min_rows: int = 150) -> dict[tuple, np.ndarray]:
    """Dev-only blend coefficients ``logit P = a + b·logit(mid) + c·logit(p)`` per cell (falls
    back to the pooled kind x lead fit when a cell has fewer than ``min_rows`` dev rows)."""
    from albert.research.fundamental import fit_logistic, logit

    dev = df.filter(~pl.col("holdout"))
    out: dict[tuple, np.ndarray] = {}
    for key, part in dev.group_by(*cells):
        if part.height < min_rows:
            continue
        X = np.column_stack([np.ones(part.height), logit(part["mid"].to_numpy()), logit(part["p"].to_numpy())])
        out[tuple(key)] = fit_logistic(X, part["y"].to_numpy().astype(float), part["event"].to_numpy())[0]
    return out


def apply_blend(df: pl.DataFrame, coefs: dict[tuple, np.ndarray], cells: list[str],
                fallback: dict[tuple, np.ndarray], fb_cells: list[str]) -> pl.DataFrame:
    from albert.research.fundamental import logit

    rows = df.select(*cells, *[c for c in fb_cells if c not in cells]).to_dicts()
    a = np.full(df.height, np.nan)
    X = np.column_stack([np.ones(df.height), logit(df["mid"].to_numpy()), logit(df["p"].to_numpy())])
    for i, r in enumerate(rows):
        beta = coefs.get(tuple(r[c] for c in cells))
        if beta is None:
            beta = fallback.get(tuple(r[c] for c in fb_cells))
        if beta is not None:
            a[i] = X[i] @ beta
    return df.with_columns(pl.Series("p_fair", 1 / (1 + np.exp(-a))).fill_nan(None))


def window_extremes(pred: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Per (ticker, t): the lowest YES price printed by a NO-taker and the highest printed by a
    YES-taker in (t, end]. A YES bid at b fills by trade-through iff ``min_no < b``; a NO bid at
    n iff ``max_yes > 1 − n`` (exactly the harness rule, vectorised)."""
    tr = trades.select("ticker", pl.col("created_time").alias("ts"), "yes_price", "taker_side")
    j = (pred.select("ticker", "t", "end").unique().join(tr, on="ticker", how="inner")
         .filter((pl.col("ts") > pl.col("t")) & (pl.col("ts") <= pl.col("end"))))
    return j.group_by("ticker", "t").agg(
        pl.col("yes_price").filter(pl.col("taker_side") == "no").min().alias("min_no"),
        pl.col("yes_price").filter(pl.col("taker_side") == "yes").max().alias("max_yes"))


def candle_ohlc(s) -> pl.DataFrame:
    """Hourly traded-price low / high (and volume) of every leg whose candles this study cached
    (re-read from the raw candle responses under data/cache/cbb/raw; ``kalshi.candle_rows``
    keeps only close and mean)."""
    from albert.http import read_raw
    from albert.venues.kalshi import parse_ts

    path = s.data_dir / "cache" / "fundamental_cbb_candles_ohlc.parquet"
    raw = cbb_settings(s).raw_dir / "kalshi"
    rows = []
    for f in sorted(raw.glob("*.json.zst")):
        rec = read_raw(f)
        url = rec.get("url", "")
        if "candlesticks" not in url:
            continue
        resp = rec["response"]
        pages = ([(url.rsplit("/markets/", 1)[1].split("/")[0], resp.get("candlesticks") or [])]
                 if "/markets/candlesticks" not in url else
                 [(m.get("market_ticker"), m.get("candlesticks") or []) for m in resp.get("markets") or []])
        for ticker, cs in pages:
            for c in cs:
                vol = float(c.get("volume") or c.get("volume_fp") or 0)
                pr = c.get("price") or {}
                lo, hi = pr.get("low") or pr.get("low_dollars"), pr.get("high") or pr.get("high_dollars")
                if vol <= 0 or lo is None or hi is None:
                    continue
                rows.append({"ticker": ticker, "end_ts": parse_ts(c.get("end_period_ts")), "price_low": float(lo),
                             "price_high": float(hi), "volume": vol})
    out = pl.DataFrame(rows, schema={"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "price_low": pl.Float64,
                                     "price_high": pl.Float64, "volume": pl.Float64}).unique(["ticker", "end_ts"])
    out.write_parquet(path)
    return out


def candle_extremes(pred: pl.DataFrame, ohlc: pl.DataFrame) -> pl.DataFrame:
    """Per (ticker, t): lowest and highest traded price in hourly periods that START at or after
    ``t`` and end by ``end`` (the cached candles stop at tip − 1 h, so the last hour before tip
    is not seen). For a RESTING bid any print strictly below it implies it was filled (a seller
    would have hit it first; an ask below it would have crossed it), so the low / high play the
    roles of the harness's NO-taker low / YES-taker high."""
    j = (pred.select("ticker", "t", "end").unique().join(ohlc, on="ticker", how="inner")
         .filter((pl.col("end_ts") - pl.duration(hours=1) >= pl.col("t")) & (pl.col("end_ts") <= pl.col("end"))))
    return j.group_by("ticker", "t").agg(pl.col("price_low").min().alias("min_no"), pl.col("price_high").max().alias("max_yes"))


def maker_fills_all(df: pl.DataFrame, trades: pl.DataFrame | None, attrs: list[str], *,
                    ext: pl.DataFrame | None = None,
                    margins: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08),
                    rules: tuple[str, ...] = ("through", "at")) -> pl.DataFrame:
    """Maker fills around ``p_fair`` for every margin and fill rule, vectorised: from ``t`` to
    tip rest a YES bid at floor(fair − margin) and a NO bid at floor(1 − fair − margin), each
    capped one tick inside the opposite quote (post-only), held to settlement with the series
    maker fee. ``"through"`` = the harness rule (a later print strictly through our price);
    ``"at"`` = lenient bound (a print at our price also fills: we posted at ``t`` and could lead
    that level's queue). Rows carry the attributes, mid and fair value. ``ext`` (ticker, t,
    min_no, max_yes) replaces the trade tape, e.g. :func:`candle_extremes`."""
    from albert.research.fundamental import maker_fee

    base = df.select("ticker", "event", "t", "end", pl.col("p_fair").alias("fair"), "bid", "ask", "y", "fee_type",
                     "fee_multiplier", "mid", "holdout", *attrs).filter(pl.col("fair").is_not_null())
    if ext is None:
        ext = window_extremes(base, trades)
    base = base.join(ext, on=["ticker", "t"], how="left")
    out = []
    for m in margins:
        yb = pl.min_horizontal(((pl.col("fair") - m) * 100 + 1e-9).floor() / 100, (pl.col("ask") - 0.01).round(2))
        nb = pl.min_horizontal(((1 - pl.col("fair") - m) * 100 + 1e-9).floor() / 100, (1 - pl.col("bid") - 0.01).round(2))
        for side, price in (("yes", yb), ("no", nb)):
            o = base.with_columns(price.round(2).alias("price"), pl.lit(side).alias("side"), pl.lit(m).alias("margin"))
            o = o.filter(pl.col("price") >= 0.01)
            for rule in rules:
                if side == "yes":
                    hit = (pl.col("min_no") < pl.col("price") - 1e-9) if rule == "through" else (pl.col("min_no") <= pl.col("price") + 1e-9)
                else:
                    hit = ((pl.col("max_yes") > 1 - pl.col("price") + 1e-9) if rule == "through"
                           else (pl.col("max_yes") >= 1 - pl.col("price") - 1e-9))
                out.append(o.with_columns(hit.fill_null(False).alias("filled"), pl.lit(rule).alias("rule")))
    F = pl.concat(out)
    fee = [maker_fee(p, ft, mu or 1.0) for p, ft, mu in zip(F["price"], F["fee_type"], F["fee_multiplier"], strict=True)]
    won = pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y"))
    F = F.with_columns(pl.Series("fee", fee)).with_columns(
        pl.when(pl.col("filled")).then(won - pl.col("price") - pl.col("fee")).otherwise(0.0).alias("pnl"),
        (pl.col("fair") - pl.col("mid")).alias("edge_mid"))
    return F.drop("min_no", "max_yes")


def maker_table(F: pl.DataFrame, by: list[str], *, gates: tuple[float, ...] = (0.0, 0.02, 0.04),
                min_fills: int = 30) -> pl.DataFrame:
    """Maker test per segment from :func:`maker_fills_all`. Variants: two-sided at every row
    (gate 0), and rows where |fair − mid| ≥ gate, two-sided or one-sided (only the side the fair
    value favours). The margin is chosen per segment x variant on dev (best PnL per fill with
    ≥ ``min_fills`` fills); the holdout is reported at that margin."""
    from albert.research.fundamental import clustered_mean

    margins = sorted(F["margin"].unique().to_list())
    rows = []
    for gate in gates:
        for sided in (("two", "one") if gate > 0 else ("two",)):
            G = F.filter(pl.col("edge_mid").abs() >= gate - 1e-9)
            if sided == "one":
                G = G.filter(((pl.col("side") == "yes") & (pl.col("edge_mid") > 0)) | ((pl.col("side") == "no") & (pl.col("edge_mid") < 0)))
            for key, seg in G.group_by(*by, maintain_order=True):
                cells = {}
                for (m, h), part in seg.group_by("margin", "holdout"):
                    filled = part.filter("filled")
                    r = {"orders": part.height, "fills": filled.height}
                    if filled.height:
                        mu, t, k = clustered_mean(filled["pnl"].to_numpy(), filled["event"].to_numpy())
                        r.update({"pnl_fill_c": round(mu * 100, 2), "t": None if t is None else round(t, 2), "events": k,
                                  "pnl_order_c": round(float(part["pnl"].sum()) / part.height * 100, 3)})
                    cells[(m, h)] = r
                best = None
                for m in margins:
                    d = cells.get((m, False))
                    if d and d["fills"] >= min_fills and (best is None or d["pnl_fill_c"] > best[1]):
                        best = (m, d["pnl_fill_c"])
                if best is None:
                    continue
                d, h = cells.get((best[0], False), {}), cells.get((best[0], True), {})
                rows.append({**dict(zip(by, key, strict=True)), "gate": gate, "sides": sided, "margin": best[0],
                             "dev_orders": d.get("orders"), "dev_fills": d.get("fills"), "dev_pnl_fill_c": d.get("pnl_fill_c"),
                             "dev_t": d.get("t"), "dev_pnl_order_c": d.get("pnl_order_c"),
                             "hold_orders": h.get("orders"), "hold_fills": h.get("fills"), "hold_events": h.get("events"),
                             "hold_pnl_fill_c": h.get("pnl_fill_c"), "hold_t": h.get("t"), "hold_pnl_order_c": h.get("pnl_order_c")})
    return pl.DataFrame(rows, infer_schema_length=None)


def summary_row(r: dict[str, Any], **keys: Any) -> dict[str, Any]:
    from albert.research.ncaaf import _summary_row

    return _summary_row(r, **keys)


def eval_grid(df: pl.DataFrame, by: list[str], *, min_dev: int = 60) -> pl.DataFrame:
    """Harness ``evaluate`` per segment (blend c on dev, its holdout refit, taker at dev θ)."""
    from albert.research.fundamental import evaluate

    rows = []
    for key, part in df.group_by(*by, maintain_order=True):
        if part.filter(~pl.col("holdout")).height < min_dev:
            continue
        rows.append(summary_row(evaluate(part, HOLDOUT_FROM), **dict(zip(by, key, strict=True))))
    return pl.DataFrame(rows, infer_schema_length=None).sort(by)


def price_band(mid: pl.Expr) -> pl.Expr:
    return (pl.when(mid < 0.1).then(pl.lit("a <10c")).when(mid < 0.3).then(pl.lit("b 10-30c"))
            .when(mid < 0.7).then(pl.lit("c 30-70c")).when(mid < 0.9).then(pl.lit("d 70-90c"))
            .otherwise(pl.lit("e >=90c")))


def spread_band(spr: pl.Expr) -> pl.Expr:
    """Quoted spread at ``t`` (ask − bid): the coordinator's bands, shared across studies."""
    return (pl.when(spr <= 0.055).then(pl.lit("a <=5c")).when(spr <= 0.105).then(pl.lit("b 6-10c"))
            .when(spr <= 0.155).then(pl.lit("c 11-15c")).otherwise(pl.lit("d >15c")))


def model_frames(inp: dict[str, Any], runs: dict[str, pl.DataFrame], choice: dict[str, str], law: str,
                 candles: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Contract rows at both leads: moneyline / spread legs priced by ``choice["margin"]``,
    total legs by ``choice["totals"]``, residual law ``law`` ("normal" / "t") fitted on dev."""
    from albert.research.ncaaf import fit_dist

    mg = inp["mgames"]
    legs = study_legs(inp).filter(pl.col("ticker").is_in(candles["ticker"].unique().implode()))
    frames, laws = [], {}
    for part, name, kinds in (("margin", choice["margin"], ["ml", "spread"]), ("totals", choice["totals"], ["total"])):
        pred = runs[name]
        fits = {k2: fit_dist(v) for k2, v in dev_residuals(pred, mg).items()}
        laws[part] = fits
        dists = {k2: f[law] for k2, f in fits.items()}
        for lead in LEADS_H:
            df = contract_frame(legs.filter(pl.col("kind").is_in(kinds)), pred, dists, candles, lead)
            frames.append(df)
    df = pl.concat(frames, how="diagonal_relaxed").with_columns(
        price_band(pl.col("mid")).alias("band"), spread_band(pl.col("ask") - pl.col("bid")).alias("spr_band"))
    return df, laws


def dev_selection(df_by_model: dict[tuple[str, str], pl.DataFrame]) -> pl.DataFrame:
    """Dev rows only: per (model, law, lead, kind) log loss of model vs mid and blend c."""
    from albert.research.fundamental import evaluate

    rows = []
    for (name, law), df in df_by_model.items():
        dev = df.filter(~pl.col("holdout"))
        for (lead, kind), part in dev.group_by("lead", "kind"):
            if part.height < 60:
                continue
            r = evaluate(part, HOLDOUT_FROM)
            sc, b = r["scores"][0], r.get("blend", {})
            rows.append({"model": name, "law": law, "lead": lead, "kind": kind, "n": sc["n"], "games": sc["events"],
                         "brier_model": sc["brier_model"], "brier_mid": sc["brier_mid"], "ll_model": sc["logloss_model"],
                         "ll_mid": sc["logloss_mid"], "c": b.get("c_model"), "t_c": b.get("t_c")})
    return pl.DataFrame(rows).sort("kind", "lead", "ll_model")


def taker_by(df: pl.DataFrame, theta: float, by: list[str]) -> pl.DataFrame:
    """Taker PnL (cents per contract, t clustered by game) per segment at a fixed θ."""
    from albert.research.fundamental import clustered_mean, taker_pnl

    tr = taker_pnl(df, theta)
    if not tr.height:
        return pl.DataFrame()
    rows = []
    for key, part in tr.group_by("holdout", *by):
        mu, t, kk = clustered_mean(part["pnl"].to_numpy(), part["event"].to_numpy())
        rows.append({**dict(zip(["holdout", *by], key, strict=True)), "trades": part.height, "games": kk,
                     "pnl_c": round(mu * 100, 2), "t": None if t is None else round(t, 2)})
    return pl.DataFrame(rows).sort("holdout", *by)


def holdout_study(s, inp: dict[str, Any], df: pl.DataFrame, trades: pl.DataFrame, laws: dict[str, Any],
                  extra: dict[str, Any], ohlc: pl.DataFrame | None = None) -> tuple[Path, dict, dict]:
    """Stage 2 (frozen model, law and blend): harness evaluation per lead x kind (x tier),
    blend fair value, taker on raw model and on the fair value, maker around the fair value,
    segments, and the report. Saves holdout prediction rows for cross-study comparison."""
    from albert.reports import md_table
    from albert.research.fundamental import evaluate, report

    sections: list[tuple[str, pl.DataFrame]] = []
    sections.append(("Kalshi scores per lead x kind x tier (model vs mid; spread_c = mean ask − bid)",
                     kalshi_scores(df, ["lead", "kind", "tier"])))
    g1 = eval_grid(df, ["lead", "kind"])
    sections.append(("Harness per lead x kind: blend c (dev) and holdout refit; taker on the RAW model at the dev θ", g1))
    g2 = eval_grid(df, ["lead", "kind", "tier"])
    sections.append(("Harness per lead x kind x tier (θ chosen on each cell's dev rows)", g2))
    g3 = eval_grid(df, ["lead", "tier"])
    sections.append(("Harness per lead x tier, all contract kinds pooled", g3))
    sections.append(("Rows per quoted-spread band at t (ask − bid)", df.group_by("lead", "kind", "tier", "spr_band").len()
                     .sort("lead", "kind", "tier", "spr_band")))
    sections.append(("Harness per quoted-spread band x lead x kind (blend c, refit and θ fitted within each band)",
                     eval_grid(df, ["spr_band", "lead", "kind"], min_dev=40)))
    sections.append(("Harness per quoted-spread band x tier (all kinds and leads pooled)",
                     eval_grid(df, ["spr_band", "tier"], min_dev=40)))
    sections.append(("Harness per quoted-spread band (everything pooled)", eval_grid(df, ["spr_band"], min_dev=40)))
    # Fair value: dev-fitted blend per kind x lead x tier (pooled kind x lead fallback).
    cells, fb = ["kind", "lead", "tier"], ["kind", "lead"]
    coefs, fall = fit_blends(df, cells), fit_blends(df, fb)
    df = apply_blend(df, coefs, cells, fall, fb)
    sections.append(("Blend coefficients (dev) used as fair value: logit P = a + b·logit(mid) + c·logit(model)",
                     pl.DataFrame([{**dict(zip(cells, kk, strict=True)), "a": round(float(v[0]), 4), "b_mid": round(float(v[1]), 4),
                                    "c_model": round(float(v[2]), 4)} for kk, v in sorted(coefs.items())])))
    fair = df.with_columns(pl.col("p_fair").alias("p"))
    rows = []
    for key, part in fair.group_by("lead", "kind", "tier", maintain_order=True):
        if part.filter(~pl.col("holdout")).height < 60:
            continue
        r = evaluate(part, HOLDOUT_FROM)
        sr = summary_row(r, lead=key[0], kind=key[1], tier=key[2])
        rows.append({k2: sr[k2] for k2 in ("lead", "kind", "tier", "brier_model_hold", "brier_mid_hold", "theta",
                                            "dev_trades", "dev_pnl_c", "dev_t", "hold_trades", "hold_pnl_c", "hold_t")})
    sections.append(("Taker on the FAIR value (dev-fitted blend), θ chosen on dev per cell", pl.DataFrame(rows).sort("lead", "kind", "tier")))
    rows = []
    for key, part in fair.group_by("spr_band", "kind", maintain_order=True):
        if part.filter(~pl.col("holdout")).height < 40:
            continue
        sr = summary_row(evaluate(part, HOLDOUT_FROM), spr_band=key[0], kind=key[1])
        rows.append({k2: sr[k2] for k2 in ("spr_band", "kind", "theta", "dev_trades", "dev_pnl_c", "dev_t",
                                            "hold_trades", "hold_pnl_c", "hold_t")})
    sections.append(("Taker on the FAIR value per quoted-spread band x kind (θ chosen on dev per cell)",
                     pl.DataFrame(rows).sort("spr_band", "kind")))
    # Segments of the raw-model taker at a fixed θ = 0.03 (by price band, by quoted spread).
    for by in ("band", "spr_band"):
        sections.append((f"Raw-model taker at θ = 0.03 by {by} (low-major only)",
                         taker_by(df.filter(pl.col("tier") == "low"), 0.03, ["kind", by])))
    sections.append(("Raw-model taker at θ = 0.03 by quoted-spread band (all tiers)", taker_by(df, 0.03, ["lead", "spr_band"])))
    sections.append(("Scores by quoted spread (all tiers)", kalshi_scores(df, ["kind", "spr_band"])))
    # Maker around the fair value (trade-through fills).
    if ohlc is not None:
        # Fill source: hourly traded-price ranges from the cached candles (Kalshi's shared rate
        # limit made a full trade pull impossible); validated against the public trade tape on a
        # seeded subsample (trade-tape fills when ``trades`` covers it).
        F = maker_fills_all(df, None, ["lead", "kind", "tier", "spr_band"], ext=candle_extremes(df, ohlc))
        sections.append(("Maker fill source", pl.DataFrame({"source": [
            "hourly traded-price low / high from the cached candles: periods starting >= t and ending <= tip - 1 h "
            "(the last hour before tip is not seen)"]})))
        if trades.height:
            sub = df.filter(pl.col("ticker").is_in(trades["ticker"].unique().implode())).with_columns(
                (pl.col("tip") - pl.duration(hours=1)).alias("end"))
            Ft = maker_fills_all(sub, trades, ["lead"], rules=("through",))
            Fc = maker_fills_all(sub, None, ["lead"], ext=candle_extremes(sub, ohlc), rules=("through",))
            agree = (Ft.select("ticker", "t", "side", "margin", pl.col("filled").alias("fill_trades"))
                     .join(Fc.select("ticker", "t", "side", "margin", pl.col("filled").alias("fill_candles")),
                           on=["ticker", "t", "side", "margin"])
                     .group_by("fill_trades", "fill_candles").len().sort("fill_trades", "fill_candles"))
            sections.append((f"Fill-proxy validation: trade-through fills from the public trade tape vs candle ranges, "
                             f"same window (t, tip − 1 h], {sub['game_id'].n_unique()} seeded low-major games, "
                             f"{trades.height} prints", agree))
        F.write_parquet(s.data_dir / "cache" / "fundamental_cbb_maker_fills.parquet")
        sections.append(("Maker around the fair value per lead x kind x tier, trade-THROUGH fills (margin chosen on "
                         "dev per row; gate = rest only when |fair − mid| ≥ gate; sides one = only the side the fair "
                         "value favours)", maker_table(F.filter(pl.col("rule") == "through"), ["lead", "kind", "tier"])))
        sections.append(("Same, lenient trade-AT fills (a print at our price fills us)",
                         maker_table(F.filter(pl.col("rule") == "at"), ["lead", "kind", "tier"])))
        sections.append(("Maker per lead x kind (all tiers), both fill rules", maker_table(F, ["rule", "lead", "kind"])))
        sections.append(("Maker per tier (all kinds and leads pooled), both fill rules", maker_table(F, ["rule", "tier"])))
        sections.append(("Maker per quoted-spread band at t x lead (all kinds and tiers), both fill rules",
                         maker_table(F, ["rule", "spr_band", "lead"])))
        sections.append(("Maker per quoted-spread band x tier, trade-through fills",
                         maker_table(F.filter(pl.col("rule") == "through"), ["spr_band", "tier"])))
    # Residual laws.
    laws_rows = [{"part": part, "variable": v, "n": f["n"], "ll_normal": f["ll_normal"], "ll_t": f["ll_t"], "t_df": f["t"].df,
                  "t_scale": f["t"].scale, "normal_scale": f["normal"].scale} for part, fits in laws.items() for v, f in fits.items()]
    sections.append(("Residual laws on dev games (standardised by the model's predictive SD)", pl.DataFrame(laws_rows)))
    hold = df.filter(pl.col("holdout"))
    hold.select("ticker", "t", "p_fair", "mid", pl.col("p").alias("p_model"), "bid", "ask", "y", "lead", "kind", "tier",
                "game_id").write_parquet(s.data_dir / "cache" / "fundamental_cbb_holdout_preds.parquet")
    pooled = evaluate(df.filter(pl.col("lead") == 2), HOLDOUT_FROM)
    path = report(s, "cbb", pooled, {**extra, "primary_frame": "all contract kinds pooled, 2 h before tip"})
    text = path.read_text()
    for title, frame in sections:
        text += f"\n## {title}\n\n{md_table(frame, max_rows=80)}\n"
    path.write_text(text)
    df.write_parquet(path.parent / "contracts.parquet")
    return path, coefs, fall


# ---- current-season smoke test (no orders) ------------------------------------------------------
def smoke(s, inp: dict[str, Any], prm_margin: EffParams, prm_totals: EffParams, law: str,
          coefs: dict[tuple, np.ndarray], fall: dict[tuple, np.ndarray], *, now: datetime | None = None,
          days_ahead: int = 2) -> dict[str, pl.DataFrame]:
    """Run the frozen models on the 2026-27 season so far (games through ``now``) and price
    (a) Nov 2026 games already settled, 2 h before tip, against the live hourly candles, and
    (b) every open KXNCAAMBGAME / SPREAD / TOTAL leg against its current listed bid / ask.
    Read-only: public market data, no orders."""
    from albert.research.fundamental import attach_quotes
    from albert.research.ncaaf import fit_dist, leg_probabilities
    from albert.venues import kalshi as k

    now = now or datetime.now(UTC)
    tag = now.strftime("%Y%m%dT%H")
    rows: list[dict[str, Any]] = []
    d = date(2026, 11, 1)
    while d <= (now + timedelta(days=days_ahead)).date():
        live = d >= (now - timedelta(days=1)).date()
        rows.extend(game_rows(espn_get(s, "scoreboard", {"dates": d.strftime("%Y%m%d"), "groups": "50", "limit": "500"},
                                       tag=tag if live else None)))
        d += timedelta(days=1)
    cur = pl.DataFrame(rows, schema=GAME_SCHEMA).unique("game_id", keep="last")
    games = pl.concat([inp["games"].filter(pl.col("season") < 2027).drop("result_at", strict=False), cur],
                      how="diagonal_relaxed").filter(~pl.col("status").is_in(["STATUS_CANCELED", "STATUS_POSTPONED"]))
    tiers = team_tiers(games.filter(pl.col("completed") | (pl.col("season") == 2027)))
    mg = model_games(games.filter(pl.col("tip") + GAME_LENGTH <= now), tiers)
    # Kalshi: settled Nov 2026 legs (markets cache) and open legs (live listing).
    api = _api(s)
    info = series_info(s)
    open_rows: list[dict[str, Any]] = []
    for series in SERIES[:3]:
        for page in api.series_markets(series, "open", cache=False):
            open_rows.extend(k.market_rows(page.payload, info, "cbb-open"))
    opn = pl.DataFrame(open_rows, schema=k.MARKET_SCHEMA).with_columns(pl.lit(0, dtype=pl.Int8).alias("y"))
    settled = inp["markets"].filter(pl.col("close_time") >= datetime(2026, 11, 1, tzinfo=UTC))
    allm = pl.concat([settled, opn], how="diagonal_relaxed").unique("ticker", keep="first")
    match = match_events(allm.filter(pl.col("series_ticker") == "KXNCAAMBGAME"), games)
    legs = ladder_legs(allm, match, games).join(game_tier(games, tiers), on="game_id", how="left")
    legs = legs.join(allm.select("ticker", pl.col("status").alias("mstatus"), pl.col("yes_bid").alias("bid_now"),
                                 pl.col("yes_ask").alias("ask_now")), on="ticker", how="left")
    is_open = pl.col("mstatus").is_in(["active", "open", "initialized"])
    q_set = legs.filter(~is_open).select("game_id", "tip").unique().with_columns((pl.col("tip") - pl.duration(hours=2)).alias("t"))
    q_open = legs.filter(is_open & (pl.col("tip") > now)).select("game_id", "tip").unique().with_columns(pl.lit(now).alias("t"))
    qq = pl.concat([q_set, q_open]).join(games.select("game_id", "season", "home_id", "away_id", "neutral"), on="game_id")
    out: dict[str, pl.DataFrame] = {"match": match}
    preds, dists = {}, {}
    for part, prm in (("margin", prm_margin), ("totals", prm_totals)):
        hist = run_model(inp["mgames"], queries_for(inp["mgames"].filter(
            (pl.col("tip") >= datetime(2024, 11, 1, tzinfo=UTC)) & (pl.col("tip") < HOLDOUT_FROM)), (2,)).drop("lead"),
            prm, inp["tiers"]).with_columns(pl.lit(2).alias("lead"))
        fits = {k2: fit_dist(v) for k2, v in dev_residuals(hist, inp["mgames"]).items()}
        dists[part] = {k2: f[law] for k2, f in fits.items()}
        preds[part] = run_model(mg, qq.drop("tip"), prm, tiers)
    frames = []
    for part, kinds in (("margin", ["ml", "spread"]), ("totals", ["total"])):
        lg = legs.filter(pl.col("kind").is_in(kinds)).join(qq.select("game_id", "t"), on="game_id")
        lg = lg.filter((is_open & (pl.col("t") == now)) | (~is_open & (pl.col("t") < now)))
        frames.append(leg_probabilities(lg, preds[part], dists[part]))
    df = pl.concat(frames, how="diagonal_relaxed").with_columns(pl.lit(2).alias("lead"))
    # (a) settled moneylines (home leg): live hourly candles 2 h before tip, fetched in batches of
    # 40 tickers over the whole window (a handful of requests; the public rate limit is shared).
    st = df.filter(~is_open & (pl.col("open_time") <= pl.col("t")) & (pl.col("kind") == "ml")
                   & (pl.col("team_side") == "home"))
    if st.height:
        cpath = s.data_dir / "cache" / "fundamental_cbb_smoke_candles.parquet"
        tick = sorted(st["ticker"].unique().to_list())
        lo, hi = st["t"].min() - timedelta(hours=3), st["t"].max() + timedelta(hours=1)
        crow: list[dict[str, Any]] = []
        for i in range(0, len(tick), 40):
            page = api.candles_batch(tick[i:i + 40], lo, hi, 60)
            for mk in page.payload.get("markets") or []:
                crow.extend(k.candle_rows(mk, mk.get("market_ticker"), 60))
        c = pl.DataFrame(crow, schema=k.CANDLE_SCHEMA)
        c.write_parquet(cpath)
        st = attach_quotes(st, c).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                                               pl.col("game_id").alias("event"), pl.lit(True).alias("holdout"))
        out["settled"] = apply_blend(st, coefs, ["kind", "lead", "tier"], fall, ["kind", "lead"])
    # (b) open legs priced now.
    op = df.filter(is_open & pl.col("bid_now").is_not_null() & pl.col("ask_now").is_not_null()
                   & (pl.col("ask_now") > pl.col("bid_now")))
    op = op.with_columns(pl.col("bid_now").alias("bid"), pl.col("ask_now").alias("ask"),
                         ((pl.col("bid_now") + pl.col("ask_now")) / 2).alias("mid"))
    out["open"] = apply_blend(op, coefs, ["kind", "lead", "tier"], fall, ["kind", "lead"])
    return out


# ---- driver -------------------------------------------------------------------------------------
def dev_stage(s, inp: dict[str, Any], candles: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Stage 1 (dev rows only): every model x residual law, Kalshi log loss and blend c per
    lead x kind. Holdout rows are dropped before any score is computed."""
    runs = {name: cached_run(s, inp, name) for name in MODELS}
    dfs = {}
    for name, prm in MODELS.items():
        for law in ("normal", "t"):
            choice = {"margin": name, "totals": name}
            if prm.mode == "massey":
                df, _ = model_frames(inp, {name: runs[name]}, choice, law, candles)
            else:
                df, _ = model_frames(inp, runs, choice, law, candles)
            dfs[(name, law)] = df.filter(~pl.col("holdout"))
    return dev_selection(dfs), runs


def main(stage: str = "all") -> None:
    """``stage="dev"``: model / law selection on dev; ``"holdout"``: the frozen choice on dev +
    holdout (run once); ``"all"``: holdout plus the current-season smoke test."""
    s = Settings()
    inp = load_inputs(s)
    candles = pl.read_parquet(s.data_dir / "cache" / "fundamental_cbb_candles.parquet")
    if stage == "dev":
        sel, _ = dev_stage(s, inp, candles)
        sel.write_parquet(s.data_dir / "cache" / "fundamental_cbb_dev_selection.parquet")
        with pl.Config(tbl_rows=100, tbl_cols=20):
            print(sel)
        return
    runs = {name: cached_run(s, inp, name) for name in {CHOSEN["margin"], CHOSEN["totals"]}}
    df, laws = model_frames(inp, runs, CHOSEN, CHOSEN["law"], candles)
    tpath = s.data_dir / "cache" / "fundamental_cbb_trades.parquet"
    trades = pl.read_parquet(tpath) if tpath.exists() else pl.DataFrame()
    ohlc = candle_ohlc(s)
    extra = {"models": {k2: str(MODELS[v]) for k2, v in CHOSEN.items() if k2 != "law"}, "law": CHOSEN["law"],
             "dev": "games tipping 2025-11-01 .. 2026-01-31 ET", "holdout": "2026-02-01 .. 2026-04-15",
             "decision_times": "24 h and 2 h before ESPN tip", "sample": {"ml": ML_PER_CELL, "ladder": LADDER_PER_CELL}}
    path, coefs, fall = holdout_study(s, inp, df, trades, laws, extra, ohlc)
    print(path, flush=True)
    if stage == "holdout":
        return
    # Current-season smoke test (2026-27 season, games of Nov 3-5 2026 and open markets; no orders).
    from albert.reports import md_table

    sm = smoke(s, inp, MODELS[CHOSEN["margin"]], MODELS[CHOSEN["totals"]], CHOSEN["law"], coefs, fall)
    text = path.read_text() + "\n## Smoke test: 2026-27 season (first games Nov 3-5 2026), no orders\n\n"
    st = sm.get("settled")
    if st is not None and st.height:
        text += ("Settled Nov 2026 legs, 2 h before tip (live hourly candles):\n\n"
                 + md_table(kalshi_scores(st, ["kind"]), max_rows=20) + "\n\n")
    op = sm["open"].with_columns((pl.col("p") - pl.col("mid")).alias("model_minus_mid"),
                                 (pl.col("p_fair") - pl.col("mid")).alias("fair_minus_mid"))
    op.write_parquet(s.data_dir / "cache" / "fundamental_cbb_smoke_open.parquet")
    summ = op.group_by("kind", "tier").agg(
        pl.len().alias("legs"), pl.col("game_id").n_unique().alias("games"), (pl.col("ask") - pl.col("bid")).mean().alias("spread"),
        pl.col("model_minus_mid").abs().mean().alias("mean_abs_model_minus_mid"),
        pl.col("fair_minus_mid").abs().mean().alias("mean_abs_fair_minus_mid"),
        pl.corr("p", "mid").alias("corr_model_mid")).sort("kind", "tier")
    text += (f"Open legs priced at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC (listed bid / ask):\n\n" + md_table(summ, max_rows=20)
             + "\n\nLargest |fair − mid| among open legs:\n\n"
             + md_table(op.sort(pl.col("fair_minus_mid").abs(), descending=True).select(
                 "ticker", "tier", "kind", "tip", "bid", "ask", "p", "p_fair").head(15), max_rows=15) + "\n")
    path.write_text(text)
    print("smoke done", flush=True)
