"""Thin Kalshi sports series vs simple public-results rating models (study ``thin_sports``).

Question (round 2 of the fundamental-model swarm): round 1 found a real model adds information
to the Kalshi price only in thin, wide-spread markets (FCS college football). Does that hold in
other thin but numerous Kalshi sports series, and does a taker or a maker quoting around a
model + price blend pay after fees on the October 2026 holdout?

Series (chosen from a scan of ``kalshi_markets`` Aug 15 - Oct 31 2026: many events, low volume
per event, a public results feed with dates and scores, not covered by an earlier study):

* ``KXNCAAWVMATCH`` women's college volleyball (1,004 events, ~2.4k contracts per event);
* ``KXNCAAHGAME`` men's college hockey (167 events, ~1.3k per event; October only);
* ``KXENGNL*`` English National League, ``KXUSL*`` USL Championship and ``KXARGNACB*``
  Argentina Primera Nacional soccer: GAME (home / tie / away after 90 min), SPREAD ("X wins by
  more than k.5 goals"), TOTAL ("over k.5 goals") and BTTS legs.

Results come from ESPN's public site API (``site.api.espn.com/apis/site/v2/sports/<sport>/
<league>/scoreboard?dates=YYYYMMDD``): one request per date on the league's own calendar, cached
under ``data/cache/thin_sports/espn`` (zstd JSON), at least 0.6 s between network requests.

As-of discipline (no look-ahead)
--------------------------------
A prediction at decision time ``t`` (24 h or 2 h before ESPN's scheduled start) uses only games
whose start + ``RESULT_LAG`` (4 h: a volleyball match, hockey or soccer game is over and its
score public well within that) is at or before ``t``. The previous season's games give the
preseason prior (down-weighted by ``kappa``). Hyperparameters (time decay, ridge strength,
previous-season weight, volleyball's probability scale ``k``) are chosen on ESPN games before
2026-10-01 (hockey: before 2026-10-17). Kalshi quotes are the last hourly candle that closed by
``t`` (harness). ESPN final scores are used to confirm which ESPN game a Kalshi game is (every
settled leg must agree) and never as a model input for that game.

Splits: dev = decision times before 2026-10-01, holdout = October. Hockey's Kalshi series only
exists in October, so its dev is Oct 3-16 and its holdout Oct 17-31.

Models
------
* **Volleyball:** ridge (Massey-type) rating on each match's point-share logit
  ``z = logit(points_home / points_total) = h·home + r_home − r_away + ε`` with exponential time
  decay, then ``P(win) = σ(k·ẑ)`` with ``k`` fitted on dev games.
* **Hockey and soccer:** log-linear Poisson goals model ``log λ_home = μ + h + a_home − d_away``,
  ``log λ_away = μ + a_away − d_home`` (ridge towards 0, time decay), giving every contract's
  probability (independent Poisson; hockey regulation+OT ties split 50/50 for the shootout).

Fair value and maker test
-------------------------
``fair`` = dev-fitted blend ``logit P = a + b·logit(mid) + c·logit(model)`` per (league, kind);
``fair_band`` = the same per (league, spread > 15c); ``fair_mid`` = the mid recalibrated alone
(control). Bids rest at floor(fair − margin) from the 24 h decision to the 2 h one, and from the
2 h decision to the start. Fills: Kalshi's shared rate limit was saturated on 2026-11-05
(~10 requests/min), so one trades request per market (~7,400) was not feasible; fills are judged
from hourly candles' printed-price ranges instead (:func:`candle_fill_orders`), and checked
against real trade prints (:func:`fill_orders`, ``fetch_trades``) on a 300-market sample.

Run: ``fetch_games`` → ``matched_legs`` → ``select_params`` per league → ``leg_predictions`` →
``fetch_candles`` (QUOTE_WINDOW, then TAIL_WINDOW as ``candles_tail``) → ``run`` →
``write_report``.
"""

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

from albert.research import fundamental as F
from albert.venues.kalshi import ticker_start

NAME = "thin_sports"
HOLDOUT = datetime(2026, 10, 1, tzinfo=UTC)
END = datetime(2026, 11, 1, tzinfo=UTC)
RESULT_LAG = timedelta(hours=4)
LEADS = (24, 2)
ESPN = "https://site.api.espn.com/apis/site/v2/sports"

# league key -> ESPN path, Kalshi series, a date inside each season (to read the season calendar)
LEAGUES: dict[str, dict[str, Any]] = {
    "wvb": {"path": "volleyball/womens-college-volleyball", "kind": "volleyball",
            "series": ("KXNCAAWVMATCH",), "seasons": ("20250915", "20260915")},
    "ncaah": {"path": "hockey/mens-college-hockey", "kind": "hockey",
              "series": ("KXNCAAHGAME",), "seasons": ("20251115", "20261017")},
    "engnl": {"path": "soccer/eng.5", "kind": "soccer",
              "series": ("KXENGNLGAME", "KXENGNLSPREAD", "KXENGNLTOTAL", "KXENGNLBTTS"),
              "seasons": ("20250920", "20260920")},
    "usl": {"path": "soccer/usa.usl.1", "kind": "soccer",
            "series": ("KXUSLGAME", "KXUSLSPREAD", "KXUSLTOTAL", "KXUSLBTTS"),
            "seasons": ("20250601", "20260601")},
    "argnacb": {"path": "soccer/arg.2", "kind": "soccer",
                "series": ("KXARGNACBGAME", "KXARGNACBSPREAD", "KXARGNACBTOTAL", "KXARGNACBBTTS"),
                "seasons": ("20250601", "20260601")},
}


def cache_dir(s) -> Path:
    return s.data_dir / "cache" / NAME


def table_path(s, name: str) -> Path:
    return s.data_dir / "cache" / f"fundamental_{NAME}_{name}.parquet"


def write_table(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- ESPN fetch + parse -----------------------------------------------------------------------
_last = [0.0]


def espn_get(s, path: str, params: dict[str, Any], *, min_interval: float = 0.6) -> dict[str, Any]:
    """GET one ESPN site-API JSON; later calls read the zstd cache under data/cache/thin_sports."""
    import httpx

    from albert.http import read_raw, write_raw

    key = hashlib.sha256((path + "?" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
    file = cache_dir(s) / "espn" / f"{re.sub(r'[^a-z0-9]+', '-', path.lower()).strip('-')}-{key}.json.zst"
    if file.exists():
        return read_raw(file)["response"]
    for attempt in range(4):
        wait = min_interval - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        r = httpx.get(f"{ESPN}/{path}", params=params, timeout=30,
                      headers={"User-Agent": "albert-research (public data, cached)"})
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    else:
        r.raise_for_status()
    payload = r.json()
    write_raw(file, {"url": f"{ESPN}/{path}", "params": params, "fetched_at": datetime.now(UTC).isoformat(),
                     "response": payload})
    return payload


def _ts(x: str | None) -> datetime | None:
    if not x:
        return None
    x = x.replace("Z", "+00:00")
    if re.match(r".*T\d\d:\d\d\+", x):
        x = x.replace("+", ":00+", 1)
    return datetime.fromisoformat(x).astimezone(UTC)


GAME_SCHEMA = {
    "league": pl.String, "game_id": pl.String, "season": pl.Int64, "start": pl.Datetime("us", "UTC"),
    "completed": pl.Boolean, "status": pl.String, "neutral": pl.Boolean,
    "home_id": pl.String, "home_abbr": pl.String, "home_loc": pl.String, "home_name": pl.String,
    "home_short": pl.String, "home_score": pl.Float64, "home_pts": pl.Float64,
    "away_id": pl.String, "away_abbr": pl.String, "away_loc": pl.String, "away_name": pl.String,
    "away_short": pl.String, "away_score": pl.Float64, "away_pts": pl.Float64, "periods": pl.Int64,
}


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def game_rows(payload: dict[str, Any], league: str) -> list[dict[str, Any]]:
    """One row per ESPN event: scheduled start, status, teams, final score and, for volleyball,
    total points over all sets (sum of the competitors' linescores)."""
    out = []
    for e in payload.get("events") or []:
        comp = (e.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if set(teams) != {"home", "away"}:
            continue
        st = (comp.get("status") or e.get("status") or {}).get("type") or {}
        row: dict[str, Any] = {
            "league": league, "game_id": str(e.get("id")), "season": (e.get("season") or {}).get("year"),
            "start": _ts(e.get("date")), "completed": bool(st.get("completed")), "status": st.get("name"),
            "neutral": bool(comp.get("neutralSite")),
        }
        periods = 0
        for side, c in teams.items():
            t = c.get("team") or {}
            ls = [_num(x.get("value")) for x in c.get("linescores") or []]
            periods = max(periods, len(ls))
            row.update({f"{side}_id": str(t.get("id")), f"{side}_abbr": t.get("abbreviation"),
                        f"{side}_loc": t.get("location"), f"{side}_name": t.get("displayName"),
                        f"{side}_short": t.get("shortDisplayName"), f"{side}_score": _num(c.get("score")),
                        f"{side}_pts": float(sum(x for x in ls if x is not None)) if ls else None})
        row["periods"] = periods
        out.append(row)
    return out


def season_dates(payload: dict[str, Any]) -> list[str]:
    """Query values for every game of the season in a scoreboard payload's league calendar: its
    game days (YYYYMMDD) for day calendars, or the months (YYYYMM) spanned by the entries of a
    list calendar (USL), which ESPN's ``dates`` parameter also accepts."""
    cal = ((payload.get("leagues") or [{}])[0]).get("calendar") or []
    out: set[str] = set()
    for c in cal:
        if isinstance(c, str):
            out.add(c[:10].replace("-", ""))
            continue
        for e in (c.get("entries") or [c]) if isinstance(c, dict) else []:
            lo, hi = _ts(e.get("startDate")), _ts(e.get("endDate"))
            if lo is None or hi is None:
                continue
            y, m = lo.year, lo.month
            while (y, m) <= (hi.year, hi.month):
                out.add(f"{y}{m:02d}")
                y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return sorted(out)


def fetch_games(s, leagues: tuple[str, ...] = tuple(LEAGUES), *, last: date = date(2026, 10, 31)) -> pl.DataFrame:
    """Every ESPN game of each league's last two seasons up to ``last`` (cached per request)."""
    rows: list[dict[str, Any]] = []
    for lg in leagues:
        cfg = LEAGUES[lg]
        for anchor in cfg["seasons"]:
            first = espn_get(s, f"{cfg['path']}/scoreboard", {"dates": anchor, "limit": 1000})
            days = [d for d in season_dates(first) if d <= last.strftime("%Y%m%d")[:len(d)]]
            print(f"{lg} season {anchor[:4]}: {len(days)} dates", flush=True)
            for d in days:
                rows.extend(game_rows(espn_get(s, f"{cfg['path']}/scoreboard", {"dates": d, "limit": 1000}), lg))
    games = pl.DataFrame(rows, schema=GAME_SCHEMA).unique(["league", "game_id"], keep="last").sort("league", "start")
    write_table(games, table_path(s, "games"))
    return games


# ---- Kalshi legs ------------------------------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_GK_RX = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})(\d{4})?([A-Z0-9]+)$")
_VS_RX = re.compile(r"(?:wins the|result of the|in the) (.+?) vs (.+?) (?:professional |women's |men's )?"
                    r"(?:English|USL|Argentine|college|soccer|match|game)")
_SPREAD_RX = re.compile(r"^(.+?) wins by more than (\d+)\.5 goals", re.IGNORECASE)
_TOTAL_RX = re.compile(r"^Over (\d+)\.5 goals", re.IGNORECASE)


def leg_kind(series: str) -> str:
    for k in ("SPREAD", "TOTAL", "BTTS"):
        if series.endswith(k):
            return k.lower()
    return "win"


def leg_row(r: dict[str, Any], league: str) -> dict[str, Any] | None:
    """One settled Kalshi market (a ``kalshi_markets`` row) as a study leg: the game key shared
    by every series of one game (ticker part after the series), contract kind, strike, the team
    the leg is about and the two names of the rules text ("A vs B")."""
    gk = r["event_ticker"].split("-", 1)[1]
    g = _GK_RX.match(gk)
    if not g:
        return None
    yy, mon, dd, hm, teams = g.groups()
    kind = leg_kind(r["series_ticker"])
    suffix = r["ticker"].rsplit("-", 1)[1]
    vs = _VS_RX.search(r["rules_primary"] or "")
    side_name, strike = None, None
    if kind == "win":
        side_name = r["yes_sub_title"]
    elif kind == "spread":
        sp = _SPREAD_RX.match(r["yes_sub_title"] or "")
        if not sp:
            return None
        side_name, strike = sp.group(1), int(sp.group(2))
    elif kind == "total":
        tt = _TOTAL_RX.match(r["yes_sub_title"] or "")
        if not tt:
            return None
        strike = int(tt.group(1))
    return {
        "ticker": r["ticker"], "event_ticker": r["event_ticker"], "series": r["series_ticker"],
        "league": league, "game_key": f"{league}:{gk}", "kind": kind,
        "code": re.sub(r"\d+$", "", suffix) if kind in ("win", "spread") else None,
        "side_name": side_name, "strike": strike,
        "name_a": vs.group(1) if vs else None, "name_b": vs.group(2) if vs else None,
        "teams": teams, "ticker_date": date(2000 + int(yy), _MONTHS[mon], int(dd)),
        "ticker_start": ticker_start(r["event_ticker"]) if hm else None,
        "y": r["y"], "open_time": r["open_time"], "close_time": r["close_time"],
        "exp_time": r["expected_expiration_time"], "volume": r["volume"],
        "fee_type": r["fee_type"], "fee_multiplier": r["fee_multiplier"],
    }


def kalshi_legs(s, leagues: tuple[str, ...] = tuple(LEAGUES)) -> pl.DataFrame:
    """Every settled leg of the studied series (volume-blind) as :func:`leg_row` rows."""
    rows = []
    for lg in leagues:
        m = F.settled_markets(s, list(LEAGUES[lg]["series"]), datetime(2026, 8, 15, tzinfo=UTC), END)
        rows.extend(x for x in (leg_row(r, lg) for r in m.iter_rows(named=True)) if x is not None)
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- Kalshi game <-> ESPN game ------------------------------------------------------------------
# Kalshi name (normalised) -> ESPN location (normalised), for names no string measure links.
ALIASES = {
    "connecticut": "uconn", "louisiana monroe": "ul monroe", "texas am commerce": "east texas am",
    "university of texas rio grande valley": "ut rio grande valley", "ut san antonio": "utsa",
    "houston baptist": "houston christian", "southern california": "usc", "nebraska omaha": "omaha",
    "miami ohio": "miami oh", "hawaii": "hawai i", "mississippi": "ole miss", "pittsburgh": "pittsburgh",
    "southern methodist": "smu", "central florida": "ucf", "brigham young": "byu", "texas christian": "tcu",
    "louisiana state": "lsu", "parana": "patronato",
}


def _norm(x: str | None) -> str:
    from albert.research.ncaaf import norm_name

    n = norm_name(x)
    return ALIASES.get(n, n)


def name_sim(kname: str | None, row: dict[str, Any], side: str, code: str | None = None) -> float:
    """Similarity of a Kalshi team name to an ESPN side (location, short or full name): the best of
    a character ratio and a club-token overlap, plus a bonus when the leg code equals ESPN's
    abbreviation."""
    from difflib import SequenceMatcher

    from albert.research.soccer import name_score

    k = _norm(kname)
    best = 0.0
    for v in (row[f"{side}_loc"], row[f"{side}_short"], row[f"{side}_name"]):
        n = _norm(v)
        if not n or not k:
            continue
        if k == n:
            return 1.0
        r = SequenceMatcher(None, k, n).ratio()
        if n.startswith(k + " ") or k.startswith(n + " ") or f" {k} " in f" {n} ":
            r = max(r, 0.9)
        best = max(best, r, name_score(kname or "", v or ""))
    if code and code.upper() == (row[f"{side}_abbr"] or "").upper():
        best = max(best, 0.85) + 0.1
    return min(best, 1.0)


def leg_outcome(kind: str, side: str | None, strike: int | None, hs: float, as_: float) -> int | None:
    """Settlement implied by a final score (home ``hs``, away ``as_``) for one leg."""
    if kind == "win":
        if side == "tie":
            return int(hs == as_)
        if side in ("home", "away"):
            mine, other = (hs, as_) if side == "home" else (as_, hs)
            return int(mine > other)
        return None
    if kind == "spread" and side in ("home", "away"):
        mine, other = (hs, as_) if side == "home" else (as_, hs)
        return int(mine - other > strike + 0.5 - 1e-9)
    if kind == "total":
        return int(hs + as_ > strike + 0.5 - 1e-9)
    if kind == "btts":
        return int(hs > 0 and as_ > 0)
    return None


def _leg_side(leg: dict[str, Any], names: dict[str, str]) -> str | None:
    if leg["kind"] not in ("win", "spread"):
        return None
    n = leg["side_name"]
    if leg["kind"] == "win" and (leg["code"] == "TIE" or (n or "").lower() == "tie"):
        return "tie"
    for side in ("home", "away"):
        if n == names[side]:
            return side
    from difflib import SequenceMatcher

    sims = {side: SequenceMatcher(None, _norm(n), _norm(names[side])).ratio() for side in ("home", "away")}
    return max(sims, key=sims.get)


def match_games(legs: pl.DataFrame, games: pl.DataFrame, *, min_name: float = 0.7,
                max_lag_h: float = 3.0) -> pl.DataFrame:
    """Map each Kalshi game (``game_key``) to the ESPN game of the same league within ±1 day of
    the ticker date whose two teams best match the names in the rules text ("A vs B"), whose
    start is within ``max_lag_h`` of the ticker's start time (when the ticker has one) and whose
    final score agrees with every settled leg (the score is used only to confirm the identity of
    the game, never as a model input). Returns one row per leg with game_id and the leg's side."""
    games = games.with_columns(pl.col("start").dt.convert_time_zone("America/New_York").dt.date().alias("et_date"))
    by = {}
    for r in games.iter_rows(named=True):
        by.setdefault((r["league"], r["et_date"]), []).append(r)
    out = []
    for (gk,), g in legs.group_by(["game_key"]):
        lr = g.to_dicts()
        first = lr[0]
        a, b = first["name_a"], first["name_b"]
        codes = {x["side_name"]: x["code"] for x in lr if x["kind"] == "win" and x["code"] != "TIE"}
        day = first["ticker_date"]
        best = None
        for off in (0, -1, 1):
            for row in by.get((first["league"], day + timedelta(days=off)), []):
                for orient in (("home", "away"), ("away", "home")):
                    sa = name_sim(a, row, orient[0], codes.get(a))
                    sb = name_sim(b, row, orient[1], codes.get(b))
                    names = {orient[0]: a, orient[1]: b}
                    bad = n = 0
                    if row["completed"] and row["home_score"] is not None and row["away_score"] is not None:
                        for x in lr:
                            o = leg_outcome(x["kind"], _leg_side(x, names), x["strike"], row["home_score"], row["away_score"])
                            if o is not None:
                                n += 1
                                bad += int(o != x["y"])
                    lag_h = (abs((row["start"] - first["ticker_start"]).total_seconds()) / 3600
                             if first["ticker_start"] is not None and row["start"] is not None else 0.0)
                    score = (min(sa, sb) + 0.02 * (sa + sb) - 0.05 * abs(off) - 0.02 * min(lag_h, 24)
                             - (bad / n if n else 0.5))
                    if best is None or score > best[0]:
                        best = (score, row, names, min(sa, sb), bad, n, lag_h)
        if best is None or best[3] < min_name or best[4] > 0 or best[6] > max_lag_h:
            out.extend({"ticker": x["ticker"], "game_id": None, "side": None,
                        "match_name": None if best is None else round(best[3], 3),
                        "match_bad": None if best is None else best[4]} for x in lr)
            continue
        _, row, names, nm, bad, n, _ = best
        out.extend({"ticker": x["ticker"], "game_id": row["game_id"], "side": _leg_side(x, names),
                    "match_name": round(nm, 3), "match_bad": bad} for x in lr)
    return legs.join(pl.DataFrame(out, infer_schema_length=None), on="ticker", how="left")


# ---- rating models ------------------------------------------------------------------------------
def _weights(start_s: np.ndarray, season: np.ndarray, t_s: float, season_now: int, half_life: float,
             kappa: float) -> np.ndarray:
    age = (t_s - start_s) / 86400.0
    return np.exp(-math.log(2) * age / half_life) * np.where(season < season_now, kappa, 1.0)


def _game_arrays(games: pl.DataFrame, teams: dict[str, int]) -> dict[str, np.ndarray]:
    g = games.filter(pl.col("completed") & pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    return {
        "start": g["start"].dt.epoch("s").to_numpy().astype(float),
        "season": g["season"].to_numpy(),
        "hi": np.array([teams[x] for x in g["home_id"]]), "ai": np.array([teams[x] for x in g["away_id"]]),
        "home": (~g["neutral"]).to_numpy().astype(float),
        "hs": g["home_score"].to_numpy().astype(float), "as": g["away_score"].to_numpy().astype(float),
        "hp": g["home_pts"].fill_null(0).to_numpy().astype(float), "ap": g["away_pts"].fill_null(0).to_numpy().astype(float),
    }


def team_index(games: pl.DataFrame) -> dict[str, int]:
    ids = sorted(set(games["home_id"].to_list()) | set(games["away_id"].to_list()))
    return {x: i for i, x in enumerate(ids)}


def season_of(t: datetime, league: str) -> int:
    """ESPN season year of a date: soccer leagues and volleyball use the calendar year of their
    2026 anchor season; hockey's 2026-27 season is labelled by ESPN with its ending year."""
    return t.year + 1 if league == "ncaah" and t.month >= 8 else t.year


def ridge_fit(A: dict[str, np.ndarray], n: int, t_s: float, season_now: int, prm: dict[str, float]) -> np.ndarray:
    """Massey/ridge on the volleyball point-share logit: z = h·home + r_home − r_away.
    Returns [r_0..r_{n-1}, h]."""
    use = A["start"] + RESULT_LAG.total_seconds() <= t_s
    use &= (A["hp"] + A["ap"]) > 0
    hi, ai, home = A["hi"][use], A["ai"][use], A["home"][use]
    z = np.log(np.clip(A["hp"][use], 1, None) / np.clip(A["ap"][use], 1, None))
    w = _weights(A["start"][use], A["season"][use], t_s, season_now, prm["half_life"], prm["kappa"])
    M = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    np.add.at(M, (hi, hi), w)
    np.add.at(M, (ai, ai), w)
    np.add.at(M, (hi, ai), -w)
    np.add.at(M, (ai, hi), -w)
    np.add.at(M, (hi, n), w * home)
    np.add.at(M, (n, hi), w * home)
    np.add.at(M, (ai, n), -w * home)
    np.add.at(M, (n, ai), -w * home)
    M[n, n] += float((w * home * home).sum()) + 1.0
    np.add.at(b, hi, w * z)
    np.add.at(b, ai, -w * z)
    b[n] = float((w * home * z).sum())
    M[np.arange(n), np.arange(n)] += prm["lam"]
    return np.linalg.solve(M, b)


def poisson_fit(A: dict[str, np.ndarray], n: int, t_s: float, season_now: int, prm: dict[str, float],
                init: np.ndarray | None = None) -> np.ndarray:
    """Weighted ridge Poisson goals model, Newton steps. Parameters [mu, h, a_0..a_{n-1}, d_0..d_{n-1}]:
    log λ_home = mu + h·home + a_home − d_away, log λ_away = mu + a_away − d_home."""
    use = A["start"] + RESULT_LAG.total_seconds() <= t_s
    hi, ai, home = A["hi"][use], A["ai"][use], A["home"][use]
    w = _weights(A["start"][use], A["season"][use], t_s, season_now, prm["half_life"], prm["kappa"])
    m = len(hi)
    rows = np.arange(2 * m)
    X = np.zeros((2 * m, 2 * n + 2))
    X[:, 0] = 1
    X[:m, 1] = home
    X[rows[:m], 2 + hi] = 1
    X[rows[:m], 2 + n + ai] = -1
    X[rows[m:], 2 + ai] = 1
    X[rows[m:], 2 + n + hi] = -1
    y = np.concatenate([A["hs"][use], A["as"][use]])
    ww = np.concatenate([w, w])
    pen = np.full(2 * n + 2, prm["lam"])
    pen[:2] = 1e-6
    beta = init.copy() if init is not None else np.zeros(2 * n + 2)
    if init is None:
        beta[0] = math.log(max(float((ww * y).sum() / max(ww.sum(), 1e-9)), 0.1))
    for _ in range(30):
        eta = np.clip(X @ beta, -5, 5)
        lam = np.exp(eta)
        g = X.T @ (ww * (y - lam)) - pen * beta
        H = X.T @ (X * (ww * lam)[:, None]) + np.diag(pen)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def predict_queries(games: pl.DataFrame, queries: pl.DataFrame, league: str, prm: dict[str, float]) -> pl.DataFrame:
    """Model output for each query (game_id, t, home_id, away_id, neutral), refitted at every
    distinct ``t`` on games that ended (start + RESULT_LAG) by ``t``. Volleyball returns the
    predicted point-share logit ``zhat``; hockey and soccer return ``lam_h`` and ``lam_a``."""
    kind = LEAGUES[league]["kind"]
    teams = team_index(pl.concat([games.select("home_id", "away_id"), queries.select("home_id", "away_id")]))
    A = _game_arrays(games, teams)
    n = len(teams)
    out = []
    beta = None
    for (t,), q in queries.sort("t").group_by(["t"], maintain_order=True):
        t_s = t.timestamp()
        season_now = season_of(t, league)
        hi = np.array([teams[x] for x in q["home_id"]])
        ai = np.array([teams[x] for x in q["away_id"]])
        home = (~q["neutral"]).to_numpy().astype(float)
        if kind == "volleyball":
            beta = ridge_fit(A, n, t_s, season_now, prm)
            zhat = home * beta[n] + beta[hi] - beta[ai]
            out.append(q.with_columns(pl.Series("zhat", zhat)))
        else:
            beta = poisson_fit(A, n, t_s, season_now, prm, beta)
            lh = np.exp(beta[0] + beta[1] * home + beta[2 + hi] - beta[2 + n + ai])
            la = np.exp(beta[0] + beta[2 + ai] - beta[2 + n + hi])
            out.append(q.with_columns(pl.Series("lam_h", lh), pl.Series("lam_a", la)))
    return pl.concat(out)


def score_pmf(lam_h: float, lam_a: float, max_goals: int = 12) -> np.ndarray:
    """Joint pmf of (home goals, away goals) under independent Poisson."""
    k = np.arange(max_goals + 1)
    fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
    ph = np.exp(-lam_h) * lam_h ** k / fact
    pa = np.exp(-lam_a) * lam_a ** k / fact
    return np.outer(ph, pa)


def leg_prob(kind: str, side: str | None, strike: int | None, pmf: np.ndarray, ot_home: float | None = None) -> float | None:
    """Probability of YES for one leg from the score pmf. ``ot_home`` (hockey): share of tied
    regulation-plus-OT scores the home side wins in the shootout."""
    i, j = np.indices(pmf.shape)
    if kind == "win":
        if side == "tie":
            return float(pmf[i == j].sum())
        hw, aw, tie = float(pmf[i > j].sum()), float(pmf[i < j].sum()), float(pmf[i == j].sum())
        if ot_home is not None:
            hw, aw = hw + ot_home * tie, aw + (1 - ot_home) * tie
        return hw if side == "home" else aw if side == "away" else None
    if kind == "spread" and side in ("home", "away"):
        d = (i - j) if side == "home" else (j - i)
        return float(pmf[d >= strike + 1].sum())
    if kind == "total":
        return float(pmf[(i + j) >= strike + 1].sum())
    if kind == "btts":
        return float(pmf[(i >= 1) & (j >= 1)].sum())
    return None


# ---- hyperparameters (chosen on games before the holdout only) -----------------------------------
GRID = {"half_life": (45.0, 90.0, 180.0, 365.0), "lam": (1.0, 3.0, 10.0, 30.0), "kappa": (0.25, 0.5, 1.0)}
# ESPN games scored when choosing hyperparameters: [from, to). Hockey's Kalshi series is October
# only, so its Kalshi dev/holdout split is inside October (see HOCKEY_SPLIT).
SELECT_WINDOW = {"wvb": (datetime(2026, 8, 27, tzinfo=UTC), HOLDOUT),
                 "ncaah": (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 10, 17, tzinfo=UTC)),
                 "engnl": (datetime(2026, 1, 1, tzinfo=UTC), HOLDOUT),
                 "usl": (datetime(2026, 1, 1, tzinfo=UTC), HOLDOUT),
                 "argnacb": (datetime(2026, 1, 1, tzinfo=UTC), HOLDOUT)}
HOCKEY_SPLIT = datetime(2026, 10, 17, tzinfo=UTC)
OT_HOME = 0.5


def holdout_from(league: str) -> datetime:
    return HOCKEY_SPLIT if league == "ncaah" else HOLDOUT


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def fit_k(zhat: np.ndarray, y: np.ndarray) -> float:
    """Slope of P(home win) = σ(k·ẑ), by Newton on the log likelihood."""
    k = 5.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-k * zhat))
        g = float((zhat * (y - p)).sum())
        h = float((zhat ** 2 * p * (1 - p)).sum()) + 1e-9
        k += g / h
        if abs(g / h) < 1e-8:
            break
    return k


def selection_queries(games: pl.DataFrame, league: str) -> pl.DataFrame:
    """Completed ESPN games inside the league's selection window, predicted as of the UTC
    midnight before ``start − 2 h`` (coarser than the Kalshi decision times, so it is fast)."""
    lo, hi = SELECT_WINDOW[league]
    g = games.filter((pl.col("league") == league) & pl.col("completed") & (pl.col("start") >= lo)
                     & (pl.col("start") < hi) & pl.col("home_score").is_not_null())
    if LEAGUES[league]["kind"] == "volleyball":
        g = g.filter((pl.col("home_pts") + pl.col("away_pts")) > 0)
    return g.select("game_id", "home_id", "away_id", "neutral", "home_score", "away_score",
                    (pl.col("start") - timedelta(hours=2)).dt.truncate("1d").alias("t"))


def selection_score(pred: pl.DataFrame, league: str) -> dict[str, float]:
    kind = LEAGUES[league]["kind"]
    hs, as_ = pred["home_score"].to_numpy(), pred["away_score"].to_numpy()
    if kind == "volleyball":
        y = (hs > as_).astype(float)
        z = pred["zhat"].to_numpy()
        k = fit_k(z, y)
        return {"logloss": _ll(1 / (1 + np.exp(-k * z)), y), "k": k, "n": len(y)}
    lh, la = pred["lam_h"].to_numpy(), pred["lam_a"].to_numpy()
    if kind == "hockey":
        p = np.array([leg_prob("win", "home", None, score_pmf(a, b), OT_HOME) for a, b in zip(lh, la)])
        y = (hs > as_).astype(float)
        keep = hs != as_
        return {"logloss": _ll(p[keep], y[keep]), "n": int(keep.sum())}
    P = np.array([[leg_prob("win", s_, None, score_pmf(a, b)) for s_ in ("home", "tie", "away")] for a, b in zip(lh, la)])
    Y = np.column_stack([hs > as_, hs == as_, hs < as_]).astype(float)
    return {"logloss": float(-(Y * np.log(np.clip(P, 1e-4, 1))).sum(axis=1).mean()), "n": len(hs)}


def select_params(games: pl.DataFrame, league: str) -> tuple[dict[str, float], pl.DataFrame]:
    """Grid search on pre-holdout ESPN games; returns the best parameters and the grid."""
    q = selection_queries(games, league)
    lg = games.filter(pl.col("league") == league)
    rows = []
    for hl in GRID["half_life"]:
        for lam in GRID["lam"]:
            for kappa in GRID["kappa"]:
                prm = {"half_life": hl, "lam": lam, "kappa": kappa}
                sc = selection_score(predict_queries(lg, q, league, prm), league)
                rows.append({**prm, **sc})
    grid = pl.DataFrame(rows).sort("logloss")
    best = grid.row(0, named=True)
    return {k: best[k] for k in ("half_life", "lam", "kappa", *(["k"] if "k" in best else []))}, grid


# ---- Kalshi quotes and trades (cached under data/cache/thin_sports) -----------------------------
def _capped_sleep(seconds: float) -> None:
    """Retry waits capped at ~1.5 s: the API rate limit is shared with other studies, so 429s
    arrive at random and a doubling backoff (up to minutes) would stall the fetch."""
    import random

    time.sleep(min(seconds, 1.0) + 0.5 * random.random())


def _api(s):
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, cache_dir(s), min_interval=0.25, max_retries=40,
                                 sleep=_capped_sleep))


QUOTE_WINDOW = (timedelta(hours=27), timedelta(hours=1))
TAIL_WINDOW = (timedelta(hours=3), timedelta(hours=0))   # the last hours before the start (maker fills)


def fetch_candles(s, legs: pl.DataFrame, *, workers: int = 2, max_tickers: int = 100,
                  historical: bool = False, window: tuple[timedelta, timedelta] = QUOTE_WINDOW,
                  name: str = "candles") -> pl.DataFrame:
    """Hourly candles of every leg over [start − 27 h, start − 1 h]. Live markets (settled after
    Kalshi's historical cutoff, 2026-09-06) go in batches of up to 100 tickers sharing one start
    date; historical ones need one request each and are skipped unless ``historical`` (the shared
    rate limit was saturated by other studies on 2026-11-05, so soccer dev starts at the cutoff).
    Rows cached in fundamental_thin_sports_candles."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    path = table_path(s, name)
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=k.CANDLE_SCHEMA)
    done_path = table_path(s, f"{name}_done")
    done = set(pl.read_parquet(done_path)["ticker"].to_list()) if done_path.exists() else set()
    todo = legs.filter(~pl.col("ticker").is_in(list(done)) & pl.col("start").is_not_null()).unique("ticker")
    if not todo.height:
        return have
    api = _api(s)
    cutoff = api.cutoff().get("market_settled_ts")
    jobs: list[tuple[str, list[str], datetime, datetime]] = []
    hist = todo.filter(pl.col("close_time") < cutoff) if cutoff and historical else todo.clear()
    live = todo.filter(pl.col("close_time") >= cutoff) if cutoff else todo
    todo = pl.concat([hist, live])
    for r in hist.iter_rows(named=True):
        jobs.append(("hist", [r["ticker"]], r["start"] - window[0], r["start"] - window[1]))
    live = live.with_columns(pl.col("start").dt.date().alias("d")).sort("d", "start")
    for (_,), grp in live.group_by(["d"], maintain_order=True):
        tick = grp["ticker"].to_list()
        for i in range(0, len(tick), max_tickers):
            part = grp.slice(i, max_tickers)
            jobs.append(("live", part["ticker"].to_list(), part["start"].min() - window[0],
                         part["start"].max() - window[1]))

    def one(job):
        kind, tickers, lo, hi = job
        for attempt in range(5):
            try:
                if kind == "hist":
                    page = api.candles(tickers[0], lo, hi, 60, True)
                    return k.candle_rows(page.payload, tickers[0], 60)
                page = api.candles_batch(tickers, lo, hi, 60)
                out = []
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
            if i % 100 == 0:
                print(f"  candle jobs {i}/{len(jobs)}, {len(rows)} rows", flush=True)
    new = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else pl.DataFrame(schema=k.CANDLE_SCHEMA)
    out = pl.concat([have, new]).unique(["ticker", "end_ts", "period_minutes"])
    write_table(out, path)
    write_table(pl.DataFrame({"ticker": sorted(done | set(todo["ticker"].to_list()))}), done_path)
    return out


def fetch_trades(s, windows: pl.DataFrame, *, workers: int = 2) -> pl.DataFrame:
    """Every public trade of each market inside [t_from, t_to] (same logic as
    ``fundamental.fetch_trades``; raw pages cached under data/cache/thin_sports/kalshi)."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    schema = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
              "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}
    path, done_path = table_path(s, "trades"), table_path(s, "trades_windows")
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=schema)
    done = pl.read_parquet(done_path) if done_path.exists() else pl.DataFrame(
        schema={"ticker": pl.String, "t_from": pl.Datetime("us", "UTC"), "t_to": pl.Datetime("us", "UTC")})
    todo = windows.select("ticker", "t_from", "t_to").unique().join(done, on=["ticker", "t_from", "t_to"], how="anti")
    if not todo.height:
        return have
    api = _api(s)
    cut = api.cutoff()
    cutoff = cut.get("trades_created_ts") or cut.get("market_settled_ts")

    def one(r):
        historical = bool(cutoff and r["t_to"] < cutoff)
        for attempt in range(4):
            try:
                out = []
                for page in api.trades(r["ticker"], historical, min_ts=r["t_from"], max_ts=r["t_to"]):
                    out.extend(k.trade_rows(page.payload))
                return out
            except Exception as error:  # noqa: BLE001  (429 storms on the shared rate limit)
                if attempt == 3:
                    print(f"skip trades {r['ticker']}: {str(error)[:120]}")
                    return []
                time.sleep(10 * (attempt + 1))
        return []

    todo_rows = list(todo.iter_rows(named=True))
    for i in range(0, len(todo_rows), 500):
        part = todo_rows[i:i + 500]
        got: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(one, part):
                got.extend(r)
        new = pl.DataFrame(got, schema=schema) if got else pl.DataFrame(schema=schema)
        have = pl.concat([have, new]).unique("trade_id")
        done = pl.concat([done, pl.DataFrame(part).select("ticker", "t_from", "t_to")])
        write_table(have, path)
        write_table(done, done_path)
        print(f"  trades {min(i + 500, len(todo_rows))}/{len(todo_rows)} windows, {have.height} trades", flush=True)
    return have


# ---- pipeline -------------------------------------------------------------------------------------
def load_games(s, *, fetch: bool = False) -> pl.DataFrame:
    path = table_path(s, "games")
    return fetch_games(s) if fetch or not path.exists() else pl.read_parquet(path)


def matched_legs(s, games: pl.DataFrame) -> pl.DataFrame:
    """Kalshi legs matched to ESPN games, with the ESPN start and team ids (cached)."""
    path = table_path(s, "legs")
    if path.exists():
        return pl.read_parquet(path)
    m = match_games(kalshi_legs(s), games).filter(pl.col("game_id").is_not_null())
    m = m.join(games.select("league", "game_id", "start", "home_id", "away_id", "neutral", "home_score", "away_score"),
               on=["league", "game_id"], how="inner")
    write_table(m, path)
    return m


def game_queries(legs: pl.DataFrame, leads: tuple[int, ...] = LEADS) -> pl.DataFrame:
    g = legs.unique(["league", "game_id"]).select("league", "game_id", "game_key", "start", "home_id", "away_id", "neutral")
    return pl.concat([g.with_columns(pl.lit(h).alias("lead"), (pl.col("start") - timedelta(hours=h)).alias("t"))
                      for h in leads])


def leg_predictions(legs: pl.DataFrame, games: pl.DataFrame, params: dict[str, dict[str, float]]) -> pl.DataFrame:
    """One row per (leg, lead): the harness prediction frame plus league / kind / lead / start."""
    out = []
    for lg in legs["league"].unique().sort():
        q = game_queries(legs.filter(pl.col("league") == lg))
        pr = predict_queries(games.filter(pl.col("league") == lg), q, lg, params[lg])
        kind = LEAGUES[lg]["kind"]
        lgl = legs.filter(pl.col("league") == lg).select(
            "ticker", "game_id", "game_key", "series", "kind", "side", "strike", "y", "fee_type", "fee_multiplier",
            "open_time", "close_time", "volume")
        x = lgl.join(pr.select("game_id", "lead", "t", "start", *(["zhat"] if kind == "volleyball" else ["lam_h", "lam_a"])),
                     on="game_id")
        ps = []
        for r in x.iter_rows(named=True):
            if kind == "volleyball":
                ph = 1 / (1 + math.exp(-params[lg]["k"] * r["zhat"]))
                ps.append(ph if r["side"] == "home" else 1 - ph if r["side"] == "away" else None)
            else:
                pmf = score_pmf(r["lam_h"], r["lam_a"])
                ps.append(leg_prob(r["kind"], r["side"], r["strike"], pmf, OT_HOME if kind == "hockey" else None))
        out.append(x.with_columns(pl.Series("p", ps, dtype=pl.Float64), pl.lit(lg).alias("league"),
                                  pl.col("game_key").alias("event")))
    return pl.concat(out, how="diagonal_relaxed").filter(pl.col("p").is_not_null())


# ---- maker test (trade-print fills) ---------------------------------------------------------------
GATES = (0.0, 0.02, 0.04, 0.06)


def maker_orders(pred: pl.DataFrame, margin: float, *, sides: str = "both", gate: float = 0.0) -> pl.DataFrame:
    """Resting bids around the fair value ``fair`` from ``t`` to ``end``: YES at
    floor(fair − margin) and NO at floor(1 − fair − margin) in whole cents, each capped one tick
    inside the opposite quote (post-only). ``sides="favoured"`` posts only the side the fair value
    favours against the mid; ``gate`` posts only when |fair − mid| ≥ gate."""
    d = pred.filter((pl.col("fair") - pl.col("mid")).abs() >= gate - 1e-12)
    yb = pl.min_horizontal(((pl.col("fair") - margin) * 100 + 1e-9).floor() / 100, (pl.col("ask") - 0.01).round(2))
    nb = pl.min_horizontal(((1 - pl.col("fair") - margin) * 100 + 1e-9).floor() / 100, (1 - pl.col("bid") - 0.01).round(2))
    base = d.select("ticker", "event", "league", "kind", "lead", "t", "end", "bid", "ask", "mid", "fair", "y",
                    "fee_type", "fee_multiplier")
    out = [base.with_columns(pl.lit("yes").alias("side"), yb.alias("price"), (yb > pl.col("bid") + 1e-9).alias("improves")),
           base.with_columns(pl.lit("no").alias("side"), nb.alias("price"), (nb > 1 - pl.col("ask") + 1e-9).alias("improves"))]
    o = pl.concat(out).filter(pl.col("price") >= 0.01)
    if sides == "favoured":
        o = o.filter(pl.when(pl.col("side") == "yes").then(pl.col("fair") > pl.col("mid")).otherwise(pl.col("fair") < pl.col("mid")))
    return o


def fill_orders(orders: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """First fill time of each order. Strict: a later print strictly THROUGH our price (a YES
    seller below our YES bid; a YES buyer above 1 − our NO bid). Lenient (only when our bid
    improved the best bid at ``t``, so we would be alone at the front of the queue): a print AT
    our price also fills. PnL held to settlement, series maker fee."""
    tr = trades.select("ticker", pl.col("created_time").alias("ts"), "yes_price", "taker_side")
    o = orders.with_row_index("oid")
    j = o.select("oid", "ticker", "t", "end", "side", "price").join(tr, on="ticker", how="inner").filter(
        (pl.col("ts") > pl.col("t")) & (pl.col("ts") <= pl.col("end")))
    yes_px = pl.when(pl.col("side") == "yes").then(pl.col("price")).otherwise(1 - pl.col("price"))
    hit_side = pl.when(pl.col("side") == "yes").then(pl.col("taker_side") == "no").otherwise(pl.col("taker_side") == "yes")
    through = pl.when(pl.col("side") == "yes").then(pl.col("yes_price") < yes_px - 1e-9).otherwise(pl.col("yes_price") > yes_px + 1e-9)
    at = pl.when(pl.col("side") == "yes").then(pl.col("yes_price") <= yes_px + 1e-9).otherwise(pl.col("yes_price") >= yes_px - 1e-9)
    j = j.filter(hit_side)
    strict = j.filter(through).group_by("oid").agg(pl.col("ts").min().alias("fill_strict"))
    lenient = j.filter(at).group_by("oid").agg(pl.col("ts").min().alias("fill_at"))
    o = o.join(strict, on="oid", how="left").join(lenient, on="oid", how="left")
    o = o.with_columns(pl.when(pl.col("improves")).then(pl.col("fill_at")).otherwise(pl.col("fill_strict")).alias("fill_lenient"))
    fee = pl.struct("price", "fee_type", "fee_multiplier").map_elements(
        lambda r: F.maker_fee(r["price"], r["fee_type"], r["fee_multiplier"] or 1.0), return_dtype=pl.Float64)
    won = pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y"))
    return o.with_columns((won - pl.col("price") - fee).alias("pnl_if_filled"))


def maker_summary(f: pl.DataFrame, fill_col: str, holdout_from_: datetime) -> list[dict[str, Any]]:
    out = []
    f = f.with_columns((pl.col("t") >= holdout_from_).alias("holdout"))
    for split, part in (("dev", f.filter(~pl.col("holdout"))), ("holdout", f.filter(pl.col("holdout")))):
        filled = part.filter(pl.col(fill_col).is_not_null())
        row: dict[str, Any] = {"split": split, "orders": part.height, "fills": filled.height,
                               "fill_rate": round(filled.height / part.height, 3) if part.height else None}
        if filled.height:
            mu, tt, k = F.clustered_mean(filled["pnl_if_filled"].to_numpy(), filled["event"].to_numpy())
            row.update({"events": k, "pnl_per_fill_c": round(mu * 100, 2), "t": None if tt is None else round(tt, 2),
                        "pnl_per_order_c": round(float(filled["pnl_if_filled"].sum()) / part.height * 100, 3),
                        "yes_share": round(float((filled["side"] == "yes").mean()), 3)})
        out.append(row)
    return out


def maker_grid(pred: pl.DataFrame, trades: pl.DataFrame | None, holdout_from_: datetime, *,
               ranges: pl.DataFrame | None = None, margins: tuple[float, ...] = F.MARGINS,
               gates: tuple[float, ...] = GATES, min_fills: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every (sides, fill rule, gate, margin) on dev and holdout; per (sides, fill rule) the
    (gate, margin) with the best dev PnL per fill (≥ ``min_fills`` dev fills) is the choice.
    Fills come from public trade prints (``trades``) or, when given, hourly candle trade ranges
    (``ranges``, see :func:`candle_fill_orders`)."""
    rows = []
    for sides in ("both", "favoured"):
        for gate in gates:
            for m in margins:
                o = maker_orders(pred, m, sides=sides, gate=gate)
                if not o.height:
                    continue
                f = candle_fill_orders(o, ranges) if ranges is not None else fill_orders(o, trades)
                for rule, col in (("strict", "fill_strict"), ("lenient", "fill_lenient")):
                    for r in maker_summary(f, col, holdout_from_):
                        rows.append({"sides": sides, "fill": rule, "gate": gate, "margin": m, **r})
    grid = pl.DataFrame(rows, infer_schema_length=None)
    chosen = []
    for (sides, rule), g in grid.group_by(["sides", "fill"], maintain_order=True):
        dev = g.filter((pl.col("split") == "dev") & (pl.col("fills") >= min_fills))
        if not dev.height:
            continue
        b = dev.sort("pnl_per_fill_c", descending=True).row(0, named=True)
        h = g.filter((pl.col("split") == "holdout") & (pl.col("gate") == b["gate"]) & (pl.col("margin") == b["margin"]))
        chosen.append({"sides": sides, "fill": rule, "gate": b["gate"], "margin": b["margin"],
                       "dev_fills": b["fills"], "dev_pnl_fill_c": b["pnl_per_fill_c"], "dev_t": b["t"],
                       **({f"hold_{k}": v for k, v in h.row(0, named=True).items()
                           if k in ("orders", "fills", "fill_rate", "events", "pnl_per_fill_c", "t", "pnl_per_order_c")}
                          if h.height else {})})
    return grid, pl.DataFrame(chosen, infer_schema_length=None)


# ---- evaluation ----------------------------------------------------------------------------------
def quotes(s, pred: pl.DataFrame) -> pl.DataFrame:
    """Prediction rows with the executable quote at ``t`` (last hourly candle closed by ``t``)."""
    c = pl.read_parquet(table_path(s, "candles"))
    q = F.attach_quotes(pred, c)
    q = q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"))
    return q.with_columns(pl.when(pl.col("spread") <= 0.05 + 1e-9).then(pl.lit("a: <=5c"))
                          .when(pl.col("spread") <= 0.15 + 1e-9).then(pl.lit("b: 6-15c"))
                          .when(pl.col("spread") <= 0.50 + 1e-9).then(pl.lit("c: 16-50c"))
                          .otherwise(pl.lit("d: >50c")).alias("spread_band"))


def blend_coefs(dev: pl.DataFrame, *, use_model: bool = True) -> np.ndarray:
    X = [np.ones(dev.height), F.logit(dev["mid"].to_numpy())]
    if use_model:
        X.append(F.logit(dev["p"].to_numpy()))
    beta, _ = F.fit_logistic(np.column_stack(X), dev["y"].to_numpy().astype(float), dev["event"].to_numpy())
    return beta


def apply_blend(df: pl.DataFrame, beta: np.ndarray) -> np.ndarray:
    X = [np.ones(df.height), F.logit(df["mid"].to_numpy())]
    if len(beta) == 3:
        X.append(F.logit(df["p"].to_numpy()))
    return 1 / (1 + np.exp(-np.column_stack(X) @ beta))


def fair_values(q: pl.DataFrame, *, min_dev: int = 150) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Dev-fitted fair values: ``fair`` = blend of mid and model, fitted per (league, kind) when
    that cell has ≥ ``min_dev`` dev rows, else per league; ``fair_band`` = the same blend fitted
    per (league, spread > 15c at t), since the mid means little in a wide book; ``fair_mid`` = the
    mid recalibrated alone (the control). Coefficients never see holdout rows."""
    out, coefs = [], []
    for (lg,), part in q.group_by(["league"], maintain_order=True):
        hf = holdout_from(lg)
        dev_all = part.filter(pl.col("t") < hf)
        for (kind,), cell in part.group_by(["kind"], maintain_order=True):
            dev = cell.filter(pl.col("t") < hf)
            src = dev if dev.height >= min_dev else dev_all
            if src.height < 30:
                continue
            b, b0 = blend_coefs(src), blend_coefs(src, use_model=False)
            coefs.append({"league": lg, "kind": kind, "fit_on": "kind" if src is dev else "league", "dev_rows": src.height,
                          "a": round(float(b[0]), 3), "b_mid": round(float(b[1]), 3), "c_model": round(float(b[2]), 3)})
            out.append(cell.with_columns(pl.Series("fair", apply_blend(cell, b)), pl.Series("fair_mid", apply_blend(cell, b0)),
                                         (pl.col("t") >= hf).alias("holdout")))
    fv = pl.concat(out).with_columns((pl.col("spread") > 0.15 + 1e-9).alias("wide"))
    band = []
    for (lg, wide), part in fv.group_by(["league", "wide"], maintain_order=True):
        dev = part.filter(~pl.col("holdout"))
        if dev.height < 30:
            band.append(part.with_columns(pl.col("fair").alias("fair_band")))
            continue
        b = blend_coefs(dev)
        coefs.append({"league": lg, "kind": "all, wide" if wide else "all, tight", "fit_on": "band", "dev_rows": dev.height,
                      "a": round(float(b[0]), 3), "b_mid": round(float(b[1]), 3), "c_model": round(float(b[2]), 3)})
        band.append(part.with_columns(pl.Series("fair_band", apply_blend(part, b))))
    return pl.concat(band), coefs


def taker_table(q: pl.DataFrame, by: list[str]) -> list[dict[str, Any]]:
    """Harness ``evaluate`` per group: Brier model vs mid, blend c (dev) and its holdout refit,
    taker PnL at the dev-chosen θ on the holdout."""
    rows = []
    for key, part in q.group_by(by, maintain_order=True):
        lg = part["league"][0]
        r = F.evaluate(part, holdout_from(lg))
        sc = {x["split"]: x for x in r.get("scores", [])}
        bl = r.get("blend", {})
        th = r.get("theta_chosen_on_dev")
        tk = {(x["theta"], x["split"]): x for x in r.get("taker", [])}
        row = dict(zip(by, key))
        for split in ("dev", "holdout"):
            x = sc.get(split, {})
            row.update({f"{split}_n": x.get("n"), f"{split}_events": x.get("events"),
                        f"{split}_brier_model": x.get("brier_model"), f"{split}_brier_mid": x.get("brier_mid")})
        row.update({"c": bl.get("c_model"), "t_c": bl.get("t_c"), "b_mid": bl.get("b_mid"),
                    "c_hold_refit": bl.get("c_model_holdout_refit"), "t_c_hold_refit": bl.get("t_c_holdout_refit"),
                    "brier_blend_hold": bl.get("brier_blend_holdout"), "brier_mid_hold": bl.get("brier_mid_holdout"),
                    "theta": th})
        for split in ("dev", "holdout"):
            x = tk.get((th, split), {}) if th is not None else {}
            row.update({f"taker_{split}_trades": x.get("trades"), f"taker_{split}_pnl_c": x.get("pnl_c"),
                        f"taker_{split}_t": x.get("t")})
        rows.append(row)
    return rows


def maker_frame(fv: pl.DataFrame, fair_col: str = "fair") -> pl.DataFrame:
    """Rows for the maker test: the 24 h order rests until the 2 h decision, the 2 h order until
    the scheduled start (one bid per side per market at any moment)."""
    end = pl.when(pl.col("lead") == 24).then(pl.col("start") - timedelta(hours=2)).otherwise(pl.col("start"))
    return fv.with_columns(end.alias("end"), pl.col(fair_col).alias("fair"))


def volume_table(legs: pl.DataFrame, ranges: pl.DataFrame) -> pl.DataFrame:
    """Per league: Kalshi lifetime volume per game (all legs) and the contracts printed in the
    24 h before the start (hourly candle volume) — what a 1-lot could meet."""
    pre = ranges.join(legs.select("ticker", "league", "game_key", "start"), on="ticker").filter(
        (pl.col("end_ts") > pl.col("start") - timedelta(hours=24)) & (pl.col("end_ts") <= pl.col("start")))
    pre_leg = pre.group_by("ticker").agg(pl.col("volume").sum().alias("pre_contracts"),
                                         (pl.col("volume") > 0).sum().alias("pre_active_hours"))
    lg = legs.join(pre_leg, on="ticker", how="left").fill_null(0)
    g = lg.group_by("league", "game_key").agg(pl.col("volume").sum(), pl.len().alias("legs"), pl.col("pre_contracts").sum())
    per_leg = lg.group_by("league").agg(pl.col("volume").median().round(0).alias("vol_per_leg_median"),
                                        pl.col("pre_contracts").median().round(0).alias("pre24h_contracts_per_leg_median"),
                                        pl.col("pre_active_hours").median().alias("pre24h_hours_with_prints_per_leg_median"))
    return g.group_by("league").agg(
        pl.len().alias("games"), pl.col("legs").mean().round(1).alias("legs_per_game"),
        pl.col("volume").median().round(0).alias("vol_per_game_median"), pl.col("volume").mean().round(0).alias("vol_per_game_mean"),
        pl.col("pre_contracts").median().round(0).alias("pre24h_contracts_per_game_median"),
    ).join(per_leg, on="league").sort("league")


def save_holdout(s, fv: pl.DataFrame) -> Path:
    path = s.data_dir / "cache" / f"fundamental_{NAME}_holdout_preds.parquet"
    h = fv.filter(pl.col("holdout")).select("ticker", "t", pl.col("fair").alias("p_fair"), "mid", pl.col("p").alias("p_model"),
                                            "bid", "ask", "y", "league", "kind", "lead", "event")
    write_table(h, path)
    return path


def run(s, *, fetch_trades_too: bool = False) -> dict[str, Any]:
    """Score every league (taker + blend), build dev-fitted fair values, run the maker grid and
    write the report. Inputs: the cached games, legs, predictions and candles (``fetch_games``,
    ``matched_legs``, ``leg_predictions``, ``fetch_candles`` with QUOTE_WINDOW and TAIL_WINDOW).
    Maker fills come from hourly candle trade ranges unless ``fetch_trades_too`` (one trades
    request per market: ~7,400 requests, too slow on the shared rate limit of 2026-11-05)."""
    pred = pl.read_parquet(table_path(s, "pred"))
    q = quotes(s, pred)
    out: dict[str, Any] = {"rows_with_quotes": q.height, "rows_total": pred.height}
    out["taker_league"] = taker_table(q, ["league"])
    out["taker_league_kind"] = taker_table(q, ["league", "kind"])
    out["taker_league_lead"] = taker_table(q, ["league", "lead"])
    out["taker_league_spread"] = taker_table(q.sort("league", "spread_band"), ["league", "spread_band"])
    fv, coefs = fair_values(q)
    out["fair_coefs"] = coefs
    save_holdout(s, fv)
    legs = pl.read_parquet(table_path(s, "legs"))
    ranges = candle_ranges(s)
    have_c = set(ranges["ticker"].unique().to_list())   # live markets (candles fetched); historical ones would read as 0
    out["volume"] = volume_table(legs.filter(pl.col("ticker").is_in(list(have_c))), ranges).to_dicts()
    trades = None
    if fetch_trades_too:
        win = legs.filter(pl.col("ticker").is_in(fv["ticker"].unique().to_list())).select(
            "ticker", (pl.col("start") - timedelta(hours=24)).alias("t_from"), pl.col("start").alias("t_to"))
        trades = fetch_trades(s, win)
    makers, chosen = [], []
    for (lg,), part in fv.group_by(["league"], maintain_order=True):
        for fair_col in ("fair", "fair_band", "fair_mid"):
            grid, ch = maker_grid(maker_frame(part, fair_col), trades, holdout_from(lg), ranges=None if trades is not None else ranges)
            makers.append(grid.with_columns(pl.lit(lg).alias("league"), pl.lit(fair_col).alias("fair_value")))
            if ch.height:
                chosen.append(ch.with_columns(pl.lit(lg).alias("league"), pl.lit(fair_col).alias("fair_value")))
    out["maker_grid"] = pl.concat(makers, how="diagonal_relaxed")
    out["maker_chosen"] = pl.concat(chosen, how="diagonal_relaxed") if chosen else pl.DataFrame()
    write_table(out["maker_grid"], table_path(s, "maker_grid"))
    return out


def write_report(s, out: dict[str, Any], params: dict[str, Any]) -> Path:
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    rep.json("Run", {"rows_total": out["rows_total"], "rows_with_quotes": out["rows_with_quotes"],
                     "holdout_from": HOLDOUT.isoformat(), "hockey_holdout_from": HOCKEY_SPLIT.isoformat(),
                     "leads_h": list(LEADS), "result_lag_h": RESULT_LAG.total_seconds() / 3600, "params": params})
    rep.table("Volume per game (contracts): lifetime, and printed in the 24 h before the start", pl.DataFrame(out["volume"]))
    rep.table("Taker + blend by league (blend logit P = a + b·logit(mid) + c·logit(model), fitted on dev)",
              pl.DataFrame(out["taker_league"], infer_schema_length=None))
    rep.table("By league and contract", pl.DataFrame(out["taker_league_kind"], infer_schema_length=None))
    rep.table("By league and lead (hours before start)", pl.DataFrame(out["taker_league_lead"], infer_schema_length=None))
    rep.table("By league and quoted spread at t (the information sits where the book is wide)",
              pl.DataFrame(out["taker_league_spread"], infer_schema_length=None))
    rep.table("Fair-value blend coefficients (dev)", pl.DataFrame(out["fair_coefs"]))
    if out["maker_chosen"].height:
        rep.table("Maker: (gate, margin) chosen on dev per variant; holdout at that choice", out["maker_chosen"])
    if out.get("fill_validation") is not None and out["fill_validation"].height:
        rep.table("Fill proxy check: hourly-candle fills vs real trade prints (markets with a cached full trades window)",
                  out["fill_validation"])
    rep.table("Maker grid (all variants)", out["maker_grid"])
    return rep.save()


# ---- maker fills from hourly candle trade ranges (rate-limit fallback) -----------------------------
def _dollars(node: dict[str, Any] | None, field: str) -> float | None:
    if not node:
        return None
    v = node.get(f"{field}_dollars", node.get(field))
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def candle_ranges(s) -> pl.DataFrame:
    """Hourly trade-price ranges (low / high of printed prices, volume) for every candle this
    study fetched, re-read from the raw responses cached under data/cache/thin_sports/kalshi."""
    from albert.http import read_raw

    rows = []
    for f in (cache_dir(s) / "kalshi").glob("*.json.zst"):
        rec = read_raw(f)
        if "candlesticks" not in rec.get("url", ""):
            continue
        resp = rec.get("response") or {}
        if "markets" in resp:
            parts = [(m.get("market_ticker"), m.get("candlesticks") or []) for m in resp["markets"]]
        else:
            parts = [(rec["url"].rstrip("/").split("/")[-2], resp.get("candlesticks") or [])]
        for tk, cs in parts:
            for c in cs:
                pr = c.get("price") or {}
                vol = c.get("volume_fp", c.get("volume"))
                rows.append({"ticker": tk, "end_ts": datetime.fromtimestamp(c["end_period_ts"], UTC),
                             "low": _dollars(pr, "low"), "high": _dollars(pr, "high"),
                             "volume": float(vol) if vol is not None else 0.0})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "low": pl.Float64,
                                      "high": pl.Float64, "volume": pl.Float64}).unique(["ticker", "end_ts"])


def candle_fill_orders(orders: pl.DataFrame, ranges: pl.DataFrame, *, period: timedelta = timedelta(hours=1)) -> pl.DataFrame:
    """Fills judged from hourly candles fully inside (t, end]: strict = a print strictly through
    our price (low < YES bid; high > 1 − NO bid); a resting best bid is always hit before any
    lower print, so this implies a fill without needing the taker side. Lenient = a print at our
    price when our bid improved the best bid at ``t``. Hourly granularity drops the partial hours
    at both ends (conservative)."""
    r = ranges.filter(pl.col("volume") > 0).select("ticker", "end_ts", "low", "high")
    o = orders.with_row_index("oid")
    j = o.select("oid", "ticker", "t", "end", "side", "price").join(r, on="ticker", how="inner").filter(
        (pl.col("end_ts") - period >= pl.col("t")) & (pl.col("end_ts") <= pl.col("end")))
    through = pl.when(pl.col("side") == "yes").then(pl.col("low") < pl.col("price") - 1e-9).otherwise(
        pl.col("high") > 1 - pl.col("price") + 1e-9)
    at = pl.when(pl.col("side") == "yes").then(pl.col("low") <= pl.col("price") + 1e-9).otherwise(
        pl.col("high") >= 1 - pl.col("price") - 1e-9)
    strict = j.filter(through).group_by("oid").agg(pl.col("end_ts").min().alias("fill_strict"))
    lenient = j.filter(at).group_by("oid").agg(pl.col("end_ts").min().alias("fill_at"))
    o = o.join(strict, on="oid", how="left").join(lenient, on="oid", how="left")
    o = o.with_columns(pl.when(pl.col("improves")).then(pl.col("fill_at")).otherwise(pl.col("fill_strict")).alias("fill_lenient"))
    fee = pl.struct("price", "fee_type", "fee_multiplier").map_elements(
        lambda x: F.maker_fee(x["price"], x["fee_type"], x["fee_multiplier"] or 1.0), return_dtype=pl.Float64)
    won = pl.when(pl.col("side") == "yes").then(pl.col("y")).otherwise(1 - pl.col("y"))
    return o.with_columns((won - pl.col("price") - fee).alias("pnl_if_filled"))


def cached_trades(s) -> pl.DataFrame:
    """Trade prints of every complete (single-page, no cursor) trades response in this study's raw
    cache, with the requested window — for checking the candle fill proxy against real prints."""
    from albert.http import read_raw
    from albert.venues import kalshi as k

    rows, wins = [], []
    for f in (cache_dir(s) / "kalshi").glob("*.json.zst"):
        rec = read_raw(f)
        if not rec.get("url", "").endswith("/trades") or (rec.get("response") or {}).get("cursor"):
            continue
        prm = rec.get("params") or {}
        if "cursor" in prm or "min_ts" not in prm:
            continue
        wins.append({"ticker": prm["ticker"], "t_from": datetime.fromtimestamp(int(prm["min_ts"]), UTC),
                     "t_to": datetime.fromtimestamp(int(prm["max_ts"]), UTC)})
        rows.extend(k.trade_rows(rec["response"]))
    schema = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
              "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}
    return (pl.DataFrame(rows, schema=schema).unique("trade_id"),
            pl.DataFrame(wins, schema={"ticker": pl.String, "t_from": pl.Datetime("us", "UTC"),
                                       "t_to": pl.Datetime("us", "UTC")}).unique())


def validate_fills(fv: pl.DataFrame, ranges: pl.DataFrame, trades: pl.DataFrame, windows: pl.DataFrame,
                   margins: tuple[float, ...] = (0.0, 0.03, 0.08)) -> pl.DataFrame:
    """Candle-range fills vs trade-print fills on the markets whose full trades window is cached
    (orders posted around the dev-fitted ``fair``, both sides)."""
    rows = []
    for m in margins:
        o = maker_orders(maker_frame(fv, "fair"), m).join(windows, on="ticker").filter(
            (pl.col("t_from") <= pl.col("t")) & (pl.col("t_to") >= pl.col("end"))).drop("t_from", "t_to")
        if not o.height:
            continue
        ft = fill_orders(o, trades).select("oid", "league", "pnl_if_filled", "fill_strict", "fill_lenient")
        fc = candle_fill_orders(o, ranges).select("oid", pl.col("fill_strict").alias("c_strict"),
                                                  pl.col("fill_lenient").alias("c_lenient"))
        j = ft.join(fc, on="oid")
        for (lg,), g in j.group_by(["league"], maintain_order=True):
            row = {"league": lg, "margin": m, "orders": g.height}
            for rule in ("strict", "lenient"):
                tcol, ccol = f"fill_{rule}", f"c_{rule}"
                both = g.filter(pl.col(tcol).is_not_null() & pl.col(ccol).is_not_null())
                row.update({f"{rule}_trade_fills": int(g[tcol].is_not_null().sum()),
                            f"{rule}_candle_fills": int(g[ccol].is_not_null().sum()), f"{rule}_both": both.height,
                            f"{rule}_trade_pnl_c": round(float(g.filter(pl.col(tcol).is_not_null())["pnl_if_filled"].mean() or 0) * 100, 2),
                            f"{rule}_candle_pnl_c": round(float(g.filter(pl.col(ccol).is_not_null())["pnl_if_filled"].mean() or 0) * 100, 2)})
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)
