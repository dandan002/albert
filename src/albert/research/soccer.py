"""Soccer 1X2: goal models and an outside-market consensus versus Kalshi home / tie / away legs.

Question: does a Dixon-Coles goal model, or a de-vigged outside price, disagree with Kalshi's
soccer moneylines in a way that pays after fees, out of sample (October 2026)?

Data and as-of timing (every input must be public before the decision time ``t``):

* **Kalshi** ``*GAME`` series (home / away / tie legs). The kickoff is
  ``expected_expiration_time − 3h``: checked on all 77 EPL games against the published fixture
  times (offset exactly 3.0h for every game) and on the Polymarket ``gameStartTime`` of matched
  games (98.9% within 15 min; the rest were rescheduled). The first team code in the event
  ticker is the home side (77/77 EPL fixtures).
* **Results** for the goal model: openfootball/football.json (public domain, GitHub) for past
  seasons, plus current-season scores rebuilt from Kalshi's own settled TOTAL ("over k.5 goals")
  and SPREAD ("wins by more than k.5") ladders together with the GAME result. A result enters a
  fit only if its kickoff is at least ``RESULT_LAG`` before the fit time (a match is over and
  reported within ~2.5h of kickoff).
* **Outside consensus:** football-data.co.uk (the source the study brief named) disallows Claude
  agents in robots.txt (``Claude-Web``, ``ClaudeBot``, ``anthropic-ai``), so it is not used. The
  public substitute is Polymarket international (data only; never traded): its home / draw /
  away "Yes" prices from ``/prices-history``, read at the last bucket that *ended* by ``t``,
  de-vigged three ways (multiplicative, power, Shin). Polymarket soccer books are made by
  market makers who quote off the sharp books, so this is the nearest public stand-in for a
  sharp consensus; it is a market price, not a bookmaker's.

Decision times: kickoff − 24h and kickoff − 2h (both strictly before kickoff), quotes from
the harness (last hourly candle that closed by ``t``).
"""

import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research import fundamental as F

NAME = "soccer"
HOLDOUT = datetime(2026, 10, 1, tzinfo=UTC)
RESULT_LAG = timedelta(hours=3)
KICKOFF_OFFSET = timedelta(hours=3)          # GAME expected_expiration_time − kickoff
LEADS = (24, 2)                              # hours before kickoff
OF_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"

# Kalshi GAME series -> (league key, openfootball league code or None, kickoff timezone of the
# openfootball fixture times). Domestic leagues only for the goal model; cups/UEFA use the
# Polymarket comparison only.
LEAGUES: dict[str, tuple[str, str | None]] = {
    "KXEPLGAME": ("EPL", "en.1"),
    "KXEFLCHAMPIONSHIPGAME": ("ENG2", "en.2"),
    "KXEFLL1GAME": ("ENG3", "en.3"),
    "KXBUNDESLIGAGAME": ("GER1", "de.1"),
    "KXBUNDESLIGA2GAME": ("GER2", "de.2"),
    "KXLALIGAGAME": ("ESP1", "es.1"),
    "KXLALIGA2GAME": ("ESP2", "es.2"),
    "KXSERIEAGAME": ("ITA1", "it.1"),
    "KXSERIEBGAME": ("ITA2", "it.2"),
    "KXLIGUE1GAME": ("FRA1", "fr.1"),
    "KXLIGUE2GAME": ("FRA2", "fr.2"),
    "KXEREDIVISIEGAME": ("NED1", "nl.1"),
    "KXLIGAPORTUGALGAME": ("POR1", "pt.1"),
    "KXBELGIANPLGAME": ("BEL1", "be.1"),
    "KXSCOTTISHPREMGAME": ("SCO1", "sco.1"),
    "KXSUPERLIGGAME": ("TUR1", "tr.1"),
    "KXSLGREECEGAME": ("GRE1", "gr.1"),
    "KXAUTBUNDESGAME": ("AUT1", "at.1"),
    "KXBRASILEIROGAME": ("BRA1", "br.1"),
    "KXMLSGAME": ("MLS", None),
    "KXLIGAMXGAME": ("MEX1", None),
    "KXARGPREMDIVGAME": ("ARG1", None),
    "KXUCLGAME": ("UCL", None),
    "KXUELGAME": ("UEL", None),
    "KXUECLGAME": ("UECL", None),
}
OF_SEASONS = ("2023-24", "2024-25", "2025-26", "2026-27")
OF_YEARS = ("2025", "2026")                  # calendar-year leagues (Brazil)


# ---- Kalshi games ----------------------------------------------------------------------------
_EV_RX = re.compile(r"^(?P<series>KX[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<teams>[A-Z0-9]+)$")


def split_home_away(teams: str, codes: list[str]) -> tuple[str, str] | None:
    """Home and away leg codes from the ticker's concatenated team codes (home first).

    ``split_home_away("AVLBRE", ["AVL", "BRE"]) == ("AVL", "BRE")``. Returns None when the codes
    do not concatenate to ``teams`` in either order."""
    if len(codes) != 2:
        return None
    a, b = codes
    if teams == a + b:
        return a, b
    if teams == b + a:
        return b, a
    return None


def game_frame(markets: pl.DataFrame) -> pl.DataFrame:
    """One row per settled three-leg game: home / away / tie tickers, names, result, kickoff."""
    rows = []
    for (ev,), g in markets.group_by(["event_ticker"]):
        m = _EV_RX.match(ev or "")
        if not m or g.height != 3:
            continue
        legs = {t.rsplit("-", 1)[1]: r for t, r in zip(g["ticker"], g.iter_rows(named=True))}
        if "TIE" not in legs:
            continue
        ha = split_home_away(m["teams"], [c for c in legs if c != "TIE"])
        if ha is None:
            continue
        h, a, tie = legs[ha[0]], legs[ha[1]], legs["TIE"]
        won = [k for k, r in (("H", h), ("A", a), ("D", tie)) if r["result"] == "yes"]
        if len(won) != 1:
            continue
        exp = tie["expected_expiration_time"]
        rows.append({
            "event": ev, "series": m["series"], "league": LEAGUES.get(m["series"], (m["series"], None))[0],
            "kickoff": exp - KICKOFF_OFFSET if exp else None,
            "home": h["yes_sub_title"], "away": a["yes_sub_title"],
            "home_ticker": h["ticker"], "away_ticker": a["ticker"], "tie_ticker": tie["ticker"],
            "outcome": won[0], "open_time": min(r["open_time"] for r in (h, a, tie)),
            "close_time": max(r["close_time"] for r in (h, a, tie)),
        })
    return pl.DataFrame(rows).sort("kickoff") if rows else pl.DataFrame()


def score_from_ladders(outcome: str, totals: dict[int, bool], margins: dict[int, bool]) -> tuple[int, int] | None:
    """Final score from settled ladders: ``totals[k]`` = "over k−0.5 goals" settled YES,
    ``margins[k]`` = "the winner wins by more than k−0.5" settled YES (winner's own spread legs).

    Each settled strike bounds the total T (YES: T ≥ k, NO: T ≤ k − 1) and the winning margin
    M (draw: M = 0; otherwise M ≥ 1 and the winner's spread strikes bound it the same way);
    M has the parity of T and M ≤ T. Returns the score only when exactly one (T, M) pair is
    consistent with every settled strike, otherwise None."""
    if not totals or outcome not in ("H", "A", "D"):
        return None
    t_lo = max([k for k, v in totals.items() if v], default=0)
    t_hi = min([k - 1 for k, v in totals.items() if not v], default=None)
    if t_hi is None or t_lo > t_hi:
        return None
    m_lo, m_hi = (0, 0) if outcome == "D" else (1, None)
    if outcome != "D":
        m_lo = max([k for k, v in margins.items() if v] + [1])
        m_hi = min([k - 1 for k, v in margins.items() if not v], default=None)
    pairs = [(t, m) for t in range(t_lo, t_hi + 1)
             for m in range(m_lo, (t if m_hi is None else min(m_hi, t)) + 1) if (t - m) % 2 == 0]
    if len(pairs) != 1:
        return None
    t, m = pairs[0]
    w, l_ = (t + m) // 2, (t - m) // 2
    return (w, l_) if outcome in ("H", "D") else (l_, w)


def kalshi_scores(markets: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Scores rebuilt from each game's settled TOTAL and SPREAD ladders (same event suffix)."""
    games = games.with_columns(pl.col("event").str.split_exact("-", 1).struct.field("field_1").alias("suffix"))
    by_suffix = {r["suffix"]: r for r in games.iter_rows(named=True)}
    tot: dict[str, dict[int, bool]] = defaultdict(dict)
    spr: dict[str, dict[str, dict[int, bool]]] = defaultdict(lambda: defaultdict(dict))
    for r in markets.select("ticker", "series_ticker", "result").iter_rows(named=True):
        parts = r["ticker"].split("-")
        if len(parts) != 3 or parts[1] not in by_suffix or r["result"] not in ("yes", "no"):
            continue
        g = by_suffix[parts[1]]
        base = g["series"][:-4]
        if r["series_ticker"] == base + "TOTAL" and parts[2].isdigit():
            tot[parts[1]][int(parts[2])] = r["result"] == "yes"
        elif r["series_ticker"] == base + "SPREAD":
            mm = re.match(r"^([A-Z0-9]+?)(\d+)$", parts[2])
            if mm:
                spr[parts[1]][mm.group(1)][int(mm.group(2))] = r["result"] == "yes"
    rows = []
    for suf, g in by_suffix.items():
        code = {"H": g["home_ticker"], "A": g["away_ticker"]}.get(g["outcome"], "").rsplit("-", 1)[-1]
        sc = score_from_ladders(g["outcome"], tot.get(suf, {}), spr[suf].get(code, {}) if suf in spr else {})
        if sc is not None:
            rows.append({"event": g["event"], "hg": sc[0], "ag": sc[1]})
    return pl.DataFrame(rows, schema={"event": pl.String, "hg": pl.Int64, "ag": pl.Int64})


# ---- openfootball results --------------------------------------------------------------------
def openfootball_dir(s) -> Path:
    return s.data_dir / "cache" / NAME / "openfootball"


def fetch_openfootball(s, codes: list[str]) -> list[Path]:
    """Download (once) each season file of these leagues; missing files are skipped."""
    import time

    import httpx

    out = openfootball_dir(s)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "albert-research/0.2"}) as http:
        for code in codes:
            for season in OF_SEASONS + OF_YEARS:
                path = out / f"{season}.{code}.json"
                if not path.exists():
                    r = http.get(f"{OF_BASE}/{season}/{code}.json")
                    time.sleep(0.5)
                    if r.status_code != 200:
                        continue
                    path.write_bytes(r.content)
                paths.append(path)
    return paths


def openfootball_rows(payload: dict[str, Any], code: str, season: str) -> list[dict[str, Any]]:
    """Played matches (full-time score present) of one openfootball season file."""
    rows = []
    for m in payload.get("matches") or []:
        ft = (m.get("score") or {}).get("ft") if isinstance(m.get("score"), dict) else None
        if not ft or len(ft) != 2 or m.get("date") is None:
            continue
        rows.append({"code": code, "season": season, "date": date.fromisoformat(m["date"]),
                     "home": m["team1"], "away": m["team2"], "hg": int(ft[0]), "ag": int(ft[1])})
    return rows


def openfootball_matches(s) -> pl.DataFrame:
    rows = []
    for path in sorted(openfootball_dir(s).glob("*.json")):
        season, code = path.name.split(".", 1)[0], path.name.split(".", 1)[1][:-5]
        rows += openfootball_rows(json.loads(path.read_text()), code, season)
    return pl.DataFrame(rows).unique(["code", "date", "home", "away"]) if rows else pl.DataFrame()


# ---- team names ------------------------------------------------------------------------------
_STOP = {"fc", "afc", "cf", "sc", "ac", "as", "ss", "us", "sv", "vfl", "vfb", "tsg", "fk", "sk", "cd",
         "ud", "rc", "rcd", "sd", "club", "de", "del", "la", "1", "1.", "calcio", "ssc", "acf", "bk",
         "if", "kv", "krc", "rsc", "royal", "sporting", "futebol", "clube", "sad", "the", "and"}


def norm_tokens(name: str) -> set[str]:
    x = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    toks = {t for t in x.split() if t and t not in _STOP and not t.isdigit()}
    return toks or set(x.split())


def name_score(a: str, b: str) -> float:
    """Similarity of two club names: token overlap, with prefix matches counted ("Nottingham"
    vs "Nottingham Forest FC", "Leverkusen" vs "Bayer 04 Leverkusen")."""
    ta, tb = norm_tokens(a), norm_tokens(b)
    if not ta or not tb:
        return 0.0
    hit = 0.0
    for t in ta:
        if t in tb:
            hit += 1
        elif any(len(t) >= 4 and (u.startswith(t) or t.startswith(u)) for u in tb if len(u) >= 4):
            hit += 0.8
    return hit / min(len(ta), len(tb)) - 0.05 * abs(len(ta) - len(tb))


def map_names(kalshi: list[str], ref: list[str], overrides: dict[str, str] | None = None,
              threshold: float = 0.75) -> dict[str, str]:
    """Kalshi club name -> reference name (best unique match above ``threshold``)."""
    out = dict(overrides or {})
    taken = set(out.values())
    pairs = sorted(((name_score(k, r), k, r) for k in kalshi if k not in out for r in ref), reverse=True)
    for sc, k, r in pairs:
        if sc < threshold:
            break
        if k in out or r in taken:
            continue
        out[k] = r
        taken.add(r)
    return out


# ---- Polymarket international (data only) ----------------------------------------------------
_PM_GAME_RX = re.compile(r"^[a-z0-9]+-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}$")
_PM_WIN_RX = re.compile(r"^Will (?P<team>.+?) win on \d{4}-\d{2}-\d{2}\?$")


def pm_clients(s, min_interval: float = 0.5):
    """Polymarket intl Gamma and Data API clients; responses cached under data/cache/soccer/raw."""
    from albert.http import CachedClient

    raw = s.data_dir / "cache" / NAME / "raw"
    return (CachedClient("pm_gamma", s.pm_gamma_base, raw, min_interval=min_interval),
            CachedClient("pm_data", s.pm_data_base, raw, min_interval=min_interval))


def pm_game_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Main 1X2 events only: two "Will <team> win on <date>?" markets plus one draw market.
    Returns the YES token of each leg and the team names as Polymarket writes them."""
    from albert.venues.polymarket import _json_list, _parse_loose

    rows = []
    for e in events:
        if not _PM_GAME_RX.match(e.get("slug") or "") or len(e.get("markets") or []) != 3:
            continue
        teams, draw, start = [], None, None
        for m in e["markets"]:
            toks = [str(t) for t in _json_list(m.get("clobTokenIds"))]
            outs = [str(o) for o in _json_list(m.get("outcomes"))]
            if len(toks) != 2 or outs[:1] != ["Yes"]:
                break
            start = start or _parse_loose(m.get("gameStartTime"))
            q = m.get("question") or ""
            w = _PM_WIN_RX.match(q)
            if w:
                teams.append((w["team"], toks[0]))
            elif "draw" in q.lower():
                draw = toks[0]
        if len(teams) != 2 or draw is None or start is None:
            continue
        rows.append({"pm_slug": e["slug"], "pm_title": e.get("title"), "pm_start": start,
                     "pm_team1": teams[0][0], "pm_tok1": teams[0][1], "pm_team2": teams[1][0],
                     "pm_tok2": teams[1][1], "pm_draw": draw})
    return rows


def pm_games(s, start: datetime, end: datetime) -> pl.DataFrame:
    """Every closed Polymarket soccer 1X2 event that ended in [start, end) (cached pages)."""
    gamma, _ = pm_clients(s)
    rows, cursor = [], None
    while True:
        page = gamma.get("/events/keyset", {
            "closed": "true", "limit": 100, "tag_slug": "soccer", "after_cursor": cursor,
            "end_date_min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date_max": end.strftime("%Y-%m-%dT%H:%M:%SZ")})
        evs = page.payload.get("events") or []
        rows += pm_game_rows(evs)
        cursor = page.payload.get("next_cursor")
        if not cursor or not evs:
            break
    return pl.DataFrame(rows).unique("pm_slug") if rows else pl.DataFrame()


# ---- Kalshi quotes ---------------------------------------------------------------------------
QUOTE_WINDOW = (timedelta(hours=27), timedelta(hours=0))   # [kickoff − 27h, kickoff]


def fetch_game_candles(s, games: pl.DataFrame, *, workers: int = 2, max_tickers: int = 99) -> pl.DataFrame:
    """Hourly candles of all three legs of every game over ``QUOTE_WINDOW``, cached in
    ``data/cache/fundamental_soccer_candles.parquet`` (same schema as the harness cache).

    Games settled after the historical cutoff are fetched with the batch endpoint (up to 99
    tickers of games kicking off the same UTC day per request); older ones per market."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = s.data_dir / "cache" / f"fundamental_{NAME}_candles.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["ticker"].unique()) if have is not None and have.height else set()
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.1))
    cutoff = api.cutoff().get("market_settled_ts")
    jobs: list[tuple[str, list[str], datetime, datetime]] = []
    live: dict[date, list[tuple[list[str], datetime, datetime]]] = defaultdict(list)
    for g in games.iter_rows(named=True):
        ticks = [t for t in (g["home_ticker"], g["away_ticker"], g["tie_ticker"]) if t not in done]
        if not ticks or g["kickoff"] is None:
            continue
        lo, hi = g["kickoff"] - QUOTE_WINDOW[0], g["kickoff"] - QUOTE_WINDOW[1]
        if cutoff and g["close_time"] < cutoff:
            jobs += [("hist", [t], lo, hi) for t in ticks]
        else:
            live[g["kickoff"].date()].append((ticks, lo, hi))
    for day_games in live.values():
        cur: list[tuple[list[str], datetime, datetime]] = []
        for item in day_games + [None]:
            if item is not None and sum(len(x[0]) for x in cur) + len(item[0]) <= max_tickers:
                cur.append(item)
                continue
            if cur:
                jobs.append(("live", [t for x in cur for t in x[0]], min(x[1] for x in cur), max(x[2] for x in cur)))
            cur = [item] if item is not None else []

    def one(job: tuple[str, list[str], datetime, datetime]) -> list[dict[str, Any]]:
        kind, tickers, lo, hi = job
        for attempt in range(6):
            try:
                if kind == "hist":
                    return k.candle_rows(api.candles(tickers[0], lo, hi, 60, True).payload, tickers[0], 60)
                page = api.candles_batch(tickers, lo, hi, 60)
                return [r for mk in page.payload.get("markets") or []
                        for r in k.candle_rows(mk, mk.get("market_ticker"), 60)]
            except RuntimeError as error:
                if "429" in str(error):
                    time.sleep(5 * (attempt + 1))
                    continue
                print(f"skip candles {tickers[0]} (+{len(tickers) - 1}): {str(error)[:160]}", flush=True)
                return []
        return []

    jobs.sort(key=lambda j: j[0] == "hist")   # batch jobs first: one request covers ~30 games
    print(f"candle jobs: {len(jobs)} ({sum(j[0] == 'hist' for j in jobs)} historical)", flush=True)
    rows: list[dict[str, Any]] = []

    def save() -> pl.DataFrame:
        new = pl.DataFrame(rows, schema=k.CANDLE_SCHEMA) if rows else None
        frames = [f for f in (have, new) if f is not None and f.height]
        out = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts", "period_minutes"]) if frames else pl.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        out.write_parquet(tmp)
        tmp.replace(path)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(one, jobs), 1):
            rows.extend(r)
            if i % 100 == 0:
                print(f"  candle jobs {i}/{len(jobs)}, {len(rows)} rows", flush=True)
                save()
    return save()


# ---- Dixon-Coles -----------------------------------------------------------------------------
NAME_OVERRIDES: dict[str, dict[str, str]] = {
    "EPL": {"Nottingham": "Nottingham Forest FC"},
    "ENG2": {"QPR": "Queens Park Rangers FC"},
    "GER1": {"Bayern Munich": "FC Bayern München", "M´gladbach": "Borussia Mönchengladbach"},
    "ESP1": {"Bilbao": "Athletic Club"},
    "FRA1": {"PSG": "Paris Saint-Germain FC"},
    "NED1": {"Alkmaar": "AZ", "Eindhoven": "PSV", "Enschede": "FC Twente '65", "GA Eagles": "Go Ahead Eagles",
             "Nijmegen": "NEC"},
    "POR1": {"SL Benfica": "Sport Lisboa e Benfica", "Sporting CP": "Sporting Clube de Portugal",
             "Vicente Barcelos": "Gil Vicente FC", "Madeira": "CS Marítimo"},
    "GRE1": {"Olympiacos": "Olympiakos Piraeus", "Aris Thessaloniki": "Aris Saloniki", "OFI Crete": "OFI Heraklion",
             "Panaitolikos": "Panetolikos"},
    "BRA1": {"Atletico Mineiro": "CA Mineiro"},
    "BEL1": {"St. Truidense": "Sint-Truidense VV"},
    "TUR1": {"Kasimpasa": "Kasımpaşa SK"},
    "AUT1": {"Austria": "Austria Wien"},
}
# League above each league (a newcomer that played there last season was relegated).
PARENT = {"en.2": "en.1", "en.3": "en.2", "de.2": "de.1", "es.2": "es.1", "it.2": "it.1", "fr.2": "fr.1"}
SEASON_START = {"br.1": date(2026, 1, 1)}   # others: 2026-07-01


def season_start(code: str) -> date:
    return SEASON_START.get(code, date(2026, 7, 1))


def fit_results(of: pl.DataFrame, games: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    """Results for the goal model, one league at a time: openfootball rows (known from the
    morning after their date: ``known = date + 30h`` UTC, conservative for late local kickoffs)
    plus Kalshi-rebuilt scores (``known = kickoff + RESULT_LAG``), Kalshi names mapped to
    openfootball names. A Kalshi game that openfootball already lists (same pair within a day)
    is dropped."""
    out = []
    for lg, code in {v[0]: v[1] for v in LEAGUES.values() if v[1]}.items():
        ref = of.filter(pl.col("code") == code)
        kg = games.filter(pl.col("league") == lg).join(scores, on="event", how="inner")
        names = sorted(set(kg["home"]) | set(kg["away"]))
        mp = map_names(names, sorted(set(ref["home"]) | set(ref["away"])), NAME_OVERRIDES.get(lg))
        a = ref.select(pl.lit(lg).alias("league"), pl.lit(code).alias("code"), "home", "away", "hg", "ag",
                       (pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(hours=12)).alias("kickoff"),
                       (pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(hours=30)).alias("known"),
                       pl.lit("openfootball").alias("src"))
        b = kg.select(pl.lit(lg).alias("league"), pl.lit(code).alias("code"),
                      pl.col("home").replace(mp), pl.col("away").replace(mp), "hg", "ag", "kickoff",
                      (pl.col("kickoff") + RESULT_LAG).alias("known"), pl.lit("kalshi").alias("src"))
        if a.height:
            dup = b.join(a.select("home", "away", pl.col("kickoff").alias("k2")), on=["home", "away"], how="inner").filter(
                (pl.col("kickoff") - pl.col("k2")).abs() <= timedelta(hours=36))
            b = b.join(dup.select("home", "away", "kickoff"), on=["home", "away", "kickoff"], how="anti")
        out.append(pl.concat([a, b.cast(a.schema)]) if a.height else b)
    return pl.concat(out).sort("league", "kickoff")


def _dc_tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.clip(t, 1e-6, None)


def fit_dc(hi: np.ndarray, ai: np.ndarray, hg: np.ndarray, ag: np.ndarray, w: np.ndarray, n: int,
           prior_att: np.ndarray, prior_def: np.ndarray, ridge: float = 2.0) -> dict[str, Any]:
    """Weighted Dixon-Coles fit. Goals: home ~ Poisson(exp(c + h + att[i] + dfn[j])), away ~
    Poisson(exp(c + att[j] + dfn[i])) (``dfn`` high = leaky defence); low-score correction
    ``rho`` by profile likelihood on a grid. Teams are tied to their prior means by a ridge
    penalty (``ridge`` ≈ pseudo-matches of information), which also identifies the model."""
    from scipy.optimize import minimize

    def unpack(z):
        return z[0], z[1], z[2:2 + n], z[2 + n:]

    def nll(z):
        c, h, att, dfn = unpack(z)
        ll = c + h + att[hi] + dfn[ai]
        la = c + att[ai] + dfn[hi]
        el, ea = np.exp(ll), np.exp(la)
        f = -(w * (hg * ll - el + ag * la - ea)).sum()
        f += ridge * (((att - prior_att) ** 2).sum() + ((dfn - prior_def) ** 2).sum())
        rh, ra = w * (hg - el), w * (ag - ea)
        g = np.zeros_like(z)
        g[0] = -(rh.sum() + ra.sum())
        g[1] = -rh.sum()
        g_att = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * ridge * (att - prior_att)
        g_def = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * ridge * (dfn - prior_def)
        g[2:2 + n], g[2 + n:] = g_att, g_def
        return f, g

    z0 = np.concatenate([[math.log(1.3), 0.25], prior_att, prior_def])
    res = minimize(nll, z0, jac=True, method="L-BFGS-B")
    c, h, att, dfn = unpack(res.x)
    lam, mu = np.exp(c + h + att[hi] + dfn[ai]), np.exp(c + att[ai] + dfn[hi])
    grid = np.linspace(-0.25, 0.15, 41)
    lls = [(w * np.log(_dc_tau(hg, ag, lam, mu, r))).sum() for r in grid]
    return {"c": c, "h": h, "att": att, "dfn": dfn, "rho": float(grid[int(np.argmax(lls))])}


def dc_outcome_probs(lam: float, mu: float, rho: float, max_goals: int = 10) -> tuple[float, float, float]:
    """P(home win), P(draw), P(away win) from the Dixon-Coles score matrix."""
    from scipy.stats import poisson

    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m /= m.sum()
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def newcomer_priors(res: pl.DataFrame, xi: float, ridge: float) -> dict[str, tuple[float, float]]:
    """Mean fitted (att, dfn) of 2025-26 newcomers (absent from the league in 2024-25): teams
    promoted from below, and teams relegated from the league above. Fit on 2025-26 only."""
    acc: dict[str, list[tuple[float, float]]] = {"up": [], "down": []}
    for code in res["code"].unique().to_list():
        r = res.filter(pl.col("code") == code)
        prev = r.filter((pl.col("kickoff") >= datetime(2024, 7, 1, tzinfo=UTC)) & (pl.col("kickoff") < datetime(2025, 7, 1, tzinfo=UTC)))
        cur = r.filter((pl.col("kickoff") >= datetime(2025, 7, 1, tzinfo=UTC)) & (pl.col("kickoff") < datetime(2026, 7, 1, tzinfo=UTC)))
        if prev.height < 100 or cur.height < 150:
            continue
        teams = sorted(set(cur["home"]) | set(cur["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        fit = fit_dc(cur["home"].replace_strict(idx).to_numpy(), cur["away"].replace_strict(idx).to_numpy(),
                     cur["hg"].to_numpy().astype(float), cur["ag"].to_numpy().astype(float),
                     np.ones(cur.height), n, np.zeros(n), np.zeros(n), ridge)
        old = set(prev["home"]) | set(prev["away"])
        parent = res.filter((pl.col("code") == PARENT.get(code, "")) & (pl.col("kickoff") >= datetime(2024, 7, 1, tzinfo=UTC))
                            & (pl.col("kickoff") < datetime(2025, 7, 1, tzinfo=UTC)))
        above = set(parent["home"]) | set(parent["away"])
        for t in teams:
            if t not in old:
                acc["down" if t in above else "up"].append((float(fit["att"][idx[t]]), float(fit["dfn"][idx[t]])))
    return {k: (float(np.mean([a for a, _ in v])), float(np.mean([d for _, d in v]))) if v else (0.0, 0.0)
            for k, v in acc.items()} | {"n_up": len(acc["up"]), "n_down": len(acc["down"])}


def dc_predictions(res: pl.DataFrame, games: pl.DataFrame, decisions: pl.DataFrame, *, xi: float,
                   ridge: float = 2.0, priors: dict[str, Any] | None = None,
                   history_days: int = 1100) -> pl.DataFrame:
    """P(home), P(draw), P(away) at each decision time from a fit on results *known* before
    the start of the decision's UTC day (one fit per league-day; weights exp(−xi·age_days))."""
    priors = priors or {"up": (0.0, 0.0), "down": (0.0, 0.0)}
    gl = games.select("event", "league", "home", "away")
    d = decisions.filter(pl.col("t").is_not_null()).join(gl, on="event").with_columns(pl.col("t").dt.truncate("1d").alias("day"))
    out = []
    for (lg, day), part in d.group_by(["league", "day"]):
        r = res.filter((pl.col("league") == lg) & (pl.col("known") <= day)
                       & (pl.col("kickoff") >= day - timedelta(days=history_days)))
        if r.height < 50:
            continue
        code = r["code"][0]
        mp = map_names(sorted(set(part["home"]) | set(part["away"])), sorted(set(r["home"]) | set(r["away"])),
                       NAME_OVERRIDES.get(lg))
        part = part.with_columns(pl.col("home").replace(mp).alias("h_ref"), pl.col("away").replace(mp).alias("a_ref"))
        teams = sorted(set(r["home"]) | set(r["away"]) | set(part["h_ref"]) | set(part["a_ref"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        start = datetime.combine(season_start(code), datetime.min.time(), UTC)
        prev = r.filter((pl.col("kickoff") < start) & (pl.col("kickoff") >= start - timedelta(days=365)))
        old = set(prev["home"]) | set(prev["away"])
        parent = res.filter((pl.col("code") == PARENT.get(code, "")) & (pl.col("kickoff") < start)
                            & (pl.col("kickoff") >= start - timedelta(days=365)))
        above = set(parent["home"]) | set(parent["away"])
        pa, pd_ = np.zeros(n), np.zeros(n)
        for t, i in idx.items():
            if t not in old and prev.height:
                pa[i], pd_[i] = priors["down" if t in above else "up"]
        age = (day - r["kickoff"]).dt.total_seconds().to_numpy() / 86400
        fit = fit_dc(r["home"].replace_strict(idx).to_numpy(), r["away"].replace_strict(idx).to_numpy(),
                     r["hg"].to_numpy().astype(float), r["ag"].to_numpy().astype(float),
                     np.exp(-xi * age), n, pa, pd_, ridge)
        for row in part.iter_rows(named=True):
            i, j = idx[row["h_ref"]], idx[row["a_ref"]]
            lam = math.exp(fit["c"] + fit["h"] + fit["att"][i] + fit["dfn"][j])
            mu = math.exp(fit["c"] + fit["att"][j] + fit["dfn"][i])
            ph, pd, pa_ = dc_outcome_probs(lam, mu, fit["rho"])
            out.append({"event": row["event"], "t": row["t"], "lead": row["lead"], "dc_h": ph, "dc_d": pd, "dc_a": pa_,
                        "dc_lam": lam, "dc_mu": mu, "dc_n": r.height})
    return pl.DataFrame(out)


def pm_match(games: pl.DataFrame, pm: pl.DataFrame, *, max_gap: timedelta = timedelta(hours=3),
             threshold: float = 0.75) -> pl.DataFrame:
    """Kalshi game -> Polymarket 1X2 event: kickoffs within ``max_gap`` and both club names
    similar (:func:`name_score`). Returns the Polymarket YES tokens as home / draw / away."""
    pm_rows = pm.sort("pm_start").to_dicts()
    starts = np.array([r["pm_start"].timestamp() for r in pm_rows])
    out = []
    for g in games.filter(pl.col("kickoff").is_not_null()).iter_rows(named=True):
        k = g["kickoff"].timestamp()
        lo, hi = np.searchsorted(starts, k - max_gap.total_seconds()), np.searchsorted(starts, k + max_gap.total_seconds(), "right")
        best = None
        for r in pm_rows[lo:hi]:
            same_time = abs(r["pm_start"].timestamp() - k) <= 900
            for swap in (False, True):
                t1, t2 = (r["pm_team2"], r["pm_team1"]) if swap else (r["pm_team1"], r["pm_team2"])
                s1, s2 = name_score(g["home"], t1), name_score(g["away"], t2)
                sc = min(s1, s2)
                if sc < threshold and same_time and max(s1, s2) >= 0.9:
                    sc = threshold - 0.01  # one club matches exactly at the same kickoff: same game
                if sc >= threshold - 0.01 and (best is None or sc > best[0]):
                    best = (sc, r, swap)
        if best is None:
            continue
        sc, r, swap = best
        out.append({"event": g["event"], "pm_slug": r["pm_slug"], "pm_start": r["pm_start"], "pm_score": sc,
                    "pm_home": r["pm_tok2"] if swap else r["pm_tok1"], "pm_away": r["pm_tok1"] if swap else r["pm_tok2"],
                    "pm_draw": r["pm_draw"], "pm_home_name": r["pm_team2"] if swap else r["pm_team1"]})
    m = pl.DataFrame(out)
    # one Polymarket event per Kalshi game and vice versa (keep the best name score)
    return m.sort("pm_score", descending=True).unique("pm_slug", keep="first").unique("event", keep="first") if m.height else m


PM_BUCKET = 300


def fetch_pm_history(s, matched: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Polymarket price history (``PM_BUCKET``-second buckets) of the three YES tokens over
    ``QUOTE_WINDOW`` before each matched kickoff, cached in
    ``data/cache/fundamental_soccer_pm_history.parquet``."""
    from albert.venues.polymarket import price_history_rows

    path = s.data_dir / "cache" / f"fundamental_{NAME}_pm_history.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(have["token_id"].unique()) if have is not None and have.height else set()
    import threading
    from concurrent.futures import ThreadPoolExecutor

    # two workers, each throttled to one request per second: at most 2 requests/s in total
    clients = [pm_clients(s, min_interval=1.0)[1] for _ in range(2)]
    local = threading.local()
    counter = iter(range(len(clients)))
    lock = threading.Lock()
    kick = dict(zip(games["event"], games["kickoff"]))
    todo = [(r["event"], tok) for r in matched.iter_rows(named=True) for tok in (r["pm_home"], r["pm_draw"], r["pm_away"])
            if tok not in done and kick.get(r["event"])]
    print(f"pm history: {len(todo)} tokens", flush=True)

    def one(item: tuple[str, str]) -> list[dict[str, Any]]:
        if not hasattr(local, "client"):
            with lock:
                local.client = clients[next(counter)]
        ev, tok = item
        lo, hi = kick[ev] - QUOTE_WINDOW[0], kick[ev] - QUOTE_WINDOW[1]
        try:
            page = local.client.get("/prices-history", {"token_id": tok, "start": int(lo.timestamp()),
                                                        "end": int(hi.timestamp()), "bucket_seconds": PM_BUCKET})
        except RuntimeError as error:
            print(f"skip pm {tok[:12]}: {str(error)[:160]}", flush=True)
            return []
        return price_history_rows(page.payload, tok)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        for i, r in enumerate(pool.map(one, todo), 1):
            rows += r
            if i % 300 == 0:
                print(f"  pm tokens {i}/{len(todo)}", flush=True)
    new = pl.DataFrame(rows, schema={"token_id": pl.String, "ts": pl.Datetime("us", "UTC"), "price": pl.Float64}) if rows else None
    frames = [f for f in (have, new) if f is not None and f.height]
    out = pl.concat(frames).unique(["token_id", "ts"]) if frames else pl.DataFrame()
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    return out


# ---- de-vigging ------------------------------------------------------------------------------
def devig(q: np.ndarray, method: str) -> np.ndarray:
    """Fair probabilities from quoted implied probabilities ``q`` (one row per event, summing
    to more than 1): ``mult`` (divide by the sum), ``power`` (p_i = q_i^k with k so that they
    sum to 1) or ``shin`` (Shin 1993 insider-share model, z solved by bisection)."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        return devig(q[None, :], method)[0]
    s = q.sum(1, keepdims=True)
    if method == "mult":
        return q / s
    out = np.empty_like(q)
    for r in range(q.shape[0]):
        qi = q[r]
        if method == "power":
            lo, hi = 0.2, 5.0
            for _ in range(80):
                k = (lo + hi) / 2
                if (qi ** k).sum() > 1:
                    lo = k
                else:
                    hi = k
            p = qi ** ((lo + hi) / 2)
        elif method == "shin":
            S_ = qi.sum()

            def ps(z: float) -> np.ndarray:
                return (np.sqrt(z * z + 4 * (1 - z) * qi * qi / S_) - z) / (2 * (1 - z))
            lo, hi = 0.0, 0.5
            for _ in range(80):
                z = (lo + hi) / 2
                if ps(z).sum() > 1:
                    lo = z
                else:
                    hi = z
            p = ps((lo + hi) / 2)
        else:
            raise ValueError(method)
        out[r] = p / p.sum()
    return out


# ---- prediction frames -----------------------------------------------------------------------
LEGS = (("home", "home_ticker", "H"), ("tie", "tie_ticker", "D"), ("away", "away_ticker", "A"))


def decision_frame(games: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, lead): ``t = kickoff − lead`` hours rounded *down* to the hour, only
    while the market was open. On the hour, the Kalshi candle that closes at ``t`` is the book
    at ``t`` exactly, so the Kalshi quote and the outside price are read at the same instant (a
    fresher outside price against a stale Kalshi quote would manufacture an edge)."""
    g = games.filter(pl.col("kickoff").is_not_null())
    return pl.concat([g.select("event", (pl.col("kickoff") - timedelta(hours=h)).dt.truncate("1h").alias("t"), pl.lit(h).alias("lead"),
                               "open_time") for h in LEADS]).filter(pl.col("open_time") <= pl.col("t")).drop("open_time")


def long_frame(wide: pl.DataFrame, games: pl.DataFrame, cols: tuple[str, str, str], model: str) -> pl.DataFrame:
    """(event, t, lead, P(home), P(draw), P(away)) -> harness rows ``ticker, event, t, p, y``."""
    g = games.select("event", "league", "outcome", "home_ticker", "tie_ticker", "away_ticker")
    w = wide.join(g, on="event")
    parts = [w.select(pl.col(tk).alias("ticker"), "event", "t", "lead", "league", pl.lit(leg).alias("leg"),
                      pl.col(c).alias("p"), (pl.col("outcome") == o).cast(pl.Int8).alias("y"), pl.lit(model).alias("model"))
             for (leg, tk, o), c in zip(LEGS, cols)]
    return pl.concat(parts).filter(pl.col("p").is_not_null())


def pm_wide(matched: pl.DataFrame, hist: pl.DataFrame, decisions: pl.DataFrame,
            max_age: timedelta = timedelta(hours=2)) -> pl.DataFrame:
    """Polymarket home / draw / away prices at each decision time: the last bucket that *ended*
    at or before ``t`` (bucket start + ``PM_BUCKET`` ≤ t) and at most ``max_age`` old; then
    de-vigged by each method (``pm_<method>_{h,d,a}``) and the raw sum (overround)."""
    h = hist.with_columns((pl.col("ts") + timedelta(seconds=PM_BUCKET)).alias("avail")).sort("token_id", "avail")
    d = decisions.join(matched.select("event", "pm_home", "pm_draw", "pm_away"), on="event")
    for leg in ("home", "draw", "away"):
        j = (d.select("event", "t", pl.col(f"pm_{leg}").alias("token_id")).sort("token_id", "t")
             .join_asof(h.select("token_id", "avail", "price"), left_on="t", right_on="avail", by="token_id",
                        strategy="backward", check_sortedness=False)
             .filter((pl.col("t") - pl.col("avail")) <= max_age)
             .select("event", "t", pl.col("price").alias(f"q_{leg}")))
        d = d.join(j, on=["event", "t"], how="left")
    d = d.drop_nulls(["q_home", "q_draw", "q_away"]).filter(
        (pl.col("q_home") > 0) & (pl.col("q_draw") > 0) & (pl.col("q_away") > 0))
    q = d.select("q_home", "q_draw", "q_away").to_numpy()
    cols = {"pm_overround": q.sum(1)}
    for m in ("mult", "power", "shin"):
        p = devig(q, m)
        cols |= {f"pm_{m}_h": p[:, 0], f"pm_{m}_d": p[:, 1], f"pm_{m}_a": p[:, 2]}
    return d.with_columns(*[pl.Series(k, v) for k, v in cols.items()])


def outcome_matrix(outcomes: pl.Series) -> np.ndarray:
    return np.column_stack([(outcomes == k).to_numpy() for k in "HDA"]).astype(float)


def multi_scores(P: np.ndarray, Y: np.ndarray) -> dict[str, float]:
    """Three-way log loss and Brier (sum over legs) and ranked probability score."""
    P = np.clip(P, 1e-6, 1)
    rps = (((np.cumsum(P, 1) - np.cumsum(Y, 1))[:, :2]) ** 2).sum(1) / 2
    return {"n": len(P), "logloss": round(float(-np.log((P * Y).sum(1)).mean()), 5),
            "brier3": round(float(((P - Y) ** 2).sum(1).mean()), 5), "rps": round(float(rps.mean()), 5)}


def fit_pool(feats: list[np.ndarray], Y: np.ndarray, iters: int = 200) -> np.ndarray:
    """Conditional-logit pool of several three-way forecasts: score_k = a_k + Σ_m b_m·log p_mk,
    P = softmax(score) (a_away = 0). Returns [a_home, a_draw, b_1, ..., b_M]; fitted by
    L-BFGS on log loss."""
    from scipy.optimize import minimize

    L = [np.log(np.clip(f, 1e-6, 1)) for f in feats]

    def nll(z):
        s = z[2 + 0] * L[0] if L else 0
        for m in range(1, len(L)):
            s = s + z[2 + m] * L[m]
        s = s + np.array([z[0], z[1], 0.0])
        s = s - s.max(1, keepdims=True)
        lp = s - np.log(np.exp(s).sum(1, keepdims=True))
        return -(lp * Y).sum() / len(Y)

    z0 = np.concatenate([[0.0, 0.0], np.ones(len(L)) / max(len(L), 1)])
    return minimize(nll, z0, method="L-BFGS-B").x


def apply_pool(z: np.ndarray, feats: list[np.ndarray]) -> np.ndarray:
    s = sum(z[2 + m] * np.log(np.clip(f, 1e-6, 1)) for m, f in enumerate(feats)) + np.array([z[0], z[1], 0.0])
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


# ---- three-way (ladder) optimisation ---------------------------------------------------------
def kelly_portfolio(p: np.ndarray, bid: np.ndarray, ask: np.ndarray, *, fraction: float = 0.25,
                    cap: float = 0.25) -> tuple[np.ndarray, float]:
    """Bankroll fractions for six instruments (YES home/draw/away at the ask, NO home/draw/away
    at 1 − bid, each plus the Kalshi taker fee) maximising E[log wealth] under the model's
    outcome probabilities ``p`` (H, D, A); then scaled by ``fraction`` (fractional Kelly) and
    capped at ``cap`` of bankroll in total. Returns (fractions, expected log growth at full
    Kelly)."""
    from scipy.optimize import minimize

    cost = np.array([a + F.taker_fee(a) for a in ask] + [(1 - b) + F.taker_fee(1 - b) for b in bid])
    win = np.zeros((3, 6))           # win[o, j] = 1 if instrument j pays in outcome o
    for o in range(3):
        win[o, o] = 1
        win[o, [3 + k for k in range(3) if k != o]] = 1
    R = win / cost[None, :] - 1      # return per $ of instrument j in outcome o
    edge = p @ R
    if edge.max() <= 0:
        return np.zeros(6), 0.0

    def nll(f):
        w = 1 + R @ f
        if (w <= 1e-9).any():
            return 1e9, np.zeros(6)
        return -(p * np.log(w)).sum(), -(R.T @ (p / w))

    res = minimize(nll, np.where(edge > 0, 0.01, 0.0), jac=True, method="SLSQP", bounds=[(0, 1)] * 6,
                   constraints=[{"type": "ineq", "fun": lambda f: 0.999 - f.sum()}])
    f = np.clip(res.x, 0, None)
    g = -res.fun if res.success else 0.0
    f = f * fraction
    if f.sum() > cap:
        f *= cap / f.sum()
    return f, float(g)


def ladder_backtest(rows: pl.DataFrame, pcols: tuple[str, str, str], *, fraction: float = 0.25,
                    min_growth: float = 0.0) -> pl.DataFrame:
    """Per game: Kelly portfolio over the three legs (both sides) vs the single best instrument
    (largest expected return per $, same Kelly sizing on that one instrument alone). Rows need
    the three legs' bid / ask (``bid_h``…) and the outcome. Returns per-game returns on a unit
    bankroll (no compounding)."""
    out = []
    for r in rows.iter_rows(named=True):
        p = np.array([r[c] for c in pcols])
        bid = np.array([r["bid_h"], r["bid_d"], r["bid_a"]])
        ask = np.array([r["ask_h"], r["ask_d"], r["ask_a"]])
        o = "HDA".index(r["outcome"])
        f, g = kelly_portfolio(p, bid, ask, fraction=fraction)
        cost = np.array([a + F.taker_fee(a) for a in ask] + [(1 - b) + F.taker_fee(1 - b) for b in bid])
        pay = np.zeros(6)
        pay[o] = 1
        pay[[3 + k for k in range(3) if k != o]] = 1
        ret = pay / cost - 1
        edge = np.array([p[k] for k in range(3)] + [1 - p[k] for k in range(3)]) / cost - 1
        j = int(np.argmax(edge))
        single = np.zeros(6)
        if edge[j] > 0:
            # Kelly for one binary bet: f* = edge / (1/cost − 1)
            single[j] = min(fraction * edge[j] / (1 / cost[j] - 1), 0.25)
        if g < min_growth:
            f = np.zeros(6)
        out.append({"event": r["event"], "t": r["t"], "lead": r["lead"], "league": r["league"],
                    "stake_ladder": float(f.sum()), "ret_ladder": float(f @ ret), "n_legs": int((f > 1e-4).sum()),
                    "stake_single": float(single.sum()), "ret_single": float(single @ ret), "exp_growth": g})
    return pl.DataFrame(out)


# ---- study -----------------------------------------------------------------------------------
XI_GRID = (0.0015, 0.003, 0.006)        # Dixon-Coles time decay per day (half-life 460/230/115 days)
DEVIG = ("mult", "power", "shin")
BIG = ("EPL", "ESP1", "ITA1", "GER1", "FRA1", "UCL")


def _three(w: pl.DataFrame, prefix: str) -> np.ndarray:
    return w.select(f"{prefix}_h", f"{prefix}_d", f"{prefix}_a").to_numpy()


def quotes_wide(games: pl.DataFrame, decisions: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Bid / ask of all three legs at each decision time (harness rule: last candle closed by
    ``t``, at most 2h old); rows with any leg missing are dropped."""
    out = decisions
    for leg, tk, _ in LEGS:
        q = F.attach_quotes(decisions.join(games.select("event", pl.col(tk).alias("ticker")), on="event"), candles)
        out = out.join(q.select("event", "t", pl.col("bid").alias(f"bid_{leg[0]}"), pl.col("ask").alias(f"ask_{leg[0]}")),
                       on=["event", "t"], how="inner")
    tie = {"bid_t": "bid_d", "ask_t": "ask_d"}
    return out.rename(tie).with_columns(*[((pl.col(f"bid_{x}") + pl.col(f"ask_{x}")) / 2).alias(f"mid_{x}") for x in "hda"])


def run(s, *, holdout_from: datetime = HOLDOUT) -> dict[str, Any]:
    """The whole study from cached inputs (fetching whatever is missing)."""
    mk = F.settled_markets(s, list(LEAGUES))
    games = game_frame(mk)
    bases = [x[:-4] for x in LEAGUES]
    ladders = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet")
               .filter(pl.col("series_ticker").is_in([b + "TOTAL" for b in bases] + [b + "SPREAD" for b in bases]))
               .select("ticker", "series_ticker", "result").collect())
    scores = kalshi_scores(ladders, games)
    fetch_openfootball(s, sorted({v[1] for v in LEAGUES.values() if v[1]} | {"mls"}))
    of = openfootball_matches(s)
    res = fit_results(of, games, scores)
    priors = newcomer_priors(res, 0.0, 2.0)
    dec = decision_frame(games)
    extra: dict[str, Any] = {"games": games.height, "scores_rebuilt": scores.height, "fit_results": res.height,
                             "newcomer_priors": priors}

    # Dixon-Coles: pick the decay on dev games (three-way log loss vs outcomes)
    dcs, dc_dev = {}, []
    y_of = dict(zip(games["event"], games["outcome"]))
    for xi in XI_GRID:
        p = dc_predictions(res, games, dec, xi=xi, priors=priors)
        dcs[xi] = p
        dev = p.filter(pl.col("t") < holdout_from)
        dc_dev.append({"xi": xi, **multi_scores(_three(dev, "dc"), outcome_matrix(dev["event"].replace_strict(y_of)))})
    xi = min(dc_dev, key=lambda r: r["logloss"])["xi"]
    dc = dcs[xi]
    extra["dc_xi_chosen_on_dev"] = xi

    # Polymarket consensus
    pm = pm_games(s, datetime(2026, 8, 14, tzinfo=UTC), datetime(2026, 11, 3, tzinfo=UTC))
    matched = pm_match(games, pm)
    hist = fetch_pm_history(s, matched, games).filter(pl.col("price") > 0)
    pmw = pm_wide(matched, hist, dec)
    pm_dev = []
    for m in DEVIG:
        dev = pmw.filter(pl.col("t") < holdout_from)
        pm_dev.append({"method": m, **multi_scores(_three(dev, f"pm_{m}"), outcome_matrix(dev["event"].replace_strict(y_of)))})
    method = min(pm_dev, key=lambda r: r["logloss"])["method"]
    extra["pm_devig_chosen_on_dev"] = method
    extra["pm_matched_games"] = matched.height
    gap = games.join(matched, on="event").select(((pl.col("pm_start") - pl.col("kickoff")).dt.total_minutes().abs() <= 15).mean())
    extra["kickoff_check_share_within_15min_of_polymarket"] = round(float(gap.item()), 4)

    # Kalshi quotes, all three legs
    candles = fetch_game_candles(s, games)
    qw = quotes_wide(games, dec, candles)

    # wide frame of all forecasts on the same rows
    w = (qw.join(games.select("event", "league", "outcome"), on="event")
         .join(dc.select("event", "t", "dc_h", "dc_d", "dc_a"), on=["event", "t"], how="left")
         .join(pmw.select("event", "t", "pm_overround", *[f"pm_{method}_{x}" for x in "hda"]), on=["event", "t"], how="left")
         .rename({f"pm_{method}_{x}": f"pm_{x}" for x in "hda"}))
    mid = _three(w, "mid")
    w = w.with_columns(*[pl.Series(f"kmid_{x}", (mid / mid.sum(1, keepdims=True))[:, i]) for i, x in enumerate("hda")])
    # pooled forecast (conditional logit), fitted on dev rows that have both inputs
    both = w.filter(pl.col("dc_h").is_not_null() & pl.col("pm_h").is_not_null())
    devb = both.filter(pl.col("t") < holdout_from)
    z = fit_pool([_three(devb, "pm"), _three(devb, "dc")], outcome_matrix(devb["outcome"]))
    pooled = apply_pool(z, [_three(both, "pm"), _three(both, "dc")])
    w = w.join(both.select("event", "t").with_columns(*[pl.Series(f"pool_{x}", pooled[:, i]) for i, x in enumerate("hda")]),
               on=["event", "t"], how="left")
    extra["pool_coef_dev"] = {"a_home": round(z[0], 4), "a_draw": round(z[1], 4), "b_pm": round(z[2], 4), "b_dc": round(z[3], 4)}

    # three-way scores: model vs Kalshi (normalised mids) on identical rows
    tw = []
    for name in ("dc", "pm", "pool"):
        rows = w.filter(pl.col(f"{name}_h").is_not_null())
        for lead in LEADS:
            for split, part in (("dev", rows.filter((pl.col("lead") == lead) & (pl.col("t") < holdout_from))),
                                ("holdout", rows.filter((pl.col("lead") == lead) & (pl.col("t") >= holdout_from)))):
                if not part.height:
                    continue
                Y = outcome_matrix(part["outcome"])
                a, b = multi_scores(_three(part, name), Y), multi_scores(_three(part, "kmid"), Y)
                tw.append({"model": name, "lead": lead, "split": split, "games": a["n"], "logloss_model": a["logloss"],
                           "logloss_kalshi": b["logloss"], "rps_model": a["rps"], "rps_kalshi": b["rps"]})

    # harness evaluation per model and lead (each leg is a row; clusters = games)
    results, seg_rows = {}, []
    quotes_long = pl.concat([w.select(pl.col("event"), "t", pl.col(f"bid_{x}").alias("bid"), pl.col(f"ask_{x}").alias("ask"),
                                      pl.lit(leg).alias("leg")) for (leg, _, _), x in zip(LEGS, "hda")])
    for name in ("dc", "pm", "pool"):
        lf = long_frame(w.filter(pl.col(f"{name}_h").is_not_null()).select("event", "t", "lead", f"{name}_h", f"{name}_d", f"{name}_a"),
                        games, (f"{name}_h", f"{name}_d", f"{name}_a"), name).join(quotes_long, on=["event", "t", "leg"])
        for lead in LEADS:
            part = lf.filter(pl.col("lead") == lead)
            r = F.evaluate(part, holdout_from)
            results[(name, lead)] = r
            theta = r.get("theta_chosen_on_dev")
            for seg_name, seg in [("leg", "leg"), ("league", "grp")]:
                pp = part.with_columns(pl.when(pl.col("league").is_in(BIG)).then(pl.lit("big5+EPL+UCL")).otherwise(pl.lit("other")).alias("grp"))
                for (val,), sp in pp.group_by([seg]):
                    rr = F.evaluate(sp, holdout_from, min_trades=10)
                    tk = [x for x in rr["taker"] if x["theta"] == (theta if theta is not None else 0.03) and x["split"] == "holdout"]
                    bl = rr.get("blend", {})
                    sc = {x["split"]: x for x in rr["scores"]}
                    seg_rows.append({"model": name, "lead": lead, "segment": seg_name, "value": val,
                                     "brier_model_hold": sc.get("holdout", {}).get("brier_model"),
                                     "brier_mid_hold": sc.get("holdout", {}).get("brier_mid"),
                                     "c_dev": bl.get("c_model"), "t_c_dev": bl.get("t_c"),
                                     "c_hold_refit": bl.get("c_model_holdout_refit"), "t_c_hold": bl.get("t_c_holdout_refit"),
                                     "theta": theta, "hold_trades": tk[0].get("trades") if tk else None,
                                     "hold_pnl_c": tk[0].get("pnl_c") if tk else None, "hold_t": tk[0].get("t") if tk else None})

    # ladder (three-way Kelly) vs single best instrument, per model
    lad = []
    for name in ("dc", "pm", "pool"):
        rows = w.filter(pl.col(f"{name}_h").is_not_null())
        bt = ladder_backtest(rows, (f"{name}_h", f"{name}_d", f"{name}_a"))
        for lead in LEADS:
            for split, part in (("dev", bt.filter((pl.col("lead") == lead) & (pl.col("t") < holdout_from))),
                                ("holdout", bt.filter((pl.col("lead") == lead) & (pl.col("t") >= holdout_from)))):
                for kind in ("ladder", "single"):
                    bets = part.filter(pl.col(f"stake_{kind}") > 0)
                    if not bets.height:
                        lad.append({"model": name, "lead": lead, "split": split, "kind": kind, "bets": 0})
                        continue
                    ret = bets[f"ret_{kind}"].to_numpy()
                    _mu, tt, _ = F.clustered_mean(ret, bets["event"].to_numpy())
                    lad.append({"model": name, "lead": lead, "split": split, "kind": kind, "bets": bets.height,
                                "staked": round(float(bets[f"stake_{kind}"].sum()), 3),
                                "pnl": round(float(ret.sum()), 4), "roi": round(float(ret.sum() / bets[f"stake_{kind}"].sum()), 4),
                                "t": None if tt is None else round(tt, 2), "log_growth": round(float(np.log1p(ret).sum()), 4),
                                "mean_legs": round(float(bets["n_legs"].mean()), 2) if kind == "ladder" else 1.0})
    return {"games": games, "wide": w, "results": results, "extra": extra, "dc_dev": dc_dev, "pm_dev": pm_dev,
            "three_way": tw, "segments": seg_rows, "ladder": lad, **artefact_tables(w, holdout_from)}


def artefact_tables(w: pl.DataFrame, holdout_from: datetime) -> dict[str, list[dict[str, Any]]]:
    """Checks before believing any edge. (1) Calibration per leg: mean Kalshi mid, model and
    realised frequency (is the tie leg mispriced on average?). (2) Disagreement buckets: the
    model minus the Kalshi mid, |d| bucketed; ``dir_gain_c`` = mean sign(d)·(y − mid) in cents,
    i.e. how much the side the model prefers beats the mid (before spread and fees)."""
    cal, dis = [], []
    rows = w.with_columns((pl.col("t") >= holdout_from).alias("hold"))
    for (leg, _, o), x in zip(LEGS, "hda"):
        for (lead, hold), part in rows.group_by(["lead", "hold"]):
            y = (part["outcome"] == o).cast(pl.Float64)
            cal.append({"leg": leg, "lead": lead, "split": "holdout" if hold else "dev", "n": part.height,
                        "kalshi_mid": round(float(part[f"mid_{x}"].mean()), 4),
                        "spread_c": round(float((part[f"ask_{x}"] - part[f"bid_{x}"]).mean() * 100), 2),
                        "pm": round(float(part[f"pm_{x}"].mean()), 4) if part[f"pm_{x}"].null_count() < part.height else None,
                        "dc": round(float(part[f"dc_{x}"].mean()), 4) if part[f"dc_{x}"].null_count() < part.height else None,
                        "realised": round(float(y.mean()), 4)})
    for name in ("dc", "pm", "pool"):
        for (leg, _, o), x in zip(LEGS, "hda"):
            part = rows.filter(pl.col(f"{name}_{x}").is_not_null())
            d = (part[f"{name}_{x}"] - part[f"mid_{x}"]).to_numpy()
            y = (part["outcome"] == o).cast(pl.Float64).to_numpy()
            gain = np.sign(d) * (y - part[f"mid_{x}"].to_numpy())
            hold = part["hold"].to_numpy()
            for lo, hi in ((0, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, 1.0)):
                for split, msk in (("dev", ~hold), ("holdout", hold)):
                    m = msk & (np.abs(d) >= lo) & (np.abs(d) < hi)
                    if not m.any():
                        continue
                    mu, tt, _ = F.clustered_mean(gain[m], part["event"].to_numpy()[m])
                    dis.append({"model": name, "leg": leg, "split": split, "abs_diff": f"[{lo:.2f},{hi:.2f})",
                                "n": int(m.sum()), "dir_gain_c": round(mu * 100, 2), "t": None if tt is None else round(tt, 2)})
    per = []
    for (lg, hold), part in rows.filter(pl.col("lead") == LEADS[-1]).group_by(["league", "hold"]):
        Y = outcome_matrix(part["outcome"])
        r = {"league": lg, "split": "holdout" if hold else "dev", "games": part.height,
             "ll_kalshi": multi_scores(_three(part, "kmid"), Y)["logloss"]}
        for name in ("dc", "pm"):
            q = part.filter(pl.col(f"{name}_h").is_not_null())
            if q.height:
                Yq = outcome_matrix(q["outcome"])
                r[f"n_{name}"] = q.height
                r[f"ll_{name}_minus_kalshi"] = round(multi_scores(_three(q, name), Yq)["logloss"]
                                                     - multi_scores(_three(q, "kmid"), Yq)["logloss"], 4)
        per.append(r)
    return {"calibration": cal, "disagreement": dis,
            "per_league": sorted(per, key=lambda r: (r["league"], r["split"]))}


def write_report(s, out: dict[str, Any]) -> Path:
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    rep.json("Run", out["extra"])
    rep.table("Dixon-Coles decay chosen on dev (three-way scores vs outcomes)", pl.DataFrame(out["dc_dev"]))
    rep.table("Polymarket de-vig method chosen on dev", pl.DataFrame(out["pm_dev"]))
    rep.table("Three-way scores: model vs Kalshi normalised mids on identical rows", pl.DataFrame(out["three_way"]))
    for (name, lead), r in out["results"].items():
        if r.get("scores"):
            rep.table(f"[{name}, {lead}h] scores per leg: model vs mid", pl.DataFrame(r["scores"]))
        if r.get("blend"):
            rep.json(f"[{name}, {lead}h] blend logit P = a + b·logit(mid) + c·logit(model)", r["blend"])
        if r.get("taker"):
            rep.table(f"[{name}, {lead}h] taker (theta chosen on dev = {r.get('theta_chosen_on_dev')})",
                      pl.DataFrame(r["taker"], infer_schema_length=None))
    rep.table("Segments (holdout; theta from the model's dev choice)", pl.DataFrame(out["segments"], infer_schema_length=None))
    rep.table("Three-way Kelly ladder vs single best instrument (quarter Kelly, unit bankroll per game)",
              pl.DataFrame(out["ladder"], infer_schema_length=None))
    rep.table("Calibration per leg (mean Kalshi mid vs realised)", pl.DataFrame(out["calibration"]).sort("leg", "lead", "split"))
    rep.table("Disagreement buckets: does the side the model prefers beat the Kalshi mid?",
              pl.DataFrame(out["disagreement"], infer_schema_length=None))
    rep.table(f"Per league, {LEADS[-1]}h: three-way log loss of each model minus Kalshi's (same games)",
              pl.DataFrame(out["per_league"], infer_schema_length=None))
    return rep.save()
