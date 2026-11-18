"""Behavioural mispricing in Kalshi politics / elections / news / world markets.

Mechanisms (who trades inefficiently and why) are listed, with their frozen classification rules,
in ``data/research/behav_politics/hypotheses.md``; that file was written before any price of this
family was opened and carries the pre-registered rules. This module builds the data, the dev
exploration, the holdout tests and the capped-book ledgers.

Pipeline (``python -m albert.research.behav_politics <step>``; outputs in
``data/research/behav_politics/``):

* ``universe``: every settled market of the Politics / Elections / World series, from the cached
  ``/historical/markets?series_ticker=`` pages (the ``flb_xs`` frame), the shared
  ``kalshi_markets`` table (closes 2026-08-15..10-31) and ``fetch-nov``; frozen labels
  (``label_markets``): partisan lean (H1), drama / procedural lexicon (H2, H7), foreign election and
  populist-right list (H6), roll-call vote party line (H17), ballot measure (H9), election type.
* ``fetch-nov``: family markets settled 2026-11-01..11-18 (``/markets``) plus the still-open or
  closed-unsettled ones (``/events``), so the 11-03 midterms are in the holdout.
* ``candles``: hourly candles for the study set (whole life, capped at 5,000 hours before the
  close), reusing ``xcat`` / ``pm_hazard`` caches; historical endpoint one market per request,
  live batch endpoint after the cutoff. Sequential, <= 1 request/s (shared IP with the live bot).
* ``panel``: per-market quotes at T - 7 d / 1 d / 6 h (``lead_panel``) and the election-night leader at
  fixed ET slots (``night_panel``), in chunks of whole events; ``explore``: dev descriptives;
  ``test``: the two pre-registered rules (``PREREG``) on dev and holdout with capped-book ledgers;
  ``report``: ``report.md`` from the saved results.

No look-ahead: a decision at time t uses the last hourly candle that *ended* at or before t; the
split is by decision time (dev: t and settlement before 2026-10-01; holdout: t in
2026-10-01..11-17); anchors are scheduled times (election day, deadline parsed from the rules),
never the actual close, except where flagged.
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

OUT = Path("data/research/behav_politics")
TABLES = Path("data/tables")
FAMILY = ("Politics", "Elections", "World")
ET = ZoneInfo("America/New_York")
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)
HOLD_END = datetime(2026, 11, 18, tzinfo=UTC)          # decisions through 2026-11-17 (UTC)
NOV_START = datetime(2026, 11, 1, tzinfo=UTC)
MIDTERM_DAY = datetime(2026, 11, 3, tzinfo=ET).date()
MIN_INTERVAL = 1.05                                     # seconds between Kalshi requests (<= 1/s)
MAX_PERIODS = 5000
MAX_BATCH_CANDLES = 10000
MAX_BATCH_TICKERS = 50

CANDLE_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "bid": pl.Float64,
                 "ask": pl.Float64, "price": pl.Float64, "volume": pl.Float64, "oi": pl.Float64}


# ---- frozen classification rules (copied from hypotheses.md; do not tune) ------------------------
R_RX = re.compile(r"\b(republican|republicans|gop)\b", re.IGNORECASE)
D_RX = re.compile(r"\b(democrat|democrats|democratic|democratics)\b", re.IGNORECASE)
PRIMARY_RX = re.compile(r"nominee|nomination|primary|primaries|runoff|caucus", re.IGNORECASE)
LADDER_RX = re.compile(r"\b(seats?|margin|by \d+|at least \d+|above \d+|below \d+|more than \d+|fewer than \d+|"
                       r"less than \d+|between \d+|\d+(\.\d+)?%|percentage points|popular vote margin|turnout|"
                       r"vote count)\b", re.IGNORECASE)
WIN_RX = re.compile(r"\bwin|\bcontrol|\bkeep|\bhold|\bflip|sworn in|be elected|become", re.IGNORECASE)
PRES_R = re.compile(r"\b(donald trump|trump|jd vance|j\.d\. vance)\b", re.IGNORECASE)
PRES_D = re.compile(r"\b(kamala harris|harris|joe biden|biden)\b", re.IGNORECASE)
DRAMA_RX = re.compile(
    r"resign|impeach|remov(e|es|ed) from office|leaves? office|out as|steps? down|fire[ds]?\b|fired|ousted|arrest|"
    r"indict|charged with|convict|jail|prison|assassinat|coup|martial law|invade|invasion|declare war|"
    r"war with|military strike|strikes? on|airstrike|attack|nuclear|25th amendment|insurrection act|secede|"
    r"overthrow|regime|collapse|default|shutdown|killed|\bdies\b|death|ceasefire|annex", re.IGNORECASE)
PROC_RX = re.compile(
    r"confirm|\bsign|pass(es)?\b|approve|nominat|\bmeet|visit|announce|release|endorse|attend|call with|"
    r"\bspeak|veto|hearing|vote on|testify|tariff|executive order|summit", re.IGNORECASE)
DEFAULT_YES_RX = re.compile(
    r"be confirmed|confirm|remain|still be|\bstay|certif|"
    r"sign the .*(funding|appropriations|ndaa|continuing resolution)|pass the .*(funding|appropriations|ndaa|"
    r"continuing resolution)", re.IGNORECASE)
BALLOT_RX = re.compile(r"\bprop(osition)?\b|ballot measure|amendment \d|referendum|ballot question|\bmeasure\b", re.IGNORECASE)
VOTE_RX = re.compile(r"^wil+ (.+?) vote (for|to|in favor|against|yes)", re.IGNORECASE)
POPULIST = [
    "poilievre", "conservative party of canada", "dutton", "weidel", "afd", "alternative für deutschland",
    "alternative fur deutschland", "le pen", "bardella", "rassemblement", "national rally", "orbán", "orban",
    "fidesz", "nawrocki", "pis", "law and justice", "simion", "georgescu", "aur", "kast", "milei",
    "la libertad avanza", "bolsonaro", "tarcísio", "tarcisio", "farage", "reform uk", "netanyahu", "likud",
    "wilders", "pvv", "asfura", "lópez aliaga", "lopez aliaga", "fujimori", "noboa", "de la espriella",
    "babiš", "babis", "ano", "kickl", "fpö", "fpo", "meloni", "fratelli d'italia", "abascal", "vox", "ventura",
    "chega", "kaczyński", "kaczynski",
]
POPULIST_RX = re.compile(r"\b(" + "|".join(re.escape(x) for x in POPULIST) + r")\b", re.IGNORECASE)
US_RX = re.compile(
    r"\b(alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|"
    r"illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
    r"mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|"
    r"north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|"
    r"utah|vermont|virginia|washington|west virginia|wisconsin|wyoming|nyc|u\.s\.|us house|house race|"
    r"senate race|governorship|electoral college|[A-Z]{2}-\d{1,2}|"
    r"state senate|state house)\b", re.IGNORECASE)
FOREIGN_RX = re.compile(
    r"\b(canad|brazil|brasil|german|bundestag|france|french|uk\b|united kingdom|british|by-election|britain|"
    r"japan|korea|australia|poland|polish|romania|chile|argentin|colombia|peru|mexic|hungar|israel|knesset|"
    r"netherlands|dutch|portug|spain|spanish|ital|ireland|irish|scotland|scottish|wales|welsh|czech|austria|"
    r"slovenia|slovak|denmark|danish|norway|norweg|sweden|swedish|finland|finnish|ecuador|bolivia|honduras|"
    r"costa rica|venezuel|philippin|india|taiwan|thailand|indonesia|south africa|nigeria|kenya|ghana|benin|"
    r"quebec|ontario|alberta|british columbia|mississauga|hamilton|toronto|london|holborn|clacton|bosnia|serbia|"
    r"croatia|bulgaria|greece|greek|cyprus|turkey|turkish|iran|iraq|lebanon|ukrain|russia|georgia's|moldova|"
    r"armenia|azerbaijan|kosovo|albania|belgi|swiss|switzerland|luxembourg|estonia|latvia|lithuania|iceland|"
    r"new zealand|singapore|malaysia|vietnam|bangladesh|pakistan|sri lanka|nepal|uruguay|paraguay|guatemala|"
    r"el salvador|panama|dominican|haiti|jamaica|trinidad|prime minister|parliament|chancellor|pope)\w*", re.IGNORECASE)

# 119th Congress senators by party (frozen 2026-11-18 before any vote-market price was opened).
# Independents who caucus with Democrats (King, Sanders) count as D for the party line.
SENATORS_R = ["tuberville", "britt", "sullivan", "murkowski", "cotton", "boozman", "moody", "scott", "rubio", "husted", "moreno", "marshall", "moran", "mcconnell", "paul", "kennedy", "cassidy", "hawley", "schmitt", "daines", "sheehy", "fischer", "ricketts", "budd", "tillis", "cramer", "hoeven", "mullin", "lankford", "graham", "thune", "rounds", "blackburn", "hagerty", "cornyn", "cruz", "lee", "curtis", "capito", "justice", "johnson", "barrasso", "lummis", "ernst", "grassley", "risch", "crapo", "young", "banks", "hyde-smith", "wicker", "collins", "mccormick"]
SENATORS_D = ["kelly", "gallego", "padilla", "schiff", "bennet", "hickenlooper", "blumenthal", "murphy", "blunt", "rochester", "coons", "ossoff", "warnock", "schatz", "hirono", "durbin", "duckworth", "alsobrooks", "van", "hollen", "markey", "warren", "peters", "slotkin", "klobuchar", "smith", "rosen", "cortez", "masto", "shaheen", "hassan", "booker", "kim", "heinrich", "lujan", "luján", "schumer", "gillibrand", "fetterman", "reed", "whitehouse", "kaine", "warner", "sanders", "welch", "murray", "cantwell", "baldwin", "king"]
PARTY_LINE_RX = re.compile(r"^KX(VOTE|OTE)", re.IGNORECASE)
FUNDING_RX = re.compile(r"(?i)funding|spending bill|omnibus|minibus|appropriations|continuing resolution|shutdown")
BIPARTISAN_RX = re.compile(r"epstein|cheney|ukraine aid|israel aid", re.IGNORECASE)


def is_primary(title: str | None, sub: str | None, rules: str | None) -> bool:
    return bool(PRIMARY_RX.search(f"{title or ''} | {sub or ''}")
                or re.search(r"primary|runoff|nomination for|caucus", rules or "", re.IGNORECASE))


def lean(title: str | None, sub: str | None, rules: str | None) -> str | None:
    """H1: 'R' / 'D' when the YES outcome is a party (or Trump / Harris) winning a general, special or
    chamber-control election; None for primaries, ladders and everything else."""
    t, s = title or "", sub or ""
    text = f"{t} | {s} | {rules or ''}"
    if is_primary(t, s, rules) or LADDER_RX.search(f"{t} | {s}"):
        return None
    if not (WIN_RX.search(t) or "presidential election" in text.lower() or "sworn in" in text.lower()):
        return None
    r_s, d_s = bool(R_RX.search(s)), bool(D_RX.search(s))
    if r_s != d_s:
        return "R" if r_s else "D"
    r_t, d_t = bool(R_RX.search(t)), bool(D_RX.search(t))
    if r_t != d_t:
        return "R" if r_t else "D"
    if "presidential election" in text.lower() or re.search(r"presiden", t, re.IGNORECASE):
        pr, pd = bool(PRES_R.search(s)), bool(PRES_D.search(s))
        if pr != pd:
            return "R" if pr else "D"
    return None


def approval_lean(series: str, title: str | None, sub: str | None, close: datetime | None) -> str | None:
    """H1 (approval / generic ballot): 'R' when YES is the Trump-favourable / GOP-favourable leg."""
    text = f"{title or ''} | {sub or ''}".lower()
    up = bool(re.search(r"\babove\b|more than|greater|at least|higher|increase", text))
    down = bool(re.search(r"\bbelow\b|less than|lower|decrease|under\b", text))
    if up == down:
        return None
    if re.search(r"generic ballot|\(d\s*[−-]\s*r\)|d − r|d - r", text):
        return "D" if up else "R"
    if re.search(r"approv", text):
        biden_era = close is not None and close < datetime(2025, 1, 20, tzinfo=UTC)
        pro_pres = "D" if biden_era or "biden" in text else "R"
        other = "R" if pro_pres == "D" else "D"
        return pro_pres if up else other
    return None


def drama(title: str | None, rules: str | None) -> bool:
    return bool(DRAMA_RX.search(f"{title or ''} | {rules or ''}"))


def procedural(title: str | None, rules: str | None) -> bool:
    return bool(PROC_RX.search(f"{title or ''} | {rules or ''}"))


def default_yes(title: str | None, rules: str | None) -> bool:
    return bool(DEFAULT_YES_RX.search(f"{title or ''}")) and not drama(title, None)


def is_foreign(category: str | None, title: str | None, sub: str | None, rules: str | None) -> bool:
    text = f"{title or ''} | {sub or ''} | {rules or ''}"
    return category in ("Elections", "World", "Politics") and bool(FOREIGN_RX.search(text)) and not US_RX.search(
        f"{title or ''} | {sub or ''}")


def populist(title: str | None, sub: str | None) -> bool:
    """H6: the YES outcome names a listed Trump-aligned / populist-right candidate or party."""
    s = sub or ""
    if POPULIST_RX.search(s):
        return True
    return bool(POPULIST_RX.search(title or "")) and not s


def member_party(member: str) -> str | None:
    """Party of a senator named in a vote market (last-name match on the frozen 119th list)."""
    words = re.findall(r"[a-záéíóúñ\-]+", member.lower())
    if not words:
        return None
    last = words[-1]
    two = " ".join(words[-2:])
    if two in SENATORS_D or (last in SENATORS_D and last not in SENATORS_R):
        return "D"
    if two in SENATORS_R or last in SENATORS_R:
        return "R"
    return None


def vote_party_line(series: str, title: str | None, rules: str | None) -> str | None:
    """H17: 'yes' / 'no' = the party-line side of a roll-call market on a Trump nominee or GOP-led
    bill (R members: yes; D members: no). None if not a vote market, bipartisan, or unknown member."""
    t = title or ""
    m = VOTE_RX.search(t)
    if not m or BIPARTISAN_RX.search(f"{t} {rules or ''}"):
        return None
    if not (PARTY_LINE_RX.search(series) or "vote" in t.lower()):
        return None
    if re.search(r"\bHouse\b", f"{t} {rules or ''}") and not re.search(r"\bSenate\b|senator", f"{t} {rules or ''}"):
        return None                                   # House members: no frozen party list
    party = member_party(m.group(1))
    if party is None:
        return None
    against = m.group(2).lower() == "against"
    yes_line = party == "R"
    if against:
        yes_line = not yes_line
    return "yes" if yes_line else "no"


def election_kind(category: str | None, title: str | None, sub: str | None, rules: str | None) -> str | None:
    """'general' (US general / special / control), 'primary' (US), 'foreign' (non-US election), 'ballot'."""
    text = f"{title or ''} | {sub or ''} | {rules or ''}"
    low = text.lower()
    if category not in ("Elections", "Politics", "World"):
        return None
    if not re.search(r"elect|win the|nominee|primary|runoff|by-election|race|governorship|mayor|seat|"
                     r"presiden|prime minister|parliament|chancellor|vote", low):
        return None
    if BALLOT_RX.search(title or "") and re.search(r"pass|approve|vote yes|ballot", low):
        return "ballot"
    if is_foreign(category, title, sub, rules):
        return "foreign"
    if is_primary(title, sub, rules):
        return "primary"
    if re.search(r"general election|special election|house race|senate race|governorship|presidential election|"
                 r"win the (house|senate)|control|mayor|sworn in|electoral", low):
        return "general"
    return None


def label_markets(m: pl.DataFrame) -> pl.DataFrame:
    """Frozen labels per market (see module docstring / hypotheses.md)."""
    rows = []
    for r in m.select("ticker", "series_ticker", "category", "title", "yes_sub_title", "rules_primary",
                      "close_time").iter_rows(named=True):
        t, s, ru = r["title"], r["yes_sub_title"], r["rules_primary"]
        ek = election_kind(r["category"], t, s, ru)
        is_ladder = bool(LADDER_RX.search(f"{t or ''} | {s or ''}"))
        rows.append({
            "ticker": r["ticker"],
            "lean": lean(t, s, ru) if (ek == "general" or (ek is None and US_RX.search(f"{t or ''} | {s or ''}")))
            else None,
            "approval_lean": approval_lean(r["series_ticker"] or "", t, s, r["close_time"]),
            "drama": drama(t, ru), "procedural": procedural(t, ru), "default_yes": default_yes(t, ru),
            "ekind": ek, "ladder_text": is_ladder,
            "populist": populist(t, s) if ek == "foreign" else None,
            "party_line": vote_party_line(r["series_ticker"] or "", t, ru),
        })
    return m.join(pl.DataFrame(rows, schema={
        "ticker": pl.String, "lean": pl.String, "approval_lean": pl.String, "drama": pl.Boolean,
        "procedural": pl.Boolean, "default_yes": pl.Boolean, "ekind": pl.String, "ladder_text": pl.Boolean,
        "populist": pl.Boolean, "party_line": pl.String}), on="ticker", how="left")


# ---- API -----------------------------------------------------------------------------------------
def _api(min_interval: float = MIN_INTERVAL):
    from albert.config import settings
    from albert.research.pm_hazard import _api as hz_api

    return hz_api(settings(), min_interval, nap_cap=30.0, retries=40)


def _series_info() -> dict[str, dict[str, Any]]:
    ser = pl.read_parquet(TABLES / "kalshi_series.parquet")
    return {r["series_ticker"]: r for r in ser.iter_rows(named=True)}


# ---- universe ------------------------------------------------------------------------------------
def historical_family(k, info: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Settled markets of every family series in the ``flb_xs`` frame, re-read from the cached
    ``/historical/markets?series_ticker=`` pages (same params, so no new requests when cached)."""
    from albert.venues.kalshi import MARKET_SCHEMA, market_rows

    cached = OUT / "hist_family.parquet"
    if cached.exists():
        return pl.read_parquet(cached)
    fr = pl.scan_parquet("data/research/flb_xs/frame/part-*.parquet").select("series_ticker").unique().collect()
    series = sorted(s for s in fr["series_ticker"].to_list() if (info.get(s) or {}).get("category") in FAMILY)
    rows: list[dict[str, Any]] = []
    fetched = 0
    for i, sr in enumerate(series):
        for page in k._cursor_pages("/historical/markets", {"series_ticker": sr, "limit": 1000}, "markets"):
            fetched += not page.from_cache
            rows.extend(x for x in market_rows(page.payload, info, "historical") if x["result"] in ("yes", "no"))
        if i % 500 == 0:
            print(f"  historical series {i}/{len(series)} ({fetched} fetched)", flush=True)
    out = pl.DataFrame(rows, schema=MARKET_SCHEMA)
    out.write_parquet(cached)
    return out


def fetch_nov(k=None) -> pl.DataFrame:
    """Family markets of events settled since 11-01 plus every open or closed-unsettled family event
    (the 11-03 midterms), from ``/events?with_nested_markets=true`` (200 events per request)."""
    from albert.venues.kalshi import MARKET_SCHEMA, market_rows

    k = k or _api()
    info = _series_info()
    rows: list[dict[str, Any]] = []
    for status in ("settled", "closed"):     # open events (2027-28 questions) have no holdout outcome
        params: dict[str, Any] = {"status": status, "limit": 200, "with_nested_markets": "true"}
        if status == "settled":
            params["min_close_ts"] = int(NOV_START.timestamp())
        n_ev = 0
        for i, page in enumerate(k._cursor_pages("/events", params, "events", cache=status == "settled"), 1):
            for ev in page.payload.get("events", []):
                cat = ev.get("category") or (info.get(ev.get("series_ticker")) or {}).get("category")
                if cat in FAMILY and not (ev.get("series_ticker") or "").startswith("KXMVE"):
                    n_ev += 1
                    rows.extend(market_rows({"markets": ev.get("markets") or []}, info, f"live-{status}"))
            if i % 25 == 0:
                print(f"  {status} event pages {i}, family events {n_ev}, rows {len(rows)}", flush=True)
        print(f"  {status}: {n_ev} family events", flush=True)
    df = pl.DataFrame(rows, schema=MARKET_SCHEMA).unique("ticker", keep="first")
    df = df.filter(pl.col("category").is_in(FAMILY) & ((pl.col("close_time") >= NOV_START)
                                                       | ~pl.col("status").is_in(["settled", "finalized"])))
    df.write_parquet(OUT / "nov_markets.parquet")
    print(f"fetch-nov: {df.height} family markets ({df['status'].value_counts().to_dicts()})", flush=True)
    return df


def build_universe() -> pl.DataFrame:
    from albert.venues.kalshi import MARKET_SCHEMA

    k = _api()
    info = _series_info()
    hist = historical_family(k, info)
    tab = (pl.read_parquet(TABLES / "kalshi_markets.parquet").filter(pl.col("category").is_in(FAMILY))
           .select(list(MARKET_SCHEMA)))
    parts = [hist, tab]
    nov_path = OUT / "nov_markets.parquet"
    if nov_path.exists():
        parts.append(pl.read_parquet(nov_path).select(list(MARKET_SCHEMA)))
    u = (pl.concat(parts, how="vertical_relaxed").filter(pl.col("market_type").fill_null("binary") == "binary")
         .unique("ticker", keep="last"))
    u = label_markets(u)
    u.write_parquet(OUT / "universe.parquet")
    print(f"universe: {u.height} markets, {u['event_ticker'].n_unique()} events", flush=True)
    return u


# ---- candles -------------------------------------------------------------------------------------
def _norm(df: pl.LazyFrame, bid: str, ask: str, price: str, oi: str) -> pl.LazyFrame:
    return df.select("ticker", pl.col("end_ts").cast(pl.Datetime("us", "UTC")), pl.col(bid).alias("bid"),
                     pl.col(ask).alias("ask"), pl.col(price).alias("price"), pl.col("volume").cast(pl.Float64),
                     pl.col(oi).cast(pl.Float64).alias("oi"))


def candle_sources() -> list[pl.LazyFrame]:
    """Hourly candles (ticker, end_ts, bid, ask, price, volume, oi) from the shared caches and ours."""
    src = []
    if list(Path("data/research/xcat/hourly").glob("part-*.parquet")):
        src.append(_norm(pl.scan_parquet("data/research/xcat/hourly/part-*.parquet"),
                         "bid_close", "ask_close", "p_close", "oi"))
    if Path("data/research/pm_hazard/candles.parquet").exists():
        src.append(_norm(pl.scan_parquet("data/research/pm_hazard/candles.parquet"),
                         "yes_bid_close", "yes_ask_close", "price_close", "open_interest"))
    if list((OUT / "candles").glob("part-*.parquet")):
        src.append(pl.scan_parquet(OUT / "candles" / "part-*.parquet").select(list(CANDLE_SCHEMA)))
    return src


def load_candles(tickers: list[str]) -> pl.DataFrame:
    frames = [s.filter(pl.col("ticker").is_in(tickers)) for s in candle_sources()]
    if not frames:
        return pl.DataFrame(schema=CANDLE_SCHEMA)
    return (pl.concat(frames, how="vertical_relaxed").collect()
            .unique(["ticker", "end_ts"], keep="last").sort("ticker", "end_ts"))


def covered_windows() -> pl.DataFrame:
    """Per ticker: first and last candle end in the existing caches (ours excluded)."""
    frames = [s for s in candle_sources()[:2]]
    if not frames:
        return pl.DataFrame(schema={"ticker": pl.String, "c_lo": pl.Datetime("us", "UTC"),
                                    "c_hi": pl.Datetime("us", "UTC")})
    return (pl.concat(frames, how="vertical_relaxed").group_by("ticker")
            .agg(pl.col("end_ts").min().alias("c_lo"), pl.col("end_ts").max().alias("c_hi")).collect())


def candle_rows(payload: dict[str, Any], ticker: str) -> list[dict[str, Any]]:
    from albert.venues.kalshi import _field, num, parse_ts

    out = []
    for c in payload.get("candlesticks") or []:
        p, b, a = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}
        out.append({"ticker": ticker, "end_ts": parse_ts(c.get("end_period_ts")), "bid": num(_field(b, "close")),
                    "ask": num(_field(a, "close")), "price": num(_field(p, "close")),
                    "volume": num(_field(c, "volume")), "oi": num(_field(c, "open_interest"))})
    return out


PRIMARY_SAMPLE_EVENTS = 150
PROC_SAMPLE_FRAC = 0.25            # procedural-only one-off events kept (seeded); drama / default-YES all
SEED = 20261118
GROUP_ORDER = ("lean", "vote", "general", "ballot", "foreign", "oneoff", "primary")


def study_group() -> pl.Expr:
    return (pl.when(pl.col("lean").is_not_null()).then(pl.lit("lean"))
            .when(pl.col("party_line").is_not_null()).then(pl.lit("vote"))
            .when(pl.col("ekind").is_not_null()).then(pl.col("ekind")).otherwise(pl.lit("oneoff")))


def study_set(u: pl.DataFrame) -> pl.DataFrame:
    """Markets each hypothesis needs (frozen groups); one row per market with its fetch window.
    US primaries: a seeded random sample of ``PRIMARY_SAMPLE_EVENTS`` events (all legs)."""
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    non_ladder = ~pl.col("ladder_text").fill_null(False)
    one_off = (pl.col("ekind").is_null() & non_ladder
               & (pl.col("drama") | pl.col("procedural") | pl.col("default_yes")))
    want = (pl.col("lean").is_not_null() | pl.col("party_line").is_not_null()
            | (pl.col("ekind").is_in(["general", "foreign", "ballot", "primary"]) & non_ladder) | one_off)
    s = u.filter(want & (pl.col("close_time") >= datetime(2024, 7, 1, tzinfo=UTC))).with_columns(
        study_group().alias("group"))
    prim = s.filter(pl.col("group") == "primary")["event_ticker"].unique().sort()
    keep = prim.sample(min(PRIMARY_SAMPLE_EVENTS, prim.len()), seed=SEED).to_list()
    s = s.filter((pl.col("group") != "primary") | pl.col("event_ticker").is_in(keep))
    proc_only = (pl.col("group") == "oneoff") & ~pl.col("drama") & ~pl.col("default_yes")
    pev = s.filter(proc_only)["event_ticker"].unique().sort()
    pkeep = pev.sample(int(PROC_SAMPLE_FRAC * pev.len()), seed=SEED).to_list()
    s = s.filter(~proc_only | pl.col("event_ticker").is_in(pkeep))
    hi = pl.min_horizontal(pl.col("close_time") + timedelta(hours=1), pl.lit(now))
    s = s.with_columns(hi.alias("hi")).with_columns(
        pl.max_horizontal(pl.col("open_time"), pl.col("hi") - timedelta(hours=MAX_PERIODS - 1)).alias("lo"))
    # markets closing in the holdout only serve holdout decisions (and their marks): from 8 days before it
    s = s.with_columns(pl.when(pl.col("close_time") >= DEV_END)
                       .then(pl.max_horizontal(pl.col("lo"), pl.lit(DEV_END - timedelta(days=8))))
                       .otherwise(pl.col("lo")).alias("lo"))
    # one-off contracts only need the last 30 days before the close; elections the whole life
    need_lo = pl.when(pl.col("group") == "oneoff").then(
        pl.max_horizontal(pl.col("lo"), pl.col("hi") - timedelta(days=30))).otherwise(pl.col("lo"))
    return s.with_columns(need_lo.alias("need_lo")).filter(pl.col("hi") > pl.col("lo"))


def fetch_candles(limit: int | None = None) -> None:
    """Sequential (<= 1 request/s): historical endpoint per market, live batch per event."""
    d = OUT / "candles"
    d.mkdir(parents=True, exist_ok=True)
    u = pl.read_parquet(OUT / "universe.parquet")
    s = study_set(u)
    cov = covered_windows()
    s = s.join(cov, on="ticker", how="left").filter(
        pl.col("c_lo").is_null() | (pl.col("c_lo") > pl.col("need_lo") + timedelta(days=2))
        | (pl.col("c_hi") < pl.col("hi") - timedelta(hours=3)))
    done_p = d / "done.parquet"
    done = set(pl.read_parquet(done_p)["ticker"]) if done_p.exists() else set()
    s = s.filter(~pl.col("ticker").is_in(list(done)))
    k = _api()
    cutoff = k.cutoff()["market_settled_ts"]
    hist = s.filter(pl.col("settlement_ts").is_not_null() & (pl.col("settlement_ts") < cutoff))
    live = s.filter(pl.col("settlement_ts").is_null() | (pl.col("settlement_ts") >= cutoff))
    order = pl.col("group").replace_strict({g: i for i, g in enumerate(GROUP_ORDER)}, default=99)
    jobs: list[tuple[str, list[dict[str, Any]]]] = []
    # live markets packed across events by window (the cap is periods x tickers per request)
    cur: list[dict[str, Any]] = []
    for r in live.sort("lo", "hi", "ticker").iter_rows(named=True):
        lo = min([x["lo"] for x in cur] + [r["lo"]])
        hi = max([x["hi"] for x in cur] + [r["hi"]])
        span = math.ceil((hi - lo).total_seconds() / 3600) + 1
        if cur and (span * (len(cur) + 1) > MAX_BATCH_CANDLES or len(cur) >= MAX_BATCH_TICKERS):
            jobs.append(("batch", cur))
            cur = []
        cur.append(r)
    if cur:
        jobs.append(("batch", cur))
    jobs += [("hist", [r]) for r in hist.with_columns(order.alias("o")).sort("o", "close_time").iter_rows(named=True)]
    if limit:
        jobs = jobs[:limit]
    print(f"candles: {s.height} markets to fetch in {len(jobs)} requests (cutoff {cutoff:%Y-%m-%d})", flush=True)
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    part = len(list(d.glob("part-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, fin, part
        if buf:
            pl.DataFrame(buf, schema=CANDLE_SCHEMA).write_parquet(d / f"part-{part:05d}.parquet")
            part += 1
        done.update(fin)
        pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}).write_parquet(done_p)
        buf, fin = [], []

    for i, (kind, rows) in enumerate(jobs, 1):
        try:
            if kind == "batch" and len(rows) > 1:
                lo, hi = min(r["lo"] for r in rows), max(r["hi"] for r in rows)
                page = k.candles_batch([r["ticker"] for r in rows], lo, hi, 60)
                got = {m["market_ticker"]: m for m in page.payload.get("markets") or []}
                for r in rows:
                    if r["ticker"] in got:
                        buf.extend(candle_rows(got[r["ticker"]], r["ticker"]))
                        fin.append(r["ticker"])
            else:
                r = rows[0]
                for h in ((kind == "hist"), (kind != "hist")):
                    try:
                        page = k.candles(r["ticker"], r["lo"], r["hi"], 60, h)
                        buf.extend(candle_rows(page.payload, r["ticker"]))
                        fin.append(r["ticker"])
                        break
                    except RuntimeError as error:
                        if "404" not in str(error):
                            raise
        except RuntimeError as error:
            print(f"  skip {rows[0]['ticker']}: {str(error)[-200:]}", flush=True)
        if i % 200 == 0 or i == len(jobs):
            flush()
            el = time.time() - t0
            print(f"  {i}/{len(jobs)} requests, {len(done)} markets, {i / el:.2f} req/s, "
                  f"eta {(len(jobs) - i) / max(i / el, 1e-9) / 60:.0f} min", flush=True)
    flush()
    print("candles done", flush=True)


# ---- anchors -------------------------------------------------------------------------------------
_MON = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                         "NOV", "DEC"))}
_TDATE = re.compile(r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})")
_DISTRICT = re.compile(r"\b([A-Z]{2})-\d{1,2}\b")
US_GENERAL_DAYS = {"24": datetime(2024, 11, 5).date(), "25": datetime(2025, 11, 4).date(),
                   "26": datetime(2026, 11, 3).date()}
STATES = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
          "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
          "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
          "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
          "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
          "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
          "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
          "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
          "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
          "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY"}
_STATE_RX = re.compile(r"\b(" + "|".join(sorted(STATES, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def ticker_date(event_ticker: str) -> Any:
    m = _TDATE.search(event_ticker or "")
    if not m:
        return None
    try:
        return datetime(2000 + int(m.group(1)), _MON[m.group(2)], int(m.group(3))).date()
    except ValueError:
        return None


def us_state(title: str | None, sub: str | None, series: str | None) -> str | None:
    text = f"{title or ''} | {sub or ''}"
    m = _DISTRICT.search(text)
    if m and m.group(1) in STATES.values():
        return m.group(1)
    m = _STATE_RX.search(text)
    if m:
        return STATES[m.group(1).lower()]
    sm = re.match(r"^KX([A-Z]{2})PRIMARY", series or "")
    return sm.group(1) if sm and sm.group(1) in STATES.values() else None


def _call_date(close: datetime | None) -> Any:
    """ET date of an election night from a call / close time (calls before 14:00 ET belong to the
    previous evening)."""
    if close is None:
        return None
    loc = close.astimezone(ET)
    return (loc - timedelta(days=1)).date() if loc.hour < 14 else loc.date()


def election_dates(u: pl.DataFrame) -> pl.DataFrame:
    """Per election event: the election day (ET date), from information known before the election.

    * ``ticker``: a date code in the event ticker (``-26OCT04``);
    * ``general``: US general / special elections on the federal election days of 2024-26 (year
      from the event ticker or the rules), unless a ticker date says otherwise;
    * ``state_mode``: US primaries: the modal call date of *all* primary markets of the same state,
      year and round (runoff or not). The election day was public in advance; the mode over many
      races recovers it without depending on how fast any one race was called.
    Foreign elections without a ticker date get none (excluded from the election-night tests).
    """
    ev = (u.filter(pl.col("ekind").is_not_null())
          .group_by("event_ticker").agg(pl.col("series_ticker").first(), pl.col("ekind").first(),
                                       pl.col("title").first(), pl.col("yes_sub_title").first(),
                                       pl.col("rules_primary").first(), pl.col("close_time").max().alias("ev_close")))
    rows = []
    for r in ev.iter_rows(named=True):
        d, src = ticker_date(r["event_ticker"]), "ticker"
        st = us_state(r["title"], r["yes_sub_title"], r["series_ticker"])
        yr = None
        m = re.search(r"-(2[4-7])(?:\b|[A-Z]|$)", r["event_ticker"])
        if m:
            yr = m.group(1)
        if yr is None:
            m = re.search(r"\b20(2[4-7])\b", f"{r['title']} {r['rules_primary'] or ''}")
            yr = m.group(1) if m else None
        runoff = bool(re.search(r"runoff", f"{r['title']} {r['rules_primary'] or ''}", re.IGNORECASE))
        if d is None and r["ekind"] in ("general", "ballot") and yr in US_GENERAL_DAYS and not runoff:
            special = bool(re.search(r"special", f"{r['title']} {r['rules_primary'] or ''}", re.IGNORECASE))
            if not special:
                d, src = US_GENERAL_DAYS[yr], "general"
        rows.append({"event_ticker": r["event_ticker"], "ekind": r["ekind"], "state": st, "yr": yr,
                     "runoff": runoff, "e_date": d, "e_src": src if d is not None else None,
                     "call_date": _call_date(r["ev_close"])})
    e = pl.DataFrame(rows, schema={"event_ticker": pl.String, "ekind": pl.String, "state": pl.String,
                                   "yr": pl.String, "runoff": pl.Boolean, "e_date": pl.Date, "e_src": pl.String,
                                   "call_date": pl.Date})
    prim = e.filter((pl.col("ekind") == "primary") & pl.col("state").is_not_null() & pl.col("yr").is_not_null())
    mode = (prim.group_by("state", "yr", "runoff", "call_date").len()
            .sort("len", descending=True).group_by("state", "yr", "runoff").agg(
                pl.col("call_date").first().alias("mode_date"), pl.col("len").first().alias("mode_n"),
                pl.col("len").sum().alias("n_events")))
    e = e.join(mode, on=["state", "yr", "runoff"], how="left")
    use_mode = pl.col("e_date").is_null() & (pl.col("ekind") == "primary") & (pl.col("mode_n") >= 3)
    return e.with_columns(pl.when(use_mode).then(pl.col("mode_date")).otherwise(pl.col("e_date")).alias("e_date"),
                          pl.when(use_mode).then(pl.lit("state_mode")).otherwise(pl.col("e_src")).alias("e_src"))


def scheduled_end(u: pl.DataFrame) -> pl.DataFrame:
    """Deadline parsed from the rules (``pm_hazard.parse_window``) as the scheduled end of one-off
    contracts; falls back to ``expected_expiration_time`` (set at listing)."""
    from albert.research.pm_hazard import parse_window

    rows = []
    for t, ru, exp in u.select("ticker", "rules_primary", "expected_expiration_time").iter_rows():
        try:
            _, T, _ = parse_window(ru)
        except (ValueError, KeyError):
            T = None
        rows.append({"ticker": t, "T_sched": T or exp, "T_src": "rules" if T else ("expected" if exp else None)})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "T_sched": pl.Datetime("us", "UTC"), "T_src": pl.String})


# ---- quotes and statistics -----------------------------------------------------------------------
def quotes_at(c: pl.DataFrame, req: pl.DataFrame, max_age_h: float = 24 * 7) -> pl.DataFrame:
    """Adds bid, ask, mid, spread, q_ts: the last hourly candle that ended at or before ``t``."""
    cc = c.select("ticker", pl.col("end_ts").alias("q_ts"), "bid", "ask", "price", "volume").sort("q_ts")
    out = req.sort("t").join_asof(cc, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                   tolerance=timedelta(hours=max_age_h), check_sortedness=False)
    bid = pl.col("bid").fill_null(0.0)
    ask = pl.when(pl.col("ask").is_null() | (pl.col("ask") <= 0)).then(1.0).otherwise(pl.col("ask"))
    out = out.with_columns(bid.alias("bid"), ask.alias("ask"))
    return out.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                            (pl.col("ask") - pl.col("bid")).alias("spread"))


def cluster_t(x: np.ndarray, *groups: np.ndarray) -> float | None:
    from albert.research.counts import cluster_se

    if len(x) < 3 or any(len(np.unique(g)) < 2 for g in groups):
        return None
    se = cluster_se(x, *groups)
    return float(x.mean() / se) if se and math.isfinite(se) and se > 1e-12 else None


def summary(df: pl.DataFrame, col: str, *clusters: str) -> dict[str, Any]:
    """Mean of ``col`` (cents) with t clustered by each key in ``clusters`` and two-way by the first two."""
    if df.is_empty():
        return {"n": 0}
    x = df[col].to_numpy().astype(float)
    out: dict[str, Any] = {"n": len(x), "mean_c": round(100 * float(x.mean()), 2)}
    for k in clusters:
        g = df[k].cast(pl.String).fill_null("?").to_numpy()
        out[f"n_{k}"] = len(np.unique(g))
        t = cluster_t(x, g)
        out[f"t_{k}"] = None if t is None else round(t, 2)
    if len(clusters) >= 2:
        g1 = df[clusters[0]].cast(pl.String).fill_null("?").to_numpy()
        g2 = df[clusters[1]].cast(pl.String).fill_null("?").to_numpy()
        t = cluster_t(x, g1, g2)
        out["t_2way"] = None if t is None else round(t, 2)
    return out


def taker_fee_order(n: pl.Expr, price: pl.Expr, mult: pl.Expr) -> pl.Expr:
    """Kalshi taker fee 0.07 x mult x n x P(1-P), rounded up to the cent per order (dollars)."""
    raw = 0.07 * mult.fill_null(1.0) * n * price * (1 - price)
    return (raw * 100).round(9).ceil() / 100


def take(df: pl.DataFrame, side: str = "side", stake: float = 5.0) -> pl.DataFrame:
    """Taker fill of ``side`` ('yes' buys YES at the ask, 'no' buys NO at 1 - bid), ~``stake`` dollars per
    order, fee rounded up per order, held to settlement. Adds price, n, fee, pnl (per contract, $)."""
    yes = pl.col(side) == "yes"
    price = pl.when(yes).then(pl.col("ask")).otherwise(1 - pl.col("bid"))
    payoff = pl.when(yes).then(pl.col("y")).otherwise(1 - pl.col("y"))
    mid_side = pl.when(yes).then(pl.col("mid")).otherwise(1 - pl.col("mid"))
    df = df.with_columns(price.alias("price"), payoff.alias("payoff"), mid_side.alias("mid_side"))
    df = df.filter((pl.col("price") > 0.005) & (pl.col("price") < 0.995))
    df = df.with_columns(pl.max_horizontal(pl.lit(1), (stake / pl.col("price")).floor()).cast(pl.Int64).alias("n"))
    df = df.with_columns(taker_fee_order(pl.col("n"), pl.col("price"), pl.col("fee_multiplier")).alias("fee_order"))
    return df.with_columns((pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n")).alias("pnl"),
                           (pl.col("payoff") - pl.col("mid_side")).alias("gross_mid"),
                           ((pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n"))
                            / (pl.col("price") + pl.col("fee_order") / pl.col("n"))).alias("ret"))


def bucket(col: str = "mid_side") -> pl.Expr:
    return (pl.when(pl.col(col) < 0.10).then(pl.lit("a <10c")).when(pl.col(col) < 0.30).then(pl.lit("b 10-30c"))
            .when(pl.col(col) < 0.50).then(pl.lit("c 30-50c")).when(pl.col(col) < 0.70).then(pl.lit("d 50-70c"))
            .when(pl.col(col) < 0.90).then(pl.lit("e 70-90c")).otherwise(pl.lit("f >=90c")))


# ---- panels --------------------------------------------------------------------------------------
LEADS_H = (168, 24, 6)
NIGHT_SLOTS = (("d0_20", 0, 20), ("d0_21", 0, 21), ("d0_22", 0, 22), ("d0_23", 0, 23), ("d1_00", 1, 0),
               ("d1_02", 1, 2), ("d1_06", 1, 6), ("d1_12", 1, 12), ("d2_12", 2, 12), ("d7_12", 7, 12))
PROXY_TS = datetime(2026, 11, 17, 23, 0, tzinfo=UTC)


def _split(t: pl.Expr, settle: pl.Expr) -> pl.Expr:
    return (pl.when((t < DEV_END) & (settle < DEV_END)).then(pl.lit("dev"))
            .when((t >= DEV_END) & (t < HOLD_END)).then(pl.lit("holdout")).otherwise(pl.lit("gap")))


def outcomes(u: pl.DataFrame, c: pl.DataFrame) -> pl.DataFrame:
    """y per market: settled result; for markets unsettled at the data end (11-03 midterms) a proxy
    from the 11-17 23:00 UTC mid (>= 97c -> 1, <= 3c -> 0, else unknown), flagged ``y_proxy``."""
    last = quotes_at(c, u.select("ticker").with_columns(pl.lit(PROXY_TS).alias("t")), max_age_h=24 * 30)
    last = last.with_columns(pl.when((pl.col("bid") > 0) | (pl.col("ask") < 1)).then(pl.col("mid"))
                             .otherwise(pl.col("price")).alias("m_last")).select("ticker", "m_last")
    x = u.join(last, on="ticker", how="left")
    settled = pl.col("result").is_in(["yes", "no"])
    proxy = pl.when(pl.col("m_last") >= 0.97).then(1).when(pl.col("m_last") <= 0.03).then(0)
    return x.with_columns(
        pl.when(settled).then((pl.col("result") == "yes").cast(pl.Int8)).otherwise(proxy.cast(pl.Int8)).alias("y"),
        (~settled).alias("y_proxy"),
        pl.when(settled).then(pl.col("settlement_ts")).otherwise(pl.lit(HOLD_END)).alias("settle_ts"))


def lead_panel(u: pl.DataFrame, c: pl.DataFrame, ed: pl.DataFrame) -> pl.DataFrame:
    """One row per (market, lead): quote at T - lead, T = election day 00:00 ET (elections with an
    ex-ante date), else the rules deadline / expected expiration, else (flagged) the event close."""
    sch = scheduled_end(u)
    x = (u.join(sch, on="ticker", how="left").join(ed.select("event_ticker", "e_date", "e_src"), on="event_ticker",
                                                    how="left")
         .with_columns(pl.col("close_time").max().over("event_ticker").alias("ev_close")))
    e0 = pl.col("e_date").cast(pl.Datetime("us")).dt.replace_time_zone(str(ET)).dt.convert_time_zone("UTC")
    # roll-call votes have no date in the rules: the vote (= event close) is the anchor (flagged 'close');
    # votes are scheduled days ahead (cloture filings), so leads of 1-7 d before it are known in advance
    one_off = pl.col("ekind").is_null() & pl.col("party_line").is_null() & pl.col("T_sched").is_not_null()
    T = (pl.when(pl.col("e_date").is_not_null()).then(e0).when(one_off).then(pl.col("T_sched"))
         .otherwise(pl.col("ev_close")))
    Tsrc = (pl.when(pl.col("e_date").is_not_null()).then(pl.col("e_src")).when(one_off).then(pl.col("T_src"))
            .otherwise(pl.lit("close")))
    x = x.with_columns(T.alias("T"), Tsrc.alias("T_src2"))
    req = (x.select("ticker", "T").join(pl.DataFrame({"lead_h": list(LEADS_H)}), how="cross")
           .with_columns((pl.col("T") - pl.duration(hours=pl.col("lead_h"))).alias("t")))
    q = quotes_at(c, req.filter(pl.col("t").is_not_null()))
    q = q.join(x.drop("T"), on="ticker", how="inner").filter(
        (pl.col("open_time") <= pl.col("t") - timedelta(hours=1)) & (pl.col("close_time") > pl.col("t"))
        & pl.col("q_ts").is_not_null() & pl.col("y").is_not_null())
    return q.with_columns(_split(pl.col("t"), pl.col("settle_ts")).alias("split"),
                          pl.col("t").dt.convert_time_zone(str(ET)).dt.date().alias("date"))


def night_panel(u: pl.DataFrame, c: pl.DataFrame, ed: pl.DataFrame) -> pl.DataFrame:
    """Election-night slots: one row per (event, slot) with the leader (highest-mid open leg, or the
    favoured side of a one-leg event) at e_date + day offset at the slot's ET hour."""
    ev = ed.filter(pl.col("e_date").is_not_null())
    legs = u.join(ev.select("event_ticker", "e_date", "e_src", "state"), on="event_ticker", how="inner")
    slots = pl.DataFrame({"slot": [s[0] for s in NIGHT_SLOTS], "dd": [s[1] for s in NIGHT_SLOTS],
                          "hh": [s[2] for s in NIGHT_SLOTS]})
    req = legs.select("ticker", "event_ticker", "e_date").join(slots, how="cross").with_columns(
        ((pl.col("e_date").cast(pl.Datetime("us")) + pl.duration(days=pl.col("dd"), hours=pl.col("hh")))
         .dt.replace_time_zone(str(ET)).dt.convert_time_zone("UTC")).alias("t"))
    q = quotes_at(c, req.select("ticker", "event_ticker", "slot", "t"))
    q = q.join(legs.drop("event_ticker"), on="ticker", how="inner").filter(
        (pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t")) & pl.col("q_ts").is_not_null()
        & (pl.col("ask") > pl.col("bid")) & pl.col("y").is_not_null())
    n_legs = u.group_by("event_ticker").len().rename({"len": "n_legs"})
    q = q.join(n_legs, on="event_ticker", how="left")
    # leader: one-leg events -> the side with mid >= 0.5; multi-leg -> the open leg with the highest mid
    q = q.with_columns(pl.col("mid").rank("ordinal", descending=True).over("event_ticker", "slot").alias("rk"))
    lead = q.filter((pl.col("n_legs") == 1) | (pl.col("rk") == 1)).with_columns(
        pl.when((pl.col("n_legs") > 1) | (pl.col("mid") >= 0.5)).then(pl.lit("yes")).otherwise(pl.lit("no"))
        .alias("side"))
    return lead.with_columns(_split(pl.col("t"), pl.col("settle_ts")).alias("split"),
                             pl.col("t").dt.convert_time_zone(str(ET)).dt.date().alias("date"))


PANEL_CHUNK_EVENTS = 400


def build_panels() -> None:
    """Panels in chunks of whole events (candles for one chunk at a time keep the process < ~1 GB)."""
    u = pl.read_parquet(OUT / "universe.parquet")
    s = study_set(u)
    ed = election_dates(s)                       # needs no prices
    ed.write_parquet(OUT / "election_dates.parquet")
    events = s["event_ticker"].unique().sort().to_list()
    studies, lps, nps = [], [], []
    n_c = 0
    for i in range(0, len(events), PANEL_CHUNK_EVENTS):
        sc = s.filter(pl.col("event_ticker").is_in(events[i:i + PANEL_CHUNK_EVENTS]))
        c = load_candles(sc["ticker"].to_list())
        n_c += c.height
        sc = outcomes(sc, c)
        studies.append(sc)
        lps.append(lead_panel(sc, c, ed))
        nps.append(night_panel(sc, c, ed))
        del c
    s = pl.concat(studies, how="vertical_relaxed")
    s.write_parquet(OUT / "study.parquet")
    lp = pl.concat(lps, how="vertical_relaxed")
    lp.write_parquet(OUT / "lead_panel.parquet")
    npn = pl.concat(nps, how="vertical_relaxed")
    npn.write_parquet(OUT / "night_panel.parquet")
    print(f"panel: {s.height} study markets, {n_c} candles; lead panel {lp.height} rows "
          f"({lp['split'].value_counts().sort('split').to_dicts()}); night panel {npn.height} rows", flush=True)


# ---- exploration (descriptive, any split) ---------------------------------------------------------
USABLE_SPREAD = 0.15


def _usable(df: pl.DataFrame, max_spread: float = USABLE_SPREAD) -> pl.DataFrame:
    return df.filter((pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("spread") <= max_spread))


def side_rows(df: pl.DataFrame, side: pl.Expr) -> pl.DataFrame:
    """Per row the residual of the named side: r_side = y_side - mid_side (dollars)."""
    df = df.with_columns(side.alias("side"))
    yes = pl.col("side") == "yes"
    return df.filter(pl.col("side").is_in(["yes", "no"])).with_columns(
        pl.when(yes).then(pl.col("mid")).otherwise(1 - pl.col("mid")).alias("mid_side"),
        pl.when(yes).then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64).alias("y_side")).with_columns(
        (pl.col("y_side") - pl.col("mid_side")).alias("r_side"))


def hyp_frames(lp: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The rows each hypothesis looks at, with the side the mechanism says is *underpriced*
    (``side``; ``r_side`` > 0 supports the mechanism)."""
    lp = _usable(lp)
    out: dict[str, pl.DataFrame] = {}
    # H1 lean: side = the D side (positive r_side = pro-D underpriced = pro-R overpriced); one leg per event
    h1 = lp.filter(pl.col("lean").is_in(["R", "D"])).sort("ticker").unique(["event_ticker", "lead_h"], keep="first")
    out["H1_lean"] = side_rows(h1, pl.when(pl.col("lean") == "D").then(pl.lit("yes")).otherwise(pl.lit("no")))
    h1a = lp.filter(pl.col("approval_lean").is_in(["R", "D"]))
    out["H1_approval"] = side_rows(h1a, pl.when(pl.col("approval_lean") == "D").then(pl.lit("yes"))
                                   .otherwise(pl.lit("no")))
    one_off = lp.filter(pl.col("ekind").is_null() & pl.col("party_line").is_null()
                        & ~pl.col("ladder_text") & pl.col("T_src2").is_in(["rules", "expected"]))
    out["H2_drama"] = side_rows(one_off.filter(pl.col("drama")), pl.lit("no"))
    out["H2_control"] = side_rows(one_off.filter(~pl.col("drama") & pl.col("procedural")), pl.lit("no"))
    out["H7_default_yes"] = side_rows(one_off.filter(pl.col("default_yes")), pl.lit("yes"))
    fo = lp.filter((pl.col("ekind") == "foreign") & ~pl.col("ladder_text"))
    out["H6_populist"] = side_rows(fo.filter(pl.col("populist").fill_null(False)), pl.lit("no"))
    out["H6_other"] = side_rows(fo.filter(~pl.col("populist").fill_null(False)), pl.lit("no"))
    out["H9_ballot"] = side_rows(lp.filter(pl.col("ekind") == "ballot"), pl.lit("no"))
    pl_rows = lp.filter(pl.col("party_line").is_in(["yes", "no"]))
    funding = pl.col("title").str.contains(FUNDING_RX.pattern)
    out["H17_party_line"] = side_rows(pl_rows.filter(~funding), pl.col("party_line"))
    out["H17_funding"] = side_rows(pl_rows.filter(funding), pl.col("party_line"))
    return out


def theme_expr() -> pl.Expr:
    """Correlation cluster: the election day for elections, else the series (one question family)."""
    return pl.col("e_date").cast(pl.String).fill_null(pl.col("series_ticker")).alias("theme")


def describe(frames: dict[str, pl.DataFrame], split: str) -> list[dict[str, Any]]:
    rows = []
    for name, f in frames.items():
        f = f.filter(pl.col("split") == split)
        if f.is_empty():
            rows.append({"hyp": name, "split": split, "cut": "all", "n": 0})
            continue
        f = f.with_columns(bucket("mid_side").alias("bkt"), theme_expr())
        for lead in [None, *LEADS_H]:
            g = f if lead is None else f.filter(pl.col("lead_h") == lead)
            rows.append({"hyp": name, "split": split, "cut": "all" if lead is None else f"lead {lead}h",
                         **summary(g, "r_side", "event_ticker", "date", "theme")})
        for b in sorted(f["bkt"].unique().to_list()):
            rows.append({"hyp": name, "split": split, "cut": f"bucket {b}",
                         **summary(f.filter(pl.col("bkt") == b), "r_side", "event_ticker", "date", "theme")})
    return rows


# ---- election night ------------------------------------------------------------------------------
def night_rows(npn: pl.DataFrame, lo: float = 0.60, hi: float = 0.97, max_spread: float = 0.10) -> pl.DataFrame:
    """Leader bought at the ask (taker, fee per order) at each slot; held to settlement."""
    f = npn.filter((pl.col("spread") <= max_spread) & (pl.col("bid") > 0) & (pl.col("ask") < 1))
    f = take(f, "side")
    f = f.filter((pl.col("mid_side") >= lo) & (pl.col("mid_side") <= hi))
    return f.with_columns(((pl.col("settle_ts") - pl.col("t")).dt.total_hours() / 24).alias("hold_d"),
                          bucket("mid_side").alias("bkt"),
                          theme_expr())


def night_describe(npn: pl.DataFrame, split: str) -> list[dict[str, Any]]:
    rows = []
    f = night_rows(npn).filter(pl.col("split") == split)
    for kind in ["all", "general", "primary", "foreign", "ballot"]:
        g = f if kind == "all" else f.filter(pl.col("ekind") == kind)
        for slot in [x[0] for x in NIGHT_SLOTS]:
            h = g.filter(pl.col("slot") == slot)
            if h.is_empty():
                continue
            rows.append({"split": split, "ekind": kind, "slot": slot, **summary(h, "gross_mid", "event_ticker", "theme"),
                         "net_c": round(100 * float(h["pnl"].mean()), 2), "ret_pct": round(100 * float(h["ret"].mean()), 2),
                         "t_net_ev": cluster_t(h["pnl"].to_numpy(), h["event_ticker"].to_numpy()),
                         "hold_d_med": round(float(h["hold_d"].median()), 1)})
    return rows


def count_order(u: pl.DataFrame, c: pl.DataFrame, ed: pl.DataFrame, split: str) -> list[dict[str, Any]]:
    """H15: per leg, the early-count move (20:00 -> 23:00 ET on election night) and what follows
    (23:00 ET -> settlement, i.e. y - mid at 23:00). Reversal = negative slope / sign flip."""
    ev = ed.filter(pl.col("e_date").is_not_null() & pl.col("ekind").is_in(["general", "primary", "ballot"]))
    legs = u.join(ev.select("event_ticker", "e_date"), on="event_ticker", how="inner")
    mk = []
    for name, dd, hh in (("m20", 0, 20), ("m23", 0, 23)):
        req = legs.select("ticker", "e_date").with_columns(
            ((pl.col("e_date").cast(pl.Datetime("us")) + pl.duration(days=dd, hours=hh))
             .dt.replace_time_zone(str(ET)).dt.convert_time_zone("UTC")).alias("t"))
        q = quotes_at(c, req.select("ticker", "t"), max_age_h=48).filter(
            (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("spread") <= 0.15))
        mk.append(q.select("ticker", pl.col("mid").alias(name), pl.col("t").alias(f"t_{name}")))
    x = legs.join(mk[0], on="ticker").join(mk[1], on="ticker").filter(pl.col("close_time") > pl.col("t_m23"))
    x = x.with_columns((pl.col("m23") - pl.col("m20")).alias("move1"), (pl.col("y") - pl.col("m23")).alias("after"),
                       _split(pl.col("t_m23"), pl.col("settle_ts")).alias("split"), theme_expr(),
                       pl.col("t_m23").dt.convert_time_zone(str(ET)).dt.date().alias("date"))
    x = x.filter(pl.col("split") == split)
    rows = []
    for lab, cond in (("up >= 10c", pl.col("move1") >= 0.10), ("down <= -10c", pl.col("move1") <= -0.10),
                      ("flat |move| < 3c", pl.col("move1").abs() < 0.03)):
        g = x.filter(cond)
        rows.append({"split": split, "early_move": lab, **summary(g, "after", "event_ticker", "theme")})
    # signed: after-move in the direction of the early move (negative = reversal)
    big = x.filter(pl.col("move1").abs() >= 0.10).with_columns(
        (pl.col("after") * pl.col("move1").sign()).alias("cont"))
    rows.append({"split": split, "early_move": "|move| >= 10c, signed continuation",
                 **summary(big, "cont", "event_ticker", "theme")})
    return rows


# ---- ledgers (capped $100 book) and Sharpe / vol ------------------------------------------------------
BANKROLL = 100.0
EVENT_CAP = 5.0


def capped_ledger(tr: pl.DataFrame, strategy: str, variant: str, split: str,
                  bankroll: float = BANKROLL, event_cap: float = EVENT_CAP) -> pl.DataFrame:
    """Trades in time order on a $``bankroll`` book: taken only while open cost (incl. this one) stays
    within the bankroll and the event's open cost within ``event_cap``; capital returns at settlement.
    An order that does not fit is shrunk to the contracts that do (fee recomputed)."""
    import heapq

    heap: list[tuple[Any, float, str]] = []
    used, per_ev, rows = 0.0, {}, []
    for r in tr.sort("t", "ticker").iter_rows(named=True):   # ties (same slot): by ticker, outcome-blind
        while heap and heap[0][0] <= r["t"]:
            _, cst, e = heapq.heappop(heap)
            used -= cst
            per_ev[e] -= cst
        room = min(bankroll - used, event_cap - per_ev.get(r["event_ticker"], 0.0))
        n = int(r["n"])
        mult = r.get("fee_multiplier") or 1.0
        while n > 0:
            fee = math.ceil(round(0.07 * mult * n * r["price"] * (1 - r["price"]) * 100, 9)) / 100
            if n * r["price"] + fee <= room + 1e-9:
                break
            n -= 1
        if n <= 0:
            continue
        cost = n * r["price"] + fee
        settle = max(r["settle_ts"], r["t"] + timedelta(hours=1))
        used += cost
        per_ev[r["event_ticker"]] = per_ev.get(r["event_ticker"], 0.0) + cost
        heapq.heappush(heap, (settle, cost, r["event_ticker"]))
        rows.append({"strategy": strategy, "variant": variant, "split": split,
                     "struct_id": f"{r['ticker']}|{r['t'].isoformat()}", "event_ticker": r["event_ticker"],
                     "ticker": r["ticker"], "side": r["side"], "n": n, "t_in": r["t"], "cost_usd": cost,
                     "settle_ts": settle, "payoff_usd": n * float(r["payoff"])})
    return pl.DataFrame(rows, schema={"strategy": pl.String, "variant": pl.String, "split": pl.String,
                                      "struct_id": pl.String, "event_ticker": pl.String, "ticker": pl.String,
                                      "side": pl.String, "n": pl.Int64, "t_in": pl.Datetime("us", "UTC"),
                                      "cost_usd": pl.Float64, "settle_ts": pl.Datetime("us", "UTC"),
                                      "payoff_usd": pl.Float64})


def marks_for(c: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    x = c.filter(pl.col("ticker").is_in(tickers))
    two = (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    return (x.with_columns(pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).otherwise(pl.col("price"))
                           .alias("yes_mid"))
            .filter(pl.col("yes_mid").is_not_null()).select("ticker", pl.col("end_ts").alias("ts"), "yes_mid"))


def ledger_stats(led: pl.DataFrame, c: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    if led.is_empty():
        return {"positions": 0}
    m = ledger_metrics(led, marks_for(c, led["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe",
            "sharpe_ci90", "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd",
            "weeks_positive_pct", "skew", "first_day", "last_day", "days")
    return {k: m.get(k) for k in keep}


def drama_regression(fr: dict[str, pl.DataFrame], split: str) -> dict[str, Any]:
    """H2: r_NO on a drama dummy with price-bucket x lead fixed effects (one-off contracts), t by event."""
    a = pl.concat([fr["H2_drama"].with_columns(pl.lit(1.0).alias("dr")),
                   fr["H2_control"].with_columns(pl.lit(0.0).alias("dr"))], how="diagonal_relaxed")
    a = a.filter(pl.col("split") == split).with_columns(bucket("mid_side").alias("bkt"))
    if a.height < 20 or a["dr"].n_unique() < 2:
        return {"n": a.height}
    cell = pl.concat_str([pl.col("bkt"), pl.col("lead_h").cast(pl.String)], separator="|")
    a = a.with_columns(cell.alias("cell"))
    # within-cell demeaning (FE), then OLS of r on the demeaned dummy
    a = a.with_columns((pl.col("r_side") - pl.col("r_side").mean().over("cell")).alias("ry"),
                       (pl.col("dr") - pl.col("dr").mean().over("cell")).alias("dx"))
    x, y = a["dx"].to_numpy(), a["ry"].to_numpy()
    sxx = float((x * x).sum())
    if sxx <= 0:
        return {"n": a.height}
    b = float((x * y).sum() / sxx)
    e = y - b * x
    g = a["event_ticker"].to_numpy()
    _, inv = np.unique(g, return_inverse=True)
    sc = np.bincount(inv, weights=x * e)
    k = len(sc)
    se = math.sqrt((sc ** 2).sum() * k / (k - 1)) / sxx if k > 1 else float("nan")
    return {"n": a.height, "events": k, "drama_minus_control_c": round(100 * b, 2), "t_event": round(b / se, 2)}


def taker_summary(fr: dict[str, pl.DataFrame], split: str, max_spread: float = 0.10) -> list[dict[str, Any]]:
    rows = []
    for name, f in fr.items():
        f = f.filter((pl.col("split") == split) & (pl.col("spread") <= max_spread))
        if f.is_empty():
            continue
        t = take(f, "side").with_columns(theme_expr())
        for lead in [None, *LEADS_H]:
            g = t if lead is None else t.filter(pl.col("lead_h") == lead)
            if g.is_empty():
                continue
            rows.append({"hyp": name, "split": split, "cut": "all" if lead is None else f"lead {lead}h",
                         **summary(g, "pnl", "event_ticker", "date", "theme"),
                         "gross_mid_c": round(100 * float(g["gross_mid"].mean()), 2),
                         "ret_pct": round(100 * float(g["ret"].mean()), 2)})
    return rows


def explore(split: str = "dev") -> None:
    lp = pl.read_parquet(OUT / "lead_panel.parquet")
    npn = pl.read_parquet(OUT / "night_panel.parquet")
    fr = hyp_frames(lp)
    d = pl.DataFrame(describe(fr, split), infer_schema_length=None)
    d.write_csv(OUT / f"explore_{split}_resid.csv")
    tk = pl.DataFrame(taker_summary(fr, split), infer_schema_length=None)
    tk.write_csv(OUT / f"explore_{split}_taker.csv")
    nd = pl.DataFrame(night_describe(npn, split), infer_schema_length=None)
    nd.write_csv(OUT / f"explore_{split}_night.csv")
    ed = pl.read_parquet(OUT / "election_dates.parquet")
    us_ev = ed.filter(pl.col("e_date").is_not_null() & pl.col("ekind").is_in(["general", "primary", "ballot"]))
    st = pl.read_parquet(OUT / "study.parquet").filter(pl.col("event_ticker").is_in(us_ev["event_ticker"].implode()))
    co = pl.DataFrame(count_order(st, load_candles(st["ticker"].to_list()), ed, split), infer_schema_length=None)
    co.write_csv(OUT / f"explore_{split}_count_order.csv")
    reg = drama_regression(fr, split)
    (OUT / f"explore_{split}_h2_regression.json").write_text(json.dumps(reg, indent=1))
    with pl.Config(tbl_rows=200, tbl_width_chars=250, tbl_cols=20):
        print(d)
        print(tk)
        print(nd)
        print(co)
    print("H2 regression", reg)


# ---- rule builders (parameterised; the frozen parameters live in PREREG) -----------------------------
def rule_night(npn: pl.DataFrame, slots: tuple[str, ...], lo: float, hi: float, kinds: tuple[str, ...],
               max_spread: float = 0.05, us_only: bool = True) -> pl.DataFrame:
    """Election night / post-call: buy the leader at the ask at the first listed slot where its mid is
    in [lo, hi] and the spread <= max_spread; one entry per event; held to settlement."""
    f = npn.filter(pl.col("slot").is_in(list(slots)) & pl.col("ekind").is_in(list(kinds)))
    if us_only:
        f = f.filter(pl.col("state").is_not_null() | (pl.col("e_src") == "general"))
    f = f.filter((pl.col("spread") <= max_spread) & (pl.col("bid") > 0) & (pl.col("ask") < 1))
    f = take(f, "side").filter((pl.col("mid_side") >= lo) & (pl.col("mid_side") <= hi))
    f = f.sort("t").unique("event_ticker", keep="first")
    return f.with_columns(theme_expr(), ((pl.col("settle_ts") - pl.col("t")).dt.total_hours() / 24).alias("hold_d"))


def rule_lead(lp: pl.DataFrame, hyp: str, lead_h: int, lo: float, hi: float, max_spread: float = 0.10) -> pl.DataFrame:
    """Lead-based rules: buy the side ``hyp_frames`` calls underpriced at T - lead when that side's mid
    is in [lo, hi]; taker at the ask (NO at 1 - bid), fee per order, held to settlement."""
    f = hyp_frames(lp)[hyp].filter((pl.col("lead_h") == lead_h) & (pl.col("spread") <= max_spread))
    f = take(f, "side").filter((pl.col("mid_side") >= lo) & (pl.col("mid_side") <= hi))
    return f.with_columns(theme_expr())


# ---- evaluation of a rule (dev and holdout) -----------------------------------------------------------
def evaluate(tr: pl.DataFrame, name: str, c: pl.DataFrame, splits: tuple[str, ...] = ("dev", "holdout"),
             extra: dict[str, pl.Expr] | None = None) -> list[dict[str, Any]]:
    """Per split: per-contract net PnL (c) with t by event / ET date / theme and two-way, $ return,
    and the capped-book ledger (Sharpe / vol via ``vol_metrics``). Writes the ledgers."""
    out = []
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    cuts: dict[str, pl.Expr] = {"all": pl.lit(True), **(extra or {})}
    for split in splits:
        f = tr.filter(pl.col("split") == split)
        for cut, expr in cuts.items():
            g = f.filter(expr)
            if g.is_empty():
                out.append({"rule": name, "split": split, "cut": cut, "n": 0})
                continue
            row: dict[str, Any] = {"rule": name, "split": split, "cut": cut,
                                   **summary(g, "pnl", "event_ticker", "date", "theme"),
                                   "gross_mid_c": round(100 * float(g["gross_mid"].mean()), 2),
                                   "usd_staked": round(float((g["n"] * g["price"] + g["fee_order"]).sum()), 2),
                                   "usd_pnl": round(float((g["n"] * g["pnl"]).sum()), 2),
                                   "win_pct": round(100 * float((g["payoff"] > 0.5).mean()), 1)}
            row["ret_per_usd_pct"] = round(100 * row["usd_pnl"] / row["usd_staked"], 2) if row["usd_staked"] else None
            if cut == "all":
                led = capped_ledger(g, name, cut, split)
                led.write_parquet(OUT / "ledgers" / f"ledger_{name}_{split}.parquet")
                st = ledger_stats(led, c)
                row.update({f"book_{k}": v for k, v in st.items()})
            out.append(row)
    return out


# ---- pre-registered rules (frozen in hypotheses.md before the holdout was opened) ----------------------
PREREG: dict[str, dict[str, Any]] = {
    "R1_night_leader": {"slots": ("d0_22", "d0_23", "d1_00"), "lo": 0.60, "hi": 0.95,
                        "kinds": ("general", "ballot"), "us_only": True},
    "R2_dayafter_leader": {"slots": ("d1_12",), "lo": 0.80, "hi": 0.985,
                           "kinds": ("general", "ballot", "primary", "foreign"), "us_only": False},
}


def run_tests() -> None:
    npn = pl.read_parquet(OUT / "night_panel.parquet")
    lp = pl.read_parquet(OUT / "lead_panel.parquet")
    traded = pl.concat([rule_night(npn, sp["slots"], sp["lo"], sp["hi"], sp["kinds"], us_only=sp["us_only"])["ticker"]
                        for sp in PREREG.values()]).unique().to_list()
    c = load_candles(traded)                     # marks for the ledgers only
    midterm = pl.col("e_date") == MIDTERM_DAY
    extra = {"midterm 11-03": midterm, "other elections": ~midterm, "settled only": ~pl.col("y_proxy"),
             "general": pl.col("ekind") == "general", "primary": pl.col("ekind") == "primary",
             "foreign": pl.col("ekind") == "foreign", "price >= 90c": pl.col("price") >= 0.90,
             "price < 90c": pl.col("price") < 0.90}
    rows: list[dict[str, Any]] = []
    trades = []
    for name, spec in PREREG.items():
        tr = rule_night(npn, spec["slots"], spec["lo"], spec["hi"], spec["kinds"], us_only=spec["us_only"])
        trades.append(tr.with_columns(pl.lit(name).alias("rule")))
        rows += evaluate(tr, name, c, extra=extra)
    res = pl.DataFrame(rows, infer_schema_length=None)
    res.with_columns(pl.col(pl.List(pl.Float64)).cast(pl.List(pl.String)).list.join(", ")).write_csv(
        OUT / "test_results.csv")
    keep = ["rule", "ticker", "event_ticker", "ekind", "e_date", "slot", "t", "side", "price", "mid_side", "n",
            "fee_order", "payoff", "pnl", "gross_mid", "ret", "hold_d", "y_proxy", "split", "title", "yes_sub_title"]
    pl.concat(trades, how="diagonal_relaxed").select([k for k in keep if k in trades[0].columns]).write_parquet(
        OUT / "test_trades.parquet")
    # descriptive holdout checks (not rules)
    fr = hyp_frames(lp)
    pl.DataFrame(describe(fr, "holdout"), infer_schema_length=None).write_csv(OUT / "holdout_descriptive.csv")
    (OUT / "holdout_h2_regression.json").write_text(json.dumps(drama_regression(fr, "holdout"), indent=1))
    pl.DataFrame(night_describe(npn, "holdout"), infer_schema_length=None).write_csv(OUT / "holdout_night.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=260, tbl_cols=40):
        print(res.select([x for x in ["rule", "split", "cut", "n", "n_event_ticker", "mean_c", "t_event_ticker",
                                      "t_2way", "gross_mid_c", "ret_per_usd_pct", "win_pct", "book_sharpe",
                                      "book_sharpe_ci90", "book_sharpe_ex_top5", "book_usd_per_week",
                                      "book_max_dd_usd"] if x in res.columns]))


# ---- report ----------------------------------------------------------------------------------------
VERDICTS = """| # | Mechanism | Verdict |
|---|---|---|
| H1 | Partisan wishful thinking | Rejected as a bias: sign = the cycle's polling error (D side 2024 -8.5c, 2025 +10c, 2026 midterms +4.3c on 531 races); longshots by party symmetric |
| H2 | Drama / action bias in one-offs | Rejected: drama -3.4c vs procedural -3.0c YES residual (pm_hazard panel, 4.9k quotes); drama - control dev +0.2c (t 0.1), holdout -2.2c (t -2.2, wrong sign) |
| H3 | Fade rumour up-jumps in drama contracts | Not run (momentum / spikes covered; no drama premium to fade) |
| H4 | Election-night denial, leader underpriced | Pre-registered R1 failed (midterms +0.2c, t 0.06); primaries ~0 in dev |
| H5 | Post-call capital lock-up | Pre-registered R2 failed (-3.1c, t -0.8); 2026 markets settle in days, not months |
| H6 | Projection onto foreign populists | Rejected: dev -6.4c, holdout +17.6c (sign flips), pooled +0.7c t 0.1 |
| H7 | Status-quo YES underpriced | Rejected: default-YES overpriced -3.8c (t -1.7): generic YES bias |
| H8 | Round-strike anchoring | No evidence, underpowered (72 quotes, t < 1.1) |
| H9 | Ballot-measure No bias | Untestable (dev 1 measure; holdout 4 measures, YES underpriced 17c) |
| H10-H14, H16 | Momentum, attention, lock-up pre-event, stale ladders, name salience, cross-venue | Covered by earlier studies; not re-run |
| H15 | Count-order reversal | Rejected: early election-night moves continue (+5c, 3 nights) |
| H17 | Roll-call defection hype | Dev party line +2.5c (t 0.9, 18 votes), taker +0.2c; no holdout votes |
"""


def _md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report() -> None:
    """``report.md`` from the saved results (``test`` and ``explore`` must have run)."""
    res = pl.read_csv(OUT / "test_results.csv")
    main = res.filter(pl.col("cut") == "all").select(
        "rule", "split", pl.col("n").alias("trades"), pl.col("n_event_ticker").alias("events"),
        pl.col("mean_c").alias("net_c"), pl.col("t_event_ticker").alias("t_event"), pl.col("gross_mid_c"),
        pl.col("ret_per_usd_pct").alias("ret_pct"), "win_pct", pl.col("book_sharpe").alias("sharpe"),
        pl.col("book_sharpe_ci90").alias("sharpe_ci90"), pl.col("book_sharpe_ex_top5").alias("sharpe_ex_top5"),
        pl.col("book_usd_per_week").alias("usd_per_week"), pl.col("book_max_dd_usd").alias("max_dd_usd"),
        pl.col("book_days").alias("days"))
    cuts = res.filter(pl.col("cut") != "all", pl.col("n") > 0).select(
        "rule", "split", "cut", "n", pl.col("mean_c").alias("net_c"), pl.col("t_event_ticker").alias("t_event"),
        "win_pct")
    night = pl.read_csv(OUT / "holdout_night.csv").filter(pl.col("ekind") == "all").select(
        "slot", "n", pl.col("mean_c").alias("gross_c"), "net_c", pl.col("t_net_ev").round(2).alias("t_event"),
        "hold_d_med")
    desc = pl.read_csv(OUT / "holdout_descriptive.csv").filter(pl.col("cut") == "all", pl.col("n") > 0).select(
        "hyp", "n", "mean_c", pl.col("n_event_ticker").alias("events"), pl.col("t_event_ticker").alias("t_event"))
    text = f"""# Behavioural mispricing in Kalshi politics / elections / news / world (generated by `behav_politics report`)

Mechanisms, frozen rules and the pre-registration: `hypotheses.md` (written before prices, and before
the holdout respectively). Dev = decisions and settlement before 2026-10-01; holdout = decisions
2026-10-01..11-17 (includes the 2026-11-03 midterms; every holdout trade settled, no outcome proxy).
Taker at the hourly candle's closing ask, ~$5 orders, Kalshi taker fee rounded up per order, held to
settlement; book $100, <= $5 per event, daily MTM at the candle mid (`vol_metrics`).

## Verdicts

{VERDICTS}
## Pre-registered rules, dev -> holdout

{_md_table(main)}

Both rules fail the pre-registered bar (net > 0 with t_event >= 2). Holdout clusters by date / theme
are degenerate (one election night). Only 23 of R1's 72 same-night holdout trades fit the $100 book.

Reported cuts (descriptive):

{_md_table(cuts)}

## Election night 2026-11-03 and the following days (leader at mid 60-97c, all holdout elections)

{_md_table(night)}

## Descriptive holdout checks (residual of the side the mechanism calls underpriced, c)

{_md_table(desc)}

## Multiple testing

About 215 dev cells (residual tables ~120, taker ~30, election-night slots ~50, count order 4,
rule variants 5, round strikes 6) and on the holdout 2 pre-registered tests + 16 cuts + ~60
descriptive cells: ~300 in all; a family-wise 5% bar is |t| ~ 3.8. Nothing pre-registered clears 2.

## Post-hoc leads (forward-monitor only)

1. Election-night leaders at ask 90-96c (R1 slots): dev 17/17 +5.6c, holdout 36/36 +6.0c net; three
   nights only; 95% Wilson lower bound on the win rate 93.2% < ~94.1% breakeven.
2. Leaders still open 2+ days after an election (slow counts): holdout +6.6c / +4.4c (22 / 13 events),
   dev +2.5c / +1.8c. Same population that lost R2 money a day earlier.

Election nights are too rare for steady weekly returns (~$0.2-0.5/week at $100 even if real).

Rerun: `python -m albert.research.behav_politics fetch-nov | universe | candles | panel | explore | test | report`
"""
    (OUT / "report.md").write_text(text)
    print(f"wrote {OUT / 'report.md'}", flush=True)


# ---- driver --------------------------------------------------------------------------------------
STEPS: dict[str, Any] = {"candles": fetch_candles, "panel": build_panels, "explore": explore, "test": run_tests,
                         "report": write_report}


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="albert.research.behav_politics")
    ap.add_argument("step", choices=["fetch-nov", "universe", *STEPS])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "fetch-nov":
        fetch_nov()
    elif a.step == "universe":
        build_universe()
    else:
        STEPS[a.step]()


if __name__ == "__main__":
    main()
