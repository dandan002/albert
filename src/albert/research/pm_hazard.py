"""Reduced-form hazard (credit-curve) model for Kalshi "will X happen by date T" contracts.

Model (shared notation). Before the event, ``p_t = 1 − E_t[exp(−∫_t^T h_u du)]``. With a
deterministic hazard ``p_t = 1 − exp(−H(t, T))``: absent the event the price decays at
``dp/dt = −h_t (1 − p_t)`` (theta) and at the event it jumps to 1; the compensated jump makes p a
martingale. Implied average hazard ``h̄_t = −ln(1 − p_t) / (T − t)``; forward hazards along a strip
``h(T_i, T_j) = −ln[(1 − p_j)/(1 − p_i)] / (T_j − T_i)``, and ``p(T_i) ≤ p(T_j)`` is a static
arbitrage bound. A CIR (Cox) intensity adds news diffusion; its closed-form survival is here too.

Study pipeline (``python -m albert.research.pm_hazard <step>``):

* ``scan``: every settled non-financial single-leg market from ``/historical/markets`` (newest
  first, back to ``START``) plus live settled markets closing just after the shared table's end,
  kept when its rules contain a deadline phrase. Cached under ``data/research/pm_hazard/``.
* ``universe``: shared ``kalshi_markets`` rows + scans, deadline ``T`` parsed from the rules text
  (never from ``close_time``/``settlement_ts``, which reveal an early YES), classified by text.
* ``candles``: hourly candles of each universe market over its last ``LOOKBACK`` before T.
* ``analyze``: tests 1-4 (calibration, theta, term structure, economic rule vs price control),
  report and ``summary.json``.

Look-ahead guards: only markets with ``T ≤ END`` and alive at some ``t ≥ START`` enter, so both
outcomes are observed (a deadline after the data end would only show its early YESes). Quotes come
from candles that ended strictly before the decision time. Parameters are chosen on dev
(``T`` before 2026-10-01) and reported on holdout (decisions in October 2026).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

ET = ZoneInfo("America/New_York")
START = datetime(2026, 1, 1, tzinfo=UTC)          # oldest decision time / scan horizon
END = datetime(2026, 11, 2, 5, 0, tzinfo=UTC)     # latest deadline (all such markets settled)
HOLDOUT = datetime(2026, 10, 1, 4, 0, tzinfo=UTC)  # dev: T < HOLDOUT; holdout: t >= HOLDOUT
TABLE_END = datetime(2026, 10, 31, 23, 59, 59, tzinfo=UTC)
LOOKBACK = timedelta(days=62)
BATCH_FROM = datetime(2026, 8, 15, tzinfo=UTC)   # the live batch candle endpoint is empty before this
LEADS_H = (1, 3, 6, 12, 24, 48, 72, 120, 168, 240, 336, 504, 720, 1080, 1440)
LEAD_BUCKETS = ((0, 6, "<6h"), (6, 24, "6-24h"), (24, 72, "1-3d"), (72, 168, "3-7d"),
                (168, 336, "7-14d"), (336, 720, "14-30d"), (720, 1e9, "30-60d"))
PRICE_BUCKETS = ((0.0, 0.05, "0-5"), (0.05, 0.15, "5-15"), (0.15, 0.35, "15-35"),
                 (0.35, 0.65, "35-65"), (0.65, 0.85, "65-85"), (0.85, 1.01, "85-100"))
MAX_SPREAD = 0.10       # fixed up front: quotes wider than this are not "priced" for tests 1-4
DESCRIPTIVE_MAX_MID = 0.9   # shape/theta use the proxy event time; above this it can predate the event
EPS = 0.005


def out_dir(s) -> Path:
    d = s.data_dir / "research" / "pm_hazard"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- hazard math -------------------------------------------------------------------------------
def implied_hazard(p: Any, tau: Any) -> Any:
    """Average hazard over ``tau`` implied by price ``p``: −ln(1 − p)/tau (p clipped to (0, 1))."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -np.log1p(-p) / np.asarray(tau, dtype=float)


def prob_from_hazard(h: Any, tau: Any) -> Any:
    """P(event within tau) under a flat hazard h: 1 − exp(−h·tau)."""
    return -np.expm1(-np.asarray(h, dtype=float) * np.asarray(tau, dtype=float))


def no_event_price(p: Any, h: Any, dt: Any) -> Any:
    """Deterministic-hazard price after ``dt`` with no event: 1 − (1 − p)·exp(h·dt) (theta path)."""
    return 1 - (1 - np.asarray(p, dtype=float)) * np.exp(np.asarray(h, dtype=float) * np.asarray(dt, dtype=float))


def forward_hazards(ps: list[float] | np.ndarray, taus: list[float] | np.ndarray) -> np.ndarray:
    """Forward hazards between consecutive deadlines (taus ascending, ps the matching prices):
    h_i = −ln[(1 − p_{i+1})/(1 − p_i)]/(tau_{i+1} − tau_i); the first is the spot h̄ to tau_0.
    A negative forward hazard is a monotonicity (static-arbitrage) violation."""
    p = np.clip(np.asarray(ps, dtype=float), EPS, 1 - EPS)
    tau = np.asarray(taus, dtype=float)
    S = np.log1p(-p)
    H = np.concatenate([[0.0], -S])
    T = np.concatenate([[0.0], tau])
    return np.diff(H) / np.diff(T)


def hazard_multiplier(p: np.ndarray, y: np.ndarray, lo: float = 0.02, hi: float = 50.0) -> float:
    """k such that Σ[1 − (1 − p)^k] = Σ y: realized cumulative hazard / implied (k < 1: implied
    hazard too high). Bisection on log k; clipped to [lo, hi]."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    target = float(np.sum(y))
    if target <= 0:
        return lo
    if target >= len(p):
        return hi
    a, b = math.log(lo), math.log(hi)
    for _ in range(80):
        m = (a + b) / 2
        if float(np.sum(1 - (1 - p) ** math.exp(m))) < target:
            a = m
        else:
            b = m
    return math.exp((a + b) / 2)


def cir_survival(lam: float, tau: float, kappa: float, theta: float, sigma: float) -> float:
    """E[exp(−∫_0^tau λ_u du)] for a CIR intensity dλ = κ(θ − λ)dt + σ√λ dW: A·exp(−B·λ0)."""
    if sigma <= 1e-12:   # deterministic mean-reverting path
        if kappa <= 1e-12:
            return math.exp(-lam * tau)
        integral = theta * tau + (lam - theta) * (1 - math.exp(-kappa * tau)) / kappa
        return math.exp(-integral)
    g = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    e = math.exp(g * tau) - 1
    den = (g + kappa) * e + 2 * g
    B = 2 * e / den
    A = (2 * g * math.exp((kappa + g) * tau / 2) / den) ** (2 * kappa * theta / sigma ** 2)
    return A * math.exp(-B * lam)


def realized_hazard(events: np.ndarray, exposure: np.ndarray) -> float:
    """Poisson MLE of a constant hazard: events / exposure time."""
    ex = float(np.sum(exposure))
    return float(np.sum(events)) / ex if ex > 0 else float("nan")


# ---- deadline parsing and classification -------------------------------------------------------
_MON = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
        "oct": 10, "nov": 11, "dec": 12}
_DATE = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
         r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+20\d\d")
_TIME = r"(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?(?:\s*(?:ET|EST|EDT))?)?"
_DATE_RX = re.compile(rf"({_DATE})({_TIME})", re.IGNORECASE)
_END_RX = re.compile(
    rf"\b(?P<kw>before(?:\s+before)?|prior to|by|through|until)\s+(?P<d>{_DATE})(?P<tm>{_TIME})"
    rf"|\bbetween\s+(?:{_DATE}){_TIME}\s+and\s+(?P<d2>{_DATE})(?P<tm2>{_TIME})", re.IGNORECASE)
_START_RX = re.compile(rf"\b(?:after|on or after|starting(?:\s+\d{{1,2}}:\d{{2}}\s*(?:am|pm))?|from|between)\s+"
                       rf"(?P<d>{_DATE})(?P<tm>{_TIME})", re.IGNORECASE)
DEADLINE_HINT = re.compile(rf"\b(before|by|through|until|prior to|between)\s+(the end of\s+)?{_DATE}", re.IGNORECASE)


def _date(text: str, tm: str | None) -> tuple[datetime, bool]:
    """Parse 'Oct 1, 2026' (+ optional ' at 10:00am ET') as US Eastern; returns (dt, has_time)."""
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", text.strip())
    if not m:
        raise ValueError(text)
    mon = _MON[m.group(1)[:3].lower()]
    base = datetime(int(m.group(3)), mon, int(m.group(2)), tzinfo=ET)
    tm = (tm or "").strip()
    if not tm:
        return base, False
    t = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", tm, re.IGNORECASE)
    hh, mm = int(t.group(1)), int(t.group(2) or 0)
    ap = (t.group(3) or "").lower().replace(".", "")
    if ap == "pm" and hh < 12:
        hh += 12
    if ap == "am" and hh == 12:
        hh = 0
    return base.replace(hour=hh % 24, minute=mm), True


def parse_window(rules: str | None) -> tuple[datetime | None, datetime | None, str | None]:
    """Contract window from the rules text: (start or None, deadline T, keyword). ``before D`` and
    ``prior to D`` end at D 00:00 ET; ``by``/``through``/``until``/``between … and D`` include D
    (end at D+1 00:00 ET); an explicit time overrides. The latest end phrase wins."""
    if not rules:
        return None, None, None
    text = re.sub(r"\s+", " ", rules.split("then the market resolves")[0])
    best: tuple[datetime, str] | None = None
    for m in _END_RX.finditer(text):
        if m.group("d2"):
            d, has = _date(m.group("d2"), m.group("tm2"))
            kw = "between"
        else:
            d, has = _date(m.group("d"), m.group("tm"))
            kw = m.group("kw").lower().split()[0]
            kw = "prior" if kw == "prior" else kw
        if not has and kw in ("by", "through", "until", "between"):
            d = d + timedelta(days=1)
        if best is None or d > best[0]:
            best = (d, kw)
    if best is None:
        return None, None, None
    start = None
    for m in _START_RX.finditer(text):
        d, _ = _date(m.group("d"), m.group("tm"))
        if d < best[0] and (start is None or d > start):
            start = d
    return (start.astimezone(UTC) if start else None), best[0].astimezone(UTC), best[1]


_EXCLUDE = (
    ("competing", re.compile(
        r"next (?:team|club)|\bfirst (?:new )?person\b|first such subject|is the first storm|"
        r"first person to serve|is the first|becomes .{0,80}(?:following|as a result)|headline the|"
        r"is confirmed as the developer|"
        r"following the 20\d\d .{0,40}election", re.IGNORECASE)),
    ("measured_at_T", re.compile(
        r"as checked on|total receipts|total number of tokens|distinct days|precipitation|"
        r"temperature|consecutive|exactly \d|is exactly|between \d+ and \d+ (?:persons|people)|"
        r"debut|scores? above|arena", re.IGNORECASE)),
    ("scheduled_vote", re.compile(
        r"number of [a-z ]{0,40}voting|votes? (?:for|yea)\b|voting yea|\bvote for the\b|"
        r"\d+ (?:democrats|republicans) vote", re.IGNORECASE)),
    ("scheduled_show_or_election", re.compile(
        r"perform(?:s)? .{0,60} at the 20\d\d|qualifies for the finale|wins at least one seat|"
        r"lose their (?:respective )?primary|members? that lose", re.IGNORECASE)),
)
_COMPETING_TITLE = re.compile(r"^(?:who|what|where|which)\b|next (?:team|club)|\bis confirmed as the\b", re.IGNORECASE)
_COUNT = re.compile(r"\b(?:at least|more than|above|over)\s+\$?[\d.,]*\d(?![\d.,])(?!\s*(?:mph|miles|km|kilometers|"
                    r"seconds?|minutes?|hours?|%|percent|degrees|°))|\d+ or more", re.IGNORECASE)


def classify(category: str | None, title: str | None, rules: str | None) -> str:
    """Text-only contract type: 'mention' | 'elimination' | 'count' | 'event' for hazard
    contracts (YES iff an occurrence before T; can close early on YES), else an exclusion code."""
    text = re.sub(r"\s+", " ", (rules or "").split("then the market resolves")[0])
    if _COMPETING_TITLE.search(title or ""):
        return "x_competing"
    for code, rx in _EXCLUDE:
        if rx.search(text):
            return f"x_{code}"
    if category == "Mentions" or re.search(r"\bis stated by\b", text, re.IGNORECASE):
        return "mention"
    if re.search(r"\beliminated from\b", text, re.IGNORECASE):
        return "elimination"
    if _COUNT.search(text):
        return "count"
    return "event"


HAZARD_TYPES = ("event", "mention", "elimination", "count")


def strip_key(series: str, rules: str | None) -> str:
    """Same question with the dates removed (legs of one deadline strip share this key)."""
    text = re.sub(r"\s+", " ", (rules or "").split("then the market resolves")[0]).lower()
    text = _DATE_RX.sub("<d>", text)
    text = re.sub(r"after (?:market )?issuance and ", "", text)
    return f"{series}|{text.strip()}"


# ---- data: scan, universe, candles -------------------------------------------------------------
def _series_info(s) -> dict[str, dict[str, Any]]:
    df = pl.read_parquet(s.tables_dir / "kalshi_series.parquet")
    return {r["series_ticker"]: r for r in df.iter_rows(named=True)}


def _candidate(r: dict[str, Any], info: dict[str, dict[str, Any]]) -> bool:
    from albert.categories import kalshi_series_is_financial

    si = info.get(r["series_ticker"]) or {}
    if kalshi_series_is_financial(r["series_ticker"], si.get("category"), si.get("title")):
        return False
    return bool(DEADLINE_HINT.search(r.get("rules_primary") or ""))


def _api(s, min_interval: float = 0.2, *, nap_cap: float = 30.0, retries: int = 40):
    """Cached public-data client. 429s back off exponentially, each wait capped at ``nap_cap``."""
    import random
    import time

    from albert.http import CachedClient
    from albert.venues import kalshi as k

    def nap(x: float) -> None:
        time.sleep(min(x, nap_cap) * (0.75 + 0.5 * random.random()))
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval,
                                 max_retries=retries, sleep=nap))


def _save(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


SERIES_HINT = re.compile(
    r"\b(out\b|leav|release|launch|visit|meet|announce|retire|resign|fire[ds]?\b|confirm|endorse|before|"
    r"when will|apolog|arrest|indict|pardon|traded?\b|deal\b|agreement|ceasefire|become law|pass(es)? |sign|"
    r"happen|this year|in 2026|by (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|say|mention|eliminat|"
    r"approv|nominat|declar|ban\b|step down|drop out|run for|join|return|reopen|shutdown|impeach|rename|"
    r"recogni|sanction|invade|strike|attack|resume|end of)", re.IGNORECASE)


def scan_series(s) -> pl.DataFrame:
    """Series to fetch whole: tier 1 = series with a hazard-typed market in the shared table (also
    fetched live after the table's end); tier 2 = non-financial one-off/custom/periodic series whose
    title has a deadline-ish word (historical only: catches one-offs that ended before 08-15)."""
    from albert.categories import kalshi_series_is_financial

    u = pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").select(
        "series_ticker", "category", "title", "rules_primary").filter(
        pl.col("rules_primary").str.contains(DEADLINE_HINT.pattern)).collect()
    kinds = [classify(c, t, r) for c, t, r in u.select("category", "title", "rules_primary").iter_rows()]
    tier1 = set(u.with_columns(pl.Series("kind", kinds)).filter(pl.col("kind").is_in(HAZARD_TYPES))["series_ticker"])
    ser = pl.read_parquet(s.tables_dir / "kalshi_series.parquet")
    tier2 = set()
    for r in ser.iter_rows(named=True):
        if r["series_ticker"] in tier1 or kalshi_series_is_financial(r["series_ticker"], r["category"], r["title"]):
            continue
        if r["frequency"] in ("one_off", "custom", "weekly", "monthly", "annual") and SERIES_HINT.search(r["title"] or ""):
            tier2.add(r["series_ticker"])
    return pl.DataFrame({"series_ticker": sorted(tier1) + sorted(tier2),
                         "tier": [1] * len(tier1) + [2] * len(tier2)})


def scan(s) -> None:
    """Every settled market of the scan series (``/historical/markets?series_ticker=`` and, for
    tier 1, ``/markets?series_ticker=&status=settled`` closing after the table's end), kept when
    the rules carry a deadline phrase. Checkpointed in ``scan_series.parquet``."""
    from albert.venues import kalshi as k

    d = out_dir(s)
    plan = scan_series(s)
    _save(plan, d / "scan_plan.parquet")
    path, done_path = d / "scan_series.parquet", d / "scan_series_done.parquet"
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=k.MARKET_SCHEMA)
    done = set(pl.read_parquet(done_path)["series_ticker"]) if done_path.exists() else set()
    api = _api(s)
    info = _series_info(s)
    rows: list[dict[str, Any]] = []
    todo = [r for r in plan.iter_rows(named=True) if r["series_ticker"] not in done]
    print(f"scan: {len(todo)} series ({plan.height} planned, {len(done)} done)", flush=True)
    for i, r in enumerate(todo, 1):
        sr = r["series_ticker"]
        try:
            for page in api._cursor_pages("/historical/markets", {"series_ticker": sr, "limit": 1000}, "markets"):
                rows.extend(x for x in k.market_rows(page.payload, info, "pmh-historical") if _candidate(x, info))
            if r["tier"] == 1:
                for page in api.series_markets(sr, "settled", min_close=TABLE_END - timedelta(hours=1), cache=True):
                    rows.extend(x for x in k.market_rows(page.payload, info, "pmh-live") if _candidate(x, info))
        except RuntimeError as error:
            print(f"skip {sr}: {str(error)[:120]}", flush=True)
            continue
        done.add(sr)
        if i % 100 == 0 or i == len(todo):
            new = pl.DataFrame(rows, schema=k.MARKET_SCHEMA)
            have = pl.concat([have, new]).unique("ticker", keep="last")
            rows = []
            _save(have, path)
            _save(pl.DataFrame({"series_ticker": sorted(done)}), done_path)
            print(f"  series {i}/{len(todo)}: {have.height} candidate markets", flush=True)


def build_universe(s) -> pl.DataFrame:
    """All candidate markets (shared table + scans), with window, type, strip key, y."""
    from albert.categories import kalshi_series_is_financial
    from albert.venues import kalshi as k

    d = out_dir(s)
    cols = list(k.MARKET_SCHEMA)
    tab = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").select(cols)
           .filter(pl.col("rules_primary").str.contains(DEADLINE_HINT.pattern)).collect())
    frames = [tab.with_columns(pl.lit("table").alias("source"))]
    for name in ("scan_hist.parquet", "scan_series.parquet"):
        if (d / name).exists():
            frames.append(pl.read_parquet(d / name).select(cols))
    m = pl.concat(frames, how="diagonal_relaxed").unique("ticker", keep="first", maintain_order=True)
    info = _series_info(s)
    fin = {r: kalshi_series_is_financial(r, (info.get(r) or {}).get("category"), (info.get(r) or {}).get("title"))
           for r in m["series_ticker"].unique().to_list()}
    m = m.filter(pl.col("series_ticker").map_elements(lambda x: not fin.get(x, True), return_dtype=pl.Boolean))
    # market-title money filter (tariffs, Fed, crypto ... inside Politics series)
    m = m.filter(~pl.col("title").map_elements(lambda t: kalshi_series_is_financial(None, "Politics", t),
                                               return_dtype=pl.Boolean))
    recs = []
    for r in m.select("ticker", "category", "title", "rules_primary", "series_ticker").iter_rows(named=True):
        try:
            st, T, kw = parse_window(r["rules_primary"])
        except (ValueError, KeyError):
            st, T, kw = None, None, None
        recs.append({"ticker": r["ticker"], "w_start": st, "T": T, "kw": kw,
                     "kind": classify(r["category"], r["title"], r["rules_primary"]),
                     "skey": strip_key(r["series_ticker"], r["rules_primary"])})
    meta = pl.DataFrame(recs, schema={"ticker": pl.String, "w_start": pl.Datetime("us", "UTC"),
                                      "T": pl.Datetime("us", "UTC"), "kw": pl.String, "kind": pl.String,
                                      "skey": pl.String})
    u = m.join(meta, on="ticker").with_columns(
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.col("fee_multiplier").fill_null(1.0))
    _save(u, d / "universe_all.parquet")
    return u


def in_scope(u: pl.DataFrame) -> pl.DataFrame:
    """Hazard contracts whose both outcomes are observable: parsed T ≤ END, T > START, settled
    yes/no, and open at some time in [START, T)."""
    return u.filter(pl.col("kind").is_in(HAZARD_TYPES) & pl.col("T").is_not_null()
                    & (pl.col("T") <= END) & (pl.col("T") > START + timedelta(hours=1))
                    & pl.col("result").is_in(["yes", "no"])
                    & (pl.col("close_time") > START) & (pl.col("open_time") < pl.col("T")))


def fetch_candles(s, u: pl.DataFrame, *, workers: int = 1, min_interval: float = 0.15) -> None:
    """Hourly candles of each market over [max(open, T − LOOKBACK, START − 1d), min(close, T)].
    One request per market (historical endpoint before the cutoff). Checkpointed."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.venues import kalshi as k

    d = out_dir(s)
    path, done_path = d / "candles.parquet", d / "candles_done.parquet"
    have = pl.read_parquet(path) if path.exists() else None
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = [r for r in u.select("ticker", "open_time", "close_time", "T").iter_rows(named=True) if r["ticker"] not in done]
    # reuse the shared hourly candles where they already cover a market's whole window
    w = pl.DataFrame(todo, schema={"ticker": pl.String, "open_time": pl.Datetime("us", "UTC"),
                                   "close_time": pl.Datetime("us", "UTC"), "T": pl.Datetime("us", "UTC")}).with_columns(
        pl.max_horizontal("open_time", pl.col("T") - LOOKBACK, pl.lit(START - timedelta(days=1))).alias("lo"),
        (pl.min_horizontal("close_time", "T") + timedelta(hours=1)).alias("hi"))
    shared = (pl.scan_parquet(s.tables_dir / "kalshi_candles.parquet")
              .filter(pl.col("ticker").is_in(w["ticker"].implode()))
              .select("ticker", "end_ts", "yes_bid_close", "yes_ask_close", "price_close", "volume", "open_interest")
              .collect().join(w.select("ticker", "lo", "hi"), on="ticker")
              .filter((pl.col("end_ts") >= pl.col("lo")) & (pl.col("end_ts") <= pl.col("hi"))))
    cov = shared.group_by("ticker").agg(pl.col("end_ts").min().alias("a"), pl.col("end_ts").max().alias("b"),
                                         pl.len().alias("n"), pl.col("lo").first(), pl.col("hi").first())
    full = cov.filter((pl.col("a") <= pl.col("lo") + timedelta(hours=2)) & (pl.col("b") >= pl.col("hi") - timedelta(hours=3))
                      & (pl.col("n") >= ((pl.col("b") - pl.col("a")).dt.total_hours() * 0.9)))
    reuse = set(full["ticker"])
    if reuse:
        part = shared.filter(pl.col("ticker").is_in(list(reuse))).drop("lo", "hi")
        have = pl.concat([f for f in (have, part) if f is not None]).unique(["ticker", "end_ts"])
        done |= reuse
        _save(have, path)
        _save(pl.DataFrame({"ticker": sorted(done)}), done_path)
    todo = [r for r in todo if r["ticker"] not in reuse]
    print(f"candles: {len(reuse)} reused from the shared table, {len(todo)} markets to fetch ({len(done)} done)", flush=True)
    if not todo:
        return
    api = _api(s, min_interval, nap_cap=3.0, retries=120)
    cutoff = api.cutoff().get("market_settled_ts")
    schema = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "yes_bid_close": pl.Float64,
              "yes_ask_close": pl.Float64, "price_close": pl.Float64, "volume": pl.Float64,
              "open_interest": pl.Float64}
    keep = list(schema)
    w = w.filter(pl.col("ticker").is_in([r["ticker"] for r in todo]))
    meta = u.select("ticker", "event_ticker", "close_time").join(w, on="ticker")

    def per_market(r: dict[str, Any]) -> list[dict[str, Any]] | None:
        if r["lo"] >= r["hi"]:
            return []
        try:
            page = api.candles(r["ticker"], r["lo"], r["hi"], 60, bool(cutoff and r["close_time"] < cutoff))
        except RuntimeError as error:
            print(f"skip {r['ticker']}: {str(error)[:120]}", flush=True)
            return None
        return [{c: x[c] for c in keep} for x in k.candle_rows(page.payload, r["ticker"], 60)]

    def job(ev_rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]] | None]]:
        """One event: live batch request(s) when every leg starts after BATCH_FROM and the union
        window is short; legs missing from (or truncated in) the batch fall back to per-market."""
        out: dict[str, list[dict[str, Any]]] = {}
        lo, hi = min(r["lo"] for r in ev_rows), max(r["hi"] for r in ev_rows)
        if lo >= BATCH_FROM and (hi - lo) <= timedelta(days=10) and len(ev_rows) > 1:
            ticks = sorted(r["ticker"] for r in ev_rows)
            for j in range(0, len(ticks), 50):
                try:
                    page = api.candles_batch(ticks[j:j + 50], lo, hi, 60)
                except RuntimeError as error:
                    print(f"skip batch {ev_rows[0]['event_ticker']}: {str(error)[:120]}", flush=True)
                    continue
                for mk in page.payload.get("markets") or []:
                    rows = [{c: x[c] for c in keep} for x in k.candle_rows(mk, mk.get("market_ticker"), 60)]
                    if rows:
                        out[mk["market_ticker"]] = rows
        res = []
        for r in ev_rows:
            got = out.get(r["ticker"])
            if got and min(x["end_ts"] for x in got) <= r["lo"] + timedelta(hours=3):
                res.append((r["ticker"], [x for x in got if r["lo"] <= x["end_ts"] <= r["hi"]]))
            else:
                res.append((r["ticker"], per_market(r)))
        return res

    jobs = [g.to_dicts() for _, g in meta.group_by("event_ticker")]
    jobs.sort(key=lambda rows: -min(r["lo"] for r in rows).timestamp())   # newest (batchable) first
    rows: list[dict[str, Any]] = []
    fin: list[str] = []

    def save() -> None:
        nonlocal have, rows
        new = pl.DataFrame(rows, schema=schema) if rows else None
        frames = [f for f in (have, new) if f is not None and f.height]
        have = pl.concat(frames).unique(["ticker", "end_ts"]) if frames else pl.DataFrame(schema=schema)
        rows = []
        _save(have, path)
        _save(pl.DataFrame({"ticker": sorted(done | set(fin))}), done_path)

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(job, jobs):
            for tk, got in res:
                n += 1
                if got is None:
                    continue          # failed: not marked done, retried on the next run
                rows.extend(got)
                fin.append(tk)
                if n % 250 == 0:
                    print(f"  candles {n}/{len(todo)}", flush=True)
                    save()
    save()
    print(f"candles done: {len(fin)} fetched, {len(todo) - len(fin)} failed", flush=True)


# ---- panel ---------------------------------------------------------------------------------------
def fee_vec(price: Any, mult: Any, rate: float = 0.07) -> np.ndarray:
    """Kalshi fee per contract, rate × multiplier × P(1 − P), rounded up to $0.0001."""
    P = np.asarray(price, dtype=float)
    return np.ceil(rate * np.asarray(mult, dtype=float) * P * (1 - P) * 1e4 - 1e-9) / 1e4


def _bucket(expr: pl.Expr, buckets: tuple, name: str) -> pl.Expr:
    out = pl.lit(None, dtype=pl.String)
    for lo, hi, lab in reversed(buckets):
        out = pl.when((expr >= lo) & (expr < hi)).then(pl.lit(lab)).otherwise(out)
    return out.alias(name)


def event_times(u: pl.DataFrame, c: pl.DataFrame, level: float = 0.97) -> pl.DataFrame:
    """Proxy event time of YES markets: the earlier of close_time and the end of the first hourly
    candle of the final run of mids ≥ ``level`` (some YES markets keep trading at 99c until T); an
    upper bound on the event time. Descriptive only: it uses the future and is never an input to a
    trading rule."""
    yes = u.filter(pl.col("y") == 1).select("ticker", "close_time")
    cc = (c.join(yes, on="ticker").filter(pl.col("end_ts") <= pl.col("close_time") + timedelta(hours=1))
          .with_columns(pl.when(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null())
                        .then((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2)
                        .otherwise(pl.col("price_close")).alias("m"))
          .sort("ticker", "end_ts")
          .with_columns(pl.col("m").fill_null(0.0).reverse().cum_min().reverse().over("ticker").alias("fwd_min")))
    # the event happened inside the hour that ENDS at the run's first candle: its end is an upper
    # bound, so "ev_ts <= t" (post-event) never drops an observation whose interval holds the jump
    first = (cc.filter(pl.col("fwd_min") >= level).group_by("ticker")
             .agg(pl.col("end_ts").min().alias("ev_c")))
    out = yes.join(first, on="ticker", how="left").with_columns(
        pl.min_horizontal("close_time", "ev_c").alias("ev_ts"))
    return out.select("ticker", "ev_ts")


def build_panel(u: pl.DataFrame, c: pl.DataFrame, leads: tuple[int, ...] = LEADS_H) -> pl.DataFrame:
    """One row per (market, lead L): decision t = T − L, the last hourly quote that ended strictly
    before t (≤ 2 h old), mid, implied hazard over the remaining window, and the split:
    dev (T < HOLDOUT), holdout (t ≥ HOLDOUT) or straddle (excluded from split tables)."""
    ev = event_times(u, c)
    base = (u.select("ticker", "event_ticker", "series_ticker", "category", "kind", "skey", "T", "w_start",
                     "open_time", "close_time", "y", "fee_type", "fee_multiplier")
            .join(ev, on="ticker", how="left"))
    grid = base.join(pl.DataFrame({"lead_h": list(leads)}), how="cross").with_columns(
        (pl.col("T") - pl.duration(hours=pl.col("lead_h"))).alias("t"))
    grid = grid.filter((pl.col("t") >= START) & (pl.col("t") >= pl.col("open_time") + timedelta(hours=1))
                       & (pl.col("t") < pl.col("close_time")))
    q = (c.filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
                  & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
         .select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("yes_bid_close").alias("bid"),
                 pl.col("yes_ask_close").alias("ask"))
         .sort("ticker", "q_ts"))
    grid = grid.with_columns((pl.col("t") - timedelta(seconds=1)).alias("t_q")).sort("ticker", "t_q")
    pn = grid.join_asof(q, left_on="t_q", right_on="q_ts", by="ticker", strategy="backward", check_sortedness=False)
    pn = pn.filter(pl.col("q_ts").is_not_null() & ((pl.col("t") - pl.col("q_ts")) <= timedelta(hours=2)))
    pn = pn.with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("T") - pl.max_horizontal("t", pl.col("w_start").fill_null(pl.col("t")))).dt.total_seconds() / 86400)
        .alias("tau_d"),
        pl.when(pl.col("T") < HOLDOUT).then(pl.lit("dev"))
        .when(pl.col("t") >= HOLDOUT).then(pl.lit("holdout")).otherwise(pl.lit("straddle")).alias("split"),
        (pl.col("ev_ts").is_not_null() & (pl.col("ev_ts") <= pl.col("t"))).alias("post_event"))
    pn = pn.with_columns(
        pl.Series("hbar", implied_hazard(pn["mid"].to_numpy(), np.maximum(pn["tau_d"].to_numpy(), 1 / 24))),
        _bucket(pl.col("lead_h"), LEAD_BUCKETS, "lead_b"), _bucket(pl.col("mid"), PRICE_BUCKETS, "price_b"),
        pl.when(pl.col("kind") == "elimination").then(pl.lit("elimination")).otherwise(pl.lit("core")).alias("group"))
    return pn.drop("t_q").sort("ticker", "lead_h", descending=[False, True])


def add_next(pn: pl.DataFrame) -> pl.DataFrame:
    """Attach the next grid point of each observation: t2 = T − (next shorter lead), or T for the
    1 h lead ('final'). t2 is fixed by the grid, never by whether the market was still open, so
    intervals in which a YES market resolved and closed are kept ('jump'). mid2/bid2/ask2 are the
    quote at t2 when the market was still open there."""
    desc = sorted(LEADS_H, reverse=True)
    nxt_lead = {L: (desc[i + 1] if i + 1 < len(desc) else 0) for i, L in enumerate(desc)}
    pn = pn.with_columns(pl.col("lead_h").replace_strict(nxt_lead, return_dtype=pl.Int64).alias("lead_grid_next"))
    pn = pn.with_columns((pl.col("T") - pl.duration(hours=pl.col("lead_grid_next"))).alias("t2"),
                         (pl.col("lead_grid_next") == 0).alias("final"))
    nxt = pn.select("ticker", pl.col("lead_h").alias("lead_grid_next"), pl.col("mid").alias("mid2"),
                    pl.col("bid").alias("bid2"), pl.col("ask").alias("ask2"))
    pn = pn.join(nxt, on=["ticker", "lead_grid_next"], how="left")
    return pn.with_columns(((pl.col("y") == 1) & (pl.col("final") | (pl.col("ev_ts").is_not_null()
                                                                     & (pl.col("ev_ts") <= pl.col("t2"))))).alias("jump"))


# ---- statistics helpers -------------------------------------------------------------------------
def _cm(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None, int]:
    from albert.research.fundamental import clustered_mean

    return clustered_mean(np.asarray(x, dtype=float), np.asarray(g))


def _r(x: float | None, d: int = 3) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), d)


def k_ci(mid: np.ndarray, y: np.ndarray, g: np.ndarray, reps: int = 200, seed: int = 0) -> tuple[float, float]:
    """5-95% interval of the hazard multiplier by resampling clusters (events) with replacement."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(g, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]
    if len(groups) < 3:
        return float("nan"), float("nan")
    ks = []
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        ks.append(hazard_multiplier(mid[idx], y[idx]))
    return float(np.quantile(ks, 0.05)), float(np.quantile(ks, 0.95))


def calib_table(pn: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Implied vs realized by cell: mean mid, YES rate, (y − mid) in cents (t by event), and the
    hazard multiplier k with Σ[1 − (1 − mid)^k] = Σy, i.e. realized / implied hazard under the
    flat model (k < 1: implied hazard too high), with a 5-95% cluster-bootstrap interval, and the
    median implied hazard per day."""
    rows = []
    for key, g in pn.group_by(by, maintain_order=False):
        y = g["y"].to_numpy().astype(float)
        mid = g["mid"].to_numpy()
        ev = g["event_ticker"].to_numpy()
        mu, t, k = _cm((y - mid) * 100, ev)
        lo, hi = k_ci(mid, y, ev)
        rows.append({**dict(zip(by, key, strict=True)), "n": g.height, "markets": g["ticker"].n_unique(), "events": k,
                     "mean_mid": _r(mid.mean()), "yes_rate": _r(y.mean()), "y_minus_mid_c": _r(mu, 2),
                     "t": _r(t, 2), "k_mult": _r(hazard_multiplier(mid, y)), "k_lo": _r(lo), "k_hi": _r(hi),
                     "h_implied_med_d": _r(float(g["hbar"].median()), 4)})
    out = pl.DataFrame(rows, infer_schema_length=None)
    order = {lab: i for i, (_, _, lab) in enumerate(LEAD_BUCKETS)} | {lab: i for i, (_, _, lab) in enumerate(PRICE_BUCKETS)}
    sort_cols = [pl.col(b).replace_strict(order, default=99, return_dtype=pl.Int64) if b in ("lead_b", "price_b") else pl.col(b)
                 for b in by]
    return out.sort(sort_cols)


# ---- tests ---------------------------------------------------------------------------------------
def _by_kind_and_core(x: pl.DataFrame) -> pl.DataFrame:
    """Stack each kind with a pooled 'core' (event + mention + count) copy, as column grp."""
    return pl.concat([x.with_columns(pl.col("kind").alias("grp")),
                      x.filter(pl.col("group") == "core").with_columns(pl.lit("core").alias("grp"))])


def test_shape(pn: pl.DataFrame) -> pl.DataFrame:
    """Hazard shape: per grid interval (t, t2], realized events vs the flat-implied expectation
    1 − exp(−h̄·Δ) (pre-event obs). ratio(L) < 1 at long leads and > 1 near T = events concentrate
    near the deadline relative to the flat hazard the price implies."""
    x = pn.filter(~pl.col("post_event") & (pl.col("spread") <= MAX_SPREAD) & (pl.col("mid") < DESCRIPTIVE_MAX_MID))
    x = x.with_columns(((pl.col("t2") - pl.max_horizontal("t", pl.col("w_start").fill_null(pl.col("t"))))
                        .dt.total_seconds().clip(0) / 86400).alias("dt_d"))
    x = x.with_columns(pl.Series("q", prob_from_hazard(x["hbar"].to_numpy(), x["dt_d"].to_numpy())),
                       pl.col("jump").cast(pl.Float64).alias("hit"))
    rows = []
    for (grp, lb), g in _by_kind_and_core(x).group_by("grp", "lead_b"):
        mu, t, k = _cm(((g["hit"] - g["q"]) * 100).to_numpy(), g["event_ticker"].to_numpy())
        rows.append({"group": grp, "lead_b": lb, "n": g.height, "events": k, "realized": int(g["hit"].sum()),
                     "expected_flat": _r(float(g["q"].sum()), 2),
                     "ratio": _r(float(g["hit"].sum()) / float(g["q"].sum()) if g["q"].sum() > 0 else None),
                     "hit_minus_q_c": _r(mu, 3), "t": _r(t, 2)})
    order = {lab: i for i, (_, _, lab) in enumerate(LEAD_BUCKETS)}
    return pl.DataFrame(rows).sort("group", pl.col("lead_b").replace_strict(order, return_dtype=pl.Int64))


def test_theta(pn: pl.DataFrame) -> pl.DataFrame:
    """Martingale and theta per lead bucket over the next grid step (t, t2]: drift E[p2 − p] with
    p2 = 1 after an event, else the mid at t2; no-event decay vs the flat-hazard model
    (1 − p)(1 − e^{h̄Δ}); jump frequency vs 1 − e^{−h̄Δ}; implied-hazard volatility."""
    x = pn.filter(~pl.col("post_event") & (pl.col("spread") <= MAX_SPREAD) & (pl.col("mid") < DESCRIPTIVE_MAX_MID))
    x = x.with_columns(((pl.col("t2") - pl.max_horizontal("t", pl.col("w_start").fill_null(pl.col("t"))))
                        .dt.total_seconds().clip(0) / 86400).alias("dt_d"))
    x = x.with_columns(pl.when(pl.col("final") & ~pl.col("jump")).then(0.0).otherwise(pl.col("mid2")).alias("mid2"))
    x = x.filter(pl.col("jump") | pl.col("mid2").is_not_null())
    x = x.with_columns(pl.when(pl.col("jump")).then(1.0).otherwise(pl.col("mid2")).alias("p2"))
    hb, dt, p = x["hbar"].to_numpy(), x["dt_d"].to_numpy(), x["mid"].to_numpy()
    x = x.with_columns(pl.Series("q", prob_from_hazard(hb, dt)),
                       pl.Series("decay_model", (1 - p) * (1 - np.exp(hb * dt))))
    rows = []
    for (grp, lb), g in _by_kind_and_core(x).group_by("grp", "lead_b"):
        ev = g["event_ticker"].to_numpy()
        drift, t_drift, k = _cm(((g["p2"] - g["mid"]) * 100).to_numpy(), ev)
        nj = g.filter(~pl.col("jump"))
        dec, _t_dec, _ = _cm(((nj["mid2"] - nj["mid"]) * 100).to_numpy(), nj["event_ticker"].to_numpy())
        dmod = float(nj["decay_model"].mean()) * 100 if nj.height else float("nan")
        gap, t_gap, _ = _cm(((nj["mid2"] - nj["mid"]) - nj["decay_model"]).to_numpy() * 100, nj["event_ticker"].to_numpy())
        # implied-hazard volatility (Cox diffusion), no-event steps, both mids interior
        ok = nj.filter((pl.col("mid") > 0.02) & (pl.col("mid") < 0.98) & (pl.col("mid2") > 0.02) & (pl.col("mid2") < 0.98)
                       & (pl.col("dt_d") > 0))
        if ok.height > 5:
            h2 = implied_hazard(ok["mid2"].to_numpy(), np.maximum(ok["tau_d"].to_numpy() - ok["dt_d"].to_numpy(), 1 / 24))
            vol = float(np.std(np.log(h2 / ok["hbar"].to_numpy()) / np.sqrt(ok["dt_d"].to_numpy())))
        else:
            vol = float("nan")
        rows.append({"group": grp, "lead_b": lb, "n": g.height, "events": k,
                     "drift_c": _r(drift, 3), "t_drift": _r(t_drift, 2),
                     "jumps": int(g["jump"].sum()), "jumps_model": _r(float(g["q"].sum()), 2),
                     "decay_c": _r(dec, 3), "decay_model_c": _r(dmod, 3),
                     "decay_ratio": _r(dec / dmod if dmod and math.isfinite(dmod) and abs(dmod) > 1e-9 else None, 2),
                     "decay_minus_model_c": _r(gap, 3), "t_gap": _r(t_gap, 2),
                     "loghazard_vol_per_sqrt_d": _r(vol, 3)})
    order = {lab: i for i, (_, _, lab) in enumerate(LEAD_BUCKETS)}
    return pl.DataFrame(rows).sort("group", pl.col("lead_b").replace_strict(order, return_dtype=pl.Int64))


def no_returns(pn: pl.DataFrame) -> pl.DataFrame:
    """Hold NO from lead L: bought at the NO ask (1 − yes bid) with the taker fee. Settlement PnL,
    its market-implied expectation at the mid (bid − mid − fee), the excess (= mid − y), and a fixed
    exit at the next grid point (sold at the NO bid 1 − ask2 with the fee; 0 after a YES event)."""
    x = pn.filter(pl.col("spread") <= MAX_SPREAD)
    mult = x["fee_multiplier"].to_numpy()
    nb = 1 - x["bid"].to_numpy()
    f_in = fee_vec(nb, mult)
    y = x["y"].to_numpy().astype(float)
    pnl = (1 - y) - nb - f_in
    jump = x["jump"].to_numpy()
    final = x["final"].to_numpy()
    has2 = x["ask2"].is_not_null().to_numpy()
    sell = np.where(has2, 1 - np.nan_to_num(x["ask2"].to_numpy(), nan=1.0), 0.0)
    f_out = np.where(has2 & ~jump & ~final, fee_vec(sell, mult), 0.0)
    exit_val = np.where(jump, 0.0, np.where(final, 1.0, np.where(has2, sell - f_out, np.nan)))
    x = x.with_columns(pl.Series("pnl", pnl), pl.Series("expected", x["bid"].to_numpy() - x["mid"].to_numpy() - f_in),
                       pl.Series("pnl_exit", exit_val - nb - f_in))
    rows = []
    for (grp, split, lb), g in x.group_by("group", "split", "lead_b"):
        ev = g["event_ticker"].to_numpy()
        m1, t1, k = _cm(g["pnl"].to_numpy() * 100, ev)
        m2, t2, _ = _cm((g["mid"] - g["y"]).to_numpy() * 100, ev)
        ge = g.filter(pl.col("pnl_exit").is_not_nan())
        m3, t3, _ = _cm(ge["pnl_exit"].to_numpy() * 100, ge["event_ticker"].to_numpy()) if ge.height else (float("nan"), None, 0)
        rows.append({"group": grp, "split": split, "lead_b": lb, "n": g.height, "events": k,
                     "mean_no_ask": _r(float((1 - g["bid"]).mean())), "pnl_settle_c": _r(m1, 2), "t": _r(t1, 2),
                     "expected_c": _r(float(g["expected"].mean()) * 100, 2), "excess_c": _r(m2, 2), "t_excess": _r(t2, 2),
                     "pnl_exit_next_c": _r(m3, 2), "t_exit": _r(t3, 2)})
    order = {lab: i for i, (_, _, lab) in enumerate(LEAD_BUCKETS)}
    return pl.DataFrame(rows).sort("group", "split", pl.col("lead_b").replace_strict(order, return_dtype=pl.Int64))


def _quotes(c: pl.DataFrame) -> pl.DataFrame:
    return (c.filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null()
                     & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
            .select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("yes_bid_close").alias("bid"),
                    pl.col("yes_ask_close").alias("ask")).sort("ticker", "q_ts"))


def strip_pairs(u: pl.DataFrame) -> pl.DataFrame:
    """Adjacent legs (T1 < T2, the nearest later deadline) of one question (same strip key) whose
    windows are nested, single-occurrence kinds only (event, mention); counts and eliminations are
    not nested."""
    legs = u.filter(pl.col("kind").is_in(["event", "mention"])).select(
        "ticker", "event_ticker", "skey", "T", "w_start", "open_time", "close_time", "y", "fee_multiplier")
    multi = legs.group_by("skey").agg(pl.col("T").n_unique().alias("nT")).filter(pl.col("nT") >= 2)
    legs = legs.join(multi.select("skey"), on="skey").sort("skey", "T")
    # nested windows: the far leg must not start later than the near one (windows without an explicit
    # start run from issuance, so the far leg must also have been issued no later)
    pairs = legs.join(legs, on="skey", suffix="_2").filter(
        (pl.col("T_2") > pl.col("T"))
        & pl.when(pl.col("w_start").is_null() & pl.col("w_start_2").is_null())
        .then(pl.col("open_time_2") <= pl.col("open_time"))
        .otherwise(pl.col("w_start_2").fill_null(pl.col("open_time_2"))
                   <= pl.col("w_start").fill_null(pl.col("open_time"))))
    nearest = pairs.group_by("ticker").agg(pl.col("T_2").min().alias("Tn"))
    return pairs.join(nearest, on="ticker").filter(pl.col("T_2") == pl.col("Tn")).drop("Tn")


def test_term(u: pl.DataFrame, c: pl.DataFrame, pn: pl.DataFrame) -> dict[str, Any]:
    """Term structure along strips: (a) hourly monotonicity violations p(T1) > p(T2) by mid and
    executable net of taker fees (buy YES far at ask2, NO near at 1 − bid1: payoff ≥ 1 in every
    state); (b) calendar spreads at the near leg's decision grid: is the near leg overpriced relative
    to the far one, and does either calendar trade pay after fees (held to settlement)?"""
    pairs = strip_pairs(u)
    if not pairs.height:
        return {"pairs": 0}
    q = _quotes(c)
    a = pairs.join(q, on="ticker").join(q.rename({"ticker": "ticker_2", "bid": "bid_2", "ask": "ask_2"}),
                                        on=["ticker_2", "q_ts"])
    # both alive, before the near deadline, windows nested at that time
    a = a.filter((pl.col("q_ts") < pl.col("close_time")) & (pl.col("q_ts") < pl.col("close_time_2"))
                 & (pl.col("q_ts") < pl.col("T")) & (pl.col("q_ts") >= START)
                 & (pl.col("w_start_2").is_null()
                    | (pl.col("w_start_2") <= pl.max_horizontal("q_ts", pl.col("w_start").fill_null(pl.col("q_ts"))))))
    fee_far = fee_vec(a["ask_2"].to_numpy(), a["fee_multiplier_2"].to_numpy())
    fee_near = fee_vec(1 - a["bid"].to_numpy(), a["fee_multiplier"].to_numpy())
    prof = a["bid"].to_numpy() - a["ask_2"].to_numpy() - fee_far - fee_near
    a = a.with_columns(((pl.col("bid") + pl.col("ask")) / 2 > (pl.col("bid_2") + pl.col("ask_2")) / 2).alias("viol_mid"),
                       pl.Series("arb_profit", prof))
    ex = a.filter(pl.col("arb_profit") > 0)
    mono = {"pairs": pairs.height, "strips": pairs["skey"].n_unique(), "pair_hours": a.height,
            "pairs_with_quotes": a["ticker"].n_unique(),
            "viol_mid_share": _r(float(a["viol_mid"].mean()) if a.height else None, 4),
            "viol_mid_hours": int(a["viol_mid"].sum()),
            "exec_arb_hours": ex.height, "exec_arb_pairs": ex["ticker"].n_unique(),
            "exec_arb_strips": ex["skey"].n_unique(),
            "exec_arb_mean_profit_c": _r(float(ex["arb_profit"].mean()) * 100 if ex.height else None, 2),
            "exec_arb_median_profit_c": _r(float(ex["arb_profit"].median()) * 100 if ex.height else None, 2),
            "examples": ex.sort("arb_profit", descending=True).unique("ticker", keep="first").head(8)
            .select("ticker", "ticker_2", "q_ts", "bid", "ask_2", (pl.col("arb_profit") * 100).round(2).alias("profit_c"))
            .with_columns(pl.col("q_ts").cast(pl.String)).to_dicts()}
    # forward hazards at the hourly common quotes (mids)
    mid1 = ((a["bid"] + a["ask"]) / 2).to_numpy()
    mid2 = ((a["bid_2"] + a["ask_2"]) / 2).to_numpy()
    tau1 = ((a["T"] - a["q_ts"]).dt.total_seconds() / 86400).to_numpy()
    tau2 = ((a["T_2"] - a["q_ts"]).dt.total_seconds() / 86400).to_numpy()
    fh = -(np.log1p(-np.clip(mid2, EPS, 1 - EPS)) - np.log1p(-np.clip(mid1, EPS, 1 - EPS))) / np.maximum(tau2 - tau1, 1e-6)
    sh = implied_hazard(mid1, np.maximum(tau1, 1 / 24))
    mono["median_spot_hazard_d"] = _r(float(np.median(sh)), 4) if len(sh) else None
    mono["median_forward_hazard_d"] = _r(float(np.median(fh)), 4) if len(fh) else None
    mono["share_forward_lt_half_spot"] = _r(float(np.mean(fh < 0.5 * sh)), 3) if len(fh) else None
    # (b) calendar at the near leg's grid (quotes strictly before t for both legs)
    near = pn.join(pairs.select("ticker", "ticker_2", "T_2", "w_start_2", "close_time_2", "y_2", "fee_multiplier_2"),
                   on="ticker")
    near = near.filter(pl.col("t") < pl.col("close_time_2")).with_columns((pl.col("t") - timedelta(seconds=1)).alias("t_q"))
    q2 = q.rename({"ticker": "ticker_2", "q_ts": "q2_ts", "bid": "bidf", "ask": "askf"})
    cal = near.sort("ticker_2", "t_q").join_asof(q2, left_on="t_q", right_on="q2_ts", by="ticker_2", strategy="backward",
                                                 check_sortedness=False)
    cal = cal.filter(pl.col("q2_ts").is_not_null() & ((pl.col("t") - pl.col("q2_ts")) <= timedelta(hours=2))
                     & (pl.col("spread") <= MAX_SPREAD) & ((pl.col("askf") - pl.col("bidf")) <= MAX_SPREAD)
                     & ~pl.col("post_event")
                     & (pl.col("w_start_2").is_null()
                        | (pl.col("w_start_2") <= pl.max_horizontal("t", pl.col("w_start").fill_null(pl.col("t"))))))
    rows = []
    if cal.height:
        m1 = cal["mid"].to_numpy()
        bf, af = cal["bidf"].to_numpy(), cal["askf"].to_numpy()
        m2 = (bf + af) / 2
        y1, y2 = cal["y"].to_numpy().astype(float), cal["y_2"].to_numpy().astype(float)
        mu1, mu2 = cal["fee_multiplier"].to_numpy(), cal["fee_multiplier_2"].to_numpy()
        b1, a1 = cal["bid"].to_numpy(), cal["ask"].to_numpy()
        long_fwd = (1 - y1) - (1 - b1) - fee_vec(1 - b1, mu1) + y2 - af - fee_vec(af, mu2)
        short_fwd = y1 - a1 - fee_vec(a1, mu1) + (1 - y2) - (1 - bf) - fee_vec(1 - bf, mu2)
        qf = np.clip((m2 - m1) / np.maximum(1 - m1, 1e-6), 0, 1)
        cal = cal.with_columns(pl.Series("diff_c", ((m1 - y1) - (m2 - y2)) * 100), pl.Series("long_fwd", long_fwd * 100),
                               pl.Series("short_fwd", short_fwd * 100), pl.Series("q_fwd", qf),
                               pl.Series("fwd_hit", ((y2 == 1) & (y1 == 0)).astype(float)))
        for split in ("dev", "holdout", "straddle"):
            g = cal.filter(pl.col("split") == split)
            if not g.height:
                continue
            cl = g["skey"].to_numpy()
            d, td, k = _cm(g["diff_c"].to_numpy(), cl)
            lf, tl, _ = _cm(g["long_fwd"].to_numpy(), cl)
            sf, ts, _ = _cm(g["short_fwd"].to_numpy(), cl)
            g0 = g.filter(pl.col("y") == 0)
            rows.append({"split": split, "n": g.height, "pairs": g["ticker"].n_unique(), "strips": k,
                         "near_mid": _r(float(g["mid"].mean())), "near_yes": _r(float(g["y"].mean())),
                         "far_mid": _r(float(np.mean(m2[(cal["split"] == split).to_numpy()]))),
                         "far_yes": _r(float(g["y_2"].mean())),
                         "near_minus_far_mispricing_c": _r(d, 2), "t": _r(td, 2),
                         "fwd_implied": _r(float(g0["q_fwd"].mean()) if g0.height else None),
                         "fwd_realized": _r(float(g0["fwd_hit"].mean()) if g0.height else None),
                         "long_fwd_pnl_c": _r(lf, 2), "t_long": _r(tl, 2), "short_fwd_pnl_c": _r(sf, 2), "t_short": _r(ts, 2)})
    return {"monotonicity": mono, "calendar": rows}


# ---- descriptive modelling ---------------------------------------------------------------------
def _p0_bucket(col: str = "p0") -> pl.Expr:
    return (pl.when(pl.col(col) < 0.15).then(pl.lit("longshot<15"))
            .when(pl.col(col) < 0.5).then(pl.lit("15-50")).otherwise(pl.lit("50-90")).alias("p0_b"))


def term_shapes(u: pl.DataFrame, c: pl.DataFrame) -> dict[str, Any]:
    """Implied hazard term structures across strips: one snapshot per strip per day (the 16:00 UTC
    candle), legs alive with windows started and nested (a later deadline never starts later), mids
    of quotes with spread ≤ MAX_SPREAD. Forward hazards are bootstrapped along the strip; the shape
    is the last forward segment over the spot hazard to the first deadline: rising (> 1.25), flat,
    front-loaded (< 0.8), or inverted (a negative forward = monotonicity violation by mid)."""
    legs = u.filter(pl.col("kind").is_in(["event", "mention"])).select(
        "ticker", "skey", "category", "kind", "T", "w_start", "open_time", "close_time")
    multi = legs.group_by("skey").agg(pl.col("T").n_unique().alias("nT")).filter(pl.col("nT") >= 2)
    legs = legs.join(multi.select("skey"), on="skey").with_columns(
        pl.max_horizontal("open_time", pl.col("w_start").fill_null(pl.col("open_time"))).alias("start"))
    q = _quotes(c).filter((pl.col("q_ts").dt.hour() == 16) & ((pl.col("ask") - pl.col("bid")) <= MAX_SPREAD))
    snap = (legs.join(q, on="ticker")
            .filter((pl.col("q_ts") < pl.col("close_time")) & (pl.col("q_ts") < pl.col("T"))
                    & (pl.col("start") <= pl.col("q_ts")) & (pl.col("q_ts") >= START))
            .with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                          ((pl.col("T") - pl.col("q_ts")).dt.total_seconds() / 86400).alias("tau")))
    rows = []
    for (sk, ts), g in snap.group_by("skey", "q_ts"):
        g = g.sort("T").unique("T", keep="first", maintain_order=True)
        if g.height < 2:
            continue
        st = g["start"].to_list()
        if any(st[j] > st[i] for i in range(len(st)) for j in range(i + 1, len(st))):
            continue                       # windows not nested
        fh = forward_hazards(g["mid"].to_numpy(), g["tau"].to_numpy())
        h0, hl = fh[0], fh[-1]
        viol = bool(np.any(fh[1:] < 0))
        r = hl / h0 if h0 > 0 else float("nan")
        shape = ("inverted" if viol else "rising" if r > 1.25 else "front-loaded" if r < 0.8 else "flat")
        rows.append({"skey": sk, "q_ts": ts, "category": g["category"][0], "kind": g["kind"][0], "legs": g.height,
                     "tau1_d": float(g["tau"][0]), "tauN_d": float(g["tau"][-1]), "h_spot": float(h0),
                     "h_last_fwd": float(hl), "ratio": r, "shape": shape,
                     "fwd_over_spot_seg2": float(fh[1] / h0) if h0 > 0 else float("nan")})
    sn = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    if not sn.height:
        return {"snapshots": 0}
    out = []
    for key, g in sn.group_by("category"):
        cnt = g["shape"].value_counts()
        share = {r["shape"]: r["count"] / g.height for r in cnt.iter_rows(named=True)}
        out.append({"category": key[0], "snapshots": g.height, "strips": g["skey"].n_unique(),
                    "median_legs": _r(float(g["legs"].median()), 1), "median_tau1_d": _r(float(g["tau1_d"].median()), 2),
                    "median_tauN_d": _r(float(g["tauN_d"].median()), 2),
                    "median_h_spot_d": _r(float(g["h_spot"].median()), 4),
                    "median_fwd_over_spot": _r(float(g.filter(pl.col("ratio").is_finite())["ratio"].median()), 3),
                    **{f"share_{k}": _r(share.get(k, 0.0), 3) for k in ("rising", "flat", "front-loaded", "inverted")}})
    tot = sn["shape"].value_counts()
    return {"snapshots": sn.height, "strips": sn["skey"].n_unique(),
            "overall_share": {r["shape"]: _r(r["count"] / sn.height, 3) for r in tot.iter_rows(named=True)},
            "overall_median_fwd_over_spot": _r(float(sn.filter(pl.col("ratio").is_finite())["ratio"].median()), 3),
            "by_category": sorted(out, key=lambda r: -r["snapshots"])}


def decay_paths(pn: pl.DataFrame, anchors: tuple[int, ...] = (168, 72, 24), by: str = "kind") -> pl.DataFrame:
    """Realized no-event decay against the flat-hazard path. From an anchor lead A (priced, mid <
    DESCRIPTIVE_MAX_MID, pre-event) to each later grid lead L at which no event has happened yet:
    realized p(L)/p(A) vs model [1 − (1 − p_A)^{τ_L/τ_A}]/p_A, and the implied hazard h̄(L)/h̄(A)
    (1 = flat; > 1 = the market's implied hazard rises toward the deadline). stickiness = mean
    realized / mean model (> 1 = slower decay than theta). Mids cannot fall below ~0.5-1c (1c tick),
    which caps longshot decay near T. Conditioning on no event to L uses the future and is
    descriptive only."""
    base = pn.filter(pl.col("spread") <= MAX_SPREAD)
    rows = []
    for A in anchors:
        a = base.filter((pl.col("lead_h") == A) & (pl.col("mid") < DESCRIPTIVE_MAX_MID) & ~pl.col("post_event")
                        & (pl.col("mid") > 0.005)).select(
            "ticker", "event_ticker", "kind", "group", "category", pl.col("mid").alias("p0"),
            pl.col("tau_d").alias("tau0"), pl.col("hbar").alias("h0"))
        later = base.filter(pl.col("lead_h") < A).select("ticker", "lead_h", "t", "mid", "tau_d", "hbar", "ev_ts")
        j = a.join(later, on="ticker").filter(pl.col("ev_ts").is_null() | (pl.col("ev_ts") > pl.col("t")))
        if not j.height:
            continue
        p0, tau0, tl = j["p0"].to_numpy(), j["tau0"].to_numpy(), j["tau_d"].to_numpy()
        model = (1 - (1 - p0) ** np.clip(tl / tau0, 0, 1)) / p0
        j = j.with_columns(pl.Series("model_ratio", model), (pl.col("mid") / pl.col("p0")).alias("real_ratio"),
                           (pl.col("hbar") / pl.col("h0")).alias("h_ratio"), _p0_bucket())
        if by == "kind":
            j = pl.concat([j.with_columns(pl.col("kind").alias("grp")),
                           j.filter(pl.col("group") == "core").with_columns(pl.lit("core").alias("grp"))])
        else:
            j = j.filter(pl.col("group") == "core").with_columns(pl.col(by).alias("grp"), pl.lit("all").alias("p0_b"))
        for (grp, pb, L), g in j.group_by("grp", "p0_b", "lead_h"):
            mr, mm = float(g["real_ratio"].mean()), float(g["model_ratio"].mean())
            rows.append({"anchor_h": A, "group": grp, "p0_b": pb, "lead_h": L, "n": g.height,
                         "markets": g["ticker"].n_unique(), "mean_p0": _r(float(g["p0"].mean())),
                         "mean_pL_c": _r(float(g["mid"].mean()) * 100, 2),
                         "model_pL_c": _r(float((g["model_ratio"] * g["p0"]).mean()) * 100, 2),
                         "realized_ratio": _r(mr), "model_ratio": _r(mm),
                         "stickiness": _r(mr / mm if mm > 0 else None, 2),
                         "median_h_ratio": _r(float(g["h_ratio"].median()), 2)})
    return pl.DataFrame(rows, infer_schema_length=None).sort("anchor_h", "group", "p0_b", "lead_h",
                                                             descending=[True, False, False, True])


def event_timing(pn: pl.DataFrame) -> dict[str, Any]:
    """Where YES resolutions fall inside the remaining window, against the flat implied hazard.
    Anchor: each YES market's earliest priced pre-event grid quote (mid < DESCRIPTIVE_MAX_MID).
    u = elapsed fraction of the window (anchor → T) at the proxy event time; under a flat hazard
    implied by the anchor price the conditional CDF is F(u) = [1 − (1 − p0)^u]/p0, so the PIT F(u)
    is uniform if the implied hazard shape is right; mean PIT > 0.5 = events later (bunched toward
    the deadline) than implied."""
    a = (pn.filter((pl.col("y") == 1) & (pl.col("spread") <= MAX_SPREAD) & (pl.col("mid") < DESCRIPTIVE_MAX_MID)
                   & ~pl.col("post_event") & pl.col("ev_ts").is_not_null())
         .sort("ticker", "lead_h", descending=[False, True]).unique("ticker", keep="first"))
    a = a.with_columns(pl.max_horizontal("t", pl.col("w_start").fill_null(pl.col("t"))).alias("s0"))
    a = a.with_columns(((pl.col("ev_ts") - pl.col("s0")).dt.total_seconds()
                        / (pl.col("T") - pl.col("s0")).dt.total_seconds()).clip(0, 1).alias("u"))
    p0 = np.clip(a["mid"].to_numpy(), EPS, 1 - EPS)
    uu = a["u"].to_numpy()
    a = a.with_columns(pl.Series("pit", (1 - (1 - p0) ** uu) / p0),
                       pl.Series("pred_last10", 1 - (1 - (1 - p0) ** 0.9) / p0), _p0_bucket("mid"))

    def summ(g: pl.DataFrame) -> dict[str, Any]:
        pit = np.sort(g["pit"].to_numpy())
        n = len(pit)
        ks = float(np.max(np.maximum(np.arange(1, n + 1) / n - pit, pit - np.arange(n) / n))) if n else float("nan")
        mu, t, k = _cm(g["pit"].to_numpy() - 0.5, g["event_ticker"].to_numpy())
        hist = np.histogram(pit, bins=5, range=(0, 1))[0] / max(n, 1)
        return {"n": n, "events": k, "mean_p0": _r(float(g["mid"].mean())), "median_anchor_lead_h": _r(float(g["lead_h"].median()), 0),
                "mean_u": _r(float(g["u"].mean())), "median_u": _r(float(g["u"].median())),
                "mean_pit": _r(mu + 0.5), "t_pit": _r(t, 2), "ks": _r(ks),
                "share_last10_realized": _r(float((g["u"] > 0.9).mean())),
                "share_last10_flat_pred": _r(float(g["pred_last10"].mean())),
                "share_last1d_realized": _r(float(((g["T"] - g["ev_ts"]).dt.total_seconds() < 86400).mean())),
                "pit_quintiles": [_r(float(x)) for x in hist]}

    out = {"all_core": summ(a.filter(pl.col("group") == "core")),
           "by_kind": {k: summ(g) for (k,), g in a.group_by("kind")},
           "by_category": {k: summ(g) for (k,), g in a.filter(pl.col("group") == "core").group_by("category") if g.height >= 10},
           "by_p0": {k: summ(g) for (k,), g in a.filter(pl.col("group") == "core").group_by("p0_b")}}
    return out


def jump_diffusion(u: pl.DataFrame, c: pl.DataFrame, *, big: float = 0.2) -> pl.DataFrame:
    """Quadratic variation of the hourly mid (quotes with spread ≤ MAX_SPREAD) from the first quote
    in the last LOOKBACK before T to settlement, split into: diffusion (hourly changes < ``big``
    before resolution), news jumps (hourly changes ≥ ``big`` before resolution), the YES
    resolution jump (1 − mid)² from the last mid ≥ 2 h before the proxy event (the proxy is an
    upper bound and a jump can straddle two candles), and the NO deadline step (last mid before T
    → 0)². For a martingale E[QV] = p0(1 − p0); Σ QV / Σ p0(1 − p0) checks that (bid-ask bounce
    inflates it)."""
    info = u.select("ticker", "event_ticker", "kind", "category", "T", "y", "close_time")
    ev = event_times(u, c)
    q = (_quotes(c).filter((pl.col("ask") - pl.col("bid")) <= MAX_SPREAD).join(info, on="ticker").join(ev, on="ticker", how="left")
         .filter((pl.col("q_ts") < pl.col("T")) & (pl.col("q_ts") <= pl.col("close_time"))
                 & (pl.col("ev_ts").is_null() | (pl.col("q_ts") <= pl.col("ev_ts") - timedelta(hours=2))))
         .with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("m")).sort("ticker", "q_ts"))
    q = q.with_columns(pl.col("m").diff().over("ticker").alias("dm"))
    per = (q.group_by("ticker").agg(
        pl.col("m").first().alias("p0"), pl.col("m").last().alias("m_last"),
        (pl.when(pl.col("dm").abs() < big).then(pl.col("dm") ** 2).otherwise(0.0)).sum().alias("D"),
        (pl.when(pl.col("dm").abs() >= big).then(pl.col("dm") ** 2).otherwise(0.0)).sum().alias("N"),
        (pl.col("dm").abs() >= big).sum().alias("n_news_jumps"), pl.len().alias("hours"),
        pl.col("kind").first(), pl.col("category").first(), pl.col("y").first(), pl.col("event_ticker").first()))
    per = per.filter(pl.col("hours") >= 6).with_columns(
        pl.when(pl.col("y") == 1).then((1 - pl.col("m_last")) ** 2).otherwise(0.0).alias("J"),
        pl.when(pl.col("y") == 0).then(pl.col("m_last") ** 2).otherwise(0.0).alias("Z"),
        (pl.col("p0") * (1 - pl.col("p0"))).alias("V0"))
    per = per.with_columns((pl.col("D") + pl.col("N") + pl.col("J") + pl.col("Z")).alias("QV"))
    per = pl.concat([per.with_columns(pl.col("kind").alias("grp")),
                     per.filter(pl.col("kind") != "elimination").with_columns(pl.lit("core").alias("grp"))])
    rows = []
    for (grp,), g in per.group_by("grp"):
        tq = float(g["QV"].sum())
        yes, no = g.filter(pl.col("y") == 1), g.filter(pl.col("y") == 0)
        rows.append({"group": grp, "markets": g.height, "yes_rate": _r(float(g["y"].mean())),
                     "median_hours": _r(float(g["hours"].median()), 0),
                     "share_diffusion": _r(float(g["D"].sum()) / tq), "share_news_jumps": _r(float(g["N"].sum()) / tq),
                     "share_yes_resolution": _r(float(g["J"].sum()) / tq),
                     "share_no_deadline_step": _r(float(g["Z"].sum()) / tq),
                     "news_jumps_per_market": _r(float(g["n_news_jumps"].mean()), 2),
                     "qv_over_p0var": _r(tq / float(g["V0"].sum()), 2),
                     "yes_mean_jump": _r(float((1 - yes["m_last"]).mean()) if yes.height else None),
                     "yes_pre_resolution_qv_share": _r(float((yes["D"] + yes["N"]).sum() / yes["QV"].sum()) if yes.height else None),
                     "no_mean_last_mid": _r(float(no["m_last"].mean()) if no.height else None)})
    return pl.DataFrame(rows).sort("group")


def cir_fit(pn: pl.DataFrame) -> pl.DataFrame:
    """CIR-type intensity on the implied average hazard λ = h̄ (per day) over no-event grid steps:
    Δλ/√λ = κθ·Δt/√λ − κ·√λ·Δt + σ√Δt·ε by OLS (no intercept); σ² = mean(resid²/Δt). Also the
    plain implied-hazard vol std(Δln h̄/√Δt). h̄ is an average to T, so a deadline-shaped hazard
    shows up as drift; treat κ, θ as descriptive."""
    x = pn.filter(~pl.col("post_event") & (pl.col("spread") <= MAX_SPREAD) & pl.col("mid2").is_not_null()
                  & ~pl.col("jump") & ~pl.col("final") & pl.col("mid").is_between(0.02, 0.98)
                  & pl.col("mid2").is_between(0.02, 0.98))
    x = x.with_columns(((pl.col("t2") - pl.max_horizontal("t", pl.col("w_start").fill_null(pl.col("t"))))
                        .dt.total_seconds().clip(0) / 86400).alias("dt_d")).filter(pl.col("dt_d") > 0.02)
    h2 = implied_hazard(x["mid2"].to_numpy(), np.maximum(x["tau_d"].to_numpy() - x["dt_d"].to_numpy(), 1 / 24))
    x = x.with_columns(pl.Series("h2", h2))
    x = pl.concat([x.with_columns(pl.col("kind").alias("grp")),
                   x.filter(pl.col("group") == "core").with_columns(pl.lit("core").alias("grp"))])
    rows = []
    for (grp, far), g in x.with_columns((pl.col("lead_h") >= 72).alias("far")).group_by("grp", "far"):
        if g.height < 30:
            continue
        lam, lam2, dt = g["hbar"].to_numpy(), g["h2"].to_numpy(), g["dt_d"].to_numpy()
        sl = np.sqrt(lam)
        yv = (lam2 - lam) / sl
        X = np.column_stack([dt / sl, -sl * dt])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        kappa = float(beta[1])
        theta = float(beta[0] / kappa) if abs(kappa) > 1e-9 else float("nan")
        sigma = float(np.sqrt(np.mean(resid ** 2 / dt)))
        logvol = float(np.std(np.log(lam2 / lam) / np.sqrt(dt)))
        rows.append({"group": grp, "leads": ">=3d" if far else "<3d", "steps": g.height, "markets": g["ticker"].n_unique(),
                     "median_h_d": _r(float(np.median(lam)), 4), "kappa_d": _r(kappa, 3), "theta_d": _r(theta, 4),
                     "sigma": _r(sigma, 3), "half_life_d": _r(math.log(2) / kappa if kappa > 0 else None, 2),
                     "feller_2kt_s2": _r(2 * kappa * theta / sigma ** 2 if sigma > 0 else None, 2),
                     "loghazard_vol_sqrt_d": _r(logvol, 3)})
    return pl.DataFrame(rows).sort("group", "leads")


def arb_by_category(u: pl.DataFrame, c: pl.DataFrame) -> pl.DataFrame:
    """Static-arbitrage frequency per category: hourly adjacent nested pairs, share with
    p(T1) > p(T2) by mid, and share executable after taker fees on both legs."""
    pairs = strip_pairs(u)
    if not pairs.height:
        return pl.DataFrame()
    q = _quotes(c)
    cat = u.select("ticker", "category")
    a = (pairs.join(q, on="ticker").join(q.rename({"ticker": "ticker_2", "bid": "bid_2", "ask": "ask_2"}), on=["ticker_2", "q_ts"])
         .filter((pl.col("q_ts") < pl.col("close_time")) & (pl.col("q_ts") < pl.col("close_time_2"))
                 & (pl.col("q_ts") < pl.col("T")) & (pl.col("q_ts") >= START)
                 & (pl.col("w_start_2").is_null()
                    | (pl.col("w_start_2") <= pl.max_horizontal("q_ts", pl.col("w_start").fill_null(pl.col("q_ts"))))))
         .join(cat, on="ticker"))
    prof = (a["bid"].to_numpy() - a["ask_2"].to_numpy() - fee_vec(a["ask_2"].to_numpy(), a["fee_multiplier_2"].to_numpy())
            - fee_vec(1 - a["bid"].to_numpy(), a["fee_multiplier"].to_numpy()))
    a = a.with_columns(((pl.col("bid") + pl.col("ask")) / 2 > (pl.col("bid_2") + pl.col("ask_2")) / 2).alias("viol_mid"),
                       (pl.col("bid") > pl.col("ask_2")).alias("cross_gross"), pl.Series("exec", prof > 0))
    rows = []
    for (cg,), g in a.group_by("category"):
        rows.append({"category": cg, "pairs": g["ticker"].n_unique(), "pair_hours": g.height,
                     "share_viol_mid": _r(float(g["viol_mid"].mean()), 4),
                     "share_crossed_before_fees": _r(float(g["cross_gross"].mean()), 4),
                     "share_exec_after_fees": _r(float(g["exec"].mean()), 4),
                     "pairs_ever_exec": g.filter("exec")["ticker"].n_unique()})
    return pl.DataFrame(rows).sort("pair_hours", descending=True)


# ---- economic rule ------------------------------------------------------------------------------
def fit_k(dev: pl.DataFrame, levels: list[list[str]], n0: float = 50.0) -> dict[tuple, dict]:
    """Hierarchical hazard multipliers from dev: each level's cell k is shrunk in log toward its
    parent cell (the previous, coarser level) with ``n0`` pseudo-observations."""
    tables: dict[tuple, dict] = {}
    parent: dict = {(): math.log(hazard_multiplier(dev["mid"].to_numpy(), dev["y"].to_numpy()))}
    tables[()] = parent
    prev: list[str] = []
    for cols in levels:
        cur = {}
        for key, g in dev.group_by(cols):
            pk = tuple(key[: len(prev)])
            lk_parent = parent.get(pk, parent.get((), 0.0)) if prev else parent[()]
            lk = math.log(hazard_multiplier(g["mid"].to_numpy(), g["y"].to_numpy()))
            cur[tuple(key)] = (g.height * lk + n0 * lk_parent) / (g.height + n0)
        tables[tuple(cols)] = cur
        parent, prev = cur, cols
    return tables


def apply_k(pn: pl.DataFrame, tables: dict[tuple, dict], levels: list[list[str]]) -> np.ndarray:
    """p_model = 1 − (1 − mid)^k with k from the finest level that has the cell."""
    lk = np.full(pn.height, tables[()][()])
    for cols in levels:
        tab = tables[tuple(cols)]
        keys = list(pn.select(cols).iter_rows())
        got = np.array([tab.get(tuple(k), np.nan) for k in keys], dtype=float)
        lk = np.where(np.isfinite(got), got, lk)
    mid = np.clip(pn["mid"].to_numpy(), EPS, 1 - EPS)
    return 1 - (1 - mid) ** np.exp(lk)


def _trades_stats(tr: pl.DataFrame, pnl_col: str = "pnl") -> dict[str, Any]:
    if not tr.height:
        return {"trades": 0}
    mu, t, k = _cm(tr[pnl_col].to_numpy() * 100, tr["event_ticker"].to_numpy())
    return {"trades": tr.height, "markets": tr["ticker"].n_unique(), "events": k, "pnl_c": _r(mu, 2), "t": _r(t, 2),
            "mean_no_ask": _r(float((1 - tr["bid"]).mean()))}


def econ(pn: pl.DataFrame, *, min_trades: int = 30) -> dict[str, Any]:
    """Buy NO at the NO ask when the dev-calibrated model's NO edge after the taker fee ≥ θ
    (θ chosen on dev), one contract per (market, lead), held to settlement; t by event. Models:
    hazard = k by kind → kind×lead → kind×lead×price; price-only = k by price bucket. Control:
    NO on every quote with YES mid ≤ x (x chosen on dev), same leads. Also the first signal per
    market and a logistic blend test of the hazard model against the mid."""
    from albert.research.fundamental import THETAS, fit_logistic, logit

    x = pn.filter((pl.col("spread") <= MAX_SPREAD) & pl.col("split").is_in(["dev", "holdout"]))
    nb = 1 - x["bid"].to_numpy()
    fee = fee_vec(nb, x["fee_multiplier"].to_numpy())
    x = x.with_columns(pl.Series("fee_no", fee),
                       pl.Series("pnl", (1 - x["y"].to_numpy()) - nb - fee))
    dev = x.filter(pl.col("split") == "dev")
    models = {"hazard": [["kind"], ["kind", "lead_b"], ["kind", "lead_b", "price_b"]],
              "price_only": [["price_b"]]}
    out: dict[str, Any] = {"rows": x.height, "dev_rows": dev.height, "holdout_rows": x.height - dev.height}
    for name, levels in models.items():
        tab = fit_k(dev, levels)
        pm = apply_k(x, tab, levels)
        xm = x.with_columns(pl.Series("p_model", pm),
                            pl.Series("edge", (1 - pm) - nb - fee))
        out[f"{name}_p"] = xm["p_model"]
        grid, best = [], None
        for th in THETAS:
            for split in ("dev", "holdout"):
                tr = xm.filter((pl.col("split") == split) & (pl.col("edge") >= th))
                st = {"theta": th, "split": split, **_trades_stats(tr)}
                grid.append(st)
                if split == "dev" and st.get("trades", 0) >= min_trades and (best is None or st["pnl_c"] > best[1]):
                    best = (th, st["pnl_c"])
        th = best[0] if best else None
        res = {"theta_chosen_on_dev": th, "grid": grid}
        if th is not None:
            for split in ("dev", "holdout"):
                tr = xm.filter((pl.col("split") == split) & (pl.col("edge") >= th))
                first = tr.sort("ticker", "t").unique("ticker", keep="first")
                res[f"{split}_first_per_market"] = _trades_stats(first)
                res[f"{split}_by_kind"] = [{"kind": k, **_trades_stats(g)} for (k,), g in tr.group_by("kind")]
                res[f"{split}_by_lead"] = [{"lead_b": k, **_trades_stats(g)} for (k,), g in tr.group_by("lead_b")]
        out[name] = res
    # price-only control: NO on every YES mid <= x
    grid, best = [], None
    for cap in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0):
        for split in ("dev", "holdout"):
            tr = x.filter((pl.col("split") == split) & (pl.col("mid") <= cap))
            st = {"x": cap, "split": split, **_trades_stats(tr)}
            grid.append(st)
            if split == "dev" and st.get("trades", 0) >= min_trades and (best is None or st["pnl_c"] > best[1]):
                best = (cap, st["pnl_c"])
    ctrl = {"x_chosen_on_dev": best[0] if best else None, "grid": grid}
    if best:
        for split in ("dev", "holdout"):
            tr = x.filter((pl.col("split") == split) & (pl.col("mid") <= best[0]))
            ctrl[f"{split}_first_per_market"] = _trades_stats(tr.sort("ticker", "t").unique("ticker", keep="first"))
            ctrl[f"{split}_by_kind"] = [{"kind": k, **_trades_stats(g)} for (k,), g in tr.group_by("kind")]
    out["control"] = ctrl
    # blend: does the hazard model add information beyond the mid?
    hp = out.pop("hazard_p").to_numpy()
    out.pop("price_only_p")
    xb = x.with_columns(pl.Series("p_h", hp))
    blend = {}
    for split in ("dev", "holdout"):
        g = xb.filter(pl.col("split") == split)
        if g.height < 50:
            continue
        X = np.column_stack([np.ones(g.height), logit(g["mid"].to_numpy()), logit(g["p_h"].to_numpy())])
        b, t = fit_logistic(X, g["y"].to_numpy().astype(float), g["event_ticker"].to_numpy())
        blend[split] = {"a": _r(b[0]), "b_mid": _r(b[1]), "c_hazard": _r(b[2]), "t_b": _r(t[1], 2), "t_c": _r(t[2], 2),
                        "n": g.height}
        yv = g["y"].to_numpy().astype(float)
        blend[split]["brier_mid"] = _r(float(((g["mid"].to_numpy() - yv) ** 2).mean()), 5)
        blend[split]["brier_hazard"] = _r(float(((g["p_h"].to_numpy() - yv) ** 2).mean()), 5)
    out["blend"] = blend
    return out


def maker_test(s, pn: pl.DataFrame, *, min_fills: int = 30) -> dict[str, Any]:
    """Maker variant on the window the shared trade tape covers (2026-07-30 → 10-31): rest one NO
    bid from t to the next grid point t2 at floor(1 − p − margin), capped one tick below the NO ask
    (post-only); filled only if a later print trades THROUGH it (fundamental.maker_fills). p is the
    dev-fitted hazard model or, as the control, the mid. Maker fee by series; held to settlement;
    margin chosen on dev; t by event."""
    from albert.research.fundamental import MARGINS, maker_fills

    x = pn.filter((pl.col("spread") <= MAX_SPREAD) & pl.col("split").is_in(["dev", "holdout"])
                  & (pl.col("t") >= datetime(2026, 7, 30, tzinfo=UTC)) & (pl.col("t2") <= TABLE_END))
    if not x.height:
        return {}
    dev_all = pn.filter((pl.col("spread") <= MAX_SPREAD) & (pl.col("split") == "dev"))
    levels = [["kind"], ["kind", "lead_b"], ["kind", "lead_b", "price_b"]]
    tab = fit_k(dev_all, levels)
    x = x.with_columns(pl.Series("p_h", apply_k(x, tab, levels)))
    trades = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet")
              .filter(pl.col("ticker").is_in(x["ticker"].unique().implode()))
              .select("ticker", "created_time", "yes_price", "taker_side").collect())
    out: dict[str, Any] = {"orders_window": x.height, "markets": x["ticker"].n_unique(), "trades_rows": trades.height}
    for name, col in (("hazard", "p_h"), ("mid_control", "mid")):
        pred = x.select("ticker", pl.col("event_ticker").alias("event"), "t", pl.col("t2").alias("end"),
                        pl.col(col).alias("p"), "bid", "ask", "y", "fee_type", "fee_multiplier", "split")
        table, best = [], None
        for m in MARGINS:
            f = maker_fills(pred.drop("split"), trades, m)
            if not f.height:
                continue
            f = f.filter(pl.col("side") == "no").join(pred.select("ticker", "t", "split"), on=["ticker", "t"])
            for split in ("dev", "holdout"):
                part = f.filter(pl.col("split") == split)
                filled = part.filter("filled")
                row = {"margin": m, "split": split, "orders": part.height, "fills": filled.height}
                if filled.height:
                    mu, tt, k = _cm(filled["pnl"].to_numpy() * 100, filled["event"].to_numpy())
                    row.update({"events": k, "pnl_per_fill_c": _r(mu, 2), "t": _r(tt, 2),
                                "pnl_per_order_c": _r(float(part["pnl"].sum()) / part.height * 100, 3)})
                    if split == "dev" and filled.height >= min_fills and (best is None or mu > best[1]):
                        best = (m, mu)
                table.append(row)
        out[name] = {"margin_chosen_on_dev": best[0] if best else None, "grid": table}
    return out


# ---- analysis driver ----------------------------------------------------------------------------
def classification_summary(u_all: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Counts by contract type and category (all parsed candidates and in-scope), with examples."""
    sc = in_scope(u_all)
    counts = (u_all.group_by("kind").agg(pl.len().alias("candidates"), pl.col("series_ticker").n_unique().alias("series"))
              .join(sc.group_by("kind").agg(pl.len().alias("in_scope"), pl.col("event_ticker").n_unique().alias("in_scope_events"),
                                            pl.col("y").mean().round(3).alias("in_scope_yes_rate")), on="kind", how="left")
              .sort("candidates", descending=True))
    by_cat = (sc.group_by("kind", "category").agg(pl.len().alias("markets"), pl.col("event_ticker").n_unique().alias("events"),
                                                  pl.col("series_ticker").n_unique().alias("series"),
                                                  pl.col("y").mean().round(3).alias("yes_rate"))
              .sort("kind", "markets", descending=[False, True]))
    ex: dict[str, Any] = {}
    for (kind, cat), g in u_all.group_by("kind", "category"):
        ex.setdefault(kind, {})[cat] = g.unique("series_ticker").head(3)["title"].to_list()
    # validation of the parsed deadline: NO markets close near T; YES markets resolve early
    no = sc.filter(pl.col("y") == 0).with_columns(((pl.col("close_time") - pl.col("T")).dt.total_minutes() / 60).alias("dh"))
    yes = sc.filter(pl.col("y") == 1).with_columns(((pl.col("T") - pl.col("close_time")).dt.total_minutes() / 60).alias("eh"))
    val = {"no_close_within_2h_of_T": _r(float((no["dh"].abs() <= 2).mean()), 3),
           "no_close_within_26h_of_T": _r(float((no["dh"].abs() <= 26).mean()), 3),
           "no_close_minus_T_median_h": _r(float(no["dh"].median()), 2),
           "yes_closed_early_gt6h_by_kind": {k: _r(float((g["eh"] > 6).mean()), 3) for (k,), g in yes.group_by("kind")}}
    return counts, by_cat, {"examples": ex, "deadline_validation": val}


def _lead_cat(lead_h: pl.Expr) -> pl.Expr:
    return (pl.when(lead_h < 24).then(pl.lit("<1d")).when(lead_h < 168).then(pl.lit("1-7d"))
            .otherwise(pl.lit(">=7d")).alias("lead3"))


def analyze(s) -> Path:
    from albert.reports import Report

    d = out_dir(s)
    u_all = pl.read_parquet(d / "universe_all.parquet")
    u = in_scope(u_all)
    c = pl.read_parquet(d / "candles.parquet")
    c = c.filter(pl.col("ticker").is_in(u["ticker"].implode()))
    counts, by_cat, meta = classification_summary(u_all)
    pn = add_next(build_panel(u, c))
    pn.write_parquet(d / "panel.parquet")
    priced = pn.filter(pl.col("spread") <= MAX_SPREAD)
    summary: dict[str, Any] = {
        "universe": {"in_scope_markets": u.height, "with_candles": c["ticker"].n_unique(),
                     "panel_rows": pn.height, "priced_rows": priced.height,
                     "priced_markets": priced["ticker"].n_unique(), "priced_events": priced["event_ticker"].n_unique(),
                     "split_rows": priced.group_by("split").len().to_dicts()},
        **meta}
    # descriptive modelling (lead)
    shapes = term_shapes(u, c)
    dpath = decay_paths(pn)
    dpath_cat = decay_paths(pn, anchors=(168,), by="category")
    timing = event_timing(pn)
    qv = jump_diffusion(u, c)
    cir = cir_fit(pn)
    arb = arb_by_category(u, c)
    cal_lead = calib_table(priced, ["group", "lead_b"])
    cal_kind = calib_table(priced, ["kind", "lead_b"])
    cal_cell = calib_table(priced.filter(pl.col("group") == "core"), ["lead_b", "price_b"])
    cal_catg = calib_table(priced.filter(pl.col("group") == "core").with_columns(_lead_cat(pl.col("lead_h"))),
                           ["category", "lead3"])
    cal_split = calib_table(priced, ["group", "split", "lead_b"])
    shape = test_shape(pn)
    theta = test_theta(pn)
    term = test_term(u, c, pn)
    # economic test (kept as run; not extended)
    nor = no_returns(pn)
    eco = econ(pn)
    eco["maker"] = maker_test(s, pn)
    summary.update({"term_structure_shapes": shapes, "static_arbitrage_by_category": arb.to_dicts(),
                    "calibration_by_group_lead": cal_lead.to_dicts(), "calibration_by_kind_lead": cal_kind.to_dicts(),
                    "calibration_core_lead_price": cal_cell.to_dicts(), "calibration_core_category": cal_catg.to_dicts(),
                    "calibration_by_split": cal_split.to_dicts(),
                    "decay_paths": dpath.to_dicts(), "decay_paths_by_category": dpath_cat.to_dicts(), "theta": theta.to_dicts(), "hazard_shape": shape.to_dicts(),
                    "event_timing": timing, "jump_diffusion_qv": qv.to_dicts(), "cir": cir.to_dicts(),
                    "term_structure": term, "no_holding": nor.to_dicts(), "economic": eco})
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    rep = Report(s.reports_dir, "research-pm-hazard")
    rep.text("Scope", (
        "Kalshi 'will X happen by T' contracts as a reduced-form hazard (credit-curve) model: p = 1 - exp(-H), "
        "theta dp/dt = -h(1-p) without the event, a jump to 1 at the event. T is parsed from the rules text "
        "(never close_time/settlement_ts, which reveal an early YES). In scope: hazard-typed, T in (2026-01-01, "
        "2026-11-02 05:00 UTC], settled yes/no. Quotes: last hourly candle that ended strictly before t; priced = "
        f"spread <= {MAX_SPREAD}. Decision grid: leads {list(LEADS_H)} hours before T. dev = T before 2026-10-01 04:00 "
        "UTC; holdout = decisions on/after it. t-stats clustered by event_ticker; effects in cents per contract. "
        "Descriptive tests that condition on 'no event yet' use a proxy event time (YES close, or the first candle of "
        "the final run of mids >= 0.97) and are restricted to mid < 0.9."))
    rep.table("Classification (all parsed candidates; in-scope = both outcomes observable)", counts)
    rep.table("In-scope markets by type and category", by_cat, max_rows=60)
    rep.json("Classification examples and deadline-parse validation", meta)
    rep.json("Universe", summary["universe"])
    rep.json("1. Implied hazard term structures across strips (daily 16:00 UTC snapshots; shape = last forward / spot)",
             shapes)
    rep.table("1b/3. Decay paths from anchors (no event yet): realized vs flat-theta p(L)/p(A), implied-hazard ratio",
              dpath, save_as="decay_paths", max_rows=120)
    rep.table("1c/3. Single-contract implied hazard path by category (core, anchor 7d, no event yet)", dpath_cat,
              save_as="decay_paths_category", max_rows=120)
    rep.table("2. Calibration: implied vs realized by group and lead (k = realized/implied hazard multiplier)", cal_lead,
              save_as="calib_group_lead")
    rep.table("2. Calibration by kind and lead", cal_kind, save_as="calib_kind_lead", max_rows=60)
    rep.table("2. Calibration, core hazards, lead x price bucket (longshots = 0-5, 5-15)", cal_cell, save_as="calib_cells",
              max_rows=60)
    rep.table("2. Calibration, core hazards, category x lead", cal_catg, save_as="calib_category", max_rows=60)
    rep.table("2. Calibration by split (dev / holdout / straddle)", cal_split, save_as="calib_split", max_rows=60)
    rep.table("3. Theta: martingale drift, no-event decay vs flat theta, jumps vs implied, per grid step", theta,
              save_as="theta", max_rows=60)
    rep.table("4. Hazard shape: realized events per grid interval vs the flat implied hazard at its start", shape,
              save_as="shape", max_rows=60)
    rep.json("4. Event timing: YES resolution inside the window vs the flat implied hazard (PIT uniform if right)", timing)
    rep.table("5. Jumps vs diffusion: quadratic-variation decomposition of the hourly mid to settlement", qv,
              save_as="qv")
    rep.table("5. CIR-type fit of the implied hazard (no-event steps) and implied-hazard vol", cir, save_as="cir")
    rep.table("6. Static arbitrage frequency across strips by category (hourly adjacent nested pairs)", arb,
              save_as="arb_by_category")
    rep.json("6. Term structure detail: monotonicity pooled and calendar spreads at the near leg's grid", term)
    rep.table("Economic (kept, not extended): hold NO from lead L (taker at the NO ask)", nor, save_as="no_holding",
              max_rows=80)
    rep.json("Test 4: chosen on dev", {
        "rows": {k: eco[k] for k in ("rows", "dev_rows", "holdout_rows")},
        "hazard_theta": eco["hazard"].get("theta_chosen_on_dev"),
        "price_only_theta": eco["price_only"].get("theta_chosen_on_dev"),
        "control_x": eco["control"].get("x_chosen_on_dev"),
        "first_signal_per_market": {m: {sp: eco[m].get(f"{sp}_first_per_market") for sp in ("dev", "holdout")}
                                    for m in ("hazard", "price_only", "control")},
        "blend_logit_y=a+b*logit(mid)+c*logit(p_hazard)": eco.get("blend")})
    for m, title in (("hazard", "hazard model (k by kind x lead x price, dev-fitted)"),
                     ("price_only", "price-only calibration model (k by price bucket, dev-fitted)")):
        rep.table(f"Test 4: buy NO when {title} edge after fee >= theta (dev in-sample, holdout out-of-sample)",
                  pl.DataFrame(eco[m]["grid"], infer_schema_length=None), max_rows=40)
        for sp in ("dev", "holdout"):
            if eco[m].get(f"{sp}_by_kind"):
                rep.table(f"Test 4: {m} at chosen theta, {sp}, by kind",
                          pl.DataFrame(eco[m][f"{sp}_by_kind"], infer_schema_length=None))
            if eco[m].get(f"{sp}_by_lead"):
                rep.table(f"Test 4: {m} at chosen theta, {sp}, by lead",
                          pl.DataFrame(eco[m][f"{sp}_by_lead"], infer_schema_length=None))
    rep.table("Test 4: control - buy NO on every quote with YES mid <= x (same leads)",
              pl.DataFrame(eco["control"]["grid"], infer_schema_length=None), max_rows=40)
    for sp in ("dev", "holdout"):
        if eco["control"].get(f"{sp}_by_kind"):
            rep.table(f"Test 4: control at chosen x, {sp}, by kind",
                      pl.DataFrame(eco["control"][f"{sp}_by_kind"], infer_schema_length=None))
    mk = eco.get("maker") or {}
    for m in ("hazard", "mid_control"):
        if mk.get(m):
            rep.table(f"Test 4 (maker): NO bid at floor(1 - p - margin), trade-through fills, p = {m}; "
                      f"margin chosen on dev = {mk[m]['margin_chosen_on_dev']}",
                      pl.DataFrame(mk[m]["grid"], infer_schema_length=None), max_rows=40)
    path = rep.save()
    print(f"report: {path}", flush=True)
    return path


# ---- entry point -------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from albert.config import settings

    ap = argparse.ArgumentParser(prog="albert.research.pm_hazard")
    ap.add_argument("step", choices=["scan", "universe", "candles", "analyze", "all"])
    a = ap.parse_args(argv)
    s = settings()
    if a.step in ("scan", "all"):
        scan(s)
    if a.step in ("universe", "candles", "all"):
        u = build_universe(s)
        print(u.group_by("kind").len().sort("len", descending=True), flush=True)
    if a.step in ("candles", "all"):
        fetch_candles(s, in_scope(u))
    if a.step in ("analyze", "all"):
        analyze(s)


if __name__ == "__main__":
    main()
