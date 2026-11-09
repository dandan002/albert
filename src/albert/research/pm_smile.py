"""Cross-strike implied-volatility smile of Kalshi ladders: is the implied distribution calibrated?

A Kalshi ladder of "above K" / bracket markets on one numeric underlying X (a daily high, a game
total, a spread, an approval average...) is a strip of digital options. Its survival curve
S(K) = P(X > K) implies a distribution, a Bachelier implied vol per strike and a smile.

**Legs on one latent scale** (as in ``strike_curves``): a leg pays YES iff ``lo < X <= hi``. In
bracket events (any "between"/exact leg) edges move half a grid step outwards ("63° to 64°" is
(62.5, 64.5], "71° or above" is (70.5, inf)); in threshold-only ladders "above K" is (K + u/2, inf)
for a strike grid of resolution u (0 for half-point sports lines). A spread ladder puts both teams
on one margin X = A - B ("B wins by over k" is X < -k) and is oriented per snapshot so that the
implied median is >= 0 (the favourite's margin; decided from prices only).

**Snapshots.** Non-weather ladders: 48, 24, 6 and 1 h before the scheduled
``expected_expiration_time`` (events whose expected expiration is > 3 days after their close are
unscheduled and dropped). Weather (KXHIGH*/KXLOWT*): by local standard time of the measurement
day D, whose end is the markets' close: D-1 12:00 (36 h before close), D 00:00 (24 h), 08:00
(16 h), 12:00 (12 h) and 16:00 (8 h). The quote is the close of the hourly candle that ended at
or before the snapshot (at most 2 h old) of a leg still open (close_time > snapshot).

**Implied distribution per snapshot.** Legs with spread <= 5c (bid may be 0 for a sub-5c wing).
Threshold legs give S at their strike (mid, or 1 - mid for "below"); a complete bracket set gives
S at every interior edge as the share of the summed mids above it (overround removed
multiplicatively). S is made monotone (weighted isotonic regression), F is the median (linear
interpolation of z = Phi^-1(S) between the two strikes that straddle 0.5), sigma_ATM is the local
slope there ((K_b - K_a) / (z_a - z_b)), sigma_K = (F - K) / z_K (|z_K| >= 0.25), and the
moneyness is m = (K - F) / sigma_ATM. Per snapshot (>= 3 off-ATM strikes) and pooled per group x
lead: sigma_K / sigma_ATM = 1 + s m + c m^2.

**Calibration against the realized X_T**, which is only known as the interval (L, U] implied by
which legs settled YES: (a) strike-level tail mass, implied S (or 1 - S below F) against the
realized 1{X > K} by moneyness bucket and wing, t clustered by event; (b) a fractional PIT
histogram (the realized interval's implied CDF mass spread uniformly); (c) the width ratio
lambda, the interval-censored normal MLE of (X_T - F) / sigma_ATM ~ N(mu, lambda^2) with cluster
sandwich errors; (d) the ATM term structure, slope of log sigma_ATM on log lead within ladders.

**Economic tests** (one contract, taker at the snapshot candle's ask / 1 - bid, Kalshi taker fee
by series multiplier, exact to $0.0001, held to settlement): sell the tail side of a leg (buy the
other side) when its model moneyness |m| >= m* (``model``), when its tail-side price <= x
(``price``, the naive control), or always (``all``); and calibration-map rules that price the tail
from the dev-sample realized frequency per (group, lead, moneyness bucket) or per (group, lead,
price bucket). Parameters are chosen on dev (events expiring before 2026-10-01) and reported on
the October holdout. The model only adds value if it beats the price control on holdout.

**Arbitrage**: monotonicity violations between threshold legs and bracket-sum violations, net of
taker fees, at the snapshot quotes (no displayed sizes in candles).
"""

import itertools
import json
import math
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betainc, ndtr, ndtri

from albert.research.strike_curves import isotonic_decreasing

INF = math.inf
HOLDOUT = datetime(2026, 10, 1, tzinfo=UTC)
LEADS_H = (48, 24, 6, 1)
WX_LEADS_H = (36, 24, 16, 12, 8)
MAX_SPREAD = 0.05
MAX_AGE = timedelta(hours=2)
MIN_Z = 0.25
S_CLIP = 0.005
OUT = Path("data/research/pm_smile")
GROUPS = ("wx_high", "wx_low", "sp_total", "sp_teamtotal", "sp_spread", "entertainment", "politics",
          "scitech", "elections", "climate_other")
CANDLE_SOURCES = ("tables/kalshi_candles.parquet", "cache/fundamental_ncaaf_candles.parquet",
                  "cache/fundamental_mlb_candles.parquet", "cache/fundamental_series_age_candles.parquet",
                  "cache/fundamental_soccer_candles.parquet", "cache/fundamental_thin_sports_candles.parquet",
                  "cache/fundamental_thin_sports_candles_tail.parquet",
                  "cache/fundamental_refprice_maker_candles.parquet", "cache/fundamental_approval_candles.parquet",
                  "cache/fundamental_culture_rt_candles.parquet", "cache/fundamental_culture_vercel_candles.parquet")
M_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, INF)


# ---- leg parsing ---------------------------------------------------------------------------------
_N = r"(-?[\d,]*\.?\d+)"
_MULT = {"K": 1e3, "M": 1e6, "B": 1e9}
_WX = ((re.compile(r"^(-?\d+)°? to (-?\d+)°"), "between"), (re.compile(r"^(-?\d+)°? or below"), "le"),
       (re.compile(r"^(-?\d+)°? or above"), "ge"))
_SPORT = ((re.compile(r"^(.+?) wins(?: \w+)? by over ([\d.]+)"), "spread"), (re.compile(r"^Over ([\d.]+)"), "gt"),
          (re.compile(r"^(.+?) over ([\d.]+) "), "team_gt"))
_GEN = ((re.compile(rf"^Above \$?{_N}\s*([KMB])?(?![a-z])"), "gt"), (re.compile(rf"^Below \$?{_N}\s*([KMB])?(?![a-z])"), "lt"),
        (re.compile(rf"(?:^|of )[Aa]t least \$?{_N}\s*([KMB])?(?![a-z])"), "ge"),
        (re.compile(rf"≥\s*\$?{_N}\s*([KMB])?(?![a-z])"), "ge"), (re.compile(rf"^{_N}\s*([KMB])?\+$"), "ge"),
        (re.compile(rf"^<\s*{_N}$"), "lt"), (re.compile(rf"^>\s*{_N}$"), "gt"),
        (re.compile(rf"^{_N}\s*-\s*{_N}$"), "between"), (re.compile(rf"^{_N}%$"), "eq"))


def _num(s: str, suffix: str | None = None) -> float:
    return float(s.replace(",", "")) * _MULT.get(suffix or "", 1.0)


def parse_leg(sub: str | None, group: str) -> tuple[str, float, float, str] | None:
    """(op, a, b, name) from a market's ``yes_sub_title``: op in gt/ge/lt/le/between/eq/spread/
    team_gt; ``name`` is the team for spreads and team totals. None if it is not a numeric leg."""
    s = (sub or "").strip()
    if not s:
        return None
    rules = _WX if group.startswith("wx") else _SPORT if group.startswith("sp_") else _GEN
    for rx, op in rules:
        m = rx.search(s)
        if not m:
            continue
        g = m.groups()
        if op in ("spread", "team_gt"):
            return op, float(g[1]), float(g[1]), g[0].strip()
        if op == "between":
            return op, _num(g[0]), _num(g[1]), ""
        if op == "eq":
            return op, _num(g[0]), _num(g[0]), ""
        v = _num(g[0], g[1] if len(g) > 1 else None)
        return op, v, v, ""
    return None


def _unit(vals: list[float]) -> float:
    for d in range(5):
        if all(abs(v * 10**d - round(v * 10**d)) < 1e-6 for v in vals):
            return 10.0**-d
    return 0.0


def ladder_edges(legs: list[tuple[str, float, float, str]]) -> tuple[list[tuple[float, float]], str]:
    """Latent (lo, hi] per leg and the ladder kind (bracket / threshold) for parsed legs of one
    ladder. Spreads: X = margin of the alphabetically first team."""
    ops = {lg[0] for lg in legs}
    vals = [v for lg in legs for v in lg[1:3]]
    bracket = bool(ops & {"between", "eq"})
    if bracket:
        b = sorted({(lg[1], lg[2]) for lg in legs if lg[0] in ("between", "eq")})
        gaps = [round(b[i + 1][0] - b[i][1], 9) for i in range(len(b) - 1)]
        gaps = [g for g in gaps if g > 1e-9]
        d = (min(gaps) if gaps else _unit(vals)) / 2
    elif all(abs(abs(v) % 1 - 0.5) < 1e-9 for v in vals):
        d = 0.0
    else:
        d = _unit(vals) / 2
    teams = sorted({lg[3] for lg in legs if lg[0] == "spread"})
    out = []
    for op, a, b, name in legs:
        if op == "spread":
            out.append((a + d, INF) if name == teams[0] else (-INF, -a - d))
        elif op in ("gt", "team_gt"):
            out.append((a + d, INF))
        elif op == "ge":
            out.append((a - d, INF))
        elif op == "lt":
            out.append((-INF, a - d))
        elif op == "le":
            out.append((-INF, a + d))
        else:
            out.append((a - d, b + d))
    return out, "bracket" if bracket else "threshold"


def realized_interval(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """(L, U, consistent): X_T lies in (L, U] given which legs settled YES (y = 1)."""
    L, U = -INF, INF
    for a, b, r in zip(lo, hi, y):
        if r == 1:
            L, U = max(L, a), min(U, b)
        elif np.isinf(b) and np.isfinite(a):        # NO on (a, inf): X <= a
            U = min(U, a)
        elif np.isinf(a) and np.isfinite(b):        # NO on (-inf, b]: X > b
            L = max(L, b)
    ok = L < U
    for a, b, r in zip(lo, hi, y):                  # a NO bracket must not contain (L, U]
        if r == 0 and np.isfinite(a) and np.isfinite(b) and a <= L and U <= b:
            ok = False
    return L, U, ok


# ---- universe ------------------------------------------------------------------------------------
def group_of(series: str, category: str) -> str | None:
    if series.startswith("KXHIGH"):
        return "wx_high"
    if series.startswith("KXLOWT"):
        return "wx_low"
    if category == "Sports":
        if "INNING" in series or not re.search(r"(SPREAD|TOTAL)$", series):
            return None
        if series.endswith("TEAMTOTAL"):
            return "sp_teamtotal"
        return "sp_spread" if series.endswith("SPREAD") else "sp_total"
    return {"Entertainment": "entertainment", "Politics": "politics", "Science and Technology": "scitech",
            "Elections": "elections", "Climate and Weather": "climate_other"}.get(category)


def build_ladders(tables: Path, min_legs: int = 4) -> tuple[pl.DataFrame, dict[str, Any]]:
    """One row per leg of every numeric ladder with >= ``min_legs`` parsed legs and a consistent
    realized interval: ladder, event, series, category, group, kind, lo, hi, y, open/close time,
    t_ref (snapshot reference), fee multiplier/type, realized L/U."""
    mk = (pl.scan_parquet(tables / "kalshi_markets.parquet")
          .filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("category") != "Mentions"))
          .select("ticker", "event_ticker", "series_ticker", "category", "yes_sub_title", "result",
                  "open_time", "close_time", "expected_expiration_time", "fee_type", "fee_multiplier")
          .collect())
    ser = pl.read_parquet(tables / "kalshi_series.parquet", columns=["series_ticker", "fee_multiplier"])
    fm = dict(zip(ser["series_ticker"], ser["fee_multiplier"]))
    mk = mk.with_columns(pl.struct("series_ticker", "category").map_elements(
        lambda r: group_of(r["series_ticker"], r["category"]), return_dtype=pl.String).alias("group"))
    mk = mk.filter(pl.col("group").is_not_null())
    parsed = [parse_leg(s, g) for s, g in zip(mk["yes_sub_title"], mk["group"])]
    mk = mk.with_columns(pl.Series("op", [p[0] if p else None for p in parsed], pl.String),
                         pl.Series("a", [p[1] if p else None for p in parsed], pl.Float64),
                         pl.Series("b", [p[2] if p else None for p in parsed], pl.Float64),
                         pl.Series("name", [p[3] if p else None for p in parsed], pl.String))
    stats: dict[str, Any] = {"markets_in_groups": mk.height, "parsed": int(mk["op"].is_not_null().sum())}
    # an event's ladder must be entirely numeric (a non-numeric leg means another market type)
    bad = mk.filter(pl.col("op").is_null())["event_ticker"].unique()
    mk = mk.filter(~pl.col("event_ticker").is_in(bad.implode()) & pl.col("op").is_not_null())
    mk = mk.with_columns(pl.when(pl.col("op") == "team_gt").then(pl.col("event_ticker") + "|" + pl.col("name"))
                         .otherwise(pl.col("event_ticker")).alias("ladder"))
    rows: list[dict[str, Any]] = []
    drop = defaultdict(int)
    for (lad,), g in mk.sort("ticker").group_by("ladder", maintain_order=True):
        if g.height < min_legs:
            drop["too_few_legs"] += 1
            continue
        legs = list(zip(g["op"], g["a"], g["b"], g["name"]))
        if g["op"].eq("spread").any() and len({lg[3] for lg in legs}) != 2:
            drop["spread_teams"] += 1
            continue
        edges, kind = ladder_edges(legs)
        if len(set(edges)) < len(edges):
            drop["duplicate_strikes"] += 1
            continue
        lo = np.array([e[0] for e in edges])
        hi = np.array([e[1] for e in edges])
        y = (g["result"] == "yes").cast(pl.Int8).to_numpy()
        if kind == "bracket" and y.sum() != 1:
            drop["bracket_not_one_yes"] += 1
            continue
        L, U, ok = realized_interval(lo, hi, y)
        if not ok:
            drop["inconsistent_results"] += 1
            continue
        grp = g["group"][0]
        close_max = g["close_time"].max()
        exp = g["expected_expiration_time"].drop_nulls().mode().sort().to_list()
        exp = exp[0] if exp else None
        if grp.startswith("wx"):
            t_ref = close_max
        else:
            t_ref = exp
            if t_ref is None or t_ref > close_max + timedelta(days=3):
                drop["unscheduled_expiry"] += 1
                continue
        for i, r in enumerate(g.iter_rows(named=True)):
            rows.append({"ticker": r["ticker"], "ladder": lad, "event_ticker": r["event_ticker"],
                         "series_ticker": r["series_ticker"], "category": r["category"], "group": grp,
                         "kind": kind, "op": r["op"], "lo": float(lo[i]), "hi": float(hi[i]), "y": int(y[i]),
                         "open_time": r["open_time"], "close_time": r["close_time"], "t_ref": t_ref, "exp_time": exp,
                         "fee_mult": float(r["fee_multiplier"] if r["fee_multiplier"] is not None
                                           else fm.get(r["series_ticker"]) or 1.0),
                         "fee_type": r["fee_type"], "L": L, "U": U, "n_legs": g.height})
    stats["dropped_ladders"] = dict(drop)
    df = pl.DataFrame(rows)
    stats["ladders"] = df["ladder"].n_unique()
    stats["legs"] = df.height
    return df, stats


def snapshot_times(lad: pl.DataFrame) -> pl.DataFrame:
    """(ladder, lead_h, t_snap) for every ladder: weather by local hour before the day's close,
    others before the expected expiration (whole hours, candles end on the hour)."""
    one = lad.group_by("ladder").agg(pl.col("group").first(), pl.col("t_ref").first(),
                                     pl.col("event_ticker").first())
    frames = []
    for wx, leads in ((True, WX_LEADS_H), (False, LEADS_H)):
        part = one.filter(pl.col("group").str.starts_with("wx") == wx)
        for h in leads:
            frames.append(part.with_columns(pl.lit(h).alias("lead_h"),
                                            (pl.col("t_ref") - pl.duration(hours=h)).dt.truncate("1h").alias("t_snap")))
    return pl.concat(frames).select("ladder", "event_ticker", "group", "lead_h", "t_snap", "t_ref")


# ---- candles -------------------------------------------------------------------------------------
def load_candles(data_dir: Path, tickers: list[str], extra: tuple[Path, ...] = ()) -> pl.DataFrame:
    """Hourly candle closes (ticker, end_ts, bid, ask) for ``tickers`` from every cached source."""
    tk = pl.Series(tickers).unique().implode()
    frames = []
    for src in (*[data_dir / s for s in CANDLE_SOURCES], *extra):
        if not src.exists():
            continue
        lf = pl.scan_parquet(src)
        sch = lf.collect_schema()
        if "yes_bid_close" not in sch:
            continue
        if "period_minutes" in sch:
            lf = lf.filter(pl.col("period_minutes") == 60)
        frames.append(lf.filter(pl.col("ticker").is_in(tk))
                      .select("ticker", pl.col("end_ts").cast(pl.Datetime("us", "UTC")),
                              pl.col("yes_bid_close").cast(pl.Float64).alias("bid"),
                              pl.col("yes_ask_close").cast(pl.Float64).alias("ask")).collect())
    if not frames:
        return pl.DataFrame(schema={"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "bid": pl.Float64,
                                    "ask": pl.Float64})
    return (pl.concat(frames).filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
            .unique(["ticker", "end_ts"], keep="first").sort("ticker", "end_ts"))


def quotes_at(lad: pl.DataFrame, snaps: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Every (snapshot, leg) with the last candle that ended at or before the snapshot (within
    ``MAX_AGE``), for legs open at the snapshot (open_time <= t_snap < close_time)."""
    legs = lad.join(snaps.select("ladder", "lead_h", "t_snap"), on="ladder")
    legs = legs.filter((pl.col("close_time") > pl.col("t_snap")) & (pl.col("open_time") <= pl.col("t_snap")))
    c = candles.rename({"end_ts": "q_ts"}).sort("q_ts")
    q = legs.sort("t_snap").join_asof(c, left_on="t_snap", right_on="q_ts", by="ticker", strategy="backward",
                                      tolerance=MAX_AGE)
    return q


def coverage(lad: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """Per group and lead: ladders, ladders with every open leg quoted, legs quoted."""
    per = (q.group_by("ladder", "group", "lead_h").agg(pl.len().alias("open_legs"),
                                                       pl.col("q_ts").is_not_null().sum().alias("quoted"))
           .with_columns((pl.col("quoted") == pl.col("open_legs")).alias("full")))
    return (per.group_by("group", "lead_h").agg(pl.len().alias("ladders"), pl.col("full").sum().alias("full"),
                                                (pl.col("quoted") >= 4).sum().alias("ge4_quoted"),
                                                pl.col("quoted").sum().alias("legs_quoted"))
            .sort("group", "lead_h"))


def fetch_missing(s, lad: pl.DataFrame, q: pl.DataFrame, out: Path, *, workers: int = 2,
                  select: pl.Expr | None = None, hist_select: pl.Expr | None = None,
                  max_markets: int | None = None) -> pl.DataFrame:
    """Hourly candles for the legs of ladders that miss quotes at some snapshot: window
    [t_ref - 50 h, min(t_ref, close)] per leg. Post-cutoff markets via the batch endpoint (legs of
    one UTC expiry day per request, <= 100 tickers and 10,000 candles), older ones via the
    historical per-market endpoint. Appends to ``out`` (parquet), skipping tickers already in it."""
    from concurrent.futures import ThreadPoolExecutor

    from albert.http import CachedClient, HttpError
    from albert.venues import kalshi as k

    miss = q.filter(pl.col("q_ts").is_null())
    if select is not None:
        miss = miss.filter(select)
    lads = miss["ladder"].unique()
    want = lad.filter(pl.col("ladder").is_in(lads.implode()))
    have = pl.read_parquet(out) if out.exists() else None
    if have is not None and have.height:
        want = want.filter(~pl.col("ticker").is_in(have["ticker"].unique().implode()))
    if max_markets:
        want = want.head(max_markets)
    if want.is_empty():
        return have if have is not None else pl.DataFrame()
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.25))
    cutoff = api.cutoff().get("market_settled_ts")
    max_lead = max(*LEADS_H, *WX_LEADS_H) + 3
    want = want.with_columns((pl.col("t_ref") - pl.duration(hours=max_lead)).alias("w_lo"),
                             pl.min_horizontal("t_ref", "close_time").alias("w_hi"),
                             pl.col("t_ref").dt.date().alias("day"))
    hist = want.filter(pl.col("close_time") < cutoff) if cutoff else want.head(0)
    if hist_select is not None:
        hist = hist.filter(hist_select)
    hist = hist.with_columns(pl.col("group").replace_strict(
        {g: i for i, g in enumerate(("wx_low", "politics", "scitech", "entertainment", "climate_other", "elections"))},
        default=9, return_dtype=pl.Int64).alias("prio")).sort("prio", "t_ref")
    live = want.filter(pl.col("close_time") >= cutoff) if cutoff else want
    jobs: list[tuple[str, Any]] = []
    for (day,), g in live.group_by("day"):
        lo, hi = g["w_lo"].min(), g["w_hi"].max()
        hours = int((hi - lo).total_seconds() // 3600) + 2
        per = max(1, min(100, 10_000 // hours))
        tk = sorted(g["ticker"].to_list())
        for i in range(0, len(tk), per):
            jobs.append(("batch", (tk[i:i + per], lo, hi)))
    for r in hist.iter_rows(named=True):
        jobs.append(("hist", (r["ticker"], r["w_lo"], r["w_hi"])))
    print(f"fetch: {live.height} live legs, {hist.height} historical legs, {len(jobs)} requests", flush=True)

    def one(job: tuple[str, Any]) -> list[dict[str, Any]]:
        kind, a = job
        for attempt in range(4):
            try:
                if kind == "batch":
                    page = api.candles_batch(a[0], a[1], a[2], 60)
                    rows = []
                    for m in page.payload.get("markets") or []:
                        rows.extend(k.candle_rows(m, m["market_ticker"], 60))
                    return rows
                page = api.candles(a[0], a[1], a[2], 60, True)
                return k.candle_rows(page.payload, a[0], 60)
            except (HttpError, RuntimeError) as error:
                if "429" not in str(error) or attempt == 3:
                    print(f"skip {kind} {str(a[0])[:60]}: {str(error)[-100:]}", flush=True)
                    return []
                import time
                time.sleep(10 * (attempt + 1))
        return []

    rows: list[dict[str, Any]] = []
    frames = [have] if have is not None and have.height else []

    def save() -> None:
        nonlocal rows
        if rows:
            frames.append(pl.DataFrame(rows, schema=k.CANDLE_SCHEMA))
            rows = []
        df = pl.concat(frames, how="diagonal_relaxed").unique(["ticker", "end_ts"])
        frames[:] = [df]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.replace(out)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(one, jobs), 1):
            rows.extend(got)
            if i % 200 == 0:
                save()
                print(f"  {i}/{len(jobs)} requests {datetime.now(UTC):%H:%M:%S}", flush=True)
    save()
    return frames[0]


# ---- implied distribution per snapshot -----------------------------------------------------------
def strike_points(kind: str, lo: np.ndarray, hi: np.ndarray, bid: np.ndarray, ask: np.ndarray,
                  ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(K, S, w): survival points of one snapshot sorted by strike. Threshold legs give S at their
    strike (weight 1 / max(spread, 1c)^2, duplicates merged); a bracket set needs every leg quoted
    and contiguous edges, and gives S at each interior edge as the share of summed mids above it."""
    mid = (bid + ask) / 2
    if kind == "bracket":
        if not ok.all():
            return None
        o = np.argsort(lo)
        lo_, hi_, mid_ = lo[o], hi[o], mid[o]
        if not (np.isinf(lo_[0]) and np.isinf(hi_[-1]) and np.allclose(hi_[:-1], lo_[1:])):
            return None
        tot = mid_.sum()
        if tot <= 0:
            return None
        S = np.cumsum(mid_[::-1])[::-1][1:] / tot
        return lo_[1:], S, np.ones(len(S))
    K, S, w = [], [], []
    for i in np.flatnonzero(ok):
        spr = max(ask[i] - bid[i], 0.01)
        if np.isinf(hi[i]):
            K.append(lo[i]), S.append(mid[i]), w.append(1 / spr**2)
        elif np.isinf(lo[i]):
            K.append(hi[i]), S.append(1 - mid[i]), w.append(1 / spr**2)
    if len(K) < 2:
        return None
    K_, S_, w_ = np.array(K), np.array(S), np.array(w)
    uk = np.unique(K_)
    if len(uk) < len(K_):
        S_ = np.array([np.average(S_[K_ == k], weights=w_[K_ == k]) for k in uk])
        w_ = np.array([w_[K_ == k].sum() for k in uk])
        K_ = uk
    o = np.argsort(K_)
    return K_[o], S_[o], w_[o]


def _strike_at(K: np.ndarray, z: np.ndarray, zq: float) -> float | None:
    """Strike where the (non-increasing) z = Phi^-1(S) curve crosses ``zq``, linear in z between
    the two quoted strikes around it (exact for a normal strip); None outside the quoted range."""
    idx = np.flatnonzero((z[:-1] >= zq) & (z[1:] < zq))
    if idx.size == 0:
        return None
    i = int(idx[0])
    return float(K[i] + (z[i] - zq) / (z[i] - z[i + 1]) * (K[i + 1] - K[i]))


Q75 = 0.6744897501960817


def implied(K: np.ndarray, S: np.ndarray, w: np.ndarray) -> dict[str, Any] | None:
    """Monotone survival, median F, sigma_ATM, per-strike Bachelier vols and moneyness, and the
    snapshot smile sigma_K / sigma_ATM = 1 + s m + c m^2 (>= 3 off-ATM strikes). sigma_ATM is the
    implied interquartile range / 1.349 (the local slope at F where the quartiles are not both
    inside the quoted strikes): the local slope between two adjacent strikes is dominated by
    point masses of lumpy underlyings (football key numbers). None when the quoted strikes do not
    straddle the median."""
    Si = isotonic_decreasing(S, w)
    z = ndtri(np.clip(Si, S_CLIP, 1 - S_CLIP))
    pos = np.flatnonzero(z >= 0)
    if pos.size == 0 or pos[-1] == len(z) - 1:
        return None
    a = pos[-1]
    b = a + 1
    if K[b] <= K[a] or z[a] <= z[b]:
        return None
    sig_loc = (K[b] - K[a]) / (z[a] - z[b])
    F = K[a] + z[a] * sig_loc
    k75, k25 = _strike_at(K, z, Q75), _strike_at(K, z, -Q75)
    sig = (k25 - k75) / (2 * Q75) if k75 is not None and k25 is not None and k25 > k75 else sig_loc
    m = (K - F) / sig
    inner = (Si > S_CLIP) & (Si < 1 - S_CLIP) & (np.abs(z) >= MIN_Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = np.where(inner, (F - K) / z, np.nan)
    sk[~(sk > 0)] = np.nan
    v = sk / sig
    s_ = c_ = np.nan
    fit = np.isfinite(v) & (np.abs(m) <= 4)
    fit[[a, b]] = False                     # F is interpolated between these two: their vols are the local slope
    if fit.sum() >= 3:
        X = np.column_stack([m[fit], m[fit] ** 2])
        coef, *_ = np.linalg.lstsq(X, v[fit] - 1, rcond=None)
        s_, c_ = float(coef[0]), float(coef[1])
    return {"Si": Si, "z": z, "F": float(F), "sigma": float(sig), "sigma_loc": float(sig_loc), "m": m, "sigK": sk,
            "v": v, "s": s_, "c": c_, "n_fit": int(fit.sum()), "ia": int(a), "iqr": k75 is not None and k25 is not None}


def taker_fee_v(price: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """Kalshi taker fee for one contract, 0.07 x multiplier x P(1-P) rounded up to $0.0001."""
    return np.ceil(0.07 * mult * price * (1 - price) * 1e4 - 1e-9) / 1e4


def build_snapshots(lad: pl.DataFrame, q: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """(snapshots, points, legs): one row per fitted snapshot, per survival point, per market leg.
    Spread ladders are flipped (X -> -X) when the implied median is negative."""
    q = q.with_columns(((pl.col("ask") - pl.col("bid") <= MAX_SPREAD + 1e-9) & (pl.col("ask") >= pl.col("bid"))
                        & pl.col("q_ts").is_not_null()).fill_null(False).alias("ok"))
    q = q.sort("ladder", "lead_h", "lo", "hi")
    keys = q.select("ladder", "lead_h").to_numpy()
    cols = {c: q[c].to_numpy() for c in ("lo", "hi", "bid", "ask", "ok", "y")}
    meta = q.select("ladder", "event_ticker", "series_ticker", "category", "group", "kind", "t_ref", "exp_time",
                    "t_snap", "lead_h", "L", "U", "n_legs").to_dicts()
    tick = q["ticker"].to_list()
    starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
    ends = np.r_[starts[1:], len(q)]
    snaps, pts, legs = [], [], []
    why = defaultdict(int)
    for a, b in zip(starts, ends):
        md = meta[a]
        lo, hi, bid, ask = (cols[c][a:b].astype(float) for c in ("lo", "hi", "bid", "ask"))
        ok = cols["ok"][a:b].astype(bool)
        if ok.sum() < 3:
            why["fewer_than_3_quoted"] += 1
            continue
        sp = strike_points(md["kind"], lo, hi, np.nan_to_num(bid), np.nan_to_num(ask, nan=1.0), ok)
        if sp is None or len(sp[0]) < 3:
            why["no_strip"] += 1
            continue
        K, S, w = sp
        fit = implied(K, S, w)
        if fit is None:
            why["median_outside_strikes"] += 1
            continue
        L, U = md["L"], md["U"]
        orient = 1
        if md["group"] == "sp_spread" and fit["F"] < 0:
            orient = -1
            K, S, w = -K[::-1], 1 - S[::-1], w[::-1]
            fit = implied(K, S, w)
            if fit is None:
                why["median_outside_strikes"] += 1
                continue
            L, U = -U, -L
            lo, hi = -hi, -lo
        F, sig = fit["F"], fit["sigma"]
        snap_id = len(snaps)
        y_pt = np.where(K <= L, 1.0, np.where(K >= U, 0.0, np.nan))
        Si = fit["Si"]
        ia = np.flatnonzero(K <= L)
        ib = np.flatnonzero(K >= U)
        pit_lo = 1 - Si[ia[-1]] if ia.size else 0.0
        pit_hi = 1 - Si[ib[0]] if ib.size else 1.0
        snaps.append({"snap_id": snap_id, "ladder": md["ladder"], "event_ticker": md["event_ticker"],
                      "series_ticker": md["series_ticker"], "category": md["category"], "group": md["group"],
                      "kind": md["kind"], "snapshot_ts": md["t_snap"], "t_ref": md["t_ref"],
                      "expected_expiration_time": md["exp_time"], "lead_h": md["lead_h"], "F": F, "sigma_atm": sig,
                      "sigma_loc": fit["sigma_loc"], "sigma_from_iqr": fit["iqr"],
                      "s": fit["s"], "c": fit["c"], "n_smile_pts": fit["n_fit"], "n_points": len(K),
                      "n_legs_used": int(ok.sum()), "n_legs": md["n_legs"], "orient": orient,
                      "x_lo": L, "x_hi": U, "z_lo": (L - F) / sig, "z_hi": (U - F) / sig,
                      "pit_lo": float(pit_lo), "pit_hi": float(pit_hi),
                      "sum_mid": float(((bid + ask) / 2)[ok].sum()) if md["kind"] == "bracket" else None})
        for j in range(len(K)):
            pts.append((snap_id, float(K[j]), float(S[j]), float(Si[j]), float(fit["z"][j]), float(fit["m"][j]),
                        float(fit["sigK"][j]), float(fit["v"][j]), float(y_pt[j]), j in (fit["ia"], fit["ia"] + 1)))
        for i in range(len(lo)):
            ml = (lo[i] - F) / sig if np.isfinite(lo[i]) else -INF
            mh = (hi[i] - F) / sig if np.isfinite(hi[i]) else INF
            bq, aq = bid[i], ask[i]
            legs.append({"snap_id": snap_id, "ticker": tick[a + i], "lo": float(lo[i]), "hi": float(hi[i]),
                         "m_lo": float(ml), "m_hi": float(mh), "bid": None if not np.isfinite(bq) else float(bq),
                         "ask": None if not np.isfinite(aq) else float(aq), "ok": bool(ok[i]),
                         "y": int(cols["y"][a + i])})
    snap_df = pl.DataFrame(snaps, infer_schema_length=None)
    pt_df = pl.DataFrame(pts, schema={"snap_id": pl.Int64, "K": pl.Float64, "S_raw": pl.Float64, "S": pl.Float64,
                                      "z": pl.Float64, "m": pl.Float64, "sigma_K": pl.Float64, "v": pl.Float64,
                                      "y_gt": pl.Float64, "atm_pair": pl.Boolean}, orient="row")
    for c in ("sigma_K", "v", "y_gt"):
        pt_df = pt_df.with_columns(pl.col(c).fill_nan(None))
    leg_df = pl.DataFrame(legs, infer_schema_length=None)
    return snap_df, pt_df, leg_df, dict(why)


# ---- statistics ----------------------------------------------------------------------------------
def cmean(x: np.ndarray, g: np.ndarray) -> tuple[float | None, float | None, int, int]:
    """(mean, t clustered by g, n, clusters); NaN values skipped."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    if len(x) == 0:
        return None, None, 0, 0
    mu = float(x.mean())
    _, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=x - mu)
    k = len(sums)
    if k < 2:
        return mu, None, len(x), k
    se = math.sqrt((sums**2).sum() * k / (k - 1)) / len(x)
    return mu, (mu / se if se > 0 else None), len(x), k


def ols_cluster(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and cluster-robust t-stats."""
    XtX = np.linalg.pinv(X.T @ X)
    beta = XtX @ X.T @ y
    u = y - X @ beta
    _, inv = np.unique(g, return_inverse=True)
    k = inv.max() + 1
    S = np.zeros((k, X.shape[1]))
    np.add.at(S, inv, X * u[:, None])
    V = XtX @ (S.T @ S) @ XtX * (k / max(k - 1, 1))
    se = np.sqrt(np.clip(np.diag(V), 1e-30, None))
    return beta, beta / se


def _interval_ll(theta: np.ndarray, zl: np.ndarray, zh: np.ndarray) -> np.ndarray:
    mu, lam = theta[0], math.exp(theta[1])
    p = ndtr((zh - mu) / lam) - ndtr((zl - mu) / lam)
    return np.log(np.clip(p, 1e-12, None))


def width_mle(zl: np.ndarray, zh: np.ndarray, g: np.ndarray) -> dict[str, Any]:
    """Interval-censored MLE of (X_T - F) / sigma_ATM ~ N(mu, lambda^2) with a cluster sandwich:
    lambda > 1 means the market's ATM width is too narrow, < 1 too wide."""
    ok = (zh > zl) & ~(np.isinf(zl) & np.isinf(zh))
    zl, zh, g = zl[ok], zh[ok], np.asarray(g)[ok]
    if len(zl) < 20:
        return {"n": len(zl)}
    res = minimize(lambda t: -_interval_ll(t, zl, zh).sum(), np.array([0.0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    th = res.x
    h = 1e-4
    score = np.zeros((len(zl), 2))
    H = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        score[:, j] = (_interval_ll(th + e, zl, zh) - _interval_ll(th - e, zl, zh)) / (2 * h)
        for k in range(2):
            f = np.zeros(2)
            f[k] = h
            H[j, k] = -(_interval_ll(th + e + f, zl, zh).sum() - _interval_ll(th + e - f, zl, zh).sum()
                        - _interval_ll(th - e + f, zl, zh).sum() + _interval_ll(th - e - f, zl, zh).sum()) / (4 * h * h)
    _, inv = np.unique(g, return_inverse=True)
    k = inv.max() + 1
    Sg = np.zeros((k, 2))
    np.add.at(Sg, inv, score)
    Hi = np.linalg.pinv(H)
    V = Hi @ (Sg.T @ Sg) @ Hi * (k / max(k - 1, 1))
    lam = math.exp(th[1])
    se_lam = lam * math.sqrt(max(V[1, 1], 0))
    se_mu = math.sqrt(max(V[0, 0], 0))
    return {"n": len(zl), "events": int(k), "lambda": round(lam, 3), "se_lambda": round(se_lam, 3),
            "t_lambda_eq_1": round((lam - 1) / se_lam, 2) if se_lam > 0 else None,
            "mu": round(float(th[0]), 3), "t_mu": round(float(th[0]) / se_mu, 2) if se_mu > 0 else None}


# ---- analysis ------------------------------------------------------------------------------------
def family(series: pl.Expr, group: pl.Expr) -> pl.Expr:
    """Reporting family: the league for sports ladders, else the group."""
    lg = series.str.extract(r"^KX(NCAAF|NFL|MLB|WNBA|NBA|NHL)", 1)
    return (pl.when(group.str.starts_with("sp_"))
            .then(group + "/" + lg.fill_null("other") + pl.when(series.str.contains(r"\d[HQ]|F5")).then(pl.lit("-part")).otherwise(pl.lit("")))
            .otherwise(group).alias("family"))


def _split(col: str = "t_ref") -> pl.Expr:
    return pl.when(pl.col(col) < HOLDOUT).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split")


def _mbucket(col: str) -> pl.Expr:
    e = pl.col(col).abs()
    out = pl.when(e < M_EDGES[1]).then(pl.lit(f"[0,{M_EDGES[1]})"))
    for lo_, hi_ in itertools.pairwise(M_EDGES[1:]):
        out = out.when(e < hi_).then(pl.lit(f"[{lo_},{hi_})" if math.isfinite(hi_) else f">={lo_}"))
    return out.otherwise(pl.lit(f">={M_EDGES[-2]}"))


def smile_tables(sn: pl.DataFrame, pt: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(params, curve): pooled smile per group x lead x split, sigma_K / sigma_ATM = 1 + s m + c m^2
    by OLS over all off-ATM strikes (|z| >= 0.25, |m| <= 3.5, the two strikes that define sigma_ATM
    excluded), t clustered by event; and the median ratio per moneyness bucket."""
    p = pt.join(sn.select("snap_id", "group", "lead_h", "split", "event_ticker"), on="snap_id")
    p = p.filter(pl.col("v").is_not_null() & ~pl.col("atm_pair") & (pl.col("m").abs() <= 3.5))
    rows = []
    cells = [((grp, lead, sp), g) for (grp, lead, sp), g in p.group_by("group", "lead_h", "split")]
    cells += [((grp, lead, "both"), g) for (grp, lead), g in p.group_by("group", "lead_h")]
    for (grp, lead, sp), g in cells:
        if g.height < 30:
            continue
        m = g["m"].to_numpy()
        beta, t = ols_cluster(np.column_stack([np.ones_like(m), m, m**2]), g["v"].to_numpy() - 1, g["event_ticker"].to_numpy())
        se = np.abs(beta / t)
        s_ = sn.filter((pl.col("group") == grp) & (pl.col("lead_h") == lead) & ((pl.col("split") == sp) | (sp == "both")))
        q = s_.filter(pl.col("s").is_not_nan() & pl.col("c").is_not_nan())
        rows.append({"group": grp, "lead_h": lead, "split": sp, "snapshots": s_.height,
                     "s_ci95": f"[{beta[1] - 1.96 * se[1]:.3f}, {beta[1] + 1.96 * se[1]:.3f}]",
                     "c_ci95": f"[{beta[2] - 1.96 * se[2]:.3f}, {beta[2] + 1.96 * se[2]:.3f}]",
                     "s_snap_iqr": f"[{q['s'].quantile(0.25):.3f}, {q['s'].quantile(0.75):.3f}]" if q.height >= 10 else None,
                     "c_snap_iqr": f"[{q['c'].quantile(0.25):.3f}, {q['c'].quantile(0.75):.3f}]" if q.height >= 10 else None,
                     "events": s_["event_ticker"].n_unique(), "points": g.height,
                     "sigma_atm_median": round(float(s_["sigma_atm"].median()), 4),
                     "a": round(float(beta[0]), 4), "s": round(float(beta[1]), 4), "t_s": round(float(t[1]), 2),
                     "c": round(float(beta[2]), 4), "t_c": round(float(t[2]), 2),
                     "s_snap_median": None if s_["s"].drop_nans().is_empty() else round(float(s_["s"].drop_nans().median()), 4),
                     "c_snap_median": None if s_["c"].drop_nans().is_empty() else round(float(s_["c"].drop_nans().median()), 4)})
    params = pl.DataFrame(rows).sort("group", "lead_h", "split") if rows else pl.DataFrame()
    curve = (p.with_columns(pl.when(pl.col("m") > 0).then(pl.lit("upper")).otherwise(pl.lit("lower")).alias("wing"),
                            _mbucket("m").alias("m_bucket"))
             .group_by("group", "split", "wing", "m_bucket")
             .agg(pl.len().alias("points"), pl.col("v").median().round(3).alias("v_median"),
                  pl.col("v").mean().round(3).alias("v_mean"))
             .sort("group", "split", "wing", "m_bucket"))
    return params, curve


def _z_h0(real: np.ndarray, imp: np.ndarray, g: np.ndarray) -> float | None:
    """z of realized minus implied tail mass under H0 (the implied probabilities are true): per
    event, mean realized minus mean implied over its strikes and leads, variance mean p(1-p) (the
    strikes of one event share one outcome); events independent. Valid when no tail occurred,
    where the clustered t collapses."""
    _, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    r = np.bincount(inv, weights=real) / cnt
    i = np.bincount(inv, weights=imp) / cnt
    v = np.bincount(inv, weights=imp * (1 - imp)) / cnt
    se = math.sqrt(float(v.sum()))
    return round(float((r - i).sum()) / se, 2) if se > 0 else None


def tail_calibration(sn: pl.DataFrame, pt: pl.DataFrame, by_lead: bool = True) -> pl.DataFrame:
    """Implied vs realized tail mass per strike, by group (x lead) x split x wing x |m| bucket:
    upper wing (K > F) compares S with 1{X > K}, lower wing 1 - S with 1{X <= K}. Cents, t by event."""
    p = pt.join(sn.select("snap_id", "group", "lead_h", "split", "event_ticker"), on="snap_id")
    p = p.filter(pl.col("y_gt").is_not_null())
    p = p.with_columns(pl.when(pl.col("m") > 0).then(pl.lit("upper")).otherwise(pl.lit("lower")).alias("wing"),
                       _mbucket("m").alias("m_bucket"),
                       pl.when(pl.col("m") > 0).then(pl.col("S")).otherwise(1 - pl.col("S")).alias("imp"),
                       pl.when(pl.col("m") > 0).then(pl.col("y_gt")).otherwise(1 - pl.col("y_gt")).alias("real"))
    p = p.with_columns(pl.col("m").abs().map_batches(lambda s: pl.Series(ndtr(-s.to_numpy())), return_dtype=pl.Float64).alias("gauss"))
    keys = ["group", "lead_h", "split", "wing", "m_bucket"] if by_lead else ["group", "split", "wing", "m_bucket"]
    rows = []
    for k, g in p.group_by(keys):
        if g.height < 20:
            continue
        d = (g["real"] - g["imp"]).to_numpy()
        mu, t, n, ne = cmean(d, g["event_ticker"].to_numpy())
        rows.append({**dict(zip(keys, k)), "points": n, "events": ne, "z_h0": _z_h0(g["real"].to_numpy(), g["imp"].to_numpy(), g["event_ticker"].to_numpy()),
                     "implied_c": round(float(g["imp"].mean()) * 100, 2), "realized_c": round(float(g["real"].mean()) * 100, 2),
                     "gauss_c": round(float(g["gauss"].mean()) * 100, 2),
                     "real_minus_imp_c": round(mu * 100, 2), "t": None if t is None else round(t, 2)})
    return pl.DataFrame(rows).sort(keys) if rows else pl.DataFrame()


def tail_summary(sn: pl.DataFrame, pt: pl.DataFrame) -> pl.DataFrame:
    """Implied vs realized mass beyond |m| thresholds per group x split (all leads): for every
    strike with |m| >= th on the given wing, implied tail mass vs realized frequency."""
    p = pt.join(sn.select("snap_id", "group", "split", "event_ticker"), on="snap_id").filter(pl.col("y_gt").is_not_null())
    rows = []
    for (grp, sp), g in p.group_by("group", "split"):
        for th in (1.0, 1.5, 2.0, 2.5):
            for wing in ("upper", "lower"):
                x = g.filter(pl.col("m") >= th) if wing == "upper" else g.filter(pl.col("m") <= -th)
                if x.height < 20:
                    continue
                imp = x["S"].to_numpy() if wing == "upper" else 1 - x["S"].to_numpy()
                real = x["y_gt"].to_numpy() if wing == "upper" else 1 - x["y_gt"].to_numpy()
                mu, t, n, ne = cmean(real - imp, x["event_ticker"].to_numpy())
                rows.append({"group": grp, "split": sp, "wing": wing, "m_ge": th, "points": n, "events": ne,
                             "z_h0": _z_h0(real, imp, x["event_ticker"].to_numpy()),
                             "implied_c": round(float(imp.mean()) * 100, 2), "realized_c": round(float(real.mean()) * 100, 2),
                             "gauss_c": round(float(ndtr(-np.abs(x["m"].to_numpy())).mean()) * 100, 2),
                             "real_minus_imp_c": round(mu * 100, 2), "t": None if t is None else round(t, 2)})
    return pl.DataFrame(rows).sort("group", "split", "wing", "m_ge") if rows else pl.DataFrame()


def pit_table(sn: pl.DataFrame, by_lead: bool = False) -> pl.DataFrame:
    """Fractional PIT histogram (10 bins) of the realized interval under the implied CDF, with the
    outer-bin share (expected 0.2) and its t clustered by event."""
    lo, hi = sn["pit_lo"].to_numpy(), sn["pit_hi"].to_numpy()
    edges = np.linspace(0, 1, 11)
    w = hi - lo
    mass = np.zeros((len(lo), 10))
    for j in range(10):
        ov = np.clip(np.minimum(hi, edges[j + 1]) - np.maximum(lo, edges[j]), 0, None)
        mass[:, j] = np.where(w > 1e-9, ov / np.where(w > 1e-9, w, 1), ((lo >= edges[j]) & (lo < edges[j + 1])) | ((j == 9) & (lo >= 1)))
    df = sn.select("group", "lead_h", "split", "event_ticker").with_columns(
        *[pl.Series(f"b{j}", mass[:, j]) for j in range(10)])
    keys = ["group", "lead_h", "split"] if by_lead else ["group", "split"]
    rows = []
    for k, g in df.group_by(keys):
        if g.height < 20:
            continue
        M = g.select([f"b{j}" for j in range(10)]).to_numpy()
        outer = M[:, 0] + M[:, 9]
        mu, t, n, ne = cmean(outer - 0.2, g["event_ticker"].to_numpy())
        mid = M[:, 3:7].sum(1)
        mu2, t2, _, _ = cmean(mid - 0.4, g["event_ticker"].to_numpy())
        rows.append({**dict(zip(keys, k)), "snapshots": n, "events": ne,
                     **{f"pit{j}": round(float(M[:, j].mean()), 3) for j in range(10)},
                     "outer_share": round(mu + 0.2, 3), "t_outer_vs_0.2": None if t is None else round(t, 2),
                     "center40_share": round(mu2 + 0.4, 3), "t_center_vs_0.4": None if t2 is None else round(t2, 2)})
    return pl.DataFrame(rows).sort(keys) if rows else pl.DataFrame()


def width_table(sn: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (grp, lead, sp), g in sn.group_by("group", "lead_h", "split"):
        r = width_mle(g["z_lo"].to_numpy(), g["z_hi"].to_numpy(), g["event_ticker"].to_numpy())
        rows.append({"group": grp, "lead_h": lead, "split": sp, **r})
    for (grp, sp), g in sn.group_by("group", "split"):
        r = width_mle(g["z_lo"].to_numpy(), g["z_hi"].to_numpy(), g["event_ticker"].to_numpy())
        rows.append({"group": grp, "lead_h": -1, "split": sp, **r})
    return pl.DataFrame(rows, infer_schema_length=None).sort("group", "split", "lead_h")


def term_structure(sn: pl.DataFrame) -> pl.DataFrame:
    """Within-ladder slope of log sigma_ATM on log lead (sqrt-time diffusion: 0.5), t by event.
    Sports use the pre-game leads only (the 1 h snapshot is in play)."""
    rows = []
    x = sn.filter((pl.col("sigma_atm") > 0) & ~(pl.col("group").str.starts_with("sp_") & (pl.col("lead_h") == 1)))
    x = x.with_columns(pl.col("sigma_atm").log().alias("ls"),
                                                       pl.col("lead_h").cast(pl.Float64).log().alias("ll"))
    x = x.filter(pl.col("ladder").count().over("ladder") >= 2)
    x = x.with_columns((pl.col("ls") - pl.col("ls").mean().over("ladder")).alias("dls"),
                       (pl.col("ll") - pl.col("ll").mean().over("ladder")).alias("dll"))
    for (grp,), g in x.group_by("group"):
        if g.height < 30:
            continue
        b, t = ols_cluster(g["dll"].to_numpy()[:, None], g["dls"].to_numpy(), g["event_ticker"].to_numpy())
        se = b[0] / t[0] if t[0] != 0 else float("nan")
        rows.append({"group": grp, "snapshots": g.height, "ladders": g["ladder"].n_unique(),
                     "slope_log_sigma_on_log_lead": round(float(b[0]), 3),
                     "ci95": f"[{b[0] - 1.96 * abs(se):.3f}, {b[0] + 1.96 * abs(se):.3f}]", "t_slope_vs_0": round(float(t[0]), 2),
                     "t_slope_vs_0.5": round(float((b[0] - 0.5) / se), 2) if se > 0 else None,
                     **{f"sigma_med_{int(h)}h": round(float(g.filter(pl.col("lead_h") == h)["sigma_atm"].median()), 3)
                        for h in sorted(g["lead_h"].unique().to_list())}})
    return pl.DataFrame(rows, infer_schema_length=None).sort("group") if rows else pl.DataFrame()


# ---- implied density shape: Jones-Faddy skew-t ----------------------------------------------------
def jf_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """CDF of the Jones-Faddy skew-t: F(x) = I_y(a, b), y = (1 + x / sqrt(a + b + x^2)) / 2. The left
    tail decays like a Student-t with 2a degrees of freedom, the right tail like one with 2b; a = b
    is the Student-t with 2a d.f. (scaled), a = b -> inf the normal."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    xf = np.where(fin, x, 0.0)
    y = (1 + xf / np.sqrt(a + b + xf * xf)) / 2
    return np.where(fin, betainc(a, b, np.clip(y, 0, 1)), np.where(x > 0, 1.0, 0.0))


def _jf_unpack(th: np.ndarray) -> tuple[float, float, float, float]:
    return float(th[0]), math.exp(th[1]), math.exp(np.clip(th[2], -3, 6)), math.exp(np.clip(th[3], -3, 6))


def jf_quantile_shape(a: float, b: float) -> dict[str, float]:
    """Unit-free shape summaries of a JF skew-t: Bowley skewness (q90 + q10 - 2 q50)/(q90 - q10) and
    tail weight (q97.5 - q2.5)/(q75 - q25) / 2.906 (1 for a normal)."""
    from scipy.optimize import brentq

    def q(p: float) -> float:
        return brentq(lambda x: float(jf_cdf(np.array([x]), a, b)[0]) - p, -1e4, 1e4, xtol=1e-9)

    q10, q25, q50, q75, q90, q025, q975 = (q(p) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.025, 0.975))
    return {"bowley": (q90 + q10 - 2 * q50) / (q90 - q10), "tailw": (q975 - q025) / (q75 - q25) / 2.9058}


def fit_jf_implied(m: np.ndarray, S: np.ndarray) -> np.ndarray | None:
    """Least squares in logit space of S(m) = 1 - JF((m - mu)/s; a, b) over pooled standardized
    survival points (S clipped to [1%, 99%], so tick-floored wings carry bounded weight)."""
    from scipy.optimize import least_squares

    Sc = np.clip(S, 0.01, 0.99)
    target = np.log(Sc / (1 - Sc))

    def res(th: np.ndarray) -> np.ndarray:
        mu, s, a, b = _jf_unpack(th)
        p = np.clip(1 - jf_cdf((m - mu) / s, a, b), 0.01, 0.99)
        return np.log(p / (1 - p)) - target

    th0 = np.array([0.0, math.log(1.2), math.log(3.0), math.log(3.0)])
    try:
        r = least_squares(res, th0, method="trf", max_nfev=200)
    except (ValueError, FloatingPointError):
        return None
    return r.x if r.success else None


def fit_jf_realized(zl: np.ndarray, zh: np.ndarray) -> np.ndarray | None:
    """Interval-censored MLE of the realized standardized outcome (X_T - F)/sigma_ATM under a JF
    skew-t."""
    def nll(th: np.ndarray) -> float:
        mu, s, a, b = _jf_unpack(th)
        p = jf_cdf((zh - mu) / s, a, b) - jf_cdf((zl - mu) / s, a, b)
        return -float(np.log(np.clip(p, 1e-12, None)).sum())

    r = minimize(nll, np.array([0.0, math.log(1.2), math.log(3.0), math.log(3.0)]), method="Nelder-Mead",
                 options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6})
    return r.x if np.all(np.isfinite(r.x)) else None


def jf_shapes(sn: pl.DataFrame, pt: pl.DataFrame, boot: int = 30, seed: int = 0) -> pl.DataFrame:
    """Implied (pooled survival points) and realized (interval-censored outcomes) JF skew-t shape per
    group x split (all leads) and per group x lead (both splits), in standardized units
    (K - F)/sigma_ATM: tail d.f. 2a (left) and 2b (right), Bowley skewness and tail weight, with
    cluster-bootstrap (by event) 90% intervals for the shape summaries."""
    rng = np.random.default_rng(seed)
    p = pt.join(sn.select("snap_id", "group", "lead_h", "split", "event_ticker"), on="snap_id")
    p = p.filter((pl.col("S") > S_CLIP) & (pl.col("S") < 1 - S_CLIP) & (pl.col("m").abs() <= 6))
    rows = []
    cells: list[tuple[dict, pl.DataFrame, pl.DataFrame]] = []
    for (grp, sp), g in p.group_by("group", "split"):
        cells.append(({"group": grp, "lead_h": -1, "split": sp}, g,
                      sn.filter((pl.col("group") == grp) & (pl.col("split") == sp))))
    for (grp, lead), g in p.group_by("group", "lead_h"):
        cells.append(({"group": grp, "lead_h": lead, "split": "both"}, g,
                      sn.filter((pl.col("group") == grp) & (pl.col("lead_h") == lead))))
    for key, g, s_ in cells:
        if g["event_ticker"].n_unique() < 30:
            continue
        row: dict[str, Any] = {**key, "points": g.height, "events": g["event_ticker"].n_unique()}
        th = fit_jf_implied(g["m"].to_numpy(), g["S"].to_numpy())
        zl, zh = s_["z_lo"].to_numpy(), s_["z_hi"].to_numpy()
        okz = (zh > zl) & ~(np.isnan(zl) | np.isnan(zh))
        tr = fit_jf_realized(zl[okz], zh[okz]) if okz.sum() >= 50 else None
        for tag, t_ in (("imp", th), ("real", tr)):
            if t_ is None:
                continue
            mu, s, a, b = _jf_unpack(t_)
            shp = jf_quantile_shape(a, b)
            row.update({f"{tag}_mu": round(mu, 3), f"{tag}_scale": round(s, 3), f"{tag}_df_left": round(2 * a, 2),
                        f"{tag}_df_right": round(2 * b, 2), f"{tag}_bowley": round(shp["bowley"], 3),
                        f"{tag}_tailw": round(shp["tailw"], 3)})
        if boot and th is not None and key["lead_h"] == -1:
            ev = g["event_ticker"].unique().to_numpy()
            gi = {e: i for i, e in enumerate(ev)}
            idx = np.array([gi[e] for e in g["event_ticker"].to_list()])
            order = np.argsort(idx, kind="stable")
            bounds = np.searchsorted(idx[order], np.arange(len(ev) + 1))
            m_, S_ = g["m"].to_numpy()[order], g["S"].to_numpy()[order]
            bs, tw = [], []
            for _ in range(boot):
                pick = rng.integers(0, len(ev), len(ev))
                sel = np.concatenate([np.arange(bounds[k], bounds[k + 1]) for k in pick])
                tb = fit_jf_implied(m_[sel], S_[sel])
                if tb is None:
                    continue
                _, _, a, b = _jf_unpack(tb)
                shp = jf_quantile_shape(a, b)
                bs.append(shp["bowley"]), tw.append(shp["tailw"])
            if len(bs) >= 10:
                row["imp_bowley_90ci"] = f"[{np.quantile(bs, 0.05):.3f}, {np.quantile(bs, 0.95):.3f}]"
                row["imp_tailw_90ci"] = f"[{np.quantile(tw, 0.05):.3f}, {np.quantile(tw, 0.95):.3f}]"
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort("group", "split", "lead_h") if rows else pl.DataFrame()


# ---- smile dynamics between consecutive snapshots ------------------------------------------------
def snapshot_pairs(sn: pl.DataFrame) -> pl.DataFrame:
    """Consecutive snapshots of one ladder (longer lead t -> next shorter lead u) in raw (un-flipped)
    coordinates: F, sigma_ATM, skew s, curvature c at both."""
    x = sn.with_columns((pl.col("F") * pl.col("orient")).alias("Fr"), (pl.col("s") * pl.col("orient")).alias("sr"))
    x = x.sort("ladder", "lead_h", descending=[False, True])
    nxt = {c: pl.col(c).shift(-1).over("ladder") for c in ("lead_h", "Fr", "sigma_atm", "sr", "c", "snap_id", "orient")}
    x = x.with_columns(**{f"{c}_u": e for c, e in nxt.items()}).filter(pl.col("lead_h_u").is_not_null())
    return x.with_columns((pl.col("lead_h").cast(pl.String) + "->" + pl.col("lead_h_u").cast(pl.String)).alias("pair"),
                          ((pl.col("Fr_u") - pl.col("Fr")) / pl.col("sigma_atm")).alias("dF_n"),
                          ((pl.col("sigma_atm_u") - pl.col("sigma_atm")) / pl.col("sigma_atm")).alias("dsig_n"))


def _fe_ols(df: pl.DataFrame, y: str, xs: list[str], fe: str = "fe") -> tuple[np.ndarray, np.ndarray, int]:
    """OLS of y on xs after demeaning everything within ``fe`` cells; t clustered by event."""
    d = df.drop_nulls([y, *xs]).filter(pl.all_horizontal([pl.col(c).is_finite() for c in (y, *xs)]))
    d = d.with_columns([(pl.col(c) - pl.col(c).mean().over(fe)).alias(f"_{c}") for c in (y, *xs)])
    X = d.select([f"_{c}" for c in xs]).to_numpy()
    b, t = ols_cluster(X, d[f"_{y}"].to_numpy(), d["event_ticker"].to_numpy())
    return b, t, d.height


def smile_dynamics(sn: pl.DataFrame, pt: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(atm, fixed_strike) regressions per group (fixed effects per lead pair; standardized by the
    earlier sigma_ATM; t clustered by event).

    ATM: dsigma_ATM/sigma = delta * s * dF/sigma + kappa * dF/sigma: sticky strike (the smile
    stays put in strike, the ATM point slides along it) predicts delta = 1, sticky moneyness (the
    smile moves with F) delta = 0; kappa is a level backbone. For positive underlyings the
    backbone exponent beta = d log sigma_ATM / d log F (0: normal, 0.5: Poisson-like, 1:
    lognormal). Fixed strike: dsigma_K/sigma = gamma * [-(s + 2 c m_K) dF/sigma] + v_K *
    dsigma_ATM/sigma * phi: sticky moneyness predicts gamma = 1, sticky strike gamma = 0."""
    pr = snapshot_pairs(sn).with_columns((pl.col("group") + "|" + pl.col("pair")).alias("fe"))
    pr = pr.filter(pl.col("dF_n").abs() < 5)
    # pooled smile per group x lead (oriented coordinates): v - 1 = a + s m + c m^2, so v'(m) = s + 2 c m;
    # used as an errors-in-variables-free alternative to the noisy per-snapshot (s, c)
    q = pt.join(sn.select("snap_id", "group", "lead_h", "event_ticker"), on="snap_id").filter(
        pl.col("v").is_not_null() & ~pl.col("atm_pair") & (pl.col("m").abs() <= 3.5))
    pool: dict[tuple[str, int], tuple[float, float]] = {}
    for (grp, lead), g in q.group_by("group", "lead_h"):
        if g.height >= 30:
            m_ = g["m"].to_numpy()
            b_, _ = ols_cluster(np.column_stack([np.ones_like(m_), m_, m_**2]), g["v"].to_numpy() - 1, g["event_ticker"].to_numpy())
            pool[(grp, int(lead))] = (float(b_[1]), float(b_[2]))
    sp_ = pl.DataFrame([{"group": k[0], "lead_h": k[1], "s_pool": v[0], "c_pool": v[1]} for k, v in pool.items()],
                       schema={"group": pl.String, "lead_h": pl.Int64, "s_pool": pl.Float64, "c_pool": pl.Float64})
    pr = pr.with_columns(pl.col("lead_h").cast(pl.Int64)).join(sp_, on=["group", "lead_h"], how="left").with_columns(
        (pl.col("s_pool") * pl.col("orient")).alias("sr_pool"))
    atm_rows, k_rows = [], []
    for (grp,), g in pr.group_by("group"):
        if g.height < 50:
            continue
        gs = g.filter(pl.col("sr").is_not_nan() & pl.col("sr").is_not_null()).with_columns(
            (pl.col("sr") * pl.col("dF_n")).alias("x_strike"))
        row: dict[str, Any] = {"group": grp, "pairs": g.height, "events": g["event_ticker"].n_unique(),
                               "mean_abs_dF_n": round(float(g["dF_n"].abs().mean()), 3),
                               "mean_s": round(float(gs["sr"].abs().mean()), 3) if gs.height else None}
        # plain slope vs the sticky-strike prediction (the pooled smile skew in raw coordinates; for
        # spreads raw skew changes sign with the favourite, so compare within the oriented frame)
        go = g.with_columns((pl.col("dF_n") * pl.col("orient")).alias("dF_o"))
        b0, t0, _ = _fe_ols(go, "dsig_n", ["dF_o"])
        row.update({"slope_dsig_on_dF": round(float(b0[0]), 3), "t_slope": round(float(t0[0]), 2),
                    "sticky_strike_pred_s": round(float(g["s_pool"].mean()), 3) if g["s_pool"].drop_nulls().len() else None})
        if gs.height >= 50:
            b, t, n = _fe_ols(gs, "dsig_n", ["x_strike", "dF_n"])
            row.update({"n_atm": n, "delta_sticky_strike": round(float(b[0]), 3), "t_delta": round(float(t[0]), 2),
                        "t_delta_vs_1": round(float((b[0] - 1) / (b[0] / t[0])), 2) if t[0] != 0 else None,
                        "kappa_level": round(float(b[1]), 3), "t_kappa": round(float(t[1]), 2)})
        pos = g.filter((pl.col("Fr") > 0) & (pl.col("Fr_u") > 0)).with_columns(
            (pl.col("sigma_atm_u") / pl.col("sigma_atm")).log().alias("dls"), (pl.col("Fr_u") / pl.col("Fr")).log().alias("dlF"))
        if pos.height >= 50 and not grp.startswith(("wx", "sp_spread")):
            b, t, n = _fe_ols(pos, "dls", ["dlF"])
            row.update({"n_backbone": n, "beta_backbone": round(float(b[0]), 3), "t_beta": round(float(t[0]), 2)})
            # cross-sectional level relation: log sigma on log F within series x lead
            cs = sn.filter((pl.col("group") == grp) & (pl.col("F") > 0)).with_columns(
                pl.col("sigma_atm").log().alias("ls"), pl.col("F").log().alias("lF"),
                (pl.col("series_ticker") + "|" + pl.col("lead_h").cast(pl.String)).alias("fe"))
            if cs.height >= 50:
                b2, t2, _ = _fe_ols(cs, "ls", ["lF"])
                row.update({"beta_cross_section": round(float(b2[0]), 3), "t_beta_cs": round(float(t2[0]), 2)})
        atm_rows.append(row)
    # fixed strike: points present at both snapshots
    p = pt.select("snap_id", "K", "m", "sigma_K", "v").filter(pl.col("sigma_K").is_not_null())
    o = sn.select("snap_id", "orient")
    p = p.join(o, on="snap_id").with_columns((pl.col("K") * pl.col("orient")).alias("Kr"),
                                              (pl.col("m") * pl.col("orient")).alias("mr")).drop("orient", "K", "m")
    pr2 = pr.filter(pl.col("sr").is_not_nan() & pl.col("c").is_not_nan() & pl.col("sr_pool").is_not_null())
    j = (pr2.select("snap_id", "snap_id_u", "event_ticker", "group", "fe", "sr", "c", "sr_pool", "c_pool", "dF_n",
                    "dsig_n", "sigma_atm")
         .join(p, on="snap_id")
         .join(p.select(pl.col("snap_id").alias("snap_id_u"), "Kr", pl.col("sigma_K").alias("sigma_K_u")), on=["snap_id_u", "Kr"]))
    j = j.with_columns(((pl.col("sigma_K_u") - pl.col("sigma_K")) / pl.col("sigma_atm")).alias("dsk_n"),
                       (-(pl.col("sr") + 2 * pl.col("c") * pl.col("mr")) * pl.col("dF_n")).alias("x_sm"),
                       (-(pl.col("sr_pool") + 2 * pl.col("c_pool") * pl.col("mr")) * pl.col("dF_n")).alias("x_sm_pool"),
                       (pl.col("v") * pl.col("dsig_n")).alias("x_lvl"))
    for (grp,), g in j.group_by("group"):
        if g.height < 100:
            continue
        b, t, n = _fe_ols(g, "dsk_n", ["x_sm", "x_lvl"])
        b1, t1, _ = _fe_ols(g, "dsk_n", ["dF_n", "x_lvl"])
        bp, tp, _ = _fe_ols(g, "dsk_n", ["x_sm_pool", "x_lvl"])
        k_rows.append({"group": grp, "strike_pairs": n, "events": g["event_ticker"].n_unique(),
                       "gamma_pooled_smile": round(float(bp[0]), 3), "t_gamma_pooled": round(float(tp[0]), 2),
                       "t_gamma_pooled_vs_1": round(float((bp[0] - 1) / (bp[0] / tp[0])), 2) if tp[0] != 0 else None,
                       "gamma_snapshot_smile": round(float(b[0]), 3), "t_gamma": round(float(t[0]), 2),
                       "t_gamma_vs_1": round(float((b[0] - 1) / (b[0] / t[0])), 2) if t[0] != 0 else None,
                       "phi_level": round(float(b[1]), 3), "t_phi": round(float(t[1]), 2),
                       "dsigK_on_dF": round(float(b1[0]), 3), "t_dsigK_on_dF": round(float(t1[0]), 2)})
    atm = pl.DataFrame(atm_rows, infer_schema_length=None).sort("group") if atm_rows else pl.DataFrame()
    fk = pl.DataFrame(k_rows, infer_schema_length=None).sort("group") if k_rows else pl.DataFrame()
    return atm, fk


def variance_between(sn: pl.DataFrame, boot: int = 200, seed: int = 1) -> pl.DataFrame:
    """Realized vs implied variance between consecutive snapshots (and from the last snapshot to
    settlement), standardized by the earlier sigma_ATM^2: realized (F_u - F_t)^2, implied drop
    sigma_t^2 - sigma_u^2. Under a martingale median with sigma^2 the conditional variance their
    means agree (law of total variance), so the ratio RV/IV is a variance-risk-premium analogue
    (< 1: implied variance falls faster than F moves). Cluster-bootstrap 90% CI by event; also the
    mean drift dF/sigma."""
    rng = np.random.default_rng(seed)
    pr = snapshot_pairs(sn).filter(pl.col("dF_n").abs() < 10)
    pr = pr.with_columns((pl.col("dF_n") ** 2).alias("rv"), (1 - (pl.col("sigma_atm_u") / pl.col("sigma_atm")) ** 2).alias("iv"))
    rows = []
    for (grp, pair), g in pr.group_by("group", "pair"):
        if g["event_ticker"].n_unique() < 30:
            continue
        ev = g["event_ticker"].to_numpy()
        u, inv = np.unique(ev, return_inverse=True)
        rv_e = np.bincount(inv, weights=g["rv"].to_numpy())
        iv_e = np.bincount(inv, weights=g["iv"].to_numpy())
        n_e = np.bincount(inv).astype(float)
        ratio = rv_e.sum() / iv_e.sum() if iv_e.sum() > 0 else float("nan")
        bs = []
        for _ in range(boot):
            k = rng.integers(0, len(u), len(u))
            den = iv_e[k].sum()
            if den > 0:
                bs.append(rv_e[k].sum() / den)
        mu, t, _, _ = cmean(g["dF_n"].to_numpy(), ev)
        rows.append({"group": grp, "pair": pair, "lead_t": int(g["lead_h"][0]), "pairs": g.height, "events": len(u),
                     "rv_mean": round(rv_e.sum() / n_e.sum(), 4), "iv_mean": round(iv_e.sum() / n_e.sum(), 4),
                     "rv_over_iv": round(ratio, 3),
                     "ci90": f"[{np.quantile(bs, 0.05):.3f}, {np.quantile(bs, 0.95):.3f}]" if bs else None,
                     "drift_dF_n": round(mu, 3) if mu is not None else None, "t_drift": None if t is None else round(t, 2)})
    return pl.DataFrame(rows).sort("group", "lead_t", descending=[False, True]) if rows else pl.DataFrame()


# ---- economic tests ------------------------------------------------------------------------------
M_GRID = (0.5, 1.0, 1.5, 2.0, 2.5)
X_GRID = (0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)
THETAS = (0.0, 0.01, 0.02, 0.03, 0.05)
P_BUCKETS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.501)
TM_BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, INF)


def leg_features(sn: pl.DataFrame, lg: pl.DataFrame, lad: pl.DataFrame) -> pl.DataFrame:
    """Per (snapshot, quoted leg): the tail side (the side whose event lies beyond F), its model
    moneyness ``tail_m``, its mid price ``p_tail``, the settled tail outcome, and the PnL of selling
    the tail (buying the other side as a taker) and of buying it, with the series' taker fee."""
    x = (lg.filter("ok").join(sn.select("snap_id", "group", "lead_h", "split", "event_ticker", "series_ticker"),
                              on="snap_id")
         .join(lad.select("ticker", "fee_mult", "fee_type").unique("ticker"), on="ticker", how="left"))
    gt = pl.col("hi").is_infinite() & pl.col("lo").is_finite()
    lt = pl.col("lo").is_infinite() & pl.col("hi").is_finite()
    x = x.with_columns(
        pl.when(gt).then(pl.col("m_lo") > 0).when(lt).then(pl.col("m_hi") < 0)
        .otherwise((pl.col("m_lo") >= 0) | (pl.col("m_hi") <= 0)).alias("tail_yes"),
        pl.when(gt).then(pl.col("m_lo").abs()).when(lt).then(pl.col("m_hi").abs())
        .when(pl.col("m_lo") >= 0).then(pl.col("m_lo")).when(pl.col("m_hi") <= 0).then(-pl.col("m_hi"))
        .otherwise(None).alias("tail_m"),
        pl.when(gt).then(pl.when(pl.col("m_lo") > 0).then(pl.lit("upper")).otherwise(pl.lit("lower")))
        .when(lt).then(pl.when(pl.col("m_hi") < 0).then(pl.lit("lower")).otherwise(pl.lit("upper")))
        .when(pl.col("m_lo") >= 0).then(pl.lit("upper")).when(pl.col("m_hi") <= 0).then(pl.lit("lower"))
        .otherwise(pl.lit("body")).alias("wing"),
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    x = x.with_columns(
        pl.when(pl.col("tail_yes")).then(pl.col("mid")).otherwise(1 - pl.col("mid")).alias("p_tail"),
        pl.when(pl.col("tail_yes")).then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Float64).alias("y_tail"),
        # cost of the body side (sell the tail) and of the tail side (buy the tail)
        pl.when(pl.col("tail_yes")).then(1 - pl.col("bid")).otherwise(pl.col("ask")).alias("c_body"),
        pl.when(pl.col("tail_yes")).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("c_tail"))
    mult = x["fee_mult"].fill_null(1.0).to_numpy()
    cb, ct = x["c_body"].to_numpy(), x["c_tail"].to_numpy()
    yt = x["y_tail"].to_numpy()
    sell = np.where(cb <= 0.99 + 1e-9, (1 - yt) - cb - taker_fee_v(cb, mult), np.nan)
    buy = np.where(ct <= 0.99 + 1e-9, yt - ct - taker_fee_v(ct, mult), np.nan)
    # body brackets (contain F): buy the YES at the ask
    a = x["ask"].to_numpy()
    body = np.where((x["wing"] == "body").to_numpy() & (a <= 0.99 + 1e-9),
                    x["y"].to_numpy() - a - taker_fee_v(a, mult), np.nan)
    x = x.with_columns(pl.Series("pnl_sell_tail", sell).fill_nan(None), pl.Series("pnl_buy_tail", buy).fill_nan(None),
                       pl.Series("pnl_buy_body", body).fill_nan(None),
                       pl.Series("fee_body", taker_fee_v(cb, mult)), pl.Series("fee_tail", taker_fee_v(ct, mult)))
    return x.with_columns(pl.when(pl.col("wing") == "body").then(None).otherwise(pl.col("pnl_sell_tail")).alias("pnl_sell_tail"),
                          pl.when(pl.col("wing") == "body").then(None).otherwise(pl.col("pnl_buy_tail")).alias("pnl_buy_tail"))


def body_rules(feat: pl.DataFrame) -> pl.DataFrame:
    """Buy the YES of the bracket that contains the implied median (bracket ladders), per group x
    lead: profitable if the implied distribution is too wide."""
    b = (feat.filter(pl.col("pnl_buy_body").is_not_null()).rename({"pnl_buy_body": "pnl"})
         .with_columns(pl.col("ask").alias("cost")))
    rows = []
    for (grp, lead), g in b.group_by("group", "lead_h"):
        row = {"group": grp, "lead_h": lead, "mean_ask_c": round(float(g["ask"].mean()) * 100, 2)}
        for sp in ("dev", "holdout"):
            row.update({f"{sp}_{k}": v for k, v in _stat(g.filter(pl.col("split") == sp)).items()})
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort("group", "lead_h") if rows else pl.DataFrame()


def _stat(df: pl.DataFrame, col: str = "pnl") -> dict[str, Any]:
    """n, events, mean PnL (cents per contract), t clustered by event, and PnL per dollar risked
    (mean PnL / mean entry cost incl. fee) when a ``cost`` column is present."""
    if df.is_empty():
        return {"n": 0}
    mu, t, n, k = cmean(df[col].to_numpy(), df["event_ticker"].to_numpy())
    out = {"n": n, "events": k, "pnl_c": None if mu is None else round(mu * 100, 3),
           "t": None if t is None else round(t, 2)}
    if "cost" in df.columns and mu is not None:
        out["per_dollar"] = round(mu / float(df["cost"].mean()), 4)
    if "p_risk" in df.columns and mu is not None:
        out["t_risk"] = _t_risk(df, mu)
    return out


def _t_risk(df: pl.DataFrame, mu: float) -> float | None:
    """t of the mean PnL against payoff risk: each contract's settlement sd sqrt(p(1-p)) at the
    traded tail's market price p, summed within an event (perfect correlation), events
    independent. Unlike the clustered t it does not collapse when no tail event happened."""
    p = np.clip(df["p_risk"].to_numpy().astype(float), 0, 1)
    sd = np.sqrt(p * (1 - p))
    _, inv = np.unique(df["event_ticker"].to_numpy(), return_inverse=True)
    se = math.sqrt(float((np.bincount(inv, weights=sd) ** 2).sum())) / len(p)
    return round(mu / se, 2) if se > 0 else None


def gated_rules(feat: pl.DataFrame, groups: list[str], min_n: int = 100, min_events: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sell-the-tail rules, parameters chosen on dev per group (and pooled): ``model`` (tail_m >= m*),
    ``price`` (tail-side mid <= x), ``price_yes`` (YES mid <= x, buy NO: the naive "sell cheap
    YES"), ``all`` (every wing leg). Lead set: all leads or one lead. Returns (grid, chosen)."""
    f = (feat.filter(pl.col("pnl_sell_tail").is_not_null()).rename({"pnl_sell_tail": "pnl"})
         .with_columns((pl.col("c_body") + pl.col("fee_body")).alias("cost"), pl.col("p_tail").alias("p_risk")))
    grid_rows, chosen = [], []
    for grp in [*groups, "ALL"]:
        fg = f if grp == "ALL" else f.filter(pl.col("group") == grp)
        if fg.is_empty():
            continue
        leads = [None, *sorted(fg["lead_h"].unique().to_list())]
        rules: list[tuple[str, float | None, pl.Expr]] = [("all", None, pl.lit(True))]
        rules += [("model", m, pl.col("tail_m") >= m) for m in M_GRID]
        rules += [("price", xx, pl.col("p_tail") <= xx + 1e-9) for xx in X_GRID]
        rules += [("price_yes", xx, pl.col("tail_yes") & (pl.col("mid") <= xx + 1e-9)) for xx in X_GRID]
        best: dict[str, tuple] = {}
        for name, par, cond in rules:
            for lead in leads:
                sub = fg.filter(cond) if lead is None else fg.filter(cond & (pl.col("lead_h") == lead))
                row = {"group": grp, "rule": name, "param": par, "lead_h": lead if lead is not None else -1}
                for sp in ("dev", "holdout"):
                    st = _stat(sub.filter(pl.col("split") == sp))
                    row.update({f"{sp}_{k}": v for k, v in st.items()})
                grid_rows.append(row)
                if (row.get("dev_n", 0) >= min_n and (row.get("dev_events") or 0) >= min_events
                        and row.get("dev_pnl_c") is not None
                        and (name not in best or row["dev_pnl_c"] > best[name]["dev_pnl_c"])):
                    best[name] = row
        if "model" in best:
            # price control matched to the chosen model rule: same lead, closest dev trade count
            mb = best["model"]
            cands = [r for r in grid_rows if r["group"] == grp and r["rule"] == "price" and r["lead_h"] == mb["lead_h"]
                     and r.get("dev_n")]
            if cands:
                pm = dict(min(cands, key=lambda r: abs(r["dev_n"] - mb["dev_n"])))
                pm["rule"] = "price_matched"
                best["price_matched"] = pm
        chosen.extend(best.values())
    return pl.DataFrame(grid_rows, infer_schema_length=None), pl.DataFrame(chosen, infer_schema_length=None)


def pooled_fixed(feat: pl.DataFrame) -> pl.DataFrame:
    """Pre-specified sell-the-tail rules pooled over every group and lead (no tuning): model
    moneyness |m| >= 1.5 / 2 / 2.5 against the tail-price controls <= 2 / 3 / 5 / 10c and every
    wing leg, plus the model rule restricted to the price control's universe (both conditions)."""
    f = (feat.filter(pl.col("pnl_sell_tail").is_not_null() & (pl.col("wing") != "body")).rename({"pnl_sell_tail": "pnl"})
         .with_columns((pl.col("c_body") + pl.col("fee_body")).alias("cost"), pl.col("p_tail").alias("p_risk")))
    rules = [("all", pl.lit(True))]
    rules += [(f"model m>={m}", pl.col("tail_m") >= m) for m in (1.5, 2.0, 2.5)]
    rules += [(f"price p<={int(x * 100)}c", pl.col("p_tail") <= x + 1e-9) for x in (0.02, 0.03, 0.05, 0.10)]
    rules += [(f"price p<=5c & m<{m}", (pl.col("p_tail") <= 0.05 + 1e-9) & (pl.col("tail_m") < m)) for m in (2.0,)]
    rules += [(f"price p<=5c & m>={m}", (pl.col("p_tail") <= 0.05 + 1e-9) & (pl.col("tail_m") >= m)) for m in (2.0,)]
    rows = []
    for scope, ff in (("all groups", f), ("excl. weather", f.filter(~pl.col("group").str.starts_with("wx"))),
                      ("weather", f.filter(pl.col("group").str.starts_with("wx")))):
        for name, cond in rules:
            sub = ff.filter(cond)
            row = {"scope": scope, "rule": name, "mean_p_tail_c": round(float(sub["p_tail"].mean() or 0) * 100, 2)}
            for sp in ("dev", "holdout"):
                st = _stat(sub.filter(pl.col("split") == sp))
                row.update({f"{sp}_{k}": v for k, v in st.items()})
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def calibration_maps(feat: pl.DataFrame, groups: list[str], min_n: int = 100) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Does model moneyness price the tail better than the tail's own price? Per group, fit on dev
    logit P(tail) = a + b logit(p_tail) (``price``) and + c1 tail_m + c2 tail_m^2 + lead dummies
    (``model``); report the coefficients (t by event) and holdout Brier / log loss of the market
    mid and both maps, then trade both maps' fair values (sell the tail when the body side clears
    fee + theta, buy it when the tail does; theta chosen on dev) and report holdout PnL."""
    from albert.research.fundamental import fit_logistic

    coef_rows, trade_rows = [], []
    feat = feat.filter(pl.col("wing") != "body")
    for grp in [*groups, "ALL"]:
        fg = feat if grp == "ALL" else feat.filter(pl.col("group") == grp)
        dev, hold = fg.filter(pl.col("split") == "dev"), fg.filter(pl.col("split") == "holdout")
        if dev.height < 200 or hold.height < 50:
            continue
        leads = sorted(fg["lead_h"].unique().to_list())

        def design(d: pl.DataFrame, kind: str) -> np.ndarray:
            cols = [np.ones(d.height), _logit(d["p_tail"].to_numpy())]
            if kind == "model":
                tm = np.clip(d["tail_m"].to_numpy(), 0, 5)
                cols += [tm, tm**2] + [(d["lead_h"] == h).cast(pl.Float64).to_numpy() for h in leads[1:]]
            return np.column_stack(cols)

        yd, yh = dev["y_tail"].to_numpy(), hold["y_tail"].to_numpy()
        fair: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        row: dict[str, Any] = {"group": grp, "dev_n": dev.height, "hold_n": hold.height,
                               "dev_events": dev["event_ticker"].n_unique(), "hold_events": hold["event_ticker"].n_unique()}
        pm = hold["p_tail"].to_numpy()
        row["brier_mid_hold"] = round(float(np.mean((pm - yh) ** 2)), 5)
        row["logloss_mid_hold"] = round(float(-np.mean(yh * np.log(np.clip(pm, .005, .995)) + (1 - yh) * np.log(np.clip(1 - pm, .005, .995)))), 5)
        for kind in ("price", "model"):
            beta, t = fit_logistic(design(dev, kind), yd, dev["event_ticker"].to_numpy())
            ph = 1 / (1 + np.exp(-design(hold, kind) @ beta))
            pd_ = 1 / (1 + np.exp(-design(dev, kind) @ beta))
            fair[kind] = (pd_, ph)
            row[f"{kind}_b_logit_p"] = round(float(beta[1]), 3)
            row[f"{kind}_t_b"] = round(float(t[1]), 2)
            row[f"{kind}_a"] = round(float(beta[0]), 3)
            if kind == "model":
                row["model_c1_tail_m"], row["model_t_c1"] = round(float(beta[2]), 3), round(float(t[2]), 2)
                row["model_c2_tail_m2"], row["model_t_c2"] = round(float(beta[3]), 3), round(float(t[3]), 2)
            row[f"brier_{kind}_hold"] = round(float(np.mean((ph - yh) ** 2)), 5)
            row[f"logloss_{kind}_hold"] = round(float(-np.mean(yh * np.log(np.clip(ph, 1e-6, 1)) + (1 - yh) * np.log(np.clip(1 - ph, 1e-6, 1)))), 5)
        coef_rows.append(row)
        for kind in ("price", "model"):
            best = None
            results = {}
            for th in THETAS:
                for sp, d, pf in (("dev", dev, fair[kind][0]), ("holdout", hold, fair[kind][1])):
                    cb, ct = d["c_body"].to_numpy(), d["c_tail"].to_numpy()
                    e_sell = (1 - pf) - cb - d["fee_body"].to_numpy()
                    e_buy = pf - ct - d["fee_tail"].to_numpy()
                    sell = (e_sell >= th) & (cb <= 0.99) & (e_sell >= e_buy)
                    buy = (e_buy >= th) & (ct <= 0.99) & ~sell
                    pnl = np.where(sell, d["pnl_sell_tail"].fill_null(np.nan).to_numpy(),
                                   np.where(buy, d["pnl_buy_tail"].fill_null(np.nan).to_numpy(), np.nan))
                    mu, t, n, k = cmean(pnl, d["event_ticker"].to_numpy())
                    results[(th, sp)] = {"n": n, "events": k, "pnl_c": None if mu is None else round(mu * 100, 3),
                                         "t": None if t is None else round(t, 2), "sell_share": round(float(sell.sum() / max(n, 1)), 3)}
                    if sp == "dev" and n >= min_n and mu is not None and (best is None or mu > best[1]):
                        best = (th, mu)
            for (th, sp), r in results.items():
                trade_rows.append({"group": grp, "map": kind, "theta": th, "split": sp, **r,
                                   "chosen": best is not None and th == best[0]})
    return pl.DataFrame(coef_rows, infer_schema_length=None), pl.DataFrame(trade_rows, infer_schema_length=None)


def arbitrage(sn: pl.DataFrame, lg: pl.DataFrame, lad: pl.DataFrame) -> pl.DataFrame:
    """Snapshot-quote no-arbitrage violations net of taker fees: survival monotonicity between
    threshold legs (buy S(K1) at its ask, sell S(K2 > K1) at its bid) and complete bracket sets
    (sum of asks < 1, sum of bids > 1). Legs with spread <= 5c only (the fitted universe; wider
    books are mostly stale). Quotes are hourly candle closes, not synchronous books."""
    x = lg.join(sn.select("snap_id", "group", "kind", "split", "n_legs"), on="snap_id").join(
        lad.select("ticker", "fee_mult").unique("ticker"), on="ticker", how="left")
    x = x.filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & pl.col("ok"))
    out = defaultdict(lambda: {"snapshots": 0, "violations": 0, "profit_c": []})
    for (sid,), g in x.group_by("snap_id"):
        grp, kind = g["group"][0], g["kind"][0]
        mult = float(g["fee_mult"].fill_null(1.0)[0])
        bid, ask = g["bid"].to_numpy(), g["ask"].to_numpy()
        lo, hi = g["lo"].to_numpy(), g["hi"].to_numpy()
        key = (grp, g["split"][0])
        out[key]["snapshots"] += 1
        best = 0.0
        if kind == "bracket":
            if g.height == g["n_legs"][0]:
                if (ask <= 0.99).all():
                    best = max(best, 1 - ask.sum() - taker_fee_v(ask, mult).sum())
                if (bid >= 0.01).all():
                    best = max(best, bid.sum() - 1 - taker_fee_v(1 - bid, mult).sum())
        else:
            K, sb, sa, cb, ca = [], [], [], [], []
            for i in range(len(bid)):
                if np.isinf(hi[i]):        # S(lo): buy = YES at ask, sell = NO at 1 - bid
                    K.append(lo[i]), sb.append(bid[i]), sa.append(ask[i]), ca.append(ask[i]), cb.append(1 - bid[i])
                elif np.isinf(lo[i]):      # S(hi) = 1 - P(X <= hi): buy = NO at 1 - bid, sell = YES at ask
                    K.append(hi[i]), sb.append(1 - ask[i]), sa.append(1 - bid[i]), ca.append(1 - bid[i]), cb.append(ask[i])
            K, sb, sa, cb, ca = map(np.array, (K, sb, sa, cb, ca))
            for i in range(len(K)):
                for j in range(len(K)):
                    if K[i] < K[j] and ca[i] <= 0.99 and cb[j] <= 0.99:
                        pr = sb[j] - sa[i] - taker_fee_v(np.array([ca[i]]), mult)[0] - taker_fee_v(np.array([cb[j]]), mult)[0]
                        best = max(best, pr)
        if best > 1e-9:
            out[key]["violations"] += 1
            out[key]["profit_c"].append(best * 100)
    rows = []
    for (grp, sp), r in sorted(out.items()):
        pc = r["profit_c"]
        rows.append({"group": grp, "split": sp, "snapshots": r["snapshots"], "violations": r["violations"],
                     "share": round(r["violations"] / max(r["snapshots"], 1), 4),
                     "median_profit_c": round(float(np.median(pc)), 2) if pc else None,
                     "max_profit_c": round(float(np.max(pc)), 2) if pc else None})
    return pl.DataFrame(rows)


FETCH_SPORTS = r"^KX(NFL|NBA|NHL|WNBA|MLB)(SPREAD|TOTAL|TEAMTOTAL)$"
NON_SPORT = ("wx_low", "entertainment", "politics", "scitech", "elections", "climate_other")


def step_fetch(s, max_markets: int | None = None) -> None:
    sel = pl.col("group").is_in(list(NON_SPORT)) | pl.col("series_ticker").str.contains(FETCH_SPORTS)
    hist = pl.col("group").is_in(list(NON_SPORT)) | pl.col("series_ticker").str.starts_with("KXNFL")
    lad = pl.read_parquet(OUT / "ladder_legs.parquet").filter(sel)
    snaps = snapshot_times(lad)
    extra = OUT / "candles_fetched.parquet"
    c = load_candles(s.data_dir, lad["ticker"].unique().to_list(), (extra,))
    q = quotes_at(lad, snaps, c).select("ladder", "ticker", "group", "series_ticker", "q_ts")
    del c
    fetch_missing(s, lad, q, extra, hist_select=hist, max_markets=max_markets)


def step_build(s) -> None:
    """Snapshots, survival points and legs for every group (one group at a time to bound memory)."""
    lad = pl.read_parquet(OUT / "ladder_legs.parquet")
    extra = OUT / "candles_fetched.parquet"
    sns, pts, lgs, cov, why_all = [], [], [], [], {}
    offset = 0
    for grp in GROUPS:
        lg_ = lad.filter(pl.col("group") == grp)
        if lg_.is_empty():
            continue
        snaps = snapshot_times(lg_)
        c = load_candles(s.data_dir, lg_["ticker"].unique().to_list(), (extra,))
        q = quotes_at(lg_, snaps, c)
        del c
        cov.append(coverage(lg_, q))
        sn, pt, lg, why = build_snapshots(lg_, q)
        del q
        why_all[grp] = why
        print(f"{grp}: {sn.height} snapshots {why} {datetime.now(UTC):%H:%M:%S}", flush=True)
        if sn.is_empty():
            continue
        sns.append(sn.with_columns(pl.col("snap_id") + offset))
        pts.append(pt.with_columns(pl.col("snap_id") + offset))
        lgs.append(lg.with_columns(pl.col("snap_id") + offset))
        offset += sn.height
    sn = pl.concat(sns, how="diagonal_relaxed").with_columns(_split())
    pt = pl.concat(pts)
    lg = pl.concat(lgs, how="diagonal_relaxed")
    sn.write_parquet(OUT / "snapshots.parquet")
    pt.write_parquet(OUT / "points.parquet")
    # per-leg rows for other branches: edges, quote, implied vol and moneyness at the leg's strike
    pk = pt.select("snap_id", "K", "sigma_K", "m", "S")
    legs = (lg.join(sn.select("snap_id", "event_ticker", "series_ticker", "category", "group", "snapshot_ts", "lead_h",
                              "split", "F", "sigma_atm", "orient"), on="snap_id")
            .with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread"),
                          pl.when(pl.col("hi").is_infinite()).then(pl.col("lo"))
                          .when(pl.col("lo").is_infinite()).then(pl.col("hi")).otherwise(None).alias("K")))
    legs = (legs.join(pk.rename({"S": "S_K"}), on=["snap_id", "K"], how="left")
            .join(pk.select("snap_id", pl.col("K").alias("lo"), pl.col("sigma_K").alias("sigma_lo")), on=["snap_id", "lo"], how="left")
            .join(pk.select("snap_id", pl.col("K").alias("hi"), pl.col("sigma_K").alias("sigma_hi")), on=["snap_id", "hi"], how="left"))
    legs.write_parquet(OUT / "legs.parquet")
    pl.concat(cov).write_parquet(OUT / "coverage.parquet")
    (OUT / "build_stats.json").write_text(json.dumps(why_all, indent=1))


def _df_json(df: pl.DataFrame) -> list[dict]:
    return json.loads(df.write_json()) if df.height else []


def step_analyze(s) -> dict[str, Any]:
    from albert.reports import Report

    lad = pl.read_parquet(OUT / "ladder_legs.parquet")
    sn = pl.read_parquet(OUT / "snapshots.parquet")
    pt = pl.read_parquet(OUT / "points.parquet")
    lg = pl.read_parquet(OUT / "legs.parquet", columns=["snap_id", "ticker", "lo", "hi", "m_lo", "m_hi", "bid", "ask", "ok", "y"])
    groups = [g for g in GROUPS if g in set(sn["group"].unique().to_list())]
    cover = (sn.group_by("group").agg(pl.col("event_ticker").n_unique().alias("events"), pl.col("ladder").n_unique().alias("ladders"),
                                      pl.len().alias("snapshots"), pl.col("n_legs_used").sum().alias("leg_quotes"),
                                      pl.col("n_legs_used").mean().round(1).alias("legs_per_snap"),
                                      pl.col("series_ticker").n_unique().alias("series"),
                                      (pl.col("split") == "dev").sum().alias("dev_snaps"),
                                      (pl.col("split") == "holdout").sum().alias("hold_snaps"))
             .sort("group"))
    cover_lead = sn.group_by("group", "lead_h").agg(pl.len().alias("snapshots"), pl.col("event_ticker").n_unique().alias("events")).sort("group", "lead_h")
    smile, curve = smile_tables(sn, pt)
    tails_lead = tail_calibration(sn, pt, by_lead=True)
    tails = tail_calibration(sn, pt, by_lead=False)
    tsum = tail_summary(sn, pt)
    pit = pit_table(sn)
    pit_lead = pit_table(sn, by_lead=True)
    width = width_table(sn)
    term = term_structure(sn)
    sp_fam = sn.filter(pl.col("group").str.starts_with("sp_")).with_columns(
        family(pl.col("series_ticker"), pl.col("group")).alias("group"))
    width_fam = width_table(sp_fam).filter(pl.col("lead_h") == -1)
    tsum_fam = tail_summary(sp_fam, pt).filter(pl.col("m_ge").is_in([1.5, 2.0]))
    feat = leg_features(sn, lg, lad)
    body = body_rules(feat)
    grid, chosen = gated_rules(feat.filter(pl.col("wing") != "body"), groups)
    coefs, maps = calibration_maps(feat, groups)
    pooled = pooled_fixed(feat)
    arb = arbitrage(sn, lg, lad)
    grid.write_parquet(OUT / "rules_grid.parquet")
    feat.write_parquet(OUT / "leg_features.parquet")

    shapes = jf_shapes(sn, pt)
    dyn_atm, dyn_k = smile_dynamics(sn, pt)
    varb = variance_between(sn)

    rep = Report(s.reports_dir, "research-pm-smile")
    rep.text("Question", "Descriptive study of the cross-strike implied distribution of Kalshi numeric ladders (strips "
             "of digital options): smile shape, smile dynamics (sticky strike vs sticky moneyness, backbone), ATM term "
             "structure, implied vs realized variance between snapshots, and calibration of the implied distribution "
             "against the realized outcome. Economic tests (run before the change of emphasis) are kept at the end. "
             "Dev = events expiring before 2026-10-01, holdout = October 2026. t-stats clustered by event. "
             "Moneyness m = (K - F)/sigma_ATM, sigma_ATM = implied IQR/1.349 (Bachelier units of the underlying).")
    rep.table("Coverage by group (fitted snapshots)", cover)
    rep.table("Coverage by group and lead", cover_lead, max_rows=60)
    rep.text("1. Smile shape", "sigma_K = (F - K)/Phi^-1(S(K)) per strike (|z| >= 0.25, the two strikes straddling F "
             "excluded); pooled OLS of sigma_K/sigma_ATM - 1 = a + s m + c m^2 with 95% CI (cluster by event), and the "
             "interquartile range of the per-snapshot fits (>= 3 off-ATM strikes). s > 0: upper-wing vols exceed "
             "lower-wing vols (right skew); c > 0: fat tails relative to a normal with the ATM width. Spreads are "
             "oriented to the favourite's margin. Caveat: wing strikes priced at the 1c tick inflate far-wing vols.")
    rep.table("Smile parameters by group, lead and split", smile, save_as="smile", max_rows=120)
    rep.table("Smile curve: median sigma_K/sigma_ATM by wing and |m| bucket", curve, save_as="smile_curve", max_rows=150)
    rep.table("Implied vs realized density shape: Jones-Faddy skew-t in standardized units (lead -1 = all leads; "
              "df_left/right = tail degrees of freedom, bowley = quantile skewness, tailw = tail weight, normal = 1)",
              shapes, save_as="jf_shapes", max_rows=80)
    rep.text("2. Smile dynamics", "Consecutive snapshots of one ladder (lead t -> next shorter lead u), raw coordinates, "
             "fixed effects per group x lead pair, standardized by the earlier sigma_ATM. ATM: slope of dsigma_ATM on dF "
             "against the sticky-strike prediction (the pooled smile skew s; sticky moneyness predicts 0); delta = "
             "coefficient on s*dF with per-snapshot s (1 = sticky strike; attenuated by noise in s). Backbone beta = "
             "d log sigma_ATM / d log F within ladder (positive underlyings; 0 normal, 0.5 Poisson-like, 1 lognormal) "
             "and the cross-sectional analogue within series x lead. Fixed strike: gamma = coefficient on the "
             "sticky-moneyness prediction -(s + 2 c m_K) dF (1 = sticky moneyness, 0 = sticky strike), with the "
             "pooled smile (no errors-in-variables) and with the per-snapshot smile.")
    rep.table("ATM response to the median and the backbone", dyn_atm, save_as="dyn_atm")
    rep.table("Fixed-strike vol response to the median", dyn_k, save_as="dyn_fixed_strike")
    rep.table("3. ATM term structure: within-ladder slope of log sigma_ATM on log lead (Brownian 0.5; sports use "
              "pre-game leads only)", term, save_as="term")
    rep.table("4. Implied vs realized variance between consecutive snapshots (standardized by sigma_t^2): RV = dF^2, "
              "IV = 1 - (sigma_u/sigma_t)^2; RV/IV < 1 means the implied distribution narrows faster than the median "
              "moves", varb, save_as="variance_between", max_rows=80)
    rep.text("5. Calibration against the realized outcome", "X_T is known only as the interval (L, U] implied by which "
             "legs settled YES. z_h0 is the z of realized minus implied tail mass under the hypothesis that implied "
             "probabilities are true (valid when no tail event occurred, where the clustered t collapses).")
    rep.table("Width ratio lambda: (X_T - F)/sigma_ATM ~ N(mu, lambda^2), interval-censored MLE (lead -1 = pooled)", width,
              save_as="width", max_rows=100)
    rep.table("Width ratio by sports family (pooled leads)", width_fam, save_as="width_family", max_rows=60)
    rep.table("Fractional PIT histogram (pooled leads)", pit, save_as="pit", max_rows=40)
    rep.table("PIT by lead", pit_lead, save_as="pit_by_lead", max_rows=80)
    rep.table("Tail mass beyond |m| (all leads): implied vs realized (cents)", tsum, save_as="tail_summary", max_rows=120)
    rep.table("Tail mass beyond |m| by sports family (pooled leads)", tsum_fam, save_as="tail_family", max_rows=120)
    rep.table("Tail calibration by |m| bucket (all leads)", tails, save_as="tails", max_rows=150)
    rep.table("Tail calibration by lead", tails_lead, save_as="tails_by_lead", max_rows=60)
    rep.text("6. Economic tests (kept from the first pass; not extended)", "One contract, taker at the snapshot candle's "
             "ask / 1 - bid with the Kalshi taker fee by series, held to settlement. t_risk is the t against payoff "
             "risk (market-price settlement s.d., events independent); the clustered t is meaningless when no tail hit.")
    rep.table("Sell-the-tail rules chosen on dev (best per rule family and group), with holdout", chosen,
              save_as="rules_chosen", max_rows=80)
    fixed = grid.filter(((pl.col("rule") == "model") & pl.col("param").is_in([1.5, 2.0]))
                        | ((pl.col("rule") == "price") & pl.col("param").is_in([0.05, 0.10]))
                        | (pl.col("rule") == "all")).filter(pl.col("group") != "ALL").sort("group", "lead_h", "rule", "param")
    rep.table("Fixed sell-the-tail rules by group and lead (not optimised; lead -1 = all leads)", fixed,
              save_as="rules_fixed", max_rows=200)
    rep.table("Pre-specified sell-the-tail rules pooled over groups and leads (model vs price controls)", pooled,
              save_as="pooled_rules", max_rows=60)
    rep.table("Buy the body bracket (contains F) at the ask, per lead", body, save_as="body", max_rows=60)
    rep.table("Calibration maps: logistic P(tail) on price only vs price + model moneyness (dev fit, holdout scores)",
              coefs, save_as="maps_coef")
    rep.table("Calibration-map trading (theta chosen on dev: chosen = true)", maps.filter(pl.col("chosen")) if maps.height else maps,
              save_as="maps_trade", max_rows=80)
    rep.table("Arbitrage at snapshot quotes (legs with spread <= 5c), net of taker fees", arb, save_as="arbitrage")
    path = rep.save()
    summary = {"generated": datetime.now(UTC).isoformat(), "report": str(path), "holdout_from": HOLDOUT.isoformat(),
               "coverage": _df_json(cover), "smile": _df_json(smile), "jf_shapes": _df_json(shapes),
               "dynamics_atm": _df_json(dyn_atm), "dynamics_fixed_strike": _df_json(dyn_k),
               "variance_between": _df_json(varb), "tail_summary": _df_json(tsum),
               "width": _df_json(width), "width_sports_family": _df_json(width_fam), "pit": _df_json(pit),
               "tail_sports_family": _df_json(tsum_fam), "term_structure": _df_json(term), "body": _df_json(body),
               "rules_chosen": _df_json(chosen), "pooled_rules": _df_json(pooled), "maps_coef": _df_json(coefs),
               "maps_trade_chosen": _df_json(maps.filter(pl.col("chosen")) if maps.height else maps),
               "arbitrage": _df_json(arb),
               "ladder_stats": json.loads((OUT / "ladder_stats.json").read_text()) if (OUT / "ladder_stats.json").exists() else None,
               "build_stats": json.loads((OUT / "build_stats.json").read_text()) if (OUT / "build_stats.json").exists() else None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(path, flush=True)
    return summary


def main(argv: list[str]) -> None:
    from albert.config import settings

    s = settings()
    step = argv[0] if argv else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    if step in ("ladders", "all"):
        lad, stats = build_ladders(s.tables_dir)
        lad.write_parquet(OUT / "ladder_legs.parquet")
        (OUT / "ladder_stats.json").write_text(json.dumps(stats, indent=1, default=str))
        print(json.dumps(stats, default=str), flush=True)
    if step in ("fetch", "all"):
        step_fetch(s, int(argv[1]) if len(argv) > 1 else None)
    if step in ("build", "all"):
        step_build(s)
    if step in ("analyze", "all"):
        step_analyze(s)


if __name__ == "__main__":
    main(sys.argv[1:])
