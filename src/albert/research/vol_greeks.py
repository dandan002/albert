"""Greek-based trades on Kalshi strike ladders: delta-hedged short strangles, vega trades on implied
vs forecast width, and gamma scalping (study ``vol_greeks``; research only, no orders).

**Model.** A leg pays YES iff X_T is in (lo, hi] (``pm_smile`` latent scale). With the ladder's
current implied median F and width sigma (``pm_smile.implied`` on the hourly candle quotes), a
digital D_K = P(X > K) = Phi(d), d = (F - K)/sigma, has delta phi(d)/sigma, gamma -d phi(d)/sigma^2
and vega -d phi(d)/sigma (all with respect to F or sigma); a leg is D(lo) - D(hi) and a NO holding
is 1 - leg. Flat sigma_ATM (no smile) is used for the greeks.

**Paths.** For every ladder of weather highs (``wx_high``, Aug 15 - Nov 9), sports totals, spreads
and team totals (Aug - Oct), each leg is quoted on an hourly grid from 36 h (weather) / 24 h
(sports) before the reference time to the close: the close of the last hourly candle that ended
at or before the hour, at most 2 h old (no look-ahead). Each hour with >= 3 legs at a spread <= 5c
is fitted with ``pm_smile.implied`` (F, sigma_ATM). Splits by t_ref: dev < 2026-10-01, holdout =
October, nov = 2026-11-01..11-10 (weather only: ``vol_ladders/nov``).

**Structures.** Entry at a fixed lead (snapshot hour, as ``pm_smile``). A strangle takes one leg
per side whose tail boundary nearest F has |m| = |K - F|/sigma in a band (the one nearest the band
centre); short vol holds the side that wins when X stays central (NO on a tail leg, YES on a
threshold leg that contains F), long vol the other side. Hedging trades the quoted non-core leg
with the largest |delta| (spread <= 5c): YES or NO to offset the net delta, first selling hedge
lots that add to the imbalance. Rules: ``none``; ``entry`` (delta-neutral at entry, never
rebalanced); ``every6`` (rebalanced every 6 h; "daily" equals ``entry`` for these <= 36 h holds);
``thr<x>`` (checked hourly, rebalanced when |net delta| x sigma > x per strangle unit, i.e. the
position's P&L for a 1-sigma move of F exceeds x dollars per unit). The hedge leg needs |delta| x
sigma >= 0.10 and a hedge order is at most 4 contracts per unit (8 open), which keeps broken fits
(a near-zero leg delta) from producing absurd sizes. Hold to settlement unless the
variant exits at a lead (sell every lot at the held side's bid).

**Execution.** Every trade (entry, hedge, exit) is a taker order at the hour's candle ask (YES) or
1 - bid (NO); sales at the bid / 1 - ask. Per-unit statistics use fractional hedge sizes with the
fee per contract of a 5-lot order (``kalshi_fee(5, p, conservative=True) / 5``, series multiplier).
The capped book uses whole contracts and ``kalshi_fee(n, p, conservative=True)`` per order, and
caps every order at ``VOL_SHARE`` x the leg's traded volume in the 3 hours to the quote (displayed
depth is not in the candles).

**Capped book.** Chronological, $100 bankroll, <= $5 collateral per event and <= $20 open (scale
check $10 / $100); hedged variants size entries to 75% of the room so later hedges fit. One ledger
row per lot (hedge lots and partial closes are their own rows; ``settle_ts`` = sale or settlement
time, ``payoff_usd`` = sale proceeds net of fees or settlement payoff), the ``vol_metrics`` schema.

Strategies (grids pre-declared, <= 12 variants each, chosen on dev by PnL per $ risked):
S1 short strangle, unhedged vs hedged; S2 vega trade on a dev-fitted width forecast (weather:
``vol_ladders`` NBM sigma model; sports: interval-censored width model by league x lead with a
shrink on log sigma_ATM); S3 gamma scalping (long strangle, hedged) only where the dev realized F
variance exceeds the implied variance drop (``rv`` step).

Results (2026-11-11, data/research/vol_greeks/verdict.md): nothing clears the bar. Short strangles
lose 1-3c per contract in every group (the wing premium is paid back in tails; costs 1.3-2.2c);
delta-hedging lowers the Sharpe ratio everywhere; sports width-forecast vega trades flip sign in
October; weather 12->8 h has realized/implied F variance 1.36-1.57 but long gamma loses on taker
costs. Sports wing legs barely trade pre-game (2-31% of entries pass the volume cap).

    python -m albert.research.vol_greeks run [panel|rv|s1|s2|s3|book|metrics|report|all]
"""

import heapq
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import ndtr

from albert.fees import kalshi_fee
from albert.research import pm_smile as sm
from albert.research.pm_smile import cmean

OUT = Path("data/research/vol_greeks")
SMILE = Path("data/research/pm_smile")
VL = Path("data/research/vol_ladders")
NOV = VL / "nov"
GROUPS = ("wx_high", "sp_total", "sp_spread", "sp_teamtotal")
WX_LEADS = (36, 24, 16, 12, 8)
SP_LEADS = (24, 6)
HOLDOUT = datetime(2026, 10, 1, tzinfo=UTC)
NOV_START = datetime(2026, 11, 1, tzinfo=UTC)
SPLITS = ("dev", "holdout", "nov")
MAX_AGE = timedelta(hours=2)
HEDGE_MAX_SPREAD = 0.05
ORDER_N = 5
VOL_SHARE = 0.25
VOL_WINDOW = "3h"
BANKROLL = 100.0
EVENT_CAP, OPEN_CAP = 5.0, 20.0
HEDGE_RESERVE = 0.25
MIN_HEDGE_DELTA = 0.10          # hedge leg needs |delta| x sigma >= this (phi(d) >= 0.1: within ~1.7 sigma of F)
HEDGE_MAX_UNITS = 4.0           # hedge contracts per strangle unit: per order and open in total
MIN_DEV_EVENTS = 30
SQRT2PI = math.sqrt(2 * math.pi)
EPS = 1e-9
SIGN = {"yes": 1.0, "no": -1.0}


# ---- greeks ----------------------------------------------------------------------------------------
def digital_greeks(F: float, sig: float, K: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """D = P(X > K) = Phi((F - K)/sig) and dD/dF, d2D/dF2, dD/dsig; K = -inf gives D = 1, +inf D = 0."""
    K = np.asarray(K, dtype=float)
    fin = np.isfinite(K)
    d = np.where(fin, (F - np.where(fin, K, 0.0)) / sig, 0.0)
    pdf = np.where(fin, np.exp(-0.5 * d * d) / SQRT2PI, 0.0)
    D = np.where(fin, ndtr(d), np.where(K < 0, 1.0, 0.0))
    return D, pdf / sig, -d * pdf / sig**2, -d * pdf / sig


def leg_greeks(F: float, sig: float, lo: Any, hi: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(value, delta, gamma, vega) of YES on (lo, hi] = D(lo) - D(hi)."""
    a = digital_greeks(F, sig, lo)
    b = digital_greeks(F, sig, hi)
    return a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]


@lru_cache(maxsize=200_000)
def fee_unit(price: float, mult: float, fee_type: str | None) -> float:
    """Taker fee per contract of a ``ORDER_N``-lot order, rounded up to the cent per order."""
    return kalshi_fee(ORDER_N, price, fee_type=fee_type or "quadratic", multiplier=mult, conservative=True) / ORDER_N


def split_of(t: datetime) -> str:
    return "dev" if t < HOLDOUT else "holdout" if t < NOV_START else "nov"


# ---- ladder paths ------------------------------------------------------------------------------------
@dataclass
class LadderPath:
    """Hourly quotes and fits of one ladder. ``bid``/``ask``/``vol`` are (hours, legs), NaN without a
    quote <= 2 h old; ``F``/``sig`` NaN where the fit failed."""
    ladder: str
    event: str
    group: str
    series: str
    split: str
    kind: str
    t_ref: datetime
    tickers: list[str]
    lo: np.ndarray
    hi: np.ndarray
    y: np.ndarray
    fee_mult: np.ndarray
    fee_type: list[str | None]
    settle: list[datetime]
    hours: list[datetime]
    bid: np.ndarray
    ask: np.ndarray
    vol: np.ndarray
    F: np.ndarray
    sig: np.ndarray

    def index(self, t: datetime) -> int | None:
        try:
            return self.hours.index(t)
        except ValueError:
            return None

    def fit_ok(self, i: int) -> bool:
        return bool(np.isfinite(self.F[i]) and np.isfinite(self.sig[i]) and self.sig[i] > 0)

    def held_quote(self, i: int, j: int, side: str) -> tuple[float, float, float] | None:
        """(buy price, sell price, mid) of the held side of leg j at hour i."""
        b, a = self.bid[i, j], self.ask[i, j]
        if not (np.isfinite(b) and np.isfinite(a)) or a < b:
            return None
        if side == "yes":
            return float(a), float(b), float(a + b) / 2
        return float(1 - b), float(1 - a), float(1 - (a + b) / 2)

    def lead_index(self, lead_h: int) -> int | None:
        t = (self.t_ref - timedelta(hours=lead_h)).replace(minute=0, second=0, microsecond=0)
        return self.index(t)


def load_ladders() -> pl.DataFrame:
    """pm_smile's ladder legs for ``GROUPS`` plus the November weather ladders, with settlement time
    and split."""
    from albert.config import settings

    lad = pl.read_parquet(SMILE / "ladder_legs.parquet").filter(pl.col("group").is_in(GROUPS))
    frames = [lad]
    if (NOV / "tables" / "kalshi_markets.parquet").exists():
        nov, _ = sm.build_ladders(NOV / "tables")
        frames.append(nov.filter(pl.col("group") == "wx_high").select(lad.columns))
    lad = pl.concat(frames, how="diagonal_relaxed").unique("ticker", keep="first")
    tabs = [settings().tables_dir / "kalshi_markets.parquet", NOV / "tables" / "kalshi_markets.parquet"]
    st = pl.concat([pl.scan_parquet(p).select("ticker", pl.coalesce("settlement_ts", "close_time").alias("settle_ts"))
                    .collect() for p in tabs if p.exists()]).unique("ticker", keep="last")
    lad = lad.join(st, on="ticker", how="left").with_columns(pl.coalesce("settle_ts", "close_time").alias("settle_ts"))
    return lad.with_columns(pl.when(pl.col("t_ref") < HOLDOUT).then(pl.lit("dev"))
                            .when(pl.col("t_ref") < NOV_START).then(pl.lit("holdout")).otherwise(pl.lit("nov"))
                            .alias("split"))


def load_candles(tickers: list[str]) -> pl.DataFrame:
    """Hourly (ticker, end_ts, bid, ask, vol3) from every cached candle source; vol3 = contracts
    traded in the 3 h ending at the candle."""
    tk = pl.Series(tickers).unique().implode()
    frames = []
    srcs = [Path("data") / s for s in sm.CANDLE_SOURCES] + [SMILE / "candles_fetched.parquet"]
    for src in srcs:
        if not src.exists():
            continue
        lf = pl.scan_parquet(src)
        if "period_minutes" in lf.collect_schema():
            lf = lf.filter(pl.col("period_minutes") == 60)
        frames.append(lf.filter(pl.col("ticker").is_in(tk)).select(
            "ticker", pl.col("end_ts").cast(pl.Datetime("us", "UTC")), pl.col("yes_bid_close").cast(pl.Float64).alias("bid"),
            pl.col("yes_ask_close").cast(pl.Float64).alias("ask"), pl.col("volume").cast(pl.Float64).alias("vol")).collect())
    for src in (VL / "wx_ohlc.parquet", NOV / "wx_ohlc.parquet"):
        if src.exists():
            frames.append(pl.scan_parquet(src).filter(pl.col("ticker").is_in(tk)).select(
                "ticker", pl.col("end_ts").cast(pl.Datetime("us", "UTC")), "bid", "ask",
                pl.col("volume").alias("vol")).collect())
    c = (pl.concat(frames).filter(pl.col("bid").is_not_null() & pl.col("ask").is_not_null())
         .unique(["ticker", "end_ts"], keep="first").sort("ticker", "end_ts"))
    return c.with_columns(pl.col("vol").fill_null(0.0).rolling_sum_by("end_ts", VOL_WINDOW).over("ticker").alias("vol3")
                          ).drop("vol")


def fit_hours(q: pl.DataFrame) -> pl.DataFrame:
    """(ladder, t, F, sigma, n_ok) for every ladder-hour of ``q`` (sorted ladder, t, lo, hi)."""
    q = q.with_columns(pl.struct("ladder", "t").rle_id().alias("gid"))
    gid = q["gid"].to_numpy()
    lo, hi = q["lo"].to_numpy(), q["hi"].to_numpy()
    bid, ask = q["bid"].to_numpy(), q["ask"].to_numpy()
    kind = q["kind"].to_list()
    ok = (np.isfinite(bid) & np.isfinite(ask) & (ask >= bid) & (ask - bid <= HEDGE_MAX_SPREAD + EPS))
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    ends = np.r_[starts[1:], len(q)]
    out_F, out_s, out_n = [], [], []
    for a, b in zip(starts, ends):
        F = s = np.nan
        k = ok[a:b]
        if k.sum() >= 3:
            sp = sm.strike_points(kind[a], lo[a:b], hi[a:b], np.nan_to_num(bid[a:b]), np.nan_to_num(ask[a:b], nan=1.0), k)
            if sp is not None and len(sp[0]) >= 3:
                fit = sm.implied(*sp)
                if fit is not None and fit["sigma"] > 0:
                    F, s = fit["F"], fit["sigma"]
        out_F.append(F)
        out_s.append(s)
        out_n.append(int(k.sum()))
    keys = q[starts.tolist()].select("ladder", "t")
    return keys.with_columns(pl.Series("F", out_F, pl.Float64), pl.Series("sigma", out_s, pl.Float64),
                             pl.Series("n_ok", out_n, pl.Int64))


def step_panel() -> dict[str, Any]:
    """Hourly quote and fit panels per group (``paths_<group>_{quotes,fits}.parquet``, ``paths_legs.parquet``)."""
    OUT.mkdir(parents=True, exist_ok=True)
    lad = load_ladders()
    lad.write_parquet(OUT / "paths_legs.parquet")
    stats: dict[str, Any] = {}
    for grp in GROUPS:
        g = lad.filter(pl.col("group") == grp)
        span = max(WX_LEADS) if grp.startswith("wx") else max(SP_LEADS)
        c = load_candles(g["ticker"].to_list())
        g = g.filter(pl.col("ticker").is_in(c["ticker"].unique().implode()))
        hours = (g.group_by("ladder").agg(pl.col("t_ref").first())
                 .with_columns(pl.datetime_ranges((pl.col("t_ref") - pl.duration(hours=span)).dt.truncate("1h"),
                                                  pl.col("t_ref"), interval="1h").alias("t")).explode("t").drop("t_ref"))
        q = (g.select("ladder", "ticker", "kind", "lo", "hi", "open_time", "close_time").join(hours, on="ladder")
             .filter((pl.col("open_time") <= pl.col("t")) & (pl.col("t") < pl.col("close_time"))).sort("t"))
        q = q.join_asof(c.rename({"end_ts": "q_ts"}).sort("q_ts"), left_on="t", right_on="q_ts", by="ticker",
                        strategy="backward", tolerance=MAX_AGE).drop("open_time", "close_time")
        q = q.sort("ladder", "t", "lo", "hi")
        fits = fit_hours(q)
        q.select("ladder", "t", "ticker", "bid", "ask", "vol3").write_parquet(OUT / f"paths_{grp}_quotes.parquet")
        fits.write_parquet(OUT / f"paths_{grp}_fits.parquet")
        stats[grp] = {"ladders": g["ladder"].n_unique(), "legs": g.height, "quote_rows": q.height,
                      "hours": fits.height, "fit_ok": round(float(fits["F"].is_not_nan().mean()), 3)}
        print(grp, stats[grp], flush=True)
        del c, q, fits
    (OUT / "panel.json").write_text(json.dumps(stats, indent=1))
    return stats


def load_paths(group: str, splits: tuple[str, ...] = SPLITS) -> list[LadderPath]:
    """LadderPath objects for one group from the panel files."""
    legs = pl.read_parquet(OUT / "paths_legs.parquet").filter((pl.col("group") == group) & pl.col("split").is_in(splits))
    q = pl.read_parquet(OUT / f"paths_{group}_quotes.parquet").filter(pl.col("ladder").is_in(legs["ladder"].unique().implode()))
    fits = pl.read_parquet(OUT / f"paths_{group}_fits.parquet")
    legs = legs.filter(pl.col("ticker").is_in(q["ticker"].unique().implode())).sort("ladder", "lo", "hi")
    out = []
    qd = {k[0]: g for k, g in q.group_by("ladder")}
    fd = {k[0]: g for k, g in fits.group_by("ladder")}
    for (lad,), g in legs.group_by("ladder", maintain_order=True):
        if lad not in qd or lad not in fd:
            continue
        f = fd[lad].sort("t")
        hours = f["t"].to_list()
        hix = {t: i for i, t in enumerate(hours)}
        tick = g["ticker"].to_list()
        jix = {t: j for j, t in enumerate(tick)}
        T, n = len(hours), len(tick)
        bid, ask, vol = (np.full((T, n), np.nan) for _ in range(3))
        qq = qd[lad]
        ii = np.array([hix[t] for t in qq["t"].to_list()])
        jj = np.array([jix.get(t, -1) for t in qq["ticker"].to_list()])
        m = jj >= 0
        bid[ii[m], jj[m]] = qq["bid"].to_numpy()[m]
        ask[ii[m], jj[m]] = qq["ask"].to_numpy()[m]
        vol[ii[m], jj[m]] = qq["vol3"].to_numpy()[m]
        r = g.row(0, named=True)
        out.append(LadderPath(lad, r["event_ticker"], group, r["series_ticker"], r["split"], r["kind"], r["t_ref"], tick,
                              g["lo"].to_numpy(), g["hi"].to_numpy(), g["y"].to_numpy(), g["fee_mult"].to_numpy(),
                              g["fee_type"].to_list(), g["settle_ts"].to_list(), hours, bid, ask, vol,
                              f["F"].cast(pl.Float64).to_numpy(), f["sigma"].cast(pl.Float64).to_numpy()))
    return out


# ---- positions -----------------------------------------------------------------------------------------
@dataclass
class Lot:
    leg: int
    side: str
    q: float
    i_in: int
    price: float
    fee: float
    mid: float
    role: str
    i_out: int | None = None
    out_price: float | None = None
    out_fee: float = 0.0
    out_mid: float | None = None
    payoff: float | None = None          # settlement payoff (None while open or when sold)

    def cost(self) -> float:
        return self.q * self.price + self.fee

    def value(self) -> float:
        """Sale proceeds net of fee, or the settlement payoff."""
        if self.i_out is not None:
            return self.q * self.out_price - self.out_fee
        return self.payoff if self.payoff is not None else float("nan")

    def pnl(self) -> float:
        return self.value() - self.cost()


@dataclass
class Rule:
    """Hedge rule: kind none / entry / every / thr, interval ``k`` hours, threshold ``thr``."""
    kind: str = "none"
    k: int = 6
    thr: float = 0.0

    @property
    def name(self) -> str:
        return {"none": "none", "entry": "entry"}.get(self.kind) or (f"every{self.k}" if self.kind == "every" else f"thr{self.thr:g}")

    def hours(self, i0: int, i_last: int) -> list[int]:
        if self.kind == "every":
            return list(range(i0 + self.k, i_last + 1, self.k))
        if self.kind == "thr":
            return list(range(i0 + 1, i_last + 1))
        return []


RULES = {r.name: r for r in (Rule("none"), Rule("entry"), Rule("every", k=6), Rule("thr", thr=0.10), Rule("thr", thr=0.25))}


class Position:
    """One structure on one ladder: core lots, hedge lots, cash. ``integer`` = whole contracts with
    per-order fees and volume caps (the capped book); else fractional sizes with 5-lot fees."""

    def __init__(self, p: LadderPath, core: list[tuple[int, str]], units: float = 1.0, integer: bool = False,
                 room: Any = None) -> None:
        self.p, self.core, self.units, self.integer, self.room = p, core, units, integer, room
        self.lots: list[Lot] = []
        self.cash = 0.0
        self.core_legs = {j for j, _ in core}
        self.steps: list[int] = []            # book: remaining decision hours
        self.k = 0
        self.finished = False

    def fee(self, q: float, price: float, j: int) -> float:
        if q <= 0 or price <= 0 or price >= 1:
            return 0.0
        if self.integer:
            return kalshi_fee(q, price, fee_type=self.p.fee_type[j] or "quadratic", multiplier=float(self.p.fee_mult[j]),
                              conservative=True)
        return q * fee_unit(round(price, 4), float(self.p.fee_mult[j]), self.p.fee_type[j])

    def vol_cap(self, i: int, j: int) -> float:
        if not self.integer:
            return math.inf
        v = self.p.vol[i, j]
        return math.floor(VOL_SHARE * v + EPS) if np.isfinite(v) else 0

    def buy(self, i: int, j: int, side: str, q: float, role: str) -> Lot | None:
        hq = self.p.held_quote(i, j, side)
        if hq is None or not 0 < hq[0] < 1 or q <= EPS:
            return None
        price = hq[0]
        if self.integer:
            q = min(math.floor(q + EPS), self.vol_cap(i, j))
            if self.room is not None:
                r = self.room()
                while q >= 1 and q * price + self.fee(q, price, j) > r + EPS:
                    q -= 1
            if q < 1:
                return None
        lot = Lot(j, side, q, i, price, self.fee(q, price, j), hq[2], role)
        self.lots.append(lot)
        self.cash -= lot.cost()
        return lot

    def sell(self, i: int, lot: Lot, q: float) -> bool:
        hq = self.p.held_quote(i, lot.leg, lot.side)
        if hq is None or lot.i_out is not None or q <= EPS:
            return False
        if self.integer:
            q = min(math.floor(q + EPS), lot.q, self.vol_cap(i, lot.leg))
            if q < 1:
                return False
        q = min(q, lot.q)
        if q < lot.q - EPS:                       # split: the remainder keeps its share of the entry fee
            rest = replace(lot, q=lot.q - q, fee=lot.fee * (lot.q - q) / lot.q)
            lot.fee -= rest.fee
            lot.q = q
            self.lots.append(rest)
        px = hq[1]
        lot.i_out, lot.out_price, lot.out_fee, lot.out_mid = i, px, self.fee(q, px, lot.leg), hq[2]
        self.cash += lot.value()
        return True

    def open_lots(self) -> list[Lot]:
        return [x for x in self.lots if x.i_out is None and x.payoff is None]

    def net_delta(self, i: int) -> float:
        p = self.p
        lots = self.open_lots()
        if not lots:
            return 0.0
        j = np.array([x.leg for x in lots])
        d = leg_greeks(p.F[i], p.sig[i], p.lo[j], p.hi[j])[1]
        return float(sum(SIGN[x.side] * x.q * dj for x, dj in zip(lots, d)))

    def enter(self, i: int) -> bool:
        """Buy ``units`` of every core leg (all or nothing)."""
        got = []
        for j, side in self.core:
            lot = self.buy(i, j, side, self.units, "core")
            if lot is None or abs(lot.q - self.units) > EPS:
                for x in got + ([lot] if lot is not None else []):
                    self.lots.remove(x)
                    self.cash += x.cost()
                return False
            got.append(lot)
        return True

    def hedge(self, i: int, thr: float | None = None) -> int:
        """Trade toward zero net delta at hour i (no-op without a fit or, for a threshold rule, while
        |delta| x sigma <= thr x units). Returns the number of orders."""
        p = self.p
        if not p.fit_ok(i):
            return 0
        F, s = p.F[i], p.sig[i]
        D = self.net_delta(i)
        if thr is not None and abs(D * s) <= thr * self.units:
            return 0
        orders = 0
        for lot in sorted((x for x in self.open_lots() if x.role == "hedge"), key=lambda x: x.i_in):
            dl = SIGN[lot.side] * float(leg_greeks(F, s, p.lo[lot.leg], p.hi[lot.leg])[1])
            if D * dl <= 0 or abs(dl) < EPS:
                continue
            q = min(lot.q, abs(D) / abs(dl))
            if self.integer:
                q = math.floor(q + 0.5)
            if q > EPS and self.sell(i, lot, q):
                D -= lot.q * dl
                orders += 1
            if abs(D * s) < 1e-6:
                return orders
        cand = [j for j in range(len(p.tickers)) if j not in self.core_legs and np.isfinite(p.bid[i, j])
                and np.isfinite(p.ask[i, j]) and p.ask[i, j] - p.bid[i, j] <= HEDGE_MAX_SPREAD + EPS
                and p.bid[i, j] > 0 and p.ask[i, j] < 1]
        if not cand:
            return orders
        d = leg_greeks(F, s, p.lo[cand], p.hi[cand])[1]
        k = int(np.argmax(np.abs(d)))
        dj = float(d[k])
        if abs(dj) * s < MIN_HEDGE_DELTA:
            return orders
        side = "yes" if D * dj < 0 else "no"
        open_hedge = sum(x.q for x in self.open_lots() if x.role == "hedge")
        q = min(abs(D) / abs(dj), HEDGE_MAX_UNITS * self.units, max(2 * HEDGE_MAX_UNITS * self.units - open_hedge, 0.0))
        if self.integer:
            q = math.floor(q + 0.5)
        if q > EPS and self.buy(i, cand[k], side, q, "hedge") is not None:
            orders += 1
        return orders

    def exit(self, i: int) -> None:
        for lot in self.open_lots():
            self.sell(i, lot, lot.q)

    def settle(self) -> None:
        for lot in self.open_lots():
            won = (int(self.p.y[lot.leg]) == 1) == (lot.side == "yes")
            lot.payoff = lot.q * float(won)
            self.cash += lot.payoff

    def pnl(self) -> float:
        return float(sum(x.pnl() for x in self.lots))

    def release_time(self, lot: Lot) -> datetime | None:
        if lot.i_out is not None:
            return self.p.hours[lot.i_out]
        if lot.payoff is not None:
            return self.p.settle[lot.leg]
        return None

    def attribution(self) -> dict[str, float]:
        """Mid-based split of the PnL: premium (core legs' distance from the payoff they expect at
        entry), jump (core settlement/exit outcome), hedge mid PnL, and spread+fee costs."""
        out = {"premium": 0.0, "jump": 0.0, "hedge_mid": 0.0, "core_cost": 0.0, "hedge_cost": 0.0}
        for x in self.lots:
            out_mid = x.payoff if x.i_out is None else x.q * x.out_mid
            cost = x.q * (x.price - x.mid) + x.fee
            if x.i_out is not None:
                cost += x.q * (x.out_mid - x.out_price) + x.out_fee
            if x.role == "core":
                if x.mid >= 0.5:
                    out["premium"] += x.q * (1 - x.mid)
                    out["jump"] += out_mid - x.q
                else:
                    out["premium"] -= x.q * x.mid
                    out["jump"] += out_mid
                out["core_cost"] -= cost
            else:
                out["hedge_mid"] += out_mid - x.q * x.mid
                out["hedge_cost"] -= cost
        return out


# ---- structures --------------------------------------------------------------------------------------
def tail_strike(F: float, lo: float, hi: float, kind: str) -> tuple[float, str] | None:
    """(tail boundary nearest F, short-vol side) of a leg, None for a bracket containing F."""
    inside = lo < F <= hi
    if inside:
        if kind == "bracket" or (np.isfinite(lo) and np.isfinite(hi)):
            return None
        return (lo if np.isfinite(lo) else hi), "yes"
    return (lo if lo >= F else hi), "no"


def strangle(p: LadderPath, i: int, band: tuple[float, float], short: bool = True) -> list[tuple[int, str]] | None:
    """One quoted leg per side whose tail boundary has |m| in ``band`` (nearest the band centre);
    the short-vol side (or the other side for long vol). None unless both sides exist."""
    if not p.fit_ok(i):
        return None
    F, s = p.F[i], p.sig[i]
    centre = sum(band) / 2
    best: dict[str, tuple[float, int, str]] = {}
    for j in range(len(p.tickers)):
        b, a = p.bid[i, j], p.ask[i, j]
        if not (np.isfinite(b) and np.isfinite(a)) or a < b or a - b > HEDGE_MAX_SPREAD + EPS:
            continue
        ts = tail_strike(F, p.lo[j], p.hi[j], p.kind)
        if ts is None:
            continue
        m = (ts[0] - F) / s
        if not band[0] <= abs(m) <= band[1]:
            continue
        side = ts[1] if short else ("no" if ts[1] == "yes" else "yes")
        hq = p.held_quote(i, j, side)
        if hq is None or not 0 < hq[0] < 1:
            continue
        key = "up" if m > 0 else "down"
        if key not in best or abs(abs(m) - centre) < best[key][0]:
            best[key] = (abs(abs(m) - centre), j, side)
    if len(best) < 2:
        return None
    return [(best["down"][1], best["down"][2]), (best["up"][1], best["up"][2])]


@dataclass
class Entry:
    p: LadderPath
    i0: int
    core: list[tuple[int, str]]
    lead: int
    exit_i: int | None = None
    sid: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def t0(self) -> datetime:
        return self.p.hours[self.i0]


def entries_for(paths: list[LadderPath], lead: int, band: tuple[float, float], short: bool = True,
                exit_lead: int | None = None, select: Any = None) -> list[Entry]:
    out = []
    for p in paths:
        i0 = p.lead_index(lead)
        if i0 is None:
            continue
        if select is not None and not select(p, lead):
            continue
        core = strangle(p, i0, band, short)
        if core is None:
            continue
        ex = p.lead_index(exit_lead) if exit_lead is not None else None
        if exit_lead is not None and (ex is None or ex <= i0):
            continue
        out.append(Entry(p, i0, core, lead, ex, f"{p.ladder}|{lead}"))
    return out


def run_unit(e: Entry, rule: Rule) -> dict[str, Any] | None:
    """Per-unit (fractional) simulation of one structure; None if the entry cannot be filled."""
    pos = Position(e.p, e.core)
    if not pos.enter(e.i0):
        return None
    n_orders = 0
    thr = rule.thr if rule.kind == "thr" else None
    last = (e.exit_i - 1) if e.exit_i is not None else len(e.p.hours) - 1
    if rule.kind != "none":
        n_orders += pos.hedge(e.i0)
        for i in rule.hours(e.i0, last):
            n_orders += pos.hedge(i, thr)
    if e.exit_i is not None:
        pos.exit(e.exit_i)
    pos.settle()
    att = pos.attribution()
    cost = float(sum(x.cost() for x in pos.lots))
    return {"sid": e.sid, "ladder": e.p.ladder, "event_ticker": e.p.event, "group": e.p.group, "series": e.p.series,
            "split": e.p.split, "t0": e.t0, "t_ref": e.p.t_ref, "lead": e.lead, "pnl": pos.pnl(), "cost": cost,
            "n_core": float(len(e.core)), "hedge_orders": n_orders, "cash": pos.cash, **att, **e.extra}


def stat(df: pl.DataFrame) -> dict[str, Any]:
    """n core contracts, structures, events, c per core contract, t clustered by event, PnL per $."""
    if df.is_empty():
        return {"n": 0, "structs": 0, "events": 0}
    x = (df["pnl"] / df["n_core"]).to_numpy()
    mu, t, n, k = cmean(x, df["event_ticker"].to_numpy())
    cost = float(df["cost"].sum())
    return {"n": int(df["n_core"].sum()), "structs": n, "events": k, "c": None if mu is None else round(mu * 100, 3),
            "t": None if t is None else round(t, 2), "per_dollar": round(float(df["pnl"].sum()) / cost, 4) if cost > 0 else None}


def unit_sharpe(df: pl.DataFrame) -> float | None:
    """Annualised Sharpe of the per-unit PnL booked on each structure's t_ref day (calendar-filled)."""
    if df.height < 5:
        return None
    d = df.group_by(pl.col("t_ref").dt.date().alias("d")).agg(pl.col("pnl").sum()).sort("d")
    cal = pl.DataFrame({"d": pl.date_range(d["d"].min(), d["d"].max(), "1d", eager=True)}).join(d, on="d", how="left")
    x = cal["pnl"].fill_null(0.0).to_numpy()
    sd = float(np.std(x, ddof=1))
    return round(float(np.mean(x)) / sd * math.sqrt(365), 2) if sd > 0 else None


def evaluate(entries: list[Entry], rule: Rule, variant: dict[str, Any]) -> tuple[dict[str, Any], pl.DataFrame]:
    rows = [r for e in entries if (r := run_unit(e, rule)) is not None]
    df = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    out = dict(variant)
    for sp in SPLITS:
        g = df.filter(pl.col("split") == sp) if df.height else df
        out.update({f"{sp}_{k}": v for k, v in stat(g).items()})
        out[f"{sp}_unit_sharpe"] = unit_sharpe(g) if g.height else None
        if g.height:
            out[f"{sp}_hedge_orders"] = round(float(g["hedge_orders"].mean()), 2)
            for c in ("premium", "jump", "hedge_mid", "core_cost", "hedge_cost"):
                out[f"{sp}_{c}_c"] = round(float((g[c] / g["n_core"]).mean()) * 100, 3)
    return out, df.with_columns(pl.lit(variant["variant"]).alias("variant")) if df.height else df


def choose(grid: list[dict[str, Any]], min_events: int = MIN_DEV_EVENTS) -> dict[str, Any] | None:
    ok = [g for g in grid if (g.get("dev_events") or 0) >= min_events and g.get("dev_per_dollar") is not None]
    return max(ok, key=lambda g: g["dev_per_dollar"]) if ok else None


def months(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [{"month": m, **stat(g)} for (m,), g in df.sort("t_ref").group_by(pl.col("t_ref").dt.strftime("%Y-%m"),
                                                                              maintain_order=True)]


# ---- realized vs implied F variance ---------------------------------------------------------------------
RV_WINDOWS = {"wx_high": ((36, 24), (24, 16), (16, 12), (12, 8), (8, 4)),
              "sp": ((24, 6), (6, 3), (6, 1), (3, 1))}


def rv_rows(paths: list[LadderPath], windows: tuple[tuple[int, int], ...]) -> pl.DataFrame:
    """Per ladder and window: hourly-sum and endpoint realized variance of F, implied drop
    sigma_a^2 - sigma_b^2, all in units of sigma_a^2."""
    rows = []
    for p in paths:
        for a, b in windows:
            ia, ib = p.lead_index(a), p.lead_index(b)
            if ia is None or ib is None or not (p.fit_ok(ia) and p.fit_ok(ib)):
                continue
            idx = [i for i in range(ia, ib + 1) if p.fit_ok(i)]
            F = p.F[idx]
            s2 = p.sig[ia] ** 2
            rows.append({"ladder": p.ladder, "event_ticker": p.event, "split": p.split, "window": f"{a}->{b}",
                         "rv_hourly": float(np.sum(np.diff(F) ** 2)) / s2, "rv_end": float((F[-1] - F[0]) ** 2) / s2,
                         "iv": 1 - (p.sig[ib] / p.sig[ia]) ** 2, "hours_fit": len(idx), "hours": ib - ia + 1,
                         "dF": float(F[-1] - F[0]) / p.sig[ia]})
    return pl.DataFrame(rows)


def ratio_stat(g: pl.DataFrame, col: str) -> dict[str, Any]:
    """Ratio of means sum(rv)/sum(iv) with an event-clustered delta-method standard error."""
    rv, iv = g[col].to_numpy(), g["iv"].to_numpy()
    if iv.sum() <= 0:
        return {"ratio": None}
    r = rv.sum() / iv.sum()
    u = (rv - r * iv) / iv.mean()
    _, t, _, _ = cmean(u, g["event_ticker"].to_numpy())
    se = abs(float(u.mean()) / t) if t not in (None, 0) else None
    return {"ratio": round(float(r), 3), "se": None if se is None else round(se, 3)}


def step_rv() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for grp in GROUPS:
        paths = load_paths(grp)
        r = rv_rows(paths, RV_WINDOWS["wx_high" if grp == "wx_high" else "sp"])
        if r.is_empty():
            continue
        r = r.filter(pl.col("rv_hourly") < 25)            # drop broken fits (|dF| > 5 sigma per hour sum)
        rows = []
        for (w, sp), g in r.group_by("window", "split"):
            rows.append({"window": w, "split": sp, "n": g.height, "events": g["event_ticker"].n_unique(),
                         "fit_share": round(float((g["hours_fit"] / g["hours"]).mean()), 2),
                         "iv_mean": round(float(g["iv"].mean()), 4),
                         **{f"hourly_{k}": v for k, v in ratio_stat(g, "rv_hourly").items()},
                         **{f"end_{k}": v for k, v in ratio_stat(g, "rv_end").items()},
                         "drift_dF": round(float(g["dF"].mean()), 3)})
        out[grp] = sorted(rows, key=lambda x: (x["window"], SPLITS.index(x["split"])))
        print(grp, json.dumps(out[grp], indent=0), flush=True)
    _dump("rv", out)
    return out


# ---- S1: short strangle, unhedged vs hedged --------------------------------------------------------------
BANDS = {"near": (0.5, 1.25), "far": (1.0, 2.0)}
S1_LEADS = {"wx_high": (24, 16, 12, 8), "sp_total": SP_LEADS, "sp_spread": SP_LEADS, "sp_teamtotal": SP_LEADS}
S1_HEDGES = ("entry", "every6", "thr0.1", "thr0.25")


def step_s1() -> dict[str, Any]:
    """Per group: entry grid (lead x band, unhedged) chosen on dev, then the hedge rules on it."""
    res: dict[str, Any] = {}
    for grp in GROUPS:
        paths = load_paths(grp)
        grid, frames = [], []
        cache: dict[tuple, list[Entry]] = {}
        for lead in S1_LEADS[grp]:
            for bn, band in BANDS.items():
                es = entries_for(paths, lead, band)
                cache[(lead, bn)] = es
                v = {"variant": f"lead={lead}|band={bn}|hedge=none", "lead": lead, "band": bn, "hedge": "none"}
                row, df = evaluate(es, RULES["none"], v)
                grid.append(row)
                frames.append(df)
                print(grp, _brief(row), flush=True)
        base = choose(grid)
        if base is None:
            res[grp] = {"grid": grid, "note": "no entry variant with enough dev events"}
            continue
        for h in S1_HEDGES:
            v = {"variant": f"lead={base['lead']}|band={base['band']}|hedge={h}", "lead": base["lead"],
                 "band": base["band"], "hedge": h}
            row, df = evaluate(cache[(base["lead"], base["band"])], RULES[h], v)
            grid.append(row)
            frames.append(df)
            print(grp, _brief(row), flush=True)
        hedged = choose([g for g in grid if g["hedge"] != "none"])
        best = choose(grid)
        trades = pl.concat([f for f in frames if f.height], how="diagonal_relaxed")
        trades.write_parquet(OUT / f"trades_s1_{grp}.parquet")
        res[grp] = {"grid": grid, "entry_chosen": base["variant"], "hedged_chosen": hedged["variant"] if hedged else None,
                    "chosen": best["variant"],
                    "months": {k: months(trades.filter(pl.col("variant") == k))
                               for k in {base["variant"], best["variant"], *( [hedged["variant"]] if hedged else [])}}}
        pl.DataFrame(grid, infer_schema_length=None).write_parquet(OUT / f"grid_s1_{grp}.parquet")
    _dump("s1", res)
    return res


def _brief(row: dict[str, Any]) -> str:
    f = " | ".join(f"{sp}: {row.get(f'{sp}_structs', 0)}/{row.get(f'{sp}_events', 0)} c={row.get(f'{sp}_c')} "
                   f"t={row.get(f'{sp}_t')} $={row.get(f'{sp}_per_dollar')} S={row.get(f'{sp}_unit_sharpe')}"
                   for sp in SPLITS if row.get(f"{sp}_structs"))
    return f"{row['variant']}: {f}"


# ---- S2: vega trade on implied vs forecast width ------------------------------------------------------------
def sport_family(series: str) -> str:
    m = re.match(r"^KX(NCAAF|NFL|MLB|WNBA|NBA|NHL)", series)
    lg = m.group(1) if m else "other"
    return lg + ("-part" if re.search(r"\d[HQ]|F5", series) else "")


def _width_ll(theta: np.ndarray, cell: np.ndarray, z: np.ndarray, zl: np.ndarray, zh: np.ndarray, k: int) -> np.ndarray:
    mu = theta[:k][cell]
    a = theta[k:2 * k][cell]
    lam = np.exp(a + theta[2 * k] * z)
    p = ndtr((zh - mu) / lam) - ndtr((zl - mu) / lam)
    return np.log(np.clip(p, 1e-12, None))


def fit_width(sn: pl.DataFrame) -> dict[str, Any]:
    """Dev interval-censored MLE of (X_T - F)/sigma_ATM ~ N(mu_c, lambda^2), log lambda = a_c + b z,
    cells c = family x lead (families with < 40 dev snapshots pooled into ``other``), z = log
    sigma_ATM minus its dev mean in the cell (b < 0: unusually wide markets are too wide)."""
    d = sn.filter(pl.col("split") == "dev")
    fam_n = d.group_by("family").len()
    keep = set(fam_n.filter(pl.col("len") >= 40)["family"].to_list())
    cells = sorted({f"{f if f in keep else 'other'}|{h}" for f, h in zip(d["family"], d["lead_h"])})
    cix = {c: i for i, c in enumerate(cells)}
    lmean = {c: float(np.mean(v)) for c, v in _group_vals(d, keep, "lsig").items()}
    fit_d = _width_design(d, keep, cix, lmean)
    k = len(cells)
    th0 = np.zeros(2 * k + 1)
    res = minimize(lambda t: -_width_ll(t, *fit_d, k).sum(), th0, method="L-BFGS-B")
    return {"cells": cells, "keep": sorted(keep), "lmean": lmean, "theta": res.x.tolist(), "k": k,
            "converged": bool(res.success), "n": d.height}


def _cell(fam: str, lead: int, keep: set) -> str:
    return f"{fam if fam in keep else 'other'}|{lead}"


def _group_vals(d: pl.DataFrame, keep: set, col: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for f, h, v in zip(d["family"], d["lead_h"], d[col]):
        out.setdefault(_cell(f, h, keep), []).append(v)
    return out


def _width_design(d: pl.DataFrame, keep: set, cix: dict, lmean: dict) -> tuple[np.ndarray, ...]:
    """(cell index or -1, z, z_lo, z_hi); z uses the cell's dev mean of log sigma_ATM (or the lead's)."""
    cells = [_cell(f, h, keep) for f, h in zip(d["family"], d["lead_h"])]
    cell = np.array([cix.get(c, -1) for c in cells])
    by_lead: dict[str, list[float]] = {}
    for c, v in lmean.items():
        by_lead.setdefault(c.split("|")[1], []).append(v)
    z = np.array([v - lmean.get(c, float(np.mean(by_lead.get(c.split("|")[1], [v])))) for v, c in zip(d["lsig"], cells)])
    return cell, z, d["z_lo"].to_numpy(), d["z_hi"].to_numpy()


def width_ratio(sn: pl.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    """Forecast sigma_f / sigma_ATM for every snapshot of ``sn``; a cell without dev data gets the
    mean intercept of its lead."""
    cix = {c: i for i, c in enumerate(fit["cells"])}
    cell, z, _, _ = _width_design(sn, set(fit["keep"]), cix, fit["lmean"])
    th, k = np.array(fit["theta"]), fit["k"]
    a = th[k:2 * k]
    lead_mean = {}
    for i, c in enumerate(fit["cells"]):
        lead_mean.setdefault(c.split("|")[1], []).append(a[i])
    leads = [str(h) for h in sn["lead_h"].to_list()]
    ai = np.array([a[c] if c >= 0 else float(np.mean(lead_mean.get(h, [0.0]))) for c, h in zip(cell, leads)])
    return np.exp(ai + th[2 * k] * z)


def sports_snapshots(grp: str) -> pl.DataFrame:
    sn = pl.read_parquet(SMILE / "snapshots.parquet").filter((pl.col("group") == grp) & pl.col("lead_h").is_in(SP_LEADS))
    sn = sn.filter(pl.col("z_lo").is_not_null() & pl.col("z_hi").is_not_null() & (pl.col("sigma_atm") > 0))
    return sn.with_columns(pl.col("series_ticker").map_elements(sport_family, return_dtype=pl.String).alias("family"),
                           pl.col("sigma_atm").log().alias("lsig"),
                           pl.when(pl.col("t_ref") < HOLDOUT).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"))


def wx_ratios() -> pl.DataFrame:
    """(ladder, lead_h, r_nbm, r_shrink) from the vol_ladders weather panel (dev-fitted sigma models)."""
    frames = [pl.read_parquet(VL / "wx_snapshots.parquet")]
    if (NOV / "wx_snapshots.parquet").exists():
        frames.append(pl.read_parquet(NOV / "wx_snapshots.parquet"))
    sn = pl.concat([f.select("ladder", "lead_h", "r_nbm", "r_shrink") for f in frames])
    return sn.unique(["ladder", "lead_h"], keep="first")


S2_WX_GRID = [{"lead": lead, "side": side, "thr": thr} for lead in (24, 16, 12)
              for side, thrs in (("short", (0.85, 0.95)), ("long", (1.05, 1.15))) for thr in thrs]
S2_SP_GRID = [{"lead": lead, "side": side, "thr": d} for lead in SP_LEADS for side in ("short", "long")
              for d in (0.05, 0.10, 0.15)]


def step_s2() -> dict[str, Any]:
    """Short vega when the forecast width ratio r = sigma_f / sigma_ATM is low, long when high;
    delta-neutral at entry (rule ``entry``), the S1 dev-chosen band of the group, hold to settlement."""
    s1 = json.loads((OUT / "s1.json").read_text())
    res: dict[str, Any] = {}
    for grp in GROUPS:
        band_name = (s1.get(grp, {}).get("entry_chosen") or "band=far").split("band=")[1].split("|")[0]
        band = BANDS[band_name]
        paths = load_paths(grp)
        if grp == "wx_high":
            rt = wx_ratios()
            rmap = {(a, b): c for a, b, c in zip(rt["ladder"], rt["lead_h"], rt["r_nbm"]) if c is not None}
            grid_spec = S2_WX_GRID
            fit_info: dict[str, Any] = {"model": "vol_ladders NBM sigma (dev-fitted, wx_sigma.json)"}
        else:
            sn = sports_snapshots(grp)
            fit = fit_width(sn)
            sn = sn.with_columns(pl.Series("r", width_ratio(sn, fit)))
            rmap = {(a, b): c for a, b, c in zip(sn["ladder"], sn["lead_h"], sn["r"])}
            grid_spec = S2_SP_GRID
            lam = {c: round(math.exp(fit["theta"][fit["k"] + i]), 3) for i, c in enumerate(fit["cells"])}
            fit_info = {"model": "family x lead interval MLE + shrink on log sigma_ATM", "b": round(fit["theta"][-1], 3),
                        "lambda_cell": lam, "n_dev": fit["n"],
                        "r_quantiles": {sp: [round(float(x), 3) for x in np.quantile(g["r"].to_numpy(), [0.1, 0.5, 0.9])]
                                        for (sp,), g in sn.group_by("split")}}
        grid, frames = [], []
        for v in grid_spec:
            short = v["side"] == "short"

            def sel(p: LadderPath, lead: int, v: dict = v) -> bool:
                r = rmap.get((p.ladder, lead))
                if r is None or not np.isfinite(r):
                    return False
                return r <= (v["thr"] if grp == "wx_high" else 1 - v["thr"]) if v["side"] == "short" else \
                    r >= (v["thr"] if grp == "wx_high" else 1 + v["thr"])

            es = entries_for(paths, v["lead"], band, short=short, select=sel)
            name = f"lead={v['lead']}|{v['side']}|r{'<=' if short else '>='}{v['thr'] if grp == 'wx_high' else (1 - v['thr'] if short else 1 + v['thr']):.2f}"
            row, df = evaluate(es, RULES["entry"], {"variant": name, **v, "band": band_name})
            grid.append(row)
            frames.append(df)
            print(grp, _brief(row), flush=True)
        best = choose(grid)
        trades = pl.concat([f for f in frames if f.height], how="diagonal_relaxed") if any(f.height for f in frames) else pl.DataFrame()
        if trades.height:
            trades.write_parquet(OUT / f"trades_s2_{grp}.parquet")
        res[grp] = {"fit": fit_info, "band": band_name, "grid": grid, "chosen": best["variant"] if best else None,
                    "months": months(trades.filter(pl.col("variant") == best["variant"])) if best else None}
        pl.DataFrame(grid, infer_schema_length=None).write_parquet(OUT / f"grid_s2_{grp}.parquet")
    _dump("s2", res)
    return res


# ---- S3: gamma scalping where realized F variance exceeds implied -----------------------------------------
S3_MAIN = ("wx_high", 12, 8)             # pre-declared: the strongest dev RV/IV window gets the full grid
S3_GRID = [{"band": b, "hedge": h, "exit": x} for b in BANDS for h in ("entry", "thr0.1", "thr0.25") for x in ("window", "settle")]
S3_SMALL = [{"band": b, "hedge": "thr0.1", "exit": "window"} for b in BANDS]
S3_MIN_IV = 0.10                         # implied variance released in the window, share of sigma_a^2


def step_s3() -> dict[str, Any]:
    """Long strangle at the window start, delta-hedged; exit at the window end (sell at the bid) or
    hold to settlement. Run for every group x window whose dev hourly and endpoint RV/IV both exceed
    1 with >= ``S3_MIN_IV`` implied variance released (see ``rv``): the full grid for ``S3_MAIN``,
    two variants elsewhere."""
    rv = json.loads((OUT / "rv.json").read_text())
    res: dict[str, Any] = {"eligible": []}
    for grp, rows in rv.items():
        for r in rows:
            if (r["split"] == "dev" and (r.get("hourly_ratio") or 0) > 1 and (r.get("end_ratio") or 0) > 1
                    and r["iv_mean"] >= S3_MIN_IV):
                res["eligible"].append({"group": grp, **r})
    for el in res["eligible"]:
        grp = el["group"]
        a, b = (int(x) for x in el["window"].split("->"))
        spec = S3_GRID if (grp, a, b) == S3_MAIN else S3_SMALL
        paths = load_paths(grp)
        grid, frames = [], []
        for v in spec:
            es = entries_for(paths, a, BANDS[v["band"]], short=False, exit_lead=b if v["exit"] == "window" else None)
            name = f"lead={a}->{b}|band={v['band']}|hedge={v['hedge']}|exit={v['exit']}"
            row, df = evaluate(es, RULES[v["hedge"]], {"variant": name, "lead": a, "exit_lead": b, **v})
            grid.append(row)
            frames.append(df)
            print(grp, _brief(row), flush=True)
        best = choose(grid)
        trades = pl.concat([f for f in frames if f.height], how="diagonal_relaxed")
        key = f"{grp}|{a}->{b}"
        trades.write_parquet(OUT / f"trades_s3_{grp}_{a}-{b}.parquet")
        res[key] = {"group": grp, "window": f"{a}->{b}", "grid": grid, "chosen": best["variant"] if best else None,
                    "months": months(trades.filter(pl.col("variant") == best["variant"])) if best else None}
        pl.DataFrame(grid, infer_schema_length=None).write_parquet(OUT / f"grid_s3_{grp}_{a}-{b}.parquet")
    _dump("s3", res)
    return res


# ---- capped book -------------------------------------------------------------------------------------------
LEDGER_SCHEMA = {"strategy": pl.String, "variant": pl.String, "split": pl.String, "struct_id": pl.String,
                 "event_ticker": pl.String, "ticker": pl.String, "side": pl.String, "role": pl.String, "n": pl.Int64,
                 "t_in": pl.Datetime("us", "UTC"), "cost_usd": pl.Float64, "settle_ts": pl.Datetime("us", "UTC"),
                 "payoff_usd": pl.Float64}


class Book:
    """Chronological capped book over many structures (see module docstring)."""

    def __init__(self, event_cap: float = EVENT_CAP, open_cap: float = OPEN_CAP, bankroll: float = BANKROLL) -> None:
        self.event_cap, self.open_cap, self.bankroll = event_cap, open_cap, bankroll
        self.active: list[tuple[Position, str]] = []
        self.done: list[tuple[Position, str, str]] = []
        self.realized = 0.0
        self.now: datetime | None = None
        self.skipped = 0

    def _released(self, pos: Position, lot: Lot) -> bool:
        t = pos.release_time(lot)
        return t is not None and t <= self.now

    def retire(self) -> None:
        keep = []
        for pos, ev in self.active:
            if pos.finished and all(self._released(pos, x) for x in pos.lots):
                self.realized += pos.pnl()
            else:
                keep.append((pos, ev))
        self.active = keep

    def room(self, ev: str) -> float:
        open_tot = open_ev = real = 0.0
        for pos, e in self.active:
            for x in pos.lots:
                if self._released(pos, x):
                    real += x.pnl()
                else:
                    open_tot += x.cost()
                    if e == ev:
                        open_ev += x.cost()
        return min(self.event_cap - open_ev, self.open_cap - open_tot, self.bankroll + self.realized + real - open_tot)

    def run(self, entries: list[Entry], rule: Rule, strategy: str, variant: str) -> pl.DataFrame:
        hedged = rule.kind != "none"
        heap: list[tuple] = []
        seq = itertools.count()
        for e in entries:
            heapq.heappush(heap, (e.t0, 1, next(seq), "enter", e, None))
        thr = rule.thr if rule.kind == "thr" else None
        positions: list[tuple[Position, Entry]] = []
        while heap:
            t, _, _, kind, e, pos = heapq.heappop(heap)
            self.now = t
            self.retire()
            ev = e.p.event
            if kind == "enter":
                p = e.p
                unit = sum(p.held_quote(e.i0, j, s)[0] + fee_unit(round(p.held_quote(e.i0, j, s)[0], 4),
                                                                  float(p.fee_mult[j]), p.fee_type[j]) for j, s in e.core)
                r = self.room(ev) * ((1 - HEDGE_RESERVE) if hedged else 1.0)
                n = math.floor(r / unit + EPS) if unit > 0 else 0
                for j, _ in e.core:
                    v = p.vol[e.i0, j]
                    n = min(n, math.floor(VOL_SHARE * v + EPS) if np.isfinite(v) else 0)
                while n >= 1:
                    tot = 0.0
                    for j, s in e.core:
                        px = p.held_quote(e.i0, j, s)[0]
                        tot += n * px + kalshi_fee(n, px, fee_type=p.fee_type[j] or "quadratic",
                                                   multiplier=float(p.fee_mult[j]), conservative=True)
                    if tot <= r + EPS:
                        break
                    n -= 1
                if n < 1:
                    self.skipped += 1
                    continue
                pos = Position(p, e.core, units=float(n), integer=True, room=lambda ev=ev: self.room(ev))
                self.active.append((pos, ev))
                if not pos.enter(e.i0):
                    self.active.pop()
                    self.skipped += 1
                    continue
                positions.append((pos, e))
                if hedged:
                    pos.hedge(e.i0)
                last = (e.exit_i - 1) if e.exit_i is not None else len(p.hours) - 1
                steps = rule.hours(e.i0, last) if hedged else []
                pos.steps = steps + ([e.exit_i] if e.exit_i is not None else [])
                pos.k = 0
                self._next(heap, seq, pos, e)
            else:
                i = pos.steps[pos.k]
                if e.exit_i is not None and i == e.exit_i:
                    pos.exit(i)
                else:
                    pos.hedge(i, thr)
                pos.k += 1
                self._next(heap, seq, pos, e)
        rows = []
        for pos, e in positions:
            for x in pos.lots:
                won = x.payoff is not None
                rows.append({"strategy": strategy, "variant": variant, "split": e.p.split, "struct_id": f"{strategy}|{e.sid}",
                             "event_ticker": e.p.event, "ticker": e.p.tickers[x.leg], "side": x.side, "role": x.role,
                             "n": round(x.q), "t_in": e.p.hours[x.i_in], "cost_usd": round(x.cost(), 6),
                             "settle_ts": pos.release_time(x), "payoff_usd": round(x.value(), 6) if won or x.i_out is not None else None})
        return pl.DataFrame(rows, schema=LEDGER_SCHEMA)

    def _next(self, heap: list, seq: Any, pos: Position, e: Entry) -> None:
        if pos.k < len(pos.steps):
            heapq.heappush(heap, (e.p.hours[pos.steps[pos.k]], 0, next(seq), "step", e, pos))
        else:
            pos.settle()
            pos.finished = True


def ledger_summary(led: pl.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (sp,), g in led.group_by("split"):
        days = max((g["settle_ts"].max() - g["t_in"].min()).total_seconds() / 86400, 1.0)
        pnl = float((g["payoff_usd"] - g["cost_usd"]).sum())
        out[sp] = {"lots": g.height, "structs": g["struct_id"].n_unique(), "events": g["event_ticker"].n_unique(),
                   "hedge_lots": int((g["role"] == "hedge").sum()), "pnl_usd": round(pnl, 2),
                   "usd_week": round(pnl / days * 7, 2), "mean_n_core": round(float(g.filter(pl.col("role") == "core")["n"].mean() or 0), 2)}
    return out


def book_variant(strategy: str, grp: str, variant: dict[str, Any], scale: str = "") -> dict[str, Any]:
    """Run the capped book for one grid variant and write its ledger."""
    paths = load_paths(grp)
    rule = RULES[variant["hedge"]]
    if strategy.startswith("s1"):
        es = entries_for(paths, variant["lead"], BANDS[variant["band"]])
    elif strategy.startswith("s3"):
        es = entries_for(paths, variant["lead"], BANDS[variant["band"]], short=False,
                         exit_lead=variant["exit_lead"] if variant["exit"] == "window" else None)
    else:
        raise ValueError(strategy)
    return _book(es, rule, strategy, grp, variant["variant"], scale)


def _book(es: list[Entry], rule: Rule, strategy: str, grp: str, variant: str, scale: str) -> dict[str, Any]:
    caps = (10.0, 100.0) if scale else (EVENT_CAP, OPEN_CAP)
    bk = Book(*caps)
    led = bk.run(sorted(es, key=lambda e: (e.t0, e.sid)), rule, f"{strategy}_{grp}", variant + scale)
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9.=-]+", "_", variant + scale).strip("_")
    path = OUT / "ledgers" / f"ledger_{strategy}_{grp}_{safe}.parquet"
    led.write_parquet(path)
    return {"path": str(path), "skipped_entries": bk.skipped, **ledger_summary(led)}


def step_book() -> dict[str, Any]:
    """Capped-book ledgers for the dev-chosen variants (and the unhedged / hedged S1 pair)."""
    out: dict[str, Any] = {}
    s1 = json.loads((OUT / "s1.json").read_text())
    for grp, r in s1.items():
        if "grid" not in r or not r.get("entry_chosen"):
            continue
        g = {x["variant"]: x for x in r["grid"]}
        for name in dict.fromkeys([r["entry_chosen"], r.get("hedged_chosen")]):
            if name is None:
                continue
            for scale in ("", "_cap10-100"):
                out[f"s1|{grp}|{name}{scale}"] = book_variant("s1", grp, g[name], scale)
                print(f"s1 {grp} {name}{scale}", out[f"s1|{grp}|{name}{scale}"], flush=True)
    s2 = json.loads((OUT / "s2.json").read_text()) if (OUT / "s2.json").exists() else {}
    for grp, r in s2.items():
        if not r.get("chosen"):
            continue
        v = {x["variant"]: x for x in r["grid"]}[r["chosen"]]
        out[f"s2|{grp}|{r['chosen']}"] = book_s2(grp, v, r["band"])
        print(f"s2 {grp}", out[f"s2|{grp}|{r['chosen']}"], flush=True)
    s3 = json.loads((OUT / "s3.json").read_text()) if (OUT / "s3.json").exists() else {}
    for key, r in s3.items():
        if key == "eligible" or not isinstance(r, dict) or not r.get("chosen"):
            continue
        v = {x["variant"]: x for x in r["grid"]}[r["chosen"]]
        out[f"s3|{key}|{r['chosen']}"] = book_variant("s3", r["group"], v)
        print(f"s3 {key}", out[f"s3|{key}|{r['chosen']}"], flush=True)
    _dump("book", out)
    return out


def book_s2(grp: str, v: dict[str, Any], band_name: str) -> dict[str, Any]:
    paths = load_paths(grp)
    if grp == "wx_high":
        rt = wx_ratios()
        rmap = {(a, b): c for a, b, c in zip(rt["ladder"], rt["lead_h"], rt["r_nbm"]) if c is not None}
        lo_thr, hi_thr = v["thr"], v["thr"]
    else:
        sn = sports_snapshots(grp)
        sn = sn.with_columns(pl.Series("r", width_ratio(sn, fit_width(sn))))
        rmap = {(a, b): c for a, b, c in zip(sn["ladder"], sn["lead_h"], sn["r"])}
        lo_thr, hi_thr = 1 - v["thr"], 1 + v["thr"]
    short = v["side"] == "short"

    def sel(p: LadderPath, lead: int) -> bool:
        r = rmap.get((p.ladder, lead))
        return r is not None and np.isfinite(r) and (r <= lo_thr if short else r >= hi_thr)

    es = entries_for(paths, v["lead"], BANDS[band_name], short=short, select=sel)
    return _book(es, RULES["entry"], "s2", grp, v["variant"], "")


# ---- metrics and report ------------------------------------------------------------------------------------
def step_metrics() -> dict[str, Any]:
    """vol_metrics on every ledger (whole period) and per split."""
    from albert.research import vol_metrics as vm

    vm.run([str(OUT / "ledgers" / "ledger_*.parquet")], OUT / "metrics")
    out: dict[str, Any] = {}
    for path in sorted((OUT / "ledgers").glob("ledger_*.parquet")):
        led = pl.read_parquet(path)
        if not led.height:
            continue
        marks = vm.load_marks(led["ticker"].unique().to_list())
        per = {"all": vm.ledger_metrics(led, marks)}
        for sp in SPLITS:
            g = led.filter(pl.col("split") == sp)
            if g.height >= 3:
                per[sp] = vm.ledger_metrics(g, marks)
        out[path.name] = per
        print(path.name, {k: (v["usd_per_week"], v["sharpe"], v["sharpe_ci90"]) for k, v in per.items()}, flush=True)
    _dump("metrics_by_split", out)
    return out


def step_report() -> str:
    """Compact markdown summary of the grids and ledgers (``summary.md``)."""
    lines = ["# vol_greeks summary (generated)", ""]
    for name in ("s1", "s2", "s3"):
        p = OUT / f"{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for grp, x in r.items():
            if not isinstance(x, dict) or "grid" not in x:
                continue
            lines.append(f"## {name} {grp}  (chosen: {x.get('chosen')})")
            lines.append("| variant | dev s/ev/c/t/$ | holdout s/ev/c/t/$ | nov s/ev/c/t/$ | unit Sharpe d/h/n | attribution dev (prem/jump/hedge/costs c) |")
            lines.append("|---|---|---|---|---|---|")
            for g in x["grid"]:
                cells = []
                for sp in SPLITS:
                    cells.append(f"{g.get(f'{sp}_structs', 0)}/{g.get(f'{sp}_events', 0)}/{g.get(f'{sp}_c')}/{g.get(f'{sp}_t')}/{g.get(f'{sp}_per_dollar')}")
                us = "/".join(str(g.get(f"{sp}_unit_sharpe")) for sp in SPLITS)
                att = "/".join(str(g.get(f"dev_{c}_c")) for c in ("premium", "jump", "hedge_mid")) + \
                    f"/{(g.get('dev_core_cost_c') or 0) + (g.get('dev_hedge_cost_c') or 0):.2f}"
                lines.append(f"| {g['variant']} | " + " | ".join(cells) + f" | {us} | {att} |")
            lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def _dump(name: str, obj: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))


def main(argv: list[str]) -> None:
    if not argv or argv[0] != "run":
        print(__doc__)
        return
    step = argv[1] if len(argv) > 1 else "all"
    steps = {"panel": step_panel, "rv": step_rv, "s1": step_s1, "s2": step_s2, "s3": step_s3, "book": step_book,
             "metrics": step_metrics, "report": step_report}
    for name, fn in steps.items():
        if step in (name, "all"):
            out = fn()
            if name == "report":
                print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
