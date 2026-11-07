"""Sub-minute lead-lag: Polymarket intl (data only) vs Kalshi, and between sibling Kalshi markets.

Everything runs on the paired recordings (``albert record --paired-days``): for each Kalshi game
moneyline we rebuild its top of book (bid, ask and displayed sizes) and the matching Polymarket
token's top of book (complement applied for the other team), each at our receive time and at the
venue's own server timestamp (Kalshi ``ts_ms`` on deltas, Polymarket ``timestamp``).

Analyses (``fast_lead_lag`` runs them all and writes one report):

1. **Feed paths.** Receive time minus server timestamp per venue. Both are read on one clock, so
   the difference between venues is the relative path delay; a lead shorter than its spread is
   noise.
2. **Cross-correlation** of 1-second mid changes at lags from -30 s to +30 s.
3. **Event study.** A Polymarket move (mid change of at least ``k`` within 1 s, Polymarket spread
   at most 3c, de-clustered) and Kalshi's signed mid response at 0.1 s to 60 s. The lag is the
   first time Kalshi has moved half of ``k`` in the same direction. Staleness: at our latency
   (the decision is made at our receive time of the Polymarket frame), is the Kalshi quote on the
   side to take still at or behind its pre-move price, with displayed size of at least 1?
4. **Taker.** At receive + latency, buy the stale Kalshi side when the Polymarket mid after the
   move is at least ``edge`` beyond the Kalshi ask (or below the bid), paying the ask and the taker
   fee 0.07·P·(1−P) (exact to $0.0001, this account's balance precision). Exit at an executable
   price (sell at the bid, also paying the taker fee) after 10, 30 or 60 s. Standard errors are
   clustered by game.
5. **Maker defence.** Each Kalshi trade's markout from the maker's side, split by whether
   Polymarket moved in the taker's direction in the preceding 5 s or 30 s: the adverse fills a
   ``pm_pull`` would have avoided.
6. **Siblings.** Team A and team B moneylines of one game (YES_A ≈ 1 − YES_B on two-outcome games):
   after a move in one, is the other's quote stale at our latency, and would taking it pay?

Regime: a market is *in play* from its scheduled start (``expected_expiration_time`` minus a
league duration) until ``expected_expiration_time`` plus an hour, else *pre-game*.
"""

import bisect
import heapq
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.backtest.l2 import (
    EPS,
    Event,
    SimConfig,
    Simulator,
    iter_events,
    load_pairs,
    pm_files,
    stream_files,
)
from albert.stream import read_stream

HORIZONS_S = (0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)
LATENCIES_MS = (50, 100, 300)
EXITS_S = (10, 30, 60)
GAME_HOURS = {"nba": 2.6, "nhl": 2.6, "cbb": 2.2, "cfb": 3.6, "nfl": 3.4}


def taker_fee(p: float) -> float:
    """Kalshi taker fee for one contract, rounded up to $0.0001."""
    return math.ceil(0.07 * p * (1 - p) * 1e4 - 1e-9) / 1e4


# ---- series ---------------------------------------------------------------------------------
@dataclass
class Book:
    """Top-of-book history of one market: arrays aligned by index, appended on any change."""
    t: list[int] = field(default_factory=list)        # our receive time, ns
    ts: list[int | None] = field(default_factory=list)   # venue server time, ms
    bid: list[float | None] = field(default_factory=list)
    ask: list[float | None] = field(default_factory=list)
    bsz: list[float] = field(default_factory=list)
    asz: list[float] = field(default_factory=list)

    def add(self, t: int, ts: int | None, bid, ask, bsz=0.0, asz=0.0) -> None:
        if self.bid and (self.bid[-1], self.ask[-1], self.bsz[-1], self.asz[-1]) == (bid, ask, bsz, asz):
            return
        self.t.append(t)
        self.ts.append(ts)
        self.bid.append(bid)
        self.ask.append(ask)
        self.bsz.append(bsz)
        self.asz.append(asz)

    def at(self, t: int) -> int:
        """Index of the state live at ``t`` (-1 if none yet)."""
        return bisect.bisect_right(self.t, t) - 1

    def mid(self, i: int) -> float | None:
        if i < 0 or self.bid[i] is None or self.ask[i] is None:
            return None
        return (self.bid[i] + self.ask[i]) / 2

    def mid_at(self, t: int) -> float | None:
        return self.mid(self.at(t))


def _pm_items(row: dict[str, Any]) -> Iterator[tuple[str, int | None, float | None, float | None]]:
    """(asset_id, server_ms, best_bid, best_ask) carried by one recorded Polymarket frame."""
    try:
        d = json.loads(row["frame"])
    except ValueError:
        return
    for it in d if isinstance(d, list) else [d]:
        if not isinstance(it, dict):
            continue
        ts = it.get("timestamp")
        ts = int(ts) if ts not in (None, "") else None
        if it.get("event_type", "book") == "book" and "bids" in it:
            bids = [float(x["price"]) for x in it.get("bids") or []]
            asks = [float(x["price"]) for x in it.get("asks") or []]
            yield it["asset_id"], ts, max(bids, default=None), min(asks, default=None)
        elif it.get("event_type") == "price_change":
            for ch in it.get("price_changes") or []:
                bb, ba = ch.get("best_bid"), ch.get("best_ask")
                yield (ch["asset_id"], ts, float(bb) if bb not in (None, "") else None,
                       float(ba) if ba not in (None, "") else None)


def pm_books(rows: Iterator[dict[str, Any]], pairs: dict[str, dict[str, Any]],
             paths: list[tuple[int, int]] | None = None) -> dict[str, Book]:
    """Polymarket top of book per paired Kalshi ticker, in that ticker's YES terms."""
    by_token: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for t, row in pairs.items():
        by_token[row["token_id"]].append((t, bool(row.get("complement"))))
    out: dict[str, Book] = defaultdict(Book)
    for row in rows:
        for asset, ts, bid, ask in _pm_items(row):
            if asset not in by_token:
                continue
            if paths is not None and ts is not None:
                paths.append((row["recv_ns"], ts))
            for ticker, comp in by_token[asset]:
                b, a = ((None if ask is None else round(1 - ask, 6), None if bid is None else round(1 - bid, 6))
                        if comp else (bid, ask))
                out[ticker].add(row["recv_ns"], ts, b, a)
    return out


def kalshi_books(events: Iterator[Event], markets: set[str],
                 paths: list[tuple[int, int]] | None = None) -> tuple[dict[str, Book], list[Event]]:
    """Kalshi top of book (with displayed sizes) per market, and the trades (labelled)."""
    sim = Simulator(SimConfig(writes_per_s=None))
    out: dict[str, Book] = defaultdict(Book)
    trades: list[Event] = []
    for ev in events:
        if ev.market not in markets:
            continue
        sim.apply(ev)
        ts = ev.msg.get("ts_ms")
        if paths is not None and ev.msg.get("ts_ms") is not None:
            paths.append((ev.recv_ns, int(ev.msg["ts_ms"])))
        if ev.kind == "trade":
            trades.append(ev)
            continue
        if ev.market not in sim.synced:
            continue
        q = sim.markets[ev.market].quote()
        out[ev.market].add(ev.recv_ns, int(ts) if ts is not None else None, q["yes_bid"], q["yes_ask"],
                           q["yes_bid_size"] or 0.0, q["yes_ask_size"] or 0.0)
    return out, trades


@dataclass
class Data:
    pairs: dict[str, dict[str, Any]]
    k: dict[str, Book]
    p: dict[str, Book]
    trades: list[Event]
    k_paths: list[tuple[int, int]]
    p_paths: list[tuple[int, int]]
    start_ns: dict[str, int | None]       # scheduled start per Kalshi ticker
    end_ns: int


def load(roots: list[Path], start: str | None, end: str | None,
         starts: dict[str, int | None] | None = None) -> Data:
    """Read paired recordings under each root (later roots skip hours already read)."""
    pairs: dict[str, dict[str, Any]] = {}
    for r in roots:
        pairs.update(load_pairs(r))
    k_paths: list[tuple[int, int]] = []
    p_paths: list[tuple[int, int]] = []
    k: dict[str, Book] = {}
    p: dict[str, Book] = {}
    trades: list[Event] = []
    seen_hours: set[str] = set()
    for r in roots:
        kf = [f for f in stream_files(r, start, end) if f"{f.parent.name}T{f.name[:2]}" not in seen_hours]
        pf = [f for f in pm_files(r, start, end) if f"{f.parent.name}T{f.name[:2]}" not in seen_hours]
        if not kf or not pf:
            continue
        hours = {f"{f.parent.name}T{f.name[:2]}" for f in kf}
        pf = [f for f in pf if f"{f.parent.name}T{f.name[:2]}" in hours]
        kb, tr = kalshi_books(iter_events(kf, defaultdict(int)), set(pairs), k_paths)
        rows = heapq.merge(*[read_stream(f) for f in pf], key=lambda x: x["recv_ns"])
        pb = pm_books(rows, pairs, p_paths)
        for src, dst in ((kb, k), (pb, p)):
            for t, b in src.items():
                dst.setdefault(t, Book())
                for i in range(len(b.t)):
                    dst[t].add(b.t[i], b.ts[i], b.bid[i], b.ask[i], b.bsz[i], b.asz[i])
        trades += tr
        seen_hours |= hours
    end_ns = max((b.t[-1] for b in k.values() if b.t), default=0)
    return Data(pairs, k, p, trades, k_paths, p_paths, starts or {}, end_ns)


def scheduled_starts(pairs: dict[str, dict[str, Any]], market_info: dict[str, dict[str, Any]]) -> dict[str, int | None]:
    """Scheduled start = expected_expiration_time minus the league's typical game length."""
    out: dict[str, int | None] = {}
    for t, row in pairs.items():
        m = market_info.get(t) or {}
        exp = m.get("expected_expiration_time")
        league = row.get("pm_slug", "").split("-")[0]
        if not exp or league not in GAME_HOURS:
            out[t] = None
            continue
        dt = datetime.fromisoformat(exp) - timedelta(hours=GAME_HOURS[league])
        out[t] = int(dt.timestamp() * 1e9)
    return out


def regime(d: Data, ticker: str, t_ns: int) -> str:
    s = d.start_ns.get(ticker)
    if s is None:
        return "unknown"
    league = d.pairs[ticker].get("pm_slug", "").split("-")[0]
    end = s + int((GAME_HOURS.get(league, 3.0) + 1.0) * 3600e9)
    return "in_play" if s <= t_ns <= end else "pre_game"


# ---- statistics -------------------------------------------------------------------------------
def clustered(values: list[float], clusters: list[str]) -> tuple[float | None, float | None, int, int]:
    """Mean, t-stat clustered by ``clusters``, observations, clusters."""
    if not values:
        return None, None, 0, 0
    s: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    for v, c in zip(values, clusters):
        s[c] += v
        n[c] += 1
    N = len(values)
    mean = sum(values) / N
    k = len(s)
    if k < 2:
        return mean, None, N, k
    se = math.sqrt(k / (k - 1) * sum((s[c] - mean * n[c]) ** 2 for c in s)) / N
    return mean, (mean / se if se > 0 else None), N, k


def path_stats(paths: list[tuple[int, int]]) -> dict[str, float | None]:
    """Receive time minus server time (ms): median and spread. One clock reads both venues, so
    the median difference between venues is their relative path delay (plus server clock skew)."""
    if not paths:
        return {"n": 0, "p10": None, "median": None, "p90": None}
    x = np.array([r / 1e6 - s for r, s in paths], dtype=float)
    return {"n": int(x.size), "p10": float(np.percentile(x, 10)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90))}


def games(d: Data) -> dict[str, list[str]]:
    """pm_slug -> Kalshi tickers recorded for that game (both teams when available)."""
    out: dict[str, list[str]] = defaultdict(list)
    for t, row in d.pairs.items():
        if t in d.k and d.k[t].t:
            out[row.get("pm_slug", t)].append(t)
    return out


def xcorr(d: Data, bin_s: float = 1.0, max_lag_s: int = 30) -> pl.DataFrame:
    """corr(ΔKalshi(t), ΔPM(t - lag)) on ``bin_s`` bins over periods both sides are two-sided;
    one market per game (the team whose YES is the Polymarket token itself, else the first)."""
    num = defaultdict(float)
    den_k = defaultdict(float)
    den_p = defaultdict(float)
    nobs = defaultdict(int)
    step = int(bin_s * 1e9)
    for ts in games(d).values():
        t = min(ts, key=lambda x: bool(d.pairs[x].get("complement")))
        kb, pb = d.k.get(t), d.p.get(t)
        if not kb or not pb or not kb.t or not pb.t:
            continue
        t0, t1 = max(kb.t[0], pb.t[0]), min(kb.t[-1], pb.t[-1])
        if t1 - t0 < 600e9:
            continue
        grid = np.arange(t0, t1, step)
        km = np.array([kb.mid_at(int(g)) for g in grid], dtype=object)
        pm = np.array([pb.mid_at(int(g)) for g in grid], dtype=object)
        ok = np.array([a is not None and b is not None for a, b in zip(km, pm)])
        kf = np.where(ok, np.array([a if a is not None else 0.0 for a in km], dtype=float), np.nan)
        pf = np.where(ok, np.array([b if b is not None else 0.0 for b in pm], dtype=float), np.nan)
        dk, dp = np.diff(kf), np.diff(pf)
        for lag in range(-max_lag_s, max_lag_s + 1):
            L = int(lag / bin_s)
            if L >= 0:
                a, b = dk[L:], dp[:len(dp) - L]
            else:
                a, b = dk[:len(dk) + L], dp[-L:]
            m = ~(np.isnan(a) | np.isnan(b))
            num[lag] += float(np.sum(a[m] * b[m]))
            den_k[lag] += float(np.sum(a[m] ** 2))
            den_p[lag] += float(np.sum(b[m] ** 2))
            nobs[lag] += int(m.sum())
    rows = [{"lag_s": lag, "corr": num[lag] / math.sqrt(den_k[lag] * den_p[lag]) if den_k[lag] and den_p[lag] else None,
             "bins": nobs[lag]} for lag in sorted(num)]
    return pl.DataFrame(rows, infer_schema_length=None)


def pm_moves(pb: Book, k_move: float, win_ns: int = 10**9, gap_ns: int = 30 * 10**9,
             max_spread: float = 0.03) -> list[tuple[int, int, float, float, float]]:
    """(t_recv, ts_ms, direction, mid_before, mid_after): mid change >= k within ``win_ns``
    with the Polymarket spread at most ``max_spread`` before and after; one per ``gap_ns``."""
    out = []
    last = -(2**62)
    j = 0
    for i in range(len(pb.t)):
        m = pb.mid(i)
        if m is None or pb.ask[i] - pb.bid[i] > max_spread + EPS:
            continue
        while pb.t[j] < pb.t[i] - win_ns:
            j += 1
        if pb.t[i] - last < gap_ns:
            continue
        m0 = pb.mid(max(j - 1, 0)) if j > 0 else pb.mid(j)
        if m0 is None:
            continue
        i0 = max(j - 1, 0)
        if pb.ask[i0] is None or pb.bid[i0] is None or pb.ask[i0] - pb.bid[i0] > max_spread + EPS:
            continue
        dm = m - m0
        if abs(dm) >= k_move - EPS:
            out.append((pb.t[i], pb.ts[i], 1.0 if dm > 0 else -1.0, m0, m))
            last = pb.t[i]
    return out


def event_study(d: Data, k_move: float, *, leader: str = "pm") -> pl.DataFrame:
    """One row per leader move: the follower's signed mid response at each horizon, the first
    time it moved half of ``k_move`` (receive time and exchange time), and staleness and taker
    PnL at each latency. ``leader='pm'`` follows Polymarket into the same Kalshi market;
    ``leader='sibling'`` follows the other team's Kalshi market (YES_A ≈ 1 − YES_B)."""
    rows = []
    for slug, tickers in games(d).items():
        if leader == "pm":
            # one Kalshi market per game for the PM study (both teams carry the same PM token)
            plan = [(t, d.p.get(t), 1.0) for t in tickers[:1] if d.p.get(t) and d.p[t].t]
        else:
            if len(tickers) < 2:
                continue
            a, b = tickers[0], tickers[1]
            plan = [(b, d.k[a], -1.0), (a, d.k[b], -1.0)]
        for follower, lb, sign in plan:
            kb = d.k[follower]
            for t0, ts0, direction, m0, m1 in pm_moves(lb, k_move):
                if t0 > d.end_ns - 60e9:
                    continue
                dirn = direction * sign          # direction the follower should move
                fair = m1 if sign > 0 else 1 - m1
                base_i = kb.at(t0)
                base = kb.mid(base_i)
                if base is None:
                    continue
                row: dict[str, Any] = {"game": slug, "market": follower, "t_ns": t0, "dir": dirn,
                                       "leader_move": abs(m1 - m0), "regime": regime(d, follower, t0),
                                       "k_spread": kb.ask[base_i] - kb.bid[base_i],
                                       "fair_gap": dirn * (fair - base)}
                for h in HORIZONS_S:
                    m = kb.mid_at(t0 + int(h * 1e9))
                    row[f"resp_{h}s"] = None if m is None else dirn * (m - base)
                # first follower move of half the leader move
                lag = None
                lag_x = None
                i = base_i + 1
                while i < len(kb.t) and kb.t[i] <= t0 + 60e9:
                    m = kb.mid(i)
                    if m is not None and dirn * (m - base) >= k_move / 2 - EPS:
                        lag = (kb.t[i] - t0) / 1e9
                        if kb.ts[i] is not None and ts0 is not None:
                            lag_x = (kb.ts[i] - ts0) / 1e3
                        break
                    i += 1
                row["lag_s"] = lag
                row["lag_exchange_s"] = lag_x
                # staleness and taker PnL at each latency
                for lat in LATENCIES_MS:
                    j = kb.at(t0 + lat * 10**6)
                    px = kb.ask[j] if dirn > 0 else (None if kb.bid[j] is None else 1 - kb.bid[j])   # buy YES or NO
                    size = kb.asz[j] if dirn > 0 else kb.bsz[j]
                    pre = kb.ask[base_i] if dirn > 0 else (None if kb.bid[base_i] is None else 1 - kb.bid[base_i])
                    stale = px is not None and pre is not None and px <= pre + EPS and size >= 1 - EPS
                    row[f"stale_{lat}ms"] = stale
                    fair_side = fair if dirn > 0 else 1 - fair
                    row[f"edge_{lat}ms"] = None if px is None else fair_side - px
                    for ex in EXITS_S:
                        pnl = None
                        if px is not None and size >= 1 - EPS:
                            jx = kb.at(t0 + lat * 10**6 + int(ex * 1e9))
                            out_px = kb.bid[jx] if dirn > 0 else (None if kb.ask[jx] is None else 1 - kb.ask[jx])
                            if out_px is not None:
                                pnl = out_px - px - taker_fee(px) - taker_fee(out_px)
                        row[f"pnl_{lat}ms_{ex}s"] = pnl
                rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def maker_defence(d: Data, k_move: float = 0.01) -> pl.DataFrame:
    """Kalshi trades in paired markets: maker markout at 1/10/60 s (maker = the side that
    rested), and whether Polymarket moved >= ``k_move`` in the taker's direction within 5 s / 30 s
    before the trade (at our receive times)."""
    rows = []
    for ev in d.trades:
        t = ev.market
        kb, pb = d.k.get(t), d.p.get(t)
        if not kb or not pb or not pb.t:
            continue
        side = ev.msg.get("taker_outcome_side") or ev.msg.get("taker_side")
        sgn = 1.0 if side == "yes" else -1.0
        base = kb.mid_at(ev.recv_ns - 1)
        if base is None:
            continue
        row: dict[str, Any] = {"game": d.pairs[t].get("pm_slug", t), "market": t,
                               "regime": regime(d, t, ev.recv_ns),
                               "qty": float(ev.msg.get("count_fp") or 0)}
        for w in (5, 30):
            p_now, p_then = pb.mid_at(ev.recv_ns), pb.mid_at(ev.recv_ns - w * 10**9)
            mv = None if p_now is None or p_then is None else sgn * (p_now - p_then)
            row[f"pm_with_taker_{w}s"] = mv is not None and mv >= k_move - EPS
        price = float(ev.msg["yes_price_dollars"])
        for h in (1, 10, 60):
            if ev.recv_ns + h * 10**9 > d.end_ns:
                row[f"maker_mo_{h}s"] = None
                continue
            m = kb.mid_at(ev.recv_ns + h * 10**9)
            # maker sold YES at price when taker bought YES: maker PnL = price - mid
            row[f"maker_mo_{h}s"] = None if m is None else -sgn * (m - price)
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def summarize_events(f: pl.DataFrame) -> list[dict[str, Any]]:
    out = []
    if f.is_empty():
        return out
    for reg in ["all"] + sorted(set(f["regime"].to_list())):
        g = f if reg == "all" else f.filter(pl.col("regime") == reg)
        if g.is_empty():
            continue
        r: dict[str, Any] = {"regime": reg, "moves": g.height, "games": g["game"].n_unique()}
        for h in HORIZONS_S:
            c = g.filter(pl.col(f"resp_{h}s").is_not_null())
            mean, t, _, _ = clustered(c[f"resp_{h}s"].to_list(), c["game"].to_list())
            r[f"resp_{h}s_c"] = None if mean is None else round(mean * 100, 2)
            r[f"t_{h}s"] = None if t is None else round(t, 2)
        lags = g["lag_s"].drop_nulls()
        r["followed_60s"] = round(lags.len() / g.height, 3)
        r["lag_p25_s"] = float(lags.quantile(0.25)) if lags.len() else None
        r["lag_median_s"] = float(lags.median()) if lags.len() else None
        lx = g["lag_exchange_s"].drop_nulls()
        r["lag_exchange_median_s"] = float(lx.median()) if lx.len() else None
        for lat in LATENCIES_MS:
            r[f"stale_{lat}ms"] = round(g[f"stale_{lat}ms"].mean(), 3)
        out.append(r)
    return out


def summarize_taker(f: pl.DataFrame, edges=(0.0, 0.01, 0.02)) -> list[dict[str, Any]]:
    out = []
    if f.is_empty():
        return out
    for reg in ["all"] + sorted(set(f["regime"].to_list())):
        g0 = f if reg == "all" else f.filter(pl.col("regime") == reg)
        for lat in LATENCIES_MS:
            for e in edges:
                g = g0.filter(pl.col(f"edge_{lat}ms").is_not_null() & (pl.col(f"edge_{lat}ms") >= e - EPS))
                for ex in EXITS_S:
                    c = g.filter(pl.col(f"pnl_{lat}ms_{ex}s").is_not_null())
                    mean, t, n, k = clustered(c[f"pnl_{lat}ms_{ex}s"].to_list(), c["game"].to_list())
                    out.append({"regime": reg, "latency_ms": lat, "min_edge_c": e * 100, "exit_s": ex,
                                "trades": n, "games": k,
                                "pnl_c": None if mean is None else round(mean * 100, 2),
                                "t_by_game": None if t is None else round(t, 2)})
    return out


def summarize_defence(f: pl.DataFrame) -> list[dict[str, Any]]:
    out = []
    if f.is_empty():
        return out
    for w in (5, 30):
        for flag in (True, False):
            g = f.filter(pl.col(f"pm_with_taker_{w}s") == flag)
            r: dict[str, Any] = {"window_s": w, "pm_moved_with_taker": flag, "trades": g.height}
            for h in (1, 10, 60):
                c = g.filter(pl.col(f"maker_mo_{h}s").is_not_null())
                mean, t, _, k = clustered(c[f"maker_mo_{h}s"].to_list(), c["game"].to_list())
                r[f"maker_mo_{h}s_c"] = None if mean is None else round(mean * 100, 2)
                r[f"t_{h}s"] = None if t is None else round(t, 2)
                r["games"] = k
            out.append(r)
    return out


def market_info(tickers: list[str], cache: Path) -> dict[str, dict[str, Any]]:
    """Kalshi market metadata (public REST), cached as JSON at ``cache``."""
    import httpx

    have = json.loads(cache.read_text()) if cache.exists() else {}
    with httpx.Client(timeout=20) as http:
        for t in tickers:
            if t in have:
                continue
            r = http.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t}")
            if r.status_code == 200:
                m = r.json().get("market", {})
                have[t] = {k: m.get(k) for k in ("expected_expiration_time", "occurrence_datetime",
                                                 "close_time", "status", "result")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(have, indent=1))
    return have


def fast_lead_lag(s, *, start: str | None = None, end: str | None = None,
                  roots: list[Path] | None = None, moves=(0.01, 0.02, 0.03)) -> Path:
    """Run every analysis and write report ``research-lead-lag-fast``."""
    from albert.reports import Report

    roots = roots or [s.raw_dir / "stream"]
    pairs: dict[str, dict[str, Any]] = {}
    for r in roots:
        pairs.update(load_pairs(r))
    info = market_info(sorted(pairs), s.data_dir / "cache" / "leadlag_fast_markets.json")
    d = load(roots, start, end, scheduled_starts(pairs, info))
    rep = Report(s.reports_dir, "research-lead-lag-fast")
    k_hours = sorted({datetime.fromtimestamp(t / 1e9, UTC).strftime("%dT%H")
                      for b in d.k.values() for t in b.t[:1] + b.t[-1:]})
    rep.json("Run", {"roots": [str(r) for r in roots], "start": start, "end": end,
                     "pairs": len(pairs), "games_with_both_feeds": len([g for g, ts in games(d).items()
                                                                       if any(d.p.get(t) and d.p[t].t for t in ts)]),
                     "kalshi_updates": sum(len(b.t) for b in d.k.values()),
                     "pm_updates": sum(len(b.t) for b in d.p.values()), "trades": len(d.trades),
                     "hours_span": k_hours[:1] + k_hours[-1:]})
    rep.json("Feed paths (receive minus server time, ms)",
             {"kalshi": path_stats(d.k_paths), "polymarket": path_stats(d.p_paths)})
    rep.table("Cross-correlation of 1 s mid changes (positive lag = Polymarket leads)", xcorr(d))
    for k_move in moves:
        ev = event_study(d, k_move, leader="pm")
        rep.table(f"Polymarket move >= {k_move * 100:.0f}c in 1 s: Kalshi response (c, t by game)",
                  pl.DataFrame(summarize_events(ev), infer_schema_length=None))
        rep.table(f"Taker on stale Kalshi after Polymarket move >= {k_move * 100:.0f}c (net c/contract)",
                  pl.DataFrame(summarize_taker(ev), infer_schema_length=None))
        if not ev.is_empty():
            rep.table(f"Moves {k_move * 100:.0f}c", ev.drop("t_ns"), save_as=f"moves_pm_{int(k_move * 100)}c",
                      max_rows=10)
        sib = event_study(d, k_move, leader="sibling")
        rep.table(f"Sibling move >= {k_move * 100:.0f}c: other team's Kalshi response",
                  pl.DataFrame(summarize_events(sib), infer_schema_length=None))
        rep.table(f"Taker on stale sibling >= {k_move * 100:.0f}c (net c/contract)",
                  pl.DataFrame(summarize_taker(sib), infer_schema_length=None))
    md = maker_defence(d)
    rep.table("Maker markout of Kalshi trades by preceding Polymarket move (c, t by game)",
              pl.DataFrame(summarize_defence(md), infer_schema_length=None))
    return rep.save()
