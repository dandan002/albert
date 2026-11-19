"""Mentions bot budget: expected fill cost instead of resting collateral (user request 2026-11-19).

The live bot (``live_mentions``) caps its value at risk: resting NO bids + max(today's fill cost,
unsettled fill cost) <= $15, at most 2 events at once, so the worst case (every bid fills and
loses) is -$15. Only ~25% of bids fill, so most of that budget never trades. The user asked to
"take the expected number of fills and limit that to $15 instead. Going a bit over is acceptable if
more get filled", and to "dynamically manage s.t. if stuff overfills it can cancel orders; we want
as close to the limit deployed as possible". The budget studied here:

    E(t) = max(today's fill cost, unsettled fill cost) + sum over resting bids of p_rem(bid, t) x price

* postings (start - 12 h early phase, start - 6 h main phase, as live) take legs cheapest first while
  E stays <= the target; ``ceiling`` is a hard cap on resting collateral + max(today's, unsettled
  fill cost) (the live bot's ``at_risk``: the worst case if every resting bid fills and loses);
* after fills, if E > target, resting bids are cancelled (largest p_rem x price first) until E <=
  target; once realized fills alone reach the target every bid is cancelled, so realized fill cost
  exceeds the target only by fills between two checks;
* ``topup``: while an event's phase is open and E has room, legs that were cut (budget) are posted
  later at their posting price, if the remaining window is >= 1 h (the hourly replay re-uses the
  posting-time price: a proxy of the live re-plan).

Fill probability ``p`` (one bid, whole window): logistic in phase, NO price, price^2 and
log(spread), fitted on dev (events before 2026-10-01); ``p_rem`` decays with the dev share of fills
by elapsed window fraction (fills are front-loaded: a third land in the first hour). Data and fills:
``research/mentions_ext`` cell ``L12h/m12c/extend`` (hourly candles, maker fills on prints strictly
through the bid, breaker, early-start exclusion), with the 6 h order on every leg the early phase
did not fill *in this replay*. Ledgers scored with ``vol_metrics`` on the $100 bankroll; dev picks,
holdout (2026-10-01 to 11-16) confirms.

    python -m albert.research.mentions_budget run
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/mentions_budget")
EPS = 1e-9
TOPUP_MIN_LEFT = timedelta(hours=1)
FEATURES = ("1", "late", "price", "price^2", "log_spread")


# ---- fill model ----------------------------------------------------------------------------------
def features(late: np.ndarray, price: np.ndarray, spread: np.ndarray) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    s = np.clip(np.asarray(spread, dtype=float), 0.01, 0.5)
    return np.column_stack([np.ones(len(p)), np.asarray(late, dtype=float), p, p * p, np.log(s)])


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(100):
        pr = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (pr - y) + l2 * w
        H = (X * (pr * (1 - pr))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def fill_prob(o: pl.DataFrame, w: np.ndarray) -> np.ndarray:
    X = features((o["phase"] != "early").to_numpy(), o["price"].to_numpy(), o["spread"].to_numpy())
    return 1 / (1 + np.exp(-(X @ w)))


GRID = np.linspace(0, 1, 11)


def fill_curve(o: pl.DataFrame) -> dict[str, list[float]]:
    """Share of a phase's fills by elapsed window fraction, F(x) at x = 0, 0.1, ..., 1."""
    f = o.filter("filled").with_columns(
        ((pl.col("fill_ts") - pl.col("t")).dt.total_seconds() / (pl.col("end_o") - pl.col("t")).dt.total_seconds())
        .clip(0, 1).alias("x"))
    out = {}
    for ph in ("early", "late"):
        x = np.sort(f.filter(pl.col("phase") == ph)["x"].to_numpy())
        out[ph] = [0.0] + [float(np.searchsorted(x, g, side="right") / len(x)) for g in GRID[1:-1]] + [1.0]
    return out


def p_remaining(p0: float, elapsed: float, curve: list[float]) -> float:
    """P(fill in the rest of the window | not filled by ``elapsed`` (fraction))."""
    F = float(np.interp(min(max(elapsed, 0.0), 1.0), GRID, curve))
    return p0 * (1 - F) / max(1 - p0 * F, EPS)


# ---- replay --------------------------------------------------------------------------------------
@dataclass
class Policy:
    name: str
    budget: str = "expected"          # "expected" (this study) or "collateral" (the live VaR rule)
    target: float = 15.0
    max_events: int = 2
    daily_cap: float = 15.0
    ceiling: float = math.inf         # hard cap: resting collateral + realized fill cost ($)
    cancel: bool = True               # cancel resting bids when E > target after fills
    topup: bool = False
    max_orders: int = 40


def sim_orders(o12: pl.DataFrame, o6: pl.DataFrame) -> pl.DataFrame:
    """Early (12 h) orders of the extend cell plus the 6 h order on every leg (the replay drops a 6 h
    leg the early phase filled); events whose early breaker fired before 6 h get no 6 h order."""
    early = o12.filter(pl.col("phase") == "early")
    killed = early.filter(pl.col("kill").is_not_null() & (pl.col("kill") <= pl.col("start") - timedelta(hours=6)))
    late = o6.filter(~pl.col("event").is_in(killed["event"].unique().implode())).with_columns(pl.lit("late").alias("phase"))
    out = pl.concat([early, late.select(early.columns)], how="vertical_relaxed")
    return out.filter(pl.col("price").is_not_nan() & pl.col("price").is_not_null())


def replay(o: pl.DataFrame, pol: Policy, w: np.ndarray, curve: dict[str, list[float]]) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Event-driven replay of one policy. Returns the posted bids (with ``filled`` / ``fill_ts`` /
    ``pnl`` as replayed and ``cancelled_ts``) and path statistics."""
    o = o.with_columns(pl.Series("p0", fill_prob(o, w))).sort("t", "event", "price").with_row_index("i")
    R = o.to_dicts()
    by_post: dict[tuple, list[int]] = {}
    for r in R:
        by_post.setdefault((r["t"], r["event"], r["phase"]), []).append(r["i"])
    q: list[tuple] = []           # (ts, order, seq, kind, payload)
    seq = 0

    def push(ts, kind, payload, order):
        nonlocal seq
        seq += 1
        heapq.heappush(q, (ts, order, seq, kind, payload))

    for (t, ev, ph), ids in by_post.items():
        push(t, "post", (ev, ph, ids), 4)
    resting: dict[int, dict] = {}            # i -> row
    posted: dict[int, dict] = {}
    cut: dict[tuple, list[int]] = {}         # (event, phase) -> legs cut by the budget (top-up candidates)
    ev_end: dict[str, Any] = {}
    seen: set[str] = set()
    skipped: set[str] = set()
    filled_legs: set[str] = set()
    fills: list[dict] = []                   # {ts, cost, settle}
    stats = {"peak_realized": 0.0, "peak_resting": 0.0, "peak_E": 0.0, "cancels": 0, "topups": 0,
             "skipped_events": 0, "posting_cut_legs": 0}
    peaks_by_day: dict[Any, float] = {}

    def realized(ts) -> float:
        today = sum(f["cost"] for f in fills if f["ts"] <= ts and f["ts"].date() == ts.date())
        unset = sum(f["cost"] for f in fills if f["ts"] <= ts < f["settle"])
        return max(today, unset)

    def today_cost(ts) -> float:
        return sum(f["cost"] for f in fills if f["ts"] <= ts and f["ts"].date() == ts.date())

    def exp_rest(r, ts) -> float:
        span = (r["end_o"] - r["post_ts"]).total_seconds()
        x = (ts - r["post_ts"]).total_seconds() / span if span > 0 else 1.0
        return p_remaining(r["p0"], x, curve[r["phase"]]) * r["price"]

    def cost_of(r, ts) -> float:
        return exp_rest(r, ts) if pol.budget == "expected" else r["price"]

    def committed(ts) -> float:
        return realized(ts) + sum(cost_of(r, ts) for r in resting.values())

    def resting_cash() -> float:
        return sum(r["price"] for r in resting.values())

    def place(ids, ts, ev, ph, *, topup=False) -> list[int]:
        room = pol.target - committed(ts)
        cash = pol.ceiling - resting_cash() - realized(ts)
        left = []
        for k, i in enumerate(ids[:pol.max_orders] if not topup else ids):
            r = dict(R[i])
            if r["ticker"] in filled_legs or r["end_o"] <= ts:
                continue
            if topup and r["filled"] and r["fill_ts"] is not None and r["fill_ts"] <= ts:
                continue    # the market already traded through this price: a live re-plan prices it lower
            r["post_ts"] = ts
            c = cost_of(r, ts)
            if c > room + EPS or r["price"] > cash + EPS:
                left.append(i)
                continue
            room -= c
            cash -= r["price"]
            resting[i] = r
            posted[i] = r
            r["topup"] = topup
            ev_end[ev] = max(ev_end.get(ev, r["end_o"]), r["end_o"])
            push(r["end_o"], "end", i, 0)
            if r["filled"] and r["fill_ts"] is not None and r["fill_ts"] > ts:
                push(r["fill_ts"], "fill", i, 1)
            elif r["filled"] and r["fill_ts"] is not None:          # a print in the posting hour
                push(ts + timedelta(minutes=1), "fill", i, 1)
        return left

    def enforce(ts) -> None:
        if not pol.cancel:
            return
        while resting and committed(ts) > pol.target + EPS:
            j = max(resting, key=lambda i: cost_of(resting[i], ts))
            r = resting.pop(j)
            r["cancelled_ts"] = ts
            stats["cancels"] += 1

    def topups(ts) -> None:
        if not pol.topup:
            return
        for key in list(cut):
            ev, ph = key
            ids = [i for i in cut[key] if R[i]["end_o"] - ts >= TOPUP_MIN_LEFT]
            if not ids:
                cut.pop(key)
                continue
            if ev_end.get(ev) is None or ev_end[ev] <= ts:
                continue
            before = len(resting)
            cut[key] = place(ids, ts, ev, ph, topup=True)
            stats["topups"] += len(resting) - before

    while q:
        ts, _, _, kind, payload = heapq.heappop(q)
        if kind == "end":
            resting.pop(payload, None)
            topups(ts)
        elif kind == "fill":
            r = resting.pop(payload, None)
            if r is None:
                continue
            r["sim_fill_ts"] = ts
            filled_legs.add(r["ticker"])
            fills.append({"ts": ts, "cost": r["price"], "settle": r["settlement_ts"]})
            enforce(ts)
            rz = realized(ts)
            stats["peak_realized"] = max(stats["peak_realized"], rz)
            peaks_by_day[ts.date()] = max(peaks_by_day.get(ts.date(), 0.0), rz)
        elif kind == "post":
            ev, ph, ids = payload
            if ev in skipped:
                continue
            open_ev = {e for e, end in ev_end.items() if end > ts and e != ev}
            if ev not in seen and len(open_ev) >= pol.max_events:
                skipped.add(ev)
                stats["skipped_events"] += 1
                continue
            if today_cost(ts) >= pol.daily_cap - EPS:
                if ev not in seen:
                    skipped.add(ev)
                    stats["skipped_events"] += 1
                continue
            seen.add(ev)
            left = place(ids, ts, ev, ph)
            stats["posting_cut_legs"] += len(left)
            if left and pol.topup:
                cut[(ev, ph)] = left
                push(ts + timedelta(hours=1), "tick", None, 5)
        elif kind == "tick":
            topups(ts)
            if cut:
                push(ts + timedelta(hours=1), "tick", None, 5)
        stats["peak_resting"] = max(stats["peak_resting"], resting_cash())
        stats["peak_E"] = max(stats["peak_E"], committed(ts))

    rows = []
    for r in posted.values():
        f = r.get("sim_fill_ts")
        rows.append({**{k: r[k] for k in ("event", "ticker", "phase", "split", "et_date", "start", "settlement_ts", "y",
                                         "price", "p0", "t", "end_o")},
                     "post_ts": r["post_ts"], "topup": r.get("topup", False), "filled": f is not None, "fill_ts": f,
                     "cancelled_ts": r.get("cancelled_ts"),
                     "pnl": ((1 - r["y"]) - r["price"]) if f is not None else 0.0})
    book = pl.DataFrame(rows, infer_schema_length=None)
    d = np.array(sorted(peaks_by_day.values())) if peaks_by_day else np.array([0.0])
    stats.update({"days_with_fills": len(d), "realized_peak_p90": round(float(np.quantile(d, 0.9)), 2),
                  "days_realized_over_target": int((d > pol.target + EPS).sum()),
                  "peak_realized": round(stats["peak_realized"], 2), "peak_resting": round(stats["peak_resting"], 2),
                  "peak_E": round(stats["peak_E"], 2)})
    return book, stats


def score(book: pl.DataFrame, marks: pl.DataFrame, pol: Policy, split: str, stats: dict[str, Any]) -> dict[str, Any]:
    from albert.research import mentions_ext as X

    row: dict[str, Any] = {"policy": pol.name, "split": split, "events": book["event"].n_unique() if book.height else 0,
                           "bids": book.height}
    if not book.height:
        return row
    led = X.ledger(book.with_columns(pl.col("fill_ts").cast(pl.Datetime("us", "UTC"))), "mentions_budget", pol.name, split)
    f = book.filter("filled")
    row.update({"fills": f.height, "topup_fills": int(f["topup"].sum()) if f.height else 0,
                "c_per_fill": round(float(f["pnl"].mean()) * 100, 2) if f.height else None,
                "pnl_usd": round(float(f["pnl"].sum()), 2)})
    if f.height:
        _, te, _ = X.clustered_t(f["pnl"].to_numpy(), f["event"].to_numpy())
        _, td, _ = X.clustered_t(f["pnl"].to_numpy(), f["et_date"].cast(pl.String).to_numpy())
        row.update({"t_event": None if te is None else round(te, 2), "t_date": None if td is None else round(td, 2)})
    row.update(X.book_metrics(led, marks))
    row.update({k: stats[k] for k in ("peak_realized", "realized_peak_p90", "days_realized_over_target", "peak_resting",
                                      "peak_E", "cancels", "topups", "skipped_events")})
    return row


POLICIES = (
    [Policy("live: $15 collateral, 2 events", budget="collateral", cancel=False)]
    + [Policy(f"E$15, {n} events", max_events=n) for n in (2, 3, 4, 6, 10)]
    + [Policy(f"E$15, {n} events, top-up", max_events=n, topup=True) for n in (4, 6, 10)]
    + [Policy(f"E$15, {n} events, no cancel", max_events=n, cancel=False) for n in (6,)]
    + [Policy(f"E$15, 6 events, ceiling ${c}", max_events=6, ceiling=c) for c in (30, 40, 50)]
)


def choose(T: pl.DataFrame) -> str:
    """Pre-specified (before any holdout number): among dev policies with the expected budget, the
    largest dev book PnL whose dev book Sharpe is at least the live rule's dev Sharpe x 0.8."""
    d = T.filter(pl.col("split") == "dev")
    live = d.filter(pl.col("policy").str.starts_with("live"))["sharpe"][0]
    ok = d.filter(pl.col("policy").str.starts_with("E$") & (pl.col("sharpe") >= 0.8 * live))
    return ok.sort("pnl_usd", descending=True)["policy"][0]


def run() -> dict[str, Any]:
    from albert.research import mentions_ext as X

    U = pl.read_parquet(X.OUT / "universe.parquet")
    G = U.filter(pl.col("start") >= X.GRID_FROM)
    H = X.load_hourly(G["ticker"].to_list())
    marks = X.marks_from(H)
    Qs = X.build_qs(G, H)
    o12 = X.cell_orders(Qs, H, 12, 0.12, "extend")
    o6 = X.cell_orders(Qs, H, 6, 0.12, "hold")
    o = sim_orders(o12, o6)
    dev = o.filter(pl.col("split") == "dev")
    w = fit_logit(features((dev["phase"] != "early").to_numpy(), dev["price"].to_numpy(), dev["spread"].to_numpy()),
                  dev["filled"].to_numpy().astype(float))
    curve = fill_curve(dev)
    calib = []
    for split in ("dev", "holdout"):
        s = o.filter(pl.col("split") == split)
        p = fill_prob(s, w)
        calib.append({"split": split, "bids": s.height, "fills": int(s["filled"].sum()), "pred_fills": round(float(p.sum()), 1),
                      "fill_cost": round(float(s.filter("filled")["price"].sum()), 2),
                      "pred_fill_cost": round(float((p * s["price"].to_numpy()).sum()), 2)})
    rows, books = [], {}
    for pol in POLICIES:
        for split in ("dev", "holdout"):
            s = o.filter(pl.col("split") == split)
            book, st = replay(s, pol, w, curve)
            books[(pol.name, split)] = book
            rows.append(score(book, marks, pol, split, st))
            print(json.dumps(rows[-1], default=str), flush=True)
    T = pl.DataFrame(rows, infer_schema_length=None)
    chosen = choose(T)
    w_all = fit_logit(features((o["phase"] != "early").to_numpy(), o["price"].to_numpy(), o["spread"].to_numpy()),
                      o["filled"].to_numpy().astype(float))
    res = {"w_dev": w.tolist(), "w_all": w_all.tolist(), "features": FEATURES, "curve_dev": curve,
           "curve_all": fill_curve(o), "calibration": calib, "chosen": chosen}
    OUT.mkdir(parents=True, exist_ok=True)
    T.write_csv(OUT / "policies.csv")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    for (name, split), b in books.items():
        if name in (chosen, POLICIES[0].name) and b.height:
            b.write_parquet(OUT / f"book_{name.replace(' ', '_').replace(',', '').replace('$', '')}_{split}.parquet")
    print(json.dumps(res, default=str), flush=True)
    return {"T": T, **res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
