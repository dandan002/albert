"""Portfolio: how to allocate the $100 Kalshi bankroll across the strategies that exist
(research swarm 2026-11-23, slug ``portfolio``). No new edge search.

Inputs (read only): the research replays of the live mentions rule (``mentions_ext`` /
``mentions_budget``), the culture maker NO orders with trade-print fills (``culture_flow_review``),
approval 5-15c NO (``approval2/review``), HITS -> KXALBUMEQUIV (``music_charts``), the linked lock
taker (``linked``), the live audit (``data/live/audit/prod``) and the forward paper ledgers
(``data/paper_forward``).

Steps (each writes under ``data/research/portfolio/``):

``mentions``  replay the live mentions budget (expected fill cost E, event cap, hard ceiling) on the
              ``L12h/m12c/extend`` cell for a grid of budgets; books, ledgers, hourly cash in use.
``cmn``       replay the culture maker NO paper rule (two-sided book, mid <= 5c, sell YES at
              ceil(mid + 12c), <= 3 legs per event, cheapest NO first, 1 lot, print fills) under cash
              policies: fixed collateral caps, an expected-fill budget with a cash ceiling, and a pool
              shared with the mentions bot (mentions has priority; cmn cancels resting bids for room).
``daily``     daily mark-to-market PnL of every strategy ledger (``vol_metrics.mtm_values``).
``analyze``   correlations, allocations chosen on dev and scored once on holdout, block-bootstrap and
              event-resampling Monte Carlo of monthly PnL and drawdowns, guard-cap binding, Kelly,
              promotion-gate sample sizes.
``live``      live mentions fills (audit) and forward paper ledgers against the research numbers.

    python -m albert.research.portfolio {mentions,cmn,daily,analyze,live}
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/portfolio")
SPLIT_AT = datetime(2026, 10, 1, tzinfo=UTC)
BANKROLL = 100.0
ACCOUNT_CASH = 97.0          # balance 2026-11-20 ($97.28), all on shard 0
EPS = 1e-9


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _split(ts: pl.Expr) -> pl.Expr:
    return pl.when(ts < SPLIT_AT).then(pl.lit("dev")).otherwise(pl.lit("holdout"))


# ---- cash in use over time -----------------------------------------------------------------------
def cash_intervals(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """(start, end, usd) intervals of cash reserved: a resting bid from posting to fill / cancel / end,
    then a filled leg from the fill to settlement."""
    out = []
    for r in rows:
        stop = min(x for x in (r.get("fill_ts"), r.get("cancelled_ts"), r["end"]) if x is not None)
        if stop > r["post_ts"]:
            out.append({"start": r["post_ts"], "end": stop, "usd": r["price"], "kind": "resting"})
        if r.get("fill_ts") is not None:
            out.append({"start": r["fill_ts"], "end": max(r["settle_ts"], r["fill_ts"]), "usd": r["price"],
                        "kind": "position"})
    return pl.DataFrame(out, schema={"start": pl.Datetime("us", "UTC"), "end": pl.Datetime("us", "UTC"),
                                     "usd": pl.Float64, "kind": pl.String})


def hourly_usage(iv: pl.DataFrame, t0: datetime, t1: datetime) -> pl.DataFrame:
    """Cash in use at each hour boundary in [t0, t1] (sum of intervals covering it)."""
    hours = pl.datetime_range(t0, t1, "1h", time_zone="UTC", eager=True)
    h = hours.to_numpy().astype("datetime64[us]").astype(np.int64)
    s = iv["start"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]").astype(np.int64)
    e = iv["end"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]").astype(np.int64)
    u = iv["usd"].to_numpy()
    k = (iv["kind"] == "resting").to_numpy()
    tot, rest = np.zeros(len(h)), np.zeros(len(h))
    for si, ei, ui, ki in zip(s, e, u, k, strict=True):
        a, b = np.searchsorted(h, si, "left"), np.searchsorted(h, ei, "left")
        tot[a:b] += ui
        if ki:
            rest[a:b] += ui
    return pl.DataFrame({"ts": hours, "usd": tot, "resting": rest})


def usage_stats(u: pl.DataFrame) -> dict[str, float]:
    x = u["usd"].to_numpy()
    return {"mean": round(float(x.mean()), 2), "p50": round(float(np.quantile(x, 0.5)), 2),
            "p90": round(float(np.quantile(x, 0.9)), 2), "p99": round(float(np.quantile(x, 0.99)), 2),
            "max": round(float(x.max()), 2)}


# ---- mentions: the live budget on a grid ---------------------------------------------------------
MENTION_POLICIES = (("live E15 6ev c40", 15.0, 6, 40.0), ("E10 6ev c40", 10.0, 6, 40.0),
                    ("E20 6ev c40", 20.0, 6, 40.0), ("E20 6ev c50", 20.0, 6, 50.0),
                    ("E25 6ev c50", 25.0, 6, 50.0), ("E25 6ev c60", 25.0, 6, 60.0),
                    ("E30 8ev c60", 30.0, 8, 60.0))


def run_mentions() -> dict[str, Any]:
    from albert.research import mentions_budget as B
    from albert.research import mentions_ext as X

    U = pl.read_parquet(X.OUT / "universe.parquet")
    G = U.filter(pl.col("start") >= X.GRID_FROM)
    H = X.load_hourly(G["ticker"].to_list())
    marks = X.marks_from(H)
    Qs = X.build_qs(G, H)
    o = B.sim_orders(X.cell_orders(Qs, H, 12, 0.12, "extend"), X.cell_orders(Qs, H, 6, 0.12, "hold"))
    del Qs, H
    dev = o.filter(pl.col("split") == "dev")
    w = B.fit_logit(B.features((dev["phase"] != "early").to_numpy(), dev["price"].to_numpy(), dev["spread"].to_numpy()),
                    dev["filled"].to_numpy().astype(float))
    curve = B.fill_curve(dev)
    _save(marks.filter(pl.col("ticker").is_in(o["ticker"].unique().implode())), OUT / "marks_mentions.parquet")
    rows = []
    for name, target, nev, ceil in MENTION_POLICIES:
        pol = B.Policy(name, target=target, max_events=nev, daily_cap=max(15.0, target), ceiling=ceil)
        for split in ("dev", "holdout"):
            book, st = B.replay(o.filter(pl.col("split") == split), pol, w, curve)
            book = book.with_columns(pl.col("fill_ts").cast(pl.Datetime("us", "UTC")),
                                     pl.col("cancelled_ts").cast(pl.Datetime("us", "UTC")))
            led = X.ledger(book, "mentions", name, split)
            _save(led, OUT / "ledgers" / f"mentions_{_slug(name)}_{split}.parquet")
            iv = cash_intervals([{"post_ts": r["post_ts"], "fill_ts": r["fill_ts"], "cancelled_ts": r["cancelled_ts"],
                                  "end": r["end_o"], "settle_ts": r["settlement_ts"], "price": r["price"]}
                                 for r in book.iter_rows(named=True)])
            u = hourly_usage(iv, book["post_ts"].min().replace(minute=0, second=0, microsecond=0),
                             book["settlement_ts"].max())
            _save(u, OUT / "cash" / f"mentions_{_slug(name)}_{split}.parquet")
            f = book.filter("filled")
            _, te, _ = X.clustered_t(f["pnl"].to_numpy(), f["event"].to_numpy())
            row = {"policy": name, "split": split, "target": target, "max_events": nev, "ceiling": ceil,
                   "bids": book.height, "fills": f.height, "c_per_fill": round(float(f["pnl"].mean()) * 100, 2),
                   "t_event": round(te, 2) if te is not None else None, "pnl_usd": round(float(f["pnl"].sum()), 2),
                   **{k: st[k] for k in ("peak_realized", "peak_resting", "cancels", "skipped_events")},
                   **{f"cash_{k}": v for k, v in usage_stats(u).items()}}
            row.update(X.book_metrics(led, marks))
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    T = pl.DataFrame(rows, infer_schema_length=None)
    T.write_csv(OUT / "mentions_policies.csv")
    return {"rows": rows}


def _slug(s: str) -> str:
    return s.replace(" ", "_").replace("$", "").replace(",", "")


# ---- cmn: culture maker NO under cash policies ---------------------------------------------------
CMN_MARGIN, CMN_BAND, CMN_MAX_LEGS = 0.12, 0.05, 3


def cmn_orders(split: str, root: Path = Path(".")) -> pl.DataFrame:
    """The paper rule's orders from the review's print-filled order table: real two-sided book (bid >=
    1c, ask <= 99c), mid <= 5c, sell YES at ceil(mid + 12c) above the bid; the event's 3 cheapest NO
    first. ``fill_ts`` = the first YES print strictly through, before the leg closes."""
    p = root / "data/research/culture_flow_review" / f"prints_{split}.parquet"
    d = pl.read_parquet(p, columns=["event_ticker", "ticker", "series_ticker", "family", "t", "bid", "ask", "mid", "s",
                                    "close_time", "settlement_ts", "y", "print_fill_ts", "vol_through"])
    return cmn_rule(d).with_columns(pl.lit(split).alias("split"))


def cmn_rule(d: pl.DataFrame) -> pl.DataFrame:
    two = (pl.col("bid") >= 0.01 - EPS) & (pl.col("ask") <= 0.99 + EPS) & (pl.col("ask") > pl.col("bid"))
    q = d.filter(two & (pl.col("mid") <= CMN_BAND + EPS))
    q = q.with_columns(((pl.col("mid") + CMN_MARGIN) * 100 - 1e-6).ceil().truediv(100).alias("sell_at"))
    q = q.filter((pl.col("sell_at") < 0.99) & (pl.col("sell_at") > pl.col("bid") + EPS))
    q = q.with_columns((1 - pl.col("sell_at")).round(2).alias("price"))
    q = q.sort("t", "event_ticker", "price", "ticker").with_columns(
        pl.int_range(pl.len()).over("event_ticker").alias("rank"))
    q = q.filter(pl.col("rank") < CMN_MAX_LEGS)
    fill = pl.when(pl.col("print_fill_ts").is_not_null() & (pl.col("print_fill_ts") > pl.col("t"))
                   & (pl.col("print_fill_ts") <= pl.col("close_time"))).then(pl.col("print_fill_ts"))
    return q.with_columns(fill.alias("fill_ts"), pl.col("close_time").alias("end"),
                          pl.max_horizontal("settlement_ts", "close_time").alias("settle_ts"))


def cmn_fill_model(o: pl.DataFrame) -> np.ndarray:
    """P(fill before close) for a cmn bid: logistic in [1, mid, log(spread), log(hours to close)],
    fitted on the rows given (dev)."""
    from albert.research.mentions_budget import fit_logit

    return fit_logit(_cmn_X(o), o["fill_ts"].is_not_null().to_numpy().astype(float))


def _cmn_X(o: pl.DataFrame) -> np.ndarray:
    hrs = ((o["end"] - o["t"]).dt.total_seconds() / 3600).clip(1, 400).to_numpy()
    sp = (o["ask"] - o["bid"]).clip(0.01, 0.5).to_numpy()
    return np.column_stack([np.ones(o.height), o["mid"].to_numpy(), np.log(sp), np.log(hrs)])


def cmn_p(o: pl.DataFrame, w: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(_cmn_X(o) @ w)))


@dataclass
class CashPolicy:
    name: str
    cap: float = 25.0                  # resting collateral + open positions (cash Kalshi reserves)
    expected: float = math.inf         # expected fill cost target: open positions + sum p_rem x price
    shared: bool = False               # cap(t) = account cash - mentions' cash in use(t) - reserve
    reserve: float = 5.0


def cmn_replay(o: pl.DataFrame, pol: CashPolicy, p0: np.ndarray | None = None,
               mentions_usage: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Event-driven replay of the cmn book under a cash policy. Returns the posted orders (with
    ``filled``, ``cancelled_ts``) and path statistics. ``mentions_usage`` (hourly ``usd``) is the
    mentions bot's cash in use for the shared pool; cmn cancels resting bids (largest price first)
    when the pool shrinks below its use, and has no claim on cash the mentions bot holds."""
    o = o.sort("t", "event_ticker", "price", "ticker").with_row_index("i")
    p0 = np.full(o.height, 0.1) if p0 is None else p0
    R = o.to_dicts()
    for r, p in zip(R, p0, strict=True):
        r["p0"] = float(p)
    mu_ts = mu_usd = None
    if mentions_usage is not None:
        mu_ts = mentions_usage["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]").astype(np.int64)
        mu_usd = mentions_usage["usd"].to_numpy()

    def mentions_at(ts: datetime) -> float:
        if mu_ts is None:
            return 0.0
        k = np.searchsorted(mu_ts, np.datetime64(ts.replace(tzinfo=None), "us").astype(np.int64), "right") - 1
        return float(mu_usd[k]) if 0 <= k < len(mu_usd) else 0.0

    def cap_at(ts: datetime) -> float:
        if pol.shared:
            return min(pol.cap, ACCOUNT_CASH - pol.reserve - mentions_at(ts))
        return pol.cap

    q: list[tuple] = []
    seq = 0

    def push(ts, kind, payload, order):
        nonlocal seq
        seq += 1
        heapq.heappush(q, (ts, order, seq, kind, payload))

    for r in R:
        push(r["t"], "post", r["i"], 3)
    if pol.shared and mu_ts is not None:
        for ts in mentions_usage["ts"].to_list():
            push(ts, "tick", None, 4)
    resting: dict[int, dict] = {}
    pos: dict[int, dict] = {}
    posted: dict[int, dict] = {}
    st = {"peak_used": 0.0, "cancels": 0, "skipped_cash": 0, "skipped_expected": 0, "conflict_hours": 0,
          "used_hours": 0.0, "hours": 0}

    def used() -> float:
        return sum(r["price"] for r in resting.values()) + sum(r["price"] for r in pos.values())

    def p_rem(r, ts) -> float:
        span = (r["end"] - r["t"]).total_seconds()
        x = min(max((ts - r["t"]).total_seconds() / span, 0.0), 1.0) if span > 0 else 1.0
        return r["p0"] * (1 - x) / max(1 - r["p0"] * x, EPS)     # fills uniform over the window

    def expected(ts) -> float:
        return sum(r["price"] for r in pos.values()) + sum(p_rem(r, ts) * r["price"] for r in resting.values())

    def shrink(ts) -> None:
        cap = cap_at(ts)
        while resting and used() > cap + EPS:
            j = max(resting, key=lambda i: (resting[i]["price"], i))
            r = resting.pop(j)
            r["cancelled_ts"] = ts
            st["cancels"] += 1
        if used() > cap + EPS:
            st["conflict_hours"] += 1

    while q:
        ts, _, _, kind, payload = heapq.heappop(q)
        if kind == "post":
            r = dict(R[payload])
            if used() + r["price"] > cap_at(ts) + EPS:
                st["skipped_cash"] += 1
                continue
            if expected(ts) + r["p0"] * r["price"] > pol.expected + EPS:
                st["skipped_expected"] += 1
                continue
            r["post_ts"] = ts
            resting[r["i"]] = r
            posted[r["i"]] = r
            push(r["end"], "end", r["i"], 0)
            if r["fill_ts"] is not None:
                push(r["fill_ts"], "fill", r["i"], 1)
        elif kind == "fill":
            r = resting.pop(payload, None)
            if r is None:
                continue
            r["sim_fill_ts"] = ts
            pos[payload] = r
            push(r["settle_ts"], "settle", payload, 0)
            if pol.expected < math.inf:            # over the expected budget after a fill: cancel the costliest
                while resting and expected(ts) > pol.expected + EPS:
                    j = max(resting, key=lambda i: p_rem(resting[i], ts) * resting[i]["price"])
                    resting.pop(j)["cancelled_ts"] = ts
                    st["cancels"] += 1
        elif kind == "end":
            resting.pop(payload, None)
        elif kind == "settle":
            pos.pop(payload, None)
        elif kind == "tick":
            shrink(ts)
            st["hours"] += 1
            st["used_hours"] += used()
        st["peak_used"] = max(st["peak_used"], used())
    rows = [{k: r.get(k) for k in ("event_ticker", "ticker", "family", "split", "t", "post_ts", "end", "settle_ts",
                                    "price", "y", "p0", "cancelled_ts")}
            | {"fill_ts": r.get("sim_fill_ts"), "filled": r.get("sim_fill_ts") is not None,
               "pnl": (1 - r["y"] - r["price"]) if r.get("sim_fill_ts") is not None else 0.0}
            for r in posted.values()]
    book = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    st["peak_used"] = round(st["peak_used"], 2)
    st["mean_used_shared"] = round(st["used_hours"] / st["hours"], 2) if st["hours"] else None
    return book, st


def cmn_ledger(book: pl.DataFrame, variant: str, split: str) -> pl.DataFrame:
    f = book.filter("filled")
    return f.select(pl.lit("cmn").alias("strategy"), pl.lit(variant).alias("variant"), pl.lit(split).alias("split"),
                    pl.col("ticker").alias("struct_id"), "event_ticker", "ticker", pl.lit("no").alias("side"),
                    pl.lit(1).alias("n"), pl.col("fill_ts").cast(pl.Datetime("us", "UTC")).alias("t_in"),
                    pl.col("price").alias("cost_usd"), pl.col("settle_ts").cast(pl.Datetime("us", "UTC")),
                    (1 - pl.col("y")).cast(pl.Float64).alias("payoff_usd"))


CMN_POLICIES = (CashPolicy("cap20", cap=20), CashPolicy("cap25", cap=25), CashPolicy("cap40", cap=40),
                CashPolicy("cap57", cap=57), CashPolicy("cap100", cap=100),
                CashPolicy("E1.5 cap40", cap=40, expected=1.5), CashPolicy("E3 cap57", cap=57, expected=3.0),
                CashPolicy("E3 cap100", cap=100, expected=3.0),
                CashPolicy("shared (97 - mentions - 5)", cap=100, shared=True),
                CashPolicy("shared, max 40", cap=40, shared=True))


def run_cmn() -> dict[str, Any]:
    from albert.research import mentions_ext as X
    from albert.research.culture_flow import marks_for

    od, oh = cmn_orders("dev"), cmn_orders("holdout")
    w = cmn_fill_model(od)
    rows = []
    allm = []
    for split, o in (("dev", od), ("holdout", oh)):
        p0 = cmn_p(o, w)
        mu = None
        mp = OUT / "cash" / f"mentions_live_E15_6ev_c40_{split}.parquet"
        if mp.exists():
            mu = pl.read_parquet(mp)
        uncapped = {"orders": o.height, "fills": int(o["fill_ts"].is_not_null().sum()),
                    "pred_fills": round(float(p0.sum()), 1), "events": o["event_ticker"].n_unique()}
        f = o.filter(pl.col("fill_ts").is_not_null())
        pnl = (1 - f["y"].cast(pl.Float64) - f["price"]).to_numpy()
        _, te, _ = X.clustered_t(pnl, f["event_ticker"].to_numpy())
        uncapped.update({"c_per_fill": round(float(pnl.mean()) * 100, 2), "t_event": round(te, 2) if te else None,
                         "yes": int(f["y"].sum()), "pnl_usd": round(float(pnl.sum()), 2)})
        rows.append({"policy": "uncapped (every rule order)", "split": split, **uncapped})
        for pol in CMN_POLICIES:
            if pol.shared and mu is None:
                continue
            book, st = cmn_replay(o, pol, p0, mu if pol.shared else None)
            led = cmn_ledger(book, pol.name, split)
            _save(led, OUT / "ledgers" / f"cmn_{_slug(pol.name)}_{split}.parquet")
            iv = cash_intervals([{"post_ts": r["post_ts"], "fill_ts": r["fill_ts"], "cancelled_ts": r["cancelled_ts"],
                                  "end": r["end"], "settle_ts": r["settle_ts"], "price": r["price"]}
                                 for r in book.iter_rows(named=True)])
            u = hourly_usage(iv, book["post_ts"].min().replace(minute=0, second=0, microsecond=0), book["settle_ts"].max())
            _save(u, OUT / "cash" / f"cmn_{_slug(pol.name)}_{split}.parquet")
            fb = book.filter("filled")
            x = fb["pnl"].to_numpy()
            _, te, _ = X.clustered_t(x, fb["event_ticker"].to_numpy()) if fb.height > 1 else (None, None, None)
            weeks = max((book["settle_ts"].max() - book["post_ts"].min()).total_seconds() / 604800, 1e-9)
            row = {"policy": pol.name, "split": split, "orders": book.height, "fills": fb.height,
                   "pred_fills": round(float(book["p0"].sum()), 1), "events": fb["event_ticker"].n_unique(),
                   "c_per_fill": round(float(x.mean()) * 100, 2) if fb.height else None,
                   "t_event": round(te, 2) if te else None, "yes": int(fb["y"].sum()) if fb.height else 0,
                   "pnl_usd": round(float(x.sum()), 2), "usd_per_week_raw": round(float(x.sum()) / weeks, 3),
                   **st, **{f"cash_{k}": v for k, v in usage_stats(u).items()},
                   "usd_per_week_per_100_cash": round(float(x.sum()) / weeks / max(usage_stats(u)["mean"], EPS) * 100, 2)}
            if led.height:
                m = marks_for(led["ticker"].unique().to_list())
                allm.append(m)
                row.update(X.book_metrics(led, m))
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    T = pl.DataFrame(rows, infer_schema_length=None)
    T.write_csv(OUT / "cmn_policies.csv")
    if allm:
        _save(pl.concat(allm).unique(["ticker", "ts"]), OUT / "marks_cmn.parquet")
    (OUT / "cmn_fill_model.json").write_text(json.dumps({"w_dev": w.tolist(),
                                                         "features": ["1", "mid", "log_spread", "log_hours"]}))
    return {"rows": rows}


# ---- daily MTM series for every strategy -----------------------------------------------------------
def other_ledgers(root: Path = Path(".")) -> dict[str, pl.DataFrame]:
    """The other candidates' research ledgers as named (strategy -> ledger with a split column)."""
    r = root / "data/research"
    out: dict[str, pl.DataFrame] = {}
    # approval: 1 NO on every 5-15c leg at 01:00 ET (the forward paper rule: no cap, no tie-break)
    a = pl.read_parquet(r / "approval2/review/ledgers/ledger_no_cap_pooled.parquet")
    out["apr"] = a.with_columns(_split(pl.col("t_in")).alias("split"))
    # HITS -> album ladders: the frozen primary (~$4/order, <= $5/event); the reviewer's look-ahead
    # row (SIL-30K, 2026-09-29: the table row did not exist at the decision) dropped from the holdout
    h = pl.concat([pl.read_parquet(r / f"music_charts/ledgers/ledger_ladder_preview_fast_theta_0_05_{s}.parquet")
                   .with_columns(pl.lit(s).alias("split")) for s in ("dev", "holdout")], how="diagonal_relaxed")
    out["hits"] = h.filter(~pl.col("ticker").str.contains("SIL"))
    out["hits_raw"] = h
    lk = pl.concat([pl.read_parquet(r / f"linked/ledgers/ledger_linked_taker_minute1_{s}.parquet")
                    .with_columns(pl.lit(s).alias("split")) for s in ("dev", "holdout")], how="diagonal_relaxed")
    out["linked"] = lk
    return out


def daily_series(led: pl.DataFrame, marks: pl.DataFrame, cal: list[date]) -> np.ndarray:
    from albert.research import vol_metrics as V

    if not led.height:
        return np.zeros(len(cal))
    v = V.mtm_values(led.select(V.LEDGER_COLS), marks)
    return V._on_calendar(v.group_by("day").agg(pl.col("pnl").sum()), cal)


def event_values(led: pl.DataFrame, marks: pl.DataFrame) -> pl.DataFrame:
    from albert.research import vol_metrics as V

    return V.mtm_values(led.select(V.LEDGER_COLS), marks).select("event_ticker", "day", "pnl")


def run_daily() -> dict[str, Any]:
    from albert.research import vol_metrics as V

    leds: dict[str, pl.DataFrame] = {}
    for p in sorted((OUT / "ledgers").glob("*.parquet")):
        name, split = p.stem.rsplit("_", 1)
        d = pl.read_parquet(p)
        if d.height:
            leds.setdefault(name, []).append(d.with_columns(pl.lit(split).alias("split")))
    leds = {k: pl.concat(v, how="diagonal_relaxed") for k, v in leds.items()}
    leds.update(other_ledgers())
    tick = sorted({t for d in leds.values() for t in d["ticker"].unique().to_list()})
    marks = [V.load_marks(tick)]
    for f in ("marks_mentions.parquet", "marks_cmn.parquet"):
        if (OUT / f).exists():
            marks.append(pl.read_parquet(OUT / f).select("ticker", pl.col("ts").cast(pl.Datetime("us", "UTC")), "yes_mid"))
    M = pl.concat(marks, how="vertical_relaxed").unique(["ticker", "ts"]).sort("ticker", "ts")
    rows, evs = [], []
    for name, d in leds.items():
        d = d.with_columns(pl.col("t_in").cast(pl.Datetime("us", "UTC")), pl.col("settle_ts").cast(pl.Datetime("us", "UTC")))
        v = event_values(d, M).with_columns(pl.lit(name).alias("strategy"))
        evs.append(v)
        for split in ("dev", "holdout"):
            s = d.filter(pl.col("split") == split)
            if s.height:
                rows.append({"strategy": name, "split": split, "fills": s.height, "events": s["event_ticker"].n_unique(),
                             "pnl": round(float((s["payoff_usd"] - s["cost_usd"]).sum()), 2),
                             "first": str(s["t_in"].min()), "last": str(s["settle_ts"].max())})
    E = pl.concat(evs)
    _save(E, OUT / "daily_event_pnl.parquet")    # (strategy, event, day) MTM PnL: the base of every series
    pl.DataFrame(rows).write_csv(OUT / "ledger_inventory.csv")
    print(pl.DataFrame(rows))
    return {"inventory": rows}


# ---- analysis --------------------------------------------------------------------------------------
DEV_DAYS = (date(2026, 8, 17), date(2026, 9, 30))       # every main strategy is replayed in this window
HOLD_DAYS = (date(2026, 10, 1), date(2026, 11, 16))     # the mentions replay ends 11-16


def series(E: pl.DataFrame, strategy: str, d0: date, d1: date, *, drop_events: list[str] | None = None,
           scale: float = 1.0) -> np.ndarray:
    cal = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    s = E.filter(pl.col("strategy") == strategy)
    if drop_events:
        s = s.filter(~pl.col("event_ticker").is_in(drop_events))
    s = s.filter((pl.col("day") >= d0) & (pl.col("day") <= d1)).group_by("day").agg(pl.col("pnl").sum())
    full = pl.DataFrame({"day": cal}).join(s, on="day", how="left")
    return full["pnl"].fill_null(0.0).to_numpy() * scale


def book_stats(x: np.ndarray, ev: pl.DataFrame | None = None) -> dict[str, Any]:
    from albert.research import vol_metrics as V

    st = V.series_stats(x)
    keep = ("days", "pnl_usd", "usd_per_week", "ann_vol_pct", "daily_vol_usd", "sharpe", "sharpe_ci90", "max_dd_usd",
            "worst_day_usd", "worst_week_usd", "weeks_positive_pct")
    out = {k: st[k] for k in keep}
    if ev is not None and ev.height:
        tot = ev.group_by("event_ticker").agg(pl.col("pnl").sum()).sort(["pnl", "event_ticker"], descending=[True, False])
        top = tot["event_ticker"].head(5).to_list()
        d0 = ev["day"].min()
        x5 = x.copy()
        sub = ev.filter(pl.col("event_ticker").is_in(top)).group_by("day").agg(pl.col("pnl").sum())
        for dd, p in sub.iter_rows():
            k = (dd - d0).days
            if 0 <= k < len(x5):
                x5[k] -= p
        out["sharpe_ex_top5"] = V._r(V.sharpe(x5))
    return out


def month_loss_prob(x: np.ndarray, *, month: int = 30, reps: int = 4000, block: int = 7, seed: int = 11,
                    haircut: float = 0.0) -> dict[str, float]:
    """Moving-block bootstrap of 30-day sums of a daily PnL series (joint across strategies when ``x``
    is a sum): P(month < 0), 5th percentile, and P(month < -$10). ``haircut`` removes that share of
    the series mean from every day (an out-of-sample shrink of the edge)."""
    x = x - haircut * x.mean()
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = math.ceil(month / block)
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :month]
    m = x[idx].sum(axis=1)
    return {"p_loss_month": round(float((m < 0).mean()), 4), "p_month_below_-10": round(float((m < -10).mean()), 4),
            "month_p05": round(float(np.quantile(m, 0.05)), 2), "month_median": round(float(np.median(m)), 2)}


def event_mc(ev_tables: dict[str, tuple[np.ndarray, float]], *, months: int = 12, reps: int = 20000, seed: int = 5,
             haircut: float = 0.0) -> dict[str, Any]:
    """Compound-Poisson Monte Carlo by event: each strategy draws Poisson(rate x 30 d) events a month
    and resamples whole-event PnLs (``ev_tables``: name -> (event PnLs, events per day)). Gives the
    monthly loss probability and the 12-month max drawdown on month ends (an event-level check that
    does not lean on the holdout's lack of losing days). ``haircut``: share of each strategy's mean
    event PnL removed (edge shrink)."""
    rng = np.random.default_rng(seed)
    tot = np.zeros((reps, months))
    for pnl, rate in ev_tables.values():
        if not len(pnl):
            continue
        adj = pnl - haircut * pnl.mean()
        k = rng.poisson(rate * 30, size=(reps, months))
        mx = int(k.max())
        draws = adj[rng.integers(0, len(adj), size=(reps, months, max(mx, 1)))]
        mask = np.arange(max(mx, 1))[None, None, :] < k[:, :, None]
        tot += (draws * mask).sum(axis=2)
    eq = np.cumsum(tot, axis=1)
    dd = (np.concatenate([np.zeros((reps, 1)), eq], axis=1)
          - np.maximum.accumulate(np.concatenate([np.zeros((reps, 1)), eq], axis=1), axis=1)).min(axis=1)
    return {"p_loss_month": round(float((tot < 0).mean()), 4), "month_mean": round(float(tot.mean()), 2),
            "month_p05": round(float(np.quantile(tot, 0.05)), 2), "p_losing_months_ge_3_of_12": round(float(((tot < 0).sum(axis=1) >= 3).mean()), 4),
            "dd12_median": round(float(np.median(dd)), 2), "dd12_p95": round(float(np.quantile(dd, 0.05)), 2),
            "p_dd12_below_-20": round(float((dd < -20).mean()), 4), "p_year_loss": round(float((eq[:, -1] < 0).mean()), 4)}


def kelly(r: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Fraction f of bankroll staked per event maximising mean log(1 + f r), r = event PnL per $ staked."""
    grid = np.linspace(0.001, 0.999, 999) if grid is None else grid
    g = [np.mean(np.log1p(np.maximum(f * r, -0.999999))) for f in grid]
    return float(grid[int(np.argmax(g))])


def rho_table(xs: dict[str, np.ndarray]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    names = list(xs)
    for a in names:
        out[a] = {}
        for b in names:
            x, y = xs[a], xs[b]
            ok = np.std(x) > 0 and np.std(y) > 0
            out[a][b] = round(float(np.corrcoef(x, y)[0, 1]), 3) if ok else None
    return out


def required_fills(mu: float, sd: float, deff: float = 1.0, t: float = 2.0) -> int:
    """Fills needed so that mean/se reaches ``t`` for a per-fill mean ``mu`` and sd ``sd`` (design effect ``deff``)."""
    return math.ceil((t * sd / mu) ** 2 * deff) if mu > 0 else -1


def clustered_deff(x: np.ndarray, g: np.ndarray) -> float:
    """Variance inflation of the event-clustered mean over the iid mean."""
    from albert.research.mentions_ext import clustered_t

    mu, tc, _ = clustered_t(x, g)
    if tc is None or not len(x):
        return 1.0
    se_c = mu / tc if tc else float("nan")
    se_i = float(np.std(x, ddof=1)) / math.sqrt(len(x))
    return round(float((se_c / se_i) ** 2), 2) if se_i > 0 else 1.0


def run_analyze() -> dict[str, Any]:
    E = pl.read_parquet(OUT / "daily_event_pnl.parquet")
    res: dict[str, Any] = {}
    strategies = sorted(E["strategy"].unique().to_list())
    # 1. each strategy alone, by split, on the common windows
    per: list[dict[str, Any]] = []
    for s in strategies:
        for split, (d0, d1) in (("dev", DEV_DAYS), ("holdout", HOLD_DAYS)):
            x = series(E, s, d0, d1)
            ev = E.filter((pl.col("strategy") == s) & (pl.col("day") >= d0) & (pl.col("day") <= d1))
            per.append({"strategy": s, "split": split, **book_stats(x, ev),
                        **month_loss_prob(x), "events": ev["event_ticker"].n_unique()})
    res["standalone"] = per
    # 2. correlations of daily and weekly PnL (holdout and dev), main candidates
    main = ["mentions_live_E15_6ev_c40", "cmn_cap25", "cmn_shared_(97_-_mentions_-_5)", "apr", "hits", "linked"]
    main = [m for m in main if m in strategies]
    from albert.research.vol_metrics import weekly
    res["corr"] = {}
    for split, (d0, d1) in (("dev", DEV_DAYS), ("holdout", HOLD_DAYS)):
        xs = {m: series(E, m, d0, d1) for m in main}
        res["corr"][split] = {"daily": rho_table(xs), "weekly": rho_table({k: weekly(v) for k, v in xs.items()})}
    # shared events / series between strategies
    sets = {s: set(E.filter(pl.col("strategy") == s)["event_ticker"].unique().to_list()) for s in main}
    res["shared_events"] = {f"{a}|{b}": len(sets[a] & sets[b]) for i, a in enumerate(main) for b in main[i + 1:]}
    ser = {s: {e.split("-")[0] for e in v} for s, v in sets.items()}
    res["shared_series"] = {f"{a}|{b}": sorted(ser[a] & ser[b]) for i, a in enumerate(main) for b in main[i + 1:]
                            if ser[a] & ser[b]}
    # 3. combined books (allocation scenarios)
    res["scenarios"] = scenarios(E, strategies)
    # 4. allocation chosen on dev (rules fixed before scoring the holdout), scored on the holdout once
    res["allocation"] = choose_allocation(E, strategies)
    # 5. event-level Monte Carlo of months and 12-month drawdowns per scenario
    leds = all_ledgers()
    res["event_mc"] = scenario_mc(leds)
    # 6. guard caps: cash at risk (hourly) and daily MTM loss of the combined book vs $60 / $15
    res["guard"] = guard_binding(E, leds)
    # 7. Kelly and promotion-gate sample sizes
    res["kelly"] = kelly_table(leds)
    res["kelly_binary"] = {k: binary_kelly(leds[k]) for k in ("mentions_live_E15_6ev_c40", "cmn_shared_(97_-_mentions_-_5)",
                                                               "cmn_cap40", "apr", "hits", "linked") if k in leds}
    res["gates"] = gate_sizes(leds)
    res["mentions_history_monthly"] = mentions_history_months()
    return res


def all_ledgers() -> dict[str, pl.DataFrame]:
    leds: dict[str, list[pl.DataFrame]] = {}
    for p in sorted((OUT / "ledgers").glob("*.parquet")):
        name, split = p.stem.rsplit("_", 1)
        d = pl.read_parquet(p)
        if d.height:
            leds.setdefault(name, []).append(d.with_columns(pl.lit(split).alias("split")))
    out = {k: pl.concat(v, how="diagonal_relaxed") for k, v in leds.items()}
    out.update(other_ledgers())
    return {k: v.with_columns(pl.col("t_in").cast(pl.Datetime("us", "UTC")), pl.col("settle_ts").cast(pl.Datetime("us", "UTC")))
            for k, v in out.items()}


def event_table(led: pl.DataFrame, split: str | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-event PnL, per-event cost and events per day for a ledger (one split or pooled)."""
    s = led if split is None else led.filter(pl.col("split") == split)
    if not s.height:
        return np.array([]), np.array([]), 0.0
    e = s.group_by("event_ticker").agg((pl.col("payoff_usd") - pl.col("cost_usd")).sum().alias("p"),
                                       pl.col("cost_usd").sum().alias("c"))
    days = max((s["settle_ts"].max() - s["t_in"].min()).total_seconds() / 86400, 1.0)
    return e["p"].to_numpy(), e["c"].to_numpy(), e.height / days


MC_MIXES = {
    "mentions live": ("mentions_live_E15_6ev_c40",),
    "mentions + cmn shared": ("mentions_live_E15_6ev_c40", "cmn_shared_(97_-_mentions_-_5)"),
    "mentions + cmn shared + apr": ("mentions_live_E15_6ev_c40", "cmn_shared_(97_-_mentions_-_5)", "apr"),
    "mentions + cmn shared + apr + hits": ("mentions_live_E15_6ev_c40", "cmn_shared_(97_-_mentions_-_5)", "apr", "hits"),
    "mentions + cmn $40": ("mentions_live_E15_6ev_c40", "cmn_cap40"),
    "mentions + cmn $40 + apr + hits x0.4": ("mentions_live_E15_6ev_c40", "cmn_cap40", "apr", "hits*0.4"),
    "cmn shared alone": ("cmn_shared_(97_-_mentions_-_5)",),
    "cmn $25 alone": ("cmn_cap25",),
    "apr alone": ("apr",),
    "hits alone": ("hits",),
}


def scenario_mc(leds: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    out = []
    for name, mix in MC_MIXES.items():
        if any(k.split("*")[0] not in leds for k in mix):
            continue
        for basis in ("dev", "holdout", None):
            tabs = {}
            for k in mix:
                led_name, scale = (k.split("*")[0], float(k.split("*")[1])) if "*" in k else (k, 1.0)
                pnl, _, rate = event_table(leds[led_name], basis)
                tabs[k] = (pnl * scale, rate)
            for hc in (0.0, 0.5):
                out.append({"mix": name, "basis": basis or "pooled", "haircut": hc, **event_mc(tabs, haircut=hc)})
    return out


def ledger_usage(led: pl.DataFrame) -> pl.DataFrame:
    """Hourly cash in positions of a taker ledger (cost from fill to settlement)."""
    iv = led.select(pl.col("t_in").alias("start"), pl.col("settle_ts").alias("end"), pl.col("cost_usd").alias("usd"),
                    pl.lit("position").alias("kind"))
    t0 = led["t_in"].min().replace(minute=0, second=0, microsecond=0)
    return hourly_usage(iv, t0, led["settle_ts"].max())


def guard_binding(E: pl.DataFrame, leds: dict[str, pl.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = {"mentions": "mentions_live_E15_6ev_c40", "cmn_cap25": "cmn_cap25",
             "cmn_shared": "cmn_shared_(97_-_mentions_-_5)", "cmn_cap57": "cmn_cap57", "cmn_cap40": "cmn_cap40"}
    for split, (d0, d1) in (("dev", DEV_DAYS), ("holdout", HOLD_DAYS)):
        hours = pl.datetime_range(datetime.combine(d0, datetime.min.time(), UTC),
                                  datetime.combine(d1, datetime.min.time(), UTC), "1h", time_zone="UTC", eager=True)
        base = pl.DataFrame({"ts": hours})
        cols = {}
        for k, v in names.items():
            p = OUT / "cash" / f"{_slug(v)}_{split}.parquet"
            if p.exists():
                u = pl.read_parquet(p).select("ts", pl.col("usd").alias(k))
                cols[k] = base.join(u, on="ts", how="left")[k].fill_null(0.0).to_numpy()
        for k in ("apr", "hits"):
            s = leds[k].filter(pl.col("split") == split)
            if s.height:
                u = ledger_usage(s).select("ts", pl.col("usd").alias(k))
                cols[k] = base.join(u, on="ts", how="left")[k].fill_null(0.0).to_numpy()
        if "hits" in cols:
            cols["hits_x0.4"] = 0.4 * cols["hits"]
        res = {}
        for combo in (("mentions",), ("mentions", "cmn_cap25"), ("mentions", "cmn_cap40"), ("mentions", "cmn_shared"),
                      ("mentions", "cmn_cap57"), ("mentions", "cmn_shared", "apr", "hits"),
                      ("mentions", "cmn_cap40", "apr", "hits_x0.4")):
            if all(c in cols for c in combo):
                x = sum(cols[c] for c in combo)
                res["+".join(combo)] = {"mean": round(float(x.mean()), 2), "p95": round(float(np.quantile(x, 0.95)), 2),
                                        "max": round(float(x.max()), 2),
                                        "hours_over_60": int((x > 60 + EPS).sum()),
                                        "share_hours_over_60": round(float((x > 60 + EPS).mean()), 4),
                                        "hours_over_92": int((x > ACCOUNT_CASH - 5 + EPS).sum())}
        # daily MTM PnL of the combined book (the guard's day-loss test is equity vs 00:00 UTC)
        dl = {}
        for combo, mix in (("mentions", {"mentions_live_E15_6ev_c40": 1.0}),
                           ("mentions+cmn40+apr", {"mentions_live_E15_6ev_c40": 1.0, "cmn_cap40": 1.0, "apr": 1.0}),
                           ("mentions+cmn40+apr+hits_x0.4", {"mentions_live_E15_6ev_c40": 1.0, "cmn_cap40": 1.0, "apr": 1.0,
                                                             "hits": 0.4}),
                           ("all, hits at $5/event", {"mentions_live_E15_6ev_c40": 1.0, "cmn_shared_(97_-_mentions_-_5)": 1.0,
                                                      "apr": 1.0, "hits": 1.0})):
            x = sum(series(E, m, d0, d1, scale=w) for m, w in mix.items())
            dl[combo] = {"worst_day": round(float(x.min()), 2), "p1_day": round(float(np.quantile(x, 0.01)), 2),
                         "days_below_-15": int((x < -15).sum()), "days_below_-5": int((x < -5).sum())}
        out[split] = {"cash_at_risk": res, "day_loss": dl}
    return out


def kelly_table(leds: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    out = []
    for k in ("mentions_live_E15_6ev_c40", "cmn_shared_(97_-_mentions_-_5)", "cmn_cap25", "apr", "hits", "linked"):
        if k not in leds:
            continue
        for basis in ("dev", "holdout", None):
            p, c, rate = event_table(leds[k], basis)
            if not len(p):
                continue
            r = p / np.maximum(c, EPS)
            se = float(np.std(r, ddof=1)) / math.sqrt(len(r)) if len(r) > 1 else float("inf")
            f = kelly(r)
            fh = kelly(r - 0.5 * r.mean())
            lcb = max(float(r.mean()) - se, 0.0)          # edge one standard error lower (0 if not positive)
            fl = kelly(r - r.mean() + lcb) if lcb > 0 else 0.0
            now = float(c.mean()) / BANKROLL
            out.append({"strategy": k, "basis": basis or "pooled", "events": len(p), "mean_event_cost": round(float(c.mean()), 2),
                        "stake_frac_now": round(now, 4), "ret_per_usd": round(float(r.mean()), 4),
                        "ret_per_usd_se": round(se, 4), "kelly_frac": round(f, 3), "kelly_frac_half_edge": round(fh, 3),
                        "kelly_frac_edge_minus_1se": round(fl, 3),
                        "now_over_kelly": round(now / f, 3) if f > 0 else None,
                        "now_over_kelly_lcb": round(now / fl, 3) if fl > 0 else None,
                        "events_per_week": round(rate * 7, 2), "worst_event": round(float(p.min()), 2)})
    return out


def wilson_lower(wins: int, n: int, z: float = 1.645) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def binary_kelly(led: pl.DataFrame) -> dict[str, Any]:
    """Kelly stake (fraction of bankroll) for one contract-bet at the book's mean price q:
    f = (p - q) / (1 - q), with p the win rate and its one-sided 95% Wilson lower bound."""
    d = led.with_columns((pl.col("cost_usd") / pl.col("n")).alias("q"), (pl.col("payoff_usd") > 0).alias("won"))
    q = float(d["q"].mean())
    n, w = d.height, int(d["won"].sum())
    out = {"n": n, "mean_price": round(q, 3), "win_rate": round(w / n, 4), "breakeven": round(q, 3),
           "win_rate_lb95": round(wilson_lower(w, n), 4)}
    out["kelly_at_mean"] = round(max((w / n - q) / (1 - q), 0.0), 3)
    out["kelly_at_lb95"] = round(max((out["win_rate_lb95"] - q) / (1 - q), 0.0), 3)
    out["stake_per_contract_frac"] = round(q / BANKROLL, 4)
    return out


def gate_sizes(leds: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    from albert.research.mentions_ext import clustered_t

    out = []
    for k in ("mentions_live_E15_6ev_c40", "cmn_shared_(97_-_mentions_-_5)", "apr", "hits", "linked"):
        if k not in leds:
            continue
        d = leds[k]
        x = ((d["payoff_usd"] - d["cost_usd"]) / d["n"].cast(pl.Float64)).to_numpy()
        g = d["event_ticker"].to_numpy()
        mu, t, ke = clustered_t(x, g)
        sd = float(np.std(x, ddof=1))
        deff = clustered_deff(x, g)
        fills_wk = d.height / max((d["settle_ts"].max() - d["t_in"].min()).total_seconds() / 604800, 1e-9)
        nf, nh = required_fills(mu, sd, deff), required_fills(mu / 2, sd, deff)
        out.append({"strategy": k, "fills": d.height, "events": ke, "c_per_contract": round(mu * 100, 2),
                    "sd_per_contract": round(sd, 3), "t_event": round(t, 2) if t else None, "deff": deff,
                    "fills_per_week_research": round(fills_wk, 1),
                    "fills_for_t2_full_edge": nf, "fills_for_t2_half_edge": nh,
                    "weeks_for_t2_half_edge": round(nh / fills_wk, 1) if fills_wk > 0 and nh > 0 else None})
    return out


def mentions_history_months(root: Path = Path(".")) -> dict[str, Any]:
    """Uncapped 1-lot monthly PnL of the frozen 6 h / 12c rule since Mar 2025 (mentions_history prints,
    earnings calls excluded as live): how often a calendar month loses."""
    p = root / "data/cache/fundamental_mentions_history_orders.parquet"
    if not p.exists():
        return {}
    o = pl.read_parquet(p).filter((pl.col("margin") == 0.12) & pl.col("filled") & (pl.col("sgroup") != "earnings"))
    m = (o.group_by("month").agg(pl.len().alias("fills"), pl.col("event").n_unique().alias("events"),
                                 (pl.col("pnl").mean() * 100).round(2).alias("c_per_fill"), pl.col("pnl").sum().round(2).alias("pnl"))
         .sort("month"))
    return {"months": m.to_dicts(), "n_months": m.height, "losing_months": int((m["pnl"] < 0).sum()),
            "losing_months_with_20plus_fills": int(((m["pnl"] < 0) & (m["fills"] >= 20)).sum())}


def choose_allocation(E: pl.DataFrame, strategies: list[str]) -> dict[str, Any]:
    """Pre-specified (written before any holdout scenario number was read):
    1. mentions budget: among the mentions grid, the largest dev book PnL whose dev Sharpe ex-top-5 is
       at least 0.9 x the live policy's and whose dev peak cash is <= $60 (the guard's at-risk cap);
    2. cmn cash policy: given (1), the cmn policy with the largest dev PnL whose addition keeps the
       combined dev Sharpe ex-top-5 >= 0.95 x mentions alone and the combined dev peak cash <= $92
       (the account's $97 minus a $5 reserve);
    3. apr and hits enter only if adding each keeps the combined dev Sharpe ex-top-5 >= 0.95 x the
       book without it (they are 1-lot / $5-per-event books, so cash never binds them).
    The chosen mix is then scored once on the holdout."""
    def stats(mix: list[str], d0: date, d1: date) -> dict[str, Any]:
        x = sum(series(E, k, d0, d1) for k in mix)
        ev = (E.filter(pl.col("strategy").is_in(mix) & (pl.col("day") >= d0) & (pl.col("day") <= d1))
              .with_columns((pl.col("strategy") + ":" + pl.col("event_ticker")).alias("event_ticker")))
        return book_stats(x, ev)

    def peak(mix: list[str], split: str) -> float:
        tot = None
        for k in mix:
            p = OUT / "cash" / f"{_slug(k)}_{split}.parquet"
            if not p.exists():
                continue
            u = pl.read_parquet(p).select("ts", pl.col("usd").alias(k))
            tot = u if tot is None else tot.join(u, on="ts", how="full", coalesce=True)
        if tot is None:
            return 0.0
        return float(tot.fill_null(0.0).select(pl.sum_horizontal([c for c in tot.columns if c != "ts"])).max().item())

    log = []
    men = [s for s in strategies if s.startswith("mentions_")]
    live = "mentions_live_E15_6ev_c40"
    s_live = stats([live], *DEV_DAYS)
    best_m, best_pnl = live, s_live["pnl_usd"]
    for m in men:
        s = stats([m], *DEV_DAYS)
        pk = peak([m], "dev")
        ok = (s["sharpe_ex_top5"] or 0) >= 0.9 * (s_live["sharpe_ex_top5"] or 0) and pk <= 60 + EPS
        log.append({"step": 1, "candidate": m, "dev_pnl": s["pnl_usd"], "dev_sharpe_ex5": s["sharpe_ex_top5"],
                    "dev_peak_cash": round(pk, 2), "ok": ok})
        if ok and s["pnl_usd"] > best_pnl + EPS:
            best_m, best_pnl = m, s["pnl_usd"]
    base = stats([best_m], *DEV_DAYS)
    best_c, best_cp = None, -math.inf
    for c in [s for s in strategies if s.startswith("cmn_")]:
        if "shared" in c and best_m != live:
            continue          # the shared pool was replayed against the live mentions cash path
        s = stats([best_m, c], *DEV_DAYS)
        pk = peak([best_m, c], "dev")
        cs = stats([c], *DEV_DAYS)
        ok = (s["sharpe_ex_top5"] or 0) >= 0.95 * (base["sharpe_ex_top5"] or 0) and pk <= ACCOUNT_CASH - 5 + EPS
        log.append({"step": 2, "candidate": c, "dev_cmn_pnl": cs["pnl_usd"], "combined_dev_sharpe_ex5": s["sharpe_ex_top5"],
                    "combined_dev_peak_cash": round(pk, 2), "ok": ok})
        if ok and cs["pnl_usd"] > best_cp + EPS:
            best_c, best_cp = c, cs["pnl_usd"]
    mix = [best_m] + ([best_c] if best_c else [])
    for k in ("apr", "hits"):
        if k not in strategies:
            continue
        a, b = stats(mix, *DEV_DAYS), stats(mix + [k], *DEV_DAYS)
        ok = (b["sharpe_ex_top5"] or 0) >= 0.95 * (a["sharpe_ex_top5"] or 0)
        log.append({"step": 3, "candidate": k, "without": a["sharpe_ex_top5"], "with": b["sharpe_ex_top5"], "ok": ok})
        if ok:
            mix.append(k)
    return {"chosen": mix, "log": log, "dev": stats(mix, *DEV_DAYS), "holdout": stats(mix, *HOLD_DAYS),
            "holdout_mentions_alone": stats([best_m], *HOLD_DAYS)}


SCENARIOS = {
    "A  mentions live (E15/6ev/c40)": {"mentions_live_E15_6ev_c40": 1},
    "B  A + cmn $25 cap": {"mentions_live_E15_6ev_c40": 1, "cmn_cap25": 1},
    "C  A + cmn shared pool": {"mentions_live_E15_6ev_c40": 1, "cmn_shared_(97_-_mentions_-_5)": 1},
    "D  C + apr": {"mentions_live_E15_6ev_c40": 1, "cmn_shared_(97_-_mentions_-_5)": 1, "apr": 1},
    "E  D + hits": {"mentions_live_E15_6ev_c40": 1, "cmn_shared_(97_-_mentions_-_5)": 1, "apr": 1, "hits": 1},
    "F  E + linked": {"mentions_live_E15_6ev_c40": 1, "cmn_shared_(97_-_mentions_-_5)": 1, "apr": 1, "hits": 1,
                      "linked": 1},
    "G  mentions E20/c50 + cmn $40 cap": {"mentions_E20_6ev_c50": 1, "cmn_cap40": 1},
    "H  A + cmn $57 cap (static)": {"mentions_live_E15_6ev_c40": 1, "cmn_cap57": 1},
    "I  A + cmn $40 cap": {"mentions_live_E15_6ev_c40": 1, "cmn_cap40": 1},
    "J  I + apr + hits at 40% ($2/event)": {"mentions_live_E15_6ev_c40": 1, "cmn_cap40": 1, "apr": 1, "hits": 0.4},
}


def scenarios(E: pl.DataFrame, strategies: list[str]) -> list[dict[str, Any]]:
    out = []
    for name, mix in SCENARIOS.items():
        if any(k not in strategies for k in mix):
            continue
        for split, (d0, d1) in (("dev", DEV_DAYS), ("holdout", HOLD_DAYS)):
            x = sum(series(E, k, d0, d1, scale=w) for k, w in mix.items())
            ev = (E.filter(pl.col("strategy").is_in(list(mix)) & (pl.col("day") >= d0) & (pl.col("day") <= d1))
                  .with_columns((pl.col("strategy") + ":" + pl.col("event_ticker")).alias("event_ticker")))
            row = {"scenario": name, "split": split, **book_stats(x, ev), **month_loss_prob(x),
                   **{f"half_edge_{k}": v for k, v in month_loss_prob(x, haircut=0.5).items() if k == "p_loss_month"}}
            out.append(row)
    return out


# ---- live and forward paper record -------------------------------------------------------------------
def live_fills(root: Path = Path(".")) -> list[dict[str, Any]]:
    """Every live bot fill from the guard's audit (exchange record), with its bot tag."""
    from albert.live_guard import tag_of
    from albert.venues.kalshi_trade import parse_fill

    a = root / "data/live/audit/prod"
    tags = {}
    for line in (a / "order_tags.jsonl").read_text().splitlines():
        d = json.loads(line)
        tags[d["order_id"]] = d.get("client_order_id")
    out = []
    for line in (a / "fills.jsonl").read_text().splitlines():
        f = json.loads(line)["fill"]
        tag = tag_of(tags.get(f.get("order_id")))
        if tag:
            r = parse_fill(f)
            out.append({"tag": tag, "ticker": r.ticker, "side": r.side, "price": r.price, "qty": r.qty, "fee": r.fee,
                        "ts": datetime.fromtimestamp(r.ts_ms / 1000, UTC), "created": f.get("created_time")})
    return out


def market_results(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Result / status of each market from Kalshi's public ``GET /markets`` (unauthenticated; cached
    raw under ``data/cache/portfolio``)."""
    from albert.config import Settings
    from albert.http import CachedClient

    s = Settings()
    c = CachedClient("kalshi", s.kalshi_base, Path("data/cache/portfolio/raw"), min_interval=0.5, max_retries=6)
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(tickers), 100):
        chunk = sorted(tickers)[i:i + 100]
        page = c.get("/markets", {"tickers": ",".join(chunk), "limit": 100}, tag=date.today().isoformat()).payload or {}
        for m in page.get("markets") or []:
            out[m["ticker"]] = {"result": m.get("result"), "status": m.get("status"), "event": m.get("event_ticker")}
    return out


def run_live() -> dict[str, Any]:
    from albert.research.mentions_ext import clustered_t

    fills = [f for f in live_fills() if f["tag"] == "albm"]
    res = market_results([f["ticker"] for f in fills])
    rows = []
    for f in fills:
        m = res.get(f["ticker"], {})
        won = None if m.get("result") not in ("yes", "no") else float(m["result"] == f["side"])
        rows.append({**f, "event": m.get("event"), "result": m.get("result"),
                     "pnl": None if won is None else f["qty"] * (won - f["price"]) - f["fee"]})
    d = pl.DataFrame(rows, infer_schema_length=None)
    s = d.filter(pl.col("pnl").is_not_null())
    x = s["pnl"].to_numpy()
    mu, te, ke = clustered_t(x, s["event"].to_numpy())
    out = {"albm_fills": d.height, "settled": s.height, "events": ke, "c_per_fill": round(mu * 100, 2),
           "t_event": None if te is None else round(te, 2), "pnl_usd": round(float(x.sum()), 2),
           "first": str(d["ts"].min()), "last": str(d["ts"].max()), "yes_lost": int((s["result"] != s["side"]).sum()),
           "avg_price": round(float(s["price"].mean()), 3)}
    # forward paper ledgers
    pf = {}
    for rule in ("cmn", "apr", "hits"):
        p = Path("data/paper_forward") / rule / "ledger.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(line) for line in p.read_text().splitlines()]
        orders = [r for r in recs if r["kind"] == "order"]
        booked = [r for r in orders if r.get("in_book", True)]
        fl = {r["id"] for r in recs if r["kind"] == "fill"}
        pf[rule] = {"orders": len(orders), "booked": len(booked), "shadow": len(orders) - len(booked), "fills": len(fl),
                    "booked_collateral": round(sum(r["price"] * r.get("n", 1) for r in booked if r["id"] not in fl), 2),
                    "settled": sum(1 for r in recs if r["kind"] == "settle")}
    out["paper_forward"] = pf
    d.write_csv(OUT / "live_albm_fills.csv")
    (OUT / "live.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mentions", "cmn", "daily", "analyze", "live"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "mentions":
        run_mentions()
    elif a.cmd == "cmn":
        run_cmn()
    elif a.cmd == "daily":
        run_daily()
    elif a.cmd == "analyze":
        res = run_analyze()
        (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str)[:20000])
    elif a.cmd == "live":
        run_live()


if __name__ == "__main__":
    main()
