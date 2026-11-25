"""Adversarial review of the ``approval2`` paper-trading candidate (KXTRUMPAPPROVE shoulder NO).

The claim (``data/research/approval2/report.md``): 12 h before the 1 PM ET settlement (01:00 ET)
buy NO at 1 - bid on every leg of the daily RCP ladder whose mid is in (5c, 15c] with spread
<= 10c, ~$2 per order, <= $5 per event, held to settlement.

This module only reads the agent's code and outputs (``approval2`` functions are imported, never
changed) plus cached raw data, and writes its own outputs under ``data/research/approval2/review/``.

Run (from /home/ubuntu/albert, heavy steps under rjob):

* ``python -m albert.research.approval2_review trades`` - fetch Kalshi trade prints around every
  primary fill (public API, cached, sequential, 1 req/s). Optional; ``run`` reads only the cache.
* ``python -m albert.research.approval2_review run`` - every check; writes ``review/results.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.research import approval2 as ap2

OUT = ap2.OUT / "review"
SERIES = "KXTRUMPAPPROVE"
LEAD = 12.0
SWEEP = [float(h) for h in range(1, 27)] + [0.5, 1 / 6]
STREAM = Path("data/raw/stream/kalshi")


# ---- shared frames -------------------------------------------------------------------------------
def daily_frame(leads: list[float]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(decision frame, legs, settled values, 1-minute candles) for the daily ladder, built with
    the agent's own ``legs`` / ``event_values`` / ``ladder_frame``."""
    mk = pl.read_parquet(ap2.OUT / "markets.parquet")
    lg = ap2.legs(mk, [SERIES])
    vals = ap2.event_values(lg)
    cand = ap2.load_candles(1, lg["ticker"].to_list())
    dec = ap2.ladder_frame(lg, vals, cand, leads)
    return dec, lg, vals, cand


def shoulder(dec: pl.DataFrame, lead: float) -> pl.DataFrame:
    """The primary's selection (``rule_cells`` H3b) at one lead."""
    d = ap2.priced(dec).filter(pl.col("lead_h") == lead)
    return d.filter((pl.col("mid") > 0.05) & (pl.col("mid") <= 0.15))


def fills_ordered(sel: pl.DataFrame, order: str = "ticker", per_event: float | None = 5.0,
                  seed: int = 0) -> pl.DataFrame:
    """``ap2.fills_from`` with a choice of which legs fill first under the per-event cap:
    ``ticker`` (the agent's code: sort by t, ticker), ``reverse``, ``random``, ``near_modal``
    (closest to the modal leg first), ``far_modal``. ``per_event=None`` = no cap."""
    s = sel.sort("t").unique(["ticker"], keep="first", maintain_order=True)
    if order == "ticker":
        s = s.sort("t", "ticker")
    elif order == "reverse":
        s = s.sort("t", "ticker", descending=[False, True])
    elif order == "random":
        s = s.with_columns(pl.Series("_r", np.random.default_rng(seed).random(s.height))).sort("t", "_r")
    elif order == "cheapest":                 # what ap2.fills_from's docstring says it does
        s = s.sort("t", "bid", descending=[False, True])
    elif order in ("near_modal", "far_modal"):
        s = s.with_columns(pl.col("d_modal").abs().alias("_a")).sort("t", "_a", descending=[False, order == "far_modal"])
    cap = per_event if per_event is not None else 1e9
    rows, spent = [], {}
    for r in s.iter_rows(named=True):          # same arithmetic as ap2.fills_from, order kept
        f = ap2.taker_fill("no", r["bid"], r["ask"], 2.0)
        if f is None or spent.get(r["event"], 0.0) + f["cost"] > cap + 1e-9:
            continue
        spent[r["event"]] = spent.get(r["event"], 0.0) + f["cost"]
        rows.append({"event": r["event"], "ticker": r["ticker"], "series": r["series"], "T": r["T"], "t": r["t"],
                     "lead_h": r["lead_h"], "side": "no", "n": f["n"], "price": f["price"], "fee": f["fee"],
                     "cost": f["cost"], "payoff": float(f["n"]) if r["y"] == 0 else 0.0})
    schema = {"event": pl.String, "ticker": pl.String, "series": pl.String, "T": pl.Datetime("us", "UTC"),
              "t": pl.Datetime("us", "UTC"), "lead_h": pl.Float64, "side": pl.String, "n": pl.Int64,
              "price": pl.Float64, "fee": pl.Float64, "cost": pl.Float64, "payoff": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def attach(f: pl.DataFrame, dec: pl.DataFrame, vals: pl.DataFrame) -> pl.DataFrame:
    """Fill rows + decision context (quote stamp, modal distance, previous value) + the day's
    settled change versus the previous settlement."""
    ctx = dec.select("ticker", "t", "bid", "ask", "mid", "q_ts", "open_time", "lo", "hi", "y", "d_modal", "d_prev",
                     "pv", "modal_centre")
    v = (vals.sort("T").with_columns((pl.col("value") - pl.col("value").shift(1)).alias("dv"),
                                     pl.col("value").shift(1).alias("v_prev"))
         .select(pl.col("event"), "value", "v_prev", "dv"))
    f = f.join(ctx, on=["ticker", "t"], how="left").join(v, on="event", how="left")
    return f.with_columns(
        pl.when(pl.col("d_modal") > 0).then(pl.lit("above")).when(pl.col("d_modal") < 0).then(pl.lit("below"))
        .otherwise(pl.lit("modal")).alias("side_modal"),
        pl.when(pl.col("dv") > 0).then(pl.lit("rose")).when(pl.col("dv") < 0).then(pl.lit("fell"))
        .when(pl.col("dv") == 0).then(pl.lit("same")).otherwise(pl.lit("n/a")).alias("day_dir"),
        ((pl.col("payoff") - pl.col("cost")) / pl.col("n")).alias("pc"),
        pl.when(pl.col("T") < ap2.DEV_END).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"),
        # where the settled value ended relative to the modal leg at the decision (overnight move)
        pl.when(pl.col("value") > pl.col("modal_centre")).then(pl.lit("up"))
        .when(pl.col("value") < pl.col("modal_centre")).then(pl.lit("down")).otherwise(pl.lit("at_modal")).alias("vs_modal"))


def summ(f: pl.DataFrame) -> dict[str, Any]:
    s = ap2.summarize(f) if f.height else {"fills": 0}
    if f.height:
        s["yes_settled"] = int((f["payoff"] == 0).sum())
    return s


# ---- check 1: quote timing --------------------------------------------------------------------------
def cache_only_api():
    """The repo's Kalshi client in cache-only mode (any network request raises)."""
    api = ap2._api(record_live=False)

    def _no_net(*a, **k):
        raise RuntimeError("cache miss (review runs cache-only)")

    api.client._request = _no_net
    return api


def raw_candle_check(fx: pl.DataFrame, mk: pl.DataFrame) -> dict[str, Any]:
    """For every traded row: reopen the cached raw candlestick payload the agent fetched (same
    request as ``ap2._candle_jobs``), take the last candle whose ``end_period_ts`` <= t, and
    compare its yes_bid / yes_ask close with the quote the backtest used."""
    api = cache_only_api()
    m = mk.select("ticker", "open_time", "close_time").to_dicts()
    meta = {r["ticker"]: r for r in m}
    rows = []
    for r in fx.iter_rows(named=True):
        mm = meta[r["ticker"]]
        hi = mm["close_time"]
        lo = max(mm["open_time"] or hi - timedelta(days=8), hi - timedelta(minutes=4900))
        page = None
        for hist in (False, True):
            try:
                page = api.candles(r["ticker"], lo, hi + timedelta(minutes=1), 1, hist)
                break
            except RuntimeError:
                continue
        if page is None:
            rows.append({"ticker": r["ticker"], "t": r["t"], "found": False})
            continue
        tsec = int(r["t"].timestamp())
        cs = [c for c in page.payload.get("candlesticks") or [] if c["end_period_ts"] <= tsec]
        nxt = [c for c in page.payload.get("candlesticks") or [] if c["end_period_ts"] > tsec]
        last = cs[-1] if cs else None
        bid = ap2.num(ap2._field(last.get("yes_bid") or {}, "close")) if last else None
        ask = ap2.num(ap2._field(last.get("yes_ask") or {}, "close")) if last else None
        nb = ap2.num(ap2._field(nxt[0].get("yes_bid") or {}, "close")) if nxt else None
        rows.append({"ticker": r["ticker"], "t": r["t"], "found": True, "raw_end": datetime.fromtimestamp(last["end_period_ts"], UTC) if last else None,
                     "raw_bid": bid, "raw_ask": ask, "bt_bid": r["bid"], "bt_ask": r["ask"],
                     "match": bid is not None and abs(bid - r["bid"]) < 1e-9 and ask is not None and abs(ask - r["ask"]) < 1e-9,
                     "next_bid": nb, "next_differs": nb is not None and abs(nb - r["bid"]) > 1e-9,
                     "fetched_at": str(page.fetched_at)})
    df = pl.DataFrame(rows, infer_schema_length=None)
    found = df.filter(pl.col("found"))
    return {"rows": df.height, "raw_found": found.height, "bid_ask_match": int(found["match"].sum()),
            "mismatches": found.filter(~pl.col("match")).select("ticker", "t", "raw_bid", "bt_bid", "raw_ask", "bt_ask").to_dicts()[:10],
            "raw_end_le_t": int((found["raw_end"] <= found["t"]).sum()),
            "next_candle_bid_differs": int(found["next_differs"].sum()),
            "fetched_at_range": [found["fetched_at"].min(), found["fetched_at"].max()]}


def quote_timing(fx: pl.DataFrame, dec12: pl.DataFrame, lg: pl.DataFrame) -> dict[str, Any]:
    age = (fx["t"] - fx["q_ts"]).dt.total_seconds() / 60
    # legs of each event that opened after the decision (they cannot be in the frame)
    first_open = lg.group_by("event").agg(pl.col("open_time").min().alias("o0"), pl.col("open_time").max().alias("o1"),
                                          pl.col("T").first())
    late = lg.join(first_open.select("event", "o0"), on="event").filter(pl.col("open_time") > pl.col("o0"))
    late12 = lg.with_columns((pl.col("T") - timedelta(hours=LEAD)).alias("t")).filter(
        pl.col("open_time") > pl.col("t") - timedelta(minutes=30))
    t_et = fx["t"].dt.convert_time_zone("America/New_York")
    return {"fills": fx.height,
            "q_ts_le_t": int((fx["q_ts"] <= fx["t"]).sum()),
            "open_plus30_le_t": int((fx["open_time"] + timedelta(minutes=30) <= fx["t"]).sum()),
            "decision_local_times": sorted({f"{x:%H:%M %Z}" for x in t_et.to_list()}),
            "quote_age_min": {"median": float(age.median()), "p90": float(age.quantile(0.9)), "max": float(age.max()),
                              "share_gt_10min": float((age > 10).mean()), "share_gt_60min": float((age > 60).mean())},
            "legs_opened_after_event_open": late.height,
            "legs_opened_after_t12_minus_30min": late12.height,
            "legs_opened_after_t12_minus_30min_list": late12.select("ticker", "open_time", "t").head(10).to_dicts(),
            "frame_rows_at_12h": dec12.height}


def candle_semantics(cand: pl.DataFrame, l2: pl.DataFrame) -> dict[str, Any]:
    """Compare 1-minute candle bid closes with the recorded L2 best bid at end_ts, end_ts - 60 s and
    end_ts + 60 s (only minutes where the L2 bid changed in the surrounding 2 minutes count)."""
    if l2.height == 0:
        return {"note": "no L2 rows"}
    rows = []
    for (tk,), g in l2.sort("ts").group_by(["ticker"], maintain_order=True):
        c = cand.filter(pl.col("ticker") == tk).filter((pl.col("end_ts") >= g["ts"].min() + timedelta(minutes=2))
                                                       & (pl.col("end_ts") <= g["ts"].max()))
        if c.height == 0:
            continue
        b = g.select(pl.col("ts"), pl.col("best_bid")).sort("ts")
        for off in (-60, 0, 60):
            q = c.select("ticker", "end_ts", "bid").with_columns((pl.col("end_ts") + timedelta(seconds=off)).alias("at")).sort("at")
            j = q.join_asof(b, left_on="at", right_on="ts", strategy="backward")
            rows.append(j.with_columns(pl.lit(off).alias("off")))
    if not rows:
        return {"note": "no overlapping candles"}
    j = pl.concat(rows)
    piv = j.pivot(on="off", index=["ticker", "end_ts", "bid"], values="best_bid")
    piv = piv.rename({"-60": "l2_m60", "0": "l2_0", "60": "l2_p60"})
    informative = piv.filter((pl.col("l2_m60") != pl.col("l2_0")) | (pl.col("l2_0") != pl.col("l2_p60")))
    return {"minutes_compared": piv.height,
            "match_at_end_ts": int((piv["bid"] == piv["l2_0"]).sum()),
            "informative_minutes": informative.height,
            "informative_match_end_ts": int((informative["bid"] == informative["l2_0"]).sum()),
            "informative_match_end_minus_60": int((informative["bid"] == informative["l2_m60"]).sum()),
            "informative_match_end_plus_60": int((informative["bid"] == informative["l2_p60"]).sum())}


# ---- check 2: settlement ---------------------------------------------------------------------------
def raw_markets(api) -> dict[str, dict[str, Any]]:
    """Every KXTRUMPAPPROVE market from the cached raw pages the agent's ``discover`` read
    (historical + live settled, tag 2026-11-20), following the cursors in the cache."""
    out: dict[str, dict[str, Any]] = {}
    for path, params, tag in (("/historical/markets", {"series_ticker": SERIES, "limit": 1000}, None),
                              ("/markets", {"series_ticker": SERIES, "status": "settled", "limit": 1000}, "2026-11-20")):
        cursor = None
        while True:
            page = api.client.get(path, {**params, "cursor": cursor}, cache=True, tag=tag)
            for m in page.payload.get("markets") or []:
                out[m["ticker"]] = {"result": m.get("result"), "expiration_value": m.get("expiration_value"),
                                    "status": m.get("status"), "settlement_ts": m.get("settlement_ts"),
                                    "event": m.get("event_ticker"), "sub": m.get("yes_sub_title"), "raw": str(page.raw_path)}
            cursor = page.payload.get("cursor")
            if not cursor or not page.payload.get("markets"):
                break
    return out


def settlement_check(fx: pl.DataFrame, lg: pl.DataFrame) -> dict[str, Any]:
    api = cache_only_api()
    raw = raw_markets(api)
    bad, rows = [], []
    for r in fx.iter_rows(named=True):
        m = raw.get(r["ticker"])
        if m is None:
            bad.append({"ticker": r["ticker"], "why": "not in raw"})
            continue
        won_no = m["result"] == "no"
        exp_pay = float(r["n"]) if won_no else 0.0
        b = ap2.bracket(m["sub"])
        ev = ap2._ev_tenths(m["expiration_value"])
        inside = None if (b is None or ev is None) else (b[0] <= ev <= b[1])
        rows.append({"ticker": r["ticker"], "event": r["event"], "raw_result": m["result"], "payoff_bt": r["payoff"],
                     "payoff_raw": exp_pay, "ev": ev, "inside": inside})
        if abs(exp_pay - r["payoff"]) > 1e-9 or (inside is not None and inside != (m["result"] == "yes")):
            bad.append({"ticker": r["ticker"], "raw_result": m["result"], "payoff_bt": r["payoff"], "ev": ev, "bracket": b})
    df = pl.DataFrame(rows)
    # event integrity over every leg of the traded events
    ev_tab = []
    for e in fx["event"].unique().to_list():
        legs_e = [v for v in raw.values() if v["event"] == e]
        ev_tab.append({"event": e, "legs": len(legs_e), "yes_legs": sum(v["result"] == "yes" for v in legs_e),
                       "values": len({v["expiration_value"] for v in legs_e})})
    et = pl.DataFrame(ev_tab)
    dates = fx.select(pl.col("T").dt.convert_time_zone("America/New_York").dt.date().alias("d"), "event").unique()
    return {"fills_checked": df.height, "payoff_mismatches": bad,
            "events": fx["event"].n_unique(), "settlement_dates": dates["d"].n_unique(),
            "dates_with_more_than_one_event": int((dates.group_by("d").len()["len"] > 1).sum()),
            "duplicate_tickers": int(fx.height - fx["ticker"].n_unique()),
            "events_not_exactly_one_yes": et.filter(pl.col("yes_legs") != 1).to_dicts(),
            "events_multiple_expiration_values": et.filter(pl.col("values") != 1).to_dicts(),
            "yes_fills": df.filter(pl.col("raw_result") == "yes").select("ticker", "ev").to_dicts()}


# ---- check 3: drift ----------------------------------------------------------------------------------
def drift_tables(fx: pl.DataFrame, vals: pl.DataFrame) -> dict[str, Any]:
    def agg(by: list[str]) -> list[dict[str, Any]]:
        res = []
        for key, g in fx.group_by(by, maintain_order=False):
            s = summ(g)
            res.append({**dict(zip(by, key, strict=True)), "fills": s["fills"], "events": s.get("events"),
                        "yes": s.get("yes_settled"), "c_per_contract": s.get("c_per_contract"), "t_event": s.get("t_event"),
                        "pnl_usd": s.get("pnl_usd")})
        return sorted(res, key=lambda r: tuple(str(r[b]) for b in by))

    v = vals.sort("T").with_columns((pl.col("value") - pl.col("value").shift(1)).alias("dv"))
    days = []
    for split, part in (("dev", v.filter(pl.col("T") < ap2.DEV_END)), ("holdout", v.filter(pl.col("T") >= ap2.DEV_END))):
        days.append({"split": split, "days": part.height, "rose": int((part["dv"] > 0).sum()), "fell": int((part["dv"] < 0).sum()),
                     "same": int((part["dv"] == 0).sum()), "first": part["value"][0], "last": part["value"][-1],
                     "net_tenths": int(part["value"][-1] - part["value"][0])})
    # profit decomposition: what share of holdout $ is above-modal on falling days
    rng = []
    for split, part in (("dev", vals.filter(pl.col("T") < ap2.DEV_END)), ("holdout", vals.filter(pl.col("T") >= ap2.DEV_END))):
        part = part.sort("T")
        rng.append({"split": split, "min": part["value"].min(), "max": part["value"].max(),
                    "first_date": str(part["T"][0].date()), "last_date": str(part["T"][-1].date())})
    eq = {}
    for split in ("dev", "holdout"):
        a = fx.filter((pl.col("split") == split) & (pl.col("side_modal") == "above"))["pc"].mean()
        b = fx.filter((pl.col("split") == split) & (pl.col("side_modal") == "below"))["pc"].mean()
        eq[split] = {"above_c": round(100 * a, 2), "below_c": round(100 * b, 2), "equal_weight_c": round(50 * (a + b), 2)}
    return {"day_changes": days, "value_range": rng, "side_equal_weight": eq,
            "by_split_side": agg(["split", "side_modal"]), "by_split_dir": agg(["split", "day_dir"]),
            "by_split_side_dir": agg(["split", "side_modal", "day_dir"]),
            "by_split_vs_modal": agg(["split", "vs_modal"]), "by_split_side_vs_modal": agg(["split", "side_modal", "vs_modal"])}


# ---- check 4: other hours, weekly, executability -------------------------------------------------
def lead_sweep(dec: pl.DataFrame) -> list[dict[str, Any]]:
    out = []
    for L in sorted(dec["lead_h"].unique().to_list()):
        sel = shoulder(dec, L)
        f = ap2.fills_from(sel, "no")
        allp = ap2.priced(dec).filter(pl.col("lead_h") == L)
        for split, part in (("dev", f.filter(pl.col("T") < ap2.DEV_END)), ("holdout", f.filter(pl.col("T") >= ap2.DEV_END)), ("pooled", f)):
            s = summ(part)
            out.append({"lead_h": L, "split": split, "fills": s["fills"], "events": s.get("events"), "yes": s.get("yes_settled"),
                        "c_per_contract": s.get("c_per_contract"), "t_event": s.get("t_event"), "pnl_usd": s.get("pnl_usd")})
        age = (allp["t"] - allp["q_ts"]).dt.total_seconds() / 60
        out[-1] |= {"median_quote_age_min_all_legs": float(age.median()) if allp.height else None,
                    "median_spread_c_shoulder": float(((sel["ask"] - sel["bid"]) * 100).median()) if sel.height else None}
    return out


def weekly_sweep() -> list[dict[str, Any]]:
    mk = pl.read_parquet(ap2.OUT / "markets.parquet")
    lg = ap2.legs(mk, ["KXAPRPOTUS"]).filter(pl.col("T") >= datetime(2025, 1, 20, tzinfo=UTC))
    cand = ap2.load_candles(60, lg["ticker"].to_list())
    dec = ap2.ladder_frame(lg, ap2.event_values(lg), cand, ap2.LEADS["KXAPRPOTUS"])
    out = []
    for L in sorted(dec["lead_h"].unique().to_list()):
        f = ap2.fills_from(shoulder(dec, L), "no")
        for split, part in (("dev", f.filter(pl.col("T") < ap2.DEV_END)), ("holdout", f.filter(pl.col("T") >= ap2.DEV_END)),
                            ("since_sep2026", f.filter(pl.col("T") >= datetime(2026, 9, 1, tzinfo=UTC)))):
            s = summ(part)
            out.append({"lead_h": L, "split": split, "fills": s["fills"], "events": s.get("events"), "yes": s.get("yes_settled"),
                        "c_per_contract": s.get("c_per_contract"), "t_event": s.get("t_event")})
    return out


def exec_candles(fx: pl.DataFrame, cand: pl.DataFrame) -> dict[str, Any]:
    """Per fill, from 1-minute candles of that leg: traded volume in the hour before / after t,
    minutes with a print at a YES price <= the bid we hit, and whether the bid was still >= our
    hit price at t + 1, 5, 15 minutes."""
    c = cand.select("ticker", "end_ts", "bid", "volume", "p_lo", "p_hi").sort("ticker", "end_ts") \
        .with_columns(pl.col("bid").forward_fill().over("ticker"))
    rows = []
    for r in fx.select("ticker", "t", "bid", "split").iter_rows(named=True):
        g = c.filter(pl.col("ticker") == r["ticker"])
        after = g.filter((pl.col("end_ts") > r["t"]) & (pl.col("end_ts") <= r["t"] + timedelta(hours=1)))
        before = g.filter((pl.col("end_ts") > r["t"] - timedelta(hours=1)) & (pl.col("end_ts") <= r["t"]))
        traded = after.filter(pl.col("volume") > 0)
        at_or_below = traded.filter(pl.col("p_lo") <= r["bid"] + 1e-9)

        def bid_at(m: int) -> float | None:
            x = g.filter(pl.col("end_ts") <= r["t"] + timedelta(minutes=m))
            return x["bid"][-1] if x.height else None
        rows.append({"ticker": r["ticker"], "t": r["t"], "split": r["split"], "bid": r["bid"],
                     "vol_after_1h": float(after["volume"].sum()), "vol_before_1h": float(before["volume"].sum()),
                     "minutes_traded_after": traded.height, "minutes_print_le_bid_after": at_or_below.height,
                     "vol_le_bid_after": float(at_or_below["volume"].sum()),
                     "bid_ge_at_1m": (b := bid_at(1)) is not None and b >= r["bid"] - 1e-9,
                     "bid_ge_at_5m": (b := bid_at(5)) is not None and b >= r["bid"] - 1e-9,
                     "bid_ge_at_15m": (b := bid_at(15)) is not None and b >= r["bid"] - 1e-9})
    df = pl.DataFrame(rows)
    res = {}
    for split in ("dev", "holdout"):
        d = df.filter(pl.col("split") == split)
        res[split] = {"fills": d.height, "share_any_trade_1h_after": float((d["minutes_traded_after"] > 0).mean()),
                      "share_print_le_bid_1h_after": float((d["minutes_print_le_bid_after"] > 0).mean()),
                      "median_vol_after_1h": float(d["vol_after_1h"].median()), "sum_vol_le_bid_after": float(d["vol_le_bid_after"].sum()),
                      "share_any_trade_1h_before": float((d["vol_before_1h"] > 0).mean()),
                      "share_bid_still_there_1m": float(d["bid_ge_at_1m"].mean()),
                      "share_bid_still_there_15m": float(d["bid_ge_at_15m"].mean())}
    _save(df, OUT / "exec_candles.parquet")
    return res


def trade_prints(fx: pl.DataFrame, *, fetch: bool) -> dict[str, Any]:
    """Kalshi trade prints in [t, t + 1 h] for each fill (cached; fetched only with ``fetch``):
    count of prints at a YES price <= the bid we hit (someone sold YES there) and contracts."""
    api = ap2._api(min_interval=1.0) if fetch else cache_only_api()
    cutoff = api.cutoff().get("market_settled_ts") if fetch else None
    rows, miss = [], 0
    for r in fx.select("ticker", "t", "T", "bid", "split").iter_rows(named=True):
        trades = None
        for hist in ((r["T"] < cutoff,) if cutoff else (False, True)):
            try:
                trades = []
                for page in api.trades(r["ticker"], hist, min_ts=r["t"], max_ts=r["t"] + timedelta(hours=1)):
                    trades.extend(page.payload.get("trades") or [])
                break
            except RuntimeError:
                trades = None
                continue
        if trades is None:
            miss += 1
            continue
        le = [x for x in trades if (ap2.num(ap2._field(x, "yes_price")) or 9) <= r["bid"] + 1e-9]
        rows.append({"ticker": r["ticker"], "t": r["t"], "split": r["split"], "prints_1h": len(trades),
                     "prints_le_bid": len(le), "contracts_le_bid": sum(ap2.num(ap2._field(x, "count")) or 0 for x in le),
                     "sold_yes_le_bid": sum(1 for x in le if x.get("taker_side") == "no")})
    if not rows:
        return {"note": "no cached trades; run the `trades` subcommand", "missing": miss}
    df = pl.DataFrame(rows)
    _save(df, OUT / "trade_prints.parquet")
    res = {"missing": miss}
    for split in ("dev", "holdout"):
        d = df.filter(pl.col("split") == split)
        if d.height:
            res[split] = {"fills": d.height, "share_with_any_print_1h": float((d["prints_1h"] > 0).mean()),
                          "share_with_print_le_bid_1h": float((d["prints_le_bid"] > 0).mean()),
                          "median_prints_1h": float(d["prints_1h"].median()),
                          "contracts_le_bid_total": float(d["contracts_le_bid"].sum())}
    return res


def _lines(path: Path):
    import zstandard

    with open(path, "rb") as fh:
        yield from io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")


def l2_books(days: list[str], hours: tuple[str, ...] = ("06", "07")) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rebuild KXTRUMPAPPROVE books from the stream recorder's k-connection files (the random
    sample recorder) for the given UTC days/hours: best YES bid/size and best NO bid/size after
    every message, plus trade prints."""
    from albert.stream import KalshiBook

    books, trades = [], []
    for d in days:
        for f in sorted((STREAM / d).glob("*-k*.jsonl.zst")):
            if f.name[:2] not in hours:
                continue
            kb = KalshiBook()
            for line in _lines(f):
                if SERIES not in line:
                    continue
                rec = json.loads(line)
                fr = json.loads(rec["frame"])
                typ, msg = fr.get("type"), fr.get("msg") or {}
                tk = msg.get("market_ticker", "")
                if not tk.startswith(SERIES):
                    continue
                ts = datetime.fromtimestamp(rec["recv_ns"] / 1e9, UTC)
                if typ == "trade":
                    trades.append({"ticker": tk, "ts": ts, "yes_price": float(msg.get("yes_price_dollars") or 0),
                                   "count": float(msg.get("count_fp") or 0), "taker_side": msg.get("taker_side"), "file": f.name})
                    continue
                if typ not in ("orderbook_snapshot", "orderbook_delta"):
                    continue
                kb.apply(typ, msg)
                yb = kb.levels(tk, "yes", 3)
                nb = kb.levels(tk, "no", 3)
                books.append({"ticker": tk, "ts": ts, "kind": typ, "best_bid": yb[0][0] if yb else None,
                              "bid_size": yb[0][1] if yb else None, "bid2": yb[1][0] if len(yb) > 1 else None,
                              "bid2_size": yb[1][1] if len(yb) > 1 else None,
                              "best_ask": round(1 - nb[0][0], 4) if nb else None, "ask_size": nb[0][1] if nb else None,
                              "file": f.name})
    bs = pl.DataFrame(books, schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "kind": pl.String,
                                     "best_bid": pl.Float64, "bid_size": pl.Float64, "bid2": pl.Float64, "bid2_size": pl.Float64,
                                     "best_ask": pl.Float64, "ask_size": pl.Float64, "file": pl.String})
    tr = pl.DataFrame(trades, schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "yes_price": pl.Float64,
                                      "count": pl.Float64, "taker_side": pl.String, "file": pl.String})
    return bs, tr


def l2_check(bs: pl.DataFrame, tr: pl.DataFrame, dec: pl.DataFrame) -> dict[str, Any]:
    """For every recorded KXTRUMPAPPROVE leg near 01:00 ET: the first rebuilt book after the
    recording began and the book at 01:40 ET; for legs the primary selected at 12 h, the YES-bid
    size we would have hit and the trade prints until 02:00 ET."""
    if bs.height == 0:
        return {"note": "no recorded books"}
    first = bs.sort("ts").group_by("ticker").agg(pl.all().first())
    d12 = dec.filter(pl.col("lead_h") == LEAD).select("ticker", "t", "bid", "ask", "mid", "y", "event")
    j = first.join(d12, on="ticker", how="inner")
    sel = j.filter((pl.col("mid") > 0.05) & (pl.col("mid") <= 0.15) & ((pl.col("ask") - pl.col("bid")) <= 0.10))
    rows = []
    for r in j.iter_rows(named=True):
        later = bs.filter((pl.col("ticker") == r["ticker"]) & (pl.col("ts") <= r["t"] + timedelta(minutes=40))).sort("ts")
        b40 = later[-1] if later.height else None
        prints = tr.filter((pl.col("ticker") == r["ticker"]) & (pl.col("ts") <= r["t"] + timedelta(hours=1)))
        rows.append({"ticker": r["ticker"], "t": r["t"], "first_book_ts": r["ts"], "candle_bid_at_t": r["bid"], "candle_ask_at_t": r["ask"],
                     "selected": (0.05 < (r["mid"] or 0) <= 0.15) and ((r["ask"] or 1) - (r["bid"] or 0) <= 0.10),
                     "l2_first_bid": r["best_bid"], "l2_first_bid_size": r["bid_size"], "l2_first_ask": r["best_ask"],
                     "l2_bid_at_t40": b40["best_bid"][0] if b40 is not None else None,
                     "l2_bid_size_at_t40": b40["bid_size"][0] if b40 is not None else None,
                     "prints_to_t60": prints.height, "contracts_to_t60": float(prints["count"].sum()) if prints.height else 0.0})
    df = pl.DataFrame(rows, infer_schema_length=None).sort("t", "ticker")
    _save(df, OUT / "l2_legs.parquet")
    s = df.filter(pl.col("selected"))
    return {"recorded_legs_with_12h_quote": df.height, "dates": sorted({str(x.date()) for x in df["t"].to_list()}),
            "selected_legs_recorded": s.height,
            "selected": s.select("ticker", "first_book_ts", "candle_bid_at_t", "l2_first_bid", "l2_first_bid_size",
                                 "l2_bid_at_t40", "l2_bid_size_at_t40", "prints_to_t60").to_dicts(),
            "all_recorded_shoulder_like": df.filter((pl.col("l2_first_bid") >= 0.03) & (pl.col("l2_first_bid") <= 0.16))
            .select("ticker", "l2_first_bid", "l2_first_bid_size", "l2_first_ask").to_dicts(),
            "n_selected_dropped_note": int(sel.height)}


# ---- check 5: statistics ----------------------------------------------------------------------------
def poisson_binomial_cdf(p: np.ndarray, k: int) -> float:
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    return float(dist[: k + 1].sum())


def exclusive_sim(f: pl.DataFrame, prob: str, k_obs: int, sims: int = 200_000, seed: int = 11) -> float:
    """P(total YES <= k_obs) when, within an event, at most one traded leg can settle YES (legs of
    one ladder are mutually exclusive) and leg i is YES with probability ``prob``."""
    rng = np.random.default_rng(seed)
    total = np.zeros(sims, dtype=np.int64)
    for (_e,), g in f.group_by(["event"]):
        p = np.clip(g[prob].to_numpy(), 0, 1)
        if p.sum() > 1:
            p = p / p.sum()
        u = rng.random(sims)
        total += (u < p.sum()).astype(np.int64)
    return float((total <= k_obs).mean())


def norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


def stats_block(fx: pl.DataFrame, pooled: pl.DataFrame) -> dict[str, Any]:
    res: dict[str, Any] = {}
    fx = fx.with_columns((1 - pl.col("price") - pl.col("fee") / pl.col("n")).alias("p_be"))
    for split in ("dev", "holdout", "pooled"):
        f = fx if split == "pooled" else fx.filter(pl.col("split") == split)
        pc = f["pc"].to_numpy()
        mu, t_ev, k = ap2.cluster_t(pc, f["event"].to_numpy())
        t2 = ap2.two_way_t(pc, f["event"].to_numpy(), f["T"].dt.date().cast(pl.String).to_numpy())
        p_be = (1 - f["price"] - f["fee"] / f["n"]).to_numpy()          # YES prob at which the NO buy breaks even
        p_mid = f["mid"].to_numpy()
        k_yes = int((f["payoff"] == 0).sum())
        res[f"{split}_exact" if split == "pooled" else split] = {"c_per_contract": round(100 * mu, 2), "t_event": round(t_ev, 2) if t_ev else None, "t_two_way": round(t2, 2) if t2 else None,
                      "clusters": k, "yes": k_yes, "expected_yes_at_breakeven": round(float(p_be.sum()), 2),
                      "expected_yes_at_mid": round(float(p_mid.sum()), 2),
                      "p_le_yes_given_breakeven": round(poisson_binomial_cdf(p_be, k_yes), 4),
                      "p_le_yes_given_mid": round(poisson_binomial_cdf(p_mid, k_yes), 4),
                      "p_le_yes_breakeven_exclusive_sim": round(exclusive_sim(f, "p_be", k_yes), 4),
                      "p_le_yes_mid_exclusive_sim": round(exclusive_sim(f, "mid", k_yes), 4),
                      "t_one_sided_p_normal": round(norm_sf(t_ev), 4) if t_ev else None,
                      "extra_losses_to_zero": round(float(pc.sum()), 2)}   # a win turned loss costs $1 per contract
    # Holm over the 170 dev cells (normal approximation, two-sided)
    cells = pl.read_parquet(ap2.OUT / "dev_cells.parquet")
    p = []
    for r in cells.iter_rows(named=True):
        t = r.get("t_event")
        p.append(1.0 if t is None or (isinstance(t, float) and math.isnan(t)) else 2 * norm_sf(abs(t)))
    p = np.array(p)
    m = len(p)
    order = np.argsort(p)
    holm = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[idx]))
        holm[idx] = run
    cells = cells.with_columns(pl.Series("p", p), pl.Series("p_holm", holm))
    prim = cells.filter((pl.col("series") == SERIES) & (pl.col("cell") == "H3b shoulder NO 5c<mid<=15c L12h")).row(0, named=True)
    real = cells.filter(pl.col("win_rate") < 1)
    res["holm"] = {"cells": m, "primary_dev_p": round(prim["p"], 3), "primary_dev_p_holm": round(prim["p_holm"], 3),
                   "dev_cells_p_holm_lt_0.05": int((cells["p_holm"] < 0.05).sum()),
                   "of_which_with_a_loss": int((real["p_holm"] < 0.05).sum()),
                   "t_needed_bonferroni_170": round(float(_z(0.05 / (2 * m))), 2),
                   "qualifying_daily_cells": cells.filter((pl.col("series") == SERIES) & (pl.col("fills") >= 40) & (pl.col("events") >= 15)
                                                          & (pl.col("win_rate") < 1) & (pl.col("c_per_contract") > 0))
                   .select("cell", "fills", "c_per_contract", "t_event").to_dicts()}
    # leave-one-week-out on the pooled sample
    pw = pooled.with_columns(pl.col("T").dt.convert_time_zone("America/New_York").dt.truncate("1w").dt.date().alias("wk"))
    lowo = []
    for w in sorted(pw["wk"].unique().to_list()):
        s = ap2.summarize(pw.filter(pl.col("wk") != w))
        ww = pw.filter(pl.col("wk") == w)
        lowo.append({"dropped_week": str(w), "week_fills": ww.height, "week_pnl": round(float((ww["payoff"] - ww["cost"]).sum()), 2),
                     "c_per_contract": s["c_per_contract"], "t_event": s["t_event"]})
    sp = ap2.summarize(pw)
    by_week = (pw.with_columns((pl.col("payoff") - pl.col("cost")).alias("pnl")).group_by("wk")
               .agg(pl.len().alias("fills"), pl.col("pnl").sum().round(2), (pl.col("payoff") == 0).sum().alias("yes")).sort("wk"))
    res["pooled"] = {**{k: sp[k] for k in ("fills", "events", "c_per_contract", "t_event", "t_two_way", "pnl_usd", "pct_pos_weeks")},
                     "lowo_min_c": min(x["c_per_contract"] for x in lowo), "lowo_min_t": min(x["t_event"] for x in lowo),
                     "lowo": lowo, "weeks": [{**r, "wk": str(r["wk"])} for r in by_week.to_dicts()]}
    for name, lo, hi in (("after_2026-11-05", datetime(2026, 11, 6, tzinfo=UTC), None),
                         ("after_2026-11-16", datetime(2026, 11, 17, tzinfo=UTC), None),
                         ("Oct", datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC)),
                         ("Nov", datetime(2026, 11, 1, tzinfo=UTC), None)):
        part = pooled.filter(pl.col("T") >= lo)
        if hi is not None:
            part = part.filter(pl.col("T") < hi)
        res[name] = summ(part)
    return res


def _z(p_one_sided: float) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if norm_sf(mid) > p_one_sided:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def holdout_family(dec_agent: pl.DataFrame) -> list[dict[str, Any]]:
    """Every daily dev cell scored on the holdout (post hoc, context only): is 12 h H3b special?"""
    hold = dec_agent.filter(pl.col("T") >= ap2.DEV_END)
    out = []
    for name, side, sel in ap2.rule_cells(hold):
        s = summ(ap2.fills_from(sel, side))
        out.append({"cell": name, "fills": s["fills"], "c_per_contract": s.get("c_per_contract"), "t_event": s.get("t_event"),
                    "yes": s.get("yes_settled")})
    return out


# ---- Sharpe / vol for variants ---------------------------------------------------------------------
def book_metrics(f: pl.DataFrame, marks: pl.DataFrame, name: str, split: str) -> dict[str, Any]:
    if f.height == 0:
        return {"variant": name, "split": split, "fills": 0}
    base = f.select("event", "ticker", "series", "T", "t", "lead_h", "side", "n", "price", "fee", "cost", "payoff")
    book = ap2.capped_book(base)
    led = ap2.to_ledger(book, "approval2_review", name, split)
    _save(led, OUT / "ledgers" / f"ledger_{name}_{split}.parquet")
    s = ap2.summarize(book)
    vb = ap2.vol_block(led, marks)
    return {"variant": name, "split": split, "fills": s["fills"], "events": s["events"], "c_per_contract": s["c_per_contract"],
            "t_event": s["t_event"], **{k: vb.get(k) for k in ("sharpe", "sharpe_ci90", "sharpe_ex_top5", "ann_vol_pct", "usd_per_week",
                                                                 "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct")}}


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


# ---- driver -------------------------------------------------------------------------------------------
def run(*, with_l2: bool = True) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {}
    dec, lg, vals, cand = daily_frame(SWEEP)
    mk = pl.read_parquet(ap2.OUT / "markets.parquet").filter(pl.col("series") == SERIES)

    # replicate the primary exactly as the agent's code does (per split, then pooled)
    sel12 = shoulder(dec, LEAD)
    f_dev = ap2.fills_from(sel12.filter(pl.col("T") < ap2.DEV_END), "no")
    f_hold = ap2.fills_from(sel12.filter(pl.col("T") >= ap2.DEV_END), "no")
    pooled = ap2.fills_from(sel12, "no")
    fx = attach(pl.concat([f_dev, f_hold]), dec.filter(pl.col("lead_h") == LEAD), vals)
    _save(fx, OUT / "primary_fills.parquet")
    led_h = pl.read_parquet(ap2.OUT / "ledgers" / "ledger_primary_holdout.parquet")
    res["replication"] = {"dev": summ(f_dev), "holdout": summ(f_hold),
                          "holdout_matches_agent_ledger": sorted(f_hold["ticker"].to_list()) == sorted(led_h["ticker"].to_list())}
    print("replication", json.dumps(res["replication"], default=str), flush=True)

    # 1. quotes
    res["quote_timing"] = quote_timing(fx, dec.filter(pl.col("lead_h") == LEAD), lg)
    res["raw_candles"] = raw_candle_check(fx, mk)
    print("quotes", json.dumps(res["quote_timing"], default=str)[:600], json.dumps(res["raw_candles"], default=str)[:600], flush=True)

    # 2. settlement
    res["settlement"] = {s: settlement_check(fx.filter(pl.col("split") == s), lg) for s in ("dev", "holdout")}
    print("settlement", json.dumps(res["settlement"], default=str)[:900], flush=True)

    # 3. drift
    res["drift"] = drift_tables(fx, vals)
    # per-event cap ordering (the code admits legs in ticker order: A* tail, E* ascending, U* tail)
    cap = []
    for order in ("ticker", "reverse", "cheapest", "near_modal", "far_modal"):
        for split, part in (("dev", sel12.filter(pl.col("T") < ap2.DEV_END)), ("holdout", sel12.filter(pl.col("T") >= ap2.DEV_END))):
            cap.append({"order": order, "split": split, **{k: v for k, v in summ(fills_ordered(part, order)).items()
                                                           if k in ("fills", "events", "c_per_contract", "t_event", "yes_settled", "pnl_usd")}})
    for split, part in (("dev", sel12.filter(pl.col("T") < ap2.DEV_END)), ("holdout", sel12.filter(pl.col("T") >= ap2.DEV_END))):
        cap.append({"order": "no_cap_all_legs", "split": split, **{k: v for k, v in summ(fills_ordered(part, "ticker", None)).items()
                                                                     if k in ("fills", "events", "c_per_contract", "t_event", "yes_settled", "pnl_usd")}})
        rnd = [summ(fills_ordered(part, "random", seed=s)) for s in range(200)]
        cs = np.array([r["c_per_contract"] for r in rnd])
        ts = np.array([r["t_event"] or 0 for r in rnd])
        agent = summ(fills_ordered(part, "ticker"))
        cap.append({"order": "random_x200_mean", "split": split, "c_per_contract": round(float(cs.mean()), 2),
                    "c_median": round(float(np.median(cs)), 2), "c_min": float(cs.min()), "c_max": float(cs.max()),
                    "t_mean": round(float(ts.mean()), 2), "t_median": round(float(np.median(ts)), 2),
                    "share_draws_ge_agent_c": round(float((cs >= agent["c_per_contract"] - 1e-9).mean()), 3),
                    "share_draws_t_ge_2": round(float((ts >= 2).mean()), 3)})
    nq = sel12.group_by("event").agg(pl.len().alias("q"), pl.col("T").first())
    res["cap_order"] = {"variants": cap, "events_with_ge3_qualifying_legs": {
        "dev": int(nq.filter((pl.col("T") < ap2.DEV_END) & (pl.col("q") >= 3)).height),
        "holdout": int(nq.filter((pl.col("T") >= ap2.DEV_END) & (pl.col("q") >= 3)).height)}}
    # all qualifying legs (no cap) split by side x direction
    allq = attach(ap2.fills_from(sel12, "no", per_event=1e9), dec.filter(pl.col("lead_h") == LEAD), vals)
    res["drift_all_legs"] = drift_tables(allq, vals)
    print("drift", json.dumps(res["drift"]["by_split_side"], default=str), flush=True)

    # 4. other hours / weekly / executability
    res["lead_sweep"] = lead_sweep(dec)
    res["weekly"] = weekly_sweep()
    res["exec_candles"] = exec_candles(fx, cand)
    res["trade_prints"] = trade_prints(fx, fetch=False)
    if with_l2:
        days = [f"2026-11-{d:02d}" for d in range(14, 21)]
        bs, tr = l2_books(days)
        _save(bs, OUT / "l2_books.parquet")
        _save(tr, OUT / "l2_trades.parquet")
        res["l2"] = l2_check(bs, tr, dec)
        res["candle_semantics"] = candle_semantics(cand, bs.filter(pl.col("best_bid").is_not_null()))
    print("exec", json.dumps(res["exec_candles"], default=str), flush=True)

    # 5. statistics
    res["stats"] = stats_block(fx, pooled)
    dec_agent = dec.filter(pl.col("lead_h").is_in([float(x) for x in ap2.LEADS[SERIES]]))
    res["holdout_family"] = holdout_family(dec_agent)

    # Sharpe / vol of the replica and variants (vol_metrics through the agent's vol_block)
    marks = ap2.marks_from(cand)
    hold = fx.filter(pl.col("split") == "holdout")
    devx = fx.filter(pl.col("split") == "dev")
    bm = [book_metrics(f_hold, marks, "primary", "holdout"), book_metrics(f_dev, marks, "primary", "dev"),
          book_metrics(pooled, marks, "primary", "pooled"),
          book_metrics(hold.filter(pl.col("side_modal") == "below"), marks, "below_modal", "holdout"),
          book_metrics(hold.filter(pl.col("side_modal") == "above"), marks, "above_modal", "holdout"),
          book_metrics(devx.filter(pl.col("side_modal") == "below"), marks, "below_modal", "dev"),
          book_metrics(devx.filter(pl.col("side_modal") == "above"), marks, "above_modal", "dev"),
          book_metrics(fills_ordered(sel12, "ticker", None), marks, "no_cap", "pooled")]
    res["book_metrics"] = bm
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print("wrote", OUT / "results.json")
    return res


def fetch_trades() -> None:
    dec, _lg, vals, _cand = daily_frame([LEAD])
    sel12 = shoulder(dec, LEAD)
    f = pl.concat([ap2.fills_from(sel12.filter(pl.col("T") < ap2.DEV_END), "no"),
                   ap2.fills_from(sel12.filter(pl.col("T") >= ap2.DEV_END), "no")])
    fx = attach(f, dec, vals)
    print(json.dumps(trade_prints(fx, fetch=True), indent=1, default=str))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m albert.research.approval2_review")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--no-l2", action="store_true")
    sub.add_parser("trades")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(with_l2=not a.no_l2)
    elif a.cmd == "trades":
        fetch_trades()


if __name__ == "__main__":
    main()
