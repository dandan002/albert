"""Momentum outside sports, and resolution-spike strategies, on the volume-blind ``xcat`` candles.

Question (user, 2026-11-17): does momentum work in other markets (politics, mentions,
entertainment, science/tech, elections), and does buying, or resting limit bids, as a contract's
price spikes towards resolution pay?

Pre-specified primary hypotheses (everything else is exploratory and reported as such):

* **H1 momentum (hourly layer, whole life):** 6 h mid change >= 5c, buy the direction of the move
  at the ask (taker), hold to settlement; by category. Also 1/24/72 h and 3/10c, fades, and 1 h
  "news jumps" of 10/20c.
* **H2 spike taker (minute layer):** the leading side's mid reaches >= 90c after rising >= 10c
  within 60 minutes: buy the leader at its ask, hold to settlement. Variants: 80/90/95c,
  10/20c rises over 5/15/60 min, a first touch of 97c, a slow-approach control (crossing without a
  >= 3c rise in 60 min), the fade (buy the laggard), and a one-minute execution delay.
* **H3 spike limit (minute layer):** after the H2 trigger, rest a bid for the leader 3c below its
  mid for 60 minutes; it fills only on a print strictly through the level (``p_low < L`` for a YES
  bid, ``p_high > 1 - L`` for a NO bid) and is held to settlement. Variants: 1/3/5c below, 60 min
  or until the window ends, and laggard bids.

Execution: taker at the candle's closing quote (``bid_close``/``ask_close``), ``n = floor(stake /
price)`` contracts, Kalshi taker fee (0.07) rounded up per order; maker fee 0.0175 only on
``quadratic_with_maker_fees`` series, also rounded up per order. Stake $1 for momentum (as in
``ta_calendar``) and $5 for spike trades (a 97c leg at $1 would pay a 1c fee per contract).

No look-ahead: the minute layer only covers each market's first three days after open (a window
fixed by the open time, ``xcat``), so a spike is never chosen because the market then resolved.
Quotes at t come from the candle ending at or before t (as-of joins on the sparse candles, which
the API omits when nothing changed); candles at or after the actual close are dropped. Dev /
holdout split by decision time (before / from 2026-10-01). t-stats clustered by event and by ET
date.

    python -m albert.research.spikes momentum|spikes|screens|ledgers|all
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from albert.fees import KALSHI_MAKER
from albert.research import ta_calendar as tc
from albert.research import xcat

OUT = Path("data/research/spikes")
HOLDOUT_T = datetime(2026, 10, 1, tzinfo=UTC)
SPIKE_STAKE = 5.0
MOM_STAKE = 1.0
SPIKE_SPREAD = 0.10
EVENT_CAP = 10.0


# ---- market metadata -----------------------------------------------------------------------------
def markets_meta(tables: Path = tc.TABLES) -> pl.DataFrame:
    return (pl.scan_parquet(tables / "kalshi_markets.parquet")
            .filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary"))
            .select("ticker", "event_ticker", "series_ticker", "category", "result", "open_time", "close_time",
                    "settlement_ts", "expected_expiration_time", "fee_type", "fee_multiplier")
            .collect().with_columns(xcat.stratum(pl.col("category"), pl.col("series_ticker")).alias("stratum")))


def _common(p: pl.DataFrame) -> pl.DataFrame:
    et = pl.col("t").dt.convert_time_zone(tc.ET)
    return p.with_columns(pl.col("event_ticker").alias("cluster"), (pl.col("t") >= HOLDOUT_T).alias("holdout"),
                          (pl.col("result") == "yes").cast(pl.Float64).alias("y"),
                          et.dt.date().alias("et_date"), et.dt.hour().alias("et_hour"))


# ---- execution -----------------------------------------------------------------------------------
def taker(f: pl.DataFrame, side_yes: pl.Expr, yes_ask: pl.Expr, yes_bid: pl.Expr, stake: float) -> pl.DataFrame:
    """Buy YES at ``yes_ask`` (side_yes) or NO at ``1 - yes_bid``; ~$stake order held to settlement."""
    f = f.with_columns(side_yes.alias("side_yes"),
                       pl.when(side_yes).then(yes_ask).otherwise(1 - yes_bid).alias("price"))
    f = f.filter(pl.col("price").is_between(0.01, 0.99))
    f = f.with_columns(pl.max_horizontal(pl.lit(1.0), (stake / pl.col("price")).floor()).alias("n"),
                       pl.when(pl.col("side_yes")).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"),
                       pl.when(pl.col("side_yes")).then(pl.col("mid")).otherwise(1 - pl.col("mid")).alias("mid_side"),
                       pl.when(pl.col("side_yes")).then(1).otherwise(-1).alias("dir"))
    f = f.with_columns(tc.taker_fee_expr(pl.col("n"), pl.col("price")).alias("fee_order"))
    return _pnl(f)


def maker_fee_expr(n: pl.Expr, price: pl.Expr) -> pl.Expr:
    coef = pl.when(pl.col("fee_type").fill_null("").str.contains("maker_fee")).then(KALSHI_MAKER).otherwise(0.0)
    raw = coef * pl.col("fee_multiplier").fill_null(1.0) * n * price * (1 - price)
    return (raw * 100).round(9).ceil() / 100


def _pnl(f: pl.DataFrame) -> pl.DataFrame:
    return f.with_columns(
        (pl.col("fee_order") / pl.col("n")).alias("fee_c"),
        (pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n")).alias("pnl"),
        (pl.col("price") + pl.col("fee_order") / pl.col("n")).alias("risk"),
        (pl.col("payoff") - pl.col("mid_side")).alias("gross"))


def wilson_lo(p: float, n: int, z: float = 1.96) -> float | None:
    if n <= 0:
        return None
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / (1 + z * z / n)


def score(f: pl.DataFrame, **labels: Any) -> list[dict[str, Any]]:
    """``ta_calendar.score`` plus dollars, the breakeven win rate (cost per contract) and a Wilson
    95% lower bound on the win rate with the number of *events* as the sample size (legs of one
    event are not independent). For near-certain bets a t-stat is meaningless when no bet lost;
    ``win_lo95 > breakeven`` is the bar."""
    rows = tc.score(f, by_date=True, **labels)
    for r, (_, sub) in zip(rows, (("dev", f.filter(~pl.col("holdout"))), ("holdout", f.filter(pl.col("holdout"))))):
        r["usd"] = tc._r(float((sub["pnl"] * sub["n"]).sum()) if sub.height else 0.0)
        if sub.height:
            w = float((sub["payoff"] * sub["n"]).sum() / sub["n"].sum())
            r["breakeven"] = tc._r(float((sub["risk"] * sub["n"]).sum() / sub["n"].sum()), 4)
            r["win_rate"] = tc._r(w, 4)
            r["win_lo95"] = tc._r(wilson_lo(w, sub["cluster"].n_unique()), 4)
    return rows


# ---- H1: momentum on the hourly layer ------------------------------------------------------------
def hourly_panel(meta: pl.DataFrame | None = None, candles: pl.DataFrame | None = None) -> pl.DataFrame:
    meta = markets_meta() if meta is None else meta
    c = (xcat.load("hourly") if candles is None else candles).select("ticker", "end_ts", pl.col("bid_close").alias("yes_bid_close"),
                                   pl.col("ask_close").alias("yes_ask_close"), "volume", pl.col("oi").alias("open_interest"))
    p = tc.build_panel(c, meta, prefixes=None)
    p = _common(p.drop("cluster", "holdout", "y", "et_date", "et_hour"))
    lags = [(pl.col("x") - pl.col("x").shift(L).over("ticker")).alias(f"r{L}") for L in (1, 6, 24, 72)]
    return p.with_columns(lags).with_columns(
        ((pl.col("t") - pl.col("open_time")).dt.total_hours()).alias("age_h"))


def momentum_signals() -> dict[str, tuple[pl.Expr, pl.Expr]]:
    sig = {}
    for L in (1, 6, 24, 72):
        for thr in (0.03, 0.05, 0.10):
            r = pl.col(f"r{L}")
            sig[f"tsmom{L}h {int(thr * 100)}c"] = (r >= thr, r <= -thr)
    for thr in (0.10, 0.20):
        sig[f"jump1h {int(thr * 100)}c"] = (pl.col("r1") >= thr, pl.col("r1") <= -thr)
    return sig


def momentum_trades(p: pl.DataFrame) -> pl.DataFrame:
    q = tc.entry_ok(p)
    frames = []
    for name, (up, dn) in momentum_signals().items():
        d = pl.when(up).then(1).when(dn).then(-1).otherwise(0)
        s = tc.first_per_market(q.with_columns(d.alias("sig")).filter(pl.col("sig") != 0))
        for mode, sign in (("trend", 1), ("fade", -1)):
            t = taker(s, (pl.col("sig") * sign) == 1, pl.col("ask"), pl.col("bid"), MOM_STAKE)
            frames.append(t.with_columns(pl.lit(name).alias("signal"), pl.lit(mode).alias("mode")))
    keep = ["signal", "mode", "stratum", "ticker", "event_ticker", "cluster", "t", "et_date", "holdout", "dir", "mid",
            "spread", "price", "n", "fee_order", "fee_c", "payoff", "pnl", "risk", "gross", "settlement_ts", "age_h"]
    return pl.concat([f.select(keep) for f in frames])


def momentum_tables(tr: pl.DataFrame) -> dict[str, pl.DataFrame]:
    rows = []
    for (sig, mode), f in tr.group_by(["signal", "mode"], maintain_order=True):
        rows += score(f, signal=sig, mode=mode, stratum="all")
        for (st,), g in f.group_by(["stratum"], maintain_order=True):
            rows += score(g, signal=sig, mode=mode, stratum=st)
    return {"momentum": pl.DataFrame(rows, infer_schema_length=None)}


# ---- H2/H3: resolution spikes on the minute layer ------------------------------------------------
WINDOWS_MIN = (5, 15, 60)


def minute_panel(meta: pl.DataFrame | None = None, candles: pl.DataFrame | None = None) -> pl.DataFrame:
    """Sparse minute candles before the actual close, with the leader side and its past prices."""
    meta = markets_meta() if meta is None else meta
    c = (xcat.load("minute") if candles is None else candles).rename({"end_ts": "t"})
    p = c.join(meta, on="ticker").filter(pl.col("t") < pl.col("close_time")).sort("ticker", "t")
    p = p.with_columns(pl.col("bid_close").forward_fill().over("ticker").alias("bid"),
                       pl.col("ask_close").forward_fill().over("ticker").alias("ask"))
    two = (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    p = p.with_columns(pl.when(two).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                       (pl.col("ask") - pl.col("bid")).alias("spread"))
    p = p.with_columns(pl.when(pl.col("spread") <= SPIKE_SPREAD).then(pl.col("mid")).alias("smid"))
    p = _common(p).with_columns((pl.col("smid") >= 0.5).alias("lead_yes"))
    p = p.with_columns(pl.when(pl.col("lead_yes")).then(pl.col("smid")).otherwise(1 - pl.col("smid")).alias("q"))
    base = p.select("ticker", pl.col("t").alias("t_past"), pl.col("smid").alias("smid_past")).sort("ticker", "t_past")
    for W in WINDOWS_MIN:
        p = (p.with_columns((pl.col("t") - timedelta(minutes=W)).alias("t_q")).sort("t_q")
             .join_asof(base.sort("t_past"), left_on="t_q", right_on="t_past", by="ticker", strategy="backward")
             .with_columns(pl.when(pl.col("lead_yes")).then(pl.col("smid_past")).otherwise(1 - pl.col("smid_past"))
                           .alias(f"q{W}"))
             .drop("t_q", "t_past", "smid_past"))
        p = p.with_columns((pl.col("q") - pl.col(f"q{W}")).alias(f"rise{W}"))
    # one-minute execution delay: the quote standing one minute after the decision
    late = p.select("ticker", pl.col("t").alias("t_l"), pl.col("bid").alias("bid_1m"), pl.col("ask").alias("ask_1m"))
    p = (p.with_columns((pl.col("t") + timedelta(minutes=1)).alias("t_q")).sort("t_q")
         .join_asof(late.sort("t_l"), left_on="t_q", right_on="t_l", by="ticker", strategy="backward")
         .drop("t_q", "t_l"))
    win_end = pl.min_horizontal("close_time", pl.col("open_time") + timedelta(days=xcat.MINUTE_DAYS))
    return p.with_columns(win_end.alias("win_end")).sort("ticker", "t")


def spike_triggers() -> dict[str, pl.Expr]:
    """name -> trigger condition on the leader price q (first qualifying minute per market)."""
    trig = {}
    for Z in (0.70, 0.80, 0.90, 0.95):
        for R in (0.10, 0.20):
            if Z - R < 0.5:
                continue
            for W in WINDOWS_MIN:
                trig[f"q>={int(Z * 100)} rise>={int(R * 100)}c/{W}m"] = (pl.col("q") >= Z) & (pl.col(f"rise{W}") >= R)
        trig[f"q>={int(Z * 100)} slow (rise<3c/60m)"] = (pl.col("q") >= Z) & (pl.col("rise60") < 0.03)
    trig["q>=97 first touch"] = pl.col("q") >= 0.97
    return trig


PRIMARY = "q>=90 rise>=10c/60m"


def trigger_rows(p: pl.DataFrame) -> pl.DataFrame:
    ok = p.filter(pl.col("smid").is_not_null() & (pl.col("spread") <= SPIKE_SPREAD) & (pl.col("bid") > 0) & (pl.col("ask") < 1))
    frames = []
    for name, cond in spike_triggers().items():
        frames.append(tc.first_per_market(ok.filter(cond)).with_columns(pl.lit(name).alias("trigger")))
    return pl.concat(frames)


def spike_taker(trig: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for mode, side, ask, bid in (("follow", pl.col("lead_yes"), pl.col("ask"), pl.col("bid")),
                                 ("follow +1m", pl.col("lead_yes"), pl.col("ask_1m"), pl.col("bid_1m")),
                                 ("fade", ~pl.col("lead_yes"), pl.col("ask"), pl.col("bid"))):
        frames.append(taker(trig, side, ask, bid, SPIKE_STAKE).with_columns(pl.lit(mode).alias("mode")))
    return pl.concat(frames)


def minute_momentum(p: pl.DataFrame) -> pl.DataFrame:
    """Intraday momentum away from the extremes: the mid moved >= thr over W minutes and now sits
    in [20c, 80c]; trend or fade at the ask, $1, first signal per market, held to settlement."""
    ok = p.filter(pl.col("smid").is_between(0.20, 0.80) & (pl.col("spread") <= SPIKE_SPREAD)
                  & (pl.col("bid") > 0) & (pl.col("ask") < 1))
    frames = []
    for W in (15, 60):
        past = pl.when(pl.col("lead_yes")).then(pl.col(f"q{W}")).otherwise(1 - pl.col(f"q{W}"))   # YES mid W ago
        for thr in (0.05, 0.10):
            r = pl.col("smid") - past
            s = tc.first_per_market(ok.with_columns(pl.when(r >= thr).then(1).when(r <= -thr).then(-1).otherwise(0).alias("sig"))
                                    .filter(pl.col("sig") != 0))
            for mode, sign in (("trend", 1), ("fade", -1)):
                t = taker(s, (pl.col("sig") * sign) == 1, pl.col("ask"), pl.col("bid"), MOM_STAKE)
                frames.append(t.with_columns(pl.lit(f"tsmom{W}m {int(thr * 100)}c").alias("signal"), pl.lit(mode).alias("mode")))
    return pl.concat(frames)


MAKER_OFFSETS = (0.01, 0.03, 0.05)
MAKER_HORIZONS = (("60m", timedelta(minutes=60)), ("window", None))


def spike_maker(trig: pl.DataFrame, p: pl.DataFrame, triggers: tuple[str, ...]) -> pl.DataFrame:
    """Rest a bid for the leader (or the laggard) d below its mid after the trigger; first print
    strictly through the level fills it (within the horizon); held to settlement."""
    tape = p.select("ticker", pl.col("t").alias("tf"), "p_low", "p_high", pl.col("volume").alias("v_fill"))
    rows = []
    base = trig.filter(pl.col("trigger").is_in(triggers)).with_row_index("oid")
    for side in ("leader", "laggard"):
        for d in MAKER_OFFSETS:
            for hname, H in MAKER_HORIZONS:
                o = base.with_columns(
                    (pl.col("lead_yes") if side == "leader" else ~pl.col("lead_yes")).alias("side_yes"))
                own = pl.when(pl.col("side_yes")).then(pl.col("mid")).otherwise(1 - pl.col("mid"))
                o = o.with_columns(((own - d) * 100).round(6).floor().truediv(100).alias("L"))   # cent grid; 1 - .92 - .01 = .0699..
                o = o.filter(pl.col("L").is_between(0.01, 0.99))
                end = pl.col("win_end") if H is None else pl.min_horizontal(pl.col("t") + H, pl.col("win_end"))
                o = o.with_columns(end.alias("t_end"))
                hit = (o.select("oid", "ticker", "t", "t_end", "side_yes", "L").join(tape, on="ticker")
                       .filter((pl.col("tf") > pl.col("t")) & (pl.col("tf") <= pl.col("t_end"))
                               & pl.when(pl.col("side_yes")).then(pl.col("p_low") < pl.col("L") - 1e-9)
                               .otherwise(pl.col("p_high") > 1 - pl.col("L") + 1e-9))   # strictly through, on the cent grid
                       .group_by("oid").agg(pl.col("tf").min().alias("t_fill"),
                                            pl.col("v_fill").sort_by("tf").first()))
                o = o.join(hit, on="oid", how="left").with_columns(
                    pl.lit(side).alias("side"), pl.lit(f"-{int(d * 100)}c").alias("offset"), pl.lit(hname).alias("horizon"),
                    pl.col("t_fill").is_not_null().alias("filled"))
                rows.append(o)
    o = pl.concat(rows, how="diagonal_relaxed")
    o = o.with_columns(pl.max_horizontal(pl.lit(1.0), (SPIKE_STAKE / pl.col("L")).floor()).alias("n"),
                       pl.col("L").alias("price"),
                       pl.when(pl.col("side_yes")).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"),
                       pl.when(pl.col("side_yes")).then(pl.col("mid")).otherwise(1 - pl.col("mid")).alias("mid_side"),
                       pl.when(pl.col("side_yes")).then(1).otherwise(-1).alias("dir"))
    o = o.with_columns(maker_fee_expr(pl.col("n"), pl.col("price")).alias("fee_order"))
    return _pnl(o)


def spike_tables(tk: pl.DataFrame, mk: pl.DataFrame) -> dict[str, pl.DataFrame]:
    rows = []
    for (trig, mode), f in tk.group_by(["trigger", "mode"], maintain_order=True):
        rows += score(f, trigger=trig, mode=mode, stratum="all")
        if trig == PRIMARY or mode == "follow":
            for (st,), g in f.group_by(["stratum"], maintain_order=True):
                rows += score(g, trigger=trig, mode=mode, stratum=st)
            for (ly,), g in f.group_by(["lead_yes"], maintain_order=True):
                rows += score(g, trigger=trig, mode=mode, stratum=f"leader={'YES' if ly else 'NO'}")
    taker_t = pl.DataFrame(rows, infer_schema_length=None)
    rows = []
    for (trig, side, off, hz), f in mk.group_by(["trigger", "side", "offset", "horizon"], maintain_order=True):
        fl = f.filter(pl.col("filled"))
        for split, sub, subf in (("dev", f.filter(~pl.col("holdout")), fl.filter(~pl.col("holdout"))),
                                 ("holdout", f.filter(pl.col("holdout")), fl.filter(pl.col("holdout")))):
            sc = score(subf.with_columns(pl.lit(split == "holdout").alias("holdout")), trigger=trig)[1 if split == "holdout" else 0]
            rows.append({"trigger": trig, "side": side, "offset": off, "horizon": hz, "split": split,
                         "orders": sub.height, "fill_rate": tc._r(subf.height / sub.height, 3) if sub.height else None,
                         "win_if_filled": tc._r(subf["payoff"].mean(), 3) if subf.height else None,
                         "win_all_orders": tc._r(sub["payoff"].mean(), 3) if sub.height else None,
                         **{k: sc.get(k) for k in ("trades", "clusters", "avg_price", "net_c", "t_net", "net_per_$",
                                                   "t_$", "t_$_by_date", "usd", "breakeven", "win_rate", "win_lo95")}})
    maker_t = pl.DataFrame(rows, infer_schema_length=None)
    return {"spike_taker": taker_t, "spike_maker": maker_t}


# ---- exploratory screens: new listings and hour of day -------------------------------------------
def listing_rows(p: pl.DataFrame, layer: str) -> pl.DataFrame:
    """The first takeable quote in a market's first 6 h after open, as taker trades on both sides."""
    age = (pl.col("t") - pl.col("open_time")).dt.total_minutes() / 60
    q = p.filter(pl.col("smid").is_not_null() & (pl.col("spread") <= 0.10) & pl.col("mid").is_between(0.03, 0.97)
                 & (age <= 6)).with_columns(tc.price_bucket(pl.col("mid")).alias("pb"))
    q = tc.first_per_market(q.filter(pl.col("pb").is_not_null()))
    frames = [taker(q, sy, pl.col("ask"), pl.col("bid"), MOM_STAKE).with_columns(pl.lit(side).alias("side"))
              for side, sy in (("YES", pl.lit(True)), ("NO", pl.lit(False)))]
    keep = ["side", "stratum", "pb", "ticker", "cluster", "et_date", "holdout", "t", "mid", "price", "n", "fee_order",
            "payoff", "pnl", "risk", "gross"]
    return pl.concat(frames).select(keep).with_columns(pl.lit(layer).alias("layer"))


def hod_rows(hp: pl.DataFrame) -> pl.DataFrame:
    """y - mid at the first takeable quote of each (market, ET hour) on the hourly layer."""
    q = hp.filter(pl.col("smid").is_not_null() & (pl.col("spread") <= 0.10) & pl.col("mid").is_between(0.03, 0.97))
    q = q.sort("t").unique(["ticker", "et_hour"], keep="first", maintain_order=True)
    return q.select("ticker", "cluster", "et_date", "et_hour", "holdout", "stratum", (pl.col("y") - pl.col("mid")).alias("resid"))


def screen_tables(lst: pl.DataFrame, hod: pl.DataFrame) -> dict[str, pl.DataFrame]:
    rows = []
    for (layer, side), t in lst.group_by(["layer", "side"], maintain_order=True):
        for (st, pb), f in t.group_by(["stratum", "pb"], maintain_order=True):
            rows += score(f, layer=layer, side=side, stratum=st, bucket=pb)
        rows += score(t, layer=layer, side=side, stratum="all", bucket="all")
    out = {"listing": pl.DataFrame(rows, infer_schema_length=None)}
    rows = []
    for split, sub in (("dev", hod.filter(~pl.col("holdout"))), ("holdout", hod.filter(pl.col("holdout")))):
        for h in range(24):
            f = sub.filter(pl.col("et_hour") == h)
            m, td, kd = tc.cl_mean(f, "resid", "et_date")
            _, tm, _ = tc.cl_mean(f, "resid", "cluster")
            rows.append({"split": split, "et_hour": h, "n": f.height, "dates": kd,
                         "resid_c": tc._r(None if m is None else 100 * m), "t_by_date": tc._r(td), "t_by_event": tc._r(tm),
                         "abs_resid_c": tc._r(100 * f["resid"].abs().mean()) if f.height else None})
    out["hour_of_day"] = pl.DataFrame(rows, infer_schema_length=None)
    return out


# ---- ledgers -------------------------------------------------------------------------------------
def ledger_metrics(specs: list[tuple[str, str, pl.DataFrame]]) -> pl.DataFrame:
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (strategy, variant, f) in enumerate(specs):
        f = f.with_columns(pl.col("t_fill").alias("t")) if "t_fill" in f.columns else f
        for split, sub in (("all", f), ("holdout", f.filter(pl.col("holdout")))):
            led = tc.capped_ledger(sub, f"spikes/{strategy}", variant, split, event_cap=EVENT_CAP)
            if split == "all" and led.height:
                led.write_parquet(OUT / "ledgers" / f"ledger_{i:02d}.parquet")
            rows.append({"strategy": strategy, "variant": variant, "split": split, "signals": sub.height,
                         **tc.ledger_row(led)})
            print(strategy, variant, split, rows[-1].get("sharpe"), rows[-1].get("usd_per_week"), flush=True)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("sharpe_ci90").map_elements(lambda v: f"[{v[0]}, {v[1]}]" if v is not None else "", return_dtype=pl.String))


def best_on_dev(table: pl.DataFrame, keys: list[str], min_trades: int = 100) -> dict[str, Any] | None:
    """The variant with the highest dev net per $ among those with >= ``min_trades`` dev trades."""
    d = table.filter((pl.col("split") == "dev") & (pl.col("trades") >= min_trades) & pl.col("net_per_$").is_not_null())
    if "stratum" in d.columns:
        d = d.filter(pl.col("stratum") == "all")
    return d.sort("net_per_$", descending=True).select(keys).row(0, named=True) if d.height else None


def run_ledgers() -> pl.DataFrame:
    mom = pl.read_parquet(OUT / "momentum_trades.parquet")
    tk = pl.read_parquet(OUT / "spike_taker.parquet")
    mk = pl.read_parquet(OUT / "spike_maker.parquet").filter(pl.col("filled"))
    specs: list[tuple[str, str, pl.DataFrame]] = [
        ("H1 momentum", "tsmom6h 5c trend (all non-sports)",
         mom.filter((pl.col("signal") == "tsmom6h 5c") & (pl.col("mode") == "trend"))),
        ("H2 spike taker", f"{PRIMARY} follow", tk.filter((pl.col("trigger") == PRIMARY) & (pl.col("mode") == "follow"))),
        ("H2 spike taker", f"{PRIMARY} follow +1m", tk.filter((pl.col("trigger") == PRIMARY) & (pl.col("mode") == "follow +1m"))),
        ("H3 spike limit", f"{PRIMARY} leader -3c 60m",
         mk.filter((pl.col("trigger") == PRIMARY) & (pl.col("side") == "leader") & (pl.col("offset") == "-3c")
                   & (pl.col("horizon") == "60m"))),
    ]
    for st in sorted(mom["stratum"].unique().to_list()):
        specs.append(("H1 momentum", f"tsmom6h 5c trend ({st})",
                      mom.filter((pl.col("signal") == "tsmom6h 5c") & (pl.col("mode") == "trend") & (pl.col("stratum") == st))))
    for st in sorted(tk["stratum"].unique().to_list()):
        specs.append(("H2 spike taker", f"{PRIMARY} follow ({st})",
                      tk.filter((pl.col("trigger") == PRIMARY) & (pl.col("mode") == "follow") & (pl.col("stratum") == st))))
    b = best_on_dev(pl.read_csv(OUT / "momentum.csv"), ["signal", "mode"])
    if b:
        specs.append(("best-on-dev momentum", f"{b['signal']} {b['mode']}",
                      mom.filter((pl.col("signal") == b["signal"]) & (pl.col("mode") == b["mode"]))))
    b = best_on_dev(pl.read_csv(OUT / "spike_taker.csv"), ["trigger", "mode"])
    if b:
        specs.append(("best-on-dev spike taker", f"{b['trigger']} {b['mode']}",
                      tk.filter((pl.col("trigger") == b["trigger"]) & (pl.col("mode") == b["mode"]))))
    b = best_on_dev(pl.read_csv(OUT / "spike_maker.csv"), ["trigger", "side", "offset", "horizon"])
    if b:
        specs.append(("best-on-dev spike limit", f"{b['trigger']} {b['side']} {b['offset']} {b['horizon']}",
                      mk.filter((pl.col("trigger") == b["trigger"]) & (pl.col("side") == b["side"])
                                & (pl.col("offset") == b["offset"]) & (pl.col("horizon") == b["horizon"]))))
    df = ledger_metrics(specs)
    _save("ledger_metrics", df)
    return df


# ---- driver --------------------------------------------------------------------------------------
def _save(name: str, df: pl.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / f"{name}.csv")


N_CHUNKS = 8   # every computation is per market; chunks keep the 8.5M-row hourly grid within memory


def _chunks(layer: str) -> list[list[str]]:
    tk = sorted(pl.read_parquet(xcat.OUT / layer / "done.parquet")["ticker"].to_list())
    return [tk[i::N_CHUNKS] for i in range(N_CHUNKS)]


def _load(layer: str, tickers: list[str]) -> pl.DataFrame:
    parts = sorted((xcat.OUT / layer).glob("part-*.parquet"))
    return (pl.scan_parquet(parts).filter(pl.col("ticker").is_in(tickers)).collect()
            .unique(["ticker", "end_ts"]).sort("ticker", "end_ts"))


def run_hourly(meta: pl.DataFrame) -> None:
    trades, lst, hod, stats = [], [], [], []
    for ch in _chunks("hourly"):
        hp = hourly_panel(meta, _load("hourly", ch))
        trades.append(momentum_trades(hp))
        lst.append(listing_rows(hp, "hourly"))
        hod.append(hod_rows(hp))
        stats.append(hp.group_by("ticker").agg(pl.col("stratum").first(), pl.col("event_ticker").first(), pl.len().alias("rows")))
        del hp
    tr = pl.concat(trades)
    if tr.is_empty():
        raise RuntimeError("zero momentum trades")
    tr.write_parquet(OUT / "momentum_trades.parquet")
    for k, v in momentum_tables(tr).items():
        _save(k, v)
    pl.concat(lst).write_parquet(OUT / "listing_hourly.parquet")
    pl.concat(hod).write_parquet(OUT / "hod_rows.parquet")
    st = pl.concat(stats)
    (OUT / "sample_hourly.json").write_text(json.dumps({
        "markets": st.height, "events": st["event_ticker"].n_unique(), "grid_rows": int(st["rows"].sum()),
        "by_stratum": dict(st.group_by("stratum").len().iter_rows())}, indent=1, default=str))


def run_minute(meta: pl.DataFrame) -> None:
    tks, mks, mms, lst, stats = [], [], [], [], []
    for ch in _chunks("minute"):
        mp = minute_panel(meta, _load("minute", ch))
        trig = trigger_rows(mp)
        tk = spike_taker(trig)
        mk = spike_maker(trig, mp, (PRIMARY, "q>=95 rise>=10c/60m", "q>=97 first touch"))
        drop = lambda f: f.drop([c for c in f.columns if c.startswith(("q", "rise", "bid_", "ask_", "p_"))
                                 and c not in ("bid_1m", "ask_1m")])
        tks.append(drop(tk))
        mks.append(drop(mk))
        mms.append(minute_momentum(mp).select("signal", "mode", "stratum", "ticker", "event_ticker", "cluster", "t",
                                              "et_date", "holdout", "dir", "mid", "price", "n", "fee_order", "payoff",
                                              "pnl", "risk", "gross", "settlement_ts"))
        lst.append(listing_rows(mp, "minute"))
        stats.append(mp.group_by("ticker").agg(pl.col("stratum").first(), pl.col("event_ticker").first(), pl.len().alias("rows"),
                                               (pl.col("close_time").first() <= pl.col("win_end").first()).alias("closes_in")))
        del mp
    tk, mk = pl.concat(tks, how="diagonal_relaxed"), pl.concat(mks, how="diagonal_relaxed")
    if tk.is_empty():
        raise RuntimeError("zero spike triggers")
    tk.write_parquet(OUT / "spike_taker.parquet")
    mk.write_parquet(OUT / "spike_maker.parquet")
    for k, v in spike_tables(tk, mk).items():
        _save(k, v)
    mm = pl.concat(mms)
    mm.write_parquet(OUT / "minute_momentum_trades.parquet")
    rows = []
    for (sig, mode), f in mm.group_by(["signal", "mode"], maintain_order=True):
        rows += score(f, signal=sig, mode=mode, stratum="all")
        for (st_,), g in f.group_by(["stratum"], maintain_order=True):
            rows += score(g, signal=sig, mode=mode, stratum=st_)
    _save("minute_momentum", pl.DataFrame(rows, infer_schema_length=None))
    pl.concat(lst).write_parquet(OUT / "listing_minute.parquet")
    st = pl.concat(stats)
    (OUT / "sample_minute.json").write_text(json.dumps({
        "markets": st.height, "events": st["event_ticker"].n_unique(), "candles": int(st["rows"].sum()),
        "by_stratum": dict(st.group_by("stratum").len().iter_rows()),
        "share_closing_inside_window": dict(st.group_by("stratum").agg(pl.col("closes_in").mean()).iter_rows()),
    }, indent=1, default=str))


def run_screens() -> None:
    lst = pl.concat([pl.read_parquet(OUT / "listing_hourly.parquet"), pl.read_parquet(OUT / "listing_minute.parquet")])
    for k, v in screen_tables(lst, pl.read_parquet(OUT / "hod_rows.parquet")).items():
        _save(k, v)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.spikes")
    ap.add_argument("step", choices=["momentum", "spikes", "screens", "ledgers", "all"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = markets_meta()
    if a.step in ("momentum", "all"):
        run_hourly(meta)
    if a.step in ("spikes", "all"):
        run_minute(meta)
    if a.step in ("screens", "all"):
        run_screens()
    if a.step in ("ledgers", "all"):
        run_ledgers()


if __name__ == "__main__":
    main()
