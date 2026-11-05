"""Longer-horizon momentum, reversal and drift on settled Kalshi markets (hourly candles).

Every model is scored against **settlement** with taker execution at the hourly candle's closing
quote: a long-YES signal buys YES at ``yes_ask_close``, a long-NO signal buys NO at
``1 - yes_bid_close``. Fees are Kalshi's taker fee per 1-contract order, rounded up to the cent
(conservative; ``albert.fees.kalshi_fee``). PnL per contract = payoff - price - fee, and per
dollar risked = PnL / (price + fee). Standard errors are clustered by *match* (the event ticker
without its series prefix and map suffix, so a game and its maps share a cluster).

Look-ahead rules: decisions at hour ``t`` use only candles ending at or before ``t``, and only
candles that end while the market is still trading (a candle ending after the actual close holds a
stale, untradeable quote: including those faked a +25c "PM vs Kalshi gap" edge). Time to the
event is measured to ``expected_expiration_time`` (scheduled), never to ``close_time`` (the actual
close of a settled market). The universe is the volume-blind samples: the series ingested with
``ingest kalshi-detail --series`` (every event or a seeded random set of events, no volume
filter) and the seeded Elo-series candles. Markets drawn by final volume (top-N, volume band) are
excluded. Dev / holdout split by scheduled expiration date.

Models:

1. ``tsmom``: the mid change over the last L hours (L = 1, 6, 24, 72) in the direction of the
   move, when |change| >= threshold.
2. ``xsmom``: within mutually-exclusive bracket events (weather highs, all legs), buy the leg that
   rose most over L hours, or the one that fell most (both as long-YES), vs the event average.
3. ``reversal``: after an hourly move >= threshold on below-median (or zero) volume, bet against it.
4. ``drift``: in the last hours before the scheduled expiration, buy the favourite (or longshot)
   by price bucket, split by the sign of the 6 h move.
5. ``pm``: the Polymarket intl (data only) price change over L hours on the matched outcome
   predicts Kalshi: trade Kalshi in the direction of the Polymarket move.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from albert.fees import kalshi_fee

VOLUME_BLIND_SERIES_PREFIXES = (
    # ingest kalshi-detail --series runs (every event, or a seeded random 500 per series)
    "KXNFLGAME", "KXNBAGAME", "KXNCAAFGAME", "KXMLBGAME", "KXNHLGAME", "KXWNBAGAME",
    "KXHIGH", "KXESOCCERGAME", "KXTTELITEMATCH", "KXCS2MAP", "KXCS2GAME", "KXLIGAPROMATCH",
    "KXEBASKETBALLGAME", "KXLOLMAP", "KXTTSTARMATCH", "KXDOTA2MAP", "KXNCAAWVMATCH",
    "KXDARTSMATCH", "KXT20MATCH", "KXLOLGAME", "KXEFOOTBALLGAME", "KXVALORANTMAP",
    "KXDOTA2GAME", "KXR6GAME", "KXVALORANTGAME", "KXNPBGAME", "KXKHLGAME", "KXKBOGAME",
    "KXITFMATCH", "KXITFWMATCH", "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH", "KXATPMATCH",
    "KXWTAMATCH",
)
HOLDOUT_FROM = datetime.fromisoformat("2026-10-01T00:00:00+00:00")
LAGS_H = (1, 6, 24, 72)
MAX_TAKE_SPREAD = 0.10      # only take when the quote we cross is this tight
MAX_SIGNAL_SPREAD = 0.20    # a mid used as a signal must come from a quote this tight


def match_key(event_ticker: str) -> str:
    """``KXCS2MAP-26SEP031500PAINAYAW-2`` and ``KXCS2GAME-26SEP031500PAINAYAW`` -> same key."""
    parts = event_ticker.split("-")
    if parts[0].startswith("KXHIGH"):
        return event_ticker          # one city-day; the date alone would merge every city
    if len(parts) >= 3 and parts[-1].isdigit():
        parts = parts[:-1]
    return "-".join(parts[1:]) or event_ticker


def fee_per_contract(price: float, fee_type: str | None, mult: float | None) -> float:
    return kalshi_fee(1.0, price, maker=False, fee_type=fee_type or "quadratic",
                      multiplier=mult if mult is not None else 1.0, conservative=True)


# ---- panel ----------------------------------------------------------------------------------------
def build_panel(candles: pl.DataFrame, markets: pl.DataFrame,
                prefixes: tuple[str, ...] | None = VOLUME_BLIND_SERIES_PREFIXES) -> pl.DataFrame:
    """One row per (market, hour): quote, mid, volume, scheduled time to expiration, outcome."""
    m = markets.filter(pl.col("result").is_in(["yes", "no"]))
    if prefixes is not None:
        m = m.filter(pl.col("series_ticker").str.starts_with(prefixes[0]) if len(prefixes) == 1 else
                     pl.any_horizontal([pl.col("series_ticker").str.starts_with(p) for p in prefixes]))
    m = m.select("ticker", "event_ticker", "series_ticker", "category", "result", "close_time",
                 "expected_expiration_time", "fee_type", "fee_multiplier")
    c = candles.select("ticker", pl.col("end_ts").alias("t"), pl.col("yes_bid_close").alias("bid"),
                       pl.col("yes_ask_close").alias("ask"), "volume")
    p = c.join(m, on="ticker", how="inner")
    p = p.with_columns(
        pl.when((pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid")))
        .then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        (pl.col("result") == "yes").cast(pl.Float64).alias("y"),
        ((pl.col("expected_expiration_time") - pl.col("t")).dt.total_minutes() / 60).alias("h_to_exp"),
        pl.col("event_ticker").map_elements(match_key, return_dtype=pl.String).alias("cluster"),
        (pl.col("expected_expiration_time") >= HOLDOUT_FROM).alias("holdout"),
    ).with_columns(
        pl.when(pl.col("spread") <= MAX_SIGNAL_SPREAD).then(pl.col("mid")).alias("smid"),
    )
    # A candle ending at or after the market's actual close carries a stale quote that could not
    # be traded at its end time (trading had stopped). Whether a market is still open at t is
    # observable at t, so this censoring uses no future information. close_time is never used
    # to *choose* decision times.
    return p.filter(pl.col("expected_expiration_time").is_not_null() & (pl.col("h_to_exp") > 0)
                    & (pl.col("t") < pl.col("close_time"))).drop("close_time").sort("ticker", "t")


def add_lags(p: pl.DataFrame, lags: tuple[int, ...] = LAGS_H) -> pl.DataFrame:
    """``ret_L``: signal mid now minus signal mid exactly L hours earlier (same market)."""
    out = p
    for lag in lags:
        past = p.select("ticker", (pl.col("t") + pl.duration(hours=lag)).alias("t"),
                        pl.col("smid").alias(f"smid_{lag}"))
        out = out.join(past, on=["ticker", "t"], how="left").with_columns(
            (pl.col("smid") - pl.col(f"smid_{lag}")).alias(f"ret_{lag}"))
    return out


def take(frame: pl.DataFrame, direction: pl.Expr) -> pl.DataFrame:
    """Rows with ``direction`` +1 (buy YES at the ask) or -1 (buy NO at 1 - bid); adds price, fee
    and PnL. Untakeable rows (no quote, spread wider than MAX_TAKE_SPREAD) are dropped."""
    f = frame.with_columns(direction.alias("dir")).filter(
        pl.col("dir").is_in([1, -1]) & (pl.col("bid") > 0) & (pl.col("ask") < 1)
        & (pl.col("spread") <= MAX_TAKE_SPREAD) & (pl.col("spread") > 0))
    f = f.with_columns(
        pl.when(pl.col("dir") == 1).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("price"),
        pl.when(pl.col("dir") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"),
    )
    fees = [fee_per_contract(pr, ft, fm) for pr, ft, fm in
            zip(f["price"].to_list(), f["fee_type"].to_list(), f["fee_multiplier"].to_list())]
    f = f.with_columns(pl.Series("fee", fees, dtype=pl.Float64))
    return f.with_columns(
        (pl.col("payoff") - pl.col("price") - pl.col("fee")).alias("pnl"),
        (pl.col("price") + pl.col("fee")).alias("risk"),
        # Signal before costs: settlement vs the mid on our side (no spread, no fee).
        (pl.col("payoff") - pl.when(pl.col("dir") == 1).then(pl.col("mid"))
         .otherwise(1 - pl.col("mid"))).alias("gross"),
    )


def one_per_market(f: pl.DataFrame) -> pl.DataFrame:
    """The first qualifying decision per market (no stacking of overlapping hourly signals)."""
    return f.sort("t").unique("ticker", keep="first", maintain_order=True)


def clustered(f: pl.DataFrame, col: str = "pnl", weight: str | None = None) -> dict[str, Any]:
    """Mean of ``col`` (per contract, or per $ when ``weight='risk'`` gives sum(pnl)/sum(risk))
    with a t-stat clustered by ``cluster``."""
    if f.is_empty():
        return {"n": 0, "clusters": 0, "mean": None, "t": None}
    w = pl.col(weight).sum() if weight else pl.len().cast(pl.Float64)
    g = f.group_by("cluster").agg(pl.col(col).sum().alias("s"), w.alias("w"))
    n_w, k = g["w"].sum(), g.height
    mean = g["s"].sum() / n_w
    t = None
    if k > 1:
        se = math.sqrt(k / (k - 1) * ((g["s"] - mean * g["w"]) ** 2).sum()) / n_w
        t = mean / se if se > 0 else None
    return {"n": f.height, "clusters": k, "mean": mean, "t": t}


def score(f: pl.DataFrame, **labels: Any) -> list[dict[str, Any]]:
    rows = []
    for split, sub in (("dev", f.filter(~pl.col("holdout"))), ("holdout", f.filter(pl.col("holdout")))):
        pc = clustered(sub)
        pd = clustered(sub, weight="risk")
        pg = clustered(sub, "gross")
        rows.append({**labels, "split": split, "trades": pc["n"], "clusters": pc["clusters"],
                     "avg_price": sub["price"].mean() if sub.height else None,
                     "gross_c": None if pg["mean"] is None else 100 * pg["mean"], "t_gross": pg["t"],
                     "pnl_c": None if pc["mean"] is None else 100 * pc["mean"], "t": pc["t"],
                     "pnl_per_$": pd["mean"], "t_$": pd["t"],
                     "hit": sub["payoff"].mean() if sub.height else None})
    return rows


def h_bucket(h: pl.Expr) -> pl.Expr:
    return (pl.when(h <= 3).then(pl.lit("0-3h")).when(h <= 12).then(pl.lit("3-12h"))
            .when(h <= 48).then(pl.lit("12-48h")).otherwise(pl.lit(">48h")))


def kind_of(series: pl.Expr) -> pl.Expr:
    return (pl.when(series.str.starts_with("KXHIGH")).then(pl.lit("weather"))
            .when(series.str.contains("MAP|GAME$") & series.str.contains(
                "CS2|LOL|DOTA2|VALORANT|R6|ESOCCER|EBASKET|EFOOTBALL")).then(pl.lit("esports"))
            .when(series.str.contains("ATP|WTA|ITF|TT|DARTS")).then(pl.lit("tennis/tt/darts"))
            .otherwise(pl.lit("team sports")))


# ---- models -----------------------------------------------------------------------------------------
def model_tsmom(p: pl.DataFrame, thresholds=(0.03, 0.05, 0.10)) -> list[dict[str, Any]]:
    rows = []
    for lag in LAGS_H:
        for thr in thresholds:
            for mode, sign in (("momentum", 1), ("reversal", -1)):
                d = (pl.when(pl.col(f"ret_{lag}") >= thr).then(sign)
                     .when(pl.col(f"ret_{lag}") <= -thr).then(-sign).otherwise(0))
                f = one_per_market(take(p, d))
                rows += score(f, model="tsmom", mode=mode, lag_h=lag, thr=thr, bucket="all", kind="all")
                if mode == "momentum" and thr == 0.05:
                    for (b, kd), sub in f.group_by(["hb", "kind"]):
                        rows += score(sub, model="tsmom", mode=mode, lag_h=lag, thr=thr, bucket=b, kind=kd)
    return rows


def model_xsmom(p: pl.DataFrame, lags=(6, 24)) -> list[dict[str, Any]]:
    """Weather bracket events: at each hour with every leg quoted, the leg whose mid rose most
    over L hours vs the leg that fell most. Both legs are bought as YES."""
    w = p.filter(pl.col("kind") == "weather")
    rows = []
    for lag in lags:
        col = f"ret_{lag}"
        q = w.filter(pl.col(col).is_not_null() & pl.col("smid").is_between(0.03, 0.97))
        q = q.with_columns(pl.col(col).rank("ordinal", descending=True).over(["event_ticker", "t"]).alias("rk_up"),
                           pl.col(col).rank("ordinal").over(["event_ticker", "t"]).alias("rk_dn"),
                           pl.col(col).max().over(["event_ticker", "t"]).alias("mx"),
                           pl.col(col).min().over(["event_ticker", "t"]).alias("mn"),
                           pl.len().over(["event_ticker", "t"]).alias("legs"))
        q = q.filter((pl.col("legs") >= 3) & (pl.col("mx") - pl.col("mn") >= 0.05))
        for name, cond in (("top riser", pl.col("rk_up") == 1), ("top faller", pl.col("rk_dn") == 1)):
            sel = q.filter(cond)
            f = take(sel, pl.lit(1)).sort("t").unique("event_ticker", keep="first", maintain_order=True)
            rows += score(f, model="xsmom", mode=f"{name} YES", lag_h=lag, thr=0.05, bucket="all",
                          kind="weather")
    return rows


def model_reversal(p: pl.DataFrame, thresholds=(0.05, 0.10)) -> list[dict[str, Any]]:
    rows = []
    med = p.group_by("ticker").agg(pl.col("volume").median().alias("vmed"))
    q = p.join(med, on="ticker")
    for thr in thresholds:
        for vol_name, vcond in (("zero volume", pl.col("volume") <= 0),
                                ("below-median volume", (pl.col("volume") > 0) & (pl.col("volume") < pl.col("vmed"))),
                                ("above-median volume", pl.col("volume") >= pl.col("vmed"))):
            sub = q.filter(vcond)
            d = (pl.when(pl.col("ret_1") >= thr).then(-1).when(pl.col("ret_1") <= -thr).then(1).otherwise(0))
            f = one_per_market(take(sub, d))
            rows += score(f, model="reversal", mode=vol_name, lag_h=1, thr=thr, bucket="all", kind="all")
    return rows


def model_drift(p: pl.DataFrame) -> list[dict[str, Any]]:
    """Last 0-6 h before the scheduled expiration: buy YES on favourites (or longshots) by price
    bucket, split by the sign of the 6 h move; one decision per market (first hour in window)."""
    rows = []
    q = p.filter(pl.col("h_to_exp").is_between(0, 6, closed="right") & pl.col("smid").is_not_null())
    q = q.with_columns(pl.when(pl.col("ret_6") > 0.02).then(pl.lit("rising"))
                       .when(pl.col("ret_6") < -0.02).then(pl.lit("falling")).otherwise(pl.lit("flat")).alias("mom"))
    for lo, hi in ((0.05, 0.20), (0.20, 0.50), (0.50, 0.80), (0.80, 0.95)):
        sub = q.filter(pl.col("smid").is_between(lo, hi, closed="left"))
        for mom in ("rising", "flat", "falling", None):
            s2 = sub if mom is None else sub.filter(pl.col("mom") == mom)
            f = one_per_market(take(s2, pl.lit(1)))
            for kd, s3 in [("all", f)] + list(f.group_by("kind")) if mom is None else [("all", f)]:
                kd = kd[0] if isinstance(kd, tuple) else kd
                rows += score(s3, model="drift", mode=f"buy YES {lo:.2f}-{hi:.2f} {mom or 'any'}",
                              lag_h=6, thr=None, bucket="0-6h", kind=kd)
    return rows


def model_pm(p: pl.DataFrame, pm_hist: pl.DataFrame, matches: pl.DataFrame, pm_markets: pl.DataFrame,
             lags=(1, 6, 24), thresholds=(0.03, 0.05)) -> list[dict[str, Any]]:
    """Polymarket intl price change (matched outcome, hourly) -> trade Kalshi at the next hour."""
    tok = _pm_tokens(matches, pm_markets)
    if tok.is_empty():
        return []
    # A PM print at ts belongs to the first hour boundary at or after ts, so the bar labelled t
    # (aligned with the Kalshi candle ending at t) holds only prices with ts <= t.
    h = (pm_hist.join(tok, on="token_id")
         .with_columns((pl.col("ts") + pl.duration(minutes=59, seconds=59)).dt.truncate("1h").alias("t"))
         .sort("ts")
         .group_by("kalshi_ticker", "t").agg(pl.col("price").last().alias("pm"))
         .sort("kalshi_ticker", "t"))
    # Trade only on a live PM price: not resolved (0/1 after the game, while the paused Kalshi book
    # still shows its last quote) and not a never-traded placeholder (0.495/0.505 for days), i.e.
    # inside [0.03, 0.97] and changed at least once in the last 6 hourly bars.
    h = h.with_columns((pl.col("pm") != pl.col("pm").shift(1)).over("kalshi_ticker")
                       .fill_null(False).cast(pl.Int32).alias("chg"))
    h = h.with_columns(pl.col("chg").rolling_sum(6, min_samples=1).over("kalshi_ticker").alias("chg6"),
                       pl.col("pm").is_between(0.03, 0.97).alias("live"))
    k = p.join(h.filter(pl.col("live") & (pl.col("chg6") >= 1)).rename({"kalshi_ticker": "ticker"}),
               on=["ticker", "t"], how="inner")
    rows = []
    for lag in lags:
        past = h.filter(pl.col("live") & (pl.col("chg6") >= 1)).select(
            pl.col("kalshi_ticker").alias("ticker"), (pl.col("t") + pl.duration(hours=lag)).alias("t"),
            pl.col("pm").alias("pm_l"))
        kk = k.join(past, on=["ticker", "t"], how="left").with_columns((pl.col("pm") - pl.col("pm_l")).alias("pmret"))
        for thr in thresholds:
            d = pl.when(pl.col("pmret") >= thr).then(1).when(pl.col("pmret") <= -thr).then(-1).otherwise(0)
            f = one_per_market(take(kk, d))
            rows += score(f, model="pm", mode="follow PM", lag_h=lag, thr=thr, bucket="all", kind="all")
            # PM moved but Kalshi did not (yet): the gap version
            gap = kk.with_columns((pl.col("pm") - pl.col("smid")).alias("gap"))
            d2 = pl.when(pl.col("gap") >= thr).then(1).when(pl.col("gap") <= -thr).then(-1).otherwise(0)
            f2 = one_per_market(take(gap, d2))
            if lag == lags[0]:
                rows += score(f2, model="pm", mode="PM above/below Kalshi mid", lag_h=0, thr=thr,
                              bucket="all", kind="all")
    return rows


def _pm_tokens(matches: pl.DataFrame, pm_markets: pl.DataFrame) -> pl.DataFrame:
    """kalshi_ticker -> PM token id of the matched outcome."""
    cols = pm_markets.columns
    tok_col = next((c for c in ("clob_token_ids", "token_ids", "clobTokenIds") if c in cols), None)
    if tok_col is None or "condition_id" not in cols:
        return pl.DataFrame(schema={"kalshi_ticker": pl.String, "token_id": pl.String})
    pm = pm_markets.select("condition_id", tok_col).unique("condition_id")
    j = matches.join(pm, left_on="pm_condition_id", right_on="condition_id")
    out = []
    for r in j.iter_rows(named=True):
        toks = r[tok_col]
        if isinstance(toks, str):
            import json
            try:
                toks = json.loads(toks)
            except ValueError:
                toks = None
        if toks and 0 <= r["pm_outcome_index"] < len(toks):
            out.append({"kalshi_ticker": r["kalshi_ticker"], "token_id": str(toks[r["pm_outcome_index"]])})
    return pl.DataFrame(out, schema={"kalshi_ticker": pl.String, "token_id": pl.String})


# ---- driver -----------------------------------------------------------------------------------------
def prepare(tables_dir: Path, prefixes=VOLUME_BLIND_SERIES_PREFIXES) -> pl.DataFrame:
    markets = pl.read_parquet(tables_dir / "kalshi_markets.parquet")
    candles = pl.read_parquet(tables_dir / "kalshi_candles.parquet")
    p = add_lags(build_panel(candles, markets, prefixes))
    return p.with_columns(h_bucket(pl.col("h_to_exp")).alias("hb"), kind_of(pl.col("series_ticker")).alias("kind"))


def momentum_hist(s) -> Path:
    from albert.reports import Report

    p = prepare(s.tables_dir)
    rep = Report(s.reports_dir, "research-momentum-hist")
    rep.json("Sample", {
        "markets": p["ticker"].n_unique(), "matches": p["cluster"].n_unique(), "rows": p.height,
        "first_exp": p["expected_expiration_time"].min(), "last_exp": p["expected_expiration_time"].max(),
        "holdout_from": HOLDOUT_FROM, "by_kind": dict(p.group_by("kind").agg(pl.col("ticker").n_unique()).iter_rows()),
    })
    for name, fn in (("tsmom", model_tsmom), ("xsmom", model_xsmom), ("reversal", model_reversal),
                     ("drift", model_drift)):
        rep.table(name, pl.DataFrame(fn(p), infer_schema_length=None), save_as=name, max_rows=200)
    t = s.tables_dir
    if (t / "match_kalshi_pm.parquet").exists() and (t / "pm_price_history.parquet").exists():
        allp = prepare(t, prefixes=None)
        rows = model_pm(allp, pl.read_parquet(t / "pm_price_history.parquet"),
                        pl.read_parquet(t / "match_kalshi_pm.parquet"), pl.read_parquet(t / "pm_markets.parquet"))
        rep.table("pm", pl.DataFrame(rows, infer_schema_length=None), save_as="pm", max_rows=60)
    return rep.save()
