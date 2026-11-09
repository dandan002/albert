"""Buy $1 of every Kalshi contract the first time it is priced above a threshold; hold to settlement.

Data: hourly candles (yes_bid_close, yes_ask_close) in data/tables/kalshi_candles.parquet (the
top-volume markets per category, closes 2026-08-15..10-31, non-financial categories only) and
results from kalshi_markets.parquet. No look-ahead: the decision uses the candle that closed
strictly before the market's close_time; the outcome is used only to score.

Rule: the first hourly candle where the mid of one side is above THRESH -> buy $1 of that side at
its ask (YES ask = yes_ask_close, NO ask = 1 - yes_bid_close), requiring a two-sided quote and an
ask <= 0.99. Taker fee 0.07 x fee_multiplier x C x P x (1 - P), exact to $0.0001 (direct member).

    python -m albert.research.favourites [THRESH ...]
"""
import json
import math
import sys
from pathlib import Path

import polars as pl

OUT = Path("data/research/favourites")

THRESHES = [float(x) for x in sys.argv[1:]] or [0.80]


def cluster_ratio(df: pl.DataFrame, num: str, den: str, cl: str):
    g = df.group_by(cl).agg(pl.col(num).sum().alias("s"), pl.col(den).sum().alias("c"))
    G = g.height
    tot = g["c"].sum()
    if G < 2 or tot <= 0:
        return None, None, G
    r = g["s"].sum() / tot
    v = ((g["s"] - r * g["c"]) ** 2).sum() * G / (G - 1)
    se = math.sqrt(v) / tot
    return r, (r / se if se > 0 else None), G


def summarize(d: pl.DataFrame) -> dict:
    if d.height == 0:
        return {"markets": 0}
    roi, t, ev = cluster_ratio(d, "net", "stake", "event_ticker")
    return {"markets": d.height, "events": ev, "invested_$": round(d["stake"].sum(), 2),
            "payout_$": round(d["payout"].sum(), 2), "fees_$": round(d["fee"].sum(), 2),
            "net_$": round(d["net"].sum(), 2), "roi_%": None if roi is None else round(100 * roi, 2),
            "t_by_event": None if t is None else round(t, 2),
            "win_rate": round(d["win"].mean(), 4), "avg_ask": round(d["ask"].mean(), 4),
            "avg_mid": round(d["mid"].mean(), 4),
            "median_hold_h": round(d["hold_h"].median(), 1),
            "worst_event_$": round(d.group_by("event_ticker").agg(pl.col("net").sum())["net"].min(), 2)}


def main():
    m = (pl.scan_parquet("data/tables/kalshi_markets.parquet")
         .filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary"))
         .select("ticker", "event_ticker", "series_ticker", "category", "open_time", "close_time",
                 "result", "fee_type", "fee_multiplier"))
    c = (pl.scan_parquet("data/tables/kalshi_candles.parquet")
         .select("ticker", "end_ts", "yes_bid_close", "yes_ask_close")
         .join(m, on="ticker", how="inner")
         .filter((pl.col("end_ts") < pl.col("close_time")) & (pl.col("end_ts") >= pl.col("open_time"))
                 & (pl.col("yes_bid_close") > 0) & (pl.col("yes_ask_close") < 1)
                 & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
         .with_columns(mid=(pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2)
         .collect())
    universe = c["ticker"].n_unique()
    out = {"universe_markets_with_quotes": universe, "thresholds": {}}
    for th in THRESHES:
        e = (c.with_columns(
                side=pl.when(pl.col("mid") > th).then(pl.lit("yes"))
                .when(pl.col("mid") < 1 - th).then(pl.lit("no")).otherwise(None))
             .filter(pl.col("side").is_not_null())
             .with_columns(ask=pl.when(pl.col("side") == "yes").then(pl.col("yes_ask_close"))
                           .otherwise(1 - pl.col("yes_bid_close")),
                           fav_mid=pl.when(pl.col("side") == "yes").then(pl.col("mid")).otherwise(1 - pl.col("mid")))
             .filter(pl.col("ask") <= 0.99 + 1e-9)
             .sort("end_ts").group_by("ticker", maintain_order=True).first())
        e = e.with_columns(
            stake=pl.lit(1.0),
            contracts=1.0 / pl.col("ask"),
            win=(pl.col("side") == pl.col("result")).cast(pl.Float64),
            mult=pl.col("fee_multiplier").fill_null(1.0),
        ).with_columns(
            fee=(0.07 * pl.col("mult") * pl.col("contracts") * pl.col("ask") * (1 - pl.col("ask")) * 1e4).ceil() / 1e4,
            payout=pl.col("contracts") * pl.col("win"),
            hold_h=(pl.col("close_time") - pl.col("end_ts")).dt.total_seconds() / 3600,
            month=pl.col("close_time").dt.strftime("%Y-%m"),
            band=(pl.when(pl.col("ask") <= 0.85).then(pl.lit("a 0.81-0.85"))
                  .when(pl.col("ask") <= 0.90).then(pl.lit("b 0.86-0.90"))
                  .when(pl.col("ask") <= 0.95).then(pl.lit("c 0.91-0.95"))
                  .otherwise(pl.lit("d 0.96-0.99"))),
        ).with_columns(net=pl.col("payout") - pl.col("stake") - pl.col("fee"))
        e = e.rename({"mid": "mid_yes"}).rename({"fav_mid": "mid"})
        res = {"all": summarize(e),
               "by_ask_band": {b: summarize(e.filter(pl.col("band") == b)) for b in sorted(e["band"].unique())},
               "by_category": {k: summarize(e.filter(pl.col("category") == k)) for k in sorted(e["category"].unique())},
               "by_month": {k: summarize(e.filter(pl.col("month") == k)) for k in sorted(e["month"].unique())},
               "by_hold": {k: summarize(e.filter(f)) for k, f in [
                   ("<1h", pl.col("hold_h") < 1), ("1-6h", (pl.col("hold_h") >= 1) & (pl.col("hold_h") < 6)),
                   ("6-24h", (pl.col("hold_h") >= 6) & (pl.col("hold_h") < 24)),
                   ("1-7d", (pl.col("hold_h") >= 24) & (pl.col("hold_h") < 168)), (">7d", pl.col("hold_h") >= 168)]}}
        # one $1 bet per EVENT (its first qualifying market) to remove ladder duplication
        first_ev = e.sort("end_ts").group_by("event_ticker", maintain_order=True).first()
        res["one_per_event"] = summarize(first_ev)
        out["thresholds"][str(th)] = res
        OUT.mkdir(parents=True, exist_ok=True)
        e.write_parquet(OUT / f"fav_{int(th * 100)}.parquet")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
