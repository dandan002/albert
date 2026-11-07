"""Observation-resolved weather brackets: how fast does Kalshi price a bracket the observations
have already decided?

Kalshi daily high/low markets (``KXHIGH*``, ``KXLOWT*``) settle on the station's NWS Daily
Climate Report (CLI). The rules now name "The Weather Company" as the source, but on Aug-Oct 2026
settled legs the result matches the NWS CLI value on 21,255 of 21,259 legs, so the CLI is used.
The CLI day is local *standard* midnight to midnight (``venues.weather.lst_day_bounds``).

Hourly ASOS observations bound the CLI value: the CLI max is >= the observed max on 99.8% of
station-days (hourly METARs miss the peak) and the CLI min <= the observed min on 99.1%. So once
the running observed max reaches ``hi + 1 + margin``, a high bracket ``[lo, hi]`` is (almost
surely) dead, and once it reaches ``lo + margin`` an "or above" bracket is (almost surely) won;
the mirror rules hold for lows. These are *resolution events*, known at the observation time plus
a publication lag (METARs are public within minutes).

For each event we read the market's executable quote from the hourly candles (Kalshi candle
``yes_bid_close`` / ``yes_ask_close``: the book at the period end) at the first candle end at or
after ``t_obs + lag`` and at later hours, and score a taker who sells a dead bracket (buys NO at
``1 - yes_bid``) or buys a won bracket at the ask, with the taker fee ``0.07 P (1 - P)``, held to
settlement. Candle quotes carry no size: results are per contract, at most a few contracts.
Trade prints (where we have them) give a finer reaction clock.

No look-ahead: an event uses only observations up to its own timestamp; quotes are taken strictly
after the event's availability time; the margin and the lag are chosen on the dev months
(Aug-Sep) and reported on the holdout (Oct).
"""

import math
from datetime import datetime, timedelta

import polars as pl

from albert.research.weather import _event_day, bracket
from albert.venues import weather as wx

TAKER = 0.07
LAGS_MIN = (5, 65, 125)   # quote at the first candle end >= t_obs + lag


def taker_fee(p: float) -> float:
    """Kalshi taker fee per contract at price p (exact; the account's balance precision is
    $0.0001, so per-order cent rounding does not apply)."""
    return math.ceil(round(TAKER * p * (1 - p) * 1e4, 9)) / 1e4


def legs(markets: pl.DataFrame) -> pl.DataFrame:
    """Settled KXHIGH / KXLOWT legs with station, CLI day, bracket and result."""
    m = markets.filter((pl.col("series_ticker").str.starts_with("KXHIGH")
                        | pl.col("series_ticker").str.starts_with("KXLOWT"))
                       & pl.col("result").is_in(["yes", "no"]))
    rows = []
    for r in m.iter_rows(named=True):
        code, b, day = wx.cli_code(r["rules_primary"]), bracket(r["yes_sub_title"]), _event_day(r["event_ticker"])
        if code not in wx.STATION_TZ or b is None or day is None:
            continue
        rows.append({"ticker": r["ticker"], "event": r["event_ticker"], "code": code, "day": day,
                     "kind": "high" if r["series_ticker"].startswith("KXHIGH") else "low",
                     "lo": b[0], "hi": b[1], "y": float(r["result"] == "yes")})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "event": pl.String, "code": pl.String,
                                      "day": pl.Date, "kind": pl.String, "lo": pl.Float64,
                                      "hi": pl.Float64, "y": pl.Float64})


def resolution(kind: str, lo: float, hi: float, running: list[tuple[datetime, float]],
               margin: float) -> tuple[str, datetime, float] | None:
    """First observation that decides the bracket: ("dead" | "won", obs time, obs value).
    ``running``: (time, running max for highs / running min for lows), in time order."""
    for t, v in running:
        if kind == "high":
            if math.isfinite(hi) and v >= hi + 1 + margin:
                return "dead", t, v
            if not math.isfinite(hi) and v >= lo + margin:
                return "won", t, v
        else:
            if math.isfinite(lo) and v <= lo - 1 - margin:
                return "dead", t, v
            if not math.isfinite(lo) and v <= hi - margin:
                return "won", t, v
    return None


def running_extreme(obs: pl.DataFrame, code: str, day, kind: str) -> list[tuple[datetime, float]]:
    a, b = wx.lst_day_bounds(code, day)
    g = obs.filter((pl.col("code") == code) & (pl.col("ts") >= a) & (pl.col("ts") < b)).sort("ts")
    out, cur = [], None
    for t, v in zip(g["ts"], g["tmpf"]):
        if v is None:
            continue
        cur = v if cur is None else (max(cur, v) if kind == "high" else min(cur, v))
        out.append((t, cur))
    return out


def events(lg: pl.DataFrame, obs: pl.DataFrame, margin: float) -> pl.DataFrame:
    rows = []
    cache: dict[tuple, list] = {}
    for r in lg.iter_rows(named=True):
        key = (r["code"], r["day"], r["kind"])
        if key not in cache:
            cache[key] = running_extreme(obs, *key)
        run = cache[key]
        if len(run) < 12:          # station-day without observations
            continue
        res = resolution(r["kind"], r["lo"], r["hi"], run, margin)
        if res is None:
            continue
        rows.append({**r, "what": res[0], "t_obs": res[1], "obs": res[2]})
    return pl.DataFrame(rows, infer_schema_length=None)


def quotes_after(ev: pl.DataFrame, candles: pl.DataFrame, lags_min=LAGS_MIN) -> pl.DataFrame:
    """Candle quotes just before the event and at the first candle end >= t_obs + lag."""
    c = candles.filter(pl.col("period_minutes") == 60).select(
        "ticker", "end_ts", "yes_bid_close", "yes_ask_close", "volume").sort("ticker", "end_ts")
    out = ev.sort("t_obs")
    before = out.join_asof(c.sort("end_ts"), left_on="t_obs", right_on="end_ts", by="ticker",
                           strategy="backward")
    out = out.with_columns(before["yes_bid_close"].alias("bid_before"),
                           before["yes_ask_close"].alias("ask_before"))
    for lag in lags_min:
        tmp = out.with_columns((pl.col("t_obs") + timedelta(minutes=lag)).alias("_t"))
        j = tmp.sort("_t").join_asof(c.sort("end_ts"), left_on="_t", right_on="end_ts", by="ticker",
                                     strategy="forward")
        j = j.with_columns(((pl.col("end_ts") - pl.col("_t")).dt.total_minutes() <= 60).alias("_ok"))
        out = j.select(*out.columns,
                       pl.when("_ok").then(pl.col("yes_bid_close")).alias(f"bid_{lag}"),
                       pl.when("_ok").then(pl.col("yes_ask_close")).alias(f"ask_{lag}"),
                       pl.when("_ok").then(pl.col("end_ts")).alias(f"ts_{lag}"))
    return out


def taker_trades(q: pl.DataFrame, lag: int, min_price: float = 0.02) -> pl.DataFrame:
    """Sell dead brackets at the bid / buy won brackets at the ask, held to settlement."""
    rows = []
    for r in q.iter_rows(named=True):
        bid, ask = r.get(f"bid_{lag}"), r.get(f"ask_{lag}")
        if r["what"] == "dead" and bid is not None and bid >= min_price:
            p, fee = bid, taker_fee(bid)
            pnl = p - fee - (1.0 if r["y"] == 1 else 0.0)        # sold YES at p (bought NO at 1-p)
            risk = 1 - p
        elif r["what"] == "won" and ask is not None and 0 < ask <= 1 - min_price:
            p, fee = ask, taker_fee(ask)
            pnl = (1.0 if r["y"] == 1 else 0.0) - p - fee
            risk = p
        else:
            continue
        rows.append({"ticker": r["ticker"], "cluster": f"{r['code']}-{r['day']}-{r['kind']}",
                     "day": r["day"], "kind": r["kind"], "what": r["what"], "price": p, "fee": fee,
                     "pnl": pnl, "risk": risk, "right": (r["y"] == 0) == (r["what"] == "dead")})
    return pl.DataFrame(rows, infer_schema_length=None)


def clustered(f: pl.DataFrame, col: str = "pnl", cluster: str = "cluster") -> dict:
    if f.is_empty():
        return {"n": 0}
    g = f.group_by(cluster).agg(pl.col(col).sum().alias("s"), pl.len().alias("n"),
                                pl.col("risk").sum().alias("r"))
    n, k = g["n"].sum(), g.height
    mean = g["s"].sum() / n
    se = math.sqrt(k / (k - 1) * ((g["s"] - mean * g["n"]) ** 2).sum()) / n if k > 1 else float("nan")
    return {"n": int(n), "clusters": k, "pnl_per_contract": mean, "t": mean / se if se else None,
            "pnl": g["s"].sum(), "per_dollar_risked": g["s"].sum() / g["r"].sum()}


def trade_reaction(ev: pl.DataFrame, trades: pl.DataFrame, lag_min: float = 5.0,
                   stale: float = 0.03) -> pl.DataFrame:
    """Per dead-bracket event with trade prints: prints at a YES price >= ``stale`` after the
    event became public (t_obs + lag). Taker 'no' prints there were YES bids still resting
    (someone sold the dead bracket at p, the trade we want); taker 'yes' prints were someone
    *buying* the dead bracket from a resting NO bid (the maker's side of the same edge)."""
    t = trades.select("ticker", "created_time", "yes_price", "count", "taker_side")
    d = ev.filter(pl.col("what") == "dead").select("ticker", "event", "code", "day", "kind", "y", "t_obs")
    j = d.join(t, on="ticker").filter(
        pl.col("created_time") >= pl.col("t_obs") + timedelta(minutes=lag_min))
    return j.group_by("ticker", "event", "code", "day", "kind", "y", "t_obs").agg(
        (pl.col("yes_price") >= stale).sum().alias("stale_prints"),
        pl.col("count").filter(pl.col("yes_price") >= stale).sum().alias("stale_contracts"),
        pl.col("count").filter((pl.col("yes_price") >= stale) & (pl.col("taker_side") == "yes")).sum()
        .alias("stale_buy_contracts"),
        ((pl.col("created_time").filter(pl.col("yes_price") >= stale).max() - pl.col("t_obs").first())
         .dt.total_minutes()).alias("last_stale_min"),
        pl.len().alias("prints_after"))
