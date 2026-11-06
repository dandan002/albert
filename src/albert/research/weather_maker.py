"""Passive quoting around a model + price blend in Kalshi weather markets (study ``weather_maker``).

Question: does resting one YES bid and one NO bid around a *fair value* (a dev-fitted blend of a
fundamental model and the market mid) earn after fees on the October 2026 holdout in daily rain
(``KXRAIN``, ``KXRAINWKND``, monthly ``KXRAIN*M``) or hourly temperature (``KXTEMP*``) markets,
and how many fills per week would it get?

Round 1 (``research/rain.py``, ``research/hourly_temp.py``) found the rain model adds information
to the mid on both periods but the taker loses the spread; the hourly nowcast loses to the price,
but at market open the books are 10-15c wide and the mid looked under-confident. This study reuses
both models unchanged (imported) and asks whether the information inside the spread is capturable
by a maker.

Fill rule (``fundamental.maker_fills`` semantics, re-implemented vectorised in :func:`fills`): a
bid at ``floor(fair - margin)`` (capped one tick inside the opposite quote at ``t``, post-only)
fills only if a later trade prints strictly THROUGH it (a YES-seller below our YES bid; a
YES-buyer above 1 - our NO bid) before the order is cancelled. One contract, held to settlement,
series maker fee (these series are fee type ``quadratic``: no maker fee). A lenient variant also
fills at our own price when our bid improved the best bid at ``t`` (we would be first in the queue
at a new level).

Cancel times (the outcome becomes observable): rain = the first ASOS report of >= 0.01" inside the
CLI day (valid time + ``OBS_LAG``) or the end of the local-standard-time day; hourly temperature =
the settlement minute S (or earlier, chosen on dev); monthly rain = the next daily requote or the
first CLI day that pushes the month total over the strike.

As-of rules: model probabilities are the round-1 frozen models (each input timestamped before
``t``, see those modules); the quote is the last candle that closed at or before ``t``; blends are
fitted on dev only (dev rows get out-of-fold blends, by event day) or, for the drift-robust
variant, on a rolling window of rows whose markets settled before ``t``. Rain and monthly splits go
by the event's date (an Oct 1 market decided on Sep 30 is holdout); hourly by t (S >= Oct 1 01:00).

Fill sources: rain and monthly use the round-1 hourly candles' trade-price low / high (no new
requests; "pessimistic" keeps prints in the hour that contains the cancel time), hourly temperature
the 1-minute candles. On the 261 rain markets whose exact trades were fetched, candle fills agree
with trade fills on 99.4-99.6% of orders and the pessimistic bound equals the exact-trade result;
the taker-side rule changes one fill in ~270 (18 hourly markets: 98-100% agreement).

Result (2026-11-06, report research-fundamental-weather_maker-*):

* **Rain, no.** Main test (quote at d 00z, requote at d 12z, cancel when rain is reported or the
  CLI day ends; dev-fitted per-slot blend; margin 8c chosen on dev): dev +4.2c per fill (t 7.3,
  906 fills), October +1.2c (t 1.1, 489 fills on 30 event days, ~110 fills / week, ~$1.3 / week at
  one contract). Evening or morning alone: +0.5c / +0.3c. Mid-centred quotes +0.4c. The dev profit
  was the Aug-Sep YES over-pricing (NO bids +6.4c dev, -0.3c October); the drift-robust 28-day
  rolling blend earns +0.2c in October. KXRAINWKND is positive (+2.8c) but has 4 October weekends.
* **Hourly temperature, no at the harness margins (<= 8c):** static quote from open+1 min to S
  -1.3c (t -2.9); requote at 55/40/25/10 -0.1c. With margins up to 20c dev picks the grid edge and
  October earns +1.5c per fill (t 3.0, ~1,150 fills / week): deep NO bids filled by YES-buyer
  spikes. The plain mid does the same (+1.35c) and NO-bid-only does better (+4.1c), so it is the
  YES over-buy pattern, not the nowcast.
* **By quoted spread at t** (coordinator check): rain's model information survives in real books:
  blend c in <= 3c books 0.59 (t 5.9) dev, 0.36 (t 2.1) October refit (93% of October rows), all
  from the morning slot (d 12z refit c 0.68, t 3.6; d 00z 0.01); the maker in <= 3c books earns
  +1.5c (t 1.3) in October vs +0.4c around the mid. Hourly: c ~ 0 in <= 3c books (0.01 / -0.05),
  only > 15c books carry it (0.19, t 4.9), and at 8c margins those books lose (-2.0c, t -2.9).
* **Monthly ladders (KXRAIN*M; dev = 10 September ladders, holdout = 15 October ones):** the model
  has no information (dev c -0.28); the October blend maker's +12c per fill (t 4.9, 88 fills) is a
  recalibration of a one-month regime (October was dry; mid 0.40 -> 3% YES) that the mid-only
  recalibration reproduces (+25c at 20c), and dev earned -0.3c. Not evidence of an edge.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

NAME = "weather_maker"
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
OBS_LAG = timedelta(minutes=10)


def cache_dir(s) -> Path:
    return s.data_dir / "cache" / NAME


def _table(s, name: str) -> Path:
    return s.data_dir / "cache" / f"fundamental_{NAME}_{name}.parquet"


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---- public trades (copy of fundamental.fetch_trades with the raw cache under data/cache) ----
TRADE_SCHEMA = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
                "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}


def fetch_trades(s, windows: pl.DataFrame, tag: str, *, workers: int = 2, log=print) -> pl.DataFrame:
    """Every public trade of each market inside [t_from, t_to] (columns ticker, t_from, t_to).
    Same logic as ``fundamental.fetch_trades``; raw pages are cached under
    data/cache/weather_maker/kalshi and rows in data/cache/fundamental_weather_maker_trades_<tag>."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = _table(s, f"trades_{tag}")
    done_path = _table(s, f"trades_{tag}_windows")
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=TRADE_SCHEMA)
    done = pl.read_parquet(done_path) if done_path.exists() else pl.DataFrame(
        schema={"ticker": pl.String, "t_from": pl.Datetime("us", "UTC"), "t_to": pl.Datetime("us", "UTC")})
    w = windows.select("ticker", "t_from", "t_to").unique()
    todo = w.join(done, on=["ticker", "t_from", "t_to"], how="anti")
    if not todo.height:
        return have
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, cache_dir(s), min_interval=0.1, max_retries=8))
    cut = api.cutoff()
    cutoff = cut.get("trades_created_ts") or cut.get("market_settled_ts")

    def one(r: dict[str, Any]) -> list[dict[str, Any]]:
        historical = bool(cutoff and r["t_to"] < cutoff)
        out: list[dict[str, Any]] = []
        for attempt in range(4):
            try:
                out = []
                for page in api.trades(r["ticker"], historical, min_ts=r["t_from"], max_ts=r["t_to"]):
                    out.extend(k.trade_rows(page.payload))
                return out
            except Exception as error:  # noqa: BLE001  (429 storms from the shared rate limit)
                if attempt == 3:
                    log(f"skip trades {r['ticker']}: {str(error)[:120]}")
                    return []
                import time

                time.sleep(10 * (attempt + 1))
        return out

    rows_todo = list(todo.sort("t_to", descending=True).iter_rows(named=True))   # newest (holdout) first
    chunk = 50
    for i in range(0, len(rows_todo), chunk):
        part = rows_todo[i:i + chunk]
        got: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(one, part):
                got.extend(r)
        new = pl.DataFrame(got, schema=TRADE_SCHEMA) if got else pl.DataFrame(schema=TRADE_SCHEMA)
        have = pl.concat([have, new]).unique("trade_id")
        done = pl.concat([done, pl.DataFrame(part).select("ticker", "t_from", "t_to")])
        _write(have, path)
        _write(done, done_path)
        log(f"  trades[{tag}] {min(i + chunk, len(rows_todo))}/{len(rows_todo)} windows, {have.height} trades "
            f"{datetime.now(UTC):%H:%M:%S}")
    return have


# ---- market lists and trade windows ------------------------------------------------------------
RAIN_SERIES = ("KXRAIN", "KXRAINWKND")
HOURLY_SERIES = ("KXTEMPMIAH", "KXTEMPNYCHS", "KXTEMPCHIHS", "KXTEMPLAXHS")
HOURLY_FROM = datetime(2026, 9, 22, 1, tzinfo=UTC)   # minute index archive starts 2026-09-21 22:22 UTC


def rain_windows(s) -> pl.DataFrame:
    """Every settled daily / weekend rain market over its whole life (open .. close)."""
    from albert.research import fundamental as fu

    m = fu.settled_markets(s, list(RAIN_SERIES))
    return m.select("ticker", pl.col("open_time").alias("t_from"), pl.col("close_time").alias("t_to"))


def hourly_windows(s) -> pl.DataFrame:
    """Hourly temperature markets settling from HOURLY_FROM that traded at all (volume > 0 in the
    ingested markets table; a zero-volume market has no prints to fill against)."""
    mk = pl.read_parquet(s.data_dir / "cache" / "hourly_temp" / "markets.parquet")
    vol = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet")
           .filter(pl.col("series_ticker").is_in(list(HOURLY_SERIES))).select("ticker", "volume").collect())
    j = mk.join(vol, on="ticker", how="left").filter((pl.col("close_time") >= HOURLY_FROM)
                                                     & (pl.col("volume").fill_null(1) > 0))
    return j.select("ticker", pl.col("open_time").alias("t_from"), pl.col("close_time").alias("t_to"))


# ---- rain: round-1 model rows, cancel times ----------------------------------------------------
def rain_candles(s) -> pl.DataFrame:
    """Hourly quotes the round-1 rain study fetched (harness candles + batch candles)."""
    from albert.research import rain

    harness = pl.read_parquet(s.data_dir / "cache" / f"fundamental_{rain.NAME}_candles.parquet")
    batch = rain.load(s, "candles_batch")
    cols = ["ticker", "end_ts", "yes_bid_close", "yes_ask_close"]
    return pl.concat([harness.select(cols),
                      batch.filter(~pl.col("ticker").is_in(harness["ticker"].unique().implode())).select(cols)])


def first_rain(obs: pl.DataFrame, win: pl.DataFrame) -> pl.DataFrame:
    """Per window (code, key, lo, hi): the first ASOS report with p01i >= 0.01 inside (lo, hi]
    (``wet_ts``), or null. Any report counts here, including one at lo + 51 min whose hour began
    before lo: for cancelling a quote, rain visible at the station is what matters."""
    o = obs.filter(pl.col("p01i") >= 0.01).select("code", "ts")
    j = (win.select("code", "key", "lo", "hi").join(o, on="code", how="inner")
         .filter((pl.col("ts") > pl.col("lo")) & (pl.col("ts") <= pl.col("hi")))
         .group_by("code", "key").agg(pl.col("ts").min().alias("wet_ts")))
    return win.join(j, on=["code", "key"], how="left")


def rain_rows(s, *, log=print) -> pl.DataFrame:
    """Round-1 rain model (variant ``ens``, ridge 30, expanding monthly fits; frozen) at the four
    round-1 decision slots for KXRAIN and three for KXRAINWKND, joined to the quote at t, the
    series fee, and the cancel time ``end`` = min(first reported rain + OBS_LAG, end of the CLI
    window, market close). Rows already decided at t (rain seen) are dropped, as in round 1."""
    from albert.research import fundamental as fu
    from albert.research import rain

    markets = fu.settled_markets(s, list(rain.SERIES))
    fees = markets.select("ticker", "fee_type", "fee_multiplier")
    legs = rain.market_legs(markets)
    codes = sorted(set(legs["code"].to_list()))
    feats = rain.station_day_features(s, codes, False)      # round 1: trace settles NO (203/203)
    preds = rain.expanding_predictions(feats, "ens", [date(2026, m, 1) for m in (8, 9, 10)], lam=30.0)
    candles = rain_candles(s)
    obs = rain.load(s, "obs")
    # KXRAIN
    daily = legs.filter(pl.col("series") == "KXRAIN")
    parts = []
    for slot in rain.MARKET_SLOTS:
        parts.append(daily.with_columns(pl.lit(slot).alias("slot"), pl.col("day").map_elements(
            lambda d, sl=slot: rain.slot_time(d, sl), return_dtype=pl.Datetime("us", "UTC")).alias("t")))
    mk = pl.concat(parts).filter((pl.col("t") >= pl.col("open_time") + timedelta(minutes=50))
                                 & (pl.col("t") < pl.col("close_time")))
    mk = (mk.join(feats.select("code", "day", "slot", "decided"), on=["code", "day", "slot"], how="left")
          .join(preds.select("code", "day", "slot", "p"), on=["code", "day", "slot"], how="left"))
    win = []
    for code, day in mk.select("code", "day").unique().iter_rows():
        lo, hi = rain.lst_bounds(code, day)
        win.append({"code": code, "key": day.isoformat(), "lo": lo, "hi": hi})
    wd = first_rain(obs, pl.DataFrame(win, schema={"code": pl.String, "key": pl.String,
                                                   "lo": pl.Datetime("us", "UTC"), "hi": pl.Datetime("us", "UTC")}))
    mk = mk.with_columns(pl.col("day").cast(pl.String).alias("key")).join(wd, on=["code", "key"], how="left")
    daily_rows = mk.filter(~pl.col("decided").fill_null(False) & pl.col("p").is_not_null())
    # KXRAINWKND (P(rain Sat or Sun) from the daily model, round-1 weekend_rows)
    wk_legs = legs.filter(pl.col("series") == "KXRAINWKND")
    sats = sorted(set(wk_legs["day"].to_list()))
    satf = rain.saturday_full(codes, sats, obs, False)
    seen = pl.concat([feats.select("code", "day", "slot", "decided"), satf.select("code", "day", "slot", "decided")])
    wk = rain.weekend_rows(wk_legs, preds, seen)
    wk = wk.filter((pl.col("t") >= pl.col("open_time") + timedelta(minutes=50)) & (pl.col("t") < pl.col("close_time"))
                   & ~pl.col("decided") & pl.col("p").is_not_null())
    wwin = []
    for code, ev in wk.select("code", "event").unique().iter_rows():
        sat = rain.event_day(ev)
        lo, _ = rain.lst_bounds(code, sat)
        _, hi = rain.lst_bounds(code, sat + timedelta(days=1))
        wwin.append({"code": code, "key": ev, "lo": lo, "hi": hi})
    ww = first_rain(obs, pl.DataFrame(wwin, schema={"code": pl.String, "key": pl.String,
                                                    "lo": pl.Datetime("us", "UTC"), "hi": pl.Datetime("us", "UTC")}))
    wk = (wk.with_columns(pl.col("event").alias("key"), pl.lit("KXRAINWKND").alias("series"))
          .join(ww, on=["code", "key"], how="left"))
    cols = ["ticker", "event", "series", "code", "slot", "t", "p", "y", "close_time", "lo", "hi", "wet_ts"]
    both = pl.concat([daily_rows.select(cols), wk.select(cols)], how="vertical_relaxed")
    q = fu.attach_quotes(both, candles).join(fees, on="ticker", how="left")
    end = pl.min_horizontal(pl.col("wet_ts") + OBS_LAG, pl.col("hi"), pl.col("close_time"))
    q = q.with_columns(end.alias("end"), ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                       (pl.col("t") >= HOLDOUT_FROM).alias("holdout"),
                       pl.col("event").str.extract(r"-(\d{2}[A-Z]{3}\d{2})", 1).alias("day_key"))
    log(f"rain rows {q.height} (daily {daily_rows.height}, weekend {wk.height}); "
        f"cancel before t: {(q['end'] <= q['t']).sum()}")
    return q.filter(pl.col("end") > pl.col("t"))


# ---- hourly temperature: minute candles with trade-price extremes ------------------------------
MINUTE_SCHEMA = {"ticker": pl.String, "end_ts": pl.Datetime("us", "UTC"), "p_low": pl.Float64,
                 "p_high": pl.Float64, "volume": pl.Float64, "bid": pl.Float64, "ask": pl.Float64}


def minute_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """``GET /markets/candlesticks`` (period 1) -> one row per market-minute: the trade-price low /
    high inside the minute (null when nothing traded), volume, and the closing yes bid / ask."""
    from albert.venues import kalshi as k

    rows = []
    for mk in payload.get("markets") or []:
        for c in mk.get("candlesticks") or []:
            price = c.get("price") or {}
            rows.append({"ticker": mk["market_ticker"], "end_ts": k.parse_ts(c.get("end_period_ts")),
                         "p_low": k.num(k._field(price, "low")), "p_high": k.num(k._field(price, "high")),
                         "volume": k.num(k._field(c, "volume")),
                         "bid": k.num(k._field(c.get("yes_bid") or {}, "close")),
                         "ask": k.num(k._field(c.get("yes_ask") or {}, "close"))})
    return rows


def hourly_minutes(s) -> pl.DataFrame:
    """Every cached 1-minute batch-candle response of the round-1 hourly study (no new requests),
    parsed with :func:`minute_rows`; cached as a parquet."""
    from albert.http import read_raw

    path = _table(s, "hourly_minutes")
    if path.exists():
        return pl.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for f in sorted((s.data_dir / "cache" / "hourly_temp" / "kalshi").glob("*")):
        rec = read_raw(f)
        if "/markets/candlesticks" in rec["url"] and str(rec["params"].get("period_interval")) == "1":
            rows.extend(minute_rows(rec["response"]))
    out = pl.DataFrame(rows, schema=MINUTE_SCHEMA).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
    _write(out, path)
    return out


# ---- hourly temperature: round-1 nowcast at market open and the round-1 leads -----------------
OPEN_LEAD = 59                          # the market lists at S - 60; we quote one minute later
HOURLY_LEADS = (OPEN_LEAD, 55, 40, 25, 10)


def _fit_leads(inp: dict[str, Any], fit_end: datetime, leads: tuple[int, ...]):
    """``hourly_temp.fit_all`` for an arbitrary set of leads (copied: fit_all loops over the
    module-level LEADS). Per (city, lead): the frozen model on dev minutes and one model per
    left-out UTC day (dev predictions are out-of-day)."""
    from albert.research import hourly_temp as ht

    coef = ht.fit_drift(inp["proxy"] + [dict(inp["grid"])])
    fits: dict[tuple, dict[str, Any]] = {}
    for city in ht.TZ:
        for lead in leads:
            f, dy, day = ht.training_samples(inp["grid"], inp["fc"], city, lead, coef, ht.DEV_FROM, fit_end)
            fits[(city, lead, None)] = ht.fit_variant(f, dy, ht.CITY_VARIANT[city])
            for dd in np.unique(day):
                m = day != dd
                fits[(city, lead, int(dd))] = ht.fit_variant({k: v[m] for k, v in f.items()}, dy[m],
                                                             ht.CITY_VARIANT[city])
    return coef, fits


def hourly_rows(s, *, leads: tuple[int, ...] = HOURLY_LEADS, log=print) -> pl.DataFrame:
    """Round-1 OU nowcast (unchanged) at each lead, dev out-of-day and holdout frozen on dev, with
    the 1-minute quote at t (two-sided books only). Cached."""
    from albert.research import hourly_temp as ht

    path = _table(s, "hourly_rows")
    if path.exists():
        return pl.read_parquet(path)
    inp_fit = ht.load_inputs(s, through=ht.HOLDOUT_FROM)
    coef, fits = _fit_leads(inp_fit, ht.HOLDOUT_FROM, leads)
    log(f"hourly fits {len(fits)} {datetime.now(UTC):%H:%M:%S}")
    dev_ev = ht.event_frame(inp_fit["markets"], ht.DEV_FROM + timedelta(hours=1), ht.HOLDOUT_FROM)
    dev = ht.predict_events(dev_ev, inp_fit["grid"], inp_fit["fc"], coef,
                            {k: f for k, f in fits.items() if k[2] is not None}, leads=leads)
    inp_all = ht.load_inputs(s, through=ht.HOLDOUT_TO)
    hold_ev = ht.event_frame(inp_all["markets"], ht.HOLDOUT_FROM + timedelta(hours=1), ht.HOLDOUT_TO)
    hold = ht.predict_events(hold_ev, inp_all["grid"], inp_all["fc"], coef,
                             {k: f for k, f in fits.items() if k[2] is None}, leads=leads)
    pred = pl.concat([dev.with_columns(pl.lit("dev").alias("split")),
                      hold.with_columns(pl.lit("holdout").alias("split"))], how="diagonal_relaxed")
    q = ht.quotes_at(pred, ht.load_candles(s)).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    fees = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet")
            .filter(pl.col("series_ticker").is_in(list(HOURLY_SERIES)))
            .select("ticker", "fee_type", "fee_multiplier").collect())
    q = q.join(fees, on="ticker", how="left").with_columns(
        pl.col("fee_type").fill_null("quadratic"), pl.col("fee_multiplier").fill_null(1.0),
        (pl.col("t") >= HOLDOUT_FROM).alias("holdout"))
    _write(q, path)
    log(f"hourly rows {q.height} {datetime.now(UTC):%H:%M:%S}")
    return q


# ---- fills -------------------------------------------------------------------------------------
def trade_prints(trades: pl.DataFrame) -> pl.DataFrame:
    """Public trades -> prints for the fill test (``fundamental.maker_fills`` rule): ``lo`` is a
    YES-seller's price (taker_side no; fills a YES bid strictly above it), ``hi`` a YES-buyer's
    price (taker_side yes; fills a NO bid whose YES-equivalent 1 - bid is strictly below it)."""
    return trades.select(
        "ticker", pl.col("created_time").alias("ts"),
        pl.when(pl.col("taker_side") == "no").then(pl.col("yes_price")).otherwise(None).alias("lo"),
        pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price")).otherwise(None).alias("hi"))


def candle_prints(minutes: pl.DataFrame) -> pl.DataFrame:
    """1-minute candles -> prints stamped at the minute's end: the minute's trade-price low and
    high, whichever side took. With our bid resting, any print below it means the book traded
    through our level (a resting ask below our bid would have crossed us), so the side-blind rule
    is the right counterfactual; it is slightly more lenient than the taker-side rule."""
    return minutes.filter(pl.col("p_low").is_not_null()).select(
        "ticker", pl.col("end_ts").alias("ts"), pl.col("p_low").alias("lo"), pl.col("p_high").alias("hi"))


def window_extremes(rows: pl.DataFrame, prints: pl.DataFrame) -> pl.DataFrame:
    """For each row (ticker, t, end): the lowest ``lo`` and highest ``hi`` printed in (t, end]
    (``lo_min`` / ``hi_max``; +inf / -inf when nothing printed). The fill test at any bid price
    only needs these two numbers (one contract, held to settlement)."""
    pr = prints.sort("ticker", "ts")
    by: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (tk,), g in pr.group_by("ticker"):
        by[tk] = (g["ts"].dt.epoch("us").to_numpy(), g["lo"].fill_null(np.inf).to_numpy(),
                  g["hi"].fill_null(-np.inf).to_numpy())
    tks = rows["ticker"].to_list()
    t0 = rows["t"].dt.epoch("us").to_numpy()
    t1 = rows["end"].dt.epoch("us").to_numpy()
    lo_min = np.full(len(tks), np.inf)
    hi_max = np.full(len(tks), -np.inf)
    for i, tk in enumerate(tks):
        got = by.get(tk)
        if got is None:
            continue
        ts, lo, hi = got
        a = np.searchsorted(ts, t0[i], side="right")
        b = np.searchsorted(ts, t1[i], side="right")
        if b > a:
            lo_min[i] = lo[a:b].min()
            hi_max[i] = hi[a:b].max()
    return rows.with_columns(pl.Series("lo_min", lo_min), pl.Series("hi_max", hi_max))


def orders(rows: pl.DataFrame, fair: str, margin: float, *, lenient: bool = False, gate: float = 0.0,
           one_sided: bool = False, no_only: bool = False, maker_fee_type: str | None = None) -> pl.DataFrame:
    """One YES bid and one NO bid per row around ``rows[fair]`` (needs bid, ask, y, lo_min, hi_max):
    YES at floor(fair - margin), NO at floor(1 - fair - margin), each capped one tick inside the
    opposite quote (post-only). ``gate``: rest only when |fair - mid| >= gate; ``one_sided``: only
    the side the fair value favours versus the mid; ``no_only``: only the NO bid. Filled when a
    later print went strictly through (``lenient``: also at our price when our bid improved the
    best bid at t). PnL per
    contract held to settlement minus the series maker fee (``maker_fee_type`` overrides it)."""
    p = rows[fair].to_numpy()
    bid, ask = rows["bid"].to_numpy(), rows["ask"].to_numpy()
    mid = (bid + ask) / 2
    y = rows["y"].to_numpy().astype(float)
    yb = np.minimum(np.floor((p - margin) * 100 + 1e-9) / 100, np.round(ask - 0.01, 2))
    nb = np.minimum(np.floor((1 - p - margin) * 100 + 1e-9) / 100, np.round(1 - bid - 0.01, 2))
    lo_min, hi_max = rows["lo_min"].to_numpy(), rows["hi_max"].to_numpy()
    fy = lo_min < yb - 1e-9
    fn = hi_max > 1 - nb + 1e-9
    if lenient:
        fy |= (yb > bid + 1e-9) & (lo_min <= yb + 1e-9)
        fn |= (1 - nb < ask - 1e-9) & (hi_max >= 1 - nb - 1e-9)
    live = np.isfinite(p) & (np.abs(p - mid) >= gate - 1e-12)
    every = np.ones(len(p), dtype=bool)
    ly = live & (yb >= 0.01) & ((p > mid) if one_sided else every) & (not no_only)
    ln = live & (nb >= 0.01) & ((p < mid) if one_sided else every)
    ft = (rows["fee_type"].fill_null("").to_numpy() if "fee_type" in rows.columns else np.full(rows.height, ""))
    if maker_fee_type is not None:
        ft = np.full(rows.height, maker_fee_type)
    fm = rows["fee_multiplier"].fill_null(1.0).to_numpy() if "fee_multiplier" in rows.columns else np.ones(rows.height)
    charged = np.array(["maker_fee" in str(f) for f in ft]) if len(ft) else np.zeros(0, dtype=bool)

    def fee(x: np.ndarray) -> np.ndarray:
        # fundamental.maker_fee, vectorised: 0.0175 x multiplier x P(1-P), rounded up to $0.0001
        f = np.ceil(0.0175 * fm * x * (1 - x) * 1e4 - 1e-9) / 1e4
        return np.where(charged & (x >= 0.01), f, 0.0)

    fee_y, fee_n = fee(yb), fee(nb)
    keep = [c for c in ("ticker", "event", "cluster", "t", "holdout", "slot", "lead", "code", "city", "series",
                        "band", "week", "sbucket", "mid") if c in rows.columns]
    base = rows.select(keep)
    yes = base.with_columns(pl.lit("yes").alias("side"), pl.Series("price", yb), pl.Series("posted", ly),
                            pl.Series("filled", ly & fy), pl.Series("pnl", np.where(ly & fy, y - yb - fee_y, 0.0)))
    no = base.with_columns(pl.lit("no").alias("side"), pl.Series("price", nb), pl.Series("posted", ln),
                           pl.Series("filled", ln & fn), pl.Series("pnl", np.where(ln & fn, (1 - y) - nb - fee_n, 0.0)))
    return pl.concat([yes, no]).filter("posted").drop("posted")


def first_fill_per_side(o: pl.DataFrame) -> pl.DataFrame:
    """Replace mode: one resting bid per side per market; once a side has filled, later quotes on
    that side of the same market are not posted (orders after the first fill are dropped)."""
    o = o.sort("ticker", "side", "t")
    first = o.filter("filled").group_by("ticker", "side").agg(pl.col("t").min().alias("t_fill"))
    return (o.join(first, on=["ticker", "side"], how="left")
            .filter(pl.col("t_fill").is_null() | (pl.col("t") <= pl.col("t_fill"))).drop("t_fill"))


def summarize(o: pl.DataFrame, weeks: dict[str, float], by: list[str] | None = None) -> pl.DataFrame:
    """Orders, fills, fill rate, PnL per fill (cents, t clustered by ``cluster``), PnL per order,
    fills per week and dollars per week (one contract per fill) by split (and ``by``)."""
    from albert.research.fundamental import clustered_mean

    out = []
    keys = ["holdout", *(by or [])]
    for key, g in o.group_by(keys):
        split = "holdout" if key[0] else "dev"
        f = g.filter("filled")
        row = {"split": split, **{k: v for k, v in zip(keys[1:], key[1:])}, "orders": g.height, "fills": f.height,
               "fill_rate": round(f.height / g.height, 3) if g.height else None}
        if f.height:
            mu, tt, k = clustered_mean(f["pnl"].to_numpy(), f["cluster"].to_numpy())
            row.update({"clusters": k, "pnl_per_fill_c": round(mu * 100, 2), "t": None if tt is None else round(tt, 2),
                        "pnl_per_order_c": round(float(g["pnl"].sum()) / g.height * 100, 3),
                        "fills_per_week": round(f.height / weeks[split], 1),
                        "usd_per_week": round(float(f["pnl"].sum()) / weeks[split], 2),
                        "yes_share": round(float((f["side"] == "yes").mean()), 2),
                        "mean_price": round(float(f["price"].mean()), 3)})
        out.append(row)
    return pl.DataFrame(out, infer_schema_length=None).sort(["split", *(by or [])]) if out else pl.DataFrame()


# ---- fair values -------------------------------------------------------------------------------
def _lg(x: np.ndarray) -> np.ndarray:
    from albert.research.fundamental import logit

    return logit(np.asarray(x, dtype=float))


def _design(df: pl.DataFrame, model: str | None) -> np.ndarray:
    cols = [np.ones(df.height), _lg(df["mid"].to_numpy())]
    if model is not None:
        cols.append(_lg(df[model].to_numpy()))
    return np.column_stack(cols)


def ridge_logit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 60) -> np.ndarray:
    """Logistic regression by Newton steps with a small ridge on the slopes (the intercept is not
    penalised): stable on short rolling windows where a plain fit can separate."""
    b = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], lam)
    pen[0] = 0.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - mu) - pen * b
        H = X.T @ (X * (mu * (1 - mu))[:, None]) + np.diag(pen) + 1e-9 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def static_blend(rows: pl.DataFrame, *, model: str | None = "p", group: str | None = None,
                 fold: str = "fold") -> tuple[np.ndarray, dict[str, Any]]:
    """Dev-fitted blend logit P = a + b·logit(mid) [+ c·logit(model)] per ``group``. Dev rows get
    out-of-fold predictions (leave one ``fold`` out); holdout rows the fit on all dev rows."""
    out = np.full(rows.height, np.nan)
    coefs: dict[str, Any] = {}
    idx = np.arange(rows.height)
    keys = rows[group].to_list() if group else [None] * rows.height
    hold = rows["holdout"].to_numpy()
    folds = rows[fold].to_numpy()
    y = rows["y"].to_numpy().astype(float)
    X = _design(rows, model)
    for key in sorted(set(keys), key=str):
        sel = np.array([k == key for k in keys])
        dev = sel & ~hold
        if dev.sum() < 30:
            continue
        beta = ridge_logit(X[dev], y[dev], lam=0.01)
        coefs[str(key)] = [round(float(b), 3) for b in beta]
        h = sel & hold
        out[h] = 1 / (1 + np.exp(-X[h] @ beta))
        for f in np.unique(folds[dev]):
            tr = dev & (folds != f)
            te = dev & (folds == f)
            bf = ridge_logit(X[tr], y[tr], lam=0.01)
            out[idx[te]] = 1 / (1 + np.exp(-X[te] @ bf))
    return out, coefs


def rolling_blend(rows: pl.DataFrame, *, window: timedelta, model: str | None = "p", group: str | None = None,
                  known: str = "known_ts", min_rows: int = 150, lam: float = 1.0) -> np.ndarray:
    """Drift-robust blend: for decisions on UTC day D, fit on rows of the same ``group`` decided in
    [D - window, D) whose outcome was public (``known``) by D 00:00 UTC; no dev / holdout split is
    needed because only the past is used. NaN until ``min_rows`` such rows exist."""
    out = np.full(rows.height, np.nan)
    day = rows["t"].dt.truncate("1d")
    X = _design(rows, model)
    y = rows["y"].to_numpy().astype(float)
    t = rows["t"].dt.epoch("us").to_numpy()
    kn = rows[known].dt.epoch("us").to_numpy()
    keys = np.array([str(k) for k in (rows[group].to_list() if group else [None] * rows.height)])
    dvals = day.dt.epoch("us").to_numpy()
    w = int(window.total_seconds() * 1e6)
    for d in np.unique(dvals):
        today = dvals == d
        for key in np.unique(keys[today]):
            te = today & (keys == key)
            tr = (keys == key) & (kn <= d) & (t >= d - w) & (t < d)
            if tr.sum() < min_rows:
                continue
            beta = ridge_logit(X[tr], y[tr], lam=lam)
            out[te] = 1 / (1 + np.exp(-X[te] @ beta))
    return out


def spread_bucket(bid: pl.Expr, ask: pl.Expr) -> pl.Expr:
    """Quoted spread at t in cents: <=3, 4-6, 7-15, >15 (a mid inside a >15c book is not a price)."""
    sp = ((ask - bid) * 100).round(0)
    return (pl.when(sp <= 3).then(pl.lit("s1 <=3c")).when(sp <= 6).then(pl.lit("s2 4-6c"))
            .when(sp <= 15).then(pl.lit("s3 7-15c")).otherwise(pl.lit("s4 >15c")))


def blend_by(rows: pl.DataFrame, by: str, model: str = "p") -> pl.DataFrame:
    """The round-1 blend test per segment: logit P = a + b·logit(mid) + c·logit(model), fitted
    separately on dev and on the holdout (October refit), t clustered by ``cluster``."""
    from albert.research.fundamental import fit_logistic

    out = []
    for key in [None, *sorted(rows[by].unique().to_list())]:
        for h in (False, True):
            g = rows.filter((pl.col("holdout") == h) & pl.col(model).is_not_null() & pl.col(model).is_not_nan())
            if key is not None:
                g = g.filter(pl.col(by) == key)
            row = {by: "all" if key is None else key, "split": "holdout refit" if h else "dev", "n": g.height,
                   "clusters": g["cluster"].n_unique() if g.height else 0}
            if g.height >= 40 and 0 < g["y"].mean() < 1:
                beta, t = fit_logistic(_design(g, model), g["y"].to_numpy().astype(float), g["cluster"].to_numpy())
                row.update({"b_mid": round(float(beta[1]), 3), "t_b": round(float(t[1]), 2),
                            "c_model": round(float(beta[2]), 3), "t_c": round(float(t[2]), 2)})
            out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)


def price_band(mid: pl.Expr) -> pl.Expr:
    return (pl.when(mid < 0.1).then(pl.lit("a <10c")).when(mid < 0.3).then(pl.lit("b 10-30c"))
            .when(mid < 0.7).then(pl.lit("c 30-70c")).when(mid < 0.9).then(pl.lit("d 70-90c"))
            .otherwise(pl.lit("e >=90c")))


MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)


def margin_grid(rows: pl.DataFrame, fair: str, weeks: dict[str, float], *, margins=MARGINS,
                min_fills: int = 30, replace: bool = False, **kw: Any) -> tuple[pl.DataFrame, float | None]:
    """Maker summary for each margin; the margin is chosen on dev (best PnL per fill with at least
    ``min_fills`` fills, the harness rule)."""
    tabs, best = [], None
    for m in margins:
        o = orders(rows.filter(pl.col(fair).is_not_null() & pl.col(fair).is_not_nan()), fair, m, **kw)
        if replace:
            o = first_fill_per_side(o)
        t = summarize(o, weeks)
        if not t.height:
            continue
        t = t.with_columns(pl.lit(m).alias("margin"))
        tabs.append(t)
        d = t.filter(pl.col("split") == "dev")
        if d.height and d["fills"][0] >= min_fills and "pnl_per_fill_c" in d.columns:
            v = d["pnl_per_fill_c"][0]
            if v is not None and (best is None or v > best[1]):
                best = (m, v)
    tab = pl.concat(tabs, how="diagonal_relaxed") if tabs else pl.DataFrame()
    return tab.select("margin", *[c for c in tab.columns if c != "margin"]) if tab.height else tab, (
        None if best is None else best[0])


def rain_candle_prints(s, rows: pl.DataFrame) -> pl.DataFrame:
    """Hourly trade-price low / high for every rain market from the round-1 candle requests (all
    cached: ``rain.candle_extremes`` re-reads the harness candles, the batch frame has them)."""
    from albert.research import rain

    batch = rain.load(s, "candles_batch")
    harness = pl.read_parquet(s.data_dir / "cache" / f"fundamental_{rain.NAME}_candles.parquet")
    legs = (rows.select("ticker", "series", "close_time").unique("ticker")
            .filter(pl.col("ticker").is_in(harness["ticker"].unique().implode())))
    markets = pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").filter(
        pl.col("ticker").is_in(legs["ticker"].implode())).select("ticker", "open_time").collect()
    legs = legs.join(markets, on="ticker", how="left")
    ext = rain.candle_extremes(s, legs, {"KXRAIN": timedelta(days=3), "KXRAINWKND": timedelta(days=8)})
    both = pl.concat([ext, batch.select("ticker", "end_ts", "p_low", "p_high")]).unique(["ticker", "end_ts"])
    return both.filter(pl.col("p_low").is_not_null()).select(
        "ticker", pl.col("end_ts").alias("ts"), pl.col("p_low").alias("lo"), pl.col("p_high").alias("hi"))


# ---- study drivers -----------------------------------------------------------------------------
HARNESS_MARGINS = MARGINS
WIDE_MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
COLS = ["margin", "split", "orders", "fills", "fill_rate", "pnl_per_fill_c", "t", "pnl_per_order_c",
        "fills_per_week", "usd_per_week"]


def split_weeks(rows: pl.DataFrame) -> dict[str, float]:
    """Calendar weeks covered by each split (distinct decision dates / 7)."""
    d = rows.group_by("holdout").agg(pl.col("t").dt.date().n_unique().alias("n"))
    return {("holdout" if h else "dev"): n / 7 for h, n in d.iter_rows()}


def chosen(tab: pl.DataFrame, margin: float | None, **label: Any) -> dict[str, Any]:
    """One summary row: dev and holdout at the dev-chosen margin."""
    out: dict[str, Any] = {**label, "margin": margin}
    if margin is None or not tab.height:
        return out
    for split in ("dev", "holdout"):
        r = tab.filter((pl.col("margin") == margin) & (pl.col("split") == split))
        if not r.height:
            continue
        r = r.row(0, named=True)
        pre = "d_" if split == "dev" else "h_"
        out.update({f"{pre}fills": r.get("fills"), f"{pre}pnl_fill_c": r.get("pnl_per_fill_c"), f"{pre}t": r.get("t")})
        if split == "holdout":
            out.update({"h_clusters": r.get("clusters"), "h_fills_wk": r.get("fills_per_week"),
                        "h_usd_wk": r.get("usd_per_week"), "h_pnl_order_c": r.get("pnl_per_order_c")})
    return out


def run_suite(rows: pl.DataFrame, prints: dict[str, pl.DataFrame], tests: dict[str, dict[str, Any]],
              fairs: tuple[str, ...], weeks: dict[str, float], *, log=print) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """For each test (row filter, end column, replace flag, order kwargs), fill source and fair
    value: margin grids (harness and wide) and the dev-chosen rows. Returns the summary table and
    the full grids keyed by "test | source | fair"."""
    summary, grids = [], {}
    for tname, spec in tests.items():
        base = rows.filter(spec["filter"]).with_columns(spec["end"].alias("end"))
        for src, pr in prints.items():
            r = window_extremes(base, pr)
            for fair in spec.get("fairs", fairs):
                ok = r.filter(pl.col(fair).is_not_null() & pl.col(fair).is_not_nan())
                tab, _ = margin_grid(ok, fair, weeks, margins=WIDE_MARGINS, replace=spec.get("replace", False),
                                     **spec.get("kw", {}))
                if not tab.height:
                    continue
                grids[f"{tname} | {src} | {fair}"] = tab
                for grid_name, grid in (("harness<=8c", HARNESS_MARGINS), ("wide<=20c", WIDE_MARGINS)):
                    sub = tab.filter(pl.col("margin").is_in(list(grid)))
                    best = None
                    for r0 in sub.filter(pl.col("split") == "dev").iter_rows(named=True):
                        v = r0.get("pnl_per_fill_c")
                        if r0["fills"] >= 30 and v is not None and (best is None or v > best[1]):
                            best = (r0["margin"], v)
                    summary.append(chosen(sub, None if best is None else best[0], test=tname, fills_from=src,
                                          fair=fair, grid=grid_name))
            log(f"  suite {tname} / {src} done {datetime.now(UTC):%H:%M:%S}")
    return pl.DataFrame(summary, infer_schema_length=None), grids


def segments(rows: pl.DataFrame, prints: pl.DataFrame, end: pl.Expr, fair: str, margin: float,
             weeks: dict[str, float], by: list[str], *, replace: bool = False, **kw: Any) -> pl.DataFrame:
    r = window_extremes(rows.with_columns(end.alias("end")), prints)
    o = orders(r.filter(pl.col(fair).is_not_null() & pl.col(fair).is_not_nan()), fair, margin, **kw)
    if replace:
        o = first_fill_per_side(o)
    t = summarize(o, weeks, by)
    return t.select(*[c for c in ("split", *by, "orders", "fills", "fill_rate", "pnl_per_fill_c", "t",
                                  "pnl_per_order_c", "fills_per_week", "usd_per_week", "yes_share", "mean_price")
                      if c in t.columns])


def score_table(rows: pl.DataFrame, fairs: tuple[str, ...], by: str | None = None) -> pl.DataFrame:
    """Brier of each fair value vs the mid on the rows where both exist."""
    out = []
    for f in fairs:
        d = rows.filter(pl.col(f).is_not_null() & pl.col(f).is_not_nan())
        for key, g in d.group_by(["holdout", *([by] if by else [])]):
            y = g["y"].to_numpy().astype(float)
            out.append({"split": "holdout" if key[0] else "dev", **({by: key[1]} if by else {}), "fair": f,
                        "n": g.height, "brier_fair": round(float(((g[f].to_numpy() - y) ** 2).mean()), 5),
                        "brier_mid": round(float(((g["mid"].to_numpy() - y) ** 2).mean()), 5),
                        "mean_fair": round(float(g[f].mean()), 4), "mean_y": round(float(y.mean()), 4)})
    return pl.DataFrame(out).sort("split", *([by] if by else []), "fair")


def taker_table(rows: pl.DataFrame, fairs: tuple[str, ...]) -> pl.DataFrame:
    """Taker at the ask / bid with the series fee on each fair value, theta chosen on dev."""
    from albert.research import fundamental as fu

    out = []
    for f in fairs:
        d = rows.filter(pl.col(f).is_not_null() & pl.col(f).is_not_nan()).with_columns(pl.col(f).alias("p"))
        best, res = None, {}
        for theta in fu.THETAS:
            for h in (False, True):
                tr = fu.taker_pnl(d.filter(pl.col("holdout") == h).select(
                    "ticker", "cluster", "p", "y", "bid", "ask", "fee_multiplier"), theta)
                if not tr.height:
                    continue
                mu, tt, k = fu.clustered_mean(tr["pnl"].to_numpy(), tr["cluster"].to_numpy())
                res[(theta, h)] = (tr.height, mu, tt)
                if not h and tr.height >= 30 and (best is None or mu > best[1]):
                    best = (theta, mu)
        if best is None:
            continue
        row = {"fair": f, "theta_dev": best[0]}
        for h, pre in ((False, "d_"), (True, "h_")):
            n, mu, tt = res.get((best[0], h), (0, float("nan"), None))
            row.update({f"{pre}trades": n, f"{pre}pnl_c": round(mu * 100, 2), f"{pre}t": None if tt is None else round(tt, 2)})
        out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)


RAIN_FAIRS = ("f_mid", "f_model", "f_blend", "f_midrecal", "f_roll28", "f_roll28mid")
HOURLY_FAIRS = ("f_mid", "f_model", "f_blend", "f_midrecal", "f_roll7", "f_roll7mid")


def prepare_rain(s, *, log=print) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Rain rows with every fair value. Groups: the four KXRAIN slots, and one weekend group.
    Static blends: dev out-of-fold over 5 contiguous blocks of dev days. Rolling: 28 days of rows
    whose markets had closed (+1 h) before the decision day."""
    from albert.research import rain

    q = rain_rows(s, log=log)
    # Split by the event's day (not by t): the d-1 12z decision for Oct 1 is made on Sep 30 but its
    # outcome belongs to the holdout, so it must not enter the dev fit.
    q = q.with_columns(pl.col("event").map_elements(rain.event_day, return_dtype=pl.Date).alias("event_day"))
    q = q.with_columns((pl.col("event_day") >= HOLDOUT_FROM.date()).alias("holdout"))
    q = q.with_columns(pl.when(pl.col("series") == "KXRAIN").then(pl.col("slot")).otherwise(pl.lit("wknd")).alias("grp"),
                       pl.col("t").dt.date().alias("d"), (pl.col("close_time") + timedelta(hours=1)).alias("known_ts"),
                       pl.col("event").alias("cluster"), price_band(pl.col("mid")).alias("band"),
                       spread_bucket(pl.col("bid"), pl.col("ask")).alias("sbucket"),
                       pl.col("t").dt.truncate("1w").dt.date().cast(pl.String).alias("week")).sort("t", "ticker")
    dev_days = sorted(q.filter(~pl.col("holdout"))["d"].unique().to_list())
    blocks = {d: i * 5 // len(dev_days) for i, d in enumerate(dev_days)}
    q = q.with_columns(pl.col("d").replace_strict(blocks, default=-1, return_dtype=pl.Int64).alias("fold"))
    b, cb = static_blend(q, model="p", group="grp")
    m, cm = static_blend(q, model=None, group="grp")
    q = q.with_columns(pl.Series("f_blend", b), pl.Series("f_midrecal", m),
                       pl.Series("f_roll28", rolling_blend(q, window=timedelta(days=28), group="grp", min_rows=100)),
                       pl.Series("f_roll28mid", rolling_blend(q, window=timedelta(days=28), group="grp", model=None,
                                                              min_rows=100)),
                       pl.col("mid").alias("f_mid"), pl.col("p").alias("f_model"))
    return q, {"blend_a_b_c_by_group (dev)": cb, "midrecal_a_b_by_group (dev)": cm}


def prepare_hourly(s, *, log=print) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Hourly rows with every fair value, per lead. Static blends: dev leave-one-UTC-day-out.
    Rolling: 7 days of settled hours (S + 5 min) before the decision day."""
    q = hourly_rows(s, log=log).with_columns(
        (pl.col("city") + pl.col("S").dt.strftime("%Y%m%d")).alias("cluster"),
        pl.col("t").dt.strftime("%Y%m%d").alias("fold"),
        (pl.col("S") + timedelta(minutes=5)).alias("known_ts"),
        price_band(pl.col("mid")).alias("band"), spread_bucket(pl.col("bid"), pl.col("ask")).alias("sbucket"),
        pl.col("t").dt.truncate("1w").dt.date().cast(pl.String).alias("week")).sort("t", "ticker")
    b, cb = static_blend(q, model="p", group="lead")
    m, cm = static_blend(q, model=None, group="lead")
    q = q.with_columns(pl.Series("f_blend", b), pl.Series("f_midrecal", m),
                       pl.Series("f_roll7", rolling_blend(q, window=timedelta(days=7), group="lead")),
                       pl.Series("f_roll7mid", rolling_blend(q, window=timedelta(days=7), group="lead", model=None)),
                       pl.col("mid").alias("f_mid"), pl.col("p").alias("f_model"))
    return q, {"blend_a_b_c_by_lead (dev)": cb, "midrecal_a_b_by_lead (dev)": cm}


NEXT_LEAD = {59: 55, 55: 40, 40: 25, 25: 10, 10: 0}


def rain_requote_end(q: pl.DataFrame, slots: tuple[str, ...]) -> pl.DataFrame:
    """Replace mode across ``slots``: each quote lives until the market's next slot in ``slots``
    (or its own cancel time, whichever is first)."""
    nxt = (q.filter(pl.col("slot").is_in(list(slots))).sort("ticker", "t")
           .with_columns(pl.col("t").shift(-1).over("ticker").alias("t_next")))
    return nxt.with_columns(pl.min_horizontal("end", pl.col("t_next").fill_null(pl.col("end"))).alias("end_rq"))


def _shift_pessimistic(pr: pl.DataFrame) -> pl.DataFrame:
    """Hourly candles stamped at their start (+1 µs): a candle then counts if it merely overlaps
    the resting window, i.e. fills in the hour that contains the cancel time are kept (an upper
    bound on adverse fills). Decision times are on the hour, so the first candle is unchanged."""
    return pr.with_columns(pl.col("ts") - timedelta(minutes=59, seconds=59, microseconds=999999))


def run_rain(s, *, trades: pl.DataFrame | None = None, log=print) -> dict[str, Any]:
    q, meta = prepare_rain(s, log=log)
    rq2 = rain_requote_end(q, ("d 00z", "d 12z")).select("ticker", "t", pl.col("end_rq").alias("end_rq2"))
    rq4 = rain_requote_end(q, ("d-1 12z", "d 00z", "d 12z", "d 18z")).select("ticker", "t", pl.col("end_rq").alias("end_rq4"))
    rqw = rain_requote_end(q, ("fri 12z", "sat 12z", "sun 12z")).select("ticker", "t", pl.col("end_rq").alias("end_rqw"))
    q = q.join(rq2, on=["ticker", "t"], how="left").join(rq4, on=["ticker", "t"], how="left").join(
        rqw, on=["ticker", "t"], how="left")
    cp = rain_candle_prints(s, q)
    prints = {"candles_1h_optimistic": cp, "candles_1h_pessimistic": _shift_pessimistic(cp)}
    daily = pl.col("series") == "KXRAIN"
    weeks = split_weeks(q.filter(daily))
    tests: dict[str, dict[str, Any]] = {
        "KXRAIN d 00z (previous evening) -> observable": {"filter": daily & (pl.col("slot") == "d 00z"), "end": pl.col("end")},
        "KXRAIN d 12z (morning) -> observable": {"filter": daily & (pl.col("slot") == "d 12z"), "end": pl.col("end")},
        "KXRAIN requote d 00z -> d 12z -> observable": {"filter": daily & pl.col("slot").is_in(["d 00z", "d 12z"]),
                                                        "end": pl.col("end_rq2"), "replace": True},
        "KXRAIN requote all four slots": {"filter": daily, "end": pl.col("end_rq4"), "replace": True},
        "KXRAINWKND requote fri/sat/sun 12z": {"filter": pl.col("series") == "KXRAINWKND", "end": pl.col("end_rqw"),
                                               "replace": True},
        "KXRAIN d 00z+12z requote, gate |fair-mid|>=3c": {
            "filter": daily & pl.col("slot").is_in(["d 00z", "d 12z"]), "end": pl.col("end_rq2"), "replace": True,
            "kw": {"gate": 0.03}, "fairs": ("f_blend", "f_roll28")},
        "KXRAIN d 00z+12z requote, gate 3c, favoured side only": {
            "filter": daily & pl.col("slot").is_in(["d 00z", "d 12z"]), "end": pl.col("end_rq2"), "replace": True,
            "kw": {"gate": 0.03, "one_sided": True}, "fairs": ("f_blend", "f_roll28")},
        "KXRAIN d 00z+12z requote, lenient fills (at our price if we improved the bid)": {
            "filter": daily & pl.col("slot").is_in(["d 00z", "d 12z"]), "end": pl.col("end_rq2"), "replace": True,
            "kw": {"lenient": True}, "fairs": ("f_mid", "f_blend", "f_roll28")},
        "KXRAIN d 00z+12z requote, hypothetical maker fee 0.0175": {
            "filter": daily & pl.col("slot").is_in(["d 00z", "d 12z"]), "end": pl.col("end_rq2"), "replace": True,
            "kw": {"maker_fee_type": "quadratic_with_maker_fees"}, "fairs": ("f_blend",)},
    }
    wk_weeks = split_weeks(q.filter(~daily))
    summary, grids = run_suite(q.filter(daily), prints, {k: v for k, v in tests.items() if "WKND" not in k},
                               RAIN_FAIRS, weeks, log=log)
    s2, g2 = run_suite(q.filter(~daily), prints, {k: v for k, v in tests.items() if "WKND" in k}, RAIN_FAIRS,
                       wk_weeks, log=log)
    summary = pl.concat([summary, s2], how="diagonal_relaxed")
    grids.update(g2)
    out: dict[str, Any] = {"rows": q, "meta": meta, "weeks": weeks, "summary": summary, "grids": grids}
    # validation on the markets whose trades were fetched: candles vs exact trades
    if trades is not None and trades.height:
        sub = q.filter(daily & pl.col("slot").is_in(["d 00z", "d 12z"])
                       & pl.col("ticker").is_in(trades["ticker"].unique().implode()))
        src = {"candles_1h_optimistic": cp, "candles_1h_pessimistic": _shift_pessimistic(cp),
               "trades_any_side": trade_prints_any(trades), "trades_taker_side": trade_prints(trades)}
        out["validation"] = pl.concat([
            validate_fills(sub.with_columns(pl.col("end_rq2").alias("end")), src, f, (0.0, 0.03, 0.08, 0.20),
                           weeks, replace=True).with_columns(pl.lit(f).alias("fair"))
            for f in ("f_blend", "f_mid")])
        out["validation_markets"] = sub["ticker"].n_unique()
    # segments of the main test (requote d 00z -> d 12z, blend, harness-grid choice)
    main = summary.filter((pl.col("test") == "KXRAIN requote d 00z -> d 12z -> observable")
                          & (pl.col("fills_from") == "candles_1h_pessimistic") & (pl.col("fair") == "f_blend")
                          & (pl.col("grid") == "harness<=8c"))
    m = main["margin"][0] if main.height and main["margin"][0] is not None else 0.03
    seg_rows = q.filter(daily & pl.col("slot").is_in(["d 00z", "d 12z"]))
    for by in (["code"], ["band"], ["slot"], ["side"], ["week"], ["sbucket"]):
        out[f"seg_{by[0]}"] = segments(seg_rows, _shift_pessimistic(cp), pl.col("end_rq2"), "f_blend", m, weeks, by,
                                       replace=True)
    out["seg_sbucket_mid"] = segments(seg_rows, _shift_pessimistic(cp), pl.col("end_rq2"), "f_mid", m, weeks,
                                      ["sbucket"], replace=True)
    out["blend_by_spread"] = blend_by(q.filter(daily), "sbucket")
    out["blend_by_spread_slot"] = pl.concat([blend_by(q.filter(daily & (pl.col("slot") == sl)), "sbucket")
                                             .with_columns(pl.lit(sl).alias("slot"))
                                             for sl in ("d 00z", "d 12z")], how="diagonal_relaxed")
    out["spread_mix"] = (q.filter(daily).group_by("holdout", "sbucket").agg(pl.len().alias("rows"))
                         .sort("holdout", "sbucket"))
    out["seg_margin"] = m
    fairs_sc = RAIN_FAIRS
    out["scores"] = score_table(q.filter(daily), fairs_sc)
    out["scores_slot"] = score_table(q.filter(daily), ("f_blend", "f_roll28"), by="slot")
    out["scores_wknd"] = score_table(q.filter(~daily), fairs_sc)
    out["taker"] = taker_table(q.filter(daily), fairs_sc)
    out["taker_wknd"] = taker_table(q.filter(~daily), fairs_sc)
    out["calib_mid"] = (q.filter(daily).with_columns(pl.col("mid").cut([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85]).alias("bin"))
                        .group_by("holdout", "bin").agg(pl.len().alias("n"), pl.col("mid").mean().alias("mean_mid"),
                                                       pl.col("y").mean().alias("freq"))
                        .with_columns(pl.col("bin").cast(pl.String)).sort("holdout", "bin"))
    return out


def run_hourly(s, *, trades: pl.DataFrame | None = None, log=print) -> dict[str, Any]:
    q, meta = prepare_hourly(s, log=log)
    q = q.with_columns((pl.col("S") - pl.duration(minutes=pl.col("lead").replace_strict(
        NEXT_LEAD, return_dtype=pl.Int64))).alias("end_rq"))
    cp = candle_prints(hourly_minutes(s))
    weeks = split_weeks(q)
    open_ = pl.col("lead") == OPEN_LEAD
    tests: dict[str, dict[str, Any]] = {
        "open+1 min -> S (static)": {"filter": open_, "end": pl.col("S")},
        "open+1 min -> S-10 (static)": {"filter": open_, "end": pl.col("S") - timedelta(minutes=10)},
        "open+1 min -> S-55 (first 4 minutes)": {"filter": open_, "end": pl.col("S") - timedelta(minutes=55)},
        "requote 59/55/40/25/10 -> S": {"filter": pl.lit(True), "end": pl.col("end_rq"), "replace": True},
        "requote, gate |fair-mid|>=3c": {"filter": pl.lit(True), "end": pl.col("end_rq"), "replace": True,
                                         "kw": {"gate": 0.03}, "fairs": ("f_blend", "f_roll7")},
        "requote, gate 3c, favoured side only": {"filter": pl.lit(True), "end": pl.col("end_rq"), "replace": True,
                                                 "kw": {"gate": 0.03, "one_sided": True}, "fairs": ("f_blend", "f_roll7")},
        "requote, NO bid only (sell YES to spikes)": {"filter": pl.lit(True), "end": pl.col("end_rq"), "replace": True,
                                                      "kw": {"no_only": True}, "fairs": ("f_mid", "f_blend", "f_roll7")},
        "requote, hypothetical maker fee 0.0175": {"filter": pl.lit(True), "end": pl.col("end_rq"), "replace": True,
                                                   "kw": {"maker_fee_type": "quadratic_with_maker_fees"},
                                                   "fairs": ("f_blend", "f_roll7")},
    }
    summary, grids = run_suite(q, {"candles_1m": cp}, tests, HOURLY_FAIRS, weeks, log=log)
    out: dict[str, Any] = {"rows": q, "meta": meta, "weeks": weeks, "summary": summary, "grids": grids}
    for test, filt, end, rep in (("open+1 min -> S (static)", open_, pl.col("S"), False),
                                 ("requote 59/55/40/25/10 -> S", pl.lit(True), pl.col("end_rq"), True)):
        main = summary.filter((pl.col("test") == test) & (pl.col("fair") == "f_blend") & (pl.col("grid") == "wide<=20c"))
        m = main["margin"][0] if main.height and main["margin"][0] is not None else 0.08
        key = "open" if rep is False else "requote"
        out[f"seg_margin_{key}"] = m
        for by in (["city"], ["band"], ["lead"], ["side"], ["week"], ["sbucket"]):
            if by == ["lead"] and not rep:
                continue
            out[f"seg_{key}_{by[0]}"] = segments(q.filter(filt), cp, end, "f_blend", m, weeks, by, replace=rep)
    no = summary.filter((pl.col("test") == "requote, NO bid only (sell YES to spikes)") & (pl.col("fair") == "f_blend")
                        & (pl.col("grid") == "wide<=20c"))
    m_no = no["margin"][0] if no.height and no["margin"][0] is not None else 0.15
    out["seg_margin_noonly"] = m_no
    for by in (["week"], ["band"]):
        out[f"seg_noonly_{by[0]}"] = segments(q, cp, pl.col("end_rq"), "f_blend", m_no, weeks, by, replace=True,
                                              no_only=True)
    if trades is not None and trades.height:
        sub = q.filter(pl.col("ticker").is_in(trades["ticker"].unique().implode())).with_columns(
            pl.col("end_rq").alias("end"))
        src = {"candles_1m": cp, "trades_any_side": trade_prints_any(trades), "trades_taker_side": trade_prints(trades)}
        out["validation"] = validate_fills(sub, src, "f_blend", (0.0, 0.08, 0.15, 0.20), weeks, replace=True)
        out["validation_markets"] = sub["ticker"].n_unique()
    out["blend_by_spread_open"] = blend_by(q.filter(open_), "sbucket")
    out["blend_by_spread_all_leads"] = blend_by(q, "sbucket")
    out["spread_mix"] = q.group_by("holdout", "lead", "sbucket").agg(pl.len().alias("rows")).sort("holdout", "lead", "sbucket")
    for test, filt, end, rep in (("open", open_, pl.col("S"), False), ("requote", pl.lit(True), pl.col("end_rq"), True)):
        for f in ("f_blend", "f_mid"):
            out[f"seg_{test}_sbucket_{f}_m08"] = segments(q.filter(filt), cp, end, f, 0.08, weeks, ["sbucket"],
                                                          replace=rep)
    out["scores"] = score_table(q.filter(open_), HOURLY_FAIRS)
    out["scores_lead"] = score_table(q, ("f_blend", "f_roll7"), by="lead")
    out["taker"] = taker_table(q.filter(open_), HOURLY_FAIRS)
    out["calib_mid"] = (q.filter(open_).with_columns(pl.col("mid").cut([0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]).alias("bin"))
                        .group_by("holdout", "bin").agg(pl.len().alias("n"), pl.col("mid").mean().alias("mean_mid"),
                                                       pl.col("y").mean().alias("freq"))
                        .with_columns(pl.col("bin").cast(pl.String)).sort("holdout", "bin"))
    return out


def holdout_preds(rain: dict[str, Any] | None, hourly: dict[str, Any] | None) -> pl.DataFrame:
    """October rows (ticker, t, p_fair = the static dev-fitted blend, mid) for cross-study comparison,
    with the model probability and the drift-robust rolling blend alongside."""
    parts = []
    if rain is not None:
        r = rain["rows"].filter(pl.col("holdout"))
        parts.append(r.select("ticker", "t", pl.col("f_blend").alias("p_fair"), "mid", pl.col("p").alias("p_model"),
                              pl.col("f_roll28").alias("p_fair_rolling"), "series", pl.col("slot").alias("decision"),
                              "y"))
    if hourly is not None:
        h = hourly["rows"].filter(pl.col("holdout"))
        parts.append(h.select("ticker", "t", pl.col("f_blend").alias("p_fair"), "mid", pl.col("p").alias("p_model"),
                              pl.col("f_roll7").alias("p_fair_rolling"), "series",
                              pl.format("lead{}", pl.col("lead")).alias("decision"), pl.col("y").cast(pl.Int64)))
    return pl.concat(parts, how="vertical_relaxed").with_columns(pl.lit(NAME).alias("study"))


def write_report(s, parts: dict[str, dict[str, Any] | None], extra: dict[str, Any]) -> Path:
    from albert.reports import Report

    rep = Report(s.reports_dir, f"research-fundamental-{NAME}")
    rep.json("Run", extra)
    for name, res in parts.items():
        if res is None:
            continue
        rep.json(f"{name}: fair-value fits and weeks per split", {"weeks": res["weeks"], **res["meta"]})
        rep.table(f"{name}: maker summary at the dev-chosen margin (d_ = dev, h_ = October holdout; pnl per "
                  "filled contract in cents, t clustered; fills and $ per week at one contract)",
                  res["summary"], save_as=f"{name.split()[0].lower()}_summary", max_rows=400)
        if "validation" in res:
            rep.table(f"{name}: fill-source validation on the {res['validation_markets']} markets whose trades "
                      "were fetched (same markets, candles vs exact trades with the taker-side rule)",
                      res["validation"], max_rows=200)
        rep.table(f"{name}: Brier of each fair value vs the mid", res["scores"], max_rows=60)
        for k in ("scores_slot", "scores_lead", "scores_wknd"):
            if k in res:
                rep.table(f"{name}: {k}", res[k], max_rows=80)
        rep.table(f"{name}: taker at ask/bid on each fair value (theta chosen on dev)", res["taker"])
        if "taker_wknd" in res:
            rep.table(f"{name}: KXRAINWKND taker", res["taker_wknd"])
        rep.table(f"{name}: mid calibration", res["calib_mid"], max_rows=40)
        for k, v in res.items():
            if (k.startswith("blend_by_spread") or k == "spread_mix") and isinstance(v, pl.DataFrame):
                rep.table(f"{name}: {k} (blend logit P = a + b·logit(mid) + c·logit(model) per quoted-spread "
                          "bucket at t; dev fit and October refit)" if k != "spread_mix" else f"{name}: {k}",
                          v, max_rows=80)
        for k, v in res.items():
            if k.startswith("seg_") and isinstance(v, pl.DataFrame):
                mg = 0.08 if k.endswith("_m08") else res.get("seg_margin", res.get("seg_margin_" + k.split("_")[1]))
                fair = "f_mid" if "f_mid" in k or k.endswith("_mid") else "f_blend"
                rep.table(f"{name}: segment {k} (fair {fair}, margin {mg})", v, max_rows=120)
        grid = pl.concat([g.with_columns(pl.lit(k).alias("test | fills | fair")) for k, g in res["grids"].items()],
                         how="diagonal_relaxed")
        grid.write_parquet(rep.dir / f"{name.split()[0].lower()}_grids.parquet")
        key = [k for k in res["grids"] if k.startswith(("KXRAIN requote d 00z -> d 12z -> observable | candles_1h_pessimistic",
                                                         "requote 59/55/40/25/10 -> S | candles_1m",
                                                         "open+1 min -> S (static) | candles_1m",
                                                         "KXRAIN*M daily requote at 18z -> next day / observable | candles_1h_pessimistic"))]
        for k in key:
            if k.endswith(("f_blend", "f_mid", "f_roll28", "f_roll7")):
                rep.table(f"{name}: full margin grid, {k}", res["grids"][k].select(
                    [c for c in COLS if c in res["grids"][k].columns]), max_rows=40)
    return rep.save()


def run(s=None, *, with_trades: bool = True, monthly: bool = True, log=print) -> Path:
    """Whole study: rain + weekend, hourly temperature and the monthly ladders; fill sources are
    candles (all markets) and, for validation, the exact trades already cached for a subset (no
    new requests here; :func:`fetch_trades` / :func:`monthly_markets` fetch). Writes the report and
    the October prediction rows."""
    from albert.config import settings

    s = s or settings()
    rain_tr = cached_trades(s, "KXRAIN") if with_trades else None
    temp_tr = cached_trades(s, "KXTEMP") if with_trades else None
    rain = run_rain(s, trades=rain_tr, log=log)
    hourly = run_hourly(s, trades=temp_tr, log=log)
    mon = run_monthly(s, log=log) if monthly and _table(s, "monthly_candles").exists() else None
    preds = holdout_preds(rain, hourly)
    _write(preds, s.data_dir / "cache" / f"fundamental_{NAME}_holdout_preds.parquet")
    extra = {"study": NAME, "holdout_from": HOLDOUT_FROM.isoformat(), "rain_rows": rain["rows"].height,
             "hourly_rows": hourly["rows"].height, "monthly_rows": None if mon is None else mon["rows"].height,
             "holdout_pred_rows": preds.height,
             "rain_trade_markets": None if rain_tr is None else rain_tr["ticker"].n_unique(),
             "hourly_trade_markets": None if temp_tr is None else temp_tr["ticker"].n_unique(),
             "fees": "KXRAIN, KXRAINWKND, KXRAIN*M and the four hourly series are fee_type 'quadratic' (no maker "
                     "fee); a hypothetical 0.0175 maker fee is shown as a variant"}
    return write_report(s, {"Rain (KXRAIN, KXRAINWKND)": rain, "Hourly temperature": hourly,
                            "Monthly rain ladders": mon}, extra)


# ---- fill-source validation ------------------------------------------------------------------------
def trade_prints_any(trades: pl.DataFrame) -> pl.DataFrame:
    """Side-blind prints from exact trades (any trade below a YES bid / above a YES-equivalent ask)."""
    return trades.select("ticker", pl.col("created_time").alias("ts"), pl.col("yes_price").alias("lo"),
                         pl.col("yes_price").alias("hi"))


def validate_fills(rows: pl.DataFrame, sources: dict[str, pl.DataFrame], fair: str, margins: tuple[float, ...],
                   weeks: dict[str, float], *, replace: bool) -> pl.DataFrame:
    """Same orders, different fill sources: fills, PnL per fill and agreement with the first source."""
    from albert.research.fundamental import clustered_mean

    out = []
    for m in margins:
        base = None
        for name, pr in sources.items():
            r = window_extremes(rows, pr)
            o = orders(r.filter(pl.col(fair).is_not_null() & pl.col(fair).is_not_nan()), fair, m)
            if replace:
                o = first_fill_per_side(o)
            o = o.sort("ticker", "t", "side")
            agree = None
            if base is None:
                base = o
            elif o.height == base.height:
                agree = round(float((o["filled"] == base["filled"]).mean()), 4)
            for split in (False, True):
                g = o.filter(pl.col("holdout") == split)
                f = g.filter("filled")
                if not g.height:
                    continue
                mu, tt, _ = clustered_mean(f["pnl"].to_numpy(), f["cluster"].to_numpy()) if f.height else (
                    float("nan"), None, 0)
                out.append({"margin": m, "fills_from": name, "split": "holdout" if split else "dev", "orders": g.height,
                            "fills": f.height, "pnl_per_fill_c": round(mu * 100, 2) if f.height else None,
                            "t": None if tt is None else round(tt, 2), "agree_with_first": agree})
    return pl.DataFrame(out, infer_schema_length=None)


def cached_trades(s, prefix: str) -> pl.DataFrame:
    """Trades rebuilt from the raw trade pages already cached under data/cache/weather_maker/kalshi
    (no requests), keeping only (ticker, window) page chains that reached their last page, so every
    kept window is complete. ``prefix`` selects tickers."""
    from albert.http import read_raw
    from albert.venues import kalshi as k

    chains: dict[tuple, list[dict[str, Any]]] = {}
    for f in sorted(cache_dir(s).joinpath("kalshi").glob("*")):
        rec = read_raw(f)
        par = rec.get("params") or {}
        if "trades" not in rec["url"] or not str(par.get("ticker", "")).startswith(prefix):
            continue
        chains.setdefault((par["ticker"], par.get("min_ts"), par.get("max_ts")), []).append(rec)
    rows: list[dict[str, Any]] = []
    for recs in chains.values():
        firsts = [r for r in recs if not r["params"].get("cursor")]
        cursors = {r["params"].get("cursor") for r in recs}
        nexts = {r["response"].get("cursor") or None for r in recs}
        if firsts and all(c in cursors for c in nexts if c):   # every next page was fetched
            for r in recs:
                rows.extend(k.trade_rows(r["response"]))
    return pl.DataFrame(rows, schema=TRADE_SCHEMA).unique("trade_id") if rows else pl.DataFrame(schema=TRADE_SCHEMA)


# ---- monthly rain ladders (KXRAIN*M): climatology + observed-so-far model, hourly candles --------
MONTHLY_STATION = {"KXRAINAUSM": "AUS", "KXRAINCHIM": "ORD", "KXRAINCLLM": "CLL", "KXRAINCMHM": "CMH",
                   "KXRAINDALM": "DFW", "KXRAINDENM": "DEN", "KXRAINHOUM": "HOU", "KXRAINLAXM": "LAX",
                   "KXRAINLEXM": "LEX", "KXRAINMIAM": "MIA", "KXRAINMKEM": "MKE", "KXRAINNYCM": "NYC",
                   "KXRAINPVDM": "PVD", "KXRAINSEAM": "SEA", "KXRAINSFOM": "SFO"}   # from rules_primary
MONTHLY_FROM, MONTHLY_TO = datetime(2026, 8, 25, tzinfo=UTC), datetime(2026, 11, 6, tzinfo=UTC)
MONTHLY_FAIRS = ("f_mid", "f_model", "f_blend", "f_midrecal")


def monthly_markets(s, *, log=print) -> pl.DataFrame:
    """Every settled monthly ladder market (``GET /markets?series_ticker=..&status=settled``; the
    ingested table misses the October ladders that settled after Nov 1). Cached."""
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    path = _table(s, "monthly_markets")
    if path.exists():
        return pl.read_parquet(path)
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, cache_dir(s), min_interval=0.5, max_retries=8))
    rows = []
    for ser, code in MONTHLY_STATION.items():
        for page in api.series_markets(ser, "settled", min_close=MONTHLY_FROM, max_close=MONTHLY_TO, cache=True):
            for m in page.payload.get("markets", []):
                rows.append({"ticker": m["ticker"], "event": m["event_ticker"], "series": ser, "code": code,
                             "strike": k.num(m.get("floor_strike")), "result": m.get("result"),
                             "open_time": k.parse_ts(m.get("open_time")), "close_time": k.parse_ts(m.get("close_time")),
                             "title": m.get("title"), "yes_sub_title": m.get("yes_sub_title")})
        log(f"  monthly {ser}: {len(rows)}")
    out = pl.DataFrame(rows, infer_schema_length=None).filter(pl.col("result").is_in(["yes", "no"]))
    _write(out, path)
    return out


def monthly_candles(s, markets: pl.DataFrame, *, log=print) -> pl.DataFrame:
    """Hourly candles (quotes and trade-price extremes) over each market's life: one batch request
    per event chunk for post-cutoff markets, per market for historical ones. Cached."""
    from albert.http import CachedClient
    from albert.research.rain import _candle_rows_ext
    from albert.venues import kalshi as k

    path = _table(s, "monthly_candles")
    if path.exists():
        return pl.read_parquet(path)
    api = k.Kalshi(CachedClient("kalshi", s.kalshi_base, cache_dir(s), min_interval=0.5, max_retries=8))
    cutoff = api.cutoff().get("market_settled_ts")
    rows: list[dict[str, Any]] = []
    from albert.research.hourly_temp import _patient   # retries through shared-rate-limit 429s

    for m in markets.filter(pl.col("close_time") < cutoff).iter_rows(named=True) if cutoff else []:
        page = _patient(lambda m=m: api.candles(m["ticker"], m["open_time"], m["close_time"], 60, True), tries=12,
                        wait=10.0)
        rows.extend(_candle_rows_ext(page.payload, m["ticker"]))
    live = markets.filter(pl.col("close_time") >= cutoff) if cutoff else markets
    for (_ev,), g in live.group_by("event"):
        lo, hi = g["open_time"].min(), g["close_time"].max()
        per = max(1, int(9000 // ((hi - lo).total_seconds() / 3600 + 1)))
        ticks = sorted(g["ticker"].to_list())
        for j in range(0, len(ticks), per):
            page = _patient(lambda tk=ticks[j:j + per], lo=lo, hi=hi: api.candles_batch(tk, lo, hi, 60), tries=12,
                            wait=10.0)
            for mk in page.payload.get("markets") or []:
                rows.extend(_candle_rows_ext({"candlesticks": mk.get("candlesticks") or []}, mk["market_ticker"]))
    out = pl.DataFrame(rows, infer_schema_length=None).unique(["ticker", "end_ts"])
    log(f"  monthly candles {out.height} rows, {out['ticker'].n_unique()} markets")
    _write(out, path)
    return out


def asos_hourly_totals(obs: pl.DataFrame) -> pl.DataFrame:
    """Hourly precipitation per station from METAR ``p01i``: a routine report holds the total since
    the previous routine one and a special the running partial, so an interval's total is the max
    over its reports. Intervals are keyed by the station's routine minute (its most common report
    minute). Trace counts as 0. Columns code, bucket, inch, ts_end (last report of the interval)."""
    o = obs.filter(pl.col("p01i").is_not_null()).with_columns(pl.col("ts").dt.minute().alias("mi"))
    rm = o.group_by("code").agg(pl.col("mi").mode().first().alias("rmin"))
    o = o.join(rm, on="code").with_columns(
        (pl.col("ts") - pl.duration(minutes=pl.col("rmin") + 1)).dt.truncate("1h").alias("bucket"))
    return (o.group_by("code", "bucket")
            .agg(pl.col("p01i").max().alias("inch"), pl.col("ts").max().alias("ts_end"))
            .with_columns(pl.when(pl.col("inch") < 0.01).then(0.0).otherwise(pl.col("inch")).alias("inch"))
            .sort("code", "bucket"))


def month_end(d: date) -> date:
    return (d.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)


def climo_remaining(cli: pl.DataFrame, code: str, day: date, frac: float, *,
                    years=range(2016, 2026), shifts=range(-10, 11, 2)) -> np.ndarray:
    """Samples of precipitation over the rest of ``day``'s month (``frac`` of ``day`` itself plus
    every later day of the month) from the station's CLI history: the same calendar dates in each
    of ``years``, shifted by ``shifts`` days to widen the sample. Trace = 0."""
    c = cli.filter((pl.col("code") == code) & pl.col("precip").is_not_null())
    vals = dict(zip(c["day"].to_list(), np.where(c["precip"].to_numpy() < 0.01, 0.0, c["precip"].to_numpy())))
    n_after = (month_end(day) - day).days
    out = []
    for y in years:
        for k in shifts:
            try:
                d0 = day.replace(year=y) + timedelta(days=k)
            except ValueError:
                continue
            v = [vals.get(d0 + timedelta(days=i)) for i in range(n_after + 1)]
            if any(x is None for x in v):
                continue
            out.append(frac * v[0] + sum(v[1:]))
    return np.array(out)


def monthly_rows(s, markets: pl.DataFrame, candles: pl.DataFrame, *, log=print) -> pl.DataFrame:
    """One row per market and day at 18 UTC from listing to close. Observed month-to-date total at
    t: CLI days whose report was issued before t (IEM product time; consecutive from the 1st) plus
    ASOS interval totals for later days up to t - OBS_LAG. Model: P(observed + rest-of-month
    climatology > strike). Rows where the observed total already exceeds the strike are decided and
    dropped. ``end`` = the next day's decision, the first ASOS interval that pushes the running
    total over the strike (+ OBS_LAG), or close."""
    from albert.research import fundamental as fu
    from albert.research import rain

    cli = rain.load(s, "cli").filter(pl.col("day") >= date(2016, 1, 1)).with_columns(
        pl.col("product").str.slice(0, 12).str.strptime(pl.Datetime("us"), "%Y%m%d%H%M", strict=False)
        .dt.replace_time_zone("UTC").alias("issued"))
    hrs = asos_hourly_totals(rain.load(s, "obs"))
    mk = markets.with_columns(pl.col("event").str.extract(r"-(\d{2}[A-Z]{3})$", 1).alias("ym"))
    rows = []
    for m in mk.iter_rows(named=True):
        first = date(2000 + int(m["ym"][:2]), rain._MON[m["ym"][2:]], 1)
        last = month_end(first)
        lo_month, _ = rain.lst_bounds(m["code"], first)
        _, hi_month = rain.lst_bounds(m["code"], last)
        offset = lo_month - datetime(first.year, first.month, first.day, tzinfo=UTC)   # LST offset
        c = cli.filter((pl.col("code") == m["code"]) & (pl.col("day") >= first) & (pl.col("day") <= last))
        h = hrs.filter((pl.col("code") == m["code"]) & (pl.col("bucket") >= lo_month) & (pl.col("bucket") < hi_month))
        over = h.sort("bucket").with_columns(pl.col("inch").cum_sum().alias("cum")).filter(
            pl.col("cum") > m["strike"] + 1e-9)
        yes_seen = (over["ts_end"][0] + OBS_LAG) if over.height else None
        t = datetime(m["open_time"].year, m["open_time"].month, m["open_time"].day, 18, tzinfo=UTC)
        if t < m["open_time"] + timedelta(hours=1):
            t += timedelta(days=1)
        while t < min(m["close_time"], hi_month):
            pub = c.filter(pl.col("issued") <= t).sort("day")
            want = [first + timedelta(days=i) for i in range(pub.height)]
            if pub["day"].to_list() != want:            # not consecutive from the 1st: use ASOS only
                pub = pub.clear()
            obs_cli = float(np.where(pub["precip"].to_numpy() < 0.01, 0.0, pub["precip"].to_numpy()).sum())
            after = rain.lst_bounds(m["code"], first + timedelta(days=pub.height))[0]
            obs_asos = float(h.filter((pl.col("bucket") >= after) & (pl.col("ts_end") <= t - OBS_LAG))["inch"].sum())
            observed = obs_cli + obs_asos
            if t < lo_month:
                today, frac = first, 1.0
            else:
                today = min(last, (t - offset).date())
                dlo, dhi = rain.lst_bounds(m["code"], today)
                frac = float(np.clip((dhi - max(t, dlo)).total_seconds() / 86400, 0, 1))
            rem = climo_remaining(cli, m["code"], today, frac)
            p = (float(((observed + rem) > m["strike"] + 1e-9).sum()) + 0.5) / (len(rem) + 1) if len(rem) >= 20 else None
            rows.append({"ticker": m["ticker"], "event": m["event"], "series": m["series"], "code": m["code"],
                         "strike": m["strike"], "t": t, "observed": observed, "p": p,
                         "decided": observed > m["strike"] + 1e-9 or (yes_seen is not None and yes_seen <= t),
                         "y": int(m["result"] == "yes"), "close_time": m["close_time"],
                         "end": min(t + timedelta(days=1), m["close_time"], hi_month,
                                    yes_seen if yes_seen is not None and yes_seen > t else m["close_time"])})
            t += timedelta(days=1)
    df = pl.DataFrame(rows, infer_schema_length=None)
    q = fu.attach_quotes(df.filter(~pl.col("decided") & pl.col("p").is_not_null()),
                         candles.select("ticker", "end_ts", "yes_bid_close", "yes_ask_close"))
    q = q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("t") >= HOLDOUT_FROM).alias("holdout"),
                       pl.col("event").alias("cluster"), pl.lit("quadratic").alias("fee_type"),
                       pl.lit(1.0).alias("fee_multiplier"))
    q = q.with_columns(price_band(pl.col("mid")).alias("band"))
    log(f"monthly rows {df.height}, live with quotes {q.height}")
    return q


def run_monthly(s, *, log=print) -> dict[str, Any] | None:
    """Monthly ladders: model, dev-fitted blend (dev = rows before October, out-of-fold by
    city-month), daily requote at 18 UTC, fills from hourly candles (both bounds)."""
    # September (dev) and October (holdout) ladders; August ladders are pre-cutoff (one historical
    # request each) and were skipped under the shared rate limit.
    markets = monthly_markets(s, log=log).filter(pl.col("event").str.contains("-26(SEP|OCT)$"))
    if not markets.height:
        return None
    candles = monthly_candles(s, markets, log=log)
    q = monthly_rows(s, markets, candles, log=log).sort("t", "ticker").with_columns(pl.col("event").alias("fold"))
    # dev = September ladders, holdout = October ladders (split by the event's month so no
    # October outcome enters the dev fit through late-September decisions); November is out of scope.
    ym = pl.col("event").str.extract(r"-(\d{2}[A-Z]{3})$", 1)
    q = q.filter(ym.is_in(["26SEP", "26OCT"])).with_columns((ym == "26OCT").alias("holdout"))
    b, cb = static_blend(q, model="p")
    m, cm = static_blend(q, model=None)
    q = q.with_columns(pl.Series("f_blend", b), pl.Series("f_midrecal", m), pl.col("mid").alias("f_mid"),
                       pl.col("p").alias("f_model"), spread_bucket(pl.col("bid"), pl.col("ask")).alias("sbucket"))
    cp = candles.filter(pl.col("p_low").is_not_null()).select(
        "ticker", pl.col("end_ts").alias("ts"), pl.col("p_low").alias("lo"), pl.col("p_high").alias("hi"))
    weeks = split_weeks(q)
    tests = {"KXRAIN*M daily requote at 18z -> next day / observable": {"filter": pl.lit(True), "end": pl.col("end"),
                                                                       "replace": True}}
    summary, grids = run_suite(q, {"candles_1h_optimistic": cp, "candles_1h_pessimistic": _shift_pessimistic(cp)},
                               tests, MONTHLY_FAIRS, weeks, log=log)
    out: dict[str, Any] = {
        "rows": q, "weeks": weeks, "summary": summary, "grids": grids,
        "meta": {"blend_a_b_c (dev)": cb, "midrecal_a_b (dev)": cm,
                 "events_by_split": {("holdout" if h else "dev"): n for h, n in
                                     q.group_by("holdout").agg(pl.col("event").n_unique()).iter_rows()}},
        "scores": score_table(q, MONTHLY_FAIRS), "taker": taker_table(q, MONTHLY_FAIRS)}
    main = summary.filter((pl.col("fair") == "f_blend") & (pl.col("grid") == "harness<=8c")
                          & (pl.col("fills_from") == "candles_1h_pessimistic"))
    mm = main["margin"][0] if main.height and main["margin"][0] is not None else 0.03
    out["seg_margin"] = mm
    for by in (["code"], ["band"], ["sbucket"]):
        out[f"seg_{by[0]}"] = segments(q, _shift_pessimistic(cp), pl.col("end"), "f_blend", mm, weeks, by, replace=True)
    out["blend_by_spread"] = blend_by(q, "sbucket")
    out["calib_mid"] = (q.with_columns(pl.col("mid").cut([0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]).alias("bin"))
                        .group_by("holdout", "bin").agg(pl.len().alias("n"), pl.col("mid").mean().alias("mean_mid"),
                                                       pl.col("y").mean().alias("freq"))
                        .with_columns(pl.col("bin").cast(pl.String)).sort("holdout", "bin"))
    return out
