"""Technical indicators on recurring non-financial Kalshi series (culture, AI leaderboards,
approval, executive actions, post counts, strait traffic).

Question (user, 2026-11-20): "culture markets like top Netflix shows weekly and other repetitive
markets ... let's add technical indicator analysis." Is there a repeatable, positive, small-stake
price / volume / open-interest / order-flow rule on these families, taker or maker, after Kalshi's
fees? ``ta_calendar`` already showed that textbook signals lose the ~3c taker cost on sports and
weather; this module tests them on the slower recurring families, adds maker execution with an
unconditional-maker control, cross-sectional signals inside multi-outcome events, time-of-day and
publication-day effects, and volume / OI regimes.

Sample (volume-blind): every settled binary market of each event of the listed series. Holdout =
every event whose last leg closes 2026-10-01 or later (to the latest settled). Dev = events closing
2026-05-01 .. 2026-09-30, at most ``DEV_EVENTS_PER_SERIES`` per series chosen by a seeded hash of
the event ticker (a daily series has far more events than the request budget allows). Nothing is
chosen on volume or result. All legs of a sampled event are kept.

Data: hourly candles (bid / ask / trade-price OHLC, volume, open interest) over each market's final
``WINDOW`` before close. Candles already cached by earlier studies (xcat, recurring, behav_culture,
counts) are reused for the sampled tickers only, so their selection never enters the sample; the
rest come from the API (historical endpoint per market before the cutoff, live batch per event
after) through the repo's cached, rate-limited client, written append-only under
``data/cache/ta_recurring/``.

No look-ahead: a regular hourly grid per market (Kalshi omits hours in which nothing changed; the
quote and OI are carried forward, volume is 0 there); indicators at hour t use grid hours <= t; a
decision at t trades the quote of the candle ending at t; candles at or after the market's actual
close are dropped; a leg listed mid-event enters only from its first candle.

    python -m albert.research.ta_recurring hypotheses | universe | fetch | panel | dev | freeze
    python -m albert.research.ta_recurring holdout | benchmarks | l2 | report

Result (2026-11-20): no edge (see ``VERDICT`` / report.md). None of 2,268 dev cells passes Holm;
the frozen primary (Donchian-48 trend, taker, executive actions) earns +0.15c in the holdout; the
plain maker is adversely selected (-1 to -5c per fill) and indicators improve it by only ~1c.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/ta_recurring")
CACHE = Path("data/cache/ta_recurring")
RECURRING = Path("data/research/recurring")
DEV_FROM = datetime(2026, 5, 1, tzinfo=UTC)
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
DEV_EVENTS_PER_SERIES = 60
WINDOW = timedelta(days=21)
REFRESH_FROM = datetime(2026, 11, 1, tzinfo=UTC)
ET = "America/New_York"

FAMILIES: tuple[tuple[str, str], ...] = (
    ("netflix", r"^KXNETFLIX"),
    ("billboard", (r"^(KXTOPSONG|KXTOPALBUM|KXBILLBOARDRUNNERUP(SONG|ALBUM)|KXRANKLISTSONGTOP10|KXALBUMEQUIV"
                   r"|KXPUREALBUMS|KXALBUMSTREAMS|KXALBUMDEBUT|KXRANKLIST1SONG|KXWEEKSNUM1|KXTOP10BBSPOTS)$")),
    ("youtube", r"^KXYT"),
    ("ai", (r"^(KXTOPMODEL|KXLLM1|KXTECHRANKLISTAICODE|KXCHINAAI|KXIMAGEAI|KXVIDEOAI|KXSPEECHAI|KXOPENINTAI"
            r"|KXTOPUSAGEAI)$")),
    ("ai_share", (r"^(KXOPENSHARE|KXGOOGSHARE|KXANTHSHARE|KXDEEPSHARE|KXOPENSOURCESHARE|KXTOKENUSE"
                  r"|KX[A-Z]+V(REQ|SPEND))$")),
    ("rt", r"^(KXRT|KXMC)$"),
    ("approval", (r"^(KXTRUMPAPPROVE|KXAPRPOTUS|KXAPRPOTUSD|KXTRUMPVH|KXVHGCB|KXGENERICBALLOTVOTEHUB"
                  r"|KXVOTEHUBTRUMPUPDOWN)$")),
    ("exec", (r"^(KXEOWEEK|KXMAMDANIEO|KXTRUMPNOMNUM|KXTRUMPENDORSEMENTS|KXTRUMPACT|KXTRUMPACTD|KXTRUMPPHOTO"
              r"|KXJUDGECOUNT|KXPARDONSTRUMP|KXBILLSCOUNT)$")),
    ("truth", r"^(KXTRUTHSOCIAL|KXTRUTHSOCIALD|KXPOTUSTWEETS)$"),
    ("strait", r"^(KXHORMUZWEEKLY|KXHORMUZMAX|KXHORMUZPEAK|KXBABELMANDEBWEEKLY|KXSUEZWEEKLY|KXPANAMAWEEKLY)$"),
    ("tv_other", (r"^(KXSTEAMTOPSELLER|KXART|KXPOKEMON|KXDWTSELIMINATION|KXBIGBROTHERELIMINATION"
                  r"|KXSURVIVORELIMINATION)$")),
)
REUSE_SOURCES = (   # (name, done list, part glob): candles other studies fetched, reused by ticker
    ("xcat", "data/research/xcat/hourly/done.parquet", "data/research/xcat/hourly/part-*.parquet"),
    ("recurring", "data/research/recurring/candles/done.parquet", "data/research/recurring/candles/part-*.parquet"),
    ("behav_hist", "data/research/behav_culture/candles_hist/done.parquet",
     "data/research/behav_culture/candles_hist/part-*.parquet"),
    ("behav_live", "data/research/behav_culture/candles_live/done.parquet",
     "data/research/behav_culture/candles_live/part-*.parquet"),
    ("counts", "data/research/counts/candles/done.parquet", "data/research/counts/candles/part-*.parquet"),
)
UNI_COLS = ["ticker", "event_ticker", "series_ticker", "yes_sub_title", "strike_type", "floor_strike", "cap_strike",
            "open_time", "close_time", "expected_expiration_time", "settlement_ts", "result", "volume"]


# ---- pre-registered hypotheses (fixed 2026-11-20 before any outcome of this sample was opened) ------
HYPOTHESES = """\
Each line: mechanism -> expected sign -> test. "Trend" buys the side the indicator points to, "fade"
the other side. Taker tests buy at the ask (YES) or 1 - bid (NO) of the candle ending at t, ~$3 order,
Kalshi taker fee rounded up per order, held to settlement. Maker tests rest a bid 1c inside the
spread on the chosen side and count a fill only when a later print trades strictly through it.

1. **Slow-crowd trend (textbook TA).** Culture / politics crowds are retail and slower than sports
   books, and information arrives gradually (FlixPatrol daily ranks, Luminate midweek numbers, poll
   releases), so a sustained move continues. -> trend gross > 0, larger more than 24 h before
   close. -> the 17 `ta_calendar` signals x trend/fade by family, hours-to-close and price band;
   `tsmom6 5c` as the benchmark.
2. **Thin-print overreaction.** In thin books one retail sweep moves the quote with no news and it
   reverts (`behav_micro`: small sweeps revert). -> fading a >= 5c 6 h move on <= 0.5x trailing
   volume pays; a move on >= 2x volume with rising open interest continues. -> move6 x volume
   regime x OI change, trend vs fade; 1 h >= 5c jumps faded.
3. **Maker execution of indicator signals.** The taker's ~3c killed every earlier rule; a resting
   bid on the indicated side keeps the signal and earns the spread. -> maker PnL per fill on the
   indicated side > the unconditional maker control in the same family x side x price band x
   hours-to-close cell (the difference is indicator skill; the control level is the plain maker
   premium). -> each signal as a maker order vs a control that posts the same order every 6 h on
   every eligible market-hour.
4. **Fade as maker.** Liquidity providers who lean against a spike earn the reversion plus the
   spread. -> a maker bid against a >= 5c move beats the control. -> as 3, side reversed.
5. **Cross-sectional rotation inside one-winner events.** Attention traders over-buy the leg that
   just rose most and neglect the others; a new leader that overtakes the old one is informed.
   -> (a) the top 24 h riser (price normalised by the event's sum of YES mids) is overpriced: NO on
   it pays, YES on the top faller pays; (b) a leader-to-runner-up rotation continues (buy the new
   leader). -> event-level ranking at 6 h decision points, taker and maker, events with >= 3
   quoted legs.
6. **Overround pressure.** When the sum of YES mids of a one-winner event rises above 1, the legs
   that pushed it up are the overpriced ones. -> NO on the leg that contributed most to a rise in
   the overround pays. -> event overround change x leg contribution.
7. **Overnight vs daytime.** US retail trades in ET daytime; overnight moves (0-8 ET) come on thin
   books and revert; daytime moves are better informed. -> fade pays more on overnight moves,
   trend more on daytime moves. -> 6 h moves by the ET hour of the decision.
8. **Moves after scheduled publication.** Weekly families get public intermediate data on a
   schedule (Netflix Top 10 on Tuesday afternoon ET informs the next week's legs; Billboard reveals
   charts Sunday / Monday; approval averages update on weekdays). A move in the hours after the
   publication is information and continues; moves at other times are noise and revert. -> trend
   after the family's publication window > trend elsewhere. -> moves by weekday x ET window.
9. **Open-interest build-up before close.** In the last 48 h, price up with OI up = new longs
   (informed late money), price up with OI down = short covering. -> trend on OI-up moves, fade on
   OI-down moves. -> 12 h price change x 12 h OI change, last 48 h before close.
10. **Near-close momentum** (the one `ta_calendar` cell that held). -> RSI / tsmom trend 3-12 h
    before close: gross > 0, taker net ~0, maker possibly > 0. -> replication.
11. **Price-band confound.** Fading rises on longshot legs (<= 15c) mostly collects the
    favourite-longshot bias, not indicator skill. -> fade-of-rise positive in the <= 15c band but
    no better than the unconditional NO control there. -> every cell against its control.
12. **Order-book imbalance (recorded L2, 2026-11-17 .. 11-20 only).** Top-of-book imbalance and
    microprice lead the next 15-60 min mid change, by less than the half spread. -> right sign,
    not tradable as taker. -> descriptive regression on the recorded books (too few settlements
    in four days to score a rule).

Selection rule (fixed now): every dev cell is counted; a cell is eligible with >= 40 trades (fills
for maker) and >= 15 events in dev; Holm at 5% one-sided on the event-clustered t of net PnL per $
risked over all eligible dev cells. One primary (at most two secondaries) is frozen in
primary.json before the holdout is scored. If nothing passes Holm, the best eligible dev cell by
event-clustered t is still frozen and scored, and the verdict says it failed the bar.
"""


def family_expr(series: pl.Expr) -> pl.Expr:
    out = pl.lit(None, dtype=pl.String)
    for fam, rx in reversed(FAMILIES):
        out = pl.when(series.str.contains(rx)).then(pl.lit(fam)).otherwise(out)
    return out


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _api(min_interval: float = 0.5):
    from albert.config import Settings
    from albert.research.pm_hazard import _api as hazard_api

    return hazard_api(Settings(), min_interval, nap_cap=15.0, retries=40)


def event_hash(event_ticker: str) -> str:
    return hashlib.sha256(f"ta_recurring|{event_ticker}".encode()).hexdigest()


# ---- universe ------------------------------------------------------------------------------------
def refresh_recent(series: list[str], since: datetime = REFRESH_FROM, *, log=print) -> pl.DataFrame:
    """Settled markets of ``series`` that closed since ``since`` (the recurring study's table stops
    on 2026-11-17)."""
    from albert.research.counts import market_row

    api = _api()
    rows: list[dict[str, Any]] = []
    for s in series:
        try:
            for page in api.series_markets(s, "settled", min_close=since):
                rows.extend(market_row(m, s, "live") for m in page.payload.get("markets") or [])
        except RuntimeError as error:
            log(f"refresh {s}: {str(error)[:160]}")
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None)


def build_universe(*, refresh: bool = True, log=print) -> pl.DataFrame:
    base = pl.read_parquet(RECURRING / "markets.parquet")
    base = base.with_columns(family_expr(pl.col("series_ticker")).alias("family")).filter(
        pl.col("family").is_not_null() & (pl.col("market_type") == "binary"))
    series = sorted(base["series_ticker"].unique().to_list())
    frames = [base.select(UNI_COLS)]
    if refresh:
        new = refresh_recent(series, log=log)
        if new.height:
            new = new.with_columns(pl.col("volume").cast(pl.Float64))
            frames.append(new.select(UNI_COLS))
            log(f"refresh: {new.height} markets settled since {REFRESH_FROM.date()}")
    u = (pl.concat(frames, how="diagonal_relaxed").unique("ticker", keep="last")
         .filter(pl.col("result").is_in(["yes", "no"]) & pl.col("open_time").is_not_null()
                 & pl.col("close_time").is_not_null() & (pl.col("close_time") > pl.col("open_time"))))
    u = u.with_columns(family_expr(pl.col("series_ticker")).alias("family"))
    fees = pl.read_parquet("data/tables/kalshi_series.parquet").select(
        "series_ticker", "category", "frequency", "fee_type", "fee_multiplier")
    u = u.join(fees, on="series_ticker", how="left")
    ev = u.group_by("series_ticker", "event_ticker").agg(pl.col("close_time").max().alias("event_end"),
                                                          pl.len().alias("legs"))
    ev = ev.with_columns(pl.when(pl.col("event_end") >= HOLDOUT_FROM).then(pl.lit("holdout"))
                         .when(pl.col("event_end") >= DEV_FROM).then(pl.lit("dev")).alias("split"),
                         pl.col("event_ticker").map_elements(event_hash, return_dtype=pl.String).alias("h"))
    dev = (ev.filter(pl.col("split") == "dev").sort("h")
           .with_columns(pl.int_range(pl.len()).over("series_ticker").alias("rk"))
           .filter(pl.col("rk") < DEV_EVENTS_PER_SERIES).drop("rk"))
    keep = pl.concat([dev, ev.filter(pl.col("split") == "holdout")])
    u = u.join(keep.select("event_ticker", "split", "event_end", "legs"), on="event_ticker")
    _save(u, OUT / "universe.parquet")
    summ = (u.group_by("family", "split").agg(pl.col("series_ticker").n_unique().alias("series"),
                                              pl.col("event_ticker").n_unique().alias("events"), pl.len().alias("markets"))
            .sort("family", "split"))
    summ.write_csv(OUT / "universe_counts.csv")
    log(str(summ))
    return u


# ---- candles -------------------------------------------------------------------------------------
OHLC_COLS = ["ticker", "end_ts", "period", "p_open", "p_high", "p_low", "p_close", "p_mean", "bid_open", "bid_high",
             "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close", "volume", "oi"]


def candle_plan(u: pl.DataFrame) -> pl.DataFrame:
    return u.select("ticker", "event_ticker", pl.col("settlement_ts").fill_null(pl.col("close_time")),
                    pl.max_horizontal("open_time", pl.col("close_time") - WINDOW).alias("lo"),
                    pl.col("close_time").alias("hi"), pl.lit(60).alias("period"))


def reuse_local(tickers: list[str], *, log=print) -> set[str]:
    """Copy candles other studies already fetched (whole life or final 21 days) for these tickers
    into this study's cache. Returns the tickers copied."""
    import glob

    d = CACHE / "hourly"
    d.mkdir(parents=True, exist_ok=True)
    want = set(tickers)
    got: set[str] = set()
    for name, done_path, parts in REUSE_SOURCES:
        if not Path(done_path).exists():
            continue
        have = set(pl.read_parquet(done_path)["ticker"].to_list()) & (want - got)
        if not have:
            continue
        files = sorted(glob.glob(parts))
        c = (pl.scan_parquet(files).filter(pl.col("ticker").is_in(list(have))).select(OHLC_COLS).collect())
        c = c.filter(pl.col("ticker").is_in(list(have)))
        part = len(list(d.glob("part-*.parquet")))
        c.write_parquet(d / f"part-{part:05d}-{name}.parquet")
        got |= have
        log(f"reuse {name}: {len(have)} markets, {c.height} candles")
    return got


def fetch(*, workers: int = 2, min_interval: float = 0.5, log=print) -> None:
    from albert.research import xcat

    u = pl.read_parquet(OUT / "universe.parquet")
    d = CACHE / "hourly"
    d.mkdir(parents=True, exist_ok=True)
    done_path = d / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"].to_list()) if done_path.exists() else set()
    todo_t = [t for t in u["ticker"].to_list() if t not in done]
    if todo_t:
        done |= reuse_local(todo_t, log=log)
        _save(pl.DataFrame({"ticker": sorted(done)}), done_path)
    plan = candle_plan(u.filter(~pl.col("ticker").is_in(list(done))))
    api = _api(min_interval)
    cutoff = api.cutoff()["market_settled_ts"]
    # live markets go per event into batch requests; historical ones one request each
    jobs = xcat._jobs(plan, cutoff)
    log(f"fetch: {u.height} markets, {len(done)} cached, {plan.height} to fetch in {len(jobs)} requests "
        f"(cutoff {cutoff})", flush=True)
    lock = threading.Lock()
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    failed: list[str] = []
    part = [len(list(d.glob("part-*.parquet")))]
    t0 = time.time()

    def flush() -> None:
        nonlocal buf
        if buf:
            pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).select(OHLC_COLS).write_parquet(d / f"part-{part[0]:05d}.parquet")
            part[0] += 1
        buf = []
        _save(pl.DataFrame({"ticker": sorted(done | set(fin))}), done_path)
        if failed:
            _save(pl.DataFrame({"ticker": sorted(set(failed))}), d / "failed.parquet")

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(lambda j: xcat._run_job(api, j), jobs):
            with lock:
                for tk, got in res:
                    if got is None:
                        failed.append(tk)
                        continue
                    buf.extend(got)
                    fin.append(tk)
                n += 1
                if n % 200 == 0:
                    flush()
                    rate = n / (time.time() - t0)
                    log(f"  {n}/{len(jobs)} requests, {rate:.2f} req/s, eta {(len(jobs) - n) / max(rate, 1e-9) / 60:.0f} min",
                        flush=True)
    flush()
    log(f"fetch done: {len(fin)} fetched, {len(set(failed))} failed", flush=True)


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    parts = sorted((CACHE / "hourly").glob("part-*.parquet"))
    if not parts:
        raise RuntimeError("no candles: run fetch")
    sc = pl.scan_parquet(parts)
    if tickers is not None:
        sc = sc.filter(pl.col("ticker").is_in(tickers))
    return sc.unique(["ticker", "end_ts"], keep="last").sort("ticker", "end_ts").collect()


# ---- panel ---------------------------------------------------------------------------------------
STAKE = 3.0                  # $ per order (taker and maker)
EVENT_CAP = 5.0              # open cost per event in the capped book
BANKROLL = 100.0
MAKER_LIFE_H = 24            # a resting maker order lives this long (or until the market closes)
CONTROL_EVERY_H = 6          # the unconditional maker control posts at UTC hours divisible by this
XS_EVERY_H = 6               # cross-sectional decisions at UTC hours divisible by this
PB_EDGES = ((0.0, 0.15, "<15c"), (0.15, 0.50, "15-50c"), (0.50, 0.85, "50-85c"), (0.85, 1.01, ">=85c"))
HB_ORDER = ("0-3h", "3-12h", "12-48h", ">48h")
# one-winner events: named-title legs (one #1 / runner-up / top model) and mutually exclusive brackets
MULTI_WINNER_SERIES = ("KXRANKLISTSONGTOP10", "KXTOP10BBSPOTS")


def side_band(price: pl.Expr) -> pl.Expr:
    out = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(PB_EDGES):
        out = pl.when(price.is_between(lo, hi, closed="left")).then(pl.lit(name)).otherwise(out)
    return out


def market_meta() -> pl.DataFrame:
    u = pl.read_parquet(OUT / "universe.parquet")
    return u.select("ticker", "event_ticker", "series_ticker", "family", "split", "strike_type", "result", "open_time",
                    "close_time", "expected_expiration_time", "settlement_ts", "fee_type", "fee_multiplier")


def pub_window(fam: pl.Expr, dow: pl.Expr, hour: pl.Expr) -> pl.Expr:
    """Hours (ET) just after a family's scheduled public data release (hypothesis 8). Netflix Top 10:
    Tuesday ~15:00 ET; Billboard: chart reveals Sunday afternoon to Monday; approval averages:
    weekday afternoons."""
    nf = (fam == "netflix") & (((dow == 2) & (hour >= 15)) | ((dow == 3) & (hour < 3)))
    bb = (fam == "billboard") & (((dow == 7) & (hour >= 12)) | (dow == 1))
    ap = (fam == "approval") & (dow <= 5) & hour.is_between(14, 20)
    return nf | bb | ap


def build_panel(c: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    """Hourly grid with quotes, trade-price range, volume, OI and every indicator (hours <= t)."""
    from albert.research.ta_calendar import add_indicators, with_quote_cols

    c = (c.select("ticker", pl.col("end_ts").alias("t"), pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask"),
                  pl.col("p_low").alias("plo"), pl.col("p_high").alias("phi"), pl.col("volume").fill_null(0.0),
                  pl.col("oi"))
         .join(meta.select("ticker", "close_time"), on="ticker").filter(pl.col("t") < pl.col("close_time"))
         .drop("close_time").sort("ticker", "t"))
    g = (c.with_columns(pl.lit(True).alias("real"))
         .upsample("t", every="1h", group_by="ticker", maintain_order=True)
         .with_columns(pl.col("bid", "ask", "oi").forward_fill().over("ticker"), pl.col("volume").fill_null(0.0),
                       pl.col("real").fill_null(False)))
    p = g.join(meta.drop("close_time"), on="ticker").with_columns(
        (pl.col("result") == "yes").cast(pl.Float64).alias("y"),
        ((pl.col("expected_expiration_time") - pl.col("t")).dt.total_minutes() / 60).alias("h_to_exp"),
        (pl.col("split") == "holdout").alias("holdout"),
        pl.col("event_ticker").alias("cluster"))
    # the same window for every market: the final WINDOW before the scheduled expiration (known at t)
    p = p.filter(pl.col("expected_expiration_time").is_not_null() & (pl.col("h_to_exp") > 0)
                 & (pl.col("h_to_exp") <= WINDOW.total_seconds() / 3600))
    from albert.research.momentum_hist import h_bucket
    p = with_quote_cols(p.with_columns(h_bucket(pl.col("h_to_exp")).alias("hb")).sort("ticker", "t"))
    p = add_indicators(p)
    x, T = pl.col("x"), "ticker"
    p = p.with_columns((x - x.shift(1).over(T)).alias("r1"), (x - x.shift(12).over(T)).alias("r12"),
                       (x - x.shift(24).over(T)).alias("r24"), x.shift(24).over(T).alias("x24"),
                       (pl.col("oi") - pl.col("oi").shift(12).over(T)).alias("doi12"))
    return p.with_columns(pub_window(pl.col("family"), pl.col("dow"), pl.col("et_hour")).alias("pub"))


def signals() -> dict[str, tuple[pl.Expr, pl.Expr]]:
    """name -> (up, down) conditions on the panel: the 17 ``ta_calendar`` signals plus this study's."""
    from albert.research.ta_calendar import ta_signals

    sig = ta_signals()
    r1, r6, r12, r24 = pl.col("r1"), pl.col("r6"), pl.col("r12"), pl.col("r24")
    sig["jump1 5c"] = (r1 >= 0.05, r1 <= -0.05)                                         # H2
    sig["tsmom24 10c"] = (r24 >= 0.10, r24 <= -0.10)                                    # H1 slow trend
    hv = (pl.col("v6") > 0) & (pl.col("v6") >= 2 * pl.col("vbase")) & (pl.col("doi6") > 0)
    sig["move6 5c high vol + OI up"] = ((r6 >= 0.05) & hv, (r6 <= -0.05) & hv)          # H2
    near = pl.col("h_to_exp") <= 48
    up_oi, dn_oi = pl.col("doi12") > 0, pl.col("doi12") < 0
    sig["move12 5c OI up, last 48h"] = (near & (r12 >= 0.05) & up_oi, near & (r12 <= -0.05) & up_oi)      # H9
    sig["move12 5c OI down, last 48h"] = (near & (r12 >= 0.05) & dn_oi, near & (r12 <= -0.05) & dn_oi)  # H9
    h = pl.col("et_hour")
    ovn, day = h.is_between(5, 9), h.is_between(11, 21)
    sig["move6 5c overnight (0-8 ET)"] = (ovn & (r6 >= 0.05), ovn & (r6 <= -0.05))       # H7
    sig["move6 5c daytime"] = (day & (r6 >= 0.05), day & (r6 <= -0.05))                 # H7
    pubfam = pl.col("family").is_in(["netflix", "billboard", "approval"])
    sig["move6 5c after publication"] = (pl.col("pub") & (r6 >= 0.05), pl.col("pub") & (r6 <= -0.05))   # H8
    sig["move6 5c not after publication"] = (pubfam & ~pl.col("pub") & (r6 >= 0.05),
                                             pubfam & ~pl.col("pub") & (r6 <= -0.05))                      # H8
    return sig


def entry_ok(p: pl.DataFrame) -> pl.DataFrame:
    return p.filter(pl.col("smid").is_not_null() & pl.col("mid").is_between(0.05, 0.95)
                    & (pl.col("spread") > 0) & (pl.col("spread") <= 0.10))


KEEP = ["ticker", "event_ticker", "series_ticker", "family", "cluster", "hb", "t", "et_date", "et_hour", "holdout",
        "bid", "ask", "mid", "spread", "y", "settlement_ts", "fee_type", "fee_multiplier", "h_to_exp"]


def taker(f: pl.DataFrame, stake: float = STAKE) -> pl.DataFrame:
    """Rows with ``dir`` +1 (buy YES at the ask) / -1 (buy NO at 1 - bid): a ~$stake order held to
    settlement, taker fee rounded up per order."""
    from albert.research.ta_calendar import taker_fee_expr

    f = f.with_columns(pl.when(pl.col("dir") == 1).then(pl.col("ask")).otherwise(1 - pl.col("bid")).alias("price"),
                       pl.when(pl.col("dir") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"),
                       pl.when(pl.col("dir") == 1).then(pl.col("mid")).otherwise(1 - pl.col("mid")).alias("mid_side"))
    f = f.with_columns(pl.max_horizontal(pl.lit(1.0), (stake / pl.col("price")).floor()).alias("n"))
    f = f.with_columns(taker_fee_expr(pl.col("n"), pl.col("price")).alias("fee_order"))
    return f.with_columns((pl.col("fee_order") / pl.col("n")).alias("fee_c"),
                          (pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n")).alias("pnl"),
                          (pl.col("price") + pl.col("fee_order") / pl.col("n")).alias("risk"),
                          (pl.col("payoff") - pl.col("mid_side")).alias("gross"),
                          pl.lit("taker").alias("exec"), pl.col("t").alias("t_fill"),
                          side_band(pl.col("price")).alias("pb"))


def maker_fee_order(n: pl.Expr, price: pl.Expr) -> pl.Expr:
    """Kalshi maker fee for one order (series with maker fees only), rounded up to the cent."""
    on = pl.col("fee_type").fill_null("").str.contains("maker_fee")
    raw = 0.0175 * pl.col("fee_multiplier").fill_null(1.0) * n * price * (1 - price)
    return pl.when(on).then((raw * 100).round(9).ceil() / 100).otherwise(0.0)


def maker(orders: pl.DataFrame, panel: pl.DataFrame, stake: float = STAKE, life_h: int = MAKER_LIFE_H) -> pl.DataFrame:
    """Rest one bid 1c inside the spread on side ``dir`` from t for ``life_h`` hours (or until the
    market closes). It fills only if a later hourly candle printed strictly THROUGH it (a YES trade
    below our YES bid, or above our YES-equivalent offer for a NO bid) with at least our size in
    that hour's volume. Filled orders are held to settlement. Returns filled orders only, plus a
    ``posted`` count per row in ``orders`` via the ``filled`` flag."""
    inside = pl.col("spread") >= 0.02
    o = orders.with_columns(
        pl.when(pl.col("dir") == 1).then(pl.when(inside).then(pl.col("bid") + 0.01).otherwise(pl.col("bid")))
        .otherwise(pl.when(inside).then(1 - pl.col("ask") + 0.01).otherwise(1 - pl.col("ask"))).round(2).alias("price"))
    o = o.filter(pl.col("price") >= 0.01).with_columns(
        pl.max_horizontal(pl.lit(1.0), (stake / pl.col("price")).floor()).alias("n"),
        pl.when(pl.col("dir") == 1).then(pl.col("price")).otherwise(1 - pl.col("price")).alias("lvl"),
        pl.int_range(pl.len()).alias("oid"))
    later = panel.select("ticker", pl.col("t").alias("tf"), "plo", "phi", pl.col("volume").alias("vol_f"))
    e = (o.select("oid", "ticker", "t", "dir", "lvl", "n")
         .with_columns(pl.int_ranges(1, life_h + 1).alias("k")).explode("k")
         .with_columns((pl.col("t") + pl.duration(hours=1) * pl.col("k")).alias("tf"))
         .join(later, on=["ticker", "tf"], how="inner"))
    through = pl.when(pl.col("dir") == 1).then(pl.col("plo") < pl.col("lvl") - 1e-9).otherwise(pl.col("phi") > pl.col("lvl") + 1e-9)
    hit = (e.filter(through & (pl.col("vol_f") >= pl.col("n"))).group_by("oid").agg(pl.col("tf").min().alias("t_fill")))
    o = o.join(hit, on="oid", how="left").with_columns(pl.col("t_fill").is_not_null().alias("filled"))
    o = o.with_columns(pl.when(pl.col("dir") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"),
                       pl.when(pl.col("dir") == 1).then(pl.col("mid")).otherwise(1 - pl.col("mid")).alias("mid_side"))
    o = o.with_columns(maker_fee_order(pl.col("n"), pl.col("price")).alias("fee_order"))
    return o.with_columns((pl.col("fee_order") / pl.col("n")).alias("fee_c"),
                          pl.when(pl.col("filled")).then(pl.col("payoff") - pl.col("price") - pl.col("fee_order") / pl.col("n"))
                          .otherwise(None).alias("pnl"),
                          (pl.col("price") + pl.col("fee_order") / pl.col("n")).alias("risk"),
                          (pl.col("payoff") - pl.col("mid_side")).alias("gross"),
                          pl.lit("maker").alias("exec"), side_band(pl.col("price")).alias("pb")).drop("oid", "lvl")


def first_per(f: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return f.sort("t").unique(keys, keep="first", maintain_order=True)


def ts_orders(p: pl.DataFrame) -> pl.DataFrame:
    """First firing of every time-series signal per market, in trend (+) and fade (-) mode."""
    q = entry_ok(p)
    frames = []
    for name, (up, dn) in signals().items():
        d = pl.when(up).then(1).when(dn).then(-1).otherwise(0)
        s = first_per(q.with_columns(d.alias("sig")).filter(pl.col("sig") != 0), ["ticker"])
        for mode, sign in (("trend", 1), ("fade", -1)):
            frames.append(s.select(KEEP + [(pl.col("sig") * sign).alias("dir")])
                          .with_columns(pl.lit(name).alias("signal"), pl.lit(mode).alias("mode")))
    return pl.concat(frames) if frames else pl.DataFrame()


def one_winner(p: pl.DataFrame) -> pl.Expr:
    st = pl.col("strike_type").fill_null("custom")
    return (st.is_in(["custom", "structured", "between"]) & ~pl.col("series_ticker").is_in(MULTI_WINNER_SERIES))


def xs_orders(p: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional signals inside one-winner events at UTC hours divisible by XS_EVERY_H."""
    d = p.filter(one_winner(p) & (pl.col("t").dt.hour() % XS_EVERY_H == 0) & pl.col("x").is_not_null())
    ev = ["event_ticker", "t"]
    d = d.with_columns(pl.len().over(ev).alias("legs_q"), pl.col("x").sum().over(ev).alias("S"),
                       pl.col("x24").is_not_null().all().over(ev).alias("all_lag"))
    d = d.filter(pl.col("legs_q") >= 3)
    d = d.with_columns((pl.col("x") / pl.col("S")).alias("q"))
    lag = d.filter(pl.col("all_lag")).with_columns(pl.col("x24").sum().over(ev).alias("S24"))
    lag = lag.with_columns((pl.col("q") - pl.col("x24") / pl.col("S24")).alias("dq"), (pl.col("x") - pl.col("x24")).alias("dx"),
                           (pl.col("S") - pl.col("S24")).alias("dS"))
    lag = lag.with_columns(pl.col("dq").rank("ordinal", descending=True).over(ev).alias("rk_up"),
                           pl.col("dq").rank("ordinal").over(ev).alias("rk_dn"),
                           pl.col("x").rank("ordinal", descending=True).over(ev).alias("rk_now"),
                           pl.col("x24").rank("ordinal", descending=True).over(ev).alias("rk_then"),
                           pl.col("dx").rank("ordinal", descending=True).over(ev).alias("rk_dx"))
    sigs = {
        "xs top riser (24h share)": (pl.col("rk_up") == 1) & (pl.col("dq") >= 0.03),        # +1 = the leg rose
        "xs top faller (24h share)": (pl.col("rk_dn") == 1) & (pl.col("dq") <= -0.03),
        "xs new leader": (pl.col("rk_now") == 1) & (pl.col("rk_then") > 1) & (pl.col("x") >= 0.2),
        "xs overround riser": (pl.col("rk_dx") == 1) & (pl.col("dS") >= 0.05) & (pl.col("S") > 1.0) & (pl.col("dx") > 0),
    }
    # direction of the indicator: riser / new leader / overround riser point UP, the faller points DOWN
    point = {"xs top riser (24h share)": 1, "xs top faller (24h share)": -1, "xs new leader": 1, "xs overround riser": 1}
    q = entry_ok(lag)
    frames = []
    for name, cond in sigs.items():
        s = first_per(q.filter(cond), ["ticker"])
        for mode, sign in (("trend", 1), ("fade", -1)):
            frames.append(s.select(KEEP + [pl.lit(point[name] * sign).alias("dir")])
                          .with_columns(pl.lit(name).alias("signal"), pl.lit(mode).alias("mode")))
    return pl.concat(frames) if frames else pl.DataFrame()


def control_orders(p: pl.DataFrame) -> pl.DataFrame:
    """The unconditional maker control: both sides at every eligible market-hour on a 6 h clock."""
    q = entry_ok(p).filter(pl.col("t").dt.hour() % CONTROL_EVERY_H == 0)
    return pl.concat([q.select(KEEP + [pl.lit(s).alias("dir")]) for s in (1, -1)]).with_columns(
        pl.lit("control").alias("signal"), pl.lit("all").alias("mode"))


def run_chunk(c: pl.DataFrame, meta: pl.DataFrame) -> dict[str, pl.DataFrame]:
    p = build_panel(c, meta)
    if p.is_empty():
        return {}
    o = pl.concat([x for x in (ts_orders(p), xs_orders(p)) if x.height], how="diagonal_relaxed")
    small = p.select("ticker", "t", "plo", "phi", "volume")
    out = {"taker": taker(o), "maker": maker(o, small), "control": maker(control_orders(p), small)}
    stats = p.group_by("ticker").agg(pl.len().alias("grid_h"), pl.col("real").sum().alias("real_h"),
                                     pl.col("smid").is_not_null().sum().alias("quoted_h"),
                                     (pl.col("spread") <= 0.10).sum().alias("tight_h"), pl.col("volume").sum().alias("vol"))
    out["stats"] = stats
    return out


def run_panel(chunk_markets: int = 1500, log=print) -> None:
    """Build the panel in chunks of whole events and write every order (signals x modes x exec)
    and the control to ``OUT/orders_*.parquet``."""
    meta = market_meta()
    evs = meta.select("event_ticker").unique().sort("event_ticker")["event_ticker"].to_list()
    sizes = dict(meta.group_by("event_ticker").len().iter_rows())
    chunks, cur, k = [], [], 0
    for e in evs:
        cur.append(e)
        k += sizes[e]
        if k >= chunk_markets:
            chunks.append(cur)
            cur, k = [], 0
    if cur:
        chunks.append(cur)
    acc: dict[str, list[pl.DataFrame]] = {"taker": [], "maker": [], "control": [], "stats": []}
    for i, ch in enumerate(chunks):
        m = meta.filter(pl.col("event_ticker").is_in(ch))
        c = load_candles(m["ticker"].to_list())
        res = run_chunk(c, m)
        for key, df in res.items():
            acc[key].append(df)
        log(f"  chunk {i + 1}/{len(chunks)}: {m.height} markets, {c.height} candles, "
            f"{res['taker'].height if res else 0} signal orders", flush=True)
        del c, res
    for key, frames in acc.items():
        frames = [f for f in frames if f.height]
        if not frames:
            raise RuntimeError(f"no {key} rows: investigate")
        pl.concat(frames, how="diagonal_relaxed").write_parquet(OUT / f"orders_{key}.parquet")
    st = pl.read_parquet(OUT / "orders_stats.parquet").join(meta.select("ticker", "family", "split"), on="ticker", how="right")
    cov = (st.group_by("family", "split").agg(pl.len().alias("markets"), pl.col("grid_h").is_not_null().sum().alias("with_candles"),
                                             (pl.col("quoted_h") > 0).sum().alias("ever_quoted"),
                                             (pl.col("tight_h") > 0).sum().alias("ever_spread_le_10c"),
                                             pl.col("real_h").median().alias("median_real_candle_h"),
                                             pl.col("vol").median().alias("median_volume"))
           .sort("family", "split"))
    cov.write_csv(OUT / "panel_coverage.csv")
    log(str(cov))


# ---- statistics ----------------------------------------------------------------------------------
def _r(v: Any, k: int = 2) -> Any:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), k)


def cluster_t(num: np.ndarray, den: np.ndarray, *groups: np.ndarray) -> tuple[float | None, float | None]:
    """Ratio mean sum(num)/sum(den) and its t with clustering on one group (or two-way on two:
    V = V_a + V_b - V_ab)."""
    if len(num) == 0 or den.sum() <= 0:
        return None, None
    mu = num.sum() / den.sum()
    e = num - mu * den

    def v(keys: np.ndarray) -> float:
        _, inv = np.unique(keys, return_inverse=True)
        s = np.bincount(inv, weights=e)
        k = len(s)
        return float((s ** 2).sum() * k / (k - 1)) if k > 1 else float("nan")

    if len(groups) == 1:
        var = v(groups[0])
    else:
        a, b = groups
        var = v(a) + v(b) - v(np.char.add(np.char.add(a.astype(str), "|"), b.astype(str)))
    se = math.sqrt(var) / den.sum() if var > 0 else float("nan")
    return float(mu), (float(mu / se) if se and math.isfinite(se) and se > 0 else None)


def score(f: pl.DataFrame, control: pl.DataFrame | None = None) -> dict[str, Any]:
    """One cell: per-contract and per-$ net PnL, event-clustered and event x date two-way t.
    ``f`` holds filled trades (``pnl`` not null). For maker cells, ``control`` (filled control
    orders of the same split) gives the cell-matched expected PnL; skill = pnl - matched control."""
    f = f.filter(pl.col("pnl").is_not_null())
    if f.is_empty():
        return {"trades": 0}
    w = f["n"].to_numpy().astype(float)
    pnl = f["pnl"].to_numpy() * w
    risk = f["risk"].to_numpy() * w
    gross = f["gross"].to_numpy() * w
    ev = f["event_ticker"].to_numpy()
    dt = f["t_fill"].dt.convert_time_zone(ET).dt.date().cast(pl.String).to_numpy()
    c, tc = cluster_t(pnl, w, ev)
    d, td = cluster_t(pnl, risk, ev)
    _, td2 = cluster_t(pnl, risk, ev, dt)
    g, tg = cluster_t(gross, w, ev)
    out = {"trades": f.height, "events": len(np.unique(ev)), "dates": len(np.unique(dt)),
           "avg_price": _r(f["price"].mean(), 3), "gross_c": _r(100 * g if g is not None else None), "t_gross": _r(tg),
           "net_c": _r(100 * c if c is not None else None), "t_net": _r(tc),
           "net_per_$": _r(d, 4), "t_$": _r(td), "t_$_2way": _r(td2), "hit": _r(f["payoff"].mean(), 3)}
    if control is not None and control.height:
        keys = ["family", "dir", "pb", "hb"]
        cm = control.filter(pl.col("pnl").is_not_null()).group_by(keys).agg(
            ((pl.col("pnl") * pl.col("n")).sum() / pl.col("n").sum()).alias("ctl_pnl"), pl.len().alias("ctl_n"))
        j = f.join(cm, on=keys, how="left")
        ok = j.filter(pl.col("ctl_n") >= 10)
        if ok.height:
            wk = ok["n"].to_numpy().astype(float)
            sk, tsk = cluster_t((ok["pnl"] - ok["ctl_pnl"]).to_numpy() * wk, wk, ok["event_ticker"].to_numpy())
            out |= {"matched": ok.height, "control_c": _r(100 * float((ok["ctl_pnl"].to_numpy() * wk).sum() / wk.sum())),
                    "skill_c": _r(100 * sk if sk is not None else None), "t_skill": _r(tsk)}
    return out


def slices(f: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    out = [("all", f)]
    out += [(f"family={v}", g) for (v,), g in f.group_by(["family"], maintain_order=True)]
    out += [(f"hb={v}", g) for (v,), g in f.group_by(["hb"], maintain_order=True)]
    out += [(f"pb={v}", g) for (v,), g in f.group_by(["pb"], maintain_order=True) if v is not None]
    return out


def load_orders() -> dict[str, pl.DataFrame]:
    return {k: pl.read_parquet(OUT / f"orders_{k}.parquet") for k in ("taker", "maker", "control")}


def cell_table(split: str, o: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Every (signal, mode, exec, slice) cell of one split, with fill rates for maker cells."""
    hold = split == "holdout"
    ctl = o["control"].filter(pl.col("holdout") == hold)
    rows = []
    for ex in ("taker", "maker"):
        f = o[ex].filter(pl.col("holdout") == hold)
        for (sig, mode), g in f.group_by(["signal", "mode"], maintain_order=True):
            for name, s in slices(g):
                r = {"signal": sig, "mode": mode, "exec": ex, "slice": name, "split": split, "orders": s.height}
                if ex == "maker":
                    r["fill_rate"] = _r(s["filled"].mean(), 3)
                    r |= score(s, ctl)
                else:
                    r |= score(s)
                rows.append(r)
    # the control itself, as a strategy (plain maker premium)
    for name, s in slices(ctl):
        for side, g in s.group_by(["dir"], maintain_order=True):
            rows.append({"signal": "control (unconditional maker)", "mode": "YES" if side[0] == 1 else "NO", "exec": "maker",
                         "slice": name, "split": split, "orders": g.height, "fill_rate": _r(g["filled"].mean(), 3),
                         **score(g)})
    return pl.DataFrame(rows, infer_schema_length=None)


def holm(t: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down on one-sided p-values (normal approximation). Returns a pass mask."""
    from math import erf, sqrt

    p = np.array([0.5 * (1 - erf(x / sqrt(2))) for x in t])
    order = np.argsort(p)
    m = len(p)
    ok = np.zeros(m, dtype=bool)
    for i, j in enumerate(order):
        if p[j] <= alpha / (m - i):
            ok[j] = True
        else:
            break
    return ok


MIN_TRADES, MIN_EVENTS = 40, 15


def run_dev(log=print) -> pl.DataFrame:
    o = load_orders()
    tab = cell_table("dev", o)
    elig = tab.filter((pl.col("trades") >= MIN_TRADES) & (pl.col("events") >= MIN_EVENTS) & pl.col("t_$").is_not_null()
                      & (pl.col("signal") != "control (unconditional maker)"))
    tab = tab.join(elig.select("signal", "mode", "exec", "slice").with_columns(pl.lit(True).alias("eligible")),
                   on=["signal", "mode", "exec", "slice"], how="left").with_columns(pl.col("eligible").fill_null(False))
    t = elig["t_$"].to_numpy().astype(float)
    passed = holm(t)
    tab = tab.join(elig.select("signal", "mode", "exec", "slice").with_columns(pl.Series("holm_pass", passed)),
                   on=["signal", "mode", "exec", "slice"], how="left").with_columns(pl.col("holm_pass").fill_null(False))
    tab.write_csv(OUT / "dev_cells.csv")
    summary = {"cells_total": tab.filter(pl.col("signal") != "control (unconditional maker)").height,
               "cells_eligible": elig.height, "holm_pass": int(passed.sum()),
               "holm_bar_t_first": _r(float(-np.sqrt(2) * _erfinv_upper(0.05 / max(elig.height, 1))), 2)}
    (OUT / "dev_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))
    return tab


def _erfinv_upper(p: float) -> float:
    """z with P(Z > z) = p, returned as -z/sqrt(2) helper input (keeps scipy out)."""
    lo, hi = 0.0, 40.0
    from math import erf, sqrt
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 - erf(mid / sqrt(2))) > p:
            lo = mid
        else:
            hi = mid
    return -lo / math.sqrt(2)


# ---- frozen rules, holdout, ledgers ---------------------------------------------------------------
def rule_filter(rule: dict[str, Any]) -> pl.Expr:
    e = (pl.col("signal") == rule["signal"]) & (pl.col("mode") == rule["mode"])
    sl = rule.get("slice", "all")
    if sl != "all":
        k, v = sl.split("=", 1)
        e = e & (pl.col(k) == v)
    return e


def rule_trades(rule: dict[str, Any], o: dict[str, pl.DataFrame]) -> pl.DataFrame:
    f = o[rule["exec"]].filter(rule_filter(rule))
    return f.filter(pl.col("pnl").is_not_null())


def marks_frame(tickers: list[str]) -> pl.DataFrame:
    c = load_candles(tickers)
    two = (pl.col("bid_close") > 0) & (pl.col("ask_close") < 1) & (pl.col("ask_close") > pl.col("bid_close"))
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.select("ticker", pl.col("end_ts").alias("ts"), mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .sort("ticker", "ts"))


def ledger(f: pl.DataFrame, name: str, split: str) -> pl.DataFrame:
    from albert.research.ta_calendar import capped_ledger

    g = f.with_columns(pl.col("t_fill").alias("t"))
    return capped_ledger(g, f"ta_recurring/{name}", name, split, bankroll=BANKROLL, event_cap=EVENT_CAP)


def ledger_stats(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research.vol_metrics import ledger_metrics

    if led.is_empty():
        return {"positions": 0}
    m = ledger_metrics(led, marks_frame(led["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "sortino", "max_dd_usd", "min_cum_pnl_usd",
            "weeks_positive_pct", "skew", "sharpe_settle_basis", "first_day", "last_day", "marked_share")
    return {k: m.get(k) for k in keep}


def weekly_positive(f: pl.DataFrame) -> float | None:
    """Share of ISO weeks (by fill date) with positive summed trade PnL (settlement basis)."""
    if f.is_empty():
        return None
    w = f.group_by(pl.col("t_fill").dt.convert_time_zone(ET).dt.truncate("1w").alias("wk")).agg(
        (pl.col("pnl") * pl.col("n")).sum().alias("p"))
    return _r(100 * float((w["p"] > 0).mean()), 0)


def evaluate_rule(rule: dict[str, Any], o: dict[str, pl.DataFrame], split: str) -> dict[str, Any]:
    f = rule_trades(rule, o)
    f = f.filter(pl.col("holdout") == (split == "holdout")) if split in ("dev", "holdout") else f
    ctl = o["control"]
    if split in ("dev", "holdout"):
        ctl = ctl.filter(pl.col("holdout") == (split == "holdout"))
    res = {"split": split, **score(f, ctl if rule["exec"] == "maker" else None),
           "weeks_positive_pct_trades": weekly_positive(f)}
    led = ledger(f, rule["name"], split)
    res["ledger"] = ledger_stats(led)
    return res, led


def freeze(rules: list[dict[str, Any]]) -> None:
    path = OUT / "primary.json"
    if path.exists():
        raise RuntimeError("primary.json exists: the rule is frozen; never refreeze after the holdout")
    payload = {"frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"), "holdout_from": HOLDOUT_FROM.isoformat(),
               "stake_usd": STAKE, "event_cap_usd": EVENT_CAP, "bankroll_usd": BANKROLL, "maker_life_h": MAKER_LIFE_H,
               "rules": rules}
    path.write_text(json.dumps(payload, indent=2))


def choose_and_freeze(log=print) -> list[dict[str, Any]]:
    """Mechanical choice on dev (fixed before the screen was read): the primary is the Holm-passing
    cell with the highest event-clustered t, else the best eligible cell by that t. Secondaries (at
    most two, if different): the best eligible maker cell whose skill over the matched unconditional
    maker control has t >= 2, and the best pooled (slice = all) eligible cell."""
    cells = pl.read_csv(OUT / "dev_cells.csv", infer_schema_length=None).filter("eligible")
    cells = cells.filter(pl.col("signal") != "control (unconditional maker)").sort("t_$", descending=True)
    passed = cells.filter("holm_pass")
    prim = (passed if passed.height else cells).row(0, named=True)
    picks = [("primary", prim)]
    key = lambda r: (r["signal"], r["mode"], r["exec"], r["slice"])
    mk = cells.filter((pl.col("exec") == "maker") & (pl.col("t_skill") >= 2)).sort("t_$", descending=True)
    pooled = cells.filter(pl.col("slice") == "all")
    for role, cand in (("secondary: maker skill over control", mk), ("secondary: best pooled cell", pooled)):
        for r in cand.iter_rows(named=True):
            if all(key(r) != key(x) for _, x in picks):
                picks.append((role, r))
                break
    rules = []
    for role, r in picks:
        rules.append({"name": f"{r['signal']} | {r['mode']} | {r['exec']} | {r['slice']}", "role": role,
                      "signal": r["signal"], "mode": r["mode"], "exec": r["exec"], "slice": r["slice"],
                      "dev": {k: r.get(k) for k in ("trades", "events", "net_c", "net_per_$", "t_$", "t_$_2way",
                                                     "skill_c", "t_skill", "holm_pass")}})
    freeze(rules)
    log(json.dumps(rules, indent=1))
    return rules


def run_holdout(log=print) -> dict[str, Any]:
    fr = json.loads((OUT / "primary.json").read_text())
    o = load_orders()
    out = {"frozen": fr, "results": []}
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    for i, rule in enumerate(fr["rules"]):
        for split in ("dev", "holdout", "all"):
            res, led = evaluate_rule(rule, o, split)
            out["results"].append({"rule": rule["name"], "role": rule["role"], **res})
            if split == "all" and led.height:
                led.write_parquet(OUT / "ledgers" / f"ledger_{i:02d}.parquet")
            if split == "holdout" and led.height:
                led.write_parquet(OUT / "ledgers" / f"holdout_{i:02d}.parquet")
            log(f"{rule['name']} [{split}] net {res.get('net_c')}c/ct, {res.get('net_per_$')}/$, t {res.get('t_$')}, "
                f"Sharpe {res['ledger'].get('sharpe')}", flush=True)
    hold = cell_table("holdout", o)
    hold.write_csv(OUT / "holdout_cells.csv")
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# ---- hypothesis 12: order-book imbalance on the recorded L2 books ---------------------------------
STREAM = Path("data/raw/stream/kalshi")
L2_PREFIXES = ("KXNETFLIX", "KXTOPSONG", "KXTOPALBUM", "KXBILLBOARD", "KXRANKLIST", "KXALBUM", "KXPUREALBUMS", "KXYT",
               "KXTOPMODEL", "KXLLM1", "KXTECHRANKLIST", "KXCHINAAI", "KXIMAGEAI", "KXVIDEOAI", "KXSPEECHAI",
               "KXOPENINTAI", "KXTOPUSAGEAI", "SHARE", "VREQ", "VSPEND", "KXTOKENUSE", "KXRT-", "KXMC-",
               "KXTRUMPAPPROVE", "KXAPRPOTUS", "KXTRUMPVH", "KXVHGCB", "KXGENERICBALLOT", "KXEOWEEK", "KXMAMDANIEO",
               "KXTRUMPNOMNUM", "KXTRUMPENDORSE", "KXTRUMPACT", "KXTRUMPPHOTO", "KXTRUTHSOCIAL", "KXHORMUZ",
               "KXBABELMANDEB", "KXSUEZ", "KXPANAMA", "KXSTEAMTOPSELLER", "KXART-", "KXDWTS")


def _grep_rows(path: Path, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Lines of one closed stream file that mention any pattern (zstd + grep in C, then JSON)."""
    import subprocess

    args = ["bash", "-c", "zstdcat -q \"$0\" | grep -F " + " ".join(f"-e '{p}'" for p in patterns), str(path)]
    out = subprocess.run(args, capture_output=True, text=True, check=False).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def l2_events(days: list[str], *, log=print):
    """Recorded Kalshi events (main recorder, ``k*`` files) for this study's families, merged by
    receive time hour by hour; trade-explained deltas are labelled as in ``backtest.l2``."""
    import heapq as hq
    from collections import defaultdict

    from albert.backtest.l2 import _mark_executions, events_from_rows

    by_hour: dict[str, list[Path]] = defaultdict(list)
    for d in days:
        for p in sorted((STREAM / d).glob("*-k*.jsonl.zst")):
            by_hour[f"{d}T{p.name[:2]}"].append(p)
    fam = {}
    for hour in sorted(by_hour):
        streams = []
        for p in by_hour[hour]:
            evs = events_from_rows(_grep_rows(p, L2_PREFIXES))
            keep = []
            for ev in evs:
                s = ev.market.split("-", 1)[0]
                if s not in fam:
                    fam[s] = pl.select(family_expr(pl.lit(s)).alias("f"))["f"][0]
                if fam[s] is not None:
                    keep.append(ev)
            _mark_executions(keep)
            streams.append(keep)
        log(f"  l2 {hour}: {sum(len(s) for s in streams)} events", flush=True)
        yield from hq.merge(*streams, key=lambda e: e.recv_ns)


L2_DAYS = ("2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20")


def run_l2_day(day: str, *, chunk: int = 100, log=print) -> None:
    """One recorded day per process (memory): tapes -> 5-min grid samples and event triggers,
    turned into frames in chunks of markets; written to ``l2_grid_<day>`` / ``l2_trig_<day>``.
    30-min horizons that cross midnight are lost."""
    from albert.research.momentum_l2 import collect, grid_samples, trigger_samples

    fees = {r["series_ticker"]: r for r in pl.read_parquet("data/tables/kalshi_series.parquet").iter_rows(named=True)}
    tapes, end = collect(l2_events([day], log=log))
    keys = sorted(tapes)
    grids, trigs = [], []
    for i in range(0, len(keys), chunk):
        sub = {k: tapes.pop(k) for k in keys[i:i + chunk]}
        g = pl.DataFrame(grid_samples(sub, end, every_s=300.0, fee_info=fees), infer_schema_length=None)
        if g.height:
            grids.append(g.filter(pl.col("spread") <= 0.10))
        t = pl.DataFrame(trigger_samples(sub, end, fee_info=fees), infer_schema_length=None)
        if t.height:
            trigs.append(t.filter(pl.col("spread") <= 0.10))
        del sub
    if grids:
        pl.concat(grids, how="diagonal_relaxed").write_parquet(OUT / f"l2_grid_{day}.parquet")
    if trigs:
        pl.concat(trigs, how="diagonal_relaxed").write_parquet(OUT / f"l2_trig_{day}.parquet")
    log(f"l2 {day}: {len(keys)} markets, {sum(x.height for x in grids)} grid rows", flush=True)


def run_l2(days: tuple[str, ...] = L2_DAYS) -> dict[str, Any]:
    """Each day in its own subprocess (the replay's memory is returned to the OS in between)."""
    import subprocess
    import sys

    for day in days:
        if (OUT / f"l2_grid_{day}.parquet").exists():
            continue
        subprocess.run([sys.executable, "-u", "-m", "albert.research.ta_recurring", "l2day", "--day", day], check=True)
    return l2_tables()


def l2_tables() -> dict[str, Any]:
    """Imbalance / flow / OFI -> signed future mid change, against the half spread (H12)."""
    g = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("l2_grid_*.parquet"))], how="diagonal_relaxed")
    g = g.filter(pl.col("spread") <= 0.10).with_columns(
        pl.col("market").str.split("-").list.first().alias("series"), pl.col("market").str.extract(r"^(.*)-[^-]+$").alias("event"))
    g = g.with_columns(family_expr(pl.col("series")).alias("family"))
    rows = []
    feats = {"imb": 0.5, "flowimb_300": 0.5, "ofi_60": 0.5, "mom_300": 0.02}
    for feat, thr in feats.items():
        for fam_name, sub in [("all", g)] + [(f, s) for (f,), s in g.group_by(["family"], maintain_order=True)]:
            s = sub.filter(pl.col(feat).abs() >= thr).with_columns(pl.col(feat).sign().alias("d"))
            if s.height < 30:
                continue
            r = {"feature": feat, "threshold": thr, "family": fam_name, "samples": s.height, "markets": s["market"].n_unique(),
                 "events": s["event"].n_unique(), "half_spread_c": _r(50 * s["spread"].mean())}
            for h in (60, 300, 1800):
                col = f"mid_{h}s"
                v = s.filter(pl.col(col).is_not_null())
                if v.height < 30:
                    continue
                num = (v[col] * v["d"]).to_numpy()
                mu, t = cluster_t(num, np.ones(len(num)), v["event"].to_numpy())
                r[f"dmid_{h}s_c"] = _r(100 * mu if mu is not None else None, 3)
                r[f"t_{h}s"] = _r(t)
            rows.append(r)
    imb = pl.DataFrame(rows, infer_schema_length=None)
    imb.write_csv(OUT / "l2_features.csv")
    out: dict[str, Any] = {"grid_samples": g.height, "markets": g["market"].n_unique(), "events": g["event"].n_unique(),
                           "first": str(pl.from_epoch(pl.Series([g["t_ns"].min()]), time_unit="ns")[0]),
                           "last": str(pl.from_epoch(pl.Series([g["t_ns"].max()]), time_unit="ns")[0])}
    tfiles = sorted(OUT.glob("l2_trig_*.parquet"))
    if tfiles:
        t = (pl.concat([pl.read_parquet(p) for p in tfiles], how="diagonal_relaxed").filter(pl.col("spread") <= 0.10)
             .with_columns(pl.col("market").str.extract(r"^(.*)-[^-]+$").alias("event")))
        rows = []
        for (kind,), s in t.group_by(["kind"], maintain_order=True):
            r = {"kind": kind, "samples": s.height, "events": s["event"].n_unique(), "half_spread_c": _r(50 * s["spread"].mean())}
            for h in (60, 300, 1800):
                for col in (f"mid_{h}s", f"taker_{h}s"):
                    v = s.filter(pl.col(col).is_not_null())
                    if v.height >= 30:
                        mu, tt = cluster_t(v[col].to_numpy(), np.ones(v.height), v["event"].to_numpy())
                        r[f"{col}_c"], r[f"t_{col}"] = _r(100 * mu if mu is not None else None, 3), _r(tt)
            rows.append(r)
        trg = pl.DataFrame(rows, infer_schema_length=None)
        trg.write_csv(OUT / "l2_triggers.csv")
    (OUT / "l2_summary.json").write_text(json.dumps(out, indent=2))
    return out


BENCHMARKS = (   # reference strategies (not selected on dev): the classic benchmark and the plain maker
    {"name": "benchmark: tsmom6 5c | trend | taker | all", "signal": "tsmom6 5c", "mode": "trend", "exec": "taker",
     "slice": "all"},
    {"name": "benchmark: unconditional maker, YES side, every 6 h", "signal": "control", "mode": "all", "exec": "control",
     "slice": "all", "dir": 1},
    {"name": "benchmark: unconditional maker, NO side, every 6 h", "signal": "control", "mode": "all", "exec": "control",
     "slice": "all", "dir": -1},
)


def run_benchmarks(log=print) -> pl.DataFrame:
    o = load_orders()
    rows = []
    for i, b in enumerate(BENCHMARKS):
        f = o[b["exec"]].filter(pl.col("pnl").is_not_null() & (pl.col("signal") == b["signal"]) & (pl.col("mode") == b["mode"]))
        if "dir" in b:
            f = f.filter(pl.col("dir") == b["dir"])
        for split in ("dev", "holdout", "all"):
            g = f if split == "all" else f.filter(pl.col("holdout") == (split == "holdout"))
            led = ledger(g, b["name"], split)
            if split == "all" and led.height:
                led.write_parquet(OUT / "ledgers" / f"ledger_{10 + i:02d}.parquet")
            st = ledger_stats(led)
            ci = st.get("sharpe_ci90")
            rows.append({"strategy": b["name"], "split": split, **score(g), **{k: v for k, v in st.items() if k != "sharpe_ci90"},
                         "sharpe_ci90": f"[{ci[0]}, {ci[1]}]" if ci else ""})
            log(f"{b['name']} [{split}] Sharpe {st.get('sharpe')} net {rows[-1].get('net_c')}c", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / "benchmarks.csv")
    return df


def export_marks() -> Path:
    """Ledger tickers' hourly candles in the ``vol_metrics.load_marks`` schema, so the shared
    ``vol_metrics run`` can mark these ledgers (its CANDLE_SOURCES would need this path)."""
    tick = pl.concat([pl.read_parquet(p).select("ticker") for p in sorted((OUT / "ledgers").glob("*.parquet"))]).unique()
    c = load_candles(tick["ticker"].to_list())
    path = OUT / "candles_marks.parquet"
    c.select("ticker", "end_ts", pl.col("bid_close").alias("yes_bid_close"), pl.col("ask_close").alias("yes_ask_close"),
             pl.col("p_close").alias("price_close")).write_parquet(path)
    return path


# ---- report --------------------------------------------------------------------------------------
VERDICT = """
**No edge (no_edge).** No technical-indicator rule on the recurring culture / AI / approval /
executive-action / count / strait families pays its costs out of sample, as taker or as maker.

* **Screen.** 2,268 dev cells (30 signals: the 17 `ta_calendar` ones, 9 new time-series and 4
  cross-sectional; x trend/fade x taker/maker x 20 slices; 1,858 with
  >= 40 trades and >= 15 events). None passes Holm (first bar t >= 4.04); 19 eligible cells have
  t >= 2, fewer than the ~42 chance alone gives, because most cells lose (648 have t <= -2).
* **Primary (frozen mechanically: best dev t).** Donchian-48 breakout, trend, taker, executive-
  action family: dev +12.3c per contract, +31% per $ (t 2.7 by event, 3.2 two-way), 94 events,
  ledger Sharpe 4.4 (ex top-5 1.7). Holdout +0.15c, +0.3% per $ (t 0.02), 28 events; ledger
  Sharpe 0.8 [-4.3, 4.5], ex top-5 -3.9, 33% positive weeks, max DD -$23. The executive-action
  trend effect shrank from +9c gross (median over signals) to +2c, below the taker cost.
* **Secondary, maker:** 6 h >= 5c move with falling open interest, maker bid 1c inside on the move's
  side, 15-50c. Dev +6.5c per fill (t 2.2), +8.4c over the matched unconditional maker (t 2.9).
  Holdout +4.4c, +16% per $ (t 1.2 by event, 1.15 two-way), 141 events, +7.3c over the control
  (t 2.0); ledger Sharpe 2.5 [-1.6, 6.3], **ex top-5 0.04**, vol 149%, max DD -$34, $7.2/week
  ($0.09/week without the five best events), 67% positive weeks. It fails the live bar (t < 2,
  Sharpe ex top-5 < 1), its holdout profit is five events, it is the best of ~900 maker cells,
  and its sign is the opposite of hypothesis 9 (OI-down moves were expected to revert). Not a
  paper candidate either; at most a free forward log.
* **Secondary, pooled taker:** 6 h >= 5c move on >= 2x trailing volume, trend, all families: dev
  +2.0c (t 1.9, gross +5.4c), holdout -2.0c (t -1.6); ledger Sharpe 1.3 -> -1.2.
* **Indicator vs the plain maker premium.** The unconditional maker (bid 1c inside, every 6 h,
  both sides) loses: YES side -2.5c / -4.6c per fill dev / holdout (Sharpe -3.0 / -3.6), NO side
  -1.2c / -2.7c (Sharpe -0.5 / -0.3). Trend-side maker signals beat it by +1.6c (dev) and +1.1c
  (holdout) on average (24 / 22 of 30 pooled cells positive), fade-side signals by ~0: momentum
  keeps a resting bid out of the worst fills, but not enough to turn a loss into a gain. Pooled
  maker cells average -1.0c (dev) and -2.5c (holdout) net.
* **Taker, pooled.** Trend cells average +1.8c gross / -1.5c net (dev) and -0.2c / -3.6c
  (holdout); fade cells 0 / -3.0c and -2.2c / -5.5c. Benchmark tsmom6 5c trend: Sharpe 0.3 dev,
  -2.9 holdout.
* **Hypotheses.** H1 (slow-crowd trend): gross trend +1.8c pooled dev, gone in holdout. H2 (thin
  moves revert, high-volume moves continue): right signs in dev gross, not after costs and not in
  holdout. H3/H4 (maker): see above. H5/H6 (cross-sectional rotation / overround): small and
  noisy (pooled |t| < 1.4 in dev; holdout signs flip). H7 (overnight vs daytime): every cell net
  negative; overnight fades lose slightly less (-2.5c vs -3.5c dev, -6.1c vs -6.9c holdout). H8
  (after publication): reversed: post-publication moves revert (trend -5.5c dev / -6.6c holdout vs
  -2.0c / -3.4c at other times), and fading them still loses (-2.0c / -3.4c). H9 (OI build-up):
  reversed sign, see the maker secondary. H10 (near-close RSI trend, 3-12 h): -5.4c dev / -2.6c
  holdout, does not replicate. H11 (fading rises into longshots, NO at >= 85c): taker -4.7c dev /
  -1.8c holdout, maker ~0 over the control: no favourite-longshot premium to harvest this way.
  H12 (L2 imbalance): right sign, about 1/30 of the half spread (below).
* **Compared with `ta_calendar` (sports + weather):** the same picture. There |gross| <= 1.7c
  against ~3.1c of taker cost and ledger Sharpe -4.5 to -6.9; here pooled trend gross is similar
  (+1.8c dev) and family cells reach +9c in dev but do not persist, books are wider and thinner
  (median 37-72 real candle-hours per Billboard / Netflix leg; a third of YouTube legs ever have a
  usable quote), and the maker route does not rescue it because the plain maker is itself
  adversely selected in these books.

Nothing here is worth live or paper capital. The only live edge remains the mentions NO-bid.
"""
FAMILY_NOTES = """
Median over the 26 time-series signals of each family's taker cell, net cents per contract
(trend / fade), dev -> holdout; and the unconditional maker per fill (YES / NO side).

* **netflix** (220 dev / 70 holdout events): trend -0.6 -> -5.5c, fade -2.9 -> -3.6c; maker
  control YES -2.7 / -5.6c, NO -1.5 / +0.3c. Half the legs never have a usable quote.
* **billboard** (273 / 140): trend -4.9 -> -4.7c, fade -6.4 -> -5.4c; median leg volume ~100
  contracts; maker control -1.8 to -5.7c. Nothing.
* **youtube** (371 / 342): trend -1.1 -> -5.1c, fade -5.1 -> -9.7c; a third of daily legs ever
  quoted; maker control -1.3 to -4.4c.
* **ai** leaderboards (117 / 67): trend -1.7 -> -7.4c, fade -2.5 -> -10.4c.
* **ai_share** (73 / 244): trend +0.9 -> -1.3c (gross +4.2 -> +2.6c), fade -9.1 -> -7.5c: the
  steadiest trend family, still below the taker cost; holdout is mostly new markets.
* **rt** (61 / 28): trend -1.7 -> -2.5c, fade -2.5 -> -4.4c.
* **approval** (111 / 108): trend -3.7 -> -3.6c, fade -1.7 -> -4.1c; the most liquid books (maker
  fill rate ~0.7) and the most efficient: nothing.
* **exec** actions (131 / 87): trend +6.2 -> -1.9c (gross +9.1 -> +2.2c), fade +0.3 -> -6.1c;
  the source of the frozen primary; the dev trend effect did not persist.
* **truth** counts (26 / 59): trend +3.1 -> -3.0c, fade +2.3 -> -8.0c; 15 dev events only.
* **strait** traffic (66 / 40): trend -6.1 -> -5.7c, fade -2.4 -> -5.1c.
* **tv_other** (Steam, art, Pokemon, reality-TV eliminations; 86 / 91): trend -1.8 -> -4.3c,
  fade -1.8 -> -5.7c.

L2 (hypothesis 12, 2026-11-17..20, main-recorder books of 436 markets / 141 events, 5-min grid,
spread <= 10c): a top-of-book imbalance >= 0.5 is followed by a +0.02c (5 min) / +0.05c (30 min)
mid move in its direction (t 4.7 / 3.4) against a 1.5c half spread. Quote jumps >= 2c in 30 s
revert 0.7c (t -3 to -5); same-side trade runs and prints >= 250 contracts continue ~+0.9-1.0c
over 1-5 min (t 3-9), but a round-trip taker loses 4.6-8.6c. Right signs, ~1/30 of the cost.
"""

CAVEATS = """
* Maker fills come from hourly candles: an order fills when a later hour's trade-price range goes
  strictly through it and that hour's volume covers our size. Queue position at our own price is
  never credited (conservative on fill count), but the size of the through-print itself is not
  known. Taker entries use the candle-close quote without displayed depth (orders are ~$3).
* Dev is a seeded sample (<= 60 events per series, 2026-05-01..09-30); the holdout is 7 weeks
  (2026-10-01..11-20) and its family mix differs (ai_share 244 holdout vs 73 dev events, the daily
  Truth Social series starts 2026-10-02). Decisions are limited to the final 21 days before the
  scheduled expiration.
* 2,268 dev cells were scored; with that many, single-cell t-stats near 2-3 are expected by
  chance. The L2 study covers four recorded days and is descriptive only.
* Ledger marks use this study's own candle cache (`candles_marks.parquet`); the shared
  `vol_metrics run` does not know that file and would mark these ledgers at the entry price.
"""


def _md(df: pl.DataFrame, cols: list[str] | None = None, n: int | None = None) -> str:
    if cols:
        df = df.select([c for c in cols if c in df.columns])
    if n is not None:
        df = df.head(n)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.iter_rows():
        lines.append("| " + " | ".join("" if v is None else str(v).replace("|", "/") for v in r) + " |")
    return "\n".join(lines)


CELL_COLS = ["signal", "mode", "exec", "slice", "orders", "fill_rate", "trades", "events", "avg_price", "gross_c", "net_c",
             "net_per_$", "t_$", "t_$_2way", "control_c", "skill_c", "t_skill"]


def write_report(verdict: str = VERDICT, family_notes: str = FAMILY_NOTES) -> Path:
    parts = [(f"# Technical indicators on recurring Kalshi series (`ta_recurring`)\n\nGenerated "
             f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m albert.research.ta_recurring report`. "
             "Dev = events closing 2026-05-01 .. 09-30 (at most 60 per series, seeded hash), holdout = every event "
             "closing 2026-10-01 .. latest settled.\n"), "## Verdict\n", verdict.strip() + "\n"]
    hyp = OUT / "hypotheses.md"
    parts += ["## Hypotheses (written before the screen)\n",
              (hyp.read_text().split("\n", 2)[-1] if hyp.exists() else HYPOTHESES).strip() + "\n"]
    uc = pl.read_csv(OUT / "universe_counts.csv")
    parts += ["## Sample (volume-blind)\n",
              ("Every settled binary market of each sampled event; dev events at most 60 per series by a seeded hash "
              "of the event ticker (all events where a series has fewer), holdout every event. Hourly candles over the "
              "final 21 days before the scheduled expiration (reused from earlier studies' caches for the sampled "
              "tickers only, the rest fetched).\n"), _md(uc) + "\n"]
    if (OUT / "panel_coverage.csv").exists():
        parts += ["Panel coverage (markets with a usable quote):\n", _md(pl.read_csv(OUT / "panel_coverage.csv")) + "\n"]
    if (OUT / "dev_summary.json").exists():
        ds = json.loads((OUT / "dev_summary.json").read_text())
        cells = pl.read_csv(OUT / "dev_cells.csv", infer_schema_length=None)
        ind = cells.filter(pl.col("signal") != "control (unconditional maker)")
        parts += ["## Dev screen\n",
                  (f"{ds['cells_total']} cells (signal x mode x taker/maker x slice) were scored on dev; "
                  f"{ds['cells_eligible']} had >= {MIN_TRADES} trades and >= {MIN_EVENTS} events. Holm at 5% "
                  f"(one-sided, event-clustered t of net PnL per $): first bar t >= {ds['holm_bar_t_first']}; "
                  f"{ds['holm_pass']} cells pass.\n"),
                  "Top 15 eligible dev cells by event-clustered t:\n",
                  _md(ind.filter("eligible").sort("t_$", descending=True, nulls_last=True), CELL_COLS, 15) + "\n",
                  "Pooled cells (slice = all), taker:\n",
                  _md(ind.filter((pl.col("slice") == "all") & (pl.col("exec") == "taker")).sort("signal", "mode"),
                      CELL_COLS) + "\n",
                  "Pooled cells (slice = all), maker with the cell-matched unconditional-maker control:\n",
                  _md(ind.filter((pl.col("slice") == "all") & (pl.col("exec") == "maker")).sort("signal", "mode"),
                      CELL_COLS) + "\n",
                  "The unconditional maker control itself (plain maker premium), dev:\n",
                  _md(cells.filter((pl.col("signal") == "control (unconditional maker)")
                                   & (pl.col("slice") == "all") | (pl.col("signal") == "control (unconditional maker)")
                                   & pl.col("slice").str.starts_with("family=")).sort("slice", "mode"),
                      ["mode", "slice", "orders", "fill_rate", "trades", "events", "avg_price", "net_c", "net_per_$", "t_$"])
                  + "\n"]
    if (OUT / "primary.json").exists():
        fr = json.loads((OUT / "primary.json").read_text())
        parts += ["## Frozen rules\n", f"Frozen {fr['frozen_utc']} (before the holdout was scored):\n",
                  "```json\n" + json.dumps(fr["rules"], indent=1) + "\n```\n"]
    if (OUT / "results.json").exists():
        res = json.loads((OUT / "results.json").read_text())["results"]
        rows = []
        for r in res:
            led = r.get("ledger") or {}
            ci = led.get("sharpe_ci90")
            rows.append({"rule": r["rule"], "role": r["role"], "split": r["split"], "fills": r.get("trades"),
                         "events": r.get("events"), "net_c": r.get("net_c"), "net_per_$": r.get("net_per_$"),
                         "t_event": r.get("t_$"), "t_2way": r.get("t_$_2way"), "skill_c": r.get("skill_c"),
                         "wk_pos_%": r.get("weeks_positive_pct_trades"), "sharpe": led.get("sharpe"),
                         "ci90": f"[{ci[0]}, {ci[1]}]" if ci else "", "sharpe_ex5": led.get("sharpe_ex_top5"),
                         "vol_%": led.get("ann_vol_pct"), "max_dd": led.get("max_dd_usd"),
                         "$/wk": led.get("usd_per_week"), "ledger_wk_pos_%": led.get("weeks_positive_pct")})
        parts += ["## Frozen rules: dev, holdout and all (capped $100 book, <= $5 per event, ~$3 orders)\n",
                  ("Sharpe / vol from `vol_metrics.ledger_metrics` (daily MTM at the hourly mid, x sqrt(365)); t by event "
                  "and two-way by event x ET date.\n"), _md(pl.DataFrame(rows, infer_schema_length=None)) + "\n"]
    if (OUT / "benchmarks.csv").exists():
        bm = pl.read_csv(OUT / "benchmarks.csv", infer_schema_length=None)
        parts += ["## Benchmarks (not selected on dev; same book)\n",
                  _md(bm, ["strategy", "split", "trades", "events", "net_c", "net_per_$", "t_$", "positions", "sharpe",
                           "sharpe_ci90", "sharpe_ex_top5", "ann_vol_pct", "max_dd_usd", "usd_per_week",
                           "weeks_positive_pct"]) + "\n"]
    if (OUT / "l2_features.csv").exists():
        parts += ["## Hypothesis 12: recorded L2 books (descriptive)\n"]
        if (OUT / "l2_summary.json").exists():
            parts.append("```json\n" + (OUT / "l2_summary.json").read_text() + "\n```\n")
        parts += ["Signed future mid change after a strong feature (|imbalance| >= 0.5 etc.), cents, t by event:\n",
                  _md(pl.read_csv(OUT / "l2_features.csv")) + "\n"]
        if (OUT / "l2_triggers.csv").exists():
            parts += ["Event triggers (mid move >= 2c in 30 s, same-side trade runs, single prints >= 250):\n",
                      _md(pl.read_csv(OUT / "l2_triggers.csv")) + "\n"]
    parts += ["## Per-family notes\n", family_notes.strip() + "\n", "## Caveats\n", CAVEATS.strip() + "\n",
              "## Files\n", ("Module `src/albert/research/ta_recurring.py`; tests `tests/test_ta_recurring.py`; outputs "
              "here: `universe.parquet`, `orders_{taker,maker,control}.parquet`, `dev_cells.csv`, `holdout_cells.csv`, "
              "`primary.json`, `results.json`, `benchmarks.csv`, `ledgers/`, `l2_*`; candles in "
              "`data/cache/ta_recurring/hourly/`.\n")]
    path = OUT / "report.md"
    path.write_text("\n".join(parts))
    return path


# ---- driver --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.ta_recurring")
    ap.add_argument("step", choices=["hypotheses", "universe", "fetch", "panel", "dev", "freeze", "holdout", "benchmarks", "l2", "l2day", "l2tables",
                                         "report"])
    ap.add_argument("--day")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "hypotheses":
        path = OUT / "hypotheses.md"
        if not path.exists():      # written once, before the screen; never overwritten
            path.write_text(f"# ta_recurring: hypotheses (written {datetime.now(UTC):%Y-%m-%d %H:%M} UTC, "
                            f"before any outcome was read)\n\n{HYPOTHESES}")
    elif a.step == "universe":
        build_universe(refresh=not a.no_refresh)
    elif a.step == "fetch":
        fetch()
    elif a.step == "panel":
        run_panel()
    elif a.step == "dev":
        run_dev()
    elif a.step == "freeze":
        choose_and_freeze()
    elif a.step == "holdout":
        run_holdout()
    elif a.step == "l2":
        print(json.dumps(run_l2(), indent=2))
    elif a.step == "benchmarks":
        run_benchmarks()
        print(export_marks())
    elif a.step == "l2day":
        run_l2_day(a.day)
    elif a.step == "l2tables":
        print(json.dumps(l2_tables(), indent=2))
    elif a.step == "report":
        print(write_report())


if __name__ == "__main__":
    main()
