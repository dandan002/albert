"""Cross-family price patterns and public flow in recurring culture, approval and executive-action
markets (Kalshi execution only; Polymarket intl is data only, never traded).

Families: Netflix Top 10 (KXNETFLIXRANK*, KXNETFLIXTOPVIEWS*), Billboard/Luminate (KXTOPSONG,
KXTOPALBUM, runner-ups, KXRANKLISTSONGTOP10, KXALBUMEQUIV, KXPUREALBUMS), YouTube (KXYTVIEWS*,
KXYTDAILYTOPVIDEO*), AI leaderboards (KXTOPMODEL, KXLLM1), Rotten Tomatoes (KXRT), approval
(KXTRUMPAPPROVE, KXAPRPOTUS, KXTRUMPVH, KXVHGCB), executive actions (KXEOWEEK, KXMAMDANIEO,
KXTRUMPNOMNUM, KXTRUMPENDORSEMENTS), Truth Social counts (KXTRUTHSOCIAL*), strait traffic (KXHORMUZ*).

Three angles (hypotheses and verdicts in ``data/research/culture_flow/report.md``):

1. **Kalshi ladder structure** from hourly candles of every leg (volume-blind): sum-of-YES over/under
   round by time to close, a resting NO bid above the mid on longshot legs (the mechanism of the
   live mentions edge), leader drift, newly listed legs, weekend theta. Maker fills only when a
   later hourly candle's trade range goes strictly through the quote (the mentions study checked
   this proxy against prints: 96-97% agreement); held to settlement; these series have no maker fee.
2. **Polymarket twins** (exact rule matches from ``xvenue_leadlag``: Netflix ranks/views,
   Billboard #1/#2): price-level disagreement at fixed horizons before the Kalshi close, and who is
   right when the venues disagree.
3. **Polymarket family specialists**: per-wallet PnL per $ risked and edge vs entry price in
   window A, out-of-sample persistence in window B, by family.

Dev = events closing before 2026-10-01; holdout = 2026-10-01 .. latest settled.

    python -m albert.research.culture_flow universe | candles | panel | screen
    python -m albert.research.culture_flow twins | wallets-fetch | wallets-fetch-extra | wallets | follow-dev
    python -m albert.research.culture_flow freeze | holdout | maker-book
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("data/research/culture_flow")
CACHE = Path("data/cache/culture_flow")
START = datetime(2026, 8, 1, tzinfo=UTC)
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)
FRESH = datetime(2026, 11, 17, tzinfo=UTC)
LOOKBACK = timedelta(days=14)          # candle window: the last 14 days of each leg's life

FAMILIES: tuple[tuple[str, str], ...] = (
    ("netflix", r"^KXNETFLIX(RANK(MOVIE|SHOW)(GLOBAL2?|RUNNERUP)?|TOPVIEWS(MOVIE|TV))$"),
    ("billboard", r"^(KXTOPSONG|KXTOPALBUM|KXRANKLISTSONGTOP10|KXBILLBOARDRUNNERUP(SONG|ALBUM)|KXALBUMEQUIV|KXPUREALBUMS)$"),
    ("youtube", r"^(KXYTVIEWSW|KXYTVIEWSHIGH|KXYTDAILYTOPVIDEOG?)$"),
    ("ai", r"^(KXTOPMODEL|KXLLM1)$"),
    ("rt", r"^KXRT$"),
    ("approval", r"^(KXTRUMPAPPROVE|KXAPRPOTUS|KXTRUMPVH|KXVHGCB)$"),
    ("exec", r"^(KXEOWEEK|KXMAMDANIEO|KXTRUMPNOMNUM|KXTRUMPENDORSEMENTS)$"),
    ("truth", r"^(KXTRUTHSOCIAL|KXTRUTHSOCIALD)$"),
    ("hormuz", r"^KXHORMUZ(WEEKLY|MAX|PEAK)$"),
)
CANDLE_DIRS = (Path("data/research/recurring/candles"), Path("data/research/behav_culture/candles_live"),
               Path("data/research/behav_culture/candles_hist"), Path("data/research/xcat/hourly"), CACHE / "candles")
# Hourly Kalshi candles other studies of this swarm cached (read-only; identical API data). Files that
# fail to read (still being written) are skipped. Netflix's cache uses the vol_metrics column names.
EXTERNAL_CANDLES = (Path("data/cache/ta_recurring/hourly"), Path("data/cache/music_charts/candles"),
                    Path("data/research/recurring_alt/candles"), Path("data/research/exec_actions/candles"),
                    Path("data/cache/netflix/candles"))
CANDLE_COLS = ("ticker", "end_ts", "p_high", "p_low", "p_close", "bid_close", "ask_close", "volume")
_RENAME = {"price_high": "p_high", "price_low": "p_low", "price_close": "p_close", "yes_bid_close": "bid_close",
           "yes_ask_close": "ask_close"}


def _candle_files() -> list[Path]:
    out = []
    for d in (*CANDLE_DIRS, *EXTERNAL_CANDLES):
        out += sorted(f for f in d.glob("part*.parquet"))
    return out


def _scan_candles(f: Path, tickers: list[str] | None) -> pl.DataFrame | None:
    try:
        q = pl.scan_parquet(f)
        names = q.collect_schema().names()
        q = q.rename({k: v for k, v in _RENAME.items() if k in names})
        if "period" in names:
            q = q.filter(pl.col("period") == 60)
        if "period_minutes" in names:
            q = q.filter(pl.col("period_minutes") == 60)
        if tickers is not None:
            q = q.filter(pl.col("ticker").is_in(tickers))
        return q.select([pl.col(c).cast(pl.Float64) if c not in ("ticker", "end_ts") else pl.col(c)
                         for c in CANDLE_COLS]).with_columns(pl.col("end_ts").dt.cast_time_unit("us")
                                                             .dt.convert_time_zone("UTC")).collect()
    except (pl.exceptions.PolarsError, OSError) as error:  # a part another study is still writing
        print(f"skip candle file {f}: {str(error)[:120]}")
        return None


def family_of(series: str | None) -> str | None:
    for fam, rx in FAMILIES:
        if re.match(rx, series or ""):
            return fam
    return None


def family_expr(col: str = "series_ticker") -> pl.Expr:
    e = pl.lit(None, dtype=pl.String)
    for fam, rx in reversed(FAMILIES):
        e = pl.when(pl.col(col).str.contains(rx)).then(pl.lit(fam)).otherwise(e)
    return e


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _api(min_interval: float = 0.6):
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=min_interval, max_retries=12))


# ---- universe -----------------------------------------------------------------------------------
MARKET_COLS = ["series_ticker", "event_ticker", "ticker", "yes_sub_title", "strike_type", "floor_strike",
               "cap_strike", "open_time", "close_time", "expected_expiration_time", "settlement_ts", "result",
               "volume"]


def universe(*, fetch_new: bool = True, log=print) -> pl.DataFrame:
    """Every settled leg of the family series closing on/after START: the ``recurring`` study's
    markets (to 2026-11-17) plus markets settled since, from ``/markets?status=settled`` by series."""
    from albert.research.counts import market_row

    base = pl.read_parquet("data/research/recurring/markets.parquet")
    base = base.with_columns(family_expr().alias("family")).filter(pl.col("family").is_not_null())
    series = sorted(base["series_ticker"].unique().to_list())
    frames = [base.select(MARKET_COLS)]
    if fetch_new:
        api = _api()
        rows: list[dict[str, Any]] = []
        for s in series:
            for page in api.series_markets(s, "settled", min_close=datetime(2026, 11, 8, tzinfo=UTC), cache=True):
                rows.extend(market_row(m, s, "live") for m in page.payload.get("markets") or [])
        if rows:
            new = pl.DataFrame(rows, infer_schema_length=None)
            frames.append(new.select([pl.col(c) if c in new.columns else pl.lit(None).alias(c) for c in MARKET_COLS]))
            log(f"universe: {len(rows)} recently settled markets fetched for {len(series)} series")
    mk = (pl.concat(frames, how="vertical_relaxed").unique("ticker", keep="last")
          .filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("close_time") >= START))
          .with_columns(family_expr().alias("family"), (pl.col("result") == "yes").cast(pl.Int8).alias("y")))
    ser = pl.read_parquet("data/tables/kalshi_series.parquet").select("series_ticker", "fee_type", "fee_multiplier")
    mk = mk.join(ser, on="series_ticker", how="left").with_columns(pl.col("fee_multiplier").fill_null(1.0))
    # series with no holdout event carry no test (discontinued in September)
    live = mk.filter(pl.col("close_time") >= DEV_END)["series_ticker"].unique()
    mk = mk.filter(pl.col("series_ticker").is_in(live))
    _save(mk, OUT / "markets.parquet")
    log(f"universe: {mk.height} legs, {mk['event_ticker'].n_unique()} events, {mk['series_ticker'].n_unique()} series,"
        f" close {mk['close_time'].min()} .. {mk['close_time'].max()}")
    return mk


# ---- hourly candles -----------------------------------------------------------------------------
def _have_tickers(need: pl.DataFrame | None = None) -> set[str]:
    """Tickers whose cached hourly candles start by max(open, close - 4 days) + 2 h (so every entry
    time of this study is covered); without ``need``, every cached ticker."""
    out: set[str] = set()
    own = {f for d in CANDLE_DIRS for f in d.glob("part*.parquet")}
    for f in _candle_files():
        c = _scan_candles(f, need["ticker"].to_list() if need is not None else None)
        if c is None or c.is_empty():
            continue
        if need is None or f in own:
            out |= set(c["ticker"].unique().to_list())
            continue
        first = c.group_by("ticker").agg(pl.col("end_ts").min().alias("a")).join(need, on="ticker")
        ok = first.filter(pl.col("a") <= pl.max_horizontal(pl.col("open_time"), pl.col("close_time") - timedelta(days=4))
                          + timedelta(hours=2))
        out |= set(ok["ticker"].to_list())
    return out


def fetch_candles(*, workers: int = 2, min_interval: float = 0.6, log=print) -> None:
    """Hourly candles (price/bid/ask OHLC) over each leg's last ``LOOKBACK`` for legs no earlier study
    cached. Historical legs one request each; live legs batched per event (``xcat`` helpers)."""
    from albert.research import xcat

    mk = pl.read_parquet(OUT / "markets.parquet")
    d = CACHE / "candles"
    d.mkdir(parents=True, exist_ok=True)
    have = _have_tickers(mk.select("ticker", "open_time", "close_time"))
    lo = pl.max_horizontal(pl.col("open_time"), pl.col("close_time") - LOOKBACK)
    todo = (mk.filter(~pl.col("ticker").is_in(list(have)))
            .select("ticker", "event_ticker", "settlement_ts", pl.lit(60).alias("period"), lo.alias("lo"),
                    pl.col("close_time").alias("hi")))
    api = _api(min_interval)
    cutoff = api.cutoff()["market_settled_ts"]
    jobs = xcat._jobs(todo, cutoff)
    log(f"candles: {mk.height} legs, {len(have & set(mk['ticker']))} cached, {todo.height} to fetch in {len(jobs)} requests",
        flush=True)
    lock = threading.Lock()
    buf: list[dict[str, Any]] = []
    part = len(list(d.glob("part-*.parquet")))
    failed: list[str] = []
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, part
        if buf:
            pl.DataFrame(buf, schema=xcat.OHLC_SCHEMA).write_parquet(d / f"part-{part:05d}.parquet")
            part += 1
        buf = []

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(lambda j: xcat._run_job(api, j), jobs):
            with lock:
                for tk, got in res:
                    if got is None:
                        failed.append(tk)
                    else:
                        buf.extend(got)
                n += 1
                if n % 100 == 0:
                    flush()
                    log(f"  candles {n}/{len(jobs)} requests, {n / (time.time() - t0):.2f} req/s", flush=True)
    flush()
    if failed:
        (CACHE / "candles_failed.json").write_text(json.dumps(sorted(set(failed))))
    log(f"candles done: {n} requests, {len(failed)} failed")


def load_candles(tickers: list[str] | None = None) -> pl.DataFrame:
    """Hourly candles of ``tickers`` from every source, first source wins on (ticker, end_ts)."""
    frames = [c for f in _candle_files() if (c := _scan_candles(f, tickers)) is not None and c.height]
    c = pl.concat(frames, how="vertical_relaxed")
    return c.unique(["ticker", "end_ts"], keep="first", maintain_order=True).sort("ticker", "end_ts")


# ---- angle 1: Kalshi ladder structure ------------------------------------------------------------
ENTRY_H = (72, 24, 6)              # hours before the event's scheduled close
MARGINS_C = (6, 12)                # maker NO bid sells YES at mid + m cents
BANDS = ((0.0, 0.05), (0.05, 0.15))
THRESHOLD_STRIKES = ("greater", "greater_or_equal", "less", "less_or_equal")
NON_EXCLUSIVE_SERIES = ("KXRANKLISTSONGTOP10",)
MAX_QUOTE_AGE = timedelta(days=7)  # candles are sparse: a missing hour means nothing changed


def events_table(mk: pl.DataFrame) -> pl.DataFrame:
    """One row per event: scheduled close (latest leg close), first listing, legs, exclusivity."""
    ev = mk.group_by("event_ticker").agg(
        pl.col("series_ticker").first(), pl.col("family").first(), pl.col("close_time").max().alias("ev_close"),
        pl.col("open_time").min().alias("ev_open"), pl.len().alias("legs"), pl.col("y").sum().alias("n_yes"),
        pl.col("strike_type").is_in(list(THRESHOLD_STRIKES)).any().alias("threshold"))
    return ev.with_columns(
        (~pl.col("threshold") & ~pl.col("series_ticker").is_in(list(NON_EXCLUSIVE_SERIES)) & (pl.col("legs") >= 3))
        .alias("exclusive"),
        pl.when(pl.col("ev_close") < DEV_END).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"))


def quotes_at(orders: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """Attach the last hourly candle (bid/ask close) ending at or before ``t`` (<= MAX_QUOTE_AGE)."""
    c = (candles.filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null())
         .select("ticker", pl.col("end_ts").alias("q_ts"), pl.col("bid_close").alias("bid"),
                 pl.col("ask_close").alias("ask")).sort("q_ts"))
    out = orders.sort("t").join_asof(c, left_on="t", right_on="q_ts", by="ticker", strategy="backward",
                                     check_sortedness=False)
    return out.filter(pl.col("q_ts").is_not_null() & ((pl.col("t") - pl.col("q_ts")) <= MAX_QUOTE_AGE)
                      & (pl.col("ask") > pl.col("bid")) & (pl.col("ask") <= 1.0) & (pl.col("bid") >= 0.0))


class Tape:
    """Per-leg hourly candle arrays for fill tests (end time, trade high/low)."""

    def __init__(self, candles: pl.DataFrame) -> None:
        self.by: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.vol: dict[str, np.ndarray] = {}
        c = candles.filter(pl.col("p_high").is_not_null()).select("ticker", "end_ts", "p_high", "p_low", "volume")
        for (tk,), g in c.group_by(["ticker"]):
            g = g.sort("end_ts")
            self.by[tk] = (g["end_ts"].dt.epoch("s").to_numpy(), g["p_high"].to_numpy(), g["p_low"].to_numpy())
            self.vol[tk] = g["volume"].fill_null(0.0).to_numpy()

    def first_through(self, tk: str, t0: datetime, t1: datetime, price: float, side: str) -> datetime | None:
        """First candle ending in (t0, t1] whose trade range goes strictly through ``price``: a YES
        print above it (side 'sell_yes', i.e. our NO bid at 1 - price) or below it ('buy_yes')."""
        a = self.by.get(tk)
        if a is None:
            return None
        ts, hi, lo = a
        i0 = np.searchsorted(ts, int(t0.timestamp()), side="right")
        i1 = np.searchsorted(ts, int(t1.timestamp()), side="right")
        if i1 <= i0:
            return None
        hit = hi[i0:i1] > price + 1e-9 if side == "sell_yes" else lo[i0:i1] < price - 1e-9
        if not hit.any():
            return None
        return datetime.fromtimestamp(int(ts[i0 + int(np.argmax(hit))]), UTC)

    def volume_at(self, tk: str, t: datetime) -> float:
        v = self.vol.get(tk)
        a = self.by.get(tk)
        if v is None or a is None:
            return 0.0
        i = np.searchsorted(a[0], int(t.timestamp()), side="left")
        return float(v[i]) if i < len(v) and a[0][i] == int(t.timestamp()) else 0.0


def build_orders(mk: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """(leg, entry) rows: every leg open at close - h for h in ENTRY_H, with its quote."""
    legs = mk.join(ev.select("event_ticker", "ev_close", "ev_open", "exclusive", "legs", "split"), on="event_ticker")
    rows = (legs.with_columns(pl.lit(list(ENTRY_H)).alias("h")).explode("h")
            .with_columns((pl.col("ev_close") - pl.duration(hours=pl.col("h"))).alias("t"))
            .filter((pl.col("open_time") <= pl.col("t")) & (pl.col("close_time") > pl.col("t"))))
    return rows


def maker_no(orders: pl.DataFrame, tape: Tape, margin_c: int) -> pl.DataFrame:
    """Rest a NO bid that sells YES at ceil(mid + m) (never below the ask's level is required: it is
    post-only as long as it is above the YES bid). Filled when a later hourly print goes strictly
    above it before the leg closes; held to settlement; no maker fee on these series."""
    from albert.research.fundamental import maker_fee

    rows = []
    for r in orders.iter_rows(named=True):
        mid = (r["bid"] + r["ask"]) / 2
        s = math.ceil(round((mid + margin_c / 100) * 100, 6)) / 100
        if s >= 0.99 or s <= r["bid"]:
            continue
        hit = tape.first_through(r["ticker"], r["t"], r["close_time"], s, "sell_yes")
        if hit is None:
            continue
        no_price = round(1 - s, 2)
        fee = maker_fee(no_price, r.get("fee_type"), r.get("fee_multiplier") or 1.0)
        rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "family": r["family"],
                     "series_ticker": r["series_ticker"], "split": r["split"], "h": r["h"], "t": hit, "t_order": r["t"],
                     "mid": mid, "sell_at": s, "price": no_price, "dir": -1, "y": r["y"],
                     "fill_candle_volume": tape.volume_at(r["ticker"], hit),
                     "pnl_c": 100 * ((1 - r["y"]) - no_price - fee), "fee_order": fee,
                     "settlement_ts": r["settlement_ts"], "day": hit.date()})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def taker_rows(orders: pl.DataFrame, side: str, budget: float = 4.5) -> pl.DataFrame:
    """Taker at the quote (YES at the ask / NO at 1 - bid), floor(budget / price) contracts, fee per
    order rounded up to the cent; held to settlement."""
    rows = []
    for r in orders.iter_rows(named=True):
        price = r["ask"] if side == "yes" else round(1 - r["bid"], 4)
        if price <= 0.0 or price >= 1.0:
            continue
        n = max(1, int(budget // price))
        fee = taker_fee_order(n, price, r.get("fee_multiplier") or 1.0)
        won = r["y"] if side == "yes" else 1 - r["y"]
        rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "family": r["family"],
                     "series_ticker": r["series_ticker"], "split": r["split"], "h": r.get("h"), "t": r["t"],
                     "mid": (r["bid"] + r["ask"]) / 2, "price": price, "n": n, "dir": 1 if side == "yes" else -1,
                     "y": r["y"], "fee_order": fee, "pnl_c": 100 * (won - price - fee / n),
                     "settlement_ts": r["settlement_ts"], "day": r["t"].date()})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def overround(orders: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """Per exclusive event and entry: sum of YES mids / bids / asks over legs quoted at t (events with
    every open leg quoted), and whether buying the field (all YES) or selling it (all NO) at the
    quotes beats $1 after 1-lot taker fees."""
    ex = orders.join(ev.filter(pl.col("exclusive")).select("event_ticker"), on="event_ticker")
    g = ex.group_by("event_ticker", "h").agg(
        pl.col("family").first(), pl.col("split").first(), pl.len().alias("quoted"),
        ((pl.col("bid") + pl.col("ask")) / 2).sum().alias("sum_mid"), pl.col("bid").sum().alias("sum_bid"),
        pl.col("ask").sum().alias("sum_ask"),
        (0.07 * pl.col("ask") * (1 - pl.col("ask"))).sum().alias("fee_yes"),
        (0.07 * pl.col("bid") * (1 - pl.col("bid"))).sum().alias("fee_no"), pl.col("y").sum().alias("n_yes"))
    return g.with_columns((pl.col("sum_ask") + pl.col("fee_yes") < 1 - 1e-9).alias("buy_field_arb"),
                          (pl.col("sum_bid") - pl.col("fee_no") > 1 + 1e-9).alias("sell_field_arb"))


def panel(log=print) -> None:
    """Orders with quotes (all legs x entries) and the event table, saved for screening."""
    mk = pl.read_parquet(OUT / "markets.parquet")
    ev = events_table(mk)
    orders = build_orders(mk, ev)
    c = load_candles(mk["ticker"].to_list())
    q = quotes_at(orders, c)
    log(f"panel: {orders.height} leg-entries, {q.height} quoted ({q['ticker'].n_unique()} legs, "
        f"{q['event_ticker'].n_unique()} events); candles for {c['ticker'].n_unique()} of {mk.height} legs")
    # newly listed legs: listed >= 12 h after the event's first leg; first two-sided quote within 6 h
    late = mk.join(ev.select("event_ticker", "ev_open", "ev_close", "split"), on="event_ticker").filter(
        (pl.col("open_time") >= pl.col("ev_open") + timedelta(hours=12)) & (pl.col("close_time") > pl.col("open_time") + timedelta(hours=1)))
    fq = (c.filter(pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null() & (pl.col("ask_close") > pl.col("bid_close")))
          .join(late.select("ticker", "open_time"), on="ticker")
          .filter(pl.col("end_ts") <= pl.col("open_time") + timedelta(hours=6))
          .sort("end_ts").group_by("ticker").first()
          .select("ticker", pl.col("end_ts").alias("t"), pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask")))
    late = late.join(fq, on="ticker").with_columns(pl.lit(None, dtype=pl.Int64).alias("h"))
    _save(ev, OUT / "events.parquet")
    _save(q, OUT / "orders.parquet")
    _save(late, OUT / "late_legs.parquet")
    log(f"late-listed legs with an early quote: {late.height} ({late['event_ticker'].n_unique()} events)")


FAM_ALL = ("all", "netflix", "billboard", "youtube", "ai", "rt", "approval", "exec", "truth", "hormuz", "counts")


def fam_filter(df: pl.DataFrame, fam: str) -> pl.DataFrame:
    if fam == "all":
        return df
    if fam == "counts":
        return df.filter(pl.col("family").is_in(["truth", "exec"]))
    return df.filter(pl.col("family") == fam)


def weekday_orders(mk: pl.DataFrame, ev: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    """H13 entries: every Friday 22:00 UTC and Wednesday 22:00 UTC inside a leg's life, >= 6 h
    before the event close, with the quote at that time."""
    legs = mk.join(ev.select("event_ticker", "ev_close", "split"), on="event_ticker")
    rows = []
    for r in legs.select("ticker", "open_time", "close_time", "ev_close").iter_rows(named=True):
        d = r["open_time"].replace(hour=22, minute=0, second=0, microsecond=0)
        while d < r["ev_close"] - timedelta(hours=6):
            if d >= r["open_time"] and d < r["close_time"] and d.weekday() in (2, 4):
                rows.append({"ticker": r["ticker"], "t": d, "wd": "fri" if d.weekday() == 4 else "wed"})
            d += timedelta(days=1)
    w = pl.DataFrame(rows, schema={"ticker": pl.String, "t": pl.Datetime("us", "UTC"), "wd": pl.String})
    w = w.join(legs, on="ticker").with_columns(pl.lit(None, dtype=pl.Int64).alias("h"))
    return quotes_at(w, candles)


def screen_cells(orders: pl.DataFrame, late: pl.DataFrame, wk: pl.DataFrame, tape: Tape,
                 split: str) -> tuple[list[dict[str, Any]], dict[str, pl.DataFrame]]:
    """Every cell of the angle-1 grid on one split. Returns cell rows and the trade frames by cell id."""
    o = orders.filter(pl.col("split") == split).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    cells: list[dict[str, Any]] = []
    trades: dict[str, pl.DataFrame] = {}

    def add(cid: str, hyp: str, df: pl.DataFrame, **kw: Any) -> None:
        trades[cid] = df
        cells.append({"cell": cid, "hyp": hyp, "split": split, **kw, **summarize(df)})

    # H1 / H2 / H12: maker NO bid above the mid on longshots
    for m in MARGINS_C:
        for lo, hi in BANDS:
            sub = o.filter((pl.col("mid") > lo) & (pl.col("mid") <= hi))
            fills = maker_no(sub, tape, m)
            for h in ENTRY_H:
                for fam in FAM_ALL:
                    d = fam_filter(fills.filter(pl.col("h") == h), fam) if fills.height else fills
                    orders_n = fam_filter(sub.filter(pl.col("h") == h), fam).height
                    add(f"makerNO|m{m}|b{int(hi * 100)}|h{h}|{fam}", "H1/H2/H12", d, orders=orders_n)
    # H1 control: taker NO on longshots (pooled over entries)
    for lo, hi in BANDS:
        sub = o.filter((pl.col("mid") > lo) & (pl.col("mid") <= hi))
        tk = taker_rows(sub, "no")
        for fam in FAM_ALL:
            add(f"takerNO|b{int(hi * 100)}|{fam}", "H1ctl", fam_filter(tk, fam) if tk.height else tk)
    # H4: leader (highest mid) of an exclusive event, taker YES, spread <= 10c
    ex = o.filter(pl.col("exclusive") & ((pl.col("ask") - pl.col("bid")) <= 0.10))
    lead = ex.sort("mid", descending=True).group_by("event_ticker", "h").first()
    tk = taker_rows(lead, "yes")
    for h in ENTRY_H:
        for fam in FAM_ALL:
            d = fam_filter(tk.filter(pl.col("h") == h), fam) if tk.height else tk
            add(f"leaderYES|h{h}|{fam}", "H4", d)
    # H6: late-listed legs at their first quote: taker NO and maker NO (m = 6c)
    lt = late.filter(pl.col("split") == split)
    if lt.height:
        tn = taker_rows(lt, "no")
        mn = maker_no(lt, tape, 6)
        for fam in FAM_ALL:
            add(f"lateNO_taker|{fam}", "H6", fam_filter(tn, fam) if tn.height else tn)
            add(f"lateNO_maker6|{fam}", "H6", fam_filter(mn, fam) if mn.height else mn)
    # H13: Friday-evening vs Wednesday-evening longshot NO taker
    w = wk.filter(pl.col("split") == split).with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    w = w.filter((pl.col("mid") > 0) & (pl.col("mid") <= 0.15))
    for wd in ("fri", "wed"):
        d = taker_rows(w.filter(pl.col("wd") == wd), "no")
        add(f"weekendNO|{wd}", "H13", d)
    return cells, trades


def screen(log=print) -> None:
    """Dev-only screen of the angle-1 grid (holdout rows are never touched here)."""
    orders = pl.read_parquet(OUT / "orders.parquet")
    late = pl.read_parquet(OUT / "late_legs.parquet")
    ev = pl.read_parquet(OUT / "events.parquet")
    mk = pl.read_parquet(OUT / "markets.parquet")
    dev_t = mk.filter(pl.col("close_time") < DEV_END + timedelta(days=8))["ticker"].to_list()
    c = load_candles(dev_t)
    tape = Tape(c)
    wk = weekday_orders(mk.filter(pl.col("ticker").is_in(dev_t)), ev, c)
    cells, _ = screen_cells(orders, late, wk, tape, "dev")
    ov = overround(orders.filter(pl.col("split") == "dev"), ev)
    _save(ov, OUT / "overround_dev.parquet")
    ovs = (ov.group_by("family", "h").agg(pl.len().alias("event_entries"), (pl.col("sum_mid") - 1).mean().alias("over_mid"),
                                           (pl.col("sum_ask") - 1).mean().alias("over_ask"), (pl.col("sum_bid") - 1).mean().alias("over_bid"),
                                           pl.col("buy_field_arb").sum(), pl.col("sell_field_arb").sum(),
                                           (pl.col("n_yes") == 1).mean().alias("one_winner_share"))
           .sort("family", "h", descending=[False, True]))
    pl.DataFrame(cells, infer_schema_length=None).write_csv(OUT / "cells_dev.csv")
    ovs.write_csv(OUT / "overround_dev.csv")
    log(f"screen: {len(cells)} dev cells")


# ---- freeze, holdout, ledgers ------------------------------------------------------------------------
SELECTION = ("Primary: among angle-1 dev cells with >= 60 fills/trades and >= 20 events, the cell with the "
             "highest two-way (event x date) clustered t of net c/contract, mean > 0; ties -> more fills. "
             "Secondary: the best Polymarket-twin taker cell by the same rule (>= 40 trades, >= 15 events), "
             "and the wallet top-k rule if its dev out-of-sample ROI t >= 2. A live candidate needs holdout "
             "event-clustered t >= 2, >= 60% positive weeks and Sharpe ex top-5 > 1.")
MAKER_LOTS = 2         # contracts per maker fill in the ledger (needs >= 2x that many printed in the fill hour)


def pick(cells: pl.DataFrame, min_n: int, min_ev: int) -> dict[str, Any] | None:
    c = cells.filter((pl.col("n") >= min_n) & (pl.col("events") >= min_ev) & (pl.col("mean_c") > 0)
                     & pl.col("t_2way").is_not_null())
    if c.is_empty():
        return None
    return c.sort(["t_2way", "n"], descending=True).row(0, named=True)


def freeze(log=print) -> dict[str, Any]:
    path = OUT / "primary.json"
    if path.exists():
        raise SystemExit("primary.json exists: the rules are frozen (delete it only to restart the study)")
    cells = pl.read_csv(OUT / "cells_dev.csv", infer_schema_length=None)
    prim = pick(cells, 60, 20)
    tw = json.loads((OUT / "twins.json").read_text()) if (OUT / "twins.json").exists() else {}
    tc = pl.DataFrame(tw.get("taker_cells") or [], infer_schema_length=None)
    sec_twin = None
    if tc.height:
        tdev = tc.filter(pl.col("split") == "dev").with_columns(
            (pl.lit("twin|g") + (pl.col("gap") * 100).round(0).cast(pl.Int64).cast(pl.String) + "|h"
             + pl.col("h").cast(pl.String) + "|" + pl.col("fam")).alias("cell"))
        sec_twin = pick(tdev, 40, 15)
    wd = json.loads((OUT / "wallets_dev.json").read_text()) if (OUT / "wallets_dev.json").exists() else {}
    sec_wallet = [r for r in wd.get("dev_A_to_B", []) if (r.get("top_t") or 0) >= 2 and (r.get("top_roi") or 0) > 0]
    fd = json.loads((OUT / "follow_dev.json").read_text()) if (OUT / "follow_dev.json").exists() else {}
    sec_follow = [{"fam": r["fam"], "dev": fd.get(r["fam"]),
                   "rule": (f"H10: top {TOP_K} {r['fam']} wallets ranked on window B; their fills in window C in "
                            f"Polymarket markets exactly matched to Kalshi (>= ${FOLLOW_MIN_USD:.0f} per wallet-market-side-"
                            f"hour) -> Kalshi taker {int(FOLLOW_DELAY.total_seconds() // 60)} min later at the ask / "
                            f"1 - bid (spread <= 10c, 3-97c), one entry per market-side per 6 h, held to settlement")}
                  for r in sec_wallet if fd.get(r["fam"])]
    out = {"frozen_at": datetime.now(UTC).isoformat(), "selection": SELECTION,
           "dev_cells_angle1": cells.height, "primary": prim, "secondary_twin": sec_twin,
           "secondary_wallets": sec_wallet or None, "secondary_follow": sec_follow or None,
           "follow_note": ("H10 was pre-registered as conditional on H9; its parameters were coded before its dev run. "
                           "Dev: wallets ranked on A (May-Jul), signals in B (Aug-Sep). Holdout: ranked on B, signals "
                           "in C (PM markets ending Oct 1 - Nov 11; Kalshi closes to Nov 10)."),
           "inspired_by_earlier_holdout": ("None of the angle-1 cells copies an earlier holdout result; the "
                                           "fandom NO lead (behav_culture) is not in the grid. Fresh = after 2026-11-17."),
           "execution": {"maker": f"NO bid selling YES at ceil(mid + m), fill on a later hourly print strictly through, "
                                  f"{MAKER_LOTS} contracts in the ledger, no maker fee on these series",
                         "taker": "ask / 1 - bid, floor($4.50 / price) contracts, fee 0.07 P(1-P) per order rounded up",
                         "book": "$100 capped book, <= $5 open per event, held to settlement"}}
    _save_json(out, path)
    log(json.dumps(out, indent=1, default=str))
    return out


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def split_cells(split: str) -> tuple[list[dict[str, Any]], dict[str, pl.DataFrame]]:
    """Every angle-1 cell on one split (the same code path as the screen)."""
    orders = pl.read_parquet(OUT / "orders.parquet")
    late = pl.read_parquet(OUT / "late_legs.parquet")
    ev = pl.read_parquet(OUT / "events.parquet")
    mk = pl.read_parquet(OUT / "markets.parquet")
    sel = (mk.filter(pl.col("close_time") < DEV_END + timedelta(days=8)) if split == "dev"
           else mk.filter(pl.col("close_time") >= DEV_END - timedelta(days=8)))
    c = load_candles(sel["ticker"].to_list())
    wk = weekday_orders(sel, ev, c)
    return screen_cells(orders, late, wk, Tape(c), split)


def cell_trades(cell: str, split: str) -> pl.DataFrame:
    """Recompute one angle-1 cell's trades on a split (the same code path as the screen)."""
    orders = pl.read_parquet(OUT / "orders.parquet")
    late = pl.read_parquet(OUT / "late_legs.parquet")
    ev = pl.read_parquet(OUT / "events.parquet")
    mk = pl.read_parquet(OUT / "markets.parquet")
    lo, hi = (DEV_END - timedelta(days=70), DEV_END + timedelta(days=8)) if split == "dev" else (DEV_END - timedelta(days=8), None)
    sel = mk.filter(pl.col("close_time") >= lo)
    if hi is not None:
        sel = sel.filter(pl.col("close_time") < hi)
    c = load_candles(sel["ticker"].to_list())
    tape = Tape(c)
    wk = weekday_orders(sel, ev, c) if cell.startswith("weekend") else pl.DataFrame(
        schema={"split": pl.String, "bid": pl.Float64, "ask": pl.Float64, "wd": pl.String})
    _, trades = screen_cells(orders, late, wk, tape, split)
    return trades[cell]


def marks_for(tickers: list[str]) -> pl.DataFrame:
    """Hourly YES marks: the mid of a real two-sided book (bid > 0 and ask < 1), else the last trade.
    An empty side (bid 0 or ask $1) would put a dead longshot's mark at ~50c."""
    c = load_candles(tickers)
    two = (pl.col("bid_close").is_not_null() & pl.col("ask_close").is_not_null() & (pl.col("ask_close") > pl.col("bid_close"))
           & (pl.col("bid_close") > 0) & (pl.col("ask_close") < 1))
    mid = pl.when(two).then((pl.col("bid_close") + pl.col("ask_close")) / 2).otherwise(pl.col("p_close"))
    return (c.with_columns(mid.alias("yes_mid")).filter(pl.col("yes_mid").is_not_null())
            .select("ticker", pl.col("end_ts").alias("ts"), "yes_mid"))


def book(trades: pl.DataFrame, name: str, split: str) -> dict[str, Any]:
    """$100 capped ledger (<= $5 per event) and vol_metrics statistics."""
    from albert.research.ta_calendar import capped_ledger
    from albert.research.vol_metrics import ledger_metrics

    if trades.is_empty():
        return {"positions": 0}
    t = trades
    if "n" not in t.columns:     # maker fills: MAKER_LOTS contracts when the fill hour printed >= 2x that
        t = t.filter(pl.col("fill_candle_volume") >= 2 * MAKER_LOTS).with_columns(pl.lit(MAKER_LOTS).alias("n"))
    t = t.with_columns(pl.when(pl.col("dir") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("payoff"),
                       pl.col("fee_order").fill_null(0.0))
    led = capped_ledger(t, "culture_flow", name, split, bankroll=100.0, event_cap=5.0)
    if led.is_empty():
        return {"positions": 0}
    _save(led, OUT / "ledgers" / f"ledger_{name.replace('|', '_')}_{split}.parquet")
    m = ledger_metrics(led, marks_for(led["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "sortino", "max_dd_usd", "weeks_positive_pct", "skew", "first_day", "last_day")
    return {k: m.get(k) for k in keep}


def weeks_positive(trades: pl.DataFrame) -> float | None:
    if trades.is_empty():
        return None
    w = trades.group_by(pl.col("t").dt.truncate("1w")).agg(pl.col("pnl_c").sum())
    return round(100 * float((w["pnl_c"] > 0).mean()), 0)


def maker_orders(split: str, margin_c: int = 12, h: int = 72, band: tuple[float, float] = (0.0, 0.05)) -> pl.DataFrame:
    """Every order of the maker NO rule (filled or not): placed at close - h, resting until its fill
    or the leg's close, 1 contract."""
    orders = pl.read_parquet(OUT / "orders.parquet").filter((pl.col("split") == split) & (pl.col("h") == h))
    orders = orders.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid")).filter(
        (pl.col("mid") > band[0]) & (pl.col("mid") <= band[1]))
    tape = Tape(load_candles(orders["ticker"].unique().to_list()))
    rows = []
    for r in orders.iter_rows(named=True):
        s = math.ceil(round((r["mid"] + margin_c / 100) * 100, 6)) / 100
        if s >= 0.99 or s <= r["bid"]:
            continue
        hit = tape.first_through(r["ticker"], r["t"], r["close_time"], s, "sell_yes")
        if hit is not None and tape.volume_at(r["ticker"], hit) < 2:
            hit = None                                # a 1-lot print cannot be shared: no fill
        rows.append({"ticker": r["ticker"], "event_ticker": r["event_ticker"], "family": r["family"], "t": r["t"],
                     "price": round(1 - s, 2), "fill_ts": hit, "end": hit or r["close_time"], "y": r["y"],
                     "settlement_ts": r["settlement_ts"]})
    return pl.DataFrame(rows, infer_schema_length=None)


def resting_book(o: pl.DataFrame, name: str, split: str, *, bankroll: float = 100.0, event_cap: float = 5.0,
                 rest_cap: float = 100.0) -> dict[str, Any]:
    """Maker book in which resting orders reserve collateral (Kalshi holds a resting bid's cost):
    at placement (events in time order, cheapest NO bid first) an order is kept only while resting +
    open positions stay within the bankroll, the event within $5 and resting orders within
    ``rest_cap``. A filled order becomes a 1-lot position held to settlement."""
    import heapq

    from albert.research.vol_metrics import ledger_metrics

    events: list[tuple[datetime, int, str, dict[str, Any]]] = []
    for i, r in enumerate(o.sort("t", "price", "ticker").iter_rows(named=True)):
        events.append((r["t"], 0, f"{i}", r))
    heap: list[tuple[datetime, int, float, str, str]] = []   # (release time, seq, amount, event, kind)
    used = resting = 0.0
    per_ev: dict[str, float] = {}
    led = []
    seq = 0
    for t, _, _, r in sorted(events, key=lambda x: (x[0], x[3]["price"], x[3]["ticker"])):
        while heap and heap[0][0] <= t:
            _, _, amt, ev, kind = heapq.heappop(heap)
            used -= amt
            per_ev[ev] -= amt
            if kind == "rest":
                resting -= amt
        c = r["price"]
        if (used + c > bankroll + 1e-9 or per_ev.get(r["event_ticker"], 0.0) + c > event_cap + 1e-9
                or resting + c > rest_cap + 1e-9):
            continue
        used += c
        per_ev[r["event_ticker"]] = per_ev.get(r["event_ticker"], 0.0) + c
        seq += 1
        if r["fill_ts"] is None:
            resting += c
            heapq.heappush(heap, (r["end"], seq, c, r["event_ticker"], "rest"))
            continue
        # rests until the fill, then the same collateral is the position until settlement
        settle = max(r["settlement_ts"], r["fill_ts"])
        heapq.heappush(heap, (settle, seq, c, r["event_ticker"], "pos"))
        led.append({"strategy": "culture_flow", "variant": name, "split": split,
                    "struct_id": f"{r['ticker']}|{r['t'].isoformat()}", "event_ticker": r["event_ticker"],
                    "ticker": r["ticker"], "side": "no", "n": 1, "t_in": r["fill_ts"], "cost_usd": c,
                    "settle_ts": settle, "payoff_usd": float(1 - r["y"])})
    if not led:
        return {"positions": 0, "orders_placed": seq}
    ledger = pl.DataFrame(led)
    _save(ledger, OUT / "ledgers" / f"ledger_{name.replace('|', '_')}_{split}.parquet")
    m = ledger_metrics(ledger, marks_for(ledger["ticker"].unique().to_list()))
    keep = ("positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
            "p_sharpe_pos", "sharpe_ex_top5", "sortino", "max_dd_usd", "weeks_positive_pct", "skew", "first_day", "last_day")
    return {"orders_placed": seq, **{k: m.get(k) for k in keep}}


def fandom_forward() -> dict[str, Any]:
    """behav_culture's frozen R1 (fandom NO at 2-35c, spread <= 10c, taker at 1 - bid, n = floor($5 / p))
    at scheduled close - 24 h, scored only on decisions after 2026-11-17 (its holdout was already seen)."""
    from albert.research.behav_culture import fandom_expr

    o = pl.read_parquet(OUT / "orders.parquet")
    mk = pl.read_parquet("data/research/recurring/markets.parquet", columns=["ticker", "name"])
    o = o.join(mk, on="ticker", how="left").filter(
        (pl.col("h") == 24) & (pl.col("t") >= FRESH) & pl.col("family").is_in(["netflix", "billboard", "youtube", "rt"]))
    mid = (pl.col("bid") + pl.col("ask")) / 2
    o = o.filter(mid.is_between(0.02, 0.35) & ((pl.col("ask") - pl.col("bid")) <= 0.10)
                 & (fandom_expr(pl.col("yes_sub_title")) | fandom_expr(pl.col("name"))))
    tk = taker_rows(o, "no", budget=5.0)
    return {**summarize(tk), "note": "close - 24 h slot only; decisions after 2026-11-17; universe = this study's "
                                     "Entertainment families (behav_culture used every Entertainment leg)"}


def holdout(log=print) -> dict[str, Any]:
    fr = json.loads((OUT / "primary.json").read_text())
    res: dict[str, Any] = {"frozen_at": fr["frozen_at"], "scored_at": datetime.now(UTC).isoformat()}
    prim = fr.get("primary")
    xp = json.loads((OUT / "exploratory_frozen.json").read_text())["cells"] if (OUT / "exploratory_frozen.json").exists() else {}
    for split in ("dev", "holdout"):
        cells, trades = split_cells(split)
        if split == "holdout":
            pl.DataFrame(cells, infer_schema_length=None).write_csv(OUT / "cells_holdout.csv")
        if prim:
            cell = prim["cell"]
            tr = trades[cell]
            _save(tr, OUT / f"primary_trades_{split}.parquet")
            r = {**summarize(tr), "weeks_positive_pct_fills": weeks_positive(tr), "book": book(tr, cell, split)}
            if split == "holdout" and tr.height:
                r["fresh_after_2026_11_17"] = summarize(tr.filter(pl.col("t") >= FRESH))
                r["losses"] = tr.filter(pl.col("pnl_c") < 0).height
            res[f"primary_{split}"] = r
        for cell in xp:
            tr = trades.get(cell, pl.DataFrame())
            res.setdefault("exploratory", {}).setdefault(cell, {})[split] = {
                **summarize(tr), "weeks_positive_pct_fills": weeks_positive(tr),
                "book": book(tr, cell, split) if tr.height else None}
    res["fandom_forward_check"] = fandom_forward()
    for w in fr.get("secondary_wallets") or []:
        rows = wallet_rows()
        sc = pl.read_parquet(OUT / "wallet_scores.parquet")
        res.setdefault("secondary_wallets_holdout", []).extend(
            [x for x in persistence(sc, rows, "B", "C") if x["fam"] == w["fam"]])
    for f in fr.get("secondary_follow") or []:
        tr = follow(f["fam"], "B", "C")
        _save(tr, OUT / f"follow_{f['fam']}_holdout.parquet")
        res.setdefault("secondary_follow_holdout", []).append(
            {"fam": f["fam"], **summarize(tr), "weeks_positive_pct_fills": weeks_positive(tr),
             "book": book(tr, f"follow_{f['fam']}", "holdout")})
    if fr.get("secondary_twin"):
        s = fr["secondary_twin"]
        tw = json.loads((OUT / "twins.json").read_text())
        res["secondary_twin_holdout"] = [c for c in tw["taker_cells"] if c["split"] == "holdout" and c["gap"] == s["gap"]
                                         and c["h"] == s["h"] and c["fam"] == s["fam"]]
    _save_json(res, OUT / "holdout.json")
    log(json.dumps(res, indent=1, default=str))
    return res


# ---- statistics -----------------------------------------------------------------------------------
def taker_fee_order(n: int, price: float, mult: float = 1.0) -> float:
    """Kalshi taker fee for one order of ``n`` contracts at ``price``, rounded up to the cent."""
    return math.ceil(0.07 * mult * n * price * (1 - price) * 100 - 1e-9) / 100


def cl_t(x: np.ndarray, g: np.ndarray) -> tuple[float, float | None, int]:
    from albert.research.fundamental import clustered_mean

    return clustered_mean(np.asarray(x, dtype=float), np.asarray(g))


def two_way_t(x: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> float | None:
    """t of the mean with Cameron-Gelbach-Miller two-way clustering (event and date)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()

    def v(keys: np.ndarray) -> float:
        _, inv = np.unique(keys, return_inverse=True)
        s = np.bincount(inv, weights=e)
        k = len(s)
        return float((s ** 2).sum()) * (k / (k - 1) if k > 1 else 1.0) / n ** 2

    inter = np.char.add(np.char.add(np.asarray(g1, dtype=str), "|"), np.asarray(g2, dtype=str))
    var = v(np.asarray(g1)) + v(np.asarray(g2)) - v(inter)
    if var <= 0:
        var = max(v(np.asarray(g1)), v(np.asarray(g2)))
    return float(x.mean() / math.sqrt(var)) if var > 0 else None


def summarize(df: pl.DataFrame, col: str = "pnl_c", ev: str = "event_ticker", day: str = "day") -> dict[str, Any]:
    if df.is_empty():
        return {"n": 0}
    x = df[col].to_numpy()
    mu, t_ev, k = cl_t(x, df[ev].to_numpy())
    out = {"n": df.height, "events": k, "mean_c": round(mu, 3), "t_event": None if t_ev is None else round(t_ev, 2)}
    if day in df.columns:
        t2 = two_way_t(x, df[ev].to_numpy(), df[day].cast(pl.String).to_numpy())
        out["t_2way"] = None if t2 is None else round(t2, 2)
    return out


# ---- angle 2: Polymarket twins (exact pairs from xvenue_leadlag) -----------------------------------
XV = Path("data/research/xvenue_leadlag")
TWIN_FAMILIES = ("netflix_rank", "netflix_views", "billboard_200_1", "billboard_hot100_1", "billboard_hot100_2")
TWIN_H = (24, 6, 1)
TWIN_G = (0.03, 0.05, 0.10)


def twin_panel(log=print) -> pl.DataFrame:
    """Kalshi quote and Polymarket mid at close - h for every exact culture pair (minute paths
    cached by xvenue_leadlag). Kalshi: last minute candle ending <= T (<= 60 min old); entry quote
    one minute later. Polymarket: last CLOB point stamped <= T (it shows the book ~14 s earlier)."""
    p = (pl.read_parquet(XV / "pairs.parquet").filter(pl.col("family").is_in(TWIN_FAMILIES))
         .select("pair_id", "family", "ticker", "event_ticker", "token", "k_result", "fee_multiplier", "k_close",
                 "k_settle"))
    grid = (p.with_columns(pl.lit(list(TWIN_H)).alias("h")).explode("h")
            .with_columns((pl.col("k_close") - pl.duration(hours=pl.col("h"))).dt.truncate("1s").alias("T")))
    # one part file at a time, keeping only the 61 minutes before each decision time (and the entry
    # minute), so memory stays small
    kwin = grid.select("ticker", (pl.col("T") - timedelta(minutes=61)).alias("lo"),
                       (pl.col("T") + timedelta(minutes=1)).alias("hi")).unique()
    # PM: per part file, an as-of join of each (token, T) on the file's points (tokens as small ints)
    tok = p.select("token").unique().with_row_index("tid")
    gq = grid.join(tok, on="token").select("tid", "token", pl.col("T").dt.epoch("s").alias("Ts")).unique().sort("Ts")
    kps, pps = [], []
    for f in sorted((XV / "kalshi_paths").glob("part-*.parquet")):
        d = (pl.scan_parquet(f).select("ticker", "end_ts", "bid_close", "ask_close")
             .filter(pl.col("ticker").is_in(kwin["ticker"].unique().to_list()) & pl.col("bid_close").is_not_null()
                     & pl.col("ask_close").is_not_null()).collect())
        kps.append(d.join(kwin, on="ticker").filter(pl.col("end_ts").is_between(pl.col("lo"), pl.col("hi")))
                   .drop("lo", "hi"))
        del d
    for f in sorted((XV / "pm_paths").glob("part-*.parquet")):
        d = (pl.scan_parquet(f).filter(pl.col("token").is_in(tok["token"].to_list())).collect()
             .join(tok, on="token").select("tid", "t", "p").sort("t"))
        if d.height:
            hit = gq.filter(pl.col("tid").is_in(d["tid"].unique().to_list())).join_asof(
                d, left_on="Ts", right_on="t", by="tid", strategy="backward", check_sortedness=False)
            pps.append(hit.filter(pl.col("t").is_not_null()).select("token", "Ts", "t", "p"))
        del d
    kp = pl.concat(kps).unique(["ticker", "end_ts"]).sort("ticker", "end_ts")
    # latest point at or before T across files
    pp = (pl.concat(pps).sort("t").group_by("token", "Ts").last()
          .select("token", pl.from_epoch("Ts", time_unit="s").dt.replace_time_zone("UTC").alias("T"),
                  pl.from_epoch("t", time_unit="s").dt.replace_time_zone("UTC").alias("p_ts"), pl.col("p").alias("pm")))
    log(f"twins: {p.height} pairs, {kp.height} kalshi minutes, {pp.height} pm points")
    g = grid.sort("T")
    g = g.join_asof(kp.rename({"end_ts": "k_ts", "bid_close": "kbid", "ask_close": "kask"}), left_on="T",
                    right_on="k_ts", by="ticker", strategy="backward", check_sortedness=False)
    g = g.with_columns((pl.col("T") + timedelta(minutes=1)).alias("T1")).sort("T1")
    g = g.join_asof(kp.rename({"end_ts": "e_ts", "bid_close": "ebid", "ask_close": "eask"}), left_on="T1",
                    right_on="e_ts", by="ticker", strategy="backward", check_sortedness=False)
    g = g.with_columns(pl.col("T").cast(pl.Datetime("us", "UTC"))).join(
        pp.with_columns(pl.col("T").cast(pl.Datetime("us", "UTC"))), on=["token", "T"], how="left")
    g = g.filter(pl.col("k_ts").is_not_null() & ((pl.col("T") - pl.col("k_ts")) <= timedelta(minutes=60))
                 & pl.col("p_ts").is_not_null() & ((pl.col("T") - pl.col("p_ts")) <= timedelta(minutes=60))
                 & (pl.col("kask") > pl.col("kbid")))
    g = g.with_columns(((pl.col("kbid") + pl.col("kask")) / 2).alias("kmid"),
                       (pl.col("k_result") == "yes").cast(pl.Int8).alias("y"),
                       pl.when(pl.col("k_close") < DEV_END).then(pl.lit("dev")).otherwise(pl.lit("holdout")).alias("split"),
                       pl.col("k_close").dt.date().alias("day"),
                       pl.when(pl.col("family").str.starts_with("netflix")).then(pl.lit("netflix"))
                       .otherwise(pl.lit("billboard")).alias("fam"))
    return g


def twin_taker(g: pl.DataFrame, gap: float, budget: float = 4.5) -> pl.DataFrame:
    """Take Kalshi toward Polymarket when PM's mid beats the Kalshi entry ask (or 1 - bid) by >= gap
    after the taker fee; one order of floor(budget / price) contracts, held to settlement."""
    rows = []
    for r in g.iter_rows(named=True):
        if r["eask"] is None or r["ebid"] is None or r["eask"] <= r["ebid"]:
            continue
        pm, mult = r["pm"], r["fee_multiplier"] or 1.0
        for d, price, fair in ((1, r["eask"], pm), (-1, 1 - r["ebid"], 1 - pm)):
            if price <= 0 or price >= 1:
                continue
            n = max(1, int(budget // price))
            fee = taker_fee_order(n, price, mult)
            if fair - price - fee / n < gap:
                continue
            won = r["y"] if d == 1 else 1 - r["y"]
            rows.append({**{k: r[k] for k in ("pair_id", "fam", "ticker", "event_ticker", "h", "T", "split", "day",
                                               "k_settle")}, "dir": d, "price": price, "n": n, "fee": fee,
                         "pnl_c": 100 * (won - price - fee / n)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def twins(log=print) -> dict[str, Any]:
    from albert.research.fundamental import fit_logistic, logit

    g = twin_panel(log)
    _save(g, OUT / "twin_panel.parquet")
    res: dict[str, Any] = {"panel_rows": g.height, "pairs": g["pair_id"].n_unique()}
    # agreement and blend: does PM add information given the Kalshi mid?
    blend = []
    for (split, h), d in g.group_by(["split", "h"], maintain_order=True):
        d = d.filter(pl.col("pm").is_between(0.001, 0.999))
        X = np.column_stack([np.ones(d.height), logit(d["kmid"].to_numpy()), logit(d["pm"].to_numpy())])
        beta, tt = fit_logistic(X, d["y"].to_numpy().astype(float), d["event_ticker"].to_numpy())
        gap = (d["pm"] - d["kmid"]).abs()
        blend.append({"split": split, "h": h, "n": d.height, "events": d["event_ticker"].n_unique(),
                      "b_kalshi": round(beta[1], 3), "t_b": round(tt[1], 2), "c_pm": round(beta[2], 3), "t_c": round(tt[2], 2),
                      "median_abs_gap_c": round(100 * gap.median(), 2), "share_gap_ge_5c": round(float((gap >= 0.05).mean()), 3),
                      "brier_k": round(float(((d["kmid"] - d["y"]) ** 2).mean()), 4),
                      "brier_pm": round(float(((d["pm"] - d["y"]) ** 2).mean()), 4)})
    res["blend"] = sorted(blend, key=lambda r: (r["split"], -r["h"]))
    cells = []
    for gap in TWIN_G:
        tk = twin_taker(g, gap)
        if tk.is_empty():
            continue
        for split in ("dev", "holdout"):
            for h in TWIN_H:
                for fam in ("netflix", "billboard", "all"):
                    d = tk.filter((pl.col("split") == split) & (pl.col("h") == h))
                    if fam != "all":
                        d = d.filter(pl.col("fam") == fam)
                    cells.append({"split": split, "gap": gap, "h": h, "fam": fam, **summarize(d)})
        _save(tk, OUT / f"twin_trades_g{int(gap * 100)}.parquet")
    res["taker_cells"] = cells
    res["dev_cells_counted"] = len([c for c in cells if c["split"] == "dev"])
    # H8: xvenue's PM-led follow trades (lag 1), sliced to the last 6 h before the Kalshi close
    tr = pl.read_parquet(XV / "trades.parquet", columns=["leader", "x", "n_min", "lag", "pair_id", "event_ticker",
                                                         "family", "t_in", "pnl_settle", "date", "split"])
    tr = (tr.filter((pl.col("leader") == "pm") & (pl.col("lag") == 1) & pl.col("family").is_in(TWIN_FAMILIES))
          .join(pl.read_parquet(XV / "pairs.parquet").select("pair_id", "k_close"), on="pair_id")
          .with_columns(((pl.col("k_close").dt.epoch("s") - pl.col("t_in")) / 3600).alias("htc")))
    h8 = []
    for (split, x, nm), d in tr.group_by(["split", "x", "n_min"], maintain_order=True):
        for win, dd in (("last6h", d.filter(pl.col("htc") <= 6)), ("earlier", d.filter(pl.col("htc") > 6))):
            dd = dd.with_columns((100 * pl.col("pnl_settle")).alias("pnl_c"), pl.col("date").alias("day"))
            h8.append({"split": split, "x": x, "n_min": nm, "window": win, **summarize(dd)})
    res["h8_last6h"] = sorted(h8, key=lambda r: (r["split"], r["x"], r["n_min"], r["window"]))
    (OUT / "twins.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ---- angle 3: Polymarket family specialists ---------------------------------------------------------
PM_FAMILIES: tuple[tuple[str, str], ...] = (
    ("netflix", r"netflix"),
    ("music", r"billboard|hot-100|spotify|top-album|1-song|number-1-song|album-sales"),
    ("approval", r"approval"),
    ("youtube", r"mrbeast|youtube"),
    ("ai", r"ai-model|best-ai|top-ai|lmarena|arena"),
    ("truth", r"truth-social"),
    ("rt", r"rotten-tomatoes"),
)
PM_START = datetime(2026, 5, 1, tzinfo=UTC)
PM_MAX_VOLUME = 1e6        # markets above $1M (17 AI markets) cost > 10 pages each and are skipped
PM_MAX_PAGES = 60
WINDOWS = (("A", datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC)),
           ("B", datetime(2026, 8, 1, tzinfo=UTC), DEV_END),
           ("C", DEV_END, datetime(2026, 11, 1, tzinfo=UTC)))


def pm_family_expr(col: str = "event_slug") -> pl.Expr:
    e = pl.lit(None, dtype=pl.String)
    for fam, rx in reversed(PM_FAMILIES):
        e = pl.when(pl.col(col).str.contains("(?i)" + rx)).then(pl.lit(fam)).otherwise(e)
    return e


def pm_frame() -> pl.DataFrame:
    """Closed in-scope Polymarket binary markets of the families (the flb_wallets frame: every closed
    non-financial market with a clean winner and >= $1k volume, ending before 2026-11-01)."""
    f = pl.read_parquet("data/research/flb_wallets/tables/frame.parquet",
                        columns=["condition_id", "event_slug", "question", "end_date", "closed_time", "volume",
                                 "winner_index", "neg_risk"])
    return (f.with_columns(pm_family_expr().alias("fam"))
            .filter(pl.col("fam").is_not_null() & (pl.col("end_date") >= PM_START) & (pl.col("volume") <= PM_MAX_VOLUME)
                    & pl.col("winner_index").is_in([0, 1])))


def pm_extra() -> pl.DataFrame:
    """Polymarket twins of Kalshi markets (xvenue_leadlag exact pairs) missing from the flb frame
    (volume < $1k, or ending after 2026-10-31), with the winner index from the pair's settlement."""
    p = pl.read_parquet(XV / "pairs.parquet").filter(pl.col("family").is_in(TWIN_FAMILIES))
    pm = pl.read_parquet(XV / "pm_markets.parquet").select("condition_id", "end_date", "closed_time", "token_ids",
                                                           "event_slug")
    j = p.join(pm, on="condition_id").join(pm_frame().select("condition_id"), on="condition_id", how="anti")
    j = j.with_columns(
        pl.when(pl.col("token_ids").list.get(0) == pl.col("token")).then(0).otherwise(1).alias("tok_idx"))
    return (j.with_columns(pl.when(pl.col("pm_result") == "yes").then(pl.col("tok_idx")).otherwise(1 - pl.col("tok_idx"))
                           .cast(pl.Int64).alias("winner_index"),
                           pl.when(pl.col("family").str.starts_with("netflix")).then(pl.lit("netflix"))
                           .otherwise(pl.lit("music")).alias("fam"))
            .select("condition_id", "fam", "end_date", "winner_index", "event_slug").unique("condition_id")
            .drop_nulls(["end_date", "winner_index"]))


def wallets_fetch(workers: int = 2, extra: bool = False, log=print) -> None:
    """Complete trade history (every wallet, maker and taker rows) of each family market from the Data
    API v2 ``/trades?condition=`` (cached raw, append-only; flb_wallets' cached pages are reused).
    ``extra``: the twin markets outside the frame (``pm_extra``), stored separately."""
    import os

    os.environ.setdefault("FLB_WALLETS_REQ_INTERVAL", "0.5")
    from albert.research import flb_wallets as fw

    pm = fw.clients()
    frame = pm_extra() if extra else pm_frame()
    tag = "_extra" if extra else ""
    d = CACHE / f"pm_trades{tag}"
    d.mkdir(parents=True, exist_ok=True)
    log_path = CACHE / f"pm_trades{tag}_log.jsonl"
    done = {json.loads(x)["condition_id"] for x in log_path.read_text().splitlines() if x.strip()} if log_path.exists() else set()
    todo = [c for c in frame.sort("volume" if "volume" in frame.columns else "end_date")["condition_id"].to_list() if c not in done]
    log(f"pm trades: {frame.height} markets, {len(done)} done, {len(todo)} to fetch", flush=True)
    part = len(list(d.glob("part-*.parquet")))
    buf: list[pl.DataFrame] = []
    metas: list[dict[str, Any]] = []
    t0 = time.time()

    def one(cid: str) -> tuple[pl.DataFrame, dict[str, Any]]:
        rows, pages, trunc = fw.market_trades(pm, cid, taker_only=False, max_pages=PM_MAX_PAGES)
        return pl.DataFrame(rows, schema=fw.TRADE_SCHEMA), {"condition_id": cid, "rows": len(rows), "pages": pages,
                                                             "truncated": trunc}

    def flush() -> None:
        nonlocal buf, metas, part
        if buf:
            pl.concat(buf).write_parquet(d / f"part-{part:05d}.parquet")
            part += 1
        with log_path.open("a") as fh:
            for m in metas:
                fh.write(json.dumps(m) + "\n")
        buf, metas = [], []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (f, meta) in enumerate(ex.map(one, todo), 1):
            if f.height:
                buf.append(f)
            metas.append(meta)
            if len(metas) >= 50:
                flush()
            if i % 200 == 0:
                log(f"  {i}/{len(todo)} markets, {time.time() - t0:.0f}s", flush=True)
    flush()
    log("pm trades done")


SHRINK_K = 200.0       # $ of pseudo-risk at zero edge added to every wallet's score
MIN_RISK = 50.0        # wallets need >= $50 risked and >= 3 markets in a window to be ranked
MIN_MKTS = 3
TOP_K = 20


def wallet_rows(extra: bool = False) -> pl.LazyFrame:
    """Every fill of the fetched family markets as an edge row: a BUY of outcome k at p earns
    size x (won_k - p) on size x p risked; a SELL is the complement bought at 1 - p. ``extra`` adds
    the twin markets outside the frame (used for following, never for ranking wallets)."""
    parts = []
    for tag, fr in (("", pm_frame()), ("_extra", pm_extra() if extra else None)):
        if fr is None:
            continue
        log_path = CACHE / f"pm_trades{tag}_log.jsonl"
        meta = pl.DataFrame([json.loads(x) for x in log_path.read_text().splitlines() if x.strip()])
        ok = meta.filter(~pl.col("truncated"))["condition_id"].to_list()
        t = (pl.scan_parquet(CACHE / f"pm_trades{tag}" / "part-*.parquet")
             .select("condition_id", "wallet", "buy", "outcome_index", "price", "size", "ts")
             .filter(pl.col("condition_id").is_in(ok)))
        parts.append(t.join(fr.select("condition_id", "fam", "end_date", "winner_index", "event_slug").lazy(),
                            on="condition_id"))
    t = pl.concat(parts, how="vertical_relaxed")
    return t.with_columns(*edge_exprs(),
                          pl.when(pl.col("end_date") < WINDOWS[0][2]).then(pl.lit("A"))
                          .when(pl.col("end_date") < WINDOWS[1][2]).then(pl.lit("B")).otherwise(pl.lit("C")).alias("win"))


def edge_exprs() -> list[pl.Expr]:
    won = (pl.col("outcome_index") == pl.col("winner_index")).cast(pl.Float64)
    return [pl.when(pl.col("buy")).then(pl.col("size") * (won - pl.col("price")))
            .otherwise(pl.col("size") * (pl.col("price") - won)).alias("edge"),
            pl.when(pl.col("buy")).then(pl.col("size") * pl.col("price"))
            .otherwise(pl.col("size") * (1 - pl.col("price"))).alias("risk")]


def wallet_scores(rows: pl.LazyFrame) -> pl.DataFrame:
    """Per (family, window, wallet): $ risked, edge $, markets, PnL per $ risked and shrunk score."""
    w = (rows.group_by("fam", "win", "wallet").agg(pl.col("risk").sum(), pl.col("edge").sum(),
                                                   pl.col("condition_id").n_unique().alias("markets"),
                                                   pl.len().alias("fills"))
         .collect())
    return w.with_columns((pl.col("edge") / pl.col("risk")).alias("roi"),
                          (pl.col("edge") / (pl.col("risk") + SHRINK_K)).alias("score"))


def persistence(sc: pl.DataFrame, rows: pl.LazyFrame, a: str, b: str) -> list[dict[str, Any]]:
    """Rank correlation of shrunk scores between windows a and b, by family and pooled, and the
    out-of-sample PnL per $ of window-a top-k wallets (t clustered by market)."""
    from scipy.stats import spearmanr

    out = []
    pooled = (sc.group_by("win", "wallet").agg(pl.col("risk").sum(), pl.col("edge").sum(), pl.col("markets").sum())
              .with_columns(pl.lit("pooled").alias("fam"), (pl.col("edge") / (pl.col("risk") + SHRINK_K)).alias("score")))
    for fam, s in [*sc.partition_by("fam", as_dict=True).items(), (("pooled",), pooled)]:
        fam = fam[0] if isinstance(fam, tuple) else fam
        elig = s.filter((pl.col("risk") >= MIN_RISK) & (pl.col("markets") >= MIN_MKTS))
        sa = elig.filter(pl.col("win") == a).select("wallet", pl.col("score").alias("sa"))
        sb = elig.filter(pl.col("win") == b).select("wallet", pl.col("score").alias("sb"))
        both = sa.join(sb, on="wallet")
        rho = float(spearmanr(both["sa"], both["sb"]).statistic) if both.height >= 10 else None
        top = sa.sort("sa", descending=True).head(TOP_K)["wallet"].to_list()
        bot = sa.sort("sa").head(TOP_K)["wallet"].to_list()
        r = rows.filter(pl.col("win") == b)
        if fam != "pooled":
            r = r.filter(pl.col("fam") == fam)
        res = {"fam": fam, "a": a, "b": b, "wallets_a": sa.height, "wallets_both": both.height,
               "spearman": None if rho is None else round(rho, 3)}
        for name, ws in (("top", top), ("bottom", bot)):
            x = (r.filter(pl.col("wallet").is_in(ws)).group_by("condition_id")
                 .agg(pl.col("edge").sum(), pl.col("risk").sum()).collect())
            if x.height:
                roi = float(x["edge"].sum() / x["risk"].sum())
                # t of the $-weighted ROI with markets as clusters (ratio estimator)
                e, k = x["edge"].to_numpy(), x["risk"].to_numpy()
                u = e - roi * k
                se = math.sqrt((u ** 2).sum() * len(u) / max(len(u) - 1, 1)) / k.sum()
                res |= {f"{name}_wallets_active": r.filter(pl.col("wallet").is_in(ws)).select(pl.col("wallet").n_unique()).collect().item(),
                        f"{name}_risk_usd": round(float(k.sum()), 0), f"{name}_roi": round(roi, 4),
                        f"{name}_t": round(roi / se, 2) if se > 0 else None, f"{name}_markets": x.height}
        out.append(res)
    return out


def wallets(log=print) -> dict[str, Any]:
    rows = wallet_rows()
    sc = wallet_scores(rows)
    _save(sc, OUT / "wallet_scores.parquet")
    desc = (sc.group_by("fam", "win").agg(pl.len().alias("wallets"), pl.col("risk").sum().round(0).alias("risk_usd"),
                                          (pl.col("edge").sum() / pl.col("risk").sum()).round(4).alias("roi_all"))
            .sort("fam", "win"))
    res = {"describe": desc.to_dicts(),
           "dev_A_to_B": persistence(sc, rows, "A", "B"),
           "frozen_rule": f"top {TOP_K} wallets by shrunk score (edge / (risk + ${SHRINK_K:.0f})) among wallets with "
                          f">= ${MIN_RISK:.0f} risked and >= {MIN_MKTS} markets in the earlier window; ROI in the next window"}
    (OUT / "wallets_dev.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def wallets_holdout() -> dict[str, Any]:
    rows = wallet_rows()
    sc = pl.read_parquet(OUT / "wallet_scores.parquet")
    res = {"holdout_B_to_C": persistence(sc, rows, "B", "C")}
    (OUT / "wallets_holdout.json").write_text(json.dumps(res, indent=1, default=str))
    return res


FOLLOW_DELAY = timedelta(minutes=2)
FOLLOW_MIN_USD = 10.0
FOLLOW_DEDUPE = timedelta(hours=6)


def follow(fam: str, a: str, b: str, *, delay: timedelta = FOLLOW_DELAY, log=print) -> pl.DataFrame:
    """H10: follow the top-k wallets of family ``fam`` (ranked on window ``a``) on Kalshi.

    Signal: a top wallet's fills in window ``b`` in a Polymarket market exactly matched to a Kalshi
    market (xvenue_leadlag pairs), summed per (wallet, market, side, UTC hour); kept when >= $10.
    Side: long YES = buy Yes or sell No. Entry: the first Kalshi minute candle ending in
    [t + delay, t + 30 min] and before the Kalshi close; YES at the ask / NO at 1 - bid, spread
    <= 10c, price 3-97c, floor($4.50 / price) contracts, taker fee per order rounded up; at most
    one entry per (Kalshi market, side) per 6 h; held to settlement."""
    sc = pl.read_parquet(OUT / "wallet_scores.parquet")
    el = sc.filter((pl.col("fam") == fam) & (pl.col("win") == a) & (pl.col("risk") >= MIN_RISK)
                   & (pl.col("markets") >= MIN_MKTS))
    top = el.sort("score", descending=True).head(TOP_K)["wallet"].to_list()
    pairs = (pl.read_parquet(XV / "pairs.parquet").filter(pl.col("family").is_in(TWIN_FAMILIES))
             .select("condition_id", "token", "ticker", "event_ticker", "k_close", "k_settle", "k_result", "fee_multiplier"))
    fr = pl.read_parquet("data/research/flb_wallets/tables/frame.parquet", columns=["condition_id", "token0", "outcome0"])
    pairs = pairs.join(fr, on="condition_id", how="left").with_columns(
        (pl.col("token") == pl.col("token0")).alias("yes_is_0"))
    r = (wallet_rows(extra=True).filter((pl.col("fam") == fam) & (pl.col("win") == b) & pl.col("wallet").is_in(top))
         .collect().join(pairs.filter(pl.col("yes_is_0").is_not_null()), on="condition_id"))
    if r.is_empty():
        return pl.DataFrame()
    r = r.with_columns(
        pl.from_epoch("ts", time_unit="s").dt.replace_time_zone("UTC").alias("t"),
        ((pl.col("buy") == (pl.col("outcome_index") == 0)) == pl.col("yes_is_0")).alias("long_yes"),
        (pl.col("size") * pl.col("price")).alias("usd"))
    sig = (r.group_by("wallet", "condition_id", "long_yes", pl.col("t").dt.truncate("1h").alias("hr"))
           .agg(pl.col("t").min(), pl.col("usd").sum(), pl.col("ticker").first(), pl.col("event_ticker").first(),
                pl.col("k_close").first(), pl.col("k_settle").first(), pl.col("k_result").first(),
                pl.col("fee_multiplier").first())
           .filter((pl.col("usd") >= FOLLOW_MIN_USD) & (pl.col("t") + delay < pl.col("k_close")))
           .with_columns((pl.col("t") + delay).alias("t_lo"), (pl.col("t") + timedelta(minutes=30)).alias("t_hi")))
    win = sig.select("ticker", "t_lo", "t_hi").unique()
    kps = []
    for f in sorted((XV / "kalshi_paths").glob("part-*.parquet")):
        d = (pl.scan_parquet(f).select("ticker", "end_ts", "bid_close", "ask_close")
             .filter(pl.col("ticker").is_in(win["ticker"].unique().to_list()) & pl.col("bid_close").is_not_null()
                     & pl.col("ask_close").is_not_null()).collect())
        kps.append(d.join(win, on="ticker").filter(pl.col("end_ts").is_between(pl.col("t_lo"), pl.col("t_hi"))))
    kp = pl.concat(kps).sort("end_ts")
    q = (sig.join(kp, on=["ticker", "t_lo", "t_hi"]).sort("end_ts").group_by("wallet", "condition_id", "long_yes", "hr")
         .first().filter(pl.col("end_ts") < pl.col("k_close")))
    rows, last = [], {}
    for x in q.sort("end_ts").iter_rows(named=True):
        side = "yes" if x["long_yes"] else "no"
        key = (x["ticker"], side)
        if key in last and x["end_ts"] - last[key] < FOLLOW_DEDUPE:
            continue
        bid, ask = x["bid_close"], x["ask_close"]
        price = ask if side == "yes" else round(1 - bid, 4)
        if ask - bid > 0.10 + 1e-9 or not (0.03 <= price <= 0.97):
            continue
        last[key] = x["end_ts"]
        n = max(1, int(4.5 // price))
        fee = taker_fee_order(n, price, x["fee_multiplier"] or 1.0)
        y = int(x["k_result"] == "yes")
        won = y if side == "yes" else 1 - y
        rows.append({"ticker": x["ticker"], "event_ticker": x["event_ticker"], "wallet": x["wallet"], "t_sig": x["t"],
                     "t": x["end_ts"], "dir": 1 if side == "yes" else -1, "price": price, "n": n, "fee_order": fee,
                     "y": y, "pnl_c": 100 * (won - price - fee / n), "sig_usd": x["usd"],
                     "settlement_ts": x["k_settle"], "day": x["end_ts"].date()})
    out = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    log(f"follow {fam} {a}->{b}: {len(top)} wallets, {sig.height} signals, {out.height} entries")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="culture_flow")
    ap.add_argument("cmd")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    if a.cmd == "universe":
        universe(fetch_new=not a.no_fetch)
    elif a.cmd == "candles":
        fetch_candles(workers=a.workers)
    elif a.cmd == "panel":
        panel()
    elif a.cmd == "screen":
        screen()
    elif a.cmd == "freeze":
        freeze()
    elif a.cmd == "holdout":
        holdout()
    elif a.cmd == "maker-book":
        res = {}
        for m in (12, 6):
            for split in ("dev", "holdout"):
                o = maker_orders(split, margin_c=m)
                _save(o, OUT / f"maker_orders_m{m}_{split}.parquet")
                f = o.filter(pl.col("fill_ts").is_not_null()).with_columns(
                    (100 * ((1 - pl.col("y")) - pl.col("price"))).alias("pnl_c"), pl.col("fill_ts").alias("t_fill"),
                    pl.col("fill_ts").dt.date().alias("day"))
                res[f"m{m}_{split}"] = {"orders": o.height, "fills_1lot": summarize(f), "yes_hits": int(f["y"].sum()),
                                        "book_rest_cap_100": resting_book(o, f"makerNO_m{m}_b5_h72_rest100", split),
                                        "book_rest_cap_40": resting_book(o, f"makerNO_m{m}_b5_h72_rest40", split,
                                                                         rest_cap=40.0)}
        _save_json(res, OUT / "maker_book.json")
        print(json.dumps(res, indent=1, default=str))
    elif a.cmd == "wallets-fetch":
        wallets_fetch(workers=a.workers)
    elif a.cmd == "wallets-fetch-extra":
        wallets_fetch(workers=a.workers, extra=True)
    elif a.cmd == "wallets":
        print(json.dumps(wallets(), indent=1, default=str))
    elif a.cmd == "follow-dev":
        res = {}
        for fam in ("netflix", "music"):
            tr = follow(fam, "A", "B")
            if tr.height:
                _save(tr, OUT / f"follow_{fam}_dev.parquet")
            res[fam] = {**summarize(tr), "weeks_positive_pct": weeks_positive(tr),
                        "book": book(tr, f"follow_{fam}", "dev") if tr.height else None}
        _save_json(res, OUT / "follow_dev.json")
        print(json.dumps(res, indent=1, default=str))
    elif a.cmd == "wallets-holdout":
        print(json.dumps(wallets_holdout(), indent=1, default=str))
    elif a.cmd == "twins":
        r = twins()
        print(json.dumps({k: r[k] for k in ("panel_rows", "pairs", "blend")}, indent=1, default=str))
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
