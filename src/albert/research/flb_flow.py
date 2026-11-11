"""Trade-level flow: who pays the favourite-longshot bias on Kalshi (workstream ``flow``).

Research only: public data, no orders of any kind. Part of the ``paper_flb/`` study; shared
definitions (side price ``q``, buckets, size buckets, fees, clustered inference) come from
:mod:`albert.research.flb_common`.

Every public Kalshi print names the liquidity *taker* (``taker_side``: the side that crossed the
spread). The resting *maker* holds the opposite side of the same contracts. For a print at YES
price ``p``:

* taker bought YES  -> taker side price ``q = p``,     maker holds NO  at ``1 - q``;
* taker bought NO   -> taker side price ``q = 1 - p``, maker holds YES at ``1 - q``.

Gross of fees, the taker's and maker's PnL per contract sum to zero, so "who pays" the bias is a
question of which role holds the overpriced side, and at which prices, sizes and times.

Samples
-------
* ``rs`` (main): a stratified random sample of settled, in-scope (non-financial) Kalshi markets
  that closed 2026-08-15 .. 2026-10-31. Strata are the coarse groups, with Sports split by league.
  Events are drawn uniformly at random within a stratum (volume-blind), and every traded leg of a
  drawn event is kept (at most ``MAX_LEGS`` legs, drawn at random). Complete trade histories are
  fetched from ``/historical/trades`` (before the venue's cutoff) and ``/markets/trades``.
* ``ex`` (replication): the prints already on disk (``data/tables/kalshi_trades.parquet``, game
  windows and capped detail samples; ``data/research/vol_theta/trades``, "by date" contracts),
  minus any market in ``rs``.

Pre-resolution rule
-------------------
A print is kept only if it happened before the outcome could be known from the clock alone (rule
set from the share of volume at <= 2c / >= 98c by minutes before the close, ``t_preres``):

* every print strictly before ``close_time`` and before ``expected_expiration_time``;
* every market loses its final ``FINAL_DROP`` (30 min) before the close: in sports the share of
  volume at extreme prices rises from ~10% to 20-90% inside the last 20 minutes (games decided,
  post-final sweeps);
* non-sports markets that closed early (``close_time`` more than ``EARLY_CLOSE_GAP`` before the
  scheduled expiration: the close was triggered by the event, e.g. a speech ending, an
  announcement) lose their final ``EARLY_CLOSE_DROP`` (4 h).

The strict variant keeps only sports prints before the scheduled start and other prints more than
24 h before the close (or the scheduled expiration, if earlier).

Steps (``python -m albert.research.flb_flow run <step>``)
--------------------------------------------------------
* ``plan``    stratified sample of markets -> ``sample_markets.parquet``, ``sample_plan.json``.
* ``fetch``   complete trade histories of the sample (cached raw pages, <= 3 requests/s).
* ``build``   trade panel with q, outcome, fees, sizes, timing and markouts -> ``panel/``.
* ``analyze`` every table -> ``summary.json``, ``tables/*.csv``, ``verdict.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from albert.research import flb_common as C

ET = ZoneInfo("America/New_York")
warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
SEED = 20261111
FRAME_START = datetime(2026, 8, 15, tzinfo=UTC)
FRAME_END = datetime(2026, 11, 1, tzinfo=UTC)
MAX_LEGS = 20

EARLY_CLOSE_GAP = timedelta(minutes=30)
EARLY_CLOSE_DROP = timedelta(hours=4)      # non-sports markets closed early by their event
FINAL_DROP = timedelta(minutes=30)         # every other market

# Events drawn per stratum (uniformly at random within the stratum, volume-blind).
EVENTS_PER_STRATUM: dict[str, int] = {
    "Politics": 260, "Entertainment": 260, "Mentions": 120, "Science & Tech": 220, "Weather": 300,
    "NFL": 60, "NCAAF": 80, "MLB": 100, "NBA": 50, "WNBA": 60, "NHL": 60, "Soccer": 120,
    "Tennis": 150, "Golf": 40, "Combat": 60, "Esports": 150, "Cricket": 60, "Other sports": 150,
}

LEAGUES: tuple[tuple[str, str], ...] = (
    ("NFL", r"^KXNFL"), ("NCAAF", r"^KXNCAAF"), ("MLB", r"^KXMLB"), ("WNBA", r"^KXWNBA"),
    ("NBA", r"^KXNBA"), ("NHL", r"^KXNHL"),
    ("Tennis", r"^KX(ATP|WTA|ITF|DAVIS|BILLIE|USOPEN|WIMBLEDON|FRENCHOPEN|AUSOPEN)"),
    ("Golf", r"^KX(PGA|DPWORLD|LPGA|LIV|RYDER|GOLF|MASTERS)"),
    ("Combat", r"^KX(UFC|BOXING|MMA|PFL)"),
    ("Esports", r"^KX(CS2|LOL|DOTA2|VALORANT|R6|OW|COD|RL|EFOOTBALL|EBASKETBALL|ESOCCER)"),
    ("Cricket", r"^KX(T20|WT20|ODI|CPL|IPL|TEST|BBL|CRICKET)"),
    ("Soccer", (r"^KX(EPL|LALIGA|UCL|UEL|UECL|MLS|SERIEA|SERIEB|BUNDESLIGA|LIGUE1|LIGAMX|UEFA|ARG|"
                r"DIMAYOR|BRASILEIRO|CONCACAF|INTLFRIENDLY|EREDIVISIE|SPL|LIGAPORTUGAL|EFL|FACUP|COPA|"
                r"WC|NWSL|CHAMPIONSHIP|SUPERLIG|JLEAGUE|KLEAGUE|ALEAGUE|SAUDI|CONMEBOL|USL|FIFA)")),
)
SPORTS_LEAGUES = tuple(name for name, _ in LEAGUES) + ("Other sports",)


def league_of(series_ticker: str | None) -> str:
    for name, rx in LEAGUES:
        if re.search(rx, series_ticker or ""):
            return name
    return "Other sports"


def out_dir(s: Any) -> Path:
    d = s.data_dir / "research" / "flb_flow"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, float | np.floating):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, datetime):
        return x.isoformat()
    return x


# ---- market frame and sample ---------------------------------------------------------------------
MARKET_COLS = ("ticker", "event_ticker", "series_ticker", "category", "result", "open_time",
               "close_time", "expected_expiration_time", "occurrence_time", "settlement_ts",
               "fee_type", "fee_multiplier", "volume", "can_close_early")


def market_frame(s: Any, cols: tuple[str, ...] = MARKET_COLS) -> pl.DataFrame:
    """Settled in-scope markets with their coarse group and stratum (Sports split by league)."""
    m = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").select(cols)
         .filter(pl.col("result").is_in(["yes", "no"])).collect())
    ser = pl.read_parquet(s.tables_dir / "kalshi_series.parquet", columns=["series_ticker", "title"])
    return assign_groups(m, ser)


def draw_sample(frame: pl.DataFrame, events_per_stratum: dict[str, int], *, seed: int = SEED,
                max_legs: int = MAX_LEGS) -> pl.DataFrame:
    """Uniform random events per stratum (volume-blind), then every traded leg of each drawn event
    (a random ``max_legs`` of them when the event has more). Deterministic given ``seed``."""
    f = frame.filter((pl.col("volume") > 0) & pl.col("close_time").is_between(
        FRAME_START, FRAME_END, closed="left"))
    ev = f.select("stratum", "event_ticker").unique().sort("stratum", "event_ticker")
    ev = ev.with_columns(pl.col("event_ticker").hash(seed).alias("_h")).sort("stratum", "_h")
    ev = ev.with_columns(pl.int_range(pl.len()).over("stratum").alias("_r"))
    ev = ev.join(pl.DataFrame({"stratum": list(events_per_stratum),
                               "_cap": list(events_per_stratum.values())}), on="stratum", how="inner")
    chosen = ev.filter(pl.col("_r") < pl.col("_cap")).select("event_ticker")
    legs = (f.join(chosen, on="event_ticker", how="semi")
            .with_columns(pl.col("ticker").hash(seed + 1).alias("_h")).sort("event_ticker", "_h")
            .with_columns(pl.int_range(pl.len()).over("event_ticker").alias("_r"))
            .filter(pl.col("_r") < max_legs).drop("_h", "_r"))
    return legs.sort("stratum", "event_ticker", "ticker")


def step_plan(s: Any) -> None:
    d = out_dir(s)
    frame = market_frame(s)
    sample = draw_sample(frame, EVENTS_PER_STRATUM)
    _save(sample, d / "sample_markets.parquet")
    f = frame.filter((pl.col("volume") > 0)
                     & pl.col("close_time").is_between(FRAME_START, FRAME_END, closed="left"))
    fr = f.group_by("stratum").agg(pl.len().alias("frame_markets"),
                                   pl.col("event_ticker").n_unique().alias("frame_events"),
                                   pl.col("volume").sum().alias("frame_volume"))
    sp = sample.group_by("stratum").agg(pl.len().alias("sample_markets"),
                                        pl.col("event_ticker").n_unique().alias("sample_events"),
                                        pl.col("volume").sum().alias("sample_volume"))
    tab = fr.join(sp, on="stratum", how="left").fill_null(0).sort("stratum")
    plan = {"seed": SEED, "frame": [FRAME_START, FRAME_END], "max_legs": MAX_LEGS,
            "events_per_stratum": EVENTS_PER_STRATUM, "strata": tab.to_dicts(),
            "markets": sample.height, "events": sample["event_ticker"].n_unique()}
    (d / "sample_plan.json").write_text(json.dumps(_jsonable(plan), indent=1))
    print(tab)
    print(f"sample: {sample.height} markets, {plan['events']} events")


# ---- fetch -----------------------------------------------------------------------------------------
TRADE_SCHEMA = {"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
                "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}


def segments(open_time: datetime, close_time: datetime,
             cutoff: datetime | None) -> list[tuple[bool, datetime | None, datetime | None]]:
    """(historical?, min_ts, max_ts) request segments covering a market's whole life. Markets wholly
    on one side of the venue's trade cutoff use one unbounded request (shares the raw cache with
    earlier ingests); a market that straddles it is split at the cutoff."""
    if cutoff is None or close_time <= cutoff:
        return [(True, None, None)]
    if open_time >= cutoff:
        return [(False, None, None)]
    return [(True, None, cutoff), (False, cutoff, None)]


def step_fetch(s: Any, *, workers: int = 2, min_interval: float = 0.7) -> None:
    """Complete trade histories of the sample. ``workers`` clients, each at most
    ``1/min_interval`` requests/s (2 x 0.7 s -> <= 2.9 requests/s in total)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from albert.research.pm_hazard import _api
    from albert.venues import kalshi as k

    d = out_dir(s)
    parts = d / "trades"
    parts.mkdir(exist_ok=True)
    done_path = d / "trades_done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    sample = pl.read_parquet(d / "sample_markets.parquet")
    # cheapest strata first so every group has data early; big markets last
    todo = sample.filter(~pl.col("ticker").is_in(list(done))).sort("volume")
    local = threading.local()

    def api():
        if not hasattr(local, "api"):
            local.api = _api(s, min_interval, nap_cap=10.0, retries=60)
        return local.api

    cutoff = api().cutoff().get("trades_created_ts")
    print(f"fetch: {todo.height} markets to go ({len(done)} done), cutoff {cutoff}", flush=True)

    def one(r: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
        got: list[dict[str, Any]] = []
        for hist, a, b in segments(r["open_time"], r["close_time"], cutoff):
            try:
                for page in api().trades(r["ticker"], hist, min_ts=a, max_ts=b):
                    got.extend(k.trade_rows(page.payload))
            except RuntimeError as error:
                print(f"skip {r['ticker']}: {str(error)[:160]}", flush=True)
                return r["ticker"], None
        return r["ticker"], got

    rows: list[dict[str, Any]] = []
    fin: list[str] = []
    n_part = len(list(parts.glob("part_*.parquet")))

    def flush() -> None:
        nonlocal rows, fin, n_part
        if rows:
            df = pl.DataFrame(rows, schema=TRADE_SCHEMA).unique("trade_id")
            _save(df, parts / f"part_{n_part:05d}.parquet")
            n_part += 1
        done.update(fin)
        _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
        rows, fin = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (tk, got) in enumerate(pool.map(one, todo.iter_rows(named=True)), 1):
            if got is not None:
                rows.extend(got)
                fin.append(tk)
            if i % 200 == 0 or len(rows) > 300_000:
                flush()
                print(f"  {datetime.now(UTC):%H:%M:%S} {i}/{todo.height} markets", flush=True)
    flush()
    print("fetch done", flush=True)


# ---- market metadata ---------------------------------------------------------------------------------
GAME_STRATA = ("NFL", "NCAAF", "MLB", "NBA", "WNBA", "NHL", "Soccer", "Tennis", "Combat", "Esports",
               "Cricket", "Other sports")
SCHED_OFFSET = timedelta(hours=3)


def scheduled_start(event_ticker: str | None, occurrence: datetime | None,
                    expiration: datetime | None) -> datetime | None:
    """Scheduled start of a single-game sports event. The start encoded in the ticker when present
    (MLB, esports, cricket, ...); otherwise occurrence_time - 3 h (Kalshi's convention, exact for
    597 of 599 MLB games in ``inplay_dynamics``); an occurrence rewritten to the actual end
    (non-round seconds) is not used."""
    from albert.venues.kalshi import ticker_start

    st = ticker_start(event_ticker)
    if st is not None:
        return st
    occ_ok = (occurrence is not None and occurrence.second == 0 and occurrence.microsecond == 0
              and (expiration is None or occurrence <= expiration))
    base = occurrence if occ_ok else expiration
    return None if base is None else base - SCHED_OFFSET


def resolution_cut(close: datetime, expiration: datetime | None, *,
                   sports: bool = False) -> tuple[datetime, bool]:
    """(last admissible print time, closed early?) under the pre-resolution rule (module doc)."""
    early = expiration is not None and close < expiration - EARLY_CLOSE_GAP
    end = min(close, expiration) if expiration is not None else close
    return end - (EARLY_CLOSE_DROP if early and not sports else FINAL_DROP), early


def assign_groups(m: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    sm = (m.select("series_ticker", "category").unique()
          .join(series.rename({"title": "series_title"}), on="series_ticker", how="left"))
    sm = sm.with_columns(
        pl.struct("category", "series_ticker", "series_title").map_elements(
            lambda r: C.coarse_category(r["category"], r["series_ticker"], r["series_title"]),
            return_dtype=pl.String).alias("group"))
    sm = sm.with_columns(
        pl.when(pl.col("group") == "Sports")
        .then(pl.col("series_ticker").map_elements(league_of, return_dtype=pl.String))
        .otherwise(pl.col("group")).alias("stratum"))
    return (m.join(sm.select("series_ticker", "category", "group", "stratum"),
                   on=["series_ticker", "category"], how="left")
            .filter(pl.col("group").is_not_null()))


def market_meta(s: Any) -> pl.DataFrame:
    """Every market of both samples with outcome, fees, timing anchors and design weights."""
    d = out_dir(s)
    rs = pl.read_parquet(d / "sample_markets.parquet").with_columns(pl.lit("rs").alias("sample"))
    plan = json.loads((d / "sample_plan.json").read_text())
    w = {r["stratum"]: r["frame_events"] / r["sample_events"] for r in plan["strata"]
         if r["sample_events"]}
    # inverse inclusion probability: events drawn n_s of N_s per stratum; legs of events with more
    # than MAX_LEGS traded legs drawn MAX_LEGS of L_e
    fr = market_frame(s, ("ticker", "event_ticker", "series_ticker", "category", "result", "volume",
                          "close_time")).filter((pl.col("volume") > 0) & pl.col("close_time").is_between(
        FRAME_START, FRAME_END, closed="left"))
    legs = fr.group_by("event_ticker").agg(pl.len().alias("legs_frame"))
    rs = rs.join(legs, on="event_ticker", how="left").with_columns(
        (pl.col("stratum").replace_strict(w, default=None)
         * (pl.col("legs_frame").cast(pl.Float64) / MAX_LEGS).clip(lower_bound=1.0)).alias("w_design"))
    series = pl.read_parquet(s.tables_dir / "kalshi_series.parquet", columns=["series_ticker", "title"])
    have = set(rs["ticker"])
    t1 = (pl.scan_parquet(s.tables_dir / "kalshi_trades.parquet").select("ticker").unique()
          .collect()["ticker"])
    m1 = (pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").select(MARKET_COLS)
          .filter(pl.col("ticker").is_in(t1.implode()) & pl.col("result").is_in(["yes", "no"]))
          .collect())
    vt = s.data_dir / "research" / "vol_theta" / "trades"
    parts = [m1]
    if vt.exists():
        t2 = pl.scan_parquet(vt / "part_*.parquet").select("ticker").unique().collect()["ticker"]
        u = pl.read_parquet(s.data_dir / "research" / "pm_hazard" / "universe_all.parquet")
        parts.append(u.filter(pl.col("ticker").is_in(t2.implode())
                              & pl.col("result").is_in(["yes", "no"])
                              & ~pl.col("ticker").is_in(m1["ticker"].implode())).select(MARKET_COLS))
    ex = assign_groups(pl.concat(parts, how="vertical_relaxed").unique("ticker"), series)
    ex = ex.filter(~pl.col("ticker").is_in(list(have))).with_columns(
        pl.lit("ex").alias("sample"), pl.lit(1.0).alias("w_design"))
    cols = [*MARKET_COLS, "group", "stratum", "sample", "w_design"]
    m = pl.concat([rs.select(cols), ex.select(cols)], how="vertical_relaxed")
    rows = []
    for r in m.select("ticker", "event_ticker", "group", "stratum", "close_time",
                      "expected_expiration_time", "occurrence_time").iter_rows(named=True):
        cut, early = resolution_cut(r["close_time"], r["expected_expiration_time"],
                                    sports=r["group"] == "Sports")
        game = (r["group"] == "Sports" and r["stratum"] in GAME_STRATA
                and re.match(r"^[A-Z0-9]+-\d{2}[A-Z]{3}\d{2}", r["event_ticker"] or "") is not None)
        sched = (scheduled_start(r["event_ticker"], r["occurrence_time"],
                                 r["expected_expiration_time"]) if game else None)
        # a start that is not between 30 min and 12 h before the close is not trusted
        if sched is not None and not (timedelta(minutes=30) <= r["close_time"] - sched
                                      <= timedelta(hours=12)):
            sched = None
        exp = r["expected_expiration_time"] or r["close_time"]
        strict = sched if sched is not None else min(exp, r["close_time"]) - timedelta(hours=24)
        rows.append({"ticker": r["ticker"], "cut": cut, "early": early, "sched": sched,
                     "strict_cut": min(strict, cut) if strict is not None else None})
    extra = pl.DataFrame(rows, schema={"ticker": pl.String, "cut": pl.Datetime("us", "UTC"),
                                       "early": pl.Boolean, "sched": pl.Datetime("us", "UTC"),
                                       "strict_cut": pl.Datetime("us", "UTC")})
    m = m.join(extra, on="ticker", how="left").with_columns(
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.col("fee_multiplier").fill_null(1.0).alias("mult"),
        (pl.col("fee_type").fill_null("") == "quadratic_with_maker_fees").alias("maker_fee"),
        pl.col("expected_expiration_time").fill_null(pl.col("close_time")).alias("exp"),
    )
    # primetime: single-game sports events starting 19:00-23:59 ET
    m = m.with_columns(
        pl.when(pl.col("sched").is_not_null())
        .then(pl.col("sched").dt.convert_time_zone("America/New_York").dt.hour() >= 19)
        .otherwise(None).alias("primetime"))
    return m


# ---- trade panel -------------------------------------------------------------------------------------
STALE = timedelta(minutes=60)
TIGHT = 0.05   # markout analyses: proxy spread (last ask print - last bid print) at most 5c


def taker_orders(tr: pl.DataFrame) -> pl.DataFrame:
    """Attach a taker-order id to each print. Prints in one ticker with the same timestamp and the
    same taker side are one taker order sweeping one or more resting orders (Kalshi prints one
    fill per resting order), so trade size is measured on the order, not the fill. Input sorted by
    ticker, created_time."""
    return tr.with_columns(
        ((pl.col("ticker") != pl.col("ticker").shift(1))
         | (pl.col("created_time") != pl.col("created_time").shift(1))
         | (pl.col("taker_side") != pl.col("taker_side").shift(1))).fill_null(True)
        .cum_sum().alias("order_id"))


def order_frame(pr: pl.DataFrame) -> pl.DataFrame:
    """One row per taker order: ticker, time, direction d (+1 YES taker, -1 NO taker), contracts,
    VWAP YES price, dollar notional on the taker side, and the order's last fill price."""
    return (pr.group_by("order_id", maintain_order=True).agg(
        pl.col("ticker").first(), pl.col("created_time").first().alias("t"),
        pl.col("d").first(), pl.col("count").sum().alias("contracts"),
        ((pl.col("count") * pl.col("yes_price")).sum() / pl.col("count").sum()).alias("p_vwap"),
        (pl.col("count") * pl.col("q")).sum().alias("notional"),
        pl.col("yes_price").last().alias("p_last"), pl.len().alias("fills")))


def mid_state(orders: pl.DataFrame) -> pl.DataFrame:
    """State after each order: last YES-taker price (an ask proxy) and last NO-taker YES price (a
    bid proxy), each with its time, carried forward within the ticker."""
    return orders.select(
        "ticker", "t",
        pl.when(pl.col("d") > 0).then(pl.col("p_last")).forward_fill().over("ticker").alias("ask"),
        pl.when(pl.col("d") > 0).then(pl.col("t")).forward_fill().over("ticker").alias("ask_t"),
        pl.when(pl.col("d") < 0).then(pl.col("p_last")).forward_fill().over("ticker").alias("bid"),
        pl.when(pl.col("d") < 0).then(pl.col("t")).forward_fill().over("ticker").alias("bid_t"),
        pl.col("contracts").cum_sum().over("ticker").alias("cumv"))


def mid_expr(at: pl.Expr, max_spread: float | None = None) -> pl.Expr:
    """Mid proxy (ask + bid)/2 from a ``mid_state`` row joined at time ``at``: null unless both
    sides printed within ``STALE`` of ``at`` and the proxies are not crossed (and, if given, are
    at most ``max_spread`` apart)."""
    ok = ((at - pl.col("ask_t") <= STALE) & (at - pl.col("bid_t") <= STALE)
          & (pl.col("ask") >= pl.col("bid")))
    if max_spread is not None:
        ok = ok & (pl.col("ask") - pl.col("bid") <= max_spread + 1e-9)
    return pl.when(ok).then((pl.col("ask") + pl.col("bid")) / 2).otherwise(None)


def state_at(orders: pl.DataFrame, state: pl.DataFrame, offset: timedelta, name: str,
             *, strict_before: bool = False, max_spread: float | None = None) -> pl.DataFrame:
    """Mid proxy (and cumulative volume) at each order's time + ``offset`` (as of that instant).
    ``strict_before`` evaluates just before the order (excludes it and same-time orders)."""
    at = orders.select("order_id", "ticker", (pl.col("t") + offset).alias("_at"))
    if strict_before:
        at = at.with_columns((pl.col("_at") - timedelta(microseconds=1)).alias("_at"))
    st = state.rename({"t": "_at"}).sort("ticker", "_at")
    j = at.sort("ticker", "_at").join_asof(st, on="_at", by="ticker", strategy="backward")
    return j.select("order_id", mid_expr(pl.col("_at"), max_spread).alias(f"m_{name}"),
                    (pl.col("ask") - pl.col("bid")).alias(f"sp_{name}"),
                    pl.col("cumv").fill_null(0.0).alias(f"cumv_{name}"))


def build_chunk(tr: pl.DataFrame, meta: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Print-level panel and market-hour order-flow panel for a set of complete market histories."""
    mt = meta.select("ticker", "event_ticker", "group", "stratum", "sample", "w_design", "y",
                     "mult", "maker_fee", "open_time", "close_time", "exp", "cut", "strict_cut",
                     "sched", "early", "primetime")
    tr = (tr.unique("trade_id")
          .filter((pl.col("count") > 0) & pl.col("yes_price").is_between(0.005, 0.995))
          .join(mt.select("ticker", "close_time"), on="ticker", how="inner")
          .filter(pl.col("created_time") < pl.col("close_time")).drop("close_time")
          .sort("ticker", "created_time", "trade_id"))
    tr = tr.with_columns(
        pl.when(pl.col("taker_side") == "yes").then(1).otherwise(-1).cast(pl.Int8).alias("d"),
        pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price"))
        .otherwise(1 - pl.col("yes_price")).alias("q"))
    tr = taker_orders(tr)
    od = order_frame(tr)
    st = mid_state(od)
    first_t = od.group_by("ticker").agg(pl.col("t").min().alias("first_t"))
    pre = state_at(od, st, timedelta(0), "pre", strict_before=True)
    m1 = state_at(od, st, timedelta(minutes=1), "1m", max_spread=TIGHT)
    m60 = state_at(od, st, timedelta(hours=1), "1h", max_spread=TIGHT)
    back = state_at(od, st, -timedelta(hours=1), "back", strict_before=True)
    od = (od.join(pre, on="order_id").join(m1.select("order_id", "m_1m"), on="order_id")
          .join(m60.select("order_id", "m_1h"), on="order_id")
          .join(back.select("order_id", "cumv_back"), on="order_id").join(first_t, on="ticker"))
    # volume surge: contracts in the hour before the order vs the market's own average hourly
    # volume before that hour (past data only; >= 6 h of history)
    hist_h = (pl.col("t") - timedelta(hours=1) - pl.col("first_t")).dt.total_seconds() / 3600
    od = od.with_columns(
        (pl.col("cumv_pre") - pl.col("cumv_back")).alias("v_prev1h"),
        pl.when(hist_h >= 6).then(pl.col("cumv_back") / hist_h).otherwise(None).alias("v_base"))
    od = od.with_columns(
        pl.when(pl.col("v_base") > 0).then(pl.col("v_prev1h") / pl.col("v_base"))
        .otherwise(None).alias("surge"),
        C.size_bucket_expr("notional").alias("size_bucket"))
    keep_o = od.select("order_id", "notional", "size_bucket", "fills", "m_pre", "sp_pre", "m_1m", "m_1h",
                       "surge", "contracts")
    pr = (tr.join(keep_o, on="order_id", how="left").join(mt, on="ticker", how="left"))
    local = pl.col("created_time").dt.convert_time_zone("America/New_York")
    pr = pr.with_columns(
        (pl.col("taker_side") == "yes").alias("taker_yes"),
        pl.when(pl.col("d") > 0).then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Int8).alias("won"),
        C.bucket_expr("q"),
        (C.TAKER_RATE * pl.col("mult") * pl.col("q") * (1 - pl.col("q"))).alias("fee_t"),
        pl.when(pl.col("maker_fee")).then(C.MAKER_RATE * pl.col("mult") * pl.col("q") * (1 - pl.col("q")))
        .otherwise(0.0).alias("fee_m"),
        local.dt.hour().cast(pl.Int8).alias("hour_et"),
        (local.dt.weekday() >= 6).alias("weekend"),
        ((pl.col("exp") - pl.col("created_time")).dt.total_seconds() / 3600).alias("tte_h"),
        ((pl.col("created_time") - pl.col("open_time")).dt.total_seconds() / 3600).alias("age_h"),
        pl.when(pl.col("sched").is_not_null()).then(pl.col("created_time") >= pl.col("sched"))
        .otherwise(None).alias("inplay"),
        (pl.col("created_time") < pl.col("cut")).alias("pre_res"),
        # marks at t + h only while t + h is before the pre-resolution cut
        pl.when(pl.col("created_time") + timedelta(minutes=1) <= pl.col("cut")).then(pl.col("m_1m"))
        .otherwise(None).alias("m_1m"),
        pl.when(pl.col("created_time") + timedelta(hours=1) <= pl.col("cut")).then(pl.col("m_1h"))
        .otherwise(None).alias("m_1h"),
        pl.when(pl.col("strict_cut").is_not_null()).then(pl.col("created_time") < pl.col("strict_cut"))
        .otherwise(False).alias("strict"),
    )
    pr = pr.select(
        "sample", "ticker", "event_ticker", "group", "stratum", "w_design", "created_time", "count",
        "yes_price", "d", "taker_yes", "q", "won", "y", "bucket", "fee_t", "fee_m", "maker_fee",
        "order_id", "notional", "size_bucket", "fills", "m_pre", "sp_pre", "m_1m", "m_1h", "surge", "hour_et",
        "weekend", "tte_h", "age_h", "inplay", "primetime", "early", "pre_res", "strict")
    hours = hourly_flow(od, st, mt)
    return pr, hours


FLOW_CLASSES = {"small": ("<=$10", "$10-100"), "medium": ("$100-1k",), "large": (">$1k",)}


def hourly_flow(od: pl.DataFrame, st: pl.DataFrame, mt: pl.DataFrame) -> pl.DataFrame:
    """Market-hour panel: signed taker contracts by size class, normalised by the hour's volume;
    mid proxy at the end of the hour before, of this hour and of the next hour; outcome. Hours
    after the market's pre-resolution cut are dropped (their next-hour mid can see the outcome)."""
    od = od.join(mt.select("ticker", "cut"), on="ticker").filter(pl.col("t") < pl.col("cut"))
    if od.is_empty():
        return pl.DataFrame()
    cls = (pl.when(pl.col("size_bucket").is_in(FLOW_CLASSES["small"])).then(pl.lit("small"))
           .when(pl.col("size_bucket").is_in(FLOW_CLASSES["medium"])).then(pl.lit("medium"))
           .otherwise(pl.lit("large")))
    od = od.with_columns(pl.col("t").dt.truncate("1h").alias("h"), cls.alias("cls"),
                         (pl.col("d").cast(pl.Float64) * pl.col("contracts")).alias("sv"))
    hv = od.group_by("ticker", "h").agg(
        pl.col("contracts").sum().alias("vol"),
        *[pl.col("sv").filter(pl.col("cls") == c).sum().alias(f"sv_{c}") for c in FLOW_CLASSES],
        *[pl.col("contracts").filter(pl.col("cls") == c).sum().alias(f"v_{c}") for c in FLOW_CLASSES],
        pl.len().alias("orders"))
    stt = st.rename({"t": "_at"}).sort("ticker", "_at")

    def mid_at(offset: timedelta, name: str) -> pl.DataFrame:
        at = hv.select("ticker", "h", (pl.col("h") + offset - timedelta(microseconds=1)).alias("_at"))
        j = at.sort("ticker", "_at").join_asof(stt, on="_at", by="ticker", strategy="backward")
        return j.select("ticker", "h", mid_expr(pl.col("_at"), TIGHT).alias(name))

    hv = (hv.join(mid_at(timedelta(0), "m_start"), on=["ticker", "h"])
          .join(mid_at(timedelta(hours=1), "m_end"), on=["ticker", "h"])
          .join(mid_at(timedelta(hours=2), "m_next"), on=["ticker", "h"]))
    hv = hv.join(mt.select("ticker", "event_ticker", "group", "stratum", "sample", "w_design", "y",
                           "cut"), on="ticker")
    # a mid is used only if its hour ends before the cut (it may otherwise reflect later prints)
    return hv.with_columns(
        pl.when(pl.col("h") + timedelta(hours=1) <= pl.col("cut")).then(pl.col("m_end"))
        .otherwise(None).alias("m_end"),
        pl.when(pl.col("h") + timedelta(hours=2) <= pl.col("cut")).then(pl.col("m_next"))
        .otherwise(None).alias("m_next")).drop("cut").sort("ticker", "h")


def _trade_sources(s: Any) -> dict[str, list[Path]]:
    d = out_dir(s)
    vt = s.data_dir / "research" / "vol_theta" / "trades"
    return {"rs": sorted((d / "trades").glob("part_*.parquet")),
            "ex": [s.tables_dir / "kalshi_trades.parquet", *sorted(vt.glob("part_*.parquet"))]}


def step_build(s: Any, *, chunk_prints: int = 300_000, samples: tuple[str, ...] = ("rs", "ex")) -> None:
    d = out_dir(s)
    meta = market_meta(s)
    _save(meta, d / "markets.parquet")
    for sub in ("panel", "hours"):
        (d / sub).mkdir(exist_ok=True)
        for f in (d / sub).glob("*.parquet"):
            if f.name.split("_")[0] in samples:
                f.unlink()
    for sample, files in _trade_sources(s).items():
        if not files or sample not in samples:
            continue
        tick = meta.filter(pl.col("sample") == sample)["ticker"]
        lf = pl.scan_parquet(files, schema=TRADE_SCHEMA).filter(pl.col("ticker").is_in(tick.implode()))
        counts = lf.group_by("ticker").agg(pl.len().alias("n")).collect().sort("ticker")
        chunks, cur, n = [], [], 0
        for tk, k in counts.iter_rows():
            cur.append(tk)
            n += k
            if n >= chunk_prints:
                chunks.append(cur)
                cur, n = [], 0
        if cur:
            chunks.append(cur)
        print(f"build {sample}: {counts['n'].sum()} prints, {counts.height} markets, "
              f"{len(chunks)} chunks", flush=True)
        for i, ch in enumerate(chunks):
            tr = lf.filter(pl.col("ticker").is_in(ch)).collect()
            pr, hv = build_chunk(tr, meta.filter(pl.col("ticker").is_in(ch)))
            base = (0 if sample == "rs" else 500) + i
            pr = pr.with_columns((pl.col("order_id") + base * 10**9).alias("order_id"))
            _save(pr, d / "panel" / f"{sample}_{i:03d}.parquet")
            if not hv.is_empty():
                _save(hv, d / "hours" / f"{sample}_{i:03d}.parquet")
            print(f"  {sample} chunk {i + 1}/{len(chunks)}: {pr.height} prints", flush=True)


# ---- inference from per-cluster sums -----------------------------------------------------------------
Metric = tuple[pl.Expr, pl.Expr]   # (x, weight)


def _valid(x: pl.Expr, w: pl.Expr) -> pl.Expr:
    return x.is_not_null() & x.is_finite() & w.is_not_null() & (w > 0)


def cell_stats(lf: pl.LazyFrame | pl.DataFrame, by: list[str], metrics: dict[str, Metric], *,
               cluster: str = "event_ticker", sums: dict[str, pl.Expr] | None = None) -> pl.DataFrame:
    """Weighted mean, cluster-robust SE and t of each metric in each ``by`` cell. Numerically the
    same as :func:`flb_common.clustered` on the cell's rows, but computed from per-cluster sums
    (sum w x, sum w), so the trade panel is never held in memory. ``sums`` adds plain cell totals."""
    lf = lf.lazy()
    key = list(by) or ["_all"]
    if not by:
        lf = lf.with_columns(pl.lit(1).alias("_all"))
    aggs: list[pl.Expr] = []
    for k, (x, w) in metrics.items():
        ok = _valid(x, w)
        aggs += [(w * x).filter(ok).sum().alias(f"_sx_{k}"), w.filter(ok).sum().alias(f"_sw_{k}"),
                 ok.sum().alias(f"_n_{k}")]
    aggs += [e.alias(n) for n, e in (sums or {}).items()]
    g = lf.group_by([*key, cluster]).agg(aggs).collect(engine="streaming")
    out = g.group_by(key).agg([pl.col(n).sum() for n in (sums or {})])
    for k in metrics:
        sx, sw, n = f"_sx_{k}", f"_sw_{k}", f"_n_{k}"
        g = g.with_columns((pl.col(sx).sum().over(key) / pl.col(sw).sum().over(key)).alias("_mu"))
        a = g.group_by(key).agg(
            pl.col(n).sum().alias(f"{k}_n"), (pl.col(n) > 0).sum().alias(f"{k}_clusters"),
            pl.col("_mu").first().alias(k), pl.col(sw).sum().alias("_W"),
            ((pl.col(sx) - pl.col("_mu") * pl.col(sw)) ** 2).sum().alias("_ss"))
        kc = pl.col(f"{k}_clusters").cast(pl.Float64)
        a = a.with_columns(pl.when(kc > 1).then((pl.col("_ss") * kc / (kc - 1)).sqrt() / pl.col("_W"))
                           .otherwise(None).alias(f"{k}_se")).drop("_ss", "_W")
        a = a.with_columns((pl.col(k) / pl.col(f"{k}_se")).alias(f"{k}_t"))
        out = out.join(a, on=key, how="full", coalesce=True)
    out = out.drop("_all") if not by else out.sort(key)
    return out


def diff_stats(lf: pl.LazyFrame | pl.DataFrame, by: list[str], flag: pl.Expr, x: pl.Expr,
               w: pl.Expr, *, cluster: str = "event_ticker") -> pl.DataFrame:
    """Difference of weighted means (flag true minus flag false) per ``by`` cell with a
    cluster-robust SE from the per-cluster influence functions of the two means."""
    lf = lf.lazy()
    key = list(by) or ["_all"]
    if not by:
        lf = lf.with_columns(pl.lit(1).alias("_all"))
    ok = _valid(x, w) & flag.is_not_null()
    g = (lf.filter(ok).with_columns(flag.alias("_f"), (w * x).alias("_wx"), w.alias("_w"))
         .group_by([*key, cluster]).agg(
             pl.col("_wx").filter(pl.col("_f")).sum().alias("sxa"),
             pl.col("_w").filter(pl.col("_f")).sum().alias("swa"),
             pl.col("_wx").filter(~pl.col("_f")).sum().alias("sxb"),
             pl.col("_w").filter(~pl.col("_f")).sum().alias("swb"),
             pl.col("_f").sum().alias("na"), (~pl.col("_f")).sum().alias("nb")).collect(engine="streaming"))
    g = g.with_columns((pl.col("sxa").sum().over(key) / pl.col("swa").sum().over(key)).alias("mua"),
                       (pl.col("sxb").sum().over(key) / pl.col("swb").sum().over(key)).alias("mub"),
                       pl.col("swa").sum().over(key).alias("wa"), pl.col("swb").sum().over(key).alias("wb"))
    g = g.with_columns(((pl.col("sxa") - pl.col("mua") * pl.col("swa")) / pl.col("wa")
                        - (pl.col("sxb") - pl.col("mub") * pl.col("swb")) / pl.col("wb")).alias("psi"))
    a = g.group_by(key).agg(
        pl.col("mua").first().alias("mean_a"), pl.col("mub").first().alias("mean_b"),
        pl.col("na").sum().alias("n_a"), pl.col("nb").sum().alias("n_b"), pl.len().alias("clusters"),
        (pl.col("psi") ** 2).sum().alias("_ss"))
    k = pl.col("clusters").cast(pl.Float64)
    a = a.with_columns((pl.col("mean_a") - pl.col("mean_b")).alias("diff"),
                       pl.when(k > 1).then((pl.col("_ss") * k / (k - 1)).sqrt()).otherwise(None).alias("se"))
    a = a.with_columns((pl.col("diff") / pl.col("se")).alias("t")).drop("_ss")
    return a.drop("_all") if not by else a.sort(key)


def ols_cluster(X: np.ndarray, y: np.ndarray, g: np.ndarray,
                names: list[str]) -> dict[str, dict[str, float | None]]:
    """OLS with standard errors clustered by ``g`` (CR1 small-sample factor)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y, g = X[ok], y[ok], np.asarray(g)[ok]
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    e = y - X @ beta
    _, inv = np.unique(g, return_inverse=True)
    G = int(inv.max()) + 1
    sc = np.zeros((G, k))
    np.add.at(sc, inv, X * e[:, None])
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * bread @ (sc.T @ sc) @ bread
    se = np.sqrt(np.diag(V))
    out = {nm: {"coef": float(b), "se": float(s_), "t": float(b / s_) if s_ > 0 else None}
           for nm, b, s_ in zip(names, beta, se, strict=True)}
    out["_n"] = {"coef": float(n), "se": None, "t": None}
    out["_clusters"] = {"coef": float(G), "se": None, "t": None}
    return out


# ---- analysis ----------------------------------------------------------------------------------------
LS_Q, FAV_Q = C.LONGSHOT + 1e-9, C.FAVOURITE - 1e-9
SIZES = C.SIZE_LABELS
NONSPORT = ("Weather", "Politics", "Entertainment", "Mentions", "Science & Tech", "Other")


def positions(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Two positions per print: the taker's side at q and the maker's opposite side at 1 - q."""
    tk = lf.with_columns(pl.lit("taker").alias("role"), pl.col("q").alias("qo"),
                         pl.col("won").cast(pl.Float64).alias("wo"), pl.col("fee_t").alias("fo"))
    mk = lf.with_columns(pl.lit("maker").alias("role"), (1 - pl.col("q")).alias("qo"),
                         (1 - pl.col("won").cast(pl.Float64)).alias("wo"), pl.col("fee_m").alias("fo"))
    return pl.concat([tk, mk]).with_columns(C.bucket_expr("qo").alias("bucket_o"))


def pos_metrics(wd: pl.Expr | None = None) -> dict[str, Metric]:
    """Per-contract and per-dollar returns of a position (own price qo, payout wo, fee fo)."""
    wd = pl.col("w_design") if wd is None else wd
    n = pl.col("count")
    q, w, f = pl.col("qo"), pl.col("wo"), pl.col("fo")
    return {"c_gross": (w - q, wd * n), "c_net": (w - q - f, wd * n),
            "d_gross": ((w - q) / q, wd * n * q), "d_net": ((w - q - f) / (q + f), wd * n * (q + f))}


def taker_metrics(wd: pl.Expr | None = None) -> dict[str, Metric]:
    wd = pl.col("w_design") if wd is None else wd
    n = pl.col("count")
    q, w, f = pl.col("q"), pl.col("won").cast(pl.Float64), pl.col("fee_t")
    return {"c_gross": (w - q, wd * n), "c_net": (w - q - f, wd * n),
            "d_gross": ((w - q) / q, wd * n * q), "d_net": ((w - q - f) / (q + f), wd * n * (q + f)),
            "p_net": ((w - q - f) / (q + f), wd)}   # print-weighted (each fill counts once)


def _sums(wd: pl.Expr | None = None) -> dict[str, pl.Expr]:
    wd = pl.col("w_design") if wd is None else wd
    return {"prints": pl.len(), "orders": (1.0 / pl.col("fills")).sum(),
            "contracts": pl.col("count").sum(), "dollars": (pl.col("count") * pl.col("q")).sum(),
            "w_contracts": (wd * pl.col("count")).sum(),
            "w_dollars": (wd * pl.col("count") * pl.col("q")).sum()}


def _with_all(fn, by: list[str], col: str = "group") -> pl.DataFrame:
    a = fn([col, *by])
    b = fn(by).with_columns(pl.lit("All").alias(col))
    return pl.concat([a, b.select(a.columns)], how="vertical_relaxed")


def _bucket_order(df: pl.DataFrame, col: str = "bucket") -> pl.DataFrame:
    order = {b: i for i, b in enumerate(C.BUCKET_LABELS)}
    return df.with_columns(pl.col(col).replace_strict(order, default=99).alias("_o")).sort(
        [c for c in df.columns if c in ("sample", "group", "role", "size_bucket", "taker_yes")] + ["_o"]
    ).drop("_o")


def load_panel(s: Any, sample: str = "rs") -> pl.LazyFrame:
    d = out_dir(s)
    lf = pl.scan_parquet(d / "panel" / f"{sample}_*.parquet")
    mk = pl.read_parquet(d / "markets.parquet", columns=["ticker", "close_time", "volume", "event_ticker",
                                                         "stratum", "sample"]).filter(pl.col("sample") == sample)
    ev = (mk.group_by("event_ticker", "stratum").agg(pl.col("volume").sum().alias("ev_vol"))
          .with_columns((pl.col("ev_vol") >= pl.col("ev_vol").quantile(0.9).over("stratum"))
                        .alias("marquee")).select("event_ticker", "marquee"))
    lf = lf.join(mk.select("ticker", "close_time").lazy(), on="ticker", how="left")
    lf = lf.join(ev.lazy(), on="event_ticker", how="left")
    return lf.with_columns(
        pl.when(pl.col("group") == "Sports").then(pl.lit("Sports")).otherwise(pl.lit("Non-sports"))
        .alias("sector"),
        pl.when(pl.col("hour_et") < 6).then(pl.lit("00-06")).when(pl.col("hour_et") < 12)
        .then(pl.lit("06-12")).when(pl.col("hour_et") < 18).then(pl.lit("12-18"))
        .otherwise(pl.lit("18-24")).alias("hour_block"),
        pl.when(pl.col("tte_h") < 1).then(pl.lit("a <1h")).when(pl.col("tte_h") < 6).then(pl.lit("b 1-6h"))
        .when(pl.col("tte_h") < 24).then(pl.lit("c 6-24h")).when(pl.col("tte_h") < 168)
        .then(pl.lit("d 1-7d")).otherwise(pl.lit("e >7d")).alias("tte"),
        (pl.col("age_h") < 24).alias("first24h"),
        pl.when(pl.col("surge").is_null()).then(None).when(pl.col("surge") < 0.5).then(pl.lit("a quiet"))
        .when(pl.col("surge") < 3).then(pl.lit("b normal")).when(pl.col("surge") < 10)
        .then(pl.lit("c surge 3-10x")).otherwise(pl.lit("d spike >=10x")).alias("surge_cls"),
        (pl.col("q") <= LS_Q).alias("ls"), (pl.col("q") >= FAV_Q).alias("fav"),
    )


def t_coverage(s: Any) -> pl.DataFrame:
    """Counts by sample and group after the pre-resolution filter, and completeness (contracts
    printed over the market's reported volume). Aggregated per market first (streaming), then by
    group on the small per-market frame."""
    d = out_dir(s)
    mk = pl.read_parquet(d / "markets.parquet", columns=["ticker", "event_ticker", "group", "volume"])
    rows = []
    for sample in ("rs", "ex"):
        files = sorted((d / "panel").glob(f"{sample}_*.parquet"))
        if not files:
            continue
        pre = pl.col("pre_res")
        per = (pl.scan_parquet(files).group_by("ticker").agg(
            pl.len().alias("prints_all"), pre.sum().alias("prints"),
            (1.0 / pl.col("fills")).filter(pre).sum().alias("orders"),
            pl.col("count").filter(pre).sum().alias("contracts"),
            (pl.col("count") * pl.col("q")).filter(pre).sum().alias("dollars"),
            pl.col("count").sum().alias("printed"),
            pl.col("created_time").min().alias("first"), pl.col("created_time").max().alias("last"))
               .collect(engine="streaming").join(mk, on="ticker", how="left"))
        per = per.with_columns((pl.col("printed") / pl.col("volume")).alias("ratio"))
        a = per.group_by("group").agg(
            pl.col("prints_all").sum(), pl.col("prints").sum(), pl.col("orders").sum(),
            pl.col("contracts").sum(), pl.col("dollars").sum(),
            pl.col("ticker").filter(pl.col("prints") > 0).n_unique().alias("markets"),
            pl.col("event_ticker").filter(pl.col("prints") > 0).n_unique().alias("events"),
            pl.col("first").min(), pl.col("last").max(),
            pl.col("ratio").median().alias("completeness_median"),
            (pl.col("ratio") > 0.95).mean().alias("complete_share"))
        rows.append(a.with_columns(pl.lit(sample).alias("sample")))
    return pl.concat(rows, how="vertical_relaxed").sort("sample", "group")


def t_roles(lf: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Taker against maker: returns by own-side price bucket and role, overall by role, and who
    holds the longshot side of longshot prints."""
    pos = positions(lf)
    m = pos_metrics()
    sums = {"contracts": pl.col("count").sum(), "w_contracts": (pl.col("w_design") * pl.col("count")).sum(),
            "w_dollars": (pl.col("w_design") * pl.col("count") * pl.col("qo")).sum()}
    by_bucket = _bucket_order(_with_all(lambda by: cell_stats(pos, by, m, sums=sums),
                                        ["role", "bucket_o"]).rename({"bucket_o": "bucket"}))
    overall = _with_all(lambda by: cell_stats(pos, by, m, sums=sums), ["role"])
    # longshot / favourite positions by role
    side = pos.with_columns(pl.when(pl.col("qo") <= LS_Q).then(pl.lit("longshot"))
                            .when(pl.col("qo") >= FAV_Q).then(pl.lit("favourite"))
                            .otherwise(pl.lit("middle")).alias("side"))
    by_side = _with_all(lambda by: cell_stats(side, by, m, sums=sums), ["side", "role"])
    # taker minus maker (per dollar, net) by group: both positions of the same prints
    diff = _with_all(lambda by: diff_stats(pos, by, pl.col("role") == "taker", m["d_net"][0],
                                           m["d_net"][1]), [])
    diff_ls = _with_all(lambda by: diff_stats(side.filter(pl.col("side") == "longshot"), by,
                                              pl.col("role") == "taker", m["d_net"][0], m["d_net"][1]), [])
    # pooled non-sports (design-weighted) rows, labelled as a group
    ns = pl.col("sector") == "Non-sports"
    nso = cell_stats(pos.filter(ns), ["role"], m, sums=sums).with_columns(pl.lit("Non-sports").alias("group"))
    nss = cell_stats(side.filter(ns), ["side", "role"], m, sums=sums).with_columns(
        pl.lit("Non-sports").alias("group"))
    nsd = diff_stats(pos.filter(ns), [], pl.col("role") == "taker", *m["d_net"]).with_columns(
        pl.lit("Non-sports").alias("group"))
    overall = pl.concat([overall, nso.select(overall.columns)], how="vertical_relaxed")
    by_side = pl.concat([by_side, nss.select(by_side.columns)], how="vertical_relaxed")
    diff = pl.concat([diff, nsd.select(diff.columns)], how="vertical_relaxed")
    strata = _bucket_order(cell_stats(pos, ["stratum", "role", "bucket_o"], {"d_net": m["d_net"]})
                           .rename({"bucket_o": "bucket"}))
    return {"roles_bucket": by_bucket, "roles_overall": overall, "roles_side": by_side,
            "roles_diff": diff, "roles_diff_longshot": diff_ls, "roles_bucket_strata": strata}


def t_size(lf: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Trade size (taker-order notional): longshot share of volume and taker returns by size."""
    wd = pl.col("w_design")
    share = {"ls_contracts": (pl.col("ls").cast(pl.Float64), wd * pl.col("count")),
             "ls_dollars": (pl.col("ls").cast(pl.Float64), wd * pl.col("count") * pl.col("q")),
             "ls_orders": (pl.col("ls").cast(pl.Float64), wd / pl.col("fills")),
             "fav_contracts": (pl.col("fav").cast(pl.Float64), wd * pl.col("count")),
             "mean_q": (pl.col("q"), wd * pl.col("count"))}
    shares = _with_all(lambda by: cell_stats(lf, by, share, sums=_sums()), ["size_bucket"])
    shares_sector = cell_stats(lf, ["sector", "size_bucket"], share, sums=_sums())
    m = taker_metrics()
    ret = _bucket_order(_with_all(lambda by: cell_stats(lf, by, m, sums=_sums()), ["size_bucket", "bucket"]))
    ret_sector = _bucket_order(cell_stats(lf, ["sector", "size_bucket", "bucket"], m, sums=_sums()))
    side = lf.with_columns(pl.when(pl.col("ls")).then(pl.lit("longshot")).when(pl.col("fav"))
                           .then(pl.lit("favourite")).otherwise(pl.lit("middle")).alias("side"))
    ret_side = _with_all(lambda by: cell_stats(side, by, m, sums=_sums()), ["side", "size_bucket"])
    ret_side_sector = cell_stats(side, ["sector", "side", "size_bucket"], m, sums=_sums())
    ends = lf.filter(pl.col("size_bucket").is_in([SIZES[0], SIZES[3]]))
    small_large = _with_all(lambda by: diff_stats(ends.filter(pl.col("ls")), by,
                                                  pl.col("size_bucket") == SIZES[0], *m["d_net"]), [])
    small_large_all = _with_all(lambda by: diff_stats(ends, by, pl.col("size_bucket") == SIZES[0],
                                                      *m["d_net"]), [])
    side_strata = cell_stats(side, ["stratum", "side", "size_bucket"], {"d_net": m["d_net"]})
    ret_strata = _bucket_order(cell_stats(lf, ["stratum", "size_bucket", "bucket"], {"d_net": m["d_net"]}))
    return {"size_side_strata": side_strata, "size_returns_strata": ret_strata,
            "size_shares": shares, "size_shares_sector": shares_sector, "size_returns": ret,
            "size_returns_sector": ret_sector, "size_side": ret_side, "size_side_sector": ret_side_sector,
            "size_small_vs_large_longshot": small_large, "size_small_vs_large_all": small_large_all}


def t_yesno(lf: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    m = taker_metrics()
    cells = _bucket_order(_with_all(lambda by: cell_stats(lf, by, m, sums=_sums()), ["taker_yes", "bucket"]))
    diff = _bucket_order(_with_all(lambda by: diff_stats(lf, by, pl.col("taker_yes"), *m["d_net"]),
                                   ["bucket"]))
    diff_c = _bucket_order(_with_all(lambda by: diff_stats(lf, by, pl.col("taker_yes"), *m["c_net"]),
                                     ["bucket"]))
    # bucket-adjusted difference: YES-taker minus NO-taker, both reweighted to the same q-mix is
    # approximated by pooling the within-bucket cells; reported per side class
    side = lf.with_columns(pl.when(pl.col("ls")).then(pl.lit("longshot")).when(pl.col("fav"))
                           .then(pl.lit("favourite")).otherwise(pl.lit("middle")).alias("side"))
    diff_side = _with_all(lambda by: diff_stats(side, by, pl.col("taker_yes"), *m["d_net"]), ["side"])
    share = _with_all(lambda by: cell_stats(lf, by, {"yes_share": (pl.col("taker_yes").cast(pl.Float64),
                                                                    pl.col("w_design") * pl.col("count"))}),
                      ["bucket"])
    ystrata = _bucket_order(cell_stats(lf, ["stratum", "taker_yes", "bucket"], {"d_net": m["d_net"]}))
    return {"yesno_strata": ystrata, "yesno_cells": cells, "yesno_diff": diff, "yesno_diff_contract": diff_c,
            "yesno_diff_side": diff_side, "yesno_share": _bucket_order(share)}


TIMING_DIMS = ("hour_et", "hour_block", "weekend", "inplay", "tte", "first24h", "surge_cls",
               "primetime", "marquee")


def t_timing(lf: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Taker returns, longshot share and small-order share by timing / attention cell."""
    m = taker_metrics()
    wd = pl.col("w_design")
    extra = {"ls_share": (pl.col("ls").cast(pl.Float64), wd * pl.col("count")),
             "small_share": (pl.col("size_bucket").is_in([SIZES[0], SIZES[1]]).cast(pl.Float64),
                             wd * pl.col("count")),
             "small_order_share": (pl.col("size_bucket").is_in([SIZES[0], SIZES[1]]).cast(pl.Float64),
                                   wd / pl.col("fills"))}
    side = lf.with_columns(pl.when(pl.col("ls")).then(pl.lit("longshot")).when(pl.col("fav"))
                           .then(pl.lit("favourite")).otherwise(pl.lit("middle")).alias("side"))
    out: dict[str, pl.DataFrame] = {}
    rows = []
    for dim in TIMING_DIMS:
        sub = lf.filter(pl.col(dim).is_not_null())
        a = cell_stats(sub, ["sector", dim], {**m, **extra}, sums=_sums())
        b = cell_stats(sub, [dim], {**m, **extra}, sums=_sums()).with_columns(pl.lit("All").alias("sector"))
        c = cell_stats(side.filter(pl.col(dim).is_not_null()), ["side", dim], m, sums=_sums())
        tab = pl.concat([a, b.select(a.columns)], how="vertical_relaxed").with_columns(
            pl.col(dim).cast(pl.String).alias("cell")).drop(dim)
        rows.append(tab.with_columns(pl.lit(dim).alias("dim")))
        out[f"timing_side_{dim}"] = c.with_columns(pl.col(dim).cast(pl.String).alias("cell")).drop(dim)
    out["timing"] = pl.concat(rows, how="vertical_relaxed")
    # high- against low-attention contrasts (taker net return per dollar)
    flags = {"evening_18_24": pl.col("hour_block") == "18-24", "weekend": pl.col("weekend"),
             "inplay": pl.col("inplay"), "first24h": pl.col("first24h"),
             "surge_3x": pl.col("surge") >= 3, "primetime": pl.col("primetime"),
             "marquee": pl.col("marquee"), "last_hour": pl.col("tte_h") < 1}
    con = []
    for name, f in flags.items():
        for lab, sub in (("all", side), ("longshot", side.filter(pl.col("side") == "longshot")),
                         ("favourite", side.filter(pl.col("side") == "favourite"))):
            for sector, ss in (("All", sub), ("Sports", sub.filter(pl.col("sector") == "Sports")),
                               ("Non-sports", sub.filter(pl.col("sector") == "Non-sports"))):
                dd = diff_stats(ss.filter(f.is_not_null()), [], f, *m["d_net"])
                con.append(dd.with_columns(pl.lit(name).alias("contrast"), pl.lit(lab).alias("side"),
                                           pl.lit(sector).alias("sector")))
        for lab, x, w in (("ls_share", pl.col("ls").cast(pl.Float64), wd * pl.col("count")),
                          ("small_order_share", pl.col("size_bucket").is_in([SIZES[0], SIZES[1]]).cast(pl.Float64),
                           wd / pl.col("fills"))):
            dd = diff_stats(lf.filter(f.is_not_null()), [], f, x, w)
            con.append(dd.with_columns(pl.lit(name).alias("contrast"), pl.lit(lab).alias("side"),
                                       pl.lit("All").alias("sector")))
    out["timing_contrasts"] = pl.concat(con, how="vertical_relaxed")
    return out


def t_markouts(lf: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Per-contract markouts of taker prints (cents, taker's view) by order size: effective
    half-spread, price impact at 1 min and 1 h, markouts at 1 min, 1 h and to settlement."""
    d = pl.col("d").cast(pl.Float64)
    p = pl.col("yes_price")
    have = pl.col("m_pre").is_not_null() & pl.col("m_1m").is_not_null() & pl.col("m_1h").is_not_null()
    tight = pl.col("sp_pre") <= TIGHT + 1e-9
    sub = lf.filter(have & tight)
    w = pl.col("w_design") * pl.col("count")
    wo = pl.col("w_design") / pl.col("fills")
    met = {"half_spread": (100 * d * (p - pl.col("m_pre")), w),
           "impact_1m": (100 * d * (pl.col("m_1m") - pl.col("m_pre")), w),
           "impact_1h": (100 * d * (pl.col("m_1h") - pl.col("m_pre")), w),
           "markout_1m": (100 * d * (pl.col("m_1m") - p), w),
           "markout_1h": (100 * d * (pl.col("m_1h") - p), w),
           "markout_settle": (100 * d * (pl.col("y").cast(pl.Float64) - p), w),
           "drift_1h_settle": (100 * d * (pl.col("y").cast(pl.Float64) - pl.col("m_1h")), w),
           "impact_1h_order": (100 * d * (pl.col("m_1h") - pl.col("m_pre")), wo),
           "markout_settle_order": (100 * d * (pl.col("y").cast(pl.Float64) - p), wo)}
    a = _with_all(lambda by: cell_stats(sub, by, met, sums=_sums()), ["size_bucket"], col="sector")
    mstrata = cell_stats(sub, ["stratum", "size_bucket"], met)
    loose = _with_all(lambda by: cell_stats(lf.filter(have), by, met, sums=_sums()), ["size_bucket"],
                      col="sector")
    excl = sub.filter(pl.col("q").is_between(0.05, 0.95))
    b = _with_all(lambda by: cell_stats(excl, by, met, sums=_sums()), ["size_bucket"], col="sector")
    cov = lf.select(have.mean().alias("share_with_mids"), (have & tight).mean().alias("share_tight"),
                    (pl.col("count") * (have & tight)).sum() / pl.col("count").sum()).collect(engine="streaming")
    # decomposition of the taker's gross loss per contract: d(Y - p) = d(Y - m_pre) - s
    dec = {"half_spread": met["half_spread"], "markout_settle": met["markout_settle"],
           "mid_to_settle": (100 * d * (pl.col("y").cast(pl.Float64) - pl.col("m_pre")), w),
           "fee_c": (100 * pl.col("fee_t"), w), "q_c": (100 * pl.col("q"), w),
           "immediacy_per_dollar": ((d * (p - pl.col("m_pre")) + pl.col("fee_t")) / (pl.col("q") + pl.col("fee_t")),
                                    pl.col("w_design") * pl.col("count") * (pl.col("q") + pl.col("fee_t")))}
    side = (lf.filter(pl.col("m_pre").is_not_null() & tight)
            .with_columns(pl.when(pl.col("ls")).then(pl.lit("longshot")).when(pl.col("fav"))
                          .then(pl.lit("favourite")).otherwise(pl.lit("middle")).alias("side")))
    c = _with_all(lambda by: cell_stats(side, by, dec, sums=_sums()), ["side"], col="sector")
    c2 = _with_all(lambda by: cell_stats(side, by, dec, sums=_sums()), ["side", "size_bucket"], col="sector")
    return {"markouts": a, "markouts_strata": mstrata, "markouts_mid_prices": b, "markouts_all_proxies": loose,
            "markouts_coverage": cov,
            "decomp_side": c, "decomp_side_size": c2}


def t_hourly(s: Any, sample: str = "rs") -> dict[str, Any]:
    """Market-hour regressions of later mid changes on this hour's signed taker imbalance by size
    class (imbalance = signed contracts of the class / all contracts that hour), clustered by event."""
    files = sorted((out_dir(s) / "hours").glob(f"{sample}_*.parquet"))
    if not files:
        return {}
    h = pl.read_parquet(files).filter(pl.col("vol") > 0).with_columns(
        *[(pl.col(f"sv_{c}") / pl.col("vol")).alias(f"oib_{c}") for c in FLOW_CLASSES],
        (pl.col("m_end") - pl.col("m_start")).alias("r_now"),
        (pl.col("m_next") - pl.col("m_end")).alias("r_next"),
        (pl.col("y").cast(pl.Float64) - pl.col("m_end")).alias("r_settle"),
        pl.when(pl.col("group") == "Sports").then(pl.lit("Sports")).otherwise(pl.lit("Non-sports"))
        .alias("sector"))
    names = ["const", "oib_small", "oib_medium", "oib_large"]
    out: dict[str, Any] = {"hours": h.height, "markets": h["ticker"].n_unique()}
    for sector in ("All", "Sports", "Non-sports"):
        hs = h if sector == "All" else h.filter(pl.col("sector") == sector)
        res: dict[str, Any] = {}
        for ycol, ctrl in (("r_now", False), ("r_next", True), ("r_settle", True)):
            hh = hs.filter(pl.col(ycol).is_not_null() & (pl.col("r_now").is_not_null() if ctrl else True))
            if hh.height < 50:
                continue
            X = np.column_stack([np.ones(hh.height)] + [hh[f"oib_{c}"].to_numpy() for c in FLOW_CLASSES]
                                + ([100 * hh["r_now"].to_numpy()] if ctrl else []))
            res[ycol] = ols_cluster(X, 100 * hh[ycol].to_numpy(), hh["event_ticker"].to_numpy(),
                                    names + (["r_now"] if ctrl else []))
        out[sector] = res
    # binned: next-hour and to-settlement mid change by sign of small-trade imbalance when the
    # small class dominates the hour (>= 50% of volume)
    hb = h.filter((pl.col("v_small") / pl.col("vol") >= 0.5) & pl.col("m_end").is_not_null())
    hb = hb.with_columns(pl.when(pl.col("oib_small") > 0.2).then(pl.lit("small net buy YES"))
                         .when(pl.col("oib_small") < -0.2).then(pl.lit("small net buy NO"))
                         .otherwise(pl.lit("balanced")).alias("flow"))
    met = {"r_now": (100 * pl.col("r_now"), pl.lit(1.0)), "r_next": (100 * pl.col("r_next"), pl.lit(1.0)),
           "r_settle": (100 * pl.col("r_settle"), pl.lit(1.0))}
    out["binned_small"] = cell_stats(hb, ["flow"], met).to_dicts()
    hl = h.filter((pl.col("v_large") / pl.col("vol") >= 0.5) & pl.col("m_end").is_not_null())
    hl = hl.with_columns(pl.when(pl.col("oib_large") > 0.2).then(pl.lit("large net buy YES"))
                         .when(pl.col("oib_large") < -0.2).then(pl.lit("large net buy NO"))
                         .otherwise(pl.lit("balanced")).alias("flow"))
    out["binned_large"] = cell_stats(hl, ["flow"], met).to_dicts()
    return out


def t_strata(lf: pl.LazyFrame) -> dict[str, Any]:
    """Per-stratum results. Within a stratum every event has the same design weight, so each
    stratum is an independent, self-weighting sample; a sign test across strata is robust to the
    heavy right tail of longshot payoffs, which lets a single upset dominate pooled means."""
    from math import comb

    pm = pos_metrics()
    tm = taker_metrics()
    pos = positions(lf)
    ls_pos = pos.filter(pl.col("qo") <= LS_Q)
    a = cell_stats(ls_pos, ["stratum", "role"], {"d_net": pm["d_net"], "d_gross": pm["d_gross"]})
    b = cell_stats(pos, ["stratum", "role"], {"d_net": pm["d_net"]})
    c = cell_stats(lf.filter(pl.col("ls") & (pl.col("size_bucket").is_in([SIZES[0], SIZES[1]]))), ["stratum"],
                   {"d_net": tm["d_net"]})
    sh = cell_stats(lf, ["stratum", "size_bucket"],
                    {"ls": (pl.col("ls").cast(pl.Float64), pl.col("w_design") * pl.col("count"))})
    hs = cell_stats(lf.filter(pl.col("ls") & pl.col("m_pre").is_not_null() & (pl.col("sp_pre") <= TIGHT + 1e-9)),
                    ["stratum"], {"hs": (100 * pl.col("d").cast(pl.Float64) * (pl.col("yes_price") - pl.col("m_pre")),
                                         pl.col("w_design") * pl.col("count"))})
    ev = lf.group_by("stratum").agg(pl.col("event_ticker").n_unique().alias("events")).collect(engine="streaming")
    # sensitivity: taker longshot return without the stratum's two most profitable events
    el = (lf.filter(pl.col("ls")).group_by("stratum", "event_ticker").agg(
        (pl.col("count") * (pl.col("won") - pl.col("q") - pl.col("fee_t"))).sum().alias("pnl"),
        (pl.col("count") * (pl.col("q") + pl.col("fee_t"))).sum().alias("cost")).collect(engine="streaming")
          .sort("stratum", "pnl", descending=[False, True])
          .with_columns(pl.int_range(pl.len()).over("stratum").alias("_r")))
    drop2 = (el.filter(pl.col("_r") >= 2).group_by("stratum")
             .agg((pl.col("pnl").sum() / pl.col("cost").sum()).alias("taker_ls_drop_top2")))
    rows = []
    for st in sorted(ev["stratum"].to_list(), key=lambda x: (x not in SPORTS_LEAGUES, x)):
        def g(df: pl.DataFrame, **eq: Any) -> dict[str, Any]:
            f = df.filter(pl.col("stratum") == st)
            for k, v in eq.items():
                f = f.filter(pl.col(k) == v)
            return f.row(0, named=True) if f.height else {}
        s_small, s_large = g(sh, size_bucket=SIZES[0]), g(sh, size_bucket=SIZES[3])
        rows.append({"stratum": st, "events": g(ev)["events"],
                     "taker_all": g(b, role="taker").get("d_net"), "taker_all_t": g(b, role="taker").get("d_net_t"),
                     "maker_all": g(b, role="maker").get("d_net"), "maker_all_t": g(b, role="maker").get("d_net_t"),
                     "taker_ls": g(a, role="taker").get("d_net"), "taker_ls_t": g(a, role="taker").get("d_net_t"),
                     "maker_ls": g(a, role="maker").get("d_net"), "maker_ls_t": g(a, role="maker").get("d_net_t"),
                     "small_ls": g(c).get("d_net"), "small_ls_t": g(c).get("d_net_t"),
                     "ls_share_small": s_small.get("ls"), "ls_share_large": s_large.get("ls"),
                     "ls_half_spread_c": g(hs).get("hs"),
                     "taker_ls_drop_top2": g(drop2).get("taker_ls_drop_top2")})
    df = pl.DataFrame(rows).with_columns((pl.col("taker_all") - pl.col("maker_all")).alias("taker_minus_maker"),
                                         (pl.col("taker_ls") - pl.col("maker_ls")).alias("ls_taker_minus_maker"),
                                         (pl.col("ls_share_small") - pl.col("ls_share_large")).alias("ls_share_gap"))

    def sign(col: str, positive: bool = False) -> dict[str, Any]:
        """One-sided sign test: how many strata have ``col`` < 0 (or > 0), against p = 1/2."""
        v = [x for x in df[col].to_list() if x is not None and math.isfinite(x)]
        k = sum(1 for x in v if (x > 0 if positive else x < 0))
        n = len(v)
        return {"k": k, "n": n, "p_one_sided": sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else None}

    tests = {"taker_ls_negative": sign("taker_ls"), "maker_ls_negative": sign("maker_ls"),
             "small_ls_negative": sign("small_ls"), "taker_below_maker": sign("taker_minus_maker"),
             "ls_taker_below_maker": sign("ls_taker_minus_maker"),
             "ls_share_small_gt_large": sign("ls_share_gap", positive=True)}
    return {"strata": df, "sign_tests": tests}


def headline(lf: pl.LazyFrame, wd: pl.Expr | None = None) -> dict[str, Any]:
    """Key numbers for one (sub)sample: taker and maker net return per dollar overall and on
    longshots, taker longshot return for small and large orders, and the FLB gap for takers."""
    wd = pl.col("w_design") if wd is None else wd
    m = taker_metrics(wd)
    pm = pos_metrics(wd)
    pos = positions(lf)
    out: dict[str, Any] = {}
    r = cell_stats(pos, ["role"], pm).to_dicts()
    out["role"] = {x["role"]: {k: x[k] for k in ("d_net", "d_net_t", "c_net", "c_net_t", "d_gross",
                                                  "d_gross_t", "d_net_n", "d_net_clusters")} for x in r}
    r = cell_stats(pos.filter(pl.col("qo") <= LS_Q), ["role"], pm).to_dicts()
    out["role_longshot"] = {x["role"]: {k: x[k] for k in ("d_net", "d_net_t", "d_gross", "d_gross_t",
                                                           "d_net_n")} for x in r}
    r = cell_stats(lf.filter(pl.col("ls")), ["size_bucket"], m).to_dicts()
    out["taker_longshot_by_size"] = {x["size_bucket"]: {k: x[k] for k in ("d_net", "d_net_t", "d_net_n")}
                                     for x in r}
    gap = diff_stats(lf.filter(pl.col("ls") | pl.col("fav")), [], pl.col("fav"), *m["d_net"]).to_dicts()
    out["taker_fav_minus_ls"] = gap[0] if gap else None
    sh = cell_stats(lf, ["size_bucket"], {"ls": (pl.col("ls").cast(pl.Float64), wd * pl.col("count"))}).to_dicts()
    out["ls_share_by_size"] = {x["size_bucket"]: x["ls"] for x in sh}
    return out


def t_robust(s: Any, lf: pl.LazyFrame) -> dict[str, Any]:
    d = out_dir(s)
    mk = pl.read_parquet(d / "markets.parquet").filter(pl.col("sample") == "rs")
    mid = mk["close_time"].sort()[mk.height // 2]
    out: dict[str, Any] = {"split_close_time": mid}
    out["first_half"] = headline(lf.filter(pl.col("close_time") < mid))
    out["second_half"] = headline(lf.filter(pl.col("close_time") >= mid))
    out["strict_pre_resolution"] = headline(lf.filter(pl.col("strict")))
    out["excl_1_2c_99c"] = headline(lf.filter(pl.col("q").is_between(0.025, 0.985)
                                              & (1 - pl.col("q")).is_between(0.025, 0.985)))
    out["unweighted_design"] = headline(lf, pl.lit(1.0))
    out["sports"] = headline(lf.filter(pl.col("sector") == "Sports"))
    out["non_sports"] = headline(lf.filter(pl.col("sector") == "Non-sports"))
    out["market_clusters"] = {
        "taker_vs_maker": diff_stats(positions(lf), [], pl.col("role") == "taker",
                                     *pos_metrics()["d_net"], cluster="ticker").to_dicts()}
    ex = d / "panel"
    if any(ex.glob("ex_*.parquet")):
        lx = load_panel(s, "ex").filter(pl.col("pre_res"))
        out["ex_sample"] = headline(lx)
        out["ex_sample_strict"] = headline(lx.filter(pl.col("strict")))
    return out


def t_preres(s: Any) -> dict[str, Any]:
    """Diagnostics for the pre-resolution rule: prints and volume dropped, and the share of volume
    at extreme prices (YES <= 3c or >= 97c) by minutes before the close."""
    lf = pl.scan_parquet(out_dir(s) / "panel" / "rs_*.parquet").join(
        pl.scan_parquet(out_dir(s) / "markets.parquet").select("ticker", "close_time"), on="ticker")
    mins = ((pl.col("close_time") - pl.col("created_time")).dt.total_seconds() / 60)
    b = (pl.when(mins < 5).then(pl.lit("a 0-5")).when(mins < 10).then(pl.lit("b 5-10"))
         .when(mins < 30).then(pl.lit("c 10-30")).when(mins < 60).then(pl.lit("d 30-60"))
         .when(mins < 240).then(pl.lit("e 60-240")).otherwise(pl.lit("f >240")))
    ext = (pl.col("yes_price") <= 0.03) | (pl.col("yes_price") >= 0.97)
    tab = (lf.with_columns(b.alias("before_close"), ext.alias("extreme"),
                           pl.when(pl.col("group") == "Sports").then(pl.lit("Sports"))
                           .otherwise(pl.lit("Non-sports")).alias("sector"))
           .group_by("sector", "early", "before_close").agg(
               pl.col("count").sum().alias("contracts"),
               (pl.col("count") * pl.col("extreme")).sum().alias("extreme_contracts"),
               pl.len().alias("prints")).collect(engine="streaming")
           .with_columns((pl.col("extreme_contracts") / pl.col("contracts")).alias("extreme_share"))
           .sort("sector", "early", "before_close"))
    dropped = lf.select((~pl.col("pre_res")).mean().alias("prints_dropped"),
                        (pl.col("count") * (~pl.col("pre_res"))).sum() / pl.col("count").sum()).collect(engine="streaming")
    return {"by_minutes_before_close": tab.to_dicts(), "dropped": dropped.to_dicts()[0]}


def _stream_lines(path: Path):
    """Lines of a recorded ``.jsonl.zst`` stream file, decompressed incrementally."""
    import io

    import zstandard

    with open(path, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
        yield from io.TextIOWrapper(r, encoding="utf-8")


L2_WINDOWS = (("2026-11-08T00", "2026-11-08T03"),   # Saturday evening ET: in-play college football
              ("2026-11-10T14", "2026-11-10T17"))   # Tuesday morning ET: pre-game, weather, mentions


def step_l2check_all(s: Any) -> dict[str, Any]:
    out = {"windows": [step_l2check(s, a, b) for a, b in L2_WINDOWS]}
    (out_dir(s) / "l2check.json").write_text(json.dumps(_jsonable(out), indent=1))
    return out


def step_l2check(s: Any, start: str = "2026-11-08T00", end: str = "2026-11-08T03") -> dict[str, Any]:
    """Validate the trade-based mid proxy against recorded order books (``albert record``). Each
    market lives on one connection file per hour, so files are replayed one at a time, line by
    line, with one book across files (memory stays small). At each print the book mid at its
    receipt is compared with the proxy built from the prints alone (``mid_state``/``state_at``)."""
    from albert.backtest.l2 import stream_files
    from albert.stream import KalshiBook

    files = stream_files(s.data_dir / "raw" / "stream", start, end)
    book = KalshiBook()
    kinds = {"orderbook_snapshot", "orderbook_delta"}
    frames: list[pl.DataFrame] = []
    for path in files:
        cols: dict[str, list[Any]] = {k: [] for k in ("trade_id", "ticker", "ts_ms", "count", "yes_price",
                                                       "taker_side", "book_mid", "book_spread")}
        for line in _stream_lines(path):
            if "trade" not in line and "orderbook_" not in line:
                continue
            d = json.loads(json.loads(line)["frame"])
            typ = d.get("type")
            if typ in kinds:
                book.apply(typ, d["msg"])
            elif typ == "trade":
                m = d["msg"]
                tk = m["market_ticker"]
                yb, nb = book.levels(tk, "yes", 1), book.levels(tk, "no", 1)
                cols["trade_id"].append(m["trade_id"])
                cols["ticker"].append(tk)
                cols["ts_ms"].append(m["ts_ms"])
                cols["count"].append(float(m["count_fp"]))
                cols["yes_price"].append(float(m["yes_price_dollars"]))
                cols["taker_side"].append(m["taker_side"])
                cols["book_mid"].append((yb[0][0] + 1 - nb[0][0]) / 2 if yb and nb else None)
                cols["book_spread"].append((1 - nb[0][0] - yb[0][0]) if yb and nb else None)
        if cols["trade_id"]:
            frames.append(pl.DataFrame(cols, schema_overrides={"book_mid": pl.Float64,
                                                               "book_spread": pl.Float64}))
    tr = pl.concat(frames).with_columns(
        pl.from_epoch("ts_ms", time_unit="ms").dt.replace_time_zone("UTC").dt.cast_time_unit("us")
        .alias("created_time")).sort("ticker", "created_time", "trade_id")
    tr = tr.with_columns(
        pl.when(pl.col("taker_side") == "yes").then(1).otherwise(-1).cast(pl.Int8).alias("d"),
        pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price"))
        .otherwise(1 - pl.col("yes_price")).alias("q"))
    tr = taker_orders(tr)
    od = order_frame(tr)
    st = mid_state(od)
    pre = state_at(od, st, timedelta(0), "pre", strict_before=True)
    first = tr.group_by("order_id", maintain_order=True).agg(pl.col("book_mid").first(),
                                                              pl.col("book_spread").first())
    j = od.join(pre, on="order_id").join(first, on="order_id").filter(
        pl.col("m_pre").is_not_null() & pl.col("book_mid").is_not_null())
    diff = (j["m_pre"] - j["book_mid"]).abs()
    out = {"window": [start, end], "files": len(files), "prints": tr.height, "orders": od.height,
           "markets": tr["ticker"].n_unique(), "orders_with_both": j.height,
           "share_with_proxy": j.height / max(od.height, 1),
           "corr": float(np.corrcoef(j["m_pre"], j["book_mid"])[0, 1]) if j.height > 2 else None,
           "median_abs_diff_c": 100 * float(diff.median()), "mean_abs_diff_c": 100 * float(diff.mean()),
           "share_within_1c": float((diff <= 0.01 + 1e-9).mean()),
           "share_within_2c": float((diff <= 0.02 + 1e-9).mean()),
           "median_book_spread_c": 100 * float(j["book_spread"].median()),
           "mean_signed_diff_c": 100 * float((j["m_pre"] - j["book_mid"]).mean())}
    # taker_side semantics: a taker order executes at or beyond the book mid on its own side
    tb = od.join(first, on="order_id").filter(pl.col("book_mid").is_not_null())
    side = pl.col("d").cast(pl.Float64) * (pl.col("p_vwap") - pl.col("book_mid"))
    out["share_taker_at_or_beyond_mid"] = float(tb.select((side >= -1e-9).mean()).item())
    out["share_taker_strictly_beyond_mid"] = float(tb.select((side > 1e-9).mean()).item())
    print(out, flush=True)
    return out


def _write(tabs: dict[str, pl.DataFrame], d: Path) -> None:
    (d / "tables").mkdir(exist_ok=True)
    for name, df in tabs.items():
        df.write_csv(d / "tables" / f"{name}.csv", float_precision=6)


def step_analyze(s: Any) -> None:
    import time

    d = out_dir(s)
    t0 = time.time()
    lf = load_panel(s, "rs").filter(pl.col("pre_res"))
    tabs: dict[str, pl.DataFrame] = {"coverage": t_coverage(s)}
    for fn in (t_roles, t_size, t_yesno, t_timing, t_markouts):
        tabs.update(fn(lf))
        print(f"  {fn.__name__} done {time.time() - t0:.0f}s", flush=True)
    _write(tabs, d)
    st = t_strata(lf)
    tabs["strata"] = st["strata"]
    _write({"strata": st["strata"]}, d)
    hourly = t_hourly(s)
    print(f"  hourly done {time.time() - t0:.0f}s", flush=True)
    robust = t_robust(s, lf)
    print(f"  robust done {time.time() - t0:.0f}s", flush=True)
    pre = t_preres(s)
    summary = {
        "generated": datetime.now(UTC), "sample": json.loads((d / "sample_plan.json").read_text()),
        "rule": {"early_close_gap_min": EARLY_CLOSE_GAP.total_seconds() / 60,
                 "early_close_drop_min": EARLY_CLOSE_DROP.total_seconds() / 60,
                 "final_drop_min": FINAL_DROP.total_seconds() / 60, "stale_min": STALE.total_seconds() / 60},
        "coverage": tabs["coverage"].to_dicts(), "headline": headline(lf), "hourly": hourly,
        "strata": st["strata"].to_dicts(), "sign_tests": st["sign_tests"],
        "robust": robust, "pre_resolution": pre,
        "roles_diff": tabs["roles_diff"].to_dicts(), "roles_diff_longshot": tabs["roles_diff_longshot"].to_dicts(),
        "size_small_vs_large_longshot": tabs["size_small_vs_large_longshot"].to_dicts(),
        "size_small_vs_large_all": tabs["size_small_vs_large_all"].to_dicts(),
        "yesno_diff_side": tabs["yesno_diff_side"].to_dicts(),
        "timing_contrasts": tabs["timing_contrasts"].to_dicts(),
        "markouts": tabs["markouts"].to_dicts(), "decomp_side": tabs["decomp_side"].to_dicts(),
        "l2check": json.loads((d / "l2check.json").read_text()) if (d / "l2check.json").exists() else None,
    }
    (d / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1))
    (d / "verdict.md").write_text(verdict(summary))
    print(f"analyze done {time.time() - t0:.0f}s", flush=True)


def _fmt(x: Any, nd: int = 1, pct: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:+.{nd}f}%" if pct else f"{x:+.{nd}f}"


def verdict(sm: dict[str, Any]) -> str:
    h = sm["headline"]
    tk, mk = h["role"]["taker"], h["role"]["maker"]
    st = sm.get("sign_tests", {})
    rb = sm["robust"]

    def sg(k: str) -> str:
        x = st.get(k) or {}
        return f"{x.get('k')}/{x.get('n')} strata (one-sided sign test p = {x.get('p_one_sided', 0):.3f})"

    dec = {r["side"]: r for r in sm.get("decomp_side", []) if r.get("sector") == "All"}
    ls = dec.get("longshot", {})
    lines = [
        "# flb_flow verdict: who pays the favourite-longshot bias on Kalshi",
        "",
        ("Stratified random sample of settled non-financial Kalshi markets (closes 2026-08-15..10-31; 18 strata,"
         " events drawn volume-blind), complete public trade histories, pre-resolution prints only. Returns are"
         " per dollar, net of each role's fee, dollar-weighted; t clustered by event."),
        "",
        "## Robust findings (hold across strata, halves and the replication sample)",
        (f"- Small orders buy the longshots: longshot share of taker contracts is larger in <=$10 than in >$1k"
         f" orders in {sg('ls_share_small_gt_large')}; pooled "
         + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(h["ls_share_by_size"].items())) + "."),
        (f"- Taker longshot positions lose in {sg('taker_ls_negative')}; maker longshot positions in"
         f" {sg('maker_ls_negative')}; takers do worse than makers on longshots in {sg('ls_taker_below_maker')}."),
        (f"- Immediacy cost on longshots (mean q {ls.get('q_c', 0):.1f}c): half-spread {ls.get('half_spread', 0):.2f}c"
         f" (t {ls.get('half_spread_t') or 0:.1f}) + fee {ls.get('fee_c', 0):.2f}c = "
         f"{100 * (ls.get('immediacy_per_dollar') or 0):.1f}% of the stake, borne by takers only."),
        f"- Averaged over all prices there is no taker-maker gap: takers below makers in {sg('taker_below_maker')}.",
        "",
        "## Pooled, design-weighted (imprecise: longshot payoffs are heavy-tailed; one MLB HR-prop event",
        "## carries ~40% of the weighted taker longshot profit)",
        (f"- Takers {_fmt(tk['d_net'])} per $ (t {tk['d_net_t']:.1f}); makers {_fmt(mk['d_net'])} "
         f"(t {mk['d_net_t']:.1f}). Non-sports: takers {_fmt(rb['non_sports']['role']['taker']['d_net'])}, "
         f"makers {_fmt(rb['non_sports']['role']['maker']['d_net'])}."),
    ]
    hr = sm.get("hourly", {}).get("All", {})
    for y in ("r_now", "r_next", "r_settle"):
        if y in hr:
            lines.append(f"- Order-flow imbalance -> {y} (cents per unit imbalance): " + ", ".join(
                f"{c[4:]} {hr[y][c]['coef']:+.2f} (t {hr[y][c]['t'] or 0:.1f})"
                for c in ("oib_small", "oib_medium", "oib_large")))
    lines += ["- No size class predicts outcomes; small flow neither predicts nor reverts.",
              "", ("See summary.json and tables/*.csv for every cell; the paper sections are"
                   " paper_flb/sections/*_flow.tex.")]
    return "\n".join(lines) + "\n"


# ---- LaTeX table bodies (pasted into paper_flb/sections/*_flow.tex) ---------------------------------
GROUP_ORDER = ("Sports", "Weather", "Politics", "Entertainment", "Mentions", "Science & Tech", "Non-sports",
               "All")
SIZE_TEX = {"<=$10": r"$\le$\$10", "$10-100": r"\$10--100", "$100-1k": r"\$100--1k", ">$1k": r"$>$\$1k"}


def _n(x: Any, nd: int = 1, scale: float = 100.0, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    v = scale * float(x)
    return f"${v:+.{nd}f}$".replace("+", "+") if sign else f"${v:.{nd}f}$"


def _t(x: Any) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"$({float(x):.1f})$"


def _cell(row: dict[str, Any] | None, k: str, nd: int = 1, scale: float = 100.0) -> str:
    if not row:
        return "-- &"
    return f"{_n(row.get(k), nd, scale)} {_t(row.get(k + '_t'))} &"


def _btex(b: str) -> str:
    return b.replace("-", "--").replace("c", r"\cents")


def _rows(df: pl.DataFrame, **eq: Any) -> dict[str, Any] | None:
    f = df
    for k, v in eq.items():
        f = f.filter(pl.col(k) == v)
    return f.row(0, named=True) if f.height else None


def tex_tables(s: Any) -> dict[str, str]:
    d = out_dir(s)
    T = {p.stem: pl.read_csv(p, infer_schema_length=10000) for p in (d / "tables").glob("*.csv")}
    sm = json.loads((d / "summary.json").read_text())
    out: dict[str, str] = {}
    # roles by group
    lines = []
    for g in GROUP_ORDER:
        tk = _rows(T["roles_overall"], group=g, role="taker")
        mk = _rows(T["roles_overall"], group=g, role="maker")
        df = _rows(T["roles_diff"], group=g)
        tl = _rows(T["roles_side"], group=g, side="longshot", role="taker")
        ml = _rows(T["roles_side"], group=g, side="longshot", role="maker")
        if not tk:
            continue
        if g == "Non-sports":
            lines.append(r"\midrule")
        share = (tl["w_contracts"] / (tl["w_contracts"] + ml["w_contracts"])) if tl and ml else None
        line = (f"{g.replace('&', r'\&')} & {_cell(tk, 'd_net')} {_cell(mk, 'd_net')} "
                f"{_n(df['diff'] if df else None)} {_t(df['t'] if df else None)} & "
                f"{_cell(tl, 'd_net')} {_cell(ml, 'd_net')} {_n(share, 0, sign=False)} & "
                f"{tk['d_net_clusters']:,} \\\\")
        lines.append(line.replace("& &", "&"))
    out["roles_group"] = "\n".join(lines)
    # size table
    lines = []
    sh = T["size_shares"].filter(pl.col("group") == "All")
    tot_o, tot_c = sh["orders"].sum(), sh["w_contracts"].sum()
    for sb in SIZES:
        r = _rows(sh, size_bucket=sb)
        if not r:
            continue
        a = _rows(T["size_side"], group="All", side="longshot", size_bucket=sb)
        f = _rows(T["size_side"], group="All", side="favourite", size_bucket=sb)
        lines.append(f"{SIZE_TEX[sb]} & {_n(r['orders'] / tot_o, 1, sign=False)} & "
                     f"{_n(r['w_contracts'] / tot_c, 1, sign=False)} & {_n(r['ls_contracts'], 1, sign=False)} & "
                     f"{_n(r['ls_orders'], 1, sign=False)} & {_n(r['mean_q'], 0, sign=False)} & "
                     f"{_cell(a, 'd_net')} {_cell(f, 'd_net')} {_cell(a, 'p_net')[:-2]} \\\\")
    out["size"] = "\n".join(lines)
    # size by sector (appendix)
    lines = []
    ss, sd = T["size_shares_sector"], T["size_side_sector"]
    for sec in ("Sports", "Non-sports"):
        sub = ss.filter(pl.col("sector") == sec)
        if sub.is_empty():
            continue
        tot_o, tot_c = sub["orders"].sum(), sub["w_contracts"].sum()
        for k, sb in enumerate(SIZES):
            r = _rows(sub, size_bucket=sb)
            if not r:
                continue
            a = _rows(sd, sector=sec, side="longshot", size_bucket=sb)
            f = _rows(sd, sector=sec, side="favourite", size_bucket=sb)
            lines.append(f"{sec if k == 0 else ''} & {SIZE_TEX[sb]} & {_n(r['orders'] / tot_o, 1, sign=False)} & "
                         f"{_n(r['w_contracts'] / tot_c, 1, sign=False)} & {_n(r['ls_contracts'], 1, sign=False)} & "
                         f"{_n(r['mean_q'], 0, sign=False)} & {_cell(a, 'd_net')} {_cell(f, 'd_net')[:-2]} \\\\")
    out["size_sector"] = "\n".join(lines)
    # markouts by size (cents)
    lines = []
    for sb in SIZES:
        r = _rows(T["markouts"], sector="All", size_bucket=sb)
        if not r:
            continue
        cells = " ".join(_cell(r, k, 2, 1.0) for k in ("half_spread", "impact_1m", "impact_1h", "markout_1h",
                                                      "markout_settle"))
        lines.append(f"{SIZE_TEX[sb]} & {cells[:-2]} \\\\")
    out["markouts"] = "\n".join(lines)
    # decomposition of taker gross loss (cents per contract) by side and sector
    lines = []
    dc = T.get("decomp_side")
    if dc is not None:
        for sec in ("Sports", "Non-sports", "All"):
            for side, lab in (("longshot", r"Longshots ($\q\le10\cents$)"), ("middle", "Middle"),
                              ("favourite", r"Favourites ($\q\ge90\cents$)")):
                r = _rows(dc, sector=sec, side=side)
                if not r:
                    continue
                lines.append(f"{sec if side == 'longshot' else ''} & {lab} & {_n(r['q_c'], 1, 1.0, sign=False)} & "
                             f"{_cell(r, 'markout_settle', 2, 1.0)} {_cell(r, 'mid_to_settle', 2, 1.0)} "
                             f"{_cell(r, 'half_spread', 2, 1.0)} {_n(r['fee_c'], 2, 1.0, sign=False)} & "
                             f"{_n(r['immediacy_per_dollar'], 1, sign=False)} \\\\")
    out["decomp"] = "\n".join(lines)
    # per-stratum table
    lines = []
    stt = T.get("strata")
    if stt is not None:
        for k, r in enumerate(stt.iter_rows(named=True)):
            if k and r["stratum"] not in SPORTS_LEAGUES and stt.row(k - 1, named=True)["stratum"] in SPORTS_LEAGUES:
                lines.append(r"\midrule")
            lines.append(f"{r['stratum'].replace('&', r'\&')} & {r['events']:,} & "
                         f"{_n(r['taker_ls'], 0)} {_t(r['taker_ls_t'])} & {_n(r['maker_ls'], 0)} {_t(r['maker_ls_t'])} & "
                         f"{_n(r['small_ls'], 0)} {_t(r['small_ls_t'])} & {_n(r['ls_share_small'], 0, sign=False)} & "
                         f"{_n(r['ls_share_large'], 1, sign=False)} & {_n(r['ls_half_spread_c'], 2, 1.0)} \\\\")
        st_ = sm.get("sign_tests", {})

        def _st(key: str) -> str:
            x = st_.get(key) or {}
            if not x:
                return "--"
            pv = x.get("p_one_sided") or 0.0
            return f"{x['k']}/{x['n']} " + ("($<$.001)" if pv < 0.001 else f"({pv:.3f})".replace("0.", "."))
        lines.append(r"\midrule")
        lines.append(f"Sign test, $k/n$ ($p$) & & {_st('taker_ls_negative')} & {_st('maker_ls_negative')} & "
                     f"{_st('small_ls_negative')} & \\multicolumn{{2}}{{c}}{{{_st('ls_share_small_gt_large')}}} & \\\\")
    out["strata"] = "\n".join(lines)
    # timing contrasts
    lines = []
    tc = T["timing_contrasts"].filter(pl.col("sector") == "All")
    labels = {"evening_18_24": "Evening, 18--24 ET", "weekend": "Weekend (ET)", "inplay": "In play (sports)",
              "primetime": "Primetime game (sports)", "marquee": "Marquee event (top decile)",
              "first24h": "First 24 h after listing", "surge_3x": r"Volume surge ($\ge 3\times$)",
              "last_hour": "Last hour before expiry"}
    for key, lab in labels.items():
        cells = []
        for side in ("all", "longshot", "favourite"):
            r = _rows(tc, contrast=key, side=side)
            cells.append(f"{_n(r['diff'] if r else None)} {_t(r['t'] if r else None)}")
        r = _rows(tc, contrast=key, side="all")
        lines.append(f"{lab} & {_n(r['mean_a'] if r else None)} & {_n(r['mean_b'] if r else None)} & "
                     + " & ".join(cells) + f" & {r['clusters'] if r else ''} \\\\")
    out["timing"] = "\n".join(lines)
    # timing cells (appendix): taker net return, longshot and small-order shares, volume share
    lines = []
    tm = T["timing"].filter(pl.col("sector") == "All")
    dims = [("hour_block", "Hour of day (ET)", {"00-06": "00--06", "06-12": "06--12", "12-18": "12--18",
                                                 "18-24": "18--24"}),
            ("weekend", "Day", {"false": "Weekday", "true": "Weekend"}),
            ("inplay", "Sports game phase", {"false": "Pre-game", "true": "In play"}),
            ("primetime", "Sports game start", {"false": "Before 19:00 ET", "true": "19:00--23:59 ET"}),
            ("tte", "Time to scheduled expiry", {"a <1h": r"$<1$ h", "b 1-6h": "1--6 h", "c 6-24h": "6--24 h",
                                                 "d 1-7d": "1--7 days", "e >7d": r"$>7$ days"}),
            ("first24h", "Market age", {"true": "First 24 h", "false": "Later"}),
            ("surge_cls", "Past-hour volume vs own average",
             {"a quiet": r"$<0.5\times$", "b normal": r"$0.5$--$3\times$", "c surge 3-10x": r"$3$--$10\times$",
              "d spike >=10x": r"$\ge10\times$"}),
            ("marquee", "Event volume", {"false": "Below top decile", "true": "Top decile (marquee)"})]
    for dim, title, cells in dims:
        sub = tm.filter(pl.col("dim") == dim).with_columns(pl.col("cell").cast(pl.String).str.to_lowercase())
        if sub.is_empty():
            continue
        tot = sub["w_contracts"].sum()
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for key, lab in cells.items():
            r = _rows(sub, cell=key.lower())
            if not r:
                continue
            lines.append(f"\\quad {lab} & {_n(r['w_contracts'] / tot, 1, sign=False)} & {_cell(r, 'd_net')} "
                         f"{_cell(r, 'd_gross')} {_n(r['ls_share'], 1, sign=False)} & "
                         f"{_n(r['small_order_share'], 1, sign=False)} \\\\")
    out["timing_cells"] = "\n".join(lines)
    # order-flow regressions
    lines = []
    hr = sm.get("hourly", {})
    for sector in ("All", "Sports", "Non-sports"):
        res = hr.get(sector, {})
        for y, lab in (("r_now", "Same hour"), ("r_next", "Next hour"), ("r_settle", "To settlement")):
            if y not in res:
                continue
            r = res[y]
            cells = " & ".join(f"{_n(r[c]['coef'], 2, 1.0)} {_t(r[c]['t'])}" for c in ("oib_small", "oib_medium",
                                                                                       "oib_large"))
            ctrl = f"{_n(r['r_now']['coef'], 2, 1.0)} {_t(r['r_now']['t'])}" if "r_now" in r else "--"
            lines.append(f"{sector if y == 'r_now' else ''} & {lab} & {cells} & {ctrl} & "
                         f"{int(r['_n']['coef']):,} & {int(r['_clusters']['coef']):,} \\\\")
    out["oib"] = "\n".join(lines)
    # robustness
    lines = []
    rb = sm["robust"]
    variants = [("Main sample", sm["headline"]), ("First half (by close)", rb["first_half"]),
                ("Second half", rb["second_half"]), ("Strict pre-resolution", rb["strict_pre_resolution"]),
                (r"Excluding 1--2\cents{} and 99\cents", rb["excl_1_2c_99c"]),
                ("No design weights", rb["unweighted_design"]), ("Sports", rb["sports"]),
                ("Non-sports", rb["non_sports"])]
    if "ex_sample" in rb:
        variants += [("Replication sample", rb["ex_sample"]), ("Replication, strict", rb["ex_sample_strict"])]
    for lab, h in variants:
        tk, mk = h["role"].get("taker", {}), h["role"].get("maker", {})
        tl = h["role_longshot"].get("taker", {})
        sz = h["taker_longshot_by_size"]
        small, large = sz.get("<=$10", {}), sz.get(">$1k", {})
        lines.append(f"{lab} & {_n(tk.get('d_net'))} {_t(tk.get('d_net_t'))} & {_n(mk.get('d_net'))} "
                     f"{_t(mk.get('d_net_t'))} & {_n(tl.get('d_net'))} {_t(tl.get('d_net_t'))} & "
                     f"{_n(small.get('d_net'))} {_t(small.get('d_net_t'))} & {_n(large.get('d_net'))} "
                     f"{_t(large.get('d_net_t'))} & {_n(h['ls_share_by_size'].get('<=$10'), 0, sign=False)} / "
                     f"{_n(h['ls_share_by_size'].get('>$1k'), 0, sign=False)} \\\\")
    out["robust"] = "\n".join(lines)
    # sample design (appendix)
    lines = []
    plan = sm["sample"]
    cov = {(r["sample"], r["group"]): r for r in sm["coverage"]}
    for r in sorted(plan["strata"], key=lambda x: (x["stratum"] not in SPORTS_LEAGUES, x["stratum"])):
        lines.append(f"{r['stratum'].replace('&', r'\&')} & {r['frame_events']:,} & {r['frame_markets']:,} & "
                     f"{r['sample_events']:,} & {r['sample_markets']:,} & "
                     f"{r['frame_events'] / max(r['sample_events'], 1):.1f} \\\\")
    out["design"] = "\n".join(lines)
    lines = []
    for g in GROUP_ORDER[:-1]:
        r = cov.get(("rs", g))
        if not r:
            continue
        lines.append(f"{g.replace('&', r'\&')} & {r['markets']:,} & {r['events']:,} & {r['prints']:,} & "
                     f"{int(r['orders']):,} & {r['contracts'] / 1e6:.2f} & {r['dollars'] / 1e6:.2f} & "
                     f"{100 * (1 - r['prints'] / r['prints_all']):.1f} & "
                     f"{100 * (r['completeness_median'] or 0):.0f} \\\\")
    out["coverage"] = "\n".join(lines)
    # roles by bucket (appendix), All
    lines = []
    rbk = T["roles_bucket"].filter(pl.col("group") == "All")
    for b in C.BUCKET_LABELS:
        tk, mk = _rows(rbk, role="taker", bucket=b), _rows(rbk, role="maker", bucket=b)
        if not tk or not mk:
            continue
        share = tk["w_contracts"] / (tk["w_contracts"] + mk["w_contracts"])
        lines.append(f"{_btex(b)} & {_n(share, 0, sign=False)} & {_cell(tk, 'c_gross', 2)} "
                     f"{_cell(tk, 'd_net')} {_cell(mk, 'c_gross', 2)} {_cell(mk, 'd_net')[:-2]} \\\\")
    out["roles_bucket"] = "\n".join(lines)
    # yes/no by bucket (appendix)
    lines = []
    yc, yd = T["yesno_cells"].filter(pl.col("group") == "All"), T["yesno_diff"].filter(pl.col("group") == "All")
    ys = T["yesno_share"].filter(pl.col("group") == "All")
    for b in C.BUCKET_LABELS:
        y, n_, df = _rows(yc, taker_yes=True, bucket=b), _rows(yc, taker_yes=False, bucket=b), _rows(yd, bucket=b)
        sh_ = _rows(ys, bucket=b)
        if not y or not n_:
            continue
        lines.append(f"{_btex(b)} & {_n(sh_['yes_share'] if sh_ else None, 0, sign=False)} & "
                     f"{_cell(y, 'd_net')} {_cell(n_, 'd_net')} {_n(df['diff'] if df else None)} "
                     f"{_t(df['t'] if df else None)} \\\\")
    out["yesno"] = "\n".join(lines)
    # pre-resolution diagnostics (appendix): share of contracts at <= 3c / >= 97c by minutes to close
    pr = pl.DataFrame(sm["pre_resolution"]["by_minutes_before_close"])
    lines = []
    if pr.height:
        labs = {"a 0-5": "0--5", "b 5-10": "5--10", "c 10-30": "10--30", "d 30-60": "30--60",
                "e 60-240": "60--240", "f >240": r"$>240$"}
        for b, lab in labs.items():
            cells = []
            for sec in ("Sports", "Non-sports"):
                for early in (False, True):
                    r = _rows(pr, sector=sec, early=early, before_close=b)
                    cells.append(_n(r["extreme_share"], 0, sign=False) if r else "--")
            tot = pr.filter(pl.col("before_close") == b)
            lines.append(f"{lab} & " + " & ".join(cells)
                         + f" & {100 * tot['contracts'].sum() / pr['contracts'].sum():.1f} \\\\")
    out["preres"] = "\n".join(lines)
    (d / "tex").mkdir(exist_ok=True)
    for k, v in out.items():
        (d / "tex" / f"{k}.tex").write_text(f"% source: data/research/flb_flow (generated by flb_flow tex)\n{v}\n")
    return out


# ---- CLI ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from albert.config import settings

    ap = argparse.ArgumentParser(prog="python -m albert.research.flb_flow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("step", choices=["plan", "fetch", "build", "analyze", "l2check", "tex", "all"])
    r.add_argument("--samples", default="rs,ex", help="build: comma-separated samples (rs, ex)")
    a = ap.parse_args(argv)
    s = settings()
    steps = ["plan", "fetch", "build", "analyze"] if a.step == "all" else [a.step]
    for st in steps:
        print(f"== {st}", flush=True)
        if st == "build":
            step_build(s, samples=tuple(a.samples.split(",")))
        else:
            {"plan": step_plan, "fetch": step_fetch, "analyze": step_analyze, "l2check": step_l2check_all,
             "tex": tex_tables}[st](s)


if __name__ == "__main__":
    main()
