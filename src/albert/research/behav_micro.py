"""Cross-market microstructure and flow on Kalshi: structural reasons for trading at bad prices.

Research only: public data, no orders. Hypotheses, pre-registration and verdicts live in
``data/research/behav_micro/{hypotheses,report}.md``. This module holds the data steps and every
test; each step writes its tables under ``data/research/behav_micro/``.

Data
----
* ``rs`` tape: the volume-blind stratified sample of settled non-financial markets that closed
  2026-08-15..10-31, with complete public trade histories (built by ``flb_flow``; prints in
  ``data/research/flb_flow/panel/rs_*.parquet``, market metadata in ``.../markets.parquet``).
* ``nov`` tape: the same design (events uniform within stratum, at most 20 legs per event) on
  markets that closed 2026-11-01..11-16 and settled, fetched here (``nov-plan``, ``nov-fetch``)
  at <= 1 request/s. Holdout only.
* xcat hourly candles (``data/research/xcat/hourly``) for the lock-up and pause tests.
* Recorded L2 (``data/raw/stream/kalshi``, ``k*`` connections = the stratified random recorder
  universe) for the exact-book version of the sweep test.

Splits: dev = decisions before 2026-10-01 in markets that closed before 2026-10-01; holdout =
decisions from 2026-10-01 (``rs`` October closes plus the ``nov`` tape). L2: 11-05..11-11 dev,
11-12..11-18 holdout.

Execution: taker at the printed/quoted ask or bid, fee ``ta_calendar.taker_fee_expr`` (0.07 x
multiplier x n x P(1-P), rounded up to the cent per order); maker fills only when a later print
trades strictly through our level (never at it), maker fee 0.0175 x P(1-P) on
``quadratic_with_maker_fees`` series, rounded up to the cent per order. t-stats clustered by
event and by ET date. Ledgers for ``vol_metrics``: ``data/research/behav_micro/ledgers/``.

    python -m albert.research.behav_micro <step>
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")

OUT = Path("data/research/behav_micro")
FLB = Path("data/research/flb_flow")
ET = ZoneInfo("America/New_York")
DEV_END = datetime(2026, 10, 1, tzinfo=UTC)
NOV_START = datetime(2026, 11, 1, tzinfo=UTC)
NOV_END = datetime(2026, 11, 17, tzinfo=UTC)          # closes in [11-01, 11-17)
NOV_SEED = 20261118
KALSHI_TAKER = 0.07
KALSHI_MAKER = 0.0175


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


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1))


# ---- fees ----------------------------------------------------------------------------------------
def fee_cents(n: float, price: float, mult: float = 1.0, rate: float = KALSHI_TAKER) -> float:
    """Fee of one order of ``n`` contracts at ``price`` (dollars), rounded up to the cent, in dollars."""
    raw = rate * mult * n * price * (1 - price)
    return math.ceil(round(raw * 100, 9)) / 100


def fee_expr(n: pl.Expr, price: pl.Expr, mult: pl.Expr, rate: float = KALSHI_TAKER) -> pl.Expr:
    """Polars version of :func:`fee_cents` (same rounding as ``ta_calendar.taker_fee_expr``)."""
    raw = rate * mult * n * price * (1 - price)
    return (raw * 100).round(9).ceil() / 100


# ---- data: November holdout sample (same design as flb_flow rs) -----------------------------------
def _api(min_interval: float = 1.0):
    from albert.config import settings
    from albert.research.pm_hazard import _api as hz_api
    return hz_api(settings(), min_interval, nap_cap=20.0, retries=60)


def series_info() -> dict[str, dict[str, Any]]:
    ser = pl.read_parquet("data/tables/kalshi_series.parquet")
    return {r["series_ticker"]: r for r in ser.iter_rows(named=True)}


def nov_events_per_stratum() -> dict[str, int]:
    from albert.research.flb_flow import EVENTS_PER_STRATUM
    return {k: max(10, math.ceil(v / 3)) for k, v in EVENTS_PER_STRATUM.items()}


def step_nov_plan() -> None:
    """List settled markets that closed in [NOV_START, NOV_END) and draw the volume-blind sample."""
    from albert.research import flb_flow as F
    from albert.venues import kalshi as k

    d = OUT / "nov"
    listing = d / "markets_all.parquet"
    if not listing.exists():
        api = _api()
        info = series_info()
        keep = list(F.MARKET_COLS)
        frames: list[pl.DataFrame] = []
        n = 0
        for i, page in enumerate(api.settled_markets(NOV_START, NOV_END), 1):
            rows = [r for r in k.market_rows(page.payload, info, "live") if (r["volume"] or 0) > 0]
            n += len(page.payload.get("markets") or [])
            if rows:   # compact per page: the full listing is ~10^6 markets
                frames.append(pl.DataFrame(rows, infer_schema_length=None).select(keep))
            if i % 50 == 0:
                print(f"  listing page {i}: {n} markets, {sum(f.height for f in frames)} traded", flush=True)
        _save(pl.concat(frames, how="vertical_relaxed"), listing)
    m = pl.read_parquet(listing)
    m = m.filter(pl.col("result").is_in(["yes", "no"])
                 & pl.col("close_time").is_between(NOV_START, NOV_END, closed="left"))
    ser = pl.read_parquet("data/tables/kalshi_series.parquet", columns=["series_ticker", "title"])
    frame = F.assign_groups(m.select(F.MARKET_COLS), ser)
    ev = (frame.filter(pl.col("volume") > 0).select("stratum", "event_ticker").unique()
          .with_columns(pl.col("event_ticker").hash(NOV_SEED).alias("_h")).sort("stratum", "_h")
          .with_columns(pl.int_range(pl.len()).over("stratum").alias("_r")))
    caps = nov_events_per_stratum()
    ev = ev.join(pl.DataFrame({"stratum": list(caps), "_cap": list(caps.values())}), on="stratum")
    chosen = ev.filter(pl.col("_r") < pl.col("_cap")).select("event_ticker")
    legs = (frame.filter(pl.col("volume") > 0).join(chosen, on="event_ticker", how="semi")
            .with_columns(pl.col("ticker").hash(NOV_SEED + 1).alias("_h")).sort("event_ticker", "_h")
            .with_columns(pl.int_range(pl.len()).over("event_ticker").alias("_r"))
            .filter(pl.col("_r") < F.MAX_LEGS).drop("_h", "_r").sort("stratum", "event_ticker", "ticker"))
    _save(legs, d / "sample_markets.parquet")
    tab = (frame.filter(pl.col("volume") > 0).group_by("stratum")
           .agg(pl.len().alias("frame_markets"), pl.col("event_ticker").n_unique().alias("frame_events"))
           .join(legs.group_by("stratum").agg(pl.len().alias("sample_markets"),
                                              pl.col("event_ticker").n_unique().alias("sample_events")),
                 on="stratum", how="left").fill_null(0).sort("stratum"))
    _dump({"seed": NOV_SEED, "frame": [NOV_START, NOV_END], "events_per_stratum": caps,
           "strata": tab.to_dicts(), "markets": legs.height, "events": legs["event_ticker"].n_unique()},
          d / "sample_plan.json")
    print(tab)
    print(f"nov sample: {legs.height} markets, {legs['event_ticker'].n_unique()} events")


def step_nov_fetch() -> None:
    """Complete trade histories of the November sample, one sequential client at <= 1 request/s."""
    from albert.research.flb_flow import TRADE_SCHEMA, segments
    from albert.venues import kalshi as k

    d = OUT / "nov"
    parts = d / "trades"
    parts.mkdir(parents=True, exist_ok=True)
    done_path = d / "trades_done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    sample = pl.read_parquet(d / "sample_markets.parquet")
    todo = sample.filter(~pl.col("ticker").is_in(list(done))).sort("volume")
    api = _api()
    cutoff = api.cutoff().get("trades_created_ts")
    print(f"nov fetch: {todo.height} markets to go ({len(done)} done), cutoff {cutoff}", flush=True)
    rows: list[dict[str, Any]] = []
    fin: list[str] = []
    n_part = len(list(parts.glob("part_*.parquet")))

    def flush() -> None:
        nonlocal rows, fin, n_part
        if rows:
            _save(pl.DataFrame(rows, schema=TRADE_SCHEMA).unique("trade_id"), parts / f"part_{n_part:05d}.parquet")
            n_part += 1
        done.update(fin)
        _save(pl.DataFrame({"ticker": sorted(done)}, schema={"ticker": pl.String}), done_path)
        rows, fin = [], []

    for i, r in enumerate(todo.iter_rows(named=True), 1):
        got: list[dict[str, Any]] = []
        ok = True
        for hist, a, b in segments(r["open_time"], r["close_time"], cutoff):
            try:
                for page in api.trades(r["ticker"], hist, min_ts=a, max_ts=b):
                    got.extend(k.trade_rows(page.payload))
            except RuntimeError as error:
                print(f"skip {r['ticker']}: {str(error)[:160]}", flush=True)
                ok = False
                break
        if ok:
            rows.extend(got)
            fin.append(r["ticker"])
        if i % 100 == 0 or len(rows) > 200_000:
            flush()
            print(f"  {datetime.now(UTC):%H:%M:%S} {i}/{todo.height} markets", flush=True)
    flush()
    print("nov fetch done", flush=True)


def step_fee_changes() -> None:
    """History of series and event fee changes (two public requests)."""
    api = _api()
    for path, name in (("/series/fee_changes", "series"), ("/events/fee_changes", "events")):
        try:
            page = api.client.get(path, {"show_historical": "true"})
            _dump(page.payload, OUT / f"fee_changes_{name}.json")
            print(name, {k: (len(v) if isinstance(v, list) else v) for k, v in page.payload.items()})
        except Exception as error:  # noqa: BLE001 - optional endpoint
            print(f"{path}: {error}")


# ---- statistics ----------------------------------------------------------------------------------
def cluster_stats(df: pl.DataFrame, x: str, g: str = "event_ticker", g2: str | None = None) -> dict[str, Any]:
    """Mean of ``x`` (equal weight per row) in cents and its t-stat clustered by ``g`` (and, with
    ``g2``, two-way: the larger of the one-way and the Cameron-Gelbach-Miller variance)."""
    cols = [pl.col(x).alias("x"), pl.col(g).alias("g")] + ([pl.col(g2).alias("g2")] if g2 else [])
    d = df.select(cols).drop_nulls("x")
    n = d.height
    if n < 3:
        return {"n": n, "mean_c": None, "t": None, "clusters": None}
    m = float(d["x"].mean())

    def var(by: list[str]) -> float:
        s = d.group_by(by).agg((pl.col("x") - m).sum().alias("s"))["s"].to_numpy()
        k = len(s)
        return float((s ** 2).sum()) / n ** 2 * (k / (k - 1)) if k > 1 else float("nan")

    v = var(["g"])
    if g2:
        v2 = var(["g2"])
        v12 = var(["g", "g2"])
        v = max(v, v2, v + v2 - v12)
    se = math.sqrt(v) if v > 0 else float("nan")
    return {"n": n, "mean_c": round(100 * m, 3), "t": round(m / se, 2) if se > 0 else None,
            "clusters": d["g"].n_unique(), "clusters2": d["g2"].n_unique() if g2 else None}


# ---- tape: taker orders from public prints -------------------------------------------------------
DS = (1, 2, 3, 5, 8, 10, 15)


def orders_from_trades(tr: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    """One row per taker order. Kalshi prints one fill per resting order, all with the taker
    order's timestamp, so prints in one ticker with the same ``created_time`` and taker side are
    one order. Taker-side prices ``q`` rise as the order walks the book, so ``q1`` (the lowest) is
    the best price before the order and ``qw`` its worst fill. ``cap<D>`` = contracts printed
    strictly beyond ``q1 + D`` cents (what a stub resting there would certainly have filled)."""
    tr = (tr.unique("trade_id")
          .filter((pl.col("count") > 0) & pl.col("yes_price").is_between(0.005, 0.995))
          .join(meta.select("ticker", "close_time"), on="ticker", how="inner")
          .filter(pl.col("created_time") < pl.col("close_time")).drop("close_time"))
    tr = tr.with_columns(pl.when(pl.col("taker_side") == "yes").then(1).otherwise(-1).cast(pl.Int8).alias("d"))
    tr = tr.with_columns(pl.when(pl.col("d") == 1).then(pl.col("yes_price"))
                         .otherwise(1 - pl.col("yes_price")).round(4).alias("q"))
    od = tr.group_by("ticker", "created_time", "d").agg(
        pl.col("count").sum().alias("contracts"), pl.len().alias("fills"),
        pl.col("q").min().alias("q1"), pl.col("q").max().alias("qw"),
        (pl.col("count") * pl.col("q")).sum().alias("notional"),
        *[pl.col("count").filter(pl.col("q") > pl.col("q").min() + D / 100 + 1e-9).sum().alias(f"cap{D}")
          for D in DS])
    od = od.rename({"created_time": "t"}).join(meta, on="ticker", how="inner")
    od = od.with_columns(pl.when(pl.col("d") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).cast(pl.Int8)
                         .alias("won"),
                         pl.col("t").dt.convert_time_zone("America/New_York").dt.date().alias("etd"))
    return od.sort("ticker", "t")


META_COLS = ["ticker", "event_ticker", "group", "stratum", "close_time", "settlement_ts", "y", "mult",
             "maker_fee", "sched", "exp"]


def nov_meta() -> pl.DataFrame:
    """Market metadata of the November sample in the ``flb_flow`` market-meta layout."""
    import re

    from albert.research.flb_flow import GAME_STRATA, scheduled_start
    m = pl.read_parquet(OUT / "nov" / "sample_markets.parquet")
    rows = []
    for r in m.iter_rows(named=True):
        game = (r["group"] == "Sports" and r["stratum"] in GAME_STRATA
                and re.match(r"^[A-Z0-9]+-\d{2}[A-Z]{3}\d{2}", r["event_ticker"] or "") is not None)
        sched = scheduled_start(r["event_ticker"], r["occurrence_time"], r["expected_expiration_time"]) if game else None
        if sched is not None and not (timedelta(minutes=30) <= r["close_time"] - sched <= timedelta(hours=12)):
            sched = None
        rows.append({"ticker": r["ticker"], "sched": sched})
    extra = pl.DataFrame(rows, schema={"ticker": pl.String, "sched": pl.Datetime("us", "UTC")})
    m = m.join(extra, on="ticker", how="left").with_columns(
        (pl.col("result") == "yes").cast(pl.Int8).alias("y"),
        pl.col("fee_multiplier").fill_null(1.0).alias("mult"),
        (pl.col("fee_type").fill_null("") == "quadratic_with_maker_fees").alias("maker_fee"),
        pl.col("expected_expiration_time").fill_null(pl.col("close_time")).alias("exp"))
    return m.select(META_COLS)


def step_tape(samples: tuple[str, ...] = ("rs", "nov")) -> None:
    """Taker-order tape -> ``tape/<sample>/part_*.parquet``, one file per raw trade part. Every
    market's prints sit in one raw part, so each order part holds complete market histories and
    the analyses can run part by part in bounded memory."""
    srcs = {"rs": FLB / "trades", "nov": OUT / "nov" / "trades"}
    for sample in samples:
        parts = sorted(srcs[sample].glob("part_*.parquet")) if srcs[sample].exists() else []
        if not parts:
            continue
        meta = (pl.read_parquet(FLB / "markets.parquet").filter(pl.col("sample") == "rs").select(META_COLS)
                if sample == "rs" else nov_meta())
        n = mk = 0
        for p in parts:
            od = orders_from_trades(pl.read_parquet(p), meta).with_columns(pl.lit(sample).alias("sample"))
            _save(od, OUT / "tape" / sample / p.name)
            n += od.height
            mk += od["ticker"].n_unique()
        print(f"{sample}: {n} orders, {mk} markets, {len(parts)} parts", flush=True)


def iter_orders(split: str):
    """Order parts of a split. ``dev``: orders before DEV_END in markets closed before DEV_END
    (rs). ``holdout``: orders from DEV_END on (rs October closes + the nov sample)."""
    samples = ("rs",) if split == "dev" else ("rs", "nov")
    for sample in samples:
        for p in sorted((OUT / "tape" / sample).glob("part_*.parquet")):
            od = pl.read_parquet(p)
            if split == "dev":
                od = od.filter((pl.col("t") < DEV_END) & (pl.col("close_time") < DEV_END))
            else:
                od = od.filter(pl.col("t") >= DEV_END)
            if od.height:
                yield od


# ---- H7: stub quotes filled by sweeps (prints) ---------------------------------------------------
def maker_fee_expr(price: pl.Expr, n: int = 1) -> pl.Expr:
    return pl.when(pl.col("maker_fee")).then(fee_expr(pl.lit(float(n)), price, pl.col("mult"), KALSHI_MAKER) / n
                                             ).otherwise(0.0)


def stub_fills(od: pl.DataFrame, depth_c: int, tau_s: float) -> pl.DataFrame:
    """Fills of a 1-contract stub resting ``depth_c`` cents beyond the best on each side.

    The stub on side S (selling S to an S-taker) is re-pegged to (best S price + D) with latency
    ``tau_s``: same-side orders less than ``tau_s`` apart form one run, and the stub stands at the
    run's starting best (``q1`` of its first order) + D for the whole run. It fills (one contract
    at its price L) at the first order of the run that prints strictly beyond L; after a fill it
    is reposted only after the run ends. We then hold the other side at 1 - L to settlement:
    PnL = L - won - maker fee (one contract, rounded up to the cent)."""
    D = depth_c / 100
    x = od.sort("ticker", "d", "t")
    gap = (pl.col("t").diff().over("ticker", "d").dt.total_microseconds() / 1e6).fill_null(1e12)
    x = x.with_columns((gap > tau_s).cum_sum().over("ticker", "d").alias("run"))
    x = x.with_columns(pl.col("q1").first().over("ticker", "d", "run").alias("ref"),
                       pl.int_range(pl.len()).over("ticker", "d", "run").alias("k_in_run"))
    x = x.with_columns((pl.col("ref") + D).round(4).alias("L"))
    x = x.filter((pl.col("qw") > pl.col("L") + 1e-9) & (pl.col("L") <= 0.99))
    x = x.sort("ticker", "d", "t").unique(["ticker", "d", "run"], keep="first", maintain_order=True)
    return x.with_columns(
        (pl.col("L") - pl.col("won") - maker_fee_expr(pl.col("L"))).alias("pnl"),
        (pl.col("L") - pl.col("won")).alias("gross"),
        (pl.col("k_in_run") > 0).alias("burst"),
        pl.lit(depth_c).alias("D"), pl.lit(tau_s).alias("tau"))


def refill_fills(od: pl.DataFrame, *, min_depth: float = 0.10, max_usd: float = 50.0, wait_s: float = 1.0,
                 life_min: float = 60.0, offset: float = 0.01) -> pl.DataFrame:
    """R5, post-small-sweep refill. A trigger is one taker order whose fills span at least
    ``min_depth`` (qw - q1) with notional at most ``max_usd`` (a small market order walking a thin
    book). ``wait_s`` later we rest one contract selling the swept side at P = qw - ``offset``
    (inside the emptied zone), for ``life_min`` minutes. It fills only when a later same-side
    taker order prints strictly beyond P; held to settlement: PnL = P - won - maker fee."""
    import bisect
    rows = []
    for (tk, d), g in od.sort("ticker", "d", "t").group_by(["ticker", "d"], maintain_order=True):
        tsec = [x.timestamp() for x in g["t"].to_list()]
        q1, qw, nt = g["q1"].to_list(), g["qw"].to_list(), g["notional"].to_list()
        for i in range(len(tsec)):
            if qw[i] - q1[i] < min_depth - 1e-9 or nt[i] > max_usd:
                continue
            P = round(qw[i] - offset, 4)
            if P <= 0.01:
                continue
            j = bisect.bisect_right(tsec, tsec[i] + wait_s)
            while j < len(tsec) and tsec[j] <= tsec[i] + life_min * 60:
                if qw[j] > P + 1e-9:
                    rows.append({"ticker": tk, "d": d, "row": j, "P": P, "trig_usd": nt[i]})
                    break
                j += 1
    if not rows:
        return pl.DataFrame()
    hit = pl.DataFrame(rows)
    g = od.sort("ticker", "d", "t").with_columns(pl.int_range(pl.len()).over("ticker", "d").alias("row"))
    f = hit.join(g, on=["ticker", "d", "row"], how="left")
    return f.with_columns((pl.col("P") - pl.col("won") - maker_fee_expr(pl.col("P"))).alias("pnl"),
                          pl.col("P").alias("L"))


def flb_control(split: str) -> pl.DataFrame:
    """Maker PnL (q - won) of single-level orders, by group and taker-side price cent: what an
    ordinary maker earned selling the same side at the same price (the favourite-longshot and
    YES-overbuy premium that any maker fill at that price carries)."""
    aggs = []
    for od in iter_orders(split):
        base = od.filter(pl.col("qw") == pl.col("q1")).with_columns(
            (pl.col("q1") * 100).round(0).cast(pl.Int32).alias("pc"))
        aggs.append(base.group_by("group", "pc").agg((pl.col("q1") - pl.col("won")).sum().alias("s"),
                                                     pl.len().alias("flb_n")))
    a = pl.concat(aggs).group_by("group", "pc").agg(pl.col("s").sum(), pl.col("flb_n").sum())
    return a.with_columns((pl.col("s") / pl.col("flb_n")).alias("flb")).drop("s")


def stub_split(split: str, cells: list[tuple[int, float]]) -> pl.DataFrame:
    """Stub fills of every (D, tau) cell over a split, part by part."""
    out = []
    for od in iter_orders(split):
        for D, tau in cells:
            out.append(stub_fills(od, D, tau))
    return pl.concat(out, how="vertical_relaxed").with_columns(
        (pl.col("L") * 100).round(0).cast(pl.Int32).alias("pc"), pl.lit(split).alias("split"))


def stub_table(fills: pl.DataFrame, ctrl: pl.DataFrame, cells: list[tuple[int, float]]) -> tuple[pl.DataFrame, list[dict]]:
    rows, out = [], []
    for D, tau in cells:
        f = fills.filter((pl.col("D") == D) & (pl.col("tau") == tau))
        f = f.join(ctrl, on=["group", "pc"], how="left").with_columns((pl.col("pnl") - pl.col("flb")).alias("excess"))
        out.append(f)
        r = cluster_stats(f, "pnl", "event_ticker", "etd")
        rows.append({"D": D, "tau_s": tau, "fills": r["n"], "events": r["clusters"], "dates": r["clusters2"],
                     "pnl_c": r["mean_c"], "t": r["t"],
                     "single_pnl_c": cluster_stats(f.filter(~pl.col("burst")), "pnl", "event_ticker", "etd")["mean_c"],
                     "burst_fills": f.filter(pl.col("burst")).height,
                     "burst_pnl_c": cluster_stats(f.filter(pl.col("burst")), "pnl", "event_ticker", "etd")["mean_c"],
                     "excess_c": cluster_stats(f, "excess", "event_ticker", "etd")["mean_c"],
                     "excess_t": cluster_stats(f, "excess", "event_ticker", "etd")["t"]})
    return pl.concat(out, how="vertical_relaxed"), rows


# ---- L2: exact-book stub simulation on the recorded k-universe -----------------------------------
def k_files(start: str, end: str) -> list[Path]:
    """Recorded hourly files of the stratified random recorder (``k<n>`` connections) for UTC days
    in [start, end] (YYYY-MM-DD)."""
    root = Path("data/raw/stream/kalshi")
    out = []
    for day in sorted(p.name for p in root.iterdir() if p.is_dir()):
        if start <= day <= end:
            out += sorted(p for p in (root / day).glob("*-k*.jsonl.zst") if not p.name.endswith(".part"))
    return out


def _iter_frames(path: Path):
    import io

    import zstandard as zstd
    with path.open("rb") as fh:
        reader = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in reader:
            try:
                row = json.loads(line)
                fr = json.loads(row["frame"])
            except (ValueError, KeyError):
                continue
            yield row["recv_ns"], fr


class _Book:
    """Kalshi book in 1/100 cent units: YES bids and NO bids (YES ask = 1 - best NO bid), with the
    best of each cached, and the history of (time, best YES bid, best YES ask) changes."""
    __slots__ = ("bn", "by", "hist_a", "hist_b", "hist_t", "last_t", "no", "yes")

    def __init__(self) -> None:
        self.yes: dict[int, float] = {}
        self.no: dict[int, float] = {}
        self.by: int | None = None
        self.bn: int | None = None
        self.hist_t: list[int] = []
        self.hist_b: list[int | None] = []
        self.hist_a: list[int | None] = []
        self.last_t = 0

    def snapshot(self, yes: dict[int, float], no: dict[int, float]) -> None:
        self.yes = {p: s for p, s in yes.items() if s > 1e-6}
        self.no = {p: s for p, s in no.items() if s > 1e-6}
        self.by = max(self.yes, default=None)
        self.bn = max(self.no, default=None)

    def delta(self, side: str, p: int, dq: float) -> None:
        book = self.yes if side == "yes" else self.no
        s = book.get(p, 0.0) + dq
        best = self.by if side == "yes" else self.bn
        if s > 1e-6:
            book[p] = s
            if best is None or p > best:
                best = p
        else:
            book.pop(p, None)
            if best == p:
                best = max(book, default=None)
        if side == "yes":
            self.by = best
        else:
            self.bn = best

    def best(self) -> tuple[int | None, int | None]:
        return self.by, (10000 - self.bn if self.bn is not None else None)

    def mark(self, t: int) -> None:
        t = max(t, self.last_t)
        self.last_t = t
        b, a = self.best()
        if self.hist_t and self.hist_b[-1] == b and self.hist_a[-1] == a:
            return
        if self.hist_t and self.hist_t[-1] == t:
            self.hist_b[-1], self.hist_a[-1] = b, a
            return
        self.hist_t.append(t)
        self.hist_b.append(b)
        self.hist_a.append(a)

    def at(self, t: int) -> tuple[int | None, int | None]:
        import bisect
        i = bisect.bisect_right(self.hist_t, t) - 1
        return (self.hist_b[i], self.hist_a[i]) if i >= 0 else (None, None)


def _px(v: Any) -> int:
    return round(float(v) * 10000)


def _replay(path: Path) -> tuple[dict[str, _Book], dict[tuple, list]]:
    """Replay one hourly file: book per market on the exchange clock (``ts_ms``; snapshots at the
    receive time) and the taker orders, (market, ts_ms, side) -> [q1, qw, contracts, notional]
    in taker-side 1/100-cent units."""
    books: dict[str, _Book] = {}
    trades: list[tuple] = []
    for recv_ns, fr in _iter_frames(path):
        typ = fr.get("type")
        msg = fr.get("msg") or {}
        mk = msg.get("market_ticker")
        if not mk:
            continue
        if typ == "orderbook_snapshot":
            bk = books.setdefault(mk, _Book())
            bk.snapshot({_px(p): float(s) for p, s in msg.get("yes_dollars_fp") or []},
                        {_px(p): float(s) for p, s in msg.get("no_dollars_fp") or []})
            bk.mark(recv_ns // 1_000_000)
        elif typ == "orderbook_delta":
            bk = books.get(mk)
            if bk is None:
                continue
            bk.delta(msg.get("side"), _px(msg["price_dollars"]), float(msg["delta_fp"]))
            bk.mark(int(msg.get("ts_ms") or recv_ns // 1_000_000))
        elif typ == "trade":
            ts = int(msg.get("ts_ms") or int(msg.get("ts", 0)) * 1000)
            trades.append((mk, ts, msg["taker_side"], _px(msg["yes_price_dollars"]), float(msg["count_fp"])))
    orders: dict[tuple, list] = {}
    for mk, ts, side, p, c in trades:
        o = orders.setdefault((mk, ts, side), [10**9, -1, 0.0, 0.0])
        q = p if side == "yes" else 10000 - p          # taker-side price
        o[0], o[1] = min(o[0], q), max(o[1], q)
        o[2] += c
        o[3] += c * q / 10000
    return books, orders


def l2_stub_file(path: Path, depths: tuple[int, ...], lat_ms: int) -> tuple[list[dict], list[dict]]:
    """Exact-book stub replay of one hourly file. For each depth D a 1-contract stub rests on
    each side at (best as of ``lat_ms`` before the order) + D and fills when the order prints
    strictly beyond it; after a fill that side's stub returns ``lat_ms`` later. Also returns the
    stub exposure (seconds live and collateral-seconds) per market and side."""
    books, orders = _replay(path)
    fills: list[dict] = []
    last_fill: dict[tuple, int] = {}
    for (mk, ts, side), (q1, qw, ct, notional) in sorted(orders.items(), key=lambda kv: kv[0][1]):
        bk = books.get(mk)
        if bk is None:
            continue
        rb, ra = bk.at(ts - lat_ms)                    # YES best bid / ask as of ts - latency
        ref = (ra if side == "yes" else (10000 - rb if rb is not None else None))
        cb, ca = bk.at(ts - 1)
        for D in depths:
            L_print = q1 + D * 100
            row = {"ticker": mk, "ts_ms": ts, "d": 1 if side == "yes" else -1, "D": D, "q1": q1 / 10000,
                   "qw": qw / 10000, "contracts": ct, "notional": notional,
                   "pre_bid": None if cb is None else cb / 10000, "pre_ask": None if ca is None else ca / 10000,
                   "print_fill": qw > L_print and L_print <= 9900, "L_print": L_print / 10000}
            book_fill = False
            L = None
            if ref is not None:
                L = ref + D * 100
                key = (mk, side, D)
                active = ts - last_fill.get(key, -10**15) >= lat_ms
                book_fill = active and qw > L and L <= 9900
                if book_fill:
                    last_fill[key] = ts
            row["book_fill"] = book_fill
            row["L_book"] = None if L is None else L / 10000
            if row["print_fill"] or book_fill:
                fills.append(row)
    expo: list[dict] = []
    for mk, bk in books.items():
        if not bk.hist_t:
            continue
        ts = bk.hist_t + [bk.last_t]
        for D in depths:
            sec = {"ask": 0.0, "bid": 0.0}
            col = {"ask": 0.0, "bid": 0.0}
            for i in range(len(bk.hist_t)):
                dt = (ts[i + 1] - ts[i]) / 1000
                b, a = bk.hist_b[i], bk.hist_a[i]
                if a is not None and a + D * 100 <= 9900:
                    sec["ask"] += dt
                    col["ask"] += dt * (1 - (a + D * 100) / 10000)
                if b is not None and b - D * 100 >= 100:
                    sec["bid"] += dt
                    col["bid"] += dt * (b - D * 100) / 10000
            expo.append({"ticker": mk, "D": D, "file": path.name, "day": path.parent.name,
                         "sec_ask": sec["ask"], "sec_bid": sec["bid"], "col_ask": col["ask"], "col_bid": col["bid"]})
    return fills, expo


def l2_orders_file(path: Path, post_ms: int = 1000, min_depth_c: int = 10, max_usd: float = 50.0) -> list[dict]:
    """Every taker order of one recorded hour (taker-side prices) and, for R5 triggers (span >=
    ``min_depth_c``, notional <= ``max_usd``), the YES best bid/ask ``post_ms`` after the order."""
    books, orders = _replay(path)
    rows = []
    for (mk, ts, side), (q1, qw, ct, notional) in orders.items():
        r = {"ticker": mk, "ts_ms": ts, "d": 1 if side == "yes" else -1, "q1": q1 / 10000, "qw": qw / 10000,
             "contracts": ct, "notional": notional, "post_bid": None, "post_ask": None}
        bk = books.get(mk)
        if bk is not None and qw - q1 >= min_depth_c * 100 and notional <= max_usd and ts + post_ms <= bk.last_t:
            b_, a_ = bk.at(ts + post_ms)
            r["post_bid"] = None if b_ is None else b_ / 10000
            r["post_ask"] = None if a_ is None else a_ / 10000
        rows.append(r)
    return rows


def step_l2_orders(start: str = "2026-11-07", end: str = "2026-11-18") -> None:
    """Taker orders of the k-recorder files (for the exact-book R5 test), one file at a time."""
    d = OUT / "l2" / "orders"
    d.mkdir(parents=True, exist_ok=True)
    files = [p for p in k_files(start, end)
             if not (d / f"orders_{p.parent.name}_{p.name.split('.')[0]}.parquet").exists()]
    print(f"l2 orders: {len(files)} files", flush=True)
    for i, p in enumerate(files, 1):
        rows = l2_orders_file(p)
        if rows:
            _save(pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.lit(p.parent.name).alias("day")),
                  d / f"orders_{p.parent.name}_{p.name.split('.')[0]}.parquet")
        if i % 50 == 0 or i == len(files):
            print(f"  {datetime.now(UTC):%H:%M:%S} {i}/{len(files)}", flush=True)


def l2_orders_scan() -> pl.LazyFrame:
    """All extracted L2 orders (files with no trigger store the post columns as Null: cast)."""
    files = sorted((OUT / "l2" / "orders").glob("orders_*.parquet"))
    return pl.concat([pl.scan_parquet(f).with_columns(pl.col("post_bid").cast(pl.Float64),
                                                      pl.col("post_ask").cast(pl.Float64)) for f in files],
                     how="vertical_relaxed")


def l2_refill(day_from: str, day_to: str) -> pl.DataFrame:
    """R5 on the recorded books: triggers on [day_from, day_to] whose post price P = qw - 1c lies
    strictly inside the YES spread 1 s later (in YES terms: P for a YES sweep, 1 - P for a NO
    sweep); fills from later same-side orders strictly beyond P within 60 min (all recorded
    hours, so the window crosses file boundaries)."""
    o = l2_orders_scan().collect().unique(["ticker", "ts_ms", "d"])
    o = o.with_columns(pl.from_epoch("ts_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("t"))
    trig = o.filter(pl.col("day").is_between(pl.lit(day_from), pl.lit(day_to)) & pl.col("post_bid").is_not_null()
                    & pl.col("post_ask").is_not_null())
    P = (pl.col("qw") - 0.01).round(4)
    yes_px = pl.when(pl.col("d") == 1).then(P).otherwise(1 - P)
    trig = trig.with_columns(P.alias("P")).filter((yes_px > pl.col("post_bid") + 1e-9) & (yes_px < pl.col("post_ask") - 1e-9)
                                                  & (pl.col("P") > 0.01))
    later = o.select("ticker", "d", pl.col("t").alias("t2"), pl.col("qw").alias("qw2")).sort("ticker", "d", "t2")
    j = trig.select("ticker", "d", "t", "P", pl.col("notional").alias("trig_usd")).join(later, on=["ticker", "d"])
    j = j.filter((pl.col("t2") > pl.col("t") + timedelta(seconds=1)) & (pl.col("t2") <= pl.col("t") + timedelta(minutes=60))
                 & (pl.col("qw2") > pl.col("P") + 1e-9))
    f = j.sort("t2").unique(["ticker", "d", "t"], keep="first").rename({"t": "t_trig"}).rename({"t2": "t"})
    out = pl.read_parquet(OUT / "l2" / "outcomes.parquet").filter(pl.col("result").is_in(["yes", "no"]))
    out = out.with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                           (pl.col("fee_type").fill_null("") == "quadratic_with_maker_fees").alias("maker_fee"),
                           pl.col("fee_multiplier").fill_null(1.0).alias("mult"))
    f = f.join(out.select("ticker", "event_ticker", "y", "maker_fee", "mult", "settlement_ts", "close_time"),
               on="ticker", how="left")
    f = f.with_columns(pl.when(pl.col("d") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("won"),
                       pl.col("t").dt.convert_time_zone("America/New_York").dt.date().alias("etd"), pl.col("P").alias("L"))
    return f.with_columns((pl.col("P") - pl.col("won") - maker_fee_expr(pl.col("P"))).alias("pnl"))


def step_l2_stubs(start: str = "2026-11-07", end: str = "2026-11-18", lat_ms: int = 1000,
                  depths: tuple[int, ...] = (3, 5, 10, 15)) -> None:
    """Exact-book stub replay over the k-recorder files, one file at a time (bounded memory)."""
    d = OUT / "l2"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"lat{lat_ms}"
    done_p = d / f"done_{tag}.json"
    done = set(json.loads(done_p.read_text())) if done_p.exists() else set()
    files = [p for p in k_files(start, end) if str(p) not in done]
    print(f"l2 stubs {tag}: {len(files)} files to replay", flush=True)
    for i, p in enumerate(files, 1):
        fills, expo = l2_stub_file(p, depths, lat_ms)
        stem = f"{p.parent.name}_{p.name.split('.')[0]}"
        if fills:
            _save(pl.DataFrame(fills, infer_schema_length=None).with_columns(pl.lit(p.parent.name).alias("day")),
                  d / tag / f"fills_{stem}.parquet")
        if expo:
            _save(pl.DataFrame(expo), d / tag / f"expo_{stem}.parquet")
        done.add(str(p))
        if i % 25 == 0 or i == len(files):
            done_p.write_text(json.dumps(sorted(done)))
            print(f"  {datetime.now(UTC):%H:%M:%S} {i}/{len(files)} {p}", flush=True)


def step_l2_outcomes() -> None:
    """Settlement results of every market with an L2 stub fill (public /markets, <= 1 request/s)."""
    from albert.venues import kalshi as k
    fills = pl.concat([pl.scan_parquet(str(OUT / "l2" / "lat*" / "fills_*.parquet")).select("ticker").unique().collect()]
                      + ([l2_orders_scan().filter(
                          pl.col("post_bid").is_not_null()).select("ticker").unique().collect()]
                         if any((OUT / "l2" / "orders").glob("orders_*.parquet")) else [])).unique()
    have = OUT / "l2" / "outcomes.parquet"
    known = pl.read_parquet(have) if have.exists() else None
    todo = sorted(set(fills["ticker"]) - (set(known.filter(pl.col("result").is_in(["yes", "no"]))["ticker"])
                                          if known is not None else set()))
    api = _api()
    info = series_info()
    rows: list[dict[str, Any]] = []
    day = datetime.now(UTC).strftime("%Y-%m-%dT%H")   # cache tag: refreshed results, raw kept append-only
    for i in range(0, len(todo), 100):
        page = api.client.get("/markets", {"tickers": ",".join(todo[i:i + 100]), "limit": 1000}, tag=day)
        rows.extend(k.market_rows(page.payload, info, "live"))
    new = pl.DataFrame(rows, infer_schema_length=None) if rows else None
    if new is not None:
        new = new.select("ticker", "event_ticker", "series_ticker", "category", "result", "close_time",
                         "settlement_ts", "fee_type", "fee_multiplier", "expected_expiration_time")
        out = pl.concat([known.filter(~pl.col("ticker").is_in(new["ticker"].implode())), new],
                        how="vertical_relaxed") if known is not None else new
        _save(out, have)
    print(f"outcomes: {len(todo)} requested, {0 if new is None else new.height} returned")


def l2_fills(day_from: str, day_to: str, lat_ms: int = 1000) -> pl.DataFrame:
    """L2 stub fills (book-based and print-based) on days [day_from, day_to] joined to outcomes;
    PnL per contract for the book-based stub (held to settlement, maker fee rounded up)."""
    f = pl.scan_parquet(str(OUT / "l2" / f"lat{lat_ms}" / "fills_*.parquet")).filter(
        pl.col("day").is_between(pl.lit(day_from), pl.lit(day_to))).collect()
    o = pl.read_parquet(OUT / "l2" / "outcomes.parquet").filter(pl.col("result").is_in(["yes", "no"]))
    o = o.with_columns((pl.col("result") == "yes").cast(pl.Int8).alias("y"),
                       (pl.col("fee_type").fill_null("") == "quadratic_with_maker_fees").alias("maker_fee"),
                       pl.col("fee_multiplier").fill_null(1.0).alias("mult"))
    from albert.research.flb_common import coarse_category
    o = o.with_columns(pl.struct("category", "series_ticker").map_elements(
        lambda r: coarse_category(r["category"], r["series_ticker"]), return_dtype=pl.String).alias("group"))
    f = f.join(o.select("ticker", "event_ticker", "group", "y", "maker_fee", "mult", "settlement_ts", "close_time"),
               on="ticker", how="left")
    f = f.with_columns(pl.from_epoch("ts_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("t"))
    f = f.with_columns(pl.when(pl.col("d") == 1).then(pl.col("y")).otherwise(1 - pl.col("y")).alias("won"),
                       pl.col("t").dt.convert_time_zone("America/New_York").dt.date().alias("etd"))
    return f.with_columns(
        (pl.col("L_book") - pl.col("won") - maker_fee_expr(pl.col("L_book"))).alias("pnl_book"),
        (pl.col("L_print") - pl.col("won") - maker_fee_expr(pl.col("L_print"))).alias("pnl_print"))


def l2_summary(day_from: str, day_to: str, lat_ms: int = 1000) -> dict[str, Any]:
    """Book-based versus print-based stub fills on the recorded k-universe: agreement, PnL per
    fill at settlement, fill rate per stub-hour and resting collateral."""
    f = l2_fills(day_from, day_to, lat_ms)
    e = pl.scan_parquet(str(OUT / "l2" / f"lat{lat_ms}" / "expo_*.parquet")).filter(
        pl.col("day").is_between(pl.lit(day_from), pl.lit(day_to))).collect()
    out: dict[str, Any] = {"days": [day_from, day_to], "lat_ms": lat_ms, "by_depth": []}
    for D in sorted(f["D"].unique().to_list()):
        x = f.filter(pl.col("D") == D)
        bk = x.filter(pl.col("book_fill"))
        pr = x.filter(pl.col("print_fill"))
        ex = e.filter(pl.col("D") == D)
        stub_h = (ex["sec_ask"].sum() + ex["sec_bid"].sum()) / 3600
        col_h = (ex["col_ask"].sum() + ex["col_bid"].sum()) / 3600
        settled = bk.filter(pl.col("won").is_not_null())
        r = cluster_stats(settled, "pnl_book", "event_ticker", "etd")
        rp = cluster_stats(pr.filter(pl.col("won").is_not_null()), "pnl_print", "event_ticker", "etd")
        out["by_depth"].append({
            "D": D, "book_fills": bk.height, "print_fills": pr.height,
            "both": x.filter(pl.col("book_fill") & pl.col("print_fill")).height,
            "book_only": x.filter(pl.col("book_fill") & ~pl.col("print_fill")).height,
            "print_only": x.filter(~pl.col("book_fill") & pl.col("print_fill")).height,
            "book_settled": settled.height, "book_pnl": r, "print_pnl": rp,
            "book_L_minus_print_L_c": round(100 * float((bk["L_book"] - bk["L_print"]).mean()), 3) if bk.height else None,
            "stub_hours": round(stub_h, 1), "fills_per_100_stub_hours": round(100 * bk.height / stub_h, 3) if stub_h else None,
            "mean_collateral_per_stub": round(col_h / stub_h, 3) if stub_h else None,
            "pnl_usd_per_stub_day": (round(float(settled["pnl_book"].sum()) / stub_h * 24, 4)
                                     if stub_h and settled.height else None),
        })
    return out


L2_STUB_BUDGET = 60.0     # dollars of resting stub collateral; the other $40 backs filled positions


def l2_ledger(day_from: str, day_to: str, D: int, split: str, lat_ms: int = 1000) -> tuple[pl.DataFrame, dict]:
    """$100 book for the exact-book stub: the k-universe needs more resting collateral than $60,
    so stubs rest in a seeded random share of its markets (share = budget / mean concurrent
    stub collateral); filled positions are capped at $40 open and $5 per event."""
    e = pl.scan_parquet(str(OUT / "l2" / f"lat{lat_ms}" / "expo_*.parquet")).filter(
        pl.col("day").is_between(pl.lit(day_from), pl.lit(day_to)) & (pl.col("D") == D)).collect()
    hours = e.select(pl.concat_str("day", "file")).n_unique() / max(1, e.select(
        pl.col("file").str.extract(r"-(k\d+)", 1)).n_unique())
    conc = float((e["col_ask"].sum() + e["col_bid"].sum()) / 3600 / max(hours, 1.0))
    share = min(1.0, L2_STUB_BUDGET / conc) if conc else 1.0
    f = l2_fills(day_from, day_to, lat_ms).filter((pl.col("D") == D) & pl.col("book_fill"))
    f = f.filter((pl.col("ticker").hash(NOV_SEED) % 10_000) < int(share * 10_000))
    f = f.rename({"L_book": "L_px"}).with_columns(pl.col("L_px").alias("L"))
    led = stub_ledger(f, "stub_l2", f"D{D}_lat{lat_ms}", split, open_cap=BANKROLL - L2_STUB_BUDGET)
    info = {"mean_concurrent_collateral_all": round(conc, 1), "share_of_markets": round(share, 3),
            "fills": f.height, **cluster_stats(f.filter(pl.col("won").is_not_null()), "pnl_book", "event_ticker", "etd")}
    return led, info


# ---- ledgers and risk metrics --------------------------------------------------------------------
def stub_ledger(fills: pl.DataFrame, strategy: str, variant: str, split: str, *, price: str = "L",
                per_event: float = 5.0, open_cap: float = 50.0) -> pl.DataFrame:
    """Capped $100 book: every fill is one contract of the side opposite the taker, bought at
    1 - L plus the maker fee, held to settlement. A fill is skipped if it would lift the open cost
    above ``open_cap`` (the rest of the bankroll backs the resting stubs) or the open cost in its
    event above ``per_event``."""
    import heapq
    f = fills.filter(pl.col("won").is_not_null() & pl.col(price).is_not_null()).with_columns(
        pl.coalesce("settlement_ts", "close_time").alias("settle_ts")).sort("t")
    rows = []
    heap: list[tuple] = []
    open_tot = 0.0
    open_ev: dict[str, float] = {}
    for i, r in enumerate(f.iter_rows(named=True)):
        while heap and heap[0][0] <= r["t"]:
            _, c, ev = heapq.heappop(heap)
            open_tot -= c
            open_ev[ev] -= c
        L = r[price]
        fee = fee_cents(1, L, r["mult"] or 1.0, KALSHI_MAKER) if r["maker_fee"] else 0.0
        cost = (1 - L) + fee
        if open_tot + cost > open_cap + 1e-9 or open_ev.get(r["event_ticker"], 0.0) + cost > per_event + 1e-9:
            continue
        open_tot += cost
        open_ev[r["event_ticker"]] = open_ev.get(r["event_ticker"], 0.0) + cost
        heapq.heappush(heap, (r["settle_ts"], cost, r["event_ticker"]))
        rows.append({"strategy": strategy, "variant": variant, "split": split, "struct_id": f"{variant}-{split}-{i}",
                     "event_ticker": r["event_ticker"], "ticker": r["ticker"],
                     "side": "no" if r["d"] == 1 else "yes", "n": 1, "t_in": r["t"], "cost_usd": cost,
                     "settle_ts": r["settle_ts"], "payoff_usd": float(1 - r["won"])})
    return pl.DataFrame(rows, infer_schema_length=None)


def daily_activity(split: str) -> pl.DataFrame:
    """Taker orders per market and UTC day (all orders of the split's samples, any time)."""
    out = []
    samples = ("rs",) if split == "dev" else ("rs", "nov")
    for sample in samples:
        for p in sorted((OUT / "tape" / sample).glob("part_*.parquet")):
            od = pl.read_parquet(p, columns=["ticker", "t", "close_time"])
            out.append(od.group_by("ticker", pl.col("t").dt.date().alias("day"))
                       .agg(pl.len().alias("orders"), pl.col("close_time").first()))
    return pl.concat(out)


def stub_universe(act: pl.DataFrame, k: int) -> pl.DataFrame:
    """Ex-ante stub universe: on each UTC day, the ``k`` markets (still open at the day's start)
    with the most taker orders on the previous day. Returns (ticker, day) pairs."""
    prev = act.with_columns((pl.col("day") + timedelta(days=1)).alias("day"))
    prev = prev.filter(pl.col("close_time") > pl.col("day").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    prev = prev.with_columns(pl.col("orders").rank("ordinal", descending=True).over("day").alias("rk"))
    return prev.filter(pl.col("rk") <= k).select("ticker", "day")


def stub_universe_band(act: pl.DataFrame, k: int, lo: int = 10, hi: int = 200) -> pl.DataFrame:
    """Ex-ante stub universe of the deployable rule: on each UTC day, among markets still open
    that had ``lo``..``hi`` taker orders on the previous day, the ``k`` with the most."""
    prev = act.with_columns((pl.col("day") + timedelta(days=1)).alias("day"))
    prev = prev.filter((pl.col("close_time") > pl.col("day").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
                       & pl.col("orders").is_between(lo, hi))
    prev = prev.with_columns(pl.col("orders").rank("ordinal", descending=True).over("day").alias("rk"))
    return prev.filter(pl.col("rk") <= k).select("ticker", "day")


def resting_collateral(depth_c: int, spread: float = 0.05) -> float:
    """Collateral of the two 1-contract stubs in one market: NO at 1 - (ask + D) plus YES at
    bid - D, i.e. 1 - spread - 2D (Kalshi reserves the cost of every resting buy)."""
    return 1 - spread - 2 * depth_c / 100


def ledger_summary(led: pl.DataFrame) -> dict[str, Any]:
    from albert.research import vol_metrics as vm
    if not led.height:
        return {}
    m = vm.ledger_metrics(led, vm.load_marks(led["ticker"].unique().to_list()))
    keep = ("positions", "events", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90", "sharpe_ex_top5",
            "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct", "days", "marked_share", "sharpe_settle_basis")
    return {k: m.get(k) for k in keep}


# ---- the stub study: dev and holdout tables, ledgers ---------------------------------------------
DEV_CELLS = [(D, t) for D in (3, 5, 8, 10, 15) for t in (0.0, 2.0, 60.0)]
PREREG_CELLS = [(10, 2.0), (15, 2.0)]
# ledgers only where the sample is dense (every stratum's markets are trading): dev closes start
# 2026-08-15; the holdout runs to the last nov close
BANKROLL = 100.0
UNIVERSE_K = (30, 60, 90)      # markets with resting stubs at once (collateral-bound on $100)
PRIMARY_K = 60
BAND = (10, 200)               # deployable universe: previous-day taker orders (chosen on dev)
LEDGER_WINDOW = {"dev": (datetime(2026, 8, 15, tzinfo=UTC), DEV_END), "holdout": (DEV_END, NOV_END)}
GROUP_CUTS = ("Sports", "Mentions", "Weather", "Politics", "Entertainment", "Science & Tech")


def breakdown(f: pl.DataFrame, by: str) -> list[dict]:
    out = []
    for k in sorted(f[by].drop_nulls().unique().to_list(), key=str):
        r = cluster_stats(f.filter(pl.col(by) == k), "pnl", "event_ticker", "etd")
        out.append({by: str(k), **r})
    return out


def step_stubs(split: str = "dev") -> None:
    """Stub table for a split (dev: every cell; holdout: the pre-registered cells plus the same
    diagnostics), breakdowns, ledgers and risk metrics."""
    cells = DEV_CELLS if split == "dev" else PREREG_CELLS + [(10, 0.0), (15, 0.0), (10, 60.0), (15, 60.0)]
    fills = stub_split(split, cells)
    ctrl = flb_control(split)
    act = daily_activity(split)
    fx, rows = stub_table(fills, ctrl, cells)
    _save(fx, OUT / f"stub_fills_{split}.parquet")
    res: dict[str, Any] = {"split": split, "table": rows, "breakdown": {}, "ledgers": {}}
    for D, tau in PREREG_CELLS:
        f = fx.filter((pl.col("D") == D) & (pl.col("tau") == tau)).with_columns(
            pl.col("L").cut([0.1, 0.25, 0.5, 0.75, 0.9]).cast(pl.String).alias("L_band"),
            pl.col("t").dt.strftime("%Y-%m").alias("month"),
            pl.col("notional").cut([10, 50, 200, 1000]).cast(pl.String).alias("sweep_usd"),
            pl.col("sample").alias("sample_"))
        key = f"D{D}_tau{int(tau)}"
        res["breakdown"][key] = {by: breakdown(f, by) for by in ("group", "L_band", "month", "sweep_usd", "sample_")}
        lo, hi = LEDGER_WINDOW[split]
        fw = f.filter(pl.col("t").is_between(lo, hi, closed="left"))
        led = stub_ledger(fw, "stub", f"{key}_all", split)    # no resting-collateral limit (upper bound)
        if led.height:
            res["ledgers"][f"{key}_all_markets"] = ledger_summary(led)
        for kk in UNIVERSE_K:
            uni = stub_universe(act, kk)
            fk = fw.with_columns(pl.col("t").dt.date().alias("day")).join(uni, on=["ticker", "day"], how="semi")
            cap = BANKROLL - kk * resting_collateral(D)
            led = stub_ledger(fk, "stub", f"{key}_K{kk}", split, open_cap=max(cap, 0.0))
            res["ledgers"][f"{key}_K{kk}"] = {"fills_in_universe": fk.height, "open_cap": round(cap, 2),
                                              **cluster_stats(fk, "pnl", "event_ticker", "etd"),
                                              **(ledger_summary(led) if led.height else {})}
            if led.height and kk == PRIMARY_K:
                _save(led, OUT / "ledgers" / f"ledger_stub_{key}_{split}.parquet")
        uni = stub_universe_band(act, PRIMARY_K)
        fk = fw.with_columns(pl.col("t").dt.date().alias("day")).join(uni, on=["ticker", "day"], how="semi")
        cap = BANKROLL - PRIMARY_K * resting_collateral(D)
        led = stub_ledger(fk, "stub", f"{key}_band", split, open_cap=cap)
        res["ledgers"][f"{key}_band"] = {"fills_in_universe": fk.height, "open_cap": round(cap, 2),
                                         **cluster_stats(fk, "pnl", "event_ticker", "etd"),
                                         **(ledger_summary(led) if led.height else {})}
        if led.height:
            _save(led, OUT / "ledgers" / f"ledger_stub_{key}_band_{split}.parquet")
    _dump(res, OUT / f"stubs_{split}.json")
    print(json.dumps(_jsonable(res["table"]), indent=0)[:3000])
    print(json.dumps(_jsonable(res["ledgers"]), indent=1))


def step_refill(split: str = "dev") -> None:
    """R5 (post-small-sweep refill) on the print tape: per-fill stats overall and by sample /
    group / trigger size, and the $100 ledger ($90 open cap, $5 per event)."""
    out = [f for od in iter_orders(split) if (f := refill_fills(od)).height]
    f = pl.concat(out, how="vertical_relaxed").with_columns(
        pl.col("trig_usd").cut([10, 25]).cast(pl.String).alias("trig_band"))
    _save(f, OUT / f"refill_fills_{split}.parquet")
    res: dict[str, Any] = {"split": split, "all": cluster_stats(f, "pnl", "event_ticker", "etd"),
                           "by": {by: breakdown(f, by) for by in ("sample", "group", "trig_band")}, "ledgers": {}}
    lo, hi = LEDGER_WINDOW[split]
    for name, sub in [("all", f)] + ([(smp, f.filter(pl.col("sample") == smp)) for smp in ("rs", "nov")]
                                    if split == "holdout" else []):
        led = stub_ledger(sub.filter(pl.col("t").is_between(lo, hi, closed="left")), "refill", f"R5_{name}", split,
                          open_cap=90.0)
        if led.height:
            _save(led, OUT / "ledgers" / f"ledger_refill_R5_{name}_{split}.parquet")
            res["ledgers"][name] = ledger_summary(led)
    _dump(res, OUT / f"refill_{split}.json")
    print(json.dumps(_jsonable({k: res[k] for k in ("all", "ledgers")}), indent=1))


# ---- CLI -----------------------------------------------------------------------------------------
STEPS = {
    "nov-plan": step_nov_plan,
    "nov-fetch": step_nov_fetch,
    "fee-changes": step_fee_changes,
    "tape": step_tape,
    "l2-stubs": step_l2_stubs,
    "l2-outcomes": step_l2_outcomes,
    "l2-orders": step_l2_orders,
    "stubs-dev": lambda: step_stubs("dev"),
    "stubs-holdout": lambda: step_stubs("holdout"),
    "refill-dev": lambda: step_refill("dev"),
    "refill-holdout": lambda: step_refill("holdout"),
    "tape-nov": lambda: step_tape(("nov",)),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="albert.research.behav_micro")
    ap.add_argument("step", choices=sorted(STEPS))
    a = ap.parse_args(argv)
    STEPS[a.step]()


if __name__ == "__main__":
    main()
