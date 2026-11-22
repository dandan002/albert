"""Adversarial review of culture_flow's exploratory maker-NO rule (``makerNO|m12|b5|h72|all``).

The claim (``data/research/culture_flow/report.md``): at each event's close - 72 h, on every leg
of 37 culture / approval / executive-action series whose mid is <= 5c, rest a 1-lot NO bid that
sells YES at ceil(mid + 12c), held to settlement; a fill is a later HOURLY CANDLE whose trade high
is strictly above that price. This module re-scores it without touching culture_flow's code or
outputs (it reads them; everything it writes goes to ``data/research/culture_flow_review`` and the
cache ``data/cache/culture_flow_review``):

1. fills from Kalshi's own trade prints (``/markets/trades``, 1 worker, rate-limited) for every
   order the rule placed, with the taker sweep that went through the price and the flow after it;
2. an exact-book replay (recorded L2, ``k<n>`` recorder files, FIFO queue at our price) of the
   orders whose legs the stratified recorder happened to subscribe;
3. look-ahead (actual vs scheduled close, quote age, one-sided books), selection, concentration
   and the cash-aware book with Kalshi's collateral reservation for resting bids.

Kalshi only. Never places orders.

    python -m albert.research.culture_flow_review orders
    python -m albert.research.culture_flow_review trades --split holdout
    python -m albert.research.culture_flow_review prints
    python -m albert.research.culture_flow_review stream-fetch --days 2026-11-14 2026-11-15 2026-11-16
    python -m albert.research.culture_flow_review l2
    python -m albert.research.culture_flow_review report
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

SRC = Path("data/research/culture_flow")          # the claim's outputs (read only)
OUT = Path("data/research/culture_flow_review")
CACHE = Path("data/cache/culture_flow_review")
STREAM = CACHE / "stream" / "kalshi"
MARGIN = 0.12
BAND = (0.0, 0.05)
H = 72
NAMED_AT = datetime(2026, 11, 20, 23, 31, 58, tzinfo=UTC)     # exploratory_frozen.json written_utc
FRESH = datetime(2026, 11, 17, tzinfo=UTC)
PRE_WINDOW = timedelta(hours=2)                              # prints fetched from t - 2 h
L2_FIRST_DAY = datetime(2026, 11, 14, tzinfo=UTC)            # first recorder day replayed


def _save(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def sell_price(mid: float) -> float:
    """The claim's sell price: ceil(mid + 12c) to the cent (culture_flow.maker_no)."""
    return math.ceil(round((mid + MARGIN) * 100, 6)) / 100


# ---- 0. the rule's orders ------------------------------------------------------------------------
def build_orders(split: str) -> pl.DataFrame:
    """Every order of the rule on a split, rebuilt from culture_flow's quoted panel exactly as
    ``culture_flow.maker_no`` does, plus the candle fill the claim used (from its saved fills)."""
    o = pl.read_parquet(SRC / "orders.parquet").filter((pl.col("split") == split) & (pl.col("h") == H))
    o = o.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid")).filter(
        (pl.col("mid") > BAND[0]) & (pl.col("mid") <= BAND[1]))
    o = o.with_columns(pl.col("mid").map_elements(sell_price, return_dtype=pl.Float64).alias("s"))
    o = o.filter((pl.col("s") < 0.99) & (pl.col("s") > pl.col("bid")))
    fills = pl.read_parquet(SRC / f"explore_makerNO_m12_b5_h72_{split}.parquet").select(
        "ticker", pl.col("t").alias("candle_fill_ts"), "fill_candle_volume", pl.col("pnl_c").alias("candle_pnl_c"))
    o = o.join(fills, on="ticker", how="left")
    return o.select(
        "series_ticker", "event_ticker", "ticker", "family", "split", "strike_type", "open_time", "close_time",
        "expected_expiration_time", "settlement_ts", "y", "ev_close", "ev_open", "t", "q_ts", "bid", "ask", "mid", "s",
        ((pl.col("t") - pl.col("q_ts")).dt.total_minutes() / 60).alias("quote_age_h"),
        (pl.col("bid") <= 0).alias("bid_empty"), "candle_fill_ts", "fill_candle_volume", "candle_pnl_c")


def step_orders() -> None:
    for split in ("dev", "holdout"):
        o = build_orders(split)
        _save(o, OUT / f"orders_{split}.parquet")
        print(f"{split}: {o.height} orders, {o['event_ticker'].n_unique()} events, "
              f"{o['candle_fill_ts'].is_not_null().sum()} candle fills")


# ---- 1a. trade prints ----------------------------------------------------------------------------
def _api(min_interval: float = 0.5):
    from albert.config import Settings
    from albert.http import CachedClient
    from albert.venues import kalshi as k

    s = Settings()
    return k.Kalshi(CachedClient("kalshi", s.kalshi_base, CACHE / "raw", min_interval=min_interval, max_retries=12))


def step_trades(split: str, limit: int | None = None, min_interval: float = 0.5) -> None:
    """Every trade print of each order's leg in [t - 2 h, close + 1 min], one leg at a time (1
    worker). Legs settled before Kalshi's trades cutoff go to ``/historical/trades``. Parts are
    flushed every 100 legs so a restart resumes."""
    from albert.venues import kalshi as k

    api = _api(min_interval)
    cut = api.cutoff()["trades_created_ts"]
    o = pl.read_parquet(OUT / f"orders_{split}.parquet").sort("t")
    d = CACHE / "trades" / split
    d.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    for f in d.glob("part-*.parquet"):
        done |= set(pl.read_parquet(f, columns=["leg"])["leg"].unique().to_list())
    todo = [r for r in o.iter_rows(named=True) if r["ticker"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"trades {split}: {len(todo)} legs to fetch ({len(done)} done), cutoff {cut}", flush=True)
    rows: list[dict[str, Any]] = []
    legs: list[str] = []
    nparts = len(list(d.glob("part-*.parquet")))
    t0 = datetime.now(UTC)

    def flush() -> None:
        nonlocal rows, legs, nparts
        if not legs:
            return
        df = pl.DataFrame(rows, schema={"leg": pl.String, "trade_id": pl.String, "created_time": pl.Datetime("us", "UTC"),
                                        "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String})
        # legs with no trades still get a marker row so they count as done
        empty = sorted(set(legs) - set(df["leg"].unique().to_list()))
        if empty:
            df = pl.concat([df, pl.DataFrame({"leg": empty}).with_columns(
                [pl.lit(None, dtype=df.schema[c]).alias(c) for c in df.columns if c != "leg"])], how="vertical")
        nparts += 1
        _save(df, d / f"part-{nparts:04d}.parquet")
        rows, legs = [], []

    for i, r in enumerate(todo, 1):
        hist = cut is not None and r["close_time"] < cut
        for page in api.trades(r["ticker"], hist, min_ts=r["t"] - PRE_WINDOW, max_ts=r["close_time"] + timedelta(minutes=1)):
            for tr in k.trade_rows(page.payload):
                rows.append({"leg": r["ticker"], **{c: tr[c] for c in ("trade_id", "created_time", "count", "yes_price",
                                                                      "taker_side")}})
        legs.append(r["ticker"])
        if i % 100 == 0:
            flush()
            el = (datetime.now(UTC) - t0).total_seconds()
            print(f"  {i}/{len(todo)} legs, {el / i:.2f} s/leg", flush=True)
    flush()
    print("trades done", flush=True)


def load_trades(split: str) -> pl.DataFrame:
    d = CACHE / "trades" / split
    files = sorted(d.glob("part-*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical").unique(["leg", "trade_id"], keep="first")


def print_fills(o: pl.DataFrame, tr: pl.DataFrame, *, s_col: str = "s", t_col: str = "t") -> pl.DataFrame:
    """Per order: the first print strictly above the sell price after the order time (a YES buy
    can only print above our resting YES offer once our level is gone, so this is a fill of the
    1-lot at any queue position), the taker order that went through (all prints with the same
    timestamp and side), contracts printed above the price (what a larger order could have
    filled), prints at exactly the price, through-prints in the hour before the order (what an
    hourly candle that straddles the order time would count), and the flow after the fill."""
    rows = []
    by = {k[0]: g.sort("created_time") for k, g in tr.filter(pl.col("trade_id").is_not_null()).group_by(["leg"])}
    for r in o.iter_rows(named=True):
        g = by.get(r["ticker"])
        s, t, close = r[s_col], r[t_col], r["close_time"]
        out: dict[str, Any] = {"ticker": r["ticker"], "t_used": t, "s_used": s}
        if g is None:
            rows.append({**out, "print_fill_ts": None})
            continue
        a = g.filter((pl.col("created_time") > t) & (pl.col("created_time") <= close))
        pre = g.filter((pl.col("created_time") > t - timedelta(hours=1)) & (pl.col("created_time") <= t))
        thr = a.filter(pl.col("yes_price") > s + 1e-9)
        out.update({"trades_after": a.height, "vol_after": float(a["count"].sum()),
                    "vol_through": float(thr["count"].sum()),
                    "vol_at_s_takeryes": float(a.filter(((pl.col("yes_price") - s).abs() < 1e-9) & (pl.col("taker_side") == "yes"))["count"].sum()),
                    "pre_through": pre.filter(pl.col("yes_price") > s + 1e-9).height > 0,
                    "max_yes_after": float(a["yes_price"].max()) if a.height else None,
                    "print_fill_ts": None})
        if thr.height:
            f = thr.row(0, named=True)
            ts = f["created_time"]
            sweep = a.filter((pl.col("created_time") == ts) & (pl.col("taker_side") == f["taker_side"]))
            post = a.filter(pl.col("created_time") > ts)
            post1h = post.filter(pl.col("created_time") <= ts + timedelta(hours=1))
            prev = g.filter(pl.col("created_time") < ts).tail(1)
            out.update({
                "print_fill_ts": ts, "fill_taker_side": f["taker_side"], "fill_print_price": f["yes_price"],
                "sweep_contracts": float(sweep["count"].sum()),
                "sweep_usd": float(sweep.select((pl.col("count") * pl.when(pl.col("taker_side") == "yes")
                                                 .then(pl.col("yes_price")).otherwise(1 - pl.col("yes_price"))).sum()).item()),
                "sweep_max_price": float(sweep["yes_price"].max()), "sweep_prints": sweep.height,
                "prev_print_price": float(prev["yes_price"][0]) if prev.height else None,
                "vol_post_1h": float(post1h["count"].sum()), "vol_post_close": float(post["count"].sum()),
                "vol_through_post": float(post.filter(pl.col("yes_price") > s + 1e-9)["count"].sum()),
                "vwap_post_1h": (float((post1h["count"] * post1h["yes_price"]).sum() / post1h["count"].sum())
                                 if post1h.height else None),
                "last_price_before_close": float(a["yes_price"][-1]),
                "hours_to_close_at_fill": (close - ts).total_seconds() / 3600,
            })
        rows.append(out)
    return pl.DataFrame(rows, infer_schema_length=None)


def pnl_c(s: float, y: int) -> float:
    """1-lot NO bought at 1 - s held to settlement, cents; these series carry no maker fee."""
    return 100 * ((1 - y) - (1 - s))


def step_prints() -> None:
    for split in ("holdout", "dev"):
        p = OUT / f"orders_{split}.parquet"
        tr = load_trades(split)
        if tr.is_empty():
            print(f"{split}: no trades fetched")
            continue
        o = pl.read_parquet(p)
        o = o.filter(pl.col("ticker").is_in(tr["leg"].unique().to_list()))
        pf = print_fills(o, tr)
        df = o.join(pf.drop("t_used", "s_used"), on="ticker", how="left")
        df = df.with_columns(pl.when(pl.col("print_fill_ts").is_not_null())
                             .then(100 * ((1 - pl.col("y")) - (1 - pl.col("s")))).alias("print_pnl_c"))
        _save(df, OUT / f"prints_{split}.parquet")
        print(f"{split}: {df.height} orders with prints fetched, {df['print_fill_ts'].is_not_null().sum()} print fills, "
              f"{df['candle_fill_ts'].is_not_null().sum()} candle fills")


# ---- 1b. exact book --------------------------------------------------------------------------------
def record_runs() -> list[dict[str, Any]]:
    """The ``albert record`` runs (stratified random recorder): window and subscribed tickers."""
    runs = []
    for f in sorted(Path("data/runs").glob("*-record.json")):
        m = json.loads(f.read_text())
        p = m.get("params") or {}
        if not p.get("started_at"):
            continue
        st = datetime.fromisoformat(str(p["started_at"]).replace(" ", "T"))
        runs.append({"run": f.name, "start": st, "end": st + timedelta(minutes=float(p["minutes"])),
                     "tickers": {x["ticker"] for x in p.get("kalshi_markets") or []}})
    return runs


def coverage(o: pl.DataFrame) -> pl.DataFrame:
    runs = record_runs()
    rows = []
    for r in o.iter_rows(named=True):
        for ru in runs:
            if r["ticker"] in ru["tickers"] and ru["start"] < r["close_time"] and ru["end"] > r["t"]:
                rows.append({"ticker": r["ticker"], "run": ru["run"], "cov_start": max(ru["start"], r["t"]),
                             "cov_end": min(ru["end"], r["close_time"]), "at_t": ru["start"] <= r["t"]})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "run": pl.String, "cov_start": pl.Datetime("us", "UTC"),
                                      "cov_end": pl.Datetime("us", "UTC"), "at_t": pl.Boolean})


def step_stream_fetch(days: list[str], local_days: list[str]) -> None:
    """Recorder (``k<n>``) hourly files only: archived days are downloaded from S3 and checked
    against the archive marker's SHA-256 (the same check as ``albert restore-stream``, but without
    pulling ~15 GB/day of vol-recorder files); local days are hard-linked so the daily archive
    timer cannot remove them mid-replay. Everything lands under the review cache."""
    import boto3

    root = Path("data/raw/stream/kalshi")
    s3 = boto3.client("s3", region_name="us-east-2")
    for day in days:
        m = json.loads((root / day / "_archived.json").read_text())
        dest = STREAM / day
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for r in m["files"]:
            if "-k" not in r["name"]:
                continue
            p = dest / r["name"]
            if p.exists() and p.stat().st_size == r["size"]:
                continue
            body = s3.get_object(Bucket=m["bucket"], Key=r["key"])["Body"]
            h = hashlib.sha256()
            part = p.with_name(p.name + ".part")
            with open(part, "wb") as fh:
                for chunk in iter(lambda: body.read(1 << 20), b""):
                    h.update(chunk)
                    fh.write(chunk)
            if base64.b64encode(h.digest()).decode() != r["sha256"] or part.stat().st_size != r["size"]:
                part.unlink()
                raise ValueError(f"{r['key']}: bytes do not match the archive marker")
            os.replace(part, p)
            n += 1
        print(f"{day}: {n} recorder files downloaded", flush=True)
    for day in local_days:
        dest = STREAM / day
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted((root / day).glob("*-k*.jsonl.zst")):
            q = dest / p.name
            if not q.exists():
                os.link(p, q)
                n += 1
        print(f"{day}: {n} local recorder files linked", flush=True)


class _Order:
    """One virtual 1-lot NO bid (= YES offer at ``s``) in the exact-book replay."""

    __slots__ = ("cand_s", "ckey", "dec", "fill_c", "fill_o", "level_vol", "q0", "q_opt", "s", "t", "through_ts", "tk")

    def __init__(self, tk: str, t: int, cand_s: float, ckey: str) -> None:
        self.tk, self.t, self.cand_s, self.ckey = tk, t, cand_s, ckey
        self.dec: dict[str, Any] | None = None
        self.s: int | None = None          # our YES offer in 1/100 cent
        self.q0 = self.q_opt = 0.0
        self.level_vol = 0.0
        self.fill_c = self.fill_o = self.through_ts = None


def l2_replay(o: pl.DataFrame, cov: pl.DataFrame, days: list[str]) -> list[dict[str, Any]]:
    """Exact-book replay (``behav_micro`` readers: ``_iter_frames``, ``_Book``, ``_px``) of the
    orders whose leg the recorder subscribed, one hourly file at a time and only those legs' frames.

    At the order time (if the recorder had the leg then), the decision book gives the real best
    bid/ask, whether both sides exist, the sell price ceil(mid + 12c) on the real mid, post-only
    rejection (price at or below the best YES bid), and the displayed NO size already resting at
    our price (we join behind it). Fill rules on the real price: ``through`` (a print strictly above
    our offer), ``fifo`` (through, or taker-YES volume at our price beyond the size ahead at
    placement; cancellations ahead are never credited) and ``fifo_cxl`` (cancellations credited:
    size ahead is capped by the displayed level). If the recorder picked the leg up only after the
    order time, the claim's candle price is used and the queue starts behind the whole level then."""
    from albert.research.behav_micro import _Book, _iter_frames, _px

    want = set(cov["ticker"].unique().to_list())
    orders: dict[str, _Order] = {}
    for r in o.filter(pl.col("ticker").is_in(list(want))).iter_rows(named=True):
        orders[r["ticker"]] = _Order(r["ticker"], int(r["t"].timestamp() * 1000), r["s"], r["ticker"])
    closes = {r["ticker"]: int(r["close_time"].timestamp() * 1000) for r in o.iter_rows(named=True)}
    t_day = {r["ticker"]: r["t"].strftime("%Y-%m-%d") for r in o.iter_rows(named=True)}
    # the recorder had the leg at the order time and that hour's files are in this replay
    live_at_t = {tk for tk in cov.filter(pl.col("at_t"))["ticker"].to_list() if t_day.get(tk) in set(days)}
    books: dict[str, _Book] = {}
    seen: dict[str, list[int]] = {}         # first / last ms a leg's book was observed
    files = []
    for day in days:
        files += sorted((STREAM / day).glob("*-k*.jsonl.zst"))
    hours: dict[str, list[Path]] = {}
    for f in files:
        hours.setdefault(f"{f.parent.name}T{f.name[:2]}", []).append(f)
    for hk in sorted(hours):
        msgs = []
        for f in hours[hk]:
            for recv_ns, fr in _iter_frames(f):
                msg = fr.get("msg") or {}
                mk = msg.get("market_ticker")
                if mk in want:
                    msgs.append((recv_ns, fr.get("type"), mk, msg))
        msgs.sort(key=lambda x: x[0])
        for recv_ns, typ, mk, msg in msgs:
            if typ not in ("orderbook_snapshot", "orderbook_delta", "trade"):
                continue
            ts = recv_ns // 1_000_000 if typ == "orderbook_snapshot" else int(msg.get("ts_ms") or recv_ns // 1_000_000)
            od = orders[mk]
            bk = books.get(mk)
            # decision on the book as of the order time (state before this first later message)
            if od.dec is None and bk is not None and ts >= od.t and ts <= closes[mk]:
                by, ba = bk.best()
                at_t = mk in live_at_t
                two = by is not None and ba is not None and by > 0 and ba < 10000
                mid_c = (by + ba) / 2 / 10000 if two else None
                s_book = sell_price(mid_c) if mid_c is not None else None
                s = s_book if (at_t and s_book is not None) else od.cand_s
                si = _px(s)
                od.dec = {"ticker": mk, "book_at_t": at_t, "dec_ts": datetime.fromtimestamp(max(ts, od.t) / 1000, UTC),
                          "bid": None if by is None else by / 10000, "ask": None if ba is None else ba / 10000,
                          "two_sided": two, "mid_book": mid_c, "s_book": s_book, "s_candle": od.cand_s, "s_used": s,
                          "post_only_reject": by is not None and by >= si,
                          "ask_size": bk.no.get(10000 - ba, 0.0) if ba is not None else None,
                          "bid_size": bk.yes.get(by, 0.0) if by is not None else None,
                          "q0": bk.no.get(10000 - si, 0.0), "best_ask_below_s": ba is not None and ba < si}
                od.s, od.q0 = si, od.dec["q0"]
                od.q_opt = od.q0
            if typ == "orderbook_snapshot":
                bk = books.setdefault(mk, _Book())
                bk.snapshot({_px(p): float(s) for p, s in msg.get("yes_dollars_fp") or []},
                            {_px(p): float(s) for p, s in msg.get("no_dollars_fp") or []})
                sv = seen.setdefault(mk, [ts, ts])
                sv[1] = ts
            elif typ == "orderbook_delta":
                if bk is None:
                    continue
                bk.delta(msg.get("side"), _px(msg["price_dollars"]), float(msg["delta_fp"]))
                seen[mk][1] = ts
            if (od.dec is None or od.dec["post_only_reject"] or ts > closes[mk]
                    or (od.fill_c is not None and od.fill_o is not None)):
                continue
            if typ == "trade":
                p = _px(msg["yes_price_dollars"])
                c = float(msg["count_fp"])
                if p > od.s:
                    od.through_ts = od.through_ts or ts
                    od.fill_c = od.fill_c or ts
                    od.fill_o = od.fill_o or ts
                elif p == od.s and msg.get("taker_side") == "yes":
                    od.level_vol += c
                    if od.level_vol > od.q0 + 1e-9:
                        od.fill_c = od.fill_c or ts
                    od.q_opt -= c
                    if od.q_opt < -1e-9:
                        od.fill_o = od.fill_o or ts
            elif typ == "orderbook_delta" and msg.get("side") == "no" and _px(msg["price_dollars"]) == 10000 - od.s:
                od.q_opt = min(od.q_opt, books[mk].no.get(10000 - od.s, 0.0))
    out = []
    for tk, od in orders.items():
        sv = seen.get(tk)
        out.append({"ticker": tk, "observed_from": None if sv is None else datetime.fromtimestamp(sv[0] / 1000, UTC),
                    "observed_to": None if sv is None else datetime.fromtimestamp(sv[1] / 1000, UTC),
                    **({k: v for k, v in od.dec.items() if k != "ticker"} if od.dec else {"book_at_t": None}),
                    "level_vol": od.level_vol,
                    "l2_through_ts": None if od.through_ts is None else datetime.fromtimestamp(od.through_ts / 1000, UTC),
                    "l2_fifo_ts": None if od.fill_c is None else datetime.fromtimestamp(od.fill_c / 1000, UTC),
                    "l2_fifo_cxl_ts": None if od.fill_o is None else datetime.fromtimestamp(od.fill_o / 1000, UTC)})
    return out


def step_l2(days: list[str]) -> None:
    o = pl.read_parquet(OUT / "orders_holdout.parquet")
    cov = coverage(o)
    _save(cov, OUT / "l2_coverage.parquet")
    have = set(days)
    cov = cov.filter(pl.col("cov_start").dt.strftime("%Y-%m-%d").is_in(list(have))
                     | pl.col("cov_end").dt.strftime("%Y-%m-%d").is_in(list(have)))
    print(f"l2: {cov['ticker'].n_unique()} legs with recorder coverage on {days}", flush=True)
    rows = l2_replay(o, cov, days)
    _save(pl.DataFrame(rows, infer_schema_length=None), OUT / "l2_orders.parquet")
    print(f"l2: {len(rows)} legs replayed", flush=True)


# ---- 3. look-ahead ---------------------------------------------------------------------------------
def scheduled_close(mk: pl.DataFrame) -> pl.DataFrame:
    """Scheduled event close inferred without the actual close: per series, the modal lag between
    ``expected_expiration_time`` (fixed at listing) and the close on events whose legs all closed
    together; scheduled close = expected expiration - that lag. An event closed early when its
    latest leg closed > 30 min before that."""
    ev = mk.group_by("event_ticker").agg(pl.col("series_ticker").first(), pl.col("close_time").max().alias("ev_close"),
                                         pl.col("close_time").min().alias("first_close"),
                                         pl.col("expected_expiration_time").max().alias("exp_exp"))
    ev = ev.with_columns(((pl.col("exp_exp") - pl.col("ev_close")).dt.total_minutes()).alias("lag_min"))
    lag = (ev.group_by("series_ticker", "lag_min").agg(pl.len().alias("n")).sort("n", descending=True)
           .group_by("series_ticker").first().select("series_ticker", pl.col("lag_min").alias("modal_lag_min"),
                                                     pl.col("n").alias("modal_n")))
    ev = ev.join(lag, on="series_ticker", how="left").with_columns(
        (pl.col("exp_exp") - pl.duration(minutes=pl.col("modal_lag_min"))).alias("sched_close"))
    return ev.with_columns(((pl.col("sched_close") - pl.col("ev_close")).dt.total_minutes() / 60).alias("early_h"))


def corrected_time_orders(o: pl.DataFrame, tr: pl.DataFrame, sc: pl.DataFrame) -> pl.DataFrame:
    """The rule placed at scheduled close - 72 h instead of actual close - 72 h, on events that
    closed > 30 min early: no order if the leg had closed by then; otherwise the candle quote at
    the scheduled decision time (same staleness rule as the claim), its sell price and the print
    fill after it. Orders on other events are kept as they are."""
    from albert.research.culture_flow import load_candles, quotes_at

    o = o.join(sc.select("event_ticker", "sched_close", "early_h"), on="event_ticker", how="left")
    early = o.filter(pl.col("early_h") > 0.5).with_columns((pl.col("sched_close") - timedelta(hours=H)).alias("t_s"))
    keep = o.filter(~(pl.col("early_h") > 0.5).fill_null(False))
    gone = early.filter(pl.col("t_s") >= pl.col("close_time"))
    live = early.filter(pl.col("t_s") < pl.col("close_time"))
    rows = [keep.join(print_fills(keep, tr).drop("t_used", "s_used"), on="ticker", how="left")]
    if live.height:
        c = load_candles(live["ticker"].unique().to_list())
        q = quotes_at(live.drop("t", "q_ts", "bid", "ask", "mid", "s").rename({"t_s": "t"}), c)
        q = q.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid")).filter(
            (pl.col("mid") > BAND[0]) & (pl.col("mid") <= BAND[1]))
        q = q.with_columns(pl.col("mid").map_elements(sell_price, return_dtype=pl.Float64).alias("s")).filter(
            (pl.col("s") < 0.99) & (pl.col("s") > pl.col("bid")))
        if q.height:
            rows.append(q.join(print_fills(q, tr).drop("t_used", "s_used"), on="ticker", how="left"))
    out = pl.concat(rows, how="diagonal_relaxed")
    print(f"corrected time: {early.height} orders on early-closed events -> {gone.height} never placed, "
          f"{sum(r.height for r in rows[1:])} re-placed at the scheduled time")
    return out.with_columns(pl.when(pl.col("print_fill_ts").is_not_null())
                            .then(100 * ((1 - pl.col("y")) - (1 - pl.col("s")))).alias("print_pnl_c"))


# ---- statistics ------------------------------------------------------------------------------------
def stats(df: pl.DataFrame, col: str) -> dict[str, Any]:
    from albert.research.culture_flow import summarize

    d = df.filter(pl.col(col).is_not_null())
    if d.is_empty():
        return {"n": 0}
    d = d.with_columns(pl.col("day").cast(pl.String)) if "day" in d.columns else d
    r = summarize(d, col=col)
    r["yes_hits"] = int(d["y"].sum())
    r["sum_usd_1lot"] = round(float(d[col].sum()) / 100, 2)
    return r


def event_bootstrap(df: pl.DataFrame, col: str, reps: int = 4000, seed: int = 7) -> dict[str, Any]:
    """90% CI of the mean c/fill resampling whole events (ladder legs of one event move together),
    and the share of resamples with a negative mean."""
    d = df.filter(pl.col(col).is_not_null())
    if d.height < 3:
        return {}
    g = d.group_by("event_ticker").agg(pl.col(col).sum().alias("s"), pl.len().alias("n"))
    s_, n_ = g["s"].to_numpy(), g["n"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s_), size=(reps, len(s_)))
    m = s_[idx].sum(1) / n_[idx].sum(1)
    return {"ci90": [round(float(np.quantile(m, 0.05)), 2), round(float(np.quantile(m, 0.95)), 2)],
            "p_mean_le_0": round(float((m <= 0).mean()), 4)}


def sibling_cancel(df: pl.DataFrame, lag_s: float = 60.0) -> pl.DataFrame:
    """Post-hoc sensitivity (not a tested rule): once one leg of an event fills, the event's other
    resting bids are cancelled ``lag_s`` seconds later; fills of other legs after that are dropped."""
    f = df.filter(pl.col("print_fill_ts").is_not_null())
    first = f.group_by("event_ticker").agg(pl.col("print_fill_ts").min().alias("first_fill"))
    return (f.join(first, on="event_ticker")
            .filter(pl.col("print_fill_ts") <= pl.col("first_fill") + timedelta(seconds=lag_s)))


def l2_summary() -> dict[str, Any]:
    p = OUT / "l2_orders.parquet"
    if not p.exists():
        return {}
    l2 = pl.read_parquet(p)
    pr = pl.read_parquet(OUT / "prints_holdout.parquet").select(
        "ticker", pl.col("bid").alias("c_bid"), pl.col("ask").alias("c_ask"), pl.col("s").alias("c_s"), "print_fill_ts", "y")
    cov = pl.read_parquet(OUT / "l2_coverage.parquet")
    j = l2.join(pr, on="ticker", how="left")
    at = j.filter(pl.col("book_at_t").fill_null(False))
    # a print fill inside a recorded window of that leg (the replay could have seen it)
    inwin = (pr.join(cov, on="ticker").filter(pl.col("print_fill_ts").is_between(pl.col("cov_start"), pl.col("cov_end"))
                                              & (pl.col("print_fill_ts") >= L2_FIRST_DAY))
             .select("ticker").unique())
    return {
        "legs_replayed": j.height, "legs_with_book_at_order_time": at.height,
        "at_t_two_sided": int(at["two_sided"].sum()),
        "at_t_book_mid_equals_candle_mid": int(((at["bid"] + at["ask"]) == (at["c_bid"] + at["c_ask"])).sum()),
        "at_t_sell_price_book_equals_candle": int((at["s_book"] == at["c_s"]).sum()),
        "post_only_rejects": int(j["post_only_reject"].fill_null(False).sum()),
        "queue_ahead_at_our_price_q0": j["q0"].drop_nulls().value_counts().to_dicts(),
        "best_ask_below_our_price": int(j["best_ask_below_s"].fill_null(False).sum()),
        "print_fills_inside_recorded_windows": inwin.height,
        "l2_through_fills": int(j["l2_through_ts"].is_not_null().sum()),
        "l2_fifo_fills": int(j["l2_fifo_ts"].is_not_null().sum()),
        "l2_fifo_cxl_fills": int(j["l2_fifo_cxl_ts"].is_not_null().sum()),
        "rows": j.select("ticker", "book_at_t", "bid", "ask", "c_bid", "c_ask", "s_book", "c_s", "q0", "post_only_reject",
                         "l2_through_ts", "l2_fifo_ts", "print_fill_ts", "y").to_dicts(),
    }


def concentration(df: pl.DataFrame, col: str) -> dict[str, Any]:
    d = df.filter(pl.col(col).is_not_null())
    if d.is_empty():
        return {}
    ev = d.group_by("event_ticker").agg(pl.col(col).sum().alias("c"), pl.len().alias("n")).sort("c", descending=True)
    tot = float(ev["c"].sum())
    out = {"total_c": round(tot, 1), "events": ev.height,
           "top5_events": ev.head(5).to_dicts(), "top5_share": round(float(ev.head(5)["c"].sum()) / tot, 3) if tot else None,
           "by_family": d.group_by("family").agg(pl.len().alias("fills"), pl.col("event_ticker").n_unique().alias("events"),
                                                 pl.col(col).mean().round(2).alias("mean_c"), pl.col(col).sum().round(1).alias("sum_c"),
                                                 pl.col("y").sum().alias("yes")).sort("sum_c", descending=True).to_dicts()}
    lofo = []
    for fam in d["family"].unique().to_list():
        lofo.append({"drop": fam, **stats(d.filter(pl.col("family") != fam), col)})
    out["leave_one_family_out"] = lofo
    wk = d.with_columns(pl.col("t").dt.truncate("1w").alias("wk"))
    lowo = []
    for w in sorted(wk["wk"].unique().to_list()):
        lowo.append({"drop_week": str(w.date()), "fills_dropped": wk.filter(pl.col("wk") == w).height,
                     **stats(wk.filter(pl.col("wk") != w), col)})
    out["leave_one_week_out"] = lowo
    out["by_week"] = wk.group_by("wk").agg(pl.len().alias("fills"), pl.col(col).sum().round(1).alias("sum_c"),
                                           pl.col("y").sum().alias("yes")).sort("wk").with_columns(pl.col("wk").cast(pl.String)).to_dicts()
    return out


def resting_book(o: pl.DataFrame, name: str, split: str, cash: float) -> dict[str, Any]:
    """culture_flow.resting_book (Kalshi reserves a resting bid's cost; $5/event) on this review's
    orders and fills with ``cash`` dollars available for resting bids plus open positions (the
    Sharpe and vol are still on the $100 bankroll), its ledger written under this review's directory."""
    import albert.research.culture_flow as cf

    old = cf.OUT
    cf.OUT = OUT
    try:
        return cf.resting_book(o, name, split, bankroll=cash, rest_cap=cash)
    finally:
        cf.OUT = old


def book_frame(df: pl.DataFrame, fill_col: str, price_col: str = "s") -> pl.DataFrame:
    return df.select("ticker", "event_ticker", "family", "t", (1 - pl.col(price_col)).round(2).alias("price"),
                     pl.col(fill_col).alias("fill_ts"), pl.coalesce(fill_col, "close_time").alias("end"), "y",
                     "settlement_ts")


def step_report() -> None:
    res: dict[str, Any] = {"generated_utc": datetime.now(UTC).isoformat(), "named_at": NAMED_AT.isoformat()}
    for split in ("holdout", "dev"):
        p = OUT / f"prints_{split}.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p).with_columns(pl.col("candle_pnl_c"), pl.col("print_pnl_c"))
        df = df.with_columns(pl.coalesce("print_fill_ts", "candle_fill_ts").dt.date().alias("day"))
        r: dict[str, Any] = {"orders": df.height, "events": df["event_ticker"].n_unique()}
        r["candle"] = {**stats(df, "candle_pnl_c"), **event_bootstrap(df, "candle_pnl_c")}
        r["print"] = {**stats(df, "print_pnl_c"), **event_bootstrap(df, "print_pnl_c")}
        r["fill_rate"] = {"candle": round(df["candle_fill_ts"].is_not_null().mean(), 4),
                          "print": round(df["print_fill_ts"].is_not_null().mean(), 4)}
        both = df.filter(pl.col("candle_fill_ts").is_not_null() & pl.col("print_fill_ts").is_not_null())
        r["agreement"] = {
            "both": both.height,
            "candle_only": df.filter(pl.col("candle_fill_ts").is_not_null() & pl.col("print_fill_ts").is_null()).height,
            "print_only": df.filter(pl.col("candle_fill_ts").is_null() & pl.col("print_fill_ts").is_not_null()).height,
            "candle_fill_hours_after_print_median": (float(both.select(((pl.col("candle_fill_ts") - pl.col("print_fill_ts"))
                                                                        .dt.total_minutes() / 60).median()).item())
                                                     if both.height else None),
            "candle_fill_with_pre_order_through_print": df.filter(pl.col("candle_fill_ts").is_not_null() & pl.col("pre_through")).height,
        }
        r["print_2lot"] = stats(df.filter(pl.col("vol_through") >= 2), "print_pnl_c")
        r["print_fills_taker_side"] = df.filter(pl.col("print_fill_ts").is_not_null()).group_by("fill_taker_side").agg(pl.len()).to_dicts()
        r["one_sided_at_t"] = {"orders_bid_empty": int(df["bid_empty"].sum()),
                               "print_two_sided_only": stats(df.filter(~pl.col("bid_empty")), "print_pnl_c"),
                               "candle_two_sided_only": stats(df.filter(~pl.col("bid_empty")), "candle_pnl_c")}
        r["quote_age"] = {"median_h": float(df["quote_age_h"].median()),
                          "share_over_2h": round(float((df["quote_age_h"] > 2).mean()), 3),
                          "share_over_24h": round(float((df["quote_age_h"] > 24).mean()), 3),
                          "fills_print_quote_over_24h": df.filter(pl.col("print_fill_ts").is_not_null()
                                                                  & (pl.col("quote_age_h") > 24)).height,
                          "print_fresh_quote_le2h": stats(df.filter(pl.col("quote_age_h") <= 2), "print_pnl_c")}
        f = df.filter(pl.col("print_fill_ts").is_not_null())
        if f.height:
            r["adverse"] = {
                "yes_fills": f.filter(pl.col("y") == 1).select("ticker", "s", "sweep_contracts", "sweep_usd", "vol_post_1h",
                                                               "vol_through", "max_yes_after", "hours_to_close_at_fill").to_dicts(),
                "by_sweep": f.with_columns(pl.when(pl.col("sweep_contracts") < 10).then(pl.lit("a <10"))
                                           .when(pl.col("sweep_contracts") < 100).then(pl.lit("b 10-99"))
                                           .otherwise(pl.lit("c >=100")).alias("bucket"))
                .group_by("bucket").agg(pl.len().alias("fills"), pl.col("print_pnl_c").mean().round(2).alias("mean_c"),
                                        pl.col("y").sum().alias("yes")).sort("bucket").to_dicts(),
                "by_vol_through": f.with_columns(pl.when(pl.col("vol_through") < 10).then(pl.lit("a <10"))
                                                 .when(pl.col("vol_through") < 200).then(pl.lit("b 10-199"))
                                                 .otherwise(pl.lit("c >=200")).alias("bucket"))
                .group_by("bucket").agg(pl.len().alias("fills"), pl.col("print_pnl_c").mean().round(2).alias("mean_c"),
                                        pl.col("y").sum().alias("yes")).sort("bucket").to_dicts(),
                "by_hours_to_close": f.with_columns(pl.when(pl.col("hours_to_close_at_fill") < 6).then(pl.lit("a <6h"))
                                                    .when(pl.col("hours_to_close_at_fill") < 24).then(pl.lit("b 6-24h"))
                                                    .otherwise(pl.lit("c >=24h")).alias("bucket"))
                .group_by("bucket").agg(pl.len().alias("fills"), pl.col("print_pnl_c").mean().round(2).alias("mean_c"),
                                        pl.col("y").sum().alias("yes")).sort("bucket").to_dicts(),
                "corr_yes_vs_log_sweep": float(np.corrcoef(f["y"].to_numpy(), np.log1p(f["sweep_contracts"].to_numpy()))[0, 1]),
                "corr_yes_vs_log_vol_through": float(np.corrcoef(f["y"].to_numpy(), np.log1p(f["vol_through"].to_numpy()))[0, 1]),
            }
        r["concentration_print"] = concentration(df.with_columns(pl.coalesce("print_fill_ts", "t").alias("t")), "print_pnl_c")
        r["concentration_candle"] = concentration(df, "candle_pnl_c")
        r["fresh_events_close_ge_2026_11_17"] = {"orders": df.filter(pl.col("close_time") >= FRESH).height,
                                                 "candle": stats(df.filter(pl.col("close_time") >= FRESH), "candle_pnl_c"),
                                                 "print": stats(df.filter(pl.col("close_time") >= FRESH), "print_pnl_c")}
        r["closing_after_named"] = df.filter(pl.col("close_time") > NAMED_AT).height
        if split == "holdout":
            books = {}
            for nm, col in (("candle", "candle_fill_ts"), ("print", "print_fill_ts")):
                bf = book_frame(df, col)
                for cap in (100.0, 57.0, 40.0, 20.0):
                    books[f"{nm}_cash{int(cap)}"] = resting_book(bf, f"review_{nm}_cash{int(cap)}", split, cap)
            bf = book_frame(df.filter(~pl.col("bid_empty")), "print_fill_ts")
            books["print_two_sided_cash57"] = resting_book(bf, "review_print_2s_cash57", split, 57.0)
            r["resting_books"] = books
        res[split] = r
    res["l2"] = l2_summary()
    pooled = [pl.read_parquet(OUT / f"prints_{sp}.parquet").with_columns(pl.col("print_fill_ts").dt.date().alias("day"))
              for sp in ("dev", "holdout") if (OUT / f"prints_{sp}.parquet").exists()]
    if len(pooled) == 2:
        pdf = pl.concat(pooled, how="diagonal_relaxed")
        f = pdf.filter(pl.col("print_fill_ts").is_not_null())
        res["pooled_dev_holdout_print"] = {**stats(pdf, "print_pnl_c"), **event_bootstrap(pdf, "print_pnl_c"),
                                           "yes_rate": round(float(f["y"].mean()), 4),
                                           "breakeven_yes_rate": round(float(f["s"].mean()), 4),
                                           "worst_events": f.group_by("event_ticker").agg(
                                               pl.len().alias("fills"), pl.col("y").sum().alias("yes"),
                                               pl.col("print_pnl_c").sum().round(1).alias("c")).sort("c").head(5).to_dicts(),
                                           "yes_fill_rank_by_vol_through": f.with_columns(pl.col("vol_through").rank(descending=True).alias("r"))
                                           .filter(pl.col("y") == 1)["r"].to_list()}
        sib = sibling_cancel(pdf)
        res["pooled_sibling_cancel_60s_posthoc"] = {**stats(sib, "print_pnl_c"), **event_bootstrap(sib, "print_pnl_c")}
    # look-ahead: actual vs scheduled close
    mk = pl.read_parquet(SRC / "markets.parquet")
    sc = scheduled_close(mk)
    _save(sc, OUT / "scheduled_close.parquet")
    ho = OUT / "prints_holdout.parquet"
    if ho.exists():
        df = pl.read_parquet(ho).join(sc.select("event_ticker", "sched_close", "early_h"), on="event_ticker", how="left")
        early = df.filter(pl.col("early_h") > 0.5)
        res["lookahead"] = {
            "events_closed_early_gt_30min": sc.filter(pl.col("early_h") > 0.5).select("event_ticker", "ev_close", "sched_close",
                                                                                   pl.col("early_h").round(2)).sort("early_h", descending=True).head(40).to_dicts(),
            "holdout_events_closed_early": sc.filter((pl.col("early_h") > 0.5) & (pl.col("ev_close") >= datetime(2026, 10, 1, tzinfo=UTC))).height,
            "holdout_orders_on_early_events": early.height,
            "print_fills_on_early_events": early.filter(pl.col("print_fill_ts").is_not_null()).select(
                "ticker", "t", "sched_close", "early_h", "print_fill_ts", "y", "print_pnl_c").to_dicts(),
            "print_excluding_early_events": stats(df.filter(~(pl.col("early_h") > 0.5)).with_columns(
                pl.col("print_fill_ts").dt.date().alias("day")), "print_pnl_c"),
            "legs_listed_after_t": int((df["open_time"] > df["t"]).sum()),
            "quotes_older_than_t": int((df["q_ts"] > df["t"]).sum()),
        }
        tr = load_trades("holdout")
        ords = pl.read_parquet(OUT / "orders_holdout.parquet").filter(pl.col("ticker").is_in(tr["leg"].unique().to_list()))
        ct = corrected_time_orders(ords, tr, sc)
        _save(ct, OUT / "prints_holdout_sched_time.parquet")
        ct = ct.with_columns(pl.col("print_fill_ts").dt.date().alias("day"))
        res["lookahead"]["print_at_scheduled_time"] = {"orders": ct.height, **stats(ct, "print_pnl_c")}
        res["lookahead"]["print_at_scheduled_time_two_sided"] = stats(ct.filter(pl.col("bid") > 0), "print_pnl_c")
    _dump(res, OUT / "results.json")
    print(json.dumps(res, indent=1, default=str)[:6000])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="culture_flow_review")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("orders")
    t = sub.add_parser("trades")
    t.add_argument("--split", default="holdout")
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--min-interval", type=float, default=0.5)
    sub.add_parser("prints")
    s = sub.add_parser("stream-fetch")
    s.add_argument("--days", nargs="*", default=[])
    s.add_argument("--local-days", nargs="*", default=[])
    lp = sub.add_parser("l2")
    lp.add_argument("--days", nargs="+", required=True)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "orders":
        step_orders()
    elif a.cmd == "trades":
        step_trades(a.split, a.limit, a.min_interval)
    elif a.cmd == "prints":
        step_prints()
    elif a.cmd == "stream-fetch":
        step_stream_fetch(a.days, a.local_days)
    elif a.cmd == "l2":
        step_l2(a.days)
    elif a.cmd == "report":
        step_report()


if __name__ == "__main__":
    main()
