"""Wallets workstream of the favourite-longshot study (``paper_flb/``): who buys longshots on Polymarket?

Polymarket international is **data only** for this project (AGENTS.md): this module reads public
endpoints and never places orders.

Data (all public, cached append-only through :class:`albert.http.CachedClient`)
--------------------------------------------------------------------------------
* **Frame.** Every closed, in-scope (non-financial) Polymarket event whose end date falls in
  ``[START, END)`` with event volume >= $1,000 (Gamma ``/events/keyset``), flattened to binary
  markets with a clean winner and market volume >= $1,000.
* **Sample.** A seeded stratified random sample of frame markets by coarse group x volume tier x
  resolution month (``sample_markets``; the $1M-6M tier by half-year, >$6M not sampled); each market
  carries the design weight ``N_h / n_h``.
* **Trades.** The *complete* trade history of each sampled market from the Data API v2
  (``/v2/trades?condition=...``), every wallet, maker and taker rows (``taker_only=false``), plus
  the taker-only feed to label each fill's liquidity role.
* **Users.** ``/v2/user-stats`` for the wallets whose address is 0 mod 8 (``user_subset``), in address
  order, so any prefix is a pseudo-random subset: ``join_date`` (account creation) is used as-of; the
  lifetime counters it also returns are ex post and are used only descriptively.
* **Cross-venue.** Kalshi game markets already matched to Polymarket (``match_kalshi_pm``):
  Kalshi hourly candles (``kalshi_candles``) and Polymarket ``/v2/prices-history`` (1h buckets).

robots.txt (checked 2026-11-11 with ``curl``): ``data-api.polymarket.com``, ``gamma-api.polymarket.com``
and ``clob.polymarket.com`` serve no robots.txt (HTTP 410/404: no restriction); ``polymarket.com``
is ``User-agent: * / Disallow:`` (allow all) and links an ``llms.txt`` for AI agents;
``docs.polymarket.com`` and ``docs.polymarket.us`` allow all (``ai-input=yes``).
``gateway.polymarket.us`` (Polymarket US data) *disallows* ``anthropic-ai``, ``ClaudeBot`` and
``Claude-Web``, so no Polymarket US market data is fetched or used here; only its published fee
schedule (``docs.polymarket.us/fees``; tiers 0.095 to 0.0695, whereas ``albert.fees.pmus_fee`` knows only
0.0695) enters the venue comparison.

Steps: ``python -m albert.research.flb_wallets run <step> [n ...]`` with ``frame``, ``sample``, ``trades``,
``users`` (optional: max wallets, workers, batch), ``build``, ``xvenue``, ``analyze`` (or ``all``).
Outputs: ``data/research/flb_wallets/`` (summary.json, verdict.md, CSVs read by paper_flb/scripts/fig_wallets.py;
intermediate tables in ``tables/``). Requests are throttled to <= 2.8/s per process
(``FLB_WALLETS_REQ_INTERVAL`` splits the budget between concurrent steps).
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.categories import PM_FINANCIAL_TAGS
from albert.config import Settings, settings
from albert.http import CachedClient, HttpError
from albert.research.flb_common import pm_coarse_category
from albert.venues.polymarket import _json_list, _parse_loose, winner_index

OUT = Path("data/research/flb_wallets")
TAB = OUT / "tables"
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2026, 11, 1, tzinfo=UTC)
MIN_VOLUME = 1_000.0
TIER_EDGES = (1e3, 1e4, 1e5, 1e6, 6e6)
# >$6M markets (1,340 of 636,520 in the frame, 42% of its volume) cost >150 pages each and are not sampled
TIER_LABELS = ("$1k-10k", "$10k-100k", "$100k-1M", "$1M-6M", ">$6M")
# Gamma tag ids of slugs in PM_FINANCIAL_TAGS (looked up 2026-11-11 via /tags/slug/<slug>). They only
# shrink the frame pages; the binding scope filter is pm_coarse_category() on the event's tag slugs.
FIN_TAG_IDS = (21, 1312, 235, 120, 100328, 604, 159, 1013, 107)
# shared across hosts and threads of one process: <= 2.8 requests/s for this workstream. Two concurrent
# steps must split the budget via FLB_WALLETS_REQ_INTERVAL (e.g. 0.45 s and 1.6 s).
REQ_INTERVAL = float(os.environ.get("FLB_WALLETS_REQ_INTERVAL", "0.36"))
MAX_TRADE_PAGES = 150        # 1,000 rows per page; markets beyond the cap are flagged truncated
SEED = 20261111


# ---- rate-limited cached client ------------------------------------------------------------------
class _SharedThrottle:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


_THROTTLE = _SharedThrottle(REQ_INTERVAL)


class ThrottledClient(CachedClient):
    """CachedClient whose request spacing is shared by every instance and thread of this module."""

    def _throttle(self) -> None:
        _THROTTLE.wait()


@dataclass
class PM:
    gamma: CachedClient
    data: CachedClient


def clients(s: Settings | None = None) -> PM:
    s = s or settings()
    return PM(ThrottledClient("pm_gamma", s.pm_gamma_base, s.raw_dir, timeout=90.0, max_retries=6),
              ThrottledClient("pm_data", s.pm_data_base, s.raw_dir, timeout=90.0, max_retries=6))


# ---- parsers (tested against trimmed recorded payloads) ----------------------------------------
FRAME_SCHEMA = {
    "condition_id": pl.String, "event_id": pl.String, "event_slug": pl.String, "event_title": pl.String,
    "question": pl.String, "tags": pl.List(pl.String), "group": pl.String,
    "created_at": pl.Datetime("us", "UTC"), "end_date": pl.Datetime("us", "UTC"),
    "closed_time": pl.Datetime("us", "UTC"), "game_start_time": pl.Datetime("us", "UTC"),
    "token0": pl.String, "token1": pl.String, "outcome0": pl.String, "outcome1": pl.String,
    "winner_index": pl.Int64, "volume": pl.Float64, "neg_risk": pl.Boolean,
    "fees_enabled": pl.Boolean, "fee_rate": pl.Float64, "fee_exponent": pl.Float64,
    "sports_market_type": pl.String,
}


def frame_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Binary markets of Gamma events, with event tags, coarse group, resolution and fee schedule."""
    rows = []
    for e in events:
        tags = [t.get("slug") for t in e.get("tags") or [] if t.get("slug")]
        group = pm_coarse_category(tags)
        for m in e.get("markets") or []:
            tokens = [str(t) for t in _json_list(m.get("clobTokenIds"))]
            outcomes = [str(o) for o in _json_list(m.get("outcomes"))]
            if not m.get("conditionId") or len(tokens) != 2 or len(outcomes) != 2:
                continue
            closed = bool(m.get("closed"))
            sched = m.get("feeSchedule") or {}
            rows.append({
                "condition_id": m["conditionId"], "event_id": str(e.get("id") or ""),
                "event_slug": e.get("slug"), "event_title": e.get("title"), "question": m.get("question"),
                "tags": tags, "group": group,
                "created_at": _parse_loose(m.get("createdAt")), "end_date": _parse_loose(m.get("endDate")),
                "closed_time": _parse_loose(m.get("closedTime")),
                "game_start_time": _parse_loose(m.get("gameStartTime")),
                "token0": tokens[0], "token1": tokens[1], "outcome0": outcomes[0], "outcome1": outcomes[1],
                "winner_index": winner_index(_json_list(m.get("outcomePrices")), closed),
                "volume": float(m.get("volumeNum") or m.get("volume") or 0.0),
                "neg_risk": bool(m.get("negRisk")),
                "fees_enabled": bool(m.get("feesEnabled")),
                "fee_rate": float(sched["rate"]) if m.get("feesEnabled") and sched.get("rate") is not None else 0.0,
                "fee_exponent": float(sched.get("exponent") or 1.0),
                "sports_market_type": m.get("sportsMarketType"),
            })
    return rows


TRADE_SCHEMA = {"condition_id": pl.String, "tx": pl.String, "wallet": pl.String, "buy": pl.Boolean,
                "outcome_index": pl.Int8, "price": pl.Float64, "size": pl.Float64, "ts": pl.Int64}


def trade_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact rows of a Data API v2 ``/trades`` page (wallet lower-cased; ts in epoch seconds)."""
    out = []
    for t in payload.get("data") or []:
        oi = t.get("outcome_index")
        if oi not in (0, 1) or t.get("price") is None or t.get("size") is None:
            continue
        out.append({"condition_id": t.get("condition_id"), "tx": t.get("transaction_hash"),
                    "wallet": (t.get("proxy_wallet") or "").lower(), "buy": t.get("side") == "BUY",
                    "outcome_index": oi, "price": float(t["price"]), "size": float(t["size"]),
                    "ts": int(t["timestamp"])})
    return out


def user_stats_row(payload: dict[str, Any], wallet: str) -> dict[str, Any]:
    d = payload.get("data") or {}
    pnl = d.get("all_time_pnl") or {}
    return {"wallet": wallet.lower(), "known": bool(payload.get("data")),
            "join_date": int(d["join_date"]) if d.get("join_date") else None,
            "markets_traded_now": d.get("trades"), "biggest_win_now": d.get("biggest_win"),
            "pnl_now": pnl.get("economic_pnl") if isinstance(pnl, dict) else None,
            "trade_count_now": pnl.get("trade_count") if isinstance(pnl, dict) else None}


def price_points(payload: dict[str, Any]) -> list[tuple[int, float]]:
    return [(int(p["timestamp"]), float(p["price"])) for p in payload.get("data") or []
            if p.get("timestamp") is not None and p.get("price") is not None]


# ---- step: frame -----------------------------------------------------------------------------------
def _months(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    lo = start
    while lo < end:
        hi = (lo.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield lo, min(hi, end)
        lo = hi


def _events(pm: PM, lo: datetime, hi: datetime) -> Iterator[list[dict[str, Any]]]:
    cursor = None
    while True:
        params: dict[str, Any] = {"closed": "true", "limit": 100, "after_cursor": cursor,
                                  "end_date_min": lo.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  "end_date_max": hi.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  "volume_min": MIN_VOLUME}
        # repeated exclude_tag_id parameters: encode as one key per id through httpx's list support
        page = pm.gamma.get("/events/keyset", {**params, "exclude_tag_id": list(FIN_TAG_IDS)})
        events = page.payload.get("events") or []
        yield events
        cursor = page.payload.get("next_cursor")
        if not cursor or not events:
            return


def run_frame(pm: PM | None = None) -> pl.DataFrame:
    pm = pm or clients()
    parts = TAB / "frame_parts"
    parts.mkdir(parents=True, exist_ok=True)
    for lo, hi in _months(START, END):
        if (parts / f"{lo:%Y-%m}.parquet").exists() or (parts / f"{lo:%Y-%m}.done").exists():
            continue                     # month already complete (re-runs resume)
        for stale in parts.glob(f"{lo:%Y-%m}-*.parquet"):
            stale.unlink()
        rows: list[dict[str, Any]] = []
        n_events = n_markets = chunk = 0
        for events in _events(pm, lo, hi):
            n_events += len(events)
            rows.extend(frame_rows(events))
            if len(rows) >= 20_000:      # flush often: 2026 months have >200k markets
                pl.DataFrame(rows, schema=FRAME_SCHEMA).write_parquet(parts / f"{lo:%Y-%m}-{chunk:03d}.parquet")
                n_markets += len(rows)
                chunk += 1
                rows = []
        pl.DataFrame(rows, schema=FRAME_SCHEMA).write_parquet(parts / f"{lo:%Y-%m}-{chunk:03d}.parquet")
        n_markets += len(rows)
        (parts / f"{lo:%Y-%m}.done").write_text(json.dumps({"events": n_events, "markets": n_markets}))
        print(f"frame {lo:%Y-%m}: {n_events} events, {n_markets} markets", flush=True)
    consolidate_frame()


def consolidate_frame() -> pl.DataFrame:
    """In-scope, resolved, >= $1k markets of the frame parts (lazy: the parts hold >1M markets)."""
    resolved = pl.coalesce("closed_time", "end_date")
    frame = pl.scan_parquet(TAB / "frame_parts" / "*.parquet").with_columns(resolved.alias("resolved_at")).filter(
        pl.col("group").is_not_null()                       # financial tags -> None (out of scope)
        & ~pl.col("tags").list.eval(pl.element().is_in(list(PM_FINANCIAL_TAGS))).list.any()
        & pl.col("winner_index").is_not_null() & (pl.col("volume") >= MIN_VOLUME)
        & pl.col("resolved_at").is_between(START, END, closed="left"))
    frame = frame.with_columns(
        pl.col("resolved_at").dt.strftime("%Y-%m").alias("month"),
        tier_expr()).collect().unique("condition_id", keep="first").sort("condition_id")
    _require(frame, "frame")
    frame.write_parquet(TAB / "frame.parquet")
    return frame


def _require(frame: pl.DataFrame, name: str) -> None:
    if frame.height == 0:
        raise RuntimeError(f"{name}: zero rows (a smoke test that returns nothing is a failure)")


# ---- step: sample ----------------------------------------------------------------------------------
# Markets drawn per stratum by tier. Strata are group x tier x resolution month, except the $1M-6M tier,
# whose strata are group x half-year: a complete $1M-6M history costs ~100 requests and ~10^5 rows, so
# that tier is sampled thinly and every market is re-weighted by N_h / n_h.
PER_STRATUM = {"$1k-10k": 6, "$10k-100k": 6, "$100k-1M": 2, "$1M-6M": 1, ">$6M": 0}
STRATUM = ["group", "tier", "period"]


def period_expr() -> pl.Expr:
    half = pl.when(pl.col("month").str.slice(5, 2).cast(pl.Int32) <= 6).then(pl.lit("H1")).otherwise(pl.lit("H2"))
    return (pl.when(pl.col("tier") == TIER_LABELS[-2]).then(pl.col("month").str.slice(0, 4) + half)
            .otherwise(pl.col("month")).alias("period"))


def sample_markets(frame: pl.DataFrame, per_stratum: dict[str, int] | None = None, seed: int = SEED) -> pl.DataFrame:
    """Seeded stratified random sample; ``weight`` = stratum size / markets drawn."""
    per = per_stratum or PER_STRATUM
    rng = np.random.default_rng(seed)
    f = frame.sort("condition_id").with_columns(pl.Series("u", rng.random(frame.height)), period_expr())
    f = f.with_columns(pl.col("u").rank("ordinal").over(STRATUM).alias("rk"),
                       pl.len().over(STRATUM).alias("N_h"))
    f = f.with_columns(pl.col("tier").replace_strict(per, return_dtype=pl.Int64).alias("k"))
    s = f.filter(pl.col("rk") <= pl.col("k"))
    s = s.with_columns(pl.min_horizontal("k", "N_h").alias("n_h")).with_columns(
        (pl.col("N_h") / pl.col("n_h")).alias("weight"))
    return s.drop("u", "rk", "k").sort("month", "group", "tier", "condition_id")


def tier_expr() -> pl.Expr:
    return (pl.col("volume").cut(list(TIER_EDGES[1:]), labels=list(TIER_LABELS), left_closed=True)
            .cast(pl.String).alias("tier"))


def run_sample() -> pl.DataFrame:
    frame = pl.read_parquet(TAB / "frame.parquet").with_columns(tier_expr())
    frame.write_parquet(TAB / "frame.parquet")
    s = sample_markets(frame)
    _require(s, "sample")
    s.write_parquet(TAB / "sample.parquet")
    print(s.group_by("tier").agg(pl.len(), pl.col("volume").sum()).sort("tier"))
    return s


# ---- step: trades ----------------------------------------------------------------------------------
def market_trades(pm: PM, condition_id: str, *, taker_only: bool,
                  max_pages: int = MAX_TRADE_PAGES) -> tuple[list[dict[str, Any]], int, bool]:
    """All trade rows of one market (newest first), pages fetched, and whether the cap truncated it."""
    rows: list[dict[str, Any]] = []
    cursor = None
    for pages in range(1, max_pages + 1):
        page = pm.data.get("/trades", {"condition": condition_id, "limit": 1000,
                                       "taker_only": str(taker_only).lower(), "cursor": cursor})
        rows.extend(trade_rows(page.payload))
        cursor = (page.payload.get("pagination") or {}).get("next_cursor")
        if not cursor or not page.payload.get("data"):
            return rows, pages, False
    return rows, max_pages, True


def _fetch_one(pm: PM, cid: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    rows, pages, trunc = market_trades(pm, cid, taker_only=False)
    trows, tpages, ttrunc = market_trades(pm, cid, taker_only=True)
    f = pl.DataFrame(rows, schema=TRADE_SCHEMA)
    takers = pl.DataFrame(trows, schema=TRADE_SCHEMA).select("tx", "wallet").unique()
    # no de-duplication: one wallet can fill two identical maker orders in the same transaction
    f = f.join(takers.with_columns(pl.lit(True).alias("taker")), on=["tx", "wallet"], how="left",
               maintain_order="left").with_columns(pl.col("taker").fill_null(False))
    meta = {"condition_id": cid, "rows": f.height, "taker_rows": len(trows), "pages": pages + tpages,
            "truncated": trunc or ttrunc, "fetched_at": datetime.now(UTC).isoformat()}
    return f, meta


def run_trades(limit: int | None = None, workers: int = 3, batch: int = 40) -> None:
    pm = clients()
    sample = pl.read_parquet(TAB / "sample.parquet")
    parts = TAB / "trades"
    parts.mkdir(parents=True, exist_ok=True)
    log_path = TAB / "trades_log.jsonl"
    done = set()
    if log_path.exists():
        done = {json.loads(x)["condition_id"] for x in log_path.read_text().splitlines() if x.strip()}
    # cheapest first, so an interrupted run still covers every stratum below the top tier
    todo = [c for c in sample.sort("volume")["condition_id"].to_list() if c not in done]
    if limit:
        todo = todo[:limit]
    print(f"trades: {len(done)} done, {len(todo)} to fetch", flush=True)
    buf: list[pl.DataFrame] = []
    metas: list[dict[str, Any]] = []
    n_part = len(list(parts.glob("part-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal n_part, buf, metas
        if not metas:
            return
        if buf:
            pl.concat(buf).write_parquet(parts / f"part-{n_part:05d}.parquet")
            n_part += 1
        with log_path.open("a") as fh:
            for m in metas:
                fh.write(json.dumps(m) + "\n")
        buf, metas = [], []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_one, pm, cid): cid for cid in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            cid = futures[fut]
            try:
                f, meta = fut.result()
            except HttpError as err:
                print(f"  {cid}: {err}", flush=True)
                continue
            if f.height:
                buf.append(f)
            metas.append(meta)
            if len(metas) >= batch:
                flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} markets, {time.time() - t0:.0f}s", flush=True)
        flush()


def load_trades() -> pl.LazyFrame:
    return pl.scan_parquet(TAB / "trades" / "part-*.parquet")


# ---- step: users -----------------------------------------------------------------------------------
USER_MOD = 8     # account stats for wallets whose address is 0 mod USER_MOD: a fixed pseudo-random 1/8 subset


def user_subset(wallets: list[str]) -> list[str]:
    """The wallets in the account-stats subset. Selection by address, so it is the same random 1/8 of
    wallets whatever the order in which markets were fetched (and re-runs can top it up)."""
    return [w for w in wallets if len(w) == 42 and int(w[-8:], 16) % USER_MOD == 0]


def run_users(max_wallets: int = 40_000, workers: int = 3, batch: int = 0) -> None:
    """``/v2/user-stats`` (account creation date) for the address-selected subset of sampled wallets."""
    pm = clients()
    wallets = load_trades().select("wallet").unique().collect().sort("wallet")["wallet"].to_list()
    order = user_subset(wallets)[:max_wallets]
    path = TAB / "users.parquet"
    have = set(pl.read_parquet(path)["wallet"].to_list()) if path.exists() else set()
    todo = [w for w in order if w not in have]
    if batch:
        todo = todo[:batch]          # one bounded pass (lets a loop alternate with other fetches)
    print(f"users: {len(wallets)} wallets in trades, {len(have)} have stats, fetching {len(todo)}", flush=True)
    rows: list[dict[str, Any]] = []

    def one(w: str) -> dict[str, Any]:
        return user_stats_row(pm.data.get("/user-stats", {"user": w}).payload, w)

    def save() -> None:
        nonlocal rows
        if not rows:
            return
        new = pl.DataFrame(rows, infer_schema_length=None)
        old = pl.read_parquet(path) if path.exists() else None
        out = pl.concat([old, new], how="diagonal_relaxed") if old is not None else new
        out.unique("wallet", keep="last").write_parquet(path)
        rows = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, w) for w in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except HttpError as err:
                print(f"  user: {err}", flush=True)
            if i % 2000 == 0:
                save()
                print(f"  users {i}/{len(todo)}", flush=True)
    save()


# ---- accounting, as-of features, classes ---------------------------------------------------------
def normalize(f: pl.DataFrame | pl.LazyFrame) -> Any:
    """YES-equivalent exposure ``expo`` (shares of outcome 0) and cash ``cash`` of each fill.

    Buying outcome 1 at p is +s(1 - Y) = s - sY: exposure -s and cash +s(1 - p). In these units a
    split or merge is PnL-neutral, so a wallet's PnL in a market is ``sum(cash) + Y0 * sum(expo)``
    from its trades alone, and the sum over all wallets of a fully observed market is zero.
    """
    s, p = pl.col("size"), pl.col("price")
    yes, buy = pl.col("outcome_index") == 0, pl.col("buy")
    expo = pl.when(buy == yes).then(s).otherwise(-s)
    cash = (pl.when(yes & buy).then(-s * p).when(yes & ~buy).then(s * p)
            .when(~yes & buy).then(s * (1 - p)).otherwise(-s * (1 - p)))
    return f.with_columns(expo.alias("expo"), cash.alias("cash"))


def asof_features(f: pl.DataFrame) -> pl.DataFrame:
    """Per fill, the wallet's history in the sample strictly before it (no look-ahead).

    ``n_prev`` fills, ``maker_prev`` maker fills, ``mkts_prev`` markets entered, ``two_prev`` of those
    already two-sided (the wallet has traded both YES-equivalent directions), ``typ_size`` the median
    notional of its last <= 200 fills including this one, ``first_seen`` its first fill in the sample.
    """
    f = f.with_columns((pl.col("price") * pl.col("size")).alias("notional"),
                       pl.when(pl.col("buy") == (pl.col("outcome_index") == 0)).then(1).otherwise(-1)
                       .cast(pl.Int8).alias("d")).sort("wid", "ts", "mid")
    mk = (~pl.col("taker")).cast(pl.Int32)
    f = f.with_columns(
        pl.int_range(pl.len(), dtype=pl.Int32).over("wid").alias("n_prev"),
        (mk.cum_sum().over("wid") - mk).alias("maker_prev"),
        pl.col("notional").rolling_median(window_size=200, min_samples=1).over("wid").alias("typ_size"),
        pl.col("ts").min().over("wid").alias("first_seen"))
    wm = f.group_by("wid", "mid").agg(
        pl.col("ts").min().alias("t_first"),
        pl.col("ts").filter(pl.col("d") != pl.col("d").sort_by("ts").first()).min().alias("t_two"))
    ev = pl.concat([
        wm.select("wid", pl.col("t_first").alias("ts"), pl.lit(1, pl.Int32).alias("m"), pl.lit(0, pl.Int32).alias("t")),
        wm.filter(pl.col("t_two").is_not_null()).select(
            "wid", pl.col("t_two").alias("ts"), pl.lit(0, pl.Int32).alias("m"), pl.lit(1, pl.Int32).alias("t"))])
    ev = ev.group_by("wid", "ts").agg(pl.col("m").sum(), pl.col("t").sum()).sort("wid", "ts").with_columns(
        pl.col("m").cum_sum().over("wid").alias("mkts_prev"),
        pl.col("t").cum_sum().over("wid").alias("two_prev")).drop("m", "t")
    f = f.sort("ts").join_asof(ev.sort("ts"), on="ts", by="wid", strategy="backward", allow_exact_matches=False,
                         check_sortedness=False)
    return f.with_columns(pl.col("mkts_prev").fill_null(0).cast(pl.Int32),
                          pl.col("two_prev").fill_null(0).cast(pl.Int32)).drop("d")


CLASSES = ("Retail", "Active retail", "Mid", "Whale", "MM")
DEFAULT_TH: dict[str, float] = {"mm_min_prev": 20, "mm_maker": 0.7, "mm_two": 0.5, "whale": 1000.0,
                                "small": 100.0, "active_prev": 20}
# robustness grid: one cut-off moved at a time
TH_VARIANTS: dict[str, dict[str, float]] = {
    "default": DEFAULT_TH,
    "small=$50": {**DEFAULT_TH, "small": 50.0}, "small=$250": {**DEFAULT_TH, "small": 250.0},
    "whale=$500": {**DEFAULT_TH, "whale": 500.0}, "whale=$5k": {**DEFAULT_TH, "whale": 5000.0},
    "active>=10": {**DEFAULT_TH, "active_prev": 10}, "active>=50": {**DEFAULT_TH, "active_prev": 50},
    "MM loose": {**DEFAULT_TH, "mm_maker": 0.5, "mm_two": 0.3, "mm_min_prev": 10},
    "MM strict": {**DEFAULT_TH, "mm_maker": 0.9, "mm_two": 0.7, "mm_min_prev": 50},
}


def class_expr(th: dict[str, float]) -> pl.Expr:
    """Wallet class of each fill from as-of features (hierarchical: MM, whale, active retail, retail, mid)."""
    n = pl.col("n_prev")
    maker = pl.col("maker_prev") / pl.max_horizontal(n, 1)
    two = pl.col("two_prev") / pl.max_horizontal(pl.col("mkts_prev"), 1)
    mm = (n >= th["mm_min_prev"]) & (maker >= th["mm_maker"]) & (two >= th["mm_two"])
    size = pl.col("typ_size")
    return (pl.when(mm).then(pl.lit("MM")).when(size >= th["whale"]).then(pl.lit("Whale"))
            .when((size <= th["small"]) & (n >= th["active_prev"])).then(pl.lit("Active retail"))
            .when(size <= th["small"]).then(pl.lit("Retail")).otherwise(pl.lit("Mid")).alias("cls"))


AGE_LABELS = ("<7d", "7-30d", "30-180d", ">180d")
COARSE_LABELS = ("<=10c", "11-35c", "36-65c", "66-89c", ">=90c")


def coarse_bucket_expr(q: str = "q") -> pl.Expr:
    c = pl.col(q)
    return (pl.when(c <= 0.10).then(pl.lit(COARSE_LABELS[0])).when(c <= 0.35).then(pl.lit(COARSE_LABELS[1]))
            .when(c <= 0.65).then(pl.lit(COARSE_LABELS[2])).when(c < 0.90).then(pl.lit(COARSE_LABELS[3]))
            .otherwise(pl.lit(COARSE_LABELS[4])).cast(pl.Enum(list(COARSE_LABELS))).alias("cbucket"))


def age_bucket_expr(days: str = "age_days") -> pl.Expr:
    c = pl.col(days)
    return (pl.when(c.is_null()).then(None).when(c < 7).then(pl.lit(AGE_LABELS[0])).when(c < 30)
            .then(pl.lit(AGE_LABELS[1])).when(c < 180).then(pl.lit(AGE_LABELS[2])).otherwise(pl.lit(AGE_LABELS[3]))
            .alias("age_bucket"))


# ---- inference helpers ---------------------------------------------------------------------------
def logit_slope(q: Any, y: Any, g: Any, w: Any | None = None) -> dict[str, Any]:
    """Logistic regression ``logit P(y=1) = a + b logit(q)`` with SEs clustered by ``g``.

    ``b = 1`` is calibration; ``b > 1`` means prices are compressed toward 1/2 (longshots too dear,
    favourites too cheap: the favourite-longshot bias).
    """
    q = np.clip(np.asarray(q, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    g = np.asarray(g)
    wt = np.ones_like(q) if w is None else np.asarray(w, float)
    ok = np.isfinite(q) & np.isfinite(y) & np.isfinite(wt)
    q, y, g, wt = q[ok], y[ok], g[ok], wt[ok]
    if len(q) < 10:
        return {"n": len(q), "clusters": 0, "a": None, "b": None, "se_a": None, "se_b": None, "t_b1": None}
    x = np.log(q / (1 - q))
    X = np.column_stack([np.ones_like(x), x])
    wt = wt / wt.mean()
    beta = np.array([0.0, 1.0])
    empty = {"n": len(q), "clusters": 0, "a": None, "b": None, "se_a": None, "se_b": None, "t_b1": None}
    try:
        for _ in range(100):
            mu = 1 / (1 + np.exp(-(X @ beta)))
            H = X.T @ (X * (wt * mu * (1 - mu))[:, None])
            step = np.linalg.solve(H, X.T @ (wt * (y - mu)))
            beta = beta + step
            if np.abs(step).max() < 1e-10:
                break
        else:
            return empty
    except np.linalg.LinAlgError:
        return empty                      # separation (e.g. a tiny subsample): no finite slope
    if not np.isfinite(beta).all() or abs(beta[1]) > 50:
        return empty
    mu = 1 / (1 + np.exp(-(X @ beta)))
    H = X.T @ (X * (wt * mu * (1 - mu))[:, None])
    _, inv = np.unique(g, return_inverse=True)
    S = np.zeros((inv.max() + 1, 2))
    np.add.at(S, inv, X * (wt * (y - mu))[:, None])
    k = S.shape[0]
    Hi = np.linalg.inv(H)
    V = Hi @ (S.T @ S * k / max(k - 1, 1)) @ Hi
    se = np.sqrt(np.diag(V))
    return {"n": len(q), "clusters": int(k), "a": float(beta[0]), "b": float(beta[1]),
            "se_a": float(se[0]), "se_b": float(se[1]), "t_b1": float((beta[1] - 1) / se[1])}


def gini(x: Any) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return float("nan")
    return float(2 * (np.arange(1, n + 1) * x).sum() / (n * x.sum()) - (n + 1) / n)


def lorenz(x: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, float))
    return np.linspace(0, 1, len(x) + 1), np.r_[0.0, np.cumsum(x) / x.sum()]


def lsv_herding(k: Any, n: Any, pbar: Any) -> np.ndarray:
    """Lakonishok-Shleifer-Vishny herding of each window: |k/n - pbar| - E|K/n - pbar|, K ~ Bin(n, pbar)."""
    from scipy.stats import binom
    k, n, pbar = (np.asarray(v) for v in (k, n, pbar))
    out = np.abs(k / n - pbar)
    adj = np.empty(len(n), float)
    for i, (ni, pi) in enumerate(zip(n.astype(int), pbar.astype(float), strict=True)):
        if ni > 300:
            adj[i] = math.sqrt(pi * (1 - pi) / ni) * math.sqrt(2 / math.pi)
        else:
            j = np.arange(ni + 1)
            adj[i] = float((binom.pmf(j, ni, pi) * np.abs(j / ni - pi)).sum())
    return out - adj


def _cl(x: Any, g: Any, w: Any | None = None) -> dict[str, Any]:
    from albert.research.flb_common import clustered
    r = clustered(x, g, w)
    return {k: (float(v) if isinstance(v, float | np.floating) else v) for k, v in r.items()}


# ---- step: build (markets with information cut-offs; fills with ids and as-of features) -----------
HALF_SPLIT = datetime(2025, 6, 1, tzinfo=UTC)     # H1: 2024-01..2025-05, H2: 2025-06..2026-10 (17 months each)
N_PARTS = 8
EARLY_BUFFER_H = 24


def market_table() -> pl.DataFrame:
    """Sampled markets with fetched trades, the information cut-off ``t_info`` and the cluster id.

    ``t_info`` is the scheduled game start for sports games and the scheduled end date otherwise; for a
    market that closed before that schedule (the event happened early) it is the close minus 24 h.
    """
    s = pl.read_parquet(TAB / "sample.parquet")
    log = pl.read_ndjson(TAB / "trades_log.jsonl").unique("condition_id", keep="last")
    s = s.join(log.select("condition_id", pl.col("rows").alias("n_rows"), "truncated"), on="condition_id")
    s = s.sort("condition_id").with_row_index("mid").with_columns(pl.col("mid").cast(pl.Int32))
    game = (pl.col("group") == "Sports") & pl.col("game_start_time").is_not_null()
    s = s.with_columns(pl.when(game).then(pl.col("game_start_time")).otherwise(pl.col("end_date")).alias("t_sched"),
                       game.alias("is_game"))
    early = (pl.col("closed_time") < pl.col("t_sched")).fill_null(False)
    s = s.with_columns(early.alias("closed_early"),
                       pl.when(early).then(pl.col("closed_time") - pl.duration(hours=EARLY_BUFFER_H))
                       .otherwise(pl.col("t_sched")).alias("t_info"))
    cluster = pl.when(pl.col("event_id") != "").then(pl.col("event_id")).otherwise(pl.col("event_slug"))
    s = s.with_columns(pl.col("t_info").dt.epoch("s").alias("t_info_s"),
                       pl.coalesce("closed_time", "end_date").dt.epoch("s").alias("t_close_s"),
                       cluster.alias("cluster"),
                       pl.when(pl.col("resolved_at") < HALF_SPLIT).then(pl.lit("H1")).otherwise(pl.lit("H2")).alias("half"))
    return s.with_columns(pl.col("cluster").rank("dense").cast(pl.Int32).alias("ev"))


def run_build() -> None:
    mk = market_table()
    _require(mk, "markets")
    mk.write_parquet(TAB / "markets.parquet")
    tr = load_trades()
    wallets = (tr.select("wallet").unique().collect().sort("wallet").with_row_index("wid")
               .with_columns(pl.col("wid").cast(pl.Int32)))
    wallets.write_parquet(TAB / "wallets.parquet")
    out = TAB / "fills"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part-*.parquet"):
        old.unlink()
    n = 0
    for j in range(N_PARTS):
        part = (tr.join(mk.lazy().select("condition_id", "mid"), on="condition_id")
                .join(wallets.lazy(), on="wallet").filter(pl.col("wid") % N_PARTS == j)
                .select("mid", "wid", "buy", pl.col("outcome_index").cast(pl.Int8), "price", "size", "ts", "taker")
                .collect())
        feat = asof_features(part)
        feat.write_parquet(out / f"part-{j}.parquet")
        n += feat.height
        print(f"  build part {j}: {feat.height} fills", flush=True)
        del part, feat
    print(f"build: {mk.height} markets, {wallets.height} wallets, {n} fills", flush=True)


def _fills() -> pl.LazyFrame:
    return pl.scan_parquet(TAB / "fills" / "part-*.parquet")


def _fill_parts() -> list[pl.LazyFrame]:
    """The fills, one lazy frame per wallet partition (every fill of a wallet is in exactly one part)."""
    return [pl.scan_parquet(f) for f in sorted((TAB / "fills").glob("part-*.parquet"))]


def _users() -> pl.DataFrame | None:
    path = TAB / "users.parquet"
    if not path.exists():
        return None
    u = pl.read_parquet(path).join(pl.read_parquet(TAB / "wallets.parquet"), on="wallet")
    return u.select("wid", "join_date", "known", "markets_traded_now")


def _leaderboard() -> pl.DataFrame:
    """Wallet ids on the all-time Polymarket leaderboards (overall, politics, sports; top 100 each, fetched
    2026-11-03 into ``pm_leaderboard``). Membership is measured after the fact: descriptive only."""
    lb = pl.read_parquet(settings().tables_dir / "pm_leaderboard.parquet").select("wallet").unique()
    return pl.read_parquet(TAB / "wallets.parquet").join(lb, on="wallet").select("wid", pl.lit("Leaderboard").alias("lb"))


GROUP_ENUM = pl.Enum(["Sports", "Weather", "Politics", "Entertainment", "Mentions", "Science & Tech", "Other"])
CLASS_ENUM = pl.Enum(list(CLASSES))


def _mk_cols(mk: pl.DataFrame) -> pl.DataFrame:
    return mk.select("mid", "ev", pl.col("group").cast(GROUP_ENUM), "weight", "winner_index", "t_info_s", "t_close_s",
                     "fees_enabled", "fee_rate", "half", "volume", "tier", "is_game", "truncated")


def buys_frame(mk: pl.DataFrame, users: pl.DataFrame | None, *, pre: bool = True) -> pl.DataFrame:
    """BUY fills (pre-information by default) with side price, outcome, returns, class and account age."""
    from albert.research.flb_common import bucket_expr, size_bucket_expr
    f = (_fills().filter(pl.col("buy")).join(_mk_cols(mk).lazy(), on="mid")
         .filter(~pl.col("truncated")))
    f = f.filter(pl.col("ts") < pl.col("t_info_s")) if pre else f.filter(pl.col("ts") < pl.col("t_close_s"))
    q = pl.col("price")
    fee = pl.when(pl.col("taker") & pl.col("fees_enabled")).then(pl.col("fee_rate") * q * (1 - q)).otherwise(0.0)
    won = (pl.col("outcome_index") == pl.col("winner_index")).cast(pl.Float64)
    f = f.with_columns(q.alias("q"), won.alias("won"), fee.alias("fee_ps")).with_columns(
        ((pl.col("won") - pl.col("q")) / pl.col("q")).alias("ret_g"),
        ((pl.col("won") - pl.col("q") - pl.col("fee_ps")) / (pl.col("q") + pl.col("fee_ps"))).alias("ret_n"),
        bucket_expr("q"), size_bucket_expr("notional"), class_expr(DEFAULT_TH).cast(CLASS_ENUM),
        (pl.col("notional") * pl.col("weight")).alias("w_usd"))
    f = f.join(_leaderboard().lazy(), on="wid", how="left").with_columns(pl.col("lb").fill_null("Other"))
    if users is not None:
        f = f.join(users.lazy().select("wid", "join_date"), on="wid", how="left").with_columns(
            ((pl.col("ts") - pl.col("join_date")) / 86400).alias("age_days")).with_columns(age_bucket_expr())
    from albert.research.flb_common import BUCKET_LABELS, SIZE_LABELS
    f32 = pl.Float32
    keep = ["mid", "wid", "ev", "group", pl.col("half").cast(pl.Enum(["H1", "H2"])), "ts", pl.col("q").cast(f32),
            pl.col("won").cast(f32), "taker", "notional", "weight", "w_usd", pl.col("fee_ps").cast(f32),
            pl.col("ret_g").cast(f32), pl.col("ret_n").cast(f32), pl.col("bucket").cast(pl.Enum(list(BUCKET_LABELS))),
            pl.col("size_bucket").cast(pl.Enum(list(SIZE_LABELS))), "cls", "n_prev", "maker_prev", "mkts_prev",
            "two_prev", pl.col("typ_size").cast(f32), "is_game", "outcome_index",
            pl.col("lb").cast(pl.Enum(["Leaderboard", "Other"])), "fees_enabled",
            (pl.col("q") <= 0.10).cast(f32).alias("is_ls"), (pl.col("q") >= 0.90).cast(f32).alias("is_fav"),
            coarse_bucket_expr("q")]
    if users is not None:
        keep += [pl.col("age_days").cast(f32), pl.col("age_bucket").cast(pl.Enum(list(AGE_LABELS)))]
    return f.select(keep).collect(engine="streaming")


# ---- analyses --------------------------------------------------------------------------------------
GROUPS_ORDER = ("Sports", "Politics", "Weather", "Entertainment", "Mentions", "Science & Tech", "Other")
LEADS_H = (1, 24, 168)


def _np(df: pl.DataFrame, *cols: str) -> list[np.ndarray]:
    return [df[c].to_numpy() for c in cols]


def _cell(df: pl.DataFrame, x: str, w: str | None, **tags: Any) -> dict[str, Any]:
    if df.height == 0:
        return {**tags, "n": 0, "clusters": 0, "mean": None, "se": None, "t": None}
    return {**tags, **_cl(df[x].to_numpy(), df["ev"].to_numpy(), None if w is None else df[w].to_numpy())}


def ols_cl(y: Any, X: Any, g: Any, w: Any | None = None) -> dict[str, Any]:
    """(Weighted) least squares with an intercept prepended and SEs clustered by ``g``."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    wt = np.ones(len(y)) if w is None else np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(X).all(1) & np.isfinite(wt)
    y, X, g, wt = y[ok], X[ok], np.asarray(g)[ok], wt[ok]
    if len(y) <= X.shape[1] + 1:
        return {"n": len(y), "beta": None, "se": None, "t": None, "clusters": 0}
    wt = wt / wt.mean()
    XtXi = np.linalg.inv(X.T @ (X * wt[:, None]))
    b = XtXi @ X.T @ (wt * y)
    e = y - X @ b
    _, inv = np.unique(g, return_inverse=True)
    S = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(S, inv, X * (wt * e)[:, None])
    k = S.shape[0]
    V = XtXi @ (S.T @ S * k / max(k - 1, 1)) @ XtXi
    se = np.sqrt(np.diag(V))
    return {"n": len(y), "clusters": int(k), "beta": b.tolist(), "se": se.tolist(), "t": (b / se).tolist()}


def yes_prices(mk: pl.DataFrame, *, pre: bool = True) -> pl.DataFrame:
    """YES-equivalent trade prices (mid, ts, p) of non-truncated markets, sorted; pre-information by default."""
    f = (_fills().select("mid", "ts", "price", "outcome_index").join(
        mk.lazy().select("mid", "t_info_s", "t_close_s", "truncated"), on="mid").filter(~pl.col("truncated")))
    f = f.filter(pl.col("ts") < (pl.col("t_info_s") if pre else pl.col("t_close_s")))
    return (f.select("mid", "ts", pl.when(pl.col("outcome_index") == 0).then(pl.col("price"))
                     .otherwise(1 - pl.col("price")).cast(pl.Float32).alias("p")).collect(engine="streaming").sort("mid", "ts"))


def calibration(mk: pl.DataFrame, px: pl.DataFrame) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    """Market-level calibration from the last trade at fixed leads before ``t_info`` (both sides)."""
    from albert.research.flb_common import BUCKET_LABELS, bucket_expr
    rows, slopes, summ = [], [], {}
    mcols = _mk_cols(mk).select("mid", "ev", "group", "weight", "volume", "winner_index", "t_info_s", "half")
    for lead in LEADS_H:
        last = (px.join(mcols.select("mid", "t_info_s"), on="mid").filter(pl.col("ts") <= pl.col("t_info_s") - lead * 3600)
                .group_by("mid").agg(pl.col("p").sort_by("ts").last(), pl.col("ts").max().alias("last_ts")))
        d = last.join(mcols, on="mid").with_columns(
            (pl.col("winner_index") == 0).cast(pl.Float64).alias("y"),
            ((pl.col("t_info_s") - lead * 3600 - pl.col("last_ts")) / 3600).alias("stale_h"),
            (pl.col("weight") * pl.col("volume")).alias("wv"), pl.lit(1.0).alias("one"))
        cols = ["mid", "ev", "group", "half", "weight", "wv", "one"]
        both = pl.concat([d.select(*cols, pl.col("p").alias("q"), pl.col("y").alias("won")),
                          d.select(*cols, (1 - pl.col("p")).alias("q"), (1 - pl.col("y")).alias("won"))])
        both = both.with_columns(bucket_expr("q"), ((pl.col("won") - pl.col("q")) / pl.col("q")).alias("ret"))
        scopes = [("All", both, d), *[(g, both.filter(pl.col("group") == g), d.filter(pl.col("group") == g))
                                      for g in GROUPS_ORDER]]
        scopes += [(f"All {h}", both.filter(pl.col("half") == h), d.filter(pl.col("half") == h)) for h in ("H1", "H2")]
        for name, b, dd in scopes:
            for lab in BUCKET_LABELS:
                c = b.filter(pl.col("bucket") == lab)
                for wname, w in (("market", "weight"), ("volume", "wv"), ("sample", "one")):
                    rows.append(_cell(c, "ret", w, lead_h=lead, scope=name, bucket=lab, weighting=wname))
            for wname, w in (("market", "weight"), ("volume", "wv"), ("sample", "one")):
                for side, cond in (("LS", pl.col("q") <= 0.10), ("FAV", pl.col("q") >= 0.90)):
                    rows.append(_cell(b.filter(cond), "ret", w, lead_h=lead, scope=name, bucket=side, weighting=wname))
            for wname, w in (("market", "weight"), ("sample", None)):
                s = logit_slope(dd["p"].to_numpy(), dd["y"].to_numpy(), dd["ev"].to_numpy(),
                                None if w is None else dd[w].to_numpy())
                slopes.append({"lead_h": lead, "scope": name, "weighting": wname, **s, "markets": dd.height,
                               "median_stale_h": float(dd["stale_h"].median()) if dd.height else None})
        summ[f"lead_{lead}h"] = {"markets": d.height, "median_stale_h": float(d["stale_h"].median())}
    return summ, pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(slopes, infer_schema_length=None)


def grouped_cl(df: pl.DataFrame | pl.LazyFrame, by: list[str], x: str, w: str | None,
               cluster: str = "ev") -> pl.DataFrame:
    """``clustered`` for every group of ``by`` in one pass, from per-cluster sums (no per-cell copies).

    Per cluster c of a group: s_c = sum(w x) - mu sum(w); se = sqrt(sum s_c^2 k/(k-1)) / sum(w).
    """
    wexpr = pl.col(w).cast(pl.Float64) if w else pl.lit(1.0)
    xe = pl.col(x).cast(pl.Float64)
    g = (df.lazy().filter(xe.is_finite()).group_by([*by, cluster])
         .agg(pl.len().alias("n"), wexpr.sum().alias("sw"), (wexpr * xe).sum().alias("swx")))
    tot = g.group_by(by).agg(pl.col("n").sum(), pl.len().alias("clusters"), pl.col("sw").sum().alias("SW"),
                             pl.col("swx").sum().alias("SWX")).with_columns((pl.col("SWX") / pl.col("SW")).alias("mean"))
    s = (g.join(tot.select(*by, "mean"), on=by, nulls_equal=True)
         .with_columns((pl.col("swx") - pl.col("mean") * pl.col("sw")).alias("s"))
         .group_by(by).agg((pl.col("s") ** 2).sum().alias("ss")))
    out = tot.join(s, on=by, nulls_equal=True).with_columns(
        pl.when(pl.col("clusters") > 1).then((pl.col("ss") * pl.col("clusters") / (pl.col("clusters") - 1)).sqrt()
                                             / pl.col("SW")).alias("se"))
    out = out.with_columns((pl.col("mean") / pl.col("se")).alias("t")).select(*by, "n", "clusters", "mean", "se", "t")
    return out.collect()


def _stack(parts: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat([p for p in parts if p.height], how="diagonal_relaxed")


def trade_flb(b: pl.DataFrame) -> pl.DataFrame:
    """Hold-to-settlement return per dollar of BUY fills by side-price bucket (gross / net of taker fees)."""
    side = pl.when(pl.col("q") <= 0.10).then(pl.lit("LS")).when(pl.col("q") >= 0.90).then(pl.lit("FAV"))
    lf = b.lazy().select("ev", "group", "half", "taker", "fees_enabled", "ret_g", "ret_n", "w_usd", "weight",
                         pl.col("bucket").cast(pl.String), side.alias("side"),
                         pl.when(pl.col("taker")).then(pl.lit("Taker")).otherwise(pl.lit("Maker")).alias("role"),
                         pl.when(pl.col("fees_enabled")).then(pl.lit("Fee markets")).otherwise(pl.lit("Fee-free markets"))
                         .alias("fees"),
                         (pl.lit("All ") + pl.col("half").cast(pl.String)).alias("halfs"),
                         pl.col("group").cast(pl.String).alias("grp"))
    scopes = {"All": (lf, None), "role": (lf, "role"), "grp": (lf, "grp"), "halfs": (lf, "halfs"), "fees": (lf, "fees"),
              "taker_fee": (lf.filter(pl.col("taker") & pl.col("fees_enabled")), None)}
    parts = []
    for key, (src, col) in scopes.items():
        for lab_col in ("bucket", "side"):
            by = [lab_col] if col is None else [col, lab_col]
            for ret in ("ret_g", "ret_n"):
                for wname, w in (("dollar", "w_usd"), ("fill", "weight")):
                    r = grouped_cl(src.filter(pl.col(lab_col).is_not_null()), by, ret, w)
                    name = (pl.lit("All") if key == "All" else pl.lit("Taker, fee markets") if key == "taker_fee"
                            else pl.col(col))
                    parts.append(r.select(name.alias("scope"), pl.col(lab_col).alias("bucket"), pl.lit(ret).alias("ret"),
                                          pl.lit(wname).alias("weighting"), "n", "clusters", "mean", "se", "t"))
    return _stack(parts)


def who_buys(b: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Longshot / favourite share of buying and returns by wallet class, fill size, leaderboard and account age."""
    dims = ["cls", "size_bucket", "lb"] + (["age_bucket"] if "age_bucket" in b.columns else [])
    share = []
    lf = b.lazy()
    for dim in dims:
        for half in ("All", "H1", "H2"):
            src = lf if half == "All" else lf.filter(pl.col("half") == half)
            src = src.filter(pl.col(dim).is_not_null()).with_columns(pl.col(dim).cast(pl.String).alias("label"))
            base = src.group_by("label").agg(pl.len().alias("fills"), pl.col("wid").n_unique().alias("wallets"),
                                             pl.col("w_usd").sum().alias("usd_w")).collect()
            cells = []
            for x in ("is_ls", "is_fav"):
                for wname, w in (("dollar", "w_usd"), ("fill", "weight")):
                    cells.append(grouped_cl(src, ["label"], x, w).with_columns(pl.lit(x).alias("stat"), pl.lit(wname).alias("weighting")))
            for ret in ("ret_g", "ret_n"):
                for wname, w in (("dollar", "w_usd"), ("fill", "weight")):
                    cells.append(grouped_cl(src.filter(pl.col("is_ls") == 1), ["label"], ret, w).with_columns(
                        pl.lit(f"LS {ret}").alias("stat"), pl.lit(wname).alias("weighting")))
            share.append(base.join(_stack(cells), on="label").with_columns(pl.lit(dim).alias("dim"), pl.lit(half).alias("half")))
    share_df = _stack(share).select("dim", "label", "half", "fills", "wallets", "usd_w", "stat", "weighting",
                                    "n", "clusters", "mean", "se", "t")
    cb = _stack([grouped_cl(lf, ["cls", "bucket"], ret, "w_usd").with_columns(pl.lit(ret).alias("ret"), pl.lit("dollar").alias("weighting"))
                 for ret in ("ret_g", "ret_n")]).with_columns(pl.col("cls").cast(pl.String), pl.col("bucket").cast(pl.String))
    ccb = _stack([grouped_cl(lf, ["cls", "cbucket"], ret, w).with_columns(pl.lit(ret).alias("ret"), pl.lit(wname).alias("weighting"))
                  for ret in ("ret_g", "ret_n") for wname, w in (("dollar", "w_usd"), ("fill", "weight"))]).with_columns(
        pl.col("cls").cast(pl.String), pl.col("cbucket").cast(pl.String))
    var = []
    base_cols = lf.select("wid", "ev", "is_ls", "ret_n", "w_usd", "n_prev", "maker_prev", "mkts_prev", "two_prev", "typ_size")
    for name, th in TH_VARIANTS.items():
        v = base_cols.with_columns(class_expr(th))
        cnt = v.group_by("cls").agg(pl.len().alias("fills"), pl.col("wid").n_unique().alias("wallets")).collect()
        a = grouped_cl(v, ["cls"], "is_ls", "w_usd").rename({k: f"ls_share_{k}" for k in ("n", "clusters", "mean", "se", "t")})
        r = grouped_cl(v.filter(pl.col("is_ls") == 1), ["cls"], "ret_n", "w_usd").rename(
            {k: f"ls_ret_{k}" for k in ("n", "clusters", "mean", "se", "t")})
        var.append(cnt.join(a, on="cls", how="left").join(r, on="cls", how="left").with_columns(pl.lit(name).alias("variant")))
    var_df = _stack(var)
    var_df = var_df.select("variant", *[c for c in var_df.columns if c != "variant"])
    return {"share": share_df, "class_bucket": cb.select("cls", "bucket", "ret", "weighting", "n", "clusters", "mean", "se", "t"),
            "class_cbucket": ccb.select("cls", "cbucket", "ret", "weighting", "n", "clusters", "mean", "se", "t"),
            "variants": var_df}


def ls_flow(mk: pl.DataFrame) -> pl.DataFrame:
    """Net longshot dollars bought by class (pre-information fills, buys and sells; zero-sum across classes).

    A fill in a token priced <= 10c is longshot flow at its notional (buy +, sell -); a fill in a token
    priced >= 90c is the opposite longshot flow at the complement's cost (buying the favourite sells the
    longshot). The per-dollar figure is the ratio sum(net * w) / sum(notional * w), w the design weight,
    with SEs clustered by event; it is built from per-cluster sums one wallet partition at a time."""
    p, s = pl.col("price"), pl.col("size")
    sig = (pl.when((p <= 0.10) & pl.col("buy")).then(p * s).when(p <= 0.10).then(-p * s)
           .when((p >= 0.90) & pl.col("buy")).then(-(1 - p) * s).when(p >= 0.90).then((1 - p) * s).otherwise(0.0))
    mc = _mk_cols(mk).lazy()
    sums = []
    for part in _fill_parts():
        f = (part.join(mc, on="mid").filter(~pl.col("truncated") & (pl.col("ts") < pl.col("t_info_s")))
             .with_columns(sig.alias("ls_net"), class_expr(DEFAULT_TH).cast(pl.String),
                           pl.when(pl.col("taker")).then(pl.lit("taker")).otherwise(pl.lit("maker")).alias("role"))
             .group_by("half", "cls", "role", "ev")
             .agg(pl.len().alias("n"), (pl.col("ls_net") * pl.col("weight")).sum().alias("swx"),
                  (pl.col("notional") * pl.col("weight")).sum().alias("sw"))
             .collect(engine="streaming"))
        sums.append(f)
    cs = pl.concat(sums).group_by("half", "cls", "role", "ev").agg(pl.col("n", "swx", "sw").sum())
    views = [cs.with_columns(pl.lit("All").alias("half")), cs]
    cs = pl.concat([v.with_columns(pl.col("half").cast(pl.String)) for v in views])
    cs = pl.concat([cs, cs.with_columns(pl.lit("all").alias("role"))]).group_by("half", "cls", "role", "ev").agg(
        pl.col("n", "swx", "sw").sum())
    tot = cs.group_by("half", "cls", "role").agg(pl.col("n").sum().alias("fills"), pl.len().alias("per_usd_clusters"),
                                                 pl.col("swx").sum().alias("net_ls_usd_w"), pl.col("sw").sum().alias("usd_w"))
    tot = tot.with_columns((pl.col("net_ls_usd_w") / pl.col("usd_w")).alias("per_usd_mean"))
    ss = (cs.join(tot.select("half", "cls", "role", "per_usd_mean"), on=["half", "cls", "role"])
          .with_columns((pl.col("swx") - pl.col("per_usd_mean") * pl.col("sw")).alias("s"))
          .group_by("half", "cls", "role").agg((pl.col("s") ** 2).sum().alias("ss")))
    out = tot.join(ss, on=["half", "cls", "role"]).with_columns(
        pl.when(pl.col("per_usd_clusters") > 1).then(
            (pl.col("ss") * pl.col("per_usd_clusters") / (pl.col("per_usd_clusters") - 1)).sqrt() / pl.col("usd_w"))
        .alias("per_usd_se"))
    tot_abs = float(out.filter((pl.col("half") == "All") & (pl.col("role") == "all"))["net_ls_usd_w"].abs().sum()) or 1.0
    out = out.with_columns((pl.col("per_usd_mean") / pl.col("per_usd_se")).alias("per_usd_t"),
                           (pl.col("net_ls_usd_w") / tot_abs).alias("share_of_abs_net"), pl.col("fills").alias("per_usd_n"))
    order = {c: i for i, c in enumerate(CLASSES)}
    return (out.with_columns(pl.col("cls").replace_strict(order, return_dtype=pl.Int32).alias("_o"),
                             pl.col("half").replace_strict({"All": 0, "H1": 1, "H2": 2}, return_dtype=pl.Int32).alias("_h"),
                             pl.col("role").replace_strict({"all": 0, "taker": 1, "maker": 2}, return_dtype=pl.Int32).alias("_r"))
            .sort("_h", "_o", "_r")
            .select("half", "cls", "role", "fills", "net_ls_usd_w", "usd_w", "share_of_abs_net", "per_usd_n",
                    "per_usd_clusters", "per_usd_mean", "per_usd_se", "per_usd_t"))


def hold_exit(mk: pl.DataFrame) -> pl.DataFrame:
    """Positions opened by a buy before ``t_info``: held to settlement vs (partly) sold before close."""
    from albert.research.flb_common import bucket_expr
    mc = _mk_cols(mk).lazy()
    f = pl.concat([(part.join(mc, on="mid").filter(~pl.col("truncated") & (pl.col("ts") < pl.col("t_close_s")))
         .group_by("wid", "mid", "outcome_index").agg(
             pl.col("buy").sort_by("ts").first().alias("first_buy"), pl.col("ts").min().alias("t0"),
             pl.col("t_info_s").first(), pl.col("ev").first(), pl.col("weight").first(),
             (pl.col("winner_index").first() == pl.col("outcome_index").first()).cast(pl.Float64).alias("won"),
             pl.col("size").filter(pl.col("buy")).sum().alias("B"), pl.col("notional").filter(pl.col("buy")).sum().alias("cost"),
             pl.col("size").filter(~pl.col("buy")).sum().alias("S"),
             pl.col("notional").filter(~pl.col("buy")).sum().alias("proceeds"),
             *[pl.col(c).sort_by("ts").first() for c in ("n_prev", "maker_prev", "mkts_prev", "two_prev", "typ_size")])
         .filter(pl.col("first_buy") & (pl.col("B") > 0) & (pl.col("cost") > 0) & (pl.col("t0") < pl.col("t_info_s")))
         .collect(engine="streaming")) for part in _fill_parts()])
    f = f.with_columns(pl.min_horizontal("S", "B").alias("sold"),
                       (pl.col("cost") / pl.col("B")).alias("q_entry"), class_expr(DEFAULT_TH)).with_columns(
        (pl.col("proceeds") * pl.when(pl.col("S") > pl.col("B")).then(pl.col("B") / pl.col("S")).otherwise(1.0)).alias("proc"),
        bucket_expr("q_entry"))
    f = f.with_columns(
        ((pl.col("proc") - pl.col("sold") * pl.col("won")) / pl.col("cost")).alias("exit_va"),
        (pl.col("sold") / pl.col("B") >= 0.5).cast(pl.Float64).alias("exited"),
        ((pl.col("B") * pl.col("won") - pl.col("cost")) / pl.col("cost")).alias("hold_ret"),
        ((pl.col("proc") + (pl.col("B") - pl.col("sold")) * pl.col("won") - pl.col("cost")) / pl.col("cost")).alias("real_ret"),
        (pl.col("cost") * pl.col("weight")).alias("wc"))
    rows = []
    for c in (*CLASSES, "All"):
        s = f if c == "All" else f.filter(pl.col("cls") == c)
        for entry, cond in (("all", pl.lit(True)), ("LS", pl.col("q_entry") <= 0.10), ("mid", pl.col("q_entry").is_between(0.10, 0.90, closed="none")),
                            ("FAV", pl.col("q_entry") >= 0.90)):
            ss = s.filter(cond)
            rows.append({"cls": c, "entry": entry, "positions": ss.height,
                         **{f"exited_{k}": v for k, v in _cell(ss, "exited", "wc").items()},
                         **{f"exit_va_{k}": v for k, v in _cell(ss, "exit_va", "wc").items()},
                         **{f"hold_{k}": v for k, v in _cell(ss, "hold_ret", "wc").items()},
                         **{f"real_{k}": v for k, v in _cell(ss, "real_ret", "wc").items()}})
    return pl.DataFrame(rows, infer_schema_length=None)


def wallet_markets(mk: pl.DataFrame) -> pl.DataFrame:
    """Exact trade-based PnL of each wallet in each fully fetched market (net of taker fees)."""
    p = pl.col("price")
    fee = pl.when(pl.col("taker") & pl.col("fees_enabled")).then(pl.col("fee_rate") * p * (1 - p) * pl.col("size")).otherwise(0.0)
    mc = _mk_cols(mk).lazy()
    wm = pl.concat([
        normalize(part.join(mc, on="mid").filter(~pl.col("truncated")))
        .with_columns(fee.alias("fee"), class_expr(DEFAULT_TH).cast(CLASS_ENUM))
        .group_by("wid", "mid").agg(
            pl.col("cash").sum(), pl.col("expo").sum(), pl.col("fee").sum().alias("fees"),
            pl.col("notional").filter(pl.col("buy")).sum().alias("bought"), pl.col("notional").sum().alias("traded"),
            pl.len().alias("fills"), pl.col("cls").sort_by("ts").first().alias("cls"),
            pl.col("ts").min().alias("t0"))
        .collect(engine="streaming") for part in _fill_parts()])
    wm = wm.join(mk.select("mid", "ev", "half", "weight", "winner_index", "volume", "group", "fees_enabled"),
                 on="mid").with_columns(
        (pl.col("cash") + (pl.col("winner_index") == 0).cast(pl.Float64) * pl.col("expo")).alias("pnl_g")).with_columns(
        (pl.col("pnl_g") - pl.col("fees")).alias("pnl"))
    return wm


def concentration(wm: pl.DataFrame) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    # In fee-free markets every fill nets to zero across wallets; with fees, takers' buy sizes are net
    # of the fee taken in shares, so the residual there is the protocol's fee income.
    zs = wm.group_by("mid").agg(pl.col("pnl_g").sum().alias("sum_pnl"), pl.col("expo").sum().alias("sum_expo"),
                                pl.col("traded").sum(), pl.col("volume").first(), pl.col("fees_enabled").first())
    zero_sum = {}
    for name, z in (("fee_free", zs.filter(~pl.col("fees_enabled"))), ("fee_enabled", zs.filter(pl.col("fees_enabled")))):
        if z.height == 0:
            continue
        zero_sum[name] = {"markets": z.height,
                          "share_exact_expo_lt_1e-6": float((z["sum_expo"].abs() < 1e-6).mean()),
                          "share_abs_sum_pnl_lt_1e-6_traded": float((z["sum_pnl"].abs() < 1e-6 * z["traded"]).mean()),
                          "share_abs_sum_pnl_lt_1pct_traded": float((z["sum_pnl"].abs() < 0.01 * z["traded"]).mean()),
                          "median_abs_sum_pnl_over_traded": float((z["sum_pnl"].abs() / z["traded"]).median())}
    w = wm.group_by("wid").agg(pl.col("pnl").sum(), pl.col("pnl_g").sum(), pl.col("bought").sum(), pl.col("traded").sum(),
                               pl.len().alias("markets"), pl.col("fees").sum(), pl.col("cls").mode().first().alias("cls_mode"))
    w = w.join(_leaderboard(), on="wid", how="left")
    lbw = w.filter(pl.col("lb").is_not_null())
    tot_g = float(w.filter(pl.col("pnl") > 0)["pnl"].sum())
    leaderboard = {"wallets": lbw.height, "pnl": float(lbw["pnl"].sum()), "traded": float(lbw["traded"].sum()),
                   "share_of_gains": float(lbw.filter(pl.col("pnl") > 0)["pnl"].sum()) / tot_g if tot_g else None,
                   "share_of_traded": float(lbw["traded"].sum() / w["traded"].sum()),
                   "share_profitable": float((lbw["pnl"] > 0).mean()) if lbw.height else None}
    pnl = w["pnl"].to_numpy()
    gains = np.sort(pnl[pnl > 0])[::-1]
    losses = np.sort(-pnl[pnl < 0])[::-1]
    n = len(pnl)
    top = {f"top{int(p * 1000) / 10:g}pct_wallets_share_of_gains": float(gains[:max(1, round(p * n))].sum() / gains.sum())
           for p in (0.001, 0.01, 0.10)}
    summ = {"zero_sum_check": zero_sum, "wallets": n, "leaderboard_wallets": leaderboard,
            "share_profitable": float((pnl > 0).mean()), "share_zero": float((pnl == 0).mean()),
            "pnl_quantiles": {str(q): float(np.quantile(pnl, q)) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
            "total_gains": float(gains.sum()), "total_losses": float(losses.sum()), "total_fees": float(w["fees"].sum()),
            "gini_gains_among_winners": gini(gains), "gini_losses_among_losers": gini(losses),
            "gini_traded_all": gini(w["traded"].to_numpy()), **top,
            "wallets_for_half_of_gains": int(np.searchsorted(np.cumsum(gains), gains.sum() / 2) + 1),
            "wallets_for_half_of_losses": int(np.searchsorted(np.cumsum(losses), losses.sum() / 2) + 1)}
    lz = []
    for name, x in (("gains", gains), ("losses", losses), ("traded", w["traded"].to_numpy())):
        xs, ys = lorenz(x)
        idx = np.unique(np.linspace(0, len(xs) - 1, 401).astype(int))
        lz += [{"series": name, "kind": "lorenz", "x": float(xs[i]), "y": float(ys[i])} for i in idx]
        top = np.sort(np.asarray(x, float))[::-1]
        cum = np.cumsum(top) / top.sum()
        for f in np.unique(np.r_[np.geomspace(1e-4, 1, 120), 0.001, 0.01, 0.1]):
            k = max(1, int(np.ceil(f * len(top))))
            lz.append({"series": name, "kind": "top_share", "x": float(f), "y": float(cum[k - 1])})
    # PnL per dollar traded by class (class at the wallet's first fill in each market), design-weighted
    cls_rows = []
    wm2 = wm.filter(pl.col("traded") > 0).with_columns((pl.col("pnl") / pl.col("traded")).alias("r"),
                                                        (pl.col("traded") * pl.col("weight")).alias("wt"),
                                                        (pl.col("pnl") * pl.col("weight")).alias("pnl_w"))
    tot_gain_w = float(wm2.filter(pl.col("pnl_w") > 0)["pnl_w"].sum())
    for c in CLASSES:
        for half in ("All", "H1", "H2"):
            s = wm2.filter(pl.col("cls") == c)
            s = s if half == "All" else s.filter(pl.col("half") == half)
            cls_rows.append({"cls": c, "half": half, "wallet_markets": s.height, "wallets": s["wid"].n_unique(),
                             "pnl_w": float(s["pnl_w"].sum()), "traded_w": float(s["wt"].sum()),
                             "pnl_w_over_total_gains": float(s["pnl_w"].sum()) / tot_gain_w,
                             **{f"roi_{k}": v for k, v in _cell(s, "r", "wt").items()}})
    return summ, pl.DataFrame(lz), pl.DataFrame(cls_rows, infer_schema_length=None)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic) if len(a) > 3 else float("nan")


def persistence(wm: pl.DataFrame, b_all: pl.DataFrame, p_close: pl.DataFrame, sims: int = 200,
                min_markets: int = 5, min_bought: float = 50.0) -> tuple[dict[str, Any], pl.DataFrame]:
    """Rank wallets on H1, evaluate on H2: raw and shrunk ROI, price-adjusted edge, and a luck placebo.

    Placebo: each market's outcome is redrawn as Bernoulli(last pre-information YES price), so
    wallets keep their trades and prices but have no outcome-forecasting skill beyond that price.
    """
    wh = wm.group_by("wid", "half").agg(pl.col("pnl").sum(), pl.col("bought").sum(), pl.len().alias("markets"))
    K = float(wh.filter(pl.col("bought") > 0)["bought"].median())
    wh = wh.with_columns((pl.col("pnl") / pl.col("bought")).alias("roi"), (pl.col("pnl") / (pl.col("bought") + K)).alias("roi_shr"))
    ok = wh.filter((pl.col("markets") >= min_markets) & (pl.col("bought") >= min_bought))
    a = ok.filter(pl.col("half") == "H1").select("wid", pl.col("roi").alias("roi_a"), pl.col("roi_shr").alias("shr_a"))
    bb = ok.filter(pl.col("half") == "H2").select("wid", pl.col("roi").alias("roi_b"), pl.col("pnl").alias("pnl_b"),
                                                   pl.col("bought").alias("bought_b"))
    pair = a.join(bb, on="wid")
    # price-adjusted edge: buy return minus the mean return of its price bucket in that half (dollar-weighted)
    b_all = b_all.select("wid", "half", "bucket", "ret_g", "notional")
    bench = b_all.group_by("half", "bucket").agg(((pl.col("ret_g") * pl.col("notional")).sum() / pl.col("notional").sum()).alias("bench"))
    edge = (b_all.join(bench, on=["half", "bucket"]).with_columns((pl.col("ret_g") - pl.col("bench")).alias("abn"))
            .group_by("wid", "half").agg(((pl.col("abn") * pl.col("notional")).sum() / pl.col("notional").sum()).alias("edge")))
    pair = (pair.join(edge.filter(pl.col("half") == "H1").select("wid", pl.col("edge").alias("edge_a")), on="wid", how="left")
            .join(edge.filter(pl.col("half") == "H2").select("wid", pl.col("edge").alias("edge_b")), on="wid", how="left")
            .sort("wid"))          # deterministic row order for the seeded bootstraps
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {"wallets_both_halves": pair.height, "shrink_K_usd": K,
                           "min_markets_per_half": min_markets, "min_bought_per_half": min_bought}
    if pair.height < 20:
        return out, pl.DataFrame({"series": [], "quintile": []})
    for score in ("roi_a", "shr_a"):
        out[f"spearman_{score}_roi_b"] = _spearman(pair[score].to_numpy(), pair["roi_b"].to_numpy())
        boots = []
        n = pair.height
        for _ in range(500):
            i = rng.integers(0, n, n)
            boots.append(_spearman(pair[score].to_numpy()[i], pair["roi_b"].to_numpy()[i]))
        out[f"spearman_{score}_ci95"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    e = pair.drop_nulls(["edge_a", "edge_b"])
    out["spearman_edge"] = _spearman(e["edge_a"].to_numpy(), e["edge_b"].to_numpy())
    out["wallets_edge"] = e.height
    pair = pair.with_columns(pl.col("shr_a").qcut(5, labels=[f"Q{i}" for i in range(1, 6)]).alias("quint"))
    qrows = []
    for qn in [f"Q{i}" for i in range(1, 6)]:
        s = pair.filter(pl.col("quint") == qn)
        qrows.append({"series": "actual", "quintile": qn, "wallets": s.height,
                      "roi_b_dollar": float(s["pnl_b"].sum() / s["bought_b"].sum()),
                      "roi_b_mean": float(s["roi_b"].mean()), "roi_b_median": float(s["roi_b"].median()),
                      "shr_a_mean": float(s["shr_a"].mean())})
    # bootstrap SE of the Q5 - Q1 dollar-weighted ROI spread (resampling wallets)
    q = pair["quint"].cast(pl.String).to_numpy()
    pb, bt = pair["pnl_b"].to_numpy(), pair["bought_b"].to_numpy()
    spreads = []
    for _ in range(500):
        i = rng.integers(0, len(q), len(q))
        hi, lo = i[q[i] == "Q5"], i[q[i] == "Q1"]
        spreads.append(pb[hi].sum() / bt[hi].sum() - pb[lo].sum() / bt[lo].sum())
    for k, qn in enumerate([f"Q{i}" for i in range(1, 6)]):
        idx = np.flatnonzero(q == qn)
        bs = []
        for _ in range(300):
            j = rng.choice(idx, len(idx))
            bs.append(pb[j].sum() / bt[j].sum())
        qrows[k]["roi_b_se"] = float(np.std(bs))
    spread = qrows[4]["roi_b_dollar"] - qrows[0]["roi_b_dollar"]
    out["q5_minus_q1_roi_b"] = spread
    out["q5_minus_q1_se"] = float(np.std(spreads))
    out["q5_minus_q1_t"] = spread / float(np.std(spreads))
    # luck placebo on the same wallets and trades
    wmp = wm.join(p_close, on="mid", how="inner")
    mids = wmp["mid"].to_numpy()
    umid, inv = np.unique(mids, return_inverse=True)
    pm_ = wmp.group_by("mid").agg(pl.col("p_close").first()).sort("mid")
    pvec = dict(zip(pm_["mid"].to_list(), pm_["p_close"].to_list(), strict=True))
    pv = np.array([pvec[m] for m in umid])
    cash, expo, fees = (wmp[c].to_numpy() for c in ("cash", "expo", "fees"))
    widx = wmp["wid"].to_numpy()
    half = (wmp["half"] == "H2").to_numpy().astype(int)
    bought = wmp["bought"].to_numpy()
    nm = np.ones(len(widx))
    key = widx.astype(np.int64) * 2 + half
    ukey, kinv = np.unique(key, return_inverse=True)
    B = np.bincount(kinv, bought)
    M = np.bincount(kinv, nm)
    elig = (M >= min_markets) & (B >= min_bought)
    kw, kh = ukey // 2, ukey % 2
    h1 = {w: i for i, (w, h, e_) in enumerate(zip(kw, kh, elig, strict=True)) if h == 0 and e_}
    h2 = {w: i for i, (w, h, e_) in enumerate(zip(kw, kh, elig, strict=True)) if h == 1 and e_}
    both = sorted(set(h1) & set(h2))
    i1 = np.array([h1[w] for w in both], dtype=int)
    i2 = np.array([h2[w] for w in both], dtype=int)
    plc, plc_t2 = [], []
    for _ in range(sims):
        y = (rng.random(len(pv)) < pv).astype(float)
        P = np.bincount(kinv, cash + y[inv] * expo - fees)
        plc.append(_spearman(P[i1] / (B[i1] + K), P[i2] / B[i2]))
    out["placebo_spearman_mean"] = float(np.mean(plc))
    out["placebo_spearman_p95"] = float(np.percentile(plc, 95))
    out["placebo_spearman_p05"] = float(np.percentile(plc, 5))
    out["placebo_wallets"] = len(both)
    out["placebo_sims"] = sims
    out["actual_spearman_same_wallets"] = None
    if len(both):
        P = np.bincount(kinv, wmp["pnl"].to_numpy())
        out["actual_spearman_same_wallets"] = _spearman(P[i1] / (B[i1] + K), P[i2] / B[i2])
        out["placebo_rank_of_actual"] = float(np.mean(np.array(plc) < out["actual_spearman_same_wallets"]))
    # share of wallets with a t-statistic above 2 over their markets (>= 20 markets), actual vs placebo
    wmt = wmp.filter(pl.col("bought") > 0)
    r = (wmt["pnl"] / wmt["bought"]).to_numpy()
    wid_t = wmt["wid"].to_numpy()
    _uw, winv = np.unique(wid_t, return_inverse=True)
    cnt = np.bincount(winv)
    sel = cnt >= 20

    def frac_t2(rr: np.ndarray) -> float:
        s1, s2 = np.bincount(winv, rr), np.bincount(winv, rr * rr)
        mu = s1 / cnt
        sd = np.sqrt(np.maximum(s2 / cnt - mu * mu, 1e-12) * cnt / np.maximum(cnt - 1, 1))
        t = mu / (sd / np.sqrt(cnt))
        return float(np.mean(t[sel] > 2))

    out["wallets_20plus_markets"] = int(sel.sum())
    out["share_t_gt_2_actual"] = frac_t2(r)
    inv_t = np.searchsorted(umid, wmt["mid"].to_numpy())
    c_t, e_t, f_t, b_t = (wmt[c].to_numpy() for c in ("cash", "expo", "fees", "bought"))
    for _ in range(min(sims, 100)):
        y = (rng.random(len(pv)) < pv).astype(float)
        plc_t2.append(frac_t2((c_t + y[inv_t] * e_t - f_t) / b_t))
    out["share_t_gt_2_placebo_mean"] = float(np.mean(plc_t2))
    out["share_t_gt_2_placebo_p95"] = float(np.percentile(plc_t2, 95))
    qrows += [{"series": "placebo", "quintile": "spearman", "wallets": len(both), "roi_b_dollar": float(v),
               "roi_b_mean": None, "roi_b_median": None, "shr_a_mean": None} for v in plc]
    return out, pl.DataFrame(qrows, infer_schema_length=None)


# Salient events, fixed in advance as the most widely covered dated events in the window (not chosen
# from the data): the 2024 US presidential election, Super Bowls LIX and LX, the 2025 papal conclave
# result, and the 2025 New York City mayoral election.
SALIENT = {"US election 2024": "2024-11-05", "Super Bowl LIX": "2025-02-09", "Papal conclave 2025": "2025-05-08",
           "NYC mayor 2025": "2025-11-04", "Super Bowl LX": "2026-02-08"}


def dynamics(b: pl.DataFrame, users: pl.DataFrame | None) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    """New-account inflow (join dates of sampled wallets), new-wallet buying by month, event windows."""
    out: dict[str, Any] = {}
    tabs: dict[str, pl.DataFrame] = {}
    if users is None:
        return out, tabs
    u = users.filter(pl.col("join_date").is_not_null()).with_columns(
        pl.from_epoch("join_date", time_unit="s").dt.replace_time_zone("UTC").alias("joined"))
    out["users_with_stats"] = users.height
    out["users_known"] = int(users["known"].sum())
    out["users_with_join_date"] = u.height
    weekly = (u.group_by(pl.col("joined").dt.truncate("1w").alias("week")).len().sort("week")
              .filter(pl.col("week") >= datetime(2023, 1, 1, tzinfo=UTC)))
    tabs["inflow_weekly"] = weekly
    daily = u.group_by(pl.col("joined").dt.date().alias("day")).len()
    dmap = dict(zip(daily["day"].to_list(), daily["len"].to_list(), strict=True))
    ev_rows = []
    for name, ds in SALIENT.items():
        e = datetime.fromisoformat(ds).date()
        win = sum(dmap.get(e + timedelta(days=k), 0) for k in range(-7, 7))
        base = sum(dmap.get(e + timedelta(days=k), 0) for k in range(-56, -28))
        rate_w, rate_b = win / 14, base / 28
        ratio = rate_w / rate_b if rate_b else None
        se_log = math.sqrt(1 / win + 1 / base) if win and base else None
        ev_rows.append({"event": name, "date": ds, "joins_window": win, "joins_baseline": base,
                        "ratio": ratio, "ci_lo": ratio * math.exp(-1.96 * se_log) if ratio and se_log else None,
                        "ci_hi": ratio * math.exp(1.96 * se_log) if ratio and se_log else None})
    tabs["inflow_events"] = pl.DataFrame(ev_rows)
    # monthly: share of buying by accounts < 30 days old, and longshot share of new vs older accounts
    bb = b.filter(pl.col("age_days").is_not_null()).with_columns(
        pl.from_epoch("ts", time_unit="s").dt.strftime("%Y-%m").alias("m"),
        (pl.col("age_days") < 30).alias("new"), (pl.col("q") <= 0.10).cast(pl.Float64).alias("is_ls"))
    rows = []
    for m in sorted(bb["m"].unique().to_list()):
        s = bb.filter(pl.col("m") == m)
        tot = float(s["w_usd"].sum())
        new, old = s.filter(pl.col("new")), s.filter(~pl.col("new"))

        def fshare(x: pl.DataFrame) -> float | None:     # longshot share of design-weighted buy fills
            return float((x["is_ls"] * x["weight"]).sum() / x["weight"].sum()) if x.height >= 30 else None
        rows.append({"month": m, "fills": s.height, "new_fills": new.height,
                     "new_usd_share": float(new["w_usd"].sum()) / tot if tot else None,
                     "new_wallets": new["wid"].n_unique(), "wallets": s["wid"].n_unique(),
                     "ls_share_new": fshare(new), "ls_share_old": fshare(old)})
    tabs["newwallet_monthly"] = pl.DataFrame(rows, infer_schema_length=None)
    # new-vs-old longshot share difference, pooled, clustered by event (dollar-weighted)
    d = bb.with_columns(pl.col("new").cast(pl.Float64))
    ls_new = _cell(d.filter(pl.col("new") == 1), "is_ls", "w_usd")
    ls_old = _cell(d.filter(pl.col("new") == 0), "is_ls", "w_usd")
    out["ls_share_new_lt30d"] = ls_new
    out["ls_share_old_ge30d"] = ls_old
    # fill-level linear probability model of the longshot indicator on the new-account dummy
    out["ls_new_minus_old_ols"] = ols_cl(d["is_ls"].to_numpy(), d["new"].to_numpy()[:, None], d["ev"].to_numpy(),
                                         d["weight"].to_numpy())
    out["_note"] = ("ls_new_minus_old_ols: fill-level linear probability model of is_ls on new (<30d), design-weighted "
                    "(fill-weighted, like the age shares), clustered by event")
    return out, tabs


def herding(b: pl.DataFrame, px: pl.DataFrame, mk: pl.DataFrame) -> tuple[dict[str, Any], pl.DataFrame]:
    """Same-side clustering of small buys in 1-hour windows (LSV) and whether it predicts reversals."""
    out: dict[str, Any] = {}
    rows = []
    groups = {"small retail": (pl.col("cls").is_in(["Retail", "Active retail"])) & (pl.col("notional") <= 100),
              "whale": pl.col("cls") == "Whale", "MM": pl.col("cls") == "MM"}
    win_tabs = {}
    for name, cond in groups.items():
        s = b.filter(cond & pl.col("taker")).with_columns((pl.col("ts") // 3600).alias("h"),
                                                         (pl.col("outcome_index") == 0).cast(pl.Float64).alias("yes"))
        pbar = s.group_by("mid").agg(pl.col("yes").mean().alias("pbar"), pl.len().alias("n_m"))
        w = (s.group_by("mid", "h").agg(pl.col("yes").sum().alias("k"), pl.len().alias("n"), pl.col("ev").first(),
                                        (pl.col("notional") * (2 * pl.col("yes") - 1)).sum().alias("imb_usd"),
                                        pl.col("notional").sum().alias("usd"))
             .join(pbar, on="mid").filter((pl.col("n") >= 5) & (pl.col("pbar") > 0) & (pl.col("pbar") < 1)))
        if w.height == 0:
            continue
        hv = lsv_herding(w["k"].to_numpy(), w["n"].to_numpy(), w["pbar"].to_numpy())
        w = w.with_columns(pl.Series("H", hv), (pl.col("imb_usd") / pl.col("usd")).alias("imb"))
        cell = _cl(hv, w["ev"].to_numpy())
        out[f"lsv_{name}"] = {**cell, "windows": w.height}
        rows.append({"group": name, **cell, "windows": w.height})
        win_tabs[name] = w
    # reversal: future YES-price change after small-retail imbalance, controlling for the past change
    w = win_tabs.get("small retail")
    if w is not None:
        t_end = (w["h"] + 1) * 3600
        grid = w.select("mid", "ev", "imb", pl.Series("t", t_end.to_numpy()))
        mt = mk.select("mid", "t_info_s", (pl.col("winner_index") == 0).cast(pl.Float64).alias("y0"))
        pxs = px.sort("ts")

        def p_at(offset: int) -> pl.Series:
            g = grid.with_columns((pl.col("t") + offset).alias("tq")).with_row_index("i").sort("tq")
            j = g.join_asof(pxs.rename({"ts": "tq"}), on="tq", by="mid", strategy="backward",
                             check_sortedness=False).sort("i")
            return j["p"]

        grid = grid.with_columns(p_at(0).alias("p0"), p_at(-3600).alias("pm1"),
                                 *[p_at(k * 3600).alias(f"p{k}") for k in (1, 6, 24)]).join(mt, on="mid")
        for k in (1, 6, 24):
            g = grid.filter(pl.col("t") + k * 3600 <= pl.col("t_info_s"))
            y = (g[f"p{k}"] - g["p0"]).to_numpy()
            X = np.column_stack([g["imb"].to_numpy(), (g["p0"] - g["pm1"]).to_numpy()])
            out[f"reversal_{k}h"] = ols_cl(y, X, g["ev"].to_numpy())
        y = (grid["y0"] - grid["p0"]).to_numpy()
        out["outcome_minus_price_on_imb"] = ols_cl(y, grid["imb"].to_numpy()[:, None], grid["ev"].to_numpy())
        out["_reversal_note"] = ("beta[1] = imbalance (+1 all YES-side small buys); beta[2] = past 1h price change; "
                                 "dependent = YES price change over the next k hours (pre-information only)")
    return out, pl.DataFrame(rows, infer_schema_length=None)


# ---- cross-venue (Kalshi vs Polymarket intl on matched games) ---------------------------------------
LEADS_X = (1, 3, 6, 12, 24)


def run_xvenue_fetch() -> None:
    """Polymarket hourly price history (data-api) for the matched outcome token of every matched game."""
    pm = clients()
    s = settings()
    mk = pl.read_parquet(s.tables_dir / "match_kalshi_pm.parquet")
    pmm = pl.read_parquet(s.tables_dir / "pm_markets.parquet").select(
        "condition_id", "token_ids", pl.coalesce("game_start_time", "closed_time").alias("t0"))
    toks = mk.join(pmm, left_on="pm_condition_id", right_on="condition_id").with_columns(
        pl.col("token_ids").list.get(pl.col("pm_outcome_index")).alias("token"))
    rows = []
    for r in toks.iter_rows(named=True):
        page = pm.data.get("/prices-history", {"token_id": r["token"], "interval": "max", "bucket_seconds": 3600})
        pts = price_points(page.payload)
        if not pts and r["t0"] is not None:
            # interval=max serves nothing for some older tokens; an explicit window (<= 15 days) does
            g = int(r["t0"].timestamp())
            page = pm.data.get("/prices-history", {"token_id": r["token"], "start": g - 3 * 86400, "end": g + 86400,
                                                   "bucket_seconds": 3600})
            pts = price_points(page.payload)
        rows += [{"kalshi_ticker": r["kalshi_ticker"], "token": r["token"], "ts": t, "p": p} for t, p in pts]
    f = pl.DataFrame(rows, schema={"kalshi_ticker": pl.String, "token": pl.String, "ts": pl.Int64, "p": pl.Float64})
    _require(f, "xvenue prices")
    f.write_parquet(TAB / "xvenue_pm_prices.parquet")
    # fee schedules of the matched Polymarket markets (Gamma /markets by condition id)
    cids = mk["pm_condition_id"].unique().sort().to_list()
    fees = []
    for i in range(0, len(cids), 50):
        page = pm.gamma.get("/markets", {"condition_ids": cids[i:i + 50], "limit": 100, "closed": "true"})
        for m in page.payload:
            sched = m.get("feeSchedule") or {}
            fees.append({"condition_id": m.get("conditionId"), "fees_enabled": bool(m.get("feesEnabled")),
                         "fee_rate": float(sched.get("rate") or 0.0) if m.get("feesEnabled") else 0.0,
                         "game_start_time": _parse_loose(m.get("gameStartTime"))})
    pl.DataFrame(fees, infer_schema_length=None).write_parquet(TAB / "xvenue_pm_fees.parquet")
    print(f"xvenue: {f['kalshi_ticker'].n_unique()} tickers with Polymarket prices, {f.height} points")


def xvenue() -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    from albert.research.flb_common import BUCKET_LABELS, bucket_expr
    s = settings()
    mk = pl.read_parquet(s.tables_dir / "match_kalshi_pm.parquet")
    km = pl.scan_parquet(s.tables_dir / "kalshi_markets.parquet").filter(
        pl.col("ticker").is_in(mk["kalshi_ticker"].to_list())).select("ticker", "result", "fee_multiplier").collect()
    kc = (pl.scan_parquet(s.tables_dir / "kalshi_candles.parquet").filter(pl.col("ticker").is_in(mk["kalshi_ticker"].to_list()))
          .filter(pl.col("period_minutes") == 60).select("ticker", "end_ts", "yes_bid_close", "yes_ask_close", "price_close").collect())
    pmw = pl.read_parquet(s.tables_dir / "pm_markets.parquet").select("condition_id", "winner_index", "game_start_time")
    fees = pl.read_parquet(TAB / "xvenue_pm_fees.parquet")
    pp = pl.read_parquet(TAB / "xvenue_pm_prices.parquet")
    base = (mk.join(km, left_on="kalshi_ticker", right_on="ticker")
            .join(pmw, left_on="pm_condition_id", right_on="condition_id")
            .join(fees.select("condition_id", "fee_rate", pl.col("game_start_time").alias("gst2")), left_on="pm_condition_id",
                  right_on="condition_id", how="left")
            .with_columns(pl.coalesce("game_start_time", "gst2").alias("t0"),
                          (pl.col("result") == "yes").cast(pl.Float64).alias("y_k"),
                          (pl.col("winner_index") == pl.col("pm_outcome_index")).cast(pl.Float64).alias("y_pm"))
            .filter(pl.col("t0").is_not_null() & pl.col("result").is_in(["yes", "no"])))
    out: dict[str, Any] = {"pairs_matched": mk.height, "pairs_with_result_and_start": base.height,
                           "resolution_agreement": float((base["y_k"] == base["y_pm"]).mean())}
    rows = []
    for lead in LEADS_X:
        t = base.with_columns(((pl.col("t0").dt.epoch("s") // 3600) * 3600 - lead * 3600).alias("tq"))
        kk = kc.with_columns(pl.col("end_ts").dt.epoch("s").alias("tq"))
        j = t.join(kk, left_on=["kalshi_ticker", "tq"], right_on=["ticker", "tq"], how="inner")
        j = j.join(pp.select("kalshi_ticker", pl.col("ts").alias("tq"), pl.col("p").alias("p_pm")), on=["kalshi_ticker", "tq"])
        rows.append(j.with_columns(pl.lit(lead).alias("lead_h")))
    x = pl.concat(rows, how="diagonal_relaxed").filter(
        pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null() & (pl.col("yes_ask_close") > pl.col("yes_bid_close")))
    x = x.with_columns(((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("p_k"),
                       ((pl.col("yes_ask_close") - pl.col("yes_bid_close")) / 2).alias("half_spread_k"),
                       pl.col("pm_condition_id").rank("dense").alias("ev")).with_columns(
        (pl.col("p_pm") - pl.col("p_k")).alias("gap"), ((pl.col("p_pm") + pl.col("p_k")) / 2).alias("q_avg"))
    mult = pl.col("fee_multiplier").fill_null(1.0)
    x = x.with_columns(bucket_expr("q_avg"),
                       (0.07 * mult * pl.col("p_k") * (1 - pl.col("p_k"))).alias("fee_k"),
                       (pl.col("fee_rate").fill_null(0.0) * pl.col("p_pm") * (1 - pl.col("p_pm"))).alias("fee_pm"),
                       (0.0695 * pl.col("p_pm") * (1 - pl.col("p_pm"))).alias("fee_pmus_equiv")).with_columns(
        (pl.col("gap").abs() > pl.col("fee_k") + pl.col("fee_pm")).cast(pl.Float64).alias("exceeds_fees"),
        (pl.col("gap").abs() > pl.col("fee_k") + pl.col("fee_pm") + pl.col("half_spread_k")).cast(pl.Float64).alias("exceeds_fees_spread"))
    x.write_parquet(TAB / "xvenue_obs.parquet")
    gap_rows, cal_rows = [], []
    for lead in LEADS_X:
        xl = x.filter(pl.col("lead_h") == lead)
        out[f"lead_{lead}h"] = {"pairs": xl.height, "games": xl["ev"].n_unique(),
                                "mean_abs_gap": float(xl["gap"].abs().mean()) if xl.height else None,
                                "share_exceeds_fees": float(xl["exceeds_fees"].mean()) if xl.height else None,
                                "share_exceeds_fees_spread": float(xl["exceeds_fees_spread"].mean()) if xl.height else None,
                                "corr_p": float(np.corrcoef(xl["p_pm"], xl["p_k"])[0, 1]) if xl.height > 2 else None}
        for lab in (*BUCKET_LABELS, "LS", "FAV"):
            c = (xl.filter(pl.col("q_avg") <= 0.10) if lab == "LS" else xl.filter(pl.col("q_avg") >= 0.90) if lab == "FAV"
                 else xl.filter(pl.col("bucket") == lab))
            gap_rows.append({"lead_h": lead, "bucket": lab, **_cell(c, "gap", None),
                             "share_exceeds_fees": float(c["exceeds_fees"].mean()) if c.height else None,
                             "mean_fee_sum": float((c["fee_k"] + c["fee_pm"]).mean()) if c.height else None,
                             "mean_half_spread_k": float(c["half_spread_k"].mean()) if c.height else None})
        for venue, pcol, ycol in (("Kalshi", "p_k", "y_k"), ("Polymarket", "p_pm", "y_pm")):
            v = xl.with_columns(pl.col(pcol).alias("q"), pl.col(ycol).alias("won")).with_columns(
                bucket_expr("q"), ((pl.col("won") - pl.col("q")) / pl.col("q")).alias("ret"))
            sl = logit_slope(v["q"].to_numpy(), v["won"].to_numpy(), v["ev"].to_numpy())
            cal_rows.append({"lead_h": lead, "venue": venue, "bucket": "slope", **{k: sl[k] for k in ("n", "clusters")},
                             "mean": sl["b"], "se": sl["se_b"], "t": sl["t_b1"]})
            for lab, cond in (("q<=10c", pl.col("q") <= 0.10), ("10-50c", pl.col("q").is_between(0.10, 0.50, closed="right")),
                              ("50-90c", pl.col("q").is_between(0.50, 0.90, closed="right")), ("q>90c", pl.col("q") > 0.90)):
                cal_rows.append({"lead_h": lead, "venue": venue, "bucket": lab, **_cell(v.filter(cond), "ret", None)})
    # gap persistence: correlation of a pair's gap across leads, and hour-to-hour autocorrelation pre-game
    wide = x.select("kalshi_ticker", "lead_h", "gap").pivot(on="lead_h", index="kalshi_ticker", values="gap",
                                                             aggregate_function="first")
    if "24" in wide.columns and "1" in wide.columns:
        ww = wide.drop_nulls(["24", "1"])
        out["corr_gap_24h_1h"] = float(np.corrcoef(ww["24"], ww["1"])[0, 1]) if ww.height > 2 else None
        out["pairs_gap_24h_1h"] = ww.height
    hh = (base.select("kalshi_ticker", "t0").join(kc.with_columns(pl.col("end_ts").dt.epoch("s").alias("tq")),
                                                   left_on="kalshi_ticker", right_on="ticker")
          .join(pp.select("kalshi_ticker", pl.col("ts").alias("tq"), "p"), on=["kalshi_ticker", "tq"])
          .filter((pl.col("tq") < pl.col("t0").dt.epoch("s")) & (pl.col("tq") >= pl.col("t0").dt.epoch("s") - 48 * 3600))
          .filter(pl.col("yes_bid_close").is_not_null() & pl.col("yes_ask_close").is_not_null())
          .with_columns((pl.col("p") - (pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2).alias("gap"))
          .sort("kalshi_ticker", "tq").with_columns(pl.col("gap").shift(1).over("kalshi_ticker").alias("gap_l1"),
                                                     pl.col("tq").shift(1).over("kalshi_ticker").alias("tq_l1"))
          .filter((pl.col("tq") - pl.col("tq_l1")) == 3600))
    if hh.height > 10:
        out["gap_ar1_hourly"] = float(np.corrcoef(hh["gap"], hh["gap_l1"])[0, 1])
        out["gap_ar1_obs"] = hh.height
        rho = out["gap_ar1_hourly"]
        out["gap_half_life_h"] = float(math.log(0.5) / math.log(rho)) if 0 < rho < 1 else None
    return out, {"xvenue_gap": pl.DataFrame(gap_rows, infer_schema_length=None),
                 "xvenue_calib": pl.DataFrame(cal_rows, infer_schema_length=None)}


def fee_table() -> pl.DataFrame:
    """Taker fee per contract by venue at a few side prices (official schedules, checked 2026-11-11).

    Kalshi 0.07 q(1-q) (flb_common.fee_pc); Polymarket intl theta q(1-q) with theta 0.05 sports,
    0.04 politics/mentions/tech, 0.05 culture/weather/other, 0 geopolitics (docs.polymarket.com/trading/fees);
    Polymarket US theta q(1-q) with theta 0.095 at tier 1 (< $10k trailing 30-day taker volume) down to
    0.0695 at tier 4 (docs.polymarket.us/fees, effective 2026-10-28; albert.fees.pmus_fee uses tier 4).
    """
    from albert.fees import pmus_fee
    rows = []
    for q in (0.02, 0.05, 0.10, 0.25, 0.50):
        v = q * (1 - q)
        rows.append({"q": q, "kalshi_taker": 0.07 * v, "pm_intl_sports": 0.05 * v, "pm_intl_politics": 0.04 * v,
                     "pm_intl_geopolitics": 0.0, "pmus_tier1": 0.095 * v, "pmus_tier4": 0.0695 * v,
                     "pmus_tier4_100_contracts_rounded": pmus_fee(100, q) / 100})
    return pl.DataFrame(rows)


# ---- step: analyze ---------------------------------------------------------------------------------
ROBOTS = {
    "checked": "2026-11-11",
    "data-api.polymarket.com": "no robots.txt (/robots.txt -> HTTP 410, /v2/robots.txt -> 404): unrestricted; used",
    "gamma-api.polymarket.com": "no robots.txt (HTTP 404): unrestricted; used",
    "polymarket.com": "User-agent: * / Disallow: (allow all); links llms.txt for AI agents; not fetched",
    "docs.polymarket.com": "allow all, Content-Signal ai-input=yes; fee page read",
    "docs.polymarket.us": "allow all, Content-Signal ai-input=yes; fee schedule (albert.fees.pmus_fee)",
    "gateway.polymarket.us": "Disallow: / for anthropic-ai, ClaudeBot, Claude-Web: NOT used (no Polymarket US prices)",
}


def coverage(mk: pl.DataFrame, users: pl.DataFrame | None) -> dict[str, Any]:
    frame = (pl.scan_parquet(TAB / "frame.parquet").select("event_id", "group", "tier", "month", "volume", "fees_enabled")
             .with_columns(pl.col("event_id").hash()).collect())
    sample = pl.read_parquet(TAB / "sample.parquet", columns=["condition_id"])
    log = pl.read_ndjson(TAB / "trades_log.jsonl")
    fills = _fills().select(pl.len().alias("n"), pl.col("wid").n_unique().alias("w"), pl.col("taker").sum().alias("t")).collect()
    pre = (_fills().join(mk.lazy().select("mid", "t_info_s"), on="mid")
           .select((pl.col("ts") < pl.col("t_info_s")).mean().alias("pre")).collect(engine="streaming"))
    by = (frame.group_by("group").agg(pl.len().alias("frame_markets"), pl.col("volume").sum().alias("frame_volume"))
          .join(mk.group_by("group").agg(pl.len().alias("sample_markets"), pl.col("n_rows").sum().alias("rows")), on="group", how="left"))
    tiers = (frame.group_by("tier").agg(pl.len().alias("frame_markets"))
             .join(mk.group_by("tier").agg(pl.len().alias("sample_markets"), pl.col("n_rows").sum().alias("rows"),
                                           pl.col("truncated").sum().alias("truncated")), on="tier", how="left"))
    return {
        "window": [START.date().isoformat(), END.date().isoformat()], "min_market_volume_usd": MIN_VOLUME,
        "frame_markets": frame.height, "frame_events": frame["event_id"].n_unique(),
        "frame_volume_usd": float(frame["volume"].sum()),
        "strata_nonempty": frame.select(pl.struct("group", "tier", "month").n_unique()).item(),
        "sample_markets": sample.height, "markets_fetched": mk.height, "markets_truncated": int(mk["truncated"].sum()),
        "sample_events": mk["ev"].n_unique(), "requests_trades": int(log["pages"].sum()),
        "fills": int(fills["n"][0]), "taker_fills": int(fills["t"][0]), "wallets": int(fills["w"][0]),
        "share_fills_pre_information": float(pre["pre"][0]),
        "markets_closed_early": int(mk["closed_early"].sum()), "sports_games_pre_game_cutoff": int(mk["is_game"].sum()),
        "users_fetched": users.height if users is not None else 0,
        "users_with_join_date": int(users["join_date"].is_not_null().sum()) if users is not None else 0,
        "by_group": by.sort("frame_markets", descending=True).to_dicts(), "by_tier": tiers.sort("tier").to_dicts(),
        "per_stratum": PER_STRATUM, "seed": SEED, "half_split": HALF_SPLIT.date().isoformat(),
        "frame_over_6m": {"markets": int((frame["volume"] > 6e6).sum()),
                          "volume_share": float(frame.filter(pl.col("volume") > 6e6)["volume"].sum() / frame["volume"].sum())},
        "fee_enabled_share_by_month": dict(frame.group_by("month").agg(pl.col("fees_enabled").mean()).sort("month").iter_rows()),
        "sports_share": {"markets": float((frame["group"] == "Sports").mean()),
                         "volume": float(frame.filter(pl.col("group") == "Sports")["volume"].sum() / frame["volume"].sum())},
    }


def _pick(df: pl.DataFrame, **eq: Any) -> dict[str, Any] | None:
    s = df
    for k, v in eq.items():
        s = s.filter(pl.col(k) == v)
    return s.row(0, named=True) if s.height else None


def _w(name: str, df: pl.DataFrame) -> None:
    df.write_csv(OUT / f"{name}.csv")


def run_analyze() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    mk = pl.read_parquet(TAB / "markets.parquet")
    users = _users()
    summary: dict[str, Any] = {"generated_at": datetime.now(UTC).isoformat(), "robots_txt": ROBOTS,
                               "coverage": coverage(mk, users), "thresholds_default": DEFAULT_TH}
    print("coverage done", flush=True)
    px = yes_prices(mk)
    cal_s, cal, slopes = calibration(mk, px)
    _w("calib_market", cal)
    _w("calib_slopes", slopes)
    summary["calibration"] = {**cal_s, "slopes": slopes.filter(pl.col("scope").is_in(["All", "All H1", "All H2", *GROUPS_ORDER])).to_dicts(),
                              "LS_FAV": cal.filter(pl.col("bucket").is_in(["LS", "FAV"]) & pl.col("scope").is_in(["All", "All H1", "All H2"])).to_dicts()}
    print("calibration done", flush=True)
    p_close = px.group_by("mid").agg(pl.col("p").sort_by("ts").last().alias("p_close"))
    b = buys_frame(mk, users)
    flb = trade_flb(b)
    _w("trade_flb", flb)
    summary["trade_flb"] = flb.filter(pl.col("bucket").is_in(["LS", "FAV", "1-2c", "3-5c", "6-10c", "99c"])
                                      & pl.col("scope").is_in(["All", "Taker", "Maker", "All H1", "All H2"])).to_dicts()
    wb = who_buys(b)
    for k, v in wb.items():
        _w(f"who_{k}", v)
    summary["who_buys"] = {"share": wb["share"].filter(pl.col("half") == "All").to_dicts(),
                           "variants": wb["variants"].to_dicts()}
    print("who-buys done", flush=True)
    flow = ls_flow(mk)
    _w("ls_flow", flow)
    summary["ls_flow"] = flow.to_dicts()
    he = hold_exit(mk)
    _w("hold_exit", he)
    summary["hold_exit"] = he.to_dicts()
    print("flow / hold-exit done", flush=True)
    wm = wallet_markets(mk)
    conc, lz, cls = concentration(wm)
    _w("lorenz", lz)
    _w("class_pnl", cls)
    summary["concentration"] = conc
    summary["class_pnl"] = cls.to_dicts()
    pers, pq = persistence(wm, b, p_close)
    _w("persistence", pq)
    summary["persistence"] = pers
    print("concentration / persistence done", flush=True)
    del wm
    dyn, dtabs = dynamics(b, users)
    for k, v in dtabs.items():
        _w(k, v)
    summary["dynamics"] = dyn
    herd, hd = herding(b, px, mk)
    _w("herding", hd)
    summary["herding"] = herd
    print("dynamics / herding done", flush=True)
    if (TAB / "xvenue_pm_prices.parquet").exists():
        xv, xt = xvenue()
        for k, v in xt.items():
            _w(k, v)
        summary["xvenue"] = xv
    _w("fees", fee_table())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (OUT / "verdict.md").write_text(verdict(summary))
    print("analyze done", flush=True)
    return summary


def _fmt(c: dict[str, Any] | None, key: str = "mean", pct: bool = True) -> str:
    if not c or c.get(key) is None:
        return "n/a"
    v = c[key] * (100 if pct else 1)
    return f"{v:+.1f}{'%' if pct else ''} (t {c['t']:.1f})" if c.get("t") is not None else f"{v:+.1f}"


def verdict(s: dict[str, Any]) -> str:
    cov = s["coverage"]
    lines = ["# Wallets workstream (flb_wallets): verdict", "",
             f"Generated {s['generated_at'][:16]} UTC from `data/research/flb_wallets/` (see summary.json).", "",
             "## Data",
             (f"- Frame: {cov['frame_markets']:,} resolved in-scope Polymarket markets (>= $1k volume), "
             f"{cov['window'][0]} to {cov['window'][1]}; stratified sample of {cov['sample_markets']:,} markets "
             f"({cov['markets_truncated']} truncated at {MAX_TRADE_PAGES} pages)."),
             (f"- {cov['fills']:,} fills ({cov['taker_fills']:,} taker) by {cov['wallets']:,} wallets; "
             f"{cov['users_with_join_date']:,} wallets with an account creation date."),
             ("- robots.txt: data-api / gamma-api serve none (allowed); gateway.polymarket.us disallows Anthropic "
             "agents, so Polymarket US prices are not used."), "", "## Findings"]
    sl = next((r for r in s["calibration"]["slopes"] if r["scope"] == "All" and r["lead_h"] == 24), None)
    if sl and sl.get("b") is not None:
        lines.append(f"- Polymarket calibration slope (24h before the information time): b = {sl['b']:.3f} "
                     f"(SE {sl['se_b']:.3f}; t vs 1 = {sl['t_b1']:.1f}; {sl['markets']} markets).")
    ls = next((r for r in s["trade_flb"] if r["scope"] == "All" and r["bucket"] == "LS" and r["ret"] == "ret_n"
               and r["weighting"] == "dollar"), None)
    fav = next((r for r in s["trade_flb"] if r["scope"] == "All" and r["bucket"] == "FAV" and r["ret"] == "ret_n"
                and r["weighting"] == "dollar"), None)
    lines.append(f"- Buys at <= 10c return {_fmt(ls)} per dollar (net, dollar-weighted); buys at >= 90c {_fmt(fav)}.")
    share = {r["label"]: r for r in s["who_buys"]["share"] if r["dim"] == "cls" and r.get("stat") == "is_ls"
             and r.get("weighting") == "fill"}
    lines.append("- Longshot share of buy fills by class: " + ", ".join(
        f"{c} {share[c]['mean'] * 100:.1f}%" for c in CLASSES if c in share and share[c].get("mean") is not None) + ".")
    flow = {r["cls"]: r for r in s["ls_flow"] if r["half"] == "All" and r["role"] == "all"}
    lines.append("- Net longshot flow per $ traded (+ = net buyer): " + ", ".join(
        f"{c} {flow[c]['per_usd_mean'] * 100:+.2f}% (t {flow[c]['per_usd_t']:.1f})" for c in CLASSES if c in flow) + ".")
    conc = s["concentration"]
    lines.append(f"- {conc['share_profitable'] * 100:.1f}% of {conc['wallets']:,} wallets profit; the top 1% of wallets "
                 f"take {conc['top1pct_wallets_share_of_gains'] * 100:.0f}% of all gains (Gini of gains "
                 f"{conc['gini_gains_among_winners']:.2f}).")
    p = s["persistence"]
    if "spearman_shr_a_roi_b" in p:
        lines.append(f"- Persistence: Spearman(H1 shrunk ROI, H2 ROI) = {p['spearman_shr_a_roi_b']:.3f} "
                     f"(95% CI {p['spearman_shr_a_ci95'][0]:.3f} to {p['spearman_shr_a_ci95'][1]:.3f}; "
                     f"{p['wallets_both_halves']} wallets); luck placebo mean {p['placebo_spearman_mean']:.3f}, "
                     f"95th pct {p['placebo_spearman_p95']:.3f}.")
    h = s.get("herding", {}).get("lsv_small retail")
    if h:
        lines.append(f"- Small-retail herding (LSV, 1h windows): {h['mean']:.3f} (t {h['t']:.1f}); it does not predict "
                     "reversals or outcomes (summary.json:herding).")
    d = s.get("dynamics", {})
    if d.get("ls_share_new_lt30d"):
        lines.append(f"- Longshot share of dollars bought: accounts <30 days {d['ls_share_new_lt30d']['mean'] * 100:.1f}%, "
                     f"older {d['ls_share_old_ge30d']['mean'] * 100:.1f}% ({d['users_with_join_date']} dated wallets).")
    if "xvenue" in s:
        x = s["xvenue"].get("lead_1h", {})
        lines.append(f"- Cross-venue (Kalshi vs Polymarket, 1h pre-game): {x.get('pairs')} team-sides, mean |gap| "
                     f"{(x.get('mean_abs_gap') or 0) * 100:.2f}c; {(x.get('share_exceeds_fees') or 0) * 100:.0f}% of gaps exceed "
                     "the two venues' taker fees.")
    lines += ["", "Descriptive results only; no trading recommendation."]
    return "\n".join(lines) + "\n"


# ---- CLI -------------------------------------------------------------------------------------------
STEPS = {"frame": run_frame, "sample": run_sample, "trades": run_trades, "users": run_users, "build": run_build,
         "xvenue": run_xvenue_fetch, "analyze": run_analyze}


def main(argv: list[str] | None = None) -> None:
    """``run <step> [int]``: the optional integer is the step's first argument (e.g. wallets for ``users``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] != "run" or args[1] not in {*STEPS, "all"}:
        raise SystemExit(f"usage: python -m albert.research.flb_wallets run <{'|'.join(STEPS)}|all> [n]")
    extra = [int(a) for a in args[2:]]
    for step in (list(STEPS) if args[1] == "all" else [args[1]]):
        print(f"== {step}", flush=True)
        STEPS[step](*extra)


if __name__ == "__main__":
    main()
