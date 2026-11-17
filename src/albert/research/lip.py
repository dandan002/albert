"""Kalshi Liquidity Incentive Program (LIP): can a $100 maker earn the rewards after adverse selection?

Terms (verified 2026-11-17 on help.kalshi.com, whose robots.txt allows ``User-agent: *``; kalshi.com
itself sits behind a Vercel challenge, so its robots.txt and the regulatory-notices page with the
formal terms could not be read):

* https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program: ends 2027-01-01 (can be
  ended or changed at any time); most U.S. members eligible, no opt-in mentioned (the combo-legs
  program says "no opt-in is required"); a book snapshot once per second at a random moment.
  Per side (YES bids, NO bids): the **Reference Price** is the first level, walking down from the
  best bid, at which cumulative resting size reaches 1/5 of the **Target Size**; every resting order
  that "helps reach the Target Size" on its side qualifies; score = size x 1.0 at or better than the
  reference price, size x DF^ticks below it; shares are normalised per side and a snapshot is worth
  share_yes + share_no (at most 2.0 over all participants). A snapshot is **excluded** unless both
  sides hold at least Target Size of resting orders (and the market is open). Reward = period score
  share x pool x valid/total snapshots, rounded down to the cent; **under $1.00 per market program
  is not paid** and sub-dollar programs do not combine. Pools $1-$1,000 per market per day, periods
  up to 31 days; paid by a daily run after the period closes.
* https://help.kalshi.com/en/articles/16076644 (where to find them) and docs.kalshi.com (robots:
  ``User-agent: *``, ``ai-input=yes``): ``GET /incentive_programs`` (public; ``period_reward`` and
  ``max_reward_per_account`` in centi-cents, ``discount_factor_bps``, ``target_size_fp``) and
  ``GET /incentive_schedules`` (recurring series schedules).

So the payout per second of a period is ``pool / period_seconds x (s_yes + s_no) / 2`` on valid
seconds, and 0 otherwise. Which orders at the level where cumulative size crosses Target Size
"help reach" it is not stated: ``mode="level"`` counts the whole crossing level (matches the
practitioner example in .planning/research/market-making.md), ``mode="truncate"`` counts only the
part needed to reach Target Size, pro rata.

Pipeline (outputs in ``data/research/lip/``, report ``report.md``)::

    python -m albert.research.lip programs     # liquidity programs + schedules
    python -m albert.research.lip poll 12 15   # batch order-book cross-sections of active programs
    python -m albert.research.lip cross        # competition census, static $100 allocations, persistence
    python -m albert.research.lip cross_fills  # public prints at those allocations' prices
    python -m albert.research.lip replay       # recorded L2 (albert record sample), Simulator fills
    python -m albert.research.lip replay_exit  # same, selling every fill straight back (exit/)
    python -m albert.research.lip reactions    # how fast large top-of-window orders are matched
    python -m albert.research.lip finalize     # refresh settlements, write report.md + ledgers

Result (2026-11-17, see report.md): every variant loses after fills, even with rewards that assume
no competitor ever reacts; the cheap (longshot) side is the only place a $100 book earns a reward
share, and its fills are informed (they settle at ~0).
"""

from __future__ import annotations

import heapq
import json
import math
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

OUT = Path("data/research/lip")
TICK = 0.01
EPS = 1e-9
BANKROLL = 100.0
MIN_PAYOUT = 1.00

PROGRAM_SCHEMA = {
    "id": pl.String, "market_ticker": pl.String, "series_ticker": pl.String, "event_ticker": pl.String,
    "incentive_type": pl.String, "incentive_description": pl.String,
    "start": pl.Datetime("us", "UTC"), "end": pl.Datetime("us", "UTC"),
    "pool_usd": pl.Float64, "discount_factor": pl.Float64, "target_size": pl.Float64,
    "max_reward_usd": pl.Float64, "paid_out": pl.Boolean,
}


# ---- parsers (tests/fixtures/kalshi/lip_*.json) -------------------------------------------------
def _ts(v: Any) -> datetime | None:
    if not v:
        return None
    return datetime.fromisoformat(str(v)).astimezone(UTC)


def program_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """``GET /incentive_programs`` -> one row per program. Money fields are centi-cents."""
    out = []
    for p in payload.get("incentive_programs") or []:
        df = p.get("discount_factor_bps")
        ts = p.get("target_size_fp")
        cap = p.get("max_reward_per_account")
        out.append({
            "id": p["id"], "market_ticker": p["market_ticker"], "series_ticker": p.get("series_ticker"),
            "event_ticker": p.get("event_ticker") or None, "incentive_type": p.get("incentive_type"),
            "incentive_description": p.get("incentive_description") or None,
            "start": _ts(p.get("start_date")), "end": _ts(p.get("end_date")),
            "pool_usd": p["period_reward"] / 1e4, "discount_factor": None if df is None else df / 1e4,
            "target_size": None if ts is None else float(ts), "max_reward_usd": None if cap is None else cap / 1e4,
            "paid_out": bool(p.get("paid_out")),
        })
    return out


def programs_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    df = pl.DataFrame(rows, schema=PROGRAM_SCHEMA).unique("id", keep="first")
    return df.with_columns(
        ((pl.col("end") - pl.col("start")).dt.total_seconds() / 86400).alias("days"),
    ).with_columns((pl.col("pool_usd") / pl.col("days")).alias("pool_per_day"))


def schedule_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """``GET /incentive_schedules`` -> one row per recurring schedule."""
    out = []
    for s in payload.get("schedules") or []:
        df = s.get("discount_factor_bps")
        ts = s.get("target_size_fp")
        out.append({
            "series_ticker": s["series_ticker"], "series_category": s.get("series_category"),
            "event_ticker": s.get("event_ticker") or None, "incentive_type": s.get("incentive_type"),
            "source": s.get("source"), "reward_usd": s["reward"] / 1e4, "window_days": s.get("window_days"),
            "discount_factor": None if df is None else df / 1e4, "target_size": None if ts is None else float(ts),
            "ends": _ts(s.get("ends")), "all_markets": bool(s.get("all_markets")),
            "posted": _ts(s.get("posted_ts")), "updated": _ts(s.get("updated_ts")),
            "description": s.get("description"),
        })
    return out


def book_levels(payload: dict[str, Any]) -> Iterator[tuple[str, str, float, float]]:
    """``GET /markets/orderbooks`` (or one ``/markets/{t}/orderbook``) -> (ticker, side, price, size).

    Both sides are *bids*: a YES bid at p, a NO bid at q (= YES ask 1 - q)."""
    books = payload.get("orderbooks")
    if books is None:
        books = [payload]
    for b in books:
        t = b.get("ticker")
        ob = b.get("orderbook_fp") or b.get("orderbook") or {}
        for side in ("yes", "no"):
            for p, q in ob.get(f"{side}_dollars") or []:
                if float(q) > 0:
                    yield t, side, float(p), float(q)


# ---- scoring ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SideScore:
    depth: float          # all resting size on the side (incl. ours)
    ref: float | None     # reference price (None when the side never reaches target/5)
    total: float          # qualifying score of everyone
    ours: float           # our qualifying score
    qual_size: float      # qualifying size of everyone (unweighted)

    @property
    def share(self) -> float:
        return self.ours / self.total if self.total > EPS else 0.0


def score_side(others: dict[float, float], ours: dict[float, float] | None, target: float, df: float,
               mode: str = "level") -> SideScore:
    """Score one side of the book under the LIP rules.

    ``others``/``ours``: price -> resting size (bids). Our size sits at the back of its level (time
    priority), which only matters for ``mode="truncate"``. A side whose depth is below ``target``
    scores nothing (the whole snapshot is then excluded; see :func:`score_snapshot`)."""
    ours = ours or {}
    prices = sorted(set(others) | set(ours), reverse=True)
    depth = sum(others.values()) + sum(ours.values())
    ref, cum = None, 0.0
    for p in prices:
        cum += others.get(p, 0.0) + ours.get(p, 0.0)
        if cum >= target / 5 - EPS:
            ref = p
            break
    if depth < target - EPS or ref is None:
        return SideScore(depth, ref, 0.0, 0.0, 0.0)
    total = mine = qual = 0.0
    cum = 0.0
    for p in prices:
        o, u = others.get(p, 0.0), ours.get(p, 0.0)
        mult = 1.0 if p >= ref - EPS else df ** round((ref - p) / TICK)
        need = target - cum
        if mode == "truncate" and o + u > need + EPS:
            o_q = min(o, need)                 # others are ahead of us in the queue at this level
            u_q = min(u, need - o_q)
        else:
            o_q, u_q = o, u
        total += (o_q + u_q) * mult
        mine += u_q * mult
        qual += o_q + u_q
        cum += o + u
        if cum >= target - EPS:
            break
    return SideScore(depth, ref, total, mine, qual)


def score_snapshot(yes: dict[float, float], no: dict[float, float], target: float, df: float,
                   our_yes: dict[float, float] | None = None, our_no: dict[float, float] | None = None,
                   mode: str = "level") -> tuple[bool, float, float]:
    """(valid, share_yes, share_no) of one snapshot. Our orders count toward validity."""
    y = score_side(yes, our_yes, target, df, mode)
    n = score_side(no, our_no, target, df, mode)
    valid = y.depth >= target - EPS and n.depth >= target - EPS and y.total > 0 and n.total > 0
    return (valid, y.share, n.share) if valid else (False, 0.0, 0.0)


def ref_price(others: dict[float, float], target: float) -> float | None:
    """Reference price of the side from others' orders alone (None: they never reach target/5)."""
    cum = 0.0
    for p in sorted(others, reverse=True):
        cum += others[p]
        if cum >= target / 5 - EPS:
            return p
    return None


def full_credit_price(others: dict[float, float], opp_best: float | None, target: float) -> float | None:
    """Cheapest bid price that still scores 1.0x: the others' reference level, or 1c when they never
    reach target/5 (then our own size sets the reference). Never crosses the other side's best bid
    (post-only: price + opp_best < 1)."""
    p = ref_price(others, target)
    p = TICK if p is None else p
    if opp_best is not None and p + opp_best >= 1 - EPS:
        return None
    return round(p, 4)


def reward_rate(pool_usd: float, period_s: float, share_yes: float, share_no: float) -> float:
    """Dollars per valid second for the given side shares."""
    return pool_usd / period_s * (share_yes + share_no) / 2


def paid(amount: float) -> float:
    """Kalshi's payout of one market program: rounded down to the cent, nothing under $1."""
    a = math.floor(amount * 100 + 1e-9) / 100
    return a if a >= MIN_PAYOUT - EPS else 0.0


# ---- fetching -------------------------------------------------------------------------------------
def _api():
    from albert.config import settings
    from albert.research.pm_hazard import _api as hazard_api

    return hazard_api(settings(), min_interval=0.5)


def fetch_programs(api, statuses: Iterable[str] = ("active", "upcoming", "closed", "paid_out"),
                   max_pages: dict[str, int] | None = None) -> pl.DataFrame:
    """Liquidity programs by status (live requests, each raw response kept append-only). Paid-out
    programs come newest first; ``max_pages`` bounds that history (10,000 per page)."""
    max_pages = {"paid_out": 4, "closed": 3, **(max_pages or {})}
    rows: list[dict[str, Any]] = []
    for st in statuses:
        cursor, pages = None, 0
        while True:
            f = api.client.get("/incentive_programs", {"status": st, "type": "liquidity", "limit": 10000,
                                                       "cursor": cursor}, cache=False)
            got = program_rows(f.payload)
            rows += [{**r, "status": st} for r in got]
            pages += 1
            cursor = f.payload.get("next_cursor")
            if not cursor or not got or pages >= max_pages.get(st, 50):
                break
    df = programs_frame([{k: v for k, v in r.items() if k != "status"} for r in rows])
    status = {r["id"]: r["status"] for r in rows}
    return df.with_columns(pl.col("id").replace_strict(status, default=None).alias("status"))


def fetch_books(api, tickers: list[str], t: datetime | None = None) -> pl.DataFrame:
    """Full-depth books of ``tickers`` via the public batch endpoint (100 per request)."""
    rows = []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        f = api.client.get("/markets/orderbooks", {"tickers": chunk}, cache=False)
        t_i = f.fetched_at
        rows += [(t or t_i, *r) for r in book_levels(f.payload)]
    return pl.DataFrame(rows, schema={"t": pl.Datetime("us", "UTC"), "ticker": pl.String, "side": pl.String,
                                      "price": pl.Float64, "size": pl.Float64}, orient="row")


def cmd_programs() -> pl.DataFrame:
    api = _api()
    OUT.mkdir(parents=True, exist_ok=True)
    df = fetch_programs(api)
    df.write_parquet(OUT / "programs.parquet")
    f = api.client.get("/incentive_schedules", {}, cache=False)
    sch = pl.DataFrame(schedule_rows(f.payload), infer_schema_length=None)
    sch.write_parquet(OUT / "schedules.parquet")
    print(df.group_by("status").agg(pl.len(), pl.col("pool_per_day").sum()), sch.height, "schedules")
    return df


def cmd_poll(rounds: int, minutes: float) -> None:
    """``rounds`` cross-sections of every active program's book, ``minutes`` apart. Each round
    re-lists the active programs, so programs that start or end mid-run are handled."""
    api = _api()
    d = OUT / "polls"
    d.mkdir(parents=True, exist_ok=True)
    for r in range(rounds):
        t0 = time.time()
        stamp = datetime.now(UTC)
        progs = fetch_programs(api, statuses=("active",))
        tickers = sorted(progs["market_ticker"].unique().to_list())
        books = fetch_books(api, tickers, t=stamp)
        tag = stamp.strftime("%Y%m%dT%H%M%SZ")
        progs.with_columns(pl.lit(stamp).alias("t")).write_parquet(d / f"programs_{tag}.parquet")
        books.write_parquet(d / f"books_{tag}.parquet")
        print(f"poll {r + 1}/{rounds} {tag}: {len(tickers)} markets, {books.height} levels, "
              f"{time.time() - t0:.0f}s", flush=True)
        if r + 1 < rounds:
            time.sleep(max(0.0, minutes * 60 - (time.time() - t0)))


# ---- cross-section: what would a $100 farmer earn against the books as they stand? -------------
Q_GRID = (10, 25, 50, 100, 200, 400, 700, 1000, 1500, 2500)
WINDOW_DAYS = 7.0                      # horizon over which the $1 minimum is checked


def series_meta() -> dict[str, dict[str, Any]]:
    """series -> {category, title, fee_type, fee_multiplier, financial} from the cached series table."""
    from albert.categories import kalshi_series_is_financial

    ser = pl.read_parquet("data/tables/kalshi_series.parquet")
    out = {}
    for r in ser.iter_rows(named=True):
        out[r["series_ticker"]] = {**r, "financial": kalshi_series_is_financial(r["series_ticker"], r["category"], r["title"])}
    return out


def books_dict(books: pl.DataFrame) -> dict[str, dict[str, dict[float, float]]]:
    out: dict[str, dict[str, dict[float, float]]] = {}
    for t, s, p, q in books.select("ticker", "side", "price", "size").iter_rows():
        side = out.setdefault(t, {"yes": {}, "no": {}})[s]
        side[p] = side.get(p, 0.0) + q
    return out


def candidate_prices(others: dict[float, float], opp_best: float | None, target: float,
                     improve: bool = True) -> list[float]:
    """Bid prices worth trying on one side: 1c, and every resting level down to the one where the
    others' cumulative size reaches ``target`` (deeper orders never qualify) plus one tick above
    each; never crossing the other side (post-only), never above the best bid + one tick."""
    cands = {TICK}
    cum = 0.0
    for p in sorted(others, reverse=True):
        cands.add(round(p, 4))
        cands.add(round(p + TICK, 4))
        cum += others[p]
        if cum >= target - EPS:
            break
    best = max(others) if others else 0.0
    top = best + (TICK if improve else 0.0)
    return sorted(p for p in cands if TICK - EPS <= p <= top + EPS and p <= 0.99 + EPS
                  and (opp_best is None or p + opp_best < 1 - EPS))


def side_options(others: dict[float, float], opp_best: float | None, target: float, df: float,
                 side_pool_per_day: float, mode: str = "level", q_grid: Iterable[int] = Q_GRID) -> list[dict[str, Any]]:
    """Every (price, size) placement on one side with its share, cost and reward per day (the side's
    half of the pool times our share; validity of the *other* side is checked by the caller)."""
    best = max(others) if others else None
    out = []
    for p in candidate_prices(others, opp_best, target):
        for q in q_grid:
            sc = score_side(others, {p: float(q)}, target, df, mode)
            if sc.ours <= 0:
                continue
            out.append({"price": p, "q": q, "cost": q * p, "share": sc.share, "ref": sc.ref,
                        "rpd": side_pool_per_day * sc.share,
                        "dist": None if best is None else round((best - p) / TICK)})
    return out


def market_options(progs: pl.DataFrame, books: dict, meta: dict, mode: str = "level") -> pl.DataFrame:
    """Placement options for every non-financial active program market, both sides."""
    rows = []
    for r in progs.iter_rows(named=True):
        m = meta.get(r["series_ticker"]) or {}
        if m.get("financial", True):
            continue
        b = books.get(r["market_ticker"]) or {"yes": {}, "no": {}}
        ts, df = r["target_size"], r["discount_factor"]
        depth = {s: sum(b[s].values()) for s in ("yes", "no")}
        for side, opp in (("yes", "no"), ("no", "yes")):
            if depth[opp] < ts - EPS:          # the other side must already be valid
                continue
            opp_best = max(b[opp]) if b[opp] else None
            base = score_side(b[side], {}, ts, df, mode)
            for o in side_options(b[side], opp_best, ts, df, r["pool_per_day"] / 2, mode):
                rows.append({"ticker": r["market_ticker"], "series": r["series_ticker"], "event": r["event_ticker"],
                             "category": m.get("category"), "side": side, "pool_per_day": r["pool_per_day"],
                             "end": r["end"], "others_S": base.total, "others_depth": depth[side],
                             "others_ref": base.ref, "best": max(b[side]) if b[side] else None, "opp_best": opp_best, **o})
    return pl.DataFrame(rows, infer_schema_length=None)


def allocate(opts: pl.DataFrame, budget: float = BANKROLL, t: datetime | None = None,
             min_payout: float = MIN_PAYOUT, max_dist: int | None = None, min_dist: int | None = None) -> pl.DataFrame:
    """Multiple-choice knapsack by Lagrangian bisection: each (ticker, side) takes the option that
    maximises rpd - lam * cost; lam is the smallest rate whose picks fit the budget. Options whose
    payout over min(remaining period, a week) is under ``min_payout`` are dropped (a market's two
    sides share one program, so this is a per-side approximation that errs conservative)."""
    o = opts
    if max_dist is not None:
        o = o.filter(pl.col("dist") <= max_dist)
    if min_dist is not None:
        o = o.filter(pl.col("dist") >= min_dist)
    if t is not None:
        days = ((pl.col("end") - pl.lit(t)).dt.total_seconds() / 86400).clip(0, WINDOW_DAYS)
        o = o.with_columns(days.alias("days_left")).filter(pl.col("rpd") * pl.col("days_left") >= min_payout)
    if o.is_empty():
        return o

    def pick(lam: float) -> pl.DataFrame:
        x = o.with_columns((pl.col("rpd") - lam * pl.col("cost")).alias("v")).filter(pl.col("v") > 0)
        return x.sort("v", descending=True).unique(["ticker", "side"], keep="first")

    lo, hi = 0.0, float(o.select((pl.col("rpd") / pl.col("cost")).max()).item()) + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if pick(mid)["cost"].sum() > budget:
            lo = mid
        else:
            hi = mid
    return pick(hi).sort("rpd", descending=True)


def evaluate(alloc: pl.DataFrame, progs: pl.DataFrame, books: dict, mode: str = "level") -> pl.DataFrame:
    """Realised reward per day of fixed orders (``alloc``: ticker, side, price, q) against another
    book: our size is added to that side, validity needs both sides at target (ours included)."""
    pr = {r["market_ticker"]: r for r in progs.iter_rows(named=True)}
    rows = []
    for a in alloc.iter_rows(named=True):
        r = pr.get(a["ticker"])
        b = books.get(a["ticker"])
        if r is None or b is None:
            rows.append({"ticker": a["ticker"], "side": a["side"], "rpd_real": 0.0, "share_real": 0.0, "valid": False})
            continue
        ours = {a["side"]: {a["price"]: float(a["q"])}}
        opp = "no" if a["side"] == "yes" else "yes"
        crossed = bool(b[opp]) and a["price"] + max(b[opp]) >= 1 - EPS
        valid, sy, sn = score_snapshot(b["yes"], b["no"], r["target_size"], r["discount_factor"],
                                       ours.get("yes"), ours.get("no"), mode)
        share = sy if a["side"] == "yes" else sn
        rows.append({"ticker": a["ticker"], "side": a["side"], "share_real": share, "valid": valid,
                     "crossed": crossed, "rpd_real": r["pool_per_day"] / 2 * share if valid else 0.0})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- recorded L2 replay ---------------------------------------------------------------------------
REPLAY_START, REPLAY_END = "2026-11-07T00", None      # local stream days (11-05/06 are archived)
DECIDE_S = 300                                          # re-allocate every 5 minutes
MARKOUT_S = (60, 300, 3600)


@dataclass
class Variant:
    name: str
    min_dist: int | None = None       # ticks behind the side's best bid (0 = join, <0 = improve)
    ref_only: bool = False            # only at the others' reference price (never step ahead)
    budget: float = BANKROLL
    exit_fill: bool = False           # sell every fill straight back at the best bid (taker), up to its size


VARIANTS = (Variant("opt"), Variant("noimprove", min_dist=0), Variant("behind2", min_dist=2),
            Variant("ref", min_dist=0, ref_only=True))
EXIT_VARIANTS = (Variant("opt_x", exit_fill=True), Variant("noimprove_x", min_dist=0, exit_fill=True),
                 Variant("behind2_x", min_dist=2, exit_fill=True), Variant("ref_x", min_dist=0, ref_only=True, exit_fill=True))


def recorded_programs(s, progs: pl.DataFrame) -> pl.DataFrame:
    """Programs of markets in the ``albert record`` sample, clipped to each run's window."""
    runs = []
    for p in sorted((s.data_dir / "runs").glob("*-record.json")):
        m = json.loads(p.read_text())
        st = datetime.fromisoformat(str(m["params"]["started_at"]).replace(" ", "T")).astimezone(UTC)
        en = datetime.fromisoformat(m["created_at"]).astimezone(UTC)
        runs += [{"ticker": r["ticker"], "run_start": st, "run_end": en} for r in m["params"].get("kalshi_markets") or []]
    if not runs:
        return pl.DataFrame()
    r = pl.DataFrame(runs).with_columns(pl.col("run_start", "run_end").dt.cast_time_unit("us"))
    j = r.join(progs.rename({"market_ticker": "ticker"}), on="ticker")
    return j.filter((pl.col("end") > pl.col("run_start")) & (pl.col("start") < pl.col("run_end")))


def market_info(api, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """ticker -> result / status / close / settlement fields (public ``GET /markets?tickers``)."""
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        f = api.client.get("/markets", {"tickers": ",".join(chunk), "limit": 100}, cache=False)
        for m in f.payload.get("markets") or []:
            out[m["ticker"]] = {"result": m.get("result") or None, "status": m.get("status"),
                                "close_time": _ts(m.get("close_time")), "settlement_ts": _ts(m.get("settlement_ts")),
                                "event_ticker": m.get("event_ticker")}
    missing = set(tickers) - set(out)
    if missing:
        raise RuntimeError(f"{len(missing)} tickers missing from /markets, e.g. {sorted(missing)[:3]}")
    return out


class Farmer:
    """One variant's orders and bookkeeping. All variants share one Simulator: our orders never
    queue ahead of each other (only recorded size counts as ahead), so each fills as if alone."""

    def __init__(self, v: Variant) -> None:
        self.v = v
        self.orders: dict[tuple[str, str], int] = {}          # (ticker, side) -> oid
        self.writes = 0
        self.pos: dict[str, dict[str, float]] = {}            # ticker -> yes/no qty and cost

    def ours(self, sim, ticker: str) -> tuple[dict[float, float], dict[float, float]]:
        out: dict[str, dict[float, float]] = {"yes": {}, "no": {}}
        for side in ("yes", "no"):
            oid = self.orders.get((ticker, side))
            if oid is None:
                continue
            o = sim.orders.get(oid)
            if o is not None and o.status in ("pending", "active") and o.rem > EPS:
                out[side][o.price] = out[side].get(o.price, 0.0) + o.rem
        return out["yes"], out["no"]

    def locked(self, settled: set[str] | None = None) -> float:
        """Capital held by unsettled positions (a YES+NO pair is worth $1 and frees its cost)."""
        tot = 0.0
        for t, p in self.pos.items():
            if settled and t in settled:
                continue
            pair = min(p["yes"], p["no"])
            tot += max(0.0, p["yes_cost"] + p["no_cost"] - pair)
        return tot


def _others(sim, ticker: str) -> tuple[dict[float, float], dict[float, float]]:
    m = sim.markets.get(ticker)
    if m is None:
        return {}, {}
    out = []
    for side in ("yes", "no"):
        d = {}
        for p, lev in m.levels[side].items():
            x = lev.ext_total()
            if x > EPS:
                d[p] = x
        out.append(d)
    return out[0], out[1]


def replay(s, start: str = REPLAY_START, end: str | None = None, variants: tuple[Variant, ...] = VARIANTS,
           latency_ms: float = 20.0) -> dict[str, Any]:
    """Portfolio replay of LIP farmers on the recorded books of the ``albert record`` sample.

    Every ``DECIDE_S`` each variant re-solves :func:`allocate` on the then-current books (others'
    orders only) of the non-financial program markets being replayed, with the budget left after
    unsettled positions, and moves its orders to the new plan (unchanged orders stay, keeping queue
    position). Between decisions orders rest; fills come from the Simulator (FIFO queue, 20 ms
    latency, post-only, trade-through and cross fills, maker fee by series ``fee_type`` rounded up
    to the cent per fill). Rewards accrue per second from the exact scoring of our orders against
    the recorded others' book (``level`` and ``truncate`` modes). Others never react to our orders:
    the reward side is an upper bound in that respect."""
    from albert.backtest.l2 import SimConfig, stream_files
    from albert.fees import kalshi_fee
    from albert.research.longshot_maker import (
        _grep_iter,
        _prune,
        closed_end,
        fast_simulator,
        file_events,
    )

    end = end or closed_end()
    progs = pl.read_parquet(OUT / "programs.parquet")
    rp = recorded_programs(s, progs)
    meta = series_meta()
    rp = rp.filter(pl.col("series_ticker").map_elements(lambda x: not (meta.get(x) or {}).get("financial", True),
                                                        return_dtype=pl.Boolean))
    prog_by_t: dict[str, list[dict[str, Any]]] = {}
    for r in rp.unique(["ticker", "id"]).iter_rows(named=True):
        prog_by_t.setdefault(r["ticker"], []).append(r)
    runs_by_t: dict[str, list[tuple[datetime, datetime]]] = {}
    for r in rp.unique(["ticker", "run_start"]).iter_rows(named=True):
        runs_by_t.setdefault(r["ticker"], []).append((r["run_start"], r["run_end"]))
    info = market_info(_api(), sorted(prog_by_t))
    (OUT / "market_info.json").write_text(json.dumps(info, default=str))
    fee_info = {k: v for k, v in meta.items()}
    sim = fast_simulator(SimConfig(latency_ms=latency_ms, fee_info=fee_info, writes_per_s=None))
    farmers = [Farmer(v) for v in variants]
    files = stream_files(s.raw_dir / "stream", start, end)
    by_hour: dict[str, list[Path]] = {}
    for p in files:
        if p.name[3] == "k":                      # albert record connections only
            by_hour.setdefault(f"{p.parent.name}T{p.name[:2]}", []).append(p)
    # lazy accrual state, per market
    last_ns: dict[str, int] = {}
    cache: dict[str, Any] = {}                    # ticker -> (others snapshot stats, per-variant shares)
    rewards: dict[tuple[str, str, str, str], float] = {}   # (variant, mode, ticker, program id) -> $
    stat_rows: list[dict[str, Any]] = []
    hstat: dict[str, dict[str, float]] = {}
    fills_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    order_meta: dict[int, tuple[str, dict[str, Any]]] = {}
    seen_fills = 0
    alloc_rows: list[dict[str, Any]] = []
    pat = OUT / "patterns.txt"
    t0 = time.time()

    def active_programs(t: str, a: int, b: int) -> list[tuple[dict[str, Any], int]]:
        """(program, seconds of [a, b) ns it covers)."""
        out = []
        for p in prog_by_t.get(t, []):
            ps, pe = int(p["start"].timestamp() * 1e9), int(p["end"].timestamp() * 1e9)
            n = (min(b, pe) - max(a, ps)) // 10**9
            if n > 0:
                out.append((p, n))
        return out

    def state(t: str) -> dict[str, Any]:
        st = cache.get(t)
        if st is not None:
            return st
        oy, on = _others(sim, t)
        p0 = (prog_by_t.get(t) or [None])[0]
        ts, df = (p0["target_size"], p0["discount_factor"]) if p0 else (1000.0, 0.5)
        base_y, base_n = score_side(oy, {}, ts, df), score_side(on, {}, ts, df)
        st = {"valid": base_y.depth >= ts - EPS and base_n.depth >= ts - EPS and base_y.total > 0 and base_n.total > 0,
              "S_yes": base_y.total, "S_no": base_n.total, "d_yes": base_y.depth, "d_no": base_n.depth, "sh": {}}
        for f in farmers:
            uy, un = f.ours(sim, t)
            if not uy and not un:
                continue
            for mode in ("level", "truncate"):
                st["sh"][(f.v.name, mode)] = score_snapshot(oy, on, ts, df, uy, un, mode)
        cache[t] = st
        return st

    def accrue(t: str, now: int) -> None:
        a = last_ns.get(t)
        last_ns[t] = now
        if a is None or now <= a:
            return
        st = state(t)
        n_all = (now - a) / 1e9
        h = hstat.setdefault(t, {"secs": 0.0, "valid_s": 0.0, "S_yes": 0.0, "S_no": 0.0, "versions": 0})
        h["secs"] += n_all
        if st["valid"]:
            h["valid_s"] += n_all
            h["S_yes"] += st["S_yes"] * n_all
            h["S_no"] += st["S_no"] * n_all
        if not st["sh"]:
            return
        for p, n in active_programs(t, a, now):
            rate = p["pool_usd"] / ((p["end"] - p["start"]).total_seconds())
            for (vn, mode), (valid, sy, sn) in st["sh"].items():
                if valid:
                    k = (vn, mode, t, p["id"])
                    rewards[k] = rewards.get(k, 0.0) + n * rate * (sy + sn) / 2

    def touch(t: str) -> None:
        cache.pop(t, None)
        hstat.setdefault(t, {"secs": 0.0, "valid_s": 0.0, "S_yes": 0.0, "S_no": 0.0, "versions": 0})["versions"] += 1

    def decide(now: int, active: list[str]) -> None:
        tnow = datetime.fromtimestamp(now / 1e9, UTC)
        for t in active:
            accrue(t, now)
        rows = []
        for t in active:
            if t not in sim.synced:
                continue
            for p in prog_by_t.get(t, []):
                if p["start"] <= tnow < p["end"]:
                    rows.append({**p, "market_ticker": t})
        if not rows:
            opts = pl.DataFrame()
        else:
            pr = pl.DataFrame(rows, infer_schema_length=None).group_by("market_ticker").agg(
                pl.col("series_ticker", "event_ticker", "target_size", "discount_factor", "end").first(),
                (pl.col("pool_usd") / ((pl.col("end") - pl.col("start")).dt.total_seconds() / 86400)).sum().alias("pool_per_day"))
            books = {t: dict(zip(("yes", "no"), _others(sim, t), strict=True)) for t in pr["market_ticker"].to_list()}
            opts = market_options(pr, books, meta)
        for f in farmers:
            plan = pl.DataFrame()
            if opts.height:
                o = opts.filter(pl.col("price") == pl.col("others_ref")) if f.v.ref_only else opts
                settled = {t for t, i in info.items() if i["settlement_ts"] is not None and i["settlement_ts"] <= tnow}
                plan = allocate(o, f.v.budget - f.locked(settled), tnow, min_dist=f.v.min_dist)
            want = {(r["ticker"], r["side"]): (r["price"], float(r["q"])) for r in plan.iter_rows(named=True)} if plan.height else {}
            for (t, side), oid in list(f.orders.items()):
                o = sim.orders.get(oid)
                live = o is not None and o.status in ("pending", "active") and o.rem > EPS
                w = want.get((t, side))
                if live and w is not None and abs(o.price - w[0]) < EPS and o.rem >= 0.5 * w[1]:
                    want.pop((t, side))           # keep the resting order and its queue place
                    continue
                if live:
                    accrue(t, now)
                    sim.cancel(oid)
                    f.writes += 1
                    touch(t)
                del f.orders[(t, side)]
            for (t, side), (price, q) in want.items():
                accrue(t, now)
                oid = sim.place(t, side, price, q)
                f.writes += 1
                if oid is not None:
                    f.orders[(t, side)] = oid
                    order_meta[oid] = (f.v.name, {"placed_ns": now})
                touch(t)
            if plan.height:
                alloc_rows.extend({"t": tnow, "variant": f.v.name, **{k: r[k] for k in ("ticker", "side", "price", "q", "cost", "share", "rpd", "dist", "others_S")}}
                                  for r in plan.iter_rows(named=True))

    def book_fill(fl) -> None:
        vn, m = order_meta.get(fl.oid, ("?", {}))
        f = next(x for x in farmers if x.v.name == vn)
        p = f.pos.setdefault(fl.market, {"yes": 0.0, "no": 0.0, "yes_cost": 0.0, "no_cost": 0.0})
        p[fl.side] += fl.qty
        p[f"{fl.side}_cost"] += fl.qty * fl.price + fl.fee
        series = fl.market.split("-", 1)[0]
        fi = fee_info.get(series, {})
        exact = kalshi_fee(fl.qty, fl.price, maker=True, fee_type=fi.get("fee_type") or "quadratic",
                           multiplier=fi.get("fee_multiplier") or 1.0, conservative=False)
        xp = xq = xfee = 0.0
        if f.v.exit_fill:
            # taker sale at the best bid of the side we now hold, at most its displayed size; the
            # book is the one right after our fill (no later information)
            bids = _others(sim, fl.market)[0 if fl.side == "yes" else 1]
            if bids:
                xp = max(bids)
                xq = min(fl.qty, bids[xp])
                xfee = kalshi_fee(xq, xp, maker=False, multiplier=fi.get("fee_multiplier") or 1.0)
                unit = (fl.qty * fl.price + fl.fee) / fl.qty
                p[fl.side] -= xq
                p[f"{fl.side}_cost"] -= xq * unit
        pending.append({"variant": vn, "t_ns": fl.t_ns, "ticker": fl.market, "side": fl.side, "price": fl.price,
                        "qty": fl.qty, "fee": fl.fee, "fee_exact": exact, "how": fl.how,
                        "exit_price": xp, "exit_qty": xq, "exit_fee": xfee, **m})

    def markouts(now_ns: int, final: bool) -> None:
        nonlocal pending
        keep = []
        for r in pending:
            if not final and r["t_ns"] + max(MARKOUT_S) * 10**9 >= now_ns:
                keep.append(r)
                continue
            for h in MARKOUT_S:
                tt = r["t_ns"] + h * 10**9
                mid = sim.mid_at(r["ticker"], tt) if tt <= now_ns else None
                r[f"mo_{h}s"] = None if mid is None else ((mid if r["side"] == "yes" else 1 - mid) - r["price"])
            fills_rows.append(r)
        pending = keep

    prev: set[str] = set()
    hour_end = 0
    for hour in sorted(by_hour):
        h0 = datetime.fromisoformat(hour + ":00:00+00:00")
        h1 = h0 + timedelta(hours=1)
        hour_start = int(h0.timestamp() * 1e9)
        hour_end = int(h1.timestamp() * 1e9)
        active = sorted(t for t, ps in prog_by_t.items()
                        if any(p["start"] < h1 and p["end"] > h0 for p in ps)
                        and any(a < h1 and b > h0 for a, b in runs_by_t.get(t, [])))
        for t in prev - set(active):              # not replayed this hour: pull our orders blind
            accrue(t, hour_start)
            for f in farmers:
                for side in ("yes", "no"):
                    oid = f.orders.pop((t, side), None)
                    o = sim.orders.get(oid) if oid else None
                    if o is not None and o.status in ("pending", "active"):
                        lev = sim.markets[t].levels[o.side].get(o.price)
                        if lev:
                            lev.q = [c for c in lev.q if c[1] != oid]
                        o.status = "cancelled"
            last_ns.pop(t, None)
            cache.pop(t, None)
            sim.synced.discard(t)
        for t in set(active) - prev:
            sim.synced.discard(t)
            last_ns.pop(t, None)
            cache.pop(t, None)
        prev = set(active)
        if not active:
            continue
        pat.write_text("".join(f'market_ticker\\":\\"{t}\\"\n' for t in active))
        streams = [file_events(_grep_iter(p, pat)) for p in sorted(by_hour[hour])]
        next_decide = hour_start + DECIDE_S * 10**9
        for ev in heapq.merge(*streams, key=lambda e: e.recv_ns):
            while ev.recv_ns >= next_decide:
                decide(next_decide, [t for t in active if t in sim.synced])
                next_decide += DECIDE_S * 10**9
            t = ev.market
            if t in last_ns:
                accrue(t, ev.recv_ns)
            sim.apply(ev)
            if ev.kind == "snapshot" and t not in last_ns:
                last_ns[t] = ev.recv_ns
            touch(t)
            if len(sim.fills) > seen_fills:
                for fl in sim.fills[seen_fills:]:
                    book_fill(fl)
                seen_fills = len(sim.fills)
        while next_decide < hour_end:
            decide(next_decide, [t for t in active if t in sim.synced])
            next_decide += DECIDE_S * 10**9
        for t in active:
            if t in last_ns:
                accrue(t, hour_end)
        for t, h in hstat.items():
            if h["secs"] > 0:
                stat_rows.append({"hour": hour, "ticker": t, **h})
        hstat.clear()
        markouts(hour_end, final=False)
        _prune(sim, keep_s=max(MARKOUT_S) + 600, now_ns=hour_end)
        tot = {f.v.name: round(sum(v for (vn, md, _, _), v in rewards.items() if vn == f.v.name and md == "level"), 2)
               for f in farmers}
        print(f"replay {hour}: {len(active)} markets, {len(fills_rows) + len(pending)} fills, rewards {tot}, "
              f"{time.time() - t0:.0f}s", flush=True)
    markouts(hour_end, final=True)
    pat.unlink(missing_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{"variant": k[0], "mode": k[1], "ticker": k[2], "program_id": k[3], "reward": v}
                  for k, v in rewards.items()], infer_schema_length=None).write_parquet(OUT / "replay_rewards.parquet")
    pl.DataFrame(fills_rows, infer_schema_length=None).write_parquet(OUT / "replay_fills.parquet")
    pl.DataFrame(stat_rows, infer_schema_length=None).write_parquet(OUT / "replay_stats.parquet")
    pl.DataFrame(alloc_rows, infer_schema_length=None).write_parquet(OUT / "replay_alloc.parquet")
    last = {t: sim.quote_at(t, 2**62) for t in prog_by_t}
    (OUT / "replay_last_quote.json").write_text(json.dumps({t: q for t, q in last.items() if q != (None, None)}))
    run = {"start": start, "end": end, "hours": len(by_hour), "markets": len(prog_by_t), "decide_s": DECIDE_S,
           "latency_ms": latency_ms, "writes": {f.v.name: f.writes for f in farmers}, "diag": dict(sim.diag),
           "seconds": round(time.time() - t0)}
    (OUT / "replay.json").write_text(json.dumps(run, indent=1, default=str))
    return run


# ---- competitor reaction: how fast is a new top-of-window order contested? -------------------------
BIG = 500.0           # an order this large can take most of a 1,000-contract window by itself
MAX_OPEN = 200        # unresolved large orders tracked per market side


def reaction_events(lines: Iterable[str], big: float = BIG, horizon_s: float = 3600.0) -> list[dict[str, Any]]:
    """Large resting orders (a positive delta >= ``big``) that join or improve a side's best bid,
    and what happens next on that side: the first later large order at the same or a better price
    (``counter_s``) and whether a cancel of exactly that size at that price follows
    (``withdraw_s``; executions are not cancels). Each market's lines must be in receive order.
    Books start from each market's snapshot; events before the first snapshot are skipped."""
    from albert.research.longshot_maker import line_event

    books: dict[str, dict[str, dict[float, float]]] = {}
    open_ev: dict[tuple[str, str], list[dict[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    for line in lines:
        ev = line_event(line)
        if ev is None:
            continue
        t = ev.market
        now = ev.recv_ns
        if ev.kind == "snapshot":
            books[t] = {s: {float(p): float(q) for p, q in ev.msg.get(f"{s}_dollars_fp") or [] if float(q) > 0}
                        for s in ("yes", "no")}
            continue
        if ev.kind != "delta" or t not in books:
            continue
        side, price, d = ev.msg["side"], float(ev.msg["price_dollars"]), float(ev.msg["delta_fp"])
        book = books[t][side]
        best = max(book) if book else None
        best_size = book.get(best, 0.0) if best is not None else 0.0
        evs = open_ev.get((t, side), [])
        for e in evs:
            age = (now - e["t_ns"]) / 1e9
            if age > horizon_s:
                continue
            if e["counter_s"] is None and d >= big - EPS and price >= e["price"] - EPS:
                e["counter_s"] = age
                e["counter_improves"] = price > e["price"] + EPS
            if e["withdraw_s"] is None and d < 0 and abs(-d - e["size"]) < 1e-6 and abs(price - e["price"]) < EPS:
                e["withdraw_s"] = age
        if d >= big - EPS and (best is None or price >= best - EPS):
            e = {"t_ns": now, "ticker": t, "side": side, "price": price, "size": d,
                 "kind": "improve" if best is None or price > best + EPS else "join",
                 "prev_best": best, "prev_best_size": best_size, "counter_s": None, "counter_improves": None,
                 "withdraw_s": None}
            out.append(e)
            evs.append(e)
        if evs:   # keep only unresolved events inside the horizon (bounded: requoting bots post thousands)
            evs = [x for x in evs if (now - x["t_ns"]) / 1e9 <= horizon_s
                   and (x["counter_s"] is None or x["withdraw_s"] is None)][-MAX_OPEN:]
            open_ev[(t, side)] = evs
        book[price] = book.get(price, 0.0) + d
        if book[price] <= EPS:
            book.pop(price)
    return out


def reactions(s, start: str = REPLAY_START, end: str | None = None) -> pl.DataFrame:
    """``reaction_events`` over every recorded ``albert record`` hour, program markets only. A
    market's events within one connection-file are in order, and each hour opens with snapshots."""
    from albert.backtest.l2 import stream_files
    from albert.research.longshot_maker import _grep_iter, closed_end

    end = end or closed_end()
    progs = pl.read_parquet(OUT / "programs.parquet")
    rp = recorded_programs(s, progs)
    meta = series_meta()
    tickers = sorted(t for t, ser in rp.select("ticker", "series_ticker").unique().iter_rows()
                     if not (meta.get(ser) or {}).get("financial", True))
    pat = OUT / "patterns_react.txt"
    pat.write_text("".join(f'market_ticker\\":\\"{t}\\"\n' for t in tickers))
    schema = {"t_ns": pl.Int64, "ticker": pl.String, "side": pl.String, "price": pl.Float64, "size": pl.Float64,
              "kind": pl.String, "prev_best": pl.Float64, "prev_best_size": pl.Float64, "counter_s": pl.Float64,
              "counter_improves": pl.Boolean, "withdraw_s": pl.Float64}
    parts: list[pl.DataFrame] = []     # one compact frame per file: requoting bots make millions of rows
    for p in stream_files(s.raw_dir / "stream", start, end):
        if p.name[3] != "k":
            continue
        parts.append(pl.DataFrame(reaction_events(_grep_iter(p, pat)), schema=schema))
    pat.unlink(missing_ok=True)
    df = pl.concat(parts) if parts else pl.DataFrame(schema=schema)
    if df.height:
        df = df.with_columns(pl.from_epoch("t_ns", time_unit="ns").dt.replace_time_zone("UTC").alias("t"))
    df.write_parquet(OUT / "reactions.parquet")
    return df


# ---- cross-section report over the polls -----------------------------------------------------------
def poll_tags() -> list[str]:
    return sorted(p.name[len("books_"):-len(".parquet")] for p in (OUT / "polls").glob("books_*.parquet"))


def load_poll(tag: str) -> tuple[pl.DataFrame, dict]:
    progs = pl.read_parquet(OUT / "polls" / f"programs_{tag}.parquet")
    books = books_dict(pl.read_parquet(OUT / "polls" / f"books_{tag}.parquet"))
    return progs, books


def side_census(progs: pl.DataFrame, books: dict, meta: dict) -> pl.DataFrame:
    """One row per non-financial program side: depth, others' score, and how the 1,000-contract
    window is held (``defended``: the best bid alone holds >= Target Size)."""
    rows = []
    for r in progs.iter_rows(named=True):
        m = meta.get(r["series_ticker"]) or {}
        if m.get("financial", True):
            continue
        b = books.get(r["market_ticker"]) or {"yes": {}, "no": {}}
        ts, df = r["target_size"], r["discount_factor"]
        valid = all(sum(b[s].values()) >= ts - EPS for s in ("yes", "no"))
        for side in ("yes", "no"):
            o = b[side]
            sc = score_side(o, {}, ts, df)
            best = max(o) if o else None
            rows.append({"ticker": r["market_ticker"], "series": r["series_ticker"], "category": m.get("category"),
                         "side": side, "pool_per_day": r["pool_per_day"], "valid": valid, "depth": sc.depth,
                         "S": sc.total, "St": score_side(o, {}, ts, df, "truncate").total, "best": best,
                         "best_size": o.get(best, 0.0) if best is not None else 0.0, "ref": sc.ref,
                         "defended": best is not None and o[best] >= ts - EPS})
    return pl.DataFrame(rows, infer_schema_length=None)


CROSS_VARIANTS = {"opt": {}, "noimprove": {"min_dist": 0}, "behind2": {"min_dist": 2}, "ref": {"min_dist": 0, "ref_only": True}}


def cross() -> dict[str, Any]:
    """Static $100 allocations on each poll, and the same orders left resting until the next poll
    (others' later book, our size added): the drop measures how fast the opportunity moves."""
    meta = series_meta()
    tags = poll_tags()
    if not tags:
        raise RuntimeError("no polls: run `poll` first")
    census, allocs, persist = [], [], []
    prev: dict[str, pl.DataFrame] = {}
    for tag in tags:
        progs, books = load_poll(tag)
        t = progs["t"][0]
        c = side_census(progs, books, meta).with_columns(pl.lit(t).alias("t"))
        census.append(c)
        opts = market_options(progs, books, meta)
        for name, kw in CROSS_VARIANTS.items():
            o = opts.filter(pl.col("price") == pl.col("others_ref")) if kw.get("ref_only") else opts
            a = allocate(o, BANKROLL, t, min_dist=kw.get("min_dist"))
            a = a.with_columns(pl.lit(t).alias("t"), pl.lit(name).alias("variant"))
            allocs.append(a)
            if name in prev and prev[name].height:
                ev = evaluate(prev[name], progs, books)
                persist.append({"variant": name, "t_from": prev[name]["t"][0], "t_to": t,
                                "rpd_planned": float(prev[name]["rpd"].sum()), "rpd_next_poll": float(ev["rpd_real"].sum()),
                                "crossed": int(ev["crossed"].sum()) if "crossed" in ev.columns else 0,
                                "cost": float(prev[name]["cost"].sum()), "sides": prev[name].height})
            prev[name] = a
    census_df = pl.concat(census, how="diagonal_relaxed")
    alloc_df = pl.concat([a for a in allocs if a.height], how="diagonal_relaxed")
    census_df.write_parquet(OUT / "census.parquet")
    alloc_df.write_parquet(OUT / "cross_alloc.parquet")
    pers = pl.DataFrame(persist, infer_schema_length=None)
    pers.write_parquet(OUT / "cross_persist.parquet")
    return {"polls": len(tags), "census_rows": census_df.height, "alloc_rows": alloc_df.height, "persist_rows": pers.height}


# ---- replay report: rewards - adverse selection - fees, ledgers, Sharpe ------------------------------
def replay_pnl(info: dict[str, dict[str, Any]], last_quote: dict[str, Any], end_ts: datetime,
               mode: str = "level", d: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(summary per variant, ledger rows) from the replay outputs. Fills settle at the market result
    when known, else they are sold at the last recorded bid of their side (an exit, not a mark).
    Rewards are paid per (market, program) with Kalshi's rounding and $1 minimum on what accrued
    inside the replay window, credited at the program end (or the window end, if later)."""
    d = d or OUT
    progs = pl.read_parquet(OUT / "programs.parquet").select("id", "end", "pool_usd", "series_ticker")
    rw = pl.read_parquet(d / "replay_rewards.parquet").filter(pl.col("mode") == mode)
    fills = pl.read_parquet(d / "replay_fills.parquet")
    rw = rw.join(progs, left_on="program_id", right_on="id", how="left")
    rw = rw.with_columns(pl.col("reward").map_elements(paid, return_dtype=pl.Float64).alias("paid"))
    legs = []
    for r in rw.filter(pl.col("paid") > 0).iter_rows(named=True):
        when = min(r["end"], end_ts)
        legs.append({"strategy": "lip", "variant": r["variant"], "split": "replay",
                     "struct_id": f"reward|{r['ticker']}|{r['program_id']}", "event_ticker": (info.get(r["ticker"]) or {}).get("event_ticker") or r["ticker"],
                     "ticker": f"REWARD|{r['ticker']}", "side": "yes", "n": 1, "t_in": when, "cost_usd": 0.0,
                     "settle_ts": when, "payoff_usd": r["paid"], "kind": "reward"})
    for f in fills.iter_rows(named=True):
        i = info.get(f["ticker"]) or {}
        t_in = datetime.fromtimestamp(f["t_ns"] / 1e9, UTC)
        res = i.get("result")
        unit = (f["qty"] * f["price"] + f["fee"]) / f["qty"]
        xq = f.get("exit_qty") or 0.0
        base = {"strategy": "lip", "variant": f["variant"], "split": "replay", "event_ticker": i.get("event_ticker") or f["ticker"],
                "ticker": f["ticker"], "side": f["side"], "t_in": t_in, "kind": "fill"}
        if xq > EPS:
            legs.append({**base, "struct_id": f"exit|{f['ticker']}|{f['t_ns']}|{f['side']}", "n": xq,
                         "cost_usd": xq * unit, "settle_ts": t_in, "payoff_usd": xq * f["exit_price"] - f["exit_fee"],
                         "settled": True})
        held = f["qty"] - xq
        if held <= EPS:
            continue
        if res in ("yes", "no") and i.get("settlement_ts") is not None:
            payoff = held * (1.0 if res == f["side"] else 0.0)
            settle = max(i["settlement_ts"], t_in)
        else:
            yb, ya = (last_quote.get(f["ticker"]) or [None, None])
            exit_px = (yb if f["side"] == "yes" else (None if ya is None else 1 - ya)) or 0.0
            payoff, settle = held * exit_px, max(end_ts, t_in)
        legs.append({**base, "struct_id": f"fill|{f['ticker']}|{f['t_ns']}|{f['side']}", "n": held,
                     "cost_usd": held * unit, "settle_ts": settle, "payoff_usd": payoff, "settled": res in ("yes", "no")})
    led = pl.DataFrame(legs, infer_schema_length=None)
    summ = []
    for v in sorted(set(rw["variant"].to_list()) | set(fills["variant"].to_list() if fills.height else [])):
        lv = led.filter(pl.col("variant") == v) if led.height else led
        fl = lv.filter(pl.col("kind") == "fill") if lv.height else lv
        fv = fills.filter(pl.col("variant") == v) if fills.height else fills
        summ.append({"variant": v, "accrued": float(rw.filter(pl.col("variant") == v)["reward"].sum()),
                     "paid": float(rw.filter(pl.col("variant") == v)["paid"].sum()),
                     "programs_paid": int(rw.filter((pl.col("variant") == v) & (pl.col("paid") > 0)).height),
                     "programs_unpaid": int(rw.filter((pl.col("variant") == v) & (pl.col("paid") <= 0)).height),
                     "fills": fv.height, "contracts": float(fv["qty"].sum()) if fv.height else 0.0,
                     "fill_cost": float(fl["cost_usd"].sum()) if fl.height else 0.0,
                     "fees": float(fv["fee"].sum()) if fv.height else 0.0,
                     "fill_pnl": float((fl["payoff_usd"] - fl["cost_usd"]).sum()) if fl.height else 0.0,
                     "fill_settled_share": float(fl["settled"].mean()) if fl.height else None,
                     "mo_60s_c": 100 * float((fv["mo_60s"] * fv["qty"]).sum() / fv["qty"].sum()) if fv.height and fv["mo_60s"].drop_nulls().len() else None,
                     "mo_3600s_c": 100 * float((fv.drop_nulls("mo_3600s")["mo_3600s"] * fv.drop_nulls("mo_3600s")["qty"]).sum() / max(fv.drop_nulls("mo_3600s")["qty"].sum(), 1e-9)) if fv.height else None})
    s = pl.DataFrame(summ).with_columns((pl.col("paid") + pl.col("fill_pnl")).alias("net"))
    return s, led


def ledger_metrics_by_variant(led: pl.DataFrame) -> pl.DataFrame:
    """Sharpe / vol per variant from the capped-book ledger (``albert.research.vol_metrics``)."""
    from albert.research.ta_calendar import ledger_row

    rows = []
    for v in sorted(led["variant"].unique().to_list()):
        lv = led.filter(pl.col("variant") == v).with_columns(pl.col("n").cast(pl.Float64))
        rows.append({"variant": v, **ledger_row(lv)})
    return pl.DataFrame(rows, infer_schema_length=None)


def _md(df: pl.DataFrame, digits: int = 2) -> str:
    if df.is_empty():
        return "(none)\n"
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        out.append("| " + " | ".join(f"{x:.{digits}f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


TERMS_MD = """## Terms (verified 2026-11-17)

Sources: https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program,
https://help.kalshi.com/en/articles/16076644 (where to find programs),
https://help.kalshi.com/en/articles/17184676 (combo-legs program: "no opt-in is required"),
https://docs.kalshi.com/api-reference/incentive-programs/get-incentives and .../get-incentive-schedules,
https://docs.kalshi.com/changelog. robots.txt: help.kalshi.com and docs.kalshi.com allow `User-agent: *`;
kalshi.com (formal terms at /regulatory/notices) answers with a Vercel challenge, so neither its
robots.txt nor the formal terms were read.

- Ends 2027-01-01, may be changed or ended at any time. Most U.S. members; excluded: Kalshi
  affiliates/employees, IBs/FCMs and their customers, non-U.S. users. No sign-up step is described.
- Snapshot once per second at a random moment. Per side (YES bids, NO bids): Reference Price = first
  level from the best bid where cumulative size reaches Target Size / 5; orders that help reach Target
  Size qualify (size x 1.0 at or better than the reference, x DF^ticks below); shares normalised per
  side; snapshot score = share_yes + share_no.
- A snapshot is excluded unless both sides hold >= Target Size (and the market is open); payout =
  period score share x pool x valid/total snapshots, rounded down to the cent; under $1.00 per market
  program is not paid. Pools $1-$1,000 per market per day; periods up to 31 days; paid in a daily run
  (~06:00 ET) after the period closes. No per-account cap on liquidity programs
  (`max_reward_per_account` absent).
- Every active program today: Target Size 1,000, DF 0.50 (programs that started before ~Nov 4 used 300-5,000 and DF 0.05-0.90).
"""


CAVEATS_MD = """## Caveats

- Rewards in the replay are an upper bound: the recorded farmers never react to our orders, but large
  orders at the best bid are matched within 10 minutes 70-89% of the time (table above).
- The replay budget is $100 minus open positions; realised losses are not deducted, so a real $100
  account would have stopped (every variant's drawdown exceeded $100 within 10.7 days).
- The replay covers only the `albert record` random sample (~120-230 program markets per hour of
  ~3,300 non-financial ones); the live cross-section has more opportunities but the same structure.
- Exits (`*_x`) sell at the best bid right after the fill, at most its displayed size, with the taker
  fee rounded up per order; they are not an optimised exit.
- Scoring at the Target Size boundary is not stated by Kalshi: `level` counts the whole crossing level,
  `truncate` only what reaches Target Size. Conclusions are the same in both.
- 10.7 days: Sharpe confidence intervals are not defined (7-day block bootstrap).
"""


def write_report() -> Path:
    """data/research/lip/report.md from the saved outputs (polls, census, replay, reactions)."""
    meta = series_meta()
    parts = ["# Kalshi LIP for a $100 maker\n", TERMS_MD]
    progs = pl.read_parquet(OUT / "programs.parquet")
    act = progs.filter(pl.col("status") == "active").with_columns(
        pl.col("series_ticker").map_elements(lambda x: (meta.get(x) or {}).get("category"), return_dtype=pl.String).alias("category"),
        pl.col("series_ticker").map_elements(lambda x: (meta.get(x) or {}).get("financial", True), return_dtype=pl.Boolean).alias("financial"))
    parts.append("## Active liquidity programs (programs.parquet)\n")
    parts.append(_md(act.group_by("financial", "category").agg(pl.len().alias("markets"), pl.col("pool_per_day").sum().round(0).alias("pool_per_day"),
                                                               pl.col("days").median().alias("period_days")).sort("pool_per_day", descending=True)))
    parts.append(_md(act.select(pl.col("target_size").unique().alias("target_sizes").cast(pl.String), pl.col("discount_factor").unique().cast(pl.String).alias("dfs")).head(3)))
    if (OUT / "census.parquet").exists():
        c = pl.read_parquet(OUT / "census.parquet")
        parts.append("## Competition in the polled books (non-financial sides)\n")
        parts.append(_md(c.group_by("t").agg(pl.len().alias("sides"), pl.col("valid").mean().alias("valid_share"),
                                             pl.col("defended").mean().alias("defended_share"),
                                             pl.col("S").median().alias("S_median"), pl.col("best_size").median().alias("best_size_median"),
                                             pl.col("depth").median().alias("depth_median")).sort("t")))
    if (OUT / "cross_alloc.parquet").exists():
        a = pl.read_parquet(OUT / "cross_alloc.parquet")
        parts.append("## Static $100 allocation per poll (others do not react; rpd = $/day)\n")
        parts.append(_md(a.group_by("variant", "t").agg(pl.len().alias("sides"), pl.col("cost").sum(), pl.col("rpd").sum(),
                                                        pl.col("dist").median().alias("dist_median")).sort("variant", "t")))
        p = pl.read_parquet(OUT / "cross_persist.parquet")
        if p.height:
            parts.append("Same orders against the next poll's book:\n\n")
            parts.append(_md(p.group_by("variant").agg(pl.len().alias("pairs"), pl.col("rpd_planned").mean(), pl.col("rpd_next_poll").mean(),
                                                       pl.col("crossed").sum())))
    if (OUT / "cross_fills.parquet").exists():
        cf = pl.read_parquet(OUT / "cross_fills.parquet")
        parts.append("Public prints at or through the static orders' prices until the next poll (upper bound on fills):\n\n")
        parts.append(_md(cf.group_by("variant").agg(pl.len().alias("orders"), (pl.col("hit_contracts") > 0).mean().alias("share_hit"),
                                                    (pl.col("hit_contracts").clip(upper_bound=pl.col("q")) / pl.col("q")).mean().alias("frac_of_q_15m"))
                         .sort("variant"), 3))
    info = {k: {**v, "settlement_ts": _ts(v.get("settlement_ts")), "close_time": _ts(v.get("close_time"))}
            for k, v in json.loads((OUT / "market_info_final.json").read_text()).items()} \
        if (OUT / "market_info_final.json").exists() else {}
    ledgers = []
    for d in (OUT, OUT / "exit"):
        if not (d / "replay_rewards.parquet").exists():
            continue
        run = json.loads((d / "replay.json").read_text())
        lq = json.loads((d / "replay_last_quote.json").read_text())
        end_ts = datetime.fromisoformat(run["end"] + ":00:00+00:00")
        days = (end_ts - datetime.fromisoformat(run["start"] + ":00:00+00:00")).total_seconds() / 86400
        parts.append(f"## Recorded-L2 replay `{d.name}`: {run['start']} .. {run['end']} ({days:.1f} days, "
                     f"{run['markets']} program markets in the albert-record sample)\n")
        for mode in ("level", "truncate"):
            s, led = replay_pnl(info, lq, end_ts, mode, d)
            s = s.with_columns((pl.col("net") / days * 7).alias("net_per_week"), (pl.col("paid") / days * 7).alias("paid_per_week"))
            parts.append(f"### Scoring mode `{mode}`\n")
            parts.append(_md(s))
            if mode == "level" and led.height:
                ledgers.append(led)
    if ledgers:
        led = pl.concat(ledgers, how="diagonal_relaxed")
        (OUT / "ledgers").mkdir(exist_ok=True)
        led.write_parquet(OUT / "ledgers" / "ledger_lip_level.parquet")
        parts.append("## Capped-book ledger metrics (vol_metrics; rewards credited at program end)\n")
        parts.append(_md(ledger_metrics_by_variant(led).select(
            "variant", "positions", "pnl_usd", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90", "sharpe_ex_top5",
            "max_dd_usd", "first_day", "last_day")))
    if (OUT / "reactions.parquet").exists():
        r = pl.read_parquet(OUT / "reactions.parquet").with_columns(
            (pl.col("withdraw_s") <= 5).fill_null(False).alias("flicker"), (pl.col("price") <= 0.03).alias("cheap"))
        if r.height:
            c = pl.col("counter_s")
            parts.append("## Competitor reaction: a new order >= 500 contracts at the best bid of a program side\n")
            parts.append(f"{r.height:,} such orders on {r['ticker'].n_unique()} markets in 10.6 days; "
                         f"{100 * r['flicker'].mean():.0f}% cancelled within 5 s (requoting). Non-flicker orders:\n\n")
            parts.append(_md(r.filter(~pl.col("flicker")).group_by("kind", "cheap").agg(
                pl.len().alias("events"), (c <= 60).fill_null(False).mean().alias("matched_1m"),
                (c <= 600).fill_null(False).mean().alias("matched_10m"), (c <= 3600).fill_null(False).mean().alias("matched_1h"),
                pl.col("size").median().alias("size_median")).sort("kind", "cheap")))
    parts.append(CAVEATS_MD)
    path = OUT / "report.md"
    path.write_text("\n".join(parts))
    return path


# ---- forward public prints at the static allocations' prices ----------------------------------------
def trade_hits(trades: pl.DataFrame, side: str, price: float) -> pl.DataFrame:
    """Prints that would reach a resting ``side`` bid at ``price``: a YES bid is hit by takers buying
    NO at yes_price <= price; a NO bid by takers buying YES at no price (1 - yes_price) <= price."""
    if side == "yes":
        return trades.filter((pl.col("taker_side") == "no") & (pl.col("yes_price") <= price + EPS))
    return trades.filter((pl.col("taker_side") == "yes") & ((1 - pl.col("yes_price")) <= price + EPS))


def cross_fills() -> pl.DataFrame:
    """For each static allocation (poll t, side, price, q): contracts printed at or through our price
    on our side between t and the next poll (an upper bound on what a resting order there fills;
    a joiner at the back of a level fills less) and the price of the next print after them."""
    from albert.venues.kalshi import trade_rows

    a = pl.read_parquet(OUT / "cross_alloc.parquet")
    tags = poll_tags()
    times = sorted(a["t"].unique().to_list())
    nxt = {t: times[i + 1] for i, t in enumerate(times[:-1])}
    a = a.filter(pl.col("t").is_in(list(nxt)))
    api = _api()
    t0, t1 = min(times), max(times)
    rows = []
    for tk in sorted(a["ticker"].unique().to_list()):
        tr = []
        for page in api.trades(tk, historical=False, min_ts=t0, max_ts=t1):
            tr += trade_rows(page.payload)
        rows.append(pl.DataFrame(tr, schema={"trade_id": pl.String, "ticker": pl.String, "created_time": pl.Datetime("us", "UTC"),
                                             "count": pl.Float64, "yes_price": pl.Float64, "taker_side": pl.String}))
    trades = pl.concat(rows) if rows else pl.DataFrame()
    trades.write_parquet(OUT / "cross_trades.parquet")
    out = []
    for r in a.iter_rows(named=True):
        w = trades.filter((pl.col("ticker") == r["ticker"]) & (pl.col("created_time") > r["t"]) & (pl.col("created_time") <= nxt[r["t"]]))
        h = trade_hits(w, r["side"], r["price"])
        out.append({"variant": r["variant"], "t": r["t"], "ticker": r["ticker"], "side": r["side"], "price": r["price"],
                    "q": r["q"], "dist": r["dist"], "prints": w.height, "hit_prints": h.height,
                    "hit_contracts": float(h["count"].sum()) if h.height else 0.0,
                    "window_min": (nxt[r["t"]] - r["t"]).total_seconds() / 60, "polls": len(tags)})
    df = pl.DataFrame(out, infer_schema_length=None)
    df.write_parquet(OUT / "cross_fills.parquet")
    return df


def finalize() -> Path:
    """Refresh settlement fields of every replayed program market, then write the report."""
    info = json.loads((OUT / "market_info.json").read_text())
    tickers: set[str] = set()
    for d in (OUT, OUT / "exit"):
        if (d / "replay_fills.parquet").exists():
            tickers |= set(pl.read_parquet(d / "replay_fills.parquet")["ticker"].to_list())
    tickers = sorted(tickers)
    if tickers:
        info.update({k: {**v, "settlement_ts": v["settlement_ts"], "close_time": v["close_time"]}
                     for k, v in market_info(_api(), tickers).items()})
    (OUT / "market_info_final.json").write_text(json.dumps(info, default=str))
    return write_report()


def main(argv: list[str]) -> None:
    from albert.config import settings

    try:   # if memory runs short, the kernel kills this process before the live recorders
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass
    step = argv[0] if argv else "report"
    s = settings()
    if step == "programs":
        cmd_programs()
    elif step == "poll":
        cmd_poll(int(argv[1]) if len(argv) > 1 else 12, float(argv[2]) if len(argv) > 2 else 15.0)
    elif step == "cross":
        print(cross())
    elif step == "cross_fills":
        print(cross_fills())
    elif step == "replay":
        print(replay(s))
    elif step == "replay_exit":
        global OUT
        OUT = OUT / "exit"
        OUT.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(OUT.parent / "programs.parquet", OUT / "programs.parquet")
        print(replay(s, variants=EXIT_VARIANTS))
    elif step == "reactions":
        print(reactions(s).height)
    elif step == "report":
        print(write_report())
    elif step == "finalize":
        print(finalize())
    else:
        raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    main(sys.argv[1:])
