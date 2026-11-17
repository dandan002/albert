"""Passive longshot seller on recorded Kalshi L2 (does the FLB maker residual survive the queue?).

The favourite-longshot study (paper_flb, ``flb_structural``) found that the longshot gap (side
price q <= 10c) is inside the taker no-arbitrage band market-weighted (-0.45c, t -4.3) and exceeds
it by +0.79c (t 2.7) volume-weighted, and bounded what a resting seller of longshots could earn at
R^mk = 2.2-2.4c per contract, before queue position and adverse selection. This study asks whether
a maker who *rests* orders to sell longshots captures any of that on our own recorded books.

Strategy (one contract per order, held to settlement):

* The **longshot side** of a market is the side whose best ask q is at most ``LONGSHOT`` (10c):
  YES when ``yes_ask <= 0.10``, NO when ``yes_bid >= 0.90`` (NO ask = 1 - yes_bid). Selling the
  longshot at q is a bid for the favourite at 1 - q: a NO bid at 1 - yes_ask, or a YES bid at yes_bid.
* ``join``: rest at the longshot ask (back of the favourite-bid queue). ``improve``: rest one cent
  better (longshot ask q - 1c, favourite bid + 1c) when that does not cross the other side, so
  we are alone at the front of a new level. Orders re-peg when the target price moves and are
  pulled when the market leaves the longshot zone or the per-market cap (``MAX_POS``) is reached.
* Both variants run in one :class:`~albert.backtest.l2.Simulator`: the simulator never lets our own
  orders queue ahead of one another (only recorded size counts as "ahead"), so each variant fills
  exactly as if it ran alone. Fills, queue, latency (20 ms), post-only rejects, trade-throughs and
  crosses are the simulator's. Maker fees follow each series' ``fee_type`` (only
  ``quadratic_with_maker_fees`` pays); ``fee`` is the per-order round-up to the cent (AGENTS rule 5,
  the simulator default), ``fee_exact`` the unrounded fee.
* Bucket (q at placement: <=2c, 3-5c, 6-10c), family (vol-recorder family, ``other`` for the
  random sample of ``albert record``) and sports phase (pre-game / in-play / after, from the
  league game's scheduled window, ``volatile._in_play_bounds`` convention) are recorded on each
  order at placement, so every cut uses only information available when the order was sent.

I/O plan (the recordings are ~45 GB of zstd JSON lines, ladders 3/4 of it):

1. ``scan``: ``zstd -dc | grep`` keeps only snapshot and trade lines; from them, the first hour in
   which each market's longshot ask is <= ``SCAN_MARGIN`` at a snapshot or a print is at <= 10c.
2. ``sim``: replays only those markets (``grep -F -f`` on their tickers) from the hour *before*
   that first qualifying hour (each hour starts with a fresh snapshot of every market), so every
   entry into the zone is seen with an exact book.

Outputs: data/research/longshot_maker/{scan,fills}.parquet, ledgers/ledger_<variant>.parquet,
report ``research-longshot-maker``. Run: ``python -m albert.research.longshot_maker scan|sim|report|all``.
"""

from __future__ import annotations

import heapq
import json
import math
import subprocess
import sys
import time
import zlib
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from albert.backtest.l2 import (
    EPS,
    Event,
    Market,
    SimConfig,
    Simulator,
    delta_matches,
    stream_files,
    trade_key,
)

OUT = Path("data/research/longshot_maker")
START, END = "2026-11-12T00", None          # END: last closed hour, set at run time
LONGSHOT = 0.10
SCAN_MARGIN = 0.12
MAX_POS = 3
LADDER_RATE = 0.10                          # share of ladder events replayed (I/O and CPU budget)
VARIANTS = ("join", "improve")
HORIZONS = (10, 60, 300)
BUCKETS = ((0.0, 0.02, "<=2c"), (0.02, 0.05, "3-5c"), (0.05, 0.10, "6-10c"))
CONFLATED_FROM = "2026-11-16T14"            # ladders on c<n> connections at 1 s from here
BANKROLL = 100.0
_SNAP = "orderbook_snapshot"
_TRADE = 'type\\":\\"trade'


def bucket_of(q: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo < q <= hi + EPS:
            return name
    return ">10c"


def closed_end(now: datetime | None = None) -> str:
    """Hour key of the current (open) hour: files of earlier hours are closed."""
    now = now or datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H")


def _grep(path: Path, args: list[str]) -> list[str]:
    z = subprocess.Popen(["zstd", "-dcq", str(path)], stdout=subprocess.PIPE)
    g = subprocess.run(["grep", "-F", *args], stdin=z.stdout, capture_output=True, text=True, check=False)
    z.stdout.close()
    z.wait()
    return g.stdout.splitlines()


def _grep_iter(path: Path, pattern_file: Path) -> Iterator[str]:
    """Lines of one stream file matching any pattern, streamed (bounded memory)."""
    z = subprocess.Popen(["zstd", "-dcq", str(path)], stdout=subprocess.PIPE)
    g = subprocess.Popen(["grep", "-F", "-f", str(pattern_file)], stdin=z.stdout, stdout=subprocess.PIPE,
                         text=True, bufsize=1 << 20)
    z.stdout.close()
    try:
        yield from g.stdout
    finally:
        g.stdout.close()
        g.wait()
        if z.wait() != 0:      # an unreadable file must not pass as a quiet hour
            raise RuntimeError(f"zstd failed on {path}")


_KINDS = {"orderbook_snapshot": "snapshot", "orderbook_delta": "delta", "trade": "trade"}


def line_event(line: str) -> Event | None:
    """``events_from_rows`` for one recorded line with a single JSON decode: the escaped frame is
    unescaped by hand when it holds only ``\\"`` and the trailing ``\\n`` escapes (always, on the
    book and trade channels); anything else goes through the two-step decode."""
    i = line.find('"frame":"')
    j = line.rfind('"}')
    raw = line[i + 9:j] if i > 0 and j > i else ""
    if raw and "\\\\" not in raw and "\\u" not in raw and "\\t" not in raw:
        recv_ns = int(line[line.index(":") + 1:line.index(",")])
        d = json.loads(raw.replace('\\"', '"').replace("\\n", ""))
    else:
        row = json.loads(line)
        recv_ns, d = row["recv_ns"], json.loads(row["frame"])
    kind = _KINDS.get(d.get("type"))
    return Event(recv_ns, kind, d["msg"]["market_ticker"], d["msg"]) if kind else None


class StreamMarker:
    """``_mark_executions`` run forward over a stream: the same rule (a negative delta and a trade
    print with the same ts_ms, maker side, price and size), each event released once it is
    ``holdback_ms`` of receive time old, when no later print can still label it. Unmatched deltas
    older than 2 s of exchange time are dropped as the offline version drops them at each print."""

    def __init__(self, holdback_ms: float = 3000.0) -> None:
        self.hold = int(holdback_ms * 1e6)
        self.buf: deque[Event] = deque()
        self.deltas: dict[str, list[Event]] = defaultdict(list)
        self.waiting: dict[str, list[tuple[str, float, float, int]]] = defaultdict(list)

    def push(self, ev: Event) -> None:
        self.buf.append(ev)
        if ev.kind == "delta" and float(ev.msg["delta_fp"]) < 0:
            waiting = self.waiting[ev.market]
            hit = next((i for i, k in enumerate(waiting) if delta_matches(ev.msg, k)), None)
            if hit is not None:
                ev.is_exec = True
                del waiting[hit]
            else:
                cands = self.deltas[ev.market]
                cands.append(ev)
                if len(cands) > 256:
                    ts = ev.msg.get("ts_ms") or 0
                    self.deltas[ev.market] = [d for d in cands if (d.msg.get("ts_ms") or ts) >= ts - 2000]
        elif ev.kind == "trade":
            k = trade_key(ev.msg)
            cands = self.deltas[ev.market]
            hit = next((i for i, d in enumerate(cands) if delta_matches(d.msg, k)), None)
            if hit is not None:
                cands[hit].is_exec = True
                del cands[hit]
            else:
                self.waiting[ev.market].append(k)
            ts = k[3]
            self.deltas[ev.market] = [d for d in cands if (d.msg.get("ts_ms") or ts) >= ts - 2000]
            self.waiting[ev.market] = [w for w in self.waiting[ev.market] if w[3] >= ts - 2000]

    def release(self, now_ns: int) -> Iterator[Event]:
        cut = now_ns - self.hold
        while self.buf and self.buf[0].recv_ns <= cut:
            yield self.buf.popleft()


def file_events(lines: Iterable[str], holdback_ms: float = 3000.0) -> Iterator[Event]:
    """Events of one connection-hour in receive order with executions labelled as
    ``_mark_executions`` labels them, streamed so a 100 MB file never sits in memory."""
    mk = StreamMarker(holdback_ms)
    for line in lines:
        ev = line_event(line)
        if ev is None:
            continue
        mk.push(ev)
        yield from mk.release(ev.recv_ns)
    yield from mk.release(2**62)


def sampled(ticker: str, family: str, ladder_rate: float) -> bool:
    """Ladder events are sampled at ``ladder_rate`` by a hash of the event ticker (all strikes of an
    event together); every other family is kept whole."""
    if family != "ladder" or ladder_rate >= 1:
        return True
    return zlib.crc32(ticker.rsplit("-", 1)[0].encode()) % 10_000 < ladder_rate * 10_000


def game_legs(ctx: dict[str, dict[str, Any]]) -> set[str]:
    """One leg per game-winner event (the first alphabetical non-TIE leg, as in the HF study): the
    two team legs of a game share one outcome, so the second adds events to replay but little
    independent information (and in-play game books are 60% of the replayed events)."""
    by_event: dict[str, list[str]] = defaultdict(list)
    for t, c in ctx.items():
        if c.get("family") == "game" and not t.endswith("-TIE"):
            by_event[c.get("event") or t.rsplit("-", 1)[0]].append(t)
    return {min(ts) for ts in by_event.values()}


def longshot_quote(yes_bid: float | None, yes_ask: float | None) -> tuple[str, float] | None:
    """(longshot side, its ask) when one side's ask is the cheaper of the two, else None."""
    cands = []
    if yes_ask is not None:
        cands.append(("yes", yes_ask))
    if yes_bid is not None:
        cands.append(("no", round(1 - yes_bid, 6)))
    return min(cands, key=lambda x: x[1]) if cands else None


def _snap_top(msg: dict[str, Any]) -> tuple[float | None, float | None]:
    yb = [float(p) for p, q in msg.get("yes_dollars_fp") or [] if float(q) > 0]
    nb = [float(p) for p, q in msg.get("no_dollars_fp") or [] if float(q) > 0]
    return (max(yb) if yb else None), (round(1 - max(nb), 6) if nb else None)


# ---- pass 1: where are the longshots ---------------------------------------------------------
def scan_lines(lines: list[str], hour: str) -> list[dict[str, Any]]:
    """Per market in one file-hour: min longshot ask over its snapshots and min longshot-side
    trade price (min(p, 1 - p)) over its prints."""
    acc: dict[str, dict[str, Any]] = {}
    for line in lines:
        row = json.loads(line)
        d = json.loads(row["frame"])
        msg = d.get("msg") or {}
        t = msg.get("market_ticker")
        if t is None:
            continue
        a = acc.setdefault(t, {"hour": hour, "ticker": t, "snap_ls": None, "trade_ls": None, "trades": 0})
        if d.get("type") == _SNAP:
            ls = longshot_quote(*_snap_top(msg))
            if ls is not None and (a["snap_ls"] is None or ls[1] < a["snap_ls"]):
                a["snap_ls"] = ls[1]
        elif d.get("type") == "trade":
            p = float(msg["yes_price_dollars"])
            q = min(p, 1 - p)
            a["trades"] += 1
            if a["trade_ls"] is None or q < a["trade_ls"]:
                a["trade_ls"] = q
    return list(acc.values())


def scan(s, start: str = START, end: str | None = None) -> pl.DataFrame:
    end = end or closed_end()
    files = stream_files(s.raw_dir / "stream", start, end)
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, f in enumerate(files):
        rows += scan_lines(_grep(f, ["-e", _SNAP, "-e", _TRADE]), f"{f.parent.name}T{f.name[:2]}")
        if i % 200 == 0:
            print(f"scan {i}/{len(files)} {f.parent.name}/{f.name} {time.time() - t0:.0f}s", flush=True)
    schema = {"hour": pl.Utf8, "ticker": pl.Utf8, "snap_ls": pl.Float64, "trade_ls": pl.Float64, "trades": pl.Int64}
    df = (pl.DataFrame(rows, schema=schema)
          .group_by("hour", "ticker").agg(pl.col("snap_ls").min(), pl.col("trade_ls").min(), pl.col("trades").sum())
          .sort("hour", "ticker"))
    df.write_parquet(OUT / "scan.parquet")
    (OUT / "scan.json").write_text(json.dumps({"start": start, "end": end, "files": len(files)}))
    return df


def active_hours(scan_df: pl.DataFrame) -> dict[str, set[str]]:
    """ticker -> hours to replay: hour H when the snapshot opening H or H+1 shows a longshot ask
    <= SCAN_MARGIN (the market is in the zone at H's start, or enters it during H) or a print
    during H is at <= LONGSHOT. Outside these hours a resting longshot order can only have filled
    against a print at <= 10c, which would have made the hour active, so skipping them loses no
    fill (bar a rare cross by a new bid); each active stretch starts from that hour's snapshot."""
    hours = sorted(scan_df["hour"].unique().to_list())
    nxt = {h: hours[i + 1] for i, h in enumerate(hours[:-1])}
    zone = scan_df.filter(pl.col("snap_ls") <= SCAN_MARGIN).select("ticker", "hour")
    snap = {(t, h) for t, h in zone.iter_rows()}
    out: dict[str, set[str]] = defaultdict(set)
    for t, h, tl in scan_df.select("ticker", "hour", "trade_ls").iter_rows():
        if (t, h) in snap or (t, nxt.get(h)) in snap or (tl is not None and tl <= LONGSHOT + EPS):
            out[t].add(h)
    return dict(out)


# ---- market context ----------------------------------------------------------------------------
def market_context(s) -> dict[str, dict[str, Any]]:
    """ticker -> {family, series, event, exp} from the run manifests (vol recorder) and the
    ``albert record`` sample (family ``other``)."""
    ctx: dict[str, dict[str, Any]] = {}
    for p in sorted((s.data_dir / "runs").glob("*-record.json")):
        m = json.loads(p.read_text())
        for r in m["params"].get("kalshi_markets") or []:
            ctx.setdefault(r["ticker"], {"family": "other", "event": r.get("event_ticker"),
                                         "category": r.get("category"), "exp": None})
    for p in sorted((s.data_dir / "runs").glob("*-record-vol.json")):
        m = json.loads(p.read_text())
        for r in m["params"]["kalshi_markets"]:
            ctx[r["ticker"]] = {"family": r["family"], "event": r["event_ticker"], "category": r["category"],
                                "exp": r.get("expected_expiration_time")}
    return ctx


def inplay_windows(ctx: dict[str, dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """Sports event suffix (``<league>|<date+teams>``) -> scheduled in-play window of its game
    market, so halves and ladders of the same game share it."""
    from albert.research.volatile import _in_play_bounds
    from albert.stream import VOL_LEAGUES

    games = {t: {"expected_expiration_time": c["exp"]} for t, c in ctx.items() if c["family"] == "game" and c["exp"]}
    out = {}
    for t, (lo, hi) in _in_play_bounds(games).items():
        series, rest = t.split("-", 1)
        lg = next((x for x in VOL_LEAGUES if series == f"KX{x}GAME"), None)
        if lg:
            out[f"{lg}|{rest.rsplit('-', 1)[0]}"] = (lo, hi)
    return out


def league_key(ticker: str) -> str | None:
    from albert.stream import VOL_LEAGUES, VOL_SPORT_SUFFIXES

    series, rest = ticker.split("-", 1) if "-" in ticker else (ticker, "")
    for lg in sorted(VOL_LEAGUES, key=len, reverse=True):
        for suf in VOL_SPORT_SUFFIXES:
            if series == f"KX{lg}{suf}":
                return f"{lg}|{rest.split('-', 1)[0]}"
    return None


# ---- simulator speed-up ------------------------------------------------------------------------
class FastMarket(Market):
    """``Market`` whose ``best`` stops at the highest price with external size instead of summing
    every level (the same answer; ~4x faster on the deep ladder books, where it dominated)."""

    def best(self, side: str) -> tuple[float, float] | None:
        lv = self.levels[side]
        if not lv:
            return None
        top = max(lv)
        size = lv[top].ext_total()
        if size > EPS:
            return top, size
        for p in sorted(lv, reverse=True):
            size = lv[p].ext_total()
            if size > EPS:
                return p, size
        return None


def fast_simulator(cfg: SimConfig) -> Simulator:
    sim = Simulator(cfg)
    sim.markets = defaultdict(FastMarket)
    return sim


# ---- the strategy ------------------------------------------------------------------------------
class LongshotSeller:
    """All variants in one simulator; see the module docstring."""

    def __init__(self, ctx: dict[str, dict[str, Any]], windows: dict[str, tuple[int, int]],
                 variants: tuple[str, ...] = VARIANTS, max_pos: int = MAX_POS) -> None:
        self.ctx, self.windows, self.variants, self.max_pos = ctx, windows, variants, max_pos
        self.live: dict[tuple[str, str], int] = {}            # (variant, market) -> oid
        self.filled: dict[tuple[str, str], float] = defaultdict(float)
        self.meta: dict[int, dict[str, Any]] = {}            # oid -> context at placement
        self.writes: dict[str, dict[int, int]] = {v: defaultdict(int) for v in variants}
        self._lk: dict[str, str | None] = {}
        self._seen = 0                                          # sim.fills already counted
        self._last: dict[str, tuple] = {}                       # market -> (yes_bid, yes_ask, fills) acted on

    def phase(self, market: str, t_ns: int) -> str:
        if market not in self._lk:
            self._lk[market] = league_key(market)
        k = self._lk[market]
        w = self.windows.get(k) if k else None
        if w is None:
            return "n/a"
        return "pre" if t_ns < w[0] else "in" if t_ns <= w[1] else "after"

    def target(self, variant: str, q: dict[str, float | None]) -> tuple[str, float, float] | None:
        """(our bid side, our bid price, longshot ask we offer) or None."""
        yb, ya = q["yes_bid"], q["yes_ask"]
        ls = longshot_quote(yb, ya)
        if ls is None or ls[1] > LONGSHOT + EPS or ls[1] < 0.01 - EPS:
            return None
        side_ls, ask = ls
        if variant == "improve":
            ask = round(ask - 0.01, 6)
            other_bid = yb if side_ls == "yes" else (None if ya is None else round(1 - ya, 6))
            if ask < 0.01 - EPS or (other_bid is not None and ask <= other_bid + EPS):
                return None
        bid_side = "no" if side_ls == "yes" else "yes"
        return bid_side, round(1 - ask, 6), ask

    def on_event(self, sim: Simulator, market: str) -> None:
        if len(sim.fills) > self._seen:
            for f in sim.fills[self._seen:]:
                m = self.meta.get(f.oid)
                if m is not None:
                    self.filled[(m["variant"], f.market)] += f.qty
            self._seen = len(sim.fills)
        if market not in sim.synced:
            return
        vals = sim.quotes[market][1]
        if not vals:
            return
        state = (*vals[-1], self._seen)
        if self._last.get(market) == state:     # same top of book, no new fill: nothing to change
            return
        self._last[market] = state
        q = {"yes_bid": vals[-1][0], "yes_ask": vals[-1][1]}
        for v in self.variants:
            key = (v, market)
            oid = self.live.get(key)
            o = sim.orders.get(oid) if oid is not None else None
            if o is not None and o.status not in ("pending", "active"):
                o, self.live[key] = None, None
            want = self.target(v, q) if self.filled[key] < self.max_pos - EPS else None
            if o is not None and want is not None and o.side == want[0] and abs(o.price - want[1]) < EPS:
                continue
            if o is not None:
                sim.cancel(o.oid)
                self.writes[v][sim.now // 10**9] += 1
                self.live[key] = None
            if want is not None:
                new = sim.place(market, want[0], want[1], 1.0)
                self.writes[v][sim.now // 10**9] += 1
                if new is not None:
                    self.live[key] = new
                    c = self.ctx.get(market, {})
                    self.meta[new] = {"variant": v, "q": want[2], "bucket": bucket_of(want[2]),
                                      "family": c.get("family", "other"), "phase": self.phase(market, sim.now),
                                      "placed_ns": sim.now, "yes_bid": q["yes_bid"], "yes_ask": q["yes_ask"]}


    def prune(self, sim: Simulator) -> None:
        """Forget placement context of orders the simulator has dropped (fills are captured first)."""
        self.meta = {oid: m for oid, m in self.meta.items() if oid in sim.orders}


def _prune(sim: Simulator, keep_s: int = 900, now_ns: int | None = None) -> None:
    """Bound memory on multi-day replays: drop finished orders nothing still references and quote
    history older than ``keep_s`` (keeping the last quote before the cut)."""
    sched = {oid for _, oid, _ in sim._sched}
    dead = [oid for oid, o in sim.orders.items()
            if o.status in ("cancelled", "rejected", "filled") and oid not in sched]
    for oid in dead:
        del sim.orders[oid]
    dead_set = set(dead)
    for m, ids in sim.by_market.items():
        if dead_set.intersection(ids):
            sim.by_market[m] = [i for i in ids if i not in dead_set]
    cut = (sim.now if now_ns is None else now_ns) - keep_s * 10**9
    for m, (times, vals) in list(sim.quotes.items()):
        i = max(0, int(np.searchsorted(times, cut, side="right")) - 1)
        if i > 0:
            sim.quotes[m] = (times[i:], vals[i:])


def pull(sim: Simulator, strat: LongshotSeller, market: str) -> None:
    """Withdraw our resting orders in a market we stop replaying (it left the longshot zone)."""
    for o in sim.live_orders(market):
        lev = sim.markets[market].levels[o.side].get(o.price)
        if lev:
            lev.q = [c for c in lev.q if c[1] != o.oid]
        o.status = "cancelled"
    for v in strat.variants:
        strat.live[(v, market)] = None
    strat._last.pop(market, None)


def simulate(s, start: str = START, end: str | None = None, *, latency_ms: float = 20.0,
             queue: str = "risk_averse", ladder_rate: float = LADDER_RATE) -> dict[str, Any]:
    """Replay the candidate markets connection-hour by connection-hour. Markets never interact in
    this simulation (no shared write budget; the bankroll cap is applied to the fills afterwards),
    and each market lives on one connection, so files are processed one at a time in hour order:
    every market still sees its own events in receive order."""
    from albert.fees import kalshi_fee

    end = end or json.loads((OUT / "scan.json").read_text())["end"]
    ctx = market_context(s)
    first_leg = game_legs(ctx)
    cand = {t: hs for t, hs in active_hours(pl.read_parquet(OUT / "scan.parquet")).items()
            if sampled(t, ctx.get(t, {}).get("family", "other"), ladder_rate)
            and (ctx.get(t, {}).get("family") != "game" or t in first_leg)}
    windows = inplay_windows(ctx)
    fee_info = {r["series_ticker"]: r for r in pl.read_parquet(s.tables_dir / "kalshi_series.parquet").iter_rows(named=True)}
    sim = fast_simulator(SimConfig(latency_ms=latency_ms, queue=queue, fee_info=fee_info, writes_per_s=None))
    strat = LongshotSeller(ctx, windows)
    files = stream_files(s.raw_dir / "stream", start, end)
    by_hour: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_hour[f"{p.parent.name}T{p.name[:2]}"].append(p)
    stats: dict[str, int] = defaultdict(int)
    fills_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    seen_fills = 0
    pat = OUT / "patterns.txt"
    t0 = time.time()

    def markouts(now_ns: int, final: bool) -> None:
        nonlocal pending
        keep = []
        for r in pending:
            if not final and r["t_ns"] + max(HORIZONS) * 10**9 >= now_ns:
                keep.append(r)
                continue
            for h in HORIZONS:
                t = r["t_ns"] + h * 10**9
                mid = sim.mid_at(r["market"], t) if t <= now_ns else None
                r[f"mo_{h}s"] = None if mid is None else ((mid if r["side"] == "yes" else 1 - mid) - r["price"])
            fills_rows.append(r)
        pending = keep

    hour_end = 0
    prev: set[str] = set()
    for hour in sorted(by_hour):
        hour_end = int((datetime.fromisoformat(hour + ":00:00+00:00").timestamp() + 3600) * 1e9)
        now_active = {t for t, hs in cand.items() if hour in hs}
        for t in prev - now_active:              # leaves the zone: our orders are pulled
            pull(sim, strat, t)
        for t in now_active - prev:              # (re)enters: rebuild the book from this hour's snapshot
            sim.synced.discard(t)
            strat._last.pop(t, None)
        prev = now_active
        stats["market_hours"] += len(now_active)
        active = sorted(now_active)
        if not active:
            continue
        pat.write_text("".join(f'market_ticker\\":\\"{t}\\"\n' for t in active))
        # a market can sit on two connections in a recorder's restart hour: older run first
        for p in sorted(by_hour[hour], key=lambda q: q.stat().st_mtime):
            for ev in file_events(_grep_iter(p, pat)):
                sim.apply(ev)
                stats["events"] += 1
                stats["trades"] += ev.kind == "trade"
                stats["executions"] += ev.is_exec
                strat.on_event(sim, ev.market)
                if len(sim.fills) > seen_fills:
                    for f in sim.fills[seen_fills:]:
                        m = strat.meta.get(f.oid, {})
                        info = fee_info.get(f.market.split("-", 1)[0], {})
                        exact = kalshi_fee(f.qty, f.price, maker=True, fee_type=info.get("fee_type") or "quadratic",
                                           multiplier=info.get("fee_multiplier") or 1.0, conservative=False)
                        pending.append({"t_ns": f.t_ns, "market": f.market, "side": f.side, "price": f.price,
                                        "qty": f.qty, "fee": f.fee, "fee_exact": exact, "how": f.how, "oid": f.oid,
                                        "fee_type": info.get("fee_type"), **m})
                    seen_fills = len(sim.fills)
        markouts(hour_end, final=False)
        _prune(sim, now_ns=hour_end)
        strat.prune(sim)
        nf = len(fills_rows) + len(pending)
        print(f"sim {hour}: {len(active)} markets, {stats['events']:,} events, {nf} fills, "
              f"{len(sim.orders)} orders held, {time.time() - t0:.0f}s", flush=True)
    markouts(hour_end, final=True)
    pat.unlink(missing_ok=True)
    fills = pl.DataFrame(fills_rows, infer_schema_length=None) if fills_rows else pl.DataFrame()
    if fills.height:
        fills = fills.with_columns(((pl.col("t_ns") - pl.col("placed_ns")) / 1e9).alias("wait_s")).drop("placed_ns").with_columns(
            pl.from_epoch(pl.col("t_ns"), time_unit="ns").dt.replace_time_zone("UTC").alias("t_in"))
    fills.write_parquet(OUT / "fills.parquet")
    last_mid = {t: sim.mid_at(t, 2**62) for t in (fills["market"].unique().to_list() if fills.height else [])}
    (OUT / "last_mid.json").write_text(json.dumps({t: v for t, v in last_mid.items() if v is not None}))
    peak = {v: (max(w.values()) if w else 0) for v, w in strat.writes.items()}
    p99 = {v: (float(np.percentile(list(w.values()), 99)) if w else 0) for v, w in strat.writes.items()}
    fam = defaultdict(int)
    for t in cand:
        fam[ctx.get(t, {}).get("family", "other")] += 1
    run = {"start": start, "end": end, "candidates": len(cand), "candidates_by_family": dict(fam),
           "game_legs": "first alphabetical non-TIE leg per game",
           "market_hours": stats["market_hours"],
           "ladder_rate": ladder_rate, "events": stats["events"], "trades": stats["trades"],
           "executions_labelled": stats["executions"], "orders": sim._next - 1, "fills": fills.height,
           "diag": dict(sim.diag), "writes_peak_per_s": peak, "writes_p99_per_s": p99,
           "latency_ms": latency_ms, "queue": queue, "max_pos": MAX_POS, "seconds": round(time.time() - t0)}
    (OUT / "sim.json").write_text(json.dumps(run, indent=1))
    return run


# ---- settlement, ledgers, report --------------------------------------------------------------
def settle_fields(page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """ticker -> settlement fields of one ``GET /markets`` page."""
    keys = ("result", "status", "close_time", "settlement_ts", "expected_expiration_time")
    return {m["ticker"]: {k: m.get(k) for k in keys} for m in page.get("markets", [])}


def settle_info(events: list[str], cache: Path, *, tries: int = 6) -> dict[str, dict[str, Any]]:
    """ticker -> {result, status, settle_ts, close_time} from public ``GET /markets?event_ticker``,
    cached; events whose markets are not all settled are re-fetched on every run. 429, 5xx and
    network errors are retried with backoff; an event still missing raises after the cache is
    saved (a rerun resumes), so rate limits can never quietly shrink the sample."""
    import httpx

    have = json.loads(cache.read_text()) if cache.exists() else {}
    done = {e for e, ms in have.items() if ms and all(m.get("result") in ("yes", "no") for m in ms.values())}
    failed: dict[str, str] = {}
    with httpx.Client(timeout=20) as http:
        for e in events:
            if e in done:
                continue
            why = ""
            for i in range(tries):
                try:
                    r = http.get("https://api.elections.kalshi.com/trade-api/v2/markets",
                                 params={"event_ticker": e, "limit": 1000})
                except httpx.HTTPError as exc:
                    why = repr(exc)
                else:
                    if r.status_code == 200:
                        have[e] = settle_fields(r.json())
                        break
                    why = f"HTTP {r.status_code}"
                    if r.status_code != 429 and r.status_code < 500:
                        break
                time.sleep(min(30.0, 0.5 * 2**i))
            else:
                failed[e] = why
            if e not in have and e not in failed:
                failed[e] = why
            if len(have) % 200 == 0:           # checkpoint: an interrupted run resumes
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(have))
            time.sleep(0.05)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(have, indent=1))
    if failed:
        raise RuntimeError(f"market metadata missing for {len(failed)} events (cache saved; rerun "
                           f"resumes): {dict(list(failed.items())[:5])}")
    return {t: v for ms in have.values() for t, v in ms.items()}


def event_of(ticker: str, ctx: dict[str, dict[str, Any]] | None = None) -> str:
    """The market's event ticker from the recorder manifests (tickers such as player props carry
    extra hyphens), else the ticker minus its last segment."""
    e = (ctx or {}).get(ticker, {}).get("event")
    return e or ticker.rsplit("-", 1)[0]


def attach_settlement(fills: pl.DataFrame, info: dict[str, dict[str, Any]], last_mid: dict[str, float],
                      data_end: datetime, ctx: dict[str, dict[str, Any]] | None = None) -> pl.DataFrame:
    """Per fill: ``won`` (held side won; None if unsettled), ``final_value`` (1/0, or the last
    recorded mid of the held side when unsettled), ``settle_ts`` and PnL per contract (fee per
    order rounded up, and exact)."""
    missing = sorted(set(fills["market"].to_list()) - set(info))
    if missing:
        raise RuntimeError(f"no settlement metadata for {len(missing)} filled markets, e.g. {missing[:5]}")
    rows = []
    for t, side in fills.select("market", "side").unique().iter_rows():
        m = info[t]
        res = m.get("result")
        if res in ("yes", "no"):
            won, val, settled = res == side, float(res == side), True
            ts = m.get("settlement_ts") or m.get("close_time") or m.get("expected_expiration_time")
            sts = datetime.fromisoformat(ts) if ts else data_end
        else:
            mid = last_mid.get(t)
            val = None if mid is None else (mid if side == "yes" else 1 - mid)
            won, settled, sts = None, False, data_end
        rows.append({"market": t, "side": side, "won": won, "final_value": val, "settled": settled, "settle_ts": sts})
    st = pl.DataFrame(rows, schema={"market": pl.Utf8, "side": pl.Utf8, "won": pl.Boolean, "final_value": pl.Float64,
                                    "settled": pl.Boolean, "settle_ts": pl.Datetime("us", "UTC")})
    return fills.join(st, on=["market", "side"], how="left").with_columns(
        (pl.col("final_value") - pl.col("price")).alias("gross_pc"),
        (pl.col("final_value") - pl.col("price") - pl.col("fee") / pl.col("qty")).alias("net_pc"),
        (pl.col("final_value") - pl.col("price") - pl.col("fee_exact") / pl.col("qty")).alias("net_exact_pc"),
        pl.col("market").map_elements(lambda t: event_of(t, ctx), return_dtype=pl.Utf8).alias("event"))


def clustered(df: pl.DataFrame, col: str, by: str = "event") -> dict[str, Any]:
    """Contract-weighted mean of ``col`` with a t-stat clustered by ``by``."""
    f = df.filter(pl.col(col).is_not_null())
    if f.is_empty():
        return {"n": 0, "mean": None, "t": None, "clusters": 0}
    g = f.group_by(by).agg((pl.col(col) * pl.col("qty")).sum().alias("s"), pl.col("qty").sum().alias("n"))
    n = g["n"].sum()
    mean = g["s"].sum() / n
    k = g.height
    se = math.sqrt(k / (k - 1) * ((g["s"] - mean * g["n"]) ** 2).sum()) / n if k > 1 else None
    return {"n": float(n), "mean": mean, "t": (mean / se if se else None), "clusters": k}


def fair_t(df: pl.DataFrame, col: str = "net_exact_pc") -> float | None:
    """Mean of ``col`` over its binomial standard error if prices were fair: the sold longshot wins
    with probability q, so a contract's payoff has variance q(1 - q); fills in one market share
    one outcome (perfectly correlated). Unlike the clustered t, this does not explode when a
    sample of rare-loss bets happens to contain no loss."""
    f = df.filter(pl.col(col).is_not_null())
    if f.is_empty():
        return None
    g = f.group_by("market").agg(pl.col("qty").sum().alias("n"), (1 - pl.col("price")).mean().alias("q"))
    n = f["qty"].sum()
    se = math.sqrt((g["n"] ** 2 * g["q"] * (1 - g["q"])).sum()) / n
    mean = (f[col] * f["qty"]).sum() / n
    return mean / se if se > 0 else None


def capped_ledger(f: pl.DataFrame, variant: str, bankroll: float = BANKROLL, fee_col: str = "fee_exact") -> pl.DataFrame:
    """Fills in time order on a $``bankroll`` book: a fill is taken only if the cost of open
    positions plus its own stays within the bankroll; capital returns at settlement. (Collateral
    of resting, unfilled orders is not counted: an optimistic simplification.)"""
    open_: list[tuple[datetime, float]] = []
    used, rows = 0.0, []
    for r in f.filter(pl.col("final_value").is_not_null()).sort("t_in").iter_rows(named=True):
        while open_ and open_[0][0] <= r["t_in"]:
            used -= heapq.heappop(open_)[1]
        cost = r["qty"] * r["price"] + r[fee_col]
        if used + cost > bankroll + EPS:
            continue
        used += cost
        heapq.heappush(open_, (r["settle_ts"], cost))
        rows.append({"strategy": "longshot_maker", "variant": variant, "split": "nov",
                     "struct_id": str(r["oid"]), "event_ticker": r["event"], "ticker": r["market"],
                     "side": r["side"], "n": r["qty"], "t_in": r["t_in"], "cost_usd": cost,
                     "settle_ts": max(r["settle_ts"], r["t_in"]), "payoff_usd": r["qty"] * r["final_value"]})
    return pl.DataFrame(rows)


def _cuts(f: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    out = []
    for v in VARIANTS:
        fv = f.filter(pl.col("variant") == v)
        out.append((v, fv))
        for b in [x[2] for x in BUCKETS]:
            out.append((f"{v}|{b}", fv.filter(pl.col("bucket") == b)))
        for fam in sorted(fv["family"].unique().to_list()):
            out.append((f"{v}|{fam}", fv.filter(pl.col("family") == fam)))
        for ph in ("pre", "in"):
            out.append((f"{v}|sports-{ph}", fv.filter(pl.col("phase") == ph)))
        out.append((f"{v}|no-maker-fee", fv.filter(pl.col("fee_type") != "quadratic_with_maker_fees")))
    return out


def write_report(s) -> Path:
    from albert.reports import Report
    from albert.research.vol_metrics import block_bootstrap, ledger_metrics, load_marks, mtm_daily

    run = json.loads((OUT / "sim.json").read_text())
    fills = pl.read_parquet(OUT / "fills.parquet")
    if fills.is_empty():
        raise RuntimeError("zero fills: the smoke run failed, investigate before reporting")
    data_end = datetime.fromisoformat(run["end"] + ":00:00+00:00")
    ctx = market_context(s)
    info = settle_info(sorted({event_of(t, ctx) for t in fills["market"].unique().to_list()}), OUT / "markets.json")
    last_mid = json.loads((OUT / "last_mid.json").read_text()) if (OUT / "last_mid.json").exists() else {}
    f = attach_settlement(fills, info, last_mid, data_end, ctx)
    f.write_parquet(OUT / "fills_settled.parquet")
    (OUT / "ledgers").mkdir(exist_ok=True)
    marks = load_marks(sorted(f["market"].unique().to_list()))
    rows = []
    for name, part in _cuts(f):
        if part.is_empty():
            continue
        gross, rnd, net = clustered(part, "gross_pc"), clustered(part, "net_pc"), clustered(part, "net_exact_pc")
        sett = part.filter(pl.col("final_value").is_not_null())
        cost = (sett["qty"] * sett["price"] + sett["fee_exact"]).sum()
        pnl = (sett["qty"] * sett["net_exact_pc"]).sum()
        r = {"cut": name, "fills": part.height, "contracts": round(part["qty"].sum(), 1),
             "markets": part["market"].n_unique(), "events": part["event"].n_unique(),
             "settled_share": round(part["settled"].mean(), 3), "mean_q_c": round(100 * (1 - part["price"]).mean(), 2),
             "gross_c": _c(gross["mean"]), "net_c": _c(net["mean"]), "net_t": _r(net["t"]),
             "net_t_fair": _r(fair_t(part)), "net_rounded_c": _c(rnd["mean"]), "net_rounded_t": _r(rnd["t"]),
             "pnl_per_$": _r(pnl / cost, 4) if cost else None,
             "loss_rate": _r(part.filter(pl.col("won").is_not_null())["won"].not_().mean(), 4),
             **{f"mo_{h}s_c": _c(clustered(part, f"mo_{h}s")["mean"]) for h in HORIZONS}}
        led = capped_ledger(part, name)
        if led.height:
            led.write_parquet(OUT / "ledgers" / f"ledger_{name.replace('|', '_')}.parquet")
            m = ledger_metrics(led, marks)
            # fewer than 14 days: the 7-day block bootstrap is undefined, so resample days i.i.d.
            boot = m["sharpe_ci90"] or _ci(block_bootstrap(mtm_daily(led, marks), block=1))
            r.update({"capped_fills": led.height, "capped_pnl_usd": m["pnl_usd"], "days": m["days"],
                      "ann_return_pct": m["ann_return_pct"], "ann_vol_pct": m["ann_vol_pct"], "sharpe": m["sharpe"],
                      "sharpe_ci90": str(boot), "sharpe_ex_top5": m["sharpe_ex_top5"],
                      "max_dd_usd": m["max_dd_usd"], "skew": m["skew"]})
        rows.append(r)
    tab = pl.DataFrame(rows, infer_schema_length=None)
    tab.write_parquet(OUT / "summary.parquet")
    rep = Report(s.reports_dir, "research-longshot-maker")
    rep.text("Question", __doc__.split("\n\n")[1])
    rep.text("Headline", headline(tab, f))
    rep.text("Reading the table", (
        "Per contract, in cents: `gross_c` = settlement value of the favourite held minus its price; "
        "`net_c` = gross minus the exact maker fee (Kalshi charges to $0.000001 and rebates rounding), "
        "`net_rounded_c` with the fee rounded up per order. `net_t` clusters by event; `net_t_fair` "
        "divides by the binomial error the sold longshots would have if priced fairly (rare-loss "
        "samples with no loss make the clustered t meaningless). `pnl_per_$` = net PnL / capital "
        "(price + fee) on settled-or-marked fills. `mo_*s_c` = mid of the held side h seconds after "
        "the fill minus the price paid (negative = adverse selection). Capped book: fills taken in "
        "time order while open cost <= $100 (resting-order collateral ignored), daily MTM via "
        "vol_metrics; with under 14 days the Sharpe CI resamples days i.i.d. Sports cuts use the "
        "league game's scheduled window. Cuts by bucket, family and phase are subsets of the two "
        "strategies' fills (the per-market cap of 3 couples them slightly)."))
    rep.json("Run", run)
    rep.table("Variants and cuts (per contract in cents; t clustered by event; Sharpe from the "
              "$100 capped-book ledger, daily MTM via vol_metrics)", tab)
    by_how = f.group_by("variant", "how").agg(pl.len().alias("fills"), (100 * pl.col("net_exact_pc").mean()).alias("net_c"),
                                              (100 * pl.col("mo_60s").mean()).alias("mo_60s_c"))
    rep.table("Fills by how they happened", by_how.sort("variant", "how"))
    return rep.save()


def headline(tab: pl.DataFrame, f: pl.DataFrame) -> str:
    """One paragraph per strategy from the summary table and the fills."""
    out = []
    for v in VARIANTS:
        r = tab.filter(pl.col("cut") == v).row(0, named=True)
        part = f.filter(pl.col("variant") == v)
        se = abs(r["net_c"] / r["net_t"]) if r["net_t"] else None
        hi = None if se is None else r["net_c"] + 1.96 * se
        q = part.group_by("how").agg(pl.len().alias("n"), (100 * pl.col("net_exact_pc").mean()).alias("c"),
                                     (100 * pl.col("mo_60s").mean()).alias("mo"))
        hows = "; ".join(f"{h}: {n} fills, net {c:+.2f}c, 60 s markout {mo:+.2f}c"
                         for h, n, c, mo in q.sort("how").iter_rows())
        out.append(
            f"**{v}**: {r['fills']:,} fills ({r['contracts']:,.0f} contracts, {r['events']:,} events, mean q "
            f"{r['mean_q_c']:.2f}c, longshot hit rate {100 * r['loss_rate']:.2f}%). Net {r['net_c']:+.2f}c per "
            f"contract (t {r['net_t']:+.2f} by event, {r['net_t_fair']:+.2f} binomial; per-order rounded fees "
            f"{r['net_rounded_c']:+.2f}c); upper 95% bound {hi:+.2f}c against the 2.2-2.4c maker bound of the "
            f"FLB study. Mid after the fill: {r['mo_10s_c']:+.2f}c at 10 s, {r['mo_60s_c']:+.2f}c at 60 s, "
            f"{r['mo_300s_c']:+.2f}c at 5 min. By fill type: {hows}. $100 capped book: {r['capped_fills']} fills, "
            f"${r['capped_pnl_usd']:+.2f} over {r['days']} days, Sharpe {r['sharpe']} (90% CI {r['sharpe_ci90']}, "
            f"ex top-5 events {r['sharpe_ex_top5']}), annualised vol {r['ann_vol_pct']}%: five days of marks "
            "cannot pin a Sharpe ratio down.")
    return "\n\n".join(out)


def _ci(boot: np.ndarray) -> list[float] | None:
    return [round(float(np.quantile(boot, 0.05)), 2), round(float(np.quantile(boot, 0.95)), 2)] if len(boot) else None


def _c(x: float | None) -> float | None:
    return None if x is None else round(100 * x, 3)


def _r(x: float | None, k: int = 2) -> float | None:
    return None if x is None else round(x, k)


def main(argv: list[str]) -> None:
    from albert.config import settings

    s = settings()
    try:   # if memory runs short, the kernel kills this replay before the live recorders
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass
    step = argv[0] if argv else "all"
    if step in ("scan", "all"):
        print(scan(s).height, "scan rows", flush=True)
    if step in ("sim", "all"):
        print(simulate(s), flush=True)
    if step in ("report", "all"):
        print(write_report(s), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
