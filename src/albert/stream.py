"""Order-book stream recorder: Kalshi and Polymarket intl WebSockets -> append-only raw files.

Layout (see .planning/research/data-and-backtesting.md section 2.2)::

    data/raw/stream/<venue>/<YYYY-MM-DD>/<HH>-<conn>[-n].jsonl.zst
        one line per received frame: {"recv_ns": ..., "conn": ..., "frame": "<verbatim text>"}
        written as ``<name>.part``, flushed every few seconds, closed at the end of the UTC
        hour, fsync'd and renamed. A closed file is never touched again.
    data/runs/<ts>-record.json
        manifest: params, market lists, closed files (sha256, lines), seq gaps, reconnects,
        REST cross-checks of the rebuilt book.

Kalshi: one ``orderbook_delta`` subscription per connection (a documented limit), so markets
are split into chunks across connections. Each connection also subscribes to ``trade`` for the
same markets. Deltas carry a per-subscription ``seq``; on a gap, and at the start of every UTC
hour, we ask for fresh snapshots (so each hourly file can be replayed on its own).
Polymarket intl sends whole ``book`` messages plus ``price_change`` updates and has no sequence
numbers, so we only record it (no gap audit).
"""

import asyncio
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import zstandard
from websockets.asyncio.client import connect

from albert.categories import kalshi_is_non_financial, pm_is_non_financial
from albert.venues import kalshi as k
from albert.venues.kalshi_auth import KalshiSigner

# Direct us-east-2 hosts (see venues/kalshi_trade.ENVS); the CloudFront host added ~10-20 ms.
KALSHI_WS = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
KALSHI_PRIVATE_WS = "wss://warp-ws.kalshi.com/trade-api/ws/v2"   # fill and user_orders only
KALSHI_WS_PATH = "/trade-api/ws/v2"
KALSHI_REST = "https://api.elections.kalshi.com/trade-api/v2"   # market data: cached shared host
PM_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
FLUSH_SECONDS = 30.0
HOUR_NS = 3600 * 10**9
ZSTD_LEVEL = 9  # measured on pilot data: ~25% smaller than level 3, negligible CPU


# ---- append-only hourly writer ----------------------------------------------------------------
@dataclass
class ClosedFile:
    path: str
    lines: int
    bytes: int
    sha256: str


class HourlyWriter:
    """Writes one ``.jsonl.zst`` file per UTC hour. Never reopens or overwrites a closed file."""

    def __init__(self, root: Path, venue: str, conn: str) -> None:
        self.root, self.venue, self.conn = root, venue, conn
        self.closed: list[ClosedFile] = []
        self._hour: datetime | None = None
        self._f = None
        self._zw = None
        self._path: Path | None = None
        self._lines = 0
        self._last_flush = 0.0

    def _open(self, hour: datetime) -> None:
        base = self.root / self.venue / hour.strftime("%Y-%m-%d")
        base.mkdir(parents=True, exist_ok=True)
        n = 0
        while True:
            suffix = f"-{n}" if n else ""
            path = base / f"{hour:%H}-{self.conn}{suffix}.jsonl.zst"
            if not path.exists() and not path.with_name(path.name + ".part").exists():
                break
            n += 1
        self._path, self._hour, self._lines = path, hour, 0
        self._f = open(path.with_name(path.name + ".part"), "wb")  # noqa: SIM115 - closed on hour roll
        self._zw = zstandard.ZstdCompressor(level=ZSTD_LEVEL).stream_writer(self._f, closefd=False)

    def write(self, recv_ns: int, frame: str) -> None:
        hour = datetime.fromtimestamp(recv_ns / 1e9, UTC).replace(minute=0, second=0, microsecond=0)
        if hour != self._hour:
            self.close()
            self._open(hour)
        line = json.dumps({"recv_ns": recv_ns, "conn": self.conn, "frame": frame},
                          separators=(",", ":")) + "\n"
        self._zw.write(line.encode())
        self._lines += 1
        now = time.monotonic()
        if now - self._last_flush > FLUSH_SECONDS:
            # A block flush makes everything so far decodable from the .part file after a crash.
            self._zw.flush(zstandard.FLUSH_BLOCK)
            self._f.flush()
            self._last_flush = now

    def close(self) -> None:
        if self._zw is None:
            return
        self._zw.flush(zstandard.FLUSH_FRAME)
        self._zw.close()
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        part = self._path.with_name(self._path.name + ".part")
        part.rename(self._path)
        data = self._path.read_bytes()
        self.closed.append(ClosedFile(str(self._path), self._lines, len(data),
                                      hashlib.sha256(data).hexdigest()))
        self._zw = self._f = None
        self._hour = None


def read_stream(path: Path) -> list[dict[str, Any]]:
    """Read every line of a closed ``.jsonl.zst`` (or a crashed ``.part``) stream file."""
    with open(path, "rb") as f:
        text = zstandard.ZstdDecompressor().stream_reader(f).read().decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---- Kalshi book state ------------------------------------------------------------------------
class KalshiBook:
    """Per-market YES/NO bid ladders rebuilt from ``orderbook_snapshot`` + ``orderbook_delta``.

    Prices and sizes stay as Kalshi's fixed-point strings converted to float; levels at or below
    zero are dropped.
    """

    def __init__(self) -> None:
        self.books: dict[str, dict[str, dict[float, float]]] = {}

    def apply(self, msg_type: str, msg: dict[str, Any]) -> None:
        ticker = msg["market_ticker"]
        if msg_type == "orderbook_snapshot":
            self.books[ticker] = {
                side: {float(p): float(q) for p, q in msg.get(f"{side}_dollars_fp") or []}
                for side in ("yes", "no")
            }
        elif msg_type == "orderbook_delta":
            ladder = self.books.setdefault(ticker, {"yes": {}, "no": {}})[msg["side"]]
            price = float(msg["price_dollars"])
            size = round(ladder.get(price, 0.0) + float(msg["delta_fp"]), 6)
            if size > 0:
                ladder[price] = size
            else:
                ladder.pop(price, None)

    def levels(self, ticker: str, side: str, depth: int) -> list[tuple[float, float]]:
        ladder = self.books.get(ticker, {}).get(side, {})
        return sorted(ladder.items(), reverse=True)[:depth]


def rest_levels(payload: dict[str, Any], side: str, depth: int) -> list[tuple[float, float]]:
    book = payload.get("orderbook_fp") or {}
    rows = [(float(p), float(q)) for p, q in book.get(f"{side}_dollars") or []]
    return sorted(rows, reverse=True)[:depth]


# ---- stats ----------------------------------------------------------------------------------
@dataclass
class ConnStats:
    frames: int = 0
    bytes: int = 0
    by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    connects: int = 0
    gaps: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    first_ns: int | None = None
    last_ns: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"frames": self.frames, "bytes": self.bytes, "by_type": dict(self.by_type),
                "connects": self.connects, "gaps": self.gaps, "errors": self.errors[-50:],
                "error_count": len(self.errors)}


# ---- universe selection -----------------------------------------------------------------------
def select_kalshi(series_info: dict[str, dict], n: int, *, days: float = 7.0, max_spread: float = 0.25,
                  seed: int = 0, rest: str = KALSHI_REST, min_spread: float = 0.0,
                  vol24h: tuple[float, float] | None = None, transport=None) -> list[dict[str, Any]]:
    """Open non-financial markets closing within ``days`` that currently show a two-sided quote
    with ``min_spread <= spread <= max_spread`` and, if ``vol24h`` = (lo, hi) is given, a trailing
    24 h volume (contracts, ``volume_24h_fp``) in [lo, hi]: the illiquid universe.

    Selection uses only the *current* quote and trailing volume (no future information), then a
    seeded random sample stratified round-robin by category so one category cannot fill the universe.
    """
    now = datetime.now(UTC)
    rows: list[dict] = []
    with httpx.Client(timeout=30, transport=transport) as http:
        cursor = None
        while True:
            params = {"status": "open", "mve_filter": "exclude", "limit": 1000,
                      "min_close_ts": int(now.timestamp()),
                      "max_close_ts": int((now + timedelta(days=days)).timestamp())}
            if cursor:
                params["cursor"] = cursor
            for attempt in range(5):
                r = http.get(f"{rest}/markets", params=params)
                if r.status_code != 429:
                    break
                time.sleep(1 + attempt)
            r.raise_for_status()
            page = r.json()
            got = k.market_rows(page, series_info, "live")
            v24 = {m["ticker"]: float(m.get("volume_24h_fp") or 0) for m in page.get("markets") or []}
            for m in got:
                m["volume_24h"] = v24.get(m["ticker"], 0.0)
            rows.extend(got)
            cursor = page.get("cursor")
            if not cursor or not page.get("markets"):
                break
    eligible = [m for m in rows if kalshi_is_non_financial(m["category"])
                and m["yes_bid"] and m["yes_ask"] and 0 < m["yes_ask"] - m["yes_bid"] <= max_spread + 1e-9
                and m["yes_ask"] - m["yes_bid"] >= min_spread - 1e-9
                and (vol24h is None or vol24h[0] <= m["volume_24h"] <= vol24h[1])]
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for m in eligible:
        by_cat[m["category"]].append(m)
    for ms in by_cat.values():
        rng.shuffle(ms)
    picked: list[dict] = []
    while len(picked) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop())
    return picked


def select_paired(days: float = 3.0) -> list[dict[str, Any]]:
    """Kalshi game moneylines played within ``days`` paired with the Polymarket intl moneyline
    token for the same team (exact match via ``albert.matching`` conventions; data only on intl).

    Polymarket slugs list the two team codes in an order we cannot read off the Kalshi ticker
    (codes vary in length), so each split of the concatenated codes is tried."""
    from albert.matching import KALSHI_GAME_LEAGUES, parse_kalshi_game

    today = datetime.now(UTC).date()
    last = (datetime.now(UTC) + timedelta(days=days)).date().isoformat()
    games: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    with httpx.Client(timeout=30) as http:
        for series in KALSHI_GAME_LEAGUES:
            cursor = None
            while True:
                params = {"series_ticker": series, "status": "open", "limit": 1000}
                if cursor:
                    params["cursor"] = cursor
                r = http.get(f"{KALSHI_REST}/markets", params=params)
                r.raise_for_status()
                page = r.json()
                for m in page.get("markets") or []:
                    g = parse_kalshi_game(m["ticker"])
                    if g and today.isoformat() <= g["date"] <= last:
                        games[(g["league"], g["date"], g["teams"])].append(m["ticker"])
                cursor = page.get("cursor")
                if not cursor or not page.get("markets"):
                    break
        out: list[dict[str, Any]] = []
        for (league, date, teams), tickers in sorted(games.items()):
            for i in range(2, len(teams) - 1):
                a, b = teams[:i], teams[i:]
                slug = f"{league}-{a}-{b}-{date}"
                r = http.get("https://gamma-api.polymarket.com/events", params={"slug": slug})
                r.raise_for_status()
                events = r.json()
                if not events:
                    continue
                ml = next((m for m in events[0].get("markets") or [] if m.get("slug") == slug), None)
                if not ml or ml.get("closed") or not ml.get("enableOrderBook"):
                    break
                tokens = json.loads(ml.get("clobTokenIds") or "[]")
                for t in tickers:
                    pick = t.rsplit("-", 1)[1].lower()
                    if pick in (a, b) and len(tokens) == 2:
                        # one recorded token per game (team a); team b's price is its complement
                        out.append({"kalshi_ticker": t, "token_id": tokens[0], "complement": pick == b,
                                    "pm_slug": slug, "question": ml.get("question")})
                break
    return out


def select_pm(n: int, *, events: int = 200) -> list[dict[str, Any]]:
    """Polymarket intl YES tokens from the most active open non-financial events (data only)."""
    out: list[dict[str, Any]] = []
    with httpx.Client(timeout=30) as http:
        r = http.get("https://gamma-api.polymarket.com/events",
                     params={"active": "true", "closed": "false", "order": "volume24hr",
                             "ascending": "false", "limit": events})
        r.raise_for_status()
        for e in r.json():
            if not pm_is_non_financial([t.get("slug") for t in e.get("tags") or []]):
                continue
            for m in e.get("markets") or []:
                if not m.get("enableOrderBook") or m.get("closed"):
                    continue
                tokens = json.loads(m.get("clobTokenIds") or "[]")
                if tokens and m.get("bestBid") and m.get("bestAsk"):
                    out.append({"token_id": tokens[0], "question": m.get("question"),
                                "event_slug": e.get("slug")})
                if len(out) >= n:
                    return out
    return out


# ---- recorders ------------------------------------------------------------------------------
class Recorder:
    def __init__(self, root: Path, signer: KalshiSigner | None,
                 on_frame: Callable[[str, int, str], None] | None = None,
                 kalshi_ws: str = KALSHI_WS, kalshi_private_ws: str = KALSHI_PRIVATE_WS) -> None:
        """``on_frame(venue, recv_ns, frame)`` is called after each frame is written (used by
        the live paper and live traders, which record exactly the input they trade on).
        ``kalshi_ws`` selects the environment (the demo exchange has its own URL and keys)."""
        self.root = root
        self.kalshi_ws = kalshi_ws
        self.kalshi_private_ws = kalshi_private_ws
        self.on_frame = on_frame
        self.signer = signer
        self.stop = asyncio.Event()
        self.writers: dict[str, HourlyWriter] = {}
        self.stats: dict[str, ConnStats] = {}
        self.book = KalshiBook()
        self.checks: list[dict[str, Any]] = []
        self.stopped_reason = "duration"

    def _writer(self, venue: str, conn: str) -> HourlyWriter:
        if conn not in self.writers:
            self.writers[conn] = HourlyWriter(self.root, venue, conn)
            self.stats[conn] = ConnStats()
        return self.writers[conn]

    def _record(self, venue: str, conn: str, frame: str) -> int:
        recv_ns = time.time_ns()
        self._writer(venue, conn).write(recv_ns, frame)
        st = self.stats[conn]
        st.frames += 1
        st.bytes += len(frame)
        st.first_ns = st.first_ns or recv_ns
        st.last_ns = recv_ns
        if self.on_frame is not None and conn != "rest":
            self.on_frame(venue, recv_ns, frame)
        return recv_ns

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def kalshi_conn(self, conn: str, tickers: list[str]) -> None:
        self._writer("kalshi", conn)
        st = self.stats[conn]
        backoff = 1.0
        while not self.stop.is_set():
            try:
                headers = self.signer.headers("GET", KALSHI_WS_PATH)
                async with connect(self.kalshi_ws, additional_headers=headers, max_size=2**26,
                                   open_timeout=20, ping_interval=20) as ws:
                    st.connects += 1
                    backoff = 1.0
                    cmd_id = 0

                    async def send(cmd: str, params: dict[str, Any]) -> None:
                        nonlocal cmd_id
                        cmd_id += 1
                        await ws.send(json.dumps({"id": cmd_id, "cmd": cmd, "params": params}))

                    await send("subscribe", {"channels": ["orderbook_delta"], "market_tickers": tickers})
                    await send("subscribe", {"channels": ["trade"], "market_tickers": tickers})
                    last_seq: dict[int, int] = {}
                    book_sid: int | None = None
                    snap_hour = time.time_ns() // HOUR_NS
                    stopper = asyncio.create_task(self.stop.wait())
                    while True:
                        recv = asyncio.create_task(ws.recv())
                        done, _ = await asyncio.wait({recv, stopper}, return_when=asyncio.FIRST_COMPLETED)
                        if stopper in done:
                            recv.cancel()
                            return
                        frame = recv.result()
                        if isinstance(frame, bytes):
                            frame = frame.decode()
                        recv_ns = self._record("kalshi", conn, frame)
                        d = json.loads(frame)
                        typ = d.get("type", "?")
                        st.by_type[typ] += 1
                        if typ == "subscribed" and d.get("msg", {}).get("channel") == "orderbook_delta":
                            book_sid = d["msg"].get("sid")
                        elif typ == "error":
                            st.errors.append(json.dumps(d.get("msg")))
                        sid, seq = d.get("sid"), d.get("seq")
                        if sid is not None and seq is not None:
                            prev = last_seq.get(sid)
                            if prev is not None and seq != prev + 1:
                                st.gaps.append({"recv_ns": recv_ns, "sid": sid, "expected": prev + 1,
                                                "got": seq, "type": typ})
                                if sid == book_sid:
                                    await send("update_subscription", {
                                        "sids": [sid], "market_tickers": tickers,
                                        "action": "get_snapshot"})
                            last_seq[sid] = seq
                        if typ in ("orderbook_snapshot", "orderbook_delta"):
                            self.book.apply(typ, d["msg"])
                        if book_sid is not None and recv_ns // HOUR_NS != snap_hour:
                            # Fresh snapshots at the start of each hour make every hourly file
                            # replayable on its own (verified live: the command re-sends one
                            # orderbook_snapshot per market within the same seq stream).
                            snap_hour = recv_ns // HOUR_NS
                            await send("update_subscription", {
                                "sids": [book_sid], "market_tickers": tickers,
                                "action": "get_snapshot"})
            except Exception as exc:  # noqa: BLE001 - recorder must survive any disconnect
                st.errors.append(f"{type(exc).__name__}: {exc}"[:300])
            if not self.stop.is_set():
                await self._sleep_or_stop(backoff)
                backoff = min(backoff * 2, 60.0)

    async def kalshi_private_conn(self, conn: str,
                                  channels: tuple[str, ...] = ("user_orders", "fill"),
                                  url: str | None = None) -> None:
        """The account's own order, fill and order-group updates (authenticated channels; no
        market filter, so every order on the account is seen, including manual ones). ``fill``
        and ``user_orders`` go to the low-latency warp host by default; that host rejects every
        other channel, so ``order_group_updates`` needs ``url=self.kalshi_ws``."""
        url = url or self.kalshi_private_ws
        self._writer("kalshi", conn)
        st = self.stats[conn]
        backoff = 1.0
        while not self.stop.is_set():
            try:
                headers = self.signer.headers("GET", KALSHI_WS_PATH)
                async with connect(url, additional_headers=headers, max_size=2**26,
                                   open_timeout=20, ping_interval=20) as ws:
                    st.connects += 1
                    backoff = 1.0
                    await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                              "params": {"channels": list(channels)}}))
                    last_seq: dict[int, int] = {}
                    stopper = asyncio.create_task(self.stop.wait())
                    while True:
                        recv = asyncio.create_task(ws.recv())
                        done, _ = await asyncio.wait({recv, stopper}, return_when=asyncio.FIRST_COMPLETED)
                        if stopper in done:
                            recv.cancel()
                            return
                        frame = recv.result()
                        if isinstance(frame, bytes):
                            frame = frame.decode()
                        recv_ns = self._record("kalshi", conn, frame)
                        d = json.loads(frame)
                        typ = d.get("type", "?")
                        st.by_type[typ] += 1
                        if typ == "error":
                            st.errors.append(json.dumps(d.get("msg")))
                        sid, seq = d.get("sid"), d.get("seq")
                        if sid is not None and seq is not None:
                            prev = last_seq.get(sid)
                            if prev is not None and seq != prev + 1:
                                st.gaps.append({"recv_ns": recv_ns, "sid": sid, "expected": prev + 1,
                                                "got": seq, "type": typ})
                            last_seq[sid] = seq
            except Exception as exc:  # noqa: BLE001 - must survive any disconnect
                st.errors.append(f"{type(exc).__name__}: {exc}"[:300])
            if not self.stop.is_set():
                await self._sleep_or_stop(backoff)
                backoff = min(backoff * 2, 60.0)

    async def pm_conn(self, conn: str, tokens: list[str]) -> None:
        self._writer("pm", conn)
        st = self.stats[conn]
        backoff = 1.0
        while not self.stop.is_set():
            try:
                async with connect(PM_WS, max_size=2**26, open_timeout=20, ping_interval=None) as ws:
                    st.connects += 1
                    backoff = 1.0
                    await ws.send(json.dumps({"assets_ids": tokens, "type": "market"}))

                    async def heartbeat() -> None:
                        while True:
                            await asyncio.sleep(10)
                            await ws.send("PING")

                    hb = asyncio.create_task(heartbeat())
                    stopper = asyncio.create_task(self.stop.wait())
                    try:
                        while True:
                            recv = asyncio.create_task(ws.recv())
                            done, _ = await asyncio.wait({recv, stopper}, return_when=asyncio.FIRST_COMPLETED)
                            if stopper in done:
                                recv.cancel()
                                return
                            frame = recv.result()
                            if isinstance(frame, bytes):
                                frame = frame.decode()
                            if frame == "PONG":
                                continue
                            self._record("pm", conn, frame)
                            try:
                                d = json.loads(frame)
                            except json.JSONDecodeError:
                                st.by_type["non_json"] += 1
                                continue
                            for item in d if isinstance(d, list) else [d]:
                                st.by_type[item.get("event_type", "?")] += 1
                    finally:
                        hb.cancel()
            except Exception as exc:  # noqa: BLE001
                st.errors.append(f"{type(exc).__name__}: {exc}"[:300])
            if not self.stop.is_set():
                await self._sleep_or_stop(backoff)
                backoff = min(backoff * 2, 60.0)

    async def book_checks(self, every: float, sample: int, depth: int = 5) -> None:
        """Every ``every`` seconds compare the rebuilt book with REST ``/orderbook`` for a few
        markets. The REST response is recorded too. A mismatch can be a race (a delta in flight),
        so each market is re-checked once after a short pause before it counts as a mismatch."""
        rng = random.Random(1)
        async with httpx.AsyncClient(timeout=15) as http:
            while not self.stop.is_set():
                await self._sleep_or_stop(every)
                if self.stop.is_set():
                    return
                tickers = [t for t, b in self.book.books.items() if b["yes"] or b["no"]]
                for t in rng.sample(tickers, min(sample, len(tickers))):
                    result = {"ticker": t, "checked_at": datetime.now(UTC).isoformat(), "match": False}
                    for attempt in range(2):
                        try:
                            r = await http.get(f"{KALSHI_REST}/markets/{t}/orderbook", params={"depth": depth})
                        except httpx.HTTPError as exc:
                            result["error"] = str(exc)[:200]
                            break
                        self._record("kalshi", "rest", json.dumps(
                            {"type": "rest_orderbook", "ticker": t, "status": r.status_code,
                             "body": r.text}))
                        if r.status_code != 200:
                            result["error"] = f"HTTP {r.status_code}"
                            break
                        payload = r.json()
                        ok = all(rest_levels(payload, s, depth) == self.book.levels(t, s, depth)
                                 for s in ("yes", "no"))
                        if ok:
                            result["match"] = True
                            result["attempts"] = attempt + 1
                            break
                        result["rest_top"] = {s: rest_levels(payload, s, 2) for s in ("yes", "no")}
                        result["ws_top"] = {s: self.book.levels(t, s, 2) for s in ("yes", "no")}
                        await asyncio.sleep(1.5)
                    self.checks.append(result)

    def close(self) -> list[ClosedFile]:
        out: list[ClosedFile] = []
        for w in self.writers.values():
            w.close()
            out.extend(w.closed)
        return out


async def disk_guard(rec: Recorder, min_free_gb: float, every: float = 60.0) -> None:
    """Stop recording before the disk fills up."""
    while not rec.stop.is_set():
        free = shutil.disk_usage(rec.root.parent if not rec.root.exists() else rec.root).free / 1e9
        if free < min_free_gb:
            rec.stopped_reason = f"disk free {free:.1f} GB < {min_free_gb} GB"
            rec.stop.set()
            return
        await rec._sleep_or_stop(every)


async def record(rec: Recorder, *, minutes: float, kalshi_markets: list[str],
                 pm_tokens: list[str], chunk: int, check_every: float, check_sample: int,
                 min_free_gb: float = 10.0) -> None:
    """Run every connection until ``minutes`` pass or ``rec.stop`` is set (e.g. by SIGTERM)."""
    tasks = [disk_guard(rec, min_free_gb)]
    for i in range(0, len(kalshi_markets), chunk):
        tasks.append(rec.kalshi_conn(f"k{i // chunk}", kalshi_markets[i:i + chunk]))
    for i in range(0, len(pm_tokens), chunk):
        tasks.append(rec.pm_conn(f"p{i // chunk}", pm_tokens[i:i + chunk]))
    if kalshi_markets and check_sample:
        tasks.append(rec.book_checks(check_every, check_sample))

    async def timer() -> None:
        await rec._sleep_or_stop(minutes * 60)
        rec.stop.set()

    await asyncio.gather(timer(), *tasks)
