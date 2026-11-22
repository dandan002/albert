"""Forward paper tests of the 2026-11-21 swarm leads on live Kalshi data. No order is ever sent.

Three rules, fixed before any forward data, each booked against its own paper bankroll (default
$25: the user asked for a $20-30 start). Specs: ``data/research/culture_flow/review.md``,
``data/research/music_charts/review.md`` and ``data/research/approval2/review.md``.

``cmn`` culture maker NO. At each event's scheduled close - 72 h (the latest leg close_time while the
    event is open) in the 37 culture_flow series, read the live REST book of every cheap leg. A leg
    qualifies with a real two-sided book (best YES bid >= 1c, best YES ask <= 99c) and mid <= 5c.
    Rest a 1-contract post-only NO bid that sells YES at ceil(mid + 12c): cheapest NO first, at most
    3 legs per event, while resting collateral plus open positions stay within the bankroll. Legs
    past either cap become shadow orders (tracked, not booked). Fill: the first YES-taker print
    strictly above the sell price after placement and before the order ends
    (``paper_trader.through``; queue-free, so the 1 lot fills at any queue position). An order ends
    at its leg's close, or when the event's close moves or the leg leaves the open listing. Fills are
    held to settlement; the maker fee follows the series' fee_type, rounded up to the cent.
``hits`` HITS preview -> KXALBUMEQUIV taker. On first sight of the first HITS chart preview of a
    tracking week (listing pages polled every 5 min Tue 13:00-21:00 ET, every 30 min otherwise; the
    post's raw HTML is frozen at first sight), price every open KXALBUMEQUIV strike of that week:
    P(Luminate > K) = t5 survival of (ln K - ln Activity - mu) / sd with the frozen dev fit. Buy YES
    at the best ask, or NO at 1 - best bid, when the edge after the per-order taker fee is >= 5c, up to
    the displayed size, 10 contracts and 4% of the bankroll per order, 5% per event (best edge
    first). Quotes of every priced strike are logged at +1 h, +6 h and +24 h (drift check).
``apr`` approval NO. 12 h before the 1 PM ET settlement of each KXTRUMPAPPROVE event, buy 1 NO
    contract at 1 - best YES bid on every leg with 5c < mid <= 15c and spread <= 10c whose bid shows at
    least 1 contract. Every qualifying leg is taken, so no tie-break is needed (the brackets are
    exclusive: at most one leg per event loses).

State is an append-only ledger per rule (``data/paper_forward/<rule>/ledger.jsonl``), replayed on
start, so a restart resumes. ``marks.jsonl`` holds hourly quotes of every open paper position for
daily marks. Stop with ``touch data/paper_forward/STOP``. Score with ``python -m albert.paper_forward
report``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import signal
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from albert.fees import kalshi_fee
from albert.paper_trader import through

OUT = Path("data/paper_forward")
ET = ZoneInfo("America/New_York")
BANKROLL = 25.0
DECIDE_WINDOW = timedelta(minutes=30)      # a decision is taken at the first pass inside [t, t + window]

# culture_flow's 37 series (data/research/culture_flow/markets.parquet, 9 families)
CMN_SERIES = (
    "KXALBUMEQUIV", "KXAPRPOTUS", "KXBILLBOARDRUNNERUPALBUM", "KXBILLBOARDRUNNERUPSONG", "KXEOWEEK", "KXHORMUZMAX",
    "KXHORMUZPEAK", "KXHORMUZWEEKLY", "KXLLM1", "KXMAMDANIEO", "KXNETFLIXRANKMOVIE", "KXNETFLIXRANKMOVIEGLOBAL",
    "KXNETFLIXRANKMOVIEGLOBAL2", "KXNETFLIXRANKMOVIERUNNERUP", "KXNETFLIXRANKSHOW", "KXNETFLIXRANKSHOWGLOBAL",
    "KXNETFLIXRANKSHOWGLOBAL2", "KXNETFLIXRANKSHOWRUNNERUP", "KXNETFLIXTOPVIEWSMOVIE", "KXNETFLIXTOPVIEWSTV",
    "KXPUREALBUMS", "KXRANKLISTSONGTOP10", "KXRT", "KXTOPALBUM", "KXTOPMODEL", "KXTOPSONG", "KXTRUMPAPPROVE",
    "KXTRUMPENDORSEMENTS", "KXTRUMPNOMNUM", "KXTRUMPVH", "KXTRUTHSOCIAL", "KXTRUTHSOCIALD", "KXVHGCB",
    "KXYTDAILYTOPVIDEO", "KXYTDAILYTOPVIDEOG", "KXYTVIEWSHIGH", "KXYTVIEWSW",
)
CMN_LEAD = timedelta(hours=72)
CMN_MARGIN = 0.12
CMN_BAND = 0.05
CMN_MAX_LEGS = 3
CMN_PREFILTER_ASK = 0.15        # books are read only for legs whose listing ask is at most this

HITS_SERIES = "KXALBUMEQUIV"
HITS_FIT = {"mu": 0.044279564821151765, "sd": 0.06465645776537911}   # fit_dev.json "equiv|preview", frozen 2026-11-20
HITS_THETA = 0.05
HITS_MAX_N = 10
HITS_ORDER_FRAC = 0.04
HITS_EVENT_FRAC = 0.05
HITS_FOLLOW_H = (1, 6, 24)
HITS_LISTINGS = ("/", "/news")

APR_SERIES = "KXTRUMPAPPROVE"
APR_LEAD = timedelta(hours=12)
APR_BAND = (0.05, 0.15)
APR_MAX_SPREAD = 0.10


def iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat()


def parse_t(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s).astimezone(UTC)


def cents_up(x: float) -> float:
    return math.ceil(round(x * 100, 6)) / 100


def _f(v: Any) -> float | None:
    return None if v in (None, "") else float(v)


# ---- market data -------------------------------------------------------------------------------
class Public:
    """Kalshi public market data: unauthenticated GETs, never cached, nothing written to data/raw
    (the ledgers keep the books each decision used)."""

    def __init__(self, client: Any = None) -> None:
        if client is None:
            from albert.config import Settings
            from albert.http import CachedClient

            s = Settings()
            client = CachedClient("kalshi", s.kalshi_base, s.raw_dir, min_interval=0.25, max_retries=4,
                                  record_live=False)
        self.c = client
        self.fees: dict[str, tuple[str | None, float]] = {}

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.c.get(path, params, cache=False).payload or {}

    def _pages(self, path: str, params: dict[str, Any], key: str) -> list[dict[str, Any]]:
        out, cursor = [], None
        while True:
            page = self.get(path, {**params, "cursor": cursor})
            out += page.get(key) or []
            cursor = page.get("cursor")
            if not cursor or not page.get(key):
                return out

    def open_markets(self, series: str) -> list[dict[str, Any]]:
        return self._pages("/markets", {"series_ticker": series, "status": "open", "limit": 1000}, "markets")

    def markets(self, tickers: list[str]) -> list[dict[str, Any]]:
        out = []
        for i in range(0, len(tickers), 100):
            out += self._pages("/markets", {"tickers": ",".join(tickers[i:i + 100]), "limit": 1000}, "markets")
        return out

    def market(self, ticker: str) -> dict[str, Any]:
        return self.get(f"/markets/{ticker}").get("market") or {}

    def book(self, ticker: str) -> dict[str, Any]:
        return self.get(f"/markets/{ticker}/orderbook", {"depth": 10})

    def trades(self, ticker: str, min_ts: float) -> list[dict[str, Any]]:
        return self._pages("/markets/trades", {"ticker": ticker, "limit": 1000, "min_ts": int(min_ts) - 1}, "trades")

    def fee(self, series: str) -> tuple[str | None, float]:
        if series not in self.fees:
            info = self.get(f"/series/{series}").get("series") or {}
            self.fees[series] = (info.get("fee_type"), float(info.get("fee_multiplier") or 1.0))
        return self.fees[series]


def listing_quote(m: dict[str, Any]) -> dict[str, Any]:
    """Top of book from a /markets row (a zero bid or a $1 ask means that side is empty)."""
    from albert.venues.kalshi import _field

    bid, ask = _f(_field(m, "yes_bid")), _f(_field(m, "yes_ask"))
    return {"ticker": m.get("ticker"), "yes_bid": bid if bid and bid > 0 else None,
            "yes_ask": ask if ask is not None and 0 < ask < 1 else None,
            "yes_bid_size": _f(_field(m, "yes_bid_size")), "yes_ask_size": _f(_field(m, "yes_ask_size")),
            "last": _f(_field(m, "last_price")), "status": m.get("status"), "close_time": m.get("close_time")}


def book_top(payload: dict[str, Any]) -> dict[str, float | None]:
    """Best YES bid/ask and displayed sizes from an ``orderbook_fp`` payload (a NO bid at q is a YES ask at 1 - q)."""
    from albert.venues.kalshi import orderbook_top

    top = orderbook_top(payload)
    return {k: top.get(k) for k in ("yes_bid", "yes_bid_size", "yes_ask", "yes_ask_size")}


def no_size_at(payload: dict[str, Any], no_price: float) -> float:
    """Displayed NO-bid size already resting at ``no_price`` (the queue ahead of a new bid there)."""
    book = payload.get("orderbook_fp") or payload.get("orderbook") or {}
    return sum(float(s) for p, s in (book.get("no_dollars") or book.get("no") or []) if abs(float(p) - no_price) < 1e-6)


def two_sided(top: dict[str, Any]) -> bool:
    bid, ask = top.get("yes_bid"), top.get("yes_ask")
    return bid is not None and ask is not None and bid >= 0.01 - 1e-9 and ask <= 0.99 + 1e-9 and ask > bid


def mid_of(top: dict[str, Any]) -> float | None:
    return (top["yes_bid"] + top["yes_ask"]) / 2 if two_sided(top) else None


# ---- rules (pure) ------------------------------------------------------------------------------
def cmn_quote(top: dict[str, Any], margin: float = CMN_MARGIN, band: float = CMN_BAND) -> dict[str, float] | None:
    """The maker NO bid for one leg, or None: two-sided book, mid <= band, sell YES at ceil(mid + margin)."""
    mid = mid_of(top)
    if mid is None or mid > band + 1e-9:
        return None
    s = cents_up(mid + margin)
    if s >= 0.99 or s <= top["yes_bid"] + 1e-9:     # post-only: never at or through the YES bid
        return None
    return {"mid": mid, "sell_at": s, "price": round(1 - s, 2)}


def cmn_rank(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cheapest NO first (ties by ticker): the order in which an event's legs enter the book."""
    return sorted(cands, key=lambda c: (c["price"], c["ticker"]))


def first_through(side: str, price: float, t0: datetime, end: datetime, prints: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first print in (t0, end] that trades strictly through a ``side`` bid at ``price``."""
    for t in sorted(prints, key=lambda p: p["created_time"]):
        ts = parse_t(t["created_time"])
        if t0 < ts <= end and through(side, price, float(t["yes_price_dollars"]), t.get("taker_side", "")):
            return t
    return None


def taker_order_size(price: float, budget: float, cap: int) -> int:
    return max(1, min(cap, math.floor(budget / max(price, 0.01) + 1e-9)))


def taker_pick(p: float, top: dict[str, Any], *, budget: float, max_n: int = HITS_MAX_N, theta: float = HITS_THETA,
               mult: float = 1.0) -> dict[str, Any] | None:
    """Buy YES at the ask or NO at 1 - bid, whichever has the larger edge after the per-order fee,
    sized to the budget, ``max_n`` and the displayed size; None below ``theta``."""
    cands = []
    ask, ask_n = top.get("yes_ask"), int(top.get("yes_ask_size") or 0)
    if ask is not None and 0 < ask < 1 and ask_n >= 1:
        n = min(taker_order_size(ask, budget, max_n), ask_n)
        fee = kalshi_fee(n, ask, multiplier=mult)
        cands.append({"side": "yes", "price": ask, "n": n, "fee": fee, "edge": p - ask - fee / n})
    bid, bid_n = top.get("yes_bid"), int(top.get("yes_bid_size") or 0)
    if bid is not None and 0 < bid < 1 and bid_n >= 1:
        q = round(1 - bid, 4)
        n = min(taker_order_size(q, budget, max_n), bid_n)
        fee = kalshi_fee(n, q, multiplier=mult)
        cands.append({"side": "no", "price": q, "n": n, "fee": fee, "edge": (1 - p) - q - fee / n})
    best = max(cands, key=lambda c: c["edge"], default=None)
    return best if best is not None and best["edge"] >= theta - 1e-9 else None


def cap_event(orders: list[dict[str, Any]], cap_usd: float) -> list[dict[str, Any]]:
    """Best edge first while the event's cumulative cost (n x price + fee) stays within the cap."""
    keep, used = [], 0.0
    for o in sorted(orders, key=lambda o: -o["edge"]):
        c = o["n"] * o["price"] + o["fee"]
        if used + c > cap_usd + 1e-9:
            break
        used += c
        keep.append(o)
    return keep


def hits_prices(table: list[dict[str, Any]], markets: list[dict[str, Any]], week_end: date,
                fit: dict[str, float] = HITS_FIT) -> list[dict[str, Any]]:
    """Model price of every open ladder strike of ``week_end`` whose album matches a row of the HITS
    table (music_charts.match_hits: title score >= 0.6, artist overlap unless the title is exact)."""
    import polars as pl

    from albert.research import music_charts as mc

    rows = []
    for m in markets:
        terms = mc.ladder_terms(m.get("rules_primary"))
        if terms is None or terms[2] != "equiv" or mc.event_date(m.get("event_ticker") or "") != week_end:
            continue
        rows.append({"ticker": m["ticker"], "event_ticker": m["event_ticker"], "album": terms[0], "artist": terms[1],
                     "strike": _f(m.get("floor_strike")), "close_time": m.get("close_time")})
    tab = [r for r in table if (r.get("activity") or 0) > 0]
    if not rows or not tab:
        return []
    mk = pl.DataFrame([{"week_end": week_end, "album": r["album"], "artist": r["artist"]} for r in rows]).unique()
    hits = pl.DataFrame([{"week_end": week_end, "album": r["album"], "artist": r["artist"],
                          "activity": float(r["activity"])} for r in tab])
    mt = mc.match_hits(mk, hits)
    act = {(a, art): v for a, art, v in hits.select("album", "artist", "activity").iter_rows()}
    key = {(r["album"], r["artist"]): (r["h_album"], r["h_artist"], r["match_score"]) for r in mt.iter_rows(named=True)}
    out = []
    for r in rows:
        k = key.get((r["album"], r["artist"]))
        if k is None or not r["strike"]:
            continue
        hval = act[(k[0], k[1])]
        out.append({**r, "h_album": k[0], "h_artist": k[1], "match_score": k[2], "activity": hval,
                    "p_model": mc.p_above(r["strike"], hval, fit["mu"], fit["sd"])})
    return out


def apr_pick(top: dict[str, Any], mult: float = 1.0) -> dict[str, Any] | None:
    """1 NO at 1 - best YES bid when 5c < mid <= 15c, spread <= 10c and the bid shows >= 1 contract."""
    bid, ask = top.get("yes_bid"), top.get("yes_ask")
    if bid is None or ask is None or not 0 < bid < ask < 1 or (top.get("yes_bid_size") or 0) < 1:
        return None
    mid = (bid + ask) / 2
    if not APR_BAND[0] + 1e-9 < mid <= APR_BAND[1] + 1e-9 or ask - bid > APR_MAX_SPREAD + 1e-9:
        return None
    q = round(1 - bid, 4)
    return {"side": "no", "price": q, "n": 1, "fee": kalshi_fee(1, q, multiplier=mult), "mid": mid}


def settle_instant(close: datetime) -> datetime:
    """KXTRUMPAPPROVE legs close at 12:59 ET; settlement is the next whole hour (approval2.settle_instant)."""
    if close.minute == 0 and close.second == 0:
        return close
    return close.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def hits_fast_window(t: datetime) -> bool:
    e = t.astimezone(ET)
    return e.weekday() == 1 and 13 <= e.hour < 21


# ---- ledger --------------------------------------------------------------------------------------
class Book:
    """One rule's append-only ledger and the state replayed from it.

    Kinds: ``order`` (id, event, ticker, side, price, n, fee, t0, end, in_book, ...), ``fill`` (id, ts,
    print), ``end`` (id, ts, reason), ``settle`` (ticker, result, ts), ``decision`` (key, ...) and free-form
    diagnostics (``post``, ``quote``, ``seed``, ``error``)."""

    def __init__(self, name: str, root: Path = OUT, *, bankroll: float = BANKROLL,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.name = name
        self.bankroll = bankroll
        self.path = root / name / "ledger.jsonl"
        self.clock = clock or (lambda: datetime.now(UTC))
        self.orders: dict[str, dict[str, Any]] = {}
        self.settled: dict[str, dict[str, Any]] = {}
        self.decided: set[str] = set()
        self.records: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    def log(self, kind: str, **rec: Any) -> dict[str, Any]:
        r = {"kind": kind, "logged": iso(self.clock()), **rec}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")
        self._apply(r)
        return r

    def _apply(self, r: dict[str, Any]) -> None:
        self.records.append(r)
        k = r["kind"]
        if k == "order":
            self.orders[r["id"]] = {**r, "fill": None, "end_reason": None}
        elif k == "fill" and r["id"] in self.orders:
            self.orders[r["id"]]["fill"] = r
        elif k == "end" and r["id"] in self.orders:
            o = self.orders[r["id"]]
            if parse_t(r["ts"]) < parse_t(o["end"]):
                o["end"], o["end_reason"] = r["ts"], r["reason"]
        elif k == "settle":
            self.settled[r["ticker"]] = r
        elif k == "decision":
            self.decided.add(r["key"])

    def resting(self, t: datetime) -> list[dict[str, Any]]:
        return [o for o in self.orders.values() if o["fill"] is None and parse_t(o["end"]) > t]

    def open_positions(self) -> list[dict[str, Any]]:
        return [o for o in self.orders.values() if o["fill"] is not None and o["ticker"] not in self.settled]

    def cash_used(self, t: datetime) -> float:
        """Collateral of booked resting bids plus the cost of booked unsettled positions."""
        rest = sum(o["n"] * o["price"] for o in self.resting(t) if o["in_book"])
        pos = sum(o["n"] * o["price"] + o["fee"] for o in self.open_positions() if o["in_book"])
        return rest + pos


# ---- the rules on live data -------------------------------------------------------------------------
class Runner:
    def __init__(self, api: Public | None = None, root: Path = OUT, *, bankroll: float = BANKROLL,
                 clock: Callable[[], datetime] | None = None, log: Callable[[str], None] = print,
                 http: Any = None) -> None:
        self.api = api or Public()
        self.root = root
        self.clock = clock or (lambda: datetime.now(UTC))
        self.say = log
        self.http = http
        self.books = {n: Book(n, root, bankroll=bankroll, clock=self.clock) for n in ("cmn", "hits", "apr")}
        self.polled: dict[str, datetime] = {}

    def now(self) -> datetime:
        return self.clock()

    def _order(self, b: Book, *, event: str, ticker: str, side: str, price: float, n: int, fee: float, t0: datetime,
               end: str, in_book: bool, **extra: Any) -> dict[str, Any]:
        oid = f"{b.name}-{ticker}-{t0:%Y%m%dT%H%M%S}"
        return b.log("order", id=oid, event=event, ticker=ticker, side=side, price=price, n=n, fee=fee,
                     t0=iso(t0), end=end, in_book=in_book, **extra)

    # -- cmn --
    def cmn_scan(self) -> None:
        b, now = self.books["cmn"], self.now()
        open_by_ticker: dict[str, dict[str, Any]] = {}
        events: dict[str, list[dict[str, Any]]] = {}
        listed: set[str] = set()
        for s in CMN_SERIES:
            try:
                ms = self.api.open_markets(s)
            except Exception as error:  # noqa: BLE001 - one series failing must not stop the scan
                self.say(f"cmn list {s}: {error}")
                continue
            listed.add(s)
            for m in ms:
                open_by_ticker[m["ticker"]] = m
                events.setdefault(m["event_ticker"], []).append(m)
        # end resting orders whose leg left the open listing or whose event's close moved
        for o in b.resting(now):
            if o["series"] not in listed:
                continue
            m = open_by_ticker.get(o["ticker"])
            ev = events.get(o["event"])
            if m is None:
                b.log("end", id=o["id"], ts=iso(now), reason="left open listing")
            elif ev and abs((max(parse_t(x["close_time"]) for x in ev) - parse_t(o["ev_close"])).total_seconds()) > 60:
                b.log("end", id=o["id"], ts=iso(now), reason="event close moved")
        for ev_t, legs in sorted(events.items()):
            if ev_t in b.decided:
                continue
            close = max(parse_t(m["close_time"]) for m in legs)
            t_dec = close - CMN_LEAD
            if not t_dec <= now < t_dec + DECIDE_WINDOW:
                continue
            self._cmn_decide(b, ev_t, legs, close, now)

    def _cmn_decide(self, b: Book, ev_t: str, legs: list[dict[str, Any]], close: datetime, now: datetime) -> None:
        series = legs[0]["event_ticker"].split("-")[0]
        fee_type, mult = self.api.fee(series)
        seen, cands = [], []
        for m in legs:
            q = listing_quote(m)
            row = {"ticker": m["ticker"], "listing": q}
            if q["yes_ask"] is not None and q["yes_ask"] <= CMN_PREFILTER_ASK:
                ob = self.api.book(m["ticker"])
                top = book_top(ob)
                row["book"] = top
                quote = cmn_quote(top)
                if quote is not None:
                    quote["ahead"] = no_size_at(ob, quote["price"])
                    cands.append({"ticker": m["ticker"], "close_time": m["close_time"], **quote, "top": top})
                    row["quote"] = quote
                else:
                    row["one_sided_cheap"] = (not two_sided(top) and top.get("yes_ask") is not None
                                              and top["yes_ask"] <= 2 * CMN_BAND)
            seen.append(row)
        placed = 0
        for c in cmn_rank(cands):
            fee = kalshi_fee(1, c["price"], maker=True, fee_type=fee_type, multiplier=mult)
            in_book = placed < CMN_MAX_LEGS and b.cash_used(now) + c["price"] <= b.bankroll + 1e-9
            placed += in_book
            self._order(b, event=ev_t, ticker=c["ticker"], side="no", price=c["price"], n=1, fee=fee, t0=now,
                        end=c["close_time"], in_book=in_book, series=series, ev_close=iso(close), mid=c["mid"],
                        sell_at=c["sell_at"], ahead=c["ahead"], top=c["top"], fee_type=fee_type,
                        close_time=c["close_time"])
        b.log("decision", key=ev_t, ts=iso(now), series=series, ev_close=iso(close), legs=len(legs),
              candidates=len(cands), booked=placed, seen=seen)
        if cands:
            self.say(f"cmn {ev_t}: {len(cands)} qualifying legs, {placed} booked")

    def cmn_match(self) -> None:
        b, now = self.books["cmn"], self.now()
        live = [o for o in b.orders.values() if o["fill"] is None
                and parse_t(o["end"]) > self.polled.get(o["ticker"], parse_t(o["t0"]))]
        for ticker in sorted({o["ticker"] for o in live}):
            mine = [o for o in live if o["ticker"] == ticker]
            since = self.polled.get(ticker, min(parse_t(o["t0"]) for o in mine))
            try:
                prints = self.api.trades(ticker, since.timestamp())
            except Exception as error:  # noqa: BLE001
                self.say(f"cmn trades {ticker}: {error}")
                continue
            self.polled[ticker] = now
            for o in mine:
                end = min(parse_t(o["end"]), now)
                hit = first_through("no", o["price"], parse_t(o["t0"]), end, prints)
                if hit is not None:
                    after = sum(float(p.get("count_fp") or 0) for p in prints
                                if parse_t(p["created_time"]) >= parse_t(hit["created_time"])
                                and through("no", o["price"], float(p["yes_price_dollars"]), p.get("taker_side", "")))
                    b.log("fill", id=o["id"], ts=hit["created_time"], print_id=hit["trade_id"],
                          print_yes=float(hit["yes_price_dollars"]), print_count=_f(hit.get("count_fp")),
                          through_count_so_far=after)
                    self.say(f"cmn fill {o['ticker']} NO at {o['price']:.2f} ({'book' if o['in_book'] else 'shadow'})")

    # -- hits --
    def _http(self):
        if self.http is None:
            import httpx

            from albert.research.music_charts import UA

            self.http = httpx.Client(timeout=60, headers={"User-Agent": UA}, follow_redirects=True)
        return self.http

    def hits_poll(self) -> None:
        from albert.research import music_charts as mc

        b, now = self.books["hits"], self.now()
        seen = {r["slug"] for r in b.records if r["kind"] in ("post", "seed")}
        slugs: set[str] = set()
        for path in HITS_LISTINGS:
            r = self._http().get(f"{mc.HITS}{path}")
            if r.status_code == 200:
                slugs |= set(mc.hits_listing_slugs(r.text))
            time.sleep(1)
        first_run = not any(r["kind"] in ("post", "seed") for r in b.records)
        for slug in sorted(slugs - seen):
            if first_run:
                b.log("seed", slug=slug, ts=iso(now))
                continue
            r = self._http().get(f"{mc.HITS}/news/charts/{slug}")
            if r.status_code != 200:
                continue
            seen_t = self.now()
            raw = self.root / "hits" / "raw" / f"{slug}_{seen_t:%Y%m%dT%H%M%SZ}.html.gz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(gzip.compress(r.content))
            d = mc.parse_hits_post(r.text, slug)
            pub = parse_t(d["published_at"])
            kind = d["kind"] if not re.search(mc.NON_ALBUM_TABLES, slug) else "non_album"
            week = mc.week_end_for_post(pub or seen_t, kind) if kind in ("preview", "final") else None
            b.log("post", slug=slug, ts=iso(seen_t), published_at=d["published_at"], post_kind=kind, title=d["title"],
                  week_end=str(week) if week else None, items=d["items"], raw=str(raw),
                  lag_s=(seen_t - pub).total_seconds() if pub else None)
            self.say(f"hits post {slug}: {kind}, week {week}, {len(d['items'])} rows")
            if kind == "preview" and d["items"] and week is not None and f"week:{week}" not in b.decided:
                self._hits_decide(b, slug, d["items"], week)
            time.sleep(1)
        if first_run:
            # previews already up at the first run are not first sightings: their weeks are skipped
            for r in b.records:
                if r["kind"] == "seed":
                    m = re.search(r"-(20\d\d-\d\d-\d\d)$", r["slug"])
                    if m and re.search(r"preview|projection", r["slug"]):
                        wk = mc.week_end_for_post(datetime.fromisoformat(m.group(1)).replace(hour=20, tzinfo=UTC), "preview")
                        if f"week:{wk}" not in b.decided:
                            b.log("decision", key=f"week:{wk}", ts=iso(now), skipped="preview already up at first run")

    def _hits_decide(self, b: Book, slug: str, table: list[dict[str, Any]], week: date) -> None:
        now = self.now()
        markets = self.api.open_markets(HITS_SERIES)
        priced = hits_prices(table, markets, week)
        _, mult = self.api.fee(HITS_SERIES)
        budget, ev_cap = HITS_ORDER_FRAC * b.bankroll, HITS_EVENT_FRAC * b.bankroll
        by_ev: dict[str, list[dict[str, Any]]] = {}
        for r in priced:
            top = book_top(self.api.book(r["ticker"]))
            r["top"] = top
            pick = taker_pick(r["p_model"], top, budget=budget, mult=mult)
            if pick is not None:
                by_ev.setdefault(r["event_ticker"], []).append({**r, **pick})
        n_orders = 0
        for ev_t, cands in sorted(by_ev.items()):
            for o in cap_event(cands, ev_cap):
                in_book = b.cash_used(now) + o["n"] * o["price"] + o["fee"] <= b.bankroll + 1e-9
                rec = self._order(b, event=ev_t, ticker=o["ticker"], side=o["side"], price=o["price"], n=o["n"],
                                  fee=o["fee"], t0=now, end=iso(now), in_book=in_book, edge=o["edge"],
                                  p_model=o["p_model"], top=o["top"], slug=slug, week_end=str(week),
                                  close_time=o["close_time"], activity=o["activity"])
                b.log("fill", id=rec["id"], ts=iso(now), taker=True)
                n_orders += 1
        b.log("decision", key=f"week:{week}", ts=iso(now), slug=slug, priced=priced, orders=n_orders,
              follow=[iso(now + timedelta(hours=h)) for h in HITS_FOLLOW_H])
        self.say(f"hits week {week}: {len(priced)} strikes priced, {n_orders} orders")

    def hits_follow(self) -> None:
        b, now = self.books["hits"], self.now()
        done = {(r["key"], r["label"]) for r in b.records if r["kind"] == "quote"}
        for d in [r for r in b.records if r["kind"] == "decision" and r.get("priced")]:
            for h, due in zip(HITS_FOLLOW_H, d["follow"], strict=True):
                label = f"+{h}h"
                if (d["key"], label) in done or parse_t(due) > now:
                    continue
                qs = [listing_quote(m) for m in self.api.markets([r["ticker"] for r in d["priced"]])]
                b.log("quote", key=d["key"], label=label, ts=iso(now), quotes=qs)

    # -- apr --
    def apr_scan(self) -> None:
        b, now = self.books["apr"], self.now()
        events: dict[str, list[dict[str, Any]]] = {}
        for m in self.api.open_markets(APR_SERIES):
            events.setdefault(m["event_ticker"], []).append(m)
        for ev_t, legs in sorted(events.items()):
            if ev_t in b.decided:
                continue
            settle = settle_instant(max(parse_t(m["close_time"]) for m in legs))
            t_dec = settle - APR_LEAD
            if not t_dec <= now < t_dec + DECIDE_WINDOW:
                continue
            _, mult = self.api.fee(APR_SERIES)
            seen, n = [], 0
            for m in sorted(legs, key=lambda m: m["ticker"]):
                top = book_top(self.api.book(m["ticker"]))
                pick = apr_pick(top, mult)
                seen.append({"ticker": m["ticker"], "top": top, "pick": pick is not None})
                if pick is None:
                    continue
                in_book = b.cash_used(now) + pick["price"] + pick["fee"] <= b.bankroll + 1e-9
                rec = self._order(b, event=ev_t, ticker=m["ticker"], side="no", price=pick["price"], n=1, fee=pick["fee"],
                                  t0=now, end=iso(now), in_book=in_book, mid=pick["mid"], top=top,
                                  close_time=m["close_time"])
                b.log("fill", id=rec["id"], ts=iso(now), taker=True)
                n += 1
            b.log("decision", key=ev_t, ts=iso(now), settle=iso(settle), legs=len(legs), orders=n, seen=seen)
            self.say(f"apr {ev_t}: {n} NO orders")

    # -- shared --
    def settle(self) -> None:
        now = self.now()
        for b in self.books.values():
            for o in b.open_positions():
                close = parse_t(o.get("close_time"))
                if close is not None and close > now:
                    continue
                if o["ticker"] in b.settled:
                    continue
                m = self.api.market(o["ticker"])
                if m.get("result") in ("yes", "no"):
                    b.log("settle", ticker=o["ticker"], result=m["result"], ts=iso(now),
                          settled_at=m.get("settlement_ts") or m.get("expiration_time") or iso(now))

    def marks(self) -> None:
        now = self.now()
        tickers = sorted({o["ticker"] for b in self.books.values()
                          for o in b.resting(now) + b.open_positions()})
        if not tickers:
            return
        rows = [{"ts": iso(now), **listing_quote(m)} for m in self.api.markets(tickers)]
        path = self.root / "marks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


# ---- loop --------------------------------------------------------------------------------------
def run(root: Path = OUT, *, bankroll: float = BANKROLL, log: Callable[[str], None] = print) -> None:
    stop = root / "STOP"
    root.mkdir(parents=True, exist_ok=True)
    stop.unlink(missing_ok=True)
    term = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: term.__setitem__("flag", True))
    r = Runner(root=root, bankroll=bankroll, log=lambda s: log(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {s}"))
    tasks: list[tuple[str, Callable[[], None], Callable[[datetime], float]]] = [
        ("cmn.scan", r.cmn_scan, lambda t: 600),
        ("cmn.match", r.cmn_match, lambda t: 600),
        ("apr.scan", r.apr_scan, lambda t: 300),
        ("hits.poll", r.hits_poll, lambda t: 300 if hits_fast_window(t) else 1800),
        ("hits.follow", r.hits_follow, lambda t: 300),
        ("settle", r.settle, lambda t: 3600),
        ("marks", r.marks, lambda t: 3600),
    ]
    due = {name: 0.0 for name, _, _ in tasks}
    r.say(f"paper_forward start: bankroll ${bankroll:.0f} per rule, ledgers under {root}")
    while not stop.exists() and not term["flag"]:
        for name, fn, every in tasks:
            if time.time() < due[name] or stop.exists() or term["flag"]:
                continue
            try:
                fn()
            except Exception as error:  # noqa: BLE001 - one task failing must not stop the others
                r.say(f"{name} failed: {type(error).__name__}: {str(error)[:300]}")
            due[name] = time.time() + every(datetime.now(UTC))
        for _ in range(12):
            if stop.exists() or term["flag"]:
                break
            time.sleep(5)
    r.say("paper_forward stop")


# ---- scoring -----------------------------------------------------------------------------------
def load_marks(root: Path = OUT):
    """Hourly YES marks from ``marks.jsonl``: the mid of a real two-sided book, else the last price."""
    import polars as pl

    path = root / "marks.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    out = []
    for r in rows:
        mid = mid_of(r)
        v = mid if mid is not None else r.get("last")
        if v is not None:
            out.append({"ticker": r["ticker"], "ts": parse_t(r["ts"]), "yes_mid": float(v)})
    return pl.DataFrame(out, schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "yes_mid": pl.Float64})


def positions(b: Book) -> list[dict[str, Any]]:
    """One row per filled order: cost, payoff (None until settled), PnL per contract."""
    out = []
    for o in b.orders.values():
        if o["fill"] is None:
            continue
        s = b.settled.get(o["ticker"])
        won = None if s is None else float((s["result"] == "yes") == (o["side"] == "yes"))
        cost = o["n"] * o["price"] + o["fee"]
        out.append({"id": o["id"], "event": o["event"], "ticker": o["ticker"], "side": o["side"], "n": o["n"],
                    "price": o["price"], "fee": o["fee"], "in_book": o["in_book"], "t_in": parse_t(o["fill"]["ts"]),
                    "cost": cost, "payoff": None if won is None else o["n"] * won, "won": won,
                    "settle_ts": parse_t(s["settled_at"]) if s else None,
                    "pnl_c": None if won is None else 100 * (won - o["price"] - o["fee"] / o["n"])})
    return out


def summary(b: Book, root: Path = OUT, *, book_only: bool = False) -> dict[str, Any]:
    import numpy as np
    import polars as pl

    from albert.research.counts import cluster_se
    from albert.research.vol_metrics import ledger_metrics

    now = b.clock()
    orders = [o for o in b.orders.values() if o["in_book"] or not book_only]
    pos = [p for p in positions(b) if p["in_book"] or not book_only]
    done = [p for p in pos if p["payoff"] is not None]
    out: dict[str, Any] = {"orders": len(orders), "fills": len(pos), "settled": len(done),
                           "resting": len([o for o in b.resting(now) if o in orders]),
                           "cash_used_now": round(b.cash_used(now), 2), "decisions": len(b.decided)}
    if not done:
        return out
    x = np.array([p["pnl_c"] for p in done]) / 100
    se = cluster_se(x, np.array([p["event"] for p in done])) if len(done) > 1 else None
    out.update({"events": len({p["event"] for p in done}), "c_per_contract": round(100 * float(x.mean()), 2),
                "t_event": round(float(x.mean()) / se, 2) if se else None,
                "losing_fills": int(sum(p["won"] == 0 for p in done)),
                "pnl_usd": round(sum(p["payoff"] - p["cost"] for p in done), 2),
                "pnl_per_usd": round(sum(p["payoff"] - p["cost"] for p in done) / sum(p["cost"] for p in done), 3)})
    led = pl.DataFrame([{"struct_id": p["id"], "event_ticker": p["event"], "ticker": p["ticker"], "side": p["side"],
                         "n": p["n"], "t_in": p["t_in"], "cost_usd": p["cost"],
                         "settle_ts": max(p["settle_ts"], p["t_in"]), "payoff_usd": p["payoff"]}
                        for p in done if p["in_book"]])
    if led.height >= 2:
        m = ledger_metrics(led, load_marks(root), bankroll=b.bankroll)
        out["book"] = {k: m.get(k) for k in ("positions", "usd_per_week", "ann_vol_pct", "sharpe", "sharpe_ci90",
                                             "sharpe_ex_top5", "max_dd_usd", "weeks_positive_pct", "first_day",
                                             "last_day", "marked_share")}
    return out


def report(root: Path = OUT, *, bankroll: float = BANKROLL) -> dict[str, Any]:
    res = {}
    for name in ("cmn", "hits", "apr"):
        b = Book(name, root, bankroll=bankroll)
        res[name] = {"all": summary(b, root), "book": summary(b, root, book_only=True)}
        if name == "hits":
            res[name]["weeks"] = sorted(r["key"] for r in b.records if r["kind"] == "decision")
            res[name]["first_sight_lag_s"] = [r["lag_s"] for r in b.records
                                              if r["kind"] == "post" and r.get("post_kind") == "preview"]
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m albert.paper_forward")
    ap.add_argument("cmd", choices=("run", "report", "once"))
    ap.add_argument("--bankroll", type=float, default=BANKROLL, help="paper bankroll per rule ($)")
    ap.add_argument("--root", default=str(OUT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == "run":
        run(root, bankroll=a.bankroll)
    elif a.cmd == "once":
        r = Runner(root=root, bankroll=a.bankroll)
        for fn in (r.cmn_scan, r.cmn_match, r.apr_scan, r.hits_poll, r.hits_follow, r.settle, r.marks):
            fn()
    else:
        print(json.dumps(report(root, bankroll=a.bankroll), indent=1, default=str))


if __name__ == "__main__":
    main()
