"""Kalshi public market data: series, markets, trades, candlesticks, order books.

Field names follow the live API as recorded in tests/fixtures/kalshi (2026-11). Prices come back
as ``*_dollars`` strings and sizes as ``*_fp`` strings. Historical candlesticks drop both
suffixes, so each parser accepts either spelling.
"""

from collections.abc import Iterator
import re
from zoneinfo import ZoneInfo
from datetime import UTC, datetime
from typing import Any

import polars as pl

from albert.categories import kalshi_is_non_financial
from albert.http import CachedClient, Fetched

MARKET_SCHEMA = {
    "ticker": pl.String,
    "event_ticker": pl.String,
    "series_ticker": pl.String,
    "category": pl.String,
    "title": pl.String,
    "yes_sub_title": pl.String,
    "market_type": pl.String,
    "status": pl.String,
    "open_time": pl.Datetime("us", "UTC"),
    "close_time": pl.Datetime("us", "UTC"),
    "occurrence_time": pl.Datetime("us", "UTC"),
    # Scheduled at listing; unlike close_time it does not move when the market closes early.
    "expected_expiration_time": pl.Datetime("us", "UTC"),
    "can_close_early": pl.Boolean,
    "settlement_ts": pl.Datetime("us", "UTC"),
    "result": pl.String,
    "settlement_value": pl.Float64,
    "last_price": pl.Float64,
    "yes_bid": pl.Float64,
    "yes_ask": pl.Float64,
    "volume": pl.Float64,
    "open_interest": pl.Float64,
    "rules_primary": pl.String,
    "fee_type": pl.String,
    "fee_multiplier": pl.Float64,
    "source": pl.String,
}

TRADE_SCHEMA = {
    "trade_id": pl.String,
    "ticker": pl.String,
    "created_time": pl.Datetime("us", "UTC"),
    "count": pl.Float64,
    "yes_price": pl.Float64,
    "taker_side": pl.String,
}

CANDLE_SCHEMA = {
    "ticker": pl.String,
    "end_ts": pl.Datetime("us", "UTC"),
    "period_minutes": pl.Int64,
    "yes_bid_close": pl.Float64,
    "yes_ask_close": pl.Float64,
    "price_close": pl.Float64,
    "price_mean": pl.Float64,
    "volume": pl.Float64,
    "open_interest": pl.Float64,
}


def parse_ts(value: Any) -> datetime | None:
    if value in (None, "", "0001-01-01T00:00:00Z"):
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _field(node: dict[str, Any], name: str) -> Any:
    """Read ``name`` from a node that may use ``name_dollars`` / ``name_fp`` / ``name``."""
    for key in (f"{name}_dollars", f"{name}_fp", name):
        if key in node:
            return node[key]
    return None


def series_of(event_ticker: str) -> str:
    return event_ticker.split("-", 1)[0]


class Kalshi:
    def __init__(self, client: CachedClient) -> None:
        self.client = client

    # ---- raw fetchers -------------------------------------------------------------------
    def cutoff(self) -> dict[str, datetime | None]:
        page = self.client.get("/historical/cutoff", cache=False)
        return {k: parse_ts(v) for k, v in page.payload.items()}

    def series(self) -> Fetched:
        # Returns every series in one response (~16k rows). Refreshed at most daily.
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        return self.client.get("/series", tag=day)

    def _cursor_pages(self, path: str, params: dict[str, Any], key: str,
                      cache: bool = True, max_pages: int | None = None) -> Iterator[Fetched]:
        cursor = None
        pages = 0
        while True:
            page = self.client.get(path, {**params, "cursor": cursor}, cache=cache)
            yield page
            pages += 1
            cursor = page.payload.get("cursor")
            if not cursor or not page.payload.get(key) or (max_pages and pages >= max_pages):
                return

    def settled_markets(self, start: datetime, end: datetime) -> Iterator[Fetched]:
        """Settled single-leg markets that closed in [start, end), routed by the cutoff.

        The live ``/markets`` endpoint honours ``min_close_ts``/``max_close_ts``. The historical
        endpoint ignores them but returns newest-first, so we page until we pass ``start``.
        """
        cutoff = self.cutoff()["market_settled_ts"]
        if cutoff is None or end > cutoff:
            live_start = max(start, cutoff) if cutoff else start
            yield from self._cursor_pages(
                "/markets",
                {
                    "status": "settled",
                    "mve_filter": "exclude",
                    "limit": 1000,
                    "min_close_ts": int(live_start.timestamp()),
                    "max_close_ts": int(end.timestamp()),
                },
                "markets",
            )
        if cutoff is not None and start < cutoff:
            for page in self._cursor_pages(
                "/historical/markets", {"mve_filter": "exclude", "limit": 1000}, "markets"
            ):
                yield page
                closes = [parse_ts(m.get("close_time")) for m in page.payload.get("markets", [])]
                closes = [c for c in closes if c is not None]
                if closes and min(closes) < start:
                    return

    def open_markets(self) -> Iterator[Fetched]:
        yield from self._cursor_pages(
            "/markets", {"status": "open", "mve_filter": "exclude", "limit": 1000}, "markets",
            cache=False,
        )

    def trades(self, ticker: str, historical: bool, max_pages: int | None = None,
               min_ts: datetime | None = None, max_ts: datetime | None = None) -> Iterator[Fetched]:
        """Trades newest-first; ``max_pages`` truncates very active markets to their final trades.

        ``min_ts``/``max_ts`` restrict to a time window (both endpoints honour them), which gives
        complete coverage of that window regardless of how busy the market is.
        """
        path = "/historical/trades" if historical else "/markets/trades"
        params: dict[str, Any] = {"ticker": ticker, "limit": 1000}
        if min_ts is not None:
            params["min_ts"] = int(min_ts.timestamp())
        if max_ts is not None:
            params["max_ts"] = int(max_ts.timestamp())
        yield from self._cursor_pages(path, params, "trades", max_pages=max_pages)

    def candles(self, ticker: str, start: datetime, end: datetime, period_minutes: int,
                historical: bool) -> Fetched:
        if historical:
            path = f"/historical/markets/{ticker}/candlesticks"
        else:
            path = f"/series/{series_of(ticker)}/markets/{ticker}/candlesticks"
        return self.client.get(path, {
            "start_ts": int(start.timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": period_minutes,
        })

    def candles_batch(self, tickers: list[str], start: datetime, end: datetime,
                      period_minutes: int) -> Fetched:
        """Live (post-cutoff) candles for many markets in one request: ``{"markets": [{
        "market_ticker", "candlesticks"}]}``. Unknown tickers are silently omitted."""
        return self.client.get("/markets/candlesticks", {
            "market_tickers": ",".join(sorted(tickers)),
            "start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()),
            "period_interval": period_minutes,
        })

    def event(self, event_ticker: str) -> Fetched:
        return self.client.get(f"/events/{event_ticker}")

    def orderbook(self, ticker: str, depth: int = 10) -> Fetched:
        return self.client.get(f"/markets/{ticker}/orderbook", {"depth": depth}, cache=False)


# ---- normalizers ----------------------------------------------------------------------------
_EASTERN = ZoneInfo("America/New_York")
_MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
_START_RX = re.compile(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})(\d{4})(?=[A-Z])")


def ticker_start(event_ticker: str | None) -> datetime | None:
    """Scheduled start encoded in many game tickers, in US Eastern time.

    ``KXKBOGAME-26NOV040430SAMKTW`` -> 2026-11-04 04:30 ET (09:30 UTC). Validated on ~80k
    settled games: the market closes a game's length after this time (1.4% close before it,
    i.e. postponements). Tickers without a time (``KXNBAGAME-26NOV06SASSAC``) return None.
    """
    m = _START_RX.match(event_ticker or "")
    if not m:
        return None
    yy, mon, dd, hm = m.groups()
    hour, minute = int(hm[:2]), int(hm[2:])
    if mon not in _MONTHS or hour > 23 or minute > 59:
        return None
    try:
        local = datetime(2000 + int(yy), _MONTHS[mon], int(dd), hour, minute, tzinfo=_EASTERN)
    except ValueError:
        return None
    return local.astimezone(UTC)


EVENT_SCHEMA = {"event_ticker": pl.String, "series_ticker": pl.String, "category": pl.String,
                "title": pl.String, "mutually_exclusive": pl.Boolean, "n_markets": pl.Int64}


def event_row(payload: dict[str, Any]) -> dict[str, Any]:
    """One row per event; ``n_markets`` counts every leg the venue lists, settled or not."""
    e = payload.get("event") or {}
    return {
        "event_ticker": e.get("event_ticker"), "series_ticker": e.get("series_ticker"),
        "category": e.get("category"), "title": e.get("title"),
        "mutually_exclusive": e.get("mutually_exclusive"),
        "n_markets": len(payload.get("markets") or e.get("markets") or []),
    }


def series_frame(payload: dict[str, Any]) -> pl.DataFrame:
    rows = [
        {
            "series_ticker": s.get("ticker"),
            "category": s.get("category"),
            "title": s.get("title"),
            "frequency": s.get("frequency"),
            "fee_type": s.get("fee_type"),
            "fee_multiplier": num(s.get("fee_multiplier")),
        }
        for s in payload.get("series", [])
    ]
    return pl.DataFrame(rows, schema={
        "series_ticker": pl.String, "category": pl.String, "title": pl.String,
        "frequency": pl.String, "fee_type": pl.String, "fee_multiplier": pl.Float64,
    })


def market_rows(payload: dict[str, Any], series_info: dict[str, dict[str, Any]],
                source: str) -> list[dict[str, Any]]:
    rows = []
    for m in payload.get("markets", []):
        if m.get("mve_collection_ticker") or m.get("mve_selected_legs"):
            continue  # multi-leg combos (parlays)
        series = series_of(m.get("event_ticker") or m["ticker"])
        info = series_info.get(series, {})
        rows.append({
            "ticker": m["ticker"],
            "event_ticker": m.get("event_ticker"),
            "series_ticker": series,
            "category": info.get("category"),
            "title": m.get("title"),
            "yes_sub_title": m.get("yes_sub_title"),
            "market_type": m.get("market_type"),
            "status": m.get("status"),
            "open_time": parse_ts(m.get("open_time")),
            "close_time": parse_ts(m.get("close_time")),
            "occurrence_time": parse_ts(m.get("occurrence_datetime")),
            "expected_expiration_time": parse_ts(m.get("expected_expiration_time")),
            "can_close_early": m.get("can_close_early"),
            "settlement_ts": parse_ts(m.get("settlement_ts")),
            "result": m.get("result") or None,
            "settlement_value": num(_field(m, "settlement_value")),
            "last_price": num(_field(m, "last_price")),
            "yes_bid": num(_field(m, "yes_bid")),
            "yes_ask": num(_field(m, "yes_ask")),
            "volume": num(_field(m, "volume")),
            "open_interest": num(_field(m, "open_interest")),
            "rules_primary": m.get("rules_primary"),
            "fee_type": info.get("fee_type"),
            "fee_multiplier": info.get("fee_multiplier"),
            "source": source,
        })
    return rows


def markets_frame(rows: list[dict[str, Any]], *, non_financial_only: bool = True) -> pl.DataFrame:
    frame = pl.DataFrame(rows, schema=MARKET_SCHEMA)
    if non_financial_only:
        mask = [kalshi_is_non_financial(c) for c in frame["category"].to_list()]
        frame = frame.filter(pl.Series(mask, dtype=pl.Boolean))
    return frame


def trade_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "trade_id": t["trade_id"],
            "ticker": t["ticker"],
            "created_time": parse_ts(t.get("created_time")),
            "count": num(_field(t, "count")),
            "yes_price": num(_field(t, "yes_price")),
            "taker_side": t.get("taker_side"),
        }
        for t in payload.get("trades", [])
    ]


def candle_rows(payload: dict[str, Any], ticker: str, period_minutes: int) -> list[dict[str, Any]]:
    rows = []
    for c in payload.get("candlesticks", []):
        bid = c.get("yes_bid") or {}
        ask = c.get("yes_ask") or {}
        price = c.get("price") or {}
        rows.append({
            "ticker": ticker,
            "end_ts": parse_ts(c.get("end_period_ts")),
            "period_minutes": period_minutes,
            "yes_bid_close": num(_field(bid, "close")),
            "yes_ask_close": num(_field(ask, "close")),
            "price_close": num(_field(price, "close")),
            "price_mean": num(_field(price, "mean")),
            "volume": num(_field(c, "volume")),
            "open_interest": num(_field(c, "open_interest")),
        })
    return rows


def orderbook_top(payload: dict[str, Any]) -> dict[str, float | None]:
    """Best yes bid/ask and displayed size from an ``orderbook_fp`` payload.

    Kalshi only shows bids. A NO bid at q is a YES ask at 1 - q.
    """
    book = payload.get("orderbook_fp") or payload.get("orderbook") or {}
    yes = [(float(p), float(s)) for p, s in (book.get("yes_dollars") or book.get("yes") or [])]
    no = [(float(p), float(s)) for p, s in (book.get("no_dollars") or book.get("no") or [])]
    best_yes = max(yes, default=None)
    best_no = max(no, default=None)
    return {
        "yes_bid": best_yes[0] if best_yes else None,
        "yes_bid_size": best_yes[1] if best_yes else None,
        "yes_ask": round(1 - best_no[0], 4) if best_no else None,
        "yes_ask_size": best_no[1] if best_no else None,
    }
