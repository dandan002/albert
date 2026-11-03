"""Polymarket US (CFTC-regulated, docs.polymarket.us): public market data from the gateway.

Each market has two instrument sides. ``long=True`` is the side quoted as ``longPx`` (for yes/no
markets that is "Yes"; for moneylines it is the first team). We treat the long side as YES
throughout, so a settlement of 1 means the long side won.
"""

from collections.abc import Iterator
from typing import Any

import polars as pl

from albert.http import CachedClient, Fetched
from albert.venues.kalshi import parse_ts
from albert.venues.polymarket import _json_list

MARKET_SCHEMA = {
    "slug": pl.String,
    "market_id": pl.String,
    "question": pl.String,
    "title": pl.String,
    "category": pl.String,
    "market_type": pl.String,
    "long_side": pl.String,
    "short_side": pl.String,
    "start_date": pl.Datetime("us", "UTC"),
    "end_date": pl.Datetime("us", "UTC"),
    "game_start_time": pl.Datetime("us", "UTC"),
    "active": pl.Boolean,
    "closed": pl.Boolean,
    "status": pl.String,
    "long_price": pl.Float64,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "fee_coefficient": pl.Float64,
    "min_trade_qty": pl.Float64,
    "tick_size": pl.Float64,
    "settlement": pl.Float64,
}


def _px(node: Any) -> float | None:
    if isinstance(node, dict):
        node = node.get("value")
    if node in (None, ""):
        return None
    return float(node)


# Non-financial categories observed on the gateway (2026-11). finance/crypto are excluded.
NON_FINANCIAL_CATEGORIES = ("sports", "politics", "culture", "science")


def window_params(start, end) -> list[dict[str, Any]]:
    """Query passes for one window. Sports is limited to game moneylines (props run to
    hundreds of thousands of markets per fortnight); other categories are taken whole."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    dates = {
        "endDateMin": start.strftime(fmt) if start else None,
        "endDateMax": end.strftime(fmt) if end else None,
    }
    return [
        {**dates, "categories": ["sports"], "marketTypes": ["moneyline"]},
        {**dates, "categories": [c for c in NON_FINANCIAL_CATEGORIES if c != "sports"]},
    ]


class PolymarketUS:
    def __init__(self, client: CachedClient) -> None:
        self.client = client

    def markets(self, *, closed: bool, page_size: int = 500, cache: bool = True,
                extra: dict[str, Any] | None = None) -> Iterator[Fetched]:
        offset = 0
        while True:
            page = self.client.get("/markets", {
                "limit": page_size, "offset": offset, "closed": str(closed).lower(),
                "orderBy": "id", "orderDirection": "asc", **(extra or {}),
            }, cache=cache)
            yield page
            # Pages can come back one short (hidden markets), so only stop on an empty page.
            if not page.payload.get("markets"):
                return
            offset += page_size

    def events(self, *, closed: bool, page_size: int = 100, cache: bool = True,
               extra: dict[str, Any] | None = None) -> Iterator[Fetched]:
        offset = 0
        while True:
            page = self.client.get("/events", {
                "limit": page_size, "offset": offset, "closed": str(closed).lower(),
                "orderBy": "id", "orderDirection": "asc", **(extra or {}),
            }, cache=cache)
            yield page
            if not page.payload.get("events"):
                return
            offset += page_size

    def bbo(self, slug: str) -> Fetched:
        return self.client.get(f"/markets/{slug}/bbo", cache=False)

    def book(self, slug: str) -> Fetched:
        return self.client.get(f"/markets/{slug}/book", cache=False)

    def settlement(self, slug: str) -> Fetched:
        return self.client.get(f"/markets/{slug}/settlement")

    def price_history(self, slug: str, start_ts: int, end_ts: int,
                      fidelity_minutes: int = 60) -> Fetched:
        return self.client.get("/price-history", {
            "symbol": slug,
            "timestamp.startTimestamp": start_ts,
            "timestamp.endTimestamp": end_ts,
            "fidelity": fidelity_minutes,
        })


def market_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for m in payload.get("markets") or []:
        sides = m.get("marketSides") or []
        long_side = next((s for s in sides if s.get("long")), None)
        short_side = next((s for s in sides if not s.get("long")), None)
        closed = bool(m.get("closed"))
        # /settlement is missing for many resolved markets, so read the resolution from
        # outcomePrices once the status is RESOLVED. Index 0 is the long side.
        settlement = None
        if m.get("status") == "MARKET_STATUS_RESOLVED":
            prices = _json_list(m.get("outcomePrices"))
            if prices:
                settlement = float(prices[0])
        rows.append({
            "slug": m.get("slug"),
            "market_id": str(m.get("id")),
            "question": m.get("question"),
            "title": m.get("title"),
            "category": m.get("category"),
            "market_type": m.get("marketType"),
            "long_side": long_side.get("description") if long_side else None,
            "short_side": short_side.get("description") if short_side else None,
            "start_date": parse_ts(m.get("startDate")),
            "end_date": parse_ts(m.get("endDate")),
            "game_start_time": parse_ts(m.get("gameStartTime")),
            "active": bool(m.get("active")),
            "closed": closed,
            "status": m.get("status"),
            "long_price": _px(long_side.get("price")) if long_side else None,
            "best_bid": _px(m.get("bestBidQuote")),
            "best_ask": _px(m.get("bestAskQuote")),
            "fee_coefficient": m.get("feeCoefficient"),
            "min_trade_qty": m.get("minimumTradeQty"),
            "tick_size": m.get("orderPriceMinTickSize"),
            "settlement": settlement,
        })
    return [r for r in rows if r["slug"]]


def event_slug_map(payload: dict[str, Any]) -> dict[str, str]:
    """market slug -> event slug (the event slug matches Polymarket intl event slugs)."""
    mapping = {}
    for event in payload.get("events") or []:
        for m in event.get("markets") or []:
            if m.get("slug"):
                mapping[m["slug"]] = event.get("slug")
    return mapping


def bbo_top(payload: dict[str, Any]) -> dict[str, float | None]:
    md = payload.get("marketData") or {}
    return {
        "yes_bid": _px(md.get("bestBid")),
        "yes_ask": _px(md.get("bestAsk")),
        "last": _px(md.get("lastTradePx")),
        "state": md.get("state"),
    }


def book_top(payload: dict[str, Any]) -> dict[str, float | None]:
    md = payload.get("marketData") or {}
    bids = [(_px(b.get("px")), float(b.get("qty") or 0)) for b in md.get("bids") or []]
    asks = [(_px(a.get("px")), float(a.get("qty") or 0)) for a in md.get("offers") or md.get("asks") or []]
    best_bid = max(bids, default=None)
    best_ask = min(asks, default=None)
    return {
        "yes_bid": best_bid[0] if best_bid else None,
        "yes_bid_size": best_bid[1] if best_bid else None,
        "yes_ask": best_ask[0] if best_ask else None,
        "yes_ask_size": best_ask[1] if best_ask else None,
    }


def price_history_rows(payload: dict[str, Any], slug: str) -> list[dict[str, Any]]:
    return [
        {"slug": slug, "ts": parse_ts(p["timestamp"]), "price": float(p["longPrice"])}
        for p in payload.get("history") or []
        if p.get("timestamp") is not None and p.get("longPrice") is not None
    ]
