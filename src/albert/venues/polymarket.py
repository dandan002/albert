"""Polymarket international (polymarket.com): public data only, never order placement.

US residents are geoblocked from trading here. This venue supplies market metadata (Gamma) and
on-chain wallet activity (Data API v2), which is the source of the smart-wallet signals.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import polars as pl

from albert.http import CachedClient, Fetched
from albert.venues.kalshi import parse_ts

MARKET_SCHEMA = {
    "condition_id": pl.String,
    "question": pl.String,
    "slug": pl.String,
    "event_slug": pl.String,
    "event_title": pl.String,
    "tags": pl.List(pl.String),
    "start_date": pl.Datetime("us", "UTC"),
    "end_date": pl.Datetime("us", "UTC"),
    "closed_time": pl.Datetime("us", "UTC"),
    "game_start_time": pl.Datetime("us", "UTC"),
    "closed": pl.Boolean,
    "outcomes": pl.List(pl.String),
    "token_ids": pl.List(pl.String),
    "winner_index": pl.Int64,
    "volume": pl.Float64,
    "neg_risk": pl.Boolean,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
}

TRADE_SCHEMA = {
    "transaction_hash": pl.String,
    "wallet": pl.String,
    "condition_id": pl.String,
    "token_id": pl.String,
    "outcome_index": pl.Int64,
    "outcome": pl.String,
    "side": pl.String,
    "price": pl.Float64,
    "size": pl.Float64,
    "ts": pl.Datetime("us", "UTC"),
    "slug": pl.String,
    "event_slug": pl.String,
}

LEADERBOARD_SCHEMA = {
    "wallet": pl.String,
    "rank": pl.Int64,
    "pnl": pl.Float64,
    "volume": pl.Float64,
    "user_name": pl.String,
    "time_period": pl.String,
    "category": pl.String,
}


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return json.loads(value)


def winner_index(outcome_prices: list[Any], closed: bool) -> int | None:
    """Index of the outcome that resolved to 1.0, or None if unresolved or ambiguous (e.g. 50/50)."""
    if not closed or not outcome_prices:
        return None
    prices = [float(p) for p in outcome_prices]
    winners = [i for i, p in enumerate(prices) if p >= 0.99]
    return winners[0] if len(winners) == 1 else None


class Polymarket:
    def __init__(self, gamma: CachedClient, data: CachedClient) -> None:
        self.gamma = gamma
        self.data = data

    def events(self, *, closed: bool, end_min: datetime | None = None,
               end_max: datetime | None = None, page_size: int = 100,
               cache: bool = True) -> Iterator[Fetched]:
        # /events caps offset at 2000; /events/keyset pages with after_cursor instead.
        cursor = None
        while True:
            page = self.gamma.get("/events/keyset", {
                "closed": str(closed).lower(),
                "limit": page_size,
                "after_cursor": cursor,
                "end_date_min": _iso_z(end_min),
                "end_date_max": _iso_z(end_max),
            }, cache=cache)
            yield page
            cursor = page.payload.get("next_cursor")
            if not cursor or not page.payload.get("events"):
                return

    def _cursor(self, path: str, params: dict[str, Any], max_pages: int | None,
                cache: bool = True) -> Iterator[Fetched]:
        cursor = None
        pages = 0
        while True:
            page = self.data.get(path, {**params, "cursor": cursor}, cache=cache)
            yield page
            pages += 1
            cursor = (page.payload.get("pagination") or {}).get("next_cursor")
            if not cursor or not page.payload.get("data") or (max_pages and pages >= max_pages):
                return

    def wallet_trades(self, wallet: str, start: datetime | None = None,
                      end: datetime | None = None, max_pages: int | None = 50) -> Iterator[Fetched]:
        params = {"user": wallet, "limit": 1000,
                  "start": int(start.timestamp()) if start else None,
                  "end": int(end.timestamp()) if end else None}
        yield from self._cursor("/trades", params, max_pages)

    def recent_wallet_trades(self, wallet: str, since: datetime) -> Iterator[Fetched]:
        params = {"user": wallet, "limit": 200, "start": int(since.timestamp())}
        yield from self._cursor("/trades", params, max_pages=5, cache=False)

    def market_trades(self, condition_id: str, max_pages: int | None = 20) -> Iterator[Fetched]:
        yield from self._cursor("/trades", {"condition": condition_id, "limit": 1000}, max_pages)

    def leaderboard(self, time_period: str = "all", category: str = "overall",
                    sort_by: str = "PNL", limit: int = 500) -> Iterator[Fetched]:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        cursor = None
        fetched = 0
        while fetched < limit:
            page = self.data.get("/leaderboard", {
                "time_period": time_period, "category": category, "sort_by": sort_by,
                "limit": min(1000, limit), "cursor": cursor,
            }, tag=day)
            yield page
            fetched += len(page.payload.get("data") or [])
            cursor = (page.payload.get("pagination") or {}).get("next_cursor")
            if not cursor or not page.payload.get("data"):
                return

    def price_history(self, token_id: str, start: datetime, end: datetime,
                      bucket_seconds: int = 300) -> Iterator[Fetched]:
        """Price history in chunks, since the API caps each request at a 15-day span."""
        chunk = 14 * 86400
        lo = int(start.timestamp())
        hi = int(end.timestamp())
        while lo < hi:
            yield self.data.get("/prices-history", {
                "token_id": token_id, "start": lo, "end": min(hi, lo + chunk),
                "bucket_seconds": bucket_seconds,
            })
            lo += chunk


# ---- normalizers ----------------------------------------------------------------------------
def market_rows(events_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events_payload:
        tags = [t.get("slug") for t in event.get("tags") or [] if t.get("slug")]
        for m in event.get("markets") or []:
            closed = bool(m.get("closed"))
            rows.append({
                "condition_id": m.get("conditionId"),
                "question": m.get("question"),
                "slug": m.get("slug"),
                "event_slug": event.get("slug"),
                "event_title": event.get("title"),
                "tags": tags,
                "start_date": parse_ts(m.get("startDate")),
                "end_date": parse_ts(m.get("endDate")),
                "closed_time": _parse_loose(m.get("closedTime")),
                "game_start_time": _parse_loose(m.get("gameStartTime")),
                "closed": closed,
                "outcomes": [str(o) for o in _json_list(m.get("outcomes"))],
                "token_ids": [str(t) for t in _json_list(m.get("clobTokenIds"))],
                "winner_index": winner_index(_json_list(m.get("outcomePrices")), closed),
                "volume": float(m.get("volumeNum") or m.get("volume") or 0),
                "neg_risk": bool(m.get("negRisk")),
                "best_bid": _float(m.get("bestBid")),
                "best_ask": _float(m.get("bestAsk")),
            })
    return [r for r in rows if r["condition_id"]]


def trade_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "transaction_hash": t.get("transaction_hash"),
            "wallet": (t.get("proxy_wallet") or "").lower(),
            "condition_id": t.get("condition_id"),
            "token_id": t.get("token_id"),
            "outcome_index": t.get("outcome_index"),
            "outcome": t.get("outcome"),
            "side": t.get("side"),
            "price": _float(t.get("price")),
            "size": _float(t.get("size")),
            "ts": parse_ts(t.get("timestamp")),
            "slug": t.get("slug"),
            "event_slug": t.get("event_slug") or None,
        }
        for t in payload.get("data") or []
    ]


def leaderboard_rows(payload: dict[str, Any], time_period: str, category: str) -> list[dict]:
    return [
        {
            "wallet": (r.get("user_id") or "").lower(),
            "rank": r.get("rank"),
            "pnl": _float(r.get("pnl")),
            "volume": _float(r.get("volume")),
            "user_name": r.get("user_name"),
            "time_period": time_period,
            "category": category,
        }
        for r in payload.get("data") or []
    ]


def price_history_rows(payload: dict[str, Any], token_id: str) -> list[dict[str, Any]]:
    points = payload.get("data") or payload.get("history") or []
    rows = []
    for p in points:
        ts = p.get("t", p.get("timestamp", p.get("ts")))
        price = p.get("p", p.get("price"))
        if ts is None or price is None:
            continue
        rows.append({"token_id": token_id, "ts": parse_ts(ts), "price": float(price)})
    return rows


def _iso_z(value: datetime | None) -> str | None:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if value else None


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_loose(value: Any) -> datetime | None:
    """Gamma mixes ISO formats, e.g. '2026-09-16 20:08:24+00'."""
    if not value:
        return None
    text = str(value).replace(" ", "T").replace("Z", "+00:00")
    if text.endswith("+00"):
        text += ":00"
    try:
        return datetime.fromisoformat(text).astimezone(UTC)
    except ValueError:
        return None
