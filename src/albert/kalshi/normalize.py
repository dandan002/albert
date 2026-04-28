from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from albert.kalshi.schemas import (
    KalshiMarketRow,
    KalshiOrderbookObservationRow,
    KalshiOutcomeRow,
    KalshiPriceObservationRow,
    KalshiQualitySummaryRow,
    KalshiSkippedMarketRow,
    KalshiTradeRow,
)


ONE_DOLLAR = Decimal("1.00")


def _parse_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError(f"unsupported datetime value: {value!r}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid decimals")
    return Decimal(str(value))


def _price_node_decimal(node: dict[str, Any] | None, name: str) -> Decimal | None:
    if not node:
        return None
    return _decimal_or_none(node.get(f"{name}_dollars", node.get(name)))


def _market_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("markets"), list):
        return list(payload["markets"])
    if isinstance(payload.get("market"), dict):
        return [payload["market"]]
    return []


def _market_is_eligible_btc_15m(market: dict[str, Any], series_ticker: str) -> bool:
    if market.get("series_ticker") != series_ticker:
        return False

    open_time = _parse_utc_datetime(market.get("open_time"))
    expiration_time = _parse_utc_datetime(market.get("expiration_time"))
    if open_time is None or expiration_time is None:
        return False

    interval_seconds = (expiration_time - open_time).total_seconds()
    return interval_seconds == 15 * 60


def _skip_reason_for_market(market: dict[str, Any], series_ticker: str) -> str:
    if market.get("series_ticker") != series_ticker:
        return f"series_ticker_not_{series_ticker.lower()}"
    open_time = _parse_utc_datetime(market.get("open_time"))
    expiration_time = _parse_utc_datetime(market.get("expiration_time"))
    if open_time is None or expiration_time is None:
        return "missing_15m_interval_timestamps"
    if (expiration_time - open_time).total_seconds() != 15 * 60:
        return "interval_not_15m"
    return "ambiguous_btc_market_mapping"


def _market_row(
    market: dict[str, Any], *, source_endpoint: str, source_raw_path: Path
) -> KalshiMarketRow:
    return KalshiMarketRow.model_validate(
        {
            "market_ticker": market.get("ticker"),
            "event_ticker": market.get("event_ticker"),
            "series_ticker": market.get("series_ticker"),
            "title": market.get("title"),
            "subtitle": market.get("subtitle"),
            "status": market.get("status"),
            "open_time_utc": _parse_utc_datetime(market.get("open_time")),
            "close_time_utc": _parse_utc_datetime(market.get("close_time")),
            "settlement_time_utc": _parse_utc_datetime(market.get("settlement_time")),
            "expiration_time_utc": _parse_utc_datetime(market.get("expiration_time")),
            "can_close_early": market.get("can_close_early"),
            "price_level_structure": market.get("price_level_structure"),
            "fractional_trading_enabled": market.get("fractional_trading_enabled"),
            "source_endpoint": source_endpoint,
            "source_raw_path": source_raw_path,
        }
    )


def _outcome_row(
    market: dict[str, Any], *, source_endpoint: str, source_raw_path: Path
) -> KalshiOutcomeRow:
    result = market.get("result")
    settlement_value = market.get("settlement_value_dollars")
    settlement_ts = _parse_utc_datetime(market.get("settlement_ts"))
    outcome_missing = result is None and settlement_value is None and settlement_ts is None
    return KalshiOutcomeRow.model_validate(
        {
            "market_ticker": market.get("ticker"),
            "settlement_time_utc": _parse_utc_datetime(market.get("settlement_time")),
            "outcome": result,
            "result": result,
            "settlement_value_dollars": settlement_value,
            "settlement_ts": settlement_ts,
            "outcome_missing": outcome_missing,
            "source_endpoint": source_endpoint,
            "source_raw_path": source_raw_path,
        }
    )


def normalize_markets_page(
    payload: dict[str, Any],
    *,
    source_endpoint: str,
    source_raw_path: Path,
    series_ticker: str = "KXBTC15M",
) -> tuple[list[KalshiMarketRow], list[KalshiOutcomeRow], list[KalshiSkippedMarketRow]]:
    markets: list[KalshiMarketRow] = []
    outcomes: list[KalshiOutcomeRow] = []
    skipped: list[KalshiSkippedMarketRow] = []

    for market in _market_records(payload):
        if not _market_is_eligible_btc_15m(market, series_ticker):
            skipped.append(
                KalshiSkippedMarketRow.model_validate(
                    {
                        "market_ticker": market.get("ticker"),
                        "event_ticker": market.get("event_ticker"),
                        "series_ticker": market.get("series_ticker"),
                        "skip_reason": payload.get("skip_reason")
                        or _skip_reason_for_market(market, series_ticker),
                        "raw_title": market.get("title"),
                        "raw_subtitle": market.get("subtitle"),
                        "source_endpoint": source_endpoint,
                        "source_raw_path": source_raw_path,
                    }
                )
            )
            continue

        markets.append(
            _market_row(market, source_endpoint=source_endpoint, source_raw_path=source_raw_path)
        )
        outcomes.append(
            _outcome_row(market, source_endpoint=source_endpoint, source_raw_path=source_raw_path)
        )

    return markets, outcomes, skipped


def normalize_candlesticks_page(
    payload: dict[str, Any],
    *,
    market_ticker: str,
    series_ticker: str,
    source_endpoint: str,
    source_raw_path: Path,
    period_interval_minutes: int = 1,
) -> list[KalshiPriceObservationRow]:
    rows: list[KalshiPriceObservationRow] = []
    interval = int(payload.get("period_interval", period_interval_minutes))
    for candle in payload.get("candlesticks", []):
        yes_bid = candle.get("yes_bid") or {}
        yes_ask = candle.get("yes_ask") or {}
        price = candle.get("price") or {}
        rows.append(
            KalshiPriceObservationRow.model_validate(
                {
                    "market_ticker": market_ticker,
                    "series_ticker": series_ticker,
                    "observed_at_utc": _parse_utc_datetime(candle.get("start_period_ts")),
                    "period_interval_minutes": interval,
                    "yes_bid_open": _price_node_decimal(yes_bid, "open"),
                    "yes_bid_high": _price_node_decimal(yes_bid, "high"),
                    "yes_bid_low": _price_node_decimal(yes_bid, "low"),
                    "yes_bid_close": _price_node_decimal(yes_bid, "close"),
                    "yes_ask_open": _price_node_decimal(yes_ask, "open"),
                    "yes_ask_high": _price_node_decimal(yes_ask, "high"),
                    "yes_ask_low": _price_node_decimal(yes_ask, "low"),
                    "yes_ask_close": _price_node_decimal(yes_ask, "close"),
                    "last_price_open": _price_node_decimal(price, "open"),
                    "last_price_close": _price_node_decimal(price, "close"),
                    "volume_contracts": _decimal_or_none(candle.get("volume")),
                    "open_interest_contracts": _decimal_or_none(candle.get("open_interest")),
                    "source_endpoint": source_endpoint,
                    "source_raw_path": source_raw_path,
                }
            )
        )
    return rows


def normalize_trades_page(
    payload: dict[str, Any],
    *,
    source_endpoint: str,
    source_raw_path: Path,
) -> list[KalshiTradeRow]:
    return [
        KalshiTradeRow.model_validate(
            {
                "trade_id": trade.get("trade_id"),
                "market_ticker": trade.get("ticker"),
                "created_time_utc": _parse_utc_datetime(trade.get("created_time")),
                "count_contracts": _decimal_or_none(trade.get("count")),
                "yes_price": _decimal_or_none(trade.get("yes_price")),
                "no_price": _decimal_or_none(trade.get("no_price")),
                "taker_side": trade.get("taker_side"),
                "source_endpoint": source_endpoint,
                "source_raw_path": source_raw_path,
            }
        )
        for trade in payload.get("trades", [])
    ]


def _best_bid(levels: list[list[Any]]) -> Decimal | None:
    prices = [_decimal_or_none(level[0]) for level in levels if level]
    return max((price for price in prices if price is not None), default=None)


def normalize_orderbooks_page(
    payload: dict[str, Any],
    *,
    source_endpoint: str,
    source_raw_path: Path,
    snapshot_time: datetime,
) -> list[KalshiOrderbookObservationRow]:
    snapshot = _parse_utc_datetime(payload.get("snapshot_time")) or _parse_utc_datetime(snapshot_time)
    if snapshot is None:
        raise ValueError("snapshot_time is required")

    rows: list[KalshiOrderbookObservationRow] = []
    for item in payload.get("orderbooks", []):
        orderbook = item.get("orderbook") or {}
        yes_levels = orderbook.get("yes") or []
        no_levels = orderbook.get("no") or []
        implied_asks = {
            "yes": ONE_DOLLAR - no_best if (no_best := _best_bid(no_levels)) is not None else None,
            "no": ONE_DOLLAR - yes_best if (yes_best := _best_bid(yes_levels)) is not None else None,
        }

        for side, levels in (("yes", yes_levels), ("no", no_levels)):
            for level_index, level in enumerate(levels):
                bid_price = level[0] if len(level) > 0 else None
                bid_count = level[1] if len(level) > 1 else None
                rows.append(
                    KalshiOrderbookObservationRow.model_validate(
                        {
                            "market_ticker": item.get("ticker"),
                            "snapshot_time_utc": snapshot,
                            "side": side,
                            "level_index": level_index,
                            "bid_price": _decimal_or_none(bid_price),
                            "bid_count_contracts": _decimal_or_none(bid_count),
                            "implied_ask_price": implied_asks[side],
                            "source_endpoint": source_endpoint,
                            "source_raw_path": source_raw_path,
                        }
                    )
                )
    return rows


def build_quality_summary(
    run_id: str,
    markets: list[KalshiMarketRow],
    skipped_markets: list[KalshiSkippedMarketRow],
    outcomes: list[KalshiOutcomeRow],
    price_observations: list[KalshiPriceObservationRow],
    orderbook_observations: list[KalshiOrderbookObservationRow],
    historical_window_count: int,
    live_window_count: int,
) -> KalshiQualitySummaryRow:
    from albert.kalshi.quality import assess_market_quality

    counts = assess_market_quality(
        markets=markets,
        outcomes=outcomes,
        price_observations=price_observations,
        orderbook_observations=orderbook_observations,
    )
    return KalshiQualitySummaryRow.model_validate(
        {
            "run_id": run_id,
            "market_count": len(markets),
            "eligible_market_count": len(markets),
            "skipped_market_count": len(skipped_markets),
            "missing_outcome_count": counts["missing_outcome_count"],
            "missing_price_history_count": counts["missing_price_history_count"],
            "missing_orderbook_count": counts["missing_orderbook_count"],
            "historical_window_count": historical_window_count,
            "live_window_count": live_window_count,
        }
    )
