from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from albert.kalshi.client import ALLOWED_ENDPOINT_SEGMENTS


@dataclass(frozen=True)
class HistoricalCutoff:
    market_settled_ts: datetime
    trades_created_ts: datetime


@dataclass(frozen=True)
class EndpointWindow:
    kind: Literal["historical", "live"]
    endpoint_segment: str
    path: str
    start: datetime
    end: datetime
    parameter_names: tuple[str, str]


def parse_historical_cutoff(payload: Mapping[str, object]) -> HistoricalCutoff:
    return HistoricalCutoff(
        market_settled_ts=_parse_timestamp(
            payload.get("market_settled_ts"),
            "market_settled_ts",
        ),
        trades_created_ts=_parse_timestamp(
            payload.get("trades_created_ts"),
            "trades_created_ts",
        ),
    )


def route_by_cutoff(
    start: datetime,
    end: datetime,
    cutoff: datetime,
    *,
    historical_segment: str,
    live_segment: str,
    historical_path: str,
    live_path: str,
    parameter_names: tuple[str, str],
) -> list[EndpointWindow]:
    start = _aware_utc(start, "start")
    end = _aware_utc(end, "end")
    cutoff = _aware_utc(cutoff, "cutoff")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    _validate_endpoint_segment(historical_segment)
    _validate_endpoint_segment(live_segment)

    if end <= cutoff:
        return [
            EndpointWindow(
                kind="historical",
                endpoint_segment=historical_segment,
                path=historical_path,
                start=start,
                end=end,
                parameter_names=parameter_names,
            )
        ]
    if start >= cutoff:
        return [
            EndpointWindow(
                kind="live",
                endpoint_segment=live_segment,
                path=live_path,
                start=start,
                end=end,
                parameter_names=parameter_names,
            )
        ]
    return [
        EndpointWindow(
            kind="historical",
            endpoint_segment=historical_segment,
            path=historical_path,
            start=start,
            end=cutoff,
            parameter_names=parameter_names,
        ),
        EndpointWindow(
            kind="live",
            endpoint_segment=live_segment,
            path=live_path,
            start=cutoff,
            end=end,
            parameter_names=parameter_names,
        ),
    ]


def plan_market_windows(
    start: datetime,
    end: datetime,
    cutoff: HistoricalCutoff,
) -> list[EndpointWindow]:
    return route_by_cutoff(
        start,
        end,
        cutoff.market_settled_ts,
        historical_segment="historical_markets",
        live_segment="live_markets",
        historical_path="/historical/markets",
        live_path="/markets",
        parameter_names=("min_close_ts", "max_close_ts"),
    )


def plan_candlestick_windows(
    ticker: str,
    start: datetime,
    end: datetime,
    cutoff: HistoricalCutoff,
) -> list[EndpointWindow]:
    return route_by_cutoff(
        start,
        end,
        cutoff.market_settled_ts,
        historical_segment="historical_candlesticks",
        live_segment="live_candlesticks",
        historical_path=f"/historical/markets/{ticker}/candlesticks",
        live_path=f"/markets/{ticker}/candlesticks",
        parameter_names=("start_ts", "end_ts"),
    )


def plan_trade_windows(
    start: datetime,
    end: datetime,
    cutoff: HistoricalCutoff,
) -> list[EndpointWindow]:
    return route_by_cutoff(
        start,
        end,
        cutoff.trades_created_ts,
        historical_segment="historical_trades",
        live_segment="live_trades",
        historical_path="/historical/trades",
        live_path="/markets/trades",
        parameter_names=("min_ts", "max_ts"),
    )


def _parse_timestamp(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return _aware_utc(value, field_name)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError(f"{field_name} must be an ISO datetime") from error
        return _aware_utc(parsed, field_name)
    raise ValueError(f"{field_name} must be an ISO datetime or epoch seconds")


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_endpoint_segment(endpoint_segment: str) -> None:
    if endpoint_segment not in ALLOWED_ENDPOINT_SEGMENTS:
        raise ValueError(f"unsupported Kalshi endpoint segment: {endpoint_segment}")


__all__ = [
    "EndpointWindow",
    "HistoricalCutoff",
    "parse_historical_cutoff",
    "plan_candlestick_windows",
    "plan_market_windows",
    "plan_trade_windows",
    "route_by_cutoff",
]
