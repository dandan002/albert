from datetime import datetime, timezone

from albert.kalshi.routing import (
    EndpointWindow,
    HistoricalCutoff,
    plan_candlestick_windows,
    plan_market_windows,
    plan_trade_windows,
    route_by_cutoff,
)


UTC = timezone.utc


def _dt(day: int, hour: int) -> datetime:
    return datetime(2026, 4, day, hour, tzinfo=UTC)


def test_splits_spanning_market_window_into_historical_and_live() -> None:
    cutoff = HistoricalCutoff(
        market_settled_ts=_dt(1, 12),
        trades_created_ts=_dt(2, 12),
    )

    windows = plan_market_windows(_dt(1, 10), _dt(1, 14), cutoff)

    assert windows == [
        EndpointWindow(
            kind="historical",
            endpoint_segment="historical_markets",
            path="/historical/markets",
            start=_dt(1, 10),
            end=_dt(1, 12),
            parameter_names=("min_close_ts", "max_close_ts"),
        ),
        EndpointWindow(
            kind="live",
            endpoint_segment="live_markets",
            path="/markets",
            start=_dt(1, 12),
            end=_dt(1, 14),
            parameter_names=("min_close_ts", "max_close_ts"),
        ),
    ]
    assert {window.kind for window in windows} == {"historical", "live"}


def test_candlestick_windows_use_market_settled_cutoff() -> None:
    cutoff = HistoricalCutoff(
        market_settled_ts=_dt(1, 12),
        trades_created_ts=_dt(2, 12),
    )

    windows = plan_candlestick_windows(
        "KXBTC15M-26APR011200",
        _dt(1, 10),
        _dt(1, 14),
        cutoff,
    )

    assert [window.endpoint_segment for window in windows] == [
        "historical_candlesticks",
        "live_candlesticks",
    ]
    assert [window.path for window in windows] == [
        "/historical/markets/KXBTC15M-26APR011200/candlesticks",
        "/markets/KXBTC15M-26APR011200/candlesticks",
    ]
    assert all(window.parameter_names == ("start_ts", "end_ts") for window in windows)


def test_uses_trades_created_cutoff_for_trade_windows() -> None:
    cutoff = HistoricalCutoff(
        market_settled_ts=_dt(1, 12),
        trades_created_ts=_dt(2, 12),
    )

    windows = plan_trade_windows(_dt(2, 10), _dt(2, 14), cutoff)

    assert [window.kind for window in windows] == ["historical", "live"]
    assert [window.endpoint_segment for window in windows] == [
        "historical_trades",
        "live_trades",
    ]
    assert all(window.parameter_names == ("min_ts", "max_ts") for window in windows)
    assert windows[0].end == cutoff.trades_created_ts
    assert windows[1].start == cutoff.trades_created_ts


def test_route_by_cutoff_returns_single_historical_or_live_window() -> None:
    historical = route_by_cutoff(
        _dt(1, 9),
        _dt(1, 10),
        _dt(1, 12),
        historical_segment="historical_markets",
        live_segment="live_markets",
        historical_path="/historical/markets",
        live_path="/markets",
        parameter_names=("min_close_ts", "max_close_ts"),
    )
    live = route_by_cutoff(
        _dt(1, 12),
        _dt(1, 13),
        _dt(1, 12),
        historical_segment="historical_markets",
        live_segment="live_markets",
        historical_path="/historical/markets",
        live_path="/markets",
        parameter_names=("min_close_ts", "max_close_ts"),
    )

    assert [window.kind for window in historical] == ["historical"]
    assert [window.kind for window in live] == ["live"]
