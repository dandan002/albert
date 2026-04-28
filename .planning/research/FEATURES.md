# Feature Research: Albert

## Table Stakes

For this kind of strategy research tool, v1 needs:

- Data ingestion for each source with pagination, time-window filters, rate-limit handling, and cache manifests.
- A canonical market table that maps each Kalshi contract to its open/close/settlement timestamps and outcome.
- A normalized derivatives table with BTC index, futures, options, mark prices, implied volatility when available, and timestamps.
- Time alignment that ensures the model only uses data available before the simulated decision time.
- A transparent probability model that can be inspected and calibrated.
- A conservative execution simulator with fees, spread crossing, minimum liquidity, and position limits.
- Reports that make positive/negative EV clear at the aggregate and per-contract level.

## Differentiators

Potential differentiators after the baseline works:

- Polymarket comparison prices and cross-market discrepancy checks.
- Scenario-suite backtests with optimistic, base, and conservative execution.
- Paper-trading monitor that logs hypothetical orders against live Kalshi order books.
- Automated strategy search over thresholds, holding rules, and risk limits with walk-forward validation.
- Dashboard for reviewing contracts, signal distributions, and error cases.

## Anti-Features

Avoid in v1:

- Live trading controls.
- Complex portfolio optimization before single-strategy EV is proven.
- Overfitted model search without holdout windows.
- Treating midpoint prices as executable fills in thin markets.
- Silent fallback from failed data fetches to partial data.

## Dependencies Between Features

- Backtesting depends on Kalshi market/outcome ingestion and derivatives time alignment.
- Calibration depends on model output plus realized outcomes.
- Conservative execution depends on order-book or candlestick bid/ask data and a fee model.
- Reporting depends on complete run manifests so charts can be traced back to exact inputs.

## Sources

- Kalshi markets API: https://docs.kalshi.com/python-sdk/api/MarketsApi
- Kalshi historical data: https://docs.kalshi.com/getting_started/historical_data
- Deribit chart data: https://docs.deribit.com/api-reference/market-data/public-get_tradingview_chart_data
- Deribit instrument trades: https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_instrument_and_time
