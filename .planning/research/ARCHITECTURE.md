# Architecture Research: Albert

## Component Boundaries

Albert should be structured as a local research pipeline:

1. **Source clients** fetch raw API payloads from Kalshi and Deribit.
2. **Cache layer** persists raw responses and normalized tables with manifests.
3. **Normalizer** converts source-specific payloads into canonical market, quote, trade, candle, instrument, and settlement records.
4. **Alignment layer** joins Kalshi contracts to Deribit observations available before each decision timestamp.
5. **Model layer** converts derivatives inputs into contract-level probabilities.
6. **Backtester** applies trading rules, conservative fills, fees, liquidity filters, and risk limits.
7. **Reporter** emits tables, plots, and run summaries.

## Data Flow

Raw API response -> raw cache -> normalized Parquet -> aligned feature set -> probability predictions -> simulated trades -> reports.

Every transformation should include:

- Source endpoint and parameters.
- Fetch timestamp.
- Covered time range.
- Code/config version where available.
- Row counts and validation warnings.

## Build Order

1. Establish project skeleton, CLI, config, cache paths, and tests.
2. Build Kalshi market discovery and historical/live stitching.
3. Build Deribit BTC derivatives ingestion and normalization.
4. Implement the first probability model and calibration diagnostics.
5. Implement conservative backtesting and risk controls.
6. Add reports and end-to-end validation fixtures.

## Critical Design Notes

- Time should be represented in UTC with explicit timezone-aware conversions at boundaries.
- Backtests must enforce no look-ahead: features at decision time cannot include settlement-period data or later order-book observations.
- Historical Kalshi data must route to live or historical endpoints based on cutoff behavior, not a hardcoded assumption.
- Normalized schemas should be stable even if raw vendor fields differ between live and historical endpoints.

## Sources

- Kalshi historical data: https://docs.kalshi.com/getting_started/historical_data
- Kalshi market candlesticks: https://docs.kalshi.com/api-reference/market/get-market-candlesticks
- Kalshi historical market candlesticks: https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks
- Deribit chart data: https://docs.deribit.com/api-reference/market-data/public-get_tradingview_chart_data
