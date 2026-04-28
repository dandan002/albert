# Stack Research: Albert

## Recommendation

Use a Python data/research stack with explicit API clients, local file-backed caching, deterministic transformations, and CLI entrypoints.

## Core Stack

| Area | Choice | Rationale | Confidence |
|------|--------|-----------|------------|
| Language | Python 3.12+ | Strong data ecosystem and fast research iteration | High |
| CLI | Typer or Click | Simple command grouping for ingest, model, backtest, report | High |
| HTTP | httpx | Sync/async support, retries/timeouts via explicit client wrappers | High |
| Data frames | polars plus pyarrow | Efficient local analytics and Parquet IO | High |
| Numeric/modeling | numpy, scipy, pandas only where needed | Standard tools for probability transforms and diagnostics | High |
| Plots | matplotlib or plotly static exports | Good enough for backtest reports without building a UI | Medium |
| Storage | Local Parquet plus JSON manifests | Reproducible, easy to inspect, no database required for v1 | High |
| Testing | pytest with fixture payloads | Covers data normalization and backtest math without live API dependence | High |

## API Surfaces

- Kalshi API docs expose market discovery, market details, order books, trades, and market candlesticks. The historical-data docs note that older markets, candlesticks, trades, and orders are partitioned into historical endpoints.
- Deribit API docs expose public market data, including TradingView-compatible OHLC chart data and last-trades-by-instrument/time endpoints. Option trade payloads include fields such as mark price, index price, and implied volatility when available.
- Polymarket docs describe Gamma, Data, and CLOB APIs. They remain useful research context but are deferred from v1.

## What Not To Use Yet

- A web dashboard: adds product work before the research loop is validated.
- A full database service: local Parquet is enough until the data volume or query patterns prove otherwise.
- Live order-placement SDKs: v1 explicitly excludes live and paper trading.
- A large ML framework: the first model should be interpretable and auditable before adding black-box complexity.

## Sources

- Kalshi API documentation: https://docs.kalshi.com/
- Kalshi historical data: https://docs.kalshi.com/getting_started/historical_data
- Kalshi market candlesticks: https://docs.kalshi.com/api-reference/market/get-market-candlesticks
- Deribit chart data: https://docs.deribit.com/api-reference/market-data/public-get_tradingview_chart_data
- Deribit instrument trades: https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_instrument_and_time
- Polymarket API reference: https://docs.polymarket.com/api-reference
