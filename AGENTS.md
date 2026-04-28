<!-- GSD:project-start source:PROJECT.md -->
## Project

**Albert**

Albert is a Python research CLI for evaluating whether crypto derivatives data can predict and trade Kalshi 15-minute Bitcoin event contracts. It ingests Kalshi market data and Deribit BTC derivatives data, computes short-horizon probabilities, and backtests conservative trading rules before any live or paper trading work.

The first milestone is not a trading bot. It is an evidence engine that can tell whether the proposed edge is positive after realistic market frictions.

**Core Value:** Produce a reproducible, conservative backtest showing whether derivatives-implied Bitcoin probabilities create positive expected value on Kalshi after costs.

### Constraints

- **Strategy scope**: Start with Kalshi plus Deribit only - keeps v1 focused on proving the derivatives-implied probability signal.
- **Backtest horizon**: Target the last 12 months - provides more observations than the live-data window and requires deliberate historical-data stitching.
- **Execution realism**: Use conservative fills at available bid/ask with fees and liquidity constraints - mid-price backtests are too optimistic for thin binary markets.
- **Trading safety**: No live or paper trading in v1 - avoid execution risk until the offline evidence is strong.
- **Interface**: Build a Python research CLI first - lower overhead and better fit for reproducible research than an early dashboard.
- **Data provenance**: Every run must be traceable to source endpoints, timestamps, and cached raw inputs - prevents accidental look-ahead and irreproducible results.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommendation
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
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
