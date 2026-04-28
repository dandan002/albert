# Albert

## What This Is

Albert is a Python research CLI for evaluating whether crypto derivatives data can predict and trade Kalshi 15-minute Bitcoin event contracts. It ingests Kalshi market data and Deribit BTC derivatives data, computes short-horizon probabilities, and backtests conservative trading rules before any live or paper trading work.

The first milestone is not a trading bot. It is an evidence engine that can tell whether the proposed edge is positive after realistic market frictions.

## Core Value

Produce a reproducible, conservative backtest showing whether derivatives-implied Bitcoin probabilities create positive expected value on Kalshi after costs.

## Requirements

### Validated

- [x] Phase 1 validated cache/config/run-manifest foundations for DATA-04: raw and normalized cache path helpers, provenance-rich run manifests, deterministic config hashing, and local verification commands.

### Active

- [ ] Discover and retrieve Kalshi 15-minute Bitcoin markets, pricing, order books, trades, candlesticks, and settlement outcomes.
- [ ] Retrieve Deribit BTC futures/options data needed to estimate short-horizon terminal probabilities.
- [ ] Compute derivatives-implied probabilities for eligible Kalshi contracts before the simulated decision time.
- [ ] Backtest trades using conservative fills, fees, liquidity assumptions, and explicit risk limits.
- [ ] Report PnL, calibration, drawdown, turnover, exposure, and per-contract attribution.

### Out of Scope

- Live Kalshi trading - v1 must prove an offline edge before introducing order placement risk.
- Paper trading - useful after v1, but not required to establish the research loop.
- Polymarket integration - deferred until Kalshi plus Deribit produces a validated baseline.
- Multi-exchange derivatives aggregation - deferred to avoid widening ingestion and data-quality risk before the core signal is tested.
- Production deployment, alerting, and hosted dashboards - v1 is a local research CLI.

## Context

The initial strategy hypothesis is that short-dated crypto derivatives prices contain enough information about near-term Bitcoin movement to price Kalshi's 15-minute Bitcoin contracts better than the market. The first user-facing artifact should be a command-line research workflow that can fetch/cache data, run backtests, and emit tables and plots.

Phase 1 is complete. The repository now has an installable Python package, Typer CLI skeleton, deterministic config loading, safe cache path helpers, provenance manifest models, reusable fixtures, and Makefile verification commands. Phase 2 starts Kalshi market data ingestion on top of those contracts.

Relevant ecosystem facts as of 2026-04-28:

- Kalshi exposes market discovery, trades, order books, candlesticks, and historical endpoints through its API documentation.
- Kalshi partitions older exchange data into historical endpoints for markets, market candlesticks, trades, and orders.
- Deribit exposes public BTC futures/options market data, including OHLC chart data and instrument trade history.
- Polymarket has public Gamma/Data APIs and CLOB APIs, but Polymarket is not part of v1 implementation.

## Constraints

- **Strategy scope**: Start with Kalshi plus Deribit only - keeps v1 focused on proving the derivatives-implied probability signal.
- **Backtest horizon**: Target the last 12 months - provides more observations than the live-data window and requires deliberate historical-data stitching.
- **Execution realism**: Use conservative fills at available bid/ask with fees and liquidity constraints - mid-price backtests are too optimistic for thin binary markets.
- **Trading safety**: No live or paper trading in v1 - avoid execution risk until the offline evidence is strong.
- **Interface**: Build a Python research CLI first - lower overhead and better fit for reproducible research than an early dashboard.
- **Data provenance**: Every run must be traceable to source endpoints, timestamps, and cached raw inputs - prevents accidental look-ahead and irreproducible results.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build a Python research CLI first | Fastest path to reproducible ingestion, modeling, backtesting, and reports | - Pending |
| Scope v1 to Kalshi plus Deribit | Focuses on the core signal before broader cross-market complexity | - Pending |
| Defer Polymarket to v2 | Polymarket is useful for comparison later but not needed to test the first hypothesis | - Pending |
| Use the last 12 months for initial backtests | More observations than a short live-data window, with explicit historical stitching | - Pending |
| Simulate conservative bid/ask fills with fees and liquidity checks | Reduces the chance of approving a strategy that only works under optimistic execution | - Pending |
| Require positive EV after costs as the v1 success bar | Aligns model work with tradable strategy evidence | - Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? Move to Out of Scope with reason
2. Requirements validated? Move to Validated with phase reference
3. New requirements emerged? Add to Active
4. Decisions to log? Add to Key Decisions
5. "What This Is" still accurate? Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-28 after Phase 1 completion*
