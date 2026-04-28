# Research Summary: Albert

## Key Findings

**Stack:** Python CLI with httpx, polars/pyarrow, local Parquet caches, pytest, and explicit source clients is the right v1 stack.

**Table stakes:** Data provenance, time alignment, conservative execution, and calibration reporting are not optional. They are the difference between a useful research tool and a misleading backtest.

**Watch out for:** Look-ahead bias, optimistic fills, historical/live endpoint differences, overfit thresholds, and mismatched BTC reference prices.

## Recommended v1 Shape

Build a narrow but rigorous offline pipeline:

1. Fetch and normalize Kalshi 15-minute BTC contract data.
2. Fetch and normalize Deribit BTC futures/options observations.
3. Align Deribit observations to Kalshi decision windows.
4. Estimate contract probabilities from derivatives inputs.
5. Backtest conservative trade rules with risk limits.
6. Report whether the edge is positive after costs.

## Deferred

- Polymarket integration.
- Paper trading.
- Live trading.
- Hosted dashboard.
- Multi-exchange derivatives aggregation.

## Sources

- Kalshi API documentation: https://docs.kalshi.com/
- Kalshi historical data: https://docs.kalshi.com/getting_started/historical_data
- Deribit API documentation: https://docs.deribit.com/
- Polymarket API documentation: https://docs.polymarket.com/api-reference
