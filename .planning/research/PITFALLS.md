# Pitfalls Research: Albert

## Pitfall: Look-Ahead Bias

**Warning signs:** Features use close-period prices, settlement-period data, or candles that end after the simulated trade decision.

**Prevention:** Use explicit decision timestamps and enforce source data `timestamp <= decision_timestamp` checks in tests.

**Phase:** Phase 4 and Phase 5.

## Pitfall: Optimistic Execution

**Warning signs:** Backtests enter at midpoint, ignore fees, ignore size, or assume all quoted liquidity is available.

**Prevention:** Fill at executable bid/ask, include fee assumptions, reject trades below liquidity thresholds, and report skipped signals.

**Phase:** Phase 5.

## Pitfall: Historical Endpoint Gaps

**Warning signs:** Backtests silently omit older Kalshi markets or mix live and historical payload shapes without normalization checks.

**Prevention:** Query or model the historical cutoff, use endpoint-specific fetchers, and validate expected market counts per date range.

**Phase:** Phase 2.

## Pitfall: Overfitting Strategy Thresholds

**Warning signs:** Many thresholds are tried on the full period and only the best result is reported.

**Prevention:** Separate exploratory and holdout windows, report all configured runs, and keep the first model intentionally simple.

**Phase:** Phase 4 and Phase 6.

## Pitfall: Contract Definition Drift

**Warning signs:** Strategy assumes all 15-minute BTC contracts have identical settlement rules or ticker formats.

**Prevention:** Parse and store settlement source, open/close times, event metadata, and contract terms from Kalshi market metadata.

**Phase:** Phase 2.

## Pitfall: Source Price Mismatch

**Warning signs:** Deribit BTC index levels and Kalshi settlement references differ enough that a model appears calibrated but targets the wrong underlying.

**Prevention:** Store Kalshi settlement methodology metadata, compare BTC reference series where available, and include basis/error diagnostics.

**Phase:** Phase 4 and Phase 6.

## Sources

- Kalshi historical data: https://docs.kalshi.com/getting_started/historical_data
- Kalshi glossary: https://docs.kalshi.com/getting_started/terms
- Kalshi crypto markets help: https://help.kalshi.com/markets/popular-markets/crypto-markets
- Deribit instrument trades: https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_instrument_and_time
