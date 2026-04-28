# Requirements: Albert

**Defined:** 2026-04-28
**Core Value:** Produce a reproducible, conservative backtest showing whether derivatives-implied Bitcoin probabilities create positive expected value on Kalshi after costs.

## v1 Requirements

### Data

- [ ] **DATA-01**: User can run a CLI command that discovers Kalshi 15-minute Bitcoin markets over a requested date range.
- [ ] **DATA-02**: User can retrieve Kalshi market metadata, order-book snapshots or bid/ask candles, trades, and settlement outcomes for discovered markets.
- [ ] **DATA-03**: User can retrieve Deribit BTC futures/options observations needed for short-horizon probability modeling.
- [x] **DATA-04**: User can cache raw source responses and normalized Parquet tables with run manifests that record endpoint, parameters, fetch time, and covered time range.

### Modeling

- [ ] **MODEL-01**: User can generate a feature set that aligns each eligible Kalshi contract with only Deribit observations available before the simulated decision time.
- [ ] **MODEL-02**: User can compute a derivatives-implied probability for each eligible Kalshi contract.
- [ ] **MODEL-03**: User can inspect calibration diagnostics comparing predicted probabilities to realized contract outcomes.

### Backtesting

- [ ] **BACK-01**: User can run a backtest that enters trades only when predicted edge exceeds a configurable threshold.
- [ ] **BACK-02**: User can simulate conservative fills using bid/ask prices, fees, liquidity constraints, and contract settlement outcomes.
- [ ] **BACK-03**: User can configure bankroll, max position size, max daily loss, and minimum edge threshold.

### Reporting

- [ ] **REPORT-01**: User can generate summary tables for PnL, hit rate, drawdown, turnover, exposure, skipped trades, and per-contract attribution.
- [ ] **REPORT-02**: User can generate plots for equity curve, edge distribution, calibration, and performance by time bucket.
- [ ] **REPORT-03**: User can reproduce a prior report from cached inputs and the saved run manifest.

## v2 Requirements

### Data

- **DATA-05**: User can add Polymarket market/pricing data for cross-market comparison.
- **DATA-06**: User can add additional crypto derivatives exchanges beyond Deribit.

### Execution

- **EXEC-01**: User can run a paper-trading monitor against live Kalshi market data.
- **EXEC-02**: User can place live Kalshi orders behind explicit risk gates and manual enablement.

### Interface

- **UI-01**: User can review contracts, signals, and backtest reports in a local dashboard.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live trading in v1 | Offline evidence must exist before execution risk is introduced |
| Paper trading in v1 | Backtest quality is the first bottleneck |
| Polymarket in v1 | Deferred until Kalshi plus Deribit baseline is validated |
| Hosted dashboard | CLI is enough for v1 research and reduces product overhead |
| Mid-price-only backtests | Too optimistic for thin or fast binary markets |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 3 | Pending |
| DATA-04 | Phase 1 | Complete |
| MODEL-01 | Phase 4 | Pending |
| MODEL-02 | Phase 4 | Pending |
| MODEL-03 | Phase 4 | Pending |
| BACK-01 | Phase 5 | Pending |
| BACK-02 | Phase 5 | Pending |
| BACK-03 | Phase 5 | Pending |
| REPORT-01 | Phase 6 | Pending |
| REPORT-02 | Phase 6 | Pending |
| REPORT-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-04-28*
*Last updated: 2026-04-28 after initial definition*
