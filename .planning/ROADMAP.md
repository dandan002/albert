# Roadmap: Albert

## Overview

Albert v1 builds a reproducible research pipeline from source data to strategy evidence. The work starts with a reliable CLI and cache contract, adds Kalshi and Deribit ingestion, then layers probability modeling, conservative backtesting, and reports that can answer whether the strategy has positive EV after costs.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Research CLI Foundation** - Create the Python project skeleton, CLI, config, cache layout, manifests, and test harness.
- [ ] **Phase 2: Kalshi Market Data** - Discover and normalize Kalshi 15-minute Bitcoin markets and outcomes across the target history.
- [ ] **Phase 3: Deribit Derivatives Data** - Retrieve and normalize Deribit BTC futures/options data for the same windows.
- [ ] **Phase 4: Probability Model** - Align data by decision time and compute/calibrate contract probabilities.
- [ ] **Phase 5: Conservative Backtester** - Simulate strategy rules with realistic fills, fees, liquidity, and risk limits.
- [ ] **Phase 6: Strategy Reports** - Generate reproducible reports that summarize edge, performance, and failure modes.

## Phase Details

### Phase 1: Research CLI Foundation
**Goal**: A tested Python CLI can manage config, cache paths, run manifests, and deterministic research commands.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-04
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. User can run the CLI help command and see ingest, model, backtest, and report command groups.
  2. User can create a run manifest that records config, source parameters, timestamps, and output locations.
  3. User can write and read cached raw and normalized data paths through project helpers.
  4. Tests cover config loading, cache path construction, manifest creation, and CLI command wiring.
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Create Python project skeleton, packaging, CLI entrypoints, and config loading.
- [x] 01-02-PLAN.md — Implement cache path conventions, run manifests, and source metadata models.
- [x] 01-03-PLAN.md — Add test harness, fixtures, and baseline CI-style verification commands.

### Phase 2: Kalshi Market Data
**Goal**: Kalshi 15-minute Bitcoin markets can be discovered, fetched, cached, normalized, and linked to outcomes.
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. User can discover Kalshi 15-minute Bitcoin markets for a requested date range.
  2. User can fetch metadata, trades, bid/ask pricing data, order-book data where available, and settlement outcomes.
  3. User can route older data through historical endpoints when required.
  4. Normalized Kalshi tables include market ticker, event/series identifiers, open/close/settlement times, prices, liquidity fields, and outcome.
**Plans**: 7 plans

Plans:
- [x] 02-00-PLAN.md — Add Phase 2 dependencies, Kalshi package marker, and dependency import verification.
- [x] 02-04-PLAN.md — Create Kalshi fixture payloads for live/historical data, outcomes, missing data, and ambiguous markets.
- [x] 02-05-PLAN.md — Create contract tests for client, routing, normalization, CLI, cache, manifest, and quality behavior.
- [x] 02-01-PLAN.md — Build Kalshi client, pagination, retry/time-window handling, and raw cache writes.
- [x] 02-02-PLAN.md — Implement live versus historical endpoint selection and fixture-based response normalization.
- [x] 02-03-PLAN.md — Add market discovery command and data-quality checks for missing outcomes or price history.
- [ ] 02-06-PLAN.md — Close verification gaps for trade cursor pagination and empty normalized table schemas.

### Phase 3: Deribit Derivatives Data
**Goal**: Deribit BTC futures/options data can be fetched, cached, and normalized for Kalshi contract windows.
**Depends on**: Phase 1
**Requirements**: DATA-03
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. User can fetch BTC futures/options instrument metadata and market observations for requested windows.
  2. User can retrieve OHLC and trade-level observations needed by the first probability model.
  3. Normalized Deribit tables include timestamp, instrument, expiry, strike, option type, mark price, index price, implied volatility when available, and volume/open-interest fields where available.
  4. Tests cover fixture payloads for futures, options, and missing optional fields.
**Plans**: 2 plans

Plans:
- [ ] 03-01: Build Deribit client, raw cache writes, and instrument/window selection.
- [ ] 03-02: Normalize futures/options observations and validate schemas against fixtures.

### Phase 4: Probability Model
**Goal**: The system can align Kalshi contracts to pre-decision derivatives observations and produce calibrated probabilities.
**Depends on**: Phase 2, Phase 3
**Requirements**: MODEL-01, MODEL-02, MODEL-03
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. User can build a feature set that excludes observations after the simulated decision timestamp.
  2. User can compute a probability estimate for each eligible Kalshi contract.
  3. User can inspect calibration tables and plots comparing predictions to realized outcomes.
  4. Tests fail if post-decision or settlement-period data leaks into features.
**Plans**: 3 plans

Plans:
- [ ] 04-01: Implement contract-window alignment and look-ahead guards.
- [ ] 04-02: Implement the first derivatives-implied probability model.
- [ ] 04-03: Add calibration diagnostics and leakage-focused tests.

### Phase 5: Conservative Backtester
**Goal**: Strategy rules can be simulated with realistic execution assumptions and explicit risk controls.
**Depends on**: Phase 4
**Requirements**: BACK-01, BACK-02, BACK-03
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. User can run a backtest with configurable edge threshold, bankroll, position size, and daily loss limit.
  2. Trades enter only when executable bid/ask, liquidity, fees, and risk limits permit.
  3. Settled outcomes produce deterministic PnL and skipped-trade attribution.
  4. Synthetic tests prove expected fills, fees, settlement PnL, and risk-limit behavior.
**Plans**: 3 plans

Plans:
- [ ] 05-01: Implement strategy rule configuration and signal-to-order simulation.
- [ ] 05-02: Implement conservative fill, fee, liquidity, settlement, and PnL accounting.
- [ ] 05-03: Add bankroll/risk controls and synthetic backtest test cases.

### Phase 6: Strategy Reports
**Goal**: Reports make the strategy's edge, robustness, and weaknesses clear from cached inputs.
**Depends on**: Phase 5
**Requirements**: REPORT-01, REPORT-02, REPORT-03
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. User can generate tables for PnL, hit rate, drawdown, turnover, exposure, skipped trades, and per-contract attribution.
  2. User can generate plots for equity curve, edge distribution, calibration, and performance by time bucket.
  3. User can reproduce a prior report from the saved manifest and cached inputs.
  4. The final v1 report states whether results are positive after costs and identifies the main residual risks.
**Plans**: 2 plans

Plans:
- [ ] 06-01: Implement report tables, plots, and manifest-based reproduction.
- [ ] 06-02: Add end-to-end validation run and final strategy readout.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Research CLI Foundation | 3/3 | Complete | 2026-04-28 |
| 2. Kalshi Market Data | 0/6 | Not started | - |
| 3. Deribit Derivatives Data | 0/2 | Not started | - |
| 4. Probability Model | 0/3 | Not started | - |
| 5. Conservative Backtester | 0/3 | Not started | - |
| 6. Strategy Reports | 0/2 | Not started | - |
