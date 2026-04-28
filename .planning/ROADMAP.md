# Roadmap: Albert Trading System

**Defined:** 2026-04-12
**Granularity:** coarse
**Core Value:** Automated, risk-managed trading on prediction markets with pluggable strategies and unified order execution across exchanges.

---

## Phases

- [ ] **Phase 1: Polymarket Production Readiness** — Fix critical ECDSA authentication and adapter registration blockers
- [ ] **Phase 2: Observability & Resilience** — Add health monitoring and graceful shutdown
- [x] **Phase 3: Strategy Expansion** — Add momentum and mean reversion strategies with edge calculation

---

## Phase Details

### Phase 1: Polymarket Production Readiness

**Goal:** Polymarket adapter can authenticate with ECDSA and execute orders via official API

**Depends on:** Nothing (first phase)

**Requirements:** PM-01, PM-02, PM-03

**Success Criteria** (what must be TRUE):

1. User can submit Polymarket orders via CLI and they execute successfully
2. Polymarket adapter registered in execution engine and responds to order intents
3. WebSocket connection established and market data flowing to strategy engine

**Plans:** 1 plan

- [x] 01-01-PLAN.md — Implement Polymarket SDK authentication and register adapter

---

### Phase 2: Observability & Resilience

**Goal:** System can gracefully shut down and reports health status of all components

**Depends on:** Phase 1

**Requirements:** RES-01, RES-02, RES-03

**Success Criteria** (what must be TRUE):

1. User can signal shutdown (Ctrl+C) and all pending state persists to database
2. User can query system health and receive status of all components
3. Circuit breaker halts trading when daily loss limit is reached

**Plans:** 1 plan

- [x] 02-01-PLAN.md — Implement graceful shutdown, health status, and circuit breaker

---

### Phase 3: Strategy Expansion

**Goal:** Users can use momentum and mean reversion strategies with proper edge calculation

**Depends on:** Phase 2

**Requirements:** STR-01, STR-02, STR-03

**Success Criteria** (what must be TRUE):

1. User can load mean reversion strategy from configuration
2. User can load momentum strategy from configuration
3. Strategies emit orders only when calculated edge exceeds configured threshold

**Plans:** 1 plan

- [x] 03-01-PLAN.md — Implement momentum and mean reversion strategies with edge calculation

---

### Phase 4: Strategy Backtesting
**Goal:** Run strategies against historical orderbook snapshots to validate edge before live trading

**Depends on:** Phase 3

**Requirements:** STR-04, STR-05

**Success Criteria** (what must be TRUE):
1. User can run a backtest for a specific market and strategy using `python -m albert.backtest`
2. Backtest produces a summary of total returns, win rate, and max drawdown

**Plans:** TBD

---

### Phase 5: Fix Critical Resilience Bugs
**Goal:** Close critical gaps where graceful shutdown hangs and circuit breaker never fires

**Depends on:** Phase 2

**Requirements:** RES-01, RES-03

**Gap Closure:** Closes gaps from v1.0 milestone audit

**Success Criteria** (what must be TRUE):
1. SIGINT/SIGTERM causes all ingestors and engine tasks to exit cleanly within 5 seconds
2. `RiskChecker.check()` is async and `await`s `EventBus.publish()`, delivering `StrategyHaltedEvent`
3. All integration tests for shutdown and circuit breaker pass

**Plans:** 2 plans

- [x] 05-01-PLAN.md — Fix graceful shutdown propagation and RiskChecker async publish bug
- [x] 05-02-PLAN.md — Add integration tests for shutdown and circuit breaker

---

### Phase 6: Complete Health Monitoring
**Goal:** Populate health check with adapter liveness, ingestor connectivity, and engine task status

**Depends on:** Phase 5

**Requirements:** RES-02

**Gap Closure:** Closes gaps from v1.0 milestone audit

**Success Criteria** (what must be TRUE):
1. `python -m albert health` reports adapter connectivity to exchanges
2. Health output includes ingestor WebSocket connection status
3. Health output includes whether StrategyEngine, ExecutionEngine, and PortfolioTracker tasks are alive

**Plans:** 2 plans

- [x] 06-01-PLAN.md — Build core health monitoring infrastructure (adapter checks, ingestor tracking, HealthMonitor)
- [x] 06-02-PLAN.md — Wire HealthMonitor into main loop and update health CLI

---

### Phase 7: Verify Polymarket Integration
**Goal:** Create formal verification artifacts and complete pending UAT for Polymarket adapter and ingestor

**Depends on:** Phase 1

**Requirements:** PM-01, PM-02, PM-03

**Gap Closure:** Closes gaps from v1.0 milestone audit

**Success Criteria** (what must be TRUE):
1. `VERIFICATION.md` exists for Phase 1 with evidence of SDK auth, adapter registration, and ingestor wiring
2. All 5 pending UAT tests in `UAT.md` are executed and pass
3. REQUIREMENTS.md checkboxes for PM-01, PM-02, PM-03 are checked

**Plans:** 2/2 plans complete

- [x] 07-01-PLAN.md — Create verification artifact for Phase 1 Polymarket integration
- [x] 07-02-PLAN.md — Execute UAT tests and close requirements

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1 - Polymarket Production Readiness | 1/1 | ✓ Complete | 2026-04-12 |
| 2 - Observability & Resilience | 1/1 | ✓ Complete | 2026-04-13 |
| 3 - Strategy Expansion | 1/1 | ✓ Complete | 2026-04-14 |
| 4 - Strategy Backtesting | 0/1 | Not started | - |
| 5 - Fix Critical Resilience Bugs | 2/2 | ✓ Complete | 2026-04-27 |
| 6 - Complete Health Monitoring | 2/2 | ✓ Complete | 2026-04-27 |
| 7 - Verify Polymarket Integration | 2/2 | Complete   | 2026-04-28 |

---

*Roadmap defined: 2026-04-12*