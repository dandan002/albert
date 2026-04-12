# Roadmap: Albert Trading System

**Defined:** 2026-04-12
**Granularity:** coarse
**Core Value:** Automated, risk-managed trading on prediction markets with pluggable strategies and unified order execution across exchanges.

---

## Phases

- [ ] **Phase 1: Polymarket Production Readiness** — Fix critical ECDSA authentication and adapter registration blockers
- [ ] **Phase 2: Observability & Resilience** — Add health monitoring and graceful shutdown
- [ ] **Phase 3: Strategy Expansion** — Add momentum and mean reversion strategies with edge calculation

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

**Plans:** TBD

---

### Phase 2: Observability & Resilience

**Goal:** System can gracefully shut down and reports health status of all components

**Depends on:** Phase 1

**Requirements:** RES-01, RES-02, RES-03

**Success Criteria** (what must be TRUE):

1. User can signal shutdown (Ctrl+C) and all pending state persists to database
2. User can query system health and receive status of all components
3. Circuit breaker halts trading when daily loss limit is reached

**Plans:** TBD

---

### Phase 3: Strategy Expansion

**Goal:** Users can use momentum and mean reversion strategies with proper edge calculation

**Depends on:** Phase 2

**Requirements:** STR-01, STR-02, STR-03

**Success Criteria** (what must be TRUE):

1. User can load mean reversion strategy from configuration
2. User can load momentum strategy from configuration
3. Strategies emit orders only when calculated edge exceeds configured threshold

**Plans:** TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1 - Polymarket Production Readiness | 0/1 | Not started | - |
| 2 - Observability & Resilience | 0/1 | Not started | - |
| 3 - Strategy Expansion | 0/1 | Not started | - |

---

*Roadmap defined: 2026-04-12*