# Albert Trading System

## What This Is

A fully automated prediction market trading bot that connects to Kalshi and Polymarket exchanges, ingests real-time orderbook data via WebSockets, runs pluggable trading strategies with Kelly criterion position sizing, and tracks portfolio P&L.

## Core Value

Automated, risk-managed trading on prediction markets with pluggable strategies and unified order execution across exchanges.

## Requirements

### Validated

- ✓ Real-time market data ingestion via WebSocket (Kalshi, Polymarket) — existing
- ✓ Strategy engine with hot-reloadable configs — existing
- ✓ Kelly criterion position sizing — existing
- ✓ Pre-trade risk checks (debounce, daily loss, max notional) — existing
- ✓ Exchange adapters (KalshiAdapter, PolymarketAdapter) — existing
- ✓ Portfolio tracking with positions and P&L — existing
- ✓ SQLite persistence (markets, orderbook, positions, fills, strategies, daily_pnl) — existing

### Active

- [ ] Add Polymarket as additional trading/data source (documented in design spec, current implementation has issues)
- [ ] Improve Polymarket adapter to use official API authentication properly
- [ ] Expand strategy library with additional trading strategies
- [ ] Add backtesting capabilities

### Out of Scope

- Web dashboard — for later milestone
- Multi-account support — complexity not needed
- Cloud deployment — run locally for v1

## Context

**Existing Documentation:**
- `docs/superpowers/plans/2026-03-29-albert-trading-system.md` — Implementation plan with detailed tasks
- `docs/superpowers/specs/2026-03-29-trading-design.md` — Design spec approved 2026-03-29

**Current State:**
- Core trading system implemented with test coverage
- Polymarket integration partially working but needs improvement
- CLI entry point: `python -m albert status`

**Technical Environment:**
- Python 3.11+
- asyncio-based event-driven architecture
- SQLite with WAL mode for persistence
- WebSockets for data ingestion, REST APIs for order execution

## Constraints

- **Tech Stack**: Python 3.11+, websockets, httpx, pytest — fixed
- **Single Process**: Runs as one asyncio process — architectural decision
- **Event-Driven**: Modules communicate via asyncio queues — architectural decision
- **Risk Limits**: Configurable via config.json — hard limits enforced

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Event-driven async architecture | Decouples ingestion, strategy, execution, and tracking | — Pending |
| SQLite for all persistence | Simple, embedded, no external dependencies | ✓ Good |
| Kelly criterion for position sizing | Mathematically optimal for edge-based betting | — Pending |
| Pluggable strategy interface | Allows dynamic loading and hot-reload of strategies | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-12 after project initialization*