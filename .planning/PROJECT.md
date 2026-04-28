# Albert Trading System

## What This Is

A fully automated prediction market trading bot that connects to Kalshi and Polymarket exchanges, ingests real-time orderbook data via WebSockets, runs pluggable trading strategies with Kelly criterion position sizing, tracks portfolio P&L, and provides health monitoring and backtesting capabilities.

## Core Value

Automated, risk-managed trading on prediction markets with pluggable strategies and unified order execution across exchanges.

## Requirements

### Validated

- ✓ Real-time market data ingestion via WebSocket (Kalshi, Polymarket) — v1.0
- ✓ Strategy engine with hot-reloadable configs — v1.0
- ✓ Kelly criterion position sizing — v1.0
- ✓ Pre-trade risk checks (debounce, daily loss, max notional) — v1.0
- ✓ Exchange adapters (KalshiAdapter, PolymarketAdapter with SDK auth) — v1.0
- ✓ Portfolio tracking with positions and P&L — v1.0
- ✓ SQLite persistence (markets, orderbook, positions, fills, strategies, daily_pnl, health_status) — v1.0
- ✓ Graceful shutdown with bounded 5-second exit — v1.0 (Phase 5)
- ✓ Circuit breaker that halts strategies via EventBus — v1.0 (Phase 5)
- ✓ Health monitoring with adapter/ingestor/engine status — v1.0 (Phase 6)
- ✓ Mean reversion and momentum strategies with edge calculation — v1.0 (Phase 3)
- ✓ Backtesting engine against historical orderbook snapshots — v1.0 (Phase 4)

### Active

- [ ] Paper trading mode with simulated execution
- [ ] Whale alert system for large trades detection
- [ ] TWAP/VWAP order types for large orders
- [ ] Web dashboard for monitoring positions and P&L
- [ ] Real-time position and P&L updates via dashboard

### Out of Scope

- Multi-account support — complexity not needed for personal trading
- Cloud deployment — run locally for v1
- Market creation — exchange handles this
- Oracle integration — not part of core trading
- NLP pipelines — overcomplex for v1

## Context

**Shipped v1.0:** 7 phases, 10 plans, ~3,782 LOC Python, 65 commits (2026-03-29 → 2026-04-28).

**Current State:**
- Core trading system fully implemented with test coverage (84 tests passing)
- Polymarket integration verified with formal verification artifact and 5/5 UAT tests
- Critical resilience bugs fixed (graceful shutdown, circuit breaker async publish)
- Health monitoring reports adapter connectivity, ingestor WebSocket status, and engine task liveness
- CLI entry points: `python -m albert status`, `python -m albert health`

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
| Event-driven async architecture | Decouples ingestion, strategy, execution, and tracking | ✓ Good — scales cleanly to 7 subsystems |
| SQLite for all persistence | Simple, embedded, no external dependencies | ✓ Good — WAL mode handles concurrent reads |
| Kelly criterion for position sizing | Mathematically optimal for edge-based betting | ✓ Good — fractional Kelly (0.25) balances growth and safety |
| Pluggable strategy interface | Allows dynamic loading and hot-reload of strategies | ✓ Good — importlib loading works for mean reversion and momentum |
| Explicit task creation + cancel for shutdown | `asyncio.gather()` without timeouts can hang forever | ✓ Good — all tasks exit within 5 seconds on SIGINT |
| Async RiskChecker.check() | Synchronous publish created garbage-collected coroutine, never delivering StrategyHaltedEvent | ✓ Good — circuit breaker now halts strategies correctly |
| HealthMonitor with SQLite upserts | SQLite ON CONFLICT enables simple persistent health history | ✓ Good — `python -m albert health` reports full pipeline status |
| time.perf_counter() for adapter latency | Sub-millisecond measurement for exchange reachability | ✓ Good — detects adapter health accurately |

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
*Last updated: 2026-04-28 after v1.0 milestone completion*
