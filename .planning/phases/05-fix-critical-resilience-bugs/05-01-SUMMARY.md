---
phase: 05-fix-critical-resilience-bugs
plan: 01
status: complete
completed: "2026-04-27"
---

# Plan 05-01 Summary: Fix Graceful Shutdown and RiskChecker Async Publish Bug

## Objective
Fix two critical resilience bugs identified in the v1.0 milestone audit:
1. Graceful shutdown hangs because `asyncio.gather()` blocks forever on tasks that never exit
2. `RiskChecker.check()` calls async `EventBus.publish()` without `await`, so `StrategyHaltedEvent` is never delivered

## What Was Built

### Task 1: Graceful Shutdown Propagation
- **`albert/__main__.py`**: Replaced `await asyncio.gather(...)` with explicit `asyncio.create_task()` for each service, `await shutdown_event.wait()` to block until signal, task cancellation loop, and `asyncio.wait_for(gather, timeout=5.0)` to ensure clean exit within 5 seconds.
- **`albert/ingestor/base.py`**: Added `shutdown_event` parameter to `BaseIngestor.__init__` and check `self._shutdown_event.is_set()` at the top of the `run()` loop before reconnect.
- **`albert/ingestor/kalshi.py` & `polymarket.py`**: Forward `shutdown_event` through to `BaseIngestor`.
- **`albert/execution/engine.py`**: Added shutdown checks before and after `queue.get()` in both `handle_market_data()` and `handle_orders()`.
- **`albert/portfolio/tracker.py`**: Added shutdown checks before and after `queue.get()` in both `handle_fills()` and `handle_market_data()`.
- **`albert/strategies/engine.py`**: Added shutdown check after `await self._queue.get()`.

### Task 2: RiskChecker Async Publish Fix
- **`albert/execution/risk.py`**: Changed `def check(...)` to `async def check(...)` and `self._bus.publish(...)` to `await self._bus.publish(...)` inside the circuit breaker block.
- **`albert/execution/engine.py`**: Changed `self._risk.check(intent, size_usd)` to `await self._risk.check(intent, size_usd)`.
- **`tests/test_risk.py`**: Updated all 5 existing tests to `async def` with `await checker.check(...)`. Added new `test_circuit_breaker_publishes_halted_event` that verifies `StrategyHaltedEvent` is actually delivered on the EventBus when the circuit breaker triggers.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Explicit task creation + cancel | `asyncio.gather()` without timeouts can hang forever if child tasks don't exit; explicit tasks with `cancel()` + `wait_for(gather, timeout=5.0)` guarantees bounded shutdown |
| Shutdown checks after `queue.get()` | When a task is cancelled while blocked on `queue.get()`, it raises `CancelledError`. Additional `is_set()` checks after unblocking ensure the task returns immediately without processing stale events during shutdown |

## Verification

- `python -m pytest tests/test_risk.py -x -v` — **6 passed** (including new circuit breaker publish test)
- `python -m pytest tests/ -x -v` — **56 passed, 0 failed** (no regressions)

## Artifacts

| File | Change |
|------|--------|
| `albert/__main__.py` | Task-based main loop with explicit cancel on shutdown |
| `albert/ingestor/base.py` | Shutdown-aware reconnect loop |
| `albert/ingestor/kalshi.py` | Forward shutdown_event |
| `albert/ingestor/polymarket.py` | Forward shutdown_event |
| `albert/execution/engine.py` | Shutdown checks in queue consumers; await risk.check() |
| `albert/execution/risk.py` | Async risk check with awaited publish |
| `albert/portfolio/tracker.py` | Shutdown checks in queue consumers |
| `albert/strategies/engine.py` | Shutdown check after queue.get() |
| `tests/test_risk.py` | Updated async tests + circuit breaker publish verification |

## Issues Encountered

- `kalshi.py` and `polymarket.py` needed `import asyncio` added because their `__init__` signatures now reference `asyncio.Event | None`.

## Deviation Log

No deviations from plan.
