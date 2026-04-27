---
phase: 05-fix-critical-resilience-bugs
plan: 02
status: complete
completed: "2026-04-27"
---

# Plan 05-02 Summary: Add Integration Tests for Shutdown and Circuit Breaker

## Objective
Add integration tests that verify the two critical resilience fixes from Plan 01:
1. All asyncio tasks exit cleanly within 5 seconds when shutdown_event is set
2. Circuit breaker publishes StrategyHaltedEvent when daily loss limit violations exceed threshold

## What Was Built

### Task 1: Integration Tests for Shutdown and Circuit Breaker

**`tests/test_integration.py`** — Two new integration tests added:

- **`test_graceful_shutdown_cancels_all_tasks_within_5_seconds`**:
  - Creates in-memory SQLite DB with one enabled strategy
  - Instantiates `StrategyEngine`, `ExecutionEngine`, and `PortfolioTracker` with a shared `asyncio.Event()`
  - Starts all three as `asyncio.create_task()`
  - Sets `shutdown_event.set()` after 0.1s startup
  - Publishes dummy events to unblock any `queue.get()` calls
  - Asserts `asyncio.wait_for(gather, timeout=5.0)` completes without `TimeoutError`
  - Asserts all tasks are `task.done()`

- **`test_circuit_breaker_publishes_strategy_halted_event`**:
  - Creates in-memory SQLite DB, seeds `daily_pnl` with -250.0 (below -200.0 limit)
  - Creates `RiskChecker` with real `EventBus`, subscribes to `"strategy_halted"` channel
  - Calls `await checker.check(intent, 10.0)` twice to exceed `circuit_breaker_violations=2`
  - Asserts `StrategyHaltedEvent` is received on the queue within 1.0s
  - Asserts `event.strategy_id == "s1"` and `"circuit_breaker" in event.reason`

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Mock adapter with AsyncMock | Avoids real HTTP/WebSocket I/O; keeps test fast and deterministic |
| Publish dummy events after shutdown | Unblocks `queue.get()` calls so tasks can see `shutdown_event.is_set()` and return |
| `order_debounce_seconds=0` | Prevents debounce from interfering with repeated `checker.check()` calls |

## Verification

- `python -m pytest tests/test_integration.py -x -v` — **3 passed**
- `python -m pytest tests/ -x -v` — **58 passed, 0 failed** (no regressions)

## Artifacts

| File | Change |
|------|--------|
| `tests/test_integration.py` | +146 lines: 2 new integration tests for shutdown and circuit breaker |

## Issues Encountered

None.

## Deviation Log

No deviations from plan.
