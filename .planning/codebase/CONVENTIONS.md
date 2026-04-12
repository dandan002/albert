# Code Conventions

**Analysis Date:** 2026-04-12

## Style & Formatting

- **Indentation:** 4 spaces (Python standard)
- **Line length:** No enforced limit observed; lines generally stay under 120 characters
- **Strings:** Double quotes for f-strings and regular strings; single quotes for dict keys and short literals (both used inconsistently)
- **Trailing commas:** Not consistently used; absent from multi-line function calls like `MarketDataEvent(...)` constructors
- **Blank lines:** 2 blank lines between top-level definitions, 1 blank line between method definitions (PEP 8)
- **No formatter configured:** No black, ruff, or yapf config found in `pyproject.toml`; formatting relies on developer convention
- **No linter configured:** No flake8, pylint, or ruff linting rules defined in `pyproject.toml`

## Naming Patterns

**Classes:**
- PascalCase — `BaseIngestor`, `KalshiIngestor`, `ExecutionEngine`, `RiskChecker`, `PortfolioTracker`
- Abstract base classes prefixed with `Base` — `BaseIngestor`, `BaseStrategy`, `ExchangeAdapter`

**Functions/Methods:**
- snake_case — `kelly_size()`, `get_connection()`, `cmd_status()`, `_make_auth_headers()`
- Private methods prefixed with single underscore — `_connect_and_stream()`, `_handle_intent()`, `_persist_fill()`
- Module-level private functions prefixed with underscore — `_check_env()`, `_setup_logging()`, `_load_private_key()`

**Variables:**
- snake_case — `market_ids`, `yes_bid`, `fill_price`, `unrealized_pnl`
- Private instance attributes prefixed with underscore — `self._bus`, `self._conn`, `self._strategies`, `self._price_cache`
- Constants in UPPER_SNAKE_CASE at module level — `_WS_URL`, `_MAX_RETRIES`, `_DEFAULTS`, `_SCHEMA`, `DB_PATH`

**Type/Model Names:**
- Data classes use PascalCase — `MarketDataEvent`, `OrderIntent`, `FillEvent`, `StrategyHaltedEvent`
- No suffix conventions like `Model`, `Dto`, or `Data`; just nouns

**File names:**
- snake_case — `portfolio/tracker.py`, `execution/engine.py`, `execution/kelly.py`

## Type Annotations

- **Comprehensive:** All function signatures include return type annotations and parameter types
- **Modern Python syntax:** Uses `dict[str, BaseStrategy]` instead of `Dict[str, BaseStrategy]`; `list[str]` instead of `List[str]`; `str | Path` instead of `Union[str, Path]` (Python 3.11+ style)
- **Optional returns explicitly typed:** `_normalize() -> MarketDataEvent | None` in `albert/ingestor/base.py`
- **Dataclass-based models:** All event types use `@dataclass` with typed fields in `albert/events.py`
- **Literal types for constrained strings:** `Literal["kalshi", "polymarket"]` and `Literal["yes", "no"]` used in event definitions
- **No typing imports:** No `from typing import ...` for basic types; uses built-in generics (Python 3.11 style)

## Error Handling Patterns

- **Broad exception handling in runners:** `BaseIngestor.run()` catches all `Exception` and reconnects; `StrategyEngine.run()` catches `Exception` per strategy to prevent one strategy crash from taking down the engine
- **`CancelledError` explicitly re-raised:** In `BaseIngestor.run()`, `asyncio.CancelledError` is re-raised before the generic `Exception` handler
- **Logging over raising:** Most components log exceptions with `logger.exception()` rather than raising; errors are swallowed with reconnection/retry
- **No custom exception classes:** No domain-specific exceptions defined; uses built-in exceptions only
- **Retry with exponential backoff:** `KalshiAdapter._post_with_retry()` and `PolymarketAdapter._post_with_retry()` use 3 retries with `2 ** attempt` second delays
- **Strategy halting on failure:** `ExecutionEngine._halt_strategy()` sets `enabled = 0` in DB and publishes `StrategyHaltedEvent` when an order fails
- **Graceful degradation:** Risk checker returns `False` (skip order) on limits rather than raising; ingestors reconnect on disconnect

## Async Patterns

- **asyncio throughout:** All core components use `async/await` — `BaseIngestor.run()`, `StrategyEngine.run()`, `ExecutionEngine.run()`, `PortfolioTracker.run()`
- **EventBus with asyncio.Queue:** Pub/sub via `EventBus` class using `asyncio.Queue` per subscriber; `subscribe()` returns a queue, `publish()` is async
- **`asyncio.gather` for concurrent tasks:** `__main__._main()` uses `asyncio.gather()` to run all services concurrently; `ExecutionEngine.run()` and `PortfolioTracker.run()` use `asyncio.gather()` for multiple queue consumers
- **Long-running loops:** Services run infinite `while True` loops, consuming from queues with `await self._queue.get()`
- **Task cancellation:** Tests use `asyncio.create_task()` + `task.cancel()` pattern; `CancelledError` caught explicitly
- **httpx.AsyncClient:** HTTP calls use `httpx.AsyncClient` for non-blocking I/O with `websockets.connect()` async context managers
- **`asyncio_mode = "auto"`** in pytest config — all async test functions are auto-detected as coroutines

## Import Conventions

- **Order:** Standard library → third-party → local, separated by blank lines
- **Standard library imports:** `import asyncio`, `import json`, `import logging`, `import sqlite3`, etc.
- **Third-party imports:** `import httpx`, `import websockets`, `from cryptography...`
- **Relative imports within package:** Used for sibling modules — `from .base import BaseIngestor` in `albert/ingestor/kalshi.py`
- **Absolute imports for cross-package:** `from albert.events import EventBus, MarketDataEvent`, `from albert.execution.adapters.kalshi import _load_private_key`
- **No `__all__` exports:** None of the `__init__.py` files define `__all__`; they are all empty
- **Import of private symbols across packages:** `KalshiIngestor` imports `_load_private_key` from `albert.execution.adapters.kalshi` — cross-package access to a private function

## Documentation Patterns

- **Docstrings on public functions only:** `kelly_size()` in `albert/execution/kelly.py` has a full docstring with Args/Returns; most other public functions have no docstrings
- **Abstract methods use `...` ellipsis:** `BaseStrategy.on_market_data()`, `ExchangeAdapter.place_order()` use `...` instead of `pass`
- **Brief docstrings on abstract methods:** `BaseIngestor.run()` has a one-line docstring: `"Connect and stream indefinitely, reconnecting on failure."`
- **No module-level docstrings:** None of the modules have module docstrings except `albert/__main__.py` and `albert/cli.py` which use `# albert/__main__.py` style comments
- **Inline comments for business logic:** Code uses comments to explain non-obvious logic, e.g., `# Only trade if net odds are at least 1:1`, `# Exact close or flip`
- **No type hint comments:** All type information is in annotations, not comments

## Logging Patterns

- **Standard library `logging`:** All modules use `logging.getLogger(__name__)` pattern
- **Structured-ish format:** JSON-like format string in `__main__.py`: `'{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": "%(message)s"}'`
- **Log levels used consistently:**
  - `logger.info()` for normal operations — order fills, strategy loads, debounce skips
  - `logger.warning()` for non-critical issues — missing orderbook data, HTTP retry failures
  - `logger.error()` for serious problems — strategy halted, no adapter found
  - `logger.exception()` for caught exceptions with traceback — disconnected ingestor, strategy errors, bankroll errors
- **Log message format:** Components prefix with module context — `"execution:fill"`, `"execution:strategy_halted"`, `"risk:debounce"`, `"risk:daily_loss_limit"`, `"portfolio:fill"`, `"kalshi POST %s attempt %d failed"`
- **Rotating file handler:** 10MB max, 3 backups, writing to `albert.log`; also streams to stdout
- **No lazy %-formatting:** Uses `%` style format args with logger (e.g., `logger.info("loaded strategy %s", sid)`) rather than f-strings in log calls

## Module Design

- **Package structure with empty `__init__.py`:** All packages (`albert/`, `albert/ingestor/`, `albert/strategies/`, `albert/strategies/examples/`, `albert/execution/`, `albert/execution/adapters/`, `albert/portfolio/`) use empty `__init__.py`
- **No re-exports:** Clients import directly from submodules, e.g., `from albert.strategies.engine import StrategyEngine`
- **Class-per-file pattern:** Most modules contain a single primary class with the same name as the file — `tracker.py` → `PortfolioTracker`, `risk.py` → `RiskChecker`
- **Helper functions are module-private:** `_load_private_key()`, `_check_env()`, `_setup_logging()`, `_make_auth_headers()` are module-level private functions
- **Constants at module top:** `_WS_URL`, `_BASE_URL`, `_MAX_RETRIES`, `_SCHEMA`, `_DEFAULTS`, `_REQUIRED_ENV`, `_LOG_FORMAT`, `DB_PATH` all defined at the top of their respective modules

---

*Convention analysis: 2026-04-12*