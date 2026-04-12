<!-- GSD:project-start source:PROJECT.md -->
## Project

**Albert Trading System**

A fully automated prediction market trading bot that connects to Kalshi and Polymarket exchanges, ingests real-time orderbook data via WebSockets, runs pluggable trading strategies with Kelly criterion position sizing, and tracks portfolio P&L.

**Core Value:** Automated, risk-managed trading on prediction markets with pluggable strategies and unified order execution across exchanges.

### Constraints

- **Tech Stack**: Python 3.11+, websockets, httpx, pytest — fixed
- **Single Process**: Runs as one asyncio process — architectural decision
- **Event-Driven**: Modules communicate via asyncio queues — architectural decision
- **Risk Limits**: Configurable via config.json — hard limits enforced
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Language & Runtime
- Python 3.11+ (requires `>=3.11` per `pyproject.toml`) — used for all application code
- Python 3.12.2 detected on development system
- Async Python (`asyncio`) — the entire application is async-first; all core loops use `async def run()` and `asyncio.gather`
- No web framework (pure async event loop, not ASGI/WSGI)
## Package Manager & Build
- pip (standard Python packaging)
- setuptools >=68 as build backend per `pyproject.toml`
- `pyproject.toml` — modern PEP 621 metadata, setuptools build backend
- Package installed in editable mode (`albert.egg-info` present)
- No lockfile committed (no `requirements.txt`, no `poetry.lock`, no `pdm.lock`)
- `config.json` — runtime configuration (risk limits, intervals)
- `.env` file — secrets loaded at startup via `albert/config.py:load_project_env()`
- No `.env.example` committed (listed in `.gitignore`)
## Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| websockets | >=12.0 | WebSocket connections to Kalshi and Polymarket orderbook feeds |
| httpx | >=0.27 | Async HTTP client for exchange REST APIs (order placement, balance queries) |
| cryptography | >=42.0 | RSA/ECDSA key handling for Kalshi PSS-signed request auth and PEM key loading |
## Dev Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=8.0 | Test runner |
| pytest-asyncio | >=0.23 | Async test support (`asyncio_mode = "auto"` in pyproject.toml) |
## Infrastructure
- SQLite 3 — embedded database, file-based (`albert.db`)
- WAL mode enabled (`PRAGMA journal_mode=WAL`) in `albert/db.py`
- Schema managed via in-code `_SCHEMA` string in `albert/db.py`, executed with `executescript` (no migration tool)
- Tables: `markets`, `orderbook_snapshots`, `positions`, `fills`, `strategies`, `daily_pnl`
- Python stdlib `logging` with structured JSON format to `albert.log` (rotating, 10MB, 3 backups)
- Also logs to stdout with same JSON format
- Single-process, multi-coroutine async application
- `asyncio.gather()` runs all subsystems concurrently: ingestors, strategy engine, execution engine, portfolio tracker, TTL cleanup
- No Redis, no message broker, no external database
- All state is SQLite + in-memory Python dicts
- WebSocket connections are outbound-only to exchanges
## Key Decisions
- **SQLite over Postgres/Redis**: Embeddable, zero-ops, WAL mode for concurrent reads. Appropriate for single-instance deployment.
- **EventBus pattern (in-process pub/sub)**: `asyncio.Queue`-based channels replace a message broker. All components communicate via `EventBus.publish()` / `EventBus.subscribe()`. Channels: `market_data`, `order_intents`, `fills`, `strategy_halted`.
- **Strategy hot-reload**: Strategies are loaded from DB (`strategies` table) and dynamically imported via `importlib.import_module()` with configurable reload interval.
- **Fractional Kelly sizing**: Position sizing uses quarter-Kelly (default 0.25 fraction) adjusted by confidence and capped at `max_position_usd`.
- **Exchange-agnostic adapter pattern**: `ExchangeAdapter` ABC with `KalshiAdapter` and `PolymarketAdapter` implementations. New exchanges need only implement the adapter interface.
- **RSA PSS authentication for Kalshi**: Private key stored in env var, used to sign every HTTP request and WebSocket connection with PSS+SHA256.
- **Polymarket auth incomplete**: `PolymarketAdapter` has a TODO noting production requires per-request ECDSA signing via `api_secret`/`api_passphrase`, which is not yet implemented (currently only API key header auth).
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Style & Formatting
- **Indentation:** 4 spaces (Python standard)
- **Line length:** No enforced limit observed; lines generally stay under 120 characters
- **Strings:** Double quotes for f-strings and regular strings; single quotes for dict keys and short literals (both used inconsistently)
- **Trailing commas:** Not consistently used; absent from multi-line function calls like `MarketDataEvent(...)` constructors
- **Blank lines:** 2 blank lines between top-level definitions, 1 blank line between method definitions (PEP 8)
- **No formatter configured:** No black, ruff, or yapf config found in `pyproject.toml`; formatting relies on developer convention
- **No linter configured:** No flake8, pylint, or ruff linting rules defined in `pyproject.toml`
## Naming Patterns
- PascalCase — `BaseIngestor`, `KalshiIngestor`, `ExecutionEngine`, `RiskChecker`, `PortfolioTracker`
- Abstract base classes prefixed with `Base` — `BaseIngestor`, `BaseStrategy`, `ExchangeAdapter`
- snake_case — `kelly_size()`, `get_connection()`, `cmd_status()`, `_make_auth_headers()`
- Private methods prefixed with single underscore — `_connect_and_stream()`, `_handle_intent()`, `_persist_fill()`
- Module-level private functions prefixed with underscore — `_check_env()`, `_setup_logging()`, `_load_private_key()`
- snake_case — `market_ids`, `yes_bid`, `fill_price`, `unrealized_pnl`
- Private instance attributes prefixed with underscore — `self._bus`, `self._conn`, `self._strategies`, `self._price_cache`
- Constants in UPPER_SNAKE_CASE at module level — `_WS_URL`, `_MAX_RETRIES`, `_DEFAULTS`, `_SCHEMA`, `DB_PATH`
- Data classes use PascalCase — `MarketDataEvent`, `OrderIntent`, `FillEvent`, `StrategyHaltedEvent`
- No suffix conventions like `Model`, `Dto`, or `Data`; just nouns
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
- **Log message format:** Components prefix with module context — `"execution:fill"`, `"execution:strategy_halted"`, `"risk:debounce"`, `"risk:daily_loss_limit"`, `"portfolio:fill"`, `"kalshi POST %s attempt %d failed"`
- **Rotating file handler:** 10MB max, 3 backups, writing to `albert.log`; also streams to stdout
- **No lazy %-formatting:** Uses `%` style format args with logger (e.g., `logger.info("loaded strategy %s", sid)`) rather than f-strings in log calls
## Module Design
- **Package structure with empty `__init__.py`:** All packages (`albert/`, `albert/ingestor/`, `albert/strategies/`, `albert/strategies/examples/`, `albert/execution/`, `albert/execution/adapters/`, `albert/portfolio/`) use empty `__init__.py`
- **No re-exports:** Clients import directly from submodules, e.g., `from albert.strategies.engine import StrategyEngine`
- **Class-per-file pattern:** Most modules contain a single primary class with the same name as the file — `tracker.py` → `PortfolioTracker`, `risk.py` → `RiskChecker`
- **Helper functions are module-private:** `_load_private_key()`, `_check_env()`, `_setup_logging()`, `_make_auth_headers()` are module-level private functions
- **Constants at module top:** `_WS_URL`, `_BASE_URL`, `_MAX_RETRIES`, `_SCHEMA`, `_DEFAULTS`, `_REQUIRED_ENV`, `_LOG_FORMAT`, `DB_PATH` all defined at the top of their respective modules
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
## Module Architecture
### `albert.events` — Event Bus (Publish/Subscribe)
- **Purpose:** In-process async message bus using `asyncio.Queue` per subscriber
- **Key class:** `EventBus` — channels: `"market_data"`, `"order_intents"`, `"fills"`, `"strategy_halted"`
- **Event types:** `MarketDataEvent`, `OrderIntent`, `FillEvent`, `StrategyHaltedEvent` (all dataclasses)
- **Pattern:** Fan-out — each subscriber gets its own queue; publishing to a channel delivers to all subscribers
### `albert.ingestor` — Data Ingestion Layer
- **Purpose:** Connect to exchange WebSockets, normalize market data, and publish `MarketDataEvent`s
- **Base class:** `BaseIngestor` (`albert/ingestor/base.py`) — abstract ABC with auto-reconnect loop
- **Implementations:**
- **Data flow:** Raw WebSocket message → `_normalize()` → `MarketDataEvent` → `bus.publish("market_data", event)`
- **Reconnect:** `BaseIngestor.run()` catches all exceptions and reconnects after configurable delay (default 5s)
### `albert.strategies` — Strategy Engine
- **Purpose:** Dynamically load and run trading strategies that analyze market data and emit order intents
- **Base class:** `BaseStrategy` (`albert/strategies/base.py`) — abstract ABC requiring `on_market_data()` and `estimate_edge()`
- **Engine:** `StrategyEngine` (`albert/strategies/engine.py`)
- **Example strategy:** `MomentumV1` (`albert/strategies/examples/momentum.py`) — buys YES when `yes_ask < 0.5 - min_edge`
### `albert.execution` — Execution Layer
- **Purpose:** Validate risk, compute Kelly sizing, route orders to exchange adapters, and persist fills
- **Components:**
- **Exchange Adapters** (`albert/execution/adapters/`):
- **Market ID routing:** `market_id` format is `"{exchange}:{ticker}"` — prefix determines which adapter to use
- **Strategy halting:** On order placement failure, the engine sets `enabled = 0` in the `strategies` table and publishes a `StrategyHaltedEvent`
### `albert.portfolio` — Portfolio Tracking
- **Purpose:** Maintain position state and compute P&L from fills and market data
- **Key class:** `PortfolioTracker` (`albert/portfolio/tracker.py`)
### `albert.db` — Database Layer
- **Purpose:** SQLite storage with WAL mode for concurrent read/write
- **Key function:** `get_connection()` returns `sqlite3.Connection` with `row_factory = sqlite3.Row`
- **Migration:** `migrate()` runs idempotent `CREATE TABLE IF NOT EXISTS` statements
- **Tables:** `markets`, `orderbook_snapshots`, `positions`, `fills`, `strategies`, `daily_pnl`
### `albert.config` — Configuration
- **Purpose:** Load configuration from `config.json` (overrides defaults) and `.env` (environment variables)
- **Defaults:** `max_total_notional_usd=10000`, `daily_loss_limit_usd=-500`, `order_debounce_seconds=10`, `orderbook_ttl_days=7`, `strategy_reload_interval=30.0`
- **Env loading:** `.env` file parsed manually; does not override existing env vars
### `albert.cli` — CLI Interface
- **Purpose:** `python -m albert status` command prints a formatted table of strategy positions and P&L
- **Key function:** `cmd_status()` queries `positions` and `daily_pnl` tables and prints a summary
## Key Design Patterns
### 1. Event-Driven Architecture (Publish/Subscribe)
### 2. Strategy Plugin Pattern
### 3. Adapter Pattern (Exchange Abstraction)
### 4. Template Method Pattern (Ingestor)
### 5. Fractional Kelly Criterion (Position Sizing)
### 6. Risk Gate Pattern
## Data Flow
### Main Trading Loop
### State Management
- **Runtime state:** In-memory (`EventBus` queues, `StrategyEngine._strategies`, `ExecutionEngine._price_cache`, `RiskChecker._last_order_time`)
- **Persistent state:** SQLite (`albert.db`) with WAL mode
- **Configuration:** `config.json` file + `.env` environment variables
## Entry Points
### Main Entry Point
- **Location:** `albert/__main__.py`
- **Invocation:** `python -m albert` (long-running async service) or `python -m albert status` (one-shot status check)
- **Startup sequence:**
### Status CLI
- **Location:** `albert/cli.py`
- **Function:** `cmd_status(conn)` — queries strategy P&L and prints formatted table
## Error Handling Strategy
### Reconnection (Ingestors)
### Retry (Exchange Adapters)
### Strategy Halting
### Strategy Error Isolation
### Risk Guard Rails
### Database Error Handling
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
