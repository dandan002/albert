# Project Structure

**Analysis Date:** 2026-04-12

## Directory Tree

```
albert/                          # Project root
├── albert/                      # Main package
│   ├── __init__.py              # Empty (package marker)
│   ├── __main__.py              # Entry point: asyncio.gather() service launcher
│   ├── cli.py                   # CLI: `status` command for P&L summary
│   ├── config.py                # Configuration: config.json + .env loader
│   ├── db.py                    # SQLite: connection, migration, schema
│   ├── events.py                # EventBus + event dataclasses
│   ├── execution/               # Order execution subsystem
│   │   ├── __init__.py          # Empty (package marker)
│   │   ├── engine.py            # ExecutionEngine: risk + Kelly + order routing
│   │   ├── risk.py              # RiskChecker: debounce, loss limit, notional cap
│   │   ├── kelly.py             # Fractional Kelly criterion position sizer
│   │   └── adapters/            # Exchange adapter implementations
│   │       ├── __init__.py      # Empty (package marker)
│   │       ├── base.py          # ExchangeAdapter ABC
│   │       ├── kalshi.py        # Kalshi REST adapter (RSA auth)
│   │       └── polymarket.py    # Polymarket REST adapter (API key auth)
│   ├── ingestor/                # Market data ingestion subsystem
│   │   ├── __init__.py          # Empty (package marker)
│   │   ├── base.py              # BaseIngestor ABC (reconnect loop)
│   │   ├── kalshi.py            # Kalshi WebSocket ingestor
│   │   └── polymarket.py        # Polymarket WebSocket ingestor
│   ├── portfolio/               # Position & P&L tracking subsystem
│   │   ├── __init__.py          # Empty (package marker)
│   │   └── tracker.py           # PortfolioTracker (position management)
│   └── strategies/              # Strategy subsystem
│       ├── __init__.py          # Empty (package marker)
│       ├── base.py              # BaseStrategy ABC
│       ├── engine.py            # StrategyEngine (hot-reload, dispatch)
│       └── examples/            # Example strategy implementations
│           ├── __init__.py      # Empty (package marker)
│           └── momentum.py     # MomentumV1: buy YES below 0.5
├── tests/                       # Test suite
│   ├── conftest.py              # Empty (no shared fixtures)
│   ├── test_adapters.py         # KalshiAdapter + PolymarketAdapter tests
│   ├── test_cli.py              # cmd_status output tests
│   ├── test_db.py               # Schema migration tests
│   ├── test_events.py           # EventBus pub/sub tests
│   ├── test_execution_engine.py # ExecutionEngine integration tests
│   ├── test_ingestor.py         # KalshiIngestor + PolymarketIngestor tests
│   ├── test_kelly.py            # Kelly sizing unit tests
│   ├── test_main.py             # Config loading tests
│   ├── test_portfolio_tracker.py # Position/P&L tracking tests
│   ├── test_risk.py             # RiskChecker unit tests
│   ├── test_strategy_base.py    # BaseStrategy + MomentumV1 tests
│   └── test_strategy_engine.py  # StrategyEngine integration tests
├── config.json                  # Runtime configuration overrides
├── pyproject.toml               # Project metadata and dependencies
├── albert.db                    # SQLite database (runtime, gitignored)
└── albert.log                   # Rotating log file (runtime, gitignored)
```

## File Inventory

| Path | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `albert/__main__.py` | Application entry point | `_main()`, `_check_env()`, `_setup_logging()`, `_ttl_cleanup()` |
| `albert/cli.py` | CLI status display | `cmd_status()` |
| `albert/config.py` | Configuration loading | `load_global_config()`, `load_project_env()` |
| `albert/db.py` | Database setup | `get_connection()`, `migrate()`, `_SCHEMA` |
| `albert/events.py` | Event system | `EventBus`, `MarketDataEvent`, `OrderIntent`, `FillEvent`, `StrategyHaltedEvent` |
| `albert/execution/engine.py` | Order execution orchestration | `ExecutionEngine` |
| `albert/execution/risk.py` | Risk management | `RiskChecker` |
| `albert/execution/kelly.py` | Position sizing | `kelly_size()` |
| `albert/execution/adapters/base.py` | Exchange adapter interface | `ExchangeAdapter` (ABC) |
| `albert/execution/adapters/kalshi.py` | Kalshi exchange adapter | `KalshiAdapter`, `_load_private_key()` |
| `albert/execution/adapters/polymarket.py` | Polymarket exchange adapter | `PolymarketAdapter` |
| `albert/ingestor/base.py` | Ingestor base class | `BaseIngestor` (ABC) |
| `albert/ingestor/kalshi.py` | Kalshi WebSocket ingestor | `KalshiIngestor` |
| `albert/ingestor/polymarket.py` | Polymarket WebSocket ingestor | `PolymarketIngestor` |
| `albert/portfolio/tracker.py` | Position & P&L tracking | `PortfolioTracker` |
| `albert/strategies/base.py` | Strategy interface | `BaseStrategy` (ABC) |
| `albert/strategies/engine.py` | Strategy loading & dispatch | `StrategyEngine` |
| `albert/strategies/examples/momentum.py` | Example strategy | `MomentumV1` |
| `tests/test_events.py` | EventBus tests | Tests for pub/sub, multi-subscriber |
| `tests/test_db.py` | Database tests | Tests for migration, idempotency |
| `tests/test_kelly.py` | Kelly criterion tests | Edge cases, caps, scaling |
| `tests/test_risk.py` | Risk checker tests | Debounce, loss limit, notional cap |
| `tests/test_cli.py` | CLI output tests | Status table formatting |
| `tests/test_adapters.py` | Exchange adapter tests | Mock HTTP tests for Kalshi/Polymarket |
| `tests/test_ingestor.py` | Ingestor tests | Mock WebSocket tests |
| `tests/test_strategy_base.py` | Strategy unit tests | BaseStrategy, MomentumV1 |
| `tests/test_strategy_engine.py` | Strategy engine tests | Hot-reload, disabled strategies |
| `tests/test_execution_engine.py` | Execution engine tests | Order flow, fill persistence |
| `tests/test_portfolio_tracker.py` | Portfolio tests | Position creation, P&L calculation |
| `tests/test_main.py` | Config loading tests | Defaults, JSON override, .env parsing |
| `tests/conftest.py` | Shared test fixtures | (empty — no shared fixtures) |

## Module Dependencies

### Internal dependency graph (imports)

```
albert/__main__.py
  ├── albert.config        → load_global_config, load_project_env
  ├── albert.db            → get_connection, migrate
  ├── albert.events        → EventBus
  ├── albert.execution.adapters.kalshi → KalshiAdapter
  ├── albert.execution.engine → ExecutionEngine
  ├── albert.ingestor.kalshi  → KalshiIngestor
  ├── albert.ingestor.polymarket → PolymarketIngestor
  ├── albert.portfolio.tracker → PortfolioTracker
  ├── albert.strategies.engine → StrategyEngine
  └── albert.cli           → cmd_status

albert/execution/engine.py
  ├── albert.events        → EventBus, FillEvent, MarketDataEvent, OrderIntent, StrategyHaltedEvent
  ├── albert.execution.adapters.base → ExchangeAdapter
  ├── albert.execution.kelly → kelly_size
  └── albert.execution.risk  → RiskChecker

albert/execution/risk.py
  └── albert.events        → OrderIntent

albert/execution/adapters/kalshi.py
  ├── albert.events        → FillEvent, OrderIntent
  └── albert.execution.adapters.base → ExchangeAdapter

albert/execution/adapters/polymarket.py
  ├── albert.events        → FillEvent, OrderIntent
  └── albert.execution.adapters.base → ExchangeAdapter

albert/ingestor/kalshi.py
  ├── albert.events        → EventBus, MarketDataEvent
  ├── albert.execution.adapters.kalshi → _load_private_key (shared utility)
  └── albert.ingestor.base → BaseIngestor

albert/ingestor/polymarket.py
  ├── albert.events        → EventBus, MarketDataEvent
  └── albert.ingestor.base → BaseIngestor

albert/portfolio/tracker.py
  └── albert.events        → EventBus, FillEvent, MarketDataEvent

albert/strategies/engine.py
  ├── albert.events       → EventBus, MarketDataEvent
  └── albert.strategies.base → BaseStrategy

albert/strategies/examples/momentum.py
  ├── albert.strategies.base → BaseStrategy
  └── albert.events        → MarketDataEvent, OrderIntent
```

### External dependencies (from `pyproject.toml`)

| Package | Used By |
|---------|---------|
| `websockets>=12.0` | `albert/ingestor/kalshi.py`, `albert/ingestor/polymarket.py` |
| `httpx>=0.27` | `albert/execution/adapters/kalshi.py`, `albert/execution/adapters/polymarket.py` |
| `cryptography>=42.0` | `albert/execution/adapters/kalshi.py`, `albert/ingestor/kalshi.py` |

### Cross-module coupling note

`albert/ingestor/kalshi.py` imports `_load_private_key` from `albert/execution/adapters/kalshi.py`. This is a shared utility for RSA key parsing used by both the REST adapter and the WebSocket ingestor.

## Where to Add New Code

### New Exchange Adapter
- Implementation: `albert/execution/adapters/<exchange>.py`
  - Subclass `ExchangeAdapter` from `albert/execution/adapters/base.py`
  - Implement `place_order()`, `cancel_order()`, `get_bankroll()`
- Register in `albert/__main__.py` → `_main()` adapters dict
- Tests: `tests/test_adapters.py`

### New Data Ingestor (WebSocket Source)
- Implementation: `albert/ingestor/<exchange>.py`
  - Subclass `BaseIngestor` from `albert/ingestor/base.py`
  - Implement `_connect_and_stream()` and `_normalize()`
- Launch in `albert/__main__.py` → `_main()` asyncio.gather
- Tests: `tests/test_ingestor.py`

### New Trading Strategy
1. Implement a class in `albert/strategies/examples/<name>.py` (or any importable path)
   - Subclass `BaseStrategy` from `albert/strategies/base.py`
   - Implement `on_market_data()` → `list[OrderIntent] | None`
   - Implement `estimate_edge()` → `float`
2. Insert row into `strategies` DB table: `(strategy_id, name, class_path, config, enabled)`
   - e.g., `class_path = "albert.strategies.examples.momentum.MomentumV1"`
3. `StrategyEngine` will hot-load it on next reload cycle
- Tests: `tests/test_strategy_base.py`, `tests/test_strategy_engine.py`

### New Event Type
- Add dataclass to `albert/events.py`
- Subscribe to the new channel name via `bus.subscribe("channel_name")`
- Publish via `await bus.publish("channel_name", event_instance)`

### New Database Table
- Add `CREATE TABLE IF NOT EXISTS` statement to `_SCHEMA` in `albert/db.py`
- New migration is idempotent (uses `IF NOT EXISTS`)

## Size Metrics

| Module | Lines | Description |
|--------|-------|-------------|
| `albert/execution/engine.py` | 121 | Execution engine — order routing, Kelly sizing, risk checks |
| `albert/execution/adapters/kalshi.py` | 121 | Kalshi REST adapter with RSA signing |
| `albert/portfolio/tracker.py` | 108 | Position management and P&L tracking |
| `albert/ingestor/kalshi.py` | 81 | Kalshi WebSocket ingestor |
| `albert/db.py` | 79 | Database schema and migration |
| `albert/strategies/engine.py` | 73 | Strategy engine with hot-reload |
| `albert/__main__.py` | 90 | Application entry point |
| `albert/ingestor/polymarket.py` | 59 | Polymarket WebSocket ingestor |
| `albert/events.py` | 59 | EventBus and event dataclasses |
| `albert/execution/risk.py` | 52 | Risk checker (debounce, loss limit, notional) |
| `albert/execution/kelly.py` | 47 | Kelly criterion position sizing |
| `albert/cli.py` | 43 | CLI status command |
| `albert/config.py` | 43 | Configuration and env loading |
| `albert/execution/adapters/polymarket.py` | 89 | Polymarket REST adapter |
| `albert/ingestor/base.py` | 33 | Base ingestor ABC |
| `albert/strategies/examples/momentum.py` | 27 | Example momentum strategy |
| `albert/execution/adapters/base.py` | 19 | Exchange adapter ABC |
| `albert/strategies/base.py` | 18 | Strategy base class ABC |
| **Total source** | **~1,100** | All `albert/` Python files |
| **Total tests** | **~900** | All `tests/` Python files |

---

*Structure analysis: 2026-04-12*