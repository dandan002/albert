# Architecture

**Analysis Date:** 2026-04-12

## System Overview

Albert is an automated prediction market trading bot that connects to prediction market exchanges (Kalshi, Polymarket), ingests live orderbook data via WebSockets, runs configurable trading strategies against the data, executes trades through exchange adapters, and tracks portfolio positions and P&L — all running as a single long-lived asyncio process.

The system follows an **event-driven architecture** with a central `EventBus` that connects five concurrent async services: two data ingestors, a strategy engine, an execution engine, and a portfolio tracker.

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
  - `KalshiIngestor` (`albert/ingestor/kalshi.py`) — WebSocket to `wss://api.elections.kalshi.com/trade-api/ws/v2`, RSA-signed auth headers
  - `PolymarketIngestor` (`albert/ingestor/polymarket.py`) — WebSocket to `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- **Data flow:** Raw WebSocket message → `_normalize()` → `MarketDataEvent` → `bus.publish("market_data", event)`
- **Reconnect:** `BaseIngestor.run()` catches all exceptions and reconnects after configurable delay (default 5s)

### `albert.strategies` — Strategy Engine

- **Purpose:** Dynamically load and run trading strategies that analyze market data and emit order intents
- **Base class:** `BaseStrategy` (`albert/strategies/base.py`) — abstract ABC requiring `on_market_data()` and `estimate_edge()`
- **Engine:** `StrategyEngine` (`albert/strategies/engine.py`)
  - Subscribes to `"market_data"` channel
  - Loads strategies from `strategies` DB table using `class_path` (dynamic `importlib.import_module`)
  - Hot-reloads strategy config every `reload_interval` seconds (default 30s)
  - For each `MarketDataEvent`, calls every enabled strategy's `on_market_data()` and publishes resulting `OrderIntent`s
- **Example strategy:** `MomentumV1` (`albert/strategies/examples/momentum.py`) — buys YES when `yes_ask < 0.5 - min_edge`

### `albert.execution` — Execution Layer

- **Purpose:** Validate risk, compute Kelly sizing, route orders to exchange adapters, and persist fills
- **Components:**
  - `ExecutionEngine` (`albert/execution/engine.py`) — subscribes to `"order_intents"` and `"market_data"`, caches prices, resolves exchange adapter, computes position size via Kelly criterion, runs risk checks, places orders, persists fills, publishes `FillEvent`s
  - `RiskChecker` (`albert/execution/risk.py`) — enforces order debounce, daily loss limit, and max total notional exposure
  - `kelly_size()` (`albert/execution/kelly.py`) — fractional Kelly criterion position sizer; returns USD position size given edge, ask price, bankroll, and config
- **Exchange Adapters** (`albert/execution/adapters/`):
  - `ExchangeAdapter` ABC (`base.py`) — `place_order()`, `cancel_order()`, `get_bankroll()`
  - `KalshiAdapter` (`kalshi.py`) — REST API with PSS-signed RSA auth, exponential backoff retries
  - `PolymarketAdapter` (`polymarket.py`) — REST API with API key auth, exponential backoff retries (TODO: ECDSA signing not implemented)
- **Market ID routing:** `market_id` format is `"{exchange}:{ticker}"` — prefix determines which adapter to use
- **Strategy halting:** On order placement failure, the engine sets `enabled = 0` in the `strategies` table and publishes a `StrategyHaltedEvent`

### `albert.portfolio` — Portfolio Tracking

- **Purpose:** Maintain position state and compute P&L from fills and market data
- **Key class:** `PortfolioTracker` (`albert/portfolio/tracker.py`)
  - Subscribes to `"fills"` and `"market_data"` channels
  - On fill: Creates/updates/closes positions in `positions` table; computes realized P&L on partial/full closes; records to `daily_pnl` table
  - On market data: Updates `current_price` and `unrealized_pnl` for all positions in that market
  - Uses weighted average entry price for position tracking

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
All inter-component communication flows through `EventBus` with typed dataclass events. Components subscribe to named channels and process events asynchronously. This decouples producers from consumers entirely.

### 2. Strategy Plugin Pattern
Strategies are loaded dynamically from the `strategies` DB table using `class_path` (e.g., `"albert.strategies.examples.momentum.MomentumV1"`). This allows adding new strategies without code changes — just insert a DB row. The `StrategyEngine` reloads config at runtime via configurable interval.

### 3. Adapter Pattern (Exchange Abstraction)
`ExchangeAdapter` ABC defines `place_order()`, `cancel_order()`, `get_bankroll()`. Concrete adapters (`KalshiAdapter`, `PolymarketAdapter`) implement exchange-specific API details. `ExecutionEngine` routes by market ID prefix to the correct adapter.

### 4. Template Method Pattern (Ingestor)
`BaseIngestor` defines the `run()` reconnect loop and delegates `_connect_and_stream()` and `_normalize()` to subclasses. New exchanges only need to implement these two methods.

### 5. Fractional Kelly Criterion (Position Sizing)
`kelly_size()` computes optimal position size using `f* = (p*b - q) / b` scaled by `kelly_fraction` and `confidence`, capped by `max_position_usd`. Conservative defaults (quarter-Kelly at 0.25).

### 6. Risk Gate Pattern
`RiskChecker.check()` acts as a gate before order placement, enforcing three constraints: debounce period, daily loss limit, and max notional exposure. Returns `bool` — `True` passes, `False` blocks.

## Data Flow

### Main Trading Loop

1. **Data Ingestion:** `KalshiIngestor` / `PolymarketIngestor` connect to exchange WebSockets, normalize data, and publish `MarketDataEvent` on `"market_data"` channel
2. **Strategy Evaluation:** `StrategyEngine` receives `MarketDataEvent`, iterates enabled strategies, each strategy's `on_market_data()` returns optional `list[OrderIntent]`
3. **Order Routing:** `OrderIntent`s published on `"order_intents"` channel
4. **Risk & Sizing:** `ExecutionEngine` receives `OrderIntent`, looks up ask price from price cache, fetches bankroll, computes Kelly size, runs risk checks
5. **Order Placement:** If risk passes, `ExecutionEngine` calls `adapter.place_order()` which returns a `FillEvent`
6. **Fill Processing:** `ExecutionEngine` persists fill to DB and publishes `FillEvent` on `"fills"` channel
7. **Position Update:** `PortfolioTracker` receives `FillEvent`, creates/updates/deletes position in DB and records realized P&L
8. **Price Update:** `PortfolioTracker` also subscribes to `"market_data"`, updating `current_price` and `unrealized_pnl` on open positions

### State Management
- **Runtime state:** In-memory (`EventBus` queues, `StrategyEngine._strategies`, `ExecutionEngine._price_cache`, `RiskChecker._last_order_time`)
- **Persistent state:** SQLite (`albert.db`) with WAL mode
- **Configuration:** `config.json` file + `.env` environment variables

## Entry Points

### Main Entry Point
- **Location:** `albert/__main__.py`
- **Invocation:** `python -m albert` (long-running async service) or `python -m albert status` (one-shot status check)
- **Startup sequence:**
  1. CLI arg check: `"status"` → `cmd_status()` then exit
  2. Setup logging (rotating file handler + stdout, JSON format)
  3. Load `.env` via `load_project_env()`
  4. Check required environment variables (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY`)
  5. Initialize SQLite connection and migrate
  6. Load global config from `config.json`
  7. Launch `asyncio.gather()` with all services: KalshiIngestor, PolymarketIngestor, StrategyEngine, ExecutionEngine, PortfolioTracker, and TTL cleanup task

### Status CLI
- **Location:** `albert/cli.py`
- **Function:** `cmd_status(conn)` — queries strategy P&L and prints formatted table

## Error Handling Strategy

### Reconnection (Ingestors)
`BaseIngestor.run()` wraps the streaming loop in a `while True` with generic exception catch. On any disconnect/error, it logs the exception and retries after `reconnect_delay` (default 5s). `asyncio.CancelledError` is re-raised to permit clean shutdown.

### Retry (Exchange Adapters)
Both `KalshiAdapter` and `PolymarketAdapter` implement exponential backoff retries (`_post_with_retry`) with `_MAX_RETRIES = 3` and `2^attempt` delay.

### Strategy Halting
When `adapter.place_order()` raises an exception in `ExecutionEngine`, the engine halts that strategy by setting `enabled = 0` in the database and publishing a `StrategyHaltedEvent`. The strategy will not be reloaded on the next reload cycle.

### Strategy Error Isolation
`StrategyEngine` catches exceptions per-strategy in the market data loop. A failing strategy does not affect other strategies.

### Risk Guard Rails
Three layers of risk protection in `RiskChecker`:
1. **Debounce:** Prevents rapid repeated orders on same market/strategy pair
2. **Daily loss limit:** Blocks all orders if daily P&L drops below threshold
3. **Max notional:** Blocks orders that would exceed total portfolio exposure cap

### Database Error Handling
Not explicitly handled — SQLite operations assume success. Connection uses `check_same_thread=False` for shared access across async tasks.

---

*Architecture analysis: 2026-04-12*