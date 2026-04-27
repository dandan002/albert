# Albert Trading System

A fully automated prediction market trading bot that connects to **Kalshi** and **Polymarket** exchanges, ingests real-time orderbook data via WebSockets, runs pluggable trading strategies with Kelly criterion position sizing, and tracks portfolio P&L.

> **Core Value:** Automated, risk-managed trading on prediction markets with pluggable strategies and unified order execution across exchanges.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [CLI Commands](#cli-commands)
- [Project Structure](#project-structure)
- [Development](#development)

---

## Features

- **Multi-Exchange Support:** Connects to Kalshi and Polymarket with exchange-agnostic adapter pattern
- **Real-Time Data Ingestion:** WebSocket orderbook feeds with auto-reconnection and data normalization
- **Pluggable Strategies:** Hot-reloadable trading strategies loaded dynamically from the database
- **Kelly Criterion Sizing:** Fractional Kelly (default 0.25) position sizing adjusted by confidence
- **Risk Management:** Configurable daily loss limits, position caps, and order debouncing
- **Portfolio Tracking:** Real-time P&L computation from fills and market data
- **Event-Driven Architecture:** Async pub/sub via in-process EventBus using `asyncio.Queue`
- **SQLite Persistence:** WAL-mode SQLite for zero-ops embedded storage

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Concurrency | `asyncio` (single process, multi-coroutine) |
| WebSocket | `websockets` >=12.0 |
| HTTP Client | `httpx` >=0.27 |
| Cryptography | `cryptography` >=42.0 (RSA PSS signing for Kalshi) |
| Database | SQLite 3 (WAL mode) |
| Testing | `pytest` >=8.0, `pytest-asyncio` >=0.23 |

---

## Architecture

Albert follows an **event-driven, single-process async architecture**. All subsystems run concurrently via `asyncio.gather()` and communicate through an in-process publish/subscribe bus.

### High-Level Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Kalshi WS      │────▶│                  │     │                 │
│  Polymarket WS  │────▶│  Data Ingestors  │────▶│  EventBus       │
└─────────────────┘     └──────────────────┘     │  (market_data)  │
                                                  └────────┬────────┘
                                                           │
                              ┌────────────────────────────┼────────────────────────────┐
                              │                            │                            │
                              ▼                            ▼                            ▼
                    ┌─────────────────┐          ┌──────────────────┐          ┌─────────────────┐
                    │ Strategy Engine │          │ Portfolio Tracker│          │ Execution Engine│
                    │ (on_market_data)│          │ (positions, P&L) │          │ (risk → order)  │
                    └────────┬────────┘          └──────────────────┘          └────────┬────────┘
                             │                                                         │
                             ▼ order_intents                                           ▼ fills
                    ┌─────────────────┐                                      ┌──────────────────┐
                    │ Execution Engine│                                      │ Portfolio Tracker│
                    │ (Kelly sizing)  │                                      │ (update P&L)     │
                    └────────┬────────┘                                      └──────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Exchange Adapter│
                    │ (Kalshi/Poly)   │
                    └─────────────────┘
```

### Key Components

| Module | Purpose |
|--------|---------|
| `albert.events` | Event bus with channels: `market_data`, `order_intents`, `fills`, `strategy_halted` |
| `albert.ingestor` | WebSocket ingestion with auto-reconnect and normalization |
| `albert.strategies` | Dynamic strategy loading and execution engine |
| `albert.execution` | Risk checks, Kelly sizing, order routing, fill persistence |
| `albert.portfolio` | Position tracking and P&L computation |
| `albert.db` | SQLite schema and connection management |
| `albert.config` | Configuration and environment loading |
| `albert.cli` | Status CLI command |

### Design Patterns

- **Event-Driven Pub/Sub:** `EventBus` using `asyncio.Queue` per subscriber
- **Strategy Plugin:** Strategies loaded dynamically via `importlib` from the database
- **Adapter Pattern:** `ExchangeAdapter` ABC with `KalshiAdapter` and `PolymarketAdapter`
- **Template Method:** `BaseIngestor` defines reconnect loop; subclasses implement `_normalize()`
- **Risk Gate:** `RiskChecker` validates orders before execution
- **Fractional Kelly:** Position sizing capped by `max_position_usd`

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd albert

# Install in editable mode
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root with your exchange credentials:

```bash
# Kalshi
KALSHI_KEY_ID=your_kalshi_key_id
KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"

# Polymarket
POLYMARKET_API_KEY=your_polymarket_api_key
POLYMARKET_API_SECRET=your_polymarket_secret
POLYMARKET_API_PASSPHRASE=your_polymarket_passphrase
```

> **Note:** `.env` is gitignored. Never commit credentials.

---

## Configuration

Runtime behavior is controlled via `config.json` (optional — sensible defaults are provided):

```json
{
  "max_total_notional_usd": 10000,
  "daily_loss_limit_usd": -500,
  "order_debounce_seconds": 10,
  "orderbook_ttl_days": 7,
  "strategy_reload_interval": 30.0
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `max_total_notional_usd` | 10000 | Maximum total position exposure |
| `daily_loss_limit_usd` | -500 | Halt trading when daily P&L falls below this |
| `order_debounce_seconds` | 10 | Minimum time between orders for the same market |
| `orderbook_ttl_days` | 7 | How long to retain orderbook snapshots |
| `strategy_reload_interval` | 30.0 | Seconds between strategy hot-reload checks |

---

## Running the Bot

### Start the Trading Service

```bash
python -m albert
```

This starts the full async trading loop:
1. Migrates the SQLite database (`albert.db`)
2. Loads strategies from the `strategies` table
3. Spins up ingestors, strategy engine, execution engine, and portfolio tracker
4. Runs indefinitely until interrupted

### Check Status

```bash
python -m albert status
```

Prints a formatted table of current strategy positions and P&L.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `python -m albert` | Start the trading service (long-running) |
| `python -m albert status` | Query and display strategy positions and P&L |

---

## Project Structure

```
albert/
├── __init__.py
├── __main__.py           # Entry point: starts the async trading loop
├── cli.py                # CLI commands (status)
├── config.py             # Config loading from config.json and .env
├── db.py                 # SQLite schema and connection management
├── events.py             # EventBus and event dataclasses
├── ingestor/
│   ├── __init__.py
│   ├── base.py           # BaseIngestor (auto-reconnect loop)
│   ├── kalshi.py         # Kalshi WebSocket ingestor
│   └── polymarket.py     # Polymarket WebSocket ingestor
├── strategies/
│   ├── __init__.py
│   ├── base.py           # BaseStrategy ABC
│   ├── engine.py         # StrategyEngine (hot-reload, execution loop)
│   └── examples/
│       └── momentum.py   # MomentumV1 example strategy
├── execution/
│   ├── __init__.py
│   ├── engine.py         # ExecutionEngine (risk → Kelly → order)
│   ├── kelly.py          # Kelly criterion sizing logic
│   ├── risk.py           # RiskChecker (limits, debounce)
│   └── adapters/
│       ├── __init__.py
│       ├── base.py       # ExchangeAdapter ABC
│       ├── kalshi.py     # Kalshi REST adapter + RSA PSS auth
│       └── polymarket.py # Polymarket REST adapter
├── portfolio/
│   ├── __init__.py
│   └── tracker.py        # PortfolioTracker (positions, P&L)
config.json               # Runtime configuration (optional)
albert.db                 # SQLite database (created at runtime)
albert.log                # Rotating JSON logs (10MB, 3 backups)
```

---

## Development

### Running Tests

```bash
pytest
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"`. The test suite covers:
- Event bus pub/sub
- Ingestor normalization and reconnection
- Strategy engine loading and execution
- Execution engine risk checks and order flow
- Kelly criterion calculations
- Portfolio tracking and P&L computation

### Adding a New Exchange Adapter

1. Create a new file in `albert/execution/adapters/`
2. Subclass `ExchangeAdapter` and implement:
   - `place_order()`, `cancel_order()`, `get_balance()`
3. Implement a corresponding ingestor in `albert/ingestor/`
4. Register routing in `ExecutionEngine` based on `market_id` prefix

### Adding a New Strategy

1. Create a Python module with a class subclassing `BaseStrategy`
2. Implement `on_market_data()` and `estimate_edge()`
3. Insert a row into the `strategies` table with:
   - `strategy_id`: unique identifier
   - `module_path`: import path (e.g., `albert.strategies.examples.momentum`)
   - `class_name`: strategy class name
   - `enabled`: 1 to activate
   - `config_json`: JSON string of strategy parameters

The `StrategyEngine` will hot-reload it on the next interval.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
