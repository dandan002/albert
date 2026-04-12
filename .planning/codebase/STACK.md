# Tech Stack

**Analysis Date:** 2026-04-12

## Language & Runtime

**Primary:**
- Python 3.11+ (requires `>=3.11` per `pyproject.toml`) — used for all application code
- Python 3.12.2 detected on development system

**Runtime:**
- Async Python (`asyncio`) — the entire application is async-first; all core loops use `async def run()` and `asyncio.gather`
- No web framework (pure async event loop, not ASGI/WSGI)

## Package Manager & Build

**Package Manager:**
- pip (standard Python packaging)
- setuptools >=68 as build backend per `pyproject.toml`

**Build:**
- `pyproject.toml` — modern PEP 621 metadata, setuptools build backend
- Package installed in editable mode (`albert.egg-info` present)
- No lockfile committed (no `requirements.txt`, no `poetry.lock`, no `pdm.lock`)

**Configuration:**
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

**Database:**
- SQLite 3 — embedded database, file-based (`albert.db`)
- WAL mode enabled (`PRAGMA journal_mode=WAL`) in `albert/db.py`
- Schema managed via in-code `_SCHEMA` string in `albert/db.py`, executed with `executescript` (no migration tool)
- Tables: `markets`, `orderbook_snapshots`, `positions`, `fills`, `strategies`, `daily_pnl`

**Logging:**
- Python stdlib `logging` with structured JSON format to `albert.log` (rotating, 10MB, 3 backups)
- Also logs to stdout with same JSON format

**Process Model:**
- Single-process, multi-coroutine async application
- `asyncio.gather()` runs all subsystems concurrently: ingestors, strategy engine, execution engine, portfolio tracker, TTL cleanup

**No External Services Required for Core:**
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

---

*Stack analysis: 2026-04-12*