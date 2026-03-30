# Albert Trading System — Design Spec
**Date:** 2026-03-29
**Status:** Approved

---

## Overview

Albert is a fully automated prediction market trading system that ingests real-time data from Kalshi and Polymarket, evaluates strategies, sizes positions using Kelly criterion, and executes orders via exchange REST APIs. It is implemented as a modular Python monolith with an internal async event bus.

---

## Architecture

Single Python process (`python -m albert`) with five modules communicating over an internal asyncio event bus. One SQLite database for persistence.

```
┌─────────────────────────────────────────────────────────┐
│                     albert (main process)                │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │   Ingestor   │    │       Event Bus (asyncio)    │   │
│  │              │───▶│  market_data / fills / risk  │   │
│  │ Kalshi WS    │    └──────────────┬───────────────┘   │
│  │ Polymarket WS│                   │                   │
│  └──────────────┘          ┌────────┼────────┐          │
│                            ▼        ▼        ▼          │
│                    ┌───────────┐ ┌──────┐ ┌──────────┐  │
│                    │ Strategy  │ │ Port │ │Execution │  │
│                    │ Engine    │ │Track │ │ Engine   │  │
│                    └───────────┘ └──────┘ └──────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              SQLite (persistence)               │    │
│  │  markets | orderbook | positions | fills | pnl  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Modules

1. **Ingestor** — connects to Kalshi and Polymarket websockets, normalizes events to a common `MarketDataEvent` schema, publishes to event bus
2. **Strategy Engine** — subscribes to market data, loads strategy classes + SQLite configs, evaluates signals, publishes `OrderIntent` events
3. **Execution Engine** — receives order intents, applies Kelly sizing and risk checks, calls exchange REST APIs, publishes `FillEvent`
4. **Portfolio Tracker** — subscribes to fills and market data, maintains live P&L, positions, and returns per strategy
5. **Event Bus** — thin asyncio queue wrapper with typed channels; keeps modules fully decoupled

---

## Data Model

Five SQLite tables:

### `markets`
Canonical record of each tradeable market across both exchanges.

| Column | Type | Notes |
|--------|------|-------|
| market_id | TEXT PK | exchange-prefixed, e.g. `kalshi:BTCZ-24` |
| exchange | TEXT | `kalshi` or `polymarket` |
| title | TEXT | human-readable market name |
| close_time | DATETIME | market resolution time |
| status | TEXT | `open`, `closed`, `resolved` |
| metadata | JSON | exchange-specific extra fields |

### `orderbook_snapshots`
Time-series of best bid/ask from the websocket stream. Pruned on a configurable TTL.

| Column | Type |
|--------|------|
| market_id | TEXT FK |
| timestamp | DATETIME |
| yes_bid | REAL |
| yes_ask | REAL |
| no_bid | REAL |
| no_ask | REAL |
| last_price | REAL |
| volume | REAL |

### `positions`
Current open positions, one row per (market, strategy).

| Column | Type |
|--------|------|
| market_id | TEXT FK |
| strategy_id | TEXT FK |
| side | TEXT | `yes` or `no` |
| contracts | REAL |
| avg_entry_price | REAL |
| current_price | REAL |
| unrealized_pnl | REAL |
| opened_at | DATETIME |

### `fills`
Immutable record of every executed order.

| Column | Type |
|--------|------|
| fill_id | TEXT PK |
| market_id | TEXT FK |
| strategy_id | TEXT FK |
| side | TEXT |
| contracts | REAL |
| fill_price | REAL |
| fee | REAL |
| filled_at | DATETIME |

### `strategies`
Registry of deployed strategies with runtime-configurable parameters.

| Column | Type | Notes |
|--------|------|-------|
| strategy_id | TEXT PK | |
| name | TEXT | human-readable |
| class_path | TEXT | e.g. `albert.strategies.momentum.MomentumV1` |
| config | JSON | runtime params, e.g. `{"min_edge": 0.05, "kelly_fraction": 0.25}` |
| enabled | BOOLEAN | |
| created_at | DATETIME | |

---

## Strategy Interface

### BaseStrategy

Every strategy inherits from `BaseStrategy`:

```python
class BaseStrategy:
    id: str          # matches strategies.strategy_id
    config: dict     # loaded from DB at startup, hot-reloadable

    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        """Called on every orderbook update. Return order intents or None."""
        ...

    def estimate_edge(self, event: MarketDataEvent) -> float:
        """Return estimated probability edge (0.0–1.0). Used by Kelly sizer."""
        ...
```

Strategies express intent only — they never call exchange APIs directly. All sizing and execution is handled downstream.

### OrderIntent

```python
@dataclass
class OrderIntent:
    market_id: str
    strategy_id: str
    side: str           # "yes" | "no"
    edge: float         # estimated probability edge
    confidence: float   # 0.0–1.0, scales Kelly fraction
```

### Config Hot-Reload

Strategies poll their `config` row in SQLite every N seconds (configurable, default 30s). Changes to `min_edge`, `kelly_fraction`, `max_position_usd`, etc. take effect without restarting the process.

---

## Execution Engine

### Kelly Position Sizing

```
f* = (edge * b - (1 - edge)) / b
position_size = bankroll * f* * kelly_fraction * confidence
```

Where:
- `b` = implied odds from the current ask price
- `bankroll` = total portfolio value (cash + mark-to-market positions)
- `kelly_fraction` = configurable fraction per strategy (e.g. 0.25 for quarter-Kelly)
- `confidence` = from `OrderIntent`, allows strategies to scale sizing

A hard cap (`max_position_usd` in strategy config) acts as a ceiling regardless of Kelly output.

### Pre-Trade Risk Checks

Applied before every order, in order:
1. Max position per market (contracts) — per strategy config
2. Max total notional exposure across all open positions — global config
3. Daily loss limit — if realized + unrealized PnL crosses a floor, halt all new orders
4. Duplicate order guard — debounce window prevents re-entering the same market within N seconds

Any failed check logs a structured rejection record and discards the intent.

### Exchange Adapters

```python
class ExchangeAdapter:
    async def place_order(self, intent: OrderIntent) -> Fill: ...
    async def cancel_order(self, order_id: str) -> None: ...
    async def get_positions(self) -> list[Position]: ...
```

One adapter per exchange (`KalshiAdapter`, `PolymarketAdapter`). Adding a new exchange requires only a new adapter implementation.

### Error Handling

- Transient API failures: retry with exponential backoff, 3 attempts
- Persistent failures: disable affected strategy, emit `StrategyHaltedEvent`
- No silent failures — every error produces a structured log record

---

## Portfolio Tracker

Subscribes to `FillEvent` and `MarketDataEvent`:

- **Positions**: updated on every fill; current price refreshed on every market data tick
- **Unrealized P&L**: `(current_price - avg_entry) * contracts` per position
- **Realized P&L**: recorded when a fill reduces position contracts to zero
- **Returns**: daily and since-inception per strategy, written to a `daily_pnl` table

---

## Observability

- **Structured logging**: JSON output via Python `logging`, to stdout + rotating file
- Every significant event logs `strategy_id`, `market_id`, `timestamp`, and relevant fields
- **CLI status command**: `python -m albert status` prints live snapshot:

```
Strategy        Positions  Unrealized PnL  Realized PnL  Today
─────────────────────────────────────────────────────────────
momentum_v1     3          +$42.10         +$120.00      +$18.40
mean_revert     1          -$8.20          +$55.00       -$8.20
─────────────────────────────────────────────────────────────
TOTAL           4          +$33.90         +$175.00      +$10.20
```

No web UI in initial scope.

---

## Out of Scope (v1)

- Web dashboard
- Backtesting engine
- Multi-account support
- Cloud deployment / process supervision (run locally)
