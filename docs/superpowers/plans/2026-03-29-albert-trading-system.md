# Albert Trading System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated prediction market trading system that ingests real-time Kalshi/Polymarket data, evaluates pluggable strategies, sizes positions with Kelly criterion, executes orders, and tracks portfolio P&L.

**Architecture:** Modular Python monolith with five async modules (Ingestor, Strategy Engine, Execution Engine, Portfolio Tracker, Event Bus) communicating through typed asyncio queues. One SQLite database for all persistence.

**Tech Stack:** Python 3.11+, `websockets`, `httpx`, `pytest`, `pytest-asyncio`, SQLite (stdlib)

---

## File Structure

```
albert/
├── __init__.py
├── __main__.py                        # entry point: `python -m albert [status]`
├── config.py                          # global config loader (env vars + config.json)
├── db.py                              # SQLite connection, schema, migrate()
├── events.py                          # all event dataclasses + EventBus
├── ingestor/
│   ├── __init__.py
│   ├── base.py                        # BaseIngestor with reconnect loop
│   ├── kalshi.py                      # KalshiIngestor (WS)
│   └── polymarket.py                  # PolymarketIngestor (WS)
├── strategies/
│   ├── __init__.py
│   ├── base.py                        # BaseStrategy ABC + OrderIntent dataclass
│   ├── engine.py                      # StrategyEngine: load, dispatch, hot-reload
│   └── examples/
│       ├── __init__.py
│       └── momentum.py                # MomentumV1: example strategy
├── execution/
│   ├── __init__.py
│   ├── kelly.py                       # kelly_size() pure function
│   ├── risk.py                        # RiskChecker
│   ├── engine.py                      # ExecutionEngine: intent → fill
│   └── adapters/
│       ├── __init__.py
│       ├── base.py                    # ExchangeAdapter ABC
│       ├── kalshi.py                  # KalshiAdapter (REST)
│       └── polymarket.py              # PolymarketAdapter (REST)
├── portfolio/
│   ├── __init__.py
│   └── tracker.py                     # PortfolioTracker
└── cli.py                             # cmd_status()

tests/
├── conftest.py                        # shared fixtures: in-memory DB, EventBus
├── test_db.py
├── test_events.py
├── test_kelly.py
├── test_risk.py
├── test_strategy_engine.py
├── test_execution_engine.py
├── test_portfolio_tracker.py
├── test_ingestor.py
└── test_cli.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `albert/__init__.py`
- Create: `albert/strategies/__init__.py`
- Create: `albert/strategies/examples/__init__.py`
- Create: `albert/ingestor/__init__.py`
- Create: `albert/execution/__init__.py`
- Create: `albert/execution/adapters/__init__.py`
- Create: `albert/portfolio/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the failing test for package imports**

```python
# tests/test_imports.py
def test_package_imports():
    import albert
    import albert.events
    import albert.db
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_imports.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert'`

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "albert"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "websockets>=12.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["albert*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 4: Create all `__init__.py` files**

```bash
mkdir -p albert/ingestor albert/strategies/examples albert/execution/adapters albert/portfolio tests
touch albert/__init__.py
touch albert/ingestor/__init__.py
touch albert/strategies/__init__.py
touch albert/strategies/examples/__init__.py
touch albert/execution/__init__.py
touch albert/execution/adapters/__init__.py
touch albert/portfolio/__init__.py
```

- [ ] **Step 5: Create `tests/conftest.py` (empty for now)**

```python
# tests/conftest.py
```

- [ ] **Step 6: Install in editable mode**

```bash
pip install -e ".[dev]"
```
Expected: `Successfully installed albert-0.1.0`

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest tests/test_imports.py -v
```
Expected: `PASSED`

- [ ] **Step 8: Delete the temporary test file**

```bash
rm tests/test_imports.py
```

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml albert/ tests/
git commit -m "chore: scaffold project structure"
```

---

## Task 2: Database Schema

**Files:**
- Create: `albert/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py
import sqlite3
import pytest
from albert.db import get_connection, migrate

def test_migrate_creates_tables():
    conn = get_connection(":memory:")
    migrate(conn)
    tables = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert tables == {"markets", "orderbook_snapshots", "positions", "fills", "strategies", "daily_pnl"}

def test_migrate_is_idempotent():
    conn = get_connection(":memory:")
    migrate(conn)
    migrate(conn)  # should not raise

def test_strategies_table_columns():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
        ("s1", "Test", "albert.strategies.examples.momentum.MomentumV1", '{"min_edge": 0.05}', 1)
    )
    row = conn.execute("SELECT * FROM strategies WHERE strategy_id = 's1'").fetchone()
    assert row["name"] == "Test"
    assert row["enabled"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.db'`

- [ ] **Step 3: Create `albert/db.py`**

```python
# albert/db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("albert.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    market_id   TEXT PRIMARY KEY,
    exchange    TEXT NOT NULL,
    title       TEXT NOT NULL,
    close_time  DATETIME,
    status      TEXT NOT NULL DEFAULT 'open',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id   TEXT NOT NULL,
    timestamp   DATETIME NOT NULL,
    yes_bid     REAL,
    yes_ask     REAL,
    no_bid      REAL,
    no_ask      REAL,
    last_price  REAL,
    volume      REAL
);

CREATE TABLE IF NOT EXISTS positions (
    market_id       TEXT NOT NULL,
    strategy_id     TEXT NOT NULL,
    side            TEXT NOT NULL,
    contracts       REAL NOT NULL DEFAULT 0,
    avg_entry_price REAL NOT NULL DEFAULT 0,
    current_price   REAL,
    unrealized_pnl  REAL NOT NULL DEFAULT 0,
    opened_at       DATETIME NOT NULL,
    PRIMARY KEY (market_id, strategy_id)
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id     TEXT PRIMARY KEY,
    market_id   TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    side        TEXT NOT NULL,
    contracts   REAL NOT NULL,
    fill_price  REAL NOT NULL,
    fee         REAL NOT NULL DEFAULT 0,
    filled_at   DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    class_path  TEXT NOT NULL,
    config      TEXT NOT NULL DEFAULT '{}',
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    date        TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    realized_pnl    REAL NOT NULL DEFAULT 0,
    unrealized_pnl  REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (date, strategy_id)
);
"""


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```
Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/db.py tests/test_db.py
git commit -m "feat: add SQLite schema and migrate()"
```

---

## Task 3: Event Types and Event Bus

**Files:**
- Create: `albert/events.py`
- Create: `tests/test_events.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_events.py
import asyncio
import pytest
from datetime import datetime
from albert.events import EventBus, MarketDataEvent, OrderIntent, FillEvent, StrategyHaltedEvent

def make_market_event():
    return MarketDataEvent(
        market_id="kalshi:TEST-24",
        exchange="kalshi",
        timestamp=datetime.utcnow(),
        yes_bid=0.45,
        yes_ask=0.47,
        no_bid=0.53,
        no_ask=0.55,
        last_price=0.46,
        volume=1000.0,
    )

async def test_publish_subscribe_delivers_event():
    bus = EventBus()
    q = bus.subscribe("market_data")
    event = make_market_event()
    await bus.publish("market_data", event)
    received = q.get_nowait()
    assert received.market_id == "kalshi:TEST-24"
    assert received.yes_ask == 0.47

async def test_multiple_subscribers_each_receive_event():
    bus = EventBus()
    q1 = bus.subscribe("market_data")
    q2 = bus.subscribe("market_data")
    await bus.publish("market_data", make_market_event())
    assert not q1.empty()
    assert not q2.empty()

async def test_publish_to_unsubscribed_channel_does_not_raise():
    bus = EventBus()
    await bus.publish("order_intents", OrderIntent(
        market_id="kalshi:TEST-24",
        strategy_id="s1",
        side="yes",
        edge=0.08,
        confidence=0.9,
    ))

def test_fill_event_fields():
    fill = FillEvent(
        fill_id="f1",
        market_id="kalshi:TEST-24",
        strategy_id="s1",
        side="yes",
        contracts=10.0,
        fill_price=0.47,
        fee=0.01,
        filled_at=datetime.utcnow(),
    )
    assert fill.contracts == 10.0
    assert fill.fill_price == 0.47
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_events.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.events'`

- [ ] **Step 3: Create `albert/events.py`**

```python
# albert/events.py
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class MarketDataEvent:
    market_id: str
    exchange: Literal["kalshi", "polymarket"]
    timestamp: datetime
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    last_price: float
    volume: float


@dataclass
class OrderIntent:
    market_id: str
    strategy_id: str
    side: Literal["yes", "no"]
    edge: float
    confidence: float


@dataclass
class FillEvent:
    fill_id: str
    market_id: str
    strategy_id: str
    side: Literal["yes", "no"]
    contracts: float
    fill_price: float
    fee: float
    filled_at: datetime


@dataclass
class StrategyHaltedEvent:
    strategy_id: str
    reason: str
    timestamp: datetime


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(channel, []).append(q)
        return q

    async def publish(self, channel: str, event: object) -> None:
        for q in self._queues.get(channel, []):
            await q.put(event)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_events.py -v
```
Expected: all 4 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/events.py tests/test_events.py
git commit -m "feat: add event dataclasses and EventBus"
```

---

## Task 4: BaseStrategy and Example Strategy

**Files:**
- Create: `albert/strategies/base.py`
- Create: `albert/strategies/examples/momentum.py`
- Create: `tests/conftest.py` (update with shared fixtures)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_strategy_base.py
import pytest
from datetime import datetime
from albert.strategies.base import BaseStrategy
from albert.events import MarketDataEvent, OrderIntent

class DoubleEdgeStrategy(BaseStrategy):
    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        edge = self.estimate_edge(event)
        if edge <= 0:
            return None
        return [OrderIntent(
            market_id=event.market_id,
            strategy_id=self.id,
            side="yes",
            edge=edge,
            confidence=1.0,
        )]

    def estimate_edge(self, event: MarketDataEvent) -> float:
        return max(0.0, 0.5 - event.yes_ask)


def make_event(yes_ask: float) -> MarketDataEvent:
    return MarketDataEvent(
        market_id="kalshi:X",
        exchange="kalshi",
        timestamp=datetime.utcnow(),
        yes_bid=yes_ask - 0.02,
        yes_ask=yes_ask,
        no_bid=0.0,
        no_ask=0.0,
        last_price=yes_ask,
        volume=0.0,
    )


async def test_strategy_returns_intent_when_edge_positive():
    s = DoubleEdgeStrategy(strategy_id="s1", config={"min_edge": 0.0})
    intents = await s.on_market_data(make_event(0.40))
    assert intents is not None
    assert len(intents) == 1
    assert intents[0].side == "yes"
    assert intents[0].edge == pytest.approx(0.10)

async def test_strategy_returns_none_when_no_edge():
    s = DoubleEdgeStrategy(strategy_id="s1", config={})
    intents = await s.on_market_data(make_event(0.55))
    assert intents is None

async def test_momentum_strategy_returns_intent_below_threshold():
    from albert.strategies.examples.momentum import MomentumV1
    s = MomentumV1(strategy_id="momentum_v1", config={"min_edge": 0.05})
    intents = await s.on_market_data(make_event(0.30))
    assert intents is not None
    assert intents[0].edge == pytest.approx(0.20)

async def test_momentum_strategy_returns_none_above_threshold():
    from albert.strategies.examples.momentum import MomentumV1
    s = MomentumV1(strategy_id="momentum_v1", config={"min_edge": 0.05})
    intents = await s.on_market_data(make_event(0.50))
    assert intents is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_strategy_base.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.strategies.base'`

- [ ] **Step 3: Create `albert/strategies/base.py`**

```python
# albert/strategies/base.py
from abc import ABC, abstractmethod
from albert.events import MarketDataEvent, OrderIntent


class BaseStrategy(ABC):
    def __init__(self, strategy_id: str, config: dict) -> None:
        self.id = strategy_id
        self.config = config

    @abstractmethod
    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        """Called on every orderbook update. Return order intents or None."""
        ...

    @abstractmethod
    def estimate_edge(self, event: MarketDataEvent) -> float:
        """Return estimated probability edge (0.0–1.0)."""
        ...
```

- [ ] **Step 4: Create `albert/strategies/examples/momentum.py`**

```python
# albert/strategies/examples/momentum.py
from albert.strategies.base import BaseStrategy
from albert.events import MarketDataEvent, OrderIntent


class MomentumV1(BaseStrategy):
    """
    Example strategy: buy YES when the ask is below 0.5 by at least min_edge.
    Edge = 0.5 - yes_ask (assumes true probability is 0.5 for any event near even odds).
    """

    async def on_market_data(self, event: MarketDataEvent) -> list[OrderIntent] | None:
        min_edge = self.config.get("min_edge", 0.05)
        edge = self.estimate_edge(event)
        if edge < min_edge:
            return None
        return [OrderIntent(
            market_id=event.market_id,
            strategy_id=self.id,
            side="yes",
            edge=edge,
            confidence=min(1.0, edge / 0.2),
        )]

    def estimate_edge(self, event: MarketDataEvent) -> float:
        if event.yes_ask <= 0:
            return 0.0
        return max(0.0, 0.5 - event.yes_ask)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_strategy_base.py -v
```
Expected: all 4 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add albert/strategies/base.py albert/strategies/examples/momentum.py tests/test_strategy_base.py
git commit -m "feat: add BaseStrategy interface and MomentumV1 example"
```

---

## Task 5: Strategy Engine

**Files:**
- Create: `albert/strategies/engine.py`
- Create: `tests/test_strategy_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_strategy_engine.py
import asyncio
import json
import pytest
from datetime import datetime
from albert.db import get_connection, migrate
from albert.events import EventBus, MarketDataEvent, OrderIntent
from albert.strategies.engine import StrategyEngine


def make_db():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
        ("m1", "Momentum", "albert.strategies.examples.momentum.MomentumV1", json.dumps({"min_edge": 0.05}), 1)
    )
    conn.commit()
    return conn


def make_event(yes_ask: float) -> MarketDataEvent:
    return MarketDataEvent(
        market_id="kalshi:X",
        exchange="kalshi",
        timestamp=datetime.utcnow(),
        yes_bid=yes_ask - 0.02,
        yes_ask=yes_ask,
        no_bid=0.0,
        no_ask=0.0,
        last_price=yes_ask,
        volume=0.0,
    )


async def test_engine_publishes_intent_for_active_strategy():
    conn = make_db()
    bus = EventBus()
    intents_queue = bus.subscribe("order_intents")
    engine = StrategyEngine(bus, conn, reload_interval=999)

    market_data_queue = bus.subscribe("market_data")
    await bus.publish("market_data", make_event(0.30))

    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert not intents_queue.empty()
    intent: OrderIntent = intents_queue.get_nowait()
    assert intent.strategy_id == "m1"
    assert intent.side == "yes"


async def test_engine_skips_disabled_strategy():
    conn = make_db()
    conn.execute("UPDATE strategies SET enabled = 0 WHERE strategy_id = 'm1'")
    conn.commit()

    bus = EventBus()
    intents_queue = bus.subscribe("order_intents")
    engine = StrategyEngine(bus, conn, reload_interval=999)

    await bus.publish("market_data", make_event(0.30))
    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert intents_queue.empty()


async def test_engine_hot_reloads_config():
    conn = make_db()
    bus = EventBus()
    intents_queue = bus.subscribe("order_intents")
    engine = StrategyEngine(bus, conn, reload_interval=0.01)

    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.02)

    # Update config to require very high edge — strategy should no longer emit
    conn.execute("UPDATE strategies SET config = ? WHERE strategy_id = 'm1'", (json.dumps({"min_edge": 0.99}),))
    conn.commit()
    await asyncio.sleep(0.05)

    # drain existing intents
    while not intents_queue.empty():
        intents_queue.get_nowait()

    await bus.publish("market_data", make_event(0.30))
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert intents_queue.empty()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_strategy_engine.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.strategies.engine'`

- [ ] **Step 3: Create `albert/strategies/engine.py`**

```python
# albert/strategies/engine.py
import asyncio
import importlib
import json
import logging
import sqlite3

from albert.events import EventBus, MarketDataEvent
from albert.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyEngine:
    def __init__(
        self,
        bus: EventBus,
        conn: sqlite3.Connection,
        reload_interval: float = 30.0,
    ) -> None:
        self._bus = bus
        self._conn = conn
        self._reload_interval = reload_interval
        self._strategies: dict[str, BaseStrategy] = {}
        self._last_reload: float = -1.0

    def _load_strategies(self) -> None:
        rows = self._conn.execute(
            "SELECT strategy_id, class_path, config FROM strategies WHERE enabled = 1"
        ).fetchall()
        active_ids = {row["strategy_id"] for row in rows}

        # Remove disabled strategies
        for sid in list(self._strategies):
            if sid not in active_ids:
                del self._strategies[sid]

        for row in rows:
            sid = row["strategy_id"]
            config = json.loads(row["config"])
            if sid not in self._strategies:
                module_path, class_name = row["class_path"].rsplit(".", 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self._strategies[sid] = cls(strategy_id=sid, config=config)
                logger.info("loaded strategy %s", sid)
            else:
                self._strategies[sid].config = config

    async def run(self) -> None:
        queue = self._bus.subscribe("market_data")
        loop = asyncio.get_event_loop()

        while True:
            now = loop.time()
            if now - self._last_reload >= self._reload_interval:
                self._load_strategies()
                self._last_reload = now

            event: MarketDataEvent = await queue.get()

            for strategy in list(self._strategies.values()):
                try:
                    intents = await strategy.on_market_data(event)
                    if intents:
                        for intent in intents:
                            await self._bus.publish("order_intents", intent)
                except Exception:
                    logger.exception("strategy %s raised on market data", strategy.id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_strategy_engine.py -v
```
Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/strategies/engine.py tests/test_strategy_engine.py
git commit -m "feat: add StrategyEngine with hot-reload"
```

---

## Task 6: Kelly Sizing

**Files:**
- Create: `albert/execution/kelly.py`
- Create: `tests/test_kelly.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kelly.py
import pytest
from albert.execution.kelly import kelly_size


def test_positive_edge_returns_positive_size():
    size = kelly_size(
        edge=0.10,
        ask_price=0.40,
        bankroll=10000.0,
        kelly_fraction=0.25,
        confidence=1.0,
        max_position_usd=500.0,
    )
    assert size > 0

def test_negative_edge_returns_zero():
    size = kelly_size(
        edge=0.01,
        ask_price=0.60,  # b = 0.667, f* = (0.01*0.667 - 0.99)/0.667 < 0
        bankroll=10000.0,
        kelly_fraction=0.25,
        confidence=1.0,
        max_position_usd=500.0,
    )
    assert size == 0.0

def test_capped_by_max_position_usd():
    size = kelly_size(
        edge=0.30,
        ask_price=0.20,
        bankroll=1_000_000.0,
        kelly_fraction=1.0,
        confidence=1.0,
        max_position_usd=100.0,
    )
    assert size == pytest.approx(100.0)

def test_confidence_scales_output():
    full = kelly_size(0.10, 0.40, 10000.0, 0.25, 1.0, 500.0)
    half = kelly_size(0.10, 0.40, 10000.0, 0.25, 0.5, 500.0)
    assert half == pytest.approx(full * 0.5)

def test_invalid_ask_price_returns_zero():
    assert kelly_size(0.10, 0.0, 10000.0, 0.25, 1.0, 500.0) == 0.0
    assert kelly_size(0.10, 1.0, 10000.0, 0.25, 1.0, 500.0) == 0.0
    assert kelly_size(0.10, 1.1, 10000.0, 0.25, 1.0, 500.0) == 0.0

def test_zero_edge_returns_zero():
    assert kelly_size(0.0, 0.40, 10000.0, 0.25, 1.0, 500.0) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_kelly.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.execution.kelly'`

- [ ] **Step 3: Create `albert/execution/kelly.py`**

```python
# albert/execution/kelly.py


def kelly_size(
    edge: float,
    ask_price: float,
    bankroll: float,
    kelly_fraction: float,
    confidence: float,
    max_position_usd: float,
) -> float:
    """
    Compute fractional Kelly position size in USD.

    Args:
        edge: Estimated probability edge above ask price (0.0–1.0).
        ask_price: Current ask price for the side being bought (0–1 exclusive).
        bankroll: Total available capital in USD.
        kelly_fraction: Fraction of full Kelly to use (e.g. 0.25 for quarter-Kelly).
        confidence: Strategy confidence scalar (0.0–1.0).
        max_position_usd: Hard cap on position size regardless of Kelly output.

    Returns:
        Position size in USD, >= 0.
    """
    if ask_price <= 0.0 or ask_price >= 1.0:
        return 0.0
    if edge <= 0.0:
        return 0.0

    b = (1.0 - ask_price) / ask_price  # net odds on a $1 bet
    f_star = (edge * b - (1.0 - edge)) / b
    if f_star <= 0.0:
        return 0.0

    size = bankroll * f_star * kelly_fraction * confidence
    return min(size, max_position_usd)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kelly.py -v
```
Expected: all 6 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/execution/kelly.py tests/test_kelly.py
git commit -m "feat: add Kelly criterion position sizing"
```

---

## Task 7: Risk Checker

**Files:**
- Create: `albert/execution/risk.py`
- Create: `tests/test_risk.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_risk.py
import time
import pytest
from datetime import date
from albert.db import get_connection, migrate
from albert.events import OrderIntent
from albert.execution.risk import RiskChecker


def make_intent(market_id: str = "kalshi:X", strategy_id: str = "s1") -> OrderIntent:
    return OrderIntent(
        market_id=market_id,
        strategy_id=strategy_id,
        side="yes",
        edge=0.10,
        confidence=1.0,
    )


def make_checker(conn, overrides: dict = {}) -> RiskChecker:
    config = {
        "max_total_notional_usd": 1000.0,
        "daily_loss_limit_usd": -200.0,
        "order_debounce_seconds": 5,
        **overrides,
    }
    return RiskChecker(conn, config)


def test_allows_normal_order():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn)
    assert checker.check(make_intent(), position_size_usd=50.0) is True


def test_blocks_on_debounce():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn, {"order_debounce_seconds": 60})
    intent = make_intent()
    assert checker.check(intent, 50.0) is True
    assert checker.check(intent, 50.0) is False  # second call within debounce window


def test_allows_after_debounce_expires():
    conn = get_connection(":memory:")
    migrate(conn)
    checker = make_checker(conn, {"order_debounce_seconds": 0})
    intent = make_intent()
    assert checker.check(intent, 50.0) is True
    assert checker.check(intent, 50.0) is True  # debounce=0 means always allowed


def test_blocks_when_daily_loss_limit_hit():
    conn = get_connection(":memory:")
    migrate(conn)
    today = date.today().isoformat()
    conn.execute(
        "INSERT INTO daily_pnl (date, strategy_id, realized_pnl, unrealized_pnl) VALUES (?, ?, ?, ?)",
        (today, "s1", -250.0, 0.0)
    )
    conn.commit()
    checker = make_checker(conn, {"daily_loss_limit_usd": -200.0})
    assert checker.check(make_intent(), 50.0) is False


def test_blocks_when_max_notional_exceeded():
    conn = get_connection(":memory:")
    migrate(conn)
    import datetime
    conn.execute(
        "INSERT INTO positions (market_id, strategy_id, side, contracts, avg_entry_price, current_price, unrealized_pnl, opened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("kalshi:Y", "s1", "yes", 10.0, 0.50, 0.50, 0.0, datetime.datetime.utcnow().isoformat())
    )
    conn.commit()
    # current notional = 10 * 0.50 = 5.0 USD
    checker = make_checker(conn, {"max_total_notional_usd": 10.0})
    assert checker.check(make_intent(), position_size_usd=6.0) is False
    assert checker.check(make_intent(), position_size_usd=4.0) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_risk.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.execution.risk'`

- [ ] **Step 3: Create `albert/execution/risk.py`**

```python
# albert/execution/risk.py
import logging
import sqlite3
import time
from datetime import date

from albert.events import OrderIntent

logger = logging.getLogger(__name__)


class RiskChecker:
    def __init__(self, conn: sqlite3.Connection, global_config: dict) -> None:
        self._conn = conn
        self._config = global_config
        self._last_order_time: dict[tuple[str, str], float] = {}

    def check(self, intent: OrderIntent, position_size_usd: float) -> bool:
        key = (intent.market_id, intent.strategy_id)
        debounce = self._config.get("order_debounce_seconds", 10)
        now = time.monotonic()

        if debounce > 0 and now - self._last_order_time.get(key, 0.0) < debounce:
            logger.info(
                "risk:debounce market=%s strategy=%s", intent.market_id, intent.strategy_id
            )
            return False

        today = date.today().isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl + unrealized_pnl), 0) AS total FROM daily_pnl WHERE date = ?",
            (today,),
        ).fetchone()
        daily_pnl = row["total"]
        limit = self._config.get("daily_loss_limit_usd", -500.0)
        if daily_pnl < limit:
            logger.warning("risk:daily_loss_limit pnl=%.2f limit=%.2f", daily_pnl, limit)
            return False

        row = self._conn.execute(
            "SELECT COALESCE(SUM(contracts * current_price), 0) AS notional FROM positions"
        ).fetchone()
        current_notional = row["notional"]
        max_notional = self._config.get("max_total_notional_usd", 10000.0)
        if current_notional + position_size_usd > max_notional:
            logger.info(
                "risk:max_notional current=%.2f new=%.2f max=%.2f",
                current_notional, position_size_usd, max_notional,
            )
            return False

        self._last_order_time[key] = now
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_risk.py -v
```
Expected: all 5 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/execution/risk.py tests/test_risk.py
git commit -m "feat: add pre-trade risk checker"
```

---

## Task 8: Exchange Adapter Interface and KalshiAdapter

**Files:**
- Create: `albert/execution/adapters/base.py`
- Create: `albert/execution/adapters/kalshi.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_adapters.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from albert.events import OrderIntent, FillEvent
from albert.execution.adapters.kalshi import KalshiAdapter


def make_intent() -> OrderIntent:
    return OrderIntent(
        market_id="kalshi:BTC-24",
        strategy_id="s1",
        side="yes",
        edge=0.10,
        confidence=1.0,
    )


async def test_kalshi_place_order_returns_fill():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "order": {
            "order_id": "ord_abc123",
            "count": 5,
            "yes_price": 47,
            "fee": 2,
        }
    }

    with patch.dict("os.environ", {"KALSHI_API_TOKEN": "test_token"}):
        adapter = KalshiAdapter()
        with patch.object(adapter._client, "post", new=AsyncMock(return_value=mock_response)):
            fill = await adapter.place_order(make_intent(), contracts=5, price=0.47)

    assert fill.fill_id == "ord_abc123"
    assert fill.contracts == 5
    assert fill.fill_price == pytest.approx(0.47)
    assert fill.strategy_id == "s1"


async def test_kalshi_get_bankroll_returns_float():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"balance": {"available": 50000}}

    with patch.dict("os.environ", {"KALSHI_API_TOKEN": "test_token"}):
        adapter = KalshiAdapter()
        with patch.object(adapter._client, "get", new=AsyncMock(return_value=mock_response)):
            bankroll = await adapter.get_bankroll()

    assert bankroll == pytest.approx(500.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_adapters.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.execution.adapters.kalshi'`

- [ ] **Step 3: Create `albert/execution/adapters/base.py`**

```python
# albert/execution/adapters/base.py
from abc import ABC, abstractmethod
from albert.events import OrderIntent, FillEvent


class ExchangeAdapter(ABC):
    @abstractmethod
    async def place_order(self, intent: OrderIntent, contracts: int, price: float) -> FillEvent:
        """Place a limit order. Returns a FillEvent on success."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        """Cancel an open order by exchange order ID."""
        ...

    @abstractmethod
    async def get_bankroll(self) -> float:
        """Return available balance in USD."""
        ...
```

- [ ] **Step 4: Create `albert/execution/adapters/kalshi.py`**

```python
# albert/execution/adapters/kalshi.py
import asyncio
import logging
import os
from datetime import datetime

import httpx

from albert.events import FillEvent, OrderIntent
from .base import ExchangeAdapter

logger = logging.getLogger(__name__)

_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
_MAX_RETRIES = 3


class KalshiAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        token = os.environ["KALSHI_API_TOKEN"]
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )

    async def place_order(self, intent: OrderIntent, contracts: int, price: float) -> FillEvent:
        ticker = intent.market_id.removeprefix("kalshi:")
        price_cents = round(price * 100)
        payload = {
            "ticker": ticker,
            "action": "buy",
            "side": intent.side,
            "count": contracts,
            "type": "limit",
            f"{intent.side}_price": price_cents,
        }
        data = await self._post_with_retry("/portfolio/orders", payload)
        order = data["order"]
        fill_price = order.get(f"{intent.side}_price", price_cents) / 100
        return FillEvent(
            fill_id=order["order_id"],
            market_id=intent.market_id,
            strategy_id=intent.strategy_id,
            side=intent.side,
            contracts=float(order["count"]),
            fill_price=fill_price,
            fee=order.get("fee", 0) / 100,
            filled_at=datetime.utcnow(),
        )

    async def cancel_order(self, order_id: str) -> None:
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.delete(f"/portfolio/orders/{order_id}")
                r.raise_for_status()
                return
            except httpx.HTTPError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def get_bankroll(self) -> float:
        r = await self._client.get("/portfolio/balance")
        r.raise_for_status()
        return r.json()["balance"]["available"] / 100

    async def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.post(path, json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                last_exc = e
                logger.warning("kalshi POST %s attempt %d failed: %s", path, attempt + 1, e)
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_exc
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_adapters.py -v
```
Expected: both `PASSED`

- [ ] **Step 6: Commit**

```bash
git add albert/execution/adapters/base.py albert/execution/adapters/kalshi.py tests/test_adapters.py
git commit -m "feat: add ExchangeAdapter interface and KalshiAdapter"
```

---

## Task 9: PolymarketAdapter

**Files:**
- Create: `albert/execution/adapters/polymarket.py`
- Modify: `tests/test_adapters.py`

- [ ] **Step 1: Add failing tests for Polymarket adapter**

Append to `tests/test_adapters.py`:

```python
from albert.execution.adapters.polymarket import PolymarketAdapter


async def test_polymarket_place_order_returns_fill():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"orderID": "poly_xyz789", "status": "matched"}

    env = {
        "POLYMARKET_API_KEY": "k",
        "POLYMARKET_API_SECRET": "s",
        "POLYMARKET_API_PASSPHRASE": "p",
        "POLYMARKET_ADDRESS": "0xABC",
    }
    intent = OrderIntent(
        market_id="polymarket:cond123:token456",
        strategy_id="s1",
        side="yes",
        edge=0.10,
        confidence=1.0,
    )
    with patch.dict("os.environ", env):
        adapter = PolymarketAdapter()
        with patch.object(adapter._client, "post", new=AsyncMock(return_value=mock_response)):
            fill = await adapter.place_order(intent, contracts=10, price=0.45)

    assert fill.fill_id == "poly_xyz789"
    assert fill.contracts == 10
    assert fill.fill_price == pytest.approx(0.45)
    assert fill.market_id == "polymarket:cond123:token456"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_adapters.py::test_polymarket_place_order_returns_fill -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.execution.adapters.polymarket'`

- [ ] **Step 3: Create `albert/execution/adapters/polymarket.py`**

```python
# albert/execution/adapters/polymarket.py
import asyncio
import logging
import os
from datetime import datetime

import httpx

from albert.events import FillEvent, OrderIntent
from .base import ExchangeAdapter

logger = logging.getLogger(__name__)

_BASE_URL = "https://clob.polymarket.com"
_MAX_RETRIES = 3


class PolymarketAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "POLY_ADDRESS": os.environ["POLYMARKET_ADDRESS"],
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )
        self._api_key = os.environ["POLYMARKET_API_KEY"]
        self._api_secret = os.environ["POLYMARKET_API_SECRET"]
        self._api_passphrase = os.environ["POLYMARKET_API_PASSPHRASE"]

    def _token_id(self, market_id: str) -> str:
        # market_id format: polymarket:<condition_id>:<token_id>
        parts = market_id.removeprefix("polymarket:").split(":")
        return parts[1] if len(parts) > 1 else parts[0]

    async def place_order(self, intent: OrderIntent, contracts: int, price: float) -> FillEvent:
        token_id = self._token_id(intent.market_id)
        payload = {
            "orderType": "GTC",
            "tokenID": token_id,
            "price": str(price),
            "size": str(contracts),
            "side": "BUY",
        }
        data = await self._post_with_retry("/order", payload)
        return FillEvent(
            fill_id=data.get("orderID", "unknown"),
            market_id=intent.market_id,
            strategy_id=intent.strategy_id,
            side=intent.side,
            contracts=float(contracts),
            fill_price=price,
            fee=0.0,
            filled_at=datetime.utcnow(),
        )

    async def cancel_order(self, order_id: str) -> None:
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.delete(f"/order/{order_id}")
                r.raise_for_status()
                return
            except httpx.HTTPError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def get_bankroll(self) -> float:
        r = await self._client.get("/balance")
        r.raise_for_status()
        return float(r.json().get("balance", 0.0))

    async def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.post(path, json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                last_exc = e
                logger.warning("polymarket POST %s attempt %d failed: %s", path, attempt + 1, e)
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_exc
```

- [ ] **Step 4: Run all adapter tests to verify they pass**

```bash
pytest tests/test_adapters.py -v
```
Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/execution/adapters/polymarket.py tests/test_adapters.py
git commit -m "feat: add PolymarketAdapter"
```

---

## Task 10: Execution Engine

**Files:**
- Create: `albert/execution/engine.py`
- Create: `tests/test_execution_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_execution_engine.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from albert.db import get_connection, migrate
from albert.events import EventBus, OrderIntent, FillEvent, StrategyHaltedEvent
from albert.execution.engine import ExecutionEngine
from albert.execution.adapters.base import ExchangeAdapter


def make_db_with_strategy(config: dict = None):
    conn = get_connection(":memory:")
    migrate(conn)
    cfg = config or {"kelly_fraction": 0.25, "max_position_usd": 500.0}
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
        ("s1", "Test", "albert.strategies.examples.momentum.MomentumV1", json.dumps(cfg), 1)
    )
    conn.execute(
        "INSERT INTO orderbook_snapshots (market_id, timestamp, yes_bid, yes_ask, no_bid, no_ask, last_price, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("kalshi:X", datetime.utcnow().isoformat(), 0.45, 0.47, 0.53, 0.55, 0.46, 1000.0)
    )
    conn.commit()
    return conn


def make_mock_adapter(bankroll: float = 10000.0) -> ExchangeAdapter:
    adapter = MagicMock(spec=ExchangeAdapter)
    adapter.get_bankroll = AsyncMock(return_value=bankroll)
    adapter.place_order = AsyncMock(return_value=FillEvent(
        fill_id="f1",
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        contracts=5.0,
        fill_price=0.47,
        fee=0.01,
        filled_at=datetime.utcnow(),
    ))
    return adapter


async def test_execution_engine_places_order_and_publishes_fill():
    conn = make_db_with_strategy()
    bus = EventBus()
    fills_queue = bus.subscribe("fills")
    adapter = make_mock_adapter()

    engine = ExecutionEngine(
        bus=bus,
        conn=conn,
        adapters={"kalshi": adapter},
        global_config={"max_total_notional_usd": 100000, "daily_loss_limit_usd": -10000, "order_debounce_seconds": 0},
    )

    intent = OrderIntent(market_id="kalshi:X", strategy_id="s1", side="yes", edge=0.10, confidence=1.0)
    await bus.publish("order_intents", intent)

    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert not fills_queue.empty()
    fill: FillEvent = fills_queue.get_nowait()
    assert fill.strategy_id == "s1"
    adapter.place_order.assert_awaited_once()


async def test_execution_engine_persists_fill_to_db():
    conn = make_db_with_strategy()
    bus = EventBus()
    adapter = make_mock_adapter()

    engine = ExecutionEngine(
        bus=bus,
        conn=conn,
        adapters={"kalshi": adapter},
        global_config={"max_total_notional_usd": 100000, "daily_loss_limit_usd": -10000, "order_debounce_seconds": 0},
    )

    intent = OrderIntent(market_id="kalshi:X", strategy_id="s1", side="yes", edge=0.10, confidence=1.0)
    await bus.publish("order_intents", intent)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    task.cancel()

    fill_row = conn.execute("SELECT * FROM fills WHERE fill_id = 'f1'").fetchone()
    assert fill_row is not None
    assert fill_row["strategy_id"] == "s1"


async def test_execution_engine_skips_unknown_exchange():
    conn = make_db_with_strategy()
    bus = EventBus()
    fills_queue = bus.subscribe("fills")
    adapter = make_mock_adapter()

    engine = ExecutionEngine(
        bus=bus,
        conn=conn,
        adapters={"kalshi": adapter},
        global_config={"max_total_notional_usd": 100000, "daily_loss_limit_usd": -10000, "order_debounce_seconds": 0},
    )

    # polymarket intent but no polymarket adapter
    intent = OrderIntent(market_id="polymarket:X:Y", strategy_id="s1", side="yes", edge=0.10, confidence=1.0)
    await bus.publish("order_intents", intent)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    task.cancel()

    assert fills_queue.empty()
    adapter.place_order.assert_not_awaited()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_execution_engine.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.execution.engine'`

- [ ] **Step 3: Create `albert/execution/engine.py`**

```python
# albert/execution/engine.py
import asyncio
import json
import logging
import sqlite3
from datetime import datetime

from albert.events import EventBus, FillEvent, OrderIntent, StrategyHaltedEvent
from albert.execution.adapters.base import ExchangeAdapter
from albert.execution.kelly import kelly_size
from albert.execution.risk import RiskChecker

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(
        self,
        bus: EventBus,
        conn: sqlite3.Connection,
        adapters: dict[str, ExchangeAdapter],
        global_config: dict,
    ) -> None:
        self._bus = bus
        self._conn = conn
        self._adapters = adapters
        self._risk = RiskChecker(conn, global_config)

    async def run(self) -> None:
        queue = self._bus.subscribe("order_intents")
        while True:
            intent: OrderIntent = await queue.get()
            await self._handle_intent(intent)

    async def _handle_intent(self, intent: OrderIntent) -> None:
        exchange = intent.market_id.split(":")[0]
        adapter = self._adapters.get(exchange)
        if not adapter:
            logger.error("execution:no_adapter exchange=%s", exchange)
            return

        row = self._conn.execute(
            "SELECT yes_ask, no_ask FROM orderbook_snapshots WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1",
            (intent.market_id,),
        ).fetchone()
        if not row:
            logger.warning("execution:no_orderbook market=%s", intent.market_id)
            return

        ask_price = row["yes_ask"] if intent.side == "yes" else row["no_ask"]
        if not ask_price:
            return

        strategy_row = self._conn.execute(
            "SELECT config FROM strategies WHERE strategy_id = ?",
            (intent.strategy_id,),
        ).fetchone()
        strategy_config = json.loads(strategy_row["config"]) if strategy_row else {}
        kelly_fraction = strategy_config.get("kelly_fraction", 0.25)
        max_position_usd = strategy_config.get("max_position_usd", 500.0)

        try:
            bankroll = await adapter.get_bankroll()
        except Exception:
            logger.exception("execution:bankroll_error strategy=%s", intent.strategy_id)
            return

        size_usd = kelly_size(intent.edge, ask_price, bankroll, kelly_fraction, intent.confidence, max_position_usd)
        if size_usd <= 0:
            return

        if not self._risk.check(intent, size_usd):
            return

        contracts = max(1, round(size_usd / ask_price))

        try:
            fill = await adapter.place_order(intent, contracts=contracts, price=ask_price)
        except Exception:
            logger.exception("execution:order_failed strategy=%s market=%s", intent.strategy_id, intent.market_id)
            await self._halt_strategy(intent.strategy_id, "order placement failed after retries")
            return

        self._persist_fill(fill)
        await self._bus.publish("fills", fill)
        logger.info(
            "execution:fill fill_id=%s strategy=%s market=%s contracts=%s price=%.4f",
            fill.fill_id, fill.strategy_id, fill.market_id, fill.contracts, fill.fill_price,
        )

    def _persist_fill(self, fill: FillEvent) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO fills
               (fill_id, market_id, strategy_id, side, contracts, fill_price, fee, filled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (fill.fill_id, fill.market_id, fill.strategy_id, fill.side,
             fill.contracts, fill.fill_price, fill.fee, fill.filled_at.isoformat()),
        )
        self._conn.commit()

    async def _halt_strategy(self, strategy_id: str, reason: str) -> None:
        self._conn.execute(
            "UPDATE strategies SET enabled = 0 WHERE strategy_id = ?",
            (strategy_id,),
        )
        self._conn.commit()
        await self._bus.publish("strategy_halted", StrategyHaltedEvent(
            strategy_id=strategy_id,
            reason=reason,
            timestamp=datetime.utcnow(),
        ))
        logger.error("execution:strategy_halted strategy=%s reason=%s", strategy_id, reason)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_execution_engine.py -v
```
Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/execution/engine.py tests/test_execution_engine.py
git commit -m "feat: add ExecutionEngine with Kelly sizing and risk checks"
```

---

## Task 11: Base Ingestor and KalshiIngestor

**Files:**
- Create: `albert/ingestor/base.py`
- Create: `albert/ingestor/kalshi.py`
- Create: `tests/test_ingestor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingestor.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from albert.events import EventBus, MarketDataEvent
from albert.ingestor.kalshi import KalshiIngestor


async def test_kalshi_ingestor_publishes_market_data_event():
    bus = EventBus()
    queue = bus.subscribe("market_data")

    ws_message = json.dumps({
        "type": "orderbook_snapshot",
        "msg": {
            "market_ticker": "BTC-24",
            "yes": {"bid": 45, "ask": 47},
            "no": {"bid": 53, "ask": 55},
            "last_price": 46,
            "volume": 1000,
        }
    })

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.send = AsyncMock()

    async def fake_aiter():
        yield ws_message
        await asyncio.sleep(999)  # hang to simulate live connection

    mock_ws.__aiter__ = fake_aiter

    with patch.dict("os.environ", {"KALSHI_API_TOKEN": "test"}):
        with patch("albert.ingestor.kalshi.websockets.connect", return_value=mock_ws):
            ingestor = KalshiIngestor(bus=bus, market_ids=["kalshi:BTC-24"])
            task = asyncio.create_task(ingestor.run())
            await asyncio.sleep(0.05)
            task.cancel()

    assert not queue.empty()
    event: MarketDataEvent = queue.get_nowait()
    assert event.market_id == "kalshi:BTC-24"
    assert event.yes_ask == pytest.approx(0.47)
    assert event.no_bid == pytest.approx(0.53)


async def test_kalshi_ingestor_ignores_non_orderbook_messages():
    bus = EventBus()
    queue = bus.subscribe("market_data")

    ws_message = json.dumps({"type": "heartbeat", "msg": {}})

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.send = AsyncMock()

    async def fake_aiter():
        yield ws_message
        await asyncio.sleep(999)

    mock_ws.__aiter__ = fake_aiter

    with patch.dict("os.environ", {"KALSHI_API_TOKEN": "test"}):
        with patch("albert.ingestor.kalshi.websockets.connect", return_value=mock_ws):
            ingestor = KalshiIngestor(bus=bus, market_ids=["kalshi:BTC-24"])
            task = asyncio.create_task(ingestor.run())
            await asyncio.sleep(0.05)
            task.cancel()

    assert queue.empty()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingestor.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.ingestor.kalshi'`

- [ ] **Step 3: Create `albert/ingestor/base.py`**

```python
# albert/ingestor/base.py
import asyncio
import logging
from abc import ABC, abstractmethod

from albert.events import EventBus, MarketDataEvent

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    def __init__(self, bus: EventBus, reconnect_delay: float = 5.0) -> None:
        self._bus = bus
        self._reconnect_delay = reconnect_delay

    async def run(self) -> None:
        """Connect and stream indefinitely, reconnecting on failure."""
        while True:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "%s disconnected, reconnecting in %.1fs",
                    self.__class__.__name__, self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)

    @abstractmethod
    async def _connect_and_stream(self) -> None: ...

    @abstractmethod
    def _normalize(self, raw: dict) -> MarketDataEvent | None: ...
```

- [ ] **Step 4: Create `albert/ingestor/kalshi.py`**

```python
# albert/ingestor/kalshi.py
import json
import logging
import os
from datetime import datetime

import websockets

from albert.events import EventBus, MarketDataEvent
from .base import BaseIngestor

logger = logging.getLogger(__name__)

_WS_URL = "wss://trading-api.kalshi.com/trade-api/ws/v2"


class KalshiIngestor(BaseIngestor):
    def __init__(self, bus: EventBus, market_ids: list[str]) -> None:
        super().__init__(bus)
        self._token = os.environ["KALSHI_API_TOKEN"]
        self._tickers = [mid.removeprefix("kalshi:") for mid in market_ids if mid.startswith("kalshi:")]

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(
            _WS_URL,
            extra_headers={"Authorization": f"Bearer {self._token}"},
        ) as ws:
            for ticker in self._tickers:
                await ws.send(json.dumps({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": [ticker],
                    },
                }))
            async for raw_message in ws:
                data = json.loads(raw_message)
                event = self._normalize(data)
                if event:
                    await self._bus.publish("market_data", event)

    def _normalize(self, raw: dict) -> MarketDataEvent | None:
        if raw.get("type") not in ("orderbook_snapshot", "orderbook_delta"):
            return None
        msg = raw.get("msg", {})
        ticker = msg.get("market_ticker")
        if not ticker:
            return None
        yes = msg.get("yes", {})
        no = msg.get("no", {})
        return MarketDataEvent(
            market_id=f"kalshi:{ticker}",
            exchange="kalshi",
            timestamp=datetime.utcnow(),
            yes_bid=yes.get("bid", 0) / 100,
            yes_ask=yes.get("ask", 0) / 100,
            no_bid=no.get("bid", 0) / 100,
            no_ask=no.get("ask", 0) / 100,
            last_price=msg.get("last_price", 0) / 100,
            volume=float(msg.get("volume", 0)),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_ingestor.py -v
```
Expected: both `PASSED`

- [ ] **Step 6: Commit**

```bash
git add albert/ingestor/base.py albert/ingestor/kalshi.py tests/test_ingestor.py
git commit -m "feat: add BaseIngestor and KalshiIngestor"
```

---

## Task 12: PolymarketIngestor

**Files:**
- Create: `albert/ingestor/polymarket.py`
- Modify: `tests/test_ingestor.py`

- [ ] **Step 1: Add failing test for Polymarket ingestor**

Append to `tests/test_ingestor.py`:

```python
from albert.ingestor.polymarket import PolymarketIngestor


async def test_polymarket_ingestor_publishes_market_data_event():
    bus = EventBus()
    queue = bus.subscribe("market_data")

    ws_message = json.dumps([{
        "asset_id": "token456",
        "bid_price": "0.44",
        "ask_price": "0.46",
        "price": "0.45",
        "size": "200",
    }])

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.send = AsyncMock()

    async def fake_aiter():
        yield ws_message
        await asyncio.sleep(999)

    mock_ws.__aiter__ = fake_aiter

    with patch("albert.ingestor.polymarket.websockets.connect", return_value=mock_ws):
        ingestor = PolymarketIngestor(
            bus=bus,
            market_ids=["polymarket:cond123:token456"],
        )
        task = asyncio.create_task(ingestor.run())
        await asyncio.sleep(0.05)
        task.cancel()

    assert not queue.empty()
    event: MarketDataEvent = queue.get_nowait()
    assert event.market_id == "polymarket:cond123:token456"
    assert event.yes_ask == pytest.approx(0.46)
    assert event.no_bid == pytest.approx(0.54)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ingestor.py::test_polymarket_ingestor_publishes_market_data_event -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.ingestor.polymarket'`

- [ ] **Step 3: Create `albert/ingestor/polymarket.py`**

```python
# albert/ingestor/polymarket.py
import json
import logging
from datetime import datetime

import websockets

from albert.events import EventBus, MarketDataEvent
from .base import BaseIngestor

logger = logging.getLogger(__name__)

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/"


class PolymarketIngestor(BaseIngestor):
    def __init__(self, bus: EventBus, market_ids: list[str]) -> None:
        super().__init__(bus)
        self._market_ids = [mid for mid in market_ids if mid.startswith("polymarket:")]
        self._token_to_market: dict[str, str] = {}
        for mid in self._market_ids:
            parts = mid.removeprefix("polymarket:").split(":")
            if len(parts) > 1:
                self._token_to_market[parts[1]] = mid

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(_WS_URL) as ws:
            await ws.send(json.dumps({
                "type": "subscribe",
                "assets_ids": list(self._token_to_market.keys()),
                "channel": "price_change",
            }))
            async for raw_message in ws:
                data = json.loads(raw_message)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    event = self._normalize(item)
                    if event:
                        await self._bus.publish("market_data", event)

    def _normalize(self, raw: dict) -> MarketDataEvent | None:
        asset_id = raw.get("asset_id")
        if not asset_id:
            return None
        market_id = self._token_to_market.get(asset_id)
        if not market_id:
            return None
        yes_bid = float(raw.get("bid_price", 0))
        yes_ask = float(raw.get("ask_price", 0))
        return MarketDataEvent(
            market_id=market_id,
            exchange="polymarket",
            timestamp=datetime.utcnow(),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=round(1.0 - yes_ask, 6),
            no_ask=round(1.0 - yes_bid, 6),
            last_price=float(raw.get("price", 0)),
            volume=float(raw.get("size", 0)),
        )
```

- [ ] **Step 4: Run all ingestor tests to verify they pass**

```bash
pytest tests/test_ingestor.py -v
```
Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/ingestor/polymarket.py tests/test_ingestor.py
git commit -m "feat: add PolymarketIngestor"
```

---

## Task 13: Portfolio Tracker

**Files:**
- Create: `albert/portfolio/tracker.py`
- Create: `tests/test_portfolio_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_portfolio_tracker.py
import asyncio
import pytest
from datetime import datetime, date
from albert.db import get_connection, migrate
from albert.events import EventBus, FillEvent, MarketDataEvent
from albert.portfolio.tracker import PortfolioTracker


def make_fill(contracts: float = 5.0, price: float = 0.40, side: str = "yes") -> FillEvent:
    return FillEvent(
        fill_id="f1",
        market_id="kalshi:X",
        strategy_id="s1",
        side=side,
        contracts=contracts,
        fill_price=price,
        fee=0.0,
        filled_at=datetime.utcnow(),
    )


def make_market_data(yes_bid: float = 0.45) -> MarketDataEvent:
    return MarketDataEvent(
        market_id="kalshi:X",
        exchange="kalshi",
        timestamp=datetime.utcnow(),
        yes_bid=yes_bid,
        yes_ask=yes_bid + 0.02,
        no_bid=0.0,
        no_ask=0.0,
        last_price=yes_bid,
        volume=0.0,
    )


async def run_tracker_briefly(bus: EventBus, conn) -> None:
    tracker = PortfolioTracker(bus, conn)
    task = asyncio.create_task(tracker.run())
    await asyncio.sleep(0.05)
    task.cancel()


async def test_fill_creates_position():
    conn = get_connection(":memory:")
    migrate(conn)
    bus = EventBus()
    await bus.publish("fills", make_fill(contracts=5.0, price=0.40))
    await run_tracker_briefly(bus, conn)

    row = conn.execute("SELECT * FROM positions WHERE market_id = 'kalshi:X'").fetchone()
    assert row is not None
    assert row["contracts"] == 5.0
    assert row["avg_entry_price"] == pytest.approx(0.40)


async def test_market_data_updates_unrealized_pnl():
    conn = get_connection(":memory:")
    migrate(conn)
    bus = EventBus()
    await bus.publish("fills", make_fill(contracts=5.0, price=0.40))
    await bus.publish("market_data", make_market_data(yes_bid=0.50))
    await run_tracker_briefly(bus, conn)

    row = conn.execute("SELECT unrealized_pnl FROM positions WHERE market_id = 'kalshi:X'").fetchone()
    # (0.50 - 0.40) * 5 = 0.50
    assert row["unrealized_pnl"] == pytest.approx(0.50)


async def test_closing_fill_records_realized_pnl():
    conn = get_connection(":memory:")
    migrate(conn)
    bus = EventBus()
    # Open 5 contracts at 0.40
    await bus.publish("fills", make_fill(contracts=5.0, price=0.40))
    await run_tracker_briefly(bus, conn)

    # Close 5 contracts at 0.50 (closing fill has negative contracts)
    close_fill = FillEvent(
        fill_id="f2",
        market_id="kalshi:X",
        strategy_id="s1",
        side="yes",
        contracts=-5.0,
        fill_price=0.50,
        fee=0.0,
        filled_at=datetime.utcnow(),
    )
    bus2 = EventBus()
    await bus2.publish("fills", close_fill)
    tracker2 = PortfolioTracker(bus2, conn)
    task = asyncio.create_task(tracker2.run())
    await asyncio.sleep(0.05)
    task.cancel()

    # Position should be gone
    row = conn.execute("SELECT * FROM positions WHERE market_id = 'kalshi:X'").fetchone()
    assert row is None

    # Realized PnL recorded: (0.50 - 0.40) * 5 = 0.50
    today = date.today().isoformat()
    pnl_row = conn.execute(
        "SELECT realized_pnl FROM daily_pnl WHERE date = ? AND strategy_id = 's1'", (today,)
    ).fetchone()
    assert pnl_row is not None
    assert pnl_row["realized_pnl"] == pytest.approx(0.50)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_portfolio_tracker.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.portfolio.tracker'`

- [ ] **Step 3: Create `albert/portfolio/tracker.py`**

```python
# albert/portfolio/tracker.py
import asyncio
import logging
import sqlite3
from datetime import datetime, date

from albert.events import EventBus, FillEvent, MarketDataEvent

logger = logging.getLogger(__name__)


class PortfolioTracker:
    def __init__(self, bus: EventBus, conn: sqlite3.Connection) -> None:
        self._bus = bus
        self._conn = conn

    async def run(self) -> None:
        fills_queue = self._bus.subscribe("fills")
        market_data_queue = self._bus.subscribe("market_data")

        async def handle_fills() -> None:
            while True:
                fill: FillEvent = await fills_queue.get()
                self._handle_fill(fill)

        async def handle_market_data() -> None:
            while True:
                event: MarketDataEvent = await market_data_queue.get()
                self._handle_market_data(event)

        await asyncio.gather(handle_fills(), handle_market_data())

    def _handle_fill(self, fill: FillEvent) -> None:
        existing = self._conn.execute(
            "SELECT contracts, avg_entry_price FROM positions WHERE market_id = ? AND strategy_id = ?",
            (fill.market_id, fill.strategy_id),
        ).fetchone()

        if existing:
            new_contracts = existing["contracts"] + fill.contracts
            if new_contracts <= 0:
                realized = (fill.fill_price - existing["avg_entry_price"]) * abs(fill.contracts)
                self._conn.execute(
                    "DELETE FROM positions WHERE market_id = ? AND strategy_id = ?",
                    (fill.market_id, fill.strategy_id),
                )
                self._record_realized_pnl(fill.strategy_id, realized)
            else:
                new_avg = (
                    existing["avg_entry_price"] * existing["contracts"]
                    + fill.fill_price * fill.contracts
                ) / new_contracts
                self._conn.execute(
                    "UPDATE positions SET contracts = ?, avg_entry_price = ? WHERE market_id = ? AND strategy_id = ?",
                    (new_contracts, new_avg, fill.market_id, fill.strategy_id),
                )
        else:
            self._conn.execute(
                """INSERT INTO positions
                   (market_id, strategy_id, side, contracts, avg_entry_price, current_price, unrealized_pnl, opened_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
                (fill.market_id, fill.strategy_id, fill.side,
                 fill.contracts, fill.fill_price, fill.fill_price,
                 datetime.utcnow().isoformat()),
            )

        self._conn.commit()
        logger.info(
            "portfolio:fill market=%s strategy=%s contracts=%s price=%.4f",
            fill.market_id, fill.strategy_id, fill.contracts, fill.fill_price,
        )

    def _handle_market_data(self, event: MarketDataEvent) -> None:
        rows = self._conn.execute(
            "SELECT strategy_id, contracts, avg_entry_price, side FROM positions WHERE market_id = ?",
            (event.market_id,),
        ).fetchall()
        for row in rows:
            current_price = event.yes_bid if row["side"] == "yes" else event.no_bid
            unrealized = (current_price - row["avg_entry_price"]) * row["contracts"]
            self._conn.execute(
                "UPDATE positions SET current_price = ?, unrealized_pnl = ? WHERE market_id = ? AND strategy_id = ?",
                (current_price, unrealized, event.market_id, row["strategy_id"]),
            )
        if rows:
            self._conn.commit()

    def _record_realized_pnl(self, strategy_id: str, realized_pnl: float) -> None:
        today = date.today().isoformat()
        self._conn.execute(
            """INSERT INTO daily_pnl (date, strategy_id, realized_pnl, unrealized_pnl)
               VALUES (?, ?, ?, 0)
               ON CONFLICT(date, strategy_id) DO UPDATE SET
                   realized_pnl = realized_pnl + excluded.realized_pnl""",
            (today, strategy_id, realized_pnl),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_portfolio_tracker.py -v
```
Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/portfolio/tracker.py tests/test_portfolio_tracker.py
git commit -m "feat: add PortfolioTracker with P&L accounting"
```

---

## Task 14: CLI Status Command

**Files:**
- Create: `albert/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import io
import sys
from datetime import datetime
from albert.db import get_connection, migrate
from albert.cli import cmd_status


def make_db_with_data():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute(
        "INSERT INTO positions (market_id, strategy_id, side, contracts, avg_entry_price, current_price, unrealized_pnl, opened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("kalshi:X", "momentum_v1", "yes", 5.0, 0.40, 0.45, 0.25, datetime.utcnow().isoformat())
    )
    conn.execute(
        "INSERT INTO daily_pnl (date, strategy_id, realized_pnl, unrealized_pnl) VALUES (date('now'), 'momentum_v1', 1.50, 0.25)"
    )
    conn.commit()
    return conn


def test_status_prints_strategy_row(capsys):
    conn = make_db_with_data()
    cmd_status(conn)
    out = capsys.readouterr().out
    assert "momentum_v1" in out
    assert "1" in out  # 1 position


def test_status_prints_total_row(capsys):
    conn = make_db_with_data()
    cmd_status(conn)
    out = capsys.readouterr().out
    assert "TOTAL" in out


def test_status_empty_db_prints_totals(capsys):
    conn = get_connection(":memory:")
    migrate(conn)
    cmd_status(conn)
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "0" in out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.cli'`

- [ ] **Step 3: Create `albert/cli.py`**

```python
# albert/cli.py
import sqlite3


def cmd_status(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT
            p.strategy_id,
            COUNT(*)                                                    AS positions,
            COALESCE(SUM(p.unrealized_pnl), 0)                         AS unrealized,
            COALESCE(SUM(d.realized_pnl), 0)                           AS realized,
            COALESCE(SUM(CASE WHEN d.date = date('now') THEN d.realized_pnl ELSE 0 END), 0) AS today
        FROM positions p
        LEFT JOIN daily_pnl d ON p.strategy_id = d.strategy_id
        GROUP BY p.strategy_id
    """).fetchall()

    col_w = 22
    header = f"{'Strategy':<{col_w}} {'Positions':>10} {'Unrealized PnL':>15} {'Realized PnL':>13} {'Today':>10}"
    sep = "─" * len(header)
    print(header)
    print(sep)

    total_positions = 0
    total_unrealized = 0.0
    total_realized = 0.0
    total_today = 0.0

    for row in rows:
        u = row["unrealized"] or 0.0
        r = row["realized"] or 0.0
        t = row["today"] or 0.0
        print(
            f"{row['strategy_id']:<{col_w}} {row['positions']:>10} "
            f"{u:>+14.2f} {r:>+12.2f} {t:>+9.2f}"
        )
        total_positions += row["positions"]
        total_unrealized += u
        total_realized += r
        total_today += t

    print(sep)
    print(
        f"{'TOTAL':<{col_w}} {total_positions:>10} "
        f"{total_unrealized:>+14.2f} {total_realized:>+12.2f} {total_today:>+9.2f}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli.py -v
```
Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add albert/cli.py tests/test_cli.py
git commit -m "feat: add CLI status command"
```

---

## Task 15: Main Entrypoint and Wiring

**Files:**
- Create: `albert/__main__.py`
- Create: `albert/config.py`
- Create: `config.json` (example)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main.py
import sys
import pytest
from unittest.mock import patch, MagicMock
from albert.config import load_global_config


def test_load_global_config_returns_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_global_config()
    assert "max_total_notional_usd" in config
    assert "daily_loss_limit_usd" in config
    assert "order_debounce_seconds" in config
    assert "orderbook_ttl_days" in config


def test_load_global_config_reads_json_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text('{"max_total_notional_usd": 99999}')
    config = load_global_config()
    assert config["max_total_notional_usd"] == 99999
    # defaults still present for unspecified keys
    assert "daily_loss_limit_usd" in config
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_main.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'albert.config'`

- [ ] **Step 3: Create `albert/config.py`**

```python
# albert/config.py
import json
from pathlib import Path

_DEFAULTS = {
    "max_total_notional_usd": 10000.0,
    "daily_loss_limit_usd": -500.0,
    "order_debounce_seconds": 10,
    "orderbook_ttl_days": 7,
    "strategy_reload_interval": 30.0,
}


def load_global_config() -> dict:
    config_path = Path("config.json")
    config = dict(_DEFAULTS)
    if config_path.exists():
        overrides = json.loads(config_path.read_text())
        config.update(overrides)
    return config
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_main.py -v
```
Expected: both `PASSED`

- [ ] **Step 5: Create `albert/__main__.py`**

```python
# albert/__main__.py
import asyncio
import logging
import logging.handlers
import sqlite3
import sys
from pathlib import Path

from albert.config import load_global_config
from albert.db import get_connection, migrate
from albert.events import EventBus
from albert.execution.adapters.kalshi import KalshiAdapter
from albert.execution.adapters.polymarket import PolymarketAdapter
from albert.execution.engine import ExecutionEngine
from albert.ingestor.kalshi import KalshiIngestor
from albert.ingestor.polymarket import PolymarketIngestor
from albert.portfolio.tracker import PortfolioTracker
from albert.strategies.engine import StrategyEngine
from albert.cli import cmd_status

_LOG_FORMAT = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": "%(message)s"}'


def _setup_logging() -> None:
    handler = logging.handlers.RotatingFileHandler(
        "albert.log", maxBytes=10 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[handler, stream_handler])


async def _ttl_cleanup(conn: sqlite3.Connection, ttl_days: int) -> None:
    while True:
        await asyncio.sleep(3600)  # run every hour
        conn.execute(
            "DELETE FROM orderbook_snapshots WHERE timestamp < datetime('now', ?)",
            (f"-{ttl_days} days",),
        )
        conn.commit()


async def _main(conn: sqlite3.Connection, global_config: dict) -> None:
    bus = EventBus()

    rows = conn.execute("SELECT market_id FROM markets WHERE status = 'open'").fetchall()
    market_ids = [row["market_id"] for row in rows]

    adapters = {
        "kalshi": KalshiAdapter(),
        "polymarket": PolymarketAdapter(),
    }

    reload_interval = global_config.get("strategy_reload_interval", 30.0)

    await asyncio.gather(
        KalshiIngestor(bus, market_ids).run(),
        PolymarketIngestor(bus, market_ids).run(),
        StrategyEngine(bus, conn, reload_interval=reload_interval).run(),
        ExecutionEngine(bus, conn, adapters, global_config).run(),
        PortfolioTracker(bus, conn).run(),
        _ttl_cleanup(conn, global_config.get("orderbook_ttl_days", 7)),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        conn = get_connection()
        migrate(conn)
        cmd_status(conn)
        sys.exit(0)

    _setup_logging()
    conn = get_connection()
    migrate(conn)
    global_config = load_global_config()
    asyncio.run(_main(conn, global_config))
```

- [ ] **Step 6: Create example `config.json`**

```json
{
    "max_total_notional_usd": 10000.0,
    "daily_loss_limit_usd": -500.0,
    "order_debounce_seconds": 10,
    "orderbook_ttl_days": 7,
    "strategy_reload_interval": 30.0
}
```

- [ ] **Step 7: Run the full test suite**

```bash
pytest -v
```
Expected: all tests `PASSED`

- [ ] **Step 8: Commit**

```bash
git add albert/__main__.py albert/config.py config.json tests/test_main.py
git commit -m "feat: add main entrypoint and config loader"
```

---

## Task 16: Smoke Test — Register a Market and Strategy

This task verifies the system initializes correctly end-to-end without live exchange connections.

**Files:**
- No new files

- [ ] **Step 1: Seed the database with a test market and strategy**

```bash
python - <<'EOF'
import json
from albert.db import get_connection, migrate

conn = get_connection()
migrate(conn)

conn.execute(
    "INSERT OR IGNORE INTO markets (market_id, exchange, title, status) VALUES (?, ?, ?, ?)",
    ("kalshi:BTCZ-24DEC", "kalshi", "Will BTC close above 100k in Dec 2024?", "open")
)
conn.execute(
    "INSERT OR IGNORE INTO strategies (strategy_id, name, class_path, config, enabled) VALUES (?, ?, ?, ?, ?)",
    (
        "momentum_v1",
        "Momentum V1",
        "albert.strategies.examples.momentum.MomentumV1",
        json.dumps({"min_edge": 0.05, "kelly_fraction": 0.25, "max_position_usd": 200.0}),
        1,
    )
)
conn.commit()
print("Seeded: 1 market, 1 strategy")
EOF
```
Expected: `Seeded: 1 market, 1 strategy`

- [ ] **Step 2: Verify status command runs**

```bash
python -m albert status
```
Expected: table printed with `TOTAL` row and zero positions (no trades yet)

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "chore: add seed script output verification"
```

---

## Running the Full Test Suite

```bash
pytest -v --tb=short
```

All tests should pass before running live. To start the system live (requires API credentials in environment):

```bash
export KALSHI_API_TOKEN=your_token
export POLYMARKET_API_KEY=your_key
export POLYMARKET_API_SECRET=your_secret
export POLYMARKET_API_PASSPHRASE=your_passphrase
export POLYMARKET_ADDRESS=0xYourAddress

python -m albert
```
