# Testing

**Analysis Date:** 2026-04-12

## Test Framework

**Runner:**
- pytest 8.0+ with pytest-asyncio 0.23+
- Config: `pyproject.toml` → `[tool.pytest.ini_options] asyncio_mode = "auto"`

**Assertion Library:**
- Standard `assert` statements — no Hamcrest or custom matchers
- `pytest.approx()` for floating-point comparisons (used extensively in `tests/test_kelly.py`, `tests/test_events.py`, `tests/test_ingestor.py`, `tests/test_risk.py`)

**Run Commands:**
```bash
pytest                          # Run all tests
pytest tests/                   # Run all tests from project root
pytest tests/test_kelly.py      # Run a single test file
pytest -x                       # Stop on first failure
pytest -v                       # Verbose output
```

**No coverage command configured** — no `.coveragerc`, no `--cov` in pytest config.

## Test File Organization

**Location:**
- Co-located in a top-level `tests/` directory (not inside `albert/`)
- Pattern: `tests/test_{module_name}.py` maps to `albert/{module_name}.py` or `albert/{package}/{module_name}.py`

**Mapping:**
| Test File | Source Module |
|---|---|
| `tests/test_strategy_base.py` | `albert/strategies/base.py` + `albert/strategies/examples/momentum.py` |
| `tests/test_strategy_engine.py` | `albert/strategies/engine.py` |
| `tests/test_db.py` | `albert/db.py` |
| `tests/test_ingestor.py` | `albert/ingestor/kalshi.py`, `albert/ingestor/polymarket.py` |
| `tests/test_kelly.py` | `albert/execution/kelly.py` |
| `tests/test_risk.py` | `albert/execution/risk.py` |
| `tests/test_portfolio_tracker.py` | `albert/portfolio/tracker.py` |
| `tests/test_adapters.py` | `albert/execution/adapters/kalshi.py`, `albert/execution/adapters/polymarket.py` |
| `tests/test_events.py` | `albert/events.py` |
| `tests/test_cli.py` | `albert/cli.py` |
| `tests/test_main.py` | `albert/config.py` |
| `tests/test_execution_engine.py` | `albert/execution/engine.py` |

**Naming:**
- All test files use `test_` prefix
- Test functions use `test_` prefix with descriptive names: `test_positive_edge_returns_positive_size`, `test_blocks_on_debounce`, `test_kalshi_ingestor_publishes_market_data_event`

## Test Structure

**Suite Organization:**
```python
# Standard pattern — standalone test functions (no classes)
async def test_kalshi_ingestor_publishes_market_data_event():
    bus = EventBus()
    queue = bus.subscribe("market_data")
    # ... arrange, act, assert
    assert not queue.empty()

# Some tests define helper functions at module level
def make_event(yes_ask: float) -> MarketDataEvent:
    return MarketDataEvent(...)

def make_db():
    conn = get_connection(":memory:")
    migrate(conn)
    # ... seed data
    return conn
```

**No `conftest.py` fixtures:** `tests/conftest.py` exists but is empty. Test helpers are defined as module-level functions (e.g., `make_db()`, `make_event()`, `make_intent()`, `make_mock_adapter()`).

**Patterns:**
- **Arrange-Act-Assert** order within each test function
- **Factory functions** for creating test data: `make_event()`, `make_fill()`, `make_intent()`, `make_checker()`, `make_db()`, `make_db_with_strategy()`, `make_mock_adapter()`, `make_private_key()`, `make_market_data()`
- **No test classes used** — all tests are module-level functions (both sync and async)

## Async Testing

**Mode:** `asyncio_mode = "auto"` — all `async def test_` functions are automatically treated as coroutines by pytest-asyncio

**Pattern for testing long-running services:**
```python
async def test_engine_publishes_intent_for_active_strategy():
    conn = make_db()
    bus = EventBus()
    intents_queue = bus.subscribe("order_intents")
    engine = StrategyEngine(bus, conn, reload_interval=999)

    await bus.publish("market_data", make_event(0.30))

    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.05)
    engine_task.cancel()

    assert not intents_queue.empty()
```

Key pattern: Start service as `asyncio.create_task()`, sleep briefly, then cancel. Results are verified on the EventBus queue or database.

**CancelledError handling in services:**
```python
# Some tests catch CancelledError explicitly
task.cancel()
try:
    await task
except asyncio.CancelledError:
    pass
```

**No explicit `@pytest.mark.asyncio`** — auto mode handles this, though some tests still have `@pytest.mark.asyncio` decorator (e.g., `test_strategy_engine.py`).

## Mocking

**Framework:** `unittest.mock` (standard library) — no additional mocking libraries

**Patterns:**

1. **AsyncMock for async methods:**
```python
from unittest.mock import AsyncMock, MagicMock
mock_response = MagicMock()
mock_response.raise_for_status = MagicMock()
mock_response.json.return_value = {"order": {...}}
adapter._client.post = AsyncMock(return_value=mock_response)
```

2. **Patching websockets for ingestor tests:**
```python
with patch("albert.ingestor.kalshi.websockets.connect", return_value=mock_ws):
    ingestor = KalshiIngestor(bus=bus, market_ids=["kalshi:BTC-24"])
```

3. **Patching environment variables:**
```python
with patch.dict("os.environ", {"KALSHI_API_KEY_ID": "test_key_id", ...}):
    with patch("albert.ingestor.kalshi._load_private_key", return_value=make_private_key()):
```

4. **Mocking ExchangeAdapter with spec:**
```python
adapter = MagicMock(spec=ExchangeAdapter)
adapter.get_bankroll = AsyncMock(return_value=10000.0)
adapter.place_order = AsyncMock(return_value=FillEvent(...))
```

5. **Monkeypatch for filesystem/config tests:**
```python
def test_load_global_config_returns_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_global_config()
```

**What to Mock:**
- External HTTP calls (exchange APIs) — always mock `httpx.AsyncClient` methods
- WebSocket connections — always mock `websockets.connect()`
- Environment variables — use `patch.dict("os.environ", {...})` or `monkeypatch`
- Private key loading — mock `_load_private_key()`
- Time-dependent behavior — `RiskChecker` debounce uses `time.monotonic()` directly (not mocked in tests, uses real time with small values)

**What NOT to Mock:**
- SQLite database — use in-memory (`:memory:`) databases with real `get_connection()` and `migrate()`
- EventBus — use real `EventBus` instances
- Kelly criterion calculation — test the real function directly

## Fixtures and Helpers

**Database seeding pattern:**
```python
def make_db():
    conn = get_connection(":memory:")
    migrate(conn)
    conn.execute("INSERT INTO strategies ...", (...)
    conn.commit()
    return conn
```

**Event creation pattern:**
```python
def make_event(yes_ask: float) -> MarketDataEvent:
    return MarketDataEvent(
        market_id="kalshi:X",
        exchange="kalshi",
        timestamp=datetime.now(timezone.utc),
        yes_bid=yes_ask - 0.02,
        yes_ask=yes_ask,
        ...
    )
```

**No pytest fixtures defined.** All shared setup is done via module-level helper functions rather than `@pytest.fixture`. The `conftest.py` file exists but is empty.

**MagicMock for private key:**
```python
def make_private_key():
    key = MagicMock()
    key.sign.return_value = b"sig"
    return key
```

## Test Coverage

**Estimated coverage:** ~85-90% of core business logic

**Well-tested modules:**
- `albert/events.py` — EventBus pub/sub, event creation
- `albert/db.py` — Migration, table creation, idempotent migration
- `albert/execution/kelly.py` — All edge cases (zero edge, invalid prices, max position cap, confidence scaling)
- `albert/execution/risk.py` — Debounce, daily loss limit, max notional
- `albert/portfolio/tracker.py` — Position creation, PnL tracking, partial/full close
- `albert/config.py` — Default config, JSON override, dotenv parsing

**Partially tested modules:**
- `albert/ingestor/kalshi.py` — Tested via mock websockets, but auth header generation not independently verified
- `albert/ingestor/polymarket.py` — Message normalization tested, connection/reconnect not tested
- `albert/execution/engine.py` — Order flow tested; bankroll error and order failure paths tested; but `_persist_fill()` is tested only via end-to-end
- `albert/cli.py` — Only `cmd_status()` tested with captured stdout

**Untested areas:**
- `albert/__main__.py` — `_check_env()`, `_setup_logging()`, `_main()`, `_ttl_cleanup()` have no tests
- `albert/ingestor/base.py` — Reconnection loop (the `run()` method) not independently tested
- Adapter retry logic — `_post_with_retry()` failure paths after max retries (the final raise) not tested
- `albert/execution/adapters/kalshi.py` — `_load_private_key()` PEM parsing logic not tested independently
- `albert/execution/adapters/polymarket.py` — Authentication/ECDSA signing not tested (noted as TODO in source)
- Integration/end-to-end tests — no tests that exercise the full pipeline from ingestor through strategy to execution

## Common Patterns

**Database test pattern:**
```python
conn = get_connection(":memory:")  # isolated per test
migrate(conn)
# ... seed and test
# no teardown needed — in-memory DB is garbage collected
```

**EventBus test pattern:**
```python
bus = EventBus()
queue = bus.subscribe("channel_name")
# ... publish events
result = queue.get_nowait()
assert result.field == expected_value
```

**Asserting queue emptiness:**
```python
assert queue.empty()  # no events received
assert not queue.empty()  # events were received
event = queue.get_nowait()  # retrieve without blocking
```

**Testing with time delays (not recommended, but used):**
```python
engine_task = asyncio.create_task(engine.run())
await asyncio.sleep(0.05)  # give the event loop time to process
engine_task.cancel()
```

**Error Testing Pattern:**
Currently no tests that verify error handling paths (e.g., what happens when a strategy raises an exception). The `StrategyEngine` catches exceptions and logs them, but no test verifies this behavior.

---

*Testing analysis: 2026-04-12*