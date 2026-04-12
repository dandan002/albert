# Concerns & Risks

**Analysis Date:** 2026-04-12

## Security Concerns

### SQL Injection via f-string in TTL cleanup
- **Issue:** `albert/__main__.py` line 50 constructs a SQL DELETE query using an f-string interpolation for `ttl_days`, which comes from config (`global_config.get("orderbook_ttl_days", 7)`). While `ttl_days` is currently an integer from `config.json`, this pattern is injection-vulnerable if the config source is ever compromised or user-supplied.
  ```python
  f"DELETE FROM orderbook_snapshots WHERE timestamp < datetime('now', '-{ttl_days} days')"
  ```
- **Files:** `albert/__main__.py`
- **Current mitigation:** Config is loaded from local `config.json`; no user input path.
- **Recommendation:** Use parameterized query with `?` placeholder: `DELETE FROM orderbook_snapshots WHERE timestamp < datetime('now', '-' || ? || ' days')` with `(ttl_days,)`.

### Polymarket API authentication incomplete (acknowledged)
- **Issue:** Explicit TODO in `albert/execution/adapters/polymarket.py` line 23 states that production use requires per-request ECDSA signing via `api_secret/passphrase`. Currently only `POLY_ADDRESS` and `POLY_API_KEY` headers are sent — no cryptographic signing of requests.
- **Files:** `albert/execution/adapters/polymarket.py`
- **Impact:** Requests to Polymarket CLOB will fail authentication in production. The adapter is effectively non-functional for real trading.
- **Recommendation:** Implement L2 ECDSA request signing per Polymarket CLOB docs before enabling PolymarketAdapter in production.

### .env file loaded into process environment without validation
- **Issue:** `albert/config.py` `load_project_env()` reads `.env` file and sets all key-value pairs into `os.environ`. There is no validation, no allowlist, and no warning for unexpected keys. Any process sharing the environment can read all secrets.
- **Files:** `albert/config.py`
- **Impact:** A typo or unexpected `.env` entry silently pollutes the environment. No log of what was loaded.
- **Recommendation:** Add an allowlist of expected env vars and warn on unexpected entries.

### No `.env` file in `.gitignore` (but `.env` IS listed)
- **Files:** `.gitignore` line 43: `.env` is listed, which is correct. However, `.env.example` is also gitignored (line 44), which means example/documented env vars won't be committed. This makes it harder for new developers to know what env vars are required.
- **Recommendation:** Commit a `.env.example` file (remove it from `.gitignore`) with dummy values documenting required env vars.

### Private key handling in Kalshi adapter
- **Issue:** `albert/execution/adapters/kalshi.py` `_load_private_key()` parses a private key from an environment variable (`KALSHI_PRIVATE_KEY`). The function reconstructs PEM formatting from a space-separated single-line format, which is fragile and could silently produce an invalid key.
- **Files:** `albert/execution/adapters/kalshi.py` lines 21-42
- **Impact:** If the env var format is slightly wrong, the key loading may fail with a confusing cryptography error rather than a clear message.
- **Recommendation:** Add explicit error handling around `_load_private_key()` with a clear message on format expectations.

## Technical Debt

### Dynamic strategy loading via `importlib` is fragile
- **Issue:** `albert/strategies/engine.py` lines 42-49 uses `importlib.import_module()` and `getattr()` to dynamically load strategy classes from a `class_path` stored in the database. Errors are caught broadly with `except Exception` and only logged — no health check or alert.
- **Files:** `albert/strategies/engine.py`
- **Impact:** A typo or missing module in `class_path` silently prevents strategy activation. No validation at insert time.
- **Fix approach:** Add a `validate_class_path()` function that dry-runs the import when a strategy is inserted into the DB. Consider a strategy registry pattern instead of pure dynamic import.

### PolymarketAdapter not registered in main entry point
- **Issue:** `albert/__main__.py` line 62 only creates a `"kalshi": KalshiAdapter()` adapter dict. `PolymarketAdapter` is imported at line 16 but never instantiated or added to `adapters`. This means Polymarket orders will always fail with "no_adapter" log message even if the PolymarketIngestor is streaming market data.
- **Files:** `albert/__main__.py` lines 61-63
- **Impact:** Polymarket execution is completely broken — strategies can emit Polymarket order intents but they will be silently dropped.
- **Recommendation:** Conditionally add PolymarketAdapter to the `adapters` dict when Polymarket env vars are present (and auth is implemented).

### No graceful shutdown or cleanup
- **Issue:** `albert/__main__.py` runs coroutines via `asyncio.gather()` with no signal handling. Ctrl+C or SIGTERM will cause abrupt termination. No database connections are closed, no exchange sessions are cleaned up, and no final state is persisted.
- **Files:** `albert/__main__.py`
- **Impact:** Data loss risk on ungraceful shutdown — pending fills may not be persisted, position state could be inconsistent.
- **Fix approach:** Add `signal.signal()` handlers or use `asyncio.create_task()` with cancellation scopes to ensure `_persist_fill()` and connection cleanup run before exit.

### Hardcoded database path
- **Issue:** `albert/db.py` line 4 hardcodes `DB_PATH = Path("albert.db")`. This is relative to the working directory, which can lead to different files being used depending on where the process is started.
- **Files:** `albert/db.py`
- **Impact:** Running from a different directory creates a separate database, potentially leading to split state or missing data.
- **Fix approach:** Make `DB_PATH` configurable via environment variable or `config.json`.

### EventBus has no backpressure
- **Issue:** `albert/events.py` `EventBus` uses unbounded `asyncio.Queue` for each subscriber. During market surges, queues can grow without limit, consuming memory.
- **Files:** `albert/events.py`
- **Impact:** OOM risk during high-volume market data events.
- **Fix approach:** Use `asyncio.Queue(maxsize=N)` and decide on a backpressure strategy (drop oldest, block publisher, or alert).

## Scalability Concerns

### SQLite single-writer bottleneck
- **Issue:** The entire system uses a single SQLite database (`albert.db`) with `check_same_thread=False` for async access. SQLite supports concurrent reads but only one writer at a time. The `ExecutionEngine`, `PortfolioTracker`, `StrategyEngine`, and `_ttl_cleanup` all write to the same DB.
- **Files:** `albert/db.py`, `albert/execution/engine.py`, `albert/portfolio/tracker.py`, `albert/__main__.py`
- **Impact:** Under high trade volume, write contention will cause slowdowns and `database is locked` errors.
- **Scaling path:** Consider WAL mode (already enabled) + batching writes, or migrate to PostgreSQL for multi-writer support.

### No database indexes beyond primary keys
- **Issue:** The schema in `albert/db.py` has no indexes on commonly queried columns like `market_id` in `orderbook_snapshots`, `fills`, or `positions` tables. The `_ttl_cleanup` query scans the entire `orderbook_snapshots` table.
- **Files:** `albert/db.py`
- **Impact:** Full table scans on growing tables will degrade performance over time.
- **Fix approach:** Add indexes on `orderbook_snapshots(market_id, timestamp)`, `fills(market_id, strategy_id)`, `positions(market_id)`, and `daily_pnl(date)`.

### Orderbook TTL cleanup is blocking
- **Issue:** `_ttl_cleanup()` in `albert/__main__.py` runs every hour and executes a `DELETE` that could lock the database for the duration of the delete operation on large `orderbook_snapshots` tables.
- **Files:** `albert/__main__.py` lines 46-52
- **Impact:** Periodic latency spikes for all writers during cleanup.
- **Fix approach:** Batch the delete in chunks (e.g., `LIMIT 1000`) or use a separate connection.

### Every market data event triggers a DB write in PortfolioTracker
- **Issue:** `albert/portfolio/tracker.py` `_handle_market_data()` writes to the DB on every market data event received. For each event, it updates all positions for that market.
- **Files:** `albert/portfolio/tracker.py` lines 83-96
- **Impact:** During high-frequency market data streams, this creates a write per event per position.
- **Fix approach:** Batch market data updates (e.g., update positions every N seconds or every N events).

## Error Handling Gaps

### Broad `except Exception` swallows errors in 5 locations
- **Files and locations:**
  1. `albert/ingestor/base.py` line 22: Catches all exceptions on WebSocket stream, reconnects with delay. No distinction between transient and permanent failures.
  2. `albert/strategies/engine.py` line 48: Catches errors loading strategy from DB. Swallows and logs — the strategy simply won't activate.
  3. `albert/strategies/engine.py` line 72: Catches errors during strategy `on_market_data()`. Swallows — strategy continues receiving events even if it's broken.
  4. `albert/execution/engine.py` line 73: Catches bankroll fetch errors — silently drops the order.
  5. `albert/execution/engine.py` line 88: Catches order placement errors — halts the strategy, which is the correct behavior.
- **Impact:** Errors in strategies and ingestors are invisible except through logs. No metrics, no health checks, no structured error reporting.
- **Recommendation:** Differentiate between retriable and fatal errors. For strategies, consider disabling after N consecutive failures instead of silently continuing.

### No error handling for missing strategy config in ExecutionEngine
- **Issue:** `albert/execution/engine.py` line 67 does `json.loads(strategy_row["config"])` which could return `None` or raise `TypeError` if `strategy_row` is `None`. The fallback `strategy_config = json.loads(strategy_row["config"]) if strategy_row else {}` handles `None` but not malformed JSON.
- **Files:** `albert/execution/engine.py` lines 63-69
- **Impact:** A malformed `config` column in the strategies table would crash the execution engine.
- **Fix approach:** Wrap `json.loads()` in try/except with a log warning and fallback to defaults.

### No validation on FillEvent fields from exchange
- **Issue:** `albert/execution/adapters/kalshi.py` line 80 accesses `data["order"]` directly without checking if `"order"` key exists. `order.get(f"{intent.side}_price", price_cents)` and `order.get("count", ...)` could return unexpected types.
- **Files:** `albert/execution/adapters/kalshi.py` lines 79-91
- **Impact:** A malformed exchange response causes `KeyError` that crashes the execution's order handling loop.
- **Fix approach:** Validate the response schema before constructing `FillEvent`. Use `.get()` with meaningful defaults and type-check critical fields.

### KalshiAdapter `_post_with_retry` can raise `UnboundLocalError`
- **Issue:** `albert/execution/adapters/kalshi.py` lines 109-121: If `_MAX_RETRIES` is 0, `last_exc` will be `None` and the `raise last_exc` on line 121 would raise `None` instead of an exception. Same issue in `albert/execution/adapters/polymarket.py` lines 77-89.
- **Files:** `albert/execution/adapters/kalshi.py`, `albert/execution/adapters/polymarket.py`
- **Impact:** Edge case — unlikely since `_MAX_RETRIES = 3` is hardcoded, but if ever changed to 0, produces a confusing error.
- **Recommendation:** Guard with `if last_exc is None: raise RuntimeError("No retries attempted")`.

### No handling of WebSocket disconnections in ingestors
- **Issue:** While `BaseIngestor.run()` in `albert/ingestor/base.py` catches `Exception` and reconnects, it uses a fixed `reconnect_delay` with no exponential backoff. Continuous connection failures will DDoS the exchange API.
- **Files:** `albert/ingestor/base.py`
- **Impact:** If an exchange is down, the system will hammer it with reconnection attempts every 5 seconds forever.
- **Fix approach:** Implement exponential backoff with jitter and a max retry count with alert.

## Missing Features

### No order status tracking
- **Issue:** The system places orders and immediately records them as fills in `FillEvent`, but never checks if the order was actually filled. There is no background task monitoring open orders.
- **Files:** `albert/execution/engine.py`, `albert/execution/adapters/base.py`
- **Impact:** The `cancel_order()` method exists on adapters but is never called. If an order is partially filled or rejected, the system has no mechanism to reconcile the difference.

### No Polymarket adapter in production entry point
- **Issue:** Only `KalshiAdapter` is instantiated in `albert/__main__.py`. PolymarketIngestor streams Polymarket data but no adapter handles Polymarket execution.
- **Files:** `albert/__main__.py` lines 61-63
- **Impact:** Polymarket market data is processed through strategies, but any resulting Polymarket orders are silently dropped.

### No health check or readiness endpoint
- **Issue:** There is no HTTP health endpoint or health check mechanism. Operators cannot determine if the system is running, connected to exchanges, or processing data.
- **Impact:** In production, there is no way to monitor system health except through log files.

### No position reconciliation on startup
- **Issue:** On restart, the system loads market IDs from the DB and starts ingesting, but does not reconcile open positions against exchange state. If the system crashed with pending fills, positions in the DB may be stale.
- **Files:** `albert/__main__.py`, `albert/portfolio/tracker.py`
- **Impact:** After a crash, positions in the database may not match actual exchange positions.

### No test coverage for main entry point's `_main()` orchestration
- **Issue:** `albert/__main__.py` has no tests for `_main()`, `_ttl_cleanup()`, `_check_env()`, or the `if __name__ == "__main__"` branch. The integration between all subsystems via `asyncio.gather()` is untested.
- **Files:** `albert/__main__.py`
- **Impact:** Regressions in startup, env checking, or orchestration will not be caught by CI.

## Code Quality Issues

### Duplicated retry logic across adapters
- **Issue:** Both `albert/execution/adapters/kalshi.py` (lines 93-102, 109-121) and `albert/execution/adapters/polymarket.py` (lines 61-70, 77-89) contain nearly identical `_post_with_retry()` and `cancel_order()` retry implementations.
- **Files:** `albert/execution/adapters/kalshi.py`, `albert/execution/adapters/polymarket.py`
- **Recommendation:** Extract a shared `_retry_request()` helper into `albert/execution/adapters/base.py` or a utility module.

### Duplicated market ID prefix parsing pattern
- **Issue:** Both `PolymarketAdapter._token_id()` and `PolymarketIngestor.__init__()` parse `polymarket:<condition_id>:<token_id>` format. Similarly, `KalshiIngestor` and `KalshiAdapter` both strip `kalshi:` prefix. This parsing is ad-hoc and not centralized.
- **Files:** `albert/execution/adapters/polymarket.py` line 36-38, `albert/ingestor/polymarket.py` line 20-23, `albert/ingestor/kalshi.py` line 26, `albert/execution/adapters/kalshi.py` line 69
- **Recommendation:** Create a `MarketId` dataclass or utility to parse/construct market IDs, ensuring consistent format handling across the codebase.

### Inconsistent error handling between adapters
- **Issue:** `PolymarketAdapter.place_order()` uses `_post_with_retry()` but `PolymarketAdapter.get_bankroll()` does a single `r.raise_for_status()` with no retry. `KalshiAdapter` has the same inconsistency. Bankroll fetch failures halt strategy execution (line 72-75 of `engine.py`), so this matters.
- **Files:** `albert/execution/adapters/polymarket.py` lines 72-75, `albert/execution/adapters/kalshi.py` lines 104-107
- **Recommendation:** Apply retry logic to bankroll and other read operations, or accept that bankroll calls may fail and use the last known value.

### `asyncio.get_event_loop()` is deprecated
- **Issue:** `albert/strategies/engine.py` line 54 uses `asyncio.get_event_loop()` which is deprecated since Python 3.10. Should use `asyncio.get_running_loop()`.
- **Files:** `albert/strategies/engine.py`
- **Impact:** Will produce deprecation warnings on Python 3.10+.
- **Fix approach:** Replace `asyncio.get_event_loop()` with `asyncio.get_running_loop()`.

### `print()` used in CLI module instead of logging
- **Issue:** `albert/cli.py` uses `print()` for all output. While this is acceptable for a CLI status command, it doesn't integrate with the structured JSON logging in the rest of the application.
- **Files:** `albert/cli.py`
- **Impact:** Minor — only affects the `status` CLI command output.

## Dependency Risks

### No version pinning for transitive dependencies
- **Issue:** `pyproject.toml` uses `>=` for all dependencies (websockets>=12.0, httpx>=0.27, cryptography>=42.0). This means a `pip install` could pull a major version bump that breaks compatibility.
- **Files:** `pyproject.toml`
- **Fix approach:** Pin exact versions or use `~=`` for compatible release specifiers (e.g., `websockets~=12.0`).

### `websockets` dependency is implicit via both ingestor modules
- **Issue:** The `websockets` library is imported directly in `albert/ingestor/kalshi.py` and `albert/ingestor/polymarket.py` but is not listed with an upper bound. If `websockets` makes breaking API changes in a major version, both ingestors break simultaneously.
- **Files:** `pyproject.toml`, `albert/ingestor/kalshi.py`, `albert/ingestor/polymarket.py`
- **Recommendation:** Consider an adapter pattern for WebSocket transport to isolate interface changes.

### No `cryptography` version upper bound
- **Issue:** `cryptography>=42.0` has no upper bound. The Kalshi adapter relies heavily on specific `cryptography` APIs (PSS signing, PEM key loading). Breaking API changes could silently affect signature generation.
- **Files:** `pyproject.toml`, `albert/execution/adapters/kalshi.py`, `albert/ingestor/kalshi.py`
- **Recommendation:** Pin to a tested version range.

## Testing Gaps

### No integration tests
- **Issue:** All tests use mocks for HTTP clients, WebSocket connections, and database connections. There are no tests that exercise the full pipeline from ingestor → strategy → execution → portfolio with real (or test) exchange behavior.
- **Files:** `tests/` directory
- **Risk:** End-to-end data flow bugs (e.g., market ID format mismatches between ingestor and adapter) will not be caught.

### No tests for `PortfolioTracker._handle_market_data` PnL direction
- **Issue:** `test_portfolio_tracker.py` tests that `yes_bid` updates produce correct unrealized PnL, but does not test `no_bid` for `side="no"` positions. The code at `albert/portfolio/tracker.py` line 89 uses a different price (`no_bid`) for `side="no"` positions, but this branch is untested.
- **Files:** `tests/test_portfolio_tracker.py`, `albert/portfolio/tracker.py`
- **Risk:** Incorrect PnL calculations for `no` side positions.

### No tests for `_ttl_cleanup` or time-based operations
- **Issue:** The `_ttl_cleanup` coroutine in `albert/__main__.py` has no tests. The `RiskChecker` debounce logic uses `time.monotonic()` which is only tested in `test_risk.py` with `debounce=0` or `debounce=60`.
- **Files:** `tests/` — no test file for `__main__.py`
- **Risk:** Stale data accumulation; TTL cleanup may not work as expected.

### No tests for edge cases in `_load_private_key()`
- **Issue:** The `_load_private_key()` function in `albert/execution/adapters/kalshi.py` has complex PEM reconstruction logic (handling missing headers, space-separated keys) — none of this is tested directly.
- **Files:** `tests/test_adapters.py` — `_load_private_key` is mocked out
- **Risk:** Production key loading may fail with unclear errors for valid key formats.

### No concurrent access tests
- **Issue:** SQLite is accessed from multiple async coroutines (`ExecutionEngine`, `PortfolioTracker`, `StrategyEngine`, `_ttl_cleanup`) with `check_same_thread=False`. No tests verify concurrent writes don't cause `database is locked` errors.
- **Files:** `albert/db.py`, `tests/`
- **Risk:** Production deadlocks or data corruption under load.

### `test_portfolio_tracker.py` has `pass` statements in teardown
- **Issue:** Lines 44, 60, 82, 118, 162 contain `pass` statements inside `except asyncio.CancelledError` blocks that do nothing. While harmless (the cancellation propagates), they add noise and suggest incomplete cleanup patterns.
- **Files:** `tests/test_portfolio_tracker.py`
- **Impact:** Minor — no functional impact but indicates the test helper pattern could be cleaner.