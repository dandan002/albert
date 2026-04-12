# Integrations

**Analysis Date:** 2026-04-12

## External APIs

| Service | Purpose | Auth Method | Direction |
|---------|---------|-------------|-----------|
| Kalshi (elections API) | Prediction market orderbook streaming, order placement, balance queries | RSA PSS-signed requests (`KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY` env vars) | Outbound (REST + WebSocket) |
| Polymarket (CLOB API) | Prediction market price streaming, order placement, balance queries | API key headers (`POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE`, `POLYMARKET_ADDRESS` env vars) | Outbound (REST + WebSocket) |

### Kalshi Integration Details

**WebSocket (Ingestor):**
- URL: `wss://api.elections.kalshi.com/trade-api/ws/v2`
- Auth: RSA PSS signature on `{timestamp}GET/trade-api/ws/v2` sent as headers
- Implementation: `albert/ingestor/kalshi.py` → `KalshiIngestor._connect_and_stream()`
- Subscribes to `orderbook_delta` channel for configured market tickers
- Normalizes prices from cents (0–100) to decimal (0–1)

**REST API (Adapter):**
- Base URL: `https://api.elections.kalshi.com/trade-api/v2`
- Auth: Every request signed via `httpx` event hook in `albert/execution/adapters/kalshi.py`
- Endpoints used:
  - `POST /portfolio/orders` — place limit orders
  - `DELETE /portfolio/orders/{order_id}` — cancel orders
  - `GET /portfolio/balance` — check available balance
- Retry: Exponential backoff (2^attempt seconds), max 3 retries

### Polymarket Integration Details

**WebSocket (Ingestor):**
- URL: `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Auth: None (public WebSocket)
- Implementation: `albert/ingestor/polymarket.py` → `PolymarketIngestor._connect_and_stream()`
- Subscribes to `price_change` channel by `assets_ids`
- Market IDs use format `polymarket:<condition_id>:<token_id>`
- Derives NO prices from YES prices: `no_bid = 1 - yes_ask`, `no_ask = 1 - yes_bid`

**REST API (Adapter):**
- Base URL: `https://clob.polymarket.com`
- Auth: Headers `POLY_ADDRESS`, `POLY_API_KEY` — **INCOMPLETE**: per TODO comment, production requires per-request ECDSA signing
- Endpoints used:
  - `POST /order` — place GTC limit orders
  - `DELETE /order/{order_id}` — cancel orders
  - `GET /balance` — check balance
- Retry: Same exponential backoff pattern, max 3 retries

## Internal Module Integration Points

### EventBus (Central Nervous System)

All inter-module communication flows through `albert/events.py:EventBus`, an in-process pub/sub using `asyncio.Queue`.

**Channels:**

| Channel | Publisher | Subscriber(s) | Event Type |
|---------|-----------|---------------|-------------|
| `market_data` | `KalshiIngestor`, `PolymarketIngestor` | `StrategyEngine`, `ExecutionEngine`, `PortfolioTracker` | `MarketDataEvent` |
| `order_intents` | `StrategyEngine` | `ExecutionEngine` | `OrderIntent` |
| `fills` | `ExecutionEngine` | `PortfolioTracker` | `FillEvent` |
| `strategy_halted` | `ExecutionEngine` | (logged, no current subscriber) | `StrategyHaltedEvent` |

### Data Flow Between Modules

```
Kalshi WS ──┐
             ├──→ market_data ──→ StrategyEngine ──→ order_intents ──→ ExecutionEngine ──→ fills ──→ PortfolioTracker
Polymarket WS┘                   (reloads from DB)                     (RiskChecker)              (updates positions/pnl)
                                     │                                    │
                                     └── Strategy DB table               ├── KalshiAdapter (REST)
                                                                            └── PolymarketAdapter (REST)
```

### Strategy Loading (Dynamic Import)

- `StrategyEngine` loads strategy classes from `strategies.class_path` column in the `strategies` SQLite table
- Format: `module.path.ClassName` (e.g., `albert.strategies.examples.momentum.MomentumV1`)
- Uses `importlib.import_module()` for runtime discovery
- Hot-reloads every `strategy_reload_interval` seconds (default 30s)
- Disabling a strategy in DB removes it from active strategies

## Data Flows

### Market Data Ingestion Flow

1. WebSocket connects with auth headers (Kalshi) or unauthenticated (Polymarket)
2. Raw JSON messages received and normalized via `_normalize()` method
3. `MarketDataEvent` published to `market_data` EventBus channel
4. `BaseIngestor.run()` wraps streaming in reconnect loop (5s delay on disconnect)
5. `ExecutionEngine` caches latest ask prices in `_price_cache` dict for order sizing
6. `PortfolioTracker` updates position market values on each tick

### Order Execution Flow

1. `StrategyEngine` receives `MarketDataEvent`, calls `strategy.on_market_data(event)`
2. Strategy returns `list[OrderIntent]` (or `None`)
3. `StrategyEngine` publishes intents to `order_intents` channel
4. `ExecutionEngine` receives intent, determines exchange from market_id prefix (`kalshi:` or `polymarket:`)
5. Fetches bankroll from exchange adapter
6. Computes position size via `kelly_size()` (edge, price, bankroll, kelly_fraction, confidence, max_position_usd)
7. Passes through `RiskChecker.check()` (debounce, daily loss limit, max notional)
8. Places order via exchange adapter, receives `FillEvent`
9. Persists fill to SQLite `fills` table
10. Publishes `FillEvent` to `fills` channel

### Portfolio Tracking Flow

1. `PortfolioTracker` subscribes to `fills` and `market_data` channels
2. On fill: updates `positions` table (weighted average entry, contract count, realized PnL on closes)
3. On market data: updates `current_price` and `unrealized_pnl` for all positions on that market
4. Realized PnL recorded in `daily_pnl` table

## Third-Party Libraries Integration

### `websockets` (>=12.0)
- Used in both `KalshiIngestor` and `PolymarketIngestor`
- Called as `async with websockets.connect(url, ...) as ws:` context manager
- Kalshi: passes `additional_headers` for auth; Polymarket: no auth headers
- Both follow the `BaseIngestor` pattern: `run()` → `_connect_and_stream()` → `_normalize()`

### `httpx` (>=0.27)
- Used in `KalshiAdapter` and `PolymarketAdapter` for REST API calls
- `httpx.AsyncClient` with configurable `base_url` and `timeout=10.0`
- Kalshi: uses `event_hooks={"request": [self._sign_request]}` to auto-sign every request
- Polymarket: static headers for API key
- Both adapters implement exponential backoff `_post_with_retry()` with max 3 retries

### `cryptography` (>=42.0)
- Used exclusively in `albert/execution/adapters/kalshi.py`
- `serialization.load_pem_private_key()` — loads RSA private key from env var string
- `padding.PSS()` with `hashes.SHA256()` — signs request payloads for Kalshi auth
- `_load_private_key()` handles single-line PEM keys (newlines replaced by spaces) by reconstructing proper PEM format

## Environment Configuration

**Required env vars:**

| Variable | Used In | Purpose |
|----------|---------|---------|
| `KALSHI_API_KEY_ID` | `KalshiAdapter`, `KalshiIngestor` | Kalshi API key identifier |
| `KALSHI_PRIVATE_KEY` | `KalshiAdapter`, `KalshiIngestor` | RSA private key PEM string for request signing |
| `POLYMARKET_API_KEY` | `PolymarketAdapter` | Polymarket CLOB API key |
| `POLYMARKET_API_SECRET` | `PolymarketAdapter` | Polymarket CLOB API secret (ECDSA — TODO: not yet used for signing) |
| `POLYMARKET_API_PASSPHRASE` | `PolymarketAdapter` | Polymarket CLOB API passphrase (TODO: not yet used) |
| `POLYMARKET_ADDRESS` | `PolymarketAdapter` | Wallet address for Polymarket header |

**Config file (`config.json`):**

| Key | Default | Purpose |
|-----|---------|---------|
| `max_total_notional_usd` | 10000.0 | Max total position notional across all strategies |
| `daily_loss_limit_usd` | -500.0 | Daily loss threshold (negative = loss) |
| `order_debounce_seconds` | 10 | Min seconds between orders on same market+strategy |
| `orderbook_ttl_days` | 7 | Days to retain orderbook snapshots before cleanup |
| `strategy_reload_interval` | 30.0 | Seconds between strategy config reloads from DB |

---

*Integration audit: 2026-04-12*