# Technology Stack — Prediction Market Trading Integration

**Project:** Albert Trading System  
**Researched:** 2026-04-12  
**Context:** Improving Polymarket integration and expanding trading capabilities

---

## Recommended Stack

### Primary Exchange SDKs

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **polymarket-us** | >=0.1.2 | Polymarket US API integration | Official SDK with full API coverage, Ed25519 auth, async support, and typed exceptions. Requires Python 3.10+. Source: [PyPI](https://pypi.org/project/polymarket-us/), [GitHub](https://github.com/Polymarket/polymarket-us-python) |
| **pykalshi** | latest | Kalshi API integration | Unofficial but production-ready SDK with WebSocket streaming, automatic retry/backoff, pandas integration, and local orderbook management. Far superior DX compared to official `kalshi-python`. Source: [GitHub](https://github.com/ArshKA/pykalshi) |

**Confidence:** HIGH — Both are actively maintained (Polymarket US SDK released Jan 2026, pykalshi updated Mar 2026).

**Note:** If trading on original Polymarket (not Polymarket US), use `py-clob-client` instead:
- `pip install py-clob-client` (500K+ monthly downloads)
- EVM wallet authentication (Polygon)
- Source: [GitHub](https://github.com/Polymarket/py-clob-client)

---

### Core HTTP & WebSocket

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **httpx** | >=0.27.0 | Async HTTP client | Used by polymarket-us SDK internally. Async-first, connection pooling, retry support. |
| **websockets** | >=12.0 | WebSocket client | Official Polymarket WebSocket library dependency. Handles heartbeats, reconnection, async context managers. Source: [pypi](https://pypi.org/project/websockets/) |

**Confidence:** HIGH — Standard libraries, stable versions.

---

### Data Validation & Models

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **pydantic** | >=2.0 | Data validation | Type-safe data models. Used by both polymarket-us and pykalshi for request/response validation. |
| **pydantic-settings** | >=2.0 | Settings management | Environment-based configuration with validation. |

**Confidence:** HIGH — Industry standard for Python data validation.

---

### Database

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **aiosqlite** | >=0.19.0 | Async SQLite | Already using SQLite. `aiosqlite` provides async interface matching existing architecture. Use directly, not via ORM. |

**Confidence:** HIGH — Existing choice, confirmed working.

**Alternative:** SQLAlchemy 2.0 with async — Adds unnecessary complexity. Stick with aiosqlite and direct SQL.

---

### Job Scheduling

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **APScheduler** | >=3.10.0 | Task scheduling | For periodic tasks (config polling, market discovery, health checks). Async-compatible, multiple backends. |

**Confidence:** MEDIUM — Alternative is simple `asyncio.sleep()` loops, but APScheduler provides more control.

---

### Testing

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **pytest** | >=8.0 | Test framework | Already in use. |
| **pytest-asyncio** | >=0.23.0 | Async test support | For testing async code. Required for polymarket-us dev deps. |
| **pytest-httpx** | >=0.30.0 | HTTP mock | For mocking exchange API calls in tests. |
| **pytest-mock** | >=3.12.0 | Test mocking | General mocking utilities. |

**Confidence:** HIGH — Standard testing stack.

---

### Optional: Data Analysis

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **pandas** | >=2.0 | Dataframes | Historical analysis, backtesting, position reporting |
| **numpy** | >=1.26 | Numerical ops | Kelly criterion calculations, statistical analysis |

**Confidence:** MEDIUM — Defer unless building backtesting engine.

---

## Dependencies Analysis

### polymarket-us SDK Dependencies

From [PyPI](https://pypi.org/project/polymarket-us/):

| Package | Constraint |
|---------|-----------|
| httpx | >=0.27.0 |
| pynacl | >=1.5.0 |
| websockets | >=12.0 |

### pykalshi SDK Dependencies

| Package | Constraint |
|---------|-----------|
| httpx | >=0.27.0 |
| websockets | >=12.0 |
| aiohttp | >=3.9.0 |
| pandas | optional |
| pydantic | >=2.0 |

---

## Installation

```bash
# Core exchange SDKs
pip install polymarket-us>=0.1.2
pip install pykalshi

# Core dependencies (if not pulled by SDKs)
pip install httpx>=0.27.0 websockets>=12.0
pip install pydantic>=2.0 pydantic-settings>=2.0
pip install aiosqlite>=0.19.0

# Scheduling
pip install APScheduler>=3.10.0

# Testing
pip install -D pytest>=8.0 pytest-asyncio>=0.23.0 pytest-httpx>=0.30.0 pytest-mock>=3.12.0

# Optional: data analysis
pip install pandas numpy
```

---

## Integration Patterns

### WebSocket Connection Pattern

```python
import asyncio
import websockets
import json

async def stream_market_data(token_ids: list[str]):
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "subscribe",
            "channel": "market",
            "assets": token_ids,
        }))
        async for message in ws:
            data = json.parse(message)
            # Process orderbook update
```

### Error Handling Pattern

```python
from polymarket_us import (
    RateLimitError,
    APITimeoutError,
    AuthenticationError,
)

async def place_order_with_retry(client, order_params, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.orders.create(order_params)
        except RateLimitError:
            wait = 2 ** attempt  # Exponential backoff
            await asyncio.sleep(wait)
        except (APITimeoutError, AuthenticationError):
            raise
```

---

## Anti-Patterns to Avoid

1. **Using legacy `py-clob-client` for Polymarket US** — Use `polymarket-us` instead. Different API, different auth.
2. **Polling REST endpoints for real-time data** — WebSocket connections don't count against rate limits. Use WebSockets for market data.
3. **Blocking synchronous HTTP in async code** — Always use async libraries (httpx, aiohttp). Requests is blocking.
4. **Hardcoding API keys** — Use environment variables with `pydantic-settings`.
5. **Ignoring heartbeat requirements** — Polymarket WebSocket requires PING every 10 seconds (market) or 5 seconds (RTDS).
6. **No rate limit handling** — Implement exponential backoff. Polymarket enforces strict limits.

---

## Stack Summary

| Category | Recommended | Avoid |
|----------|------------|-------|
| Polymarket US SDK | polymarket-us >=0.1.2 | py-clob-client (different API) |
| Kalshi SDK | pykalshi | kalshi-python (limited features) |
| HTTP Client | httpx >=0.27.0 | requests (blocking) |
| WebSocket | websockets >=12.0 | websocket-client (deprecated) |
| Data Validation | pydantic >=2.0 | dataclasses (manual validation) |
| Database | aiosqlite >=0.19.0 | SQLAlchemy (overkill) |
| Scheduling | APScheduler >=3.10.0 | celery (overkill) |

---

## Sources

- Polymarket US Quickstart: https://docs.polymarket.us/api-reference/sdks/python/quickstart
- Polymarket US SDK GitHub: https://github.com/Polymarket/polymarket-us-python
- pykalshi GitHub: https://github.com/ArshKA/pykalshi
- Polymarket WebSocket Guide: https://agentbets.ai/guides/polymarket-websocket-guide/
- Polymarket API Tutorial: https://agentbets.ai/guides/polymarket-api-guide/
- PyPI polymarket-us: https://pypi.org/project/polymarket-us/