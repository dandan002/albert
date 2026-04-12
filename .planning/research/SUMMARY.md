# Project Research Summary

**Project:** Albert Trading System  
**Domain:** Automated Prediction Market Trading (Polymarket, Kalshi)  
**Researched:** 2026-04-12  
**Confidence:** HIGH

---

## Executive Summary

Albert is an automated trading system for prediction markets (Polymarket and Kalshi) built with Python asyncio. Research confirms the existing architecture—five async modules (Ingestor, StrategyEngine, ExecutionEngine, PortfolioTracker, EventBus) communicating via typed events—aligns with production trading system patterns. The immediate priority is fixing critical authentication and adapter registration issues blocking Polymarket production trading, then expanding the strategy library and adding paper trading mode. Key risks include incomplete ECDSA implementation (blocks Polymarket orders), lack of edge calculation in strategies (amplifies losses via Kelly sizing), and thin market liquidity (destroys theoretical edges through slippage).

---

## Key Findings

### Recommended Stack

**Core technologies:**
- **polymarket-us >=0.1.2** — Official SDK for Polymarket US API; includes Ed25519 auth, async support, typed exceptions. Use instead of legacy `py-clob-client`.
- **pykalshi** — Production-ready Kalshi SDK with WebSocket streaming, auto-retry, pandas integration. Far superior to official `kalshi-python`.
- **httpx >=0.27.0** — Async HTTP client (used by both SDKs); supports connection pooling and retry.
- **websockets >=12.0** — WebSocket client for real-time market data; handles heartbeats (10s for market, 5s for RTDS) and reconnection.
- **pydantic >=2.0** — Type-safe data validation for request/response models.
- **aiosqlite >=0.19.0** — Async SQLite interface; current choice confirmed appropriate.

**Avoid:**
- `py-clob-client` — For original Polymarket only; Polymarket US requires `polymarket-us`
- `requests` — Blocking; use httpx in async code
- `websocket-client` — Deprecated; use `websockets`
- SQLAlchemy — Overkill for this use case; stick with aiosqlite + direct SQL

### Expected Features

**Must have (table stakes):**
- Real-time WebSocket data ingestion — Already implemented for Kalshi + Polymarket
- Kelly criterion position sizing — Mathematically optimal for edge-based betting
- Pre-trade risk checks (max position, daily loss limit, duplicate debounce) — Already implemented
- Portfolio P&L tracking with mark-to-market — Already implemented
- Strategy hot-reload without restart — Already implemented
- Multi-exchange support (Polymarket + Kalshi) — Core to competitive advantage

**Should have (competitive differentiators):**
- **Paper trading mode** — Enables user onboarding without capital risk; high value, medium complexity
- **Strategy library expansion** — Add momentum and mean reversion strategies; medium complexity
- **Backtesting engine** — #1 requested feature for serious traders; high complexity, defer to v2
- **Cross-platform arbitrage** — Exploit price discrepancies between Polymarket and Kalshi; high complexity

**Defer (v2+):**
- Web dashboard — Initially out of scope per PROJECT.md
- Multi-account management — Adds significant complexity
- Advanced order types (TWAP/VWAP) — Requires large position sizes to benefit
- Whale tracking — External API dependency

### Architecture Approach

Five async modules communicate through `EventBus` (asyncio.Queue-based pub/sub):

1. **Ingestor Layer** — WebSocket connections to exchanges, normalizes to typed MarketDataEvent
2. **StrategyEngine** — Loads strategies from DB, evaluates on market data, emits OrderIntent
3. **ExecutionEngine** — Kelly sizing, risk checks, adapter routing, persist fills
4. **PortfolioTracker** — Maintains positions, computes P&L on fills and price updates
5. **EventBus** — Fan-out pub/sub with typed channels

**Build order:** Tier 1 (DB, EventBus, Config) → Tier 2 (Ingestors) → Tier 3 (Strategy) → Tier 4 (Execution) → Tier 5 (Tracking)

**Production patterns to add (future phases):**
- Circuit breakers (consecutive loss, drawdown protection)
- HealthMonitor (component heartbeat watchdog)
- Position state machine (explicit lifecycle states)
- Multi-exchange router with liquidity checking

### Critical Pitfalls

1. **Polymarket ECDSA Authentication Not Implemented** — Adapter lacks per-request ECDSA signing required by Polymarket CLOB API. Orders fail with 401 errors. Prevention: Implement L2 ECDSA signing with env vars (`POLY_ADDRESS`, `POLY_API_KEY`, `POLY_API_SECRET`, `POLY_PASSPHRASE`).

2. **PolymarketAdapter Not Registered in Main Entry Point** — Adapter imported but never instantiated in `albert/__main__.py`. Orders silently drop with "no_adapter" log. Prevention: Conditionally add to adapters dict when env vars present.

3. **Position Sizing Without Edge Calculation** — Kelly criterion amplifies losses when strategies trade without genuine edge. Prevention: Require `estimate_edge()` to compare strategy probability to market price; only emit orders when edge exceeds threshold (e.g., `min_edge: 0.05`).

4. **Ignoring Liquidity in Thin Markets** — Enter at reasonable price, exit at 50-70% of mark due to lack of counterparties. A 10% edge becomes 30% loss. Prevention: Add pre-trade liquidity check; reject if depth < 2x position size or total volume < $5,000.

5. **No Graceful Shutdown** — Ctrl+C causes abrupt termination; pending fills not persisted, position state becomes inconsistent. Prevention: Add signal handlers for SIGINT/SIGTERM; use asyncio.TaskGroup with cancellation scopes.

---

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 71: Polymarket Production Readiness
**Rationale:** Critical blockers must be fixed before any Polymarket trading works. Current adapter cannot execute orders.
**Delivers:** Functional Polymarket order execution with ECDSA auth, proper adapter registration
**Addresses:** Pitfall 1, Pitfall 2 — ECDSA implementation and adapter registration
**Avoids:** Silent order failures, authentication errors in production

### Phase 72: Observability & Resilience
**Rationale:** Production deployment requires system health monitoring and graceful shutdown.
**Delivers:** HealthMonitor component, structured logging, graceful signal handling
**Addresses:** Pitfall 5 — No graceful shutdown; missing component heartbeat monitoring
**Uses:** APScheduler for health check scheduling

### Phase 73: Risk Enhancements & Circuit Breakers
**Rationale:** Protect bankroll from consecutive losses and excessive drawdown.
**Delivers:** ConsecutiveLossBreaker, DrawdownBreaker, position state machine
**Addresses:** Architecture patterns from research; protects against cascade losses
**Implements:** Circuit breaker pattern between RiskChecker and ExecutionEngine

### Phase 74: Paper Trading & Strategy Expansion
**Rationale:** Low complexity, high value differentiation. Paper trading enables user onboarding without risk.
**Delivers:** Paper trading mode with mock adapter, virtual portfolio, momentum/mean reversion strategies
**Addresses:** Feature differentiators — paper trading mode, expanded strategy library
**Requires:** Strategy.estimate_edge() properly implemented in all strategies

### Phase 75: Multi-Venue & Liquidity Intelligence
**Rationale:** Add liquidity checking before orders, prepare for cross-exchange capabilities.
**Delivers:** Pre-trade liquidity validation, MultiExchangeRouter with venue health monitoring
**Addresses:** Pitfall 4 — liquidity ignored; adds intelligent routing
**Implements:** Architecture pattern for multi-exchange fallback

### Phase Ordering Rationale

- **Why this order:** Phase 71 fixes critical blockers (no Polymarket trading works without it). Phase 72 adds observability needed for production debugging. Phase 73 adds risk protection before expanding strategy complexity. Phase 74 delivers user value with paper trading. Phase 75 adds production-hardening for multi-venue trading.
- **Grouping rationale:** Phases 71-72 are prerequisites for any production use. Phase 73 protects capital during active trading. Phase 74 delivers differentiation. Phase 75 scales to multi-exchange.
- **How this avoids pitfalls:** Each phase explicitly addresses pitfalls from research (71→Pitfall1,2; 72→Pitfall5; 73→risk amplification; 74→Pitfall3; 75→Pitfall4)

### Research Flags

**Phases likely needing deeper research during planning:**
- **Phase 71 (Polymarket Auth):** Complex ECDSA implementation; may need standalone research on signing flow
- **Phase 75 (Multi-Venue):** Cross-exchange arbitrage patterns; need API research for routing logic

**Phases with standard patterns (skip research-phase):**
- **Phase 72 (Observability):** Health monitoring is well-documented; standard patterns from production systems
- **Phase 73 (Circuit Breakers):** QuantLabs patterns well-established; can proceed without additional research

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official SDKs (Polymarket US released Jan 2026, pykalshi updated Mar 2026); verified active maintenance |
| Features | HIGH | Based on existing implementation validation + market leader research (AgentBets, PredictEngine, OpenClaw) |
| Architecture | HIGH | Verified against codebase and design spec; build order derived from dependency analysis |
| Pitfalls | HIGH | Based on codebase analysis (CONCERNS.md), documented API requirements, and ecosystem pitfalls |

**Overall confidence:** HIGH

### Gaps to Address

- **Edge calculation implementation:** Need to verify all existing strategies properly implement `estimate_edge()` before Phase 74
- **Historical backtesting data:** No historical market data pipeline yet; needed for Phase 76 (deferred)
- **Multi-exchange routing logic:** Algorithm for best venue selection not specified; needs planning-phase research

---

## Sources

### Primary (HIGH confidence)
- polymarket-us SDK (PyPI, GitHub) — Official SDK with API documentation
- pykalshi GitHub — Production-ready Kalshi SDK with WebSocket support
- Polymarket CLOB API docs — ECDSA authentication requirements
- Albert codebase (ARCHITECTURE.md, CONCERNS.md) — Implementation verification

### Secondary (MEDIUM confidence)
- AgentBets.ai — "Prediction Market Trading Layer: How Agents Execute Trades in 2026"
- SimpleFunctions — Six-layer trading architecture model
- QuantLabs — Circuit breaker patterns for trading bots
- Polymatics — Intelligent trading infrastructure patterns

### Tertiary (LOW confidence)
- Various blog posts on prediction market pitfalls — Need validation during implementation

---

*Research completed: 2026-04-12*
*Ready for roadmap: yes*