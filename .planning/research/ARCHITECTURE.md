# Prediction Market Trading System Architecture

**Researched:** 2026-04-12
**Domain:** Prediction Market Trading Systems
**Research Mode:** Architecture — Component Boundaries, Data Flow, Build Order

## Executive Summary

Prediction market trading systems require a carefully structured event-driven architecture that separates data ingestion, strategy logic, risk management, order execution, and portfolio tracking into distinct components. The Albert system follows this pattern with five async modules communicating through an event bus. Research reveals that production systems benefit from additional layers: circuit breakers for drawdown protection, position state machines for lifecycle management, and health monitoring for detecting stalled components. The recommended build order follows data flow—from ingestion through strategy to execution—with risk controls interleaved between strategy output and order placement.

## Component Architecture

### Core Components (Current Implementation)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        albert (main process)                         │
│                                                                      │
│  ┌──────────────────┐                      ┌──────────────────┐   │
│  │   Ingestor      │                      │    Event Bus     │   │
│  │   Layer        │─────────────────────▶│  (asyncio.Queue)│   │
│  │                 │    market_data      │                  │   │
│  │  ┌───────────┐  │                     └────────┬────────┘   │
│  │  │  Kalshi   │  │                              │              │
│  │  │ Ingestor  │  │         ┌────────────────────┼────────┐      │
│  │  └───────────┘  │         ▼                ▼        ▼        │
│  │                 │    ┌─────────┐   ┌──────────┐  ┌─────────┐ │
│  │  ┌───────────┐  │    │Strategy│   │Execution│  │Portfolio│ │
│  │  │Polymarket│  │    │Engine  │   │ Engine  │  │ Tracker │ │
│  │  │ Ingestor │  │    └─────────┘   └──────────┘  └─────────┘ │
│  │  └───────────┘  │         │           │            │        │
│  └──────────────────┘         │    order_intents    fills         │
│                               ▼                  ▼                │
│                        ┌─────────────────────────────────────┐     │
│                        │         SQLite Database              │     │
│                        │  markets | orderbook | positions     │     │
│                        │  fills | strategies | daily_pnl      │     │
│                        └─────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With | Boundary Interface |
|-----------|---------------|-------------------|-------------------|
| **KalshiIngestor** | WebSocket connect, normalize to MarketDataEvent | EventBus (publish: market_data) | `_normalize(raw) → MarketDataEvent` |
| **PolymarketIngestor** | WebSocket connect, normalize to MarketDataEvent | EventBus (publish: market_data) | `_normalize(raw) → MarketDataEvent` |
| **StrategyEngine** | Load strategies, evaluate on market_data, emit OrderIntent | Subscribes: market_data; Publishes: order_intents | `on_market_data(event) → list[OrderIntent]` |
| **ExecutionEngine** | Kelly sizing, risk checks, adapter routing, persist fills | Subscribes: order_intents, market_data; Publishes: fills | `execute(intent) → Fill or raises` |
| **PortfolioTracker** | Maintain positions, compute P&L on fills and price updates | Subscribes: fills, market_data | `_update_position(fill)`, `_update_prices(event)` |
| **RiskChecker** | Debounce, daily loss, max notional enforcement | Called by ExecutionEngine | `check(intent) → bool` |
| **EventBus** | Fan-out pub/sub with typed channels | All components | `publish(channel, event)`, `subscribe(channel, queue)` |

### Data Flow Direction

**Primary Flow (Market Data → Signal → Order → Fill → Position)**

```
1. [Ingestor] WebSocket message → _normalize() → MarketDataEvent
2. [Ingestor] bus.publish("market_data", event)
3. [StrategyEngine] bus.subscribe receives MarketDataEvent
4. [StrategyEngine] strategy.on_market_data(event) → list[OrderIntent]
5. [StrategyEngine] bus.publish("order_intents", intent)
6. [ExecutionEngine] bus.subscribe receives OrderIntent
7. [ExecutionEngine] lookup price from cache
8. [ExecutionEngine] kelly_size(edge, ask, bankroll, config)
9. [ExecutionEngine] RiskChecker.check(intent) → bool
10. [ExecutionEngine] adapter.place_order(intent) → Fill
11. [ExecutionEngine] persist fill to DB
12. [ExecutionEngine] bus.publish("fills", fill_event)
13. [PortfolioTracker] bus.subscribe receives FillEvent
14. [PortfolioTracker] create/update/close position in DB
15. [PortfolioTracker] _update_daily_pnl(fill)

Secondary Flow (Price Updates for Mark-to-Market)
16. [Ingestor] bus.publish("market_data", event)
17. [PortfolioTracker] bus.subscribe receives MarketDataEvent
18. [PortfolioTracker] update current_price, unrealized_pnl for positions in market
```

### Build Order Implications

The system components have a natural dependency order that should inform phase planning:

**Tier 1 — Foundation (must build first)**
1. Database layer — all other components depend on persistence
2. EventBus — all async communication flows through it
3. Configuration loader — runtime parameters needed everywhere

**Tier 2 — Data Ingestion (depends on Tier 1)**
4. BaseIngestor + event types — establishes the data contract
5. Individual exchange ingestors (Kalshi, Polymarket) — normalize to typed events
6. Orderbook snapshots — time-series store for price data

**Tier 3 — Strategy (depends on Tier 2)**
7. BaseStrategy + strategy interface — defines contract
8. StrategyEngine — loads and executes strategies
9. Core strategies (MomentumV1, etc.) — initial strategy implementations

**Tier 4 — Execution (depends on Tier 3)**
10. ExchangeAdapter interface + implementations — routing abstraction
11. Kelly sizing module — mathematical position sizing
12. RiskChecker — pre-trade guards (debounce, loss limit, max notional)
13. ExecutionEngine — orchestrates sizing, risk, order placement, fill persistence

**Tier 5 — Tracking (depends on Tier 4)**
14. PortfolioTracker — positions, P&L, daily tracking
15. CLI status command — human-readable output

**Interleaved Risk Controls**
The RiskChecker sits between StrategyEngine (produces OrderIntent) and ExecutionEngine (places orders). This means risk logic can be enhanced independently without modifying strategy or execution code.

## Architectural Patterns from Research

### Pattern 1: Layered Trading Architecture

Based on research into SimpleFunctions and AgentBets stacks, production prediction market systems often organize into six layers:

| Layer | Purpose | In Albert |
|-------|---------|----------|
| 1. Raw Exchange Data | WebSocket streams | Ingestor layer |
| 2. Normalized Data | Typed MarketDataEvent | Ingestor → EventBus |
| 3. Context Enrichment | Adding external data (news, sentiment) | Not implemented (future) |
| 4. Causal Reasoning | Why price should differ from probability | Not implemented (future) |
| 5. Edge Detection | Estimate probability edge | Strategy.estimate_edge() |
| 6. Execution | Place orders, manage risk, track P&L | ExecutionEngine + PortfolioTracker |

**Implication for Roadmap:** The current architecture covers layers 1, 2, 5, and 6. Future phases could add layers 3-4 for enhanced decision-making.

### Pattern 2: Circuit Breaker for Drawdown Protection

Research from QuantLabs and other futures trading bots reveals circuit breakers as essential protection:

**Components to add:**
- `ConsecutiveLossBreaker` — halts trading after N consecutive losses
- `DrawdownBreaker` — halts trading when drawdown exceeds threshold
- `VolumeWindowBreaker` — halts during unusual volume spikes

**Integration point:** Run after RiskChecker but before adapter.place_order()

### Pattern 3: Position State Machine

Production bots track positions through explicit states:

```
┌─────────┐    fill_received     ┌──────────┐    partial_fill    ┌─────────┐
│  OPEN   │ ──────────────────▶ │ PARTIAL │ ────────────────▶ │  OPEN   │
└─────────┘                     └──────────┘                   └─────────┘
     ▲                                                             │
     │                    ┌──────────┐    fully_closed           │
     └───────────────────│  CLOSING  │──────────────────────────────┘
            trade_closed └──────────┘
                          │  CLOSED  │────────────────────────────
                          └──────────┘                           (kept for history)
```

**Current gap:** Albert handles partial fills but not explicit state transitions. The `positions` table tracks avg_entry_price, which implicitly handles partial closes.

### Pattern 4: Health Monitoring Component

Research reveals production systems benefit from a watchdog that monitors all components:

```
HealthMonitor:
  - Ingestor heartbeat (last received timestamp)
  - Execution queue depth
  - Strategy evaluation latency
  - Database connection health
  - Alerts on component stall (no heartbeats > threshold)
```

**Current gap:** No heartbeat monitoring. Ingestor disconnects would only be detected on next publish attempt.

### Pattern 5: Multi-Exchange Routing with Fallback

Research from AgentBets shows multi-platform bots need intelligent routing:

```
┌──────────────┐
│  Market      │
│  Discovery   │
└──────┬───────┘
       ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Check        │    │ Check        │    │ Check        │
│ Polymarket   │───▶│   Kalshi     │───▶│  Fallback    │
│ liquidity    │    │ availability │    │  venue      │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Current gap:** Albert routes by market_id prefix directly to adapter. No liquidity checking before order.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Synchronous Blocking in Async Loop

Never use blocking `time.sleep()` or synchronous database calls in async event handlers. Use `asyncio.sleep()` and async DB operations with aiosqlite.

### Anti-Pattern 2: Shared Mutable State Without Locks

Redis and threading model in research bots use locks for shared state. In asyncio, use `asyncio.Lock()` only when absolutely necessary—prefer message passing through EventBus.

### Anti-Pattern 3: Strategy Directly Calls Exchange API

Strategies should never place trades directly. All execution flows through ExecutionEngine with risk checks. Violating this bypasses risk controls.

### Anti-Pattern 4: Storing Sensitive Credentials in Code

Environment variables only—never commit API keys, private keys, or secrets to repository.

### Anti-Pattern 5: No State Persistence

Never rely on in-memory state alone. Research bots emphasize crash recovery through persistent state. Albert's SQLite is essential.

## Scalability Considerations

| Scale | Architecture Impact |
|-------|-------------------|
| **100 users / day** | Single process sufficient, SQLite with WAL mode |
| **10K daily trades** | Consider connection pooling for DB |
| **100K+ daily trades** | May need streaming DB (RisingWave pattern), separate execution process |
| **Cross-exchange** | Add adapter registry with priority routing |

## Recommended Architecture for Future Phases

### Additional Components to Consider

| Component | Purpose | Add in Phase |
|-----------|---------|-------------|
| HealthMonitor | Watchdog for component heartbeats | 72.x |
| CircuitBreaker | Drawdown + consecutive loss protection | 73.x |
| PositionStateMachine | Explicit position lifecycle states | 73.x |
| MultiExchangeRouter | Route to best venue by liquidity | 74.x |
| ContextEnricher | Add external data to market events | 75.x (intelligence layer) |

### Enhanced Component Boundaries

```
                    ┌──────────────────────────────────┐
                    │         HealthMonitor            │
                    │  (watchdog, all component        │
                    │   heartbeats)                  │
                    └──────────────────────────────────┘
                                    │
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                 ▼           ▼           ▼                 ▼
┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐
│  Kalshi  │  │Polymarket │  │Strategy │  │Execution │  │Portfolio│
│Ingestor  │  │Ingestor  │  │ Engine │  │ Engine   │  │ Tracker │
└──────────┘  └───────────┘  └──────────┘  └───────────┘  └──────────┘
```

## Phase Build Order Recommendations

Based on dependency analysis, here is the optimal phase ordering:

### Phase 71 (Foundation)
- Complete existing core functionality
- Verify all five modules running with test coverage

### Phase 72 (Observability)
- Add HealthMonitor component
- Add structured logging throughout
- Add metrics collection (positions checked, orders placed, fills received)

### Phase 73 (Risk Enhancements)
- Add CircuitBreaker for drawdown protection
- Add ConsecutiveLossBreaker
- Add position state machine for explicit lifecycle tracking
- Integrate circuit breakers between RiskChecker and ExecutionEngine

### Phase 74 (Multi-Venue)
- Add MultiExchangeRouter for liquidity-based routing
- Enhance PolymarketAdapter with proper ECDSA signing
- Add venue health monitoring (detect API degradation)

### Phase 75 (Intelligence Layer)
- Add ContextEnricher for external data integration
- Add thesis engine for causal reasoning on prices
- Add historical backtesting data pipeline

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Component boundaries | HIGH | Verified in codebase ARCHITECTURE.md and design spec |
| Data flow | HIGH | Verified through event type definitions and module signatures |
| Build order | MEDIUM | Derived from dependency analysis; may need adjustment based on phase priorities |
| Anti-patterns | HIGH | Well-established patterns from research sources |
| Scalability | MEDIUM | Research from similar systems; may need tuning for specific workloads |

## Sources

- **Medium: How Prediction Market APIs Work** (2026-01-21) — Architecture layers
- **RisingWave: Real-Time Prediction Market** (2025-12-15) — Streaming database patterns
- **AgentBets.ai: Trading Layer** (2026-03-04) — Polymarket/Kalshi comparison
- **SimpleFunctions: Prediction Market Data Stack** (2026-04-02) — Six-layer model
- **Oracle3: Wang Transform Pricing** (GitHub 2026-02-28) — Multi-exchange architecture
- **QuantLabs: Multi-Asset Futures Bot Suite** (2026-02-24) — Circuit breaker patterns
- **QuantMuse: Trading Engine Architecture** (GitHub) — Threading model reference
- **Chudi.dev: Production Trading Bot** (2026-03-09) — Paper-to-live progression

---

*Architecture research complete: 2026-04-12*