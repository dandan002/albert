# Feature Landscape: Prediction Market Trading Systems

**Domain:** Automated trading bots for prediction markets (Polymarket, Kalshi)
**Researched:** 2026-04-12
**Project Context:** Albert Trading System — improving Polymarket integration and expanding capabilities
**Overall Confidence:** HIGH

---

## Executive Summary

Prediction market trading systems in 2026 have evolved beyond simple bet размещение. The ecosystem now expects institutional-grade infrastructure: real-time WebSocket data, sophisticated position sizing, multi-exchange support, and advanced risk management. Albert's current foundation (WebSocket ingestion, Kelly sizing, risk checks, portfolio tracking) covers table stakes, but competitive differentiation requires expanded strategy library, backtesting, cross-platform arbitrage, and advanced order types.

This research categorizes features into three tiers:

- **Table Stakes:** Features users expect; missing = product feels incomplete or unusable
- **Differentiators:** Features that set the product apart; not expected but highly valued
- **Anti-Features:** Deliberate omissions that preserve focus and avoid scope creep

---

## Table Stakes

Features that users expect in any prediction market trading system. Missing these makes the product feel fundamentally incomplete.

### Real-Time Market Data Ingestion

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Prediction markets move on news, data releases, and events — seconds matter. Stale data means trading at outdated prices. |
| **Complexity** | Medium |
| **Notes** | Requires WebSocket connections to exchanges (Polymarket CLOB, Kalshi). Must handle reconnection, message parsing, and normalization. Albert already has this. |

**Dependencies:** Exchange API credentials, WebSocket infrastructure, message schema normalization

---

### Order Execution with Risk Checks

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Users won't trust a bot that places unlimited orders. Pre-trade risk checks are the minimum safeguard. |
| **Complexity** | Low |
| **Notes** | Includes: max position per market, max total notional, daily loss limit, duplicate order debounce. Albert already has this. |

**Dependencies:** Exchange adapters, position tracking, config-based limits

---

### Kelly Criterion Position Sizing

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Mathematically optimal for edge-based betting; prevents overbetting on uncertain outcomes. Expected by sophisticated users. |
| **Complexity** | Medium |
| **Notes** | Formula: `f* = (edge * b - (1 - edge)) / b`. Requires edge estimation from strategy. Albert already has this. |

**Dependencies:** Strategy edge estimation, bankroll tracking, Kelly fraction config

---

### Portfolio & P&L Tracking

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Users need to know their exposure, realized/unrealized P&L, and returns per strategy. Without this, trading is blind. |
| **Complexity** | Low |
| **Notes** | Must track positions, compute mark-to-market P&L, record realized P&L on close. Albert already has this. |

**Dependencies:** Fill events, market data price updates, position database

---

### Multi-Exchange Support (Polymarket + Kalshi)

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Single-exchange bots are limited — you miss cross-platform arbitrage and have lower signal diversity. |
| **Complexity** | Medium |
| **Notes** | Requires separate adapters with normalized interfaces. Polymarket integration is the active improvement area for Albert. |

**Dependencies:** ExchangeAdapter interface, API credentials for each exchange, normalized data schemas

---

### Strategy Hot-Reload

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Markets change; strategies need tuning without restarting the bot. This is standard for production trading systems. |
| **Complexity** | Low |
| **Notes** | Poll strategy config from database; apply changes on next evaluation cycle. Albert already has this. |

**Dependencies:** SQLite strategy registry, config polling mechanism

---

### Structured Logging & CLI Status

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Users need to inspect bot state, diagnose issues, and verify it's running. |
| **Complexity** | Low |
| **Notes** | JSON logging with context (strategy_id, market_id, timestamp). CLI status command for snapshot. Albert already has this. |

**Dependencies:** Logging configuration, CLI entry point

---

## Differentiators

Features that set products apart. Not expected by average users, but highly valued and create competitive advantage.

### Expanded Strategy Library

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Momentum** | Trade on price velocity and order flow patterns | Medium | Identifies markets where price is trending with volume confirmation |
| **Mean Reversion** | Fade extreme prices expecting return to equilibrium | Medium | Works on binary markets with wide spreads |
| **Sentiment Analysis** | Aggregate news/social signals for directional conviction | High | Requires NLP pipeline, news API integration |
| **Arbitrage (Cross-Platform)** | Exploit price discrepancies between Polymarket and Kalshi | High | Simultaneous leg execution, partial fill handling |
| **Whale Tracking** | Follow smart money wallet activity | Medium | Requires on-chain wallet analytics |

**Dependencies:** Each strategy needs: strategy class implementation, config schema, edge estimation logic

---

### Backtesting Engine

| Attribute | Details |
|-----------|---------|
| **Value Proposition** | Users want to validate strategies on historical data before risking capital. This is the #1 requested feature for serious traders. |
| **Complexity** | High |
| **Notes** | Requires: historical market data storage, simulation engine, performance metrics (Sharpe, drawdown, win rate). |

**Dependencies:** Historical orderbook data, strategy replay mechanism, performance analytics

---

### Advanced Order Types (TWAP, VWAP, Iceberg)

| Attribute | Details |
|-----------|---------|
| **Value Proposition** | Large orders move markets. Institutional traders expect algo execution to minimize slippage. |
| **Complexity** | High |
| **Notes** | Not natively supported by Polymarket (limit orders only), must be implemented client-side. |

**Dependencies:** Order splitting logic, execution scheduler, price impact estimation

---

### Paper Trading / Simulation Mode

| Attribute | Details |
|-----------|---------|
| **Value Proposition** | Users want to test strategies with fake money before committing capital. 14-day trials are standard in commercial bots. |
| **Complexity** | Medium |
| **Notes** | Simulates fills at current market price without actual order placement. |

**Dependencies:** Separate execution path (mock adapter), virtual portfolio tracking

---

### Cross-Market Correlation Analysis

| Attribute | Details |
|-----------|---------|
| **Value Proposition** | Hedged positions across correlated markets reduce risk. Advanced traders actively seek this. |
| **Complexity** | High |
| **Notes** | Requires correlation data, position aggregation across markets, hedging logic. |

**Dependencies:** Market relationship database, portfolio-level risk metrics

---

### Whale Alerts & Smart Money Detection

| Attribute | Details |
|-----------|---------|
| **Value Proposition** | Detecting large wallet activity provides early signal on market direction. Commercial platforms charge for this. |
| **Complexity** | Medium |
| **Notes** | Requires: on-chain wallet tracking, large trade detection, alert system. |

**Dependencies:** On-chain data API (e.g., Falcon API, Dune Analytics), alert webhooks

---

### Web Dashboard

| Attribute | Details |
|-----------|---------|
| **Value Proposition** | Visual portfolio analytics, trade history, strategy performance charts. Expected for any modern trading tool. |
| **Complexity** | Medium |
| **Notes** | Initially out of scope for Albert v1, but needed for competitive parity. |

**Dependencies:** REST API endpoints for data, frontend framework, authentication

---

### Multi-Account Management

| Attribute | Details |
|-----------|---------|
| **Value Proposition** | Manage multiple exchange accounts (different wallets, strategies) from one dashboard. Enterprise feature. |
| **Complexity** | High |
| **Notes** | Out of scope for v1; adds significant complexity to position tracking and risk management. |

**Dependencies:** Account registry, separate position/portfolio per account, aggregated views

---

## Anti-Features

Features to deliberately NOT build. These either distract from core value, add unnecessary complexity, or are better handled externally.

### Real-Time News Sentiment (Full NLP Pipeline)

| Why Avoid | What to Do Instead |
|-----------|-------------------|
| Building a news sentiment pipeline is a product in itself. Requires constant model retraining, reliable news source APIs, and sophisticated NLP. | Integrate with external sentiment APIs (e.g., Falcon API social sentiment, or use PredictEngine's sentiment module). Keep the strategy layer focused on price/action signals. |

---

### Market Creation / Curation

| Why Avoid | What to Do Instead |
|-----------|-------------------|
| Market creation is a platform feature (Polymarket handles this). Bots are consumers, not creators. | Focus on trading existing markets. Market discovery via Gamma API is sufficient. |

---

### Automated Settlement & Oracle Integration

| Why Avoid | What to Do Instead |
|-----------|-------------------|
| Settlement is handled by the exchange (Polymarket uses UMA oracle). No bot involvement needed. | Monitor resolution via market status API. Trigger position close after resolution if desired. |

---

### Full Backtesting with Regime Detection

| Why Avoid | What to Do Instead |
|-----------|-------------------|
| Regime detection (bull/bear, high/low volatility) requires sophisticated ML and degrades to overfitting. | Keep backtesting simple: historical price replay. Don't try to identify market regimes automatically. |

---

### Decentralized Exchange (DEX) Integration Beyond Polymarket

| Why Avoid | What to Do Instead |
|-----------|-------------------|
| Polymarket is the dominant prediction market. Adding moreDEXs (e.g., Azuro, Omen) fragments focus and adds maintenance burden. | Deepen Polymarket integration first. Add Kalshi as second exchange. Don't chase marginal platforms. |

---

## Feature Dependencies

The following dependency graph determines implementation order:

```
┌─────────────────────────────────────────────────────────────────┐
│                        TABLE STAKES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Real-Time Data Ingestion ───────┐                              │
│         │                        │                              │
│         ▼                        │                              │
│  Strategy Engine ────────────────┼──────────────────────────┐    │
│         │                        │                          │    │
│         ▼                        │                          │    │
│  Kelly Sizing + Risk Checks ────┼──────────────────────────┼─┐  │
│         │                        │                          │ │  │
│         ▼                        │                          ▼ ▼  │
│  Order Execution ───────────────┴───▶ Portfolio Tracking ◀─┘ │  │
│                                              │                │
└──────────────────────────────────────────────┼────────────────┘
                                               │
                   ┌───────────────────────────┼───────────────────┐
                   │                           │                   │
                   ▼                           ▼                   ▼
┌─────────────────────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     DIFFERENTIATORS            │  │ DIFFERENTIATORS │  │ DIFFERENTIATORS │
├─────────────────────────────────┤  ├─────────────────┤  ├─────────────────┤
│                                 │  │                 │  │                 │
│  Expanded Strategy Library      │  │  Backtesting   │  │  Paper Trading │
│  - Momentum                    │  │  Engine         │  │  Mode          │
│  - Mean Reversion               │  │                 │  │                 │
│  - Arbitrage                    │  │  Dependencies:  │  │  Dependencies:  │
│  - Sentiment (external API)     │  │  Historical    │  │  Mock          │
│                                 │  │  data store    │  │  adapter       │
│  Dependencies:                  │  │  + replay      │  │  + virtual     │
│  - New strategy classes         │  │  engine        │  │  portfolio     │
│  - Config schemas               │  │                │  │                │
│                                 │  │                │  │                │
└─────────────────────────────────┘  └─────────────────┘  └─────────────────┘
```

### Critical Path to Differentiators

1. **First:** Complete Polymarket integration improvements (current active work)
2. **Second:** Paper trading mode — enables user onboarding without risk
3. **Third:** Expanded strategy library — provides variety for different market conditions
4. **Fourth:** Backtesting — validates strategies before live deployment

---

## MVP Recommendation

For Albert's next phase (Polymarket improvement + trading expansion), prioritize:

### Must Have (Table Stakes — already exists)

- Real-time WebSocket ingestion (Kalshi + Polymarket)
- Kelly criterion position sizing
- Pre-trade risk checks
- Portfolio P&L tracking
- Strategy hot-reload
- CLI status output

### Prioritize for Next Phase

1. **Paper Trading Mode** (differentiator) — Low complexity, high value for user onboarding
2. **Strategy Library Expansion** (differentiator) — Add momentum and mean reversion to existing engine
3. **Polymarket Authentication Fix** (table stakes) — Current issue blocking production use

### Defer (Out of Scope)

| Feature | Reason |
|---------|--------|
| Backtesting | High complexity; requires historical data pipeline |
| Web Dashboard | Initially out of scope per PROJECT.md |
| Multi-Account | Adds complexity; single-account focus for v1 |
| Advanced Order Types (TWAP/VWAP) | Requires large position sizes to benefit; rare edge case |
| Whale Tracking | External API dependency; can integrate later |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Table Stakes | HIGH | Based on Albert's existing implementation + design spec validation |
| Differentiators | HIGH | Researched via PredictEngine, Polymatics, OpenClaw, AgentBets — market leaders |
| Anti-Features | MEDIUM | Derived from ecosystem analysis and out-of-scope decisions in PROJECT.md |
| Dependencies | HIGH | Based on architecture in design spec and feature interdependencies |

---

## Sources

- AgentBets.ai — "Prediction Market Trading Layer: How Agents Execute Trades in 2026" (2026-03-04)
- TRUEiGTECH — "Top Prediction Market Software Features 2026" (2025-12-11)
- OpenClaw — "OpenClaw + Polymarket: The Complete Prediction Market Trading Guide" (2026-02-17)
- PredictEngine — "PredictEngine Pro: Advanced Features for Serious Traders" (2026-02-28)
- Polymatics — "Intelligent Trading Infrastructure for Prediction Markets" (polymatics.info)
- Polymarket Documentation — "API Reference", "Quickstart", "Orders API Overview" (2026)
- PolyCatalog — "Polymarket API Review (2026)" (2026-03-26)
- PRED Scanner — "Mastering Polymarket: A Comprehensive API Guide for Active Traders in 2026" (2026-02-26)
- NewsPoly — "Polymarket 2026: Complete Guide and Platform Outlook" (2026-03-30)
