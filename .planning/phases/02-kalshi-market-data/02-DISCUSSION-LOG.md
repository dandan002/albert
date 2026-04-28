# Phase 2: Kalshi Market Data - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-04-28T17:17:38Z
**Phase:** 02-kalshi-market-data
**Areas discussed:** Market Discovery Scope, Endpoint Stitching, Price And Liquidity Data, Normalization And Data Quality

---

## Market Discovery Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Only clean 15-minute BTC contracts | Only Kalshi 15-minute BTC range/event contracts that map cleanly to settlement windows | yes |
| All BTC-related markets | All Kalshi BTC-related markets, then filter later | |
| You decide | Let the planner/implementer choose | |

**User's choice:** Only Kalshi 15-minute BTC range/event contracts that map cleanly to settlement windows.
**Notes:** Discovery should accept explicit `--start-date` and `--end-date`, exclude ambiguous markets with skip logging, and output a normalized market table plus raw cached API pages.

---

## Endpoint Stitching

| Option | Description | Selected |
|--------|-------------|----------|
| Automatic endpoint selection | Choose current versus historical endpoints based on requested date/window and API availability | yes |
| User endpoint flags | Separate user flags like `--historical` versus `--current` | |
| Current only | Start with current endpoints only and defer historical support | |

**User's choice:** Automatic endpoint selection.
**Notes:** Spanning ranges should be split into endpoint-specific windows and stitched. Every source request/page should be cached separately with endpoint and parameters in the manifest. Retries should be bounded with timeouts and clear failures.

---

## Price And Liquidity Data

| Option | Description | Selected |
|--------|-------------|----------|
| Full relevant price surfaces | Trades, bid/ask pricing data or candlesticks, and order-book snapshots where available | yes |
| Trades and candlesticks only | Fetch trades and candlesticks but skip order-book snapshots | |
| Minimal metadata/outcomes | Fetch only market metadata and settlement outcomes now | |

**User's choice:** Full relevant price surfaces.
**Notes:** Thin or missing liquidity should be represented with explicit null/missing fields and quality flags, without imputation. Preserve source-native granularity. Non-negotiable fields for later backtests include bid, ask, trade price/size, timestamp, market ticker, and liquidity/depth where available.

---

## Normalization And Data Quality

| Option | Description | Selected |
|--------|-------------|----------|
| Separate normalized tables | Markets, market price observations, order-book observations, trades, and outcomes | yes |
| One denormalized table | One denormalized table per market | |
| Markets and outcomes only | Markets and outcomes now, price detail later | |

**User's choice:** Separate normalized tables.
**Notes:** Unavailable settlement outcomes should keep the market row with an `outcome_missing` quality flag. Normalized schemas should validate required fields strictly while preserving unknown raw fields only in raw cache. Data-quality results should be visible in CLI summary plus manifest/output metadata and covered by tests.

---

## the agent's Discretion

- Exact module names and helper boundaries.
- Exact retry counts and timeout values.
- Exact wording of CLI data-quality summaries.
- Exact normalized column names, provided required semantics are preserved.

## Deferred Ideas

None.
