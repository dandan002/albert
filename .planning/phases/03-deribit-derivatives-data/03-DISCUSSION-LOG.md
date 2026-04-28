# Phase 3: Deribit Derivatives Data - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-28
**Phase:** 03-deribit-derivatives-data
**Areas discussed:** Instrument selection, Data resolution & types, Historical data handling, CLI invocation pattern

---

## Instrument Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Near-dated options + futures benchmark | Fetch BTC options expiring within a configurable window around each Kalshi contract, plus BTC-PERP and nearest quarterly future for forward price | ✓ |
| All BTC derivatives in range | Fetch all BTC futures and options within the requested window | |
| User-specified instrument list | CLI accepts an explicit list of Deribit instrument names | |

**User's choice:** Near-dated options + futures benchmark
**Notes:** Casts a focused net — enough for Phase 4 probability models without pulling everything.

### Instrument window determination

| Option | Description | Selected |
|--------|-------------|----------|
| All options expiring within N days of window | Configurable window (e.g., 1-7 days) around the date range; Phase 4 selects relevant subset | ✓ |
| Options matching nearest Deribit expiry only | Only fetch options for the closest expiry to each Kalshi window | |

**User's choice:** All options expiring within N days of window
**Notes:** Captures enough strikes for probability estimation; Phase 4 does the filtering.

### Futures benchmark instruments

| Option | Description | Selected |
|--------|-------------|----------|
| BTC perpetual + nearest quarterly future | Forward curve for pricing + perp reference for spot | ✓ |
| BTC perpetual only | Simpler, but only spot reference, no forward price | |
| All BTC futures expiring within window | Complete forward curve but more data than needed | |

**User's choice:** BTC perpetual + nearest quarterly future

---

## Data Resolution & Types

### OHLC candle resolution

| Option | Description | Selected |
|--------|-------------|----------|
| 1-minute candles | Granular enough for precise pre-decision observation; more API calls but stays precise for look-ahead guards | ✓ |
| 5-minute candles | Fewer API calls; ~5-min approximation may not be precise enough | |
| 1-minute + trade-level data | Most comprehensive; maximum data, double the API calls | |

**User's choice:** 1-minute candles

### Trade-level data scope

| Option | Description | Selected |
|--------|-------------|----------|
| OHLC only for now | 1-min candles sufficient for Phase 4; trade-level can be added later | ✓ |
| OHLC + trade-level mark price/IV | Richer data but doubles API calls and normalization work | |

**User's choice:** OHLC only for now
**Notes:** Trade-level data deferred to future phase if Phase 4 needs it.

### Mark/index price source

| Option | Description | Selected |
|--------|-------------|----------|
| Deribit ticker endpoint for mark/index price | Separate fetch for mark and index price from Deribit ticker/book summary | ✓ |
| Derive from OHLC close only | Simpler but introduces inaccuracy | |
| the agent decides | Let planner pick approach | |

**User's choice:** Deribit ticker endpoint for mark/index price

---

## Historical Data Handling

### API routing approach

| Option | Description | Selected |
|--------|-------------|----------|
| Single API with time-range windows | Deribit uses one API with start/end timestamp params; no live/historical split like Kalshi | ✓ |
| Mirror Kalshi routing pattern | Build routing layer for consistency, but adds unnecessary complexity | |
| the agent decides | Let planner decide based on research | |

**User's choice:** Single API with time-range windows

### Rate limit handling

| Option | Description | Selected |
|--------|-------------|----------|
| Conservative retries with delays (Phase 2 pattern) | Bounded retries with configurable delays; consistent with existing client behavior | ✓ |
| Explicit rate-limit tracking per endpoint | More sophisticated; adds complexity Phase 3 doesn't need | |

**User's choice:** Conservative retries with delays, matching Phase 2 pattern

---

## CLI Invocation Pattern

### Command structure

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone date-range + optional Kalshi run reference | Primary: date-range params; optional: --kalshi-run-id reads Kalshi manifest for alignment | ✓ |
| Always requires Kalshi run output | Tighter coupling but less flexible | |
| Fully standalone date-range command | Maximum independence but requires manual alignment | |

**User's choice:** Standalone date-range command + optional Kalshi run reference

### Instrument selection parameters

| Option | Description | Selected |
|--------|-------------|----------|
| Automatic by date range | Discover available BTC options from Deribit, select those expiring within --near-dated-days | ✓ |
| Explicit --instruments flag | Requires knowing Deribit naming conventions | |
| Both automatic + override flag | Flexible but more CLI surface | |

**User's choice:** Automatic by date range
**Notes:** User specifies --start-date, --end-date, and --near-dated-days. No need to know Deribit instrument names.

---

## Missing Data Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Same pattern as Phase 2 | Null/missing fields with quality flags; no instruments recorded in skipped-instruments table | ✓ |
| the agent decides | Let planner decide, as long as no imputation | |

**User's choice:** Same pattern as Phase 2

---

## the agent's Discretion

- Exact module boundaries, class names, helper function names
- Exact retry counts, timeout values, delay durations
- Exact normalized column names
- Default value for --near-dated-days

## Deferred Ideas

None — discussion stayed within Phase 3 scope.