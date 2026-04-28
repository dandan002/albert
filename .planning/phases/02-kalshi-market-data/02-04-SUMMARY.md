---
phase: 02-kalshi-market-data
plan: 04
subsystem: testing
tags: [kalshi, fixtures, json, market-data, orderbooks]

requires:
  - phase: 01-research-cli-foundation
    provides: deterministic pytest fixture conventions and local verification commands
provides:
  - Fixture payloads for live and historical KXBTC15M market discovery
  - Fixture payloads for live and historical Kalshi candlesticks and trades
  - Current orderbook snapshots with YES/NO bids and implied ask expectations
  - Edge-case fixtures for missing outcomes, missing history, and ambiguous BTC mappings
affects: [02-kalshi-market-data, DATA-01, DATA-02, kalshi-normalization]

tech-stack:
  added: []
  patterns:
    - Source-like JSON fixtures under tests/fixtures/kalshi
    - Explicit quality_expectations blocks for missing or skipped source observations

key-files:
  created:
    - tests/fixtures/kalshi/live_markets_page.json
    - tests/fixtures/kalshi/historical_markets_page.json
    - tests/fixtures/kalshi/missing_outcome_market.json
    - tests/fixtures/kalshi/missing_history_market.json
    - tests/fixtures/kalshi/ambiguous_btc_market.json
    - tests/fixtures/kalshi/live_candlesticks.json
    - tests/fixtures/kalshi/historical_candlesticks.json
    - tests/fixtures/kalshi/live_trades_page.json
    - tests/fixtures/kalshi/historical_trades_page.json
    - tests/fixtures/kalshi/orderbooks.json
  modified: []

key-decisions:
  - "Represent missing outcomes and missing history as explicit fixture quality expectations instead of dropping or imputing rows."
  - "Represent current orderbook data as YES and NO bid arrays with expected implied ask prices, without synthetic historical depth."

patterns-established:
  - "Kalshi fixture pages mirror endpoint families with live/historical pairs so schema drift can be tested deterministically."
  - "Skipped BTC mappings carry skip_reason so normalizers can write kalshi_skipped_markets rows."

requirements-completed: [DATA-01, DATA-02]

duration: 3min
completed: 2026-04-28
---

# Phase 02 Plan 04: Kalshi Fixture Payload Foundation Summary

**Deterministic Kalshi JSON fixtures for live/historical markets, prices, trades, orderbooks, settlement outcomes, missing data, and skipped BTC mappings**

## Performance

- **Duration:** 3min
- **Started:** 2026-04-28T19:26:14Z
- **Completed:** 2026-04-28T19:29:14Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments

- Added eligible `KXBTC15M` live and historical market discovery payloads with event, market, timing, status, and settlement fields.
- Added edge-case fixtures for missing settlement outcome fields, missing price/orderbook history, and an ambiguous non-15-minute BTC mapping with `skip_reason`.
- Added live/historical candlestick and trade payloads plus current orderbook snapshots containing YES/NO bid arrays and expected implied ask values.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create market and settlement fixtures** - `cedde87` (feat)
2. **Task 2: Create price, trade, and orderbook fixtures** - `e7f5966` (feat)

**Plan metadata:** recorded in the final docs commit for this summary.

## Files Created/Modified

- `tests/fixtures/kalshi/live_markets_page.json` - Live `KXBTC15M` market discovery page with open 15-minute contracts.
- `tests/fixtures/kalshi/historical_markets_page.json` - Historical settled `KXBTC15M` market page with observed settlement outcome fields.
- `tests/fixtures/kalshi/missing_outcome_market.json` - Closed market fixture with outcome fields omitted and `outcome_missing` expectation.
- `tests/fixtures/kalshi/missing_history_market.json` - Settled market fixture that records missing price history and orderbook availability.
- `tests/fixtures/kalshi/ambiguous_btc_market.json` - Excluded BTC mapping fixture with explicit `skip_reason`.
- `tests/fixtures/kalshi/live_candlesticks.json` - Live candlestick payload using `_dollars` price fields.
- `tests/fixtures/kalshi/historical_candlesticks.json` - Historical candlestick payload using non-suffixed fixed-point string fields.
- `tests/fixtures/kalshi/live_trades_page.json` - Live trade page with trade identifiers, prices, counts, and taker sides.
- `tests/fixtures/kalshi/historical_trades_page.json` - Historical trade page with fixed-point count strings.
- `tests/fixtures/kalshi/orderbooks.json` - Current orderbook snapshots with YES/NO bids and derived ask expectations.

## Decisions Made

- Fixtures encode quality expectations directly where downstream normalizers need explicit missing-data behavior.
- Historical orderbook depth remains unavailable in fixtures; the plan intentionally avoids inventing historical depth.

## Deviations from Plan

None - plan executed exactly as written.

**Total deviations:** 0 auto-fixed.
**Impact on plan:** No scope changes.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None.

## Next Phase Readiness

Plan 02-05 can use these fixtures to create contract tests for Kalshi client, routing, normalization, CLI, cache, manifest, and quality behavior without live Kalshi API calls.

## Self-Check: PASSED

- Verified all 10 fixture files and this summary exist.
- Verified task commits `cedde87` and `e7f5966` exist in git history.
- Re-ran focused JSON checks for live markets, historical markets, live candlesticks, and orderbooks.

---
*Phase: 02-kalshi-market-data*
*Completed: 2026-04-28*
