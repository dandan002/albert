---
phase: 02-kalshi-market-data
plan: 01
subsystem: data-ingestion
tags: [kalshi, httpx, cache, routing, provenance]

requires:
  - phase: 01-research-cli-foundation
    provides: cache path helpers, manifest metadata models, pytest and ruff verification
  - phase: 02-kalshi-market-data
    provides: Wave 0 Kalshi client and routing contract tests
provides:
  - Kalshi HTTP client with bounded retries and raw JSON cache writes
  - Cursor pagination preserving source request filters
  - Historical/live endpoint route planning for markets, candlesticks, and trades
affects: [kalshi-market-data, data-ingestion, normalization, cli]

tech-stack:
  added: []
  patterns:
    - httpx-backed source clients write every raw page through raw_cache_path
    - endpoint routing is represented by frozen source-preserving dataclasses

key-files:
  created:
    - src/albert/kalshi/client.py
    - src/albert/kalshi/routing.py
  modified:
    - tests/test_kalshi_client.py

key-decisions:
  - "Keep the Phase 2 Kalshi client public-data only with no authentication headers or API key options."
  - "Route market and candlestick windows by market_settled_ts, and trade windows by trades_created_ts."

patterns-established:
  - "Raw Kalshi cache writes validate endpoint segments against an allowlist before calling raw_cache_path."
  - "EndpointWindow carries endpoint segment, path, covered range, and parameter name metadata for later manifests."

requirements-completed: [DATA-01, DATA-02]

duration: 4min
completed: 2026-04-28
---

# Phase 02 Plan 01: Kalshi Client And Routing Summary

**Kalshi raw page retrieval and historical/live endpoint routing with deterministic cache provenance**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-28T19:36:25Z
- **Completed:** 2026-04-28T19:40:05Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `FetchedPage` and `KalshiClient` for bounded HTTP fetching, retryable status handling, cursor pagination, and sorted raw JSON cache writes.
- Added endpoint segment allowlisting so Kalshi raw cache directories cannot be chosen by arbitrary external input.
- Added cutoff-aware routing for market, candlestick, and trade request windows with source-preserving `EndpointWindow` metadata.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement HTTP client, raw cache writes, and cursor pagination** - `b89f77f` (feat)
2. **Task 2: Implement historical cutoff route planning** - `9f8fc74` (feat)

## Files Created/Modified

- `src/albert/kalshi/client.py` - HTTPX-backed Kalshi client, retry loop, cursor iterator, endpoint allowlist, and raw page cache writes.
- `src/albert/kalshi/routing.py` - Historical cutoff parser and endpoint window planners for markets, candlesticks, and trades.
- `tests/test_kalshi_client.py` - Stabilized the `_dt()` helper so 24-hour covered windows are valid test inputs.

## Decisions Made

- Kept the client unauthenticated and public-data only, matching the Phase 2 threat model.
- Used the existing `raw_cache_path()` and `sanitize_segment()` helpers rather than constructing any cache paths inside the client.
- Split market/candlestick windows on `market_settled_ts` and trade windows on `trades_created_ts`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stabilized client test datetime helper**
- **Found during:** Task 1 (Implement HTTP client, raw cache writes, and cursor pagination)
- **Issue:** Two existing Wave 0 client tests called `_dt(24)`, which raised before the new client code could be exercised.
- **Fix:** Changed the helper to add timedeltas from midnight so 24-hour covered windows are valid.
- **Files modified:** `tests/test_kalshi_client.py`
- **Verification:** `python -m pytest tests/test_kalshi_client.py -q`
- **Committed in:** `b89f77f`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The fix was limited to a test helper needed for the planned verification. No product scope changed.

## Issues Encountered

- Ruff flagged one unused import left from the client implementation. Removed it and reran plan-level verification successfully.

## Known Stubs

None - placeholder scan found only intentional test empty-list payloads and optional `None` defaults.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Client and routing contracts are ready for downstream normalization and CLI orchestration. Raw page provenance now includes endpoint, endpoint segment, request parameters, fetch time, covered range, and raw cache path.

## Verification

- `python -m pytest tests/test_kalshi_client.py -q` - passed
- `python -m pytest tests/test_kalshi_routing.py -q` - passed
- `python -m pytest tests/test_kalshi_client.py tests/test_kalshi_routing.py -q` - passed
- `python -m ruff check src tests` - passed

## Self-Check: PASSED

- Found `src/albert/kalshi/client.py`
- Found `src/albert/kalshi/routing.py`
- Found `.planning/phases/02-kalshi-market-data/02-01-SUMMARY.md`
- Found task commit `b89f77f`
- Found task commit `9f8fc74`

---
*Phase: 02-kalshi-market-data*
*Completed: 2026-04-28*
