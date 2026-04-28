---
status: partial
phase: 02-kalshi-market-data
source: [02-VERIFICATION.md]
started: 2026-04-28T20:43:13Z
updated: 2026-04-28T20:43:13Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Live Kalshi smoke test

expected: Running `albert ingest kalshi discover --start-date 2026-04-01 --end-date 2026-04-02 --series-ticker KXBTC15M --run-id kalshi-smoke-20260401` completes successfully, writes raw cache files and all 7 normalized Parquet tables with correct schemas, records manifest sources for all fetched pages (including paginated trades), and reports quality counts.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps