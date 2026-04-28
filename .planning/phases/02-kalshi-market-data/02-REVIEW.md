---
phase: 02-kalshi-market-data
reviewed: 2026-04-28T20:04:59Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - src/albert/kalshi/client.py
  - src/albert/kalshi/routing.py
  - src/albert/kalshi/schemas.py
  - src/albert/kalshi/normalize.py
  - src/albert/kalshi/quality.py
  - src/albert/kalshi/pipeline.py
  - src/albert/commands/ingest.py
  - src/albert/cache.py
  - src/albert/manifest.py
  - tests/test_kalshi_client.py
  - tests/test_kalshi_routing.py
  - tests/test_kalshi_normalize.py
  - tests/test_kalshi_discovery_cli.py
  - tests/test_kalshi_cache_manifest.py
findings:
  critical: 0
  warning: 4
  info: 1
  total: 5
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-04-28T20:04:59Z
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Reviewed the Kalshi client, routing, normalization, discovery pipeline, cache/manifest helpers, CLI wiring, and fixture-backed tests. The path and manifest helpers are generally conservative, and focused Kalshi verification passed locally. The main risks are correctness and data completeness issues in pagination, URL construction for source-derived tickers, and normalized Parquet schema stability.

Verification run during review:

```bash
python -m pytest tests/test_kalshi_client.py tests/test_kalshi_routing.py tests/test_kalshi_normalize.py tests/test_kalshi_discovery_cli.py tests/test_kalshi_cache_manifest.py -q
# 21 passed in 0.25s

python -m ruff check src/albert/kalshi src/albert/commands/ingest.py src/albert/cache.py src/albert/manifest.py tests/test_kalshi_client.py tests/test_kalshi_routing.py tests/test_kalshi_normalize.py tests/test_kalshi_discovery_cli.py tests/test_kalshi_cache_manifest.py
# All checks passed
```

## Warnings

### WR-01: Cursor pagination drops falsey but valid cursor tokens

**File:** `src/albert/kalshi/client.py:106`

**Issue:** `iter_cursor_pages()` only adds the cursor parameter when `if cursor:` is truthy. A cursor token such as `0` or `"0"` can be valid in paginated APIs but would be skipped, causing the client to refetch the first page or stop making forward progress depending on the API response.

**Fix:**

```python
if cursor is not None and cursor != "":
    page_parameters["cursor"] = cursor
```

Add a test with `cursor=0` or `"0"` to lock the behavior.

### WR-02: Trade ingestion fetches only one page despite cursor-capable payloads

**File:** `src/albert/kalshi/pipeline.py:273`

**Issue:** `_fetch_trades()` calls `get_json_page()` once per historical/live trade window and never follows `cursor`. The live trade fixture already contains `"cursor": "live-trades-page-2"`, so real discovery runs can silently miss all subsequent trade pages. That directly affects trade completeness and any downstream liquidity or execution analysis.

**Fix:** Use cursor iteration for trades the same way market discovery does.

```python
for page in client.iter_cursor_pages(
    endpoint_segment=window.endpoint_segment,
    path=window.path,
    parameters=_window_parameters(window, {"series_ticker": series_ticker, "limit": 1000}),
    collection_key="trades",
    covered_start=window.start,
    covered_end=window.end,
):
    pages.append(page)
    rows.extend(
        normalize_trades_page(
            page.payload,
            source_endpoint=page.endpoint_segment,
            source_raw_path=page.raw_path,
        )
    )
```

### WR-03: Source-derived ticker is interpolated into request paths without escaping

**File:** `src/albert/kalshi/routing.py:128`

**Issue:** `plan_candlestick_windows()` builds URL paths with `f"/markets/{ticker}/candlesticks"`. The ticker comes from source payloads before being used for another request. If a malformed or unexpected ticker contains `/`, `?`, or other reserved URL characters, the client can request a different path than intended while still using the trusted Kalshi host. This weakens endpoint provenance and path safety.

**Fix:** URL-encode dynamic path segments before interpolation.

```python
from urllib.parse import quote

safe_ticker = quote(ticker, safe="")
historical_path = f"/historical/markets/{safe_ticker}/candlesticks"
live_path = f"/markets/{safe_ticker}/candlesticks"
```

Add a routing test for a ticker containing `/` or `?` to verify the path remains a single segment.

### WR-04: Empty normalized tables are written without schema

**File:** `src/albert/kalshi/pipeline.py:346`

**Issue:** `pl.DataFrame([_row_dict(row) for row in rows]).write_parquet(path)` produces a zero-column Parquet file when `rows` is empty. Empty trade, skipped-market, or orderbook outputs are plausible in real windows; downstream readers expecting the Phase 2 table contract will not see the expected columns.

**Fix:** Define per-table Polars schemas or derive expected columns from the Pydantic models, and use them when `rows` is empty.

```python
records = [_row_dict(row) for row in rows]
frame = pl.DataFrame(records) if records else pl.DataFrame(schema=TABLE_SCHEMAS[table_name])
frame.write_parquet(path)
```

Add a test that runs discovery with an empty table and asserts the Parquet file has the expected columns.

## Info

### IN-01: Pipeline test double masks trade pagination bug

**File:** `tests/test_kalshi_cache_manifest.py:89`

**Issue:** `FixtureKalshiClient.iter_cursor_pages()` only supports market pagination, while the trade path in the fixture payload has a non-null cursor. Because `_fetch_trades()` uses `get_json_page()`, the current integration test cannot detect missed trade pages.

**Fix:** Extend the fake client to support cursor iteration for trades and assert that multiple trade pages are represented in manifest sources and normalized output row counts.

---

_Reviewed: 2026-04-28T20:04:59Z_
_Reviewer: Codex (gsd-code-reviewer)_
_Depth: standard_
