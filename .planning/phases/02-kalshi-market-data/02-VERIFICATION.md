---
phase: 02-kalshi-market-data
verified: 2026-04-28T20:31:00Z
status: gaps_found
score: 9/11 must-haves verified
overrides_applied: 0
gaps:
  - truth: "User can fetch metadata, trades, bid/ask pricing data, order-book data where available, and settlement outcomes."
    status: partial
    reason: "Trade retrieval only fetches one page per endpoint window and ignores non-empty cursor values, so paginated Kalshi trade data can be silently incomplete."
    artifacts:
      - path: "src/albert/kalshi/pipeline.py"
        issue: "_fetch_trades() calls get_json_page() once per window at lines 273-291 instead of iter_cursor_pages(..., collection_key=\"trades\")."
      - path: "tests/test_kalshi_cache_manifest.py"
        issue: "FixtureKalshiClient.iter_cursor_pages() asserts collection_key == \"markets\", so the integration test cannot catch missing trade pagination."
    missing:
      - "Fetch trades through KalshiClient.iter_cursor_pages(..., collection_key=\"trades\") and append every returned FetchedPage to manifest sources."
      - "Extend the fixture client/test to include a multi-page trade payload and assert both pages contribute rows and manifest sources."
  - truth: "Normalized Kalshi tables include market ticker, event/series identifiers, open/close/settlement times, prices, liquidity fields, and outcome."
    status: partial
    reason: "Empty normalized outputs are written from pl.DataFrame([]), producing zero-column Parquet files instead of the declared table schema."
    artifacts:
      - path: "src/albert/kalshi/pipeline.py"
        issue: "_write_normalized_outputs() writes pl.DataFrame([_row_dict(row) for row in rows]) at line 346 without a schema for empty row sets."
    missing:
      - "Define per-table output schemas from the Pydantic row contracts or explicit Polars schemas."
      - "Write empty tables with their expected columns and add a regression test for an empty trade/orderbook/skipped table."
---

# Phase 2: Kalshi Market Data Verification Report

**Phase Goal:** Kalshi 15-minute Bitcoin markets can be discovered, fetched, cached, normalized, and linked to outcomes.  
**Verified:** 2026-04-28T20:31:00Z  
**Status:** gaps_found  
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Phase 2 runtime dependencies and package namespace exist. | VERIFIED | `pyproject.toml` declares `httpx`, `polars`, and `pyarrow`; `src/albert/kalshi/__init__.py` exists; `python -c 'import httpx, polars, pyarrow, albert.kalshi'` exited 0. |
| 2 | Fixture payloads cover eligible KXBTC15M markets, live/historical shapes, missing data, orderbooks, and skipped ambiguous mappings without live API access. | VERIFIED | Fixture files exist and are consumed by normalization/manifest tests; artifact verification passed for Plan 02-04. |
| 3 | Kalshi source pages can be fetched with bounded retries and raw cache writes through safe helpers. | VERIFIED | `KalshiClient.get_json_page()` uses retry statuses and `raw_cache_path()`; focused client tests passed. |
| 4 | Cursor pagination exists for supported client calls and preserves request filters. | VERIFIED with warning | `KalshiClient.iter_cursor_pages()` exists and market pagination tests pass; warning remains for falsey numeric cursor tokens because line 106 checks `if cursor:`. |
| 5 | Older data can be routed through historical endpoints when required. | VERIFIED | `plan_market_windows()`, `plan_candlestick_windows()`, and `plan_trade_windows()` split by `market_settled_ts` or `trades_created_ts`; routing tests passed. |
| 6 | Eligible Kalshi 15-minute BTC markets normalize into strict market/outcome rows and ambiguous mappings are skipped. | VERIFIED | `normalize_markets_page()` validates rows with Pydantic models and tests cover eligible, missing-outcome, and skipped-market fixtures. |
| 7 | Price, trade, and orderbook observations preserve source granularity, provenance, and nullable liquidity fields. | VERIFIED | Normalizer tests cover candlesticks, trades, orderbooks, Decimal fields, endpoint provenance, and implied asks. |
| 8 | User can run `albert ingest kalshi discover` for an explicit UTC date range. | VERIFIED | `src/albert/commands/ingest.py` registers `ingest kalshi discover`, validates dates, and CLI test passes. |
| 9 | Discovery writes raw page cache files, normalized Parquet outputs, quality rows, and a run manifest. | VERIFIED | `run_kalshi_discovery()` writes `RunManifest` sources/outputs through cache helpers; manifest integration test passed. |
| 10 | User can fetch all available trade data for requested windows. | FAILED | `_fetch_trades()` calls `get_json_page()` once per window and does not follow `cursor`; targeted check showed pages returned `cursor='page-2'` while `iter_cursor_pages` was never called. |
| 11 | Normalized Parquet tables retain declared columns even when a table has no rows. | FAILED | Targeted check of `_write_normalized_outputs()` with empty rows produced `columns []` for `kalshi_trades`. |

**Score:** 9/11 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | Phase 2 runtime dependencies | VERIFIED | Contains `httpx>=0.28.1`, `polars>=1.40.1`, and `pyarrow>=24.0.0`. |
| `src/albert/kalshi/__init__.py` | Kalshi package marker | VERIFIED | Exists with `__all__`. |
| `src/albert/kalshi/client.py` | HTTP client, retries, raw cache writes, cursor pagination | VERIFIED with warning | Substantive and tested; numeric falsey cursor handling remains advisory unless Kalshi emits numeric cursors. |
| `src/albert/kalshi/routing.py` | Historical/live cutoff routing | VERIFIED with warning | Substantive and tested; source-derived candlestick ticker path is not URL-escaped. |
| `src/albert/kalshi/schemas.py` | Strict normalized row models | VERIFIED | Strict Pydantic models and Decimal fields found. |
| `src/albert/kalshi/normalize.py` | Fixture-backed normalizers | VERIFIED | Market, candlestick, trade, orderbook, and quality functions found and tested. |
| `src/albert/kalshi/quality.py` | Missing-data quality checks | VERIFIED | Missing outcome, price-history, and orderbook counts implemented. |
| `src/albert/kalshi/pipeline.py` | End-to-end discovery orchestration | PARTIAL | Wired and substantive, but trade pagination and empty-table schema gaps block full DATA-02/normalization contract. |
| `src/albert/commands/ingest.py` | `albert ingest kalshi discover` CLI | VERIFIED | Command group and summary output implemented and tested. |
| `tests/test_kalshi_*.py` | Contract and integration tests | VERIFIED with coverage gap | 21 focused tests pass; trade pagination and empty-output schema regressions are not covered. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/albert/commands/ingest.py` | `src/albert/kalshi/pipeline.py` | CLI calls `run_kalshi_discovery` | WIRED | Direct import/call found. |
| `src/albert/kalshi/pipeline.py` | `src/albert/manifest.py` | `RunManifest`, `SourceMetadata`, `OutputMetadata` | WIRED | Manifest is constructed and written. |
| `src/albert/kalshi/pipeline.py` | `src/albert/cache.py` | `normalized_cache_path`, `run_manifest_path` | WIRED | Existing helpers are used for output paths. |
| `src/albert/kalshi/client.py` | `src/albert/cache.py` | `raw_cache_path` | WIRED | Every `get_json_page()` raw write goes through `raw_cache_path()`. |
| `src/albert/kalshi/pipeline.py` | trade pagination | `iter_cursor_pages(..., collection_key="trades")` | NOT WIRED | `_fetch_trades()` uses `get_json_page()` once per window. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/albert/commands/ingest.py` | CLI options and config roots | Typer options plus `load_config()` | Yes | FLOWING |
| `src/albert/kalshi/pipeline.py` | `market_rows`, `outcome_rows`, `skipped_rows` | `iter_cursor_pages(... markets ...)` plus `normalize_markets_page()` | Yes | FLOWING |
| `src/albert/kalshi/pipeline.py` | `price_rows` | `plan_candlestick_windows()` plus `get_json_page()` and `normalize_candlesticks_page()` | Yes | FLOWING |
| `src/albert/kalshi/pipeline.py` | `trade_rows` | `plan_trade_windows()` plus one `get_json_page()` per window | Partial | HOLLOW - paginated trade pages after first cursor are disconnected. |
| `src/albert/kalshi/pipeline.py` | Parquet table schemas | `pl.DataFrame([row dicts])` | Partial | HOLLOW - empty row sets produce zero-column tables. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Focused Phase 2 tests | `python -m pytest tests/test_kalshi_client.py tests/test_kalshi_routing.py tests/test_kalshi_normalize.py tests/test_kalshi_discovery_cli.py tests/test_kalshi_cache_manifest.py -q` | `21 passed in 0.27s` | PASS |
| Runtime imports | `python -c 'import httpx, polars, pyarrow, albert.kalshi'` | exit 0 | PASS |
| Empty normalized output schema | targeted `_write_normalized_outputs()` check | `columns []`, `height 0` for `kalshi_trades` | FAIL |
| Trade pagination data flow | targeted `_fetch_trades()` fake client with `cursor='page-2'` | `iter_calls 0`, cursors remained unconsumed | FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DATA-01 | All Phase 2 plans | User can run a CLI command that discovers Kalshi 15-minute Bitcoin markets over a requested date range. | SATISFIED | CLI route exists, explicit date range is validated, pipeline discovers `KXBTC15M` markets, and CLI/manifest tests pass. |
| DATA-02 | All Phase 2 plans | User can retrieve Kalshi market metadata, order-book snapshots or bid/ask candles, trades, and settlement outcomes for discovered markets. | BLOCKED | Metadata, candles, orderbooks, and outcomes are wired, but trades are incomplete when Kalshi returns paginated trade cursors. Empty Parquet outputs also break the normalized table contract for missing/empty observations. |

No additional Phase 2 requirement IDs were found in `.planning/REQUIREMENTS.md` beyond DATA-01 and DATA-02.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/albert/kalshi/pipeline.py` | 274 | `get_json_page()` for trades instead of cursor iteration | Blocker | Drops trade pages after the first page. |
| `src/albert/kalshi/pipeline.py` | 346 | `pl.DataFrame([]).write_parquet(...)` for empty row sets | Blocker | Writes zero-column normalized tables. |
| `src/albert/kalshi/client.py` | 106 | `if cursor:` | Warning | Numeric cursor token `0` is not passed to the next request. |
| `src/albert/kalshi/routing.py` | 128 | Source-derived ticker interpolated into URL path | Warning | Unexpected ticker characters can change the request path; URL-encode path segments. |

### Human Verification Required

After the blockers are fixed, run the optional live smoke command against Kalshi public data:

1. `albert ingest kalshi discover --start-date 2026-04-01 --end-date 2026-04-02 --series-ticker KXBTC15M --run-id kalshi-smoke-20260401`
2. Expected: the command completes, writes raw cache files, writes all normalized tables with schemas, records manifest sources for all fetched pages, and reports quality counts.
3. Why human: live external service behavior, endpoint availability, and real Kalshi payload pagination cannot be verified from fixture-only automated tests.

### Gaps Summary

Phase 2 has a working CLI, client, routing, normalization, cache, and manifest foundation, and the focused automated gates pass. The phase goal is not fully achieved because DATA-02 trade retrieval can silently miss paginated trade pages, and normalized output tables lose their declared schema when no rows are present. These are not deferred to later roadmap phases and should be closed before Phase 2 is marked complete.

---

_Verified: 2026-04-28T20:31:00Z_  
_Verifier: Claude (gsd-verifier)_
