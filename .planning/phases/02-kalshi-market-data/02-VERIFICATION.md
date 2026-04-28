---
phase: 02-kalshi-market-data
verified: 2026-04-28T21:15:00Z
status: human_needed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 9/11
  gaps_closed:
    - "User can fetch all available Kalshi trade data for requested windows (G-01: trade pagination now uses iter_cursor_pages)"
    - "Normalized Parquet tables retain declared columns when a table has zero rows (G-02: TABLE_SCHEMAS provides explicit schemas)"
  gaps_remaining: []
  regressions: []
gaps: []
human_verification:
  - test: "albert ingest kalshi discover --start-date 2026-04-01 --end-date 2026-04-02 --series-ticker KXBTC15M --run-id kalshi-smoke-20260401"
    expected: "Command completes, writes raw cache files, writes all 7 normalized Parquet tables with correct schemas, records manifest sources for all fetched pages (including paginated trades), and reports quality counts."
    why_human: "Live external service behavior, endpoint availability, real Kalshi payload pagination, and real data quality cannot be verified from fixture-only automated tests."
---

# Phase 2: Kalshi Market Data Verification Report

**Phase Goal:** Ingest Kalshi market data and compute short-horizon probabilities from normalized BTC event contracts. (Roadmap: "Kalshi 15-minute Bitcoin markets can be discovered, fetched, cached, normalized, and linked to outcomes.")
**Verified:** 2026-04-28T21:15:00Z
**Status:** human_needed
**Re-verification:** Yes — after gap closure by plan 02-06

## Re-verification Context

This is a re-verification. The initial verification (2026-04-28T20:31:00Z) found 2 gaps:
- **G-01:** Trade retrieval only fetched one page per endpoint window, ignoring cursor-based pagination. Plan 02-06 fixed this by replacing `_fetch_trades` single `get_json_page` call with an `iter_cursor_pages(..., collection_key="trades")` loop.
- **G-02:** Empty normalized Parquet outputs had zero columns. Plan 02-06 fixed this by adding `TABLE_SCHEMAS` and `_rows_dataframe` to preserve declared columns.

Both gaps are now **closed**. Regression tests `test_discover_paginates_trade_pages_and_records_each_source` and `test_discover_writes_empty_parquet_tables_with_declared_columns` pass.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Phase 2 runtime dependencies and package namespace exist. | ✓ VERIFIED | `pyproject.toml` declares `httpx`, `polars`, and `pyarrow`; `src/albert/kalshi/__init__.py` exists; `python -c 'import httpx, polars, pyarrow, albert.kalshi'` exits 0. |
| 2 | Fixture payloads cover eligible KXBTC15M markets, live/historical shapes, missing data, orderbooks, and skipped ambiguous mappings without live API access. | ✓ VERIFIED | 10 JSON fixtures under `tests/fixtures/kalshi/`; all consumed by normalization and manifest tests; `tests/test_kalshi_normalize.py` covers eligible, missing-outcome, skipped, candlestick, trade, and orderbook fixtures. |
| 3 | Kalshi source pages can be fetched with bounded retries and raw cache writes through safe helpers. | ✓ VERIFIED | `KalshiClient.get_json_page()` uses retry statuses `{429, 500, 502, 503, 504}` and writes via `raw_cache_path()`; `test_retries_429_then_succeeds_with_bounded_attempts` passes. |
| 4 | Cursor pagination exists for supported client calls and preserves request filters. | ✓ VERIFIED (advisory) | `KalshiClient.iter_cursor_pages()` exists; `test_iter_cursor_pages_keeps_filters_and_adds_cursor` passes. **Advisory:** Line 106 uses `if cursor:` which treats numeric `0` as falsey; Kalshi API returns string cursors (low risk). |
| 5 | Older data can be routed through historical endpoints when required. | ✓ VERIFIED | `plan_market_windows()`, `plan_candlestick_windows()`, and `plan_trade_windows()` split by `market_settled_ts` or `trades_created_ts`; routing tests pass. |
| 6 | Eligible Kalshi 15-minute BTC markets normalize into strict market/outcome rows; ambiguous mappings are skipped. | ✓ VERIFIED | `normalize_markets_page()` validates rows with Pydantic models; `test_records_ambiguous_btc_markets_as_skipped` and market/outcome normalization tests pass. |
| 7 | Price, trade, and orderbook observations preserve source granularity, provenance, and nullable liquidity fields. | ✓ VERIFIED | Normalizer tests cover candlesticks, trades, orderbooks, Decimal fields, endpoint provenance, and implied asks. |
| 8 | User can run `albert ingest kalshi discover` for an explicit UTC date range. | ✓ VERIFIED | `src/albert/commands/ingest.py` registers `ingest kalshi discover`; CLI test passes with date validation and summary output. |
| 9 | Discovery writes raw page cache files, normalized Parquet outputs, quality rows, and a run manifest. | ✓ VERIFIED | `run_kalshi_discovery()` writes `RunManifest` sources/outputs; manifest integration test asserts sources, outputs, and all 7 table names. |
| 10 | User can fetch all available Kalshi trade pages for requested windows. | ✓ VERIFIED | `_fetch_trades()` now uses `iter_cursor_pages(..., collection_key="trades")` loop; `test_discover_paginates_trade_pages_and_records_each_source` asserts cursor page in sources and `TRADE-PAGE-2` in trades Parquet. |
| 11 | Normalized Parquet tables retain declared columns when a table has zero rows. | ✓ VERIFIED | `TABLE_SCHEMAS` dict provides explicit Polars column types for all 7 tables; `_rows_dataframe` uses `pl.DataFrame(schema=TABLE_SCHEMAS[table_name])` for empty rows; `test_discover_writes_empty_parquet_tables_with_declared_columns` asserts column sets for trades, orderbooks, and skipped_markets. |

**Score:** 11/11 truths verified

### Deferred Items

No items deferred — all truths are verified within this phase.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | Phase 2 runtime dependencies | ✓ VERIFIED | Contains `httpx>=0.28.1`, `polars>=1.40.1`, `pyarrow>=24.0.0`. |
| `src/albert/kalshi/__init__.py` | Kalshi package marker | ✓ VERIFIED | Exists with `__all__`. |
| `src/albert/kalshi/client.py` | HTTP client, retries, raw cache writes, cursor pagination | ✓ VERIFIED (advisory) | 175 lines; `FetchedPage`, `KalshiClient`, retry logic, `iter_cursor_pages`, endpoint allowlist all present and tested. Advisory: `if cursor:` on line 106 treats `0` as falsey. |
| `src/albert/kalshi/routing.py` | Historical/live cutoff routing | ✓ VERIFIED (advisory) | 185 lines; `HistoricalCutoff`, `EndpointWindow`, `plan_market_windows`, `plan_candlestick_windows`, `plan_trade_windows` all present. Advisory: source-derived ticker in path is not URL-encoded. |
| `src/albert/kalshi/schemas.py` | Strict normalized row models | ✓ VERIFIED | 219 lines; 7 table constants, `StrictRow` with `ConfigDict(extra="forbid")`, `ProvenanceRow` with source fields, all row models present. |
| `src/albert/kalshi/normalize.py` | Fixture-backed normalizers | ✓ VERIFIED | 313 lines; market, candlestick, trade, orderbook, quality normalizer functions present and tested. |
| `src/albert/kalshi/quality.py` | Missing-data quality checks | ✓ VERIFIED | 47 lines; `assess_market_quality` counts missing outcomes/prices/orderbooks; `quality_summary_text` formats counts. |
| `src/albert/kalshi/pipeline.py` | End-to-end discovery orchestration | ✓ VERIFIED | 468 lines; `KalshiDiscoveryResult`, `run_kalshi_discovery`, `TABLE_SCHEMAS`, `_rows_dataframe`, `iter_cursor_pages` for trades, `_write_normalized_outputs` with schema-preserving writes. |
| `src/albert/commands/ingest.py` | `albert ingest kalshi discover` CLI | ✓ VERIFIED | 74 lines; Typer command group, `discover` command with date range, series ticker, run ID, deterministic paths. |
| `tests/test_kalshi_*.py` | Contract and integration tests | ✓ VERIFIED | 23 tests pass across 5 test files; regression tests for both previous gaps now included. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/albert/commands/ingest.py` | `src/albert/kalshi/pipeline.py` | CLI calls `run_kalshi_discovery` | ✓ WIRED | Direct import and call found at line 7 and line 47. |
| `src/albert/kalshi/pipeline.py` | `src/albert/manifest.py` | `RunManifest`, `SourceMetadata`, `OutputMetadata` | ✓ WIRED | Line 40 imports; lines 255-281 construct and write manifest. |
| `src/albert/kalshi/pipeline.py` | `src/albert/cache.py` | `normalized_cache_path`, `run_manifest_path` | ✓ WIRED | Line 9 imports; lines 439, 254 use helpers. |
| `src/albert/kalshi/client.py` | `src/albert/cache.py` | `raw_cache_path` | ✓ WIRED | Line 9 imports; lines 70-76 write through `raw_cache_path()`. |
| `src/albert/kalshi/pipeline.py` | `src/albert/kalshi/client.py` | `_fetch_trades` uses `iter_cursor_pages(..., collection_key="trades")` | ✓ WIRED | Lines 368-386: `for page in client.iter_cursor_pages(..., collection_key="trades")`. Gap G-01 CLOSED. |
| `src/albert/kalshi/pipeline.py` | `src/albert/kalshi/schemas.py` | `TABLE_SCHEMAS` maps each table to declared columns | ✓ WIRED | Lines 54-146 define `TABLE_SCHEMAS`; lines 446-449 `_rows_dataframe` uses schema for empty tables. Gap G-02 CLOSED. |
| `src/albert/kalshi/pipeline.py` | `src/albert/kalshi/routing.py` | Market/candlestick/trade window planning | ✓ WIRED | Lines 196, 221-228, 327-332 import and call routing functions. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/albert/commands/ingest.py` | CLI options and config roots | Typer options plus `load_config()` | Yes | ✓ FLOWING |
| `src/albert/kalshi/pipeline.py` | `market_rows`, `outcome_rows`, `skipped_rows` | `iter_cursor_pages(..., collection_key="markets")` plus `normalize_markets_page()` | Yes | ✓ FLOWING |
| `src/albert/kalshi/pipeline.py` | `price_rows` | `plan_candlestick_windows()` plus `get_json_page()` and `normalize_candlesticks_page()` | Yes | ✓ FLOWING |
| `src/albert/kalshi/pipeline.py` | `trade_rows` | `plan_trade_windows()` plus `iter_cursor_pages(..., collection_key="trades")` and `normalize_trades_page()` | Yes | ✓ FLOWING — Gap G-01 CLOSED |
| `src/albert/kalshi/pipeline.py` | Parquet table outputs | `_rows_dataframe(table_name, rows)` with `TABLE_SCHEMAS` | Yes | ✓ FLOWING — Gap G-02 CLOSED |
| `src/albert/kalshi/pipeline.py` | Manifest | `RunManifest` with sources and outputs | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 2 test suite | `python -m pytest tests/test_kalshi_client.py tests/test_kalshi_routing.py tests/test_kalshi_normalize.py tests/test_kalshi_discovery_cli.py tests/test_kalshi_cache_manifest.py -q` | 23 passed in 0.42s | ✓ PASS |
| Module imports | `python -c 'import httpx, polars, pyarrow, albert.kalshi; from albert.kalshi.pipeline import TABLE_SCHEMAS, run_kalshi_discovery; ...'` | exit 0 | ✓ PASS |
| Lint check | `python -m ruff check src/albert/kalshi/ tests/test_kalshi_*.py` | All checks passed | ✓ PASS |
| Trade pagination | `python -c 'import inspect; src = inspect.getsource(_fetch_trades); assert "iter_cursor_pages" in src and "collection_key=\"trades\"" in src'` | Both assertions pass | ✓ PASS |
| Empty table schema | `python -c 'from albert.kalshi.pipeline import _rows_dataframe; df = _rows_dataframe("kalshi_trades", []); assert len(df.columns) == 9 and "trade_id" in df.columns'` | Assertions pass | ✓ PASS |
| Manifest cursor coverage | Test `test_discover_paginates_trade_pages_and_records_each_source` | Cursor page in sources, TRADE-PAGE-2 in Parquet, >7 sources | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DATA-01 | All Phase 2 plans | User can run a CLI command that discovers Kalshi 15-minute Bitcoin markets over a requested date range. | ✓ SATISFIED | CLI `albert ingest kalshi discover --start-date --end-date --series-ticker KXBTC15M --run-id` routes through pipeline; explicit date validation; CLI test passes. |
| DATA-02 | All Phase 2 plans | User can retrieve Kalshi market metadata, order-book snapshots or bid/ask candles, trades, and settlement outcomes for discovered markets. | ✓ SATISFIED | `run_kalshi_discovery` fetches metadata, candlesticks, trades (now with cursor pagination), orderbooks, and outcomes; all 7 normalized Parquet tables written with proper schemas; manifest records all sources including paginated trade pages. |

No additional Phase 2 requirement IDs were found in `.planning/REQUIREMENTS.md` beyond DATA-01 and DATA-02.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/albert/kalshi/client.py` | 106 | `if cursor:` | ⚠️ Warning | Numeric cursor token `0` would be treated as falsey and not passed to the next request. Kalshi API returns string cursors (e.g., `"page-2"`), making this low risk in practice. |
| `src/albert/kalshi/routing.py` | 128 | Source-derived ticker interpolated into URL path | ⚠️ Info | `f"/historical/markets/{ticker}/candlesticks"` — unexpected ticker characters could change the request path. Downstream phases should consider URL-encoding. |

No blocker anti-patterns found. Both warnings are advisory and do not prevent goal achievement.

### Human Verification Required

After automated checks pass, run the optional live smoke command against Kalshi public data:

1. **Live Kalshi smoke test**
   - **Test:** `albert ingest kalshi discover --start-date 2026-04-01 --end-date 2026-04-02 --series-ticker KXBTC15M --run-id kalshi-smoke-20260401`
   - **Expected:** The command completes, writes raw cache files, writes all 7 normalized Parquet tables with correct schemas, records manifest sources for all fetched pages (including paginated trades), and reports quality counts.
   - **Why human:** Live external service behavior, endpoint availability, real Kalshi payload pagination, and real data quality cannot be verified from fixture-only automated tests.

### Gaps Summary

**No gaps remain.** The two blockers from the initial verification (G-01: trade pagination, G-02: empty table schemas) have been closed by plan 02-06. All 11 observable truths are verified, all artifacts pass existence, substantive, and wiring checks, and data flows through the pipeline for all collection types including paginated trades.

The remaining human verification item is a live smoke test that requires actual Kalshi API access, which cannot be replicated from automated tests alone.

---

_Verified: 2026-04-28T21:15:00Z_
_Verifier: the agent (gsd-verifier)_