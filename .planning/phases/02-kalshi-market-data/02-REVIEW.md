---
phase: 02-kalshi-market-data
reviewed: 2026-04-28T12:00:00Z
depth: quick
files_reviewed: 12
files_reviewed_list:
  - src/albert/kalshi/pipeline.py
  - src/albert/kalshi/client.py
  - src/albert/kalshi/normalize.py
  - src/albert/kalshi/schemas.py
  - src/albert/kalshi/routing.py
  - src/albert/kalshi/quality.py
  - src/albert/commands/ingest.py
  - tests/test_kalshi_cache_manifest.py
  - tests/test_kalshi_client.py
  - tests/test_kalshi_discovery_cli.py
  - tests/test_kalshi_normalize.py
  - tests/test_kalshi_routing.py
findings:
  critical: 0
  warning: 1
  info: 3
  total: 4
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-04-28T12:00:00Z
**Depth:** quick
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Reviewed 12 files across the Kalshi market data ingestion pipeline — client, routing, normalization, schemas, quality assessment, pipeline orchestration, CLI command, and their tests. The codebase is well-structured with strong schema validation (Pydantic `StrictRow` with `extra="forbid"`), proper timezone-aware datetime handling, endpoint segment allowlisting, and comprehensive test coverage including fixture-driven integration tests.

No critical security vulnerabilities or crash-causing bugs were found. One warning-level issue concerns silent data loss in quality assessment due to dict key collision, and three info-level items note dead code, missing retry backoff, and unvalidated URL interpolation.

## Warnings

### WR-01: Duplicate market_ticker silently overwrites outcome in quality assessment

**File:** `src/albert/kalshi/quality.py:16`
**Issue:** `outcome_by_ticker = {outcome.market_ticker: outcome for outcome in outcomes}` builds a dict from outcome rows keyed by `market_ticker`. If the API ever returns multiple outcome records for the same ticker (e.g., due to a data anomaly or schema change), later entries silently overwrite earlier ones. The quality check then operates on a subset of outcomes, potentially under-reporting `missing_outcome_count`. While Kalshi market tickers are expected to be unique per response, there is no assertion or guard making this invariant explicit.
**Fix:** Add a deduplication check after building the dict, or use a list-based lookup instead:

```python
outcome_by_ticker: dict[str, KalshiOutcomeRow] = {}
for outcome in outcomes:
    if outcome.market_ticker in outcome_by_ticker:
        # Log or count duplicates instead of silent overwrite
        continue
    outcome_by_ticker[outcome.market_ticker] = outcome
```

Alternatively, if uniqueness is guaranteed by API contract, add an assertion:

```python
outcome_by_ticker = {o.market_ticker: o for o in outcomes}
assert len(outcome_by_ticker) == len(outcomes), "Duplicate market_ticker in outcomes"
```

## Info

### IN-01: Unreachable RuntimeError fallback in client retry loop

**File:** `src/albert/kalshi/client.py:143-145`
**Issue:** The `RuntimeError("Kalshi request failed without a response")` at line 145 is unreachable. The `for` loop always terminates either via `return payload` (success), `raise_for_status()` raising `HTTPStatusError` (non-retryable HTTP error on any attempt, or retryable error on last attempt), or re-raising `TransportError` on the last attempt. The post-loop `if last_error is not None` branch can never be reached because `TransportError` is always re-raised on the final attempt inside the loop, and successful responses return early.
**Fix:** Consider removing the dead code or restructuring to make intent clearer. If kept as a safety net, add a comment explaining that it's defensive:

```python
# Defensive: should be unreachable given loop logic above
if last_error is not None:
    raise last_error
raise RuntimeError("Kalshi request failed without a response")
```

### IN-02: No backoff or delay between HTTP retries

**File:** `src/albert/kalshi/client.py:132`
**Issue:** The retry loop in `_request_json` immediately retries on retryable status codes (429, 500, 502, 503, 504) with no sleep or exponential backoff between attempts. For 429 (rate limiting) responses specifically, hammering the API immediately may worsen the rate limit condition and reduce the chance of recovery. The project's constraints mention "conservative" execution behavior.
**Fix:** Add a small backoff between retries:

```python
import time

for attempt in range(1, self._max_retries + 1):
    try:
        response = self._http.get(url, params=parameters)
        if response.status_code in RETRY_STATUSES and attempt < self._max_retries:
            time.sleep(0.5 * attempt)  # linear backoff
            continue
        response.raise_for_status()
        ...
```

### IN-03: URL path interpolation without ticker format validation

**File:** `src/albert/kalshi/routing.py:128-129`
**Issue:** `plan_candlestick_windows` interpolates the `ticker` parameter directly into URL paths (`f"/historical/markets/{ticker}/candlesticks"` and `f"/markets/{ticker}/candlesticks"`). While `ticker` originates from `market.market_ticker` (a Kalshi API field that should be alphanumeric), there is no explicit validation that it doesn't contain path traversal characters (e.g., `../`). The `KalshiClient._url_for` method validates that the path starts with `/`, but doesn't check for path traversal within the path.
**Fix:** Add a runtime check or use `sanitize_segment` (already used for endpoint segments) to validate the ticker doesn't contain `/`, `..`, or other path traversal sequences:

```python
def plan_candlestick_windows(ticker: str, ...) -> list[EndpointWindow]:
    if "/" in ticker or ".." in ticker:
        raise ValueError(f"Invalid ticker for URL path: {ticker!r}")
    ...
```

---

_Reviewed: 2026-04-28T12:00:00Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: quick_