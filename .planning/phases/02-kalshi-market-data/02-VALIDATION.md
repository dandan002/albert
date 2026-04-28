---
phase: 02
slug: kalshi-market-data
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-28
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x with Ruff linting |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `python -m pytest tests/test_kalshi_routing.py tests/test_kalshi_normalize.py -q` |
| **Full suite command** | `make check` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run the focused Kalshi test file for the touched module plus `python -m ruff check src tests`
- **After every plan wave:** Run `make check`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-00-01 | 00 | 0 | DATA-01, DATA-02 | T-02-03 | Phase 2 dependencies install and import before Polars-backed tests run | dependency | `python -c 'import httpx, polars, pyarrow'` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 0 | DATA-01, DATA-02 | T-02-04 | Fixture payloads cover live/historical schema drift without live network calls | fixture | `python -m json.tool tests/fixtures/kalshi/live_markets_page.json` | ❌ W0 | ⬜ pending |
| 02-05-01 | 05 | 0 | DATA-01, DATA-02 | T-02-04 | Contract tests encode client, routing, normalization, CLI, cache, and manifest behavior | unit | `python -m pytest tests/test_kalshi_routing.py tests/test_kalshi_normalize.py -q` | ❌ W0 | ⬜ pending |
| 02-01-01 | 01 | 1 | DATA-01, DATA-02 | T-02-01, T-02-02 | Raw request pages are written only through cache helpers and manifests reject secret-like params | unit | `python -m pytest tests/test_kalshi_client.py -q` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | DATA-01, DATA-02 | T-02-03 | Live and historical windows are split by the correct cutoff field for each data family | unit | `python -m pytest tests/test_kalshi_routing.py -q` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | DATA-01 | T-02-04 | Ambiguous BTC or interval mappings are skipped with explicit reasons | unit | `python -m pytest tests/test_kalshi_normalize.py::test_records_ambiguous_btc_markets_as_skipped -q` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | DATA-02 | T-02-04 | Normalizers parse fixed-point strings without float conversion and preserve endpoint provenance | unit | `python -m pytest tests/test_kalshi_normalize.py -q` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | DATA-01 | T-02-03 | CLI requires explicit UTC date ranges and writes discovery outputs plus manifest metadata | CLI | `python -m pytest tests/test_kalshi_discovery_cli.py -q` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 2 | DATA-02 | T-02-04 | Missing outcomes, missing price history, and unavailable orderbooks are reported as quality flags | integration | `python -m pytest tests/test_kalshi_cache_manifest.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` — dependencies include `httpx`, `polars`, and `pyarrow`, followed by `python -c 'import httpx, polars, pyarrow'`
- [ ] `src/albert/kalshi/__init__.py` — package marker for Kalshi modules
- [ ] `tests/fixtures/kalshi/*.json` — live/historical markets, candlesticks, trades, orderbooks, missing outcome, missing history, observed outcome fields (`result`, `settlement_value_dollars`, `settlement_ts`), unavailable historical orderbook depth, and ambiguous BTC market cases
- [ ] `tests/test_kalshi_routing.py` — cutoff field selection and spanning-window splits for markets/candlesticks/trades
- [ ] `tests/test_kalshi_normalize.py` — strict schema normalization for markets, price observations, trades, orderbooks, outcomes, skipped markets, and quality summaries
- [ ] `tests/test_kalshi_client.py` — pagination, timeout/retry status handling, and raw cache path writes with mocked HTTP transport
- [ ] `tests/test_kalshi_discovery_cli.py` — Typer command arguments, output paths, summary text, and manifest creation
- [ ] `tests/test_kalshi_cache_manifest.py` — raw pages, normalized Parquet outputs, and manifest source/output metadata

---

## Threat References

| Threat | Behavior | Mitigation |
|--------|----------|------------|
| T-02-01 | Path traversal through source, endpoint, table, or run identifiers | Reuse `sanitize_segment()`, `raw_cache_path()`, `normalized_cache_path()`, and `run_manifest_path()` for all generated paths |
| T-02-02 | Secret leakage into manifests or raw request parameters | Keep Phase 2 public-data only by default and rely on existing secret-like key rejection in config and manifest models |
| T-02-03 | Look-ahead from incorrect cutoff or timestamp filters | Preserve endpoint, request parameters, fetched time, and covered ranges per source page; test spanning windows |
| T-02-04 | Silent data corruption from Kalshi schema drift or ambiguous BTC market mapping | Use fixture-backed Pydantic/normalizer validation, keep unknown fields in raw cache only, and emit skipped-market/quality rows |

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Optional live Kalshi smoke run | DATA-01, DATA-02 | Phase tests must not depend on live API availability or current market listings | Run `albert ingest kalshi discover --start-date 2026-04-01 --end-date 2026-04-02 --series-ticker KXBTC15M` against the live API after automated tests pass, then inspect the manifest source endpoints |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
