---
phase: 03
slug: deribit-derivatives-data
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-28
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing from Phase 1) |
| **Config file** | pyproject.toml (existing) |
| **Quick run command** | `python -m pytest tests/ -k deribit -q` |
| **Full suite command** | `python -m pytest -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -k deribit -q`
- **After every plan wave:** Run `python -m pytest -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | DATA-03 | T-03-01 | Input validation on instrument names and path segments | unit | `python -m pytest tests/test_deribit_client.py -q` | ✅ | ⬜ pending |
| 03-01-02 | 01 | 1 | DATA-03 | T-03-02 | No auth credentials stored or transmitted | unit | `python -m pytest tests/test_deribit_client.py -q` | ✅ | ⬜ pending |
| 03-01-03 | 01 | 1 | DATA-03 | — | Fixture validation and schema conformity | unit | `python -m pytest tests/test_deribit_fixtures.py -q` | ✅ | ⬜ pending |
| 03-02-01 | 02 | 2 | DATA-03 | T-03-01 | Null handling for optional fields (IV, greeks) | unit | `python -m pytest tests/test_deribit_normalize.py -q` | ✅ | ⬜ pending |
| 03-02-02 | 02 | 2 | DATA-03 | — | JSON-RPC unwrap and ms-epoch timestamp conversion | unit | `python -m pytest tests/test_deribit_normalize.py -q` | ✅ | ⬜ pending |
| 03-02-03 | 02 | 2 | DATA-03 | T-03-03 | Sanity-check OHLC time ranges dont exceed request window | unit | `python -m pytest tests/test_deribit_normalize.py -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_deribit_client.py` — client, retry, and endpoint tests
- [ ] `tests/test_deribit_normalize.py` — normalization and schema tests
- [ ] `tests/test_deribit_fixtures.py` — fixture validation tests
- [ ] `tests/fixtures/deribit/` — fixture directory with sample payloads
- [ ] Deribit fixtures: futures instrument, option instrument, OHLC response, ticker response, no_data response, book summary response

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live Deribit API connectivity | DATA-03 | Real API call with network | `albert ingest deribit discover --start-date 2025-01-01 --end-date 2025-01-02` and verify output |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending