---
phase: 1
slug: research-cli-foundation
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-28
---

# Phase 1 - Validation Strategy

Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `python -m pytest tests/test_cli.py tests/test_config.py tests/test_cache.py tests/test_manifest.py -q` |
| **Full suite command** | `python -m pytest -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_cli.py tests/test_config.py tests/test_cache.py tests/test_manifest.py -q`
- **After every plan wave:** Run `python -m pytest -q` and `python -m ruff check src tests`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | DATA-04 | T-01-01 / T-01-02 | Config loading rejects invalid path-like values and does not expose secrets. | unit/smoke | `python -m pytest tests/test_cli.py tests/test_config.py -q` | No - W0 | pending |
| 01-02-01 | 02 | 1 | DATA-04 | T-01-03 | Cache helpers reject unsafe path segments and keep generated paths under configured roots. | unit | `python -m pytest tests/test_cache.py tests/test_manifest.py -q` | No - W0 | pending |
| 01-03-01 | 03 | 1 | DATA-04 | T-01-01 / T-01-02 / T-01-03 | Baseline checks prove CLI wiring, config, cache paths, and manifests are reproducible. | unit/lint | `python -m pytest -q && python -m ruff check src tests` | No - W0 | pending |

---

## Wave 0 Requirements

- [ ] `pyproject.toml` - package metadata, dev dependencies, pytest config, and Ruff config.
- [ ] `tests/test_cli.py` - CLI help and command group wiring checks.
- [ ] `tests/test_config.py` - config default, TOML, and environment override checks.
- [ ] `tests/test_cache.py` - deterministic raw and normalized cache path checks.
- [ ] `tests/test_manifest.py` - run manifest schema and JSON serialization checks.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 30 seconds
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-28
