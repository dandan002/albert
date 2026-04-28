# Phase 1: Research CLI Foundation - Research

**Researched:** 2026-04-28 [VERIFIED: current_date]
**Domain:** Python CLI scaffolding, configuration, cache path contracts, run manifests, and pytest validation [VERIFIED: .planning/ROADMAP.md]
**Confidence:** HIGH [VERIFIED: official docs, PyPI JSON, local environment probes]

## User Constraints

No phase `CONTEXT.md` exists, so there are no locked phase-specific decisions to copy verbatim. [VERIFIED: `gsd-tools init phase-op 1` returned `"has_context": false`]

Project constraints that apply to this phase: Albert must remain a Python research CLI, not a dashboard or live/paper trading system. [VERIFIED: AGENTS.md] v1 must scope data work to Kalshi plus Deribit, with Polymarket deferred. [VERIFIED: AGENTS.md] Every run must be traceable to source endpoints, timestamps, and cached raw inputs. [VERIFIED: AGENTS.md] Direct repo edits should happen through the GSD workflow. [VERIFIED: AGENTS.md]

## Project Constraints (from AGENTS.md)

- Build a Python research CLI first. [VERIFIED: AGENTS.md]
- Keep v1 focused on Kalshi plus Deribit only. [VERIFIED: AGENTS.md]
- Do not add live or paper trading in v1. [VERIFIED: AGENTS.md]
- Cache raw and normalized data with provenance sufficient to reproduce every run. [VERIFIED: AGENTS.md]
- Use conservative bid/ask, fee, and liquidity assumptions in later backtests. [VERIFIED: AGENTS.md]
- Follow existing codebase patterns; the repository is currently greenfield except planning artifacts. [VERIFIED: `find . -maxdepth 3 -type f`]
- No `CLAUDE.md` exists in the repository. [VERIFIED: `sed -n '1,220p' CLAUDE.md` failed with no such file]
- No project skill directories exist at `.claude/skills/` or `.agents/skills/`. [VERIFIED: `find .claude/skills .agents/skills -maxdepth 2 -name SKILL.md`]

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-04 | User can cache raw source responses and normalized Parquet tables with run manifests that record endpoint, parameters, fetch time, and covered time range. [VERIFIED: .planning/REQUIREMENTS.md] | Use typed cache path helpers, Pydantic manifest models, JSON raw-cache writes, normalized-table path reservations, and pytest coverage for manifest fields and deterministic path construction. [VERIFIED: .planning/ROADMAP.md; CITED: https://docs.pydantic.dev/latest/concepts/serialization/; CITED: https://docs.pytest.org/en/stable/how-to/tmp_path.html] |

</phase_requirements>

## Summary

Phase 1 should establish a small installable `albert` package with a Typer root CLI and four top-level command groups: `ingest`, `model`, `backtest`, and `report`. [VERIFIED: .planning/ROADMAP.md; CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/] The CLI foundation should not implement Kalshi, Deribit, model, or backtest behavior yet; it should create command wiring, config loading, cache path conventions, and manifest creation that later phases reuse. [VERIFIED: .planning/ROADMAP.md]

Use `pyproject.toml` project metadata, a `src/albert` package layout, Typer for composable command groups, stdlib `tomllib` for TOML reads, Pydantic for config and manifest schemas, platformdirs for OS cache defaults, pytest for tests, and Ruff for lint/format. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/; CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/; CITED: https://docs.python.org/3/library/tomllib.html; CITED: https://docs.pydantic.dev/latest/concepts/serialization/; CITED: https://platformdirs.readthedocs.io/en/latest/api.html; CITED: https://docs.pytest.org/en/stable/how-to/tmp_path.html; CITED: https://docs.astral.sh/ruff/configuration/]

**Primary recommendation:** Build the package skeleton, CLI groups, config model, cache path module, and manifest model first, then validate all of them with pytest before adding external API clients or real market normalization. [VERIFIED: .planning/ROADMAP.md]

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12+ | Runtime and stdlib features. [VERIFIED: .planning/research/STACK.md] | Local environment has Python 3.12.2, and `tomllib` is in the stdlib from Python 3.11 onward. [VERIFIED: `python3 --version`; CITED: https://docs.python.org/3/library/tomllib.html] |
| Typer | 0.25.0, uploaded 2026-04-26T08:46:15Z | CLI root app and grouped commands. [VERIFIED: PyPI JSON lookup] | Typer composes `typer.Typer()` apps with `app.add_typer()` and provides test support through `typer.testing.CliRunner`. [CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/; CITED: https://typer.tiangolo.com/tutorial/testing/] |
| Pydantic | 2.13.3, uploaded 2026-04-20T14:46:41Z | Validate config and manifest schemas. [VERIFIED: PyPI JSON lookup] | Pydantic serializes models to dictionaries and JSON strings with `model_dump()` and `model_dump_json()`. [CITED: https://docs.pydantic.dev/latest/concepts/serialization/] |
| pydantic-settings | 2.14.0, uploaded 2026-04-20T13:37:38Z | Optional `ALBERT_` environment override layer. [VERIFIED: PyPI JSON lookup] | `BaseSettings` supports environment-derived values, prefixes, and nested delimiters. [CITED: https://docs.pydantic.dev/latest/concepts/pydantic_settings/] |
| platformdirs | 4.9.6, uploaded 2026-04-09T00:04:09Z | OS-appropriate default cache/config locations. [VERIFIED: PyPI JSON lookup] | `user_cache_dir` returns a user-scoped cache directory for an application. [CITED: https://platformdirs.readthedocs.io/en/latest/api.html] |
| tomllib | stdlib in Python 3.12 | Read project TOML config files. [VERIFIED: local Python 3.12.2] | Python documents `tomllib.load()` for TOML 1.0.0 parsing and notes that it does not write TOML. [CITED: https://docs.python.org/3/library/tomllib.html] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hatchling | 1.29.0, uploaded 2026-02-23T19:42:05Z | Build backend for `pyproject.toml`. [VERIFIED: PyPI JSON lookup] | Use for a simple modern package build without `setup.py`. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/] |
| pytest | 9.0.3, uploaded 2026-04-07T17:16:16Z | Test config, cache paths, manifests, and CLI wiring. [VERIFIED: PyPI JSON lookup] | Use `tmp_path` for isolated filesystem tests. [CITED: https://docs.pytest.org/en/stable/how-to/tmp_path.html] |
| Ruff | 0.15.12, uploaded 2026-04-24T18:17:22Z | Formatting and linting. [VERIFIED: PyPI JSON lookup] | Store Ruff configuration in `pyproject.toml` and run `ruff check` plus `ruff format --check`. [CITED: https://docs.astral.sh/ruff/configuration/] |
| polars + pyarrow | Defer to Phase 2 unless Phase 1 writes a real Parquet fixture. [VERIFIED: .planning/research/STACK.md; VERIFIED: .planning/ROADMAP.md] | Future normalized table IO. [VERIFIED: .planning/research/STACK.md] | In Phase 1, reserve normalized Parquet paths but avoid dataframe dependencies unless a smoke write is explicitly required. [VERIFIED: Phase 1 success criteria in .planning/ROADMAP.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Typer | Click | Click is Typer's underlying CLI dependency, but Typer reduces boilerplate for typed command functions and exposes Click-compatible testing through `CliRunner`. [CITED: https://pypi.org/pypi/typer/json; CITED: https://typer.tiangolo.com/tutorial/testing/] |
| Pydantic models | Plain dataclasses plus manual validation | Manual validation would duplicate type parsing, error reporting, and JSON serialization behavior that Pydantic already provides. [CITED: https://docs.pydantic.dev/latest/concepts/serialization/] |
| `tomllib` | TOMLKit | `tomllib` reads TOML in the stdlib; TOMLKit is only needed when preserving formatting or writing TOML. [CITED: https://docs.python.org/3/library/tomllib.html] |
| platformdirs | Hardcoded `~/.cache/albert` | platformdirs supplies OS-aware cache paths, while hardcoded paths encode Unix assumptions. [CITED: https://platformdirs.readthedocs.io/en/latest/api.html] |
| `venv` + pip | uv | `uv` is not installed locally, so the Phase 1 plan should use stdlib `venv` and pip unless the user chooses to add uv. [VERIFIED: `command -v uv` returned missing] |

**Installation:**

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

**Version verification:** Current package versions were verified with PyPI JSON on 2026-04-28; Typer was also verified with `python3 -m pip index versions typer`, which reported latest `0.25.0`. [VERIFIED: escalated PyPI JSON lookup; VERIFIED: escalated `pip index`]

## Architecture Patterns

### Recommended Project Structure

```text
pyproject.toml                  # PEP 621 metadata, script entrypoint, pytest and Ruff config. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/]
src/
└── albert/
    ├── __init__.py             # Package version surface. [VERIFIED: Python packaging pattern from PyPA docs]
    ├── cli.py                  # Root Typer app and command-group registration. [CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/]
    ├── config.py               # TOML load + Pydantic validation. [CITED: https://docs.python.org/3/library/tomllib.html; CITED: https://docs.pydantic.dev/latest/concepts/serialization/]
    ├── cache.py                # Raw and normalized path helpers. [VERIFIED: DATA-04]
    ├── manifest.py             # RunManifest and SourceMetadata models. [VERIFIED: DATA-04]
    └── commands/
        ├── ingest.py           # Placeholder command group. [VERIFIED: Phase 1 success criteria]
        ├── model.py            # Placeholder command group. [VERIFIED: Phase 1 success criteria]
        ├── backtest.py         # Placeholder command group. [VERIFIED: Phase 1 success criteria]
        └── report.py           # Placeholder command group. [VERIFIED: Phase 1 success criteria]
tests/
├── test_cli.py                 # Help output and command-group wiring. [CITED: https://typer.tiangolo.com/tutorial/testing/]
├── test_config.py              # TOML/default/env config behavior. [CITED: https://docs.python.org/3/library/tomllib.html]
├── test_cache.py               # Deterministic cache path construction. [VERIFIED: DATA-04]
└── test_manifest.py            # Manifest schema and JSON creation. [CITED: https://docs.pydantic.dev/latest/concepts/serialization/]
```

### Pattern 1: Composable Typer Command Groups

**What:** Create one `typer.Typer()` per command group and register each group on the root app with `app.add_typer(..., name="...")`. [CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/]

**When to use:** Use this from the first commit because Phase 1 success requires help output listing `ingest`, `model`, `backtest`, and `report`. [VERIFIED: .planning/ROADMAP.md]

**Example:**

```python
# Source: https://typer.tiangolo.com/tutorial/subcommands/add-typer/
import typer

from albert.commands import backtest, ingest, model, report

app = typer.Typer(help="Albert research CLI")
app.add_typer(ingest.app, name="ingest")
app.add_typer(model.app, name="model")
app.add_typer(backtest.app, name="backtest")
app.add_typer(report.app, name="report")
```

### Pattern 2: Read TOML, Validate with Pydantic

**What:** Parse TOML with `tomllib`, then validate into a Pydantic model instead of passing dictionaries through the codebase. [CITED: https://docs.python.org/3/library/tomllib.html; CITED: https://docs.pydantic.dev/latest/concepts/serialization/]

**When to use:** Use for `AlbertConfig`, cache roots, default output roots, and later source-specific config. [VERIFIED: .planning/ROADMAP.md]

**Example:**

```python
# Source: https://docs.python.org/3/library/tomllib.html + https://docs.pydantic.dev/latest/concepts/serialization/
from pathlib import Path
import tomllib

from pydantic import BaseModel


class AlbertConfig(BaseModel):
    cache_dir: Path
    output_dir: Path
    timezone: str = "UTC"


def load_config(path: Path) -> AlbertConfig:
    with path.open("rb") as file:
        data = tomllib.load(file)
    return AlbertConfig.model_validate(data)
```

### Pattern 3: Stable Cache Paths from Sanitized Slugs and Parameter Hashes

**What:** Convert source, endpoint, table, and interval labels into validated slugs, then append a stable hash of canonical JSON parameters. [VERIFIED: DATA-04; CITED: https://owasp.org/www-community/attacks/Path_Traversal]

**When to use:** Use for raw response files and normalized table output locations so identical requests resolve to identical cache paths. [VERIFIED: .planning/ROADMAP.md]

**Example:**

```python
# Source: DATA-04 path contract + OWASP path traversal guidance.
import hashlib
import json
import re
from pathlib import Path

SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def slug(value: str) -> str:
    lowered = value.strip().lower().replace("/", "_")
    if not SAFE_SLUG.fullmatch(lowered):
        raise ValueError(f"unsafe cache path segment: {value!r}")
    return lowered


def params_hash(params: dict[str, object]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def raw_cache_path(root: Path, source: str, endpoint: str, params: dict[str, object]) -> Path:
    return root / "raw" / slug(source) / slug(endpoint) / f"{params_hash(params)}.json"
```

### Pattern 4: Inject Clock and Run ID for Deterministic Tests

**What:** Manifest creation should accept `fetched_at` and `run_id` arguments or a small clock dependency, rather than calling current time deep inside file-writing helpers. [VERIFIED: Phase goal says deterministic research commands; VERIFIED: .planning/ROADMAP.md]

**When to use:** Use in `create_run_manifest()` so tests can assert exact timestamps and output paths. [VERIFIED: Phase 1 success criteria]

### Anti-Patterns to Avoid

- **Global package state:** Do not rely on packages installed in the active Anaconda environment because local Typer, pytest, Pydantic, pydantic-settings, and platformdirs versions differ from current PyPI versions. [VERIFIED: local import/version probes; VERIFIED: PyPI JSON lookup]
- **Anonymous cache files:** Do not write raw responses without source, endpoint, request parameters, fetch timestamp, and covered interval in the manifest. [VERIFIED: DATA-04]
- **Path construction with string concatenation:** Do not build cache paths by interpolating unsanitized endpoint or market strings into filenames. [CITED: https://owasp.org/www-community/attacks/Path_Traversal]
- **Phase 1 API clients:** Do not implement Kalshi or Deribit clients in this phase because those are Phase 2 and Phase 3 responsibilities. [VERIFIED: .planning/ROADMAP.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI parser and help output | Custom `argparse` tree | Typer | Typer supports composable command groups and `CliRunner` tests. [CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/; CITED: https://typer.tiangolo.com/tutorial/testing/] |
| Manifest schema validation | Manual dictionary checks | Pydantic models | Pydantic validates model input and serializes JSON output. [CITED: https://docs.pydantic.dev/latest/concepts/serialization/] |
| TOML parsing | Custom parser or regex | `tomllib` | `tomllib` parses TOML 1.0.0 in Python 3.11+. [CITED: https://docs.python.org/3/library/tomllib.html] |
| User cache defaults | Hardcoded OS paths | platformdirs | platformdirs exposes user cache/config directory APIs. [CITED: https://platformdirs.readthedocs.io/en/latest/api.html] |
| Isolated filesystem tests | Manual temp directory cleanup | pytest `tmp_path` | `tmp_path` supplies a unique `pathlib.Path` temporary directory per test. [CITED: https://docs.pytest.org/en/stable/how-to/tmp_path.html] |
| Package entry points | Shell wrapper scripts | `[project.scripts]` | PyPA defines `[project.scripts]` as console script entry points. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/] |

**Key insight:** Phase 1 is a contract phase; custom CLI parsing, untyped manifests, and ad hoc cache filenames would become migration debt once Kalshi and Deribit ingestion starts. [VERIFIED: DATA-04; VERIFIED: .planning/ROADMAP.md]

## Common Pitfalls

### Pitfall 1: Stale Global Dependencies

**What goes wrong:** Tests pass against globally installed packages but fail in a clean environment. [VERIFIED: local Typer import reported 0.9.0 while PyPI latest is 0.25.0]

**Why it happens:** The repository has no package skeleton or lock/install workflow yet. [VERIFIED: `find . -maxdepth 3 -type f`]

**How to avoid:** Create `pyproject.toml`, install with `python -m pip install -e ".[dev]"`, and run tests through `python -m pytest`. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/]

**Warning signs:** `pytest --version` reports a package outside `.venv`, or `python -c 'import typer'` resolves to an Anaconda site package. [VERIFIED: local environment probes]

### Pitfall 2: Non-Deterministic Manifests

**What goes wrong:** Manifest tests assert shape but not exact values, so timestamp and run-id regressions slip through. [VERIFIED: Phase goal requires deterministic research commands]

**Why it happens:** Code calls `datetime.now()` inside manifest creation without injection. [ASSUMED]

**How to avoid:** Accept explicit `fetched_at`, `started_at`, and `run_id` values in manifest constructors, with CLI code supplying defaults at the boundary. [VERIFIED: Phase 1 success criteria]

**Warning signs:** Tests use substring assertions for timestamps or only check that a manifest file exists. [ASSUMED]

### Pitfall 3: Cache Paths Encode Unsafe User Input

**What goes wrong:** Endpoint names, ticker strings, or source parameters containing `/`, `..`, spaces, or platform-reserved characters create ambiguous paths. [CITED: https://owasp.org/www-community/attacks/Path_Traversal]

**Why it happens:** File paths are assembled from raw API labels instead of validated slugs plus parameter hashes. [VERIFIED: DATA-04 risk analysis]

**How to avoid:** Validate path segments with an allowlist regex and store full original parameters in the manifest. [CITED: https://owasp.org/www-community/attacks/Path_Traversal]

**Warning signs:** Tests include only simple names like `kalshi` and omit adversarial path segments like `../x`. [VERIFIED: security test design]

### Pitfall 4: Mixing Raw Cache and Normalized Outputs

**What goes wrong:** Later phases cannot tell whether a path contains source JSON or a normalized table. [VERIFIED: DATA-04]

**Why it happens:** A single `cache/` helper lacks explicit namespaces. [ASSUMED]

**How to avoid:** Reserve separate roots such as `cache/raw/{source}/...`, `cache/normalized/{source}/{table}/...`, and `runs/{run_id}/manifest.json`. [VERIFIED: .planning/research/ARCHITECTURE.md]

**Warning signs:** A manifest output location has no `kind` or table/source metadata. [VERIFIED: DATA-04]

### Pitfall 5: Pulling Dataframe Dependencies Too Early

**What goes wrong:** Phase 1 grows into partial normalization before cache contracts and CLI tests are stable. [VERIFIED: Phase 2 and Phase 3 own ingestion/normalization]

**Why it happens:** DATA-04 mentions normalized Parquet, but Phase 1 success only requires helper-level raw and normalized cache paths plus manifest creation. [VERIFIED: .planning/REQUIREMENTS.md; VERIFIED: .planning/ROADMAP.md]

**How to avoid:** In Phase 1, test normalized Parquet path construction and manifest references; add real polars/pyarrow writes in the first phase that emits normalized tables. [VERIFIED: .planning/ROADMAP.md]

**Warning signs:** Phase 1 tasks start defining Kalshi or Deribit schemas. [VERIFIED: .planning/ROADMAP.md]

## Code Examples

Verified patterns from official sources:

### CLI Help Wiring Test

```python
# Source: https://typer.tiangolo.com/tutorial/testing/
from typer.testing import CliRunner

from albert.cli import app

runner = CliRunner()


def test_root_help_lists_command_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "model", "backtest", "report"):
        assert command in result.output
```

### Manifest JSON Creation

```python
# Source: https://docs.pydantic.dev/latest/concepts/serialization/
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class SourceMetadata(BaseModel):
    source: str
    endpoint: str
    parameters: dict[str, object]
    fetched_at: datetime
    covered_start: datetime
    covered_end: datetime
    raw_path: Path


class RunManifest(BaseModel):
    run_id: str
    config_path: Path
    config_hash: str
    output_dir: Path
    sources: list[SourceMetadata]


manifest = RunManifest(
    run_id="test-run",
    config_path=Path("configs/example.toml"),
    config_hash="0" * 64,
    output_dir=Path("runs/test-run"),
    sources=[
        SourceMetadata(
            source="kalshi",
            endpoint="markets",
            parameters={"start": "2026-01-01", "end": "2026-01-02"},
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            covered_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            covered_end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            raw_path=Path("cache/raw/kalshi/markets/example.json"),
        )
    ],
)

payload = manifest.model_dump_json(indent=2)
```

### Filesystem Test Isolation

```python
# Source: https://docs.pytest.org/en/stable/how-to/tmp_path.html
def test_raw_cache_path_uses_configured_root(tmp_path):
    path = raw_cache_path(
        root=tmp_path,
        source="kalshi",
        endpoint="markets",
        params={"ticker": "KXBTCD-26APR281200"},
    )

    assert path.is_relative_to(tmp_path)
    assert path.suffix == ".json"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `setup.py`-centered metadata | `pyproject.toml` with `[project]` and `[project.scripts]` | PyPA's current project metadata specification is `pyproject.toml`. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/] | Plan should create `pyproject.toml` first. [VERIFIED: greenfield repo scan] |
| Third-party TOML reader for config reads | stdlib `tomllib` | `tomllib` was added in Python 3.11. [CITED: https://docs.python.org/3/library/tomllib.html] | Python 3.12+ does not need `tomli` for reads. [VERIFIED: local Python 3.12.2] |
| Monolithic CLI file | Typer sub-app modules | Typer documents `app.add_typer()` for nested command groups. [CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/] | Each future domain can own its command file. [VERIFIED: Phase roadmap domains] |
| `typer-slim` for smaller installs | `typer` package | Typer's PyPI metadata states `typer-slim` stopped being supported since 0.22.0 and now installs all of Typer. [CITED: https://pypi.org/pypi/typer/json] | Do not choose `typer-slim`. [CITED: https://pypi.org/pypi/typer/json] |

**Deprecated/outdated:**

- `setup.py` as the primary packaging interface is not the right starting point for this greenfield package. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/]
- `tomli` is not needed for Python 3.12 TOML reads. [CITED: https://docs.python.org/3/library/tomllib.html]
- `typer-slim` is not useful as a distinct dependency after Typer 0.22.0. [CITED: https://pypi.org/pypi/typer/json]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Manifest code often becomes non-deterministic when current time is called inside constructors. | Common Pitfalls | Planner should still require deterministic clock injection because the phase goal explicitly requires deterministic research commands. |
| A2 | A single `cache/` helper without namespaces tends to mix raw and normalized outputs. | Common Pitfalls | Planner can remove this assumption if it creates explicit raw/normalized namespaces from the start. |

## Open Questions (RESOLVED)

1. **Should Phase 1 include real Parquet writes?** [VERIFIED: DATA-04 mentions normalized Parquet; VERIFIED: Phase 1 success criteria mention cached paths]
   - RESOLVED: Phase 1 should not include real Parquet writes. It should provide normalized Parquet path contracts and manifest references only. Real polars/pyarrow writes belong to the first ingestion phase that emits normalized tables. [VERIFIED: revision decision 2026-04-28]
   - What we know: The phase success criteria require write/read helpers for raw and normalized data paths, not source-specific normalized records. [VERIFIED: .planning/ROADMAP.md]
   - Plan impact: Phase 1 tests should assert deterministic `.parquet` path construction and manifest output references, without installing polars/pyarrow or writing dataframe files. [VERIFIED: revision decision 2026-04-28]

2. **What config file path should be the default?** [VERIFIED: Phase 1 includes config loading]
   - RESOLVED: When no config file is present, default config should use platformdirs-derived cache, config, and run roots. `load_config(path)` may read explicit TOML, and `ALBERT_` environment variables may override validated config values. Phase 1 must not require a repo-local `configs/default.toml` file to exist. [VERIFIED: revision decision 2026-04-28; CITED: https://platformdirs.readthedocs.io/en/latest/api.html]
   - What we know: Project constraints require reproducibility and local CLI usage. [VERIFIED: AGENTS.md]
   - Plan impact: `default_config()` should be sufficient on a clean checkout. Tests may create explicit temporary TOML files, but Phase 1 should not create or depend on `configs/default.toml`. [VERIFIED: revision decision 2026-04-28]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Package runtime | yes | 3.12.2 | None needed. [VERIFIED: `python3 --version`] |
| pip | Editable install and dependency install | yes | 26.0.1 | None needed. [VERIFIED: `python3 -m pip --version`] |
| pytest command | Local test runner | yes | 8.3.4 globally | Use `.venv` dev install to get pytest 9.0.3. [VERIFIED: `pytest --version`; VERIFIED: PyPI JSON lookup] |
| Typer | CLI framework | partial | local imports are stale versus PyPI latest | Install into `.venv`; do not rely on global site packages. [VERIFIED: local import/version probe; VERIFIED: PyPI JSON lookup] |
| Pydantic | Config and manifest models | partial | local version is 2.12.5 versus PyPI 2.13.3 | Install into `.venv`. [VERIFIED: local import/version probe; VERIFIED: PyPI JSON lookup] |
| pydantic-settings | Env override layer | partial | local version is 2.12.0 versus PyPI 2.14.0 | Install into `.venv`. [VERIFIED: local import/version probe; VERIFIED: PyPI JSON lookup] |
| platformdirs | Cache/config defaults | partial | local version is 4.3.7 versus PyPI 4.9.6 | Install into `.venv`. [VERIFIED: local import/version probe; VERIFIED: PyPI JSON lookup] |
| hatchling | Build backend | no | missing globally | Add to `build-system.requires`; pip will install it in build isolation. [VERIFIED: local import probe; CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/] |
| Ruff | Lint/format | no | missing globally | Add to dev dependencies and run via `.venv`. [VERIFIED: `command -v ruff`; VERIFIED: PyPI JSON lookup] |
| uv | Optional package workflow | no | missing | Use stdlib `venv` plus pip. [VERIFIED: `command -v uv`] |

**Missing dependencies with no fallback:**
- None for Phase 1, because stdlib `venv` plus pip is available. [VERIFIED: local environment probes]

**Missing dependencies with fallback:**
- `uv` is missing; use `python3 -m venv` and `python -m pip install -e ".[dev]"`. [VERIFIED: `command -v uv`]
- Global Ruff is missing; install it as a dev dependency. [VERIFIED: `command -v ruff`]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest; latest verified 9.0.3, local global command 8.3.4. [VERIFIED: PyPI JSON lookup; VERIFIED: `pytest --version`] |
| Config file | none currently; add pytest config to `pyproject.toml` in Wave 0. [VERIFIED: repo scan] |
| Quick run command | `python -m pytest tests/test_cli.py tests/test_config.py tests/test_cache.py tests/test_manifest.py -q` [VERIFIED: Phase 1 success criteria] |
| Full suite command | `python -m pytest -q` [CITED: https://typer.tiangolo.com/tutorial/testing/] |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DATA-04 | Config loading resolves explicit TOML values and default cache/output roots. [VERIFIED: .planning/ROADMAP.md] | unit | `python -m pytest tests/test_config.py -q` | no - Wave 0 [VERIFIED: repo scan] |
| DATA-04 | Raw and normalized cache path helpers construct deterministic paths under the configured root. [VERIFIED: .planning/ROADMAP.md] | unit | `python -m pytest tests/test_cache.py -q` | no - Wave 0 [VERIFIED: repo scan] |
| DATA-04 | Run manifest records config path, config hash, source endpoint, parameters, fetch time, covered time range, and outputs. [VERIFIED: .planning/REQUIREMENTS.md] | unit | `python -m pytest tests/test_manifest.py -q` | no - Wave 0 [VERIFIED: repo scan] |
| DATA-04 | Root CLI help lists `ingest`, `model`, `backtest`, and `report`. [VERIFIED: .planning/ROADMAP.md] | unit/smoke | `python -m pytest tests/test_cli.py -q` | no - Wave 0 [VERIFIED: repo scan] |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/test_cli.py tests/test_config.py tests/test_cache.py tests/test_manifest.py -q` [VERIFIED: Phase 1 scope]
- **Per wave merge:** `python -m pytest -q` and `python -m ruff check src tests` after Ruff is installed. [CITED: https://docs.astral.sh/ruff/configuration/]
- **Phase gate:** Full pytest suite green, root CLI help verified through `CliRunner`, and manifest JSON snapshot/value assertions passing. [VERIFIED: Phase 1 success criteria; CITED: https://typer.tiangolo.com/tutorial/testing/]

### Wave 0 Gaps

- [ ] `pyproject.toml` - package metadata, script entrypoint, build backend, dev dependencies, pytest config, and Ruff config. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/]
- [ ] `src/albert/cli.py` - root app and command groups. [CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/]
- [ ] `tests/test_cli.py` - CLI help wiring. [CITED: https://typer.tiangolo.com/tutorial/testing/]
- [ ] `tests/test_config.py` - config loading. [CITED: https://docs.python.org/3/library/tomllib.html]
- [ ] `tests/test_cache.py` - cache path helpers. [VERIFIED: DATA-04]
- [ ] `tests/test_manifest.py` - manifest schema and JSON creation. [VERIFIED: DATA-04]

## Security Domain

Security enforcement is enabled because `.planning/config.json` does not explicitly set `security_enforcement` to `false`. [VERIFIED: .planning/config.json]

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | Phase 1 has no users, sessions, or remote service authentication. [VERIFIED: .planning/ROADMAP.md] |
| V3 Session Management | no | Phase 1 is a local CLI with no sessions. [VERIFIED: .planning/ROADMAP.md] |
| V4 Access Control | no | Phase 1 has no multi-user authorization boundary. [VERIFIED: .planning/ROADMAP.md] |
| V5 Input Validation | yes | Validate config values and cache path segments with Pydantic models and allowlisted slugs. [CITED: https://docs.pydantic.dev/latest/concepts/serialization/; CITED: https://owasp.org/www-community/attacks/Path_Traversal] |
| V6 Cryptography | no | Phase 1 should not implement cryptography or secret storage. [VERIFIED: .planning/ROADMAP.md] |

### Known Threat Patterns for Python CLI Cache Foundations

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal through source, endpoint, table, or run identifiers | Tampering / Information Disclosure | Allowlist path segments, keep all writes under configured roots, and test adversarial path segments. [CITED: https://owasp.org/www-community/attacks/Path_Traversal] |
| Manifest spoofing or missing provenance fields | Repudiation | Make manifest fields explicit and required for endpoint, parameters, fetch time, covered range, raw path, normalized path, and output path. [VERIFIED: DATA-04] |
| Secret leakage into manifests | Information Disclosure | Do not include API keys or full environment dumps in manifests; include config path and non-secret source parameters only. [VERIFIED: Project trading safety constraints; ASSUMED] |
| Unsafe file overwrite outside run/cache directories | Tampering | Resolve paths under configured roots and reject absolute or parent-traversing dynamic path segments. [CITED: https://owasp.org/www-community/attacks/Path_Traversal] |

## Sources

### Primary (HIGH confidence)

- `.planning/REQUIREMENTS.md` - DATA-04 and traceability mapping. [VERIFIED: file read]
- `.planning/ROADMAP.md` - Phase 1 scope, success criteria, and plan list. [VERIFIED: file read]
- `.planning/PROJECT.md` and `AGENTS.md` - project definition and constraints. [VERIFIED: file read]
- `.planning/research/STACK.md`, `ARCHITECTURE.md`, `PITFALLS.md`, `FEATURES.md`, and `SUMMARY.md` - existing project research context. [VERIFIED: file read]
- PyPI JSON lookup - current package versions and upload timestamps for Typer, Pydantic, pydantic-settings, platformdirs, pytest, hatchling, and Ruff. [VERIFIED: escalated command]
- Python Packaging User Guide - `pyproject.toml`, `[project]`, and `[project.scripts]`. [CITED: https://packaging.python.org/en/latest/specifications/pyproject-toml/]
- Typer docs - command groups and CLI testing. [CITED: https://typer.tiangolo.com/tutorial/subcommands/add-typer/; CITED: https://typer.tiangolo.com/tutorial/testing/]
- Python `tomllib` docs - TOML parsing in stdlib. [CITED: https://docs.python.org/3/library/tomllib.html]
- Pydantic docs - serialization and settings. [CITED: https://docs.pydantic.dev/latest/concepts/serialization/; CITED: https://docs.pydantic.dev/latest/concepts/pydantic_settings/]
- platformdirs docs - user cache directory API. [CITED: https://platformdirs.readthedocs.io/en/latest/api.html]
- pytest docs - `tmp_path`. [CITED: https://docs.pytest.org/en/stable/how-to/tmp_path.html]
- Ruff docs - configuration in `pyproject.toml`/`ruff.toml`. [CITED: https://docs.astral.sh/ruff/configuration/]

### Secondary (MEDIUM confidence)

- OWASP path traversal guidance - filesystem input validation and path traversal risk. [CITED: https://owasp.org/www-community/attacks/Path_Traversal]
- OWASP ASVS project/developer guide - ASVS category context. [CITED: https://owasp.org/www-project-application-security-verification-standard/; CITED: https://devguide.owasp.org/en/06-verification/01-guides/03-asvs/]

### Tertiary (LOW confidence)

- Assumptions A1 and A2 in the Assumptions Log are based on engineering experience and should be validated by planner task design. [ASSUMED]

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - package versions were verified against PyPI JSON and core APIs were checked against official docs. [VERIFIED: PyPI JSON lookup; CITED: official docs listed above]
- Architecture: HIGH - phase is greenfield and project roadmap defines exact CLI/config/cache/manifest scope. [VERIFIED: repo scan; VERIFIED: .planning/ROADMAP.md]
- Pitfalls: MEDIUM - path traversal and stale dependency risks were verified, while two design-risk claims are explicitly marked assumed. [CITED: https://owasp.org/www-community/attacks/Path_Traversal; VERIFIED: local environment probes; ASSUMED]
- Validation: HIGH - pytest and Typer testing patterns are documented officially and no existing test files exist. [CITED: https://typer.tiangolo.com/tutorial/testing/; VERIFIED: repo scan]

**Research date:** 2026-04-28 [VERIFIED: current_date]
**Valid until:** 2026-05-28 for package versions, unless Typer/Pydantic/Ruff release cadence changes sooner. [ASSUMED]
