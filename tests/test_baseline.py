import tomllib
from pathlib import Path

import albert
from albert.manifest import SourceMetadata


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FIXTURE = ROOT / "tests" / "fixtures" / "config" / "example.toml"
SOURCE_METADATA_FIXTURE = (
    ROOT / "tests" / "fixtures" / "manifest" / "source_metadata.json"
)
SECRET_LIKE_TERMS = ("api_key", "token", "password", "secret")


def test_project_script_entrypoint_points_to_typer_app() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)

    assert pyproject["project"]["scripts"]["albert"] == "albert.cli:app"


def test_package_version_is_importable() -> None:
    assert albert.__version__


def test_source_metadata_fixture_validates_against_manifest_model() -> None:
    payload = SOURCE_METADATA_FIXTURE.read_text(encoding="utf-8")

    source = SourceMetadata.model_validate_json(payload)

    assert source.source == "kalshi"
    assert source.endpoint == "markets"
    assert source.parameters == {"ticker": "KXBTCD-26APR281200"}
    assert str(source.raw_path) == "cache/raw/kalshi/markets/example.json"


def test_fixture_files_do_not_contain_secret_like_terms() -> None:
    for fixture_path in (CONFIG_FIXTURE, SOURCE_METADATA_FIXTURE):
        fixture_text = fixture_path.read_text(encoding="utf-8").lower()

        for term in SECRET_LIKE_TERMS:
            assert term not in fixture_text


def test_phase_one_has_no_external_api_clients_yet() -> None:
    assert not (ROOT / "src" / "albert" / "clients").exists()
