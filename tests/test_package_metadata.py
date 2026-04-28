from pathlib import Path
import tomllib


def test_package_metadata_defines_installable_albert_project() -> None:
    pyproject_path = Path("pyproject.toml")

    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)

    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert pyproject["project"]["name"] == "albert"
    assert pyproject["project"]["requires-python"] == ">=3.12"
    assert pyproject["project"]["scripts"]["albert"] == "albert.cli:app"
    assert "typer>=0.25.0" in pyproject["project"]["dependencies"]
    assert "pydantic>=2.13.3" in pyproject["project"]["dependencies"]
    assert "pydantic-settings>=2.14.0" in pyproject["project"]["dependencies"]
    assert "platformdirs>=4.9.6" in pyproject["project"]["dependencies"]
    assert "pytest>=9.0.3" in pyproject["project"]["optional-dependencies"]["dev"]
    assert "ruff>=0.15.12" in pyproject["project"]["optional-dependencies"]["dev"]


def test_package_version_surface() -> None:
    version_file = Path("src/albert/__init__.py")

    assert '__version__ = "0.1.0"' in version_file.read_text()
