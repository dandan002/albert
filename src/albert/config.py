import tomllib
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_config_dir
from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SECRET_KEY_PARTS = ("secret", "token", "password", "api_key", "apikey")


def _platform_cache_dir() -> Path:
    return Path(user_cache_dir("albert", appauthor=False))


def _platform_run_dir() -> Path:
    return _platform_cache_dir() / "runs"


def _platform_config_dir() -> Path:
    return Path(user_config_dir("albert", appauthor=False))


def _secret_like_keys(data: Any, prefix: str = "") -> list[str]:
    if not isinstance(data, dict):
        return []

    matches: list[str] = []
    for key, value in data.items():
        key_text = str(key).lower()
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if any(part in key_text for part in SECRET_KEY_PARTS):
            matches.append(dotted_key)
        matches.extend(_secret_like_keys(value, dotted_key))
    return matches


class AlbertConfig(BaseSettings):
    cache_dir: Path = Field(default_factory=_platform_cache_dir)
    run_dir: Path = Field(default_factory=_platform_run_dir)
    config_dir: Path = Field(default_factory=_platform_config_dir)
    timezone: str = "UTC"

    model_config = SettingsConfigDict(
        env_prefix="ALBERT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    @model_validator(mode="before")
    @classmethod
    def reject_secret_like_keys(cls, data: Any) -> Any:
        matches = _secret_like_keys(data)
        if matches:
            joined = ", ".join(sorted(matches))
            raise ValueError(f"secret-like config keys are not allowed: {joined}")
        return data

    @field_validator("cache_dir", "run_dir", "config_dir", mode="before")
    @classmethod
    def validate_path(cls, value: Any) -> Path:
        if value is None:
            raise ValueError("path cannot be null")
        if isinstance(value, str):
            if value == "":
                raise ValueError("path cannot be empty")
            if "\x00" in value:
                raise ValueError("path cannot contain null bytes")
        path = Path(value)
        if any(part == ".." for part in path.parts):
            raise ValueError("path cannot contain parent traversal")
        return path


def default_config() -> AlbertConfig:
    return AlbertConfig(
        cache_dir=Path(user_cache_dir("albert", appauthor=False)),
        run_dir=Path(user_cache_dir("albert", appauthor=False)) / "runs",
        config_dir=Path(user_config_dir("albert", appauthor=False)),
    )


def load_config(path: Path | None = None) -> AlbertConfig:
    if path is None:
        return default_config()

    defaults = default_config().model_dump()
    with path.open("rb") as file:
        toml_values = tomllib.load(file)

    return AlbertConfig(**(defaults | toml_values))


__all__ = ["AlbertConfig", "ValidationError", "default_config", "load_config"]
