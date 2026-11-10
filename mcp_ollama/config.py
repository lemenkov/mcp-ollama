# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Ollama MCP server.

The backend list is a JSON document, normally installed as
``/etc/sysconfig/mcp-ollama.json``.  The systemd unit passes it to the service
as a credential (``LoadCredential=backends.json:...``), so the file can stay
root-only while the service runs under ``DynamicUser``.

Every backend is addressed by its map key; the MCP tools never accept a URL
from the model, which keeps the server from becoming an SSRF proxy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_CONFIG_PATH = "/etc/sysconfig/mcp-ollama.json"


class ConfigError(RuntimeError):
    """Raised when the backend configuration cannot be loaded or validated."""


class BackendConfig(BaseModel):
    """One Ollama endpoint."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    base_url: str = Field(..., description="http:// or https:// URL, optionally with a path prefix")
    username: str | None = Field(default=None, description="HTTP Basic auth user")
    password: str | None = Field(default=None, description="HTTP Basic auth password")
    password_file: str | None = Field(default=None, description="Read the password from this file")
    default_model: str | None = Field(default=None, description="Model used when a tool omits one")
    verify_tls: bool | str = Field(default=True, description="true, false, or a CA bundle path")
    timeout_seconds: float = Field(default=600.0, ge=1.0, le=3600.0)
    connect_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    max_concurrency: int = Field(default=1, ge=1, le=8)
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    description: str | None = None

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https"):
            raise ValueError("base_url must start with http:// or https://")
        if not parts.hostname:
            raise ValueError("base_url must include a hostname")
        if parts.query or parts.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        return value

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, value: dict[str, str]) -> dict[str, str]:
        for name, header_value in value.items():
            if any(ch in name or ch in header_value for ch in "\r\n"):
                raise ValueError("header names and values must not contain newlines")
        return value

    @model_validator(mode="after")
    def _check_password_sources(self) -> "BackendConfig":
        if self.password is not None and self.password_file is not None:
            raise ValueError("set only one of password or password_file")
        if self.username is not None and not self.username:
            self.username = None
        return self


class AppConfig(BaseModel):
    """Top-level backend configuration."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    default_backend: str = Field(..., min_length=1, max_length=64)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    job_ttl_seconds: int = Field(default=3600, ge=60, le=86_400)
    max_jobs: int = Field(default=100, ge=1, le=1000)
    result_max_chars: int = Field(default=20_000, ge=1_000, le=200_000)

    @field_validator("backends")
    @classmethod
    def _validate_backend_names(cls, value: dict[str, BackendConfig]) -> dict[str, BackendConfig]:
        if not value:
            raise ValueError("at least one backend must be configured")
        for name in value:
            if not name or len(name) > 64:
                raise ValueError("backend names must be 1..64 characters")
            if not all(ch.isalnum() or ch in "._-" for ch in name):
                raise ValueError(
                    f"backend name {name!r} may contain only letters, digits, '.', '_' and '-'"
                )
        return value

    @model_validator(mode="after")
    def _check_default_backend(self) -> "AppConfig":
        if self.default_backend not in self.backends:
            raise ValueError(
                f"default_backend {self.default_backend!r} is not present in backends"
            )
        if not self.backends[self.default_backend].enabled:
            raise ValueError(f"default_backend {self.default_backend!r} is disabled")
        if not any(backend.enabled for backend in self.backends.values()):
            raise ValueError("at least one backend must be enabled")
        return self


def config_path(explicit: str | None = None) -> Path:
    """Resolve the config path, preferring the systemd credential directory."""

    if explicit:
        return Path(explicit)

    from_env = os.environ.get("OLLAMA_MCP_CONFIG")
    if from_env:
        return Path(from_env)

    credentials_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if credentials_dir:
        candidate = Path(credentials_dir) / "backends.json"
        if candidate.is_file():
            return candidate

    return Path(DEFAULT_CONFIG_PATH)


def redact_url(url: str) -> str:
    """Return *url* without userinfo, for logs and tool output."""

    parts = urlsplit(url)
    netloc = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def split_userinfo(url: str) -> tuple[str | None, str | None, str]:
    """Split ``user:pass@host`` out of *url*.

    Returns ``(username, password, clean_url)``.  The clean URL never contains
    credentials, so it is safe to log and to use as an httpx base URL.
    """

    parts = urlsplit(url)
    username = unquote(parts.username) if parts.username is not None else None
    password = unquote(parts.password) if parts.password is not None else None
    netloc = parts.netloc.rsplit("@", 1)[-1]
    clean = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    return (username or None, password, clean)


def normalize_base_url(url: str) -> str:
    """Ensure the base URL has a trailing slash so httpx joins paths correctly."""

    parts = urlsplit(url)
    netloc = parts.netloc.rsplit("@", 1)[-1]
    path = parts.path or "/"
    if not path.endswith("/"):
        path += "/"
    return urlunsplit((parts.scheme, netloc, path, "", ""))


def load_config(explicit: str | None = None) -> tuple[AppConfig, Path]:
    """Load and validate the backend configuration.

    ``password_file`` paths are resolved relative to the configuration file,
    which makes the file usable both from ``/etc/sysconfig`` and from a
    development checkout.
    """

    path = config_path(explicit)
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(
            f"backend configuration not found: {path}. "
            "Install /etc/sysconfig/mcp-ollama.json or set OLLAMA_MCP_CONFIG."
        ) from exc
    except OSError as exc:
        raise ConfigError(f"cannot read backend configuration {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid backend configuration {path}: {exc}") from exc

    for name, backend in config.backends.items():
        if backend.password_file is None:
            continue
        password_path = Path(backend.password_file)
        if not password_path.is_absolute():
            password_path = path.parent / password_path
        try:
            backend.password = password_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(
                f"backend {name!r}: cannot read password_file {password_path}: {exc}"
            ) from exc

    return config, path
