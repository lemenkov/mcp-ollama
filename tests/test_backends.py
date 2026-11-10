# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import base64

import httpx
import pytest

from mcp_ollama.backends import (
    Backend,
    BackendError,
    Registry,
    UnknownBackend,
    summarize_tags,
)
from mcp_ollama.config import AppConfig, BackendConfig


def basic(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def backend_config(**overrides) -> BackendConfig:
    data = {
        "base_url": "https://example.com/ollama",
        "username": "user",
        "password": "pass",
        "default_model": "llama3.2:3b",
    }
    data.update(overrides)
    return BackendConfig.model_validate(data)


async def test_basic_auth_and_path_prefix():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["authorization"] == basic("user", "pass")
        assert request.url.path == "/ollama/api/tags"
        return httpx.Response(200, json={"models": []})

    backend = Backend(
        "remote",
        backend_config(),
        transport=httpx.MockTransport(handler),
    )
    data = await backend.tags()
    assert data == {"models": []}
    assert backend.display_url == "https://example.com/ollama/"
    assert len(seen) == 1
    await backend.close()


async def test_userinfo_is_extracted_and_redacted():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == basic("alice", "p@ss")
        return httpx.Response(200, json={"version": "1"})

    backend = Backend(
        "remote",
        backend_config(
            base_url="https://alice:p%40ss@example.com/ollama",
            username=None,
            password=None,
        ),
        transport=httpx.MockTransport(handler),
    )
    assert "alice" not in backend.display_url
    assert "p@ss" not in backend.display_url
    assert await backend.version() == {"version": "1"}
    await backend.close()


async def test_http_401_is_actionable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="nope")

    backend = Backend(
        "remote",
        backend_config(),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BackendError) as excinfo:
        await backend.tags()
    assert excinfo.value.status_code == 401
    assert "authentication failed" in str(excinfo.value)
    await backend.close()


async def test_connect_error_is_actionable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    backend = Backend(
        "remote",
        backend_config(),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BackendError) as excinfo:
        await backend.tags()
    assert "unreachable" in str(excinfo.value)
    await backend.close()


async def test_probe_reports_status_without_model_list():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.12.11"})
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "llama3.2:3b",
                            "size": 1,
                            "details": {"parameter_size": "3B"},
                        }
                    ]
                },
            )
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3.2:3b", "size": 1, "size_vram": 0}]},
            )
        return httpx.Response(404)

    backend = Backend(
        "local",
        backend_config(base_url="http://127.0.0.1:11434", username=None, password=None),
        transport=httpx.MockTransport(handler),
    )
    probe = await backend.probe()
    assert probe["reachable"] is True
    assert probe["version"] == "0.12.11"
    assert probe["models_count"] == 1
    assert probe["loaded"][0]["name"] == "llama3.2:3b"
    assert "models" not in probe
    await backend.close()


async def test_probe_failure_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    backend = Backend(
        "remote",
        backend_config(),
        transport=httpx.MockTransport(handler),
    )
    probe = await backend.probe()
    assert probe["reachable"] is False
    assert "HTTP 503" in probe["error"]
    await backend.close()


def test_registry_unknown_backend():
    config = AppConfig.model_validate(
        {
            "default_backend": "local",
            "backends": {
                "local": {"base_url": "http://127.0.0.1:11434"},
            },
        }
    )
    registry = Registry(config)
    with pytest.raises(UnknownBackend):
        registry.get("missing")


def test_summarize_tags():
    models = summarize_tags(
        {
            "models": [
                {
                    "name": "qwen3:4b",
                    "size": 42,
                    "modified_at": "2026-01-01T00:00:00Z",
                    "details": {
                        "family": "qwen3",
                        "parameter_size": "4B",
                        "quantization_level": "Q4_K_M",
                    },
                }
            ]
        }
    )
    assert models == [
        {
            "name": "qwen3:4b",
            "size": 42,
            "modified_at": "2026-01-01T00:00:00Z",
            "family": "qwen3",
            "parameter_size": "4B",
            "quantization": "Q4_K_M",
        }
    ]
