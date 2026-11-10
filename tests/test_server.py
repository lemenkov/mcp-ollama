# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json

import httpx
import pytest
from fastmcp import Client

from mcp_ollama import server
from mcp_ollama.config import AppConfig


def basic(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def make_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "default_backend": "local",
            "backends": {
                "local": {
                    "base_url": "http://local.test",
                    "username": "u1",
                    "password": "p1",
                    "default_model": "m1",
                },
                "remote": {
                    "base_url": "https://remote.test/ollama",
                    "username": "u2",
                    "password": "p2",
                    "default_model": "m2",
                },
            },
        }
    )


def make_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "local.test":
            assert request.headers["authorization"] == basic("u1", "p1")
        elif host == "remote.test":
            assert request.headers["authorization"] == basic("u2", "p2")
        else:
            return httpx.Response(404)

        path = request.url.path
        if path.endswith("/api/version"):
            return httpx.Response(200, json={"version": "test"})
        if path.endswith("/api/tags"):
            model = "m1" if host == "local.test" else "m2"
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": model,
                            "size": 1,
                            "details": {"parameter_size": "1B"},
                        }
                    ]
                },
            )
        if path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": []})
        if path.endswith("/api/chat"):
            body = json.loads(request.content)
            assert body["model"] == "m2"
            return httpx.Response(
                200,
                json={
                    "model": "m2",
                    "message": {"role": "assistant", "content": "remote answer"},
                    "done": True,
                    "eval_count": 1,
                    "eval_duration": 100_000_000,
                },
            )
        return httpx.Response(404)

    return handler


@pytest.fixture(autouse=True)
async def reset_server():
    yield
    if server._registry is not None:
        await server._registry.close()
    server._registry = None
    server._jobs = None


async def test_backends_and_models_across_two_backends():
    server.init(make_config(), transport=httpx.MockTransport(make_handler()))

    async with Client(server.mcp) as client:
        backends = await client.call_tool("ollama_backends", {"probe": True})
        assert backends.data["default_backend"] == "local"
        assert len(backends.data["backends"]) == 2
        assert all(item["reachable"] for item in backends.data["backends"])
        assert all("models" not in item for item in backends.data["backends"])

        models = await client.call_tool("ollama_models", {})
        assert len(models.data["backends"]) == 2
        names = {
            item["backend"]: item["models"][0]["name"]
            for item in models.data["backends"]
        }
        assert names == {"local": "m1", "remote": "m2"}


async def test_chat_selects_backend_and_auth():
    server.init(make_config(), transport=httpx.MockTransport(make_handler()))

    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "ollama_chat",
            {
                "backend": "remote",
                "messages": [{"role": "user", "content": "hi"}],
                "wait_seconds": 5,
            },
        )
        assert result.data["status"] == "done"
        assert result.data["backend"] == "remote"
        assert result.data["result"]["response"] == "remote answer"


async def test_unknown_backend_is_an_error_payload():
    server.init(make_config(), transport=httpx.MockTransport(make_handler()))

    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "ollama_models", {"backend": "missing"}
        )
        assert result.data["status"] == "error"
        assert "unknown or disabled backend" in result.data["error"]


async def test_401_is_an_error_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="nope")

    config = AppConfig.model_validate(
        {
            "default_backend": "local",
            "backends": {
                "local": {
                    "base_url": "http://local.test",
                    "username": "u",
                    "password": "p",
                }
            },
        }
    )
    server.init(config, transport=httpx.MockTransport(handler))

    async with Client(server.mcp) as client:
        result = await client.call_tool("ollama_models", {})
        item = result.data["backends"][0]
        assert "authentication failed" in item["error"]


async def test_generate_job_polling():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/generate"):
            await asyncio.sleep(0.15)
            return httpx.Response(
                200,
                json={
                    "model": "m1",
                    "response": "late answer",
                    "done": True,
                    "eval_count": 2,
                    "eval_duration": 200_000_000,
                },
            )
        return httpx.Response(404)

    server.init(make_config(), transport=httpx.MockTransport(handler))

    async with Client(server.mcp) as client:
        started = await client.call_tool(
            "ollama_generate",
            {"prompt": "hi", "wait_seconds": 0},
        )
        assert started.data["status"] in ("queued", "running")
        job_id = started.data["job_id"]

        for _ in range(100):
            status = await client.call_tool(
                "ollama_job_status", {"job_id": job_id}
            )
            if status.data["status"] == "done":
                break
            await asyncio.sleep(0.02)

        assert status.data["status"] == "done"
        result = await client.call_tool(
            "ollama_job_result", {"job_id": job_id}
        )
        assert result.data["result"]["response"] == "late answer"
