# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""HTTP access to one or more Ollama endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from .config import BackendConfig, AppConfig, normalize_base_url, redact_url, split_userinfo

USER_AGENT = "mcp-ollama/0.1.0"


class BackendError(RuntimeError):
    """A backend could not be reached or returned an error."""

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.status_code = status_code


class UnknownBackend(BackendError):
    """The requested backend name is not in the configuration."""


class Backend:
    """A configured Ollama endpoint and its shared HTTP client."""

    def __init__(
        self,
        name: str,
        config: BackendConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self.config = config

        username, password = config.username, config.password
        url_user, url_password, clean_url = split_userinfo(config.base_url)
        if username is None and url_user:
            username = url_user
            password = password if password is not None else url_password
        if username is not None and password is None:
            password = ""

        self.base_url = normalize_base_url(clean_url)
        self.display_url = redact_url(self.base_url)

        auth = httpx.BasicAuth(username, password) if username else None
        timeout = httpx.Timeout(
            config.timeout_seconds,
            connect=config.connect_timeout_seconds,
        )
        headers = {"User-Agent": USER_AGENT, **config.headers}
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=auth,
            verify=config.verify_tls,
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            transport=transport,
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrency)

    async def close(self) -> None:
        await self.client.aclose()

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Issue a request and return decoded JSON, with actionable errors."""

        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise BackendError(
                f"backend {self.name!r} timed out after "
                f"{self.config.timeout_seconds:g}s ({self.display_url})",
                backend=self.name,
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendError(
                f"backend {self.name!r} is unreachable at {self.display_url}: {exc}",
                backend=self.name,
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = " ".join((exc.response.text or "").split())[:300]
            if status in (401, 403):
                hint = "authentication failed; check username/password or the reverse-proxy ACL"
            elif status == 404:
                hint = "endpoint or model not found"
            elif status == 429:
                hint = "backend is rate-limiting requests"
            elif status >= 500:
                hint = "backend error"
            else:
                hint = "request rejected"
            message = f"backend {self.name!r} returned HTTP {status} ({hint})"
            if body:
                message += f": {body}"
            raise BackendError(message, backend=self.name, status_code=status) from exc
        except httpx.RequestError as exc:
            raise BackendError(
                f"backend {self.name!r} request failed: {exc}",
                backend=self.name,
            ) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise BackendError(
                f"backend {self.name!r} returned invalid JSON: {exc}",
                backend=self.name,
            ) from exc

        if not isinstance(payload, dict):
            raise BackendError(
                f"backend {self.name!r} returned an unexpected JSON type "
                f"({type(payload).__name__})",
                backend=self.name,
            )
        return payload

    async def version(self) -> dict[str, Any]:
        return await self.request("GET", "api/version")

    async def tags(self) -> dict[str, Any]:
        return await self.request("GET", "api/tags")

    async def ps(self) -> dict[str, Any]:
        return await self.request("GET", "api/ps")

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        return await self.request("POST", "api/chat", json=payload)

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        return await self.request("POST", "api/generate", json=payload)

    async def embed(
        self,
        model: str,
        input_data: str | list[str],
        *,
        truncate: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": input_data,
            "truncate": truncate,
        }
        return await self.request("POST", "api/embed", json=payload)

    async def probe(self, timeout: float = 3.0) -> dict[str, Any]:
        """Return a compact reachability/status summary for a backend."""

        started = time.perf_counter()

        async def collect() -> dict[str, Any]:
            version = await self.version()
            tags = await self.tags()
            try:
                ps = await self.ps()
            except BackendError as exc:
                if exc.status_code == 404:
                    ps = {"models": []}
                else:
                    raise
            return {
                "version": version.get("version"),
                "models": summarize_tags(tags),
                "loaded": summarize_ps(ps),
            }

        try:
            data = await asyncio.wait_for(collect(), timeout=timeout)
        except asyncio.TimeoutError:
            return {
                "name": self.name,
                "url": self.display_url,
                "reachable": False,
                "error": f"probe timed out after {timeout:g}s",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except BackendError as exc:
            return {
                "name": self.name,
                "url": self.display_url,
                "reachable": False,
                "error": str(exc),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

        return {
            "name": self.name,
            "url": self.display_url,
            "reachable": True,
            "version": data["version"],
            "models_count": len(data["models"]),
            "loaded": data["loaded"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }


class Registry:
    """The set of configured backends."""

    def __init__(
        self,
        config: AppConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.default_backend = config.default_backend
        self.backends: dict[str, Backend] = {
            name: Backend(name, backend_config, transport=transport)
            for name, backend_config in config.backends.items()
            if backend_config.enabled
        }

    def names(self) -> list[str]:
        return sorted(self.backends)

    def get(self, name: str | None = None) -> Backend:
        key = name or self.default_backend
        backend = self.backends.get(key)
        if backend is None:
            available = ", ".join(self.names()) or "(none)"
            raise UnknownBackend(
                f"unknown or disabled backend {key!r}; available backends: {available}",
                backend=key,
            )
        return backend

    async def close(self) -> None:
        await asyncio.gather(*(backend.close() for backend in self.backends.values()))

    async def probe_all(self, timeout: float = 3.0) -> list[dict[str, Any]]:
        return await asyncio.gather(
            *(backend.probe(timeout=timeout) for backend in self.backends.values())
        )


def summarize_tags(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the useful fields from /api/tags."""

    models: list[dict[str, Any]] = []
    for item in data.get("models") or []:
        details = item.get("details") or {}
        models.append(
            {
                "name": item.get("name") or item.get("model"),
                "size": item.get("size"),
                "modified_at": item.get("modified_at"),
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
            }
        )
    return models


def summarize_ps(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the useful fields from /api/ps."""

    loaded: list[dict[str, Any]] = []
    for item in data.get("models") or []:
        loaded.append(
            {
                "name": item.get("name") or item.get("model"),
                "size": item.get("size"),
                "size_vram": item.get("size_vram"),
                "expires_at": item.get("expires_at"),
            }
        )
    return loaded


def tokens_per_second(data: dict[str, Any]) -> float | None:
    """Ollama reports durations in nanoseconds."""

    count = data.get("eval_count")
    duration = data.get("eval_duration")
    if not count or not duration:
        return None
    return round(count / (duration / 1_000_000_000), 2)


def summarize_chat(data: dict[str, Any]) -> dict[str, Any]:
    message = data.get("message") or {}
    result: dict[str, Any] = {
        "response": message.get("content", ""),
        "model": data.get("model"),
        "done": data.get("done"),
        "done_reason": data.get("done_reason"),
        "eval_count": data.get("eval_count"),
        "eval_duration_ns": data.get("eval_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
        "total_duration_ns": data.get("total_duration"),
        "load_duration_ns": data.get("load_duration"),
        "tokens_per_second": tokens_per_second(data),
    }
    if message.get("thinking"):
        result["thinking"] = message["thinking"]
    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]
    return result


def summarize_generate(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "response": data.get("response", ""),
        "model": data.get("model"),
        "done": data.get("done"),
        "done_reason": data.get("done_reason"),
        "eval_count": data.get("eval_count"),
        "eval_duration_ns": data.get("eval_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
        "total_duration_ns": data.get("total_duration"),
        "load_duration_ns": data.get("load_duration"),
        "tokens_per_second": tokens_per_second(data),
    }
    if data.get("thinking"):
        result["thinking"] = data["thinking"]
    return result


def summarize_embed(data: dict[str, Any]) -> dict[str, Any]:
    embeddings = data.get("embeddings") or []
    dimensions = len(embeddings[0]) if embeddings else 0
    return {
        "model": data.get("model"),
        "count": len(embeddings),
        "dimensions": dimensions,
        "embeddings": embeddings,
        "total_duration_ns": data.get("total_duration"),
        "load_duration_ns": data.get("load_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
    }
