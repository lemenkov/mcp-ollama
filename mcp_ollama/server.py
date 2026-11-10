# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""FastMCP server for one or more Ollama endpoints.

The backend list comes from a JSON file (normally
``/etc/sysconfig/mcp-ollama.json``).  The model chooses a backend by name; the
server never accepts an arbitrary URL, so it cannot be turned into an SSRF
proxy by a tool call.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from typing import Any, Literal

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from .backends import (
    Backend,
    BackendError,
    Registry,
    summarize_chat,
    summarize_embed,
    summarize_generate,
    summarize_tags,
)
from .config import AppConfig, ConfigError, load_config
from .jobs import JobCapacityError, JobManager, JobNotFound

LOG = logging.getLogger("mcp_ollama")

mcp = FastMCP("Ollama")

_registry: Registry | None = None
_jobs: JobManager | None = None

MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
MAX_MESSAGES = 100
MAX_MESSAGE_CHARS = 100_000
MAX_TOTAL_CHARS = 200_000
MAX_SYSTEM_CHARS = 20_000
MAX_EMBED_INPUTS = 8
MAX_EMBED_CHARS = 16_000


class ChatMessage(BaseModel):
    """One chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="Message role"
    )
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)


class GenerationOptions(BaseModel):
    """Ollama generation options; only the useful subset is exposed."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    num_predict: int | None = Field(default=None, ge=1, le=8192)
    num_ctx: int | None = Field(default=None, ge=512, le=32768)
    num_thread: int | None = Field(default=None, ge=1, le=16)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    stop: list[str] | None = Field(default=None, max_length=8)


def _get_registry() -> Registry:
    if _registry is None:
        raise RuntimeError("server is not initialized")
    return _registry


def _get_jobs() -> JobManager:
    if _jobs is None:
        raise RuntimeError("server is not initialized")
    return _jobs


def init(
    config: AppConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Registry:
    """Initialize the global registry and job manager (also used by tests)."""

    global _registry, _jobs
    _registry = Registry(config, transport=transport)
    _jobs = JobManager(
        ttl_seconds=config.job_ttl_seconds,
        max_jobs=config.max_jobs,
    )
    return _registry


def _error(exc: Exception) -> dict[str, Any]:
    LOG.warning("tool error: %s", exc)
    return {"status": "error", "error": str(exc) or type(exc).__name__}


def _wait_seconds(value: int) -> int:
    return max(0, min(int(value), 120))


def _model_name(model: str | None, backend: Backend) -> str:
    name = model or backend.config.default_model
    if not name:
        raise ValueError(
            f"no model specified and backend {backend.name!r} has no default_model; "
            "pass model explicitly"
        )
    if not MODEL_RE.match(name):
        raise ValueError(
            f"invalid model name {name!r}; expected letters, digits, '.', '_', ':', '/' or '-'"
        )
    return name


def _resolve(backend: str | None, model: str | None) -> tuple[Backend, str]:
    resolved = _get_registry().get(backend)
    return resolved, _model_name(model, resolved)


def _options_dict(options: GenerationOptions | None) -> dict[str, Any]:
    if options is None:
        return {}
    data = options.model_dump(exclude_none=True)
    if data.get("stop") == []:
        data.pop("stop")
    return data


def _validate_messages(messages: list[ChatMessage], system: str | None) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("messages must contain at least one message")
    if len(messages) > MAX_MESSAGES:
        raise ValueError(f"too many messages ({len(messages)}); maximum is {MAX_MESSAGES}")
    if system is not None and len(system) > MAX_SYSTEM_CHARS:
        raise ValueError(f"system prompt is too long; maximum is {MAX_SYSTEM_CHARS} characters")
    total = sum(len(message.content) for message in messages)
    if total > MAX_TOTAL_CHARS:
        raise ValueError(
            f"messages contain {total} characters; maximum is {MAX_TOTAL_CHARS}"
        )
    if system and any(message.role == "system" for message in messages):
        raise ValueError(
            "provide the system prompt either as a message or via the system parameter, not both"
        )

    payload = [message.model_dump() for message in messages]
    if system:
        payload.insert(0, {"role": "system", "content": system})
    return payload


async def _submit(
    *,
    kind: str,
    backend: Backend,
    model: str,
    runner,
    wait_seconds: int,
) -> dict[str, Any]:
    config = _get_registry().config
    return await _get_jobs().submit(
        kind=kind,
        backend=backend.name,
        model=model,
        runner=runner,
        wait_seconds=_wait_seconds(wait_seconds),
        result_max_chars=config.result_max_chars,
    )


@mcp.tool(
    tags={"read"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ollama_backends(probe: bool = True, timeout_seconds: float = 3.0) -> dict[str, Any]:
    """List configured Ollama backends.

    By default each backend is probed for its version, model count and currently
    loaded models.  Use the returned ``name`` as the ``backend`` argument of the
    other tools.  Probe failures are reported per backend and do not fail the
    whole call.

    Args:
        probe: Probe each backend over HTTP (default true).
        timeout_seconds: Per-backend probe timeout, capped at 10s.
    """

    try:
        registry = _get_registry()
        timeout = min(max(float(timeout_seconds), 1.0), 10.0)
        if probe:
            backends = await registry.probe_all(timeout=timeout)
        else:
            backends = [
                {
                    "name": backend.name,
                    "url": backend.display_url,
                    "default_model": backend.config.default_model,
                    "description": backend.config.description,
                    "reachable": None,
                }
                for backend in registry.backends.values()
            ]
        return {
            "default_backend": registry.default_backend,
            "backends": backends,
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    tags={"read"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ollama_models(backend: str | None = None) -> dict[str, Any]:
    """List models available on one backend, or on every backend if omitted.

    Args:
        backend: Backend name from ollama_backends; omit to query all backends.
    """

    registry = _get_registry()

    async def one(name: str, item: Backend) -> dict[str, Any]:
        try:
            data = await item.tags()
            return {
                "backend": name,
                "url": item.display_url,
                "models": summarize_tags(data),
            }
        except BackendError as exc:
            return {
                "backend": name,
                "url": item.display_url,
                "error": str(exc),
            }

    try:
        if backend is not None:
            item = registry.get(backend)
            return await one(item.name, item)

        results = await asyncio.gather(
            *(one(name, item) for name, item in registry.backends.items())
        )
        return {
            "default_backend": registry.default_backend,
            "backends": results,
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    tags={"inference"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ollama_chat(
    messages: list[ChatMessage],
    backend: str | None = None,
    model: str | None = None,
    system: str | None = None,
    options: GenerationOptions | None = None,
    wait_seconds: int = 30,
) -> dict[str, Any]:
    """Chat with an Ollama model.

    The call waits up to ``wait_seconds`` (0-120).  If the answer is ready it is
    returned directly; otherwise the result contains a ``job_id`` and the work
    continues in the background.  Poll it with ollama_job_status and fetch the
    answer with ollama_job_result.

    Args:
        messages: Ordered chat messages; roles are system, user or assistant.
        backend: Backend name from ollama_backends; defaults to the configured default.
        model: Model name; defaults to the backend's default_model.
        system: Optional system prompt (do not also put a system message in messages).
        options: Sampling and context options.
        wait_seconds: Seconds to wait for completion before returning a job id.
    """

    try:
        item, name = _resolve(backend, model)
        payload = _validate_messages(messages, system)
        opts = _options_dict(options)

        async def runner() -> dict[str, Any]:
            async with item.semaphore:
                data = await item.chat(name, payload, opts)
            return {"backend": item.name, **summarize_chat(data)}

        return await _submit(
            kind="chat",
            backend=item,
            model=name,
            runner=runner,
            wait_seconds=wait_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    tags={"inference"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ollama_generate(
    prompt: str,
    backend: str | None = None,
    model: str | None = None,
    system: str | None = None,
    options: GenerationOptions | None = None,
    wait_seconds: int = 30,
) -> dict[str, Any]:
    """Generate a completion for a single prompt.

    This is the convenient tool for rewriting/translation-style work.  It uses
    the same bounded wait and job-id behaviour as ollama_chat.

    Args:
        prompt: The user prompt.
        backend: Backend name from ollama_backends.
        model: Model name; defaults to the backend's default_model.
        system: Optional system prompt.
        options: Sampling and context options.
        wait_seconds: Seconds to wait for completion before returning a job id.
    """

    try:
        if not prompt:
            raise ValueError("prompt must not be empty")
        if len(prompt) > MAX_MESSAGE_CHARS:
            raise ValueError(f"prompt is too long; maximum is {MAX_MESSAGE_CHARS} characters")
        if system is not None and len(system) > MAX_SYSTEM_CHARS:
            raise ValueError(f"system prompt is too long; maximum is {MAX_SYSTEM_CHARS}")
        item, name = _resolve(backend, model)
        opts = _options_dict(options)

        async def runner() -> dict[str, Any]:
            async with item.semaphore:
                data = await item.generate(name, prompt, system=system, options=opts)
            return {"backend": item.name, **summarize_generate(data)}

        return await _submit(
            kind="generate",
            backend=item,
            model=name,
            runner=runner,
            wait_seconds=wait_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    tags={"inference"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ollama_embed(
    input: str | list[str],
    backend: str | None = None,
    model: str | None = None,
    truncate: bool = True,
    wait_seconds: int = 30,
) -> dict[str, Any]:
    """Compute embeddings for one or more short inputs.

    Vectors are returned inline, so this is intentionally limited to 8 inputs and
    16,000 characters total.  Use a dedicated pipeline for large corpora.

    Args:
        input: A string or up to 8 strings.
        backend: Backend name from ollama_backends.
        model: Embedding model name; defaults to the backend's default_model.
        truncate: Ask Ollama to truncate inputs longer than the context.
        wait_seconds: Seconds to wait before returning a job id.
    """

    try:
        if isinstance(input, str):
            inputs = [input] if input else []
        else:
            inputs = list(input)
        if not inputs:
            raise ValueError("input must contain at least one string")
        if len(inputs) > MAX_EMBED_INPUTS:
            raise ValueError(f"too many inputs ({len(inputs)}); maximum is {MAX_EMBED_INPUTS}")
        total = sum(len(value) for value in inputs)
        if total > MAX_EMBED_CHARS:
            raise ValueError(
                f"input contains {total} characters; maximum is {MAX_EMBED_CHARS}"
            )

        item, name = _resolve(backend, model)
        payload: str | list[str] = inputs[0] if isinstance(input, str) else inputs

        async def runner() -> dict[str, Any]:
            async with item.semaphore:
                data = await item.embed(name, payload, truncate=truncate)
            return {"backend": item.name, **summarize_embed(data)}

        return await _submit(
            kind="embed",
            backend=item,
            model=name,
            runner=runner,
            wait_seconds=wait_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    tags={"jobs"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ollama_job_status(job_id: str) -> dict[str, Any]:
    """Return the status of a job created by an inference tool.

    Args:
        job_id: The job_id returned by ollama_chat, ollama_generate or ollama_embed.
    """

    try:
        return await _get_jobs().status(job_id)
    except JobNotFound as exc:
        return _error(exc)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    tags={"jobs"},
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ollama_job_result(job_id: str, max_chars: int = 20_000) -> dict[str, Any]:
    """Fetch the result of a completed inference job.

    Long text fields are truncated to ``max_chars``; the response marks this with
    ``"truncated": true`` and ``original_chars``.

    Args:
        job_id: The job_id returned by an inference tool.
        max_chars: Maximum characters of generated text to return (1,000-200,000).
    """

    try:
        limit = min(max(int(max_chars), 1_000), 200_000)
        return await _get_jobs().result(job_id, max_chars=limit)
    except JobNotFound as exc:
        return _error(exc)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    tags={"jobs"},
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ollama_cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a queued or running inference job.

    Cancelling an already-finished job is harmless and returns its final status.

    Args:
        job_id: The job_id returned by an inference tool.
    """

    try:
        return await _get_jobs().cancel(job_id)
    except JobNotFound as exc:
        return _error(exc)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server for Ollama")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8815")))
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=os.environ.get("MCP_TRANSPORT", "streamable-http"),
    )
    parser.add_argument("--config", default=None, help="Path to backends.json")
    args = parser.parse_args()

    level_name = os.environ.get("OLLAMA_MCP_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        config, path = load_config(args.config)
    except ConfigError as exc:
        LOG.error("%s", exc)
        raise SystemExit(1) from exc

    registry = init(config)
    LOG.info(
        "loaded %d backend(s) from %s; default=%s",
        len(registry.backends),
        path,
        registry.default_backend,
    )

    if args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
