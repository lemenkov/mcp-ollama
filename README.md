<!--
SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
SPDX-License-Identifier: Apache-2.0
-->

# mcp-ollama

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

An MCP (Model Context Protocol) server for one or more [Ollama](https://ollama.com)
instances.

It exposes model discovery, chat, completion, embeddings, and asynchronous jobs as
typed MCP tools. Backends are selected by name from a configuration file; local and
remote HTTP(S) endpoints are supported, including endpoints behind HTTP Basic auth.

The server is designed for the case where inference is slow — a CPU-only machine or a
small GPU — and an agent should not hold a tool call open for several minutes.
Inference tools wait for a bounded time and return a job id when the work is still
running.

## Features

- **Several Ollama backends in one server.** Local and remote instances are addressed
  by name; the model never supplies a URL.
- **HTTP Basic auth per backend.** Works with Ollama behind nginx, Caddy, or another
  reverse proxy using an htpasswd-style user/password.
- **HTTPS and custom CA bundles.** `verify_tls` accepts `true`, `false`, or a CA file.
- **Path prefixes.** A backend may live at `https://host/ollama`, not only at the
  root of a host.
- **Asynchronous jobs.** `wait_seconds` bounds a tool call; longer generations return
  a `job_id` for `ollama_job_status` / `ollama_job_result` / `ollama_cancel_job`.
- **Per-backend concurrency limits.** One request in flight per backend by default,
  which matches a small CPU-only Ollama.
- **Streamable HTTP and stdio.** Run as a long-lived HTTP service or as a local stdio
  server.
- **Systemd-friendly.** The included unit runs with `DynamicUser`, a read-only
  filesystem, no capabilities, private devices, and the backend JSON passed as a
  systemd credential rather than an environment variable.
- **No telemetry.** The packaged service disables FastMCP update checks and the
  startup banner.

## Architecture

```text
MCP client (Claude Code, Codex, Cursor, ...)
    |
    |  Streamable HTTP  or  stdio
    v
mcp-ollama  (FastMCP)
    |
    |  named backend + optional Basic auth + TLS
    +--> local Ollama       http://127.0.0.1:11434
    +--> remote Ollama      https://ollama.example.com/ollama
    +--> another Ollama     https://other.example.com
```

Each backend has its own `httpx.AsyncClient` and an `asyncio.Semaphore`. Inference
jobs run in the server process; the answers are held in memory for
`job_ttl_seconds` (one hour by default).

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com) 0.12 or newer (tested with 0.12.11)
- `fastmcp` 4.x (tested with 4.1.7) and `httpx` 0.28+
- Linux with systemd for the included service/installer; the Python package itself
  runs anywhere

## Quick start

```bash
git clone https://github.com/lemenkov/mcp-ollama.git
cd mcp-ollama

python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

# Make sure the default model exists locally
ollama pull llama3.2:3b

# Start the MCP server on 127.0.0.1:8815
mcp-ollama --config mcp-ollama.json
```

The Streamable HTTP endpoint is then:

```text
http://127.0.0.1:8815/mcp
```

For a local stdio server instead:

```bash
mcp-ollama --config mcp-ollama.json --transport stdio
```

## Configuration

There are two files:

| File | Purpose |
| --- | --- |
| `/etc/sysconfig/mcp-ollama` | Listener and logging environment variables |
| `/etc/sysconfig/mcp-ollama.json` | Backend definitions and auth credentials |

The JSON file is normally root-owned and mode `0600`. The systemd unit passes it to
the service with `LoadCredential=`, so passwords do not appear in the environment,
`systemctl show`, or `/proc`.

### Listener environment

```ini
HOST=127.0.0.1
PORT=8815
OLLAMA_MCP_LOG_LEVEL=INFO

# Keep the packaged service fully local.
FASTMCP_CHECK_FOR_UPDATES=off
FASTMCP_SHOW_SERVER_BANNER=false
```

### Backends

```json
{
  "default_backend": "local",
  "backends": {
    "local": {
      "base_url": "http://127.0.0.1:11434",
      "default_model": "llama3.2:3b",
      "timeout_seconds": 600,
      "max_concurrency": 1
    },
    "remote": {
      "base_url": "https://ollama.example.com/ollama",
      "username": "ollama",
      "password": "change-me",
      "default_model": "qwen3:4b",
      "verify_tls": true,
      "timeout_seconds": 600,
      "max_concurrency": 1,
      "description": "Remote Ollama behind nginx Basic auth"
    }
  },
  "job_ttl_seconds": 3600,
  "max_jobs": 100,
  "result_max_chars": 20000
}
```

#### Backend fields

| Field | Default | Description |
| --- | --- | --- |
| `base_url` | required | `http://` or `https://` URL. A path prefix such as `/ollama` is allowed. Userinfo (`https://user:pass@host`) is also accepted. |
| `username` | `null` | HTTP Basic auth user. If omitted and the URL has userinfo, the URL user is used. |
| `password` | `null` | HTTP Basic auth password. |
| `password_file` | `null` | Read the password from a file. Relative paths are resolved against the JSON file's directory. Use inline passwords with the systemd credential. |
| `default_model` | `null` | Model used when a tool omits `model`. If neither is set, the tool returns an error. |
| `verify_tls` | `true` | `true`, `false`, or a path to a CA bundle. |
| `timeout_seconds` | `600` | Read timeout for inference requests. |
| `connect_timeout_seconds` | `10` | TCP/TLS connect timeout. |
| `max_concurrency` | `1` | Maximum simultaneous requests to this backend. |
| `headers` | `{}` | Extra HTTP headers for this backend. |
| `enabled` | `true` | Disabled backends are not loaded and cannot be selected. |
| `description` | `null` | Free-form note shown by `ollama_backends` when `probe=false`. |

#### Top-level fields

| Field | Default | Description |
| --- | --- | --- |
| `default_backend` | required | Backend used when a tool omits `backend`. Must exist and be enabled. |
| `job_ttl_seconds` | `3600` | How long completed jobs are retained in memory. |
| `max_jobs` | `100` | Maximum retained jobs; new submissions are rejected when the limit is reached and nothing can be evicted. |
| `result_max_chars` | `20000` | Default truncation limit for generated text returned by inference tools. |

#### Configuration precedence

The backend file is resolved in this order:

1. `--config PATH`
2. `OLLAMA_MCP_CONFIG`
3. `$CREDENTIALS_DIRECTORY/backends.json` (the systemd credential)
4. `/etc/sysconfig/mcp-ollama.json`

### Two different Basic auth layers

Do not confuse them:

1. **MCP client to this server**, when the server is behind a reverse proxy. The
   proxy checks an htpasswd file; the MCP client sends the proxy credentials.
2. **This server to an Ollama backend**, configured by `username` / `password` in
   the backend JSON.

The MCP server has no built-in authentication. Bind it to `127.0.0.1` and put an
authenticated reverse proxy in front of it for remote clients.

## Tools

| Tool | Purpose |
| --- | --- |
| `ollama_backends` | List configured backends and optionally probe them. |
| `ollama_models` | List models on one backend, or on all backends. |
| `ollama_chat` | Chat completion with a message list. |
| `ollama_generate` | Single-prompt completion. |
| `ollama_embed` | Embeddings for a small number of inputs. |
| `ollama_job_status` | Check an asynchronous job. |
| `ollama_job_result` | Fetch the result of a completed job. |
| `ollama_cancel_job` | Cancel a queued or running job. |

All inference tools use the same bounded-wait model. `wait_seconds` defaults to 30
and is capped at 120. A value of `0` returns a job id immediately.

### `ollama_backends`

```text
ollama_backends(probe=True, timeout_seconds=3.0)
```

Returns the configured backend names, their redacted URLs, and — when `probe` is
true — version, model count, loaded models, and latency. A backend that is down is
reported in its own `error` field; the call itself does not fail.

```json
{
  "default_backend": "local",
  "backends": [
    {
      "name": "local",
      "url": "http://127.0.0.1:11434/",
      "reachable": true,
      "version": "0.12.11",
      "models_count": 1,
      "loaded": [],
      "latency_ms": 12.4
    }
  ]
}
```

Use the returned `name` as the `backend` argument of the other tools.

### `ollama_models`

```text
ollama_models(backend=None)
```

With `backend` omitted, every enabled backend is queried. Each result contains the
model name, size, modification time, family, parameter size, and quantization:

```json
{
  "default_backend": "local",
  "backends": [
    {
      "backend": "local",
      "url": "http://127.0.0.1:11434/",
      "models": [
        {
          "name": "llama3.2:3b",
          "size": 2019393189,
          "modified_at": "2026-02-08T19:04:45.19249316+01:00",
          "family": "llama",
          "parameter_size": "3.2B",
          "quantization": "Q4_K_M"
        }
      ]
    }
  ]
}
```

### `ollama_chat`

```text
ollama_chat(
  messages,
  backend=None,
  model=None,
  system=None,
  options=None,
  wait_seconds=30
)
```

`messages` is a list of `{ "role": "system" | "user" | "assistant", "content": "..." }`.
If `system` is supplied, do not also put a system message in `messages`.

`options` is a `GenerationOptions` object with any of:

| Option | Range |
| --- | --- |
| `temperature` | 0.0 – 2.0 |
| `top_p` | 0.0 – 1.0 |
| `num_predict` | 1 – 8192 |
| `num_ctx` | 512 – 32768 |
| `num_thread` | 1 – 16 |
| `seed` | 0 – 2147483647 |
| `stop` | up to 8 strings |

### `ollama_generate`

```text
ollama_generate(
  prompt,
  backend=None,
  model=None,
  system=None,
  options=None,
  wait_seconds=30
)
```

The convenient tool for rewriting, translation, and other single-prompt work.

### `ollama_embed`

```text
ollama_embed(input, backend=None, model=None, truncate=True, wait_seconds=30)
```

`input` is a string or a list of up to 8 strings, with at most 16,000 characters in
total. Vectors are returned inline, so use a dedicated pipeline for large corpora.
`truncate` asks Ollama to truncate inputs longer than the model's context.

### Jobs

A completed call returns a result directly:

```json
{
  "job_id": "fd5e7cda6280475b",
  "kind": "generate",
  "backend": "local",
  "model": "llama3.2:3b",
  "status": "done",
  "created_at": "2026-11-10T20:13:39.616516+00:00",
  "started_at": "2026-11-10T20:13:39.616560+00:00",
  "finished_at": "2026-11-10T20:13:47.866026+00:00",
  "elapsed_seconds": 8.25,
  "result_ready": true,
  "error": null,
  "result": {
    "backend": "local",
    "response": "hello from MCP",
    "model": "llama3.2:3b",
    "done": true,
    "done_reason": "stop",
    "eval_count": 4,
    "eval_duration_ns": 341321693,
    "prompt_eval_count": 32,
    "prompt_eval_duration_ns": 1621996874,
    "total_duration_ns": 8244612415,
    "load_duration_ns": 6269440872,
    "tokens_per_second": 11.72
  }
}
```

If the call is still running when `wait_seconds` expires, the response has a `job_id`
and a status of `queued` or `running`:

```json
{
  "job_id": "1128ff069b2a46e6",
  "kind": "generate",
  "backend": "local",
  "model": "llama3.2:3b",
  "status": "running",
  "result_ready": false,
  "error": null,
  "message": "Job is still running. Poll ollama_job_status, then fetch the answer with ollama_job_result."
}
```

Typical client flow:

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8815/mcp") as client:
        started = await client.call_tool(
            "ollama_generate",
            {"prompt": "Explain systemd sandboxing in two sentences.", "wait_seconds": 0},
        )
        job_id = started.data["job_id"]

        while True:
            status = await client.call_tool("ollama_job_status", {"job_id": job_id})
            if status.data["status"] in ("done", "failed", "cancelled"):
                break
            await asyncio.sleep(1)

        result = await client.call_tool("ollama_job_result", {"job_id": job_id})
        print(result.data["result"]["response"])

asyncio.run(main())
```

`ollama_job_result` accepts `max_chars` (1,000 – 200,000, default 20,000). Long text
fields are truncated with `"truncated": true` and `original_chars`.

`ollama_cancel_job` cancels a queued or running job. Cancelling a finished job is
harmless.

## MCP client setup

### Streamable HTTP, local

```json
{
  "mcpServers": {
    "ollama": {
      "command": "npx",
      "args": [
        "mcp-remote@0.16.5",
        "http://127.0.0.1:8815/mcp"
      ]
    }
  }
}
```

### Streamable HTTP behind an authenticated reverse proxy

```json
{
  "mcpServers": {
    "ollama": {
      "command": "npx",
      "args": [
        "mcp-remote@0.16.5",
        "https://mcp.example.com/mcp/ollama",
        "--header-file",
        "/home/you/.config/mcp-auth-ollama.txt"
      ]
    }
  }
}
```

`mcp-auth-ollama.txt` contains the proxy's credentials, for example:

```text
Authorization: Basic <base64-user:password>
```

This is the **proxy** credential, not the Ollama backend credential.

### stdio

```json
{
  "mcpServers": {
    "ollama": {
      "command": "/opt/mcp-servers/mcp-ollama/venv/bin/mcp-ollama",
      "args": [
        "--transport", "stdio",
        "--config", "/etc/sysconfig/mcp-ollama.json"
      ]
    }
  }
}
```

A stdio server runs as the client's user and reads the JSON file directly. The
`LoadCredential` arrangement only applies to the systemd HTTP service.

## systemd deployment

The repository includes an opinionated installer for a Fedora-like host with systemd
and nginx:

```bash
sudo bash install-mcp-ollama.sh
```

Review the script before running it. It installs:

```text
/opt/mcp-servers/mcp-ollama/              code, pyproject.toml, venv/
/etc/sysconfig/mcp-ollama                 listener environment
/etc/sysconfig/mcp-ollama.json            backend definitions (0600)
/etc/systemd/system/mcp-ollama.service    hardened unit
```

It also adds a `/mcp/ollama` proxy location to `/etc/nginx/conf.d/mcp.conf`, runs
`nginx -t`, and reloads nginx. Existing sysconfig files are preserved; replaced
units and nginx configuration are backed up first.

Manual equivalent:

```bash
sudo install -d /opt/mcp-servers/mcp-ollama
sudo cp -a mcp_ollama pyproject.toml README.md LICENSE NOTICE \
    /opt/mcp-servers/mcp-ollama/
sudo python3 -m venv /opt/mcp-servers/mcp-ollama/venv
sudo /opt/mcp-servers/mcp-ollama/venv/bin/pip install /opt/mcp-servers/mcp-ollama

sudo install -m 0644 mcp-ollama.sysconfig /etc/sysconfig/mcp-ollama
sudo install -m 0600 mcp-ollama.json /etc/sysconfig/mcp-ollama.json
sudo install -m 0644 mcp-ollama.service /etc/systemd/system/mcp-ollama.service

sudo systemctl daemon-reload
sudo systemctl enable --now mcp-ollama.service
```

The service is read-only apart from its private `/tmp`. Its configuration is
delivered via `LoadCredential=backends.json:/etc/sysconfig/mcp-ollama.json`, so the
JSON can stay root-only.

Useful checks:

```bash
systemctl status mcp-ollama.service
journalctl -u mcp-ollama.service -f
ss -ltn | grep 8815
```

### nginx

If the server is already behind a server-level `auth_basic`, the location only needs
the proxy settings:

```nginx
location ^~ /mcp/ollama {
    rewrite ^/mcp/ollama(.*)$ /mcp$1 break;
    proxy_set_header Accept "application/json, text/event-stream";
    # FastMCP validates the Host header; keep it in its localhost allowlist.
    proxy_set_header Host localhost:$server_port;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Connection "";
    proxy_pass http://127.0.0.1:8815;
}
```

If authentication is not inherited, add:

```nginx
auth_basic "MCP Access";
auth_basic_user_file /etc/nginx/.htpasswd-mcp;
```

## Security model

- **Named backends only.** Tools accept a backend name, never a URL, so the model
  cannot make the server connect to arbitrary hosts.
- **Credential, not environment.** The backend JSON is passed as a systemd credential
  and is not visible in `systemctl show`, the service environment, or `/proc`.
- **Least privilege.** The packaged unit uses `DynamicUser=yes`, `ProtectSystem=strict`,
  `ProtectHome=yes`, `PrivateDevices=yes`, `PrivateTmp=yes`,
  `MemoryDenyWriteExecute=yes`, an empty capability set, and a `@system-service`
  syscall filter.
- **Local listener.** The server binds `127.0.0.1`. Put an authenticated reverse proxy
  in front of it for remote clients.
- **TLS for remote backends.** Basic auth is only as private as the transport; use
  `https://` for remote Ollama endpoints and keep `verify_tls` enabled.
- **No telemetry.** The packaged service sets `FASTMCP_CHECK_FOR_UPDATES=off` and
  `FASTMCP_SHOW_SERVER_BANNER=false`.
- **No prompt logging by default.** Logs contain backend names, model names, and
  timings, not message contents.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

The test suite uses `httpx.MockTransport`, so it needs no Ollama instance and no
network. It covers:

- configuration parsing, URL prefixes, userinfo, and password files
- Basic auth headers and actionable HTTP errors
- per-backend routing
- asynchronous job submission, polling, cancellation, failure, and truncation

To run the server against a real local Ollama:

```bash
ollama pull llama3.2:3b
mcp-ollama --config mcp-ollama.json
```

## Limitations

- **Jobs are in memory.** Restarting the MCP service loses running and completed jobs.
  The Ollama server itself keeps the loaded model, so a retry is usually cheap.
- **No token streaming.** Inference calls use `stream=false`; progress is exposed as
  job status, not partial tokens.
- **Embeddings are inline.** `ollama_embed` is capped at 8 inputs and 16,000
  characters per call. Use a dedicated batch pipeline for large corpora.
- **No model administration.** Pulling and deleting models stay in the `ollama` CLI;
  the MCP tools do not expose `pull` or `rm`.
- **One request per backend by default.** Raise `max_concurrency` only if the backend
  has enough CPU/GPU capacity.
- **CPU inference is slow.** On a four-core Haswell, a 3B Q4 model generates roughly
  10–12 tokens/s after loading. A 300-token answer takes about half a minute; a
  1,000-token answer takes minutes.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 Peter Lemenkov.

## Acknowledgements

- [Ollama](https://ollama.com) for the local inference runtime and HTTP API.
- [FastMCP](https://gofastmcp.com) for the MCP server framework.
- [httpx](https://www.python-httpx.org) for the async HTTP client.
- [Pydantic](https://docs.pydantic.dev) for configuration and tool schemas.
