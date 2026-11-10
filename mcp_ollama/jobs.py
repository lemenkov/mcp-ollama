# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Small in-memory job registry for long-running Ollama requests.

Ollama generation on this CPU can take minutes.  An MCP tool call that simply
awaits the HTTP request would eventually hit a client timeout, so inference
tools submit a job, wait for a bounded time, and return a job id when the work
is still running.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


class JobError(RuntimeError):
    """Base class for job registry errors."""


class JobNotFound(JobError):
    """The job id is unknown or has already been pruned."""


class JobCapacityError(JobError):
    """Too many jobs are in flight or retained."""


@dataclass
class Job:
    id: str
    kind: str
    backend: str
    model: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = None


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _truncate_result(result: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Truncate long text fields without mutating the stored result."""

    out = dict(result)
    for key in ("response", "thinking"):
        value = out.get(key)
        if isinstance(value, str) and len(value) > max_chars:
            original = len(value)
            out[key] = value[:max_chars] + "\n...[truncated]"
            out["truncated"] = True
            out["original_chars"] = original
    return out


class JobManager:
    def __init__(self, *, ttl_seconds: int = 3600, max_jobs: int = 100) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        *,
        kind: str,
        backend: str,
        model: str,
        runner: Callable[[], Awaitable[dict[str, Any]]],
        wait_seconds: float,
        result_max_chars: int,
    ) -> dict[str, Any]:
        """Start *runner* and wait at most *wait_seconds* for it.

        A negative or zero wait means "return immediately".  On timeout the job
        keeps running and the returned object contains its id.
        """

        async with self._lock:
            self._prune_locked()
            if len(self._jobs) >= self._max_jobs:
                self._evict_finished_locked()
            if len(self._jobs) >= self._max_jobs:
                raise JobCapacityError(
                    f"job limit reached ({self._max_jobs}); wait for a job to finish "
                    "or cancel one with ollama_cancel_job"
                )
            job_id = uuid.uuid4().hex[:16]
            while job_id in self._jobs:
                job_id = uuid.uuid4().hex[:16]
            job = Job(
                id=job_id,
                kind=kind,
                backend=backend,
                model=model,
            )
            self._jobs[job.id] = job
            job.task = asyncio.create_task(
                self._run(job, runner),
                name=f"ollama-job-{job.id}",
            )

        if wait_seconds > 0:
            try:
                # Shield the task: a client-side wait timeout must not cancel
                # the inference that is already running.
                await asyncio.wait_for(asyncio.shield(job.task), timeout=wait_seconds)
            except asyncio.TimeoutError:
                pass

        return self._describe(job, include_result=True, max_chars=result_max_chars)

    async def _run(
        self,
        job: Job,
        runner: Callable[[], Awaitable[dict[str, Any]]],
    ) -> None:
        if job.status == "cancelling":
            job.status = "cancelled"
            job.error = "job cancelled before it started"
            job.finished_at = time.time()
            return

        job.status = "running"
        job.started_at = time.time()
        try:
            job.result = await runner()
            job.status = "done"
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "job cancelled"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            job.status = "failed"
            job.error = str(exc) or type(exc).__name__
        finally:
            job.finished_at = time.time()

    def _get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(
                f"unknown job id {job_id!r}; jobs are kept for "
                f"{self._ttl_seconds}s after completion"
            )
        return job

    async def status(self, job_id: str) -> dict[str, Any]:
        async with self._lock:
            self._prune_locked()
            job = self._get(job_id)
            return self._describe(job, include_result=False)

    async def result(self, job_id: str, *, max_chars: int) -> dict[str, Any]:
        async with self._lock:
            self._prune_locked()
            job = self._get(job_id)
            return self._describe(job, include_result=True, max_chars=max_chars)

    async def cancel(self, job_id: str) -> dict[str, Any]:
        async with self._lock:
            self._prune_locked()
            job = self._get(job_id)
            if job.status in ("done", "failed", "cancelled"):
                return self._describe(job, include_result=False)
            job.status = "cancelling"
            if job.task is not None:
                job.task.cancel()
            if job.started_at is None:
                job.status = "cancelled"
                job.error = "job cancelled before it started"
                job.finished_at = time.time()
            return self._describe(job, include_result=False)

    def _describe(
        self,
        job: Job,
        *,
        include_result: bool,
        max_chars: int = 20_000,
    ) -> dict[str, Any]:
        now = time.time()
        started = job.started_at or job.created_at
        out: dict[str, Any] = {
            "job_id": job.id,
            "kind": job.kind,
            "backend": job.backend,
            "model": job.model,
            "status": job.status,
            "created_at": _iso(job.created_at),
            "started_at": _iso(job.started_at),
            "finished_at": _iso(job.finished_at),
            "elapsed_seconds": round((job.finished_at or now) - started, 2),
            "result_ready": job.result is not None,
            "error": job.error,
        }
        if include_result and job.result is not None:
            out["result"] = _truncate_result(job.result, max_chars)
        elif job.status in ("queued", "running", "cancelling"):
            out["message"] = (
                "Job is still running. Poll ollama_job_status, then fetch the "
                "answer with ollama_job_result."
            )
        return out

    def _prune_locked(self) -> None:
        cutoff = time.time() - self._ttl_seconds
        stale = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None and job.finished_at < cutoff
        ]
        for job_id in stale:
            del self._jobs[job_id]

    def _evict_finished_locked(self) -> None:
        finished = [
            (job.finished_at or job.created_at, job_id)
            for job_id, job in self._jobs.items()
            if job.status in ("done", "failed", "cancelled")
        ]
        for _, job_id in sorted(finished):
            del self._jobs[job_id]
            if len(self._jobs) < self._max_jobs:
                break
