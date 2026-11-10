# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from mcp_ollama.jobs import JobCapacityError, JobManager, JobNotFound


async def test_submit_returns_result_when_fast():
    manager = JobManager(ttl_seconds=60, max_jobs=10)

    async def runner():
        await asyncio.sleep(0.01)
        return {"response": "done"}

    result = await manager.submit(
        kind="generate",
        backend="local",
        model="m",
        runner=runner,
        wait_seconds=1,
        result_max_chars=1000,
    )
    assert result["status"] == "done"
    assert result["result"]["response"] == "done"


async def test_timeout_returns_job_id_and_finishes_later():
    manager = JobManager(ttl_seconds=60, max_jobs=10)

    async def runner():
        await asyncio.sleep(0.15)
        return {"response": "late"}

    result = await manager.submit(
        kind="generate",
        backend="local",
        model="m",
        runner=runner,
        wait_seconds=0.01,
        result_max_chars=1000,
    )
    assert result["job_id"]
    assert result["status"] in ("queued", "running")

    job_id = result["job_id"]
    for _ in range(100):
        status = await manager.status(job_id)
        if status["status"] == "done":
            break
        await asyncio.sleep(0.01)

    assert status["status"] == "done"
    fetched = await manager.result(job_id, max_chars=1000)
    assert fetched["result"]["response"] == "late"


async def test_failure_is_recorded():
    manager = JobManager(ttl_seconds=60, max_jobs=10)

    async def runner():
        raise RuntimeError("backend exploded")

    result = await manager.submit(
        kind="chat",
        backend="local",
        model="m",
        runner=runner,
        wait_seconds=1,
        result_max_chars=1000,
    )
    assert result["status"] == "failed"
    assert "backend exploded" in result["error"]


async def test_cancel_running_job():
    manager = JobManager(ttl_seconds=60, max_jobs=10)

    async def runner():
        await asyncio.sleep(10)

    result = await manager.submit(
        kind="generate",
        backend="local",
        model="m",
        runner=runner,
        wait_seconds=0,
        result_max_chars=1000,
    )
    job_id = result["job_id"]
    cancelled = await manager.cancel(job_id)
    assert cancelled["status"] == "cancelled"
    await asyncio.sleep(0.01)
    assert (await manager.status(job_id))["status"] == "cancelled"


async def test_capacity_limit():
    manager = JobManager(ttl_seconds=60, max_jobs=1)

    async def runner():
        await asyncio.sleep(10)

    first = await manager.submit(
        kind="generate",
        backend="local",
        model="m",
        runner=runner,
        wait_seconds=0,
        result_max_chars=1000,
    )
    with pytest.raises(JobCapacityError):
        await manager.submit(
            kind="generate",
            backend="local",
            model="m",
            runner=runner,
            wait_seconds=0,
            result_max_chars=1000,
        )
    await manager.cancel(first["job_id"])
    await asyncio.sleep(0.01)


async def test_result_truncation():
    manager = JobManager(ttl_seconds=60, max_jobs=10)

    async def runner():
        return {"response": "x" * 100}

    result = await manager.submit(
        kind="generate",
        backend="local",
        model="m",
        runner=runner,
        wait_seconds=1,
        result_max_chars=1000,
    )
    fetched = await manager.result(result["job_id"], max_chars=10)
    assert fetched["result"]["truncated"] is True
    assert fetched["result"]["original_chars"] == 100
    assert fetched["result"]["response"].startswith("x" * 10)


async def test_unknown_job():
    manager = JobManager(ttl_seconds=60, max_jobs=10)
    with pytest.raises(JobNotFound):
        await manager.status("nope")
