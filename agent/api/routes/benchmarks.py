"""Benchmark API routes for task timing and cost data."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["benchmarks"])


def _get_deps():
    """Get shared dependencies from main module."""
    from agent.main import state_manager, storage
    return state_manager, storage


@router.get("/api/tasks/{task_id}/benchmark")
async def get_task_benchmark(task_id: str):
    """Get benchmark data for a completed task."""
    state_manager, storage = _get_deps()

    task = await state_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task["status"] not in ("DONE", "FAILED"):
        raise HTTPException(
            status_code=400,
            detail=f"Task is {task['status']}, benchmark available after completion",
        )

    benchmark = await storage.read(f"tasks/{task_id}/benchmark")
    if benchmark is None:
        raise HTTPException(status_code=404, detail="Benchmark data not found")

    return benchmark


@router.get("/api/benchmarks")
async def list_benchmarks():
    """List all task benchmarks sorted by started_at descending."""
    _, storage = _get_deps()

    benchmarks = await storage.read_all("benchmarks")
    benchmarks.sort(key=lambda b: b.get("started_at", ""), reverse=True)
    return benchmarks
