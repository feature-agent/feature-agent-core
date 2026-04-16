"""Task API routes: create, list, and get tasks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from agent.api.models import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskListResponse,
    TaskResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _get_deps():
    """Get shared dependencies from main module."""
    from agent.main import nats_client, state_manager
    from agent.config import settings
    return state_manager, nats_client, settings


@router.post("", response_model=TaskCreateResponse, status_code=201)
async def create_task(request: TaskCreateRequest) -> TaskCreateResponse:
    """Create a new task and publish to NATS queue."""
    state_manager, nats_client, settings = _get_deps()

    source = "github_issue" if request.github_issue_url else "free_text"
    task = await state_manager.create_task(
        source=source,
        github_issue_url=request.github_issue_url,
        task_description=request.task_description,
        target_repo=request.target_repo,
    )

    try:
        await nats_client.publish(
            settings.NATS_TASK_SUBJECT,
            {"task_id": task["task_id"], "action": "new"},
        )
    except ConnectionError:
        logger.warning("NATS not available — task %s queued locally only", task["task_id"])

    return TaskCreateResponse(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
    )


@router.get("", response_model=TaskListResponse)
async def list_tasks() -> TaskListResponse:
    """List all tasks."""
    state_manager, _, _ = _get_deps()
    tasks = await state_manager.list_tasks()
    return TaskListResponse(
        tasks=[TaskResponse(**t) for t in tasks],
        total=len(tasks),
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Get a task by ID."""
    state_manager, _, _ = _get_deps()
    task = await state_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(**task)
