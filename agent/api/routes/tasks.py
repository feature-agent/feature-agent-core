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
from agent.state_manager import TaskState

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
    try:
        task = await state_manager.create_task(
            task_id=request.task_id,
            source=source,
            github_issue_url=request.github_issue_url,
            task_description=request.task_description,
            target_repo=request.target_repo,
            provider=request.provider.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

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


def _strip_provider(task: dict) -> dict:
    """Remove provider credentials before returning to client."""
    t = {k: v for k, v in task.items() if k != "provider"}
    return t


@router.get("", response_model=TaskListResponse)
async def list_tasks() -> TaskListResponse:
    """List all tasks."""
    state_manager, _, _ = _get_deps()
    tasks = await state_manager.list_tasks()
    return TaskListResponse(
        tasks=[TaskResponse(**_strip_provider(t)) for t in tasks],
        total=len(tasks),
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Get a task by ID."""
    state_manager, _, _ = _get_deps()
    task = await state_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(**_strip_provider(task))


TERMINAL_STATES = {TaskState.DONE.value, TaskState.FAILED.value, TaskState.CANCELED.value}


@router.delete("/{task_id}")
async def delete_or_cancel_task(task_id: str) -> dict:
    """DELETE semantics: if the task is running, mark CANCELED; if terminal, remove."""
    state_manager, _, _ = _get_deps()
    task = await state_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task["status"] in TERMINAL_STATES:
        await state_manager.delete_task(task_id)
        return {"task_id": task_id, "action": "deleted"}
    await state_manager.update_task(task_id, status=TaskState.CANCELED.value)
    logger.info("Task %s marked for cancellation", task_id)
    return {"task_id": task_id, "action": "canceled"}
