"""Task state manager using LocalVolumeStorage."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

from agent.storage.interface import StorageInterface

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """Task lifecycle states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
    DONE = "DONE"
    FAILED = "FAILED"


class StateManager:
    """Manages task state persistence and event storage."""

    def __init__(self, storage: StorageInterface) -> None:
        self._storage = storage

    def _task_key(self, task_id: str) -> str:
        return f"tasks/{task_id}/state"

    def _events_key(self, task_id: str) -> str:
        return f"tasks/{task_id}/events"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def create_task(
        self,
        source: str,
        github_issue_url: Optional[str],
        task_description: Optional[str],
        target_repo: str,
    ) -> dict[str, Any]:
        """Create a new task with PENDING status."""
        task_id = secrets.token_hex(4)
        now = self._now_iso()
        task = {
            "task_id": task_id,
            "status": TaskState.PENDING.value,
            "created_at": now,
            "updated_at": now,
            "source": source,
            "github_issue_url": github_issue_url,
            "task_description": task_description,
            "target_repo": target_repo,
            "requirement": None,
            "clarification": None,
            "context": {},
            "result": None,
            "error": None,
        }
        await self._storage.write(self._task_key(task_id), task)
        logger.info("Created task %s", task_id)
        return task

    async def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Get a task by ID. Returns None if not found."""
        return await self._storage.read(self._task_key(task_id))

    async def update_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update task fields and set updated_at."""
        task = await self.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        task.update(kwargs)
        task["updated_at"] = self._now_iso()
        await self._storage.write(self._task_key(task_id), task)
        logger.info("Updated task %s: %s", task_id, list(kwargs.keys()))
        return task

    async def list_tasks(self) -> List[dict[str, Any]]:
        """List all tasks sorted by created_at descending."""
        keys = await self._storage.list_keys("tasks")
        tasks = []
        for key in keys:
            if key.endswith("/state"):
                task = await self._storage.read(key)
                if task:
                    tasks.append(task)
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tasks

    async def append_event(self, task_id: str, event: dict[str, Any]) -> None:
        """Append an event to the task's event log."""
        await self._storage.append(self._events_key(task_id), event)

    async def get_events(self, task_id: str) -> List[dict[str, Any]]:
        """Get all events for a task in chronological order."""
        return await self._storage.read_all(self._events_key(task_id))

    async def set_clarification_questions(
        self, task_id: str, questions: List[dict[str, Any]]
    ) -> dict[str, Any]:
        """Set clarification questions and move to AWAITING_CLARIFICATION."""
        task = await self.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        clarification = task.get("clarification") or {}
        clarification["questions"] = questions
        return await self.update_task(
            task_id,
            clarification=clarification,
            status=TaskState.AWAITING_CLARIFICATION.value,
        )

    async def set_clarification_answers(
        self, task_id: str, answers: List[dict[str, Any]]
    ) -> dict[str, Any]:
        """Set clarification answers and move to RUNNING."""
        task = await self.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        clarification = task.get("clarification") or {}
        clarification["answers"] = answers
        return await self.update_task(
            task_id,
            clarification=clarification,
            status=TaskState.RUNNING.value,
        )
