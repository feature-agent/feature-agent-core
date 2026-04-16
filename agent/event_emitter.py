"""SSE event emission and per-task subscriber queues."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from agent.state_manager import StateManager

logger = logging.getLogger(__name__)


class EventEmitter:
    """Emits SSE events, stores them, and manages per-task subscriber queues."""

    def __init__(self, state_manager: StateManager) -> None:
        self._state = state_manager
        self._queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    async def emit(self, task_id: str, event_type: str, **kwargs: Any) -> None:
        """Build an event, persist it, and push to all subscriber queues."""
        event: dict[str, Any] = {
            "type": event_type,
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        await self._state.append_event(task_id, event)

        for queue in self._queues.get(task_id, []):
            await queue.put(event)

        logger.debug("Emitted %s for task %s", event_type, task_id)

    async def subscribe(self, task_id: str) -> AsyncGenerator[dict[str, Any], None]:
        """Yield events from a per-task queue. Used by SSE endpoint."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        if task_id not in self._queues:
            self._queues[task_id] = []
        self._queues[task_id].append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._queues[task_id].remove(queue)
            if not self._queues[task_id]:
                del self._queues[task_id]

    def get_queue(self, task_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Get or create a subscriber queue for a task."""
        if task_id not in self._queues:
            self._queues[task_id] = []
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._queues[task_id].append(queue)
        return queue
