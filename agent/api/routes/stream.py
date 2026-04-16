"""SSE streaming endpoint with event replay and keep-alive."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

PING_INTERVAL_SECONDS = 15


def _get_deps():
    """Get shared dependencies from main module."""
    from agent.main import event_emitter, state_manager
    return state_manager, event_emitter


@router.get("/api/stream/{task_id}")
async def stream_events(task_id: str) -> StreamingResponse:
    """Stream SSE events for a task with replay and keep-alive pings."""
    state_manager, event_emitter = _get_deps()

    task = await state_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Replay stored events
        stored_events = await state_manager.get_events(task_id)
        task_terminal = False
        for event in stored_events:
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("task_done", "task_failed"):
                task_terminal = True

        # If task already finished, no need to listen for more
        if task_terminal:
            return

        # 2. Subscribe to live events
        queue = event_emitter.get_queue(task_id)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=PING_INTERVAL_SECONDS
                    )
                    yield f"data: {json.dumps(event)}\n\n"

                    # Stop streaming after terminal events
                    if event.get("type") in ("task_done", "task_failed"):
                        return

                except asyncio.TimeoutError:
                    # Send ping keep-alive
                    ping = {
                        "type": "ping",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    yield f"data: {json.dumps(ping)}\n\n"
        finally:
            # Cleanup: remove queue from emitter
            if task_id in event_emitter._queues:
                if queue in event_emitter._queues[task_id]:
                    event_emitter._queues[task_id].remove(queue)
                if not event_emitter._queues[task_id]:
                    del event_emitter._queues[task_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
