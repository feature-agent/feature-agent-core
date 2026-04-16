"""Task queue consumer worker."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.queue.nats_client import NATSClient
from agent.state_manager import StateManager

logger = logging.getLogger(__name__)


class TaskConsumer:
    """Consumes tasks from NATS and dispatches to orchestrator."""

    def __init__(
        self,
        nats_client: NATSClient,
        state_manager: StateManager,
        subject: str,
        durable_name: str = "agent-worker",
    ) -> None:
        self._nats = nats_client
        self._state = state_manager
        self._subject = subject
        self._durable = durable_name

    async def start(self) -> None:
        """Subscribe to NATS subject and begin consuming tasks."""
        await self._nats.subscribe(
            self._subject,
            self._handle_message,
            self._durable,
        )
        logger.info("TaskConsumer started on %s", self._subject)

    async def _handle_message(self, msg: Any) -> None:
        """Handle an incoming NATS message."""
        try:
            data = json.loads(msg.data.decode())
            task_id = data.get("task_id")
            action = data.get("action", "new")
            logger.info("Received task %s (action=%s)", task_id, action)

            task = await self._state.get_task(task_id)
            if task is None:
                logger.error("Task %s not found — skipping", task_id)
                await msg.ack()
                return

            # Orchestrator dispatch will be wired in Phase 5
            logger.info("Task %s ready for processing", task_id)
            await msg.ack()

        except Exception:
            logger.exception("Error handling message")
            await msg.ack()
