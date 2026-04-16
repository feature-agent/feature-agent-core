"""NATS JetStream client wrapper with exponential backoff."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine

import nats
from nats.aio.client import Client as NATSConnection
from nats.js import JetStreamContext

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1.0


class NATSClient:
    """Wrapper around nats.py with JetStream support."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._nc: NATSConnection | None = None
        self._js: JetStreamContext | None = None

    @property
    def is_connected(self) -> bool:
        """Check if NATS connection is active."""
        return self._nc is not None and self._nc.is_connected

    async def connect(self) -> None:
        """Connect to NATS with exponential backoff. Max 5 retries."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._nc = await nats.connect(self._url)
                self._js = self._nc.jetstream()
                logger.info("Connected to NATS at %s", self._url)
                return
            except Exception as exc:
                delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "NATS connect attempt %d/%d failed: %s. Retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    delay,
                )
                if attempt == MAX_RETRIES:
                    raise ConnectionError(
                        f"Failed to connect to NATS after {MAX_RETRIES} attempts"
                    ) from exc
                await asyncio.sleep(delay)

    async def disconnect(self) -> None:
        """Disconnect from NATS."""
        if self._nc and self._nc.is_connected:
            await self._nc.close()
            logger.info("Disconnected from NATS")

    async def publish(self, subject: str, data: dict[str, Any]) -> None:
        """Publish a JSON message to a subject."""
        if not self._js:
            raise ConnectionError("NATS not connected")
        payload = json.dumps(data).encode()
        await self._js.publish(subject, payload)
        logger.debug("Published to %s", subject)

    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Any], Coroutine[Any, Any, None]],
        durable_name: str,
    ) -> Any:
        """Subscribe to a JetStream subject with a durable consumer."""
        if not self._js:
            raise ConnectionError("NATS not connected")
        sub = await self._js.subscribe(
            subject,
            durable=durable_name,
            cb=handler,
        )
        logger.info("Subscribed to %s (durable=%s)", subject, durable_name)
        return sub

    async def ensure_stream(
        self, stream_name: str, subjects: list[str]
    ) -> None:
        """Create or update a JetStream stream."""
        if not self._js:
            raise ConnectionError("NATS not connected")
        try:
            await self._js.find_stream_name_by_subject(subjects[0])
            logger.info("Stream %s already exists", stream_name)
        except nats.js.errors.NotFoundError:
            await self._js.add_stream(name=stream_name, subjects=subjects)
            logger.info("Created stream %s with subjects %s", stream_name, subjects)
