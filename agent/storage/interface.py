"""Abstract storage interface for pluggable backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageInterface(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def write(self, key: str, data: dict[str, Any]) -> None:
        """Write a JSON object to storage at the given key."""

    @abstractmethod
    async def read(self, key: str) -> dict[str, Any] | None:
        """Read a JSON object from storage. Returns None if missing."""

    @abstractmethod
    async def append(self, key: str, data: dict[str, Any]) -> None:
        """Append a JSON object as a new line to a JSONL file."""

    @abstractmethod
    async def read_all(self, key: str) -> list[dict[str, Any]]:
        """Read all JSON objects from a JSONL file."""

    @abstractmethod
    async def list_keys(self, prefix: str) -> list[str]:
        """List all keys matching the given prefix."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
