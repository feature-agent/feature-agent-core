"""Local filesystem storage backed by a Docker named volume."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from agent.storage.interface import StorageInterface

logger = logging.getLogger(__name__)


class LocalVolumeStorage(StorageInterface):
    """Storage implementation using local filesystem (Docker volume)."""

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a per-key asyncio lock for thread safety."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _json_path(self, key: str) -> Path:
        return self._base / f"{key}.json"

    def _jsonl_path(self, key: str) -> Path:
        return self._base / f"{key}.jsonl"

    async def write(self, key: str, data: dict[str, Any]) -> None:
        """Write a JSON object to {key}.json."""
        async with self._get_lock(key):
            path = self._json_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            logger.debug("Wrote %s", path)

    async def read(self, key: str) -> dict[str, Any] | None:
        """Read a JSON object from {key}.json. Returns None if missing."""
        path = self._json_path(key)
        if not path.exists():
            return None
        async with self._get_lock(key):
            return json.loads(path.read_text())

    async def append(self, key: str, data: dict[str, Any]) -> None:
        """Append a JSON object as a new line to {key}.jsonl."""
        async with self._get_lock(key):
            path = self._jsonl_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(data) + "\n")
            logger.debug("Appended to %s", path)

    async def read_all(self, key: str) -> list[dict[str, Any]]:
        """Read all JSON objects from {key}.jsonl."""
        path = self._jsonl_path(key)
        if not path.exists():
            return []
        async with self._get_lock(key):
            lines = path.read_text().strip().split("\n")
            return [json.loads(line) for line in lines if line.strip()]

    async def list_keys(self, prefix: str) -> list[str]:
        """List all keys matching the given prefix."""
        search_dir = self._base / prefix
        if not search_dir.exists():
            # Try as a glob prefix on the base
            results = []
            for path in self._base.rglob(f"{prefix}*"):
                rel = path.relative_to(self._base)
                key = str(rel).removesuffix(".json").removesuffix(".jsonl")
                results.append(key)
            return sorted(set(results))

        results = []
        for path in search_dir.rglob("*.json"):
            rel = path.relative_to(self._base)
            key = str(rel).removesuffix(".json")
            results.append(key)
        return sorted(results)

    async def exists(self, key: str) -> bool:
        """Check if a key exists (as .json or .jsonl)."""
        return self._json_path(key).exists() or self._jsonl_path(key).exists()
