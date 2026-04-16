"""EFS storage stub for AWS deployment.

To swap Docker volume for AWS EFS:
1. pip install aiobotocore
2. Change STORAGE_BACKEND=efs in .env
3. Set AWS_REGION and EFS_FILE_SYSTEM_ID in .env
Everything else stays identical — same interface.
"""

from __future__ import annotations

from typing import Any

from agent.storage.interface import StorageInterface


class EFSStorage(StorageInterface):
    """AWS EFS storage backend — stub implementation."""

    async def write(self, key: str, data: dict[str, Any]) -> None:
        raise NotImplementedError("EFSStorage: swap implementation here")

    async def read(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError("EFSStorage: swap implementation here")

    async def append(self, key: str, data: dict[str, Any]) -> None:
        raise NotImplementedError("EFSStorage: swap implementation here")

    async def read_all(self, key: str) -> list[dict[str, Any]]:
        raise NotImplementedError("EFSStorage: swap implementation here")

    async def list_keys(self, prefix: str) -> list[str]:
        raise NotImplementedError("EFSStorage: swap implementation here")

    async def exists(self, key: str) -> bool:
        raise NotImplementedError("EFSStorage: swap implementation here")
