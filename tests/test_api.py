"""API endpoint tests for Phase 1 — health and storage."""

import pytest

from agent.storage.local_volume import LocalVolumeStorage


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """GET /api/health returns 200 with expected fields."""
    response = await async_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "nats" in data


@pytest.mark.asyncio
async def test_index_page(async_client):
    """GET / returns the web client HTML."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "Feature Agent Core" in response.text


@pytest.mark.asyncio
async def test_storage_write_and_read(tmp_data_dir):
    """LocalVolumeStorage can write and read JSON."""
    storage = LocalVolumeStorage(tmp_data_dir)
    await storage.write("test/item", {"key": "value"})
    result = await storage.read("test/item")
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_storage_read_missing(tmp_data_dir):
    """LocalVolumeStorage returns None for missing keys."""
    storage = LocalVolumeStorage(tmp_data_dir)
    result = await storage.read("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_storage_append_and_read_all(tmp_data_dir):
    """LocalVolumeStorage can append and read JSONL."""
    storage = LocalVolumeStorage(tmp_data_dir)
    await storage.append("test/log", {"event": 1})
    await storage.append("test/log", {"event": 2})
    results = await storage.read_all("test/log")
    assert len(results) == 2
    assert results[0]["event"] == 1
    assert results[1]["event"] == 2


@pytest.mark.asyncio
async def test_storage_read_all_missing(tmp_data_dir):
    """LocalVolumeStorage returns empty list for missing JSONL."""
    storage = LocalVolumeStorage(tmp_data_dir)
    results = await storage.read_all("nonexistent")
    assert results == []


@pytest.mark.asyncio
async def test_storage_exists(tmp_data_dir):
    """LocalVolumeStorage.exists checks for key presence."""
    storage = LocalVolumeStorage(tmp_data_dir)
    assert not await storage.exists("test/item")
    await storage.write("test/item", {"key": "value"})
    assert await storage.exists("test/item")


@pytest.mark.asyncio
async def test_storage_list_keys(tmp_data_dir):
    """LocalVolumeStorage.list_keys finds keys by prefix."""
    storage = LocalVolumeStorage(tmp_data_dir)
    await storage.write("tasks/abc/state", {"id": "abc"})
    await storage.write("tasks/def/state", {"id": "def"})
    keys = await storage.list_keys("tasks")
    assert "tasks/abc/state" in keys
    assert "tasks/def/state" in keys
