"""API endpoint tests."""

import pytest

from agent.storage.local_volume import LocalVolumeStorage


# --- Phase 1: Health and storage tests ---

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


# --- Phase 2: Task API tests ---

@pytest.mark.asyncio
async def test_post_tasks_with_github_url(async_client):
    """POST /api/tasks with github_issue_url creates a task."""
    response = await async_client.post("/api/tasks", json={
        "github_issue_url": "https://github.com/owner/repo/issues/1",
        "target_repo": "owner/repo",
    })
    assert response.status_code == 201
    data = response.json()
    assert len(data["task_id"]) == 8
    assert data["status"] == "PENDING"


@pytest.mark.asyncio
async def test_post_tasks_with_description(async_client):
    """POST /api/tasks with task_description creates a task."""
    response = await async_client.post("/api/tasks", json={
        "task_description": "Add due_date field",
        "target_repo": "owner/repo",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "PENDING"


@pytest.mark.asyncio
async def test_post_tasks_requires_one_of_url_or_description(async_client):
    """POST /api/tasks fails without url or description."""
    response = await async_client.post("/api/tasks", json={
        "target_repo": "owner/repo",
    })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_tasks_requires_target_repo(async_client):
    """POST /api/tasks fails without target_repo."""
    response = await async_client.post("/api/tasks", json={
        "task_description": "Add something",
    })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_tasks_empty(async_client):
    """GET /api/tasks returns empty list when no tasks exist."""
    response = await async_client.get("/api/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["tasks"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_get_tasks_with_records(async_client):
    """GET /api/tasks returns created tasks."""
    await async_client.post("/api/tasks", json={
        "task_description": "Task 1",
        "target_repo": "owner/repo",
    })
    await async_client.post("/api/tasks", json={
        "task_description": "Task 2",
        "target_repo": "owner/repo",
    })
    response = await async_client.get("/api/tasks")
    data = response.json()
    assert data["total"] == 2


@pytest.mark.asyncio
async def test_get_task_by_id(async_client):
    """GET /api/tasks/{task_id} returns the specific task."""
    create_resp = await async_client.post("/api/tasks", json={
        "task_description": "Specific task",
        "target_repo": "owner/repo",
    })
    task_id = create_resp.json()["task_id"]

    response = await async_client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["task_id"] == task_id
    assert response.json()["task_description"] == "Specific task"


@pytest.mark.asyncio
async def test_get_task_not_found_returns_404(async_client):
    """GET /api/tasks/{task_id} returns 404 for unknown ID."""
    response = await async_client.get("/api/tasks/nonexist")
    assert response.status_code == 404
