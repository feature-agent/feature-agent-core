"""API endpoint tests."""

import json

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


# --- Phase 3: SSE streaming tests ---

@pytest.mark.asyncio
async def test_stream_returns_200_with_correct_content_type(async_client):
    """GET /api/stream/{task_id} returns 200 with text/event-stream."""
    create_resp = await async_client.post("/api/tasks", json={
        "task_description": "Stream test",
        "target_repo": "owner/repo",
    })
    task_id = create_resp.json()["task_id"]

    # Emit a terminal event so the stream ends
    import agent.main as main_module
    await main_module.event_emitter.emit(task_id, "task_done", pr_url="", pr_number=0, elapsed_ms=0)

    async with async_client.stream("GET", f"/api/stream/{task_id}") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_stream_replays_existing_events(async_client):
    """GET /api/stream/{task_id} replays stored events."""
    create_resp = await async_client.post("/api/tasks", json={
        "task_description": "Replay test",
        "target_repo": "owner/repo",
    })
    task_id = create_resp.json()["task_id"]

    # Store some events before connecting
    import agent.main as main_module
    await main_module.event_emitter.emit(task_id, "task_start")
    await main_module.event_emitter.emit(task_id, "skill_start", skill="issue_reader", skill_index=1, skill_total=7, message="Reading...")
    await main_module.event_emitter.emit(task_id, "task_done", pr_url="https://github.com/test/pr/1", pr_number=1, elapsed_ms=1000)

    # Connect and read all replayed events
    events = []
    async with async_client.stream("GET", f"/api/stream/{task_id}") as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                events.append(event)

    assert len(events) >= 3
    assert events[0]["type"] == "task_start"
    assert events[1]["type"] == "skill_start"
    assert events[2]["type"] == "task_done"


@pytest.mark.asyncio
async def test_stream_404_for_unknown_task(async_client):
    """GET /api/stream/{task_id} returns 404 for non-existent task."""
    response = await async_client.get("/api/stream/nonexist")
    assert response.status_code == 404


# --- Phase 5: Clarification API tests ---

async def _create_awaiting_task(async_client):
    """Helper: create a task and set it to AWAITING_CLARIFICATION."""
    import agent.main as main_module
    create_resp = await async_client.post("/api/tasks", json={
        "task_description": "Clarify test",
        "target_repo": "owner/repo",
    })
    task_id = create_resp.json()["task_id"]

    questions = [
        {"id": "q1", "question": "Which format?", "options": [
            {"id": "a", "label": "JSON", "value": "json"},
            {"id": "b", "label": "XML", "value": "xml"},
            {"id": "c", "label": "CSV", "value": "csv"},
            {"id": "other", "label": "Other (type your own)", "value": None},
        ]}
    ]
    await main_module.state_manager.set_clarification_questions(task_id, questions)
    return task_id


@pytest.mark.asyncio
async def test_clarify_resumes_task(async_client):
    """POST /api/tasks/{id}/clarify saves answers and returns RESUMED."""
    task_id = await _create_awaiting_task(async_client)

    response = await async_client.post(f"/api/tasks/{task_id}/clarify", json={
        "answers": [{
            "question_id": "q1",
            "question": "Which format?",
            "selected_option_id": "a",
            "selected_option_label": "JSON",
            "answer": "json",
        }],
    })
    assert response.status_code == 200
    assert response.json()["status"] == "RESUMED"


@pytest.mark.asyncio
async def test_clarify_rejects_wrong_status(async_client):
    """POST /api/tasks/{id}/clarify rejects tasks not awaiting clarification."""
    create_resp = await async_client.post("/api/tasks", json={
        "task_description": "Not awaiting",
        "target_repo": "owner/repo",
    })
    task_id = create_resp.json()["task_id"]

    response = await async_client.post(f"/api/tasks/{task_id}/clarify", json={
        "answers": [{"question_id": "q1", "question": "Q?", "selected_option_id": "a",
                      "selected_option_label": "A", "answer": "a"}],
    })
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_clarify_validates_answer_count(async_client):
    """POST /api/tasks/{id}/clarify rejects wrong number of answers."""
    task_id = await _create_awaiting_task(async_client)

    response = await async_client.post(f"/api/tasks/{task_id}/clarify", json={
        "answers": [
            {"question_id": "q1", "question": "Q1?", "selected_option_id": "a",
             "selected_option_label": "A", "answer": "a"},
            {"question_id": "q2", "question": "Q2?", "selected_option_id": "b",
             "selected_option_label": "B", "answer": "b"},
        ],
    })
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_clarify_validates_answer_not_empty(async_client):
    """POST /api/tasks/{id}/clarify rejects empty answers."""
    task_id = await _create_awaiting_task(async_client)

    response = await async_client.post(f"/api/tasks/{task_id}/clarify", json={
        "answers": [{"question_id": "q1", "question": "Q?", "selected_option_id": "a",
                      "selected_option_label": "A", "answer": ""}],
    })
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_clarify_validates_question_ids_match(async_client):
    """POST /api/tasks/{id}/clarify rejects unknown question IDs."""
    task_id = await _create_awaiting_task(async_client)

    response = await async_client.post(f"/api/tasks/{task_id}/clarify", json={
        "answers": [{"question_id": "q99", "question": "Q?", "selected_option_id": "a",
                      "selected_option_label": "A", "answer": "a"}],
    })
    assert response.status_code == 400


# --- Phase 6: Benchmark API tests ---

@pytest.mark.asyncio
async def test_get_benchmarks_empty(async_client):
    """GET /api/benchmarks returns empty list when no benchmarks exist."""
    response = await async_client.get("/api/benchmarks")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_task_benchmark_not_found(async_client):
    """GET /api/tasks/{id}/benchmark returns 404 for unknown task."""
    response = await async_client.get("/api/tasks/nonexist/benchmark")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_task_benchmark_not_complete(async_client):
    """GET /api/tasks/{id}/benchmark returns 400 for incomplete task."""
    create_resp = await async_client.post("/api/tasks", json={
        "task_description": "Benchmark test",
        "target_repo": "owner/repo",
    })
    task_id = create_resp.json()["task_id"]

    response = await async_client.get(f"/api/tasks/{task_id}/benchmark")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_task_benchmark_after_completion(async_client):
    """GET /api/tasks/{id}/benchmark returns data for completed task."""
    import agent.main as main_module

    create_resp = await async_client.post("/api/tasks", json={
        "task_description": "Completed task",
        "target_repo": "owner/repo",
    })
    task_id = create_resp.json()["task_id"]

    # Simulate completion
    await main_module.state_manager.update_task(task_id, status="DONE")
    await main_module.storage.write(f"tasks/{task_id}/benchmark", {
        "task_id": task_id,
        "total_elapsed_ms": 5000,
        "total_elapsed_human": "5s",
        "skills": [],
    })

    response = await async_client.get(f"/api/tasks/{task_id}/benchmark")
    assert response.status_code == 200
    assert response.json()["task_id"] == task_id
