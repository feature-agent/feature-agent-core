"""State manager tests."""

import pytest

from agent.state_manager import StateManager, TaskState
from agent.storage.local_volume import LocalVolumeStorage


@pytest.fixture
def state_manager(tmp_data_dir):
    storage = LocalVolumeStorage(tmp_data_dir)
    return StateManager(storage)


@pytest.mark.asyncio
async def test_create_task_uses_provided_id(state_manager):
    """Created task uses the provided task_id."""
    t1 = await state_manager.create_task("sm-1", "free_text", None, "task 1", "owner/repo")
    assert t1["task_id"] == "sm-1"
    t2 = await state_manager.create_task("sm-2", "free_text", None, "task 2", "owner/repo")
    assert t2["task_id"] == "sm-2"


@pytest.mark.asyncio
async def test_create_task_sets_pending_status(state_manager):
    """New tasks start in PENDING status."""
    task = await state_manager.create_task("sm-3", "free_text", None, "do something", "owner/repo")
    assert task["status"] == TaskState.PENDING.value


@pytest.mark.asyncio
async def test_get_task_returns_none_if_missing(state_manager):
    """Getting a non-existent task returns None."""
    result = await state_manager.get_task("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_update_task_updates_fields(state_manager):
    """update_task modifies specified fields and updated_at."""
    task = await state_manager.create_task("sm-4", "free_text", None, "test", "owner/repo")
    original_updated = task["updated_at"]

    updated = await state_manager.update_task(
        task["task_id"], status=TaskState.RUNNING.value
    )
    assert updated["status"] == TaskState.RUNNING.value
    assert updated["updated_at"] >= original_updated


@pytest.mark.asyncio
async def test_list_tasks_sorted_by_created_at(state_manager):
    """list_tasks returns tasks sorted by created_at descending."""
    await state_manager.create_task("sm-5", "free_text", None, "first", "owner/repo")
    await state_manager.create_task("sm-6", "free_text", None, "second", "owner/repo")
    tasks = await state_manager.list_tasks()
    assert len(tasks) == 2
    assert tasks[0]["created_at"] >= tasks[1]["created_at"]


@pytest.mark.asyncio
async def test_append_event_and_get_events(state_manager):
    """Events can be appended and retrieved in order."""
    task = await state_manager.create_task("sm-7", "free_text", None, "test", "owner/repo")
    tid = task["task_id"]

    await state_manager.append_event(tid, {"type": "task_start", "n": 1})
    await state_manager.append_event(tid, {"type": "skill_start", "n": 2})

    events = await state_manager.get_events(tid)
    assert len(events) == 2
    assert events[0]["n"] == 1
    assert events[1]["n"] == 2


@pytest.mark.asyncio
async def test_set_clarification_questions(state_manager):
    """Setting questions moves task to AWAITING_CLARIFICATION."""
    task = await state_manager.create_task("sm-8", "free_text", None, "test", "owner/repo")
    questions = [{"id": "q1", "question": "Which?", "options": []}]

    updated = await state_manager.set_clarification_questions(
        task["task_id"], questions
    )
    assert updated["status"] == TaskState.AWAITING_CLARIFICATION.value
    assert updated["clarification"]["questions"] == questions


@pytest.mark.asyncio
async def test_set_clarification_answers(state_manager):
    """Setting answers moves task to RUNNING."""
    task = await state_manager.create_task("sm-9", "free_text", None, "test", "owner/repo")
    questions = [{"id": "q1", "question": "Which?", "options": []}]
    await state_manager.set_clarification_questions(task["task_id"], questions)

    answers = [{"question_id": "q1", "answer": "Option A"}]
    updated = await state_manager.set_clarification_answers(task["task_id"], answers)
    assert updated["status"] == TaskState.RUNNING.value
    assert updated["clarification"]["answers"] == answers


@pytest.mark.asyncio
async def test_create_task_rejects_duplicate_id(tmp_data_dir):
    """create_task raises ValueError for duplicate task_id."""
    storage = LocalVolumeStorage(tmp_data_dir)
    sm = StateManager(storage)
    await sm.create_task("dup-1", "free_text", None, "desc", "owner/repo")
    with pytest.raises(ValueError, match="already exists"):
        await sm.create_task("dup-1", "free_text", None, "other", "owner/repo")
