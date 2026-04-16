"""Orchestrator tests with mocked skills."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm_client import LLMClient
from agent.orchestrator import Orchestrator
from agent.skill_base import Skill, SkillError
from agent.state_manager import StateManager, TaskState
from agent.storage.local_volume import LocalVolumeStorage


class MockSkill(Skill):
    """A mock skill that returns preset output."""

    def __init__(self, name: str, output: dict, should_fail: bool = False):
        self.name = name
        self._output = output
        self._should_fail = should_fail
        self.executed = False

    async def execute(self, task_id, context, llm, benchmark, emitter):
        self.executed = True
        benchmark.start_skill(self.name)
        if self._should_fail:
            benchmark.end_skill(self.name, "failed")
            raise SkillError(self.name, "Skill failed")
        benchmark.end_skill(self.name, "success")
        return self._output


def _make_skills(clarification_clear=True, tests_pass=True):
    """Create 7 mock skills for the pipeline."""
    return [
        MockSkill("issue_reader", {
            "requirement": {
                "title": "Test Feature",
                "description": "Test",
                "requirements": ["req1"],
                "acceptance_criteria": ["ac1"],
                "source": "free_text",
                "issue_number": None,
                "issue_url": None,
            }
        }),
        MockSkill("clarifier", {
            "clarification": {
                "is_clear": clarification_clear,
                "questions": [] if clarification_clear else [
                    {"id": "q1", "question": "Which?", "options": [
                        {"id": "a", "label": "A", "value": "a"},
                        {"id": "other", "label": "Other", "value": None},
                    ]}
                ],
                "reasoning": "test",
            }
        }),
        MockSkill("codebase_explorer", {
            "codebase": {
                "file_tree": ["app.py"],
                "relevant_files": [],
                "architecture_summary": "Simple app",
                "entry_points": [],
                "test_files": [],
                "total_token_estimate": 100,
            }
        }),
        MockSkill("code_writer", {
            "code_changes": [{"path": "app.py", "new_content": "pass", "change_summary": "Added"}],
            "implementation_notes": "Done",
        }),
        MockSkill("test_writer", {
            "test_changes": [{"path": "tests/test_app.py", "new_content": "pass", "change_summary": "Tests"}],
        }),
        MockSkill("test_runner", {
            "test_results": {
                "passed": tests_pass,
                "output": "1 passed" if tests_pass else "1 failed\nFAILED test_x",
                "failed_tests": [] if tests_pass else ["test_x"],
                "total_tests": 1,
                "passed_count": 1 if tests_pass else 0,
                "failed_count": 0 if tests_pass else 1,
            }
        }),
        MockSkill("pr_creator", {
            "pr_result": {
                "pr_url": "https://github.com/test/repo/pull/1",
                "pr_number": 1,
                "branch_name": "feature/agent-test",
                "commit_sha": "abc123",
            }
        }),
    ]


@pytest.fixture
def storage(tmp_data_dir):
    return LocalVolumeStorage(tmp_data_dir)


@pytest.fixture
def state(storage):
    return StateManager(storage)


@pytest.fixture
def emitter(state):
    return EventEmitter(state)


@pytest.fixture
def mock_create_llm():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.call = AsyncMock()
    mock_llm.parse_json = AsyncMock()
    with patch.object(Orchestrator, "_create_llm", return_value=mock_llm):
        yield mock_llm


@pytest.mark.asyncio
async def test_orchestrator_runs_all_skills_in_order(storage, state, emitter, mock_create_llm):
    """Orchestrator executes all 7 skills when everything succeeds."""
    skills = _make_skills()
    orch = Orchestrator(skills, state, emitter, storage)

    task = await state.create_task("test-orch-1", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}, "anthropic_api_key": "test-key"})
    await orch.run(task["task_id"])

    for skill in skills:
        assert skill.executed

    updated = await state.get_task(task["task_id"])
    assert updated["status"] == TaskState.DONE.value


@pytest.mark.asyncio
async def test_orchestrator_pauses_on_clarification_needed(storage, state, emitter, mock_create_llm):
    """Orchestrator pauses when clarifier says requirement is unclear."""
    skills = _make_skills(clarification_clear=False)
    orch = Orchestrator(skills, state, emitter, storage)

    task = await state.create_task("test-orch-2", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}, "anthropic_api_key": "test-key"})
    await orch.run(task["task_id"])

    updated = await state.get_task(task["task_id"])
    assert updated["status"] == TaskState.AWAITING_CLARIFICATION.value
    # Skills after clarifier should NOT have executed
    assert not skills[2].executed  # codebase_explorer


@pytest.mark.asyncio
async def test_orchestrator_resumes_after_clarification(storage, state, emitter, mock_create_llm):
    """Orchestrator resumes from step 3 after clarification answers."""
    skills = _make_skills(clarification_clear=False)
    orch = Orchestrator(skills, state, emitter, storage)

    task = await state.create_task("test-orch-3", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}, "anthropic_api_key": "test-key"})
    await orch.run(task["task_id"])

    # Provide answers
    await state.set_clarification_answers(
        task["task_id"],
        [{"question_id": "q1", "answer": "Option A"}],
    )

    # Resume with fresh skills
    resume_skills = _make_skills()
    orch2 = Orchestrator(resume_skills, state, emitter, storage)
    await orch2.resume_after_clarify(task["task_id"])

    updated = await state.get_task(task["task_id"])
    assert updated["status"] == TaskState.DONE.value


@pytest.mark.asyncio
async def test_orchestrator_retries_code_writer_on_test_failure(storage, state, emitter, mock_create_llm):
    """Orchestrator retries code_writer when tests fail."""
    skills = _make_skills()
    call_count = {"test_runner": 0}

    class RetryTestRunner(MockSkill):
        async def execute(self, task_id, context, llm, benchmark, emitter):
            self.executed = True
            benchmark.start_skill(self.name)
            call_count["test_runner"] += 1
            if call_count["test_runner"] < 2:
                benchmark.end_skill(self.name, "failed")
                return {"test_results": {
                    "passed": False, "output": "FAILED", "failed_tests": ["test_x"],
                    "total_tests": 1, "passed_count": 0, "failed_count": 1,
                }}
            benchmark.end_skill(self.name, "success")
            return {"test_results": {
                "passed": True, "output": "1 passed", "failed_tests": [],
                "total_tests": 1, "passed_count": 1, "failed_count": 0,
            }}

    skills[5] = RetryTestRunner("test_runner", {})
    orch = Orchestrator(skills, state, emitter, storage)

    task = await state.create_task("test-orch-4", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}, "anthropic_api_key": "test-key"})
    await orch.run(task["task_id"])

    updated = await state.get_task(task["task_id"])
    assert updated["status"] == TaskState.DONE.value
    assert call_count["test_runner"] == 2


@pytest.mark.asyncio
async def test_orchestrator_fails_after_max_retries(storage, state, emitter, mock_create_llm):
    """Orchestrator fails task after max test retries exhausted."""
    skills = _make_skills(tests_pass=False)
    orch = Orchestrator(skills, state, emitter, storage)

    task = await state.create_task("test-orch-5", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}, "anthropic_api_key": "test-key"})
    await orch.run(task["task_id"])

    updated = await state.get_task(task["task_id"])
    assert updated["status"] == TaskState.FAILED.value


@pytest.mark.asyncio
async def test_orchestrator_emits_task_done_on_success(storage, state, emitter, mock_create_llm):
    """Orchestrator emits task_done event on successful completion."""
    skills = _make_skills()
    orch = Orchestrator(skills, state, emitter, storage)

    task = await state.create_task("test-orch-6", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}, "anthropic_api_key": "test-key"})
    await orch.run(task["task_id"])

    events = await state.get_events(task["task_id"])
    event_types = [e["type"] for e in events]
    assert "task_start" in event_types
    assert "task_done" in event_types


@pytest.mark.asyncio
async def test_orchestrator_emits_task_failed_on_skill_error(storage, state, emitter, mock_create_llm):
    """Orchestrator emits task_failed event when a skill raises SkillError."""
    skills = _make_skills()
    skills[0] = MockSkill("issue_reader", {}, should_fail=True)
    orch = Orchestrator(skills, state, emitter, storage)

    task = await state.create_task("test-orch-7", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}, "anthropic_api_key": "test-key"})
    await orch.run(task["task_id"])

    events = await state.get_events(task["task_id"])
    event_types = [e["type"] for e in events]
    assert "task_failed" in event_types


@pytest.mark.asyncio
async def test_orchestrator_saves_benchmark_on_completion(storage, state, emitter, mock_create_llm):
    """Orchestrator saves benchmark data after task completion."""
    skills = _make_skills()
    orch = Orchestrator(skills, state, emitter, storage)

    task = await state.create_task("test-orch-8", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}, "anthropic_api_key": "test-key"})
    await orch.run(task["task_id"])

    benchmark_data = await storage.read(f"tasks/{task['task_id']}/benchmark")
    assert benchmark_data is not None
    assert benchmark_data["task_id"] == task["task_id"]
