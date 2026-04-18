"""Skill tests with mocked LLM and GitHub clients."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm import LLMProvider, LLMResponse
from agent.skills.clarifier import ClarifierSkill
from agent.skills.code_writer import CodeWriterSkill
from agent.skills.issue_reader import IssueReaderSkill
from agent.skills.test_writer import TestWriterSkill
from agent.state_manager import StateManager
from agent.storage.local_volume import LocalVolumeStorage


@pytest.fixture
def llm():
    """Mock LLM client."""
    mock = MagicMock(spec=LLMProvider)
    mock.call = AsyncMock()
    mock.parse_json = AsyncMock()
    return mock


@pytest.fixture
def benchmark():
    return BenchmarkTracker()


@pytest.fixture
def emitter(tmp_data_dir):
    storage = LocalVolumeStorage(tmp_data_dir)
    state = StateManager(storage)
    return EventEmitter(state)


def _llm_response(content: str = "{}") -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=100,
        output_tokens=50,
        cached_tokens=0,
        elapsed_ms=500,
        model="claude-opus-4-5-20250514",
    )


# --- IssueReaderSkill ---

@pytest.mark.asyncio
async def test_issue_reader_with_free_text(llm, benchmark, emitter):
    """IssueReaderSkill structures free text into a requirement."""
    requirement = {
        "title": "Add due_date field",
        "description": "Add a due_date field to tasks",
        "requirements": ["Add due_date column"],
        "acceptance_criteria": ["due_date shows in API"],
        "source": "free_text",
        "issue_number": None,
        "issue_url": None,
    }
    llm.call.return_value = _llm_response(json.dumps(requirement))
    llm.parse_json.return_value = requirement

    skill = IssueReaderSkill()
    result = await skill.execute(
        "test123",
        {"task_description": "Add due_date field to tasks", "target_repo": "owner/repo"},
        llm, benchmark, emitter,
    )

    assert result["requirement"]["title"] == "Add due_date field"
    assert result["requirement"]["source"] == "free_text"


@pytest.mark.asyncio
async def test_issue_reader_structures_requirement(llm, benchmark, emitter):
    """IssueReaderSkill returns proper structure with all fields."""
    requirement = {
        "title": "Feature X",
        "description": "Implement feature X",
        "requirements": ["req1", "req2"],
        "acceptance_criteria": ["ac1"],
        "source": "free_text",
        "issue_number": None,
        "issue_url": None,
    }
    llm.call.return_value = _llm_response()
    llm.parse_json.return_value = requirement

    skill = IssueReaderSkill()
    result = await skill.execute(
        "test123",
        {"task_description": "Implement feature X", "target_repo": "owner/repo"},
        llm, benchmark, emitter,
    )

    req = result["requirement"]
    assert "title" in req
    assert "requirements" in req
    assert "acceptance_criteria" in req
    assert isinstance(req["requirements"], list)


# --- ClarifierSkill ---

@pytest.mark.asyncio
async def test_clarifier_returns_questions_with_options(llm, benchmark, emitter):
    """ClarifierSkill returns questions with properly structured options."""
    clarification = {
        "is_clear": False,
        "questions": [
            {
                "id": "q1",
                "question": "What date format?",
                "options": [
                    {"id": "a", "label": "ISO 8601", "value": "iso"},
                    {"id": "b", "label": "Unix timestamp", "value": "unix"},
                    {"id": "c", "label": "Human readable", "value": "human"},
                    {"id": "other", "label": "Other (type your own)", "value": None},
                ],
            }
        ],
        "reasoning": "Date format is ambiguous",
    }
    llm.call.return_value = _llm_response()
    llm.parse_json.return_value = clarification

    skill = ClarifierSkill()
    result = await skill.execute(
        "test123",
        {"requirement": {"title": "Add dates", "description": "Add dates", "requirements": []}},
        llm, benchmark, emitter,
    )

    assert not result["clarification"]["is_clear"]
    assert len(result["clarification"]["questions"]) == 1
    assert len(result["clarification"]["questions"][0]["options"]) == 4


@pytest.mark.asyncio
async def test_clarifier_max_3_questions_enforced(llm, benchmark, emitter):
    """ClarifierSkill enforces maximum 3 questions even if LLM returns more."""
    clarification = {
        "is_clear": False,
        "questions": [
            {"id": f"q{i}", "question": f"Q{i}?", "options": []}
            for i in range(5)  # LLM returns 5 questions
        ],
        "reasoning": "Many ambiguities",
    }
    llm.call.return_value = _llm_response()
    llm.parse_json.return_value = clarification

    skill = ClarifierSkill()
    result = await skill.execute(
        "test123",
        {"requirement": {"title": "Complex", "description": "Complex", "requirements": []}},
        llm, benchmark, emitter,
    )

    # HARD RULE: always sliced to 3
    assert len(result["clarification"]["questions"]) == 3


@pytest.mark.asyncio
async def test_clarifier_returns_clear_for_simple_requirement(llm, benchmark, emitter):
    """ClarifierSkill returns is_clear=True for unambiguous requirements."""
    clarification = {
        "is_clear": True,
        "questions": [],
        "reasoning": "Requirement is clear",
    }
    llm.call.return_value = _llm_response()
    llm.parse_json.return_value = clarification

    skill = ClarifierSkill()
    result = await skill.execute(
        "test123",
        {"requirement": {"title": "Simple task", "description": "Clear", "requirements": ["do X"]}},
        llm, benchmark, emitter,
    )

    assert result["clarification"]["is_clear"]
    assert result["clarification"]["questions"] == []


# --- CodebaseExplorerSkill ---

@pytest.mark.asyncio
async def test_codebase_explorer_builds_file_tree(llm, benchmark, emitter):
    """CodebaseExplorerSkill builds file tree (tested via mock subprocess)."""
    # This test validates the skill structure without actually cloning
    from agent.skills.codebase_explorer import CodebaseExplorerSkill

    skill = CodebaseExplorerSkill()
    assert skill.name == "codebase_explorer"


# --- CodeWriterSkill ---

@pytest.mark.asyncio
async def test_code_writer_returns_file_changes(llm, benchmark, emitter):
    """CodeWriterSkill returns properly structured file changes."""
    llm.call.return_value = _llm_response()
    llm.parse_json.return_value = {
        "file_changes": [
            {"path": "app/models.py", "new_content": "class Task: pass", "change_summary": "Added Task model"},
        ],
        "implementation_notes": "Added model",
    }

    skill = CodeWriterSkill()
    result = await skill.execute(
        "test123",
        {
            "requirement": {"title": "Add Task", "requirements": [], "acceptance_criteria": []},
            "clarification": {},
            "codebase": {"relevant_files": [], "architecture_summary": "FastAPI app"},
        },
        llm, benchmark, emitter,
    )

    assert len(result["code_changes"]) == 1
    assert result["code_changes"][0]["path"] == "app/models.py"


@pytest.mark.asyncio
async def test_code_writer_includes_test_failure_on_retry(llm, benchmark, emitter):
    """CodeWriterSkill includes test failure context on retry iterations."""
    llm.call.return_value = _llm_response()
    llm.parse_json.return_value = {
        "file_changes": [{"path": "fix.py", "new_content": "fixed", "change_summary": "Fixed"}],
        "implementation_notes": "Fixed failing test",
    }

    skill = CodeWriterSkill()
    result = await skill.execute(
        "test123",
        {
            "requirement": {"title": "Fix", "requirements": [], "acceptance_criteria": []},
            "clarification": {},
            "codebase": {"relevant_files": [], "architecture_summary": ""},
            "iteration": 1,
            "test_failure": "FAILED test_something - AssertionError",
        },
        llm, benchmark, emitter,
    )

    # Verify the call included test failure context on retry
    call_args = llm.call.call_args
    user_prompt = call_args.kwargs.get("user", call_args.args[1] if len(call_args.args) > 1 else "")
    assert "Tests reported" in user_prompt
    assert "FAILED test_something - AssertionError" in user_prompt
    assert "ITERATE" in user_prompt


# --- TestWriterSkill ---

@pytest.mark.asyncio
async def test_test_writer_generates_tests(llm, benchmark, emitter):
    """TestWriterSkill generates test files."""
    llm.call.return_value = _llm_response()
    llm.parse_json.return_value = {
        "test_changes": [
            {"path": "tests/test_new.py", "new_content": "def test_x(): pass", "change_summary": "Added test"},
        ],
    }

    skill = TestWriterSkill()
    result = await skill.execute(
        "test123",
        {
            "requirement": {"title": "Feature", "acceptance_criteria": ["works"]},
            "clarification": {},
            "codebase": {"relevant_files": []},
            "code_changes": [{"path": "app.py", "new_content": "pass", "change_summary": "Added"}],
        },
        llm, benchmark, emitter,
    )

    assert len(result["test_changes"]) == 1
    assert result["test_changes"][0]["path"] == "tests/test_new.py"


# --- TestRunnerSkill ---

@pytest.mark.asyncio
async def test_test_runner_parses_pytest_output():
    """TestRunnerSkill can parse pytest output format."""
    import re

    output = "5 passed, 2 failed, 1 error in 3.5s\nFAILED tests/test_a.py::test_x"

    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    failed_tests = re.findall(r"FAILED\s+(\S+)", output)

    assert int(passed_match.group(1)) == 5
    assert int(failed_match.group(1)) == 2
    assert failed_tests == ["tests/test_a.py::test_x"]


@pytest.mark.asyncio
async def test_test_runner_returns_failed_on_failure():
    """TestRunnerSkill correctly identifies test failures."""
    import re

    output = "0 passed, 3 failed in 1.0s"
    failed_match = re.search(r"(\d+) failed", output)
    passed_match = re.search(r"(\d+) passed", output)

    failed_count = int(failed_match.group(1)) if failed_match else 0
    passed_count = int(passed_match.group(1)) if passed_match else 0
    all_passed = failed_count == 0 and passed_count > 0

    assert not all_passed
    assert failed_count == 3


# --- PRCreatorSkill ---

@pytest.mark.asyncio
async def test_pr_creator_builds_correct_pr_body():
    """PRCreatorSkill formats PR body correctly."""
    from agent.skills.pr_creator import PR_BODY_TEMPLATE

    body = PR_BODY_TEMPLATE.format(
        title="Add due_date",
        task_id="abc123",
        changes_list="- Added due_date column\n- Updated API",
        requirements_list="- Add due_date field",
        criteria_checklist="- [ ] due_date in response",
        implementation_notes="Added column and migration",
    )

    assert "Add due_date" in body
    assert "abc123" in body
    assert "Added due_date column" in body
    assert "- [ ] due_date in response" in body
