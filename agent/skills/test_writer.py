"""Skill 5: Writes pytest tests via Claude."""

from __future__ import annotations

import logging
from typing import Any

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm_client import LLMClient
from agent.skill_base import Skill, SkillError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are writing pytest tests for a new feature in an existing Python codebase.
Match the exact style of existing tests: same fixtures, same assertion patterns, same import conventions.

Return ONLY this JSON:
{
  "test_changes": [
    {
      "path": "tests/test_file.py",
      "new_content": "complete test file content",
      "change_summary": "one sentence"
    }
  ]
}
Return JSON only."""


class TestWriterSkill(Skill):
    """Writes pytest tests for the implemented feature."""

    name = "test_writer"

    async def execute(
        self,
        task_id: str,
        context: dict[str, Any],
        llm: LLMClient,
        benchmark: BenchmarkTracker,
        emitter: EventEmitter,
    ) -> dict[str, Any]:
        benchmark.start_skill(self.name)
        try:
            await self._emit_start(task_id, emitter, 5, 7, "Writing tests...")

            requirement = context.get("requirement", {})
            codebase = context.get("codebase", {})
            code_changes = context.get("code_changes", [])

            # Format code changes
            changes_str = ""
            for change in code_changes:
                changes_str += f"\n--- {change.get('path', '')} ---\n"
                changes_str += f"{change.get('new_content', '')}\n"

            # Find existing test files and conftest for style reference
            existing_test = ""
            conftest = ""
            for f in codebase.get("relevant_files", []):
                if "test_" in f.get("path", ""):
                    existing_test = f["content"]
                    break
            for f in codebase.get("relevant_files", []):
                if "conftest" in f.get("path", ""):
                    conftest = f["content"]
                    break

            response = await llm.call(
                system=SYSTEM_PROMPT,
                user=(
                    f"Write tests for this feature:\n{requirement.get('title', '')}\n\n"
                    f"Acceptance criteria to test:\n{requirement.get('acceptance_criteria', [])}\n\n"
                    f"Code changes made:\n{changes_str}\n\n"
                    f"Existing test file for reference style:\n{existing_test}\n\n"
                    f"Test fixtures available:\n{conftest}\n\n"
                    "Write comprehensive tests. JSON only."
                ),
            )
            benchmark.record_llm_call(self.name, response, "Write tests")

            result = await llm.parse_json(response.content)
            test_changes = result.get("test_changes", [])

            await self._emit_log(
                task_id, emitter, f"Generated {len(test_changes)} test file(s)"
            )

            benchmark.end_skill(self.name, "success")
            await self._emit_done(task_id, emitter, benchmark)

            return {"test_changes": test_changes}

        except SkillError:
            benchmark.end_skill(self.name, "failed")
            raise
        except Exception as exc:
            benchmark.end_skill(self.name, "failed")
            raise SkillError(self.name, str(exc), detail=str(exc)) from exc
