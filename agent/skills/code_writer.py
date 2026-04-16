"""Skill 4: Writes code changes via Claude."""

from __future__ import annotations

import logging
from typing import Any

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm_client import LLMClient
from agent.skill_base import Skill, SkillError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are implementing a software feature in an existing Python codebase. Return complete file contents — never partial diffs.

Rules:
- Follow existing code style exactly
- Follow existing patterns exactly (service layer, schemas, routers)
- Never change code unrelated to the feature
- Never add comments that were not in original
- Always include Alembic migration if model changes
- Return complete file contents for every changed file

Return ONLY this JSON:
{
  "file_changes": [
    {
      "path": "relative/path/file.py",
      "new_content": "complete file content",
      "change_summary": "one sentence"
    }
  ],
  "implementation_notes": "brief technical notes"
}
Return JSON only."""


class CodeWriterSkill(Skill):
    """Writes code changes for the feature using Claude."""

    name = "code_writer"

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
            await self._emit_start(task_id, emitter, 4, 7, "Writing code changes...")

            requirement = context.get("requirement", {})
            clarification = context.get("clarification", {})
            codebase = context.get("codebase", {})
            iteration = context.get("iteration", 0)
            test_failure = context.get("test_failure")

            # Format clarification Q&A
            qa_pairs = ""
            if clarification and clarification.get("answers"):
                answers = clarification["answers"]
                for a in answers:
                    qa_pairs += f"Q: {a.get('question', '')}\nA: {a.get('answer', '')}\n\n"

            # Format relevant files
            relevant_files_str = ""
            for f in codebase.get("relevant_files", []):
                relevant_files_str += f"\n--- {f['path']} ---\n{f['content']}\n"

            user_prompt = (
                f"Implement this feature:\n"
                f"Title: {requirement.get('title', '')}\n"
                f"Requirements: {requirement.get('requirements', [])}\n"
                f"Acceptance criteria: {requirement.get('acceptance_criteria', [])}\n\n"
                f"Clarifications:\n{qa_pairs}\n"
                f"Codebase architecture:\n{codebase.get('architecture_summary', '')}\n\n"
                f"Relevant files:\n{relevant_files_str}\n"
            )

            if iteration > 0 and test_failure:
                user_prompt += (
                    f"\nPrevious attempt failed tests:\n{test_failure}\n"
                    "Fix the implementation to make tests pass.\n"
                )

            user_prompt += "\nReturn complete file contents. JSON only."

            response = await llm.call(system=SYSTEM_PROMPT, user=user_prompt)
            benchmark.record_llm_call(self.name, response, "Write code changes")

            result = await llm.parse_json(response.content)

            code_changes = result.get("file_changes", [])
            for change in code_changes:
                await self._emit_progress(
                    task_id, emitter, f"Writing {change.get('path', '?')}"
                )

            context_update: dict[str, Any] = {
                "code_changes": code_changes,
                "implementation_notes": result.get("implementation_notes", ""),
            }

            benchmark.end_skill(self.name, "success")
            await self._emit_done(task_id, emitter, benchmark)

            return context_update

        except SkillError:
            benchmark.end_skill(self.name, "failed")
            raise
        except Exception as exc:
            benchmark.end_skill(self.name, "failed")
            raise SkillError(self.name, str(exc), detail=str(exc)) from exc
