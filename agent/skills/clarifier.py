"""Skill 2: Detects ambiguity and generates clarification questions."""

from __future__ import annotations

import logging
from typing import Any

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm import LLMProvider
from agent.skill_base import Skill, SkillError

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 3

SYSTEM_PROMPT = """You are a senior engineer reviewing a feature request before implementation. Identify only ambiguities that would directly cause incorrect implementation choices.

Rules:
- Maximum 3 questions. No more.
- Only ask if the answer changes the code.
- Generate exactly 3 options per question (plus always a 4th 'Other' option).
- Options must be concrete and mutually exclusive.
- Include a realistic example in option labels where helpful.
- Last option always: {"id": "other", "label": "Other (type your own)", "value": null}
- If requirement is clear: is_clear=true, questions=[]

Return ONLY this JSON:
{
  "is_clear": true or false,
  "questions": [
    {
      "id": "q1",
      "question": "question text",
      "options": [
        {"id": "a", "label": "option A", "value": "val_a"},
        {"id": "b", "label": "option B", "value": "val_b"},
        {"id": "c", "label": "option C", "value": "val_c"},
        {"id": "other", "label": "Other (type your own)", "value": null}
      ]
    }
  ],
  "reasoning": "one sentence"
}
Return JSON only."""


class ClarifierSkill(Skill):
    """Analyzes requirements for ambiguity and generates clarification questions."""

    name = "clarifier"

    async def execute(
        self,
        task_id: str,
        context: dict[str, Any],
        llm: LLMProvider,
        benchmark: BenchmarkTracker,
        emitter: EventEmitter,
    ) -> dict[str, Any]:
        benchmark.start_skill(self.name)
        try:
            await self._emit_start(task_id, emitter, 2, 7, "Analyzing requirement clarity...")

            requirement = context.get("requirement", {})

            response = await llm.call(
                system=SYSTEM_PROMPT,
                user=(
                    f"Feature: {requirement.get('title', '')}\n"
                    f"Description: {requirement.get('description', '')}\n"
                    f"Requirements: {requirement.get('requirements', [])}\n\n"
                    "Identify ambiguities. Return JSON only."
                ),
                max_tokens=1024,
                model="fast",
            )
            benchmark.record_llm_call(self.name, response, "Analyze requirement clarity")

            clarification = await llm.parse_json(response.content)

            # HARD RULE: max 3 questions — never trust LLM to count
            if "questions" in clarification:
                clarification["questions"] = clarification["questions"][:MAX_QUESTIONS]

            if not clarification.get("is_clear", True) and clarification.get("questions"):
                await emitter.emit(
                    task_id,
                    "clarification_needed",
                    questions=clarification["questions"],
                    question_count=len(clarification["questions"]),
                )
            else:
                clarification["is_clear"] = True
                clarification["questions"] = []
                await self._emit_log(task_id, emitter, "Requirement is clear, proceeding")

            benchmark.end_skill(self.name, "success")
            await self._emit_done(task_id, emitter, benchmark)

            return {"clarification": clarification}

        except SkillError:
            benchmark.end_skill(self.name, "failed")
            raise
        except Exception as exc:
            benchmark.end_skill(self.name, "failed")
            raise SkillError(self.name, str(exc), detail=str(exc)) from exc
