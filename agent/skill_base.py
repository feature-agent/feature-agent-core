"""Abstract skill base class and SkillError exception."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm_client import LLMClient


class SkillError(Exception):
    """Raised when a skill encounters an unrecoverable error."""

    def __init__(self, skill_name: str, message: str, detail: str = "") -> None:
        self.skill_name = skill_name
        self.message = message
        self.detail = detail
        super().__init__(f"[{skill_name}] {message}")


class Skill(ABC):
    """Abstract base class for all pipeline skills."""

    name: str = ""

    @abstractmethod
    async def execute(
        self,
        task_id: str,
        context: dict[str, Any],
        llm: LLMClient,
        benchmark: BenchmarkTracker,
        emitter: EventEmitter,
    ) -> dict[str, Any]:
        """Execute the skill and return data to merge into context."""

    async def _emit_start(
        self,
        task_id: str,
        emitter: EventEmitter,
        skill_index: int,
        skill_total: int,
        message: str,
    ) -> None:
        """Emit a skill_start event."""
        await emitter.emit(
            task_id,
            "skill_start",
            skill=self.name,
            skill_index=skill_index,
            skill_total=skill_total,
            message=message,
        )

    async def _emit_done(
        self,
        task_id: str,
        emitter: EventEmitter,
        benchmark: BenchmarkTracker,
    ) -> None:
        """Emit a skill_done event with benchmark data."""
        skills = benchmark._skills
        current = next((s for s in reversed(skills) if s.skill_name == self.name), None)
        if current:
            await emitter.emit(
                task_id,
                "skill_done",
                skill=self.name,
                elapsed_ms=current.elapsed_ms,
                llm_ms=current.llm_total_ms,
                input_tokens=current.input_tokens,
                output_tokens=current.output_tokens,
                cached_tokens=current.cached_tokens,
                cost_usd=current.estimated_cost_usd,
            )

    async def _emit_progress(
        self, task_id: str, emitter: EventEmitter, message: str
    ) -> None:
        """Emit a skill_progress event."""
        await emitter.emit(
            task_id, "skill_progress", skill=self.name, message=message
        )

    async def _emit_log(
        self,
        task_id: str,
        emitter: EventEmitter,
        message: str,
        level: str = "info",
    ) -> None:
        """Emit a log event."""
        await emitter.emit(task_id, "log", level=level, message=message)
