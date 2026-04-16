"""Benchmark tracking for skill timing, token usage, and cost."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel

from agent.llm_client import LLMResponse
from agent.storage.interface import StorageInterface

logger = logging.getLogger(__name__)

COST_INPUT_PER_TOKEN = 0.000015
COST_OUTPUT_PER_TOKEN = 0.000075
COST_CACHED_PER_TOKEN = 0.0000015


class LLMCallBenchmark(BaseModel):
    """Benchmark data for a single LLM call."""

    call_index: int
    prompt_summary: str
    elapsed_ms: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    model: str


class SkillBenchmark(BaseModel):
    """Benchmark data for a single skill execution."""

    skill_name: str
    started_at: str
    ended_at: str
    elapsed_ms: int
    llm_calls: List[LLMCallBenchmark]
    llm_total_ms: int
    non_llm_ms: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    estimated_cost_usd: float
    status: str
    retry_count: int


class TaskBenchmark(BaseModel):
    """Benchmark data for a complete task."""

    task_id: str
    started_at: str
    ended_at: str
    total_elapsed_ms: int
    total_elapsed_human: str
    skills: List[SkillBenchmark]
    total_llm_ms: int
    total_non_llm_ms: int
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    estimated_total_cost_usd: float
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    slowest_skill: str
    fastest_skill: str


def _calculate_cost(input_tokens: int, output_tokens: int, cached_tokens: int) -> float:
    """Calculate estimated cost from token counts."""
    return (
        (input_tokens - cached_tokens) * COST_INPUT_PER_TOKEN
        + output_tokens * COST_OUTPUT_PER_TOKEN
        + cached_tokens * COST_CACHED_PER_TOKEN
    )


class BenchmarkTracker:
    """Tracks timing and token usage across skills for a single task."""

    def __init__(self) -> None:
        self._skill_starts: dict[str, float] = {}
        self._skill_start_times: dict[str, str] = {}
        self._skill_llm_calls: dict[str, List[LLMCallBenchmark]] = {}
        self._skills: List[SkillBenchmark] = []
        self._task_start: float = time.monotonic()
        self._task_start_iso: str = datetime.now(timezone.utc).isoformat()

    def start_skill(self, skill_name: str) -> None:
        """Record the start time for a skill."""
        self._skill_starts[skill_name] = time.monotonic()
        self._skill_start_times[skill_name] = datetime.now(timezone.utc).isoformat()
        self._skill_llm_calls[skill_name] = []
        logger.debug("Benchmark: started %s", skill_name)

    def end_skill(self, skill_name: str, status: str, retry_count: int = 0) -> None:
        """Record the end time and compute benchmarks for a skill."""
        start = self._skill_starts.get(skill_name, time.monotonic())
        elapsed_ms = int((time.monotonic() - start) * 1000)
        ended_at = datetime.now(timezone.utc).isoformat()

        llm_calls = self._skill_llm_calls.get(skill_name, [])
        llm_total_ms = sum(c.elapsed_ms for c in llm_calls)
        input_tokens = sum(c.input_tokens for c in llm_calls)
        output_tokens = sum(c.output_tokens for c in llm_calls)
        cached_tokens = sum(c.cached_tokens for c in llm_calls)

        self._skills.append(SkillBenchmark(
            skill_name=skill_name,
            started_at=self._skill_start_times.get(skill_name, ended_at),
            ended_at=ended_at,
            elapsed_ms=elapsed_ms,
            llm_calls=llm_calls,
            llm_total_ms=llm_total_ms,
            non_llm_ms=max(0, elapsed_ms - llm_total_ms),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            estimated_cost_usd=round(_calculate_cost(input_tokens, output_tokens, cached_tokens), 6),
            status=status,
            retry_count=retry_count,
        ))
        logger.debug("Benchmark: ended %s (%dms, %s)", skill_name, elapsed_ms, status)

    def record_llm_call(
        self, skill_name: str, response: LLMResponse, prompt_summary: str
    ) -> None:
        """Record an LLM call's metrics for a skill."""
        calls = self._skill_llm_calls.setdefault(skill_name, [])
        calls.append(LLMCallBenchmark(
            call_index=len(calls) + 1,
            prompt_summary=prompt_summary[:100],
            elapsed_ms=response.elapsed_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            model=response.model,
        ))

    def get_task_benchmark(
        self,
        task_id: str,
        pr_url: Optional[str] = None,
        pr_number: Optional[int] = None,
    ) -> TaskBenchmark:
        """Aggregate all skill benchmarks into a task-level summary."""
        ended_at = datetime.now(timezone.utc).isoformat()
        total_elapsed_ms = int((time.monotonic() - self._task_start) * 1000)

        total_llm_ms = sum(s.llm_total_ms for s in self._skills)
        total_input = sum(s.input_tokens for s in self._skills)
        total_output = sum(s.output_tokens for s in self._skills)
        total_cached = sum(s.cached_tokens for s in self._skills)

        # Format human-readable time
        seconds = total_elapsed_ms // 1000
        if seconds >= 60:
            total_elapsed_human = f"{seconds // 60}m {seconds % 60}s"
        else:
            total_elapsed_human = f"{seconds}s"

        # Find slowest/fastest
        if self._skills:
            slowest = max(self._skills, key=lambda s: s.elapsed_ms).skill_name
            fastest = min(self._skills, key=lambda s: s.elapsed_ms).skill_name
        else:
            slowest = ""
            fastest = ""

        return TaskBenchmark(
            task_id=task_id,
            started_at=self._task_start_iso,
            ended_at=ended_at,
            total_elapsed_ms=total_elapsed_ms,
            total_elapsed_human=total_elapsed_human,
            skills=self._skills,
            total_llm_ms=total_llm_ms,
            total_non_llm_ms=max(0, total_elapsed_ms - total_llm_ms),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cached_tokens=total_cached,
            estimated_total_cost_usd=round(_calculate_cost(total_input, total_output, total_cached), 6),
            pr_url=pr_url,
            pr_number=pr_number,
            slowest_skill=slowest,
            fastest_skill=fastest,
        )

    async def save(self, task_id: str, storage: StorageInterface) -> None:
        """Save benchmark data to storage."""
        benchmark = self.get_task_benchmark(task_id)
        await storage.write(f"tasks/{task_id}/benchmark", benchmark.model_dump())
        await storage.append("benchmarks", benchmark.model_dump())
        logger.info("Saved benchmark for task %s", task_id)
