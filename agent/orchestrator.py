"""Skill pipeline orchestrator with clarification pause/resume and test retry."""

from __future__ import annotations

import logging
import time
from typing import Any, List

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm import LLMProvider, create_provider
from agent.skill_base import Skill, SkillError
from agent.state_manager import StateManager, TaskState
from agent.storage.interface import StorageInterface

logger = logging.getLogger(__name__)

MAX_TEST_RETRIES = 4


class TaskCanceled(Exception):
    """Raised when a task is canceled mid-pipeline."""


class Orchestrator:
    """Coordinates skill execution pipeline for a task."""

    def __init__(
        self,
        skills: List[Skill],
        state: StateManager,
        emitter: EventEmitter,
        storage: StorageInterface,
    ) -> None:
        self._skills = skills
        self._state = state
        self._emitter = emitter
        self._storage = storage

    def _create_provider(self, provider_type: str, credentials: dict) -> LLMProvider:
        """Create an LLM provider from the per-task provider config."""
        return create_provider(provider_type, credentials)

    async def run(self, task_id: str) -> None:
        """Execute the full skill pipeline for a new task."""
        task = await self._state.get_task(task_id)
        if task is None:
            logger.error("Task %s not found", task_id)
            return

        provider = task.get("provider", {})
        provider_type = provider.get("provider_type", "")
        credentials = provider.get("credentials", {})
        if not provider_type or not credentials:
            await self._handle_failure(task_id, BenchmarkTracker(), "Missing provider_type or credentials")
            return

        llm = self._create_provider(provider_type, credentials)
        benchmark = BenchmarkTracker()
        await self._state.update_task(task_id, status=TaskState.RUNNING.value)
        await self._emitter.emit(task_id, "task_start")

        context: dict[str, Any] = {
            "task_id": task_id,
            "target_repo": task["target_repo"],
            "github_token": provider.get("github_token", ""),
            "github_issue_url": task.get("github_issue_url"),
            "task_description": task.get("task_description"),
            "iteration": 0,
            "test_failure": None,
        }

        try:
            # Step 1: issue_reader
            result = await self._run_skill(
                self._skills[0], task_id, context, llm, benchmark
            )
            context.update(result)

            # Step 2: clarifier
            result = await self._run_skill(
                self._skills[1], task_id, context, llm, benchmark
            )
            context.update(result)

            # Check if clarification needed — pause pipeline
            clarification = context.get("clarification", {})
            if not clarification.get("is_clear", True) and clarification.get("questions"):
                await self._state.set_clarification_questions(
                    task_id, clarification["questions"]
                )
                # Save context to task for later resume
                await self._state.update_task(task_id, context=context)
                await benchmark.save(task_id, self._storage)
                logger.info("Task %s paused for clarification", task_id)
                return

            # Steps 3-7: continue pipeline
            await self._run_remaining_pipeline(
                task_id, context, llm, benchmark
            )

        except TaskCanceled:
            await self._handle_canceled(task_id, benchmark)
        except SkillError as exc:
            await self._handle_failure(task_id, benchmark, str(exc))
        except Exception as exc:
            await self._handle_failure(task_id, benchmark, f"Unexpected error: {exc}")

    async def resume_after_clarify(self, task_id: str) -> None:
        """Resume pipeline after clarification answers are received."""
        task = await self._state.get_task(task_id)
        if task is None:
            logger.error("Task %s not found for resume", task_id)
            return

        valid_statuses = {TaskState.AWAITING_CLARIFICATION.value, TaskState.RUNNING.value}
        if task["status"] not in valid_statuses:
            logger.error("Task %s in unexpected status %s for resume", task_id, task["status"])
            return

        provider = task.get("provider", {})
        provider_type = provider.get("provider_type", "")
        credentials = provider.get("credentials", {})
        if not provider_type or not credentials:
            await self._handle_failure(task_id, BenchmarkTracker(), "Missing provider_type or credentials")
            return

        llm = self._create_provider(provider_type, credentials)
        benchmark = BenchmarkTracker()
        await benchmark.restore_from(task_id, self._storage)
        context = task.get("context", {})
        # Ensure github_token is in context for resumed tasks
        if "github_token" not in context:
            context["github_token"] = provider.get("github_token", "")

        # Add clarification answers to context
        clarification = task.get("clarification", {})
        context["clarification"] = clarification

        await self._state.update_task(task_id, status=TaskState.RUNNING.value)
        await self._emitter.emit(
            task_id,
            "clarification_received",
            answer_count=len(clarification.get("answers", [])),
        )

        try:
            await self._run_remaining_pipeline(
                task_id, context, llm, benchmark
            )
        except TaskCanceled:
            await self._handle_canceled(task_id, benchmark)
        except SkillError as exc:
            await self._handle_failure(task_id, benchmark, str(exc))
        except Exception as exc:
            await self._handle_failure(task_id, benchmark, f"Unexpected error: {exc}")

    async def _run_remaining_pipeline(
        self,
        task_id: str,
        context: dict[str, Any],
        llm: LLMProvider,
        benchmark: BenchmarkTracker,
    ) -> None:
        """Run steps 3-7 of the pipeline (codebase_explorer through pr_creator)."""
        # Step 3: codebase_explorer
        result = await self._run_skill(
            self._skills[2], task_id, context, llm, benchmark
        )
        context.update(result)

        # Steps 4-6: code_writer, test_writer, test_runner with retry loop
        for iteration in range(MAX_TEST_RETRIES + 1):
            context["iteration"] = iteration

            # Step 4: code_writer
            result = await self._run_skill(
                self._skills[3], task_id, context, llm, benchmark
            )
            context.update(result)

            # Step 5: test_writer
            result = await self._run_skill(
                self._skills[4], task_id, context, llm, benchmark
            )
            context.update(result)

            # Step 6: test_runner
            result = await self._run_skill(
                self._skills[5], task_id, context, llm, benchmark
            )
            context.update(result)

            test_results = context.get("test_results", {})
            if test_results.get("passed", False):
                break

            if iteration < MAX_TEST_RETRIES:
                context["test_failure"] = test_results.get("output", "")
                logger.info(
                    "Task %s: tests failed, retrying (attempt %d/%d)",
                    task_id, iteration + 1, MAX_TEST_RETRIES,
                )
            else:
                raise SkillError(
                    "test_runner",
                    f"Tests failed after {MAX_TEST_RETRIES} retries",
                    detail=test_results.get("output", ""),
                )

        # Step 7: pr_creator
        result = await self._run_skill(
            self._skills[6], task_id, context, llm, benchmark
        )
        context.update(result)

        # Success
        pr_result = context.get("pr_result", {})
        start_ms = int(benchmark._task_start * 1000)
        elapsed_ms = int(time.monotonic() * 1000) - start_ms

        await self._state.update_task(
            task_id,
            status=TaskState.DONE.value,
            result=pr_result,
            context=context,
        )
        await self._emitter.emit(
            task_id,
            "task_done",
            pr_url=pr_result.get("pr_url", ""),
            pr_number=pr_result.get("pr_number", 0),
            elapsed_ms=elapsed_ms,
        )

        # Emit benchmark summary
        task_benchmark = benchmark.get_task_benchmark(
            task_id,
            pr_url=pr_result.get("pr_url"),
            pr_number=pr_result.get("pr_number"),
        )
        await self._emitter.emit(
            task_id,
            "benchmark_summary",
            total_elapsed_ms=task_benchmark.total_elapsed_ms,
            total_elapsed_human=task_benchmark.total_elapsed_human,
            total_agent_ms=task_benchmark.total_agent_ms,
            user_wait_ms=task_benchmark.user_wait_ms,
            skills=[s.model_dump() for s in task_benchmark.skills],
            total_input_tokens=task_benchmark.total_input_tokens,
            total_output_tokens=task_benchmark.total_output_tokens,
            total_cached_tokens=task_benchmark.total_cached_tokens,
            total_cost_usd=task_benchmark.estimated_total_cost_usd,
            slowest_skill=task_benchmark.slowest_skill,
            fastest_skill=task_benchmark.fastest_skill,
        )

        await benchmark.save(task_id, self._storage)
        logger.info("Task %s completed successfully", task_id)

    async def _run_skill(
        self,
        skill: Skill,
        task_id: str,
        context: dict[str, Any],
        llm: LLMProvider,
        benchmark: BenchmarkTracker,
    ) -> dict[str, Any]:
        """Execute a single skill and return its output."""
        await self._check_canceled(task_id)
        try:
            return await skill.execute(
                task_id, context, llm, benchmark, self._emitter
            )
        except SkillError:
            raise
        except Exception as exc:
            raise SkillError(
                skill.name, f"Unexpected error in {skill.name}: {exc}"
            ) from exc

    async def _check_canceled(self, task_id: str) -> None:
        """Raise TaskCanceled if the task has been marked CANCELED."""
        task = await self._state.get_task(task_id)
        if task and task.get("status") == TaskState.CANCELED.value:
            raise TaskCanceled(task_id)

    async def _handle_canceled(
        self, task_id: str, benchmark: BenchmarkTracker
    ) -> None:
        """Handle a canceled task: emit event and persist benchmark."""
        logger.info("Task %s canceled", task_id)
        elapsed_ms = int((time.monotonic() - benchmark._task_start) * 1000)
        await self._emitter.emit(
            task_id,
            "task_failed",
            reason="Task canceled by user",
            elapsed_ms=elapsed_ms,
        )
        await benchmark.save(task_id, self._storage)

    async def _handle_failure(
        self,
        task_id: str,
        benchmark: BenchmarkTracker,
        error_msg: str,
    ) -> None:
        """Handle task failure: update state, emit event, save benchmark."""
        logger.error("Task %s failed: %s", task_id, error_msg)
        elapsed_ms = int((time.monotonic() - benchmark._task_start) * 1000)

        await self._state.update_task(
            task_id,
            status=TaskState.FAILED.value,
            error=error_msg,
        )
        await self._emitter.emit(
            task_id,
            "task_failed",
            reason=error_msg,
            elapsed_ms=elapsed_ms,
        )
        await benchmark.save(task_id, self._storage)
