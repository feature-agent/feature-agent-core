"""Skill 6: Runs pytest in a cloned repo with applied changes."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm import LLMProvider
from agent.skill_base import Skill, SkillError

logger = logging.getLogger(__name__)


class TestRunnerSkill(Skill):
    """Runs pytest in a temporary copy of the target repo with changes applied."""

    name = "test_runner"

    async def execute(
        self,
        task_id: str,
        context: dict[str, Any],
        llm: LLMProvider,
        benchmark: BenchmarkTracker,
        emitter: EventEmitter,
    ) -> dict[str, Any]:
        benchmark.start_skill(self.name)
        tmp_dir = None
        try:
            await self._emit_start(task_id, emitter, 6, 7, "Running tests...")

            code_changes = context.get("code_changes", [])
            test_changes = context.get("test_changes", [])
            target_repo = context.get("target_repo", "")

            # Clone fresh copy
            tmp_dir = tempfile.mkdtemp(prefix="agent_test_")
            repo_dir = os.path.join(tmp_dir, "repo")
            github_token = context.get("github_token", "")
            clone_url = f"https://x-access-token:{github_token}@github.com/{target_repo}.git"

            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", "main", clone_url, repo_dir],
                capture_output=True, text=True, check=True,
            )

            repo_path = Path(repo_dir)

            from agent.utils.file_changes import apply_changes, ApplyChangesError
            try:
                apply_changes(repo_path, code_changes)
                apply_changes(repo_path, test_changes)
            except ApplyChangesError as exc:
                # Treat as a test failure so the retry loop can try again
                # with the error as feedback to the LLM.
                benchmark.end_skill(self.name, "failed")
                await self._emit_log(task_id, emitter, f"Edit apply failed: {exc}")
                return {
                    "test_results": {
                        "passed": False,
                        "output": (
                            f"Edit could not be applied: {exc}\n"
                            "The old_string in one of your edits did not match the "
                            "current file content. Re-check which file actually contains "
                            "the failing assertion, and provide an old_string that exists "
                            "verbatim in that file."
                        ),
                        "passed_count": 0,
                        "total_tests": 0,
                    }
                }

            # Install requirements
            req_file = repo_path / "requirements.txt"
            if req_file.exists():
                subprocess.run(
                    ["pip", "install", "-r", str(req_file)],
                    capture_output=True, text=True, cwd=repo_dir,
                )

            req_dev = repo_path / "requirements-dev.txt"
            if req_dev.exists():
                subprocess.run(
                    ["pip", "install", "-r", str(req_dev)],
                    capture_output=True, text=True, cwd=repo_dir,
                )

            # Run alembic if present
            alembic_ini = repo_path / "alembic.ini"
            if alembic_ini.exists():
                subprocess.run(
                    ["alembic", "upgrade", "head"],
                    capture_output=True, text=True, cwd=repo_dir,
                )

            # Run pytest
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header", "-q"],
                capture_output=True, text=True, cwd=repo_dir,
            )

            output = result.stdout + result.stderr
            # Log truncated output
            for line in output.split("\n")[:50]:
                if line.strip():
                    await self._emit_log(task_id, emitter, line[:200])

            # Parse results
            passed_match = re.search(r"(\d+) passed", output)
            failed_match = re.search(r"(\d+) failed", output)
            error_match = re.search(r"(\d+) error", output)

            passed_count = int(passed_match.group(1)) if passed_match else 0
            failed_count = int(failed_match.group(1)) if failed_match else 0
            error_count = int(error_match.group(1)) if error_match else 0

            total_tests = passed_count + failed_count + error_count
            all_passed = failed_count == 0 and error_count == 0 and passed_count > 0

            # Extract failed test names
            failed_tests = re.findall(r"FAILED\s+(\S+)", output)

            test_results = {
                "passed": all_passed,
                "output": output[-2000:],  # Keep last 2000 chars
                "failed_tests": failed_tests,
                "total_tests": total_tests,
                "passed_count": passed_count,
                "failed_count": failed_count,
            }

            if not all_passed:
                await emitter.emit(
                    task_id,
                    "skill_error",
                    skill=self.name,
                    message=f"Tests failed: {failed_count} failed, {error_count} errors",
                    detail=output[-1000:],
                    retry_count=context.get("iteration", 0),
                )

            benchmark.end_skill(self.name, "success" if all_passed else "failed")
            await self._emit_done(task_id, emitter, benchmark)

            return {"test_results": test_results}

        except SkillError:
            benchmark.end_skill(self.name, "failed")
            raise
        except Exception as exc:
            benchmark.end_skill(self.name, "failed")
            raise SkillError(self.name, str(exc), detail=str(exc)) from exc
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
