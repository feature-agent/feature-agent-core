"""Skill 7: Creates a GitHub branch and pull request."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent.benchmark import BenchmarkTracker
from agent.config import settings
from agent.event_emitter import EventEmitter
from agent.llm_client import LLMClient
from agent.skill_base import Skill, SkillError

logger = logging.getLogger(__name__)

PR_BODY_TEMPLATE = """## Feature: {title}

Implemented by Feature Agent

Task ID: `{task_id}`

### What changed
{changes_list}

### Requirements implemented
{requirements_list}

### Acceptance criteria
{criteria_checklist}

### How to test
```bash
pytest tests/ -v
```

### Notes
{implementation_notes}

---
*This PR was generated automatically by Feature Agent. Please review carefully.*
"""


class PRCreatorSkill(Skill):
    """Creates a GitHub branch and pull request with all changes."""

    name = "pr_creator"

    async def execute(
        self,
        task_id: str,
        context: dict[str, Any],
        llm: LLMClient,
        benchmark: BenchmarkTracker,
        emitter: EventEmitter,
    ) -> dict[str, Any]:
        benchmark.start_skill(self.name)
        tmp_dir = None
        try:
            await self._emit_start(task_id, emitter, 7, 7, "Creating GitHub PR...")

            requirement = context.get("requirement", {})
            code_changes = context.get("code_changes", [])
            test_changes = context.get("test_changes", [])
            test_results = context.get("test_results", {})
            target_repo = context.get("target_repo", "")
            impl_notes = context.get("implementation_notes", "")

            branch_name = f"feature/agent-{task_id}"

            # Clone repo
            tmp_dir = tempfile.mkdtemp(prefix="agent_pr_")
            repo_dir = os.path.join(tmp_dir, "repo")
            clone_url = f"https://x-access-token:{settings.GITHUB_TOKEN}@github.com/{target_repo}.git"

            subprocess.run(
                ["git", "clone", clone_url, repo_dir],
                capture_output=True, text=True, check=True,
            )

            await self._emit_log(task_id, emitter, f"Creating branch {branch_name}")

            # Create branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                capture_output=True, text=True, check=True, cwd=repo_dir,
            )

            repo_path = Path(repo_dir)

            # Apply code changes
            for change in code_changes:
                file_path = repo_path / change["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(change["new_content"])

            # Apply test changes
            for change in test_changes:
                file_path = repo_path / change["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(change["new_content"])

            # Git add and commit
            subprocess.run(
                ["git", "add", "-A"],
                capture_output=True, text=True, check=True, cwd=repo_dir,
            )

            # Build commit message
            change_bullets = "\n".join(
                f"- {c.get('change_summary', c.get('path', ''))}"
                for c in code_changes + test_changes
            )
            passed = test_results.get("passed_count", 0)
            total = test_results.get("total_tests", 0)

            commit_msg = (
                f"feat: {requirement.get('title', 'Feature implementation')}\n\n"
                f"Implemented by Feature Agent\n"
                f"Task ID: {task_id}\n\n"
                f"Changes:\n{change_bullets}\n\n"
                f"Tests: {passed}/{total} passing"
            )

            subprocess.run(
                ["git", "-c", "user.name=Feature Agent", "-c", "user.email=agent@feature-agent.dev",
                 "commit", "-m", commit_msg],
                capture_output=True, text=True, check=True, cwd=repo_dir,
            )

            # Push
            await self._emit_log(task_id, emitter, "Pushing changes...")
            push_result = subprocess.run(
                ["git", "push", "origin", branch_name],
                capture_output=True, text=True, cwd=repo_dir,
            )
            if push_result.returncode != 0:
                raise SkillError(
                    self.name,
                    f"Git push failed: {push_result.stderr}",
                    detail=push_result.stderr,
                )

            # Get commit SHA
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=repo_dir,
            )
            commit_sha = sha_result.stdout.strip()

            # Create PR via PyGithub
            await self._emit_log(task_id, emitter, "Opening PR...")
            from github import Github

            g = Github(settings.GITHUB_TOKEN)
            gh_repo = g.get_repo(target_repo)

            # Build PR body
            changes_list = "\n".join(
                f"- {c.get('change_summary', c.get('path', ''))}"
                for c in code_changes + test_changes
            )
            requirements_list = "\n".join(
                f"- {r}" for r in requirement.get("requirements", [])
            )
            criteria_checklist = "\n".join(
                f"- [ ] {c}" for c in requirement.get("acceptance_criteria", [])
            )

            pr_body = PR_BODY_TEMPLATE.format(
                title=requirement.get("title", ""),
                task_id=task_id,
                changes_list=changes_list,
                requirements_list=requirements_list,
                criteria_checklist=criteria_checklist,
                implementation_notes=impl_notes,
            )

            pr = gh_repo.create_pull(
                title=requirement.get("title", "Feature implementation"),
                body=pr_body,
                base="main",
                head=branch_name,
            )

            try:
                pr.add_to_labels("agent-generated")
            except Exception:
                logger.debug("Could not add label — may not exist")

            pr_result = {
                "pr_url": pr.html_url,
                "pr_number": pr.number,
                "branch_name": branch_name,
                "commit_sha": commit_sha,
            }

            await self._emit_log(
                task_id, emitter, f"PR created: {pr.html_url}", level="success"
            )

            benchmark.end_skill(self.name, "success")
            await self._emit_done(task_id, emitter, benchmark)

            return {"pr_result": pr_result}

        except SkillError:
            benchmark.end_skill(self.name, "failed")
            raise
        except Exception as exc:
            benchmark.end_skill(self.name, "failed")
            raise SkillError(self.name, str(exc), detail=str(exc)) from exc
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
