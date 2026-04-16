"""Skill 3: Clones and explores a target codebase to find relevant files."""

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

SYSTEM_PROMPT = """You are analyzing a Python codebase to find files relevant to implementing a feature.
Return ONLY this JSON:
{
  "relevant_file_paths": ["path1", "path2"],
  "architecture_summary": "one paragraph",
  "entry_points": ["main.py", "app/main.py"],
  "test_files": ["tests/test_x.py"]
}
Include only files directly needed.
Do not include tangentially related files.
Return JSON only."""


class CodebaseExplorerSkill(Skill):
    """Clones target repo and identifies relevant files for the feature."""

    name = "codebase_explorer"

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
            await self._emit_start(task_id, emitter, 3, 7, "Exploring codebase...")

            requirement = context.get("requirement", {})
            target_repo = context.get("target_repo", "")

            # Clone repo
            tmp_dir = tempfile.mkdtemp(prefix="agent_explore_")
            repo_dir = os.path.join(tmp_dir, "repo")
            clone_url = f"https://x-access-token:{settings.GITHUB_TOKEN}@github.com/{target_repo}.git"

            subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, repo_dir],
                capture_output=True, text=True, check=True,
            )
            await self._emit_progress(task_id, emitter, f"Cloned {target_repo}")

            # Build file tree and compressed contents
            file_tree = []
            compressed_contents = {}
            repo_path = Path(repo_dir)

            for py_file in sorted(repo_path.rglob("*.py")):
                rel_path = str(py_file.relative_to(repo_path))
                if "__pycache__" in rel_path or ".git" in rel_path:
                    continue
                file_tree.append(rel_path)

                lines = py_file.read_text(errors="replace").split("\n")
                if len(lines) <= 100:
                    compressed_contents[rel_path] = "\n".join(lines)
                else:
                    head = "\n".join(lines[:50])
                    tail = "\n".join(lines[-10:])
                    compressed_contents[rel_path] = (
                        f"{head}\n... ({len(lines) - 60} more lines) ...\n{tail}"
                    )

            # Ask Claude to identify relevant files
            contents_str = "\n\n".join(
                f"--- {path} ---\n{content}"
                for path, content in compressed_contents.items()
            )

            response = await llm.call(
                system=SYSTEM_PROMPT,
                user=(
                    f"Feature: {requirement.get('title', '')}\n"
                    f"File tree: {file_tree}\n"
                    f"File contents:\n{contents_str}\n\n"
                    "Find relevant files. Return JSON only."
                ),
            )
            benchmark.record_llm_call(self.name, response, "Identify relevant files")

            analysis = await llm.parse_json(response.content)

            # Read full content of relevant files
            relevant_files = []
            for path in analysis.get("relevant_file_paths", []):
                full_path = repo_path / path
                if full_path.exists():
                    content = full_path.read_text(errors="replace")
                    relevant_files.append({
                        "path": path,
                        "content": content,
                        "relevance_reason": "identified by codebase analysis",
                    })
                    await self._emit_log(task_id, emitter, f"Relevant: {path}")

            arch_summary = analysis.get("architecture_summary", "")
            await self._emit_log(task_id, emitter, f"Architecture: {arch_summary[:200]}")

            # Estimate total tokens
            total_chars = sum(len(f["content"]) for f in relevant_files)
            token_estimate = total_chars // 4

            codebase = {
                "file_tree": file_tree,
                "relevant_files": relevant_files,
                "architecture_summary": arch_summary,
                "entry_points": analysis.get("entry_points", []),
                "test_files": analysis.get("test_files", []),
                "total_token_estimate": token_estimate,
            }

            benchmark.end_skill(self.name, "success")
            await self._emit_done(task_id, emitter, benchmark)

            return {"codebase": codebase}

        except SkillError:
            benchmark.end_skill(self.name, "failed")
            raise
        except Exception as exc:
            benchmark.end_skill(self.name, "failed")
            raise SkillError(self.name, str(exc), detail=str(exc)) from exc
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
