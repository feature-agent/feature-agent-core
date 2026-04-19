"""Skill 1: Reads a GitHub issue or structures free text into a requirement."""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm import LLMProvider
from agent.skill_base import Skill, SkillError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are extracting structured requirements from a feature request.
Return ONLY this exact JSON schema:
{
  "title": "concise feature title",
  "description": "full description",
  "requirements": ["req 1", "req 2"],
  "acceptance_criteria": ["ac 1", "ac 2"],
  "source": "github_issue or free_text",
  "issue_number": null,
  "issue_url": null
}
Extract only what is explicitly stated.
Do not invent requirements.
Return JSON only."""


class IssueReaderSkill(Skill):
    """Reads a GitHub issue or free text and structures it into a requirement."""

    name = "issue_reader"

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
            await self._emit_start(task_id, emitter, 1, 7, "Reading feature request...")

            github_issue_url = context.get("github_issue_url")
            task_description = context.get("task_description")

            if github_issue_url:
                raw_content = await self._fetch_github_issue(github_issue_url, context)
            elif task_description:
                raw_content = task_description
            else:
                raise SkillError(self.name, "No issue URL or description provided")

            response = await llm.call(
                system=SYSTEM_PROMPT,
                user=f"Structure this feature request:\n{raw_content}",
                max_tokens=2048,
                model="fast",
            )
            benchmark.record_llm_call(self.name, response, "Structure feature request")

            requirement = await llm.parse_json(response.content)

            # Set source based on input
            if github_issue_url:
                requirement["source"] = "github_issue"
                match = re.search(r"/issues/(\d+)", github_issue_url)
                if match:
                    requirement["issue_number"] = int(match.group(1))
                requirement["issue_url"] = github_issue_url
            else:
                requirement["source"] = "free_text"
                requirement["issue_number"] = None
                requirement["issue_url"] = None

            await self._emit_log(task_id, emitter, f"Requirement: {requirement['title']}")
            benchmark.end_skill(self.name, "success")
            await self._emit_done(task_id, emitter, benchmark)

            return {"requirement": requirement}

        except SkillError:
            benchmark.end_skill(self.name, "failed")
            raise
        except Exception as exc:
            benchmark.end_skill(self.name, "failed")
            raise SkillError(self.name, str(exc), detail=str(exc)) from exc

    async def _fetch_github_issue(
        self, url: str, context: dict[str, Any]
    ) -> str:
        """Fetch issue content from GitHub."""
        from github import Github

        match = re.match(r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)", url)
        if not match:
            raise SkillError(self.name, f"Invalid GitHub issue URL: {url}")

        github_token = context.get("github_token", "")
        owner, repo, number = match.group(1), match.group(2), int(match.group(3))
        g = Github(github_token)
        gh_repo = g.get_repo(f"{owner}/{repo}")
        issue = gh_repo.get_issue(number)

        return f"Title: {issue.title}\n\nBody:\n{issue.body or '(no body)'}"
