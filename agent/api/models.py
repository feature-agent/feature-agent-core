"""Pydantic API models for requests, responses, and SSE events."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, model_validator


class ProviderConfig(BaseModel):
    """Provider credentials passed per-task from the client."""

    provider_type: str
    github_token: str
    credentials: dict[str, Any] = {}


class TaskCreateRequest(BaseModel):
    """Request body for creating a new task."""

    task_id: str
    github_issue_url: Optional[str] = None
    task_description: Optional[str] = None
    target_repo: str
    provider: ProviderConfig

    @model_validator(mode="after")
    def require_url_or_description(self) -> "TaskCreateRequest":
        if not self.github_issue_url and not self.task_description:
            raise ValueError("One of github_issue_url or task_description is required")
        return self


class TaskCreateResponse(BaseModel):
    """Response after creating a task."""

    task_id: str
    status: str
    created_at: str


class TaskResponse(BaseModel):
    """Full task representation."""

    task_id: str
    status: str
    created_at: str
    updated_at: str
    source: str
    github_issue_url: Optional[str] = None
    task_description: Optional[str] = None
    target_repo: str
    requirement: Optional[dict[str, Any]] = None
    clarification: Optional[dict[str, Any]] = None
    context: dict[str, Any] = {}
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response for listing tasks."""

    tasks: List[TaskResponse]
    total: int


class ClarifyAnswer(BaseModel):
    """A single clarification answer."""

    question_id: str
    question: str
    selected_option_id: str
    selected_option_label: str
    answer: str


class ClarifyRequest(BaseModel):
    """Request body for submitting clarification answers."""

    answers: List[ClarifyAnswer]


# SSE Event Models

class ClarificationOption(BaseModel):
    """Option for a clarification question."""

    id: str
    label: str
    value: Optional[str] = None


class ClarificationQuestion(BaseModel):
    """A clarification question with options."""

    id: str
    question: str
    options: List[ClarificationOption]


class SkillBenchmarkEvent(BaseModel):
    """Benchmark data for a single skill in SSE events."""

    skill: str
    elapsed_ms: int
    llm_ms: int
    non_llm_ms: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    status: str
    retry_count: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
