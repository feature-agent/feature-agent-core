"""LLM provider base class with ABC, shared response model, and parse utilities."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Response from an LLM call with token usage."""

    content: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    elapsed_ms: int
    model: str


class LLMError(Exception):
    """Raised when LLM call fails after retries."""
    pass


class ParseError(Exception):
    """Raised when JSON parsing fails after correction attempt."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def call(
        self,
        system: str,
        user: str,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Call the LLM. Must be implemented by each provider."""
        ...

    async def parse_json(
        self,
        response_text: str,
        correction_context: str = "",
    ) -> dict[str, Any]:
        """Parse JSON from response, retrying with correction prompt on failure."""
        cleaned = self._strip_markdown_fences(response_text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Retry with correction prompt
        logger.warning("JSON parse failed, attempting correction")
        try:
            correction = await self.call(
                system="You are a JSON correction assistant. Return ONLY valid JSON with no explanation, no markdown, no code fences.",
                user=f"Your previous response was not valid JSON. Return ONLY the JSON object with no explanation, no markdown, no code fences.\n\nPrevious response:\n{response_text}\n\n{correction_context}",
                use_cache=False,
            )
            cleaned = self._strip_markdown_fences(correction.content)
            return json.loads(cleaned)
        except (json.JSONDecodeError, LLMError) as exc:
            raise ParseError(f"Failed to parse JSON after correction: {exc}") from exc

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences from text."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
