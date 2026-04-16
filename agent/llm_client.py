"""Anthropic SDK wrapper with deterministic settings and retry logic."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-5-20250514"
RETRY_DELAY_SECONDS = 2.0


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


class LLMClient:
    """Wrapper around Anthropic SDK with deterministic settings."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def call(
        self,
        system: str,
        user: str,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Call Claude with temperature=0.0, top_p=1.0. Retries once on failure."""
        if use_cache:
            system_messages = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_messages = [{"type": "text", "text": system}]

        for attempt in range(2):
            try:
                start = time.monotonic()
                response = await asyncio.to_thread(
                    self._client.messages.create,
                    model=MODEL,
                    max_tokens=4096,
                    temperature=0.0,
                    top_p=1.0,
                    system=system_messages,
                    messages=[{"role": "user", "content": user}],
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)

                usage = response.usage
                cached = getattr(usage, "cache_read_input_tokens", 0) or 0

                return LLMResponse(
                    content=response.content[0].text,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_tokens=cached,
                    elapsed_ms=elapsed_ms,
                    model=response.model,
                )
            except Exception as exc:
                if attempt == 0:
                    logger.warning("LLM call failed (attempt 1): %s. Retrying...", exc)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    raise LLMError(f"LLM call failed after 2 attempts: {exc}") from exc

        raise LLMError("LLM call failed unexpectedly")

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
