"""Anthropic provider using the Anthropic SDK."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import anthropic

from agent.llm.base import LLMError, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"
RETRY_DELAY_SECONDS = 2.0


class AnthropicProvider(LLMProvider):
    """LLM provider wrapping the Anthropic SDK with deterministic settings."""

    def __init__(self, credentials: dict[str, Any]) -> None:
        api_key = credentials.get("anthropic_api_key")
        if not api_key:
            raise ValueError("credentials must include 'anthropic_api_key'")
        self._client = anthropic.Anthropic(api_key=api_key)

    async def call(
        self,
        system: str,
        user: str,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Call Claude. Retries once on failure."""
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
                    max_tokens=16384,
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
