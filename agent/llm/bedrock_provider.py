"""AWS Bedrock provider using boto3 converse API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import boto3

from agent.llm.base import LLMError, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "anthropic.claude-sonnet-4-5-20250514-v1:0"
RETRY_DELAY_SECONDS = 2.0

MODEL_ALIASES: dict[str, str] = {
    "fast": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "default": "anthropic.claude-sonnet-4-5-20250514-v1:0",
    "powerful": "anthropic.claude-sonnet-4-5-20250514-v1:0",
}


class BedrockProvider(LLMProvider):
    """LLM provider using AWS Bedrock converse API."""

    def __init__(self, credentials: dict[str, Any]) -> None:
        for key in ("aws_access_key_id", "aws_secret_access_key", "aws_region"):
            if not credentials.get(key):
                raise ValueError(f"credentials must include '{key}'")

        session_kwargs: dict[str, Any] = {
            "aws_access_key_id": credentials["aws_access_key_id"],
            "aws_secret_access_key": credentials["aws_secret_access_key"],
            "region_name": credentials["aws_region"],
        }
        if credentials.get("aws_session_token"):
            session_kwargs["aws_session_token"] = credentials["aws_session_token"]

        self._client = boto3.client("bedrock-runtime", **session_kwargs)

    def _resolve_model(self, model: str | None) -> str:
        """Resolve a model alias or explicit ID to a Bedrock model ID."""
        if model is None:
            return DEFAULT_MODEL_ID
        return MODEL_ALIASES.get(model, model)

    async def call(
        self,
        system: str,
        user: str,
        use_cache: bool = True,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        """Call Bedrock converse API with temperature=0.0, topP=1.0. Retries once."""
        resolved_model = self._resolve_model(model)
        for attempt in range(2):
            try:
                start = time.monotonic()
                response = await asyncio.to_thread(
                    self._client.converse,
                    modelId=resolved_model,
                    system=[{"text": system}],
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": user}],
                        }
                    ],
                    inferenceConfig={
                        "temperature": 0.0,
                        "topP": 1.0,
                        "maxTokens": max_tokens,
                    },
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)

                content = response["output"]["message"]["content"][0]["text"]
                usage = response["usage"]

                return LLMResponse(
                    content=content,
                    input_tokens=usage["inputTokens"],
                    output_tokens=usage["outputTokens"],
                    cached_tokens=0,
                    elapsed_ms=elapsed_ms,
                    model=resolved_model,
                )
            except Exception as exc:
                if attempt == 0:
                    logger.warning("Bedrock call failed (attempt 1): %s. Retrying...", exc)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    raise LLMError(f"Bedrock call failed after 2 attempts: {exc}") from exc

        raise LLMError("Bedrock call failed unexpectedly")
