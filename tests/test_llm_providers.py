"""Tests for LLM provider abstraction layer."""

from __future__ import annotations

import json
from typing import Any
from collections import deque

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.llm.base import LLMError, LLMProvider, LLMResponse, ParseError
from agent.llm.anthropic_provider import AnthropicProvider
from agent.llm.bedrock_provider import BedrockProvider
from agent.llm.registry import PROVIDERS, create_provider


# ---------------------------------------------------------------------------
# ConcreteProvider: test subclass with pre-configured responses
# ---------------------------------------------------------------------------

class ConcreteProvider(LLMProvider):
    """Test subclass that returns pre-configured responses."""

    def __init__(self, responses: list[LLMResponse | Exception] | None = None) -> None:
        self._responses: deque[LLMResponse | Exception] = deque(responses or [])

    async def call(
        self,
        system: str,
        user: str,
        use_cache: bool = True,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        if not self._responses:
            raise LLMError("No more pre-configured responses")
        resp = self._responses.popleft()
        if isinstance(resp, Exception):
            raise resp
        return resp


def _make_response(content: str = '{"key": "value"}') -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=20,
        cached_tokens=0,
        elapsed_ms=100,
        model="test-model",
    )


# ---------------------------------------------------------------------------
# Task 1: Base module tests
# ---------------------------------------------------------------------------

class TestLLMResponse:
    """LLMResponse model tests."""

    def test_llm_response_model(self) -> None:
        resp = LLMResponse(
            content="hello",
            input_tokens=10,
            output_tokens=20,
            cached_tokens=5,
            elapsed_ms=100,
            model="test-model",
        )
        assert resp.content == "hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 20
        assert resp.cached_tokens == 5
        assert resp.elapsed_ms == 100
        assert resp.model == "test-model"


class TestStripMarkdownFences:
    """Tests for _strip_markdown_fences static method."""

    def test_strip_markdown_fences(self) -> None:
        assert LLMProvider._strip_markdown_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert LLMProvider._strip_markdown_fences('```\n{"a": 1}\n```') == '{"a": 1}'
        assert LLMProvider._strip_markdown_fences('{"a": 1}') == '{"a": 1}'


class TestParseJson:
    """Tests for parse_json method."""

    @pytest.mark.asyncio
    async def test_parse_json_valid(self) -> None:
        provider = ConcreteProvider()
        result = await provider.parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_parse_json_strips_markdown(self) -> None:
        provider = ConcreteProvider()
        result = await provider.parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_parse_json_retries_with_correction(self) -> None:
        correction_response = _make_response('{"corrected": true}')
        provider = ConcreteProvider(responses=[correction_response])
        result = await provider.parse_json("not valid json")
        assert result == {"corrected": True}

    @pytest.mark.asyncio
    async def test_parse_json_raises_on_second_failure(self) -> None:
        bad_correction = _make_response("still not json {{{")
        provider = ConcreteProvider(responses=[bad_correction])
        with pytest.raises(ParseError, match="Failed to parse JSON after correction"):
            await provider.parse_json("not valid json")


# ---------------------------------------------------------------------------
# Task 2: AnthropicProvider tests
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_init_from_credentials(self) -> None:
        with patch("agent.llm.anthropic_provider.anthropic.Anthropic") as mock_cls:
            provider = AnthropicProvider(credentials={"anthropic_api_key": "sk-test-123"})
            mock_cls.assert_called_once_with(api_key="sk-test-123")

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="anthropic_api_key"):
            AnthropicProvider(credentials={})

    @pytest.mark.asyncio
    async def test_returns_llm_response(self) -> None:
        provider = AnthropicProvider.__new__(AnthropicProvider)
        mock_usage = MagicMock(input_tokens=100, output_tokens=50, cache_read_input_tokens=10)
        mock_content = MagicMock(text="hello world")
        mock_response = MagicMock(
            content=[mock_content],
            usage=mock_usage,
            model="claude-opus-4-5-20250514",
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        result = await provider.call(system="sys", user="usr")
        assert isinstance(result, LLMResponse)
        assert result.content == "hello world"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cached_tokens == 10
        assert result.model == "claude-opus-4-5-20250514"

    @pytest.mark.asyncio
    async def test_deprecated_sampling_params_not_sent(self) -> None:
        """Claude Opus 4.7 deprecated temperature/top_p — they must not be passed."""
        provider = AnthropicProvider.__new__(AnthropicProvider)
        mock_usage = MagicMock(input_tokens=10, output_tokens=5, cache_read_input_tokens=0)
        mock_content = MagicMock(text="ok")
        mock_response = MagicMock(content=[mock_content], usage=mock_usage, model="test")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        await provider.call(system="sys", user="usr")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs

    @pytest.mark.asyncio
    async def test_retry_on_error(self) -> None:
        provider = AnthropicProvider.__new__(AnthropicProvider)
        mock_usage = MagicMock(input_tokens=10, output_tokens=5, cache_read_input_tokens=0)
        mock_content = MagicMock(text="ok")
        mock_response = MagicMock(content=[mock_content], usage=mock_usage, model="test")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [Exception("transient"), mock_response]
        provider._client = mock_client

        with patch("agent.llm.anthropic_provider.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.call(system="sys", user="usr")
        assert result.content == "ok"
        assert mock_client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# Task 3: BedrockProvider tests
# ---------------------------------------------------------------------------

_BEDROCK_CREDS = {
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "aws_region": "us-east-1",
}


def _bedrock_converse_response() -> dict:
    return {
        "output": {
            "message": {
                "content": [{"text": "bedrock reply"}],
            },
        },
        "usage": {
            "inputTokens": 50,
            "outputTokens": 25,
        },
    }


class TestBedrockProvider:
    """Tests for BedrockProvider."""

    def test_init_from_credentials(self) -> None:
        with patch("agent.llm.bedrock_provider.boto3.client") as mock_boto:
            provider = BedrockProvider(credentials=_BEDROCK_CREDS)
            mock_boto.assert_called_once_with(
                "bedrock-runtime",
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                region_name="us-east-1",
            )

    def test_init_with_session_token(self) -> None:
        creds = {**_BEDROCK_CREDS, "aws_session_token": "FwoGZX..."}
        with patch("agent.llm.bedrock_provider.boto3.client") as mock_boto:
            provider = BedrockProvider(credentials=creds)
            call_kwargs = mock_boto.call_args
            assert call_kwargs.kwargs["aws_session_token"] == "FwoGZX..."

    def test_missing_access_key_raises(self) -> None:
        with pytest.raises(ValueError, match="aws_access_key_id"):
            BedrockProvider(credentials={})

    def test_missing_secret_key_raises(self) -> None:
        with pytest.raises(ValueError, match="aws_secret_access_key"):
            BedrockProvider(credentials={"aws_access_key_id": "x"})

    def test_missing_region_raises(self) -> None:
        with pytest.raises(ValueError, match="aws_region"):
            BedrockProvider(credentials={"aws_access_key_id": "x", "aws_secret_access_key": "y"})

    @pytest.mark.asyncio
    async def test_returns_llm_response(self) -> None:
        provider = BedrockProvider.__new__(BedrockProvider)
        mock_client = MagicMock()
        mock_client.converse.return_value = _bedrock_converse_response()
        provider._client = mock_client

        result = await provider.call(system="sys", user="usr")
        assert isinstance(result, LLMResponse)
        assert result.content == "bedrock reply"
        assert result.input_tokens == 50
        assert result.output_tokens == 25
        assert result.cached_tokens == 0
        assert result.model == "anthropic.claude-sonnet-4-5-20250514-v1:0"

    @pytest.mark.asyncio
    async def test_uses_converse_api_with_correct_params(self) -> None:
        provider = BedrockProvider.__new__(BedrockProvider)
        mock_client = MagicMock()
        mock_client.converse.return_value = _bedrock_converse_response()
        provider._client = mock_client

        await provider.call(system="sys", user="usr")
        call_kwargs = mock_client.converse.call_args.kwargs
        assert call_kwargs["modelId"] == "anthropic.claude-sonnet-4-5-20250514-v1:0"
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.0
        assert call_kwargs["inferenceConfig"]["topP"] == 1.0
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 4096

    @pytest.mark.asyncio
    async def test_retries_on_error(self) -> None:
        provider = BedrockProvider.__new__(BedrockProvider)
        mock_client = MagicMock()
        mock_client.converse.side_effect = [Exception("throttled"), _bedrock_converse_response()]
        provider._client = mock_client

        with patch("agent.llm.bedrock_provider.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.call(system="sys", user="usr")
        assert result.content == "bedrock reply"
        assert mock_client.converse.call_count == 2


# ---------------------------------------------------------------------------
# Task 4: Provider registry tests
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    """Tests for provider registry and create_provider factory."""

    def test_registry_contains_anthropic(self) -> None:
        assert "anthropic" in PROVIDERS
        assert PROVIDERS["anthropic"] is AnthropicProvider

    def test_registry_contains_bedrock(self) -> None:
        assert "bedrock" in PROVIDERS
        assert PROVIDERS["bedrock"] is BedrockProvider

    def test_create_provider_returns_anthropic(self) -> None:
        with patch("agent.llm.anthropic_provider.anthropic.Anthropic"):
            provider = create_provider("anthropic", {"anthropic_api_key": "sk-test"})
        assert isinstance(provider, AnthropicProvider)

    def test_create_provider_returns_bedrock(self) -> None:
        with patch("agent.llm.bedrock_provider.boto3.client"):
            provider = create_provider("bedrock", _BEDROCK_CREDS)
        assert isinstance(provider, BedrockProvider)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider type 'openai'"):
            create_provider("openai", {})
