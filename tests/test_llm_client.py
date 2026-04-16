"""LLM client tests with mocked Anthropic SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.llm_client import LLMClient, LLMError, ParseError


def _make_mock_client(content="test response", input_tokens=100, output_tokens=50):
    """Create a mock Anthropic client."""
    mock = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=content)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    response.usage.cache_read_input_tokens = 0
    response.model = "claude-opus-4-5-20250514"
    mock.messages.create.return_value = response
    return mock


@pytest.mark.asyncio
async def test_llm_client_uses_temperature_zero():
    """LLM client always calls with temperature=0.0 and top_p=1.0."""
    mock = _make_mock_client()
    client = LLMClient(mock)

    await client.call(system="test", user="test")

    call_kwargs = mock.messages.create.call_args
    assert call_kwargs.kwargs["temperature"] == 0.0
    assert call_kwargs.kwargs["top_p"] == 1.0


@pytest.mark.asyncio
async def test_llm_client_retries_on_api_error():
    """LLM client retries once on API error."""
    mock = MagicMock()
    good_response = MagicMock()
    good_response.content = [MagicMock(text="ok")]
    good_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    good_response.usage.cache_read_input_tokens = 0
    good_response.model = "claude-opus-4-5-20250514"

    mock.messages.create.side_effect = [Exception("API error"), good_response]

    client = LLMClient(mock)
    with patch("agent.llm_client.RETRY_DELAY_SECONDS", 0):
        result = await client.call(system="test", user="test")

    assert result.content == "ok"
    assert mock.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_parse_json_strips_markdown_fences():
    """parse_json strips markdown code fences."""
    mock = _make_mock_client()
    client = LLMClient(mock)

    result = await client.parse_json('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_parse_json_retries_on_invalid_json():
    """parse_json retries with correction prompt on invalid JSON."""
    mock = _make_mock_client(content='{"key": "corrected"}')
    client = LLMClient(mock)

    result = await client.parse_json("not json at all")
    assert result == {"key": "corrected"}


@pytest.mark.asyncio
async def test_parse_json_raises_on_second_failure():
    """parse_json raises ParseError when correction also fails."""
    mock = _make_mock_client(content="still not json")
    client = LLMClient(mock)

    with pytest.raises(ParseError):
        await client.parse_json("not json")
