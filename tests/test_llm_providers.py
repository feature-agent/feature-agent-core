"""Tests for LLM provider abstraction layer."""

from __future__ import annotations

import json
from typing import Any
from collections import deque

import pytest

from agent.llm.base import LLMError, LLMProvider, LLMResponse, ParseError


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
