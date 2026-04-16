# LLM Provider Abstraction and State Isolation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Abstract the LLM module into a provider pattern (Anthropic + Bedrock), move task ID ownership to the client, and update all imports.

**Architecture:** Strategy pattern with `LLMProvider` ABC in `agent/llm/base.py`. Concrete providers in separate files. Registry factory in `agent/llm/registry.py`. Client sends `provider_type` + `credentials` dict per task.

**Tech Stack:** Python 3.10+, FastAPI, anthropic SDK, boto3, pydantic, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `agent/llm/__init__.py` | Create | Public exports: `create_provider`, `LLMProvider`, `LLMResponse`, `LLMError`, `ParseError` |
| `agent/llm/base.py` | Create | `LLMProvider` ABC, `LLMResponse`, `LLMError`, `ParseError`, shared `parse_json` + `_strip_markdown_fences` |
| `agent/llm/anthropic_provider.py` | Create | `AnthropicProvider` — current `llm_client.py` logic |
| `agent/llm/bedrock_provider.py` | Create | `BedrockProvider` — AWS Bedrock via boto3 `converse` API |
| `agent/llm/registry.py` | Create | `PROVIDERS` dict + `create_provider()` factory |
| `agent/llm_client.py` | Delete | Replaced by `agent/llm/` package |
| `agent/api/models.py` | Modify | Update `ProviderConfig`, add `task_id` to `TaskCreateRequest` |
| `agent/api/routes/tasks.py` | Modify | Client-provided `task_id`, 409 on conflict |
| `agent/state_manager.py` | Modify | Accept `task_id` param, remove `secrets.token_hex` |
| `agent/orchestrator.py` | Modify | Use `create_provider()`, update imports |
| `agent/main.py` | Modify | Remove `anthropic` import (already gone), no LLM import needed |
| `agent/skill_base.py` | Modify | Update `LLMClient` import → `LLMProvider` |
| `agent/benchmark.py` | Modify | Update `LLMResponse` import path |
| `requirements.txt` | Modify | Add `boto3` |
| `tests/test_llm_providers.py` | Create | Tests for base, anthropic, bedrock, registry |
| `tests/test_api.py` | Modify | Update payloads with new `ProviderConfig` shape + `task_id` |
| `tests/test_orchestrator.py` | Modify | Update mock spec from `LLMClient` → `LLMProvider` |
| `tests/test_llm_client.py` | Delete | Replaced by `tests/test_llm_providers.py` |
| `tests/conftest.py` | Modify | Update `mock_llm_client` fixture |
| `tests/test_state_manager.py` | Modify | Update `create_task` calls with `task_id` param |

---

### Task 1: Create LLM base module with ABC, LLMResponse, and shared logic

**Files:**
- Create: `agent/llm/__init__.py`
- Create: `agent/llm/base.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write failing tests for base module**

Create `tests/test_llm_providers.py`:

```python
"""LLM provider tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.llm.base import LLMProvider, LLMResponse, LLMError, ParseError


class ConcreteProvider(LLMProvider):
    """Minimal concrete provider for testing base class logic."""

    def __init__(self):
        self.call_responses = []
        self._call_index = 0

    async def call(self, system: str, user: str, use_cache: bool = True) -> LLMResponse:
        if self._call_index < len(self.call_responses):
            resp = self.call_responses[self._call_index]
            self._call_index += 1
            if isinstance(resp, Exception):
                raise resp
            return resp
        raise LLMError("No more responses configured")


@pytest.mark.asyncio
async def test_parse_json_valid():
    """parse_json parses valid JSON directly."""
    provider = ConcreteProvider()
    result = await provider.parse_json('{"key": "value"}')
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_parse_json_strips_markdown_fences():
    """parse_json strips markdown code fences before parsing."""
    provider = ConcreteProvider()
    result = await provider.parse_json('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_parse_json_retries_with_correction():
    """parse_json retries with correction prompt on invalid JSON."""
    provider = ConcreteProvider()
    provider.call_responses = [
        LLMResponse(
            content='{"key": "corrected"}',
            input_tokens=10, output_tokens=5, cached_tokens=0,
            elapsed_ms=100, model="test",
        )
    ]
    result = await provider.parse_json("not json at all")
    assert result == {"key": "corrected"}


@pytest.mark.asyncio
async def test_parse_json_raises_on_second_failure():
    """parse_json raises ParseError when correction also fails."""
    provider = ConcreteProvider()
    provider.call_responses = [
        LLMResponse(
            content="still not json",
            input_tokens=10, output_tokens=5, cached_tokens=0,
            elapsed_ms=100, model="test",
        )
    ]
    with pytest.raises(ParseError):
        await provider.parse_json("not json")


def test_strip_markdown_fences():
    """_strip_markdown_fences removes code fences."""
    assert LLMProvider._strip_markdown_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert LLMProvider._strip_markdown_fences('```\nhello\n```') == "hello"
    assert LLMProvider._strip_markdown_fences("plain text") == "plain text"


def test_llm_response_model():
    """LLMResponse is a valid pydantic model."""
    resp = LLMResponse(
        content="test", input_tokens=10, output_tokens=5,
        cached_tokens=0, elapsed_ms=100, model="test-model",
    )
    assert resp.content == "test"
    assert resp.model == "test-model"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_llm_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.llm'`

- [ ] **Step 3: Create `agent/llm/base.py`**

```python
"""LLM provider base class, response model, and shared logic."""

from __future__ import annotations

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
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def call(
        self,
        system: str,
        user: str,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Call the LLM with deterministic settings. Retries once on failure."""
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
```

- [ ] **Step 4: Create `agent/llm/__init__.py`**

```python
"""LLM provider package."""

from agent.llm.base import LLMError, LLMProvider, LLMResponse, ParseError

__all__ = ["LLMProvider", "LLMResponse", "LLMError", "ParseError"]
```

Note: `create_provider` will be added here in Task 4 after registry is built.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm_providers.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add agent/llm/__init__.py agent/llm/base.py tests/test_llm_providers.py
git commit -m "feat: add LLM provider base class with ABC, LLMResponse, and shared parse_json"
```

---

### Task 2: Create AnthropicProvider

**Files:**
- Create: `agent/llm/anthropic_provider.py`
- Modify: `tests/test_llm_providers.py`

- [ ] **Step 1: Write failing tests for AnthropicProvider**

Append to `tests/test_llm_providers.py`:

```python
from agent.llm.anthropic_provider import AnthropicProvider


def _make_mock_anthropic_client(content="test response", input_tokens=100, output_tokens=50):
    """Create a mock Anthropic SDK client."""
    mock = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=content)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    response.usage.cache_read_input_tokens = 0
    response.model = "claude-opus-4-5-20250514"
    mock.messages.create.return_value = response
    return mock


@pytest.mark.asyncio
async def test_anthropic_provider_uses_temperature_zero():
    """AnthropicProvider calls with temperature=0.0 and top_p=1.0."""
    mock = _make_mock_anthropic_client()
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._client = mock

    await provider.call(system="test", user="test")

    call_kwargs = mock.messages.create.call_args
    assert call_kwargs.kwargs["temperature"] == 0.0
    assert call_kwargs.kwargs["top_p"] == 1.0


@pytest.mark.asyncio
async def test_anthropic_provider_retries_on_error():
    """AnthropicProvider retries once on API error."""
    mock = MagicMock()
    good_response = MagicMock()
    good_response.content = [MagicMock(text="ok")]
    good_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    good_response.usage.cache_read_input_tokens = 0
    good_response.model = "claude-opus-4-5-20250514"
    mock.messages.create.side_effect = [Exception("API error"), good_response]

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._client = mock

    from unittest.mock import patch as mock_patch
    with mock_patch("agent.llm.anthropic_provider.RETRY_DELAY_SECONDS", 0):
        result = await provider.call(system="test", user="test")

    assert result.content == "ok"
    assert mock.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_anthropic_provider_returns_llm_response():
    """AnthropicProvider returns a proper LLMResponse."""
    mock = _make_mock_anthropic_client(content="hello", input_tokens=50, output_tokens=25)
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._client = mock

    result = await provider.call(system="sys", user="usr")

    assert isinstance(result, LLMResponse)
    assert result.content == "hello"
    assert result.input_tokens == 50
    assert result.output_tokens == 25


def test_anthropic_provider_init_from_credentials():
    """AnthropicProvider constructs from credentials dict."""
    from unittest.mock import patch as mock_patch
    with mock_patch("agent.llm.anthropic_provider.anthropic") as mock_anthropic:
        provider = AnthropicProvider({"anthropic_api_key": "sk-test-123"})
        mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-test-123")


def test_anthropic_provider_missing_key_raises():
    """AnthropicProvider raises ValueError if api key missing."""
    with pytest.raises(ValueError, match="anthropic_api_key"):
        AnthropicProvider({})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_llm_providers.py::test_anthropic_provider_init_from_credentials -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.llm.anthropic_provider'`

- [ ] **Step 3: Create `agent/llm/anthropic_provider.py`**

```python
"""Anthropic direct API provider."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import anthropic

from agent.llm.base import LLMError, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-5-20250514"
RETRY_DELAY_SECONDS = 2.0


class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic's direct API."""

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
        """Call Claude via Anthropic API with temperature=0.0, top_p=1.0."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm_providers.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add agent/llm/anthropic_provider.py tests/test_llm_providers.py
git commit -m "feat: add AnthropicProvider wrapping existing LLM client logic"
```

---

### Task 3: Create BedrockProvider

**Files:**
- Create: `agent/llm/bedrock_provider.py`
- Modify: `tests/test_llm_providers.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add boto3 to requirements.txt**

Add `boto3>=1.34.0` to `requirements.txt` after the `anthropic` line.

- [ ] **Step 2: Write failing tests for BedrockProvider**

Append to `tests/test_llm_providers.py`:

```python
from agent.llm.bedrock_provider import BedrockProvider


def _make_mock_bedrock_client(content="test response", input_tokens=100, output_tokens=50):
    """Create a mock boto3 bedrock-runtime client."""
    mock = MagicMock()
    mock.converse.return_value = {
        "output": {"message": {"content": [{"text": content}]}},
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
        "metrics": {"latencyMs": 200},
    }
    return mock


@pytest.mark.asyncio
async def test_bedrock_provider_returns_llm_response():
    """BedrockProvider returns a proper LLMResponse from converse API."""
    mock = _make_mock_bedrock_client(content="hello", input_tokens=50, output_tokens=25)
    provider = BedrockProvider.__new__(BedrockProvider)
    provider._client = mock
    provider._model_id = "anthropic.claude-opus-4-5-20250514-v1:0"

    result = await provider.call(system="sys", user="usr")

    assert isinstance(result, LLMResponse)
    assert result.content == "hello"
    assert result.input_tokens == 50
    assert result.output_tokens == 25
    assert result.cached_tokens == 0


@pytest.mark.asyncio
async def test_bedrock_provider_uses_converse_api():
    """BedrockProvider calls boto3 converse with correct params."""
    mock = _make_mock_bedrock_client()
    provider = BedrockProvider.__new__(BedrockProvider)
    provider._client = mock
    provider._model_id = "anthropic.claude-opus-4-5-20250514-v1:0"

    await provider.call(system="test system", user="test user")

    call_kwargs = mock.converse.call_args.kwargs
    assert call_kwargs["modelId"] == "anthropic.claude-opus-4-5-20250514-v1:0"
    assert call_kwargs["inferenceConfig"]["temperature"] == 0.0
    assert call_kwargs["inferenceConfig"]["topP"] == 1.0
    assert call_kwargs["system"] == [{"text": "test system"}]
    assert call_kwargs["messages"] == [{"role": "user", "content": [{"text": "test user"}]}]


@pytest.mark.asyncio
async def test_bedrock_provider_retries_on_error():
    """BedrockProvider retries once on API error."""
    mock = MagicMock()
    good_response = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "metrics": {"latencyMs": 100},
    }
    mock.converse.side_effect = [Exception("Throttled"), good_response]

    provider = BedrockProvider.__new__(BedrockProvider)
    provider._client = mock
    provider._model_id = "anthropic.claude-opus-4-5-20250514-v1:0"

    from unittest.mock import patch as mock_patch
    with mock_patch("agent.llm.bedrock_provider.RETRY_DELAY_SECONDS", 0):
        result = await provider.call(system="test", user="test")

    assert result.content == "ok"
    assert mock.converse.call_count == 2


def test_bedrock_provider_init_from_credentials():
    """BedrockProvider constructs from credentials dict."""
    from unittest.mock import patch as mock_patch
    with mock_patch("agent.llm.bedrock_provider.boto3") as mock_boto3:
        provider = BedrockProvider({
            "aws_access_key_id": "AKIA_TEST",
            "aws_secret_access_key": "secret",
            "aws_region": "us-east-1",
        })
        mock_boto3.client.assert_called_once_with(
            "bedrock-runtime",
            aws_access_key_id="AKIA_TEST",
            aws_secret_access_key="secret",
            aws_session_token=None,
            region_name="us-east-1",
        )


def test_bedrock_provider_init_with_session_token():
    """BedrockProvider passes session token when provided."""
    from unittest.mock import patch as mock_patch
    with mock_patch("agent.llm.bedrock_provider.boto3") as mock_boto3:
        provider = BedrockProvider({
            "aws_access_key_id": "AKIA_TEST",
            "aws_secret_access_key": "secret",
            "aws_region": "us-west-2",
            "aws_session_token": "sess-tok",
        })
        mock_boto3.client.assert_called_once_with(
            "bedrock-runtime",
            aws_access_key_id="AKIA_TEST",
            aws_secret_access_key="secret",
            aws_session_token="sess-tok",
            region_name="us-west-2",
        )


def test_bedrock_provider_missing_credentials_raises():
    """BedrockProvider raises ValueError if required credentials missing."""
    with pytest.raises(ValueError, match="aws_access_key_id"):
        BedrockProvider({})

    with pytest.raises(ValueError, match="aws_secret_access_key"):
        BedrockProvider({"aws_access_key_id": "AKIA"})

    with pytest.raises(ValueError, match="aws_region"):
        BedrockProvider({"aws_access_key_id": "AKIA", "aws_secret_access_key": "s"})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_llm_providers.py::test_bedrock_provider_init_from_credentials -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.llm.bedrock_provider'`

- [ ] **Step 4: Create `agent/llm/bedrock_provider.py`**

```python
"""AWS Bedrock provider using the converse API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import boto3

from agent.llm.base import LLMError, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

MODEL_ID = "anthropic.claude-opus-4-5-20250514-v1:0"
RETRY_DELAY_SECONDS = 2.0


class BedrockProvider(LLMProvider):
    """LLM provider using AWS Bedrock converse API."""

    def __init__(self, credentials: dict[str, Any]) -> None:
        for key in ("aws_access_key_id", "aws_secret_access_key", "aws_region"):
            if not credentials.get(key):
                raise ValueError(f"credentials must include '{key}'")

        self._client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"],
            aws_session_token=credentials.get("aws_session_token"),
            region_name=credentials["aws_region"],
        )
        self._model_id = MODEL_ID

    async def call(
        self,
        system: str,
        user: str,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Call Claude via Bedrock converse API with temperature=0.0, top_p=1.0."""
        for attempt in range(2):
            try:
                start = time.monotonic()
                response = await asyncio.to_thread(
                    self._client.converse,
                    modelId=self._model_id,
                    system=[{"text": system}],
                    messages=[{
                        "role": "user",
                        "content": [{"text": user}],
                    }],
                    inferenceConfig={
                        "temperature": 0.0,
                        "topP": 1.0,
                        "maxTokens": 4096,
                    },
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)

                output = response["output"]["message"]["content"][0]["text"]
                usage = response.get("usage", {})

                return LLMResponse(
                    content=output,
                    input_tokens=usage.get("inputTokens", 0),
                    output_tokens=usage.get("outputTokens", 0),
                    cached_tokens=0,
                    elapsed_ms=elapsed_ms,
                    model=self._model_id,
                )
            except Exception as exc:
                if attempt == 0:
                    logger.warning("Bedrock call failed (attempt 1): %s. Retrying...", exc)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    raise LLMError(f"Bedrock call failed after 2 attempts: {exc}") from exc

        raise LLMError("Bedrock call failed unexpectedly")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm_providers.py -v`
Expected: 19 passed

- [ ] **Step 6: Commit**

```bash
git add agent/llm/bedrock_provider.py tests/test_llm_providers.py requirements.txt
git commit -m "feat: add BedrockProvider using boto3 converse API"
```

---

### Task 4: Create provider registry

**Files:**
- Create: `agent/llm/registry.py`
- Modify: `agent/llm/__init__.py`
- Modify: `tests/test_llm_providers.py`

- [ ] **Step 1: Write failing tests for registry**

Append to `tests/test_llm_providers.py`:

```python
from agent.llm.registry import create_provider, PROVIDERS


def test_registry_contains_anthropic_and_bedrock():
    """Registry has entries for both supported providers."""
    assert "anthropic" in PROVIDERS
    assert "bedrock" in PROVIDERS


def test_create_provider_anthropic():
    """create_provider returns AnthropicProvider for 'anthropic'."""
    from unittest.mock import patch as mock_patch
    with mock_patch("agent.llm.anthropic_provider.anthropic"):
        provider = create_provider("anthropic", {"anthropic_api_key": "test"})
        assert isinstance(provider, AnthropicProvider)


def test_create_provider_bedrock():
    """create_provider returns BedrockProvider for 'bedrock'."""
    from unittest.mock import patch as mock_patch
    with mock_patch("agent.llm.bedrock_provider.boto3"):
        provider = create_provider("bedrock", {
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "secret",
            "aws_region": "us-east-1",
        })
        assert isinstance(provider, BedrockProvider)


def test_create_provider_unknown_raises():
    """create_provider raises ValueError for unknown provider type."""
    with pytest.raises(ValueError, match="Unknown provider_type"):
        create_provider("openai", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_llm_providers.py::test_registry_contains_anthropic_and_bedrock -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.llm.registry'`

- [ ] **Step 3: Create `agent/llm/registry.py`**

```python
"""Provider registry and factory function."""

from __future__ import annotations

from typing import Any

from agent.llm.anthropic_provider import AnthropicProvider
from agent.llm.base import LLMProvider
from agent.llm.bedrock_provider import BedrockProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "bedrock": BedrockProvider,
}


def create_provider(provider_type: str, credentials: dict[str, Any]) -> LLMProvider:
    """Create an LLM provider by type name."""
    if provider_type not in PROVIDERS:
        raise ValueError(
            f"Unknown provider_type: {provider_type}. "
            f"Supported: {list(PROVIDERS.keys())}"
        )
    cls = PROVIDERS[provider_type]
    return cls(credentials)
```

- [ ] **Step 4: Update `agent/llm/__init__.py` to export `create_provider`**

```python
"""LLM provider package."""

from agent.llm.base import LLMError, LLMProvider, LLMResponse, ParseError
from agent.llm.registry import create_provider

__all__ = ["LLMProvider", "LLMResponse", "LLMError", "ParseError", "create_provider"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm_providers.py -v`
Expected: 23 passed

- [ ] **Step 6: Commit**

```bash
git add agent/llm/registry.py agent/llm/__init__.py tests/test_llm_providers.py
git commit -m "feat: add provider registry with create_provider factory"
```

---

### Task 5: Update ProviderConfig API model and client-provided task_id

**Files:**
- Modify: `agent/api/models.py`
- Modify: `agent/api/routes/tasks.py`
- Modify: `agent/state_manager.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_state_manager.py`
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Update `agent/api/models.py`**

Replace the `ProviderConfig` class (lines 10-14):

```python
class ProviderConfig(BaseModel):
    """Provider credentials passed per-task from the client."""

    provider_type: str
    github_token: str
    credentials: dict[str, Any] = {}
```

Add `task_id` to `TaskCreateRequest` (lines 17-29):

```python
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
```

- [ ] **Step 2: Update `agent/state_manager.py`**

Change `create_task` (lines 41-70) to accept `task_id` parameter and check for uniqueness:

```python
    async def create_task(
        self,
        task_id: str,
        source: str,
        github_issue_url: Optional[str],
        task_description: Optional[str],
        target_repo: str,
        provider: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a new task with PENDING status. Raises ValueError if task_id exists."""
        existing = await self.get_task(task_id)
        if existing is not None:
            raise ValueError(f"Task {task_id} already exists")
        now = self._now_iso()
        task = {
            "task_id": task_id,
            "status": TaskState.PENDING.value,
            "created_at": now,
            "updated_at": now,
            "source": source,
            "github_issue_url": github_issue_url,
            "task_description": task_description,
            "target_repo": target_repo,
            "provider": provider or {},
            "requirement": None,
            "clarification": None,
            "context": {},
            "result": None,
            "error": None,
        }
        await self._storage.write(self._task_key(task_id), task)
        logger.info("Created task %s", task_id)
        return task
```

Remove `import secrets` from line 5 (no longer needed).

- [ ] **Step 3: Update `agent/api/routes/tasks.py`**

Update `create_task` (lines 28-54) to use client `task_id` and return 409:

```python
@router.post("", response_model=TaskCreateResponse, status_code=201)
async def create_task(request: TaskCreateRequest) -> TaskCreateResponse:
    """Create a new task and publish to NATS queue."""
    state_manager, nats_client, settings = _get_deps()

    source = "github_issue" if request.github_issue_url else "free_text"
    try:
        task = await state_manager.create_task(
            task_id=request.task_id,
            source=source,
            github_issue_url=request.github_issue_url,
            task_description=request.task_description,
            target_repo=request.target_repo,
            provider=request.provider.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    try:
        await nats_client.publish(
            settings.NATS_TASK_SUBJECT,
            {"task_id": task["task_id"], "action": "new"},
        )
    except ConnectionError:
        logger.warning("NATS not available — task %s queued locally only", task["task_id"])

    return TaskCreateResponse(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
    )
```

- [ ] **Step 4: Update `tests/conftest.py`**

Update the `sample_task` fixture (lines 44-60) to include `provider`:

```python
@pytest.fixture
def sample_task() -> dict:
    """Sample task data for testing."""
    return {
        "task_id": "abcd1234",
        "status": "PENDING",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "source": "free_text",
        "github_issue_url": None,
        "task_description": "Add a due_date field to tasks",
        "target_repo": "testuser/testrepo",
        "provider": {"provider_type": "anthropic", "github_token": "test", "credentials": {"anthropic_api_key": "test"}},
        "requirement": None,
        "clarification": None,
        "context": {},
        "result": None,
        "error": None,
    }
```

- [ ] **Step 5: Update all test payloads in `tests/test_api.py`**

Every `POST /api/tasks` call needs `task_id` and the new `provider` shape. Replace all occurrences of:

```python
"provider": {"github_token": "test-token", "anthropic_api_key": "test-key"},
```

with:

```python
"provider": {"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}},
```

And add unique `task_id` to each POST call. For example, `test_post_tasks_with_github_url`:

```python
response = await async_client.post("/api/tasks", json={
    "task_id": "test-gh-url-1",
    "github_issue_url": "https://github.com/owner/repo/issues/1",
    "target_repo": "owner/repo",
    "provider": {"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}},
})
```

Each test needs a unique `task_id`. Use descriptive IDs: `"test-gh-url-1"`, `"test-desc-1"`, `"test-list-1"`, `"test-list-2"`, `"test-get-1"`, `"test-stream-1"`, `"test-replay-1"`, `"test-clarify-1"`, `"test-clarify-status-1"`, `"test-clarify-count-1"`, `"test-clarify-empty-1"`, `"test-clarify-ids-1"`, `"test-bench-1"`, `"test-bench-done-1"`.

Also add a new test for 409 conflict:

```python
@pytest.mark.asyncio
async def test_post_tasks_duplicate_id_returns_409(async_client):
    """POST /api/tasks with duplicate task_id returns 409."""
    payload = {
        "task_id": "dup-test-1",
        "task_description": "First task",
        "target_repo": "owner/repo",
        "provider": {"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}},
    }
    resp1 = await async_client.post("/api/tasks", json=payload)
    assert resp1.status_code == 201

    resp2 = await async_client.post("/api/tasks", json=payload)
    assert resp2.status_code == 409
```

The validation tests (`test_post_tasks_requires_one_of_url_or_description`, `test_post_tasks_requires_target_repo`) need `task_id` and `provider` too — but they'll still 422 because of the missing field being tested.

- [ ] **Step 6: Update `tests/test_state_manager.py`**

Every `create_task` call needs `task_id` as first arg. For example:

```python
# Before:
task = await state_manager.create_task("free_text", None, "desc", "owner/repo")

# After:
task = await state_manager.create_task("test-id-1", "free_text", None, "desc", "owner/repo")
```

Update each test to use unique task IDs. Add a test for duplicate detection:

```python
@pytest.mark.asyncio
async def test_create_task_rejects_duplicate_id(tmp_data_dir):
    """create_task raises ValueError if task_id already exists."""
    storage = LocalVolumeStorage(tmp_data_dir)
    sm = StateManager(storage)
    await sm.create_task("dup-1", "free_text", None, "desc", "owner/repo")
    with pytest.raises(ValueError, match="already exists"):
        await sm.create_task("dup-1", "free_text", None, "other", "owner/repo")
```

- [ ] **Step 7: Update `tests/test_orchestrator.py`**

Update all `create_task` calls to include `task_id` and the new provider shape:

```python
# Before:
task = await state.create_task("free_text", None, "Test", "owner/repo", provider={"anthropic_api_key": "test-key", "github_token": "test-token"})

# After:
task = await state.create_task("test-orch-1", "free_text", None, "Test", "owner/repo", provider={"provider_type": "anthropic", "github_token": "test-token", "credentials": {"anthropic_api_key": "test-key"}})
```

Use unique IDs per test: `"test-orch-1"` through `"test-orch-8"`.

- [ ] **Step 8: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: All pass (existing + new 409 test + new duplicate state test)

- [ ] **Step 9: Commit**

```bash
git add agent/api/models.py agent/api/routes/tasks.py agent/state_manager.py tests/test_api.py tests/test_state_manager.py tests/test_orchestrator.py tests/conftest.py
git commit -m "feat: client-provided task_id, updated ProviderConfig with provider_type and credentials"
```

---

### Task 6: Wire orchestrator and imports to new LLM package

**Files:**
- Modify: `agent/orchestrator.py`
- Modify: `agent/skill_base.py`
- Modify: `agent/benchmark.py`
- Modify: `agent/main.py`
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/conftest.py`
- Delete: `agent/llm_client.py`
- Delete: `tests/test_llm_client.py`

- [ ] **Step 1: Update `agent/orchestrator.py`**

Replace imports (lines 9-13):

```python
from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm import LLMProvider, create_provider
from agent.skill_base import Skill, SkillError
from agent.state_manager import StateManager, TaskState
from agent.storage.interface import StorageInterface
```

Remove the `import anthropic` line (line 9).

Replace `_create_llm` method (lines 38-41):

```python
    def _create_provider(self, provider_type: str, credentials: dict) -> LLMProvider:
        """Create an LLM provider from the per-task provider config."""
        return create_provider(provider_type, credentials)
```

In `run()` (lines 50-56), replace the provider credential extraction:

```python
        provider = task.get("provider", {})
        provider_type = provider.get("provider_type", "")
        credentials = provider.get("credentials", {})
        if not provider_type or not credentials:
            await self._handle_failure(task_id, BenchmarkTracker(), "Missing provider_type or credentials")
            return

        llm = self._create_provider(provider_type, credentials)
```

In `resume_after_clarify()` (lines 118-124), same change:

```python
        provider = task.get("provider", {})
        provider_type = provider.get("provider_type", "")
        credentials = provider.get("credentials", {})
        if not provider_type or not credentials:
            await self._handle_failure(task_id, BenchmarkTracker(), "Missing provider_type or credentials")
            return

        llm = self._create_provider(provider_type, credentials)
```

Update the type hint in `_run_remaining_pipeline` (line 155) and `_run_skill` (line 257):

```python
    # Change LLMClient → LLMProvider in both method signatures
    llm: LLMProvider,
```

- [ ] **Step 2: Update `agent/skill_base.py`**

Replace import (line 10):

```python
from agent.llm import LLMProvider
```

Update the `execute` signature (line 33):

```python
        llm: LLMProvider,
```

- [ ] **Step 3: Update `agent/benchmark.py`**

Replace import (line 12):

```python
from agent.llm import LLMResponse
```

- [ ] **Step 4: Update `agent/main.py`**

Remove the now-unused import (line 15 — `from agent.orchestrator import Orchestrator` is fine, but there's no longer any `anthropic` or `llm_client` import to worry about). Verify no stale imports remain.

- [ ] **Step 5: Update `tests/test_orchestrator.py`**

Replace imports (lines 9-11):

```python
from agent.benchmark import BenchmarkTracker
from agent.event_emitter import EventEmitter
from agent.llm import LLMProvider
from agent.orchestrator import Orchestrator
from agent.skill_base import Skill, SkillError
from agent.state_manager import StateManager, TaskState
from agent.storage.local_volume import LocalVolumeStorage
```

Update the `mock_create_llm` fixture — rename to `mock_create_provider` and update the patch target:

```python
@pytest.fixture
def mock_create_provider():
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.call = AsyncMock()
    mock_llm.parse_json = AsyncMock()
    with patch.object(Orchestrator, "_create_provider", return_value=mock_llm):
        yield mock_llm
```

Update all test function signatures from `mock_create_llm` → `mock_create_provider`.

- [ ] **Step 6: Update `tests/conftest.py`**

Update `mock_llm_client` fixture (lines 28-33):

```python
@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM provider for testing without real API."""
    from agent.llm import LLMProvider
    client = MagicMock(spec=LLMProvider)
    client.call = AsyncMock()
    client.parse_json = AsyncMock()
    return client
```

- [ ] **Step 7: Delete old files**

```bash
rm agent/llm_client.py tests/test_llm_client.py
```

- [ ] **Step 8: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: All pass. The `test_llm_client.py` tests are gone, replaced by `test_llm_providers.py`.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: wire orchestrator and all imports to new LLM provider package, delete old llm_client"
```

---

### Task 7: Docker build verification and final cleanup

**Files:**
- Verify: `docker-compose.yml`
- Verify: `Dockerfile`

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Build Docker image**

Run: `docker compose build`
Expected: Build succeeds with no import errors.

- [ ] **Step 3: Verify no stale references**

Run: `grep -r "llm_client" agent/ tests/ --include="*.py"`
Expected: No matches (all references migrated to `agent.llm`).

Run: `grep -r "from agent.config import settings" agent/skills/ --include="*.py"`
Expected: No matches in skills (credentials come from context, not settings).

- [ ] **Step 4: Commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: final cleanup and Docker build verification"
```
