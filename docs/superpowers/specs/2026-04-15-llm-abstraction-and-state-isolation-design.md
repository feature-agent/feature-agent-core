# LLM Provider Abstraction and State Isolation

**Date:** 2026-04-15
**Status:** Approved

## Goals

1. Abstract the LLM module so it can use Anthropic direct API, AWS Bedrock, and be extended for Azure and GitHub Models.
2. Move task ID ownership to the client so the agent core is stateless with respect to identity.
3. Confirm state isolation is sound for concurrent task processing.

## Decision Record

- **Provider selection:** Client passes explicit `provider_type` (not auto-detected).
- **Bedrock credentials:** Client passes `aws_access_key_id`, `aws_secret_access_key`, `aws_region`, and optional `aws_session_token`.
- **Scope:** Build Anthropic + Bedrock adapters now. Azure and GitHub Models are future — the interface makes them a one-file drop-in.
- **Architecture:** Strategy pattern with provider registry.
- **Task ID:** Client-provided, validated for uniqueness (409 on conflict).

---

## 1. LLM Provider Abstraction

### File Structure

```
agent/
  llm/
    __init__.py              # exports create_provider()
    base.py                  # LLMProvider ABC + LLMResponse + shared parse_json
    anthropic_provider.py    # AnthropicProvider
    bedrock_provider.py      # BedrockProvider
    registry.py              # PROVIDERS dict + create_provider() factory
```

The old `agent/llm_client.py` is deleted.

### Base Class (`agent/llm/base.py`)

```python
class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    elapsed_ms: int
    model: str

class LLMProvider(ABC):
    @abstractmethod
    async def call(self, system: str, user: str, use_cache: bool = True) -> LLMResponse:
        ...

    async def parse_json(self, response_text: str, correction_context: str = "") -> dict[str, Any]:
        # Shared implementation. Strips markdown fences, parses JSON.
        # On failure, calls self.call() with correction prompt, parses again.
        # Raises ParseError on second failure.
        ...

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        ...
```

`parse_json` and `_strip_markdown_fences` live in the base class because they are provider-agnostic. Every provider inherits them.

### Exceptions

```python
class LLMError(Exception):
    """LLM call failed after retries."""

class ParseError(Exception):
    """JSON parsing failed after correction attempt."""
```

These move to `base.py` alongside the classes.

### Registry (`agent/llm/registry.py`)

```python
PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "bedrock": BedrockProvider,
}

def create_provider(provider_type: str, credentials: dict) -> LLMProvider:
    if provider_type not in PROVIDERS:
        raise ValueError(f"Unknown provider_type: {provider_type}. Supported: {list(PROVIDERS.keys())}")
    cls = PROVIDERS[provider_type]
    return cls(credentials)
```

Adding a new provider: create the file, import it, add one entry to `PROVIDERS`.

### AnthropicProvider (`agent/llm/anthropic_provider.py`)

Wraps the current `llm_client.py` logic:

- **Credentials:** `{"anthropic_api_key": "sk-ant-..."}`
- **SDK:** `anthropic.Anthropic(api_key=...)`
- **Model:** `claude-opus-4-5-20250514`
- **Settings:** temperature=0.0, top_p=1.0, max_tokens=4096
- **Cache:** Ephemeral prompt caching via `cache_control` on system message when `use_cache=True`
- **Retry:** 1 retry with 2s delay on failure
- **Token mapping:** Direct from Anthropic usage fields; `cached_tokens` from `cache_read_input_tokens`

### BedrockProvider (`agent/llm/bedrock_provider.py`)

- **Credentials:** `{"aws_access_key_id": "...", "aws_secret_access_key": "...", "aws_region": "...", "aws_session_token": "..."}` (session_token optional)
- **SDK:** `boto3.client("bedrock-runtime", ...)`
- **API:** Bedrock `converse` API
- **Model:** `anthropic.claude-opus-4-5-20250514-v1:0` (Bedrock model ID format)
- **Settings:** Same deterministic settings (temperature=0.0, top_p=1.0, max_tokens=4096)
- **Retry:** Same 1 retry with 2s delay
- **Token mapping:** From Bedrock's `usage` response field; `cached_tokens` set to 0 (Bedrock does not expose Anthropic-style prompt cache metrics)
- **New dependency:** `boto3` added to `requirements.txt`

---

## 2. ProviderConfig API Changes

### Updated Model

```python
class ProviderConfig(BaseModel):
    provider_type: str    # "anthropic" or "bedrock"
    github_token: str
    credentials: dict     # opaque, passed directly to provider class
```

`credentials` is an opaque dict — the API layer passes it through. Each provider class validates what it needs at construction time. This means adding a new provider never touches the API model.

### Example Request — Anthropic

```json
{
  "task_id": "client-123",
  "task_description": "Add hello endpoint",
  "target_repo": "owner/repo",
  "provider": {
    "provider_type": "anthropic",
    "github_token": "ghp_xxx",
    "credentials": {
      "anthropic_api_key": "sk-ant-xxx"
    }
  }
}
```

### Example Request — Bedrock

```json
{
  "task_id": "client-456",
  "task_description": "Add hello endpoint",
  "target_repo": "owner/repo",
  "provider": {
    "provider_type": "bedrock",
    "github_token": "ghp_xxx",
    "credentials": {
      "aws_access_key_id": "AKIA...",
      "aws_secret_access_key": "...",
      "aws_region": "us-east-1"
    }
  }
}
```

---

## 3. Client-Provided Task ID

### Changes

- `task_id` becomes a required string field in `TaskCreateRequest`.
- `StateManager.create_task` accepts `task_id` instead of generating one.
- On creation, check if task_id already exists. Return `409 Conflict` if so.
- No format restriction on the ID. Client controls the namespace.

### What This Replaces

```python
# Before (server-generated)
task_id = secrets.token_hex(4)

# After (client-provided)
task_id = request.task_id  # validated for uniqueness
```

---

## 4. State Isolation (No Changes Needed)

The current architecture already isolates tasks correctly:

| Component | Scope | Isolation Mechanism |
|-----------|-------|-------------------|
| Storage paths | Per-task | `tasks/{task_id}/state`, `tasks/{task_id}/events`, `tasks/{task_id}/benchmark` |
| LocalVolumeStorage | Shared | Per-key asyncio locks prevent concurrent write races |
| BenchmarkTracker | Per-task | Fresh instance created in orchestrator per task |
| LLMProvider | Per-task | Fresh instance with per-task credentials |
| EventEmitter._queues | Shared | Keyed by task_id, cleanup in stream endpoint |
| Context dict | Per-task | Local to each orchestrator.run() call |
| StateManager | Shared | Task-keyed reads/writes, no cross-task state |

No architectural changes needed. Moving task_id to client-provided is the only change in this area.

---

## 5. Orchestrator Changes

The orchestrator's `_create_llm` method becomes `_create_provider`:

```python
def _create_provider(self, provider_type: str, credentials: dict) -> LLMProvider:
    return create_provider(provider_type, credentials)
```

Called in `run()` and `resume_after_clarify()` using task's stored provider config.

Skills receive the provider via the same `llm` parameter — they call `llm.call()` and `llm.parse_json()` as before. Zero changes to any skill file.

---

## 6. Test Updates

- **Orchestrator tests:** Update mock to use `LLMProvider` spec instead of `LLMClient`
- **API tests:** Update request payloads with new `ProviderConfig` shape and `task_id`
- **New tests for registry:** `create_provider("anthropic", {...})` returns `AnthropicProvider`, unknown type raises `ValueError`
- **New tests for BedrockProvider:** Mock `boto3.client` and verify `converse` call params and response mapping
- **Existing skill tests:** No changes (they mock the LLM interface, which stays the same)

---

## 7. Files Changed Summary

| File | Action |
|------|--------|
| `agent/llm_client.py` | Delete |
| `agent/llm/__init__.py` | New — exports `create_provider`, `LLMProvider`, `LLMResponse` |
| `agent/llm/base.py` | New — ABC, LLMResponse, parse_json, exceptions |
| `agent/llm/anthropic_provider.py` | New — current llm_client.py logic |
| `agent/llm/bedrock_provider.py` | New — Bedrock via boto3 |
| `agent/llm/registry.py` | New — provider map + factory |
| `agent/api/models.py` | Update ProviderConfig, add task_id to TaskCreateRequest |
| `agent/api/routes/tasks.py` | Pass client task_id, return 409 on conflict |
| `agent/state_manager.py` | Accept task_id param instead of generating |
| `agent/orchestrator.py` | Use `create_provider()` instead of `_create_llm()` |
| `agent/main.py` | Update imports from `agent.llm` |
| `requirements.txt` | Add `boto3` |
| `tests/test_api.py` | Update payloads |
| `tests/test_orchestrator.py` | Update mock spec |
| `tests/test_llm_client.py` | Rename/refactor for new module structure |
