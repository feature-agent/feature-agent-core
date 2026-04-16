"""Shared test fixtures."""

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent.main import app
from agent.queue.nats_client import NATSClient


@pytest.fixture
def mock_nats_client() -> AsyncMock:
    """Mock NATS client that doesn't require a running server."""
    client = AsyncMock(spec=NATSClient)
    client.is_connected = True
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.publish = AsyncMock()
    client.subscribe = AsyncMock()
    client.ensure_stream = AsyncMock()
    return client


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client for testing without Anthropic API."""
    client = MagicMock()
    client.call = AsyncMock()
    client.parse_json = AsyncMock()
    return client


@pytest.fixture
def tmp_data_dir() -> str:
    """Temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


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
        "requirement": None,
        "clarification": None,
        "context": {},
        "result": None,
        "error": None,
    }


@pytest_asyncio.fixture
async def async_client(mock_nats_client: AsyncMock) -> AsyncClient:
    """Async HTTP test client with mocked NATS."""
    import agent.main as main_module

    original_nats = main_module.nats_client
    main_module.nats_client = mock_nats_client

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    main_module.nats_client = original_nats
