"""Tests for AI Proxy."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_proxy.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


def test_health(client: TestClient) -> None:
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("ai_proxy.llm_client.LLMClient.classify_query")
@patch("ai_proxy.llm_client.LLMClient.chat_completion")
def test_chat_completions(mock_chat: AsyncMock, mock_classify: AsyncMock, client: TestClient) -> None:
    """Test chat completions endpoint."""
    mock_classify.return_value = "local"
    mock_chat.return_value = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello!"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
