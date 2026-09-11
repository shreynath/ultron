from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.brain.runtime import (
    CognitiveExecution,
    ExecutionRoute,
    ExecutionStatus,
)
from app.brain.schemas import fallback_assessment
from app.main import app

client = TestClient(app)


def test_brain_status_endpoint():
    response = client.get("/api/brain/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "brain_model" in data
    assert "brain_max_iterations" not in data


def test_brain_run_endpoint():
    assessment = fallback_assessment()
    mock_result = CognitiveExecution(
        answer="Direct reply.",
        route=ExecutionRoute.DIRECT_RESPONSE,
        status=ExecutionStatus.COMPLETED,
        assessment=assessment,
    )
    with (
        patch("app.routers.brain.executive.assess", new_callable=AsyncMock) as mock_assess,
        patch("app.routers.brain.runtime.handle_conversation", new_callable=AsyncMock) as mock_handle,
    ):
        mock_assess.return_value = assessment
        mock_handle.return_value = mock_result
        response = client.post(
            "/api/brain/run",
            json={"message": "Analyze system load", "history": []},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Direct reply."
    assert data["route"] == "direct_response"
    assert data["status"] == "completed"
    assert "assessment" in data
    assert data["assessment"]["user_goal"] == assessment.user_goal
    mock_assess.assert_awaited_once_with("Analyze system load", history=[])
    mock_handle.assert_awaited_once_with(
        user_input="Analyze system load",
        history=[],
        assessment=assessment,
    )
