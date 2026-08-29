from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    with patch("app.main.voicebox.health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {"status": "ok"}
        response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "voicebox" in data
    assert "tools" in data
    assert "ollama_configured" in data


def test_list_voices_unavailable():
    with patch("app.routers.voices.client.list_profiles", new_callable=AsyncMock) as mock_list:
        mock_list.side_effect = ConnectionError("Voicebox down")
        response = client.get("/api/voices")
    assert response.status_code == 502


def test_refine_without_ollama_key():
    with patch("app.config.settings.ollama_api_key", ""):
        response = client.post("/api/transcribe/refine", json={"text": "hello um world"})
    assert response.status_code == 400
