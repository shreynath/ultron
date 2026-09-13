from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_development_status_is_read_only_and_honest_when_unconfigured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ultron_home", str(tmp_path / "ultron"))
    response = TestClient(app).get("/api/local-intelligence/development-status")
    assert response.status_code == 200
    body = response.json()
    assert body["champion_id"] is None
    assert body["next_action"] == "NO_ACTION"
    assert "No immutable development policy" in body["reason"]
