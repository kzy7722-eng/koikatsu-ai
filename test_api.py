import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.delenv("APP_API_TOKEN", raising=False)


def test_healthz():
    assert client.get("/healthz").json()["status"] == "ok"


def test_profile_endpoint():
    r = client.post("/api/v1/generate/profile", json={"max_chars": 120, "me": {"occupation": "デザイナー", "hobbies": ["映画"]}})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "profile" and body["id"].startswith("gen_")
    assert len(body["result"]["drafts"]) == 3


def test_reply_endpoint_with_alias_from():
    r = client.post("/api/v1/generate/reply", json={"messages": [
        {"from": "me", "text": "こんにちは"}, {"from": "partner", "text": "こんにちは!今日は暑いですね。"}]})
    assert r.status_code == 200
    assert r.json()["result"]["state"]["distance"] == 1


def test_interest_endpoint_insufficient():
    r = client.post("/api/v1/analyze/interest", json={"messages": [{"from": "partner", "text": "こんにちは"}]})
    assert r.status_code == 200 and r.json()["result"]["status"] == "insufficient"


def test_invalid_input_envelope():
    r = client.post("/api/v1/generate/profile", json={"max_chars": 5, "me": {}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_input"


def test_reply_mode_requires_messages():
    r = client.post("/api/v1/generate/reply", json={"mode": "reply", "messages": []})
    assert r.status_code == 400


def test_auth(monkeypatch):
    monkeypatch.setenv("APP_API_TOKEN", "secret")
    body = {"me": {"hobbies": ["映画"]}}
    assert client.post("/api/v1/generate/profile", json=body).status_code == 401
    ok = client.post("/api/v1/generate/profile", json=body, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_anthropic_backend_without_key_is_503(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/api/v1/generate/profile", json={"me": {"hobbies": ["映画"]}})
    assert r.status_code == 503 and r.json()["error"]["code"] == "ai_unavailable"
