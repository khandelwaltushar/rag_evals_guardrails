"""OpenAPI / Swagger UI surface."""

from fastapi.testclient import TestClient

from api.main import app


def test_openapi_json_lists_paths() -> None:
    with TestClient(app) as client:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        body = r.json()
        assert "/health" in body["paths"]
        assert "/ingest" in body["paths"]
        assert "/query" in body["paths"]
        tags = {t["name"] for t in body.get("tags", [])}
        assert "health" in tags
        assert "ingestion" in tags
        assert "query" in tags


def test_swagger_ui_available() -> None:
    with TestClient(app) as client:
        r = client.get("/docs")
        assert r.status_code == 200
        assert "swagger" in r.text.lower() or "openapi" in r.text.lower()


def test_redoc_available() -> None:
    with TestClient(app) as client:
        r = client.get("/redoc")
        assert r.status_code == 200


def test_root_links_to_docs() -> None:
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "/docs" in r.text
