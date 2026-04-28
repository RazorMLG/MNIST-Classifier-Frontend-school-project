from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_reports_ready_storage(tmp_path: Path) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.get("/api/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["service"] == "mnist-backend"
    assert body["storage"]["ready"] is True
    assert Path(body["storage"]["root"]).resolve() == tmp_path.resolve()
    assert body["storage"]["directories"] == [
        "shipped-models",
        "custom-models",
        "registry",
    ]

    for directory in body["storage"]["directories"]:
        assert (tmp_path / directory).is_dir()