import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.reference_model import flatten, render_digit


def make_reference_training_csv(samples_per_digit: int = 4) -> str:
    header = ["label", *[f"pixel{index}" for index in range(784)]]
    rows = [",".join(header)]

    for label in range(10):
        prototype_pixels = [str(round(value * 255)) for value in flatten(render_digit(label))]

        for _ in range(samples_per_digit):
            rows.append(",".join([str(label), *prototype_pixels]))

    return "\n".join(rows)


def make_digit_one_canvas(width: int = 20, height: int = 20) -> list[int]:
    pixels = [0] * (width * height)

    def paint(x: int, y: int, value: int = 255) -> None:
        if 0 <= x < width and 0 <= y < height:
            pixels[(y * width) + x] = value

    for y in range(3, 17):
        paint(10, y)
        paint(11, y)

    for offset in range(4):
        paint(10 - offset, 3 + offset)

    for x in range(7, 15):
        paint(x, 17)
        paint(x, 18)

    return pixels


def expected_shipped_model_ids() -> list[str]:
    return [
        "reference-prototype-v1",
        "knn-classifier-v1",
        "svm-classifier-v1",
        "random-forest-classifier-v1",
    ]


def wait_for_training_job(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/api/training/jobs/{job_id}")
        assert response.status_code == 200

        body = response.json()["job"]
        if body["status"] == "completed":
            return body

        time.sleep(0.02)

    raise AssertionError("Training job did not complete within the polling window.")


def test_custom_training_job_creates_and_deletes_a_shared_model_entry(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.post(
            "/api/training/jobs",
            json={
                "model_name": "Classroom Prototype",
                "file_name": "classroom-train.csv",
                "csv_text": make_reference_training_csv(),
                "split": {
                    "train_ratio": 0.6,
                    "validation_ratio": 0.2,
                    "test_ratio": 0.2,
                },
                "seed": 17,
                "hyperparameters": {
                    "max_examples_per_label": 4,
                    "prototype_blend": 0.35,
                    "temperature": 18.0,
                },
            },
        )

        assert response.status_code == 202

        start_body = response.json()["job"]
        assert start_body["model_name"] == "Classroom Prototype"

        finished_job = wait_for_training_job(client, start_body["id"])
        assert finished_job["status"] == "completed"
        assert finished_job["model_id"] != "reference-prototype-v1"

        models_response = client.get("/api/models")
        assert models_response.status_code == 200

        models = models_response.json()["models"]
        custom_model = next(
            model for model in models if model["id"] == finished_job["model_id"]
        )

        assert custom_model["name"] == "Classroom Prototype"
        assert custom_model["kind"] == "custom"
        assert custom_model["training"]["seed"] == 17
        assert custom_model["training"]["config_snapshot"]["split"] == {
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
        }

        prediction_response = client.post(
            "/api/predict",
            json={
                "model_id": finished_job["model_id"],
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

        assert prediction_response.status_code == 200
        assert prediction_response.json()["model"]["id"] == finished_job["model_id"]

        delete_response = client.delete(f"/api/models/{finished_job['model_id']}")
        assert delete_response.status_code == 204

        deleted_models = client.get("/api/models").json()["models"]
        assert [model["id"] for model in deleted_models] == expected_shipped_model_ids()


def test_custom_training_rejects_a_second_active_job(tmp_path: Path) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        start_response = client.post(
            "/api/training/jobs",
            json={
                "model_name": "First Prototype",
                "file_name": "first.csv",
                "csv_text": make_reference_training_csv(),
                "split": {
                    "train_ratio": 0.6,
                    "validation_ratio": 0.2,
                    "test_ratio": 0.2,
                },
                "seed": 21,
                "hyperparameters": {
                    "max_examples_per_label": 4,
                    "prototype_blend": 0.2,
                    "temperature": 18.0,
                },
            },
        )

        assert start_response.status_code == 202

        second_response = client.post(
            "/api/training/jobs",
            json={
                "model_name": "Second Prototype",
                "file_name": "second.csv",
                "csv_text": make_reference_training_csv(),
                "split": {
                    "train_ratio": 0.6,
                    "validation_ratio": 0.2,
                    "test_ratio": 0.2,
                },
                "seed": 22,
                "hyperparameters": {
                    "max_examples_per_label": 4,
                    "prototype_blend": 0.2,
                    "temperature": 18.0,
                },
            },
        )

        assert second_response.status_code == 409
        assert "Only one training job can run at a time" in second_response.json()["detail"]


def test_custom_training_job_is_discarded_on_shutdown(tmp_path: Path) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.post(
            "/api/training/jobs",
            json={
                "model_name": "Interrupted Prototype",
                "file_name": "interrupted.csv",
                "csv_text": make_reference_training_csv(),
                "split": {
                    "train_ratio": 0.6,
                    "validation_ratio": 0.2,
                    "test_ratio": 0.2,
                },
                "seed": 23,
                "hyperparameters": {
                    "max_examples_per_label": 4,
                    "prototype_blend": 0.2,
                    "temperature": 18.0,
                },
            },
        )

        assert response.status_code == 202

    custom_models_root = tmp_path / "custom-models"
    assert list(custom_models_root.glob("*.json")) == []

    with TestClient(create_app(storage_root=tmp_path)) as restarted_client:
        models_response = restarted_client.get("/api/models")

    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["models"]] == expected_shipped_model_ids()