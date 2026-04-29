from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app


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


def test_reference_model_predicts_digit_one_from_canvas(tmp_path: Path) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.post(
            "/api/predict",
            json={
                "model_id": "reference-prototype-v1",
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["model"]["id"] == "reference-prototype-v1"
    assert body["prediction"]["digit"] == 1
    assert len(body["prediction"]["confidences"]) == 10
    assert max(body["prediction"]["confidences"]) == body["prediction"]["confidences"][1]
    assert abs(sum(body["prediction"]["confidences"]) - 1.0) < 1e-6


@pytest.mark.parametrize(
    "model_id",
    [
        "knn-classifier-v1",
        "svm-classifier-v1",
        "random-forest-classifier-v1",
    ],
)
def test_shipped_classical_models_predict_digit_one_from_canvas(
    tmp_path: Path,
    model_id: str,
) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.post(
            "/api/predict",
            json={
                "model_id": model_id,
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["model"]["id"] == model_id
    assert body["model"]["kind"] == "built-in"
    assert body["prediction"]["digit"] == 1
    assert len(body["prediction"]["confidences"]) == 10
    assert max(body["prediction"]["confidences"]) == body["prediction"]["confidences"][1]
    assert abs(sum(body["prediction"]["confidences"]) - 1.0) < 1e-6


def test_models_lists_all_shipped_classical_models(tmp_path: Path) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.get("/api/models")

    assert response.status_code == 200

    models = body = response.json()["models"]
    model_ids = [model["id"] for model in models]

    assert model_ids == [
        "reference-prototype-v1",
        "knn-classifier-v1",
        "svm-classifier-v1",
        "random-forest-classifier-v1",
    ]

    for model in models:
        assert model["kind"] == "built-in"
        assert model["metrics"]["accuracy"] > 0
        assert model["dataset"]["source"] == "MNIST benchmark split"
        assert model["trained_at"]
        assert len(model["evaluation"]["confusion_matrix"]) == 10

    reference_model = next(model for model in models if model["id"] == "reference-prototype-v1")
    assert reference_model["hyperparameters"]["prototype_grid_size"] == 20
    assert reference_model["evaluation"]["sample_predictions"][0]["predicted"] == 1