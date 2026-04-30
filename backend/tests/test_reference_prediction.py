import json
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from backend.app.main import create_app
from backend.app.reference_model import flatten, preprocess_canvas


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


def write_knn_fixture_artifact(storage_root: Path, predicted_digit: int) -> None:
    shipped_models_root = storage_root / "shipped-models"
    shipped_models_root.mkdir(parents=True, exist_ok=True)

    processed_canvas = preprocess_canvas(
        width=20,
        height=20,
        pixels=make_digit_one_canvas(),
    )
    target_features = flatten(processed_canvas)
    blank_features = [0.0] * len(target_features)

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit([target_features, blank_features], [predicted_digit, 0])

    runtime_artifact_path = shipped_models_root / "knn-classifier-v1.pkl"
    with runtime_artifact_path.open("wb") as artifact_file:
        pickle.dump(classifier, artifact_file)

    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    confusion_matrix[1][predicted_digit] = 1

    metadata = {
        "id": "knn-classifier-v1",
        "name": "k-NN Classifier",
        "kind": "built-in",
        "family": "knn",
        "version": "1.0.0",
        "description": "Tiny artifact-backed fixture for shipped k-NN integration tests.",
        "trained_at": "2026-04-30T00:00:00Z",
        "input": {
            "width": 20,
            "height": 20,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": "train.csv",
            "image_shape": "28x28 grayscale",
            "train_examples": 2,
            "validation_examples": 1,
            "test_examples": 1,
            "split_seed": 13,
        },
        "metrics": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "avg_inference_ms": 0.1,
        },
        "hyperparameters": {
            "neighbors": 1,
            "distance_metric": "euclidean",
            "weighting": "uniform",
            "target_size": 28,
            "content_box": 20,
            "centering": "center-of-mass",
        },
        "evaluation": {
            "confusion_matrix": confusion_matrix,
            "sample_predictions": [
                {
                    "label": 1,
                    "predicted": predicted_digit,
                    "confidence": 1.0,
                }
            ],
        },
        "artifact": {
            "version": 1,
            "serializer": "pickle",
            "estimator": "sklearn.neighbors.KNeighborsClassifier",
            "path": "shipped-models/knn-classifier-v1.pkl",
        },
    }
    (shipped_models_root / "knn-classifier-v1.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def write_svm_fixture_artifact(storage_root: Path, predicted_digit: int) -> None:
    shipped_models_root = storage_root / "shipped-models"
    shipped_models_root.mkdir(parents=True, exist_ok=True)

    processed_canvas = preprocess_canvas(
        width=20,
        height=20,
        pixels=make_digit_one_canvas(),
    )
    target_features = flatten(processed_canvas)
    training_features = []
    training_labels = []

    for digit in range(10):
        if digit == predicted_digit:
            training_features.append(target_features)
        else:
            feature_row = [0.0] * len(target_features)
            feature_row[digit] = 1.0
            training_features.append(feature_row)

        training_labels.append(digit)

    classifier = LinearSVC(random_state=13, max_iter=5000)
    classifier.fit(training_features, training_labels)

    runtime_artifact_path = shipped_models_root / "svm-classifier-v1.pkl"
    with runtime_artifact_path.open("wb") as artifact_file:
        pickle.dump(classifier, artifact_file)

    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    confusion_matrix[1][predicted_digit] = 1

    metadata = {
        "id": "svm-classifier-v1",
        "name": "SVM Classifier",
        "kind": "built-in",
        "family": "svm",
        "version": "1.0.0",
        "description": "Tiny artifact-backed fixture for shipped SVM integration tests.",
        "trained_at": "2026-04-30T00:00:00Z",
        "input": {
            "width": 20,
            "height": 20,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": "train.csv",
            "image_shape": "28x28 grayscale",
            "train_examples": 10,
            "validation_examples": 1,
            "test_examples": 1,
            "split_seed": 13,
        },
        "metrics": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "avg_inference_ms": 0.1,
        },
        "hyperparameters": {
            "classifier": "linear-svc",
            "max_iter": 5000,
            "target_size": 28,
            "content_box": 20,
            "centering": "center-of-mass",
        },
        "evaluation": {
            "confusion_matrix": confusion_matrix,
            "sample_predictions": [
                {
                    "label": 1,
                    "predicted": predicted_digit,
                    "confidence": 0.9,
                }
            ],
        },
        "artifact": {
            "version": 1,
            "serializer": "pickle",
            "estimator": "sklearn.svm.LinearSVC",
            "path": "shipped-models/svm-classifier-v1.pkl",
        },
    }
    (shipped_models_root / "svm-classifier-v1.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def write_random_forest_fixture_artifact(storage_root: Path, predicted_digit: int) -> None:
    shipped_models_root = storage_root / "shipped-models"
    shipped_models_root.mkdir(parents=True, exist_ok=True)

    processed_canvas = preprocess_canvas(
        width=20,
        height=20,
        pixels=make_digit_one_canvas(),
    )
    target_features = flatten(processed_canvas)
    training_features = []
    training_labels = []

    for digit in range(10):
        if digit == predicted_digit:
            training_features.append(target_features)
        else:
            feature_row = [0.0] * len(target_features)
            feature_row[digit] = 1.0
            training_features.append(feature_row)

        training_labels.append(digit)

    classifier = RandomForestClassifier(n_estimators=32, random_state=17)
    classifier.fit(training_features, training_labels)

    runtime_artifact_path = shipped_models_root / "random-forest-classifier-v1.pkl"
    with runtime_artifact_path.open("wb") as artifact_file:
        pickle.dump(classifier, artifact_file)

    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    confusion_matrix[1][predicted_digit] = 1

    metadata = {
        "id": "random-forest-classifier-v1",
        "name": "Random Forest Classifier",
        "kind": "built-in",
        "family": "random-forest",
        "version": "1.0.0",
        "description": "Tiny artifact-backed fixture for shipped random forest integration tests.",
        "trained_at": "2026-04-30T00:00:00Z",
        "input": {
            "width": 20,
            "height": 20,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": "train.csv",
            "image_shape": "28x28 grayscale",
            "train_examples": 10,
            "validation_examples": 1,
            "test_examples": 1,
            "split_seed": 17,
        },
        "metrics": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "avg_inference_ms": 0.1,
        },
        "hyperparameters": {
            "estimators": 32,
            "target_size": 28,
            "content_box": 20,
            "centering": "center-of-mass",
        },
        "evaluation": {
            "confusion_matrix": confusion_matrix,
            "sample_predictions": [
                {
                    "label": 1,
                    "predicted": predicted_digit,
                    "confidence": 0.9,
                }
            ],
        },
        "artifact": {
            "version": 1,
            "serializer": "pickle",
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "path": "shipped-models/random-forest-classifier-v1.pkl",
        },
    }
    (shipped_models_root / "random-forest-classifier-v1.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def write_mlp_fixture_artifact(storage_root: Path, predicted_digit: int) -> None:
    shipped_models_root = storage_root / "shipped-models"
    shipped_models_root.mkdir(parents=True, exist_ok=True)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model[3].bias[predicted_digit] = 5.0

    torch.save(
        {
            "version": 1,
            "family": "mlp",
            "input_size": 28 * 28,
            "hidden_layers": [16],
            "state_dict": model.state_dict(),
        },
        shipped_models_root / "mlp-classifier-v1.pt",
    )

    metadata = {
        "id": "mlp-classifier-v1",
        "name": "MLP Classifier",
        "kind": "built-in",
        "family": "mlp",
        "version": "1.0.0",
        "description": "Tiny artifact-backed fixture for shipped MLP integration tests.",
        "trained_at": "2026-04-30T00:00:00Z",
        "input": {
            "width": 20,
            "height": 20,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": "train.csv",
            "image_shape": "28x28 grayscale",
            "train_examples": 12,
            "validation_examples": 2,
            "test_examples": 2,
        },
        "metrics": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "avg_inference_ms": 0.2,
        },
        "hyperparameters": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.01,
            "target_size": 28,
            "content_box": 20,
            "centering": "center-of-mass",
        },
        "evaluation": {
            "confusion_matrix": [[1 if row == column else 0 for column in range(10)] for row in range(10)],
            "sample_predictions": [
                {
                    "label": 1,
                    "predicted": predicted_digit,
                    "confidence": 0.99,
                }
            ],
        },
        "deep_details": {
            "architecture_summary": [
                "Flatten -> Linear(784, 16) -> ReLU -> Linear(16, 10)",
            ],
            "epoch_curves": [
                {
                    "epoch": 1,
                    "train_loss": 0.12,
                    "validation_loss": 0.18,
                    "train_accuracy": 0.98,
                    "validation_accuracy": 0.96,
                }
            ],
        },
        "artifact": {
            "version": 1,
            "serializer": "torch",
            "estimator": "torch.nn.Module",
            "path": "shipped-models/mlp-classifier-v1.pt",
        },
    }
    (shipped_models_root / "mlp-classifier-v1.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def write_cnn_fixture_artifact(storage_root: Path, predicted_digit: int) -> None:
    shipped_models_root = storage_root / "shipped-models"
    shipped_models_root.mkdir(parents=True, exist_ok=True)

    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(8 * 7 * 7, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model[9].bias[predicted_digit] = 5.0

    torch.save(
        {
            "version": 1,
            "family": "cnn",
            "conv_channels": [4, 8],
            "dense_units": 16,
            "state_dict": model.state_dict(),
        },
        shipped_models_root / "cnn-classifier-v1.pt",
    )

    metadata = {
        "id": "cnn-classifier-v1",
        "name": "CNN Classifier",
        "kind": "built-in",
        "family": "cnn",
        "version": "1.0.0",
        "description": "Tiny artifact-backed fixture for shipped CNN integration tests.",
        "trained_at": "2026-04-30T00:00:00Z",
        "input": {
            "width": 20,
            "height": 20,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": "train.csv",
            "image_shape": "28x28 grayscale",
            "train_examples": 12,
            "validation_examples": 2,
            "test_examples": 2,
        },
        "metrics": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "avg_inference_ms": 0.2,
        },
        "hyperparameters": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.01,
            "target_size": 28,
            "content_box": 20,
            "centering": "center-of-mass",
        },
        "evaluation": {
            "confusion_matrix": [[1 if row == column else 0 for column in range(10)] for row in range(10)],
            "sample_predictions": [
                {
                    "label": 1,
                    "predicted": predicted_digit,
                    "confidence": 0.99,
                }
            ],
        },
        "deep_details": {
            "architecture_summary": [
                "Conv(1, 4, 3) -> ReLU -> MaxPool",
                "Conv(4, 8, 3) -> ReLU -> MaxPool",
                "Flatten -> Linear(392, 16) -> ReLU -> Linear(16, 10)",
            ],
            "epoch_curves": [
                {
                    "epoch": 1,
                    "train_loss": 0.08,
                    "validation_loss": 0.14,
                    "train_accuracy": 0.99,
                    "validation_accuracy": 0.97,
                }
            ],
        },
        "artifact": {
            "version": 1,
            "serializer": "torch",
            "estimator": "torch.nn.Module",
            "path": "shipped-models/cnn-classifier-v1.pt",
        },
    }
    (shipped_models_root / "cnn-classifier-v1.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


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


def test_knn_built_in_uses_fixture_metadata_and_runtime_artifact(tmp_path: Path) -> None:
    write_knn_fixture_artifact(tmp_path, predicted_digit=7)

    with TestClient(create_app(storage_root=tmp_path)) as client:
        models_response = client.get("/api/models")
        prediction_response = client.post(
            "/api/predict",
            json={
                "model_id": "knn-classifier-v1",
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

    assert models_response.status_code == 200
    assert prediction_response.status_code == 200

    knn_model = next(
        model for model in models_response.json()["models"] if model["id"] == "knn-classifier-v1"
    )
    prediction = prediction_response.json()

    assert knn_model["dataset"]["source"] == "train.csv"
    assert "artifact" not in knn_model
    assert prediction["model"]["dataset"]["source"] == "train.csv"
    assert prediction["prediction"]["digit"] == 7
    assert max(prediction["prediction"]["confidences"]) == prediction["prediction"]["confidences"][7]


def test_svm_built_in_uses_artifact_scores_in_prediction_response(tmp_path: Path) -> None:
    write_svm_fixture_artifact(tmp_path, predicted_digit=8)

    with TestClient(create_app(storage_root=tmp_path)) as client:
        prediction_response = client.post(
            "/api/predict",
            json={
                "model_id": "svm-classifier-v1",
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

    assert prediction_response.status_code == 200

    prediction = prediction_response.json()["prediction"]

    assert prediction["digit"] == 8
    assert abs(sum(prediction["confidences"]) - 1.0) < 1e-6
    assert sum(1 for value in prediction["confidences"] if value > 0.0) > 1
    assert max(prediction["confidences"]) == prediction["confidences"][8]


def test_random_forest_built_in_uses_artifact_probabilities(tmp_path: Path) -> None:
    write_random_forest_fixture_artifact(tmp_path, predicted_digit=9)

    with TestClient(create_app(storage_root=tmp_path)) as client:
        prediction_response = client.post(
            "/api/predict",
            json={
                "model_id": "random-forest-classifier-v1",
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

    assert prediction_response.status_code == 200

    prediction = prediction_response.json()["prediction"]

    assert prediction["digit"] == 9
    assert abs(sum(prediction["confidences"]) - 1.0) < 1e-6
    assert sum(1 for value in prediction["confidences"] if value > 0.0) > 1
    assert max(prediction["confidences"]) == prediction["confidences"][9]


def test_mlp_built_in_uses_torch_artifact_and_exposes_deep_details(tmp_path: Path) -> None:
    write_mlp_fixture_artifact(tmp_path, predicted_digit=4)

    with TestClient(create_app(storage_root=tmp_path)) as client:
        models_response = client.get("/api/models")
        prediction_response = client.post(
            "/api/predict",
            json={
                "model_id": "mlp-classifier-v1",
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

    assert models_response.status_code == 200
    assert prediction_response.status_code == 200

    mlp_model = next(
        model for model in models_response.json()["models"] if model["id"] == "mlp-classifier-v1"
    )
    prediction = prediction_response.json()

    assert "artifact" not in mlp_model
    assert mlp_model["deep_details"]["architecture_summary"]
    assert mlp_model["deep_details"]["epoch_curves"][0]["epoch"] == 1
    assert prediction["prediction"]["digit"] == 4
    assert abs(sum(prediction["prediction"]["confidences"]) - 1.0) < 1e-6
    assert max(prediction["prediction"]["confidences"]) == prediction["prediction"]["confidences"][4]


def test_cnn_built_in_uses_torch_artifact_and_exposes_deep_details(tmp_path: Path) -> None:
    write_cnn_fixture_artifact(tmp_path, predicted_digit=6)

    with TestClient(create_app(storage_root=tmp_path)) as client:
        models_response = client.get("/api/models")
        prediction_response = client.post(
            "/api/predict",
            json={
                "model_id": "cnn-classifier-v1",
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

    assert models_response.status_code == 200
    assert prediction_response.status_code == 200

    cnn_model = next(
        model for model in models_response.json()["models"] if model["id"] == "cnn-classifier-v1"
    )
    prediction = prediction_response.json()

    assert "artifact" not in cnn_model
    assert len(cnn_model["deep_details"]["architecture_summary"]) == 3
    assert cnn_model["deep_details"]["epoch_curves"][0]["validation_accuracy"] == 0.97
    assert prediction["prediction"]["digit"] == 6
    assert abs(sum(prediction["prediction"]["confidences"]) - 1.0) < 1e-6
    assert max(prediction["prediction"]["confidences"]) == prediction["prediction"]["confidences"][6]


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


def test_models_lists_all_shipped_models(tmp_path: Path) -> None:
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
        "mlp-classifier-v1",
        "cnn-classifier-v1",
    ]

    for model in models:
        assert model["kind"] == "built-in"
        assert model["metrics"]["accuracy"] > 0
        assert model["trained_at"]
        assert len(model["evaluation"]["confusion_matrix"]) == 10

    reference_model = next(model for model in models if model["id"] == "reference-prototype-v1")
    assert reference_model["hyperparameters"]["prototype_grid_size"] == 20
    assert reference_model["evaluation"]["sample_predictions"][0]["predicted"] == 1

    shipped_models = [model for model in models if model["id"] != "reference-prototype-v1"]
    classical_models = [
        model
        for model in shipped_models
        if model["id"] in {"knn-classifier-v1", "svm-classifier-v1", "random-forest-classifier-v1"}
    ]
    for model in classical_models:
        assert model["dataset"]["source"] == "train.csv"
        assert model["training"]["split_seed"] == 17
        assert "artifact" not in model

    deep_models = [
        model
        for model in shipped_models
        if model["id"] in {"mlp-classifier-v1", "cnn-classifier-v1"}
    ]
    for model in deep_models:
        assert model["dataset"]["source"] == "train.csv"
        assert model["training"]["split_seed"] == 17
        assert model["deep_details"]["architecture_summary"]
        assert model["deep_details"]["epoch_curves"]
        assert "artifact" not in model


def test_models_omit_broken_local_shipped_artifacts_and_prediction_fails_clearly(
    tmp_path: Path,
) -> None:
    write_knn_fixture_artifact(tmp_path, predicted_digit=7)
    (tmp_path / "shipped-models" / "knn-classifier-v1.pkl").unlink()

    with TestClient(create_app(storage_root=tmp_path)) as client:
        models_response = client.get("/api/models")
        prediction_response = client.post(
            "/api/predict",
            json={
                "model_id": "knn-classifier-v1",
                "canvas": {
                    "width": 20,
                    "height": 20,
                    "pixels": make_digit_one_canvas(),
                },
            },
        )

    assert models_response.status_code == 200
    assert "knn-classifier-v1" not in {
        model["id"] for model in models_response.json()["models"]
    }
    assert prediction_response.status_code == 400
    assert "runtime artifact is missing" in prediction_response.json()["detail"]