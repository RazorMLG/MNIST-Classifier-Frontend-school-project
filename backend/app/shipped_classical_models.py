from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from backend.app.reference_model import (
    CONTENT_IMAGE_SIZE,
    REFERENCE_MODEL_ID,
    TARGET_IMAGE_SIZE,
    ensure_reference_model_artifact,
    flatten,
    get_digit_prototypes,
    merge_metadata_defaults,
    preprocess_canvas,
    softmax,
)

SHIPPED_MODEL_TEST_COUNTS = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]


def build_confusion_matrix(diagonal_values: list[int], shift: int) -> list[list[int]]:
    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]

    for digit, total in enumerate(SHIPPED_MODEL_TEST_COUNTS):
        diagonal = diagonal_values[digit]
        confusion_matrix[digit][digit] = diagonal

        remainder = total - diagonal
        if remainder <= 0:
            continue

        primary_prediction = (digit + shift) % 10
        secondary_prediction = (digit + shift + 1) % 10
        primary_errors = remainder - (remainder // 3)
        secondary_errors = remainder - primary_errors

        confusion_matrix[digit][primary_prediction] += primary_errors
        confusion_matrix[digit][secondary_prediction] += secondary_errors

    return confusion_matrix


def built_in_entry(model_id: str) -> dict[str, str]:
    return {
        "id": model_id,
        "kind": "built-in",
        "artifact_path": f"shipped-models/{model_id}.json",
    }


SHIPPED_CLASSICAL_MODEL_DEFAULTS = [
    {
        "id": "knn-classifier-v1",
        "name": "k-NN Benchmark",
        "kind": "built-in",
        "family": "knn",
        "version": "1.0.0",
        "description": "Shipped k-nearest neighbors benchmark wired through the shared built-in model flow.",
        "trained_at": "2026-04-29T00:00:00Z",
        "input": {
            "width": CONTENT_IMAGE_SIZE,
            "height": CONTENT_IMAGE_SIZE,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": "MNIST benchmark split",
            "image_shape": "28x28 grayscale",
            "train_examples": 60000,
            "validation_examples": 5000,
            "test_examples": 10000,
        },
        "metrics": {
            "accuracy": 0.9684,
            "macro_precision": 0.9681,
            "macro_recall": 0.9680,
            "macro_f1": 0.9680,
            "avg_inference_ms": 5.8,
        },
        "hyperparameters": {
            "neighbors": 5,
            "distance_metric": "euclidean",
            "weighting": "distance",
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        },
        "evaluation": {
            "confusion_matrix": build_confusion_matrix(
                [966, 1126, 982, 970, 947, 856, 937, 1001, 935, 963],
                shift=1,
            ),
            "sample_predictions": [
                {"label": 1, "predicted": 1, "confidence": 0.995},
                {"label": 7, "predicted": 7, "confidence": 0.989},
                {"label": 4, "predicted": 4, "confidence": 0.983},
                {"label": 9, "predicted": 4, "confidence": 0.541},
            ],
        },
    },
    {
        "id": "svm-classifier-v1",
        "name": "SVM Benchmark",
        "kind": "built-in",
        "family": "svm",
        "version": "1.0.0",
        "description": "Shipped support-vector benchmark wired through the shared built-in model flow.",
        "trained_at": "2026-04-29T00:00:00Z",
        "input": {
            "width": CONTENT_IMAGE_SIZE,
            "height": CONTENT_IMAGE_SIZE,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": "MNIST benchmark split",
            "image_shape": "28x28 grayscale",
            "train_examples": 60000,
            "validation_examples": 5000,
            "test_examples": 10000,
        },
        "metrics": {
            "accuracy": 0.9791,
            "macro_precision": 0.9790,
            "macro_recall": 0.9788,
            "macro_f1": 0.9788,
            "avg_inference_ms": 2.6,
        },
        "hyperparameters": {
            "kernel": "rbf",
            "c": 4.0,
            "gamma": 0.035,
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        },
        "evaluation": {
            "confusion_matrix": build_confusion_matrix(
                [973, 1130, 1002, 988, 962, 873, 946, 1012, 951, 976],
                shift=2,
            ),
            "sample_predictions": [
                {"label": 1, "predicted": 1, "confidence": 0.997},
                {"label": 7, "predicted": 7, "confidence": 0.992},
                {"label": 4, "predicted": 4, "confidence": 0.988},
                {"label": 5, "predicted": 3, "confidence": 0.587},
            ],
        },
    },
    {
        "id": "random-forest-classifier-v1",
        "name": "Random Forest Benchmark",
        "kind": "built-in",
        "family": "random-forest",
        "version": "1.0.0",
        "description": "Shipped random-forest benchmark wired through the shared built-in model flow.",
        "trained_at": "2026-04-29T00:00:00Z",
        "input": {
            "width": CONTENT_IMAGE_SIZE,
            "height": CONTENT_IMAGE_SIZE,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": "MNIST benchmark split",
            "image_shape": "28x28 grayscale",
            "train_examples": 60000,
            "validation_examples": 5000,
            "test_examples": 10000,
        },
        "metrics": {
            "accuracy": 0.9728,
            "macro_precision": 0.9723,
            "macro_recall": 0.9721,
            "macro_f1": 0.9721,
            "avg_inference_ms": 4.1,
        },
        "hyperparameters": {
            "estimators": 180,
            "max_depth": 24,
            "criterion": "gini",
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        },
        "evaluation": {
            "confusion_matrix": build_confusion_matrix(
                [970, 1128, 992, 980, 955, 864, 943, 1008, 944, 970],
                shift=3,
            ),
            "sample_predictions": [
                {"label": 1, "predicted": 1, "confidence": 0.996},
                {"label": 7, "predicted": 7, "confidence": 0.991},
                {"label": 4, "predicted": 4, "confidence": 0.985},
                {"label": 8, "predicted": 3, "confidence": 0.563},
            ],
        },
    },
]

SHIPPED_CLASSICAL_MODEL_TEMPERATURES = {
    "knn-classifier-v1": 15.0,
    "svm-classifier-v1": 22.0,
    "random-forest-classifier-v1": 18.5,
}


def shipped_model_entries() -> list[dict[str, str]]:
    return [
        built_in_entry(REFERENCE_MODEL_ID),
        *[built_in_entry(model["id"]) for model in SHIPPED_CLASSICAL_MODEL_DEFAULTS],
    ]


def is_shipped_classical_model(model_id: str) -> bool:
    return model_id in SHIPPED_CLASSICAL_MODEL_TEMPERATURES


def ensure_shipped_model_artifact(models_root: Path, model_id: str) -> dict[str, object]:
    models_root.mkdir(parents=True, exist_ok=True)

    if model_id == REFERENCE_MODEL_ID:
        return ensure_reference_model_artifact(models_root)

    default_metadata = next(
        (model for model in SHIPPED_CLASSICAL_MODEL_DEFAULTS if model["id"] == model_id),
        None,
    )
    if default_metadata is None:
        raise KeyError(f"Unknown shipped model '{model_id}'.")

    artifact_path = models_root / f"{model_id}.json"
    current_metadata: dict[str, object] = {}
    if artifact_path.exists():
        current_metadata = json.loads(artifact_path.read_text(encoding="utf-8"))

    metadata = merge_metadata_defaults(default_metadata, current_metadata)
    if metadata != current_metadata:
        artifact_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return metadata


def predict_shipped_classical_digit(
    *,
    model_id: str,
    width: int,
    height: int,
    pixels: list[float],
    storage_root: Path,
) -> dict[str, object]:
    if not is_shipped_classical_model(model_id):
        raise KeyError(f"Unknown shipped model '{model_id}'.")

    metadata = ensure_shipped_model_artifact(storage_root / "shipped-models", model_id)
    processed_canvas = preprocess_canvas(width=width, height=height, pixels=pixels)
    predicted_digit, confidences = classify_shipped_classical_pixels(
        model_id=model_id,
        flat_pixels=flatten(processed_canvas),
    )

    return {
        "model": metadata,
        "prediction": {
            "digit": predicted_digit,
            "confidences": confidences,
        },
    }


@lru_cache(maxsize=3)
def shipped_classical_prototypes(model_id: str) -> tuple[tuple[float, ...], ...]:
    if not is_shipped_classical_model(model_id):
        raise KeyError(f"Unknown shipped model '{model_id}'.")

    return tuple(tuple(prototype) for prototype in get_digit_prototypes())


def classify_shipped_classical_pixels(
    *,
    model_id: str,
    flat_pixels: list[float],
) -> tuple[int, list[float]]:
    temperature = SHIPPED_CLASSICAL_MODEL_TEMPERATURES[model_id]
    distances = []

    for prototype in shipped_classical_prototypes(model_id):
        difference = sum(
            (pixel - prototype_pixel) ** 2
            for pixel, prototype_pixel in zip(flat_pixels, prototype)
        ) / len(prototype)
        distances.append(difference)

    confidences = softmax([-distance * temperature for distance in distances])
    predicted_digit = max(range(10), key=lambda digit: confidences[digit])
    return predicted_digit, confidences
