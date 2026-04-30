from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from backend.app.custom_training import calculate_metrics
from backend.app.reference_model import (
    CONTENT_IMAGE_SIZE,
    TARGET_IMAGE_SIZE,
    preprocess_mnist_pixels,
)
from backend.app.training_csv import parse_training_csv_rows

ARTIFACT_VERSION = 1
DEFAULT_SPLIT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}

MODEL_SPECS = {
    "knn-classifier-v1": {
        "name": "k-NN Classifier",
        "family": "knn",
        "version": "1.0.0",
        "description": "Shipped k-nearest neighbors classifier trained from the repository train.csv.",
        "hyperparameters": {
            "neighbors": 5,
            "distance_metric": "euclidean",
            "weighting": "distance",
            "pca_components": 32,
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        },
    },
    "svm-classifier-v1": {
        "name": "SVM Classifier",
        "family": "svm",
        "version": "1.0.0",
        "description": "Shipped linear SVM classifier trained from the repository train.csv.",
        "hyperparameters": {
            "classifier": "linear-svc",
            "max_iter": 5000,
            "regularization": 1.0,
            "pca_components": 48,
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        },
    },
    "random-forest-classifier-v1": {
        "name": "Random Forest Classifier",
        "family": "random-forest",
        "version": "1.0.0",
        "description": "Shipped random forest classifier trained from the repository train.csv.",
        "hyperparameters": {
            "estimators": 120,
            "max_depth": 24,
            "feature_projection": "pca",
            "pca_components": 48,
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        },
    },
}


def regenerate_shipped_classical_models(
    *,
    dataset_path: Path,
    storage_root: Path,
    model_ids: list[str] | None = None,
    seed: int = 17,
) -> list[dict[str, object]]:
    training_rows = parse_training_csv_rows(dataset_path.read_text(encoding="utf-8"))
    features = [preprocess_mnist_pixels(pixels) for _, pixels in training_rows]
    labels = [label for label, _ in training_rows]

    train_features, validation_features, test_features, train_labels, validation_labels, test_labels = (
        split_dataset(features=features, labels=labels, seed=seed)
    )

    selected_model_ids = model_ids or list(MODEL_SPECS)
    shipped_models_root = storage_root / "shipped-models"
    shipped_models_root.mkdir(parents=True, exist_ok=True)

    generated_models: list[dict[str, object]] = []
    for model_id in selected_model_ids:
        if model_id not in MODEL_SPECS:
            raise KeyError(f"Unknown shipped classical model '{model_id}'.")

        model_spec = MODEL_SPECS[model_id]
        estimator = build_estimator(model_id=model_id, seed=seed)
        estimator.fit(train_features, train_labels)

        metrics, evaluation_matrix, sample_predictions = evaluate_estimator(
            estimator=estimator,
            evaluation_features=test_features or validation_features,
            evaluation_labels=test_labels or validation_labels,
        )

        artifact_path = f"shipped-models/{model_id}.pkl.gz"
        metadata = {
            "id": model_id,
            "name": model_spec["name"],
            "kind": "built-in",
            "family": model_spec["family"],
            "version": model_spec["version"],
            "description": model_spec["description"],
            "trained_at": now_timestamp(),
            "input": {
                "width": CONTENT_IMAGE_SIZE,
                "height": CONTENT_IMAGE_SIZE,
                "encoding": "grayscale-intensity",
            },
            "dataset": {
                "source": dataset_path.name,
                "image_shape": "28x28 grayscale",
                "train_examples": len(train_labels),
                "validation_examples": len(validation_labels),
                "test_examples": len(test_labels),
            },
            "training": {
                "split_seed": seed,
                "split_counts": {
                    "train": len(train_labels),
                    "validation": len(validation_labels),
                    "test": len(test_labels),
                },
                "split_ratios": DEFAULT_SPLIT_RATIOS,
            },
            "preprocessing": {
                "target_size": TARGET_IMAGE_SIZE,
                "content_box": CONTENT_IMAGE_SIZE,
                "centering": "center-of-mass",
            },
            "metrics": metrics,
            "hyperparameters": model_spec["hyperparameters"],
            "evaluation": {
                "confusion_matrix": evaluation_matrix,
                "sample_predictions": sample_predictions,
            },
            "artifact": {
                "version": ARTIFACT_VERSION,
                "serializer": "pickle-gzip",
                "estimator": "sklearn.pipeline.Pipeline",
                "path": artifact_path,
            },
        }

        write_pickle(shipped_models_root / f"{model_id}.pkl.gz", estimator)
        write_metadata(shipped_models_root / f"{model_id}.json", metadata)
        generated_models.append(metadata)

    return generated_models


def split_dataset(
    *,
    features: list[list[float]],
    labels: list[int],
    seed: int,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[int],
    list[int],
    list[int],
]:
    train_validation_features, test_features, train_validation_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=DEFAULT_SPLIT_RATIOS["test"],
        random_state=seed,
        stratify=labels,
    )

    validation_ratio = DEFAULT_SPLIT_RATIOS["validation"] / (
        DEFAULT_SPLIT_RATIOS["train"] + DEFAULT_SPLIT_RATIOS["validation"]
    )
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        train_validation_features,
        train_validation_labels,
        test_size=validation_ratio,
        random_state=seed,
        stratify=train_validation_labels,
    )

    return (
        train_features,
        validation_features,
        test_features,
        train_labels,
        validation_labels,
        test_labels,
    )


def build_estimator(*, model_id: str, seed: int):
    if model_id == "knn-classifier-v1":
        return Pipeline(
            steps=[
                ("pca", PCA(n_components=32, random_state=seed)),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        metric="euclidean",
                        weights="distance",
                    ),
                ),
            ]
        )

    if model_id == "svm-classifier-v1":
        return Pipeline(
            steps=[
                ("pca", PCA(n_components=48, random_state=seed)),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_id == "random-forest-classifier-v1":
        return Pipeline(
            steps=[
                ("pca", PCA(n_components=48, random_state=seed)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=24,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise KeyError(f"Unknown shipped classical model '{model_id}'.")


def evaluate_estimator(
    *,
    estimator,
    evaluation_features: list[list[float]],
    evaluation_labels: list[int],
) -> tuple[dict[str, float], list[list[int]], list[dict[str, object]]]:
    if not evaluation_labels:
        raise ValueError("Shipped classical training requires at least one evaluation example.")

    predicted_digits = [int(digit) for digit in estimator.predict(evaluation_features)]
    evaluation_matrix = confusion_matrix(
        evaluation_labels,
        predicted_digits,
        labels=list(range(10)),
    ).tolist()

    sample_confidences = predict_confidences(
        estimator=estimator,
        evaluation_features=evaluation_features[:4],
        predicted_digits=predicted_digits[:4],
    )
    sample_predictions = [
        {
            "label": int(label),
            "predicted": predicted_digit,
            "confidence": round(max(confidences), 4),
        }
        for label, predicted_digit, confidences in zip(
            evaluation_labels[:4],
            predicted_digits[:4],
            sample_confidences,
        )
    ]

    timed_features = evaluation_features[: min(len(evaluation_features), 64)]
    inference_times = []
    for feature in timed_features:
        started_at = time.perf_counter()
        estimator.predict([feature])
        inference_times.append((time.perf_counter() - started_at) * 1000)

    return calculate_metrics(evaluation_matrix, inference_times), evaluation_matrix, sample_predictions


def predict_confidences(
    *,
    estimator,
    evaluation_features: list[list[float]],
    predicted_digits: list[int],
) -> list[list[float]]:
    if not evaluation_features:
        return []

    if hasattr(estimator, "predict_proba"):
        probability_rows = estimator.predict_proba(evaluation_features)
        class_labels = [int(label) for label in estimator.classes_]
        return [
            expand_probabilities(class_labels=class_labels, probabilities=probabilities)
            for probabilities in probability_rows
        ]

    return [one_hot_confidence(predicted_digit) for predicted_digit in predicted_digits]


def expand_probabilities(*, class_labels: list[int], probabilities) -> list[float]:
    confidences = [0.0] * 10
    for class_label, probability in zip(class_labels, probabilities):
        if 0 <= class_label < len(confidences):
            confidences[class_label] = float(probability)

    total_confidence = sum(confidences)
    if total_confidence <= 0:
        return confidences

    return [value / total_confidence for value in confidences]


def one_hot_confidence(predicted_digit: int) -> list[float]:
    confidences = [0.0] * 10
    confidences[predicted_digit] = 1.0
    return confidences


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_pickle(path: Path, estimator) -> None:
    with gzip.open(path, "wb") as artifact_file:
        pickle.dump(estimator, artifact_file)


def now_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate shipped classical model artifacts from a labeled MNIST CSV.",
    )
    parser.add_argument(
        "--dataset",
        default="train.csv",
        help="Path to the labeled MNIST CSV used for training.",
    )
    parser.add_argument(
        "--storage-root",
        default="data",
        help="Directory where shipped-model metadata and artifacts should be written.",
    )
    parser.add_argument(
        "--model-id",
        dest="model_ids",
        action="append",
        help="Optional shipped model id to regenerate. Repeat to regenerate a subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Deterministic split seed used for train/validation/test generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_models = regenerate_shipped_classical_models(
        dataset_path=Path(args.dataset).resolve(),
        storage_root=Path(args.storage_root).resolve(),
        model_ids=args.model_ids,
        seed=args.seed,
    )

    for metadata in generated_models:
        print(f"Generated {metadata['id']} -> {metadata['artifact']['path']}")


if __name__ == "__main__":
    main()