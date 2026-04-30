from __future__ import annotations

import gzip
import json
import pickle
import random
import re
import threading
import time
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from backend.app.reference_model import (
    CONTENT_IMAGE_SIZE,
    REFERENCE_MODEL_ID,
    TARGET_IMAGE_SIZE,
    flatten,
    get_digit_prototypes,
    preprocess_canvas,
    preprocess_mnist_pixels,
    softmax,
)
from backend.app.shipped_classical_models import (
    build_estimator_confidences,
    classify_shipped_classical_artifact,
    classify_shipped_deep_artifact,
    ensure_shipped_model_artifact,
    is_shipped_classical_model,
    is_shipped_deep_model,
    predict_shipped_classical_digit,
    predict_shipped_deep_digit,
    shipped_model_entries,
)
from backend.app.training_csv import calculate_split_counts, parse_training_csv_rows

REGISTRY_FILE_NAME = "models.json"
JOB_STAGE_DELAY_SECONDS = 0.04
CLASSICAL_MODEL_FAMILIES = {"knn", "svm", "random-forest"}
DEEP_MODEL_FAMILIES = {"mlp", "cnn"}
CUSTOM_CLASSICAL_MODEL_SPECS = {
    "knn": {
        "description": "Custom k-nearest neighbors classifier trained from an uploaded MNIST CSV.",
        "hyperparameters": {
            "distance_metric": "euclidean",
            "weighting": "distance",
        },
    },
    "svm": {
        "description": "Custom linear SVM classifier trained from an uploaded MNIST CSV.",
        "hyperparameters": {
            "classifier": "linear-svc",
        },
    },
    "random-forest": {
        "description": "Custom random forest classifier trained from an uploaded MNIST CSV.",
        "hyperparameters": {
            "feature_projection": "pca",
        },
    },
}
CUSTOM_DEEP_MODEL_SPECS = {
    "mlp": {
        "description": "Custom multilayer perceptron classifier trained from an uploaded MNIST CSV.",
        "hidden_layers": [128, 64],
        "architecture_summary": [
            "Flatten -> Linear(784, 128) -> ReLU",
            "Linear(128, 64) -> ReLU",
            "Linear(64, 10)",
        ],
    },
    "cnn": {
        "description": "Custom convolutional neural network classifier trained from an uploaded MNIST CSV.",
        "conv_channels": [8, 16],
        "dense_units": 32,
        "architecture_summary": [
            "Conv(1, 8, 3) -> ReLU -> MaxPool",
            "Conv(8, 16, 3) -> ReLU -> MaxPool",
            "Flatten -> Linear(784, 32) -> ReLU -> Linear(32, 10)",
        ],
    },
}


class ModelConflictError(ValueError):
    pass


class JobConflictError(RuntimeError):
    pass


class JobInterruptedError(RuntimeError):
    pass


class ModelDeletionError(ValueError):
    pass


def ensure_model_registry(storage_root: Path) -> dict[str, object]:
    storage_root.mkdir(parents=True, exist_ok=True)
    registry_path = storage_root / "registry" / REGISTRY_FILE_NAME
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry: dict[str, object] = {"version": 1, "models": []}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    current_entries = registry.get("models", [])
    deduped_entries: list[dict[str, str]] = []
    seen_model_ids: set[str] = set()

    for entry in [*shipped_model_entries(), *current_entries]:
        model_id = str(entry.get("id", "")).strip()
        kind = str(entry.get("kind", "")).strip()
        artifact_path = str(entry.get("artifact_path", "")).strip()

        if not model_id or not kind or not artifact_path or model_id in seen_model_ids:
            continue

        deduped_entries.append(
            {
                "id": model_id,
                "kind": kind,
                "artifact_path": artifact_path.replace("\\", "/"),
            }
        )
        seen_model_ids.add(model_id)

    next_registry = {"version": 1, "models": deduped_entries}
    if next_registry != registry:
        registry_path.write_text(
            json.dumps(next_registry, indent=2) + "\n",
            encoding="utf-8",
        )

    return next_registry


def list_available_models(storage_root: Path) -> list[dict[str, object]]:
    registry = ensure_model_registry(storage_root)
    next_entries: list[dict[str, str]] = []
    models: list[dict[str, object]] = []

    for entry in registry["models"]:
        if entry["kind"] == "built-in":
            try:
                metadata = ensure_shipped_model_artifact(
                    storage_root / "shipped-models",
                    entry["id"],
                )
            except (KeyError, ValueError):
                continue
        else:
            artifact_path = storage_root / entry["artifact_path"]
            if not artifact_path.exists():
                continue

            metadata = json.loads(artifact_path.read_text(encoding="utf-8"))
            runtime_artifact_path = resolve_runtime_artifact_path(
                storage_root=storage_root,
                metadata=metadata,
            )
            if runtime_artifact_path is not None and not runtime_artifact_path.exists():
                continue
        models.append(to_public_model_metadata(metadata))
        next_entries.append(entry)

    if next_entries != registry["models"]:
        save_model_registry(storage_root, next_entries)

    return models


def predict_available_model(
    *,
    model_id: str,
    width: int,
    height: int,
    pixels: list[float],
    storage_root: Path,
) -> dict[str, object]:
    if model_id == REFERENCE_MODEL_ID:
        from backend.app.reference_model import predict_reference_digit

        return predict_reference_digit(
            model_id=model_id,
            width=width,
            height=height,
            pixels=pixels,
            storage_root=storage_root,
        )

    if is_shipped_classical_model(model_id):
        return predict_shipped_classical_digit(
            model_id=model_id,
            width=width,
            height=height,
            pixels=pixels,
            storage_root=storage_root,
        )

    if is_shipped_deep_model(model_id):
        return predict_shipped_deep_digit(
            model_id=model_id,
            width=width,
            height=height,
            pixels=pixels,
            storage_root=storage_root,
        )

    metadata = load_model_artifact(storage_root, model_id)
    family = str(metadata.get("family", "")).strip()
    if family in CLASSICAL_MODEL_FAMILIES:
        processed_canvas = preprocess_canvas(width=width, height=height, pixels=pixels)
        predicted_digit, confidences = classify_shipped_classical_artifact(
            metadata=metadata,
            flat_pixels=flatten(processed_canvas),
            storage_root=storage_root,
        )

        return {
            "model": to_public_model_metadata(metadata),
            "prediction": {
                "digit": predicted_digit,
                "confidences": confidences,
            },
        }

    if family in DEEP_MODEL_FAMILIES:
        processed_canvas = preprocess_canvas(width=width, height=height, pixels=pixels)
        predicted_digit, confidences = classify_shipped_deep_artifact(
            metadata=metadata,
            processed_canvas=processed_canvas,
            storage_root=storage_root,
        )

        return {
            "model": to_public_model_metadata(metadata),
            "prediction": {
                "digit": predicted_digit,
                "confidences": confidences,
            },
        }

    artifact = metadata.get("artifact", {})
    prototypes = artifact.get("digit_prototypes")
    if not isinstance(prototypes, list):
        raise KeyError(f"Unknown model '{model_id}'.")

    processed_canvas = preprocess_canvas(width=width, height=height, pixels=pixels)
    predicted_digit, confidences = classify_pixels(
        flat_pixels=flatten(processed_canvas),
        prototypes=prototypes,
        temperature=float(artifact.get("temperature", 18.0)),
    )

    return {
        "model": to_public_model_metadata(metadata),
        "prediction": {
            "digit": predicted_digit,
            "confidences": confidences,
        },
    }


def delete_custom_model(storage_root: Path, model_id: str) -> None:
    if model_id == REFERENCE_MODEL_ID:
        raise ModelDeletionError("Built-in models cannot be deleted.")

    registry = ensure_model_registry(storage_root)
    next_entries: list[dict[str, str]] = []
    target_entry: dict[str, str] | None = None

    for entry in registry["models"]:
        if entry["id"] == model_id:
            target_entry = entry
            continue

        next_entries.append(entry)

    if target_entry is None:
        raise KeyError(f"Unknown model '{model_id}'.")

    if target_entry["kind"] != "custom":
        raise ModelDeletionError("Only custom models can be deleted.")

    artifact_path = storage_root / target_entry["artifact_path"]
    runtime_artifact_path: Path | None = None
    if artifact_path.exists():
        try:
            metadata = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}

        runtime_artifact_path = resolve_runtime_artifact_path(
            storage_root=storage_root,
            metadata=metadata,
        )
        artifact_path.unlink()

    if runtime_artifact_path is not None and runtime_artifact_path.exists():
        runtime_artifact_path.unlink()

    save_model_registry(storage_root, next_entries)


class TrainingJobManager:
    def __init__(self, storage_root: Path) -> None:
        self._storage_root = storage_root
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}
        self._active_job_id: str | None = None
        self._active_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        ensure_model_registry(self._storage_root)

    def start_job(
        self,
        *,
        model_name: str,
        model_family: str,
        file_name: str,
        csv_text: str,
        split: dict[str, float],
        seed: int,
        hyperparameters: dict[str, int | float],
    ) -> dict[str, object]:
        normalized_name = model_name.strip()
        if not normalized_name:
            raise ValueError("Model name is required.")

        with self._lock:
            if self._active_job_id is not None:
                active_job = self._jobs[self._active_job_id]
                if active_job["status"] in {"queued", "running"}:
                    raise JobConflictError(
                        "Only one training job can run at a time in version one."
                    )

            if model_name_exists(self._storage_root, normalized_name):
                raise ModelConflictError(
                    f"Model name '{normalized_name}' is already in use. Choose a unique name."
                )

            model_id = build_custom_model_id(self._storage_root, normalized_name)
            job_id = uuid.uuid4().hex
            job_snapshot = {
                "id": job_id,
                "model_id": model_id,
                "model_name": normalized_name,
                "model_family": model_family,
                "status": "queued",
                "progress": {
                    "percent": 0.0,
                    "stage": "Queued",
                },
                "error": None,
                "started_at": now_timestamp(),
                "completed_at": None,
            }

            self._jobs[job_id] = job_snapshot
            self._active_job_id = job_id
            self._shutdown_event.clear()
            self._active_thread = threading.Thread(
                target=self._run_job,
                args=(
                    job_id,
                    {
                        "model_id": model_id,
                        "model_name": normalized_name,
                        "model_family": model_family,
                        "file_name": file_name,
                        "csv_text": csv_text,
                        "split": split,
                        "seed": seed,
                        "hyperparameters": hyperparameters,
                    },
                ),
                daemon=True,
            )
            self._active_thread.start()

            return deepcopy(job_snapshot)

    def get_job(self, job_id: str) -> dict[str, object]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)

            return deepcopy(self._jobs[job_id])

    def shutdown(self) -> None:
        self._shutdown_event.set()

        active_thread = self._active_thread
        if active_thread is not None and active_thread.is_alive():
            active_thread.join(timeout=2)

    def _run_job(self, job_id: str, request: dict[str, object]) -> None:
        try:
            self._update_job(job_id, status="running", percent=0.08, stage="Validating dataset")
            sleep_for_stage(self._shutdown_event)
            training_rows = parse_training_csv_rows(str(request["csv_text"]))

            split = request["split"]
            split_counts = calculate_split_counts(
                example_count=len(training_rows),
                train_ratio=float(split["train_ratio"]),
                validation_ratio=float(split["validation_ratio"]),
                test_ratio=float(split["test_ratio"]),
            )

            self._update_job(job_id, percent=0.32, stage="Shuffling and splitting data")
            sleep_for_stage(self._shutdown_event)
            train_rows, validation_rows, test_rows = split_training_rows(
                training_rows=training_rows,
                split_counts=split_counts,
                seed=int(request["seed"]),
            )

            self._update_job(
                job_id,
                percent=0.6,
                stage=build_training_stage_label(str(request["model_family"])),
            )
            sleep_for_stage(self._shutdown_event)
            artifact, runtime_artifact = build_custom_artifact(
                model_id=str(request["model_id"]),
                model_name=str(request["model_name"]),
                model_family=str(request["model_family"]),
                file_name=str(request["file_name"]),
                train_rows=train_rows,
                validation_rows=validation_rows,
                test_rows=test_rows,
                split=split,
                seed=int(request["seed"]),
                hyperparameters=request["hyperparameters"],
            )

            self._update_job(job_id, percent=0.9, stage="Persisting model metadata")
            sleep_for_stage(self._shutdown_event)
            persist_custom_artifact(
                self._storage_root,
                artifact,
                runtime_artifact=runtime_artifact,
            )

            self._update_job(
                job_id,
                status="completed",
                percent=1.0,
                stage="Training complete",
                completed_at=now_timestamp(),
            )
        except JobInterruptedError:
            self._update_job(
                job_id,
                status="cancelled",
                percent=0.0,
                stage="Interrupted during shutdown",
                error="Interrupted jobs are discarded when the app closes.",
                completed_at=now_timestamp(),
            )
        except Exception as error:  # pragma: no cover - exercised through API integration tests.
            self._update_job(
                job_id,
                status="failed",
                stage="Training failed",
                error=str(error),
                completed_at=now_timestamp(),
            )
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                    self._active_thread = None

    def _update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        percent: float | None = None,
        stage: str | None = None,
        error: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]

            if status is not None:
                job["status"] = status
            if percent is not None:
                job["progress"] = {
                    **job["progress"],
                    "percent": round(percent, 3),
                }
            if stage is not None:
                job["progress"] = {
                    **job["progress"],
                    "stage": stage,
                }
            if error is not None:
                job["error"] = error
            if completed_at is not None:
                job["completed_at"] = completed_at


def persist_custom_artifact(
    storage_root: Path,
    artifact: dict[str, object],
    *,
    runtime_artifact: object | None = None,
) -> None:
    custom_models_root = storage_root / "custom-models"
    custom_models_root.mkdir(parents=True, exist_ok=True)

    runtime_artifact_path = resolve_runtime_artifact_path(
        storage_root=storage_root,
        metadata=artifact,
    )
    if runtime_artifact is not None:
        if runtime_artifact_path is None:
            raise ValueError("Runtime artifact metadata is required for this custom model.")

        runtime_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        write_runtime_artifact(
            path=runtime_artifact_path,
            metadata=artifact,
            artifact=runtime_artifact,
        )

    artifact_path = custom_models_root / f"{artifact['id']}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    registry = ensure_model_registry(storage_root)
    next_entries = [entry for entry in registry["models"] if entry["id"] != artifact["id"]]
    next_entries.append(
        {
            "id": str(artifact["id"]),
            "kind": "custom",
            "artifact_path": f"custom-models/{artifact['id']}.json",
        }
    )
    save_model_registry(storage_root, next_entries)


def build_custom_artifact(
    *,
    model_id: str,
    model_name: str,
    model_family: str,
    file_name: str,
    train_rows: list[tuple[int, list[int]]],
    validation_rows: list[tuple[int, list[int]]],
    test_rows: list[tuple[int, list[int]]],
    split: dict[str, object],
    seed: int,
    hyperparameters: dict[str, object],
) -> tuple[dict[str, object], object | None]:
    if model_family == "prototype":
        return (
            build_custom_prototype_artifact(
                model_id=model_id,
                model_name=model_name,
                file_name=file_name,
                train_rows=train_rows,
                validation_rows=validation_rows,
                test_rows=test_rows,
                split=split,
                seed=seed,
                hyperparameters=hyperparameters,
            ),
            None,
        )

    if model_family in DEEP_MODEL_FAMILIES:
        return build_custom_deep_artifact(
            model_id=model_id,
            model_name=model_name,
            model_family=model_family,
            file_name=file_name,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            split=split,
            seed=seed,
            hyperparameters=hyperparameters,
        )

    return build_custom_classical_artifact(
        model_id=model_id,
        model_name=model_name,
        model_family=model_family,
        file_name=file_name,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        split=split,
        seed=seed,
        hyperparameters=hyperparameters,
    )


def build_custom_prototype_artifact(
    *,
    model_id: str,
    model_name: str,
    file_name: str,
    train_rows: list[tuple[int, list[int]]],
    validation_rows: list[tuple[int, list[int]]],
    test_rows: list[tuple[int, list[int]]],
    split: dict[str, object],
    seed: int,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    learned_prototypes = build_digit_prototypes(
        train_rows=train_rows,
        max_examples_per_label=int(hyperparameters["max_examples_per_label"]),
        prototype_blend=float(hyperparameters["prototype_blend"]),
    )

    metrics, confusion_matrix, sample_predictions = evaluate_prototypes(
        prototypes=learned_prototypes,
        validation_rows=validation_rows,
        test_rows=test_rows,
        temperature=float(hyperparameters["temperature"]),
    )

    return {
        "id": model_id,
        "name": model_name,
        "kind": "custom",
        "family": "prototype",
        "version": "1.0.0",
        "description": "Custom prototype classifier trained from an uploaded MNIST CSV.",
        "trained_at": now_timestamp(),
        "input": {
            "width": CONTENT_IMAGE_SIZE,
            "height": CONTENT_IMAGE_SIZE,
            "encoding": "grayscale-intensity",
        },
        "dataset": {
            "source": f"Uploaded CSV: {file_name}",
            "image_shape": "28x28 grayscale",
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "test_examples": len(test_rows),
        },
        "metrics": metrics,
        "hyperparameters": {
            "max_examples_per_label": int(hyperparameters["max_examples_per_label"]),
            "prototype_blend": round(float(hyperparameters["prototype_blend"]), 3),
            "temperature": round(float(hyperparameters["temperature"]), 3),
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        },
        "training": {
            "seed": seed,
            "config_snapshot": {
                "classifier": "reference-prototype",
                "file_name": file_name,
                "split": {
                    "train_ratio": float(split["train_ratio"]),
                    "validation_ratio": float(split["validation_ratio"]),
                    "test_ratio": float(split["test_ratio"]),
                },
                "hyperparameters": {
                    "max_examples_per_label": int(hyperparameters["max_examples_per_label"]),
                    "prototype_blend": round(float(hyperparameters["prototype_blend"]), 3),
                    "temperature": round(float(hyperparameters["temperature"]), 3),
                },
            },
        },
        "evaluation": {
            "confusion_matrix": confusion_matrix,
            "sample_predictions": sample_predictions,
        },
        "artifact": {
            "digit_prototypes": learned_prototypes,
            "temperature": round(float(hyperparameters["temperature"]), 3),
        },
    }


def build_custom_classical_artifact(
    *,
    model_id: str,
    model_name: str,
    model_family: str,
    file_name: str,
    train_rows: list[tuple[int, list[int]]],
    validation_rows: list[tuple[int, list[int]]],
    test_rows: list[tuple[int, list[int]]],
    split: dict[str, object],
    seed: int,
    hyperparameters: dict[str, object],
) -> tuple[dict[str, object], object]:
    train_features, train_labels = preprocess_training_rows(train_rows)
    validation_features, validation_labels = preprocess_training_rows(validation_rows)
    test_features, test_labels = preprocess_training_rows(test_rows)

    estimator = build_classical_estimator(
        model_family=model_family,
        hyperparameters=hyperparameters,
        seed=seed,
        train_example_count=len(train_features),
    )
    estimator.fit(train_features, train_labels)

    metrics, evaluation_matrix, sample_predictions = evaluate_classical_estimator(
        estimator=estimator,
        evaluation_features=test_features or validation_features,
        evaluation_labels=test_labels or validation_labels,
    )

    spec = CUSTOM_CLASSICAL_MODEL_SPECS[model_family]
    runtime_artifact_path = f"custom-models/{model_id}.pkl.gz"
    curated_hyperparameters = build_custom_classical_metadata_hyperparameters(
        model_family=model_family,
        hyperparameters=hyperparameters,
    )

    return (
        {
            "id": model_id,
            "name": model_name,
            "kind": "custom",
            "family": model_family,
            "version": "1.0.0",
            "description": spec["description"],
            "trained_at": now_timestamp(),
            "input": {
                "width": CONTENT_IMAGE_SIZE,
                "height": CONTENT_IMAGE_SIZE,
                "encoding": "grayscale-intensity",
            },
            "dataset": {
                "source": f"Uploaded CSV: {file_name}",
                "image_shape": "28x28 grayscale",
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "test_examples": len(test_rows),
            },
            "metrics": metrics,
            "hyperparameters": curated_hyperparameters,
            "training": {
                "seed": seed,
                "config_snapshot": {
                    "classifier": model_family,
                    "file_name": file_name,
                    "split": {
                        "train_ratio": float(split["train_ratio"]),
                        "validation_ratio": float(split["validation_ratio"]),
                        "test_ratio": float(split["test_ratio"]),
                    },
                    "hyperparameters": build_requested_hyperparameters(
                        model_family=model_family,
                        hyperparameters=hyperparameters,
                    ),
                },
            },
            "evaluation": {
                "confusion_matrix": evaluation_matrix,
                "sample_predictions": sample_predictions,
            },
            "artifact": {
                "version": 1,
                "serializer": "pickle-gzip",
                "estimator": "sklearn.pipeline.Pipeline",
                "path": runtime_artifact_path,
            },
        },
        estimator,
    )


def build_custom_deep_artifact(
    *,
    model_id: str,
    model_name: str,
    model_family: str,
    file_name: str,
    train_rows: list[tuple[int, list[int]]],
    validation_rows: list[tuple[int, list[int]]],
    test_rows: list[tuple[int, list[int]]],
    split: dict[str, object],
    seed: int,
    hyperparameters: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    from backend.app.shipped_deep_training import (
        evaluate_module,
        train_cnn_model,
        train_mlp_model,
    )

    train_features, train_labels = preprocess_training_rows(train_rows)
    validation_features, validation_labels = preprocess_training_rows(validation_rows)
    test_features, test_labels = preprocess_training_rows(test_rows)

    requested_hyperparameters = build_requested_hyperparameters(
        model_family=model_family,
        hyperparameters=hyperparameters,
    )
    spec = CUSTOM_DEEP_MODEL_SPECS[model_family]

    if model_family == "mlp":
        hidden_layers = [int(layer) for layer in spec["hidden_layers"]]
        module, epoch_curves = train_mlp_model(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            hidden_layers=hidden_layers,
            epochs=int(requested_hyperparameters["epochs"]),
            batch_size=int(requested_hyperparameters["batch_size"]),
            learning_rate=float(requested_hyperparameters["learning_rate"]),
            seed=seed,
        )
        metadata_hyperparameters = {
            **requested_hyperparameters,
            "hidden_layers": "128 -> 64",
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        }
        checkpoint_payload = {
            "version": 1,
            "family": "mlp",
            "input_size": TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE,
            "hidden_layers": hidden_layers,
            "state_dict": module.state_dict(),
        }
    else:
        conv_channels = [int(channel) for channel in spec["conv_channels"]]
        dense_units = int(spec["dense_units"])
        module, epoch_curves = train_cnn_model(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            conv_channels=conv_channels,
            dense_units=dense_units,
            epochs=int(requested_hyperparameters["epochs"]),
            batch_size=int(requested_hyperparameters["batch_size"]),
            learning_rate=float(requested_hyperparameters["learning_rate"]),
            seed=seed,
        )
        metadata_hyperparameters = {
            **requested_hyperparameters,
            "conv_channels": "8 -> 16",
            "dense_units": dense_units,
            "target_size": TARGET_IMAGE_SIZE,
            "content_box": CONTENT_IMAGE_SIZE,
            "centering": "center-of-mass",
        }
        checkpoint_payload = {
            "version": 1,
            "family": "cnn",
            "conv_channels": conv_channels,
            "dense_units": dense_units,
            "state_dict": module.state_dict(),
        }

    metrics, evaluation_matrix, sample_predictions = evaluate_module(
        module=module,
        evaluation_features=test_features or validation_features,
        evaluation_labels=test_labels or validation_labels,
        batch_size=int(requested_hyperparameters["batch_size"]),
    )

    runtime_artifact_path = f"custom-models/{model_id}.pt"
    return (
        {
            "id": model_id,
            "name": model_name,
            "kind": "custom",
            "family": model_family,
            "version": "1.0.0",
            "description": spec["description"],
            "trained_at": now_timestamp(),
            "input": {
                "width": CONTENT_IMAGE_SIZE,
                "height": CONTENT_IMAGE_SIZE,
                "encoding": "grayscale-intensity",
            },
            "dataset": {
                "source": f"Uploaded CSV: {file_name}",
                "image_shape": "28x28 grayscale",
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "test_examples": len(test_rows),
            },
            "metrics": metrics,
            "hyperparameters": metadata_hyperparameters,
            "training": {
                "seed": seed,
                "config_snapshot": {
                    "classifier": model_family,
                    "file_name": file_name,
                    "split": {
                        "train_ratio": float(split["train_ratio"]),
                        "validation_ratio": float(split["validation_ratio"]),
                        "test_ratio": float(split["test_ratio"]),
                    },
                    "hyperparameters": requested_hyperparameters,
                },
            },
            "deep_details": {
                "architecture_summary": list(spec["architecture_summary"]),
                "epoch_curves": epoch_curves,
            },
            "evaluation": {
                "confusion_matrix": evaluation_matrix,
                "sample_predictions": sample_predictions,
            },
            "artifact": {
                "version": 1,
                "serializer": "torch",
                "estimator": "torch.nn.Module",
                "path": runtime_artifact_path,
            },
        },
        checkpoint_payload,
    )


def preprocess_training_rows(
    training_rows: list[tuple[int, list[int]]],
) -> tuple[list[list[float]], list[int]]:
    return (
        [preprocess_mnist_pixels(pixels) for _, pixels in training_rows],
        [label for label, _ in training_rows],
    )


def build_classical_estimator(
    *,
    model_family: str,
    hyperparameters: dict[str, object],
    seed: int,
    train_example_count: int,
):
    pca_components = int(hyperparameters["pca_components"])
    max_pca_components = min(train_example_count, CONTENT_IMAGE_SIZE * CONTENT_IMAGE_SIZE)
    if pca_components > max_pca_components:
        raise ValueError(
            f"PCA components ({pca_components}) cannot exceed {max_pca_components} for this training split."
        )

    if model_family == "knn":
        neighbors = int(hyperparameters["neighbors"])
        if neighbors > train_example_count:
            raise ValueError(
                f"k-NN neighbors ({neighbors}) cannot exceed the {train_example_count} training examples in this split."
            )

        return Pipeline(
            steps=[
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=neighbors,
                        metric="euclidean",
                        weights="distance",
                    ),
                ),
            ]
        )

    if model_family == "svm":
        return Pipeline(
            steps=[
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "classifier",
                    LinearSVC(
                        C=float(hyperparameters["regularization"]),
                        max_iter=int(hyperparameters["max_iter"]),
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_family == "random-forest":
        return Pipeline(
            steps=[
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(hyperparameters["estimators"]),
                        max_depth=int(hyperparameters["max_depth"]),
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported custom training family '{model_family}'.")


def evaluate_classical_estimator(
    *,
    estimator,
    evaluation_features: list[list[float]],
    evaluation_labels: list[int],
) -> tuple[dict[str, float], list[list[int]], list[dict[str, object]]]:
    if not evaluation_labels:
        raise ValueError("Custom training requires at least one validation or test example.")

    predicted_digits = [int(digit) for digit in estimator.predict(evaluation_features)]
    evaluation_matrix = confusion_matrix(
        evaluation_labels,
        predicted_digits,
        labels=list(range(10)),
    ).tolist()

    sample_predictions = []
    for label, predicted_digit, feature in zip(
        evaluation_labels[:4],
        predicted_digits[:4],
        evaluation_features[:4],
    ):
        confidences = build_estimator_confidences(
            estimator=estimator,
            flat_pixels=feature,
        )
        sample_predictions.append(
            {
                "label": int(label),
                "predicted": predicted_digit,
                "confidence": round(max(confidences), 4),
            }
        )

    timed_features = evaluation_features[: min(len(evaluation_features), 64)]
    inference_times = []
    for feature in timed_features:
        started_at = time.perf_counter()
        estimator.predict([feature])
        inference_times.append((time.perf_counter() - started_at) * 1000)

    return calculate_metrics(evaluation_matrix, inference_times), evaluation_matrix, sample_predictions


def build_custom_classical_metadata_hyperparameters(
    *,
    model_family: str,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    return {
        **CUSTOM_CLASSICAL_MODEL_SPECS[model_family]["hyperparameters"],
        **build_requested_hyperparameters(
            model_family=model_family,
            hyperparameters=hyperparameters,
        ),
        "target_size": TARGET_IMAGE_SIZE,
        "content_box": CONTENT_IMAGE_SIZE,
        "centering": "center-of-mass",
    }


def build_requested_hyperparameters(
    *,
    model_family: str,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    if model_family in DEEP_MODEL_FAMILIES:
        return {
            "epochs": int(hyperparameters["epochs"]),
            "batch_size": int(hyperparameters["batch_size"]),
            "learning_rate": round(float(hyperparameters["learning_rate"]), 4),
        }

    if model_family == "knn":
        return {
            "neighbors": int(hyperparameters["neighbors"]),
            "pca_components": int(hyperparameters["pca_components"]),
        }

    if model_family == "svm":
        return {
            "regularization": round(float(hyperparameters["regularization"]), 3),
            "max_iter": int(hyperparameters["max_iter"]),
            "pca_components": int(hyperparameters["pca_components"]),
        }

    if model_family == "random-forest":
        return {
            "estimators": int(hyperparameters["estimators"]),
            "max_depth": int(hyperparameters["max_depth"]),
            "pca_components": int(hyperparameters["pca_components"]),
        }

    return {
        "max_examples_per_label": int(hyperparameters["max_examples_per_label"]),
        "prototype_blend": round(float(hyperparameters["prototype_blend"]), 3),
        "temperature": round(float(hyperparameters["temperature"]), 3),
    }


def resolve_runtime_artifact_path(
    *,
    storage_root: Path,
    metadata: dict[str, object],
) -> Path | None:
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        return None

    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None

    return storage_root / artifact_path.replace("\\", "/")


def write_runtime_artifact(
    *,
    path: Path,
    metadata: dict[str, object],
    artifact: object,
) -> None:
    artifact_metadata = metadata.get("artifact")
    serializer = "pickle-gzip"
    if isinstance(artifact_metadata, dict):
        serializer = str(artifact_metadata.get("serializer", serializer)).strip()

    if serializer == "torch":
        import torch

        torch.save(artifact, path)
        return

    if serializer == "pickle-gzip":
        write_pickle(path, artifact)
        return

    raise ValueError(f"Unsupported custom runtime serializer '{serializer}'.")


def write_pickle(path: Path, artifact: object) -> None:
    with gzip.open(path, "wb") as artifact_file:
        pickle.dump(artifact, artifact_file)


def build_training_stage_label(model_family: str) -> str:
    if model_family == "prototype":
        return "Building digit prototypes"

    if model_family == "knn":
        return "Training k-NN classifier"

    if model_family == "svm":
        return "Training SVM classifier"

    if model_family == "random-forest":
        return "Training Random Forest classifier"

    if model_family == "mlp":
        return "Training MLP classifier"

    if model_family == "cnn":
        return "Training CNN classifier"

    return "Training classifier"


def build_digit_prototypes(
    *,
    train_rows: list[tuple[int, list[int]]],
    max_examples_per_label: int,
    prototype_blend: float,
) -> list[list[float]]:
    grouped_rows: dict[int, list[list[float]]] = {digit: [] for digit in range(10)}
    for label, pixels in train_rows:
        if len(grouped_rows[label]) >= max_examples_per_label:
            continue

        grouped_rows[label].append([pixel / 255.0 for pixel in pixels])

    baseline_prototypes = get_digit_prototypes()
    next_prototypes: list[list[float]] = []

    for label in range(10):
        label_rows = grouped_rows[label]
        if not label_rows:
            next_prototypes.append(baseline_prototypes[label].copy())
            continue

        learned_average = [
            sum(row[pixel_index] for row in label_rows) / len(label_rows)
            for pixel_index in range(len(label_rows[0]))
        ]
        blended = [
            ((1.0 - prototype_blend) * learned_average[pixel_index])
            + (prototype_blend * baseline_prototypes[label][pixel_index])
            for pixel_index in range(len(learned_average))
        ]
        next_prototypes.append(blended)

    return next_prototypes


def evaluate_prototypes(
    *,
    prototypes: list[list[float]],
    validation_rows: list[tuple[int, list[int]]],
    test_rows: list[tuple[int, list[int]]],
    temperature: float,
) -> tuple[dict[str, float], list[list[int]], list[dict[str, object]]]:
    evaluation_rows = test_rows or validation_rows
    if not evaluation_rows:
        raise ValueError("Custom training requires at least one validation or test example.")

    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    sample_predictions: list[dict[str, object]] = []
    inference_times: list[float] = []

    for label, pixels in evaluation_rows:
        started_at = time.perf_counter()
        predicted_digit, confidences = classify_pixels(
            flat_pixels=[pixel / 255.0 for pixel in pixels],
            prototypes=prototypes,
            temperature=temperature,
        )
        inference_times.append((time.perf_counter() - started_at) * 1000)
        confusion_matrix[label][predicted_digit] += 1

        if len(sample_predictions) < 4:
            sample_predictions.append(
                {
                    "label": label,
                    "predicted": predicted_digit,
                    "confidence": round(max(confidences), 4),
                }
            )

    return (
        calculate_metrics(confusion_matrix, inference_times),
        confusion_matrix,
        sample_predictions,
    )


def classify_pixels(
    *,
    flat_pixels: list[float],
    prototypes: list[list[float]],
    temperature: float,
) -> tuple[int, list[float]]:
    distances = []
    for prototype in prototypes:
        difference = sum(
            (pixel - prototype_pixel) ** 2
            for pixel, prototype_pixel in zip(flat_pixels, prototype)
        ) / len(prototype)
        distances.append(difference)

    confidences = softmax([-distance * temperature for distance in distances])
    predicted_digit = max(range(10), key=lambda digit: confidences[digit])
    return predicted_digit, confidences


def split_training_rows(
    *,
    training_rows: list[tuple[int, list[int]]],
    split_counts: dict[str, int],
    seed: int,
) -> tuple[list[tuple[int, list[int]]], list[tuple[int, list[int]]], list[tuple[int, list[int]]]]:
    shuffled_rows = training_rows.copy()
    random.Random(seed).shuffle(shuffled_rows)

    train_count = split_counts["train"]
    validation_count = split_counts["validation"]

    train_rows = shuffled_rows[:train_count]
    validation_rows = shuffled_rows[train_count : train_count + validation_count]
    test_rows = shuffled_rows[train_count + validation_count :]
    return train_rows, validation_rows, test_rows


def calculate_metrics(
    confusion_matrix: list[list[int]],
    inference_times: list[float],
) -> dict[str, float]:
    total_examples = sum(sum(row) for row in confusion_matrix)
    true_positives = [confusion_matrix[digit][digit] for digit in range(10)]
    predicted_totals = [sum(confusion_matrix[row][digit] for row in range(10)) for digit in range(10)]
    actual_totals = [sum(confusion_matrix[digit]) for digit in range(10)]

    precisions = []
    recalls = []
    f1_scores = []

    for digit in range(10):
        precision = true_positives[digit] / predicted_totals[digit] if predicted_totals[digit] else 0.0
        recall = true_positives[digit] / actual_totals[digit] if actual_totals[digit] else 0.0
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    accuracy = sum(true_positives) / total_examples if total_examples else 0.0
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1_scores) / len(f1_scores)

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "avg_inference_ms": round(sum(inference_times) / len(inference_times), 3),
    }


def to_public_model_metadata(metadata: dict[str, object]) -> dict[str, object]:
    public_metadata = deepcopy(metadata)
    public_metadata.pop("artifact", None)
    return public_metadata


def load_model_artifact(storage_root: Path, model_id: str) -> dict[str, object]:
    registry = ensure_model_registry(storage_root)
    for entry in registry["models"]:
        if entry["id"] != model_id:
            continue

        artifact_path = storage_root / entry["artifact_path"]
        if not artifact_path.exists():
            break

        return json.loads(artifact_path.read_text(encoding="utf-8"))

    raise KeyError(f"Unknown model '{model_id}'.")


def model_name_exists(storage_root: Path, model_name: str) -> bool:
    normalized_name = model_name.casefold()
    return any(model["name"].casefold() == normalized_name for model in list_available_models(storage_root))


def build_custom_model_id(storage_root: Path, model_name: str) -> str:
    registry = ensure_model_registry(storage_root)
    existing_model_ids = {entry["id"] for entry in registry["models"]}
    base_slug = slugify(model_name)
    next_model_id = f"custom-{base_slug}"
    suffix = 2

    while next_model_id in existing_model_ids:
        next_model_id = f"custom-{base_slug}-{suffix}"
        suffix += 1

    return next_model_id


def save_model_registry(storage_root: Path, entries: list[dict[str, str]]) -> None:
    registry_path = storage_root / "registry" / REGISTRY_FILE_NAME
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps({"version": 1, "models": entries}, indent=2) + "\n",
        encoding="utf-8",
    )


def sleep_for_stage(shutdown_event: threading.Event) -> None:
    deadline = time.perf_counter() + JOB_STAGE_DELAY_SECONDS
    while time.perf_counter() < deadline:
        if shutdown_event.is_set():
            raise JobInterruptedError()
        time.sleep(0.005)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or uuid.uuid4().hex[:8]


def now_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")