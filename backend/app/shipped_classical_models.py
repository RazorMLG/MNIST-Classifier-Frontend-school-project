from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
import shutil

from backend.app.reference_model import (
    REFERENCE_MODEL_ID,
    ensure_reference_model_artifact,
    flatten,
    preprocess_canvas,
    softmax,
)

SHIPPED_CLASSICAL_MODEL_IDS = (
    "knn-classifier-v1",
    "svm-classifier-v1",
    "random-forest-classifier-v1",
)
SUPPORTED_ARTIFACT_SERIALIZERS = {"pickle", "pickle-gzip"}
SUPPORTED_ARTIFACT_VERSION = 1
CANONICAL_STORAGE_ROOT = Path(__file__).resolve().parents[2] / "data"
CANONICAL_SHIPPED_MODELS_ROOT = CANONICAL_STORAGE_ROOT / "shipped-models"


def built_in_entry(model_id: str) -> dict[str, str]:
    return {
        "id": model_id,
        "kind": "built-in",
        "artifact_path": f"shipped-models/{model_id}.json",
    }


def to_public_shipped_model_metadata(metadata: dict[str, object]) -> dict[str, object]:
    public_metadata = dict(metadata)
    public_metadata.pop("artifact", None)
    return public_metadata


def has_runtime_artifact(metadata: dict[str, object]) -> bool:
    artifact = metadata.get("artifact")
    return isinstance(artifact, dict) and isinstance(artifact.get("path"), str)


def canonical_metadata_path(model_id: str) -> Path:
    return CANONICAL_SHIPPED_MODELS_ROOT / f"{model_id}.json"


def validate_shipped_model_metadata(
    *,
    metadata: dict[str, object],
    model_id: str,
    storage_root: Path,
) -> None:
    if str(metadata.get("id", "")) != model_id:
        raise ValueError(
            f"Shipped model '{model_id}' metadata does not match the expected model id."
        )

    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"Shipped model '{model_id}' is missing runtime artifact metadata.")

    if artifact.get("version") != SUPPORTED_ARTIFACT_VERSION:
        raise ValueError(
            f"Shipped model '{model_id}' uses incompatible runtime artifact version '{artifact.get('version')}'."
        )

    serializer = str(artifact.get("serializer", "")).strip()
    if serializer not in SUPPORTED_ARTIFACT_SERIALIZERS:
        raise ValueError(
            f"Shipped model '{model_id}' uses unsupported serializer '{serializer}'."
        )

    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError(f"Shipped model '{model_id}' is missing a runtime artifact path.")

    resolved_artifact_path = storage_root / artifact_path.replace("\\", "/")
    if not resolved_artifact_path.exists():
        raise ValueError(
            f"Shipped model '{model_id}' runtime artifact is missing at '{artifact_path}'."
        )


def materialize_canonical_shipped_model(storage_root: Path, model_id: str) -> dict[str, object]:
    metadata_path = canonical_metadata_path(model_id)
    if not metadata_path.exists():
        raise ValueError(f"Canonical shipped metadata for '{model_id}' is missing.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_shipped_model_metadata(
        metadata=metadata,
        model_id=model_id,
        storage_root=CANONICAL_STORAGE_ROOT,
    )

    destination_metadata_path = storage_root / "shipped-models" / f"{model_id}.json"
    destination_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_metadata_path.resolve() != metadata_path.resolve():
        destination_metadata_path.write_text(
            metadata_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    artifact_path = str(metadata["artifact"]["path"]).replace("\\", "/")
    source_artifact_path = CANONICAL_STORAGE_ROOT / artifact_path
    destination_artifact_path = storage_root / artifact_path
    destination_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_artifact_path.resolve() != source_artifact_path.resolve():
        shutil.copy2(source_artifact_path, destination_artifact_path)

    return metadata


def load_runtime_artifact(
    *,
    metadata: dict[str, object],
    storage_root: Path,
):
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"Shipped model '{metadata.get('id', 'unknown')}' is missing runtime artifact metadata.")

    serializer = str(artifact.get("serializer", "pickle"))
    if serializer not in SUPPORTED_ARTIFACT_SERIALIZERS:
        raise ValueError(
            f"Shipped model '{metadata.get('id', 'unknown')}' uses unsupported serializer '{serializer}'."
        )

    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError(f"Shipped model '{metadata.get('id', 'unknown')}' is missing a runtime artifact path.")

    resolved_path = storage_root / artifact_path.replace("\\", "/")
    if not resolved_path.exists():
        raise ValueError(
            f"Shipped model '{metadata.get('id', 'unknown')}' runtime artifact is missing at '{artifact_path}'."
        )

    if serializer == "pickle-gzip":
        with gzip.open(resolved_path, "rb") as artifact_file:
            return pickle.load(artifact_file)

    with resolved_path.open("rb") as artifact_file:
        return pickle.load(artifact_file)


def classify_shipped_classical_artifact(
    *,
    metadata: dict[str, object],
    flat_pixels: list[float],
    storage_root: Path,
) -> tuple[int, list[float]]:
    estimator = load_runtime_artifact(metadata=metadata, storage_root=storage_root)
    predicted_digit = int(estimator.predict([flat_pixels])[0])

    confidences = build_estimator_confidences(estimator=estimator, flat_pixels=flat_pixels)

    total_confidence = sum(confidences)
    if total_confidence <= 0:
        confidences[predicted_digit] = 1.0
    elif abs(total_confidence - 1.0) > 1e-6:
        confidences = [value / total_confidence for value in confidences]

    return predicted_digit, confidences


def build_estimator_confidences(*, estimator, flat_pixels: list[float]) -> list[float]:
    if hasattr(estimator, "predict_proba") and hasattr(estimator, "classes_"):
        probabilities = estimator.predict_proba([flat_pixels])[0]
        return expand_class_confidences(
            class_labels=estimator.classes_,
            values=probabilities,
            apply_softmax=False,
        )

    if hasattr(estimator, "decision_function") and hasattr(estimator, "classes_"):
        decision_values = estimator.decision_function([flat_pixels])[0]
        if hasattr(decision_values, "tolist"):
            decision_values = decision_values.tolist()

        if not isinstance(decision_values, list):
            decision_score = float(decision_values)
            if len(estimator.classes_) == 2:
                decision_values = [-decision_score, decision_score]
            else:
                decision_values = [decision_score]

        return expand_class_confidences(
            class_labels=estimator.classes_,
            values=decision_values,
            apply_softmax=True,
        )

    predicted_digit = int(estimator.predict([flat_pixels])[0])
    confidences = [0.0] * 10
    confidences[predicted_digit] = 1.0
    return confidences


def expand_class_confidences(*, class_labels, values, apply_softmax: bool) -> list[float]:
    numeric_values = [float(value) for value in values]
    if apply_softmax:
        numeric_values = softmax(numeric_values)

    confidences = [0.0] * 10
    for class_label, value in zip(class_labels, numeric_values):
        digit = int(class_label)
        if 0 <= digit < len(confidences):
            confidences[digit] = float(value)

    return confidences


def shipped_model_entries() -> list[dict[str, str]]:
    return [
        built_in_entry(REFERENCE_MODEL_ID),
        *[built_in_entry(model_id) for model_id in SHIPPED_CLASSICAL_MODEL_IDS],
    ]


def is_shipped_classical_model(model_id: str) -> bool:
    return model_id in SHIPPED_CLASSICAL_MODEL_IDS


def ensure_shipped_model_artifact(models_root: Path, model_id: str) -> dict[str, object]:
    models_root.mkdir(parents=True, exist_ok=True)

    if model_id == REFERENCE_MODEL_ID:
        return ensure_reference_model_artifact(models_root)

    if not is_shipped_classical_model(model_id):
        raise KeyError(f"Unknown shipped model '{model_id}'.")

    storage_root = models_root.parent
    metadata_path = models_root / f"{model_id}.json"
    if not metadata_path.exists():
        materialize_canonical_shipped_model(storage_root, model_id)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_shipped_model_metadata(
        metadata=metadata,
        model_id=model_id,
        storage_root=storage_root,
    )

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
    flat_pixels = flatten(processed_canvas)
    predicted_digit, confidences = classify_shipped_classical_artifact(
        metadata=metadata,
        flat_pixels=flat_pixels,
        storage_root=storage_root,
    )

    return {
        "model": to_public_shipped_model_metadata(metadata),
        "prediction": {
            "digit": predicted_digit,
            "confidences": confidences,
        },
    }
