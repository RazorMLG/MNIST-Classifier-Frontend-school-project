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
SHIPPED_DEEP_MODEL_IDS = (
    "mlp-classifier-v1",
    "cnn-classifier-v1",
)
SUPPORTED_ARTIFACT_SERIALIZERS = {"pickle", "pickle-gzip", "torch"}
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

    if serializer == "torch":
        import torch

        return torch.load(resolved_path, map_location="cpu")

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


def build_mlp_module(*, input_size: int, hidden_layers: list[int]):
    from torch import nn

    layers: list[nn.Module] = [nn.Flatten()]
    current_size = input_size
    for hidden_layer in hidden_layers:
        layers.append(nn.Linear(current_size, hidden_layer))
        layers.append(nn.ReLU())
        current_size = hidden_layer

    layers.append(nn.Linear(current_size, 10))
    return nn.Sequential(*layers)


def build_cnn_module(*, conv_channels: list[int], dense_units: int):
    from torch import nn

    layers: list[nn.Module] = []
    input_channels = 1
    spatial_size = 28
    for output_channels in conv_channels:
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2))
        input_channels = output_channels
        spatial_size //= 2

    layers.append(nn.Flatten())
    layers.append(nn.Linear(input_channels * spatial_size * spatial_size, dense_units))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(dense_units, 10))
    return nn.Sequential(*layers)


def build_deep_module(
    *,
    metadata: dict[str, object],
    artifact_checkpoint: dict[str, object],
):
    family = str(artifact_checkpoint.get("family") or metadata.get("family") or "").strip()
    state_dict = artifact_checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Shipped model '{metadata.get('id', 'unknown')}' is missing torch state_dict data."
        )

    if family == "mlp":
        hidden_layers = artifact_checkpoint.get("hidden_layers", [128, 64])
        if not isinstance(hidden_layers, list) or not hidden_layers:
            raise ValueError(
                f"Shipped model '{metadata.get('id', 'unknown')}' is missing MLP hidden layer configuration."
            )

        module = build_mlp_module(
            input_size=int(artifact_checkpoint.get("input_size", 28 * 28)),
            hidden_layers=[int(hidden_layer) for hidden_layer in hidden_layers],
        )
    elif family == "cnn":
        conv_channels = artifact_checkpoint.get("conv_channels", [8, 16])
        if not isinstance(conv_channels, list) or not conv_channels:
            raise ValueError(
                f"Shipped model '{metadata.get('id', 'unknown')}' is missing CNN channel configuration."
            )

        module = build_cnn_module(
            conv_channels=[int(channel) for channel in conv_channels],
            dense_units=int(artifact_checkpoint.get("dense_units", 32)),
        )
    else:
        raise ValueError(
            f"Shipped model '{metadata.get('id', 'unknown')}' uses unsupported deep family '{family}'."
        )

    module.load_state_dict(state_dict)
    module.eval()
    return module


def classify_shipped_deep_artifact(
    *,
    metadata: dict[str, object],
    processed_canvas: list[list[float]],
    storage_root: Path,
) -> tuple[int, list[float]]:
    import torch

    artifact_checkpoint = load_runtime_artifact(metadata=metadata, storage_root=storage_root)
    if not isinstance(artifact_checkpoint, dict):
        raise ValueError(
            f"Shipped model '{metadata.get('id', 'unknown')}' uses invalid torch artifact data."
        )

    module = build_deep_module(metadata=metadata, artifact_checkpoint=artifact_checkpoint)
    input_tensor = torch.tensor(processed_canvas, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = module(input_tensor)

    logits_row = logits[0].tolist()
    confidences = softmax([float(value) for value in logits_row])
    predicted_digit = max(range(10), key=lambda digit: confidences[digit])
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
        *[built_in_entry(model_id) for model_id in SHIPPED_DEEP_MODEL_IDS],
    ]


def is_shipped_classical_model(model_id: str) -> bool:
    return model_id in SHIPPED_CLASSICAL_MODEL_IDS


def is_shipped_deep_model(model_id: str) -> bool:
    return model_id in SHIPPED_DEEP_MODEL_IDS


def ensure_shipped_model_artifact(models_root: Path, model_id: str) -> dict[str, object]:
    models_root.mkdir(parents=True, exist_ok=True)

    if model_id == REFERENCE_MODEL_ID:
        return ensure_reference_model_artifact(models_root)

    if not is_shipped_classical_model(model_id) and not is_shipped_deep_model(model_id):
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


def predict_shipped_deep_digit(
    *,
    model_id: str,
    width: int,
    height: int,
    pixels: list[float],
    storage_root: Path,
) -> dict[str, object]:
    if not is_shipped_deep_model(model_id):
        raise KeyError(f"Unknown shipped model '{model_id}'.")

    metadata = ensure_shipped_model_artifact(storage_root / "shipped-models", model_id)
    processed_canvas = preprocess_canvas(width=width, height=height, pixels=pixels)
    predicted_digit, confidences = classify_shipped_deep_artifact(
        metadata=metadata,
        processed_canvas=processed_canvas,
        storage_root=storage_root,
    )

    return {
        "model": to_public_shipped_model_metadata(metadata),
        "prediction": {
            "digit": predicted_digit,
            "confidences": confidences,
        },
    }
