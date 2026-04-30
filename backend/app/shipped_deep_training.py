from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.app.custom_training import calculate_metrics
from backend.app.reference_model import (
    CONTENT_IMAGE_SIZE,
    TARGET_IMAGE_SIZE,
    preprocess_mnist_pixels,
)
from backend.app.shipped_classical_models import build_cnn_module, build_mlp_module
from backend.app.shipped_classical_training import (
    DEFAULT_SPLIT_RATIOS,
    now_timestamp,
    split_dataset,
    write_metadata,
)
from backend.app.training_csv import parse_training_csv_rows

ARTIFACT_VERSION = 1

MODEL_SPECS = {
    "mlp-classifier-v1": {
        "name": "MLP Classifier",
        "family": "mlp",
        "version": "1.0.0",
        "description": "Shipped multilayer perceptron classifier trained from the repository train.csv.",
        "hidden_layers": [128, 64],
        "epochs": 4,
        "batch_size": 32,
        "learning_rate": 0.002,
    },
    "cnn-classifier-v1": {
        "name": "CNN Classifier",
        "family": "cnn",
        "version": "1.0.0",
        "description": "Shipped convolutional neural network classifier trained from the repository train.csv.",
        "conv_channels": [8, 16],
        "dense_units": 32,
        "epochs": 4,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
}


def regenerate_shipped_deep_models(
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
            raise KeyError(f"Unknown shipped deep model '{model_id}'.")

        model_spec = MODEL_SPECS[model_id]
        if model_spec["family"] == "mlp":
            module, epoch_curves = train_mlp_model(
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                hidden_layers=model_spec["hidden_layers"],
                epochs=model_spec["epochs"],
                batch_size=model_spec["batch_size"],
                learning_rate=model_spec["learning_rate"],
                seed=seed,
            )
            hyperparameters = {
                "epochs": model_spec["epochs"],
                "batch_size": model_spec["batch_size"],
                "learning_rate": model_spec["learning_rate"],
                "hidden_layers": "128 -> 64",
                "target_size": TARGET_IMAGE_SIZE,
                "content_box": CONTENT_IMAGE_SIZE,
                "centering": "center-of-mass",
            }
            architecture_summary = [
                "Flatten -> Linear(784, 128) -> ReLU",
                "Linear(128, 64) -> ReLU",
                "Linear(64, 10)",
            ]
            checkpoint_payload = {
                "version": ARTIFACT_VERSION,
                "family": "mlp",
                "input_size": TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE,
                "hidden_layers": model_spec["hidden_layers"],
                "state_dict": module.state_dict(),
            }
        else:
            module, epoch_curves = train_cnn_model(
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                conv_channels=model_spec["conv_channels"],
                dense_units=model_spec["dense_units"],
                epochs=model_spec["epochs"],
                batch_size=model_spec["batch_size"],
                learning_rate=model_spec["learning_rate"],
                seed=seed,
            )
            hyperparameters = {
                "epochs": model_spec["epochs"],
                "batch_size": model_spec["batch_size"],
                "learning_rate": model_spec["learning_rate"],
                "conv_channels": "8 -> 16",
                "dense_units": model_spec["dense_units"],
                "target_size": TARGET_IMAGE_SIZE,
                "content_box": CONTENT_IMAGE_SIZE,
                "centering": "center-of-mass",
            }
            architecture_summary = [
                "Conv(1, 8, 3) -> ReLU -> MaxPool",
                "Conv(8, 16, 3) -> ReLU -> MaxPool",
                "Flatten -> Linear(784, 32) -> ReLU -> Linear(32, 10)",
            ]
            checkpoint_payload = {
                "version": ARTIFACT_VERSION,
                "family": "cnn",
                "conv_channels": model_spec["conv_channels"],
                "dense_units": model_spec["dense_units"],
                "state_dict": module.state_dict(),
            }

        metrics, evaluation_matrix, sample_predictions = evaluate_module(
            module=module,
            evaluation_features=test_features or validation_features,
            evaluation_labels=test_labels or validation_labels,
            batch_size=model_spec["batch_size"],
        )

        artifact_path = f"shipped-models/{model_id}.pt"
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
            "hyperparameters": hyperparameters,
            "deep_details": {
                "architecture_summary": architecture_summary,
                "epoch_curves": epoch_curves,
            },
            "evaluation": {
                "confusion_matrix": evaluation_matrix,
                "sample_predictions": sample_predictions,
            },
            "artifact": {
                "version": ARTIFACT_VERSION,
                "serializer": "torch",
                "estimator": "torch.nn.Module",
                "path": artifact_path,
            },
        }

        torch.save(checkpoint_payload, shipped_models_root / f"{model_id}.pt")
        write_metadata(shipped_models_root / f"{model_id}.json", metadata)
        generated_models.append(metadata)

    return generated_models


def train_mlp_model(
    *,
    train_features: list[list[float]],
    train_labels: list[int],
    validation_features: list[list[float]],
    validation_labels: list[int],
    hidden_layers: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    torch.manual_seed(seed)

    module = build_mlp_module(
        input_size=TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE,
        hidden_layers=hidden_layers,
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = build_loader(train_features, train_labels, batch_size=batch_size, shuffle=True)
    validation_loader = build_loader(
        validation_features,
        validation_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    epoch_curves: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        module.train()
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = module(batch_inputs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate_loader(
            module=module,
            loader=train_loader,
            loss_fn=loss_fn,
        )
        validation_loss, validation_accuracy = evaluate_loader(
            module=module,
            loader=validation_loader,
            loss_fn=loss_fn,
        )
        epoch_curves.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "validation_loss": round(validation_loss, 4),
                "train_accuracy": round(train_accuracy, 4),
                "validation_accuracy": round(validation_accuracy, 4),
            }
        )

    module.eval()
    return module, epoch_curves


def train_cnn_model(
    *,
    train_features: list[list[float]],
    train_labels: list[int],
    validation_features: list[list[float]],
    validation_labels: list[int],
    conv_channels: list[int],
    dense_units: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    torch.manual_seed(seed)

    module = build_cnn_module(conv_channels=conv_channels, dense_units=dense_units)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = build_loader(train_features, train_labels, batch_size=batch_size, shuffle=True)
    validation_loader = build_loader(
        validation_features,
        validation_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    epoch_curves: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        module.train()
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = module(batch_inputs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate_loader(
            module=module,
            loader=train_loader,
            loss_fn=loss_fn,
        )
        validation_loss, validation_accuracy = evaluate_loader(
            module=module,
            loader=validation_loader,
            loss_fn=loss_fn,
        )
        epoch_curves.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "validation_loss": round(validation_loss, 4),
                "train_accuracy": round(train_accuracy, 4),
                "validation_accuracy": round(validation_accuracy, 4),
            }
        )

    module.eval()
    return module, epoch_curves


def evaluate_module(
    *,
    module,
    evaluation_features: list[list[float]],
    evaluation_labels: list[int],
    batch_size: int,
) -> tuple[dict[str, float], list[list[int]], list[dict[str, object]]]:
    if not evaluation_labels:
        raise ValueError("Shipped deep training requires at least one evaluation example.")

    loader = build_loader(evaluation_features, evaluation_labels, batch_size=batch_size, shuffle=False)
    predicted_digits: list[int] = []
    confidence_rows: list[list[float]] = []
    inference_times: list[float] = []

    module.eval()
    with torch.no_grad():
        for batch_inputs, _batch_labels in loader:
            started_at = time.perf_counter()
            logits = module(batch_inputs)
            inference_times.append(((time.perf_counter() - started_at) * 1000) / max(len(batch_inputs), 1))
            probabilities = torch.softmax(logits, dim=1)
            predicted_digits.extend(int(digit) for digit in torch.argmax(probabilities, dim=1).tolist())
            confidence_rows.extend(probabilities.tolist())

    evaluation_matrix = confusion_matrix(
        evaluation_labels,
        predicted_digits,
        labels=list(range(10)),
    ).tolist()

    sample_predictions = [
        {
            "label": int(label),
            "predicted": int(predicted_digit),
            "confidence": round(max(confidences), 4),
        }
        for label, predicted_digit, confidences in zip(
            evaluation_labels[:4],
            predicted_digits[:4],
            confidence_rows[:4],
        )
    ]

    return calculate_metrics(evaluation_matrix, inference_times), evaluation_matrix, sample_predictions


def evaluate_loader(*, module, loader: DataLoader, loss_fn) -> tuple[float, float]:
    if len(loader.dataset) == 0:
        raise ValueError("Deep training requires at least one dataset example.")

    module.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            logits = module(batch_inputs)
            loss = loss_fn(logits, batch_labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = len(batch_labels)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == batch_labels).sum().item())
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def build_loader(
    features: list[list[float]],
    labels: list[int],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    feature_tensor = torch.tensor(features, dtype=torch.float32).view(-1, 1, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(feature_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate shipped deep MNIST models.")
    parser.add_argument("--dataset", required=True, help="Path to the labeled training CSV.")
    parser.add_argument("--storage-root", required=True, help="Directory where shipped model artifacts are written.")
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Optional shipped deep model id to regenerate. Repeat to select multiple models.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Deterministic split and training seed.")
    args = parser.parse_args()

    regenerate_shipped_deep_models(
        dataset_path=Path(args.dataset).resolve(),
        storage_root=Path(args.storage_root).resolve(),
        model_ids=args.model_ids,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()