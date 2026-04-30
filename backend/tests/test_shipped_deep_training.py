import json
from pathlib import Path

from backend.app.reference_model import flatten, render_digit
from backend.app.shipped_deep_training import regenerate_shipped_deep_models


def make_training_csv(samples_per_digit: int = 12) -> str:
    header = ["label", *[f"pixel{index}" for index in range(784)]]
    rows = [",".join(header)]

    for label in range(10):
        prototype_pixels = [str(round(value * 255)) for value in flatten(render_digit(label))]

        for _ in range(samples_per_digit):
            rows.append(",".join([str(label), *prototype_pixels]))

    return "\n".join(rows)


def test_regenerate_shipped_models_writes_mlp_artifact_and_deep_metadata(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text(make_training_csv(), encoding="utf-8")

    generated_models = regenerate_shipped_deep_models(
        dataset_path=dataset_path,
        storage_root=tmp_path,
        model_ids=["mlp-classifier-v1"],
        seed=17,
    )

    assert [model["id"] for model in generated_models] == ["mlp-classifier-v1"]

    metadata_path = tmp_path / "shipped-models" / "mlp-classifier-v1.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["family"] == "mlp"
    assert metadata["training"]["split_seed"] == 17
    assert metadata["artifact"]["serializer"] == "torch"
    assert metadata["artifact"]["path"] == "shipped-models/mlp-classifier-v1.pt"
    assert (tmp_path / metadata["artifact"]["path"]).is_file()
    assert metadata["deep_details"]["architecture_summary"]
    assert metadata["deep_details"]["epoch_curves"]
    assert metadata["metrics"]["accuracy"] >= 0.0


def test_regenerate_shipped_models_writes_cnn_artifact_and_deep_metadata(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text(make_training_csv(), encoding="utf-8")

    generated_models = regenerate_shipped_deep_models(
        dataset_path=dataset_path,
        storage_root=tmp_path,
        model_ids=["cnn-classifier-v1"],
        seed=23,
    )

    assert [model["id"] for model in generated_models] == ["cnn-classifier-v1"]

    metadata_path = tmp_path / "shipped-models" / "cnn-classifier-v1.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["family"] == "cnn"
    assert metadata["training"]["split_seed"] == 23
    assert metadata["artifact"]["serializer"] == "torch"
    assert metadata["artifact"]["path"] == "shipped-models/cnn-classifier-v1.pt"
    assert (tmp_path / metadata["artifact"]["path"]).is_file()
    assert len(metadata["deep_details"]["architecture_summary"]) >= 3
    assert metadata["deep_details"]["epoch_curves"]
    assert metadata["metrics"]["accuracy"] >= 0.0