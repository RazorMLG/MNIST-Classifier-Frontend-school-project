import json
from pathlib import Path

from backend.app.reference_model import flatten, render_digit
from backend.app.shipped_classical_training import regenerate_shipped_classical_models


def make_training_csv(samples_per_digit: int = 12) -> str:
    header = ["label", *[f"pixel{index}" for index in range(784)]]
    rows = [",".join(header)]

    for label in range(10):
        prototype_pixels = [str(round(value * 255)) for value in flatten(render_digit(label))]

        for _ in range(samples_per_digit):
            rows.append(",".join([str(label), *prototype_pixels]))

    return "\n".join(rows)


def test_regenerate_shipped_models_writes_knn_artifact_and_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text(make_training_csv(), encoding="utf-8")

    generated_models = regenerate_shipped_classical_models(
        dataset_path=dataset_path,
        storage_root=tmp_path,
        model_ids=["knn-classifier-v1"],
        seed=17,
    )

    assert [model["id"] for model in generated_models] == ["knn-classifier-v1"]

    metadata_path = tmp_path / "shipped-models" / "knn-classifier-v1.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["dataset"]["source"] == "train.csv"
    assert metadata["training"]["split_seed"] == 17
    assert metadata["training"]["split_counts"] == {
        "train": 96,
        "validation": 12,
        "test": 12,
    }
    assert metadata["artifact"]["version"] == 1
    assert metadata["artifact"]["serializer"] == "pickle-gzip"
    assert metadata["artifact"]["path"] == "shipped-models/knn-classifier-v1.pkl.gz"
    assert (tmp_path / metadata["artifact"]["path"]).is_file()
    assert metadata["metrics"]["accuracy"] >= 0.0


def test_regenerate_shipped_models_writes_svm_artifact_and_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text(make_training_csv(), encoding="utf-8")

    generated_models = regenerate_shipped_classical_models(
        dataset_path=dataset_path,
        storage_root=tmp_path,
        model_ids=["svm-classifier-v1"],
        seed=19,
    )

    assert [model["id"] for model in generated_models] == ["svm-classifier-v1"]

    metadata_path = tmp_path / "shipped-models" / "svm-classifier-v1.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["family"] == "svm"
    assert metadata["training"]["split_seed"] == 19
    assert metadata["artifact"]["path"] == "shipped-models/svm-classifier-v1.pkl.gz"
    assert (tmp_path / metadata["artifact"]["path"]).is_file()
    assert metadata["metrics"]["accuracy"] >= 0.0


def test_regenerate_shipped_models_writes_random_forest_artifact_and_metadata(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text(make_training_csv(), encoding="utf-8")

    generated_models = regenerate_shipped_classical_models(
        dataset_path=dataset_path,
        storage_root=tmp_path,
        model_ids=["random-forest-classifier-v1"],
        seed=23,
    )

    assert [model["id"] for model in generated_models] == ["random-forest-classifier-v1"]

    metadata_path = tmp_path / "shipped-models" / "random-forest-classifier-v1.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["family"] == "random-forest"
    assert metadata["training"]["split_seed"] == 23
    assert metadata["artifact"]["path"] == "shipped-models/random-forest-classifier-v1.pkl.gz"
    assert (tmp_path / metadata["artifact"]["path"]).is_file()
    assert metadata["metrics"]["accuracy"] >= 0.0