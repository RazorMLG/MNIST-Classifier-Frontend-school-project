from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app


def make_training_csv(example_count: int = 10) -> str:
    header = ["label", *[f"pixel{index}" for index in range(784)]]
    rows = [",".join(header)]

    for label in range(example_count):
        pixel_values = [str((label + offset) % 256) for offset in range(784)]
        rows.append(",".join([str(label % 10), *pixel_values]))

    return "\n".join(rows)


def test_csv_preview_reports_default_split_counts(tmp_path: Path) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.post(
            "/api/training/csv-preview",
            json={
                "file_name": "train.csv",
                "csv_text": make_training_csv(),
                "split": {
                    "train_ratio": 0.8,
                    "validation_ratio": 0.1,
                    "test_ratio": 0.1,
                },
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["file_name"] == "train.csv"
    assert body["dataset"]["example_count"] == 10
    assert body["dataset"]["label_range"] == {"min": 0, "max": 9}
    assert body["split"]["counts"] == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }


def test_csv_preview_rejects_incompatible_schema(tmp_path: Path) -> None:
    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.post(
            "/api/training/csv-preview",
            json={
                "file_name": "test.csv",
                "csv_text": "pixel0,pixel1\n0,1",
                "split": {
                    "train_ratio": 0.8,
                    "validation_ratio": 0.1,
                    "test_ratio": 0.1,
                },
            },
        )

    assert response.status_code == 400
    assert "labeled MNIST training format" in response.json()["detail"]


def test_csv_preview_rejects_labels_outside_digit_range(tmp_path: Path) -> None:
    csv_text = make_training_csv(example_count=1).replace("0,0,1,2", "12,0,1,2", 1)

    with TestClient(create_app(storage_root=tmp_path)) as client:
        response = client.post(
            "/api/training/csv-preview",
            json={
                "file_name": "train.csv",
                "csv_text": csv_text,
                "split": {
                    "train_ratio": 0.8,
                    "validation_ratio": 0.1,
                    "test_ratio": 0.1,
                },
            },
        )

    assert response.status_code == 400
    assert "label must be between 0 and 9" in response.json()["detail"]