from __future__ import annotations

import csv
import math
from io import StringIO

EXPECTED_FEATURE_COUNT = 784
EXPECTED_HEADER = ["label", *[f"pixel{index}" for index in range(EXPECTED_FEATURE_COUNT)]]


def preview_training_csv(
    *,
    file_name: str,
    csv_text: str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, object]:
    example_count, label_range = validate_training_csv(csv_text)
    split_counts = calculate_split_counts(
        example_count=example_count,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    return {
        "file_name": file_name,
        "dataset": {
            "example_count": example_count,
            "feature_count": EXPECTED_FEATURE_COUNT,
            "label_range": label_range,
        },
        "split": {
            "ratios": {
                "train": train_ratio,
                "validation": validation_ratio,
                "test": test_ratio,
            },
            "counts": split_counts,
        },
    }


def validate_training_csv(csv_text: str) -> tuple[int, dict[str, int]]:
    reader = csv.reader(StringIO(csv_text.strip()))

    try:
        header = next(reader)
    except StopIteration as error:
        raise ValueError("CSV file is empty.") from error

    if header != EXPECTED_HEADER:
        raise ValueError(
            "CSV schema must match the labeled MNIST training format: label,pixel0,...,pixel783."
        )

    example_count = 0
    min_label = 9
    max_label = 0

    for row_index, row in enumerate(reader, start=2):
        if len(row) != len(EXPECTED_HEADER):
            raise ValueError(
                f"Row {row_index} has {len(row)} columns but expected {len(EXPECTED_HEADER)}."
            )

        try:
            label = int(row[0])
        except ValueError as error:
            raise ValueError(f"Row {row_index} has a non-integer label.") from error

        if not 0 <= label <= 9:
            raise ValueError(f"Row {row_index} label must be between 0 and 9.")

        for pixel_index, raw_value in enumerate(row[1:], start=0):
            try:
                pixel = int(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"Row {row_index} pixel{pixel_index} must be an integer between 0 and 255."
                ) from error

            if not 0 <= pixel <= 255:
                raise ValueError(
                    f"Row {row_index} pixel{pixel_index} must be an integer between 0 and 255."
                )

        example_count += 1
        min_label = min(min_label, label)
        max_label = max(max_label, label)

    if example_count == 0:
        raise ValueError("CSV file must include at least one labeled example.")

    return example_count, {"min": min_label, "max": max_label}


def calculate_split_counts(
    *,
    example_count: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    train_count = math.floor(example_count * train_ratio)
    validation_count = math.floor(example_count * validation_ratio)
    test_count = example_count - train_count - validation_count

    if test_count < 0:
        raise ValueError("Split ratios produce more assigned rows than the dataset contains.")

    return {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }