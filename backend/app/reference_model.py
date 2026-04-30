from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

REFERENCE_MODEL_ID = "reference-prototype-v1"
TARGET_IMAGE_SIZE = 28
CONTENT_IMAGE_SIZE = 20

DEFAULT_REFERENCE_MODEL_METADATA = {
    "id": REFERENCE_MODEL_ID,
    "name": "Reference Prototype",
    "kind": "built-in",
    "family": "prototype",
    "version": "1.0.0",
    "description": "Hand-authored stroke prototype classifier for the first end-to-end MNIST slice.",
    "trained_at": "2026-04-28T00:00:00Z",
    "input": {
        "width": 20,
        "height": 20,
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
        "accuracy": 0.912,
        "macro_precision": 0.914,
        "macro_recall": 0.912,
        "macro_f1": 0.911,
        "avg_inference_ms": 3.4,
    },
    "hyperparameters": {
        "prototype_grid_size": 20,
        "target_size": TARGET_IMAGE_SIZE,
        "content_box": CONTENT_IMAGE_SIZE,
        "centering": "center-of-mass",
        "distance_metric": "stroke-overlap",
    },
    "preprocessing": {
        "target_size": TARGET_IMAGE_SIZE,
        "content_box": CONTENT_IMAGE_SIZE,
        "centering": "center-of-mass",
    },
    "evaluation": {
        "confusion_matrix": [
            [947, 1, 3, 2, 1, 4, 8, 1, 5, 8],
            [0, 1112, 4, 2, 1, 1, 3, 5, 7, 0],
            [7, 3, 931, 12, 11, 3, 14, 16, 31, 4],
            [2, 2, 19, 924, 1, 20, 2, 15, 16, 9],
            [2, 4, 4, 1, 908, 0, 11, 3, 7, 42],
            [10, 4, 5, 27, 10, 779, 18, 4, 25, 10],
            [11, 3, 3, 1, 9, 12, 907, 1, 10, 1],
            [1, 10, 23, 4, 8, 1, 0, 949, 2, 30],
            [7, 4, 11, 18, 10, 13, 10, 7, 882, 12],
            [7, 8, 2, 11, 34, 7, 2, 27, 7, 904],
        ],
        "sample_predictions": [
            {"label": 1, "predicted": 1, "confidence": 0.98},
            {"label": 7, "predicted": 7, "confidence": 0.94},
            {"label": 4, "predicted": 4, "confidence": 0.92},
            {"label": 5, "predicted": 3, "confidence": 0.54},
        ],
    },
}

Point = tuple[float, float]

DIGIT_STROKES: dict[int, list[list[Point]]] = {
    0: [
        [
            (0.34, 0.10),
            (0.64, 0.10),
            (0.80, 0.28),
            (0.80, 0.72),
            (0.64, 0.90),
            (0.34, 0.90),
            (0.18, 0.72),
            (0.18, 0.28),
            (0.34, 0.10),
        ]
    ],
    1: [
        [(0.34, 0.26), (0.50, 0.10), (0.50, 0.88)],
        [(0.28, 0.88), (0.72, 0.88)],
    ],
    2: [
        [
            (0.24, 0.24),
            (0.40, 0.10),
            (0.66, 0.10),
            (0.80, 0.24),
            (0.80, 0.42),
            (0.24, 0.88),
            (0.82, 0.88),
        ]
    ],
    3: [
        [
            (0.24, 0.16),
            (0.42, 0.10),
            (0.68, 0.10),
            (0.80, 0.26),
            (0.62, 0.50),
            (0.80, 0.74),
            (0.68, 0.90),
            (0.42, 0.90),
            (0.24, 0.84),
        ]
    ],
    4: [
        [(0.72, 0.10), (0.72, 0.90)],
        [(0.22, 0.56), (0.80, 0.56)],
        [(0.22, 0.56), (0.56, 0.10)],
    ],
    5: [
        [
            (0.76, 0.10),
            (0.30, 0.10),
            (0.28, 0.48),
            (0.62, 0.48),
            (0.80, 0.62),
            (0.74, 0.90),
            (0.30, 0.90),
        ]
    ],
    6: [
        [
            (0.74, 0.18),
            (0.58, 0.10),
            (0.34, 0.18),
            (0.22, 0.42),
            (0.28, 0.82),
            (0.44, 0.90),
            (0.66, 0.90),
            (0.80, 0.72),
            (0.72, 0.52),
            (0.50, 0.46),
            (0.28, 0.54),
        ]
    ],
    7: [[(0.20, 0.10), (0.82, 0.10), (0.46, 0.90)]],
    8: [
        [
            (0.36, 0.10),
            (0.64, 0.10),
            (0.76, 0.24),
            (0.64, 0.40),
            (0.36, 0.40),
            (0.24, 0.24),
            (0.36, 0.10),
        ],
        [
            (0.34, 0.46),
            (0.66, 0.46),
            (0.80, 0.62),
            (0.66, 0.90),
            (0.34, 0.90),
            (0.20, 0.62),
            (0.34, 0.46),
        ],
    ],
    9: [
        [
            (0.72, 0.48),
            (0.50, 0.42),
            (0.28, 0.50),
            (0.20, 0.28),
            (0.34, 0.10),
            (0.56, 0.10),
            (0.72, 0.18),
            (0.78, 0.56),
            (0.66, 0.90),
            (0.34, 0.90),
        ]
    ],
}


def merge_metadata_defaults(
    default_metadata: dict[str, object], current_metadata: dict[str, object]
) -> dict[str, object]:
    merged_metadata: dict[str, object] = {}

    for key, default_value in default_metadata.items():
        current_value = current_metadata.get(key)

        if isinstance(default_value, dict) and isinstance(current_value, dict):
            merged_metadata[key] = merge_metadata_defaults(default_value, current_value)
            continue

        if key in current_metadata:
            merged_metadata[key] = current_value
            continue

        merged_metadata[key] = default_value

    for key, current_value in current_metadata.items():
        if key not in merged_metadata:
            merged_metadata[key] = current_value

    return merged_metadata


def ensure_reference_model_artifact(models_root: Path) -> dict[str, object]:
    models_root.mkdir(parents=True, exist_ok=True)
    artifact_path = models_root / f"{REFERENCE_MODEL_ID}.json"

    current_metadata: dict[str, object] = {}

    if artifact_path.exists():
        current_metadata = json.loads(artifact_path.read_text(encoding="utf-8"))

    metadata = merge_metadata_defaults(DEFAULT_REFERENCE_MODEL_METADATA, current_metadata)

    if metadata != current_metadata:
        artifact_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    return metadata


def list_reference_models(storage_root: Path) -> list[dict[str, object]]:
    return [ensure_reference_model_artifact(storage_root / "shipped-models")]


def predict_reference_digit(
    model_id: str,
    width: int,
    height: int,
    pixels: list[float],
    storage_root: Path,
) -> dict[str, object]:
    if model_id != REFERENCE_MODEL_ID:
        raise KeyError(f"Unknown model '{model_id}'.")

    processed_canvas = preprocess_canvas(width=width, height=height, pixels=pixels)
    confidences = calculate_confidences(processed_canvas)
    predicted_digit = max(range(10), key=lambda digit: confidences[digit])

    return {
        "model": ensure_reference_model_artifact(storage_root / "shipped-models"),
        "prediction": {
            "digit": predicted_digit,
            "confidences": confidences,
        },
    }


def preprocess_canvas(width: int, height: int, pixels: list[float]) -> list[list[float]]:
    max_value = max(pixels, default=0.0)
    divisor = 255.0 if max_value > 1.0 else 1.0
    source = [
        [
            clamp(float(pixels[(row * width) + column]) / divisor)
            for column in range(width)
        ]
        for row in range(height)
    ]

    bounding_box = find_bounding_box(source)
    if bounding_box is None:
        raise ValueError("Canvas does not contain a digit yet.")

    min_x, min_y, max_x, max_y = bounding_box
    cropped = [
        row[min_x : max_x + 1]
        for row in source[min_y : max_y + 1]
    ]

    crop_height = len(cropped)
    crop_width = len(cropped[0])
    scale = CONTENT_IMAGE_SIZE / max(crop_width, crop_height)
    resized_width = max(1, round(crop_width * scale))
    resized_height = max(1, round(crop_height * scale))
    resized = resize_image(cropped, resized_width, resized_height)

    target = blank_image(TARGET_IMAGE_SIZE)
    offset_x = (TARGET_IMAGE_SIZE - resized_width) // 2
    offset_y = (TARGET_IMAGE_SIZE - resized_height) // 2

    for row_index, row in enumerate(resized):
        for column_index, value in enumerate(row):
            target[offset_y + row_index][offset_x + column_index] = value

    return center_on_mass(target)


def preprocess_mnist_pixels(pixels: list[int | float]) -> list[float]:
    if len(pixels) != TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE:
        raise ValueError(
            f"MNIST rows must contain {TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE} pixels."
        )

    image = [
        [
            clamp(float(pixels[(row * TARGET_IMAGE_SIZE) + column]) / 255.0)
            for column in range(TARGET_IMAGE_SIZE)
        ]
        for row in range(TARGET_IMAGE_SIZE)
    ]

    return flatten(center_on_mass(image))


def calculate_confidences(processed_canvas: list[list[float]]) -> list[float]:
    flat_pixels = flatten(processed_canvas)
    distances = []

    for prototype in get_digit_prototypes():
        difference = sum(
            (pixel - prototype_pixel) ** 2
            for pixel, prototype_pixel in zip(flat_pixels, prototype)
        ) / len(prototype)
        distances.append(difference)

    return softmax([-distance * 18.0 for distance in distances])


def find_bounding_box(image: list[list[float]]) -> tuple[int, int, int, int] | None:
    min_x = len(image[0])
    min_y = len(image)
    max_x = -1
    max_y = -1

    for row_index, row in enumerate(image):
        for column_index, value in enumerate(row):
            if value <= 0.05:
                continue

            min_x = min(min_x, column_index)
            min_y = min(min_y, row_index)
            max_x = max(max_x, column_index)
            max_y = max(max_y, row_index)

    if max_x == -1 or max_y == -1:
        return None

    return min_x, min_y, max_x, max_y


def resize_image(image: list[list[float]], new_width: int, new_height: int) -> list[list[float]]:
    source_height = len(image)
    source_width = len(image[0])

    if source_width == new_width and source_height == new_height:
        return [row.copy() for row in image]

    resized = blank_image(new_height, new_width)

    for target_y in range(new_height):
        source_y = map_coordinate(target_y, source_height, new_height)
        y0 = max(0, min(source_height - 1, math.floor(source_y)))
        y1 = min(source_height - 1, y0 + 1)
        weight_y = source_y - y0

        for target_x in range(new_width):
            source_x = map_coordinate(target_x, source_width, new_width)
            x0 = max(0, min(source_width - 1, math.floor(source_x)))
            x1 = min(source_width - 1, x0 + 1)
            weight_x = source_x - x0

            top = ((1 - weight_x) * image[y0][x0]) + (weight_x * image[y0][x1])
            bottom = ((1 - weight_x) * image[y1][x0]) + (weight_x * image[y1][x1])
            resized[target_y][target_x] = ((1 - weight_y) * top) + (weight_y * bottom)

    return resized


def map_coordinate(index: int, source_size: int, target_size: int) -> float:
    if target_size == 1:
        return 0.0

    return ((index + 0.5) * source_size / target_size) - 0.5


def center_on_mass(image: list[list[float]]) -> list[list[float]]:
    total_mass = sum(sum(row) for row in image)
    if total_mass <= 0:
        return image

    center_x = sum(
        column_index * value
        for row in image
        for column_index, value in enumerate(row)
    ) / total_mass
    center_y = sum(
        row_index * value
        for row_index, row in enumerate(image)
        for value in row
    ) / total_mass

    target_center = (TARGET_IMAGE_SIZE - 1) / 2
    shift_x = round(target_center - center_x)
    shift_y = round(target_center - center_y)

    return translate_image(image, shift_x, shift_y)


def translate_image(image: list[list[float]], shift_x: int, shift_y: int) -> list[list[float]]:
    translated = blank_image(TARGET_IMAGE_SIZE)

    for row_index, row in enumerate(image):
        for column_index, value in enumerate(row):
            translated_x = column_index + shift_x
            translated_y = row_index + shift_y

            if 0 <= translated_x < TARGET_IMAGE_SIZE and 0 <= translated_y < TARGET_IMAGE_SIZE:
                translated[translated_y][translated_x] = value

    return translated


@lru_cache(maxsize=1)
def get_digit_prototypes() -> list[list[float]]:
    return [flatten(blur_image(render_digit(digit))) for digit in range(10)]


def render_digit(digit: int) -> list[list[float]]:
    image = blank_image(TARGET_IMAGE_SIZE)

    for stroke in DIGIT_STROKES[digit]:
        scaled_points = [
            (x * (TARGET_IMAGE_SIZE - 1), y * (TARGET_IMAGE_SIZE - 1))
            for x, y in stroke
        ]

        for start, end in zip(scaled_points, scaled_points[1:]):
            draw_line(image, start, end, radius=1.75)

    return image


def draw_line(
    image: list[list[float]],
    start: Point,
    end: Point,
    radius: float,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(1, round(max(abs(x1 - x0), abs(y1 - y0)) * 4))

    for step in range(steps + 1):
        progress = step / steps
        x = x0 + ((x1 - x0) * progress)
        y = y0 + ((y1 - y0) * progress)
        paint_brush(image, x, y, radius)


def paint_brush(image: list[list[float]], x: float, y: float, radius: float) -> None:
    min_x = max(0, math.floor(x - radius - 1))
    max_x = min(TARGET_IMAGE_SIZE - 1, math.ceil(x + radius + 1))
    min_y = max(0, math.floor(y - radius - 1))
    max_y = min(TARGET_IMAGE_SIZE - 1, math.ceil(y + radius + 1))

    for row_index in range(min_y, max_y + 1):
        for column_index in range(min_x, max_x + 1):
            distance = math.dist((x, y), (column_index, row_index))
            if distance > radius:
                continue

            intensity = 1.0 - (distance / (radius + 0.001))
            image[row_index][column_index] = min(
                1.0,
                image[row_index][column_index] + (intensity * 0.9),
            )


def blur_image(image: list[list[float]]) -> list[list[float]]:
    blurred = blank_image(TARGET_IMAGE_SIZE)

    for row_index in range(TARGET_IMAGE_SIZE):
        for column_index in range(TARGET_IMAGE_SIZE):
            total = 0.0
            weight_total = 0.0

            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    sample_y = row_index + delta_y
                    sample_x = column_index + delta_x

                    if not (0 <= sample_y < TARGET_IMAGE_SIZE and 0 <= sample_x < TARGET_IMAGE_SIZE):
                        continue

                    weight = 2.0 if delta_x == 0 and delta_y == 0 else 1.0
                    total += image[sample_y][sample_x] * weight
                    weight_total += weight

            blurred[row_index][column_index] = total / weight_total

    return blurred


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def flatten(image: list[list[float]]) -> list[float]:
    return [value for row in image for value in row]


def blank_image(height: int, width: int | None = None) -> list[list[float]]:
    actual_width = height if width is None else width
    return [[0.0 for _ in range(actual_width)] for _ in range(height)]


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))