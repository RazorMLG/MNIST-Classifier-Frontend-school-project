from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from backend.app.reference_model import (
    ensure_reference_model_artifact,
    list_reference_models,
    predict_reference_digit,
)
from backend.app.training_csv import preview_training_csv

STORAGE_DIRECTORIES = ["shipped-models", "custom-models", "registry"]


class CanvasPayload(BaseModel):
    width: int = Field(ge=1, le=64)
    height: int = Field(ge=1, le=64)
    pixels: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pixels_length(self) -> "CanvasPayload":
        expected_pixels = self.width * self.height
        if len(self.pixels) != expected_pixels:
            raise ValueError(
                f"Canvas payload has {len(self.pixels)} pixels but expected {expected_pixels}."
            )

        return self


class PredictionRequest(BaseModel):
    model_id: str = Field(min_length=1)
    canvas: CanvasPayload


class TrainingSplitPayload(BaseModel):
    train_ratio: float = Field(ge=0, le=1)
    validation_ratio: float = Field(ge=0, le=1)
    test_ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_total_ratio(self) -> "TrainingSplitPayload":
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must add up to 1.0.")

        if self.train_ratio <= 0:
            raise ValueError("Training split must be greater than zero.")

        return self


class TrainingCsvPreviewRequest(BaseModel):
    file_name: str = Field(min_length=1)
    csv_text: str = Field(min_length=1)
    split: TrainingSplitPayload


def ensure_storage_structure(storage_root: Path) -> list[str]:
    storage_root.mkdir(parents=True, exist_ok=True)

    for directory in STORAGE_DIRECTORIES:
        (storage_root / directory).mkdir(parents=True, exist_ok=True)

    ensure_reference_model_artifact(storage_root / "shipped-models")

    return STORAGE_DIRECTORIES.copy()


def create_app(storage_root: Path | None = None) -> FastAPI:
    resolved_storage_root = Path(
        storage_root or Path(__file__).resolve().parents[2] / "data"
    ).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.storage_root = resolved_storage_root
        app.state.storage_directories = ensure_storage_structure(resolved_storage_root)
        yield

    app = FastAPI(title="MNIST Classifier Backend", lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        storage_directories = getattr(app.state, "storage_directories", STORAGE_DIRECTORIES)
        storage_root_path = getattr(app.state, "storage_root", resolved_storage_root)

        return {
            "status": "ok",
            "service": "mnist-backend",
            "storage": {
                "ready": all((storage_root_path / directory).is_dir() for directory in storage_directories),
                "root": str(storage_root_path),
                "directories": storage_directories,
            },
        }

    @app.get("/api/models")
    def models() -> dict[str, object]:
        storage_root_path = getattr(app.state, "storage_root", resolved_storage_root)
        return {"models": list_reference_models(storage_root_path)}

    @app.post("/api/predict")
    def predict(request: PredictionRequest) -> dict[str, object]:
        storage_root_path = getattr(app.state, "storage_root", resolved_storage_root)

        try:
            return predict_reference_digit(
                model_id=request.model_id,
                width=request.canvas.width,
                height=request.canvas.height,
                pixels=request.canvas.pixels,
                storage_root=storage_root_path,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=error.args[0]) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/training/csv-preview")
    def training_csv_preview(request: TrainingCsvPreviewRequest) -> dict[str, object]:
        try:
            return preview_training_csv(
                file_name=request.file_name,
                csv_text=request.csv_text,
                train_ratio=request.split.train_ratio,
                validation_ratio=request.split.validation_ratio,
                test_ratio=request.split.test_ratio,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()