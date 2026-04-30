from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

from backend.app.custom_training import (
    JobConflictError,
    ModelConflictError,
    ModelDeletionError,
    TrainingJobManager,
    delete_custom_model,
    list_available_models,
    predict_available_model,
)
from backend.app.reference_model import (
    ensure_reference_model_artifact,
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


TrainingModelFamily = Literal[
    "prototype",
    "knn",
    "svm",
    "random-forest",
    "mlp",
    "cnn",
]


class PrototypeTrainingHyperparameters(BaseModel):
    max_examples_per_label: int = Field(ge=1, le=5000)
    prototype_blend: float = Field(ge=0, le=1)
    temperature: float = Field(gt=0, le=40)


class KnnTrainingHyperparameters(BaseModel):
    neighbors: int = Field(ge=1, le=15)
    pca_components: int = Field(ge=4, le=64)


class SvmTrainingHyperparameters(BaseModel):
    regularization: float = Field(gt=0, le=10)
    max_iter: int = Field(ge=500, le=20000)
    pca_components: int = Field(ge=4, le=64)


class RandomForestTrainingHyperparameters(BaseModel):
    estimators: int = Field(ge=10, le=400)
    max_depth: int = Field(ge=2, le=64)
    pca_components: int = Field(ge=4, le=64)


class DeepTrainingHyperparameters(BaseModel):
    epochs: int = Field(ge=1, le=12)
    batch_size: int = Field(ge=4, le=128)
    learning_rate: float = Field(gt=0, le=0.05)


TRAINING_HYPERPARAMETER_MODELS: dict[
    TrainingModelFamily,
    type[BaseModel],
] = {
    "prototype": PrototypeTrainingHyperparameters,
    "knn": KnnTrainingHyperparameters,
    "svm": SvmTrainingHyperparameters,
    "random-forest": RandomForestTrainingHyperparameters,
    "mlp": DeepTrainingHyperparameters,
    "cnn": DeepTrainingHyperparameters,
}


class CustomTrainingRequest(BaseModel):
    model_name: str = Field(min_length=3, max_length=80)
    model_family: TrainingModelFamily = "prototype"
    file_name: str = Field(min_length=1)
    csv_text: str = Field(min_length=1)
    split: TrainingSplitPayload
    seed: int = Field(ge=0, le=1_000_000)
    hyperparameters: dict[str, object]

    @model_validator(mode="after")
    def validate_model_name(self) -> "CustomTrainingRequest":
        if not self.model_name.strip():
            raise ValueError("Model name is required.")

        hyperparameter_model = TRAINING_HYPERPARAMETER_MODELS[self.model_family]
        self.hyperparameters = hyperparameter_model.model_validate(
            self.hyperparameters
        ).model_dump()

        return self


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
        app.state.training_manager = TrainingJobManager(resolved_storage_root)
        try:
            yield
        finally:
            app.state.training_manager.shutdown()

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
        return {"models": list_available_models(storage_root_path)}

    @app.post("/api/predict")
    def predict(request: PredictionRequest) -> dict[str, object]:
        storage_root_path = getattr(app.state, "storage_root", resolved_storage_root)

        try:
            return predict_available_model(
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

    @app.post("/api/training/jobs", status_code=202)
    def create_training_job(request: CustomTrainingRequest) -> dict[str, object]:
        training_manager = getattr(app.state, "training_manager")

        try:
            job = training_manager.start_job(
                model_name=request.model_name,
                model_family=request.model_family,
                file_name=request.file_name,
                csv_text=request.csv_text,
                split=request.split.model_dump(),
                seed=request.seed,
                hyperparameters=request.hyperparameters,
            )
        except JobConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ModelConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {"job": job}

    @app.get("/api/training/jobs/{job_id}")
    def get_training_job(job_id: str) -> dict[str, object]:
        training_manager = getattr(app.state, "training_manager")

        try:
            return {"job": training_manager.get_job(job_id)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown training job '{job_id}'.") from error

    @app.delete("/api/models/{model_id}", status_code=204)
    def delete_model(model_id: str) -> Response:
        storage_root_path = getattr(app.state, "storage_root", resolved_storage_root)

        try:
            delete_custom_model(storage_root_path, model_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=error.args[0]) from error
        except ModelDeletionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

        return Response(status_code=204)

    return app


app = create_app()