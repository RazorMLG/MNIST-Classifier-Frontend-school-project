from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

STORAGE_DIRECTORIES = ["shipped-models", "custom-models", "registry"]


def ensure_storage_structure(storage_root: Path) -> list[str]:
    storage_root.mkdir(parents=True, exist_ok=True)

    for directory in STORAGE_DIRECTORIES:
        (storage_root / directory).mkdir(parents=True, exist_ok=True)

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

    return app


app = create_app()