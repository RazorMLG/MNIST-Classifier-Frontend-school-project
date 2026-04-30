from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.shipped_classical_training import (
    MODEL_SPECS as CLASSICAL_MODEL_SPECS,
    regenerate_shipped_classical_models,
)
from backend.app.shipped_deep_training import (
    MODEL_SPECS as DEEP_MODEL_SPECS,
    regenerate_shipped_deep_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate shipped MNIST model artifacts.")
    parser.add_argument("--dataset", required=True, help="Path to the labeled training CSV.")
    parser.add_argument("--storage-root", required=True, help="Directory where shipped model artifacts are written.")
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Optional shipped model id to regenerate. Repeat to select multiple models.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Deterministic split and training seed.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    storage_root = Path(args.storage_root).resolve()
    selected_model_ids = args.model_ids or []

    known_model_ids = {*CLASSICAL_MODEL_SPECS, *DEEP_MODEL_SPECS}
    unknown_model_ids = sorted(model_id for model_id in selected_model_ids if model_id not in known_model_ids)
    if unknown_model_ids:
        joined_model_ids = ", ".join(unknown_model_ids)
        raise KeyError(f"Unknown shipped model ids: {joined_model_ids}.")

    selected_classical_ids = [
        model_id for model_id in selected_model_ids if model_id in CLASSICAL_MODEL_SPECS
    ]
    selected_deep_ids = [model_id for model_id in selected_model_ids if model_id in DEEP_MODEL_SPECS]

    if not selected_model_ids or selected_classical_ids:
        regenerate_shipped_classical_models(
            dataset_path=dataset_path,
            storage_root=storage_root,
            model_ids=selected_classical_ids or None,
            seed=args.seed,
        )

    if not selected_model_ids or selected_deep_ids:
        regenerate_shipped_deep_models(
            dataset_path=dataset_path,
            storage_root=storage_root,
            model_ids=selected_deep_ids or None,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()