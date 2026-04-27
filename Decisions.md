# Decisions

Status: Draft. This file will be updated as decisions are confirmed.

## Locked Requirements

### Classifiers

- k-NN from scratch
- Softmax classifier
- Support Vector Machine
- Random Forest
- Multi Layer Perceptron
- Convolutional Neural Network

### Web GUI

- React + TypeScript frontend
- 28x28 painting area for testing models
- Leaderboard for all models
- CSV upload for extra data
- Prediction results paired with the 28x28 drawing
- Training stats shown when a leaderboard entry is opened

## Recommended Additions

- A reproducible training pipeline with saved configs and model artifacts
- A fixed validation/test split so comparisons stay honest
- More than one metric on the leaderboard: accuracy, macro F1, and inference latency
- A confusion matrix and a misclassified examples gallery for the report/demo
- A small backend API for inference instead of loading every model directly in the browser
- Exportable charts/screenshots for the final school presentation

## Confirmed Decisions

- Model strategy: hybrid implementation.
- k-NN and softmax will be implemented from scratch.
- SVM and random forest will use scikit-learn.
- MLP and CNN will use PyTorch.
- Architecture style: modular design with explicit interfaces between modules.
- Deployment style: modular monolith.
- One Python backend service will expose the public API.
- The backend will contain multiple internal modules instead of separate deployable services.
- Models will be pre-trained and available immediately in the app.
- Users can train new models from scratch during app runtime.
- Runtime training will accept user-provided hyperparameters.
- All six classifier types are expected to support user-triggered training.
- User-created model entries can be named and appear on the same leaderboard.
- Training jobs will run asynchronously.
- The UI will show training progress and status updates.
- CSV uploads will be restricted to MNIST-compatible data.
- Uploaded training/evaluation data will use 784 pixel features.
- Labels are required for training and evaluation datasets.
- Labels can be optional for prediction-only datasets.
- The training UI will expose a curated set of important hyperparameters per model.
- Every training form will provide sane defaults and validation.
- Trained model artifacts will be stored on the backend as the source of truth.
- Model metadata and leaderboard data will be persisted outside the browser.
- The browser can cache UI state, but it will not be the canonical storage layer.
- Default training presets should stay close to standard scikit-learn, scratch, and PyTorch baseline settings.
- The project does not need separate aggressive hardware-specific tuning modes in the first version.
- Upload size should have a hard limit of roughly 20 MB.
- Hyperparameter forms should show soft warnings when values look expensive or risky.
- Training forms should display explicit parameter ceilings in the UI.
- Training forms should also show recommended parameter values alongside those ceilings.
- The app will not use user accounts or authentication.
- The project should stay plug-and-play across PCs.
- Storage should live in project-relative backend files so the app state can move with the project folder.
- Startup should be script-based instead of Docker-dependent.
- The app should be launchable with a small, simple command flow on another PC.
- The project should start from one root-level command.
- The startup flow can assume Python is installed.
- The startup flow can assume Node is installed.
- The leaderboard should present multiple model metrics instead of a single bare score.
- The leaderboard should be a multi-column sortable table.
- Users should be able to sort by any visible metric instead of relying on a hidden composite score.
- The model details panel should show core metrics.
- The model details panel should show the exact training hyperparameters.
- The model details panel should show training curves when the model type supports them.
- The model details panel should show a confusion matrix.
- The model details panel should show sample predictions paired with labels.
- The model details panel should show dataset source details.
- The model details panel should show model size information where available.
- Custom trained models should support deletion.
- Each new training run should create a new leaderboard entry instead of replacing an old one.
- Deep models may expose extra capabilities beyond the shared adapter interface.
- The allowed deep-model extras are epoch curves and architecture summary data.
- The shared prediction contract should return the top predicted class plus confidence information.
- Every trained model should save an immutable training config snapshot with its artifact.
- Training runs should expose and persist a random seed for reproducibility.
- Uploaded datasets for custom training should be auto-split by default.
- The default split ratios should be visible in the UI.
- Users should be allowed to override the default split ratios.
- Built-in MNIST and uploaded datasets should remain separate for custom training runs.
- TDD should focus first on module contracts, validation, persistence, and job flow.
- Model-quality thresholds should be secondary to stable interface and workflow tests.
- Automated tests should use reduced MNIST subsets so the suite stays fast.
- Canonical data preprocessing should live in the backend.
- Canvas input should be transformed by the backend using the same preprocessing assumptions as training.
- The canvas prediction flow should target one selected model at a time.
- The prediction UI should show confidence values for all digits 0 through 9.
- Version one should allow one training job at a time.
- Model and leaderboard metadata should be stored with JSON manifests rather than SQLite.
- Version one metadata should use one central registry JSON file for simplicity.
- Target demo machines can be treated as Windows-only.
- The visible startup command for version one should be a root PowerShell script.
- Pre-trained model artifacts should ship inside the project folder as long as their size stays reasonable.
- Newly trained custom model artifacts should live in a project-local data folder.
- Generated custom artifacts should stay separate from source code files.
- Built-in pre-trained models should be protected from deletion in the UI.
- Each training request should target one classifier type.
- If the app closes during training, the in-progress job should be terminated and discarded.
- Custom model names should be unique.

Reason: this keeps the project technically credible while staying realistic for a school timeline.

Recommended modular split:

- data module: MNIST loading, preprocessing, CSV import validation
- training module: training jobs, configs, saved artifacts
- model registry: stores pre-trained and user-created named model entries
- model adapters: mostly one common interface across all classifiers, with tightly scoped deep-model exceptions
- evaluation module: metrics, confusion matrix, leaderboard data
- inference API: prediction and model stats endpoints
- frontend: React UI, canvas, leaderboard, stats views

## Next Questions

- Milestone order is intentionally deferred until the deeper module architecture and TDD approach are defined.
- Should the app auto-suggest default names for custom models before the user edits them?
- Should failed training jobs appear in a history view, or disappear entirely in version one?
- Should there be a separate setup script in addition to the root start script?

Note: because one leaderboard will contain both pre-trained and user-created models, each entry should display dataset source, training timestamp, and whether it is built-in or custom.
