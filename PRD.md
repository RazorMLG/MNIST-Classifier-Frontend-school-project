## Problem Statement

I want to build an end-of-year school project around MNIST that is strong both as a machine learning demo and as a software engineering project. The project needs to compare several classifier families, provide a usable web interface, allow live experimentation, and stay simple enough to run on ordinary Windows machines without a heavy deployment story. It also needs to show architectural discipline through modular boundaries, reproducibility, and a testing strategy that is realistic for a student project.

## Solution

Build a modular monolith with a Python backend and a React + TypeScript frontend. The backend will own canonical preprocessing, model training, inference, evaluation, artifact persistence, and leaderboard metadata. The frontend will provide a 28x28 drawing canvas, a sortable multi-metric leaderboard, model detail views, CSV-based custom training flows, and progress-aware training UX.

The product will ship with six classifier types:

1. k-NN from scikit-learn
2. Softmax classifier via scikit-learn
3. Support Vector Machine via scikit-learn
4. Random Forest via scikit-learn
5. Multi Layer Perceptron via PyTorch
6. Convolutional Neural Network via PyTorch

Pre-trained models will be available immediately. Users will also be able to train custom models from scratch at runtime on MNIST-compatible CSV uploads, choose curated hyperparameters, assign unique names, and compare those custom models on the same leaderboard as built-in models. The system will stay low-scope and portable by using project-local storage, a single training job at a time, Windows-friendly startup scripts, and JSON-based metadata.

## User Stories

1. As a student presenter, I want the project to start from one root command, so that I can demo it quickly on another PC.
2. As a student presenter, I want the app to work without accounts or login flows, so that the demo stays plug-and-play.
3. As a student presenter, I want pre-trained models to be available immediately, so that I can demonstrate predictions without waiting for training.
4. As a viewer, I want to see all supported classifier families in one app, so that I can compare classical ML and deep learning approaches.
5. As a viewer, I want to draw a digit on a 28x28 canvas, so that I can test the models interactively.
6. As a viewer, I want the app to show the prediction for my drawing, so that I can understand what the selected model thinks the digit is.
7. As a viewer, I want to see confidence values for digits 0 through 9, so that I can understand model uncertainty and confusion.
8. As a viewer, I want to choose which model I am testing before predicting, so that I can compare different trained models one at a time.
9. As a viewer, I want the leaderboard to show multiple metrics, so that model quality is not reduced to one opaque score.
10. As a viewer, I want the leaderboard to be sortable by any visible metric, so that I can decide whether accuracy, F1, or latency matters most.
11. As a viewer, I want each leaderboard entry to clearly show whether it is built-in or custom, so that I can distinguish official models from user-trained ones.
12. As a viewer, I want each leaderboard entry to show dataset source and training timestamp, so that I can interpret results in context.
13. As a viewer, I want to open a model details panel from the leaderboard, so that I can inspect how a model was trained and how it performs.
14. As a viewer, I want the details panel to show core metrics, so that I can evaluate model performance beyond ranking position.
15. As a viewer, I want the details panel to show the exact training hyperparameters, so that I can understand why one model differs from another.
16. As a viewer, I want the details panel to show training curves when available, so that I can understand learning behavior over time.
17. As a viewer, I want the details panel to show a confusion matrix, so that I can see which digits the model confuses.
18. As a viewer, I want the details panel to show sample predictions paired with labels, so that I can inspect concrete successes and failures.
19. As a viewer, I want the details panel to show model size information when available, so that I can reason about performance versus complexity.
20. As a user experimenting with data, I want to upload MNIST-compatible CSV files, so that I can train and evaluate custom models on my own data.
21. As a user experimenting with data, I want upload validation to reject incompatible CSV formats, so that I get clear feedback instead of broken training runs.
22. As a user experimenting with data, I want the system to auto-split uploaded datasets by default, so that I can train quickly without manual dataset preparation.
23. As a user experimenting with data, I want the default train, validation, and test split ratios to be visible, so that I know how my data is being used.
24. As a user experimenting with data, I want to override the default split ratios, so that I can control evaluation strategy when needed.
25. As a user experimenting with data, I want uploaded datasets to remain separate from built-in MNIST, so that comparisons stay understandable.
26. As a user training a model, I want to choose one classifier type per training run, so that progress and results stay easy to follow.
27. As a user training a model, I want to enter curated hyperparameters with defaults, so that I can experiment without facing an overwhelming ML control panel.
28. As a user training a model, I want recommended values and visible ceilings in the UI, so that I can tune models without accidentally creating absurd runs.
29. As a user training a model, I want soft warnings on risky hyperparameter choices, so that I understand when a run may be slow or unstable.
30. As a user training a model, I want to name my custom model uniquely, so that I can find it later on the leaderboard.
31. As a user training a model, I want the training job to run asynchronously, so that the UI stays responsive while work is in progress.
32. As a user training a model, I want to see training progress and status updates, so that I know whether the run is still active.
33. As a user training a model, I want each new run to create a new leaderboard entry, so that I can compare experiments instead of overwriting history.
34. As a user training a model, I want only one training job to run at a time in version one, so that the app stays predictable on normal hardware.
35. As a user training a model, I want interrupted training jobs to be discarded if the app closes, so that the first version stays simple and honest.
36. As a user managing custom models, I want to delete my custom models, so that I can keep the leaderboard and storage clean.
37. As a user managing models, I do not want built-in models to disappear accidentally, so that the default demo remains intact.
38. As a developer, I want canonical preprocessing to live in the backend, so that canvas inference, CSV ingestion, and training all use consistent assumptions.
39. As a developer, I want most model modules to share one adapter interface, so that training, inference, save, load, and evaluation flows remain uniform.
40. As a developer, I want deep models to expose only limited extra capabilities, so that the interface stays coherent while still supporting curves and architecture summaries.
41. As a developer, I want every trained model to save an immutable config snapshot, so that experiments are reproducible and explainable.
42. As a developer, I want every training run to persist a random seed, so that results can be reproduced or debugged later.
43. As a developer, I want metadata and artifacts to live in project-local storage, so that the project remains portable across machines.
44. As a developer, I want built-in artifacts shipped with the project when reasonably small, so that the demo works offline and immediately.
45. As a developer, I want custom artifacts separated from source code, so that generated state does not pollute implementation files.
46. As a developer, I want a central JSON registry for version one metadata, so that the storage model stays low-scope and easy to understand.
47. As a developer, I want tests to focus first on module contracts and workflow behavior, so that the test suite stays stable as implementations evolve.
48. As a developer, I want automated tests to use reduced MNIST subsets, so that the test suite remains fast enough to run often.
49. As a teacher or reviewer, I want the project to demonstrate modular architecture decisions clearly, so that the engineering quality is visible beyond the UI.
50. As a teacher or reviewer, I want the project to show both ML performance and system design tradeoffs, so that it feels like a complete software project rather than only a notebook demo.

## Implementation Decisions

- The system will be a modular monolith, not a microservice architecture.
- The backend will be a single Python API service responsible for preprocessing, training, inference, evaluation, persistence, and job orchestration.
- The frontend will be a React + TypeScript web application responsible for canvas interaction, leaderboard browsing, model details, and training workflows.
- The shipped classical classifier stack will use scikit-learn for k-NN, SVM, and random forest, with artifacts regenerated offline from the repository root `train.csv` rather than trained during normal startup.
- The softmax classifier and deep-learning families remain separate planned slices with their own implementation choices.
- Pre-trained models will ship with the project when their size remains reasonable for version control and portability.
- Shipped classical model evaluation will come only from deterministic train, validation, and test partitions derived from the labeled root `train.csv`; the unlabeled root `test.csv` is out of scope for built-in evaluation.
- Custom runtime training will support all six classifier families.
- Training jobs will be asynchronous, but version one will permit only one active training job at a time.
- Each training request will target a single classifier type rather than launching a multi-model batch.
- Interrupted training jobs will be terminated and discarded instead of resumed.
- The backend will own the canonical preprocessing pipeline for canvas input, CSV input, training, and inference consistency.
- CSV ingestion will be restricted to MNIST-compatible data with 784 pixel features.
- Labels will be required for training and evaluation uploads, and optional only for prediction-only uploads.
- Uploaded datasets will be auto-split by default, with visible default split ratios and optional user override.
- Uploaded datasets will remain separate from built-in MNIST rather than being mixed into one training pool.
- Model metadata and leaderboard data will persist outside the browser; browser storage may cache UI state but will not be the source of truth.
- Version one metadata will use JSON storage with a single central registry file.
- Built-in model artifacts and custom generated artifacts will be stored separately.
- Custom trained artifacts will live in a project-local data area, distinct from implementation code.
- Built-in models will be protected from deletion in the UI; custom models will support deletion.
- Each new training run will create a new leaderboard entry rather than mutating a previous one.
- Custom model names will be unique.
- The leaderboard will expose multiple metrics and allow column-based sorting instead of relying on one hidden weighted score.
- The model details view will include metrics, hyperparameters, curves where supported, confusion matrix, sample predictions, dataset source, and size-related information.
- The shared model adapter contract should support the common training and inference lifecycle consistently across classifier families.
- Deep-model exceptions will be tightly scoped to epoch curves and architecture summary data.
- Prediction responses will include the top predicted class and confidence information, and the UI will present confidence values for all digits 0 through 9.
- The training UI will expose curated hyperparameters only, with sane defaults, visible ceilings, recommended values, and soft warnings.
- The project will be no-auth and optimized for plug-and-play use on other Windows PCs.
- The startup model for version one will be script-based rather than Docker-based.
- The intended visible startup entrypoint will be a root PowerShell script.
- Shipped classical artifact regeneration will be an explicit maintainer command rather than part of the normal startup flow.
- The startup flow may assume Python and Node are already installed on the target machine.
- The major deep modules to build are the data/preprocessing module, the model adapter layer, the training job orchestrator, the model registry, the evaluation module, the inference API, and the frontend application shell.
- Milestone ordering has been intentionally deferred until the deeper architecture and TDD approach are refined further.
- Open implementation questions still remaining are whether to auto-suggest default names for custom models, whether failed jobs should appear in history in version one, and whether there should be a separate setup script in addition to the start script.

## Testing Decisions

- A good test should validate externally visible behavior, stable contracts, and user-relevant outcomes rather than internal implementation details.
- Contract tests should verify the shared expectations across model adapters, including training entrypoints, prediction output shape, evaluation output shape, save and load behavior, and error handling.
- Data module tests should verify MNIST-compatible CSV validation, split logic, label handling, and preprocessing behavior.
- Preprocessing tests should verify that canvas-originated data and dataset-originated data are transformed consistently by the backend pipeline.
- Training orchestrator tests should verify job creation, single-job concurrency limits, progress-state transitions, interruption behavior, and rejection of invalid hyperparameter requests.
- Model registry tests should verify persistence, uniqueness rules, built-in versus custom model distinctions, deletion rules, and leaderboard entry creation.
- Evaluation module tests should verify metric calculation, confusion matrix generation, and details-panel data assembly.
- Inference API tests should verify single-model prediction flows, confidence payload structure, and correct handling of missing or invalid model identifiers.
- Frontend tests should verify core user workflows such as selecting a model, drawing a digit, requesting a prediction, uploading a CSV, starting a training job, and viewing leaderboard details.
- Automated tests should rely on reduced MNIST subsets so the suite remains fast and repeatable.
- Reproducibility-oriented tests should verify config snapshot persistence and random-seed persistence, not specific exact scores on large datasets.
- Model-quality thresholds may exist later as secondary checks, but they should not dominate the first testing layer because they are more brittle than contract and workflow tests.
- There is no established prior testing pattern in the current repository yet, so the test approach will need to be introduced deliberately from scratch with contract-first discipline.

## Out of Scope

- User accounts, authentication, and ownership security boundaries
- Browser storage as the canonical source of truth for models or metadata
- Microservice deployment or multiple backend services
- Docker-first startup and deployment
- Arbitrary non-MNIST tabular CSV support
- Mixing uploaded datasets directly into the built-in MNIST training corpus
- Multiple concurrent training jobs in version one
- Multi-classifier batch training from one user action in version one
- Automatic resume or restart of interrupted training jobs
- Composite weighted leaderboard scores as the main ranking mechanism
- Deleting built-in shipped models from the UI
- Broad exposure of every low-level training option from underlying libraries
- Deep-model checkpointing
- Advanced deep-learning controls such as custom callbacks or manual optimizer micromanagement in version one

## Further Notes

- This PRD is synthesized from the current project spec and decision log.
- The current repository contains planning documents only, so this PRD defines the implementation direction rather than describing an existing codebase.
- The project is intentionally balancing ML breadth with architecture discipline: enough algorithmic variety to be credible, but with explicit scope control to keep the deliverable demoable.
- The design already favors deep modules that can be tested in isolation: preprocessing, model adapters, training orchestration, registry, and evaluation.
- The remaining open questions are small enough to resolve during implementation planning without changing the main architecture.
- This environment does not provide a GitHub issue submission tool, so this document should be treated as an issue-ready PRD draft.
