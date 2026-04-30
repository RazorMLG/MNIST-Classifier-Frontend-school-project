import { useEffect, useState, type ChangeEvent } from "react";

import "./App.css";

const DRAWING_GRID_SIZE = 20;
const DRAWING_PIXEL_COUNT = DRAWING_GRID_SIZE * DRAWING_GRID_SIZE;

const SORTABLE_METRICS = ["accuracy", "macro_f1", "avg_inference_ms"] as const;

type SortableMetricKey = (typeof SORTABLE_METRICS)[number];

type HealthPayload = {
  status: string;
  service: string;
  storage: {
    ready: boolean;
    root: string;
    directories: string[];
  };
};

type ModelMetrics = {
  accuracy?: number;
  macro_precision?: number;
  macro_recall?: number;
  macro_f1?: number;
  avg_inference_ms?: number;
};

type ModelDataset = {
  source?: string;
  image_shape?: string;
  train_examples?: number;
  validation_examples?: number;
  test_examples?: number;
};

type SamplePrediction = {
  label: number;
  predicted: number;
  confidence: number;
};

type ModelEvaluation = {
  confusion_matrix?: number[][];
  sample_predictions?: SamplePrediction[];
};

type ModelTrainingSnapshot = {
  classifier?: string;
  file_name?: string;
  split?: {
    train_ratio: number;
    validation_ratio: number;
    test_ratio: number;
  };
  hyperparameters?: Record<string, number | string>;
};

type ModelTraining = {
  seed?: number;
  config_snapshot?: ModelTrainingSnapshot;
};

type ModelSummary = {
  id: string;
  name: string;
  kind: string;
  family?: string;
  description?: string;
  trained_at?: string;
  metrics?: ModelMetrics;
  dataset?: ModelDataset;
  hyperparameters?: Record<string, number | string>;
  evaluation?: ModelEvaluation;
  training?: ModelTraining;
  input?: {
    width: number;
    height: number;
    encoding?: string;
  };
};

type ModelsPayload = {
  models: ModelSummary[];
};

type PredictionPayload = {
  model: {
    id: string;
    name: string;
  };
  prediction: {
    digit: number;
    confidences: number[];
  };
};

type BootstrapState =
  | { kind: "loading" }
  | { kind: "ready"; health: HealthPayload; models: ModelSummary[] }
  | { kind: "error"; message: string };

type PredictionState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; payload: PredictionPayload }
  | { kind: "error"; message: string };

type WorkspaceMode = "predict" | "training";

type TrainingSplitForm = {
  train: number;
  validation: number;
  test: number;
};

type TrainingPreviewPayload = {
  file_name: string;
  dataset: {
    example_count: number;
    feature_count: number;
    label_range: {
      min: number;
      max: number;
    };
  };
  split: {
    ratios: {
      train: number;
      validation: number;
      test: number;
    };
    counts: {
      train: number;
      validation: number;
      test: number;
    };
  };
};

type TrainingPreviewState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; payload: TrainingPreviewPayload }
  | { kind: "error"; message: string };

type TrainingJob = {
  id: string;
  model_id: string;
  model_name: string;
  status: string;
  progress: {
    percent: number;
    stage: string;
  };
  error?: string | null;
  started_at?: string;
  completed_at?: string | null;
};

type TrainingJobPayload = {
  job: TrainingJob;
};

type CustomTrainingForm = {
  modelName: string;
  seed: number;
  maxExamplesPerLabel: number;
  prototypeBlend: number;
  temperature: number;
};

type TrainingJobState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "running"; job: TrainingJob }
  | { kind: "completed"; job: TrainingJob }
  | { kind: "error"; message: string };

const DEFAULT_TRAINING_SPLIT: TrainingSplitForm = {
  train: 80,
  validation: 10,
  test: 10,
};

const DEFAULT_CUSTOM_TRAINING_FORM: CustomTrainingForm = {
  modelName: "",
  seed: 17,
  maxExamplesPerLabel: 4,
  prototypeBlend: 0.35,
  temperature: 18,
};

const TRAINING_POLL_INTERVAL_MS = 150;

export function App() {
  const [bootstrapState, setBootstrapState] = useState<BootstrapState>({
    kind: "loading",
  });
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("predict");
  const [leaderboardMetric, setLeaderboardMetric] =
    useState<SortableMetricKey>("accuracy");
  const [selectedModelId, setSelectedModelId] = useState("");
  const [canvasPixels, setCanvasPixels] = useState<number[]>(() =>
    createBlankCanvas(),
  );
  const [predictionState, setPredictionState] = useState<PredictionState>({
    kind: "idle",
  });
  const [isPainting, setIsPainting] = useState(false);
  const [trainingSplit, setTrainingSplit] = useState<TrainingSplitForm>(
    DEFAULT_TRAINING_SPLIT,
  );
  const [trainingFile, setTrainingFile] = useState<File | null>(null);
  const [trainingPreviewState, setTrainingPreviewState] =
    useState<TrainingPreviewState>({ kind: "idle" });
  const [customTrainingForm, setCustomTrainingForm] =
    useState<CustomTrainingForm>(DEFAULT_CUSTOM_TRAINING_FORM);
  const [trainingJobState, setTrainingJobState] = useState<TrainingJobState>({
    kind: "idle",
  });
  const [modelDeleteFeedback, setModelDeleteFeedback] = useState<string | null>(
    null,
  );
  const [isDeletingModel, setIsDeletingModel] = useState(false);

  function applyBootstrapPayload(
    healthPayload: HealthPayload,
    modelsPayload: ModelsPayload,
    preferredModelId?: string,
  ) {
    setBootstrapState({
      kind: "ready",
      health: healthPayload,
      models: modelsPayload.models,
    });

    setSelectedModelId((currentModelId) => {
      const nextSelectedModelId = preferredModelId ?? currentModelId;

      if (
        nextSelectedModelId &&
        modelsPayload.models.some((model) => model.id === nextSelectedModelId)
      ) {
        return nextSelectedModelId;
      }

      return modelsPayload.models[0]?.id ?? "";
    });
  }

  async function refreshBootstrap(preferredModelId?: string) {
    try {
      const [healthPayload, modelsPayload] = await Promise.all([
        fetchJson<HealthPayload>("/api/health"),
        fetchJson<ModelsPayload>("/api/models"),
      ]);

      applyBootstrapPayload(healthPayload, modelsPayload, preferredModelId);
      return modelsPayload;
    } catch (error) {
      setBootstrapState({
        kind: "error",
        message:
          error instanceof Error ? error.message : "Backend check failed.",
      });
      throw error;
    }
  }

  useEffect(() => {
    let isActive = true;

    async function loadAppBootstrap() {
      try {
        const [healthPayload, modelsPayload] = await Promise.all([
          fetchJson<HealthPayload>("/api/health"),
          fetchJson<ModelsPayload>("/api/models"),
        ]);

        if (!isActive) {
          return;
        }

        applyBootstrapPayload(healthPayload, modelsPayload);
      } catch (error) {
        if (!isActive) {
          return;
        }

        setBootstrapState({
          kind: "error",
          message:
            error instanceof Error ? error.message : "Backend check failed.",
        });
      }
    }

    void loadAppBootstrap();

    return () => {
      isActive = false;
    };
  }, []);

  const hasInk = canvasPixels.some((pixel) => pixel > 0);
  const selectedModel =
    bootstrapState.kind === "ready"
      ? bootstrapState.models.find((model) => model.id === selectedModelId)
      : undefined;
  const sortedModels =
    bootstrapState.kind === "ready"
      ? [...bootstrapState.models].sort((leftModel, rightModel) =>
          compareModels(leftModel, rightModel, leaderboardMetric),
        )
      : [];
  const trainingSplitTotal =
    trainingSplit.train + trainingSplit.validation + trainingSplit.test;
  const canPreviewTrainingSplit =
    bootstrapState.kind === "ready" &&
    trainingFile !== null &&
    trainingPreviewState.kind !== "loading";

  const readinessLabel =
    bootstrapState.kind === "ready" && bootstrapState.health.storage.ready
      ? "Backend ready"
      : bootstrapState.kind === "error"
        ? "Backend unavailable"
        : "Checking backend";

  async function handlePredict() {
    if (!selectedModelId || !hasInk) {
      return;
    }

    setPredictionState({ kind: "loading" });

    try {
      const payload = await fetchJson<PredictionPayload>("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_id: selectedModelId,
          canvas: {
            width: DRAWING_GRID_SIZE,
            height: DRAWING_GRID_SIZE,
            pixels: canvasPixels,
          },
        }),
      });

      setPredictionState({ kind: "ready", payload });
    } catch (error) {
      setPredictionState({
        kind: "error",
        message:
          error instanceof Error ? error.message : "Prediction request failed.",
      });
    }
  }

  function handlePaint(index: number) {
    setCanvasPixels((currentPixels) => {
      if (currentPixels[index] === 255) {
        return currentPixels;
      }

      const nextPixels = currentPixels.slice();
      nextPixels[index] = 255;
      return nextPixels;
    });

    if (predictionState.kind !== "idle") {
      setPredictionState({ kind: "idle" });
    }
  }

  function handleClearBoard() {
    setCanvasPixels(createBlankCanvas());
    setPredictionState({ kind: "idle" });
  }

  function handleTrainingFileChange(event: ChangeEvent<HTMLInputElement>) {
    setTrainingFile(event.target.files?.[0] ?? null);
    setTrainingPreviewState({ kind: "idle" });
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  function handleTrainingSplitChange(
    key: keyof TrainingSplitForm,
    value: string,
  ) {
    const numericValue = Number(value);

    setTrainingSplit((currentSplit) => ({
      ...currentSplit,
      [key]: Number.isFinite(numericValue) ? numericValue : 0,
    }));
    setTrainingPreviewState({ kind: "idle" });
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  function handleTrainingModelNameChange(value: string) {
    setCustomTrainingForm((currentForm) => ({
      ...currentForm,
      modelName: value,
    }));
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  function handleTrainingSettingChange(
    key: Exclude<keyof CustomTrainingForm, "modelName">,
    value: string,
  ) {
    const numericValue = Number(value);

    setCustomTrainingForm((currentForm) => ({
      ...currentForm,
      [key]: Number.isFinite(numericValue) ? numericValue : 0,
    }));
    setModelDeleteFeedback(null);

    if (
      trainingJobState.kind !== "running" &&
      trainingJobState.kind !== "submitting"
    ) {
      setTrainingJobState({ kind: "idle" });
    }
  }

  async function handlePreviewTrainingSplit() {
    if (!trainingFile) {
      return;
    }

    if (trainingSplit.train <= 0) {
      setTrainingPreviewState({
        kind: "error",
        message: "Training split must be greater than zero.",
      });
      return;
    }

    if (Math.abs(trainingSplitTotal - 100) > 0.001) {
      setTrainingPreviewState({
        kind: "error",
        message: "Split ratios must total 100 before preview.",
      });
      return;
    }

    setTrainingPreviewState({ kind: "loading" });

    try {
      const payload = await fetchJson<TrainingPreviewPayload>(
        "/api/training/csv-preview",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_name: trainingFile.name,
            csv_text: await readTextFromFile(trainingFile),
            split: {
              train_ratio: toSplitRatio(trainingSplit.train),
              validation_ratio: toSplitRatio(trainingSplit.validation),
              test_ratio: toSplitRatio(trainingSplit.test),
            },
          }),
        },
      );

      setTrainingPreviewState({ kind: "ready", payload });
    } catch (error) {
      setTrainingPreviewState({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "CSV preview request failed.",
      });
    }
  }

  async function handleStartTraining() {
    if (!trainingFile) {
      return;
    }

    const modelName = customTrainingForm.modelName.trim();
    if (!modelName) {
      setTrainingJobState({
        kind: "error",
        message: "Model name is required before starting training.",
      });
      return;
    }

    if (trainingPreviewState.kind !== "ready") {
      setTrainingJobState({
        kind: "error",
        message: "Preview the dataset split before starting training.",
      });
      return;
    }

    if (
      bootstrapState.kind === "ready" &&
      bootstrapState.models.some(
        (model) => model.name.toLowerCase() === modelName.toLowerCase(),
      )
    ) {
      setTrainingJobState({
        kind: "error",
        message: `Model name '${modelName}' is already in use. Choose a unique name.`,
      });
      return;
    }

    setModelDeleteFeedback(null);
    setTrainingJobState({ kind: "submitting" });

    try {
      const payload = await fetchJson<TrainingJobPayload>(
        "/api/training/jobs",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model_name: modelName,
            file_name: trainingFile.name,
            csv_text: await readTextFromFile(trainingFile),
            split: {
              train_ratio: toSplitRatio(trainingSplit.train),
              validation_ratio: toSplitRatio(trainingSplit.validation),
              test_ratio: toSplitRatio(trainingSplit.test),
            },
            seed: customTrainingForm.seed,
            hyperparameters: {
              max_examples_per_label: customTrainingForm.maxExamplesPerLabel,
              prototype_blend: customTrainingForm.prototypeBlend,
              temperature: customTrainingForm.temperature,
            },
          }),
        },
      );

      if (payload.job.status === "completed") {
        await refreshBootstrap(payload.job.model_id);
        setTrainingJobState({ kind: "completed", job: payload.job });
        return;
      }

      setTrainingJobState({ kind: "running", job: payload.job });
    } catch (error) {
      setTrainingJobState({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "Training job request failed.",
      });
    }
  }

  async function handleDeleteSelectedModel() {
    if (!selectedModel || selectedModel.kind !== "custom" || isDeletingModel) {
      return;
    }

    setIsDeletingModel(true);
    setModelDeleteFeedback(null);

    try {
      await fetchNoContent(`/api/models/${selectedModel.id}`, {
        method: "DELETE",
      });
      await refreshBootstrap();

      setModelDeleteFeedback(`Custom model deleted: ${selectedModel.name}.`);

      if (
        trainingJobState.kind === "completed" &&
        trainingJobState.job.model_id === selectedModel.id
      ) {
        setTrainingJobState({ kind: "idle" });
      }
    } catch (error) {
      setModelDeleteFeedback(
        `Delete failed: ${
          error instanceof Error
            ? error.message
            : "Custom model deletion failed."
        }`,
      );
    } finally {
      setIsDeletingModel(false);
    }
  }

  const canChooseModel =
    bootstrapState.kind === "ready" && bootstrapState.models.length > 0;
  const canStartTraining =
    bootstrapState.kind === "ready" &&
    trainingFile !== null &&
    trainingPreviewState.kind === "ready" &&
    customTrainingForm.modelName.trim().length > 0 &&
    trainingJobState.kind !== "submitting" &&
    trainingJobState.kind !== "running";
  const trainingWarnings = getCustomTrainingWarnings(customTrainingForm);
  const activeTrainingJobId =
    trainingJobState.kind === "running" ? trainingJobState.job.id : null;

  useEffect(() => {
    if (!activeTrainingJobId) {
      return;
    }

    let isActive = true;

    async function pollTrainingJob() {
      try {
        const payload = await fetchJson<TrainingJobPayload>(
          `/api/training/jobs/${activeTrainingJobId}`,
        );

        if (!isActive) {
          return;
        }

        if (payload.job.status === "completed") {
          await refreshBootstrap(payload.job.model_id);

          if (!isActive) {
            return;
          }

          setTrainingJobState({ kind: "completed", job: payload.job });
          return;
        }

        if (
          payload.job.status === "failed" ||
          payload.job.status === "cancelled"
        ) {
          setTrainingJobState({
            kind: "error",
            message:
              payload.job.error ??
              `${payload.job.model_name} did not complete successfully.`,
          });
          return;
        }

        setTrainingJobState({ kind: "running", job: payload.job });
      } catch (error) {
        if (!isActive) {
          return;
        }

        setTrainingJobState({
          kind: "error",
          message:
            error instanceof Error
              ? error.message
              : "Training status request failed.",
        });
      }
    }

    const intervalId = window.setInterval(() => {
      void pollTrainingJob();
    }, TRAINING_POLL_INTERVAL_MS);

    void pollTrainingJob();

    return () => {
      isActive = false;
      window.clearInterval(intervalId);
    };
  }, [activeTrainingJobId]);

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">MNIST model catalog</p>
          <h1>Inspect leaderboard metadata. Score a digit live.</h1>
          <p className="summary">
            This tracer bullet keeps the app bootable while exposing model
            metrics, dataset lineage, and detail metadata in the UI, then reuses
            the selected model for the existing prediction flow.
          </p>
        </div>

        <div className="hero-meta" aria-live="polite">
          <span
            className={`status-pill status-pill--${
              bootstrapState.kind === "ready"
                ? "ready"
                : bootstrapState.kind === "error"
                  ? "error"
                  : "loading"
            }`}
          >
            {readinessLabel}
          </span>
          <p className="hero-note">
            {bootstrapState.kind === "loading"
              ? "Checking backend status and loading model metadata."
              : bootstrapState.kind === "error"
                ? `The frontend could not reach the backend: ${bootstrapState.message}`
                : `${bootstrapState.models.length} model${
                    bootstrapState.models.length === 1 ? "" : "s"
                  } loaded for inspection and live scoring.`}
          </p>
        </div>
      </section>

      <section className="workspace-grid">
        <div className="workspace-primary">
          <section className="board-card">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Workspace mode</p>
                <h2>
                  {workspaceMode === "predict"
                    ? "Paint a digit sample"
                    : "Upload a labeled MNIST CSV"}
                </h2>
              </div>
              <div
                className="mode-toggle"
                role="group"
                aria-label="Workspace mode"
              >
                <button
                  className={`mode-toggle-button ${
                    workspaceMode === "predict"
                      ? "mode-toggle-button--active"
                      : ""
                  }`}
                  type="button"
                  onClick={() => setWorkspaceMode("predict")}
                >
                  Prediction mode
                </button>
                <button
                  className={`mode-toggle-button ${
                    workspaceMode === "training"
                      ? "mode-toggle-button--active"
                      : ""
                  }`}
                  type="button"
                  onClick={() => setWorkspaceMode("training")}
                >
                  Training mode
                </button>
              </div>
            </div>

            {workspaceMode === "predict" ? (
              <>
                <div className="control-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleClearBoard}
                  >
                    Clear board
                  </button>
                  <button
                    className="primary-button"
                    type="button"
                    disabled={
                      bootstrapState.kind !== "ready" ||
                      !selectedModelId ||
                      !hasInk ||
                      predictionState.kind === "loading"
                    }
                    onClick={() => {
                      void handlePredict();
                    }}
                  >
                    {predictionState.kind === "loading"
                      ? "Scoring digit"
                      : "Run prediction"}
                  </button>
                </div>

                <p className="status-copy board-copy">
                  Click or drag across the board. The frontend sends the raw
                  pixel grid and the backend performs centering and scale
                  normalization.
                </p>

                <div className="drawing-stage">
                  <div
                    className="drawing-frame"
                    onPointerLeave={() => setIsPainting(false)}
                  >
                    <div
                      className="drawing-grid"
                      role="grid"
                      aria-label="Digit drawing board"
                    >
                      {canvasPixels.map((pixel, index) => {
                        const row = Math.floor(index / DRAWING_GRID_SIZE);
                        const column = index % DRAWING_GRID_SIZE;

                        return (
                          <button
                            key={index}
                            aria-label={`Paint row ${row + 1} column ${column + 1}`}
                            aria-pressed={pixel > 0}
                            className={`pixel-button ${
                              pixel > 0 ? "pixel-button--active" : ""
                            }`}
                            type="button"
                            onClick={() => handlePaint(index)}
                            onPointerDown={(event) => {
                              event.preventDefault();
                              setIsPainting(true);
                              handlePaint(index);
                            }}
                            onPointerEnter={() => {
                              if (isPainting) {
                                handlePaint(index);
                              }
                            }}
                            onPointerUp={() => setIsPainting(false)}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                <p className="status-copy board-copy">
                  Upload a labeled CSV that matches the bundled MNIST training
                  schema, then preview how the default or adjusted split ratios
                  will divide the dataset before training starts.
                </p>

                <div className="training-intake-grid">
                  <div className="training-field training-field--wide">
                    <label className="field-label" htmlFor="training-csv">
                      Training CSV
                    </label>
                    <input
                      id="training-csv"
                      className="training-file-input"
                      type="file"
                      accept=".csv,text/csv"
                      onChange={handleTrainingFileChange}
                    />
                    <p className="helper-copy">
                      {trainingFile
                        ? `${trainingFile.name} selected for preview.`
                        : "Use the labeled training format: label followed by pixel0 through pixel783."}
                    </p>
                  </div>

                  <div className="training-split-grid">
                    <div className="training-field">
                      <label className="field-label" htmlFor="split-train">
                        Train split
                      </label>
                      <input
                        id="split-train"
                        className="training-input"
                        type="number"
                        min="0"
                        max="100"
                        step="1"
                        value={trainingSplit.train}
                        onChange={(event) => {
                          handleTrainingSplitChange(
                            "train",
                            event.target.value,
                          );
                        }}
                      />
                    </div>

                    <div className="training-field">
                      <label className="field-label" htmlFor="split-validation">
                        Validation split
                      </label>
                      <input
                        id="split-validation"
                        className="training-input"
                        type="number"
                        min="0"
                        max="100"
                        step="1"
                        value={trainingSplit.validation}
                        onChange={(event) => {
                          handleTrainingSplitChange(
                            "validation",
                            event.target.value,
                          );
                        }}
                      />
                    </div>

                    <div className="training-field">
                      <label className="field-label" htmlFor="split-test">
                        Test split
                      </label>
                      <input
                        id="split-test"
                        className="training-input"
                        type="number"
                        min="0"
                        max="100"
                        step="1"
                        value={trainingSplit.test}
                        onChange={(event) => {
                          handleTrainingSplitChange("test", event.target.value);
                        }}
                      />
                    </div>
                  </div>

                  <div className="training-config-grid">
                    <div className="training-field training-field--wide">
                      <label
                        className="field-label"
                        htmlFor="training-model-name"
                      >
                        Model name
                      </label>
                      <input
                        id="training-model-name"
                        className="training-input"
                        type="text"
                        value={customTrainingForm.modelName}
                        onChange={(event) => {
                          handleTrainingModelNameChange(event.target.value);
                        }}
                      />
                      <p className="helper-copy">
                        Use a unique name so each run lands in the shared
                        leaderboard as its own entry.
                      </p>
                    </div>

                    <div className="training-field">
                      <label className="field-label" htmlFor="training-seed">
                        Seed
                      </label>
                      <input
                        id="training-seed"
                        className="training-input"
                        type="number"
                        min="0"
                        max="1000000"
                        step="1"
                        value={customTrainingForm.seed}
                        onChange={(event) => {
                          handleTrainingSettingChange(
                            "seed",
                            event.target.value,
                          );
                        }}
                      />
                      <p className="helper-copy">
                        Saved with the model for reproducible split and
                        evaluation behavior.
                      </p>
                    </div>
                  </div>

                  <div className="training-config-grid">
                    <div className="training-field">
                      <label
                        className="field-label"
                        htmlFor="training-cap-per-label"
                      >
                        Examples per label cap
                      </label>
                      <input
                        id="training-cap-per-label"
                        className="training-input"
                        type="number"
                        min="1"
                        max="5000"
                        step="1"
                        value={customTrainingForm.maxExamplesPerLabel}
                        onChange={(event) => {
                          handleTrainingSettingChange(
                            "maxExamplesPerLabel",
                            event.target.value,
                          );
                        }}
                      />
                      <p className="helper-copy">
                        Recommended 4. Ceiling 5000 for the first portable
                        training slice.
                      </p>
                    </div>

                    <div className="training-field">
                      <label
                        className="field-label"
                        htmlFor="training-prototype-blend"
                      >
                        Prototype blend
                      </label>
                      <input
                        id="training-prototype-blend"
                        className="training-input"
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={customTrainingForm.prototypeBlend}
                        onChange={(event) => {
                          handleTrainingSettingChange(
                            "prototypeBlend",
                            event.target.value,
                          );
                        }}
                      />
                      <p className="helper-copy">
                        Recommended 0.35. Higher blend leans more on the shipped
                        reference prototype.
                      </p>
                    </div>

                    <div className="training-field">
                      <label
                        className="field-label"
                        htmlFor="training-temperature"
                      >
                        Confidence temperature
                      </label>
                      <input
                        id="training-temperature"
                        className="training-input"
                        type="number"
                        min="1"
                        max="40"
                        step="1"
                        value={customTrainingForm.temperature}
                        onChange={(event) => {
                          handleTrainingSettingChange(
                            "temperature",
                            event.target.value,
                          );
                        }}
                      />
                      <p className="helper-copy">
                        Recommended 18. Ceiling 40 to keep confidence scaling
                        readable.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="control-row">
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={!canPreviewTrainingSplit}
                    onClick={() => {
                      void handlePreviewTrainingSplit();
                    }}
                  >
                    {trainingPreviewState.kind === "loading"
                      ? "Previewing split"
                      : "Preview split"}
                  </button>
                  <button
                    className="primary-button"
                    type="button"
                    disabled={!canStartTraining}
                    onClick={() => {
                      void handleStartTraining();
                    }}
                  >
                    {trainingJobState.kind === "submitting"
                      ? "Starting training"
                      : trainingJobState.kind === "running"
                        ? "Training in progress"
                        : "Start training"}
                  </button>
                </div>

                {trainingWarnings.length > 0 ? (
                  <div className="warning-stack" aria-live="polite">
                    {trainingWarnings.map((warning) => (
                      <p className="warning-copy" key={warning}>
                        {warning}
                      </p>
                    ))}
                  </div>
                ) : null}

                {Math.abs(trainingSplitTotal - 100) > 0.001 ? (
                  <p className="status-copy">
                    Split ratios must total 100 before preview.
                  </p>
                ) : null}

                {trainingPreviewState.kind === "idle" ? (
                  <p className="status-copy">
                    Choose a labeled CSV and preview the train, validation, and
                    test counts before launching the first custom training flow.
                  </p>
                ) : null}

                {trainingPreviewState.kind === "error" ? (
                  <p className="status-copy">
                    CSV preview failed: {trainingPreviewState.message}
                  </p>
                ) : null}

                {trainingPreviewState.kind === "loading" ? (
                  <p className="status-copy">
                    Validating the CSV schema and calculating split sizes.
                  </p>
                ) : null}

                {trainingPreviewState.kind === "ready" ? (
                  <>
                    <dl className="details-grid details-grid--compact">
                      <div>
                        <dt>File</dt>
                        <dd>{trainingPreviewState.payload.file_name}</dd>
                      </div>
                      <div>
                        <dt>Examples</dt>
                        <dd>
                          {formatCount(
                            trainingPreviewState.payload.dataset.example_count,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>Features</dt>
                        <dd>
                          {formatCount(
                            trainingPreviewState.payload.dataset.feature_count,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>Labels</dt>
                        <dd>
                          {trainingPreviewState.payload.dataset.label_range.min}{" "}
                          to{" "}
                          {trainingPreviewState.payload.dataset.label_range.max}
                        </dd>
                      </div>
                    </dl>

                    <div className="preview-stat-grid" aria-live="polite">
                      <article className="preview-stat-card">
                        <p className="preview-stat-copy">
                          Train{" "}
                          {trainingPreviewState.payload.split.counts.train}{" "}
                          examples
                        </p>
                        <strong>
                          {formatPercent(
                            trainingPreviewState.payload.split.ratios.train,
                          )}
                        </strong>
                      </article>
                      <article className="preview-stat-card">
                        <p className="preview-stat-copy">
                          Validation{" "}
                          {trainingPreviewState.payload.split.counts.validation}{" "}
                          examples
                        </p>
                        <strong>
                          {formatPercent(
                            trainingPreviewState.payload.split.ratios
                              .validation,
                          )}
                        </strong>
                      </article>
                      <article className="preview-stat-card">
                        <p className="preview-stat-copy">
                          Test {trainingPreviewState.payload.split.counts.test}{" "}
                          examples
                        </p>
                        <strong>
                          {formatPercent(
                            trainingPreviewState.payload.split.ratios.test,
                          )}
                        </strong>
                      </article>
                    </div>
                  </>
                ) : null}

                {trainingPreviewState.kind === "ready" &&
                trainingJobState.kind === "idle" ? (
                  <p className="status-copy">
                    Preview is ready. Start training to create a custom model
                    entry without leaving the shared leaderboard screen.
                  </p>
                ) : null}

                {trainingJobState.kind === "submitting" ? (
                  <p className="status-copy">
                    Submitting the custom training job.
                  </p>
                ) : null}

                {trainingJobState.kind === "running" ? (
                  <div className="training-progress-card" aria-live="polite">
                    <div className="training-progress-row">
                      <p className="status-copy">
                        {trainingJobState.job.progress.stage}
                      </p>
                      <strong>
                        {Math.round(
                          trainingJobState.job.progress.percent * 100,
                        )}
                        %
                      </strong>
                    </div>
                    <div
                      aria-label="Training progress"
                      aria-valuemax={100}
                      aria-valuemin={0}
                      aria-valuenow={Math.round(
                        trainingJobState.job.progress.percent * 100,
                      )}
                      className="training-progress-bar"
                      role="progressbar"
                    >
                      <span
                        className="training-progress-fill"
                        style={{
                          width: `${Math.round(
                            trainingJobState.job.progress.percent * 100,
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                ) : null}

                {trainingJobState.kind === "completed" ? (
                  <p className="status-copy">
                    {trainingJobState.job.model_name} is now available in the
                    shared model list.
                  </p>
                ) : null}

                {trainingJobState.kind === "error" ? (
                  <p className="status-copy">
                    Custom training failed: {trainingJobState.message}
                  </p>
                ) : null}

                {modelDeleteFeedback ? (
                  <p className="status-copy">{modelDeleteFeedback}</p>
                ) : null}
              </>
            )}
          </section>

          <section
            className="leaderboard-panel"
            aria-labelledby="leaderboard-heading"
          >
            <div className="panel-heading panel-heading--tight">
              <div>
                <p className="panel-kicker">Model leaderboard</p>
                <h2 id="leaderboard-heading">Compare available entries</h2>
              </div>
              <div className="sort-control">
                <label className="field-label" htmlFor="leaderboard-sort">
                  Sort leaderboard by
                </label>
                <select
                  id="leaderboard-sort"
                  className="model-select"
                  disabled={bootstrapState.kind !== "ready"}
                  value={leaderboardMetric}
                  onChange={(event) => {
                    setLeaderboardMetric(
                      event.target.value as SortableMetricKey,
                    );
                  }}
                >
                  <option value="accuracy">Accuracy</option>
                  <option value="macro_f1">Macro F1</option>
                  <option value="avg_inference_ms">Inference latency</option>
                </select>
              </div>
            </div>

            {bootstrapState.kind === "ready" ? (
              <div className="leaderboard-table-frame">
                <table
                  className="leaderboard-table"
                  aria-label="Model leaderboard"
                >
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col">Accuracy</th>
                      <th scope="col">Macro F1</th>
                      <th scope="col">Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedModels.map((model, index) => {
                      const isSelected = model.id === selectedModelId;

                      return (
                        <tr
                          key={model.id}
                          className={`leaderboard-table-row ${
                            isSelected ? "leaderboard-table-row--selected" : ""
                          }`}
                          aria-selected={isSelected}
                          tabIndex={0}
                          onClick={() => {
                            setSelectedModelId(model.id);
                            setPredictionState({ kind: "idle" });
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              setSelectedModelId(model.id);
                              setPredictionState({ kind: "idle" });
                            }
                          }}
                        >
                          <th scope="row">
                            <div className="leaderboard-model-cell">
                              <span className="leaderboard-rank">
                                #{index + 1}
                              </span>
                              <div className="leaderboard-model-copy">
                                <span className="leaderboard-name">
                                  {model.name}
                                </span>
                                <span className="leaderboard-subline">
                                  {formatModelKind(model.kind)}
                                </span>
                              </div>
                            </div>
                          </th>
                          <td>
                            {formatMetric(model.metrics?.accuracy, "percent")}
                          </td>
                          <td>
                            {formatMetric(model.metrics?.macro_f1, "percent")}
                          </td>
                          <td>
                            {formatMetric(
                              model.metrics?.avg_inference_ms,
                              "ms",
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="status-copy">
                {bootstrapState.kind === "loading"
                  ? "The model leaderboard will populate after the backend finishes bootstrapping."
                  : "The model leaderboard is unavailable until the backend responds."}
              </p>
            )}
          </section>
        </div>

        <section className="prediction-card" aria-live="polite">
          <div className="model-picker-block">
            <label className="field-label" htmlFor="model-select">
              Model
            </label>
            <select
              id="model-select"
              className="model-select"
              disabled={!canChooseModel}
              value={canChooseModel ? selectedModelId : ""}
              onChange={(event) => {
                setSelectedModelId(event.target.value);
                setPredictionState({ kind: "idle" });
              }}
            >
              {!canChooseModel ? (
                <option value="">
                  {bootstrapState.kind === "loading"
                    ? "Loading models"
                    : bootstrapState.kind === "error"
                      ? "Backend unavailable"
                      : "No models available"}
                </option>
              ) : (
                bootstrapState.models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {`${model.name} (${formatModelKind(model.kind)})`}
                  </option>
                ))
              )}
            </select>
            {bootstrapState.kind === "ready" && selectedModel?.description ? (
              <p className="helper-copy">{selectedModel.description}</p>
            ) : null}
          </div>

          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Model details</p>
              <h2>{selectedModel?.name ?? "Select a model"}</h2>
            </div>
            {selectedModel ? (
              <div className="panel-actions">
                <span
                  className={`kind-badge ${
                    selectedModel.kind === "custom"
                      ? "kind-badge--custom"
                      : "kind-badge--built-in"
                  }`}
                >
                  {formatModelKind(selectedModel.kind)}
                </span>
                {selectedModel.kind === "custom" ? (
                  <button
                    className="secondary-button secondary-button--compact"
                    type="button"
                    disabled={isDeletingModel}
                    onClick={() => {
                      void handleDeleteSelectedModel();
                    }}
                  >
                    {isDeletingModel
                      ? "Deleting custom model"
                      : "Delete custom model"}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>

          {selectedModel ? (
            <>
              <dl className="details-grid details-grid--compact">
                <div>
                  <dt>Dataset source</dt>
                  <dd>{selectedModel.dataset?.source ?? "Pending"}</dd>
                </div>
                <div>
                  <dt>Trained at</dt>
                  <dd>{formatTimestamp(selectedModel.trained_at)}</dd>
                </div>
                <div>
                  <dt>Dataset sizes</dt>
                  <dd>{formatDatasetSizes(selectedModel.dataset)}</dd>
                </div>
                <div>
                  <dt>Input shape</dt>
                  <dd>
                    {selectedModel.dataset?.image_shape ??
                      formatInputShape(selectedModel.input)}
                  </dd>
                </div>
              </dl>

              {selectedModel.training ? (
                <section className="detail-block">
                  <div className="panel-heading panel-heading--tight">
                    <div>
                      <p className="panel-kicker">Training snapshot</p>
                      <h3>Saved config and seed</h3>
                    </div>
                  </div>
                  <dl className="details-grid details-grid--compact">
                    <div>
                      <dt>Seed</dt>
                      <dd>{selectedModel.training.seed ?? "Pending"}</dd>
                    </div>
                    <div>
                      <dt>Classifier</dt>
                      <dd>
                        {formatClassifierLabel(
                          selectedModel.training.config_snapshot?.classifier,
                        )}
                      </dd>
                    </div>
                    <div>
                      <dt>Uploaded file</dt>
                      <dd>
                        {selectedModel.training.config_snapshot?.file_name ??
                          "Pending"}
                      </dd>
                    </div>
                    <div>
                      <dt>Split</dt>
                      <dd>
                        {formatConfiguredSplit(
                          selectedModel.training.config_snapshot?.split,
                        )}
                      </dd>
                    </div>
                  </dl>
                </section>
              ) : null}

              <section className="detail-block">
                <div className="panel-heading panel-heading--tight">
                  <div>
                    <p className="panel-kicker">Metrics</p>
                    <h3>Leaderboard metrics</h3>
                  </div>
                </div>
                <dl className="details-grid details-grid--compact">
                  <div>
                    <dt>Accuracy</dt>
                    <dd>
                      {formatMetric(selectedModel.metrics?.accuracy, "percent")}
                    </dd>
                  </div>
                  <div>
                    <dt>Macro precision</dt>
                    <dd>
                      {formatMetric(
                        selectedModel.metrics?.macro_precision,
                        "percent",
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Macro recall</dt>
                    <dd>
                      {formatMetric(
                        selectedModel.metrics?.macro_recall,
                        "percent",
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Macro F1</dt>
                    <dd>
                      {formatMetric(selectedModel.metrics?.macro_f1, "percent")}
                    </dd>
                  </div>
                  <div>
                    <dt>Inference latency</dt>
                    <dd>
                      {formatMetric(
                        selectedModel.metrics?.avg_inference_ms,
                        "ms",
                      )}
                    </dd>
                  </div>
                </dl>
              </section>

              <section className="detail-block">
                <div className="panel-heading panel-heading--tight">
                  <div>
                    <p className="panel-kicker">Hyperparameters</p>
                    <h3>Reference configuration</h3>
                  </div>
                </div>
                <div className="tag-grid">
                  {Object.entries(selectedModel.hyperparameters ?? {}).map(
                    ([key, value]) => (
                      <div className="tag-card" key={key}>
                        <span className="tag-label">{formatKeyLabel(key)}</span>
                        <strong>{String(value)}</strong>
                      </div>
                    ),
                  )}
                </div>
              </section>

              <section className="detail-block">
                <div className="panel-heading panel-heading--tight">
                  <div>
                    <p className="panel-kicker">Confusion matrix</p>
                    <h3>Evaluation breakdown</h3>
                  </div>
                </div>
                <div className="matrix-frame">
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th scope="col">Actual\\Pred</th>
                        {renderDigitHeaders()}
                      </tr>
                    </thead>
                    <tbody>
                      {(selectedModel.evaluation?.confusion_matrix ?? []).map(
                        (row, rowIndex) => (
                          <tr key={rowIndex}>
                            <th scope="row">{rowIndex}</th>
                            {row.map((value, columnIndex) => (
                              <td key={columnIndex}>{value}</td>
                            ))}
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="detail-block">
                <div className="panel-heading panel-heading--tight">
                  <div>
                    <p className="panel-kicker">Sample predictions</p>
                    <h3>Held-out examples</h3>
                  </div>
                </div>
                <div className="sample-grid">
                  {(selectedModel.evaluation?.sample_predictions ?? []).map(
                    (sample, index) => (
                      <article
                        className="sample-card"
                        key={`${sample.label}-${index}`}
                      >
                        <span className="sample-chip">
                          Actual {sample.label} predicted {sample.predicted}
                        </span>
                        <strong>{formatConfidence(sample.confidence)}</strong>
                      </article>
                    ),
                  )}
                </div>
              </section>
            </>
          ) : (
            <p className="status-copy">
              Select a model from the leaderboard to inspect its metadata and
              evaluation details.
            </p>
          )}

          <div className="section-divider" />

          <div className="panel-heading panel-heading--tight">
            <div>
              <p className="panel-kicker">Prediction</p>
              <h3>Confidence profile</h3>
            </div>
          </div>

          {predictionState.kind === "idle" ? (
            <p className="status-copy">
              Paint a digit and run the selected model to see the predicted
              class and the full digit confidence spread.
            </p>
          ) : null}

          {predictionState.kind === "error" ? (
            <p className="status-copy">
              Prediction failed: {predictionState.message}
            </p>
          ) : null}

          {predictionState.kind === "loading" ? (
            <p className="status-copy">
              Normalizing the sketch and scoring it against the selected model.
            </p>
          ) : null}

          {predictionState.kind === "ready" ? (
            <>
              <p className="prediction-headline">
                {predictionState.payload.model.name} predicts{" "}
                {predictionState.payload.prediction.digit}
              </p>
              <div className="confidence-grid">
                {predictionState.payload.prediction.confidences.map(
                  (confidence, digit) => (
                    <div className="confidence-row" key={digit}>
                      <span className="confidence-digit">{digit}</span>
                      <div className="confidence-bar">
                        <span
                          className="confidence-fill"
                          style={{ width: `${Math.max(confidence * 100, 3)}%` }}
                        />
                      </div>
                      <span className="confidence-value">
                        {formatConfidence(confidence)}
                      </span>
                    </div>
                  ),
                )}
              </div>
            </>
          ) : null}
        </section>
      </section>
    </main>
  );
}

export default App;

function createBlankCanvas() {
  return Array.from({ length: DRAWING_PIXEL_COUNT }, () => 0);
}

async function fetchJson<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);

  if (!response.ok) {
    throw new Error(await readFetchErrorMessage(response, input));
  }

  return (await response.json()) as T;
}

async function fetchNoContent(input: string, init?: RequestInit) {
  const response = await fetch(input, init);

  if (!response.ok) {
    throw new Error(await readFetchErrorMessage(response, input));
  }
}

function formatConfidence(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function readTextFromFile(file: File) {
  if (typeof file.text === "function") {
    return file.text();
  }

  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }

      reject(new Error("Could not read the selected CSV file."));
    };
    reader.onerror = () => {
      reject(
        reader.error ?? new Error("Could not read the selected CSV file."),
      );
    };

    reader.readAsText(file);
  });
}

function toSplitRatio(value: number) {
  return value / 100;
}

function compareModels(
  leftModel: ModelSummary,
  rightModel: ModelSummary,
  metric: SortableMetricKey,
) {
  const leftValue = getMetricValue(leftModel, metric);
  const rightValue = getMetricValue(rightModel, metric);

  if (metric === "avg_inference_ms") {
    return (
      leftValue - rightValue || leftModel.name.localeCompare(rightModel.name)
    );
  }

  return (
    rightValue - leftValue || leftModel.name.localeCompare(rightModel.name)
  );
}

function getMetricValue(model: ModelSummary, metric: SortableMetricKey) {
  const value = model.metrics?.[metric];

  if (value == null) {
    return metric === "avg_inference_ms"
      ? Number.POSITIVE_INFINITY
      : Number.NEGATIVE_INFINITY;
  }

  return value;
}

function formatMetric(value: number | undefined, mode: "percent" | "ms") {
  if (value == null) {
    return "n/a";
  }

  if (mode === "ms") {
    return `${value.toFixed(1)} ms`;
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(value: string | undefined) {
  if (!value) {
    return "Pending";
  }

  return value.replace("T", " ").replace("Z", " UTC");
}

function formatDatasetSizes(dataset: ModelDataset | undefined) {
  if (!dataset) {
    return "Pending";
  }

  return [
    `Train ${formatCount(dataset.train_examples)}`,
    `Val ${formatCount(dataset.validation_examples)}`,
    `Test ${formatCount(dataset.test_examples)}`,
  ].join(" | ");
}

function formatCount(value: number | undefined) {
  if (value == null) {
    return "-";
  }

  return value.toLocaleString();
}

function formatInputShape(
  input:
    | {
        width: number;
        height: number;
        encoding?: string;
      }
    | undefined,
) {
  if (!input) {
    return "Pending";
  }

  return `${input.width}x${input.height}${
    input.encoding ? ` ${input.encoding}` : ""
  }`;
}

function formatKeyLabel(key: string) {
  return key
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatModelKind(kind: string) {
  return kind === "custom" ? "Custom" : "Built-in";
}

function formatClassifierLabel(classifier: string | undefined) {
  if (classifier === "reference-prototype") {
    return "Reference prototype";
  }

  return classifier ? formatKeyLabel(classifier) : "Pending";
}

function formatConfiguredSplit(
  split:
    | {
        train_ratio: number;
        validation_ratio: number;
        test_ratio: number;
      }
    | undefined,
) {
  if (!split) {
    return "Pending";
  }

  return [
    `Train ${formatPercent(split.train_ratio)}`,
    `Val ${formatPercent(split.validation_ratio)}`,
    `Test ${formatPercent(split.test_ratio)}`,
  ].join(" | ");
}

function getCustomTrainingWarnings(form: CustomTrainingForm) {
  const warnings: string[] = [];

  if (form.maxExamplesPerLabel > 1000) {
    warnings.push(
      "Large per-label caps can slow the reference training slice on classroom hardware.",
    );
  }

  if (form.prototypeBlend > 0.75) {
    warnings.push(
      "High prototype blend leans heavily on the shipped baseline and can hide dataset differences.",
    );
  }

  if (form.temperature > 30) {
    warnings.push(
      "Very high confidence temperature can make score gaps look sharper than the underlying distances.",
    );
  }

  return warnings;
}

async function readFetchErrorMessage(response: Response, input: string) {
  let message = `Request to ${input} failed with ${response.status}`;

  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) {
      message = payload.detail;
    }
  } catch {
    // Ignore malformed error payloads and keep the default message.
  }

  return message;
}

function renderDigitHeaders() {
  return Array.from({ length: 10 }, (_, digit) => (
    <th key={digit} scope="col">
      {digit}
    </th>
  ));
}
