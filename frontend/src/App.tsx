import { useEffect, useState } from "react";

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

export function App() {
  const [bootstrapState, setBootstrapState] = useState<BootstrapState>({
    kind: "loading",
  });
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

        setBootstrapState({
          kind: "ready",
          health: healthPayload,
          models: modelsPayload.models,
        });

        setSelectedModelId((currentModelId) => {
          if (currentModelId) {
            return currentModelId;
          }

          return modelsPayload.models[0]?.id ?? "";
        });
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

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">MNIST shipped models</p>
        <h1>Inspect leaderboard metadata. Score a digit live.</h1>
        <p className="summary">
          This tracer bullet keeps the app bootable while exposing shipped model
          metrics, dataset lineage, and detail metadata in the UI, then reuses
          the selected built-in model for the existing prediction flow.
        </p>
      </section>

      <section className="workspace-grid">
        <section className="status-card" aria-live="polite">
          <div className="status-row">
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
            <span className="service-name">
              {bootstrapState.kind === "ready"
                ? bootstrapState.health.service
                : "mnist-backend"}
            </span>
          </div>

          {bootstrapState.kind === "loading" ? (
            <p className="status-copy">
              Checking backend status and loading shipped model metadata.
            </p>
          ) : null}

          {bootstrapState.kind === "error" ? (
            <p className="status-copy">
              The frontend could not reach the backend: {bootstrapState.message}
            </p>
          ) : null}

          {bootstrapState.kind === "ready" ? (
            <>
              <p className="status-copy">
                The backend is reachable, the shipped-model registry is online,
                and the leaderboard can drive both model inspection and live
                predictions.
              </p>
              <dl className="details-grid">
                <div>
                  <dt>Storage root</dt>
                  <dd>{bootstrapState.health.storage.root}</dd>
                </div>
                <div>
                  <dt>Directories</dt>
                  <dd>
                    {bootstrapState.health.storage.directories.join(", ")}
                  </dd>
                </div>
                <div>
                  <dt>Loaded models</dt>
                  <dd>{bootstrapState.models.length}</dd>
                </div>
              </dl>
              <label className="field-label" htmlFor="model-select">
                Built-in model
              </label>
              <select
                id="model-select"
                className="model-select"
                value={selectedModelId}
                onChange={(event) => {
                  setSelectedModelId(event.target.value);
                  setPredictionState({ kind: "idle" });
                }}
              >
                {bootstrapState.models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
              {selectedModel?.description ? (
                <p className="helper-copy">{selectedModel.description}</p>
              ) : null}

              <section className="leaderboard-panel" aria-labelledby="leaderboard-heading">
                <div className="panel-heading panel-heading--tight">
                  <div>
                    <p className="panel-kicker">Shipped leaderboard</p>
                    <h2 id="leaderboard-heading">Compare built-in entries</h2>
                  </div>
                  <div className="sort-control">
                    <label className="field-label" htmlFor="leaderboard-sort">
                      Sort leaderboard by
                    </label>
                    <select
                      id="leaderboard-sort"
                      className="model-select"
                      value={leaderboardMetric}
                      onChange={(event) => {
                        setLeaderboardMetric(event.target.value as SortableMetricKey);
                      }}
                    >
                      <option value="accuracy">Accuracy</option>
                      <option value="macro_f1">Macro F1</option>
                      <option value="avg_inference_ms">Inference latency</option>
                    </select>
                  </div>
                </div>

                <div
                  className="leaderboard-list"
                  role="list"
                  aria-label="Shipped model leaderboard"
                >
                  {sortedModels.map((model) => (
                    <button
                      key={model.id}
                      type="button"
                      className={`leaderboard-row ${
                        model.id === selectedModelId ? "leaderboard-row--selected" : ""
                      }`}
                      aria-pressed={model.id === selectedModelId}
                      onClick={() => {
                        setSelectedModelId(model.id);
                        setPredictionState({ kind: "idle" });
                      }}
                    >
                      <div className="leaderboard-name-row">
                        <span className="leaderboard-name">{model.name}</span>
                        <span className="kind-badge">Built-in</span>
                      </div>
                      <div className="leaderboard-metrics">
                        <span>
                          Accuracy {formatMetric(model.metrics?.accuracy, "percent")}
                        </span>
                        <span>
                          Macro F1 {formatMetric(model.metrics?.macro_f1, "percent")}
                        </span>
                        <span>
                          Latency {formatMetric(model.metrics?.avg_inference_ms, "ms")}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </section>
            </>
          ) : null}
        </section>

        <section className="board-card">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Input canvas</p>
              <h2>Paint a digit sample</h2>
            </div>
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
          </div>

          <p className="status-copy">
            Click or drag across the board. The frontend sends the raw pixel
            grid and the backend performs centering and scale normalization.
          </p>

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
        </section>

        <section className="prediction-card" aria-live="polite">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Model details</p>
              <h2>{selectedModel?.name ?? "Select a built-in model"}</h2>
            </div>
            {selectedModel ? <span className="kind-badge">Built-in</span> : null}
          </div>

          {selectedModel ? (
            <>
              {selectedModel.description ? (
                <p className="status-copy">{selectedModel.description}</p>
              ) : null}

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
                  <dd>
                    {formatDatasetSizes(selectedModel.dataset)}
                  </dd>
                </div>
                <div>
                  <dt>Input shape</dt>
                  <dd>
                    {selectedModel.dataset?.image_shape ??
                      formatInputShape(selectedModel.input)}
                  </dd>
                </div>
              </dl>

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
                    <dd>{formatMetric(selectedModel.metrics?.accuracy, "percent")}</dd>
                  </div>
                  <div>
                    <dt>Macro precision</dt>
                    <dd>
                      {formatMetric(selectedModel.metrics?.macro_precision, "percent")}
                    </dd>
                  </div>
                  <div>
                    <dt>Macro recall</dt>
                    <dd>{formatMetric(selectedModel.metrics?.macro_recall, "percent")}</dd>
                  </div>
                  <div>
                    <dt>Macro F1</dt>
                    <dd>{formatMetric(selectedModel.metrics?.macro_f1, "percent")}</dd>
                  </div>
                  <div>
                    <dt>Inference latency</dt>
                    <dd>
                      {formatMetric(selectedModel.metrics?.avg_inference_ms, "ms")}
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
                      <article className="sample-card" key={`${sample.label}-${index}`}>
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
              Select a shipped model from the leaderboard to inspect its
              metadata and evaluation details.
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
              Paint a digit and run the selected shipped model to see the
              predicted class and the full digit confidence spread.
            </p>
          ) : null}

          {predictionState.kind === "error" ? (
            <p className="status-copy">
              Prediction failed: {predictionState.message}
            </p>
          ) : null}

          {predictionState.kind === "loading" ? (
            <p className="status-copy">
              Normalizing the sketch and scoring it against the shipped model.
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
    let message = `Request to ${input} failed with ${response.status}`;

    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Ignore malformed error payloads and keep the default message.
    }

    throw new Error(message);
  }

  return (await response.json()) as T;
}

function formatConfidence(value: number) {
  return `${Math.round(value * 100)}%`;
}

function compareModels(
  leftModel: ModelSummary,
  rightModel: ModelSummary,
  metric: SortableMetricKey,
) {
  const leftValue = getMetricValue(leftModel, metric);
  const rightValue = getMetricValue(rightModel, metric);

  if (metric === "avg_inference_ms") {
    return leftValue - rightValue || leftModel.name.localeCompare(rightModel.name);
  }

  return rightValue - leftValue || leftModel.name.localeCompare(rightModel.name);
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

function renderDigitHeaders() {
  return Array.from({ length: 10 }, (_, digit) => (
    <th key={digit} scope="col">
      {digit}
    </th>
  ));
}
