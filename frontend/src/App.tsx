import { useEffect, useState } from "react";

import "./App.css";

const DRAWING_GRID_SIZE = 20;
const DRAWING_PIXEL_COUNT = DRAWING_GRID_SIZE * DRAWING_GRID_SIZE;

type HealthPayload = {
  status: string;
  service: string;
  storage: {
    ready: boolean;
    root: string;
    directories: string[];
  };
};

type ModelSummary = {
  id: string;
  name: string;
  kind: string;
  family?: string;
  description?: string;
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
        <p className="eyebrow">MNIST reference slice</p>
        <h1>Draw a digit. Score it end to end.</h1>
        <p className="summary">
          This tracer bullet keeps the shell bootable while adding one shipped
          reference model, raw drawing input, backend preprocessing, and a live
          confidence readout for digits 0 through 9.
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
              Checking backend status and loading the shipped reference model.
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
                and this slice can send raw drawing pixels into the prediction
                contract.
              </p>
              <dl className="details-grid">
                <div>
                  <dt>Storage root</dt>
                  <dd>{bootstrapState.health.storage.root}</dd>
                </div>
                <div>
                  <dt>Directories</dt>
                  <dd>{bootstrapState.health.storage.directories.join(", ")}</dd>
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
            <div className="drawing-grid" role="grid" aria-label="Digit drawing board">
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
              <p className="panel-kicker">Prediction</p>
              <h2>Confidence profile</h2>
            </div>
          </div>

          {predictionState.kind === "idle" ? (
            <p className="status-copy">
              Paint a digit and run the reference model to see the predicted
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
