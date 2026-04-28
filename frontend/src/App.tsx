import { useEffect, useState } from "react";

import "./App.css";

type HealthPayload = {
  status: string;
  service: string;
  storage: {
    ready: boolean;
    root: string;
    directories: string[];
  };
};

type ViewState =
  | { kind: "loading" }
  | { kind: "ready"; payload: HealthPayload }
  | { kind: "error"; message: string };

export function App() {
  const [viewState, setViewState] = useState<ViewState>({ kind: "loading" });

  useEffect(() => {
    let isActive = true;

    async function loadHealth() {
      try {
        const response = await fetch("/api/health");

        if (!response.ok) {
          throw new Error(`Backend responded with ${response.status}`);
        }

        const payload = (await response.json()) as HealthPayload;

        if (!isActive) {
          return;
        }

        setViewState({ kind: "ready", payload });
      } catch (error) {
        if (!isActive) {
          return;
        }

        setViewState({
          kind: "error",
          message: error instanceof Error ? error.message : "Backend check failed.",
        });
      }
    }

    void loadHealth();

    return () => {
      isActive = false;
    };
  }, []);

  const readinessLabel =
    viewState.kind === "ready" && viewState.payload.storage.ready
      ? "Backend ready"
      : viewState.kind === "error"
        ? "Backend unavailable"
        : "Checking backend";

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">MNIST shell</p>
        <h1>Bootable app shell</h1>
        <p className="summary">
          This first slice wires a Windows-friendly startup path, a Python backend status
          endpoint, and a React frontend that verifies the API is reachable.
        </p>
      </section>

      <section className="status-card" aria-live="polite">
        <div className="status-row">
          <span
            className={`status-pill status-pill--${
              viewState.kind === "ready"
                ? "ready"
                : viewState.kind === "error"
                  ? "error"
                  : "loading"
            }`}
          >
            {readinessLabel}
          </span>
          <span className="service-name">
            {viewState.kind === "ready" ? viewState.payload.service : "mnist-backend"}
          </span>
        </div>

        {viewState.kind === "loading" ? (
          <p className="status-copy">Checking backend status and storage bootstrap.</p>
        ) : null}

        {viewState.kind === "error" ? (
          <p className="status-copy">The frontend could not reach the backend: {viewState.message}</p>
        ) : null}

        {viewState.kind === "ready" ? (
          <>
            <p className="status-copy">
              The backend responded successfully and prepared the project-local storage
              layout for the next implementation slices.
            </p>
            <dl className="details-grid">
              <div>
                <dt>Storage root</dt>
                <dd>{viewState.payload.storage.root}</dd>
              </div>
              <div>
                <dt>Directories</dt>
                <dd>{viewState.payload.storage.directories.join(", ")}</dd>
              </div>
            </dl>
          </>
        ) : null}
      </section>
    </main>
  );
}

export default App;
