import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("shows backend readiness from the health endpoint", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (input === "/api/health") {
        return {
          ok: true,
          json: async () => ({
            status: "ok",
            service: "mnist-backend",
            storage: {
              ready: true,
              root: "O:/Projects/MNIST Projekat/data",
              directories: ["shipped-models", "custom-models", "registry"],
            },
          }),
        };
      }

      return {
        ok: true,
        json: async () => ({
          models: [
            {
              id: "reference-prototype-v1",
              name: "Reference Prototype",
              kind: "built-in",
            },
          ],
        }),
      };
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(screen.getByText("Checking backend")).toBeInTheDocument();

    expect(await screen.findByText("Backend ready")).toBeInTheDocument();

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/health", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/models", undefined);
    expect(
      screen.getByText(/O:\/Projects\/MNIST Projekat\/data/i),
    ).toBeInTheDocument();
  });

  it("submits a drawn digit to the selected built-in model", async () => {
    let predictRequest: unknown = null;

    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        if (input === "/api/health") {
          return {
            ok: true,
            json: async () => ({
              status: "ok",
              service: "mnist-backend",
              storage: {
                ready: true,
                root: "O:/Projects/MNIST Projekat/data",
                directories: ["shipped-models", "custom-models", "registry"],
              },
            }),
          };
        }

        if (input === "/api/models") {
          return {
            ok: true,
            json: async () => ({
              models: [
                {
                  id: "reference-prototype-v1",
                  name: "Reference Prototype",
                  kind: "built-in",
                  input: { width: 20, height: 20 },
                },
              ],
            }),
          };
        }

        if (input === "/api/predict") {
          predictRequest = JSON.parse(String(init?.body));

          return {
            ok: true,
            json: async () => ({
              model: {
                id: "reference-prototype-v1",
                name: "Reference Prototype",
              },
              prediction: {
                digit: 1,
                confidences: [
                  0.01, 0.72, 0.05, 0.03, 0.03, 0.03, 0.03, 0.04, 0.03, 0.03,
                ],
              },
            }),
          };
        }

        throw new Error(`Unexpected fetch call: ${String(input)}`);
      },
    );

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(
      await screen.findByRole("combobox", { name: /built-in model/i }),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /paint row 8 column 10/i }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: /paint row 9 column 10/i }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: /paint row 10 column 10/i }),
    );

    fireEvent.click(screen.getByRole("button", { name: /run prediction/i }));

    expect(
      await screen.findByText(/reference prototype predicts 1/i),
    ).toBeInTheDocument();
    expect(predictRequest).toMatchObject({
      model_id: "reference-prototype-v1",
      canvas: { width: 20, height: 20 },
    });
  });

  it("renders a sortable leaderboard and model details from shipped metadata", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (input === "/api/health") {
        return {
          ok: true,
          json: async () => ({
            status: "ok",
            service: "mnist-backend",
            storage: {
              ready: true,
              root: "O:/Projects/MNIST Projekat/data",
              directories: ["shipped-models", "custom-models", "registry"],
            },
          }),
        };
      }

      if (input === "/api/models") {
        return {
          ok: true,
          json: async () => ({
            models: [
              {
                id: "reference-prototype-v1",
                name: "Reference Prototype",
                kind: "built-in",
                description: "First shipped reference slice.",
                trained_at: "2026-04-28T00:00:00Z",
                metrics: {
                  accuracy: 0.912,
                  macro_precision: 0.914,
                  macro_recall: 0.912,
                  macro_f1: 0.911,
                  avg_inference_ms: 3.4,
                },
                dataset: {
                  source: "MNIST benchmark split",
                  image_shape: "28x28 grayscale",
                  train_examples: 60000,
                  validation_examples: 5000,
                  test_examples: 10000,
                },
                hyperparameters: {
                  prototype_grid_size: 20,
                  distance_metric: "stroke-overlap",
                },
                evaluation: {
                  confusion_matrix: Array.from({ length: 10 }, (_, row) =>
                    Array.from({ length: 10 }, (_, column) =>
                      row === column ? 90 : 1,
                    ),
                  ),
                  sample_predictions: [
                    { label: 1, predicted: 1, confidence: 0.98 },
                  ],
                },
                input: { width: 20, height: 20 },
              },
              {
                id: "baseline-challenger-v1",
                name: "Baseline Challenger",
                kind: "built-in",
                description: "Comparator metadata for leaderboard sorting.",
                trained_at: "2026-04-27T00:00:00Z",
                metrics: {
                  accuracy: 0.905,
                  macro_precision: 0.926,
                  macro_recall: 0.919,
                  macro_f1: 0.923,
                  avg_inference_ms: 5.2,
                },
                dataset: {
                  source: "MNIST augmented benchmark",
                  image_shape: "28x28 grayscale",
                  train_examples: 62000,
                  validation_examples: 4000,
                  test_examples: 10000,
                },
                hyperparameters: {
                  prototype_grid_size: 24,
                  distance_metric: "center-weighted",
                },
                evaluation: {
                  confusion_matrix: Array.from({ length: 10 }, (_, row) =>
                    Array.from({ length: 10 }, (_, column) =>
                      row === column ? 92 : 0,
                    ),
                  ),
                  sample_predictions: [
                    { label: 6, predicted: 5, confidence: 0.54 },
                  ],
                },
                input: { width: 20, height: 20 },
              },
            ],
          }),
        };
      }

      throw new Error(`Unexpected fetch call: ${String(input)}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    const leaderboard = await screen.findByRole("list", {
      name: /shipped model leaderboard/i,
    });

    expect(within(leaderboard).getAllByText(/built-in/i)).toHaveLength(2);
    expect(within(leaderboard).getAllByRole("button")[0]).toHaveTextContent(
      /reference prototype/i,
    );

    fireEvent.change(screen.getByRole("combobox", { name: /sort leaderboard by/i }), {
      target: { value: "macro_f1" },
    });

    expect(within(leaderboard).getAllByRole("button")[0]).toHaveTextContent(
      /baseline challenger/i,
    );

    fireEvent.click(
      within(leaderboard).getByRole("button", {
        name: /baseline challenger/i,
      }),
    );

    expect(screen.getByText(/mnist augmented benchmark/i)).toBeInTheDocument();
    expect(screen.getByText(/2026-04-27/i)).toBeInTheDocument();
    expect(screen.getByText(/confusion matrix/i)).toBeInTheDocument();
    expect(screen.getByText(/sample predictions/i)).toBeInTheDocument();
  });
});
