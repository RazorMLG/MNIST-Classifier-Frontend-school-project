import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

function makeTrainingCsv(exampleCount = 10) {
  const header = [
    "label",
    ...Array.from({ length: 784 }, (_, index) => `pixel${index}`),
  ];
  const rows = [header.join(",")];

  for (let label = 0; label < exampleCount; label += 1) {
    const pixels = Array.from({ length: 784 }, (_, index) =>
      String((label + index) % 256),
    );

    rows.push([String(label % 10), ...pixels].join(","));
  }

  return rows.join("\n");
}

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

    expect(screen.getByText(/checking backend status/i)).toBeInTheDocument();

    expect(await screen.findByText("Backend ready")).toBeInTheDocument();

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/health", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/models", undefined);
    expect(
      screen.getByText(/1 model loaded for inspection/i),
    ).toBeInTheDocument();
  });

  it("submits a drawn digit to the selected model", async () => {
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
      await screen.findByRole("combobox", { name: /^model$/i }),
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

    const leaderboard = await screen.findByRole("table", {
      name: /model leaderboard/i,
    });

    expect(within(leaderboard).getAllByText(/built-in/i)).toHaveLength(2);
    expect(within(leaderboard).getAllByRole("row")[1]).toHaveTextContent(
      /reference prototype/i,
    );

    fireEvent.change(
      screen.getByRole("combobox", { name: /sort leaderboard by/i }),
      {
        target: { value: "macro_f1" },
      },
    );

    expect(within(leaderboard).getAllByRole("row")[1]).toHaveTextContent(
      /baseline challenger/i,
    );

    fireEvent.click(within(leaderboard).getByText(/baseline challenger/i));

    expect(screen.getByText(/mnist augmented benchmark/i)).toBeInTheDocument();
    expect(screen.getByText(/2026-04-27/i)).toBeInTheDocument();
    expect(screen.getByText(/confusion matrix/i)).toBeInTheDocument();
    expect(screen.getByText(/sample predictions/i)).toBeInTheDocument();
  });

  it("replaces the canvas with CSV intake and previews an adjusted split", async () => {
    let previewRequest: unknown = null;

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
                },
              ],
            }),
          };
        }

        if (input === "/api/training/csv-preview") {
          previewRequest = JSON.parse(String(init?.body));

          return {
            ok: true,
            json: async () => ({
              file_name: "train.csv",
              dataset: {
                example_count: 10,
                feature_count: 784,
                label_range: { min: 0, max: 9 },
              },
              split: {
                ratios: {
                  train: 0.7,
                  validation: 0.2,
                  test: 0.1,
                },
                counts: {
                  train: 7,
                  validation: 2,
                  test: 1,
                },
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
      await screen.findByRole("button", { name: /training mode/i }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /training mode/i }));

    expect(
      screen.queryByRole("grid", { name: /digit drawing board/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText(/train split/i)).toHaveValue(80);
    expect(screen.getByLabelText(/validation split/i)).toHaveValue(10);
    expect(screen.getByLabelText(/test split/i)).toHaveValue(10);

    fireEvent.change(screen.getByLabelText(/train split/i), {
      target: { value: "70" },
    });
    fireEvent.change(screen.getByLabelText(/validation split/i), {
      target: { value: "20" },
    });

    fireEvent.change(screen.getByLabelText(/training csv/i), {
      target: {
        files: [
          new File([makeTrainingCsv()], "train.csv", { type: "text/csv" }),
        ],
      },
    });

    fireEvent.click(screen.getByRole("button", { name: /preview split/i }));

    expect(await screen.findByText(/train 7/i)).toBeInTheDocument();
    expect(screen.getByText(/validation 2/i)).toBeInTheDocument();
    expect(screen.getByText(/test 1/i)).toBeInTheDocument();
    expect(previewRequest).toMatchObject({
      file_name: "train.csv",
      split: {
        train_ratio: 0.7,
        validation_ratio: 0.2,
        test_ratio: 0.1,
      },
    });
  });

  it("shows backend CSV validation feedback in training mode", async () => {
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
                },
              ],
            }),
          };
        }

        if (input === "/api/training/csv-preview") {
          const requestBody = JSON.parse(String(init?.body));

          expect(requestBody.file_name).toBe("bad-train.csv");

          return {
            ok: false,
            status: 400,
            json: async () => ({
              detail: "Row 2 label must be between 0 and 9.",
            }),
          };
        }

        throw new Error(`Unexpected fetch call: ${String(input)}`);
      },
    );

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    fireEvent.click(
      await screen.findByRole("button", { name: /training mode/i }),
    );
    fireEvent.change(screen.getByLabelText(/training csv/i), {
      target: {
        files: [
          new File([makeTrainingCsv(1)], "bad-train.csv", { type: "text/csv" }),
        ],
      },
    });

    fireEvent.click(screen.getByRole("button", { name: /preview split/i }));

    expect(
      await screen.findByText(
        /csv preview failed: row 2 label must be between 0 and 9/i,
      ),
    ).toBeInTheDocument();
  });

  it("runs custom training through the shared leaderboard and model details flow", async () => {
    let trainingCompleted = false;
    let modelDeleted = false;
    let jobPollCount = 0;
    let trainingRequest: unknown = null;
    let deletedModelPath = "";

    const builtInModel = {
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
          Array.from({ length: 10 }, (_, column) => (row === column ? 90 : 1)),
        ),
        sample_predictions: [{ label: 1, predicted: 1, confidence: 0.98 }],
      },
      input: { width: 20, height: 20 },
    };

    const customModel = {
      id: "custom-classroom-prototype",
      name: "Classroom Prototype",
      kind: "custom",
      description:
        "Custom prototype classifier trained from an uploaded MNIST CSV.",
      trained_at: "2026-04-29T12:00:00Z",
      metrics: {
        accuracy: 0.954,
        macro_precision: 0.953,
        macro_recall: 0.954,
        macro_f1: 0.952,
        avg_inference_ms: 1.8,
      },
      dataset: {
        source: "Uploaded CSV: classroom-train.csv",
        image_shape: "28x28 grayscale",
        train_examples: 24,
        validation_examples: 3,
        test_examples: 3,
      },
      hyperparameters: {
        max_examples_per_label: 4,
        prototype_blend: 0.35,
        temperature: 18,
      },
      training: {
        seed: 17,
        config_snapshot: {
          classifier: "reference-prototype",
          file_name: "classroom-train.csv",
          split: {
            train_ratio: 0.8,
            validation_ratio: 0.1,
            test_ratio: 0.1,
          },
          hyperparameters: {
            max_examples_per_label: 4,
            prototype_blend: 0.35,
            temperature: 18,
          },
        },
      },
      evaluation: {
        confusion_matrix: Array.from({ length: 10 }, (_, row) =>
          Array.from({ length: 10 }, (_, column) => (row === column ? 12 : 0)),
        ),
        sample_predictions: [{ label: 7, predicted: 7, confidence: 0.99 }],
      },
      input: { width: 20, height: 20 },
    };

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
              models: modelDeleted
                ? [builtInModel]
                : trainingCompleted
                  ? [builtInModel, customModel]
                  : [builtInModel],
            }),
          };
        }

        if (input === "/api/training/csv-preview") {
          return {
            ok: true,
            json: async () => ({
              file_name: "classroom-train.csv",
              dataset: {
                example_count: 30,
                feature_count: 784,
                label_range: { min: 0, max: 9 },
              },
              split: {
                ratios: {
                  train: 0.8,
                  validation: 0.1,
                  test: 0.1,
                },
                counts: {
                  train: 24,
                  validation: 3,
                  test: 3,
                },
              },
            }),
          };
        }

        if (input === "/api/training/jobs") {
          trainingRequest = JSON.parse(String(init?.body));

          return {
            ok: true,
            status: 202,
            json: async () => ({
              job: {
                id: "job-1",
                model_id: "custom-classroom-prototype",
                model_name: "Classroom Prototype",
                status: "running",
                progress: {
                  percent: 0.08,
                  stage: "Validating dataset",
                },
                error: null,
                started_at: "2026-04-29T12:00:00Z",
                completed_at: null,
              },
            }),
          };
        }

        if (input === "/api/training/jobs/job-1") {
          jobPollCount += 1;

          if (jobPollCount === 1) {
            return {
              ok: true,
              json: async () => ({
                job: {
                  id: "job-1",
                  model_id: "custom-classroom-prototype",
                  model_name: "Classroom Prototype",
                  status: "running",
                  progress: {
                    percent: 0.6,
                    stage: "Building digit prototypes",
                  },
                  error: null,
                  started_at: "2026-04-29T12:00:00Z",
                  completed_at: null,
                },
              }),
            };
          }

          trainingCompleted = true;

          return {
            ok: true,
            json: async () => ({
              job: {
                id: "job-1",
                model_id: "custom-classroom-prototype",
                model_name: "Classroom Prototype",
                status: "completed",
                progress: {
                  percent: 1,
                  stage: "Training complete",
                },
                error: null,
                started_at: "2026-04-29T12:00:00Z",
                completed_at: "2026-04-29T12:00:03Z",
              },
            }),
          };
        }

        if (
          input === "/api/models/custom-classroom-prototype" &&
          init?.method === "DELETE"
        ) {
          deletedModelPath = String(input);
          modelDeleted = true;

          return {
            ok: true,
            status: 204,
            json: async () => ({}),
          };
        }

        throw new Error(`Unexpected fetch call: ${String(input)}`);
      },
    );

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    fireEvent.click(
      await screen.findByRole("button", { name: /training mode/i }),
    );

    fireEvent.change(screen.getByLabelText(/model name/i), {
      target: { value: "Classroom Prototype" },
    });
    fireEvent.change(screen.getByLabelText(/training csv/i), {
      target: {
        files: [
          new File([makeTrainingCsv(30)], "classroom-train.csv", {
            type: "text/csv",
          }),
        ],
      },
    });

    fireEvent.click(screen.getByRole("button", { name: /preview split/i }));

    expect(await screen.findByText(/train 24 examples/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingRequest).toMatchObject({
        model_name: "Classroom Prototype",
        seed: 17,
        hyperparameters: {
          max_examples_per_label: 4,
          prototype_blend: 0.35,
          temperature: 18,
        },
      });
    });

    expect(
      await screen.findByText(/building digit prototypes/i),
    ).toBeInTheDocument();
    expect(
      await screen.findByText(
        /classroom prototype is now available in the shared model list/i,
      ),
    ).toBeInTheDocument();

    const leaderboard = screen.getByRole("table", {
      name: /model leaderboard/i,
    });

    expect(
      within(leaderboard).getByText(/classroom prototype/i),
    ).toBeInTheDocument();
    expect(within(leaderboard).getByText(/^custom$/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /delete custom model/i }),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /delete custom model/i }),
    );

    expect(deletedModelPath).toBe("/api/models/custom-classroom-prototype");
    expect(
      await screen.findByText(/custom model deleted: classroom prototype/i),
    ).toBeInTheDocument();
    expect(
      within(leaderboard).queryByText(/classroom prototype/i),
    ).not.toBeInTheDocument();
  });
});
