import { fireEvent, render, screen } from "@testing-library/react";
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

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
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
              confidences: [0.01, 0.72, 0.05, 0.03, 0.03, 0.03, 0.03, 0.04, 0.03, 0.03],
            },
          }),
        };
      }

      throw new Error(`Unexpected fetch call: ${String(input)}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(await screen.findByRole("combobox", { name: /built-in model/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /paint row 8 column 10/i }));
    fireEvent.click(screen.getByRole("button", { name: /paint row 9 column 10/i }));
    fireEvent.click(screen.getByRole("button", { name: /paint row 10 column 10/i }));

    fireEvent.click(screen.getByRole("button", { name: /run prediction/i }));

    expect(await screen.findByText(/reference prototype predicts 1/i)).toBeInTheDocument();
    expect(predictRequest).toMatchObject({
      model_id: "reference-prototype-v1",
      canvas: { width: 20, height: 20 },
    });
  });
});
