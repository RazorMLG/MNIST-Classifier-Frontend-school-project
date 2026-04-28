import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("shows backend readiness from the health endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
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
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(screen.getByText("Checking backend")).toBeInTheDocument();

    expect(await screen.findByText("Backend ready")).toBeInTheDocument();

    expect(fetchMock).toHaveBeenCalledWith("/api/health");
    expect(
      screen.getByText(/O:\/Projects\/MNIST Projekat\/data/i),
    ).toBeInTheDocument();
  });
});
