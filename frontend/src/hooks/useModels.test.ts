import { describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useModels } from "./useModels";

const { GET, POST } = vi.hoisted(() => ({ GET: vi.fn(), POST: vi.fn() }));

vi.mock("../api/client", () => ({
  apiClient: { GET, POST },
}));

describe("useModels", () => {
  it("loads the registry on mount", async () => {
    GET.mockResolvedValueOnce({
      data: {
        models: [
          {
            key: "ref_logistic",
            model_name: "Logistic Regression",
            model_type: "classifier",
            default_hyperparameters: {},
          },
        ],
      },
      error: undefined,
    });

    const { result } = renderHook(() => useModels());
    expect(result.current.loading).toBe(true);

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.models).toHaveLength(1);
    expect(result.current.error).toBeNull();
    expect(GET).toHaveBeenCalledWith("/api/models/registry", expect.anything());
  });

  it("surfaces a readable error when the registry request fails", async () => {
    GET.mockResolvedValueOnce({
      data: undefined,
      error: {
        error: true,
        code: "registry_unavailable",
        message: "Registry could not be loaded.",
        details: {},
      },
    });

    const { result } = renderHook(() => useModels());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.models).toBeNull();
    expect(result.current.error?.message).toBe("Registry could not be loaded.");
  });

  it("checkCompatibility posts the data_id and stores the response", async () => {
    GET.mockResolvedValueOnce({ data: { models: [] }, error: undefined });
    POST.mockResolvedValueOnce({
      data: {
        data_type: "classification",
        compatible: [],
        incompatible: [
          {
            key: "ref_kmeans",
            model_name: "K-Means",
            model_type: "clusterer",
            default_hyperparameters: {},
            reason: "Clusterers require unlabelled data.",
          },
        ],
      },
      error: undefined,
    });

    const { result } = renderHook(() => useModels());
    await waitFor(() => expect(result.current.loading).toBe(false));

    await result.current.checkCompatibility("data-123");

    expect(POST).toHaveBeenCalledWith("/api/models/compatible", {
      body: { data_id: "data-123" },
    });
    await waitFor(() =>
      expect(result.current.compatibility?.dataType).toBe("classification"),
    );
    expect(result.current.compatibility?.incompatible).toHaveLength(1);
  });
});
