import { useCallback, useEffect, useState } from "react";
import { apiClient } from "../api/client";
import { describeApiError } from "../api/errors";
import type { components } from "../types/api";

type ModelSummary = components["schemas"]["ModelSummary"];
type IncompatibleModelSummary = components["schemas"]["IncompatibleModelSummary"];
type DataType = components["schemas"]["CompatibilityResponse"]["data_type"];

export interface Compatibility {
  dataType: DataType;
  compatible: ModelSummary[];
  incompatible: IncompatibleModelSummary[];
}

/**
 * Wraps GET /api/models/registry and POST /api/models/compatible.
 * The registry loads once on mount; compatibility is checked on demand
 * once a dataset has been uploaded (data_id known).
 */
export function useModels() {
  const [models, setModels] = useState<ModelSummary[] | null>(null);
  const [compatibility, setCompatibility] = useState<Compatibility | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadRegistry() {
      setLoading(true);
      setError(null);
      const { data, error: fetchError } = await apiClient.GET("/api/models/registry", {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      if (fetchError) {
        setError(new Error(describeApiError(fetchError)));
      } else {
        setModels(data.models);
      }
      setLoading(false);
    }

    loadRegistry();
    return () => controller.abort();
  }, []);

  const checkCompatibility = useCallback(async (dataId: string) => {
    setError(null);
    const { data, error: fetchError } = await apiClient.POST("/api/models/compatible", {
      body: { data_id: dataId },
    });
    if (fetchError) {
      setError(new Error(describeApiError(fetchError)));
      return null;
    }
    const next: Compatibility = {
      dataType: data.data_type,
      compatible: data.compatible,
      incompatible: data.incompatible,
    };
    setCompatibility(next);
    return next;
  }, []);

  return { models, compatibility, loading, error, checkCompatibility };
}
