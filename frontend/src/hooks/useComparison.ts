import { useCallback, useState } from "react";
import { apiClient } from "../api/client";
import { describeApiError } from "../api/errors";
import type { components } from "../types/api";

type ComparisonModelRef = components["schemas"]["ComparisonModelRef"];
type ComparisonResponse = components["schemas"]["ComparisonResponse"];

/**
 * Wraps POST /api/results/comparison (BUILD_SESSIONS.md Session 4). Refs
 * are cross-run by design -- {training_id, model_key} pairs may name the
 * same training run or different ones (app/schemas/results.py).
 */
export function useComparison() {
  const [result, setResult] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const compare = useCallback(async (models: ComparisonModelRef[]) => {
    setLoading(true);
    setError(null);
    const { data, error: fetchError } = await apiClient.POST("/api/results/comparison", {
      body: { models },
    });
    setLoading(false);
    if (fetchError) {
      setError(new Error(describeApiError(fetchError)));
      return null;
    }
    setResult(data);
    return data;
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, compare, reset };
}
