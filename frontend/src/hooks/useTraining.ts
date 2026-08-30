import { useCallback, useState } from "react";
import { apiClient } from "../api/client";
import { describeApiError } from "../api/errors";
import type { components } from "../types/api";

type ModelTrainSpec = components["schemas"]["ModelTrainSpec"];
type TrainResponse = components["schemas"]["TrainResponse"];

export interface TrainRequestArgs {
  dataId: string;
  models: ModelTrainSpec[];
  testSize?: number;
}

/**
 * Wraps POST /api/training/train (synchronous — ARCHITECTURE.md §6) and
 * GET /api/training/{id}/results for re-fetching a past run.
 */
export function useTraining() {
  const [results, setResults] = useState<TrainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const train = useCallback(async ({ dataId, models, testSize }: TrainRequestArgs) => {
    setLoading(true);
    setError(null);
    // TrainRequest.test_size carries a backend default (0.2, ARCHITECTURE.md
    // §6/§7) but openapi-typescript's defaultNonNullable generates it as
    // required, so mirror the default here rather than omitting the field.
    const { data, error: fetchError } = await apiClient.POST("/api/training/train", {
      body: {
        data_id: dataId,
        models,
        test_size: testSize ?? 0.2,
      },
    });
    setLoading(false);
    if (fetchError) {
      setError(new Error(describeApiError(fetchError)));
      return null;
    }
    setResults(data);
    return data;
  }, []);

  const fetchResults = useCallback(async (trainingId: string) => {
    setLoading(true);
    setError(null);
    const { data, error: fetchError } = await apiClient.GET(
      "/api/training/{training_id}/results",
      { params: { path: { training_id: trainingId } } },
    );
    setLoading(false);
    if (fetchError) {
      setError(new Error(describeApiError(fetchError)));
      return null;
    }
    setResults(data);
    return data;
  }, []);

  // Clears a past run when the upstream selection (dataset or chosen models)
  // changes, so a stale result is never shown against a new choice.
  const reset = useCallback(() => {
    setResults(null);
    setError(null);
  }, []);

  return { results, loading, error, train, fetchResults, reset };
}
