import { useCallback, useState } from "react";
import { apiClient } from "../api/client";
import { describeApiError } from "../api/errors";
import type { components } from "../types/api";

type PredictionResponse = components["schemas"]["PredictionResponse"];
type PredictBody = components["schemas"]["Body_predict_api_prediction_predict_post"];

/**
 * Wraps POST /api/prediction/predict (Stage 6, DATA_FLOW_GUIDE.md §7):
 * run an already-trained model on freshly uploaded data. A column mismatch
 * is also rejected here (backend.md), as defense in depth behind the
 * screen 6 client-side check -- see lib/columns.ts.
 */
export function usePrediction() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const predict = useCallback(
    async (trainingId: string, modelKey: string, file: File) => {
      setLoading(true);
      setError(null);
      const { data, error: fetchError } = await apiClient.POST("/api/prediction/predict", {
        // See useDataset.upload's identical cast/bodySerializer pattern --
        // openapi-typescript has no better way to type a File in a
        // multipart schema than the OpenAPI binary-format placeholder.
        body: { training_id: trainingId, model_key: modelKey } as unknown as PredictBody,
        bodySerializer: () => {
          const formData = new FormData();
          formData.append("training_id", trainingId);
          formData.append("model_key", modelKey);
          formData.append("file", file);
          return formData;
        },
      });
      setLoading(false);
      if (fetchError) {
        setError(new Error(describeApiError(fetchError)));
        return null;
      }
      setResult(data);
      return data;
    },
    [],
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, predict, reset };
}
