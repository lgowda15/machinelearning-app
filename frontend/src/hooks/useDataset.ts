import { useCallback, useEffect, useState } from "react";
import { apiClient } from "../api/client";
import { describeApiError } from "../api/errors";
import type { components } from "../types/api";

type DataProfileResponse = components["schemas"]["DataProfileResponse"];
type SampleDatasetInfo = components["schemas"]["SampleDatasetInfo"];
type UploadBody = components["schemas"]["Body_upload_data_api_data_upload_post"];

/**
 * Wraps the Stage 1 endpoints (DATA_FLOW_GUIDE.md SS2): upload a CSV or load
 * a shipped sample, both returning the same profile shape that powers EDA.
 * The sample list loads once on mount -- the Upload screen's secondary path.
 */
export function useDataset() {
  const [samples, setSamples] = useState<SampleDatasetInfo[] | null>(null);
  const [samplesError, setSamplesError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadSamples() {
      const { data, error: fetchError } = await apiClient.GET("/api/data/samples", {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      if (fetchError) {
        setSamplesError(new Error(describeApiError(fetchError)));
      } else {
        setSamples(data.samples);
      }
    }

    loadSamples();
    return () => controller.abort();
  }, []);

  const upload = useCallback(
    async (file: File, targetColumn: string | null, hasTarget: boolean) => {
      setLoading(true);
      setError(null);
      const { data, error: fetchError } = await apiClient.POST("/api/data/upload", {
        // Body_upload_data_api_data_upload_post types `file` as `string`
        // (OpenAPI's binary-format placeholder) -- openapi-typescript has no
        // better way to represent a File in a multipart schema. The cast
        // bridges that; bodySerializer below builds the real FormData from
        // the closed-over arguments, not from this typed-but-fictional body.
        body: { target_column: targetColumn, has_target: hasTarget } as unknown as UploadBody,
        bodySerializer: () => {
          const formData = new FormData();
          formData.append("file", file);
          if (targetColumn) formData.append("target_column", targetColumn);
          formData.append("has_target", String(hasTarget));
          return formData;
        },
      });
      setLoading(false);
      if (fetchError) {
        setError(new Error(describeApiError(fetchError)));
        return null;
      }
      return data;
    },
    [],
  );

  const loadSample = useCallback(async (sampleId: string) => {
    setLoading(true);
    setError(null);
    const { data, error: fetchError } = await apiClient.POST("/api/data/samples/{sample_id}", {
      params: { path: { sample_id: sampleId } },
    });
    setLoading(false);
    if (fetchError) {
      setError(new Error(describeApiError(fetchError)));
      return null;
    }
    return data;
  }, []);

  return { samples, samplesError, loading, error, upload, loadSample };
}

export type { DataProfileResponse };
