import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CompareScreen } from "./CompareScreen";
import type { components } from "../../types/api";

type TrainResponse = components["schemas"]["TrainResponse"];

const { GET, POST } = vi.hoisted(() => ({ GET: vi.fn(), POST: vi.fn() }));

vi.mock("../../api/client", () => ({
  apiClient: { GET, POST },
}));

const trainingResults: TrainResponse = {
  training_id: "t1",
  data_id: "d1",
  test_size: 0.2,
  results: [
    {
      model_key: "logistic_regression",
      model_name: "Logistic Regression",
      model_type: "classifier",
      metrics: { accuracy: 0.9 },
      hyperparameters: {},
      training_time_seconds: 0.01,
      n_features: 4,
      feature_importance: null,
      visualization_data: null,
      plot_data: null,
    },
    {
      model_key: "other_classifier",
      model_name: "Other Classifier",
      model_type: "classifier",
      metrics: { accuracy: 0.8 },
      hyperparameters: {},
      training_time_seconds: 0.01,
      n_features: 4,
      feature_importance: null,
      visualization_data: null,
      plot_data: null,
    },
    {
      model_key: "kmeans",
      model_name: "K-Means",
      model_type: "clusterer",
      metrics: { silhouette_score: 0.5 },
      hyperparameters: {},
      training_time_seconds: 0.01,
      n_features: 4,
      feature_importance: null,
      visualization_data: null,
      plot_data: null,
    },
  ],
};

describe("CompareScreen", () => {
  it("shows a notice with no trained results", () => {
    render(<CompareScreen trainingResults={null} />);
    expect(screen.getByText("Train at least one model before comparing results.")).toBeInTheDocument();
  });

  it("explains a mixed-type selection instead of sending a request", async () => {
    const user = userEvent.setup();
    render(<CompareScreen trainingResults={trainingResults} />);

    await user.click(screen.getByText("Logistic Regression"));
    await user.click(screen.getByText("K-Means"));

    expect(
      screen.getByText(/These models aren't comparable: classifier, clusterer are different model types\./),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Compare selected models/ })).toBeDisabled();
    expect(POST).not.toHaveBeenCalled();
  });

  it("compares two same-type models and renders the metrics table", async () => {
    const user = userEvent.setup();
    POST.mockResolvedValueOnce({
      data: {
        common_metrics: ["accuracy"],
        models: [
          {
            training_id: "t1",
            model_key: "logistic_regression",
            model_name: "Logistic Regression",
            model_type: "classifier",
            metrics: { accuracy: 0.9 },
          },
          {
            training_id: "t1",
            model_key: "other_classifier",
            model_name: "Other Classifier",
            model_type: "classifier",
            metrics: { accuracy: 0.8 },
          },
        ],
      },
      error: undefined,
    });

    render(<CompareScreen trainingResults={trainingResults} />);

    await user.click(screen.getByText("Logistic Regression"));
    await user.click(screen.getByText("Other Classifier"));
    await user.click(screen.getByRole("button", { name: /Compare selected models/ }));

    await waitFor(() => expect(POST).toHaveBeenCalledTimes(1));
    expect(POST).toHaveBeenCalledWith("/api/results/comparison", {
      body: {
        models: [
          { training_id: "t1", model_key: "logistic_regression" },
          { training_id: "t1", model_key: "other_classifier" },
        ],
      },
    });

    // formatMetricValue renders a non-integer number to 4 decimal places
    // (frontend.md's mono-aligned numeric columns).
    expect(await screen.findByText("0.9000")).toBeInTheDocument();
    expect(screen.getByText("0.8000")).toBeInTheDocument();
  });
});
