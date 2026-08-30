import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ResultsScreen } from "./ResultsScreen";
import type { components } from "../../types/api";

type TrainResponse = components["schemas"]["TrainResponse"];
type TrainedModelResponse = components["schemas"]["TrainedModelResponse"];

// One fixture per reference model (BUILD_SESSIONS.md Session 7's "done
// when": all four render correctly), shaped exactly like
// app/routes/training.py's _to_response output for that model_type.
const classifierResult: TrainedModelResponse = {
  model_key: "logistic_regression",
  model_name: "Logistic Regression",
  model_type: "classifier",
  metrics: {
    accuracy: 1.0,
    precision: 1.0,
    recall: 1.0,
    f1: 1.0,
    confusion_matrix: [
      [2, 0],
      [0, 2],
    ],
    labels: ["0", "1"],
  },
  hyperparameters: {},
  training_time_seconds: 0.01,
  n_features: 4,
  feature_importance: { feature_0: 0.5, feature_1: 0.3 },
  visualization_data: null,
  plot_data: null,
};

const clustererResult: TrainedModelResponse = {
  model_key: "kmeans",
  model_name: "K-Means",
  model_type: "clusterer",
  metrics: { silhouette_score: 0.8, davies_bouldin_score: 0.2, inertia: 12.3 },
  hyperparameters: { n_clusters: 2 },
  training_time_seconds: 0.02,
  n_features: 4,
  feature_importance: null,
  visualization_data: null,
  plot_data: {
    points: [
      [1, 2],
      [3, 4],
      [5, 6],
    ],
    labels: [0, 0, -1],
  },
};

const regressorResult: TrainedModelResponse = {
  model_key: "linear_regression",
  model_name: "Linear Regression",
  model_type: "regressor",
  metrics: { mse: 1.0, rmse: 1.0, r2: 0.9, mae: 0.8 },
  hyperparameters: {},
  training_time_seconds: 0.01,
  n_features: 10,
  feature_importance: { feature_0: 0.5 },
  visualization_data: null,
  plot_data: { y_true: [1, 2, 3], y_pred: [1.1, 1.9, 3.2] },
};

const reducerResult: TrainedModelResponse = {
  model_key: "pca",
  model_name: "PCA",
  model_type: "dimensionality_reducer",
  metrics: { explained_variance_ratio: [0.7, 0.2] },
  hyperparameters: { n_components: 2 },
  training_time_seconds: 0.005,
  n_features: 4,
  feature_importance: null,
  // PCA's own get_visualization_data() -- model-owned, separate from plot_data.
  visualization_data: { explained_variance_ratio: [0.7, 0.2] },
  plot_data: null,
};

function trainResponse(results: TrainedModelResponse[]): TrainResponse {
  return { training_id: "t1", data_id: "d1", test_size: 0.2, results };
}

describe("ResultsScreen", () => {
  it("shows a notice with no trained results", () => {
    render(<ResultsScreen results={null} />);
    expect(screen.getByText("Train at least one model to see results.")).toBeInTheDocument();
  });

  it("renders the confusion matrix for a classifier, without duplicating it as raw JSON", () => {
    render(<ResultsScreen results={trainResponse([classifierResult])} />);
    expect(screen.getByText("Actual \\ Predicted")).toBeInTheDocument();
    expect(screen.getByText("4 of 4 test samples correctly classified.")).toBeInTheDocument();
    expect(screen.queryByText("confusion_matrix")).not.toBeInTheDocument();
    expect(screen.getByText("Feature importance")).toBeInTheDocument();
  });

  it("renders the cluster scatter caption for a clusterer, keeping noise labelled", () => {
    render(<ResultsScreen results={trainResponse([clustererResult])} />);
    expect(screen.getByText("1 cluster · 1 noise point")).toBeInTheDocument();
  });

  it("renders the predicted-vs-actual caption for a regressor", () => {
    render(<ResultsScreen results={trainResponse([regressorResult])} />);
    expect(
      screen.getByText("3 test samples · dashed line marks a perfect prediction (predicted = actual)."),
    ).toBeInTheDocument();
  });

  it("renders the variance-plot caption for a dimensionality reducer, and its model-owned visualization_data", () => {
    render(<ResultsScreen results={trainResponse([reducerResult])} />);
    expect(screen.getByText("0.9000 cumulative variance explained")).toBeInTheDocument();
    expect(screen.getByText("Model-provided data")).toBeInTheDocument();
    expect(screen.getByText("[2 values]")).toBeInTheDocument();
  });

  it("renders all four model types together without crashing", () => {
    render(
      <ResultsScreen
        results={trainResponse([classifierResult, clustererResult, regressorResult, reducerResult])}
      />,
    );
    expect(screen.getByText("Logistic Regression")).toBeInTheDocument();
    expect(screen.getByText("K-Means")).toBeInTheDocument();
    expect(screen.getByText("Linear Regression")).toBeInTheDocument();
    expect(screen.getByText("PCA")).toBeInTheDocument();
  });
});
