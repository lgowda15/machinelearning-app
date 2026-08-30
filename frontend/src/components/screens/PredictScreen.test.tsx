import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PredictScreen } from "./PredictScreen";
import type { DataProfileResponse } from "../../hooks/useDataset";
import type { components } from "../../types/api";

type TrainResponse = components["schemas"]["TrainResponse"];
type ColumnSummary = components["schemas"]["ColumnSummary"];

const { GET, POST } = vi.hoisted(() => ({ GET: vi.fn(), POST: vi.fn() }));

vi.mock("../../api/client", () => ({
  apiClient: { GET, POST },
}));

function column(name: string, isTarget = false): ColumnSummary {
  return {
    name,
    dtype: "numeric",
    is_target: isTarget,
    missing_count: 0,
    missing_pct: 0,
    unique_count: 5,
    distribution: { kind: "numeric", bin_edges: [0, 1], counts: [1] },
  };
}

const profile: DataProfileResponse = {
  data_id: "d1",
  source: "train.csv",
  n_rows: 100,
  n_columns: 3,
  data_type: "classification",
  target_column: "target",
  columns: [column("a"), column("b"), column("target", true)],
};

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
      n_features: 2,
      feature_importance: null,
      visualization_data: null,
      plot_data: null,
    },
  ],
};

function csvFile(content: string): File {
  return new File([content], "new.csv", { type: "text/csv" });
}

describe("PredictScreen", () => {
  it("shows a notice with no trained results", () => {
    render(<PredictScreen profile={profile} trainingResults={null} />);
    expect(
      screen.getByText("Train at least one model before predicting on new data."),
    ).toBeInTheDocument();
  });

  it("names missing and unexpected columns before sending any request", async () => {
    const user = userEvent.setup();
    render(<PredictScreen profile={profile} trainingResults={trainingResults} />);

    const input = document.getElementById("csv-upload") as HTMLInputElement;
    await user.upload(input, csvFile("a,extra\n1,2"));

    expect(await screen.findByText("This file's columns don't match the training data.")).toBeInTheDocument();
    expect(screen.getByText("b")).toBeInTheDocument(); // missing
    expect(screen.getByText("extra")).toBeInTheDocument(); // unexpected
    expect(screen.getByRole("button", { name: "Predict" })).toBeDisabled();
    expect(POST).not.toHaveBeenCalled();
  });

  it("predicts once the columns match and renders the predictions", async () => {
    const user = userEvent.setup();
    POST.mockResolvedValueOnce({
      data: {
        training_id: "t1",
        model_key: "logistic_regression",
        model_type: "classifier",
        n_samples: 2,
        predictions: [0, 1],
        probabilities: [
          [0.9, 0.1],
          [0.2, 0.8],
        ],
      },
      error: undefined,
    });

    render(<PredictScreen profile={profile} trainingResults={trainingResults} />);

    const input = document.getElementById("csv-upload") as HTMLInputElement;
    await user.upload(input, csvFile("a,b\n1,2\n3,4"));

    await waitFor(() =>
      expect(screen.queryByText("This file's columns don't match the training data.")).not.toBeInTheDocument(),
    );

    await user.click(screen.getByRole("button", { name: "Predict" }));

    await waitFor(() => expect(POST).toHaveBeenCalledTimes(1));
    expect(await screen.findByText("2 predictions")).toBeInTheDocument();
  });

  it("names the missing fields in manual mode before sending any request", async () => {
    const user = userEvent.setup();
    // POST isn't reset between tests in this file -- clear the call an
    // earlier test made so ".not.toHaveBeenCalled()" below is about this
    // action, not the accumulated total.
    POST.mockClear();
    render(<PredictScreen profile={profile} trainingResults={trainingResults} />);

    await user.click(screen.getByRole("button", { name: "Enter values" }));
    await user.type(screen.getByLabelText(/^a /), "1");
    await user.click(screen.getByRole("button", { name: "Predict" }));

    expect(
      screen.getByText("Enter a value for every feature before predicting."),
    ).toBeInTheDocument();
    expect(screen.getByText(/Missing:/)).toHaveTextContent("b"); // named as missing
    expect(POST).not.toHaveBeenCalled();
  });

  it("predicts in manual mode once every field is filled", async () => {
    const user = userEvent.setup();
    // POST isn't reset between tests in this file (matching the existing
    // pattern above) -- clear the call count this earlier test left behind
    // so this test's own assertion below is about this action, not the total.
    POST.mockClear();
    POST.mockResolvedValueOnce({
      data: {
        training_id: "t1",
        model_key: "logistic_regression",
        model_type: "classifier",
        n_samples: 1,
        predictions: [1],
        probabilities: null,
      },
      error: undefined,
    });

    render(<PredictScreen profile={profile} trainingResults={trainingResults} />);

    await user.click(screen.getByRole("button", { name: "Enter values" }));
    await user.type(screen.getByLabelText(/^a /), "1");
    await user.type(screen.getByLabelText(/^b /), "2");
    await user.click(screen.getByRole("button", { name: "Predict" }));

    await waitFor(() => expect(POST).toHaveBeenCalledTimes(1));
    expect(await screen.findByText("1 prediction")).toBeInTheDocument();
  });
});
