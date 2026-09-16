import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import type { DataProfileResponse } from "./hooks/useDataset";
import type { components } from "./types/api";

type TrainResponse = components["schemas"]["TrainResponse"];

const { GET, POST } = vi.hoisted(() => ({ GET: vi.fn(), POST: vi.fn() }));

vi.mock("./api/client", () => ({
  apiClient: { GET, POST },
}));

// Every screen below is stubbed to a minimal control surface. The point of
// this test is App's own reset wiring -- which callback clears which piece
// of lifted state -- not any individual screen's rendering, which each has
// its own test file already.

vi.mock("./components/screens/StartScreen", () => ({
  StartScreen: ({ onBegin }: { onBegin: () => void }) => (
    <button onClick={onBegin}>Begin</button>
  ),
}));

function profile(source: string): DataProfileResponse {
  return {
    data_id: source,
    source,
    n_rows: 10,
    n_columns: 2,
    data_type: "classification",
    target_column: "target",
    columns: [],
  };
}

vi.mock("./components/screens/UploadScreen", () => ({
  UploadScreen: ({
    onProfile,
  }: {
    onProfile: (p: DataProfileResponse) => void;
  }) => (
    <div>
      <button onClick={() => onProfile(profile("a.csv"))}>
        set-profile-a
      </button>
      <button onClick={() => onProfile(profile("b.csv"))}>
        set-profile-b
      </button>
    </div>
  ),
}));

vi.mock("./components/screens/EdaScreen", () => ({
  EdaScreen: () => <div>eda</div>,
}));

vi.mock("./components/screens/ModelSelectionScreen", () => ({
  ModelSelectionScreen: ({
    onToggle,
  }: {
    onToggle: (key: string) => void;
  }) => <button onClick={() => onToggle("model_a")}>toggle-model-a</button>,
}));

vi.mock("./components/screens/TrainingScreen", () => ({
  TrainingScreen: ({
    dataId,
    selectedModelKeys,
    trainingState,
  }: {
    dataId: string;
    selectedModelKeys: string[];
    trainingState: { train: (args: unknown) => void };
  }) => (
    <button
      onClick={() =>
        trainingState.train({
          dataId,
          models: selectedModelKeys.map((key) => ({ model_key: key })),
        })
      }
    >
      run-train
    </button>
  ),
}));

vi.mock("./components/screens/ResultsScreen", () => ({
  ResultsScreen: () => <div>results</div>,
}));

vi.mock("./components/screens/PredictScreen", () => ({
  PredictScreen: ({
    trainingResults,
    predictionState,
  }: {
    trainingResults: TrainResponse | null;
    predictionState: {
      result: { model_key: string } | null;
      predict: (trainingId: string, modelKey: string, file: File) => void;
    };
  }) => (
    <div>
      <p>
        predict-result:{" "}
        {predictionState.result ? predictionState.result.model_key : "none"}
      </p>
      <button
        onClick={() =>
          predictionState.predict(
            trainingResults!.training_id,
            "model_a",
            new File(["a\n1"], "x.csv"),
          )
        }
      >
        run-predict
      </button>
    </div>
  ),
}));

vi.mock("./components/screens/CompareScreen", () => ({
  CompareScreen: ({
    trainingResults,
    comparisonState,
  }: {
    trainingResults: TrainResponse | null;
    comparisonState: {
      result: { common_metrics: string[] } | null;
      compare: (models: unknown) => void;
    };
  }) => (
    <div>
      <p>
        compare-result:{" "}
        {comparisonState.result
          ? comparisonState.result.common_metrics.join(",")
          : "none"}
      </p>
      <button
        onClick={() =>
          comparisonState.compare([
            { training_id: trainingResults!.training_id, model_key: "model_a" },
            { training_id: trainingResults!.training_id, model_key: "model_b" },
          ])
        }
      >
        run-compare
      </button>
    </div>
  ),
}));

function trainResponse(dataId: string): TrainResponse {
  return {
    training_id: `t-${dataId}`,
    data_id: dataId,
    test_size: 0.2,
    results: [
      {
        model_key: "model_a",
        model_name: "Model A",
        model_type: "classifier",
        metrics: { accuracy: 0.9 },
        hyperparameters: {},
        training_time_seconds: 0.01,
        n_features: 2,
        feature_importance: null,
        visualization_data: null,
        plot_data: null,
      },
      {
        model_key: "model_b",
        model_name: "Model B",
        model_type: "classifier",
        metrics: { accuracy: 0.8 },
        hyperparameters: {},
        training_time_seconds: 0.01,
        n_features: 2,
        feature_importance: null,
        visualization_data: null,
        plot_data: null,
      },
    ],
  };
}

function mockApiClient() {
  GET.mockResolvedValue({ data: { models: [] }, error: undefined });

  POST.mockImplementation(async (url: string) => {
    if (url === "/api/models/compatible") {
      return {
        data: { data_type: "classification", compatible: [], incompatible: [] },
        error: undefined,
      };
    }
    if (url === "/api/training/train") {
      return { data: trainResponse("a.csv"), error: undefined };
    }
    if (url === "/api/prediction/predict") {
      return {
        data: {
          training_id: "t-a.csv",
          model_key: "model_a",
          model_type: "classifier",
          n_samples: 1,
          predictions: [1],
          probabilities: null,
        },
        error: undefined,
      };
    }
    if (url === "/api/results/comparison") {
      return {
        data: {
          common_metrics: ["accuracy"],
          models: [
            {
              training_id: "t-a.csv",
              model_key: "model_a",
              model_name: "Model A",
              model_type: "classifier",
              metrics: { accuracy: 0.9 },
            },
            {
              training_id: "t-a.csv",
              model_key: "model_b",
              model_name: "Model B",
              model_type: "classifier",
              metrics: { accuracy: 0.8 },
            },
          ],
        },
        error: undefined,
      };
    }
    throw new Error(`unexpected POST ${url}`);
  });
}

/** Clicks the nth (0-indexed) step button in the header's progress list. */
async function jumpToStep(user: ReturnType<typeof userEvent.setup>, index: number) {
  const stepList = screen.getByRole("list", { name: "Progress" });
  const buttons = within(stepList).getAllByRole("button");
  await user.click(buttons[index]);
}

/** Waits for the training run to land (Continue only re-enables past the
 * Training step once trainingState.results is non-null), then advances. */
async function runTrainAndContinue(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByText("run-train"));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Continue" })).not.toBeDisabled(),
  );
  await user.click(screen.getByRole("button", { name: "Continue" })); // -> results
}

/** Drives a fresh App from Start through Compare, producing a trained run
 * plus a prediction and a comparison, all against that one run. */
async function driveFullCycleWithResults(user: ReturnType<typeof userEvent.setup>) {
  render(<App />);

  await user.click(screen.getByText("Begin"));
  await user.click(screen.getByText("set-profile-a"));
  await user.click(screen.getByRole("button", { name: "Continue" })); // -> eda
  await user.click(screen.getByRole("button", { name: "Continue" })); // -> model-selection
  await user.click(screen.getByText("toggle-model-a"));
  await user.click(screen.getByRole("button", { name: "Continue" })); // -> training

  await runTrainAndContinue(user);
  await user.click(screen.getByRole("button", { name: "Continue" })); // -> predict

  await user.click(screen.getByText("run-predict"));
  await waitFor(() => expect(screen.getByText("predict-result: model_a")).toBeInTheDocument());

  await user.click(screen.getByRole("button", { name: "Continue" })); // -> compare

  await user.click(screen.getByText("run-compare"));
  await waitFor(() => expect(screen.getByText("compare-result: accuracy")).toBeInTheDocument());
}

describe("App reset wiring", () => {
  it("clears a prediction and comparison made against a training run once that run is invalidated by a model-selection change", async () => {
    mockApiClient();
    const user = userEvent.setup();

    await driveFullCycleWithResults(user);

    // Changing the model selection invalidates the run the prediction and
    // comparison above were made against -- both must be cleared, not just
    // the training result.
    await jumpToStep(user, 2); // model-selection
    await user.click(screen.getByText("toggle-model-a")); // deselect
    await user.click(screen.getByText("toggle-model-a")); // reselect, so Continue re-enables

    await user.click(screen.getByRole("button", { name: "Continue" })); // -> training

    // Re-training is required before Predict/Compare are reachable again --
    // trainingState.results was cleared along with the prediction/comparison.
    await runTrainAndContinue(user);
    await user.click(screen.getByRole("button", { name: "Continue" })); // -> predict

    expect(screen.getByText("predict-result: none")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Continue" })); // -> compare

    expect(screen.getByText("compare-result: none")).toBeInTheDocument();
  });

  it("clears a prediction and comparison made against a training run once a new dataset is uploaded", async () => {
    mockApiClient();
    const user = userEvent.setup();

    await driveFullCycleWithResults(user);

    // Re-uploading a dataset invalidates every downstream run -- go back to
    // Upload and load a different profile.
    await jumpToStep(user, 0); // upload
    await user.click(screen.getByText("set-profile-b"));

    // handleProfile resets maxStepIndexReached to 0, so the whole cycle
    // (including a fresh training run) must be redone before Predict/
    // Compare are reachable again.
    await user.click(screen.getByRole("button", { name: "Continue" })); // -> eda
    await user.click(screen.getByRole("button", { name: "Continue" })); // -> model-selection
    await user.click(screen.getByText("toggle-model-a"));
    await user.click(screen.getByRole("button", { name: "Continue" })); // -> training

    await runTrainAndContinue(user);
    await user.click(screen.getByRole("button", { name: "Continue" })); // -> predict

    expect(screen.getByText("predict-result: none")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Continue" })); // -> compare

    expect(screen.getByText("compare-result: none")).toBeInTheDocument();
  });
});
