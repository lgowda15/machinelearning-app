// The seven screens — frontend.md's layout contract.
export type StepId =
  | "upload"
  | "eda"
  | "model-selection"
  | "training"
  | "results"
  | "predict"
  | "compare";

export interface StepDefinition {
  id: StepId;
  label: string;
}

export const STEPS: StepDefinition[] = [
  { id: "upload", label: "Upload" },
  { id: "eda", label: "EDA" },
  { id: "model-selection", label: "Model selection" },
  { id: "training", label: "Training" },
  { id: "results", label: "Results" },
  { id: "predict", label: "Predict" },
  { id: "compare", label: "Compare" },
];

// The Start screen precedes step 1 and is deliberately outside the 1-7
// numbering (frontend.md) -- there's no run history to return to, so it
// isn't itself a "step" the indicator can land on.
export type View = "start" | StepId;
