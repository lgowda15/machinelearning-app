/**
 * The four-way model-type colour coding (.claude/rules/frontend.md) --
 * "use the matching type colour consistently everywhere that model's
 * results appear: its card border/accent in Model Selection, its chart
 * series in Results, its column header in Compare." One place these four
 * mappings live, so no screen invents its own.
 *
 * Class names are written out in full (not template-built) so Tailwind's
 * scanner sees them as static strings.
 */

export type ModelType = "classifier" | "clusterer" | "regressor" | "dimensionality_reducer";

export function isModelType(value: string): value is ModelType {
  return value === "classifier" || value === "clusterer" || value === "regressor" || value === "dimensionality_reducer";
}

const TYPE_COLOR_VAR: Record<ModelType, string> = {
  classifier: "var(--color-signal)",
  clusterer: "var(--color-type-cluster)",
  regressor: "var(--color-type-regress)",
  dimensionality_reducer: "var(--color-type-reduce)",
};

/** For chart fills/strokes, which take a CSS colour value, not a class. */
export function typeColorVar(modelType: string): string {
  return isModelType(modelType) ? TYPE_COLOR_VAR[modelType] : "var(--color-ink)";
}

const TYPE_CLASSES: Record<ModelType, { border: string; text: string }> = {
  classifier: { border: "border-signal", text: "text-signal" },
  clusterer: { border: "border-type-cluster", text: "text-type-cluster" },
  regressor: { border: "border-type-regress", text: "text-type-regress" },
  dimensionality_reducer: { border: "border-type-reduce", text: "text-type-reduce" },
};

export function typeBorderClass(modelType: string): string {
  return isModelType(modelType) ? TYPE_CLASSES[modelType].border : "border-ink";
}

export function typeTextClass(modelType: string): string {
  return isModelType(modelType) ? TYPE_CLASSES[modelType].text : "text-ink";
}

/**
 * EDA (screen 2) runs before any model is chosen, so there is no
 * `model_type` yet to key off -- only the detected `data_type`. A
 * classification target can only ever be trained by a classifier, and a
 * regression target only by a regressor, so those map directly; an
 * unlabelled (clustering) dataset has no target column to highlight at all.
 * This replaces the EDA screen's old always-signal target colour, which is
 * what produced the "target chart blue even for a regression target"
 * inconsistency BUILD_SESSIONS.md's Session 8 calls out.
 */
export function targetColorVarForDataType(dataType: string): string {
  if (dataType === "classification") return TYPE_COLOR_VAR.classifier;
  if (dataType === "regression") return TYPE_COLOR_VAR.regressor;
  return "var(--color-ink)";
}
