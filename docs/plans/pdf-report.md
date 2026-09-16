# Download Report as PDF — Plan

Status: **approved, not yet implemented**. Frontend-only. No changes to
`backend/models/`, `base_model.py`, or `registry.py`.

Summarizes one full user cycle — upload → EDA → model selection/split →
training results → optional comparison → optional prediction — as a
downloadable PDF.

## Investigation findings

### 1. State ownership and lifetime

Routing is a single `switch` in `App.tsx` (lines 126-225) — exactly one screen
is mounted at a time. `StepShell` is the only thing that persists across the
whole session.

| State | Owner | Survives navigation? |
|---|---|---|
| Dataset profile (EDA, `data_type`, `class_imbalance`) | `profile` — App.tsx:34 | Yes (App-level) |
| Split ratio | `testSize` — App.tsx:35 | Yes |
| Selected model keys | `selectedModelKeys` — App.tsx:36 | Yes |
| Training results (metrics, metadata, `visualization_data`, `plot_data`, hyperparameters) | `trainingState.results` via `useTraining()` — App.tsx:39 | Yes |
| Comparison result + selected keys | `useComparison()` instantiated inside `CompareScreen` (CompareScreen.tsx:41-42) | **No** — dies on unmount |
| Prediction result + uploaded file/manual values | `usePrediction()` instantiated inside `PredictScreen` (PredictScreen.tsx:36,44,47) | **No** — dies on unmount |

`toggleModel` and the profile-change effect (App.tsx:57-83) call
`resetTraining()` on re-selection/re-upload. There is only ever one live
training run in memory, never a history.

### 2. Backend recoverability

- Dataset profile: `GET /api/data/{data_id}` — fully recoverable.
- Training results (metrics, metadata, viz data, hyperparameters, test_size,
  model keys): `GET /api/training/{training_id}/results` — fully recoverable;
  `useTraining.fetchResults` (useTraining.ts:46-60) already wraps this and is
  currently unused.
- Comparison: no GET-by-id, but fully re-derivable client-side from
  `trainingState.results` since Compare always draws from the one current
  `training_id`. Only the *which-models-were-checked* selection is
  frontend-only.
- Prediction: **not recoverable at all.** No GET endpoint; the uploaded file
  is consumed inline in the POST and never persisted server-side. Once
  `PredictScreen` unmounts, that prediction is gone from both frontend and
  backend.
- `PredictionResponse` (types/api.ts:434-447) already includes `model_key`,
  `model_type`, `training_id`, `n_samples`, `predictions`, `probabilities` —
  no backend change needed to know which model a prediction came from.
- `DataProfileResponse.source` (types/api.ts:332) is the dataset
  name/identifier to use as `datasetName`; `target_column` (types/api.ts:343)
  is the target.

### 3. Chart rendering inventory

| Component | Type | Color source |
|---|---|---|
| `DistributionChart`, `FeatureImportanceChart`, `ClusterScatterChart`, `PredictedVsActualChart`, `VariancePlotChart`, `ShapValuesChart` | Recharts, `ResponsiveContainer` | `"var(--color-...)"` strings via `chartTheme.ts` + `typeColorVar()` in `lib/modelType.ts` |
| `DendrogramChart` | Custom hand-rolled `<svg>` | Same `var(--color-...)` strings as raw SVG attributes |
| `ConfusionMatrixChart` | Plain HTML `<table>`, no SVG | Tailwind classes |
| `TreeStructureChart` | Custom, flexbox `<div>`s, not SVG | Tailwind classes |

All chart color resolves through the live CSSOM (`index.css`'s single
`@theme` block, no dark-mode variant). Off-screen/detached rendering can't
rely on `var(--...)` resolving correctly through the normal cascade.

### 4. Terminal screens

Step order (`types/steps.ts:16-24`): `upload → eda → model-selection →
training → results → predict → compare`. `isStepComplete` (App.tsx:85-90) has
no entries for `results`/`predict`/`compare`, and `StepIndicator` lets the
user jump freely between any already-reached step. **Results, Predict, and
Compare are all valid stopping points.**

## Final design decisions

### State lift

Lift `useComparison()` and `usePrediction()` out of `CompareScreen` /
`PredictScreen` into `App.tsx`, same pattern as `useModels`/`useTraining`.
Pass state + actions down as props instead of each screen instantiating its
own hook. This makes the report reflect the whole cycle regardless of which
screen the user is currently on, since prediction results have no backend
recovery path once their screen unmounts.

Additionally: whenever `resetTraining()` runs, or the profile changes,
**also clear comparison and prediction state**. A stale prediction or
comparison made against a previous training run must never survive into a
new one. Covered by a dedicated test (see Testing).

### `CycleState` (owned by `App`)

```ts
interface CycleState {
  dataId: string | null;
  profile: DataProfileResponse | null;
  testSize: number;
  selectedModelKeys: string[];
  trainingResults: TrainResponse | null;            // includes training_id
  comparison: { selectedKeys: string[] } | null;    // metrics table re-derived from trainingResults, not stored separately
  prediction: { inputSummary: string; result: PredictionResponse } | null; // result.model_key identifies which model was used
}
```

No new fetch layer is required to build the report — once lifted, everything
needed is already resident in `App`. `GET /api/training/{training_id}/results`
remains available as a defensive refetch only, not part of the normal path.

### `ReportModel`

```ts
interface ReportModel {
  summary: {
    datasetName: string;      // DataProfileResponse.source
    nRows: number;
    nCols: number;
    target: string | null;    // DataProfileResponse.target_column
    dataType: DataType;
    testSize: number;
    modelDisplayNames: string[];
    generatedAt: string;      // ISO
  };
  preprocessing: {
    // static section, not derived from live state — describes the fixed
    // pipeline: median/mode imputation, encoding, standard scaling, all
    // fit on the training split only.
  };
  dataset: {
    imbalance?: { isImbalanced: boolean; message?: string };
    columns: ColumnSummary[];       // capped at 20, with an "N more columns omitted" note when truncated
    distributionCharts: ChartSpec[];
  };
  models: {
    key: string;
    displayName: string;
    modelType: ModelType;
    hyperparameters: Record<string, unknown>;
    metrics: Record<string, number>;
    charts: ChartSpec[];             // zero or more per model, not a single chart
  }[];
  comparison: {
    modelKeys: string[];
    metricsTable: TableSpec;
  } | null;                          // omitted entirely if Compare was never visited
  prediction: {
    modelKey: string;                // from PredictionResponse.model_key
    inputSummary: string;
    totalRows: number;
    outputSample: TableSpec;         // capped at 25 rows
    predictedValueDistribution: TableSpec;
  } | null;                          // omitted entirely if Predict was never visited
}
```

`ChartSpec`/`TableSpec` are plain serializable descriptors (e.g.
`{kind:'bar', data, xKey, yKey}`), not React elements or DOM nodes — this is
what makes `buildReportModel` pure and unit-testable.

Sections for skipped steps are omitted from the model entirely (`null`), not
rendered as empty sections in the PDF.

### Tree chart (`TreeStructureChart`)

No SVG conversion. `buildReportModel` produces an autotable-friendly summary:
depth, node count, leaf count, and the first 3 levels of splits as indented
text rules, e.g.:

```
feature_3 <= 0.42 → feature_1 <= -0.8 → class 1
```

### Chart colors

No `reportColors.ts`, no color-override props on chart components. Instead:
after `captureChartSvg` grabs the rendered `<svg>` element, walk its
attributes/inline styles and replace every `var(--color-*)` occurrence with
its resolved value from `getComputedStyle(document.documentElement)` before
handing the SVG to `svg2pdf`. Chart components only gain optional
`width`/`height`/`isAnimationActive` props (needed to render off-screen
without `ResponsiveContainer`, which measures 0 outside real layout) — no
color-related props.

### `captureChartSvg`

Mounts the target chart component off-screen (`position:fixed;
left:-10000px`, explicit pixel width/height, `isAnimationActive={false}`)
using `react-dom/client`'s `createRoot`, and renders with **`flushSync`**
before reading the `<svg>` out of the detached container, to guarantee the
DOM is fully painted synchronously before capture. Unmounts and removes the
container afterward.

### Download button placement

One `DownloadReportButton` in `StepShell`, enabled once `trainingResults`
exists (i.e. as soon as Results is reachable), rather than a button
duplicated across Results/Compare/Predict. It reads whatever is currently in
the lifted `CycleState` regardless of which screen is mounted.

### Filename

`report_<dataset-name>_<YYYYMMDD-HHmm>.pdf`, e.g.
`report_iris_20260916-1432.pdf`. Dataset name is `DataProfileResponse.source`,
sanitized for filesystem safety (strip path separators/extension, replace
non-alphanumerics with `_`).

## `buildReportModel` / `renderReportPdf` split

**`buildReportModel(cycle: CycleState): ReportModel`** — pure, no DOM/React,
Vitest-tested.

**`renderReportPdf(model: ReportModel): Promise<Blob>`** — impure, dynamically
imported on click (`import('./renderReportPdf')`) so `jspdf`/`svg2pdf.js`/
`jspdf-autotable`/`react-dom` never ship in the main bundle:

1. Preprocessing section and summary header via `jspdf-autotable`.
2. For each `ChartSpec`: `captureChartSvg` renders the real chart component
   off-screen at fixed dimensions, resolves `var(--color-*)` to literal
   values, then `svg2pdf()` embeds it as vector.
3. `ConfusionMatrixChart` data and comparison/prediction tables go straight
   to `jspdf-autotable` — no SVG step.
4. Tree chart sections render as the indented-text split summary described
   above, via `jspdf`'s text API (not autotable, not SVG).

## File list

**New:**
- `frontend/src/lib/report/types.ts` — `ReportModel`, `ChartSpec`, `TableSpec`
- `frontend/src/lib/report/buildReportModel.ts`
- `frontend/src/lib/report/buildReportModel.test.ts`
- `frontend/src/lib/report/renderReportPdf.ts`
- `frontend/src/lib/report/captureChartSvg.ts`
- `frontend/src/lib/report/treeSummaryTable.ts`
- `frontend/src/lib/report/resolveCssVars.ts` — walks a captured `<svg>`, replaces `var(--color-*)` with `getComputedStyle` values
- `frontend/src/components/DownloadReportButton.tsx`

**Modified:**
- `frontend/src/App.tsx` — lift `useComparison`/`usePrediction`; clear both on `resetTraining()`/profile change; pass `CycleState` down as props
- `frontend/src/components/StepShell.tsx` — render `DownloadReportButton`, enabled once `trainingResults` exists
- `frontend/src/components/screens/CompareScreen.tsx` — consume lifted state instead of local hook instantiation
- `frontend/src/components/screens/PredictScreen.tsx` — consume lifted state instead of local hook instantiation
- `frontend/src/components/charts/{DistributionChart,FeatureImportanceChart,ClusterScatterChart,PredictedVsActualChart,VariancePlotChart,ShapValuesChart,DendrogramChart}.tsx` — optional `width`/`height`/`isAnimationActive` props only; on-screen default behavior unchanged
- `frontend/package.json` — add `jspdf`, `svg2pdf.js`, `jspdf-autotable`

No backend changes.

## Testing

`buildReportModel.test.ts` (Vitest, pure — no DOM):
- results-only cycle
- results + comparison
- results + prediction
- results + comparison + prediction (full cycle)
- imbalance present / absent
- column list truncation at 20 with the "N more omitted" note
- prediction sample capped at 25 rows, total row count and value distribution present
- **stale-prediction case**: build a cycle where a prediction/comparison was made against training run A, then simulate `resetTraining()` (or a profile change) firing before `buildReportModel` is called — assert the resulting `ReportModel.prediction` and `.comparison` are `null`, proving a stale prediction from a previous training run cannot appear in the report.

`renderReportPdf`: smoke test only — call with a minimal `ReportModel`,
assert it resolves to a `Blob` of type `application/pdf` without throwing.
Not pixel-tested.

## Implementation order

1. Lift `useComparison`/`usePrediction` into `App.tsx`; wire the
   reset-on-retrain/reset-on-reprofile clearing; write the stale-prediction
   test against the lifted state (can be a plain unit/RTL test on `App`'s
   reset logic, ahead of `buildReportModel` existing).
2. `buildReportModel` + full Vitest suite listed above (results-only,
   +compare, +predict, +both, imbalance present/absent, stale-prediction).
3. `renderReportPdf` (`captureChartSvg` with `flushSync`, `resolveCssVars`,
   tree summary, autotable sections) + smoke test.
4. `DownloadReportButton` wired into `StepShell`; add the three new
   dependencies to `package.json`.

## Done criteria

- All Vitest cases above pass, including the stale-prediction test.
- A generated PDF from a **classification run** is shown and reviewed.
- A generated PDF from a **clustering run** is shown and reviewed (exercises
  the imbalance-absent / no-target / different metrics path).
- No changes under `backend/models/`, `base_model.py`, or `registry.py`.
