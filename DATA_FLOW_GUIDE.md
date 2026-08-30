# Data Flow Guide — from upload to prediction

Audience: the integration team, and anyone you're walking through the system. Purpose: trace exactly what happens to a dataset from the moment a user uploads it to the moment they see a result — what gets converted, what gets handed to a model, what comes back, and what we do with it.

Read this alongside ARCHITECTURE.md (the full stack) and CODING_STANDARDS.md (what the 12 groups must produce). This document is the thread connecting them: it explains the conversions in between.

---

## 1. The one-sentence version

We convert whatever the user uploads into one fixed shape before any model sees it, and we convert whatever a model returns into one fixed shape before the user sees it. Both conversions happen in the backend. Models never see raw data, and the frontend never sees raw model output.

Everything below is the detail behind that sentence.

---

## 2. Stage 1 — Upload → raw table

Input: a CSV file from the user, arbitrary columns, arbitrary types.

What we do:

- Parse the CSV into a pandas DataFrame.
- Reject the upload if it fails basic sanity checks: wrong format, fewer than 50 rows, more than 100 columns, no numeric columns at all.
- Identify the target column — either named by the user, or defaulted to the last column.
- Infer a data type for the problem itself:
  - target is categorical with 2–10 unique values → classification
  - target is continuous (more than 10 unique values) → regression
  - no target given → clustering

Output of this stage: a raw DataFrame plus a `data_type` label. Nothing has been cleaned yet. This is what powers the EDA screen — summary statistics, missing-value counts, distribution charts — computed directly on the raw table so the user sees their data as it actually is before we touch it.

---

## 3. Stage 2 — Raw table → model-ready array

This is the conversion every model depends on, and it is the one place in the whole system where "input format" gets decided once, for everyone.

What we do, in order:

1. **Split.** Train/test split at the ratio the user chose (default 80/20), stratified for classification so class proportions are preserved in both halves.
2. **Impute.** Missing values filled — numeric columns with the median, categorical columns with the mode. The imputer is fit on the training split only.
3. **Encode.** Categorical columns converted to numeric (label or one-hot, depending on cardinality). Same rule: fit on train, apply to test.
4. **Scale.** Numeric columns standardized (zero mean, unit variance) with a scaler fit on the training split only.
5. **Cast.** The result converted to a numpy array, dtype float64.

Why fit-on-train-only matters: if we fit the scaler or encoder on the combined train+test data, information from the test set leaks into training indirectly (the model "knows" the test set's distribution before it's evaluated). That inflates every metric we report and is exactly the mistake the coding standards forbid group models from making internally — we don't make it either, at the layer above them.

Output of this stage: `X_train`, `X_test` — both 2D float64 numpy arrays, same number of columns, no missing values, fully numeric, scaled. `y_train`, `y_test` as 1D arrays (or `None` for clustering).

This is exactly the object every one of the 12 models receives. Nothing upstream of this point is model-specific. Nothing downstream of this point touches raw data again.

---

## 4. Stage 3 — Handing data to a model

The user has picked one or more models from the (compatibility-filtered) list. For each:

1. Look up the class in the registry (`app/core/registry.py`).
2. Instantiate it, passing any hyperparameter overrides the user supplied (JSON-safe types only — see the coding standards).
3. Call `model.fit(X_train, y_train)`.
4. Call `model.predict(X_test)` and, if the model is a classifier, `model.predict_proba(X_test)`.
5. Call `model.get_metadata()` and, if present, `model.get_visualization_data()`.

Every model receives the identical `X_train`/`X_test` produced in Stage 2. We never branch this step by algorithm family — the whole point of the `BaseModel` contract is that this code doesn't know or care whether it's calling K-Means or an RNN.

The one exception: sequence and image groups (RNN, CNN) receive the same 2D array but reshape it internally, according to the column layout they document in their own README. We don't reshape for them — we just guarantee the 2D array arrives clean, and trust their fit/predict to interpret it as they've documented.

---

## 5. Stage 4 — Converting what comes back

A model hands us back three things, and each is used differently.

### 5.1 Predictions (`predict`, `predict_proba`)

Raw output: a 1D array (labels, cluster ids, or floats) and, for classifiers, a probability matrix.

We convert this into the metrics the user actually sees, and which metrics we compute depends entirely on the `model_type` string from `get_metadata()`:

| `model_type` | Metrics computed from predictions |
|---|---|
| classifier | accuracy, precision, recall, F1, confusion matrix |
| clusterer | silhouette score, Davies–Bouldin index, inertia |
| regressor | MSE, RMSE, R², MAE |
| dimensionality_reducer | explained variance ratio |

This is why the metadata contract is strict — `model_type` is not documentation, it's a switch statement. Get it wrong and the backend computes the wrong metrics, or crashes trying to compute a confusion matrix on continuous output.

### 5.2 Metadata (`get_metadata`)

Used two ways:

- **Display** — model name, hyperparameters, training time shown in the results view.
- **Routing** — `model_type` drives the metrics switch above; `feature_importance`, if present, renders as a bar chart alongside the metrics.

### 5.3 Visualization data (`get_visualization_data`, optional)

Only present for groups 1, 2, 9, 15. Whatever JSON-safe structure they return is handed to a matching frontend chart component we build for that specific shape (SHAP bar chart, dendrogram, variance plot, tree diagram). This is the one place output format is not uniform across models — by design, since these are genuinely different visual artifacts. Everything else in this document is about forcing uniformity; this is the deliberate exception.

---

## 6. Stage 5 — Class imbalance note

Computed once, right after Stage 1 (on the raw target column, classification datasets only): if any class is below 20% or above 80% of samples, the EDA response carries a flag, and the frontend shows an informational line — never a blocker, never shown again after that first screen.

This sits outside the model pipeline entirely. Models never see this flag and never adjust their behavior because of it.

---

## 7. Stage 6 — Prediction on new data

When a user later uploads a fresh CSV to get predictions from an already-trained model, it goes through the same Stage 2 conversion — impute, encode, scale — but using the encoder/scaler objects already fit during that model's original training, not refit on the new data. Then straight to `model.predict()`. Same contract, same shape guarantees, just skipping the fit step.

If the new file's columns don't match what the model was trained on, we reject it before it reaches the model, with a clear message naming the mismatch — this is caught in Stage 2, not left for the model to discover.

---

## 8. The whole path, in one picture

```
CSV upload
   │
   ▼
raw DataFrame  ──────────────►  EDA view (stats, charts, imbalance note)
   │
   ▼
[Stage 2: split → impute → encode → scale → cast to float64]
   │
   ▼
X_train, X_test, y_train, y_test   (identical shape for every model)
   │
   ▼
model.fit(X_train, y_train)
model.predict(X_test) / predict_proba(X_test)
model.get_metadata()
model.get_visualization_data()   (optional)
   │
   ▼
[Stage 4: route by model_type → compute the right metrics]
   │
   ▼
results view: metrics, confusion matrix / distribution / comparison bars,
              any model-specific chart
   │
   ▼
user chooses: predict on new data (→ Stage 6)  or  compare models
```

---

## 9. What this buys us

Every conversion above exists to answer one question the same way, regardless of which of the 12 algorithms is running: what does a model receive, and what must it return? Because that's fixed once, in the backend, the 12 groups never see each other's code and never need to agree on anything among themselves — they only need to agree with this document and with CODING_STANDARDS.md. If a model's results look wrong, the first question is always "which stage produced this," not "whose code is broken" — Stage 2 is shared infrastructure, Stage 3 is the model's own logic, Stage 4 is shared infrastructure again. That's usually enough to localize a bug before opening a single line of a group's model.
