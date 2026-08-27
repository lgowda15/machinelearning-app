# System Architecture — ML Integration Platform

**Course:** Machine Learning (UM25MB653CA2) · Trimester 3, 2025–26
**Audience:** Integration team (internal). Basis for the application documentation and the faculty submission.
**Version:** 1.0

---

## 1. What the system does

A user uploads a tabular dataset. The application profiles it (exploratory data analysis), lets the user choose one or more models to train, trains them on a user-defined train/test split, and presents evaluation metrics and visualisations. The user can then run the trained model on new data.

The engineering problem is integration: twelve models, written independently by twelve groups across different algorithm families, must run behind one uniform interface. The system's value is that it makes them interchangeable — the same upload-train-evaluate-predict flow works regardless of which model the user picks.

The platform is not an AutoML system. It does not decide which model to use; the user does. It restricts choices only to the extent of grouping models by compatibility with the uploaded data (Section 5).

---

## 2. Technology stack

| Layer | Technology | Role |
|---|---|---|
| Backend framework | FastAPI | HTTP routing, request/response validation, OpenAPI docs |
| ASGI server | Uvicorn | Runs the FastAPI application |
| Validation/schemas | Pydantic | Typed request and response models |
| ML runtime | scikit-learn 1.9, PyTorch, XGBoost, numpy, pandas | Model training and inference |
| Frontend framework | React + Vite | UI, built and served as a static bundle |
| Language (frontend) | TypeScript | Compile-time checking against the API contract |
| Styling | Tailwind CSS | Utility-first styling |
| Charts | Recharts | Confusion matrix, distributions, comparison bars |
| Containerisation | Docker + Docker Compose | Reproducible environments, deployment |
| CI | GitHub Actions | Lint, test, validate on every pull request |
| Lint/format | ruff | Single-tool linting and formatting (Python) |
| Test (Python) | pytest, pytest-cov, httpx | Unit and API integration tests |
| Test (frontend) | Vitest, React Testing Library | Component and unit tests |

FastAPI is to this project what Spring Boot is to a Java service: the web framework that owns routing, dependency injection, and request validation. Uvicorn is the server that runs it, analogous to an embedded Tomcat. Every component above is open source.

---

## 3. Repository structure

```
ml-integration/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI application entry point
│   │   ├── routes/
│   │   │   ├── data.py            # upload, EDA
│   │   │   ├── models.py          # registry, compatibility
│   │   │   ├── training.py        # train, results
│   │   │   └── prediction.py      # predict on new data
│   │   ├── core/
│   │   │   ├── preprocessing.py   # scaling, encoding, imputation
│   │   │   ├── imbalance.py       # class-imbalance detection
│   │   │   ├── metrics.py         # metric computation by model_type
│   │   │   └── registry.py        # model manifest and loader
│   │   └── schemas/               # Pydantic request/response models
│   ├── models/
│   │   ├── base_model.py          # the BaseModel interface
│   │   ├── group_01_decision_trees/
│   │   ├── group_02_random_forest_xgboost/
│   │   ├── ...
│   │   └── group_15_pca/
│   ├── tests/                     # backend unit + integration tests
│   ├── validate_submission.py     # the submission validator
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/            # DataUpload, EDA, ModelSelector,
│   │   │                          #   TrainingPanel, Results, Comparison
│   │   ├── hooks/                 # useModels, useTraining
│   │   ├── api/                   # typed API client
│   │   ├── types/                 # shared TypeScript types
│   │   └── App.tsx
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 4. Model loading — the manifest

Models are registered through an explicit manifest rather than directory scanning. The manifest is a single list the integration team controls.

```python
# app/core/registry.py
from models.group_01_decision_trees.model import DecisionTreeModel
from models.group_13_svm.model import SVMModel
# ... one import per group

REGISTRY = {
    "decision_tree": DecisionTreeModel,
    "svm": SVMModel,
    # ... one entry per group
}
```

At startup the application iterates the manifest, instantiates each class, and confirms it subclasses `BaseModel`. A model that fails is logged and omitted; the rest of the application continues to run.

This is deliberately chosen over scanning the `models/` directory and importing whatever is present. Scanning appears simpler but imports arbitrary code from every folder, fails if a group leaves a stray file, and offers no control over load order or partial failure. With a manifest, a broken submission is one commented-out line and the demo still runs with the remaining eleven models. For a graded demonstration, isolating one group's defect from the whole application is worth the small manual step.

---

## 5. Preprocessing and compatibility

### Preprocessing (backend-owned)

All preprocessing happens once, in `core/preprocessing.py`, and serves both the EDA view and the models. The pipeline: infer column types, impute missing values, encode categoricals, scale numeric features. Scalers and encoders are fit on the **training split only** and applied to the test split, which prevents data leakage. Models receive clean float64 arrays and perform no preprocessing themselves (see Coding Standards Section 4). Centralising this guarantees every model is compared on identically-prepared data.

### Compatibility filter (not recommendation)

The uploaded dataset is classified by shape and target into a data type: labelled with a categorical target (classification), labelled with a continuous target (regression), or unlabelled (clustering). The UI lists all twelve models and greys out those whose `model_type` cannot consume the detected data type — a clusterer on labelled classification data, for instance. This is a static compatibility check keyed on `model_type`, not a learned recommendation. The user retains the final choice among compatible models.

---

## 6. API contract

All endpoints exchange JSON except file uploads (multipart). Errors use a single shape: `{ "error": true, "code": "...", "message": "...", "details": {...} }`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/data/upload` | Upload CSV; returns EDA, detected data type, imbalance note |
| GET | `/api/data/{data_id}` | Retrieve stored dataset profile |
| GET | `/api/models/registry` | List all models and metadata |
| POST | `/api/models/compatible` | Given a dataset, return compatible/incompatible models |
| POST | `/api/training/train` | Train selected models at a given split; returns metrics |
| GET | `/api/training/{training_id}/results` | Retrieve training results |
| POST | `/api/results/comparison` | Side-by-side metrics for trained models |
| POST | `/api/prediction/predict` | Run a trained model on newly uploaded data |

Training is synchronous: the client submits a train request and receives metrics in the response. The train/test split defaults to 80/20 and is user-adjustable. Metrics returned depend on `model_type` — accuracy/precision/recall/F1 and a confusion matrix for classifiers; silhouette, Davies–Bouldin and inertia for clusterers; MSE/RMSE/R²/MAE for regressors; explained-variance for dimensionality reduction.

The full request/response schemas live in the Pydantic models under `app/schemas/` and are the authoritative contract; the frontend's TypeScript types in `src/types/` mirror them.

---

## 7. Frontend flow

```
Upload CSV
   → EDA view (summary statistics, distribution charts, imbalance note)
   → Model selection (all models listed; incompatible ones greyed out)
   → Train/test split control (default 80/20)
   → Train (synchronous; progress indicator)
   → Results (metrics + confusion matrix / distribution / comparison bars)
   → Choose next action:
        • Predict on new data (upload → predictions)
        • Compare models (side-by-side metrics table + bar chart)
```

State is held in `useModels` and `useTraining` hooks. The API client in `src/api/` is typed against `src/types/`, so a backend contract change that isn't reflected in the types surfaces as a compile error rather than a runtime failure.

Three charts, per the agreed scope: a confusion matrix (classifiers), feature/target distributions (EDA), and a model-comparison bar chart (comparison view).

---

## 8. Class-imbalance detection

Implemented in `core/imbalance.py`, applied to classification datasets only. If any class constitutes below 20% or above 80% of samples, the upload response carries an informational flag which the EDA view renders as: *"Class imbalance present; predictions may be biased."* It is informational, shown once at EDA, and never blocks training. Detection is entirely backend-side; model groups are not involved.

---

## 9. Continuous integration

`.github/workflows/ci.yml` runs on every pull request:

1. **ruff** — lint and format check on changed Python.
2. **pytest with coverage** — unit tests; the submission must meet the 80% threshold.
3. **validate_submission.py** — the interface contract check from the Coding Standards.
4. **Integration smoke test** — loads the submitted model through the registry, runs a fit/predict cycle on a small fixture dataset, confirms output shapes.

A pull request is not reviewed until CI is green. Merging requires two approvals, one from the integration lead. GitHub Actions is used because the repository is already on GitHub; the runner is open source and the hosted minutes are free at this scale. (Self-hosted open-source alternatives such as Woodpecker CI or Drone exist but are unnecessary here.)

---

## 10. Testing strategy

| Level | Owner | Tools | Scope |
|---|---|---|---|
| Model unit tests | Each of the 12 groups | pytest, pytest-cov | Their model in isolation, ≥80% coverage |
| Model validation | CI (automatic) | validate_submission.py | Interface conformance |
| Integration testing | Testing teams (Groups 10, 12, 14) | pytest, httpx | Model behaviour through the API |
| API integration | Integration team | pytest, httpx | Endpoint correctness end to end |
| Frontend unit | Integration team | Vitest, React Testing Library | Components, hooks |
| End-to-end (optional) | Integration/testing teams | Playwright | Full browser flow |

Testing team assignments:

- **Testing #1 (Group 14):** SVM, Regression, PCA, KNN/K-Means/GMM, HMM/Naïve Bayes, DBSCAN/Hierarchical.
- **Testing #2 (Group 10):** LDA & QDA, CNN, Random Forest + XGBoost.
- **Testing #3 (Group 12):** Decision Trees, ANN, RNN/LSTM/GRU.

---

## 11. Deployment

The application is containerised. `docker-compose.yml` defines two services: the FastAPI backend (Uvicorn) and the frontend (static bundle served behind a lightweight web server). Compose gives every team member and the deployment target an identical environment.

Sizing note: the backend image includes PyTorch (CPU build, roughly 800 MB) and holds models in memory during inference, so the host needs on the order of 1–2 GB RAM. Free tiers capped at 512 MB will run out of memory when a PyTorch model loads. Suitable targets are a container platform such as Fly.io or a small virtual private server; both run the same compose file. The deep-learning models drive this requirement — the classical models alone would fit comfortably in far less.

Build locally:

```bash
docker compose build
docker compose up
```

The backend serves the API; the frontend bundle is served alongside and calls it. For the deployed instance, the same images run on the chosen host with the backend URL supplied to the frontend at build time.

---

## 12. Timeline

Kickoff July 14; six-week build with a one-week buffer. Group submissions are staggered (scikit-learn-only models first, deep-learning models last) with an absolute deadline of August 15. The integration team builds the backend skeleton in week one, integrates models as they arrive, then builds the frontend once the backend stabilises. Testing teams validate submissions as they land and run integration testing once models are combined. Full schedule is maintained separately in the project plan.

---

## 13. Open items for the team

- Finalise `validate_submission.py` before kickoff — it is the enforceable form of the coding standards, and the whole submission process depends on it.
- Confirm the deployment host (Fly.io vs VPS) and provision it.
- Assign the four integration-team roles (two frontend, one backend, one QA/DevOps).
- Decide whether end-to-end Playwright tests are in scope or dropped for time.
