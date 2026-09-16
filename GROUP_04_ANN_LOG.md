# Group 4 (ANN) — Submission Log

Standalone from `GROUP_REMEDIATION.md`, which tracks the original nine group
PRs (plus the post-sweep CNN addition). Group 4's submission arrived later
and separately, with its own set of problems worth a dedicated record — this
file exists specifically so it can be handed to the professor as-is.

## Timeline

The original nine-group remediation sweep was declared done on 2026-08-30
(`GROUP_REMEDIATION.md`). Group 4's submission started **after** that date
and continued in pieces for almost a week:

| Date | PR | What it was |
|---|---|---|
| 2026-09-01 | #49 | A loose `.py` file dropped at the repo root — not in the group's folder, not usable as-is |
| 2026-09-02 | #51 | The real `ANNModel` implementation (`model.py` + `test.py`) |
| 2026-09-07 13:51 | #53 | `__init__.py` — needed to actually expose `ANNModel` from the package |
| 2026-09-07 14:35 | #54 | README rewrite |
| 2026-09-07 14:43 | #55 | `requirements.txt` fix (correct dependency) |
| 2026-09-07 14:56 | #56 | `requirements.txt` — accidentally overwritten with README prose |

So: a five-day gap between the first attempt (#49) and the actual model
landing (#51), then a further five-day gap before the supporting files
(package export, docs, dependencies) showed up — and even then, arriving as
four separate single-file PRs rather than one submission, with one of those
four (#56) itself a mistake. Six PRs total to deliver what every other group
delivered as one.

## Every mistake found, logged individually

### 1. PR #49 — submission outside the group's folder
Added `ANN_MODEL_CORRECTED_FINAL (1).py` at the **repository root**, not
under `backend/models/group_04_ann/`. Per CODING_STANDARDS.md §12/13,
submissions live inside the group's own folder; this file was not usable
in place and would have failed CI's folder-scope check regardless.
**Action: closed without merging** (comment posted pointing to the real PRs).

### 2. PR #56 — wrong content in `requirements.txt`
The diff appends ~118 lines of README-style markdown (Overview,
Architecture, activation functions, hyperparameters, testing, etc.) onto
`backend/models/group_04_ann/requirements.txt`, on top of the real
dependency lines. This is the same content that (correctly) landed in the
README in PR #54 — pasted into the wrong file. Merging it would have broken
`pip install -r requirements.txt` for this folder.
**Action: closed without merging** (comment posted).

### 3. Submission split across four uncoordinated single-file PRs
Rather than one PR containing the model, its test, the package export, the
README, and the dependency pin — the standard shape every other group used
— this arrived as four separate PRs (#51, #53, #54, #55), each touching
exactly one file, opened in a burst over about an hour on 2026-09-07,
five days after the model itself (#51) had already been sitting open.
Nothing about this required four PRs; `__init__.py`, `README.md`, and
`requirements.txt` are all trivial, same-session edits.
**Action: no code fix needed** — content is correct, just fragmented.
Combined into one consolidated branch/PR for merge (see below).

### 4. Dead duplicate module docstring in `model.py`
The file kept the original stub's placeholder docstring
(`"""Artificial Neural Network (MLP) — implement YourModel(BaseModel)
here. ..."""`) as the first statement in the file, then added the group's
own real docstring as a second, unassigned string-literal expression
directly below it. Only the first string in a module becomes `__doc__` in
Python — the group's actual documentation was dead code, never accessible
as the module's docstring.
**Action: fixed.** Merged into a single real docstring, stub text removed.

### 5. Stray leftover comment in `model.py`
Line 5 of the submitted file was `# models/group_04_ann/model.py` — a
self-referential path comment, almost certainly left over from copy-pasting
out of an editor tab or a scratch script, serving no purpose in the file.
**Action: fixed.** Removed as part of the docstring cleanup above.

### 6. Ruff: unsorted/unformatted import block in `model.py`
```
import time
import numpy as np
import torch
...
```
`time` (stdlib) and `numpy`/`torch` (third-party) were not separated into
groups per the project's isort convention — flagged by `ruff check`
(`I001`).
**Action: fixed.** Blank line inserted between the stdlib and third-party
import groups.

## What was *not* a mistake — confirmed working as submitted

- `ANNModel.fit()`/`predict()`/`predict_proba()`/`get_metadata()` all pass
  `validate_submission.py`'s full 16-check suite (determinism, exact
  metadata keys, `predict_proba` contract, JSON-safety, no prints, no
  hardcoded paths, coverage) — zero logic-level bugs found.
- The group's own `test.py` (12 tests) passes unmodified.
- `requirements.txt`'s real fix (PR #55: drop `scikit-learn`, add
  `torch==2.7.1`) is consistent with the project — the pin itself is
  advisory documentation of the group's dependency, not what's installed
  into the shared venv (root `backend/requirements.txt` already pins
  `torch==2.13.0+cpu` for group_03_rnn; same convention noted there
  applies here — confirmed the model imports and runs correctly against
  that version).

## Design note (logged, not a blocker)

`ANNModel` is the first group model whose `model_type` is decided
dynamically from `y`'s dtype at `fit()` time (integer/boolean → classifier,
float → regressor), rather than being a static property of the class the
way every other registered model works. Before `fit()`, `get_metadata()`
falls back to `"classifier"` so the conformance suite's fixture-generation
step (which reads `model_type` from an unfitted instance) has something
valid to key off of — meaning the generic conformance suite only ever
exercises the classifier path for this model. Accepted as the platform's
answer for this group, same spirit as Group 9's predict-on-unseen-data
resolution in `DECISIONS.md`. Not treated as a blocker per the standing
policy in `GROUP_REMEDIATION.md`.

## Resolution

Real changes (#51, #53, #54, #55) combined into one consolidated branch and
merged as a single PR from the integration account (no push access to the
contributor's fork, same constraint as every prior group). #49 and #56
closed without merging. Registered as `"ann"` in
`backend/app/core/registry.py`. Conformance suite run scoped to this group,
confirmed passing, before considering this group done.

**Zero logic-level bugs** — every issue above was scaffolding: a stray
out-of-folder file, one file overwritten with the wrong content, needless
PR fragmentation, a dead docstring, a stray comment, and one import-order
lint finding.
