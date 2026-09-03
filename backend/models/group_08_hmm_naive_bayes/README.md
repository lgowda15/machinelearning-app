# HMM, Naive Bayes

## Model

Two independent classifiers in this group (CODING_STANDARDS.md Section 6 —
one class per algorithm):

- **`NaiveBayesModel`** — Gaussian Naive Bayes. Assumes each feature is
  conditionally normal given the class and classifies by comparing the
  product of those per-feature densities across classes. Standard tabular
  input, no reshaping.
- **`HMMModel`** — a Hidden Markov Model classifier. Fits one HMM per class
  on that class's training sequences and classifies a new sequence by
  which class's HMM assigns it the highest likelihood. Sequence model
  (Section 4) — see "Sequence feature layout" below.

## Usage

```python
from models.group_08_hmm_naive_bayes import NaiveBayesModel, HMMModel

nb = NaiveBayesModel()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)

# HMMModel requires exactly 10 columns -- see the sequence layout section.
hmm = HMMModel()
hmm.fit(X_train_seq, y_train_seq)
hmm_preds = hmm.predict(X_test_seq)
```

## Hyperparameters

### `NaiveBayesModel`

| Name | Default | Controls |
|---|---|---|
| `var_smoothing` | `1e-9` | Fraction of the largest observed feature variance added to every feature's variance estimate, to avoid divide-by-zero on near-constant features. Raise it if a feature with almost no spread in the training data is producing extreme, overconfident probabilities. |
| `random_state` | `42` | Accepted for interface consistency with every other model in the platform. `GaussianNB.fit` has no stochastic component (means and variances are closed-form per-class statistics), so this value is stored but never passed to sklearn — changing it has no effect on output. |

### `HMMModel`

| Name | Default | Controls |
|---|---|---|
| `n_components` | `3` | Number of hidden states each class's HMM has available. More states let the model represent more distinct "regimes" within a sequence, at the cost of needing more training sequences per class to estimate them (fit raises if a class has fewer training sequences than `n_components`). |
| `covariance_type` | `"diag"` | Shape of each hidden state's Gaussian emission covariance: `"diag"` (independent per-feature variances — 2 parameters/state here), `"full"` (also models correlation between the 2 features/step), `"tied"` (all states share one covariance), or `"spherical"` (one variance for all features). `"diag"` is the default because it is the cheapest that still lets the two features per timestep vary independently. |
| `n_iter` | `50` | Maximum Baum-Welch (EM) iterations when fitting each class's HMM. Raise it if training log-likelihood is still improving noticeably at 50 iterations; lower it to speed up training at the cost of a less-converged fit. |
| `lookback` | `5` | Number of timesteps per sequence. Fixed at `5` for this platform's schema (Section 4) — the constructor raises if given any other value. It exists as a named argument, rather than being silently hardcoded, so the schema is visible in the hyperparameter form the same way every other hyperparameter is. |
| `random_state` | `42` | Seeds Baum-Welch's k-means-based parameter initialisation, which is stochastic. Required for determinism (Section 10) — without it, two fits on identical data can converge to different (equally valid) hidden-state labellings and produce different likelihoods. |

## Sequence feature layout (`HMMModel` only)

`HMMModel` is a sequence model under Section 4: it receives the same 2D
`(n_samples, n_features)` input as every other model and reshapes internally,
rather than the backend supplying a 3D array.

- `X` must be a 2D numpy array, dtype `float64`, shape `(n_samples, 10)`.
- `y` must be a 1D array with at least 2 distinct classes.
- Flattened column order in each row:
  `[t0_f0, t0_f1, t1_f0, t1_f1, t2_f0, t2_f1, t3_f0, t3_f1, t4_f0, t4_f1]`.
- Internal reshape performed by the model:
  `(n_samples, 10) -> (n_samples, 5, 2)`.

Timesteps: `5`
Features per timestep: `2`

This is a longer lookback than Group 3's RNN (`3` timesteps). An HMM
estimates a transition matrix between hidden states from the transitions
*within* each training sequence; 3 timesteps gives only 2 transitions per
sequence to learn from, which is thin. 5 timesteps (4 transitions per
sequence) was chosen as a better minimum for `n_components=3` states to be
estimated reasonably, while keeping sequences short enough that this stays
well inside the 5-minute CPU training ceiling.

### Rejection conditions

`HMMModel.fit`/`predict`/`predict_proba` raise `ValueError` if:
- `X` is not 2D,
- `X` dtype is not `float64`,
- `X` does not have exactly 10 columns,
- `X` contains non-finite values (`NaN`, `inf`),
- `y` is missing, non-1D, or row count mismatches `X`,
- fewer than 2 classes are present in `y`,
- any class has fewer training sequences than `n_components`.

`predict`/`predict_proba` raise `RuntimeError` if called before `fit`.

`NaiveBayesModel` has no fixed column count — any number of preprocessed
tabular features is accepted, per the standard tabular contract.

## Running the tests

```bash
cd backend
python -m pytest models/group_08_hmm_naive_bayes --cov=models/group_08_hmm_naive_bayes --cov-report=term-missing
```

## Design decisions

**Why Gaussian Naive Bayes, not Multinomial or Bernoulli.** The backend
guarantees scaled, zero-centred, continuous `float64` input (Section 4).
`MultinomialNB` assumes non-negative count-like features and `BernoulliNB`
assumes binary features; both would either raise on StandardScaler output
or silently threshold/clip it into something the input was never encoded
as. `GaussianNB`'s per-class-per-feature normal assumption is the only one
of the three that matches what the backend actually hands every model.

**Why Naive Bayes' `feature_importance` is a heuristic, not native.**
Naive Bayes has no coefficient vector or split-gain analogue to report.
What it does compute per class is `theta_`, the per-class mean of every
feature. A feature that barely shifts between classes contributes little
to telling them apart; one whose class-conditional mean moves a lot does
most of the separating. `_feature_importance` reports the standard
deviation of each feature's per-class means across classes as that signal.
It is a reasonable proxy for "how much this feature distinguishes the
classes under this model," not a magnitude comparable to a linear model's
coefficients or a tree's gain — documented here so it isn't read as more
than it is.

**Why an HMM classifies by per-class likelihood, not by treating hidden
states as clusters.** There are two standard ways to get a class label out
of an HMM. One fits a single HMM to all the data and treats the inferred
hidden-state sequence as a clustering signal; that produces state labels,
not the ground-truth classes, and would need a second mapping step from
states to classes that has no principled definition when states don't
align 1:1 with classes. The other — used here — fits one HMM per class on
that class's own sequences, and classifies a new sequence by asking each
class's HMM "how likely is this sequence under what I learned," then
picking the class whose HMM answers highest. This is the standard
generative pattern for HMM classification (the same structure as a
Gaussian-mixture or Naive-Bayes classifier: one generative model per class,
classify by likelihood) and it maps directly onto this platform's
single-`predict`-call contract, since each class's HMM is independent and
`score()` gives a well-defined per-sequence, per-class number to compare.

**Why `predict_proba` is a softmax over log-likelihoods, and why that's
exact, not an approximation.** `GaussianHMM.score()` returns a
log-likelihood, `log P(X | class)`, not a probability. Under a *uniform*
prior over classes (no class assumed more likely than another before
seeing the sequence), Bayes' rule gives `P(class | X) ∝ P(X | class)`, and
normalising a set of values proportional to `P(X | class)` by their sum is
exactly what `softmax` on their logs computes:
`softmax(log p_1, ..., log p_k)_i = p_i / sum_j(p_j)`. So this isn't a
softmax used as a generic squashing trick — it is the exact posterior under
the one prior assumption the model makes explicit (uniform), which is also
the only prior assumption available: nothing in this platform's fit
contract passes class-frequency information beyond what's implicit in the
count of sequences given to each class's HMM.

**Why `lookback` is a constructor argument fixed to one legal value,
rather than hardcoded.** Following Group 3's RNN precedent: the schema is
fixed for this repository (Section 4 requires documenting, not
negotiating, the column layout), but exposing it as a named, validated
argument keeps it visible in the same hyperparameter surface as every
other tunable, rather than being an invisible constant a reader has to go
find in the source to discover.

**Why `covariance_type="diag"` by default, not `"full"`.** `"full"`
additionally models the correlation between the 2 features at each
timestep, which is a real capability but needs more data per hidden state
to estimate without overfitting (a full covariance matrix has more free
parameters than a diagonal one). `"diag"` is the safer default for
whatever size of dataset a user uploads; `"full"` remains available as an
explicit choice for users with enough data per class to support it.

**Why `n_components < training sequences for a class` is a hard failure,
not a silent reduction.** hmmlearn will not itself refuse to fit an HMM
with more hidden states than a class has training sequences to constrain
them — it will fit *something*, silently underdetermined, and produce
numbers that look valid but reflect too little data rather than the
algorithm. Section 10's explicit-failure rule exists precisely for cases
like this: raising here, with a message naming both counts and what to do
about it, is more useful than a quietly unreliable model that would need a
testing team to notice the numbers don't reproduce.
