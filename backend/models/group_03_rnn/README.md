# RNN / LSTM / GRU

## Model
Group 3 sequence models for regression/forecasting, using backend-preprocessed
2D float64 input.

## Usage
```python
from models.group_03_rnn import RNNModel

model = RNNModel()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## Sequence feature layout (Group 3 contract)

The Group 3 sequence contract is fixed for this repository workstream:

- `X` must be a 2D numpy array, dtype `float64`, shape `(n_samples, 6)`.
- `y` must be a 1D numeric regression target, shape `(n_samples,)`.
- Flattened column order in each row:
  `[t0_f0, t0_f1, t1_f0, t1_f1, t2_f0, t2_f1]`.
- Internal reshape performed by the model:
  `(n_samples, 6) -> (n_samples, 3, 2)`.

Timesteps: `3`  
Features per timestep: `2`

### Rejection conditions

`fit`/`predict` raise `ValueError` if:
- `X` is not 2D,
- `X` dtype is not `float64`,
- `X` does not have exactly 6 columns,
- `X` contains non-finite values (`NaN`, `inf`),
- `y` is missing, non-1D, non-numeric, non-finite, or row count mismatches `X`.

`predict` raises `RuntimeError` if called before `fit`.

## Running the tests
```bash
cd backend
python -m pytest models/group_03_rnn -q
```
