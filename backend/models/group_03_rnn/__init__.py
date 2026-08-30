"""RNN group package. Set PyTorch deterministic config at import time per coding standards.
This file is intentionally minimal: it sets deterministic PyTorch flags if torch is available
and re-exports the model classes.
"""

# Set PyTorch deterministic behaviour at import time for deep-learning groups (3,4,7).
try:
    import torch
except Exception as exc:  # if torch cannot be imported, fail loudly per standards
    raise ImportError(
        "PyTorch is required for group_03_rnn and must be installed (CPU wheel). "
        "Install a compatible torch for Python 3.12 and rerun tests. Underlying error: "
        f"{type(exc).__name__}: {exc}"
    ) from exc

# Set deterministic behaviour and seed — do not swallow failures
torch.manual_seed(42)
try:
    torch.use_deterministic_algorithms(True)
except Exception as exc:
    raise RuntimeError(
        "torch.use_deterministic_algorithms(True) failed; a deterministic "
        "PyTorch configuration is required by the coding standards. "
        f"Underlying error: {type(exc).__name__}: {exc}"
    ) from exc

# Re-export model classes (files created in this package)
from .gru import GRUModel
from .lstm import LSTMModel
from .rnn import RNNModel

__all__ = ["GRUModel", "LSTMModel", "RNNModel"]
