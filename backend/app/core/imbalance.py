"""Class-imbalance detection (ARCHITECTURE.md SS8, DATA_FLOW_GUIDE.md Stage 5).

Applied to classification targets only, computed once on the raw target
column right after Stage 1. Purely informational: the EDA response carries
a flag, the frontend shows a note once, and it never blocks training or
reaches any model.
"""
import pandas as pd

from app.schemas.data import ImbalanceInfo

LOWER_THRESHOLD = 0.20
UPPER_THRESHOLD = 0.80
FLAG_MESSAGE = "Class imbalance present; predictions may be biased."


def detect_imbalance(y: pd.Series) -> ImbalanceInfo:
    """Flag imbalance if any class is below 20% or above 80% of samples."""
    clean = y.dropna()
    counts = clean.value_counts()
    total = int(counts.sum())

    class_counts = {str(label): int(count) for label, count in counts.items()}

    if total == 0:
        return ImbalanceInfo(is_imbalanced=False, class_counts=class_counts, message=None)

    proportions = counts / total
    is_imbalanced = bool(((proportions < LOWER_THRESHOLD) | (proportions > UPPER_THRESHOLD)).any())

    return ImbalanceInfo(
        is_imbalanced=is_imbalanced,
        class_counts=class_counts,
        message=FLAG_MESSAGE if is_imbalanced else None,
    )
