"""Public interface for the Group 11 LDA/QDA submission."""

from .comparison import compare_lda_qda
from .lda import LDAModel
from .qda import QDAModel
from .suitability import analyze_suitability

__all__ = ["LDAModel", "QDAModel", "analyze_suitability", "compare_lda_qda"]
