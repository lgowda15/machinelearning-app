"""Group 13 — Support Vector Machine.

Re-exports the model class so the platform imports from the package
(``models.group_13_svm``) rather than from internal file layout, per
Section 6 of the Coding Standards.
"""

from .model import SVMModel

__all__ = ["SVMModel"]
