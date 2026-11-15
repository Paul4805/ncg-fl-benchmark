"""Class-gap / NCG definitions.

Class Gap = max_c(recall_c) - min_c(recall_c)
NCG       = (ClassGap_FL - ClassGap_Centralized) / (1 - ClassGap_Centralized)

Recall vectors are ALWAYS averaged before a gap is computed; gaps are never averaged.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import recall_score


def per_class_recall(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Per-class recall with labels=range(C) and zero_division=0."""
    return np.asarray(
        recall_score(
            np.asarray(y_true).ravel(),
            np.asarray(y_pred).ravel(),
            labels=list(range(num_classes)),
            average=None,
            zero_division=0,
        ),
        dtype=np.float64,
    )


def class_gap(recall_vector) -> float:
    r = np.asarray(recall_vector, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    return float(r.max() - r.min())


def average_recall_vectors(vectors) -> np.ndarray:
    arr = np.asarray([np.asarray(v, dtype=np.float64) for v in vectors], dtype=np.float64)
    return arr.mean(axis=0)


def ncg(fl_recall, cent_recall) -> float:
    return ncg_from_gaps(class_gap(fl_recall), class_gap(cent_recall))


def ncg_from_gaps(fl_gap: float, cent_gap: float) -> float:
    denom = 1.0 - float(cent_gap)
    if abs(denom) < 1e-12:
        return float("nan")
    return float((float(fl_gap) - float(cent_gap)) / denom)


def minority_class(recall_vector) -> int:
    return int(np.asarray(recall_vector).argmin())


def mean_std(values):
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))
