from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int


def bootstrap_metric_ci(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    metric: str = "auc",
    threshold: float | None = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 2026,
) -> ConfidenceInterval:
    """Identity-trial bootstrap confidence interval for AUC or accuracy."""
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be equally sized one-dimensional arrays")
    if len(np.unique(y)) < 2:
        raise ValueError("both classes are required")
    metric = metric.lower()
    if metric not in {"auc", "accuracy"}:
        raise ValueError("metric must be 'auc' or 'accuracy'")
    if metric == "accuracy" and threshold is None:
        raise ValueError("threshold is required for accuracy")

    def calculate(index: np.ndarray) -> float:
        sample_y, sample_s = y[index], s[index]
        if metric == "auc":
            if len(np.unique(sample_y)) < 2:
                return float("nan")
            return float(roc_auc_score(sample_y, sample_s))
        prediction = (sample_s >= float(threshold)).astype(np.int32)
        return float(accuracy_score(sample_y, prediction))

    full = calculate(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [calculate(rng.integers(0, len(y), size=len(y))) for _ in range(int(n_boot))],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("bootstrap produced no valid samples")
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return ConfidenceInterval(full, float(lower), float(upper), confidence, int(values.size))


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float | int]:
    """Exact two-sided McNemar comparison for paired model decisions."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired correctness arrays must have the same one-dimensional shape")
    b_only = int(np.sum(~a & b))
    a_only = int(np.sum(a & ~b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        from scipy.stats import binomtest

        p_value = float(binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def identity_folds(identities: Sequence[str], k: int = 5, seed: int = 2026) -> list[list[str]]:
    """Create deterministic identity-disjoint folds."""
    values = sorted(set(identities))
    if k < 2 or k > len(values):
        raise ValueError("k must be between 2 and the number of identities")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(values, dtype=object)[rng.permutation(len(values))]
    return [list(chunk.astype(str)) for chunk in np.array_split(shuffled, k)]


def calibrate_similarity_threshold(labels: Sequence[int], similarities: Sequence[float]) -> dict[str, float]:
    """Return Youden and equal-error operating thresholds."""
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(similarities, dtype=np.float64)
    if y.shape != s.shape or len(np.unique(y)) < 2:
        raise ValueError("calibration requires equally sized scores from both classes")
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    fnr = 1.0 - tpr
    youden_index = int(np.argmax(tpr - fpr))
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "optimal_threshold": float(thresholds[youden_index]),
        "eer_threshold": float(thresholds[eer_index]),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "auc": float(roc_auc_score(y, s)),
    }


def summarize_folds(fold_metrics: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    values = list(fold_metrics)
    if not values:
        raise ValueError("at least one fold is required")
    keys = sorted(set.intersection(*(set(item) for item in values)))
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        numeric = np.asarray([item[key] for item in values], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(numeric)),
            "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
            "min": float(np.min(numeric)),
            "max": float(np.max(numeric)),
        }
    return summary
