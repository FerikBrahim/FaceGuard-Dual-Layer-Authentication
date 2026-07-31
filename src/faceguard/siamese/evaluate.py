from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def threshold_metrics(labels: np.ndarray, similarities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    similarities = np.asarray(similarities, dtype=np.float64)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, similarities, pos_label=1)
    false_negative_rate = 1.0 - true_positive_rate
    eer_index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    youden_index = int(np.argmax(true_positive_rate - false_positive_rate))
    optimal_threshold = float(thresholds[youden_index])
    predictions = (similarities >= optimal_threshold).astype(np.int32)
    genuine = similarities[labels == 1]
    impostor = similarities[labels == 0]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, similarities)),
        "eer": float((false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2.0),
        "eer_threshold": float(thresholds[eer_index]),
        "optimal_threshold": optimal_threshold,
        "genuine_similarity_mean": float(np.mean(genuine)),
        "genuine_similarity_std": float(np.std(genuine)),
        "impostor_similarity_mean": float(np.mean(impostor)),
        "impostor_similarity_std": float(np.std(impostor)),
        "separation": float(np.mean(genuine) - np.mean(impostor)),
    }
