import numpy as np

from faceguard.siamese.statistics import (
    bootstrap_metric_ci,
    calibrate_similarity_threshold,
    identity_folds,
    mcnemar_exact,
)


def test_threshold_calibration_separates_simple_scores():
    result = calibrate_similarity_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert result["auc"] == 1.0
    assert 0.2 < result["optimal_threshold"] <= 0.9


def test_identity_folds_are_disjoint_and_complete():
    identities = [f"id_{index}" for index in range(10)]
    folds = identity_folds(identities, k=5, seed=7)
    flattened = [identity for fold in folds for identity in fold]
    assert sorted(flattened) == sorted(identities)
    assert len(flattened) == len(set(flattened))


def test_mcnemar_and_bootstrap_outputs():
    result = mcnemar_exact([True, True, False, False], [True, False, True, False])
    assert result["discordant"] == 2
    ci = bootstrap_metric_ci([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], metric="auc", n_boot=100)
    assert ci.estimate == 1.0
    assert 0.0 <= ci.lower <= ci.upper <= 1.0
