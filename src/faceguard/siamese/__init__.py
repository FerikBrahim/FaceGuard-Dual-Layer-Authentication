"""Siamese training, verification, baselines, and statistical evaluation."""

from .evaluate import threshold_metrics
from .loss import BatchSemiHardTripletLoss, CosineTripletLoss
from .model import FaceTripletModel, SiameseVerifier
from .statistics import bootstrap_metric_ci, calibrate_similarity_threshold, identity_folds, mcnemar_exact

__all__ = [
    "BatchSemiHardTripletLoss",
    "CosineTripletLoss",
    "FaceTripletModel",
    "SiameseVerifier",
    "bootstrap_metric_ci",
    "calibrate_similarity_threshold",
    "identity_folds",
    "mcnemar_exact",
    "threshold_metrics",
]
